# -*- coding: utf-8 -*-
"""
Created on Thu Jan 15 09:55:51 2026

@author: yliu

# Run in Windows SAR_2026 env 
"""

from __future__ import annotations

import os
import pandas as pd
import pyarrow as pa
import numpy as np
from pathlib import Path
import subprocess

from typing import Sequence
from docx import Document
from IPython.display import display

import re

import oracledb
from sqlalchemy import create_engine, text, inspect
from sqlalchemy.dialects.oracle import NUMBER, VARCHAR2, CLOB, DATE

from nutbolt.sar_helpers import freq, means

pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)

# -----------------------------------------------------------------------------
# Paths / inputs (mirror your SAS macro vars)
# -----------------------------------------------------------------------------

data_folder = Path("/mnt/d/ISAR Multi-CPTs/Oct 2026/Data")
model_base = Path("/mnt/d/ISAR Multi-CPTs/Oct 2026/Multiple CPTs")

for d in [data_folder, model_base,]:
    if not Path(d).is_dir():
        d.mkdir(parents=True, exist_ok=True)
        
sar_paths = [
    Path(data_folder / "isar_2022oct_final.parquet"),
    Path(data_folder / "isar_2023oct_final.parquet"),
    Path(data_folder / "sar_2024oct_final.parquet"),
    Path(data_folder / "sar_2025oct_final.parquet"),
]        

sar1 = pd.read_parquet(sar_paths[0]).drop(columns=["delirium"])  # SAR Oct 2022 
sar2 = pd.read_parquet(sar_paths[1]).drop(columns=["delirium"])  # SAR Oct 2023 
sar3 = pd.read_parquet(sar_paths[2]).drop(columns=["delirium"])  # SAR Oct 2024 
sar4 = pd.read_parquet(sar_paths[3])  # SAR Oct 2025 
sar4["delirium"].value_counts(dropna=False)
sar4["delirium"].drop(columns=["delirium"], inplace=False)
sar4yr = [sar1, sar2, sar3, sar4]

# =============================================================================
# Load data from the previous step 
# In July SAR, this should be step3.sas7bdat 
# =============================================================================

step3 = pd.read_parquet(data_folder / "step3.parquet")
step3.columns = step3.columns.str.lower()  # convert column headers to lower case

sorted(step3.columns.tolist())
step3[["division_n", "asm1"]] = step3[["division_n", "asm1"]].astype(int)

step3[["delirium", "fhsdecline"]] = step3[["delirium", "fhsdecline"]].apply(
    pd.to_numeric, errors="coerce").astype("Int64")

step3.loc[step3["age"]>=65, ["delirium"]].value_counts(dropna=False)
step3.loc[step3["age"]>=65, ["fhsdecline"]].value_counts(dropna=False)

# =============================================================================
# Core 5-year “keep” subset + derived fields
# =============================================================================

KEEP_COLS = [
    "division","asm1","prncptx","age", "agegroup","agegroup_bucket","asaclas","fnstatpressurg",
    "oprymd","tend","hdisdt","opsdlos", 
    "postcode","score1b","compssi","computi","compdvt","compvent","comprenal","comptube",
    "compcard","comppneu","compsepsis","returnor","readmission","dsermorb","ssi",
    'ssi2', # measure colon SSI outcome
    "cnscva","postopcdiff","dehis","colon","colorectal",
    "sermorb", # this is outcome for Leandra, same definition as dsermorb except excluding death
    "orgspcssi", "wndinfd", "dssipatos", "sssipatos", "ossipatos",
    "delirium", "fhsdecline", # delirium and fhsdecline are two new Geri outcomes with age >=65
    'concpt1', 'concpt2', 'concpt3', 'concpt4', 'concpt5',
    'concpt6', 'concpt7', 'concpt8', 'concpt9', 'concpt10',
    'othcpt1', 'othcpt2', 'othcpt3', 'othcpt4', 'othcpt5',
    'othcpt6', 'othcpt7', 'othcpt8', 'othcpt9', 'othcpt10',
    ]

def read_sas(path: Path) -> pd.DataFrame:
    df=  pd.read_sas(str(path), encoding="latin-1")
    df.columns = df.columns.str.lower()
    return df

def make_asaclas_c(asaclas: pd.Series) -> pd.Series:
    # mirrors SAS string labels; adjust if your asaclas values differ
    out = pd.Series(pd.NA, index=asaclas.index, dtype="object")
    out[asaclas.eq("1-No Disturb")] = "1"
    out[asaclas.eq("2-Mild Disturb")] = "2"
    out[asaclas.eq("3-Severe Disturb")] = "3"
    out[asaclas.isin(["4-Life Threat", "5-Moribund"])] = "4-5"
    return out

def compute_opsdlos(df: pd.DataFrame) -> pd.Series:
     # SAS: if datepart(hdisdt) >= datepart(tend) use hdisdt else tend; then intck day from oprymd
     # In python: ensure datetime
     opr = pd.to_datetime(df["oprymd"], errors="coerce")
     hdis = pd.to_datetime(df["hdisdt"], errors="coerce")
     tend = pd.to_datetime(df["tend"], errors="coerce")
 
     end_date = tend.where(~(hdis >= tend), hdis)  # if hdis >= tend then hdis else tend
     los = (end_date - opr).dt.days
 
     # SAS: if (opsdlos=. or missing(hdisdt)) and scorePATOS=0 then opsdlos=30
     return los

def compute_scorePATOS(df: pd.DataFrame) -> pd.Series:
    # SAS: scorePATOS=1 if ANY complication indicator triggers
    cond = (
        (df["compssi"].fillna(0) != 0) |
        (df["compdvt"].fillna(0) != 0) |
        (df["comptube"].fillna(0) != 0) |
        (df["compvent"].fillna(0) != 0) |
        (df["comppneu"].fillna(0) != 0) |
        (df["comprenal"].fillna(0) != 0) |
        (df["computi"].fillna(0) != 0) |
        (df["compcard"].fillna(0) != 0) |
        (df["compsepsis"].fillna(0) != 0) |
        (df["cnscva"].astype("object").fillna("No Complication") != "No Complication") |
        (df["dehis"].astype("object").fillna("No Complication") != "No Complication")
    )
    return cond.astype("int8")

def compute_sermorb(df: pd.DataFrame) -> pd.Series:
    cond = (
        (df["compcard"] == 1) |
        (df["comptube"] == 1) |
        (df["comppneu"] == 1) |
        (df["comprenal"] == 1) |
        (df["computi"] == 1) |
        (df["returnor"] == 1) |
        (df["compsepsis"] == 1) |
        (
            (df["orgspcssi"] != "No Complication") &
            (df["dssipatos"] != 1) &
            (df["sssipatos"] != 1) &
            (df["ossipatos"] != 1)
        ) |
        (
            (df["wndinfd"] != "No Complication") &
            (df["dssipatos"] != 1) &
            (df["sssipatos"] != 1) &
            (df["ossipatos"] != 1)
        ) |
        (df["dehis"] != "No Complication")
    )
    return cond.astype("int8")

def assign_T_procs(df: pd.DataFrame) -> pd.Series:
    """
    Python equivalent of SAS %TLOS_def macro for currently-active mappings.
    Returns a Series T_procs with values like 'GenPanDistal', etc.

    Notes:
    - Uses sequential assignment to mimic SAS multiple IF statements.
    - Assumes df has PRNCPTX column (string or numeric). Will convert to string.
    - Only includes mappings that are NOT commented out in your macro.
    """
    p = df["prncptx"].astype(str).str.strip()

    t = pd.Series(pd.NA, index=df.index, dtype="object")

    # Gen Pan
    t.loc[p.isin(['48140', '48145', '48146'])] = "GenPanDistal"
    t.loc[p.isin(['48150', '48152', '48153', '48154', '48155'])] = "GenPanWhipple"

    # General Colon
    t.loc[p.isin([
        '44140', '44141', '44143', '44144', '44145', '44146', '44147', '44150', '44151',
        '44160', '44204', '44205', '44206', '44207', '44208', '44210'
    ])] = "GenColon"

    # General Proctectomy
    t.loc[p.isin([
        '44155', '44156', '44157', '44158', '44211', '44212',
        '45110', '45111', '45112', '45113', '45114', '45116', '45119',
        '45120', '45121', '45123', '45126', '45130', '45135', '45160',
        '45395', '45397', '45402', '45550'
    ])] = "GenProctectomy"

    # Hepatectomy
    t.loc[p.isin(['47122', '47125', '47130'])] = "GenHepMajor"
    t.loc[p.isin(['47120'])] = "GenHepPartial"

    # Esophagectomy
    t.loc[p.isin([
        '43101', '43107', '43108', '43112', '43113', '43116', '43117', '43118', '43121', 
		'43122', '43123', '43124', '43286', '43287', '43288'
    ])] = "GenEso"

    # Vascular
    t.loc[p.isin(['34830', '34831', '34832','35081', '35082', '35091', '35092', '35102', '35103'])] = "VascAAA"
    t.loc[p.isin(['34701', '34702', '34703', '34704', '34705', '34706', '34710'])] = "VascEVAR"

    t.loc[p.isin([
        '35131','35132','35331','35351','35355','35361','35363','35521','35533','35537','35538','35539',
		'35540','35558','35563','35565','35621','35623','35637','35638','35646','35647','35654','35661','35663','35665'
    ])] = "VascAIO"

    t.loc[p.isin([
        '34201', '34203', '35141', '35142', '35151', '35152', '35226', '35286', '35302', '35303', 
        '35304', '35305', '35371', '35372', '35556', '35566', '35570', '35571', '35583', '35585', 
        '35587', '35656', '35666', '35671', '35879', '35881', '35883', '35884'
    ])] = "VascLEO"

    # NSG Brain
    t.loc[p.isin(['61510', '61512', '61518', '61519', '61520', '61521', '61526', '61530', '61545', '61546'])] = "NSGBrain"

    # Urology
    t.loc[p.isin(['50220', '50225', '50230', '50234', '50236', '50240', '50543', '50545', '50546', '50548'])] = "URONeph"
    t.loc[p.isin(['51550', '51555', '51565', '51570', '51575', '51580', '51585', '51590', '51595', '51596', '51597'])] = "UROCys"

    # Thoracic lung resection
    t.loc[p.isin([
        '32440', '32442', '32445', '32480', '32482', '32484', '32486', '32488', '32491', '32503', '32504',
		'32505', '32506', '32507', '32663', '32666', '32667', '32668', '32669', '32670', '32671', '32672'
    ])] = "ThoLung"

    return t

def cptlin_5yr(df: pd.DataFrame, extra_keep=()) -> pd.DataFrame:
        
    df["sermorb"] = compute_sermorb(df)
    
    keep = KEEP_COLS + list(extra_keep)
    keep = [c.lower() for c in keep]
    df = df.loc[:, [c for c in keep if c in df.columns]].copy()
        
    df["scorepatos"] = compute_scorePATOS(df)

    df["asaclas_c"] = make_asaclas_c(df["asaclas"]) if "asaclas" in df else pd.NA

    df["opsdlos"] = compute_opsdlos(df)

    # SAS: if (opsdlos missing OR hdisdt missing) and scorePATOS=0 → 30
    missing_hdis = pd.to_datetime(df.get("hdisdt"), errors="coerce").isna() if "hdisdt" in df.columns else True
    df.loc[(df["opsdlos"].isna() | missing_hdis) & (df["scorepatos"] == 0), "opsdlos"] = 30

    df["t_procs"] = assign_T_procs(df) if "prncptx" in df.columns else pd.NA
    
    return df

sar_dfs = [cptlin_5yr(p) for p in sar4yr]
sar5 = cptlin_5yr(step3)

sar5yr = pd.concat(sar_dfs + [sar5], ignore_index=True)
sar5yr["delirium"].dtype
sar5yr["delirium"] = sar5yr["delirium"].astype("Int64")
sar5yr["delirium"].value_counts(dropna=False)
sar5yr["fhsdecline"].value_counts(dropna=False)
sar5yr["sermorb"].value_counts(dropna=False)

# SAS: proc sort nodupkey by Division asm1
sar5yr = sar5yr.sort_values(["division","asm1"]).drop_duplicates(["division","asm1"])
# sar5yr["opsdlos"] = sar5yr["opsdlos"].fillna(30)

sar5yr.to_parquet(data_folder / "sar5yr_for_cptlin.parquet")
sar5yr = pd.read_parquet(data_folder / "sar5yr_for_cptlin.parquet")

# =============================================================================
# Generate LOS75 outcome flags per procedure
# =============================================================================

los_summary = pd.read_parquet(data_folder / "tprocs_los75.parquet")
los_summary.columns = los_summary.columns.str.lower()
los_summary["los_var_name"] = los_summary["los_var_name"].str.lower()
los_summary["filter"] = los_summary["filter"].str.lower()

los_summary

_FILTER_RE_COLO = re.compile(r"^\s*where\s+colorectal\s*=\s*1\s+and\s+scorepatos\s*=\s*0\s*$", re.I)
_FILTER_RE_TPROCS = re.compile(
    r"^\s*where\s+upcase\s*\(\s*t_procs\s*\)\s*=\s*'([^']+)'\s+and\s+scorepatos\s*=\s*0\s*$",
    re.I
)

def apply_los75_rules(
    df: pd.DataFrame,
    tprocs_los75: pd.DataFrame,
    *,
    division_col: str = "division",
    asm1_col: str = "asm1",
    opsdlos_col: str = "opsdlos",
    tprocs_col: str = "t_procs",
    scorepatos_col: str = "scorepatos",
    colorectal_col: str = "colorectal",
    los_var_col: str = "los_var_name",
    cutoff_col: str = "cutoff75",
    filter_col: str = "filter",
    ) -> pd.DataFrame:
    """
    Create LOS75 outcome columns defined by tprocs_los75 table.
    
      - For each LOS_var_name:
          - initialize var for all rows (conceptually)
          - set to 1 if opsdlos > cutoff
          - apply filter so only qualifying rows retain values
      - Non-qualifying rows get missing (NaN).

    Returns a df with [Division, asm1] + all LOS_var_name columns.
    """

    # ---- normalize source df columns for robust matching ----
    d = df.copy()

    # handle possible casing differences by mapping canonical names if needed
    col_map = {c.lower(): c for c in d.columns}

    def colname(preferred: str) -> str:
        return col_map.get(preferred.lower(), preferred)

    division_col = colname(division_col)
    asm1_col = colname(asm1_col)
    opsdlos_col = colname(opsdlos_col)
    tprocs_col = colname(tprocs_col)
    scorepatos_col = colname(scorepatos_col)
    colorectal_col = colname(colorectal_col)

    # Ensure required columns exist
    required = [division_col, asm1_col, opsdlos_col, tprocs_col, colorectal_col, scorepatos_col]
    for rc in required:
        if rc not in d.columns:
            raise KeyError(f"apply_los75_rules: missing required column '{rc}'")

    # Uppercase T_procs once (SAS uses upcase)
    tprocs_up = (
        d[tprocs_col].astype("object").fillna("").astype(str).str.upper()
        if tprocs_col in d.columns else pd.Series("", index=d.index)
    )

    score0 = (pd.to_numeric(d[scorepatos_col], errors="coerce") == 0)
    ops = pd.to_numeric(d[opsdlos_col], errors="coerce")

    out = d[[division_col, asm1_col]].copy()

    # ---- loop through rules ----
    for _, r in tprocs_los75.iterrows():
        los_var = str(r[los_var_col]).strip()
        cutoff = float(r[cutoff_col])
        filt = str(r[filter_col]).strip()

        # Determine mask from Filter pattern
        m_colo = _FILTER_RE_COLO.match(filt)
        m_tprocs = _FILTER_RE_TPROCS.match(filt)

        if m_colo:
            if colorectal_col not in d.columns:
                raise KeyError(f"Filter requires '{colorectal_col}', but it is missing in df.")
            mask = (pd.to_numeric(d[colorectal_col], errors="coerce") == 1) & score0

        elif m_tprocs:
            target = m_tprocs.group(1).strip().upper()
            if tprocs_col not in d.columns:
                raise KeyError(f"Filter requires '{tprocs_col}', but it is missing in df.")
            mask = (tprocs_up == target) & score0

        else:
            raise ValueError(
                f"Unrecognized Filter format for {los_var}: '{filt}'. "
                "Extend parser if new patterns appear."
            )

        # Create series: missing unless mask; within mask, 1 if opsdlos > cutoff else 0
        vals = pd.Series(np.nan, index=d.index, dtype="float")
        vals.loc[mask] = (ops.loc[mask] > cutoff).astype("int8")

        out[los_var] = vals

    return out

# d = sar5yr.copy()
# tprocs_los75 = los_summary.copy()
# division_col = "division"
# asm1_col = "asm1"
# opsdlos_col = "opsdlos"
# tprocs_col = "t_procs"
# scorepatos_col = "scorepatos"
# colorectal_col = "colorectal"
# los_var_col = "los_var_name"
# cutoff_col = "cutoff75"
# filter_col = "filter"

# r = tprocs_los75.iloc[3,:]

los_cols = apply_los75_rules(
    sar5yr,
    los_summary,
    tprocs_col="t_procs",        # produced by assign_T_procs()
    scorepatos_col="scorepatos", # produced by compute_scorePATOS()
    )

sar5yr = sar5yr.merge(los_cols, on=["division", "asm1"], how="left")

# check
for v in los_summary["los_var_name"]:
    s = sar5yr[v]
    rate = s.mean(skipna=True)
    n = s.notna().sum()
    print(v, "n=", n, "event_rate=", rate)

sar5yr.to_parquet(data_folder / "sar5yr_for_cptlin.parquet")

# =============================================================================
# CPT linear risk random intercept by PRNCPTX for All Cases Morbidity
# come back after PyMC modeling
# =============================================================================

# The CPT linear risk portion will detour to a hierarchical PyMC 
# implementation in WSL for processing,
# conda activate nutbolt016
# jupyter lab . 

sar5yr = pd.read_parquet(data_folder / "sar5yr_for_cptlin.parquet")

# check the counts of the variables in Morbidity CPT linear risk
cols = ["agegroup_bucket", "asaclas", "fnstatpressurg"]
for col in cols:
    print(f"\nValue counts for {col}:")
    print(sar5yr[col].value_counts(dropna=False))

cpt_morb_risk = pd.read_excel(model_base / "Morbidity_pymc" / "Morbidity_pymc_random_intercept.xlsx")
cpt_morb_risk = cpt_morb_risk[["prncptx", "mean"]].copy()
cpt_morb_risk.rename(columns={"mean": "cpt_morb_risk"}, inplace=True)
cpt_morb_risk["prncptx"] = cpt_morb_risk["prncptx"].astype(str).str.strip()
cpt_morb_risk.dtypes

cpt_morb_risk.head()

sar5yr = pd.read_parquet(data_folder / "sar5yr_for_cptlin.parquet")
sar5yr.dtypes

sar5yr.shape
df = pd.merge(sar5yr, cpt_morb_risk, on="prncptx", how="inner")
df.head()

df.shape

rename_cpts = dict()

for i in range(1,11,1):
    rename_cpts[f"othcpt{i}"] = f"cpt{i}"
    rename_cpts[f"concpt{i}"] = f"cpt{i+10}"
    
df.rename(columns=rename_cpts, inplace=True)

df.columns

# first push the non-missing values from cpt1 - pt20 to smaller numbered cpt columns, push missing to the right
# then order cpt1 - cpt20 descening by cpt_risk_morb if the CPTs found in prncptx, followed by not found CPTs, 
# and the missing to the far right

# 1) Build CPT->risk lookup (if duplicate keys exist, keep max risk; change to mean/last if you want)
cpt_cols = [f"cpt{i}" for i in range(1,21)]
risk_map = df.groupby("prncptx")["cpt_morb_risk"].max()

# 2) Pull CPT matrix (object so None stays None)
cpts = df[cpt_cols].to_numpy(dtype=object, copy=True)
present = ~pd.isna(cpts)

# 3) Map each CPT cell to its lookup risk (NaN if not found)
risk_arr = np.column_stack([df[col].map(risk_map).to_numpy() for col in cpt_cols])

found = present & ~pd.isna(risk_arr)

# 4) Build per-cell sort score:
#    - found first, sorted by descending risk  (use -risk)
#    - then not-found, in original left-to-right order (use pos)
#    - then missing last
n, m = cpts.shape
pos = np.broadcast_to(np.arange(m), (n, m))

# choose a "big" constant safely larger than any risk
max_risk = np.nanmax(risk_arr) if np.isfinite(np.nanmax(risk_arr)) else 0.0
big = max_risk + 1.0

score = np.where(found, -risk_arr, big + pos)      # found by -risk; notfound by pos
score = np.where(present, score, big + m + pos)    # missing pushed to the end

order = np.argsort(score, axis=1, kind="mergesort")  # stable

df[cpt_cols] = np.take_along_axis(cpts, order, axis=1)

df.to_parquet(data_folder / "sar5yr_for_cptlin.parquet")

# assign string "Missing" to the missing value of cpt1 - cpt20
cpt_cols = [f"cpt{i}" for i in range(1,21)]
df = pd.read_parquet(data_folder / "sar5yr_for_cptlin.parquet")
df.head()
df[cpt_cols] = df[cpt_cols].fillna("Missing")

df.to_parquet(data_folder / "sar5yr_for_cptlin.parquet")

df = pd.read_parquet(data_folder / "sar5yr_for_cptlin.parquet")

# check outcomes count 
outcomes = ["postcode","score1b","compcard", "comppneu", "comptube",
            "compvent", "compdvt", "comprenal", "computi", "compssi",
             "compsepsis", "postopcdiff", "returnor","readmission",
             "dsermorb","ssi", "ssi2", "sermorb", "delirium", 
            "fhsdecline", "los75colo", "los75panwhipple", "los75pandistal",
            "los75gencole", "los75genproc", "los75hepmajor", "los75heppartial", 
            "los75geneso", "los75vascaaa", "los75vascevar", "los75vascaio", "los75vascleo", 
            "los75nsgbrain", "los75uroneph", "los75urocys", "los75tholung"            
            ]

for outcome in outcomes:
    if outcome in df.columns:
        if outcome.startswith("los") or outcome in ["delirium", "fhsdecline"]:
            df_clean = df[df[outcome].notna()]
        else:
            df_clean = df[df[outcome].isin([0,1])]
        # df_clean = df[df[outcome].isin([0,1])]
        print(df_clean[outcome].value_counts(dropna=False))
        print(df_clean[outcome].value_counts(normalize=True, dropna=False), end="\n\n")
    else:
        print(outcome, "not found in df")
       
df.loc[df["age"]>=65, "delirium"].value_counts(normalize=True, dropna=True)
df.loc[df["age"]>=65, "fhsdecline"].value_counts(normalize=True, dropna=True)

op_year = list(df["oprymd"].dt.year.unique())
cpt_cols = [f"cpt{i}" for i in range(1,21)]

for year in op_year:
    missing_rate = df.loc[df["oprymd"].dt.year == year, cpt_cols].eq("Missing").all(axis=1).mean()
    print(f"Year {year}: Missing rate for all cpt1 - cpt20 = {missing_rate*100: .2f}%")
