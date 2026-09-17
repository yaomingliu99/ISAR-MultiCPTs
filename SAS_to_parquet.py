"""
To save space, convert SAS datasets to parquet files. 
This script will convert all SAS datasets in the input directory 
to parquet files in the output directory.
"""

from __future__ import annotations

import pandas as pd
import numpy as np
from pathlib import Path

pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)
pd.set_option("display.max_colwidth", None)
pd.set_option("display.max_seq_items", None)

data_folder = Path("/mnt/d/ISAR Multi-CPTs/Oct 2026/Data")

sas_files = [f.stem for f in data_folder.iterdir() if f.is_file() and f.suffix == ".sas7bdat"]

for f in sas_files:
    print(f"Converting {f} to parquet...")
    df = pd.read_sas(data_folder / f, format="sas7bdat", encoding="latin1")
    df.columns = df.columns.str.strip().str.lower()
    df.to_parquet(data_folder / f"{f}.parquet", engine="pyarrow", index=False)
    
oct_2022 = pd.read_parquet(data_folder / "isar_2022oct_final.parquet")
oct_2023 = pd.read_parquet(data_folder / "isar_2023oct_final.parquet")
oct_2024 = pd.read_parquet(data_folder / "sar_2024oct_final.parquet")
oct_2025 = pd.read_parquet(data_folder / "sar_2025oct_final.parquet")
step3 = pd.read_parquet(data_folder / "step3.parquet")

# IT's data cut has all other and concurrent CPTs missingbetween Jan 1, 2024 - Jun 30, 2025.

# 'isar_2022oct_final': surgery date range: Apr 1, 2021 - Mar 31, 2022
# 'isar_2023oct_final': surgery date range: Apr 1, 2022 - Mar 31, 2023
# 'sar_2024oct_final': surgery date range: Apr 1, 2023 - Mar 31, 2024
# 'isar_2025oct_final': surgery date range: Apr 1, 2024 - Mar 31, 2025
# 'step3': surgery date range: Apr 1, 2025 - Mar 31, 2026

# Check the missingness of concpt1-concpt10, othcpt1-othcpt10

concpts = [f"concpt{i}" for i in range(1,11)]
othcpts = [f"othcpt{i}" for i in range(1,11)]

step3.loc[step3["oprymd"]< pd.Timestamp("2025-07-01"),concpts + othcpts].isna().sum()
# this oct 2026 ISAR data cut is fine.

oct_2025.shape
oct_2025[concpts + othcpts].isna().sum()
# oct_2025 has all 20 other concurrent CPT missing, need patch

oct_2024.loc[oct_2024["oprymd"]>= pd.Timestamp("2024-01-01")].shape
oct_2024.loc[oct_2024["oprymd"]>= pd.Timestamp("2024-01-01"),concpts + othcpts].isna().sum()
# Oct 2024 ISAR data cut is okay.

# need load the data between Apr 1, 2024 - Mar 31, 2025
# Dig out the data from SAR for the period between 4/1/2024 - 3/1/2025

sar_jul_2025 = pd.read_sas("/mnt/d/Universal Risk Cal and RTR/SAR datasets and Codes/SAR2024/sar_2025jul_final.sas7bdat", encoding="latin-1")
sar_jul_2026 = pd.read_sas("/mnt/d/Universal Risk Cal and RTR/SAR datasets and Codes/SAR2025/sar_2026jul_final.sas7bdat", encoding="latin-1")

sar_jul_2025.columns = sar_jul_2025.columns.str.strip().str.lower()
sar_jul_2026.columns = sar_jul_2026.columns.str.strip().str.lower()

sar_jul_2025 = sar_jul_2025[["division", "asm1", "oprymd"] + othcpts + concpts].copy()
sar_jul_2026 = sar_jul_2026[["division", "asm1", "oprymd"] + othcpts + concpts].copy()

patching_cpts = pd.concat([sar_jul_2025.loc[sar_jul_2025['oprymd']>= pd.Timestamp("2024-04-01")],
                           sar_jul_2026.loc[sar_jul_2026['oprymd']<= pd.Timestamp("2025-03-31")]
                           ], ignore_index=True)

patching_cpts[othcpts + concpts].isna().sum()

oct_2025[["division", "asm1", "oprymd"]].head()

oct_2025.drop(columns=othcpts + concpts, inplace=True)

n_before = oct_2025.shape[0]
oct_2025_patched = pd.merge(oct_2025, 
                            patching_cpts[["division", "asm1"] + othcpts + concpts],
                            on=["division", "asm1"], how="left")
n_after = oct_2025_patched.shape[0]

print(f"N cases: before {n_before}, after {n_after}")

oct_2025_patched.to_parquet(data_folder / "sar_2025oct_final.parquet")