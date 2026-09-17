from functools import partial
from pathlib import Path

import pandas as pd

from multiple_cpt_risk_support import (
    ModelResults,
    fit_cpt_advi_and_score as _fit_cpt_advi_and_score,
    merge_cptlin_results,
)

data_folder = Path("/mnt/d/ISAR Multi-CPTs/Oct 2026/Data")
DEFAULT_MODEL_BASE = Path("/mnt/d/ISAR Multi-CPTs/Oct 2026/Multiple CPTs")
DEFAULT_CAT_COLS = ["prncptx"] + [f"cpt{i}" for i in range(1, 21)]

model_base = DEFAULT_MODEL_BASE
df_org = pd.read_parquet(data_folder / "sar5yr_for_cptlin.parquet")
cat_cols = list(DEFAULT_CAT_COLS)

# Bind shared script settings so the repeated model calls can stay concise.
fit_cpt_advi_and_score = partial(
    _fit_cpt_advi_and_score,
    cat_cols=cat_cols,
    model_base=model_base,
)

# =============================================================================
# All cases
# =============================================================================

#Mortality
outcome = "postcode"
df_model = df_org [["division", "asm1", outcome] + cat_cols].copy()
df_model.shape

fit_cpt_advi_and_score(
    df = df_model,
    model_name = "Mortality",
    outcome = outcome,
    cptlin="postcodelin",
    seed=123,
    # Full-batch settings (small N)
    n_full = 30_000,
    lr_full = 1e-3,
    # Minibatch settings (large N)
    n_mb = 30_000,
    batch_size = 50_000, 
    lr_mb = 2e-3,
    # Priors
    alpha_sigma = 1.0,
    sigma_beta = 0.2,
    # Posterior draws for saving flexibility
    draws = 8000,
)

#Morbidity
outcome = "score1b"
df_model = df_org [["division", "asm1", outcome] + cat_cols].copy()
df_model.shape

fit_cpt_advi_and_score(
    df = df_model,
    model_name = "Morbidity",
    outcome = outcome,
    cptlin="score1blin",
    seed=123,
    # Full-batch settings (small N)
    n_full = 30_000,
    lr_full = 1e-3,
    # Minibatch settings (large N)
    n_mb = 30_000,
    batch_size = 50_000, 
    lr_mb = 1e-3,
    # Priors
    alpha_sigma = 1.0,
    sigma_beta = 0.2,
    # Posterior draws for saving flexibility
    draws = 8000,
)

#Cardiac
outcome = "compcard"
df_model = df_org [["division", "asm1", outcome] + cat_cols].copy()
df_model.shape

fit_cpt_advi_and_score(
    df = df_model,
    model_name = "Cardiac",
    outcome = outcome,
    cptlin="cardlin",
    seed=123,
    # Full-batch settings (small N)
    n_full = 30_000,
    lr_full = 1e-3,
    # Minibatch settings (large N)
    n_mb = 30_000,
    batch_size = 50_000, 
    lr_mb = 1e-3,
    # Priors
    alpha_sigma = 0.5,
    sigma_beta = 0.2,
    # Posterior draws for saving flexibility
    draws = 8000,
)

#Pneumonia
outcome = "comppneu"
df_model = df_org [["division", "asm1", outcome] + cat_cols].copy()
df_model.shape

fit_cpt_advi_and_score(
    df = df_model,
    model_name = "Pneumonia",
    outcome = outcome,
    cptlin="pneulin",
    seed=123,
    # Full-batch settings (small N)
    n_full = 30_000,
    lr_full = 1e-3,
    # Minibatch settings (large N)
    n_mb = 30_000,
    batch_size = 50_000, 
    lr_mb = 1e-3,
    # Priors
    alpha_sigma = 0.5,
    sigma_beta = 0.2,
    # Posterior draws for saving flexibility
    draws = 8000,
)

#Intubation
outcome = "comptube"
df_model = df_org [["division", "asm1", outcome] + cat_cols].copy()
df_model.shape

fit_cpt_advi_and_score(
    df = df_model,
    model_name = "Intubation",
    outcome = outcome,
    cptlin="tubelin",
    seed=123,
    # Full-batch settings (small N)
    n_full = 30_000,
    lr_full = 1e-3,
    # Minibatch settings (large N)
    n_mb = 30_000,
    batch_size = 50_000, 
    lr_mb = 1e-3,
    # Priors
    alpha_sigma = 0.5,
    sigma_beta = 0.2,
    # Posterior draws for saving flexibility
    draws = 8000,
)

#Ventilator
outcome = "compvent"
df_model = df_org [["division", "asm1", outcome] + cat_cols].copy()
df_model.shape

fit_cpt_advi_and_score(
    df = df_model,
    model_name = "Ventilator",
    outcome = outcome,
    cptlin="ventlin",
    seed=123,
    # Full-batch settings (small N)
    n_full = 30_000,
    lr_full = 1e-3,
    # Minibatch settings (large N)
    n_mb = 30_000,
    batch_size = 50_000, 
    lr_mb = 1e-3,
    # Priors
    alpha_sigma = 0.5,
    sigma_beta = 0.2,
    # Posterior draws for saving flexibility
    draws = 8000,
)

#DVT
outcome = "compdvt"
df_model = df_org [["division", "asm1", outcome] + cat_cols].copy()
df_model.shape

fit_cpt_advi_and_score(
    df = df_model,
    model_name = "DVT",
    outcome = outcome,
    cptlin="dvtlin",
    seed=123,
    # Full-batch settings (small N)
    n_full = 30_000,
    lr_full = 1e-3,
    # Minibatch settings (large N)
    n_mb = 30_000,
    batch_size = 50_000, 
    lr_mb = 1e-3,
    # Priors
    alpha_sigma = 0.5,
    sigma_beta = 0.2,
    # Posterior draws for saving flexibility
    draws = 8000,
)

#Renal
outcome = "comprenal"
df_model = df_org [["division", "asm1", outcome] + cat_cols].copy()
df_model.shape

fit_cpt_advi_and_score(
    df = df_model,
    model_name = "Renal",
    outcome = outcome,
    cptlin="renallin",
    seed=123,
    # Full-batch settings (small N)
    n_full = 30_000,
    lr_full = 1e-3,
    # Minibatch settings (large N)
    n_mb = 30_000,
    batch_size = 50_000, 
    lr_mb = 1e-3,
    # Priors
    alpha_sigma = 0.5,
    sigma_beta = 0.2,
    # Posterior draws for saving flexibility
    draws = 8000,
)

#UTI
outcome = "computi"
df_model = df_org [["division", "asm1", outcome] + cat_cols].copy()
df_model.shape

fit_cpt_advi_and_score(
    df = df_model,
    model_name = "UTI",
    outcome = outcome,
    cptlin="utilin",
    seed=123,
    # Full-batch settings (small N)
    n_full = 30_000,
    lr_full = 1e-3,
    # Minibatch settings (large N)
    n_mb = 30_000,
    batch_size = 50_000, 
    lr_mb = 1e-3,
    # Priors
    alpha_sigma = 0.5,
    sigma_beta = 0.2,
    # Posterior draws for saving flexibility
    draws = 8000,
)

#SSI
outcome = "compssi"
df_model = df_org [["division", "asm1", outcome] + cat_cols].copy()
df_model.shape

fit_cpt_advi_and_score(
    df = df_model,
    model_name = "SSI",
    outcome = outcome,
    cptlin="ssilin",
    seed=123,
    # Full-batch settings (small N)
    n_full = 30_000,
    lr_full = 1e-3,
    # Minibatch settings (large N)
    n_mb = 30_000,
    batch_size = 50_000, 
    lr_mb = 1e-3,
    # Priors
    alpha_sigma = 0.5,
    sigma_beta = 0.2,
    # Posterior draws for saving flexibility
    draws = 8000,
)

#Sepsis
outcome = "compsepsis"
df_model = df_org [["division", "asm1", outcome] + cat_cols].copy()
df_model.shape

fit_cpt_advi_and_score(
    df = df_model,
    model_name = "Sepsis",
    outcome = outcome,
    cptlin="sepsislin",
    seed=123,
    # Full-batch settings (small N)
    n_full = 30_000,
    lr_full = 1e-3,
    # Minibatch settings (large N)
    n_mb = 30_000,
    batch_size = 50_000, 
    lr_mb = 1e-3,
    # Priors
    alpha_sigma = 0.5,
    sigma_beta = 0.2,
    # Posterior draws for saving flexibility
    draws = 8000,
)

#Cdiff
outcome = "postopcdiff"
df_model = df_org [["division", "asm1", outcome] + cat_cols].copy()
df_model.shape

fit_cpt_advi_and_score(
    df = df_model,
    model_name = "Cdiff",
    outcome = outcome,
    cptlin="cdifflin",
    seed=123,
    # Full-batch settings (small N)
    n_full = 30_000,
    lr_full = 1e-3,
    # Minibatch settings (large N)
    n_mb = 30_000,
    batch_size = 50_000, 
    lr_mb = 1e-3,
    # Priors
    alpha_sigma = 0.5,
    sigma_beta = 0.2,
    # Posterior draws for saving flexibility
    draws = 8000,
)

#ReturnOR
outcome = "returnor"
df_model = df_org [["division", "asm1", outcome] + cat_cols].copy()
df_model.shape

fit_cpt_advi_and_score(
    df = df_model,
    model_name = "ReturnOR",
    outcome = outcome,
    cptlin="returnorlin",
    seed=123,
    # Full-batch settings (small N)
    n_full = 30_000,
    lr_full = 1e-3,
    # Minibatch settings (large N)
    n_mb = 30_000,
    batch_size = 50_000, 
    lr_mb = 1e-3,
    # Priors
    alpha_sigma = 0.5,
    sigma_beta = 0.2,
    # Posterior draws for saving flexibility
    draws = 8000,
)

#Readmission
outcome = "readmission"
df_model = df_org [["division", "asm1", outcome] + cat_cols].copy()
df_model.shape

fit_cpt_advi_and_score(
    df = df_model,
    model_name = "Readmission",
    outcome = outcome,
    cptlin="readmissionlin",
    seed=123,
    # Full-batch settings (small N)
    n_full = 30_000,
    lr_full = 1e-3,
    # Minibatch settings (large N)
    n_mb = 30_000,
    batch_size = 50_000, 
    lr_mb = 1e-3,
    # Priors
    alpha_sigma = 0.5,
    sigma_beta = 0.2,
    # Posterior draws for saving flexibility
    draws = 8000,
)

# =============================================================================
# ### EGS sermorb models
# =============================================================================

# EGS sermorb
outcome = "sermorb"
df_model = df_org [["division", "asm1", outcome] + cat_cols].copy()
df_model.shape

fit_cpt_advi_and_score(
    df = df_model,
    model_name = "EGS_sermorb",
    outcome = outcome,
    cptlin="egs_sermorblin",
    seed=123,
    # Full-batch settings (small N)
    n_full = 30_000,
    lr_full = 1e-3,
    # Minibatch settings (large N)
    n_mb = 30_000,
    batch_size = 50_000, 
    lr_mb = 1e-3,
    # Priors
    alpha_sigma = 0.5,
    sigma_beta = 0.2,
    # Posterior draws for saving flexibility
    draws = 8000,
)

# =============================================================================
# ### Measure models
# =============================================================================

#DSMorb
outcome = "dsermorb"
df_model = df_org [["division", "asm1", outcome] + cat_cols].copy()
df_model.shape

fit_cpt_advi_and_score(
    df = df_model,
    model_name = "DSMorb",
    outcome = outcome,
    cptlin="dsermorblin",
    seed=123,
    # Full-batch settings (small N)
    n_full = 30_000,
    lr_full = 1e-3,
    # Minibatch settings (large N)
    n_mb = 30_000,
    batch_size = 50_000, 
    lr_mb = 1e-3,
    # Priors
    alpha_sigma = 0.5,
    sigma_beta = 0.2,
    # Posterior draws for saving flexibility
    draws = 8000,
)

#ElderDSMorb
outcome = "dsermorb"
df_model = df_org.loc[df_org['agegroup'].str.strip().ne('1-<65')].copy()
df_model = df_model[["division", "asm1", outcome] + cat_cols].copy()
df_model.shape

fit_cpt_advi_and_score(
    df = df_model,
    model_name = "ElderDSMorb",
    outcome = outcome,
    cptlin="seniorlin",
    seed=123,
    # Full-batch settings (small N)
    n_full = 30_000,
    lr_full = 1e-3,
    # Minibatch settings (large N)
    n_mb = 30_000,
    batch_size = 50_000, 
    lr_mb = 1e-3,
    # Priors
    alpha_sigma = 0.5,
    sigma_beta = 0.2,
    # Posterior draws for saving flexibility
    draws = 8000,
)

#MSRSSI
outcome = "ssi"
df_model = df_org[["division", "asm1", outcome] + cat_cols].copy()
df_model.shape

fit_cpt_advi_and_score(
    df = df_model,
    model_name = "MSRSSI",
    outcome = outcome,
    cptlin="ssimsrlin",
    seed=123,
    # Full-batch settings (small N)
    n_full = 30_000,
    lr_full = 1e-3,
    # Minibatch settings (large N)
    n_mb = 30_000,
    batch_size = 50_000, 
    lr_mb = 1e-3,
    # Priors
    alpha_sigma = 0.5,
    sigma_beta = 0.2,
    # Posterior draws for saving flexibility
    draws = 8000,
)

#Colon DSM. Not for the SAR modeling
outcome = "dsermorb"
df_model = df_org.loc[df_org['colon'] == 1].copy()
df_model = df_model[["division", "asm1", outcome] + cat_cols].copy()
df_model.shape

fit_cpt_advi_and_score(
    df = df_model,
    model_name = "ColonDSM",
    outcome = outcome,
    cptlin="colondsmlin",
    seed=123,
    # Full-batch settings (small N)
    n_full = 30_000,
    lr_full = 1e-3,
    # Minibatch settings (large N)
    n_mb = 30_000,
    batch_size = 50_000, 
    lr_mb = 1e-3,
    # Priors
    alpha_sigma = 0.5,
    sigma_beta = 0.2,
    # Posterior draws for saving flexibility
    draws = 8000,
)

#Colon SSI. Not for the SAR modeling
outcome = "ssi2"
df_model = df_org.loc[(df_org['colon'] == 1) & (df_org["ssi2"].isin([0,1]))].copy()
df_model = df_model[["division", "asm1", outcome] + cat_cols].copy()
df_model.shape

fit_cpt_advi_and_score(
    df = df_model,
    model_name = "ColonSSI",
    outcome = outcome,
    cptlin="colonssilin",
    seed=123,
    # Full-batch settings (small N)
    n_full = 30_000,
    lr_full = 1e-3,
    # Minibatch settings (large N)
    n_mb = 30_000,
    batch_size = 50_000, 
    lr_mb = 1e-3,
    # Priors
    alpha_sigma = 0.5,
    sigma_beta = 0.2,
    # Posterior draws for saving flexibility
    draws = 8000,
)

# =============================================================================
# LOS
# =============================================================================

#LOS: colorectal
outcome = "los75colo"
df_model = df_org.loc[(df_org['colorectal']==1) & (df_org["scorepatos"]==0) & 
                      (~df_org[outcome].isna())].copy()
df_model = df_model[["division", "asm1", outcome] + cat_cols].copy()
df_model.shape

fit_cpt_advi_and_score(
    df = df_model,
    model_name = "LOS75Colo",
    outcome = outcome,
    cptlin="los75cololin",
    seed=123,
    large_data_threshold = 300_000,
    # Full-batch settings (small N)
    n_full = 30_000,
    lr_full = 1e-3,
    # Minibatch settings (large N)
    n_mb = 30_000,
    batch_size = 50_000, 
    lr_mb = 1e-3,
    # Priors
    alpha_sigma = 0.5,
    sigma_beta = 0.2,
    # Posterior draws for saving flexibility
    draws = 8000,
)

#LOS: Gen Pan Whipple LOS
outcome = "los75panwhipple"
df_model = df_org.loc[(df_org['t_procs'].str.upper()=="GENPANWHIPPLE") & (df_org["scorepatos"]==0) & 
                      (~df_org[outcome].isna())].copy()
df_model = df_model[["division", "asm1", outcome] + cat_cols].copy()
df_model.shape

fit_cpt_advi_and_score(
    df = df_model,
    model_name = "LOS75PanWhipple",
    outcome = outcome,
    cptlin="los75panwhipplelin",
    seed=123,
    # Full-batch settings (small N)
    large_data_threshold=50_000,
    n_full = 30_000,
    lr_full = 5e-3,
    # Minibatch settings (large N)
    n_mb = 30_000,
    batch_size = 50_000, 
    lr_mb = 1e-3,
    # Priors
    alpha_sigma = 0.5,
    sigma_beta = 0.25,
    # Posterior draws for saving flexibility
    draws = 8000,
)

#LOS: Gen Pan Distal LOS
outcome = "los75pandistal"
df_model = df_org.loc[(df_org['t_procs'].str.upper()=="GENPANDISTAL") & (df_org["scorepatos"]==0) & 
                      (~df_org[outcome].isna())].copy()
df_model = df_model[["division", "asm1", outcome] + cat_cols].copy()
df_model.shape

fit_cpt_advi_and_score(
    df = df_model,
    model_name = "LOS75PanDistal",
    outcome = outcome,
    cptlin="los75pandistallin",
    seed=123,
    # Full-batch settings (small N)
    n_full = 30_000,
    lr_full = 5e-3,
    # Minibatch settings (large N)
    n_mb = 30_000,
    batch_size = 50_000, 
    lr_mb = 1e-3,
    # Priors
    alpha_sigma = 0.5,
    sigma_beta = 0.2,
    # Posterior draws for saving flexibility
    draws = 8000,
)

#LOS: Gen Colectomy LOS
outcome = "los75gencole"
df_model = df_org.loc[(df_org['t_procs'].str.upper()=="GENCOLON") & (df_org["scorepatos"]==0) & 
                      (~df_org[outcome].isna())].copy()
df_model = df_model[["division", "asm1", outcome] + cat_cols].copy()
df_model.shape

fit_cpt_advi_and_score(
    df = df_model,
    model_name = "LOS75GenCole",
    outcome = outcome,
    cptlin="los75gencolelin",
    seed=123,
    # Full-batch settings (small N)
    n_full = 30_000,
    lr_full = 1e-3,
    # Minibatch settings (large N)
    n_mb = 30_000,
    batch_size = 50_000, 
    lr_mb = 1e-3,
    # Priors
    alpha_sigma = 0.5,
    sigma_beta = 0.2,
    # Posterior draws for saving flexibility
    draws = 8000,
)

#LOS: Gen Proctectomy LOS
outcome = "los75genproc"
df_model = df_org.loc[(df_org['t_procs'].str.upper()=="GENPROCTECTOMY") & (df_org["scorepatos"]==0) & 
                      (~df_org[outcome].isna())].copy()
df_model = df_model[["division", "asm1", outcome] + cat_cols].copy()
df_model.shape

fit_cpt_advi_and_score(
    df = df_model,
    model_name = "LOS75GenProc",
    outcome = outcome,
    cptlin="los75genproclin",
    seed=123,
    # Full-batch settings (small N)
    n_full = 30_000,
    lr_full = 5e-3,
    # Minibatch settings (large N)
    n_mb = 30_000,
    batch_size = 50_000, 
    lr_mb = 1e-3,
    # Priors
    alpha_sigma = 0.5,
    sigma_beta = 0.2,
    # Posterior draws for saving flexibility
    draws = 8000,
)

#LOS: Gen Hep Major LOS
outcome = "los75hepmajor"
df_model = df_org.loc[(df_org['t_procs'].str.upper()=="GENHEPMAJOR") & (df_org["scorepatos"]==0) & 
                      (~df_org[outcome].isna())].copy()
df_model = df_model[["division", "asm1", outcome] + cat_cols].copy()
df_model.shape

fit_cpt_advi_and_score(
    df = df_model,
    model_name = "LOS75GenHepMajor",
    outcome = outcome,
    cptlin="los75hepmajorlin",
    seed=123,
    # Full-batch settings (small N)
    n_full = 30_000,
    lr_full = 5e-3,
    # Minibatch settings (large N)
    n_mb = 30_000,
    batch_size = 50_000, 
    lr_mb = 1e-3,
    # Priors
    alpha_sigma = 0.5,
    sigma_beta = 0.2,
    # Posterior draws for saving flexibility
    draws = 8000,
)

#LOS: Gen Hep Partial LOS
outcome = "los75heppartial"
df_model = df_org.loc[(df_org['t_procs'].str.upper()=="GENHEPPARTIAL") & (df_org["scorepatos"]==0) & 
                      (~df_org[outcome].isna())].copy()
df_model = df_model[["division", "asm1", outcome] + cat_cols].copy()
df_model.shape

fit_cpt_advi_and_score(
    df = df_model,
    model_name = "LOS75GenHepPartial",
    outcome = outcome,
    cptlin="los75heppartiallin",
    seed=123,
    # Full-batch settings (small N)
    n_full = 30_000,
    lr_full = 5e-3,
    # Minibatch settings (large N)
    n_mb = 30_000,
    batch_size = 50_000, 
    lr_mb = 1e-3,
    # Priors
    alpha_sigma = 0.5,
    sigma_beta = 0.2,
    # Posterior draws for saving flexibility
    draws = 8000,
)

#LOS: Gen Eso LOS
outcome = "los75geneso"
df_model = df_org.loc[(df_org['t_procs'].str.upper()=="GENESO") & (df_org["scorepatos"]==0) & 
                      (~df_org[outcome].isna())].copy()
df_model = df_model[["division", "asm1", outcome] + cat_cols].copy()
df_model.shape

fit_cpt_advi_and_score(
    df = df_model,
    model_name = "LOS75GenEso",
    outcome = outcome,
    cptlin="los75genesolin",
    seed=123,
    # Full-batch settings (small N)
    n_full = 30_000,
    lr_full = 5e-3,
    # Minibatch settings (large N)
    n_mb = 30_000,
    batch_size = 50_000, 
    lr_mb = 1e-3,
    # Priors
    alpha_sigma = 0.5,
    sigma_beta = 0.2,
    # Posterior draws for saving flexibility
    draws = 8000,
)

#LOS: Vasc AAA LOS
outcome = "los75vascaaa"
df_model = df_org.loc[(df_org['t_procs'].str.upper()=="VASCAAA") & (df_org["scorepatos"]==0) & 
                      (~df_org[outcome].isna())].copy()
df_model = df_model[["division", "asm1", outcome] + cat_cols].copy()
df_model.shape

fit_cpt_advi_and_score(
    df = df_model,
    model_name = "LOS75VascAAA",
    outcome = outcome,
    cptlin="los75vascaaalin",
    seed=123,
    # Full-batch settings (small N)
    n_full = 30_000,
    lr_full = 5e-3,
    # Minibatch settings (large N)
    n_mb = 30_000,
    batch_size = 50_000, 
    lr_mb = 1e-3,
    # Priors
    alpha_sigma = 0.5,
    sigma_beta = 0.2,
    # Posterior draws for saving flexibility
    draws = 8000,
)

#LOS: Vasc EVAR LOS
outcome = "los75vascevar"
df_model = df_org.loc[(df_org['t_procs'].str.upper()=="VASCEVAR") & (df_org["scorepatos"]==0) & 
                      (~df_org[outcome].isna())].copy()
df_model = df_model[["division", "asm1", outcome] + cat_cols].copy()
df_model.shape

fit_cpt_advi_and_score(
    df = df_model,
    model_name = "LOS75VascEVAR",
    outcome = outcome,
    cptlin="los75vascevarlin",
    seed=123,
    # Full-batch settings (small N)
    n_full = 30_000,
    lr_full = 5e-3,
    # Minibatch settings (large N)
    n_mb = 30_000,
    batch_size = 50_000, 
    lr_mb = 1e-3,
    # Priors
    alpha_sigma = 0.5,
    sigma_beta = 0.2,
    # Posterior draws for saving flexibility
    draws = 8000,
)

#LOS: Vasc AIO LOS
outcome = "los75vascaio"
df_model = df_org.loc[(df_org['t_procs'].str.upper()=="VASCAIO") & (df_org["scorepatos"]==0) & 
                      (~df_org[outcome].isna())].copy()
df_model = df_model[["division", "asm1", outcome] + cat_cols].copy()
df_model.shape

fit_cpt_advi_and_score(
    df = df_model,
    model_name = "LOS75VascAIO",
    outcome = outcome,
    cptlin="los75vascaiolin",
    seed=123,
    # Full-batch settings (small N)
    n_full = 30_000,
    lr_full = 5e-3,
    # Minibatch settings (large N)
    n_mb = 30_000,
    batch_size = 50_000, 
    lr_mb = 1e-3,
    # Priors
    alpha_sigma = 0.5,
    sigma_beta = 0.2,
    # Posterior draws for saving flexibility
    draws = 8000,
)

#LOS: Vasc LEO LOS
outcome = "los75vascleo"
df_model = df_org.loc[(df_org['t_procs'].str.upper()=="VASCLEO") & (df_org["scorepatos"]==0) & 
                      (~df_org[outcome].isna())].copy()
df_model = df_model[["division", "asm1", outcome] + cat_cols].copy()
df_model.shape

fit_cpt_advi_and_score(
    df = df_model,
    model_name = "LOS75VascLEO",
    outcome = outcome,
    cptlin="los75vascleolin",
    seed=123,
    # Full-batch settings (small N)
    n_full = 30_000,
    lr_full = 5e-3,
    # Minibatch settings (large N)
    n_mb = 30_000,
    batch_size = 50_000, 
    lr_mb = 1e-3,
    # Priors
    alpha_sigma = 0.5,
    sigma_beta = 0.2,
    # Posterior draws for saving flexibility
    draws = 8000,
)

#LOS: NSG Brain LOS
outcome = "los75nsgbrain"
df_model = df_org.loc[(df_org['t_procs'].str.upper()=="NSGBRAIN") & (df_org["scorepatos"]==0) & 
                      (~df_org[outcome].isna())].copy()
df_model = df_model[["division", "asm1", outcome] + cat_cols].copy()
df_model.shape

fit_cpt_advi_and_score(
    df = df_model,
    model_name = "LOS75NSGBrain",
    outcome = outcome,
    cptlin="los75nsgbrainlin",
    seed=123,
    # Full-batch settings (small N)
    n_full = 30_000,
    lr_full = 5e-3,
    # Minibatch settings (large N)
    n_mb = 30_000,
    batch_size = 50_000, 
    lr_mb = 1e-3,
    # Priors
    alpha_sigma = 0.5,
    sigma_beta = 0.2,
    # Posterior draws for saving flexibility
    draws = 8000,
)

#LOS: Uro Neph LOS
outcome = "los75uroneph"
df_model = df_org.loc[(df_org['t_procs'].str.upper()=="URONEPH") & (df_org["scorepatos"]==0) & 
                      (~df_org[outcome].isna())].copy()
df_model = df_model[["division", "asm1", outcome] + cat_cols].copy()
df_model.shape

fit_cpt_advi_and_score(
    df = df_model,
    model_name = "LOS75UroNeph",
    outcome = outcome,
    cptlin="los75uronephlin",
    seed=123,
    # Full-batch settings (small N)
    n_full = 30_000,
    lr_full = 5e-3,
    # Minibatch settings (large N)
    n_mb = 30_000,
    batch_size = 50_000, 
    lr_mb = 1e-3,
    # Priors
    alpha_sigma = 0.5,
    sigma_beta = 0.2,
    # Posterior draws for saving flexibility
    draws = 8000,
)

#LOS: URO Cyst LOS
outcome = "los75urocys"
df_model = df_org.loc[(df_org['t_procs'].str.upper()=="UROCYS") & (df_org["scorepatos"]==0) & 
                      (~df_org[outcome].isna())].copy()
df_model = df_model[["division", "asm1", outcome] + cat_cols].copy()
df_model.shape

fit_cpt_advi_and_score(
    df = df_model,
    model_name = "LOS75UroCys",
    outcome = outcome,
    cptlin="los75urocyslin",
    seed=123,
    # Full-batch settings (small N)
    n_full = 30_000,
    lr_full = 5e-3,
    # Minibatch settings (large N)
    n_mb = 30_000,
    batch_size = 50_000, 
    lr_mb = 1e-3,
    # Priors
    alpha_sigma = 0.5,
    sigma_beta = 0.2,
    # Posterior draws for saving flexibility
    draws = 8000,
)

#LOS: Tho Lung LOS
outcome = "los75tholung"
df_model = df_org.loc[(df_org['t_procs'].str.upper()=="THOLUNG") & (df_org["scorepatos"]==0) & 
                      (~df_org[outcome].isna())].copy()
df_model = df_model[["division", "asm1", outcome] + cat_cols].copy()
df_model.shape

fit_cpt_advi_and_score(
    df = df_model,
    model_name = "LOS75ThoLung",
    outcome = outcome,
    cptlin="los75tholunglin",
    seed=123,
    # Full-batch settings (small N)
    n_full = 30_000,
    lr_full = 5e-3,
    # Minibatch settings (large N)
    n_mb = 30_000,
    batch_size = 50_000, 
    lr_mb = 1e-3,
    # Priors
    alpha_sigma = 0.5,
    sigma_beta = 0.2,
    # Posterior draws for saving flexibility
    draws = 8000,
)

# # =============================================================================
# Geriatric delirium and fhsdecline CPT linear risk 
# =============================================================================
# delirium
outcome = "delirium"
df_model = df_org.loc[(df_org['age']>=65) & (~df_org[outcome].isna())].copy()
df_model = df_model[["division", "asm1", outcome] + cat_cols].copy()
df_model.shape

fit_cpt_advi_and_score(
    df = df_model,
    model_name = "Delirium",
    outcome = outcome,
    cptlin="deliriumlin",
    seed=123,
    # Full-batch settings (small N)
    n_full = 30_000,
    lr_full = 5e-3,
    # Minibatch settings (large N)
    n_mb = 30_000,
    batch_size = 50_000, 
    lr_mb = 1e-3,
    # Priors
    alpha_sigma = 0.5,
    sigma_beta = 0.2,
    # Posterior draws for saving flexibility
    draws = 8000,
)

# fhsdecline
outcome = "fhsdecline"
df_model = df_org.loc[(df_org['age']>=65) & (~df_org[outcome].isna())].copy()
df_model = df_model[["division", "asm1", outcome] + cat_cols].copy()
df_model.shape

fit_cpt_advi_and_score(
    df = df_model,
    model_name = "FhsDecline",
    outcome = outcome,
    cptlin="fhsdeclinelin",
    seed=123,
    # Full-batch settings (small N)
    n_full = 30_000,
    lr_full = 5e-3,
    # Minibatch settings (large N)
    n_mb = 30_000,
    batch_size = 50_000, 
    lr_mb = 1e-3,
    # Priors
    alpha_sigma = 0.5,
    sigma_beta = 0.2,
    # Posterior draws for saving flexibility
    draws = 8000,
)

# =============================================================================
# Merge CPTs linear risk together for all outcomes
# =============================================================================

cptlin_models = [
    ModelResults("Mortality", "postcode", "postcodelin"),
    ModelResults("Morbidity", "score1b", "score1blin"),
    ModelResults("Cardiac", "compcard", "cardlin"),
    ModelResults("Pneumonia", "comppneu", "pneulin"),
    ModelResults("Intubation", "comptube", "tubelin"),
    ModelResults("Ventilator", "compvent", "ventlin"),
    ModelResults("DVT", "compdvt", "dvtlin"),
    ModelResults("Renal", "comprenal", "renallin"),
    ModelResults("UTI", "computi", "utilin"),
    ModelResults("SSI", "compssi", "ssilin"),
    ModelResults("Sepsis", "compsepsis", "sepsislin"),
    ModelResults("Cdiff", "postopcdiff", "cdifflin"),
    ModelResults("ReturnOR", "returnor", "returnorlin"),
    ModelResults("Readmission", "readmission", "readmissionlin"),
    ModelResults("DSMorb", "dsermorb", "dsermorblin"),
    ModelResults("ElderDSMorb", "dsermorb", "seniorlin"),
    ModelResults("MSRSSI", "ssi", "ssimsrlin"),
    ModelResults("ColonDSM", "dsermorb", "colondsmlin"),
    ModelResults("ColonSSI", "ssi2", "colonssilin"),
    ModelResults("LOS75Colo", "los75colo", "los75cololin"),
    ModelResults("LOS75PanWhipple", "los75panwhipple", "los75panwhipplelin"),
    ModelResults("LOS75PanDistal", "los75pandistal", "los75pandistallin"),
    ModelResults("LOS75GenCole", "los75gencole", "los75gencolelin"),
    ModelResults("LOS75GenProc", "los75genproc", "los75genproclin"),
    ModelResults("LOS75GenHepMajor", "los75hepmajor", "los75hepmajorlin"),
    ModelResults("LOS75GenHepPartial", "los75heppartial", "los75heppartiallin"),
    ModelResults("LOS75GenEso", "los75geneso", "los75genesolin"),
    ModelResults("LOS75VascAAA", "los75vascaaa", "los75vascaaalin"),
    ModelResults("LOS75VascEVAR", "los75vascevar", "los75vascevarlin"),
    ModelResults("LOS75VascAIO", "los75vascaio", "los75vascaiolin"),
    ModelResults("LOS75VascLEO", "los75vascleo", "los75vascleolin"),
    ModelResults("LOS75NSGBrain", "los75nsgbrain", "los75nsgbrainlin"),
    ModelResults("LOS75UroNeph", "los75uroneph", "los75uronephlin"),
    ModelResults("LOS75UroCys", "los75urocys", "los75urocyslin"),
    ModelResults("LOS75ThoLung", "los75tholung", "los75tholunglin"),
    ModelResults("EGS_sermorb", "sermorb", "egs_sermorblin"),
    ModelResults("Delirium", "delirium", "deliriumlin"),
    ModelResults("FhsDecline", "fhsdecline", "fhsdeclinelin"),
]

merged_cptlin_dfs, missing_cptlin_records = merge_cptlin_results(
    cptlin_models,
    df_org,
    model_base,
    output_path=model_base / "Essential_cptlin.parquet",
)

# =============================================================================
# # Colon DSM and SSI models not for the SAR modeling.
# =============================================================================
# cptlin_colon_models = [
#     ModelResults("ColonDSM", "dsermorb", "colondsmlin"),
#     ModelResults("ColonSSI", "ssi2", "colonssilin"),
# ]

# merged_colon_cptlin_dfs, missing_cptlin_records = merge_cptlin_results(
#     cptlin_colon_models,
#     df_org,
#     model_base,
#     output_path=model_base / "MSR_Colon_cptlin.parquet",
# )