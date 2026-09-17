from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc as pm
from pymc.variational.callbacks import CheckParametersConvergence
import pytensor.tensor as pt
from nutbolt import sar_helpers
from scipy.special import expit, logit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score


@dataclass
class ModelResults:
    model_name: str
    outcome: str
    cptlin: str

def merge_cptlin_results(
    cptlin_models: list[ModelResults],
    df: pd.DataFrame,
    model_base: str | Path,
    *,
    vi_method: str = "advi",
    key_cols: tuple[str, str] = ("division", "asm1"),
    merge_how: str = "outer",
    output_path: str | Path | None = None,
    print_summary: bool = True,
) -> pd.DataFrame | None:
    """
    Read row-level CPT linear risk files and merge them into one wide dataframe.

    Parameters
    ----------
    cptlin_models : list[ModelResults]
        Model folder names and CPT linear risk column names to merge.
    df: pd.DataFrame
        Original data with dependent variables
    model_base : str or Path
        Base directory containing one subdirectory per model.
    vi_method : str, default "advi"
        Suffix used in files named "{model_name}_cptlin_full_{vi_method}.csv".
    key_cols : tuple[str, str], default ("division", "asm1")
        Key columns used to merge model output files.
    merge_how : str, default "outer"
        Merge type passed to pandas merge.
    output_path : str or Path, optional
        If provided, save the merged dataframe to this CSV path.
    print_summary : bool, default True
        Print a formatted describe table for the CPT linear risk columns.

    Returns
    -------
    pd.DataFrame
        Wide dataframe with key columns plus one CPT linear risk column per model.
    """
    
    df_data = df.copy()
        
    cptlin_models = list(cptlin_models)
    if not cptlin_models:
        raise ValueError("cptlin_models must contain at least one ModelResults item.")

    model_base = Path(model_base)
    key_cols = tuple(key_cols)
    key_col_list = list(key_cols)
    
    missing_key_cols = [col for col in key_col_list if col not in df_data.columns]
    if missing_key_cols:
        raise ValueError(f"df is missing key columns: {missing_key_cols}")
    
    # Normalize key column types in original data
    if "division" in df_data.columns:
        df_data["division"] = pd.to_numeric(df_data["division"], errors="raise").astype(int)

    merged = None
    
    valid_cptlin_models = []
    missing_record_dfs = []

    for run in cptlin_models:
        print(f"\n... Processing Model: {run.model_name}")
        
        if run.outcome not in df_data.columns:
            raise ValueError(
                f"df is missing outcome column '{run.outcome}' for model '{run.model_name}'."
            )

        cptlin_path = (
            model_base
            / run.model_name
            / f"{run.model_name}_cptlin_full_{vi_method}.csv"
        )
        
        if not cptlin_path.is_file():
            print(f"Skipping missing CPT linear risk file: {cptlin_path}")
            continue
       
        df_temp = pd.read_csv(cptlin_path)

        keep_cols = key_col_list + [run.cptlin]
        missing_cols = [col for col in keep_cols if col not in df_temp.columns]
        if missing_cols:
            raise ValueError(
                f"{cptlin_path} is missing required columns: {missing_cols}"
            )
            
        # Normalize key column types in CPT output
        if "division" in df_temp.columns:
            df_temp["division"] = pd.to_numeric(
                df_temp["division"], errors="raise"
            ).astype(int)
            
         # ---------------------------------------------------------------------
        # Check records where outcome is valid but CPT linear risk is missing
        # ---------------------------------------------------------------------
        
        if run.cptlin in ["seniorlin", "deliriumlin", "fhsdeclinelin"]:
            df_senior = df_data[df_data["age"]>=65].copy()
            check_df = df_senior[key_col_list + [run.outcome]].merge(
                df_temp[keep_cols],
                on=key_col_list,
                how="left",
                validate="many_to_one",
            )
        else:
            check_df = df_data[key_col_list + [run.outcome]].merge(
                df_temp[keep_cols],
                on=key_col_list,
                how="left",
                validate="many_to_one",
            )

        missing_mask = (
            check_df[run.outcome].isin([0, 1])
            & check_df[run.cptlin].isna()
        )

        missing_count = int(missing_mask.sum())

        missing_record_dfs.append(
            {
                "model_name": run.model_name,
                "outcome": run.outcome,
                "cptlin": run.cptlin,
                "Missing_count": missing_count,
            }
        )

        print(
            f"    Missing {run.cptlin} where {run.outcome} in (0, 1): "
            f"{missing_count:,}"
        )
         
        # ---------------------------------------------------------------------
        # Merge CPT linear risk column into wide dataframe
        # ---------------------------------------------------------------------
        valid_cptlin_models.append(run)

        cptlin_df = df_temp[keep_cols].copy()

        if merged is None:
            merged = cptlin_df
        else:
            merged = merged.merge(cptlin_df, on=key_col_list, how=merge_how)

        del df_temp, cptlin_df, check_df

    missing_cptlin_records = pd.DataFrame(
        missing_record_dfs,
        columns=["model_name", "outcome", "cptlin", "Missing_count"],
        )
           
    if merged is None:
        print("No CPT linear risk files were merged.")
        return None, missing_cptlin_records

    if print_summary:
        cptlin_cols = [run.cptlin for run in valid_cptlin_models]
        describe_long = (
            merged[cptlin_cols]
            .describe()
            .T
            .rename_axis("CPT_lin")
            .reset_index()
        )

        describe_formatters = {
            "count": lambda value: f"{value:,.0f}",
            **{
                col: (lambda value: f"{value:,.2f}")
                for col in describe_long.columns
                if col not in {"CPT_lin", "count"}
            },
        }

        print(describe_long.to_string(index=False, formatters=describe_formatters))

    if "asm1" in merged.columns:
        asm1 = pd.to_numeric(merged["asm1"], errors="raise")
        merged["asm1"] = asm1.astype("Int64" if asm1.isna().any() else int)

    if "division" in merged.columns:
        def _format_division(value):
            if pd.isna(value):
                return pd.NA
            value = str(value).strip()
            if value.endswith(".0") and value[:-2].isdigit():
                value = value[:-2]
            return value.zfill(5)

        merged["division"] = merged["division"].map(_format_division)

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        merged.to_parquet(output_path)
        
        excel_output_path = output_path.parent / f"{output_path.stem}_summary.xlsx"
        with pd.ExcelWriter(excel_output_path, engine="openpyxl") as writer:
            missing_cptlin_records.to_excel(
                writer,
                sheet_name="Missingness",
                index=False,
            )

            describe_long.to_excel(
                writer,
                sheet_name="CPT_lin_summary",
                index=False,
            )

    return merged, missing_cptlin_records

def clip_probs(p, eps=1e-6):
    p = np.asarray(p, dtype=np.float64)
    return np.clip(p, eps, 1 - eps)


def calibration_intercept_slope(y, p, eps=1e-6):
    """
    Fit: y ~ Logistic( a + b * logit(p) )
    Returns:
      intercept a, slope b
    Interpretation:
      a ~ 0 and b ~ 1 => well calibrated
      a > 0 => underprediction overall (pred too low)
      a < 0 => overprediction overall (pred too high)
      b < 1 => predictions too extreme (overconfident)
      b > 1 => predictions not extreme enough (underconfident)
    """
    y = np.asarray(y, dtype=np.int8)
    p = clip_probs(p, eps=eps)
    x = np.log(p / (1 - p)).reshape(-1, 1)

    # Unpenalized fit (C=np.inf replaces the deprecated penalty=None in sklearn>=1.8).
    lr = LogisticRegression(
        C=np.inf,
        solver="lbfgs",
        max_iter=200,
    )
    lr.fit(x, y)
    intercept = float(lr.intercept_[0])
    slope = float(lr.coef_[0, 0])
    return intercept, slope


def expected_calibration_error(y, p, n_bins=10, strategy="quantile", eps=1e-6):
    """
    ECE = sum_b (|acc_b - conf_b| * weight_b)
    strategy:
      - "quantile": equal counts per bin (recommended for huge N)
      - "uniform": equal-width bins in [0, 1]
    Returns:
      ece, bin_stats_df (dict of arrays)
    """
    y = np.asarray(y, dtype=np.int8)
    p = clip_probs(p, eps=eps)

    if strategy == "quantile":
        edges = np.quantile(p, np.linspace(0, 1, n_bins + 1))
        edges[0], edges[-1] = 0.0, 1.0
        for i in range(1, len(edges)):
            if edges[i] <= edges[i - 1]:
                edges[i] = min(1.0, edges[i - 1] + 1e-12)
    elif strategy == "uniform":
        edges = np.linspace(0, 1, n_bins + 1)
    else:
        raise ValueError("strategy must be 'quantile' or 'uniform'")

    bin_id = np.digitize(p, edges[1:-1], right=True)

    ece = 0.0
    bin_count = np.zeros(n_bins, dtype=np.int64)
    bin_mean_p = np.zeros(n_bins, dtype=np.float64)
    bin_mean_y = np.zeros(n_bins, dtype=np.float64)

    n = len(y)
    for b in range(n_bins):
        mask = bin_id == b
        nb = int(mask.sum())
        if nb == 0:
            continue
        bin_count[b] = nb
        bin_mean_p[b] = float(p[mask].mean())
        bin_mean_y[b] = float(y[mask].mean())
        ece += (nb / n) * abs(bin_mean_y[b] - bin_mean_p[b])

    stats = {
        "edges": edges,
        "count": bin_count,
        "mean_pred": bin_mean_p,
        "mean_obs": bin_mean_y,
    }
    return float(ece), stats


def calibration_metrics(y, p, n_bins=20, ece_strategy="quantile", eps=1e-6):
    y = np.asarray(y, dtype=np.int8)
    p = clip_probs(p, eps=eps)

    brier = brier_score_loss(y, p)
    intercept, slope = calibration_intercept_slope(y, p, eps=eps)
    ece, ece_stats = expected_calibration_error(
        y,
        p,
        n_bins=n_bins,
        strategy=ece_strategy,
        eps=eps,
    )

    return {
        "brier": float(brier),
        "cal_intercept": float(intercept),
        "cal_slope": float(slope),
        "ece": float(ece),
        "ece_stats": ece_stats,
    }


def plot_calibration_curve(
    y,
    p,
    n_bins=20,
    save_path="calib_plot.png",
    strategy="quantile",
    sample_for_plot=500_000,
    seed=123,
    eps=1e-6,
):
    y = np.asarray(y, dtype=np.int8)
    p = clip_probs(p, eps=eps)

    if sample_for_plot is not None and len(y) > sample_for_plot:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(y), size=sample_for_plot, replace=False)
        y_plot = y[idx]
        p_plot = p[idx]
    else:
        y_plot, p_plot = y, p

    ece, stats = expected_calibration_error(
        y_plot,
        p_plot,
        n_bins=n_bins,
        strategy=strategy,
        eps=eps,
    )
    mean_pred = stats["mean_pred"]
    mean_obs = stats["mean_obs"]
    count = stats["count"]

    mask = count > 0
    mean_pred = mean_pred[mask]
    mean_obs = mean_obs[mask]

    pad = 0.02
    xmin = max(0, min(mean_pred.min(), mean_obs.min()) - pad)
    xmax = min(1, max(mean_pred.max(), mean_obs.max()) + pad)

    plt.figure(figsize=(6, 6))
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.scatter(mean_pred, mean_obs, s=10, color="blue")
    plt.xlabel("Mean predicted probability")
    plt.ylabel("Observed event rate")
    plt.title(f"Calibration curve ({strategy}, bins={n_bins}) | ECE~{ece:.4f}")
    plt.ylim(xmin, xmax)
    plt.xlim(xmin, xmax)
    plt.savefig(save_path, dpi=1200, bbox_inches="tight")
    plt.show()


def fit_cpt_advi_and_score(
    df: pd.DataFrame,
    model_name: str,
    outcome: str,
    cptlin: str,
    *,
    cat_cols: list[str],
    model_base: str | Path,
    vi_method: str = "advi",
    seed: int = 123,
    large_data_threshold: int = 50_000,
    n_full: int = 30_000,
    lr_full: float = 1e-3,
    n_mb: int = 30_000,
    batch_size: int | None = None,
    lr_mb: float = 1e-3,
    alpha_sigma: float = 0.2,
    sigma_beta: float = 0.05,
    svgd_particles: int = 100,
    svgd_jitter: float = 1.0,
    svgd_temperature: float = 1.0,
    draws: int = 1000,
    rare_threshold: int | None = None,
):
    """
    Fits additive multilevel model:
      logit(p_i) = alpha + sum_k beta_k[level_k[i]]
    Returns:
      approx, idata, df_risk (row-level), unique_df_risk (unique-pattern table), meta
    """
    model_base = Path(model_base)
    output_folder = model_base / model_name
    output_folder.mkdir(parents=True, exist_ok=True)

    df = df[df[outcome].isin([0, 1])].copy()

    if rare_threshold is not None:
        for col in cat_cols:
            s = df[col].fillna("__MISSING__")
            vc = s.value_counts()
            rare = vc[vc < rare_threshold].index
            df.loc[s.isin(rare), col] = "__RARE__"
            df[col] = df[col].fillna("__MISSING__")

    cat_codes = {}
    cat_idx_arrays = []
    n_levels = []

    for col in cat_cols:
        codes, uniques = pd.factorize(df[col].fillna("__MISSING__"), sort=True)
        cat_codes[col] = uniques
        cat_idx_arrays.append(codes.astype(np.int32))
        n_levels.append(len(uniques))

    y = df[outcome].fillna(0).astype(np.int8).to_numpy()
    n = y.shape[0]
    y_rate = float(y.mean())

    unique_keys = df[cat_cols].fillna("__MISSING__").astype(str).agg("|".join, axis=1)
    df = df.copy()
    df["pattern_key"] = unique_keys

    unique_first_idx = unique_keys.drop_duplicates().index
    unique_df = df.loc[unique_first_idx, cat_cols].fillna("__MISSING__").reset_index(drop=True)
    unique_df["pattern_key"] = unique_keys.loc[unique_first_idx].to_numpy()

    unique_idx_matrix = np.vstack(
        [
            pd.Categorical(unique_df[col], categories=cat_codes[col]).codes
            for col in cat_cols
        ]
    ).T.astype(np.int32)
    m = unique_idx_matrix.shape[0]

    use_minibatch = n > large_data_threshold

    if use_minibatch:
        if batch_size is None:
            batch_size = int(np.clip(int(0.05 * n), 50_000, 250_000))
        scale = n / batch_size
        n_fit = n_mb
        lr = lr_mb
        mode = f"MINIBATCH (batch_size={batch_size:,})"
    else:
        batch_size = None
        scale = None
        n_fit = n_full
        lr = lr_full
        mode = "FULL BATCH"

    print(
        f"[{vi_method.upper()}] N={n:,}, M={m:,}, sum_levels={sum(n_levels):,} "
        f"| mode={mode} | y.mean={y_rate:.4f}"
    )

    start = time.time()
    with pm.Model() as _:
        alpha = pm.Normal(
            "alpha",
            mu=logit(max(min(y_rate, 1 - 1e-6), 1e-6)),
            sigma=alpha_sigma,
        )

        effects = []

        sigma_global = pm.HalfNormal("sigma_global", sigma=sigma_beta)
        sigma_sigma = pm.HalfNormal("sigma_sigma", sigma=sigma_beta)

        for k, n_level in enumerate(n_levels):
            print(f"k: {k}, K: {n_level}")
            sigma_k = pm.LogNormal(
                f"sigma_{k + 1}",
                mu=pt.log(sigma_global + 1e-8),
                sigma=sigma_sigma,
            )

            z = pm.Normal(f"z_{k + 1}", 0.0, 1.0, shape=n_level)
            beta_k = pm.Deterministic(f"beta_{k + 1}", z * sigma_k)
            effects.append(beta_k)

        if use_minibatch:
            y_data = pm.Data("y_data", y)
            idx_data = [
                pm.Data(f"idx_{k}", arr)
                for k, arr in enumerate(cat_idx_arrays)
            ]
            mb_idx = pm.Minibatch(np.arange(n, dtype=np.int32), batch_size=batch_size)

            logits_mb = alpha
            for k in range(len(cat_cols)):
                logits_mb = logits_mb + effects[k][idx_data[k][mb_idx]]

            y_mb = y_data[mb_idx]
            logp_vec = pm.logp(pm.Bernoulli.dist(logit_p=logits_mb), y_mb)
            pm.Potential("likelihood", scale * pt.sum(logp_vec))
        else:
            logits = alpha
            for k in range(len(cat_cols)):
                logits = logits + effects[k][cat_idx_arrays[k]]
            pm.Bernoulli("obs", logit_p=logits, observed=y)

        fit_kwargs = dict(
            n=n_fit,
            method=vi_method,
            progressbar=True,
            random_seed=seed,
        )

        fit_kwargs["obj_optimizer"] = pm.adam(learning_rate=lr)
        fit_kwargs["callbacks"] = [
            CheckParametersConvergence(tolerance=1e-3)
        ]

        if vi_method == "svgd":
            fit_kwargs["inf_kwargs"] = {
                "n_particles": svgd_particles,
                "jitter": svgd_jitter,
                "temperature": svgd_temperature,
            }
        elif vi_method not in {"advi", "fullrank_advi"}:
            raise ValueError(f"Unsupported vi_method: {vi_method}")

        approx = pm.fit(**fit_kwargs)

    end = time.time()
    running_time = (end - start) / 60

    posterior = approx.sample(draws=draws, random_seed=seed)
    idata = posterior if isinstance(posterior, az.InferenceData) else az.from_pymc(posterior)
    post = idata.posterior

    alpha_mean = post["alpha"].mean(dim=("chain", "draw")).values.item()
    beta_means = [
        post[f"beta_{k + 1}"].mean(dim=("chain", "draw")).values
        for k in range(len(cat_cols))
    ]

    logits_rows = np.full(n, alpha_mean, dtype=np.float32)
    for k in range(len(cat_cols)):
        logits_rows += beta_means[k][cat_idx_arrays[k]]

    logits_mean = float(logits_rows.mean())
    logits_std = float(logits_rows.std() + 1e-8)

    logits_unique = np.full(m, alpha_mean, dtype=np.float32)
    for k in range(len(cat_cols)):
        logits_unique += beta_means[k][unique_idx_matrix[:, k]]

    logits_unique_z = (logits_unique - logits_mean) / logits_std

    unique_df_risk = unique_df.copy()
    unique_df_risk[f"{cptlin}_org"] = logits_unique
    unique_df_risk[f"{cptlin}"] = logits_unique_z

    df_risk = df.merge(
        unique_df_risk[["pattern_key", f"{cptlin}_org", f"{cptlin}"]],
        on="pattern_key",
        how="left",
    ).drop(columns=["pattern_key"])

    p = expit(df_risk[f"{cptlin}_org"].to_numpy())
    df_risk["pred"] = p

    auc = roc_auc_score(y, df_risk["pred"].to_numpy())
    print(f"{model_name} c-statistic: {auc}")

    hl_stat, _, hl_p, _ = sar_helpers.hosmer_lemeshow_from_pymc(
        df_risk,
        y_col=outcome,
        p_col="pred",
        g=10,
    )
    print(f"{model_name} HL-statistic: {hl_stat}, p-value: {hl_p}")

    metrics = calibration_metrics(y, p, n_bins=20, ece_strategy="quantile")
    print(
        metrics["brier"],
        metrics["cal_intercept"],
        metrics["cal_slope"],
        metrics["ece"],
    )

    plot_calibration_curve(
        y,
        p,
        n_bins=20,
        save_path=output_folder / f"{model_name}_calib.png",
        strategy="quantile",
        sample_for_plot=500_000,
        seed=seed,
    )

    hist = getattr(approx, "hist", None)

    if hist is not None and len(hist) >= 500:
        w = 500
        elbo_smooth = np.convolve(hist, np.ones(w) / w, mode="valid")

        plt.figure(figsize=(10, 4))
        plt.plot(elbo_smooth)
        plt.title(f"Smoothed optimization history ({vi_method}, window={w})")
        plt.xlabel("Iteration")
        plt.ylabel("Objective history")
        plt.savefig(output_folder / f"{model_name}_{vi_method}_smoothed_history.png", dpi=1200)
        plt.show()

        delta = elbo_smooth[1:] - elbo_smooth[:-1]

        plt.figure(figsize=(10, 4))
        plt.plot(delta)
        plt.axhline(0, linestyle="--")
        plt.title(f"Change in smoothed optimization history ({vi_method})")
        plt.xlabel("Iteration")
        plt.ylabel("Delta objective")
        plt.savefig(
            output_folder / f"{model_name}_{vi_method}_delta_smoothed_history.png",
            dpi=1200,
        )
        plt.show()

    run_meta = {
        "model": model_name,
        "seed": seed,
        "mode": mode,
        "N": n,
        "M (unique CPTs combinations)": m,
        "Sum of CPTs over 21 CPTs": sum(n_levels),
        "y_rate": y_rate,
        "cat_cols": cat_cols,
        "cat_codes": {k: v.to_list() for k, v in cat_codes.items()},
        "logits_mean": logits_mean,
        "logits_std": logits_std,
        "vi_method": vi_method,
        "vi_settings": {
            "n_fit": n_fit,
            "learning_rate": lr,
            "batch_size": batch_size,
            "large_data_threshold": large_data_threshold,
            "svgd_particles": svgd_particles if vi_method == "svgd" else None,
            "svgd_jitter": svgd_jitter if vi_method == "svgd" else None,
            "svgd_temperature": svgd_temperature if vi_method == "svgd" else None,
        },
        "priors": {"alpha_sigma": alpha_sigma, "sigma_beta": sigma_beta},
        "draws_saved": draws,
        "running-time (min)": running_time,
        "c-stat": auc,
        "hl-stat": hl_stat,
        "hl p-value": hl_p,
        "brier": metrics["brier"],
        "calibration intercept": metrics["cal_intercept"],
        "calibration slope": metrics["cal_slope"],
        "expected calibration error": metrics["ece"],
    }

    with open(output_folder / f"{model_name}_{vi_method}_meta.json", "w", encoding="utf-8") as f:
        json.dump(run_meta, f, ensure_ascii=False, indent=4)

    df_risk_out = df_risk[
        ["division", "asm1", outcome, f"{cptlin}_org", f"{cptlin}", "pred"]
    ].copy()
    df_risk_out.to_csv(output_folder / f"{model_name}_cptlin_full_{vi_method}.csv", index=False)
    unique_df_risk.drop(columns="pattern_key").to_csv(
        output_folder / f"{model_name}_cptlin_unique_{vi_method}.csv",
        index=False,
    )


__all__ = [
    "ModelResults",
    "calibration_intercept_slope",
    "calibration_metrics",
    "clip_probs",
    "expected_calibration_error",
    "fit_cpt_advi_and_score",
    "merge_cptlin_results",
    "plot_calibration_curve",
]
