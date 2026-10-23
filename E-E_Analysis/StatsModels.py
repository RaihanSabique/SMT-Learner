"""M1: linear mixed-effects models replacing the trajectory-level ANOVA (R7-3).

Model (per dataset):  z(EE ratio) ~ Cohort * Phase + z(Age) [+ Hand for UCPMP]
with a random intercept per participant. Reports standardized beta, 95% CI,
p-values with Benjamini-Hochberg FDR across the full contrast family, and
Cohen's d computed on SUBJECT-LEVEL means (the correct unit of analysis)."""

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests


def _zscore(s):
    return (s - s.mean()) / s.std()


def _fit_robust(model, methods=("lbfgs", "powell", "cg", "bfgs")):
    """Fit a MixedLM, falling back across optimizers.

    On heavy-tailed E-E distributions the random-intercept variance can be
    driven to the boundary, at which point lbfgs inverts a singular
    information matrix and raises. The fallbacks reach the same REML
    objective by a different route; the first optimizer that both converges
    and returns finite standard errors wins."""
    last_err = None
    for i, method in enumerate(methods):
        try:
            fit = model.fit(reml=True, method=method)
        except Exception as exc:                      # noqa: BLE001 - optimizer-specific
            last_err = exc
            continue
        if fit.converged and np.all(np.isfinite(fit.bse_fe)):
            if i:
                print(f"  (LMM fell back to method={method}: {type(last_err).__name__ if last_err else 'non-convergence'})")
            return fit
        last_err = RuntimeError(f"{method} did not converge")
    raise RuntimeError(f"all MixedLM optimizers failed; last error: {last_err}")


def fit_lmm(ee_df: pd.DataFrame, dataset: str, ratio_col: str = "ee_ratio"):
    df = ee_df.dropna(subset=[ratio_col, "Age"]).copy()
    df = df[df["phase"].isin(["early", "late"])]
    # winsorize the heavy right tail of the ratio before z-scoring
    hi = df[ratio_col].quantile(0.99)
    df["ee_z"] = _zscore(df[ratio_col].clip(upper=hi))
    df["age_z"] = _zscore(df["Age"].astype(float))
    df["cohort"] = df["Cohort"].astype(str)
    df["phase_"] = pd.Categorical(df["phase"], categories=["early", "late"])

    formula = "ee_z ~ C(cohort) * C(phase_) + age_z"
    if dataset.upper() == "UCPMP":
        formula += " + C(task_type)"  # paretic vs non-paretic hand

    model = smf.mixedlm(formula, df, groups=df["participant_id"])
    fit = _fit_robust(model)

    rows = []
    for name in fit.fe_params.index:
        if name == "Intercept":
            continue
        ci = fit.conf_int().loc[name]
        rows.append({
            "dataset": dataset, "term": name,
            "beta_std": fit.fe_params[name],
            "ci_low": ci[0], "ci_high": ci[1],
            "p": fit.pvalues[name],
        })
    table = pd.DataFrame(rows)
    return fit, table, df


def cohens_d_subject_level(df: pd.DataFrame, ratio_col="ee_ratio"):
    """Effect sizes on subject-level means for the key contrasts."""
    out = []
    subj = (df[df["phase"].isin(["early", "late"])]
            .groupby(["participant_id", "Cohort", "phase"])[ratio_col]
            .mean().reset_index())

    def d(a, b):
        na, nb = len(a), len(b)
        if na < 2 or nb < 2:
            return np.nan
        sp = np.sqrt(((na - 1) * a.std() ** 2 + (nb - 1) * b.std() ** 2) / (na + nb - 2))
        return (a.mean() - b.mean()) / sp if sp > 0 else np.nan

    cohorts = sorted(subj["Cohort"].unique())
    for phase in ("early", "late"):
        sub = subj[subj["phase"] == phase]
        a = sub[sub["Cohort"] == cohorts[0]][ratio_col]
        b = sub[sub["Cohort"] == cohorts[1]][ratio_col]
        out.append({"contrast": f"{cohorts[0]} vs {cohorts[1]} ({phase})",
                    "cohens_d": d(a, b), "n1": len(a), "n2": len(b)})
    for cohort in cohorts:
        sub = subj[subj["Cohort"] == cohort]
        wide = sub.pivot(index="participant_id", columns="phase", values=ratio_col).dropna()
        if len(wide) >= 2:
            diff = wide["early"] - wide["late"]
            out.append({"contrast": f"early vs late ({cohort}, paired)",
                        "cohens_d": diff.mean() / diff.std() if diff.std() > 0 else np.nan,
                        "n1": len(wide), "n2": len(wide)})
    return pd.DataFrame(out)


def run_stats(ee_tables: dict, out_csv: str):
    """ee_tables: {dataset_name: ee_df with phase labels}. Applies BH-FDR
    across the pooled contrast family from both datasets."""
    all_tables, all_d = [], []
    for name, df in ee_tables.items():
        fit, table, used = fit_lmm(df, name)
        all_tables.append(table)
        dtab = cohens_d_subject_level(used)
        dtab.insert(0, "dataset", name)
        all_d.append(dtab)
        print(f"[{name}] LMM converged: {fit.converged}, n_trials={len(used)}, "
              f"n_subjects={used['participant_id'].nunique()}")
    stats = pd.concat(all_tables, ignore_index=True)
    stats["p_fdr"] = multipletests(stats["p"], method="fdr_bh")[1]
    stats["significant_fdr05"] = stats["p_fdr"] < 0.05
    effect = pd.concat(all_d, ignore_index=True)
    stats.to_csv(out_csv, index=False)
    effect.to_csv(out_csv.replace(".csv", "_effect_sizes.csv"), index=False)
    return stats, effect
