"""M2: external clinical validity of the E-E biomarker (R7-1) — the headline
new result. Subject-level E-E summaries vs standardized clinical scores that
were never used in training:
  MSL   -> MABC-2 standard score
  UCPMP -> ABILHAND-Kids patient measure (logits)
        -> NIH Toolbox age-adjusted standard scores (DCCS, Flanker, PSM,
           Picture Vocabulary, RAVLT) and the cognition Composite Score

The NIH Toolbox panel is a second, cognitive axis of external validity: unlike
ABILHAND-Kids it is not a manual-ability instrument, so a correlation there
speaks to whether E-E indexes exploration policy rather than limb function.

Reports: Spearman rho with bootstrap 95% CI; partial Spearman controlling for
the three performance variables the model was trained with (completion time,
rmsd, success rate); leave-one-subject-out ridge probes (perf-only vs EE-only
vs both) with predictions for held-out subjects only ever made by models that
never saw them."""

import os

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.multitest import multipletests

# NIH Toolbox column -> short name. Age-adjusted standard scores are the
# comparable unit across ages (mean 100, sd 15), matching MABC-2's scaling.
NIH_SCORE_COLS = {
    "DCCS Age-Adj. Stand. Score": "nih_dccs",
    "FI Age-Adj. Stand. Score": "nih_flanker",
    "PSM Age-Adj. Stand. Score": "nih_psm",
    "PV Age-Adj. Stand. Score": "nih_picvocab",
    "RAVLT Age-Adj. Stand. Score": "nih_ravlt",
    "Composite Score": "nih_composite",
}


def perf_summaries(meta: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for pid, g in meta.groupby("participant_id"):
        row = {
            "participant_id": pid,
            "ct_mean": g["completion_time"].mean(),
            "rmsd_mean": g["rmsd"].mean(),
        }
        if (g["is_success"] >= 0).any():
            row["success_rate"] = g.loc[g["is_success"] >= 0, "is_success"].mean()
        rows.append(row)
    return pd.DataFrame(rows)


def bootstrap_spearman(x, y, n_boot=2000, seed=42):
    rho, p = spearmanr(x, y)
    rng = np.random.RandomState(seed)
    n = len(x)
    boots = []
    x, y = np.asarray(x), np.asarray(y)
    for _ in range(n_boot):
        idx = rng.randint(0, n, n)
        if len(np.unique(x[idx])) > 1 and len(np.unique(y[idx])) > 1:
            boots.append(spearmanr(x[idx], y[idx])[0])
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {"rho": rho, "p": p, "ci_low": lo, "ci_high": hi, "n": n}


def partial_spearman(x, y, controls: np.ndarray, n_boot=2000, seed=42):
    """Spearman of residuals after linear regression on controls (rank-based)."""
    from scipy.stats import rankdata

    def resid(v):
        vr = rankdata(v)
        C = np.column_stack([rankdata(c) for c in controls.T] + [np.ones(len(vr))])
        beta, *_ = np.linalg.lstsq(C, vr, rcond=None)
        return vr - C @ beta

    return bootstrap_spearman(resid(np.asarray(x)), resid(np.asarray(y)), n_boot, seed)


def loso_probe(features: pd.DataFrame, outcome: pd.Series, alpha=1.0):
    """Leave-one-subject-out ridge regression; every prediction is for a
    subject the probe never saw."""
    X = features.to_numpy(float)
    y = outcome.to_numpy(float)
    preds = np.empty(len(y))
    for train_idx, test_idx in LeaveOneOut().split(X):
        scaler = StandardScaler().fit(X[train_idx])
        model = Ridge(alpha=alpha).fit(scaler.transform(X[train_idx]), y[train_idx])
        preds[test_idx] = model.predict(scaler.transform(X[test_idx]))
    ss_res = np.sum((y - preds) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    rho, p = spearmanr(preds, y)
    return {"r2_loso": 1 - ss_res / ss_tot, "spearman_pred_true": rho, "p": p, "n": len(y)}


def run_clinical_validity(ee_subject: pd.DataFrame, meta: pd.DataFrame,
                          clinical: pd.Series, dataset: str,
                          test_pids=None) -> pd.DataFrame:
    """ee_subject: from ee_metric.subject_summaries. clinical: indexed by
    participant_id. Returns a tidy results table."""
    perf = perf_summaries(meta)
    df = ee_subject.merge(perf, on="participant_id")
    df["clinical"] = df["participant_id"].map(clinical)
    df = df.dropna(subset=["clinical"])
    print(f"[{dataset}] clinical validity n={len(df)} subjects")

    perf_cols = [c for c in ("ct_mean", "rmsd_mean", "success_rate") if c in df.columns]
    ee_cols = ["ee_mean", "ee_early", "ee_late", "ee_slope", "ee_decline"]
    controls = df[perf_cols].to_numpy(float)

    rows = []
    for col in ee_cols + perf_cols:
        sub = df.dropna(subset=[col])
        r = bootstrap_spearman(sub[col], sub["clinical"])
        r.update({"dataset": dataset, "predictor": col, "type": "raw"})
        rows.append(r)
    for col in ee_cols:
        sub = df.dropna(subset=[col] + perf_cols)
        r = partial_spearman(sub[col], sub["clinical"], sub[perf_cols].to_numpy(float))
        r.update({"dataset": dataset, "predictor": col, "type": "partial_vs_perf"})
        rows.append(r)

    probes = {
        "perf_only": perf_cols,
        "ee_only": ee_cols,
        "perf_plus_ee": perf_cols + ee_cols,
    }
    for name, cols in probes.items():
        sub = df.dropna(subset=cols)
        r = loso_probe(sub[cols], sub["clinical"])
        r.update({"dataset": dataset, "predictor": name, "type": "loso_probe"})
        rows.append(r)

    if test_pids is not None:
        held = df[df["participant_id"].isin(test_pids)]
        rows.append({"dataset": dataset, "predictor": "ee_mean", "type": "heldout_subjects_note",
                     "n": len(held),
                     "rho": spearmanr(held["ee_mean"], held["clinical"])[0] if len(held) >= 4 else np.nan})
    return pd.DataFrame(rows), df


# ---------------------------------------------------------------------------
# multi-instrument clinical panel
# ---------------------------------------------------------------------------
def clinical_targets(dataset: str, meta: pd.DataFrame, dataset_dir: str) -> dict:
    """All external clinical measures available for a dataset.

    Returns {measure_name: Series indexed by participant_id}. Every measure is
    external to training: none of them is an input to the model or to the E-E
    metric."""
    if dataset.upper() == "MSL":
        return {"mabc_standard": meta.groupby("participant_id")["mabc_standard_score"].first()}

    ha = pd.read_csv(os.path.join(dataset_dir, "UCPMP_HandAbility.csv"))
    ha = ha.rename(columns={"partcipant_id": "participant_id"}).drop_duplicates("participant_id")
    targets = {"abilhand_logits": pd.to_numeric(
        ha.set_index("participant_id")["PatientMeasure_Logits"], errors="coerce")}

    nih = pd.read_csv(os.path.join(dataset_dir, "UCPMP_NIHTOOLBOX.csv"))
    nih = nih.drop_duplicates("participant_id").set_index("participant_id")
    for col, short in NIH_SCORE_COLS.items():
        s = pd.to_numeric(nih[col], errors="coerce")
        if short == "nih_composite":
            # 0.0 is the export's placeholder for "battery not completed":
            # those four UCP participants have every subscore blank.
            s = s.replace(0.0, np.nan)
        targets[short] = s
    return targets


def run_clinical_panel(ee_subject: pd.DataFrame, meta: pd.DataFrame, dataset: str,
                       dataset_dir: str, test_pids=None, min_n: int = 8):
    """run_clinical_validity over every clinical instrument for `dataset`.

    BH-FDR is applied across the raw and partial Spearman family within each
    dataset, since the NIH Toolbox panel adds six correlated outcomes."""
    tables, merged = [], {}
    for measure, series in clinical_targets(dataset, meta, dataset_dir).items():
        n_avail = ee_subject["participant_id"].map(series).notna().sum()
        if n_avail < min_n:
            print(f"[{dataset}/{measure}] skipped: only {n_avail} subjects with a score")
            continue
        res, df = run_clinical_validity(ee_subject, meta, series, dataset, test_pids=test_pids)
        res.insert(1, "measure", measure)
        tables.append(res)
        merged[measure] = df
    if not tables:
        return pd.DataFrame(), merged

    out = pd.concat(tables, ignore_index=True)
    fam = out["type"].isin(["raw", "partial_vs_perf"]) & out["p"].notna()
    out["p_fdr"] = np.nan
    out.loc[fam, "p_fdr"] = multipletests(out.loc[fam, "p"], method="fdr_bh")[1]
    out["significant_fdr05"] = out["p_fdr"] < 0.05
    return out, merged
