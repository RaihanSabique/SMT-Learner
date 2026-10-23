"""Shared helpers for the RQ validation experiments.

Ports the analysis and figure configuration from the original notebooks
  E-E_Analysis/RQ_validation/QR1_RQ3_micro_vs_macro_and_distribution.ipynb
  E-E_Analysis/RQ_validation/QR2_MSL_vs_PM_and_Case_Study.ipynb
onto the clean repo's E-E tables, which are produced from the saved
SMT-Learner checkpoint by scripts/run_experiments.py.

Differences from the notebooks, all forced by the revision:
  - MonkeyPR is gone (dataset removed), so every cohort/dataset panel that
    carried a "Monkey" series now carries only MSL and MP.
  - The E-E column is `ee_ratio` here, `exploration_exploitation_ratio` there.
  - `actual_dist` is not in the cached meta; it is merged back from the raw
    Dataset CSVs, which carry it directly for both MSL and UCPMP.
  - The notebooks disagreed with themselves on the MSL task filter (the RQ1
    t-test cell used task_type==1 labelled "Bimanual", the distribution and
    RQ2 cells used task_type==0 labelled "Unimanual"). We use Unimanual
    (task_type==0) everywhere and report both in the RQ1 table.
"""

import ast
import os

import numpy as np
import pandas as pd
import scipy.stats as stats
import seaborn as sns
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Paths are relative to this analysis package: results/ and RQ_Validation/ live
# beside it, and the raw dataset CSVs default to the repository's Dataset/.
ROOT = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.environ.get("SMT_DATASET_DIR",
                             os.path.join(os.path.dirname(ROOT), "Dataset"))

MICRO_COL = "ee_ratio"
TIME_COL = "completion_time"
RMSD_COL = "rmsd"
DIST_COL = "actual_dist"

# ---- figure configuration, copied from the notebooks ------------------------
FONT_RC = {
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "legend.fontsize": 11,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
}
# bin sizes (QR1 cell 17)
MSL_MACRO_BIN, MSL_MICRO_BIN = 2, 12
MP_MACRO_BIN, MP_MICRO_BIN = 2, 8
EARLY_LATE_BIN = 6          # QR2 cell 9
N_SLOPE_BINS, N_BOOT, BOOT_SEED = 10, 2000, 42

# dataset geometry
MSL_TRIALS_PER_SESSION, MSL_TOTAL_TRIALS = 24, 120
MP_TRIALS_PER_SESSION, MP_TOTAL_TRIALS = 48, 96

# palettes
MP_PALETTE = {"TD": "black", "UCP": "black"}
MSL_PALETTE = {"TERM": "black", "PRETERM": "black"}
MP_STYLES = {"TD": "-", "UCP": "--"}
MSL_STYLES = {"TERM": "-", "PRETERM": "--"}
EARLY_LATE_COLORS = {"mp": "red", "msl_unimanual": "blue"}
EARLY_LATE_STYLES = {"TD": "-", "UCP": "--", "TERM": "-", "PRETERM": "--"}
SLOPE_COLORS = {"UCP": "#4f81bd", "TD": "#9ecae1", "Term": "0.65", "Preterm": "0.45"}


def apply_style():
    sns.set_style("white")
    plt.rcParams.update(FONT_RC)


def save_fig(fig, out_dir, filename, dpi=300):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, filename)
    # pad_inches above the 0.1 default: the long slope-axis label sits flush
    # with the tight bbox and loses its last character otherwise
    fig.savefig(path, dpi=dpi, bbox_inches="tight", pad_inches=0.2)
    plt.close(fig)
    print(f"Saved: {path}")
    return path


# ---- statistics -------------------------------------------------------------
def ci95_t(x):
    x = pd.to_numeric(pd.Series(x), errors="coerce").dropna().astype(float)
    n = int(x.shape[0])
    if n == 0:
        return np.nan, np.nan, np.nan, 0
    mean = float(x.mean())
    if n < 2:
        return mean, np.nan, np.nan, n
    sem = stats.sem(x, nan_policy="omit")
    h = stats.t.ppf(0.975, df=n - 1) * sem
    return mean, mean - float(h), mean + float(h), n


def fmt_mean_ci(mean, lo, hi):
    if not (np.isfinite(mean) and np.isfinite(lo) and np.isfinite(hi)):
        return "NA"
    return f"{mean:.4g} [{lo:.4g}, {hi:.4g}]"


def welch_ttest(x, y):
    a = pd.to_numeric(pd.Series(x), errors="coerce").dropna().astype(float)
    b = pd.to_numeric(pd.Series(y), errors="coerce").dropna().astype(float)
    n1, n2 = int(a.shape[0]), int(b.shape[0])
    if n1 < 2 or n2 < 2:
        return {"t": np.nan, "p": np.nan, "n1": n1, "n2": n2, "cohen_d": np.nan}
    t, p = stats.ttest_ind(a, b, equal_var=False, nan_policy="omit")
    s1, s2 = float(a.std(ddof=1)), float(b.std(ddof=1))
    sp = np.sqrt(((n1 - 1) * s1 ** 2 + (n2 - 1) * s2 ** 2) / (n1 + n2 - 2)) if (n1 + n2 - 2) > 0 else np.nan
    d = float((a.mean() - b.mean()) / sp) if np.isfinite(sp) and sp > 0 else np.nan
    return {"t": float(t), "p": float(p), "n1": n1, "n2": n2, "cohen_d": d}


def paired_ttest(x, y):
    a = pd.to_numeric(pd.Series(x), errors="coerce").astype(float)
    b = pd.to_numeric(pd.Series(y), errors="coerce").astype(float)
    ok = a.notna() & b.notna()
    a, b = a[ok], b[ok]
    if len(a) < 2:
        return {"t": np.nan, "p": np.nan, "n": len(a), "cohen_dz": np.nan}
    t, p = stats.ttest_rel(a, b)
    diff = a - b
    dz = float(diff.mean() / diff.std(ddof=1)) if diff.std(ddof=1) > 0 else np.nan
    return {"t": float(t), "p": float(p), "n": int(len(a)), "cohen_dz": dz}


# ---- data loading -----------------------------------------------------------
MSL_MERGE_KEY = ["participant_id", "session_no", "task_type", "trial_no", "day", "block", "start_time"]
MP_MERGE_KEY = ["participant_id", "session_no", "trial_no", "day", "block_no", "planet", "trial_hand", "start_time"]


def _merge_actual_dist(meta, raw_csv, key):
    """actual_dist lives only in the raw Dataset CSVs; merge it back."""
    raw = pd.read_csv(os.path.join(DATASET_DIR, raw_csv), usecols=key + [DIST_COL])
    if raw.duplicated(key).any():
        raise ValueError(f"{raw_csv}: merge key {key} is not unique")
    merged = meta.merge(raw, on=key, how="left", validate="one_to_one")
    n_missing = merged[DIST_COL].isna().sum()
    if n_missing:
        raise ValueError(f"{raw_csv}: {n_missing}/{len(merged)} rows failed to match on {key}")
    return merged


def load_ee_tables(tag="smtlearner", results_dir=None):
    """E-E tables for MSL and MP, with actual_dist, speed and rmsd_norm added.

    speed and accuracy are the macro metrics the notebooks compare the micro
    E-E ratio against: speed = actual_dist / completion_time, accuracy =
    rmsd / actual_dist (lower is better)."""
    results_dir = results_dir or os.path.join(ROOT, "results")
    msl = pd.read_csv(os.path.join(results_dir, f"MSL_ee_table_{tag}.csv"))
    mp = pd.read_csv(os.path.join(results_dir, f"UCPMP_ee_table_{tag}.csv"))
    msl = _merge_actual_dist(msl, "MSL.csv", MSL_MERGE_KEY)
    mp = _merge_actual_dist(mp, "UCPMP.csv", MP_MERGE_KEY)
    for df in (msl, mp):
        for c in (MICRO_COL, TIME_COL, RMSD_COL, DIST_COL):
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df["speed"] = np.where(df[DIST_COL].notna() & df[TIME_COL].notna() & (df[TIME_COL] > 0),
                               df[DIST_COL] / df[TIME_COL], np.nan)
        df["rmsd_norm"] = np.where(df[RMSD_COL].notna() & df[DIST_COL].notna() & (df[DIST_COL] > 0),
                                   df[RMSD_COL] / df[DIST_COL], np.nan)
    msl["group"] = msl["Cohort"].astype(str).str.strip().str.upper()
    mp["group"] = mp["Cohort"].astype(str).str.strip().str.upper()
    return msl, mp


# ---- trial indexing / binning ----------------------------------------------
def build_global_trial_index(df, unit_col, session_col, trial_col,
                             trials_per_session, expected_total_trials):
    """Global trial index in [1..expected_total_trials] per unit, built from
    session rank and within-session trial order (QR1 _build_global_trial_index)."""
    d = df[[unit_col, session_col, trial_col]].copy()
    d[trial_col] = pd.to_numeric(d[trial_col], errors="coerce")
    d[session_col] = pd.to_numeric(d[session_col], errors="coerce")
    session_rank = d.groupby(unit_col)[session_col].rank(method="dense").astype("Int64")
    within = d.groupby([unit_col, session_col])[trial_col].rank(method="first")
    idx = (session_rank - 1) * trials_per_session + within
    idx = idx.where(idx <= expected_total_trials)
    return idx.astype(float)


def binned_mean(df, group_col, unit_col, session_col, trial_col,
                trials_per_session, expected_total_trials, bin_size, metric):
    """Mean of `metric` per group in fixed-width bins of the global trial index."""
    d = df.copy()
    d["group"] = d[group_col].astype(str).str.upper()
    d[unit_col] = d[unit_col].astype(str)
    d["trial_index"] = build_global_trial_index(
        d, unit_col, session_col, trial_col, trials_per_session, expected_total_trials)
    d = d.dropna(subset=["trial_index"])
    d["trial_index"] = d["trial_index"].astype(int)
    d[metric] = pd.to_numeric(d[metric], errors="coerce")
    d = d.dropna(subset=[metric])
    d["bin"] = ((d["trial_index"] - 1) // bin_size) + 1
    d["x"] = d["bin"] * bin_size
    return d.groupby(["group", "bin", "x"], as_index=False)[metric].mean().rename(columns={metric: "mean"})


def binned_range(df, group_col, unit_col, session_col, trial_col,
                 trials_per_session, expected_total_trials, bin_size, metric, which,
                 n_trials=24):
    """Early (first n) or late (last n) trials per unit, binned (QR2 cell 9)."""
    d = df.copy()
    d["group"] = d[group_col].astype(str).str.upper()
    d[unit_col] = d[unit_col].astype(str)
    d["trial_index"] = build_global_trial_index(
        d, unit_col, session_col, trial_col, trials_per_session, expected_total_trials)
    d = d.dropna(subset=["trial_index"])
    d["trial_index"] = d["trial_index"].astype(int)
    rows = []
    for _, sub in d.groupby([unit_col, "group"]):
        sub = sub.sort_values("trial_index")
        sub = sub.iloc[:n_trials] if which == "early" else sub.iloc[-n_trials:]
        sub = sub.copy()
        sub["norm_trial"] = np.arange(1, len(sub) + 1)
        rows.append(sub)
    if not rows:
        return pd.DataFrame(columns=["group", "bin", "x", "mean"])
    d2 = pd.concat(rows)
    d2[metric] = pd.to_numeric(d2[metric], errors="coerce")
    d2 = d2.dropna(subset=[metric, "norm_trial"])
    d2["bin"] = ((d2["norm_trial"].astype(int) - 1) // bin_size) + 1
    d2["x"] = d2["bin"] * bin_size
    return d2.groupby(["group", "bin", "x"], as_index=False)[metric].mean().rename(columns={metric: "mean"})


def early_late_per_unit(df, unit_col, session_col, trial_col,
                        trials_per_session, expected_total_trials,
                        metric=MICRO_COL, n_trials=24):
    """Per-unit early and late means, for paired tests."""
    d = df.copy()
    d[unit_col] = d[unit_col].astype(str)
    d["trial_index"] = build_global_trial_index(
        d, unit_col, session_col, trial_col, trials_per_session, expected_total_trials)
    d = d.dropna(subset=["trial_index"])
    d[metric] = pd.to_numeric(d[metric], errors="coerce")
    rows = []
    for uid, sub in d.groupby(unit_col):
        sub = sub.sort_values("trial_index")
        early = sub.iloc[:n_trials][metric].mean()
        late = sub.iloc[-n_trials:][metric].mean()
        rows.append({"unit_id": uid, "group": str(sub["group"].iloc[0]).upper(),
                     "early": early, "late": late, "delta": late - early})
    return pd.DataFrame(rows)


# ---- slope distributions ----------------------------------------------------
def bin_trials_by_unit(df, unit_cols, trial_col, n_bins=N_SLOPE_BINS, ratio_col=MICRO_COL):
    """Bin trials within each unit; mean E-E per bin with trial_percentage 0-100."""
    rows = []
    d = df.copy()
    d[ratio_col] = pd.to_numeric(d[ratio_col], errors="coerce")
    d[trial_col] = pd.to_numeric(d[trial_col], errors="coerce")
    d = d.dropna(subset=[ratio_col, trial_col])
    for unit_vals, g in d.groupby(unit_cols):
        if len(g) < 5:
            continue
        g = g.sort_values(trial_col)
        max_trials = g[trial_col].max()
        if max_trials <= 1:
            continue
        bin_size = max(1, max_trials / n_bins)
        g = g.copy()
        g["trial_bin"] = np.minimum(((g[trial_col] - 1) // bin_size).astype(int), n_bins - 1)
        binned = g.groupby("trial_bin", as_index=False)[ratio_col].mean()
        binned["trial_percentage"] = (binned["trial_bin"] / (n_bins - 1)) * 100.0
        if not isinstance(unit_vals, tuple):
            unit_vals = (unit_vals,)
        binned["unit_id"] = "_".join(str(v) for v in unit_vals)
        rows.append(binned[["unit_id", "trial_percentage", ratio_col]])
    if not rows:
        return pd.DataFrame(columns=["unit_id", "trial_percentage", ratio_col])
    return pd.concat(rows, ignore_index=True)


def ols_slope(df, ratio_col=MICRO_COL):
    if df.empty or df["trial_percentage"].nunique() < 2:
        return np.nan
    x = df["trial_percentage"].to_numpy(float)
    y = df[ratio_col].to_numpy(float)
    return float(np.polyfit(x, y, 1)[0])


def bootstrap_slopes(binned_df, n_boot=N_BOOT, seed=BOOT_SEED, ratio_col=MICRO_COL):
    """Cluster bootstrap over units (QR1 cell 24)."""
    rng = np.random.default_rng(seed)
    units = binned_df["unit_id"].dropna().unique()
    if units.size == 0:
        return pd.DataFrame(columns=["slope"])
    groups = {uid: binned_df[binned_df["unit_id"] == uid] for uid in units}
    slopes = []
    for _ in range(n_boot):
        sampled = rng.choice(units, size=len(units), replace=True)
        slopes.append(ols_slope(pd.concat([groups[u] for u in sampled], ignore_index=True), ratio_col))
    return pd.DataFrame({"slope": slopes})


# ---- case study -------------------------------------------------------------
def parse_path(val):
    if isinstance(val, (list, tuple, np.ndarray)):
        arr = np.asarray(val)
    elif isinstance(val, str):
        try:
            arr = np.asarray(ast.literal_eval(val))
        except Exception:
            return None
    else:
        return None
    if arr.ndim == 1 and arr.size % 2 == 0:
        arr = arr.reshape(-1, 2)
    if arr.ndim >= 2 and arr.shape[1] >= 2:
        return arr[:, 0], arr[:, 1]
    return None
