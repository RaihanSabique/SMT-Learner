"""Exploration-Exploitation (E-E) metric on trajectory embeddings.

Faithful to the submitted paper's definition MINUS two estimator artifacts
(revision Step 0.4):
  1. REMOVED trial_position_factor = exp(-0.01 * i): it multiplied exploration
     by a decaying function of trial index, hard-coding a decline.
  2. REMOVED global min-max normalization across the whole dataframe: scores
     are now min-max scaled with bounds fit on TRAINING participants only, so
     held-out subjects' values are portable and cohort comparisons do not leak.

Kept from the original: euclidean distance in embedding space, novelty =
min distance to ALL previous trials, exploitation = max over the last
`window` trials of 1/(1+d), ratio = exploration / (exploitation + 0.01).
First trial of each sequence is NaN (no imputation; coverage is reported).
"""

import numpy as np
import pandas as pd
from scipy.spatial.distance import squareform, pdist


def group_distance_matrices(meta: pd.DataFrame, embeddings: np.ndarray,
                            group_cols=("participant_id", "task_type")):
    """Per participant x task: chronologically ordered row ids + distance matrix.

    Returns list of (group_key, row_ids_in_chron_order, dist_matrix)."""
    out = []
    for key, grp in meta.groupby(list(group_cols)):
        grp = grp.sort_values("trial_seq")
        rows = grp["row_id"].to_numpy()
        emb = embeddings[rows]
        d = squareform(pdist(emb, metric="euclidean")) if len(rows) > 1 else np.zeros((1, 1))
        out.append((key, rows, d))
    return out


def ee_scores_from_distmat(dist: np.ndarray, order: np.ndarray, window: int):
    """Exploration / exploitation per trial for one ordered sequence.

    `order` maps sequence position -> index into dist. Returns (explor, exploit)
    arrays aligned with sequence positions; position 0 is NaN."""
    n = len(order)
    explor = np.full(n, np.nan)
    exploit = np.full(n, np.nan)
    for i in range(1, n):
        prev = order[:i]
        cur = order[i]
        d_prev = dist[cur, prev]
        explor[i] = d_prev.min()  # novelty: min distance to any previous trial
        d_win = d_prev[max(0, i - window):]
        exploit[i] = (1.0 / (1.0 + d_win)).max()
    return explor, exploit


def compute_ee(meta: pd.DataFrame, embeddings: np.ndarray, train_pids,
               window: int = 6, smoothing: float = 0.01,
               group_cols=("participant_id", "task_type"),
               score_prefix: str = "") -> pd.DataFrame:
    """E-E table for all trials. Normalization bounds fit on train_pids only."""
    groups = group_distance_matrices(meta, embeddings, group_cols)
    explor = np.full(len(meta), np.nan)
    exploit = np.full(len(meta), np.nan)
    for _, rows, dist in groups:
        e, x = ee_scores_from_distmat(dist, np.arange(len(rows)), window)
        explor[rows] = e
        exploit[rows] = x

    out = meta.copy()
    p = score_prefix
    out[p + "exploration_score"] = explor
    out[p + "exploitation_score"] = exploit

    train_mask = out["participant_id"].isin(train_pids)
    for col in (p + "exploration_score", p + "exploitation_score"):
        lo = out.loc[train_mask, col].min()
        hi = out.loc[train_mask, col].max()
        if hi > lo:
            out[col] = (out[col] - lo) / (hi - lo)

    out[p + "ee_ratio"] = out[p + "exploration_score"] / (out[p + "exploitation_score"] + smoothing)
    coverage = out[p + "ee_ratio"].notna().mean()
    print(f"E-E coverage: {coverage:.1%} of trials (first trial per sequence is undefined)")
    return out


def add_phase_labels(ee_df: pd.DataFrame, group_cols=("participant_id", "task_type")) -> pd.DataFrame:
    """Early / middle / late tertiles of trial_seq within each sequence."""
    df = ee_df.copy()
    df["phase"] = ""
    for _, grp in df.groupby(list(group_cols)):
        seq = grp["trial_seq"]
        t1, t2 = seq.quantile(1 / 3), seq.quantile(2 / 3)
        lab = np.where(seq <= t1, "early", np.where(seq > t2, "late", "middle"))
        df.loc[grp.index, "phase"] = lab
    return df


def subject_summaries(ee_df: pd.DataFrame, ratio_col: str = "ee_ratio") -> pd.DataFrame:
    """Per-participant E-E summaries used for clinical validity (M2)."""
    rows = []
    for pid, grp in ee_df.groupby("participant_id"):
        g = grp.dropna(subset=[ratio_col])
        if len(g) < 4:
            continue
        early = g[g["phase"] == "early"][ratio_col].mean() if "phase" in g else np.nan
        late = g[g["phase"] == "late"][ratio_col].mean() if "phase" in g else np.nan
        slope = np.polyfit(g["trial_seq"], g[ratio_col], 1)[0]
        rows.append({
            "participant_id": pid,
            "ee_mean": g[ratio_col].mean(),
            "ee_early": early,
            "ee_late": late,
            "ee_slope": slope,
            "ee_decline": early - late,
            "n_trials": len(g),
        })
    return pd.DataFrame(rows)
