"""NIH Toolbox (uncorrected standard scores) panel for the MP / UCPMP study.

Two jobs, both driven by the seven columns the manuscript's Table
(participant characteristics) and the clinical-validity analysis were asked to
report:

    DCCS Stand. Score      Dimensional Change Card Sort  (cognitive flexibility)
    FI Stand. Score        Flanker Inhibitory Control    (inhibitory control)
    PSM Stand. Score       Picture Sequence Memory       (episodic memory)
    PV Stand. Score        Picture Vocabulary            (receptive vocabulary)
    Composite Score        Early Childhood Cognition composite
    Percentile Rank        composite, percentile units
    Standard T Score       composite, T units (M 50, SD 10)

These are the UNCORRECTED standard scores (norm-referenced to the whole
nationally representative sample), not the age-adjusted ones used by
`clinical_validity.NIH_SCORE_COLS`.  Two consequences are carried explicitly
through every function here:

  1. Uncorrected scores confound the construct with age, and the MP groups
     differ in age (UCP ~1 year older).  Every correlation is therefore also
     reported partialling out age at the PPT visit, alongside the partial
     against the three performance variables the model was trained with.
  2. `Composite Score`, `Percentile Rank` and `Standard T Score` are three
     units of the SAME composite (Spearman rho > 0.999 with one another in this
     cohort).  They are reported as separate rows because the table asks for
     them, but they are one test, not three, and the FDR family is defined over
     the four distinct subtests + one composite to avoid inflating the count.

`0.0` is the export's placeholder for "battery not completed" and is mapped to
NaN in every column (in the analysis cohort no participant is affected, but the
raw file has four such UCP rows)."""

import os

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, spearmanr, t as student_t
from statsmodels.stats.multitest import multipletests

from ClinicalValidity import (bootstrap_spearman, loso_probe,
                                        partial_spearman, perf_summaries)

# raw column -> (short key, display label, construct, is_composite_restatement)
NIH_RAW_COLS = {
    "DCCS Stand. Score": ("dccs", "DCCS", "Cognitive flexibility", False),
    "FI Stand. Score": ("flanker", "Flanker Inhibitory Control", "Inhibitory control", False),
    "PSM Stand. Score": ("psm", "Picture Sequence Memory", "Episodic memory", False),
    "PV Stand. Score": ("picvocab", "Picture Vocabulary", "Receptive vocabulary", False),
    "Composite Score": ("composite", "Cognition Composite", "Global cognition", False),
    "Percentile Rank": ("percentile", "Composite percentile rank", "Global cognition (\\%ile)", True),
    "Standard T Score": ("tscore", "Composite T score", "Global cognition (T)", True),
}

# order used in every table and figure
ORDER = ["dccs", "flanker", "psm", "picvocab", "composite", "percentile", "tscore"]
LABEL = {v[0]: v[1] for v in NIH_RAW_COLS.values()}
CONSTRUCT = {v[0]: v[2] for v in NIH_RAW_COLS.values()}
RESTATES_COMPOSITE = {v[0]: v[3] for v in NIH_RAW_COLS.values()}


def load_nih(dataset_dir: str) -> pd.DataFrame:
    """Wide NIH frame indexed by participant_id: the seven requested scores plus
    Condition and age at the PPT visit."""
    raw = pd.read_csv(os.path.join(dataset_dir, "UCPMP_NIHTOOLBOX.csv"))
    raw = raw.drop_duplicates("participant_id").set_index("participant_id")
    out = pd.DataFrame(index=raw.index)
    out["condition"] = raw["Condition"].str.strip()
    out["age_ppt"] = pd.to_numeric(raw["Age at PPT"], errors="coerce")
    for col, (short, *_rest) in NIH_RAW_COLS.items():
        s = pd.to_numeric(raw[col], errors="coerce")
        out[short] = s.replace(0.0, np.nan)  # 0.0 == battery not completed
    return out


# ---------------------------------------------------------------------------
# participant characteristics
# ---------------------------------------------------------------------------
def _mean_ci(x: pd.Series):
    x = x.dropna()
    n = len(x)
    if n == 0:
        return np.nan, np.nan, np.nan, 0, np.nan
    m, sd = x.mean(), x.std(ddof=1)
    if n < 2 or not np.isfinite(sd):
        return m, np.nan, np.nan, n, sd
    half = student_t.ppf(0.975, n - 1) * sd / np.sqrt(n)
    return m, m - half, m + half, n, sd


def characteristics(nih: pd.DataFrame, cohort: pd.Series) -> pd.DataFrame:
    """Per-group mean, SD and 95% CI for every requested score, restricted to
    the participants in `cohort` (a participant_id -> group label Series), plus
    a Mann-Whitney group comparison with rank-biserial effect size."""
    df = nih.reindex(cohort.index).assign(group=cohort)
    groups = list(pd.unique(cohort))
    rows = []
    for short in ORDER:
        row = {"measure": short, "label": LABEL[short], "construct": CONSTRUCT[short]}
        samples = {}
        for g in groups:
            x = df.loc[df["group"] == g, short]
            m, lo, hi, n, sd = _mean_ci(x)
            row[f"{g}_n"], row[f"{g}_mean"] = n, m
            row[f"{g}_sd"], row[f"{g}_lo"], row[f"{g}_hi"] = sd, lo, hi
            samples[g] = x.dropna().to_numpy()
        if len(groups) == 2:
            a, b = samples[groups[0]], samples[groups[1]]
            if len(a) >= 3 and len(b) >= 3:
                u, p = mannwhitneyu(a, b, alternative="two-sided")
                row["mwu_p"] = p
                # rank-biserial: 1 => group0 always higher
                row["rank_biserial"] = 2 * u / (len(a) * len(b)) - 1
            else:
                row["mwu_p"], row["rank_biserial"] = np.nan, np.nan
        rows.append(row)
    out = pd.DataFrame(rows)
    ok = out["mwu_p"].notna()
    out["mwu_p_fdr"] = np.nan
    out.loc[ok, "mwu_p_fdr"] = multipletests(out.loc[ok, "mwu_p"], method="fdr_bh")[1]
    return out


# ---------------------------------------------------------------------------
# clinical validity
# ---------------------------------------------------------------------------
EE_COLS = ["ee_mean", "ee_early", "ee_late", "ee_slope", "ee_decline"]


def validity_panel(ee_subject: pd.DataFrame, meta: pd.DataFrame, nih: pd.DataFrame,
                   source: str, min_n: int = 8, n_boot: int = 2000) -> tuple:
    """Spearman panel of E-E summaries against each requested NIH score.

    For every (E-E summary x NIH score) pair: raw Spearman with bootstrap 95%
    CI, partial Spearman against the performance variables (completion time,
    rmsd), and partial Spearman against age at the PPT visit.  BH-FDR is
    applied within each `type` family over the five DISTINCT instruments
    (the two composite restatements are excluded from the family, then given
    their own composite row's q-value)."""
    perf = perf_summaries(meta)
    df = ee_subject.merge(perf, on="participant_id")
    df = df.merge(nih.reset_index(), on="participant_id", how="left")
    cohort = meta.groupby("participant_id")["Cohort"].first()
    df["Cohort"] = df["participant_id"].map(cohort)

    perf_cols = [c for c in ("ct_mean", "rmsd_mean", "success_rate") if c in df.columns
                 and df[c].notna().any()]

    rows = []
    for short in ORDER:
        for ee_col in EE_COLS:
            sub = df.dropna(subset=[ee_col, short])
            if len(sub) < min_n:
                print(f"[{source}/{short}/{ee_col}] skipped: n={len(sub)}")
                continue
            base = {"source": source, "measure": short, "label": LABEL[short],
                    "predictor": ee_col}

            r = bootstrap_spearman(sub[ee_col], sub[short], n_boot=n_boot)
            rows.append({**base, "type": "raw", **r})

            s2 = sub.dropna(subset=perf_cols)
            if len(s2) >= min_n:
                r = partial_spearman(s2[ee_col], s2[short],
                                     s2[perf_cols].to_numpy(float), n_boot=n_boot)
                rows.append({**base, "type": "partial_vs_perf", **r})

            s3 = sub.dropna(subset=["age_ppt"])
            if len(s3) >= min_n:
                r = partial_spearman(s3[ee_col], s3[short],
                                     s3[["age_ppt"]].to_numpy(float), n_boot=n_boot)
                rows.append({**base, "type": "partial_vs_age", **r})

        # LOSO ridge probes for this outcome
        sub = df.dropna(subset=EE_COLS + perf_cols + [short])
        if len(sub) >= min_n:
            for name, cols in (("perf_only", perf_cols), ("ee_only", EE_COLS),
                               ("perf_plus_ee", perf_cols + EE_COLS)):
                r = loso_probe(sub[cols], sub[short])
                rows.append({"source": source, "measure": short, "label": LABEL[short],
                             "predictor": name, "type": "loso_probe", **r})

    out = pd.DataFrame(rows)
    if out.empty:
        return out, df

    # FDR within each type family, over distinct instruments only
    out["p_fdr"] = np.nan
    distinct = ~out["measure"].map(RESTATES_COMPOSITE).astype(bool)
    for typ in ("raw", "partial_vs_perf", "partial_vs_age"):
        fam = (out["type"] == typ) & distinct & out["p"].notna()
        if fam.any():
            out.loc[fam, "p_fdr"] = multipletests(out.loc[fam, "p"], method="fdr_bh")[1]
    # composite restatements inherit the composite's q-value for the same predictor
    comp = out[(out["measure"] == "composite")].set_index(["type", "predictor"])["p_fdr"]
    restate = out["measure"].map(RESTATES_COMPOSITE).astype(bool)
    out.loc[restate, "p_fdr"] = pd.MultiIndex.from_frame(
        out.loc[restate, ["type", "predictor"]]).map(comp)
    out["significant_fdr05"] = out["p_fdr"] < 0.05
    return out, df
