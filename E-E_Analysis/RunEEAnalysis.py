"""
Exploration-Exploitation (E-E) analysis reported in Section IV of the paper.

Reads the exported trial-level embeddings, takes the E-E ratio as exported,
and produces the statistics the manuscript reports:

  E-E tables        per-trial E-E with early/late phase labels
  M1  LMM contrasts the four clinical uses in Section IV-B
                    (progress monitoring, within-subject limb comparison,
                     adapting therapy, cohort contrasts)
  M2  clinical      E-E vs ABILHAND-Kids and the NIH Toolbox Flanker task,
                    Spearman with bootstrap CI, BH-FDR, partial correlations
                    controlling completion time / RMSD, and LOSO ridge probes
  Fig 8             external clinical validity figure

    python E-E_Analysis/RunEEAnalysis.py \
        --msl  embeddings/embedded_trajectories_msl.csv \
        --ucpmp embeddings/embedded_trajectories_ucpmp.csv \
        --dataset-dir Dataset

The E-E ratio is taken verbatim from the `exploration_exploitation_ratio`
column rather than recomputed, so these numbers follow the estimator used for
the reported results.
"""

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ClinicalValidity import run_clinical_panel
from EEMetric import add_phase_labels, subject_summaries
from StatsModels import run_stats

HERE = os.path.dirname(os.path.abspath(__file__))
RATIO_SRC = "exploration_exploitation_ratio"
TAG = "oldmodel"

# Only the two instruments reported in Fig. 8 of the manuscript.
REPORTED_MEASURES = ("abilhand_logits", "nih_flanker", "mabc_standard")


def load_embeddings(path, name):
    df = pd.read_csv(path)
    if name == "UCPMP":
        df = fix_ucpmp_metadata(df)
    return df


def fix_ucpmp_metadata(df, hand_ability_csv=None):
    """
    Two metadata corrections to the exported MP table. These are facts about the
    dataset rather than model outputs, and analyses that condition on them are
    wrong without the fix.

      task_type   the export hard-codes 1 ("Bimanual") for all trials; the
                  paretic (UCP) / nondominant (TD) indicator is merged back in
      is_success  the export hard-codes 1; MP has no success criterion, so the
                  -1 sentinel is restored and success terms stay off
    """
    df = df.copy()
    if hand_ability_csv and os.path.exists(hand_ability_csv):
        hand = pd.read_csv(hand_ability_csv)
        key = "participant_id"
        if key in hand.columns and "affected_side" in hand.columns and "trial_hand" in df.columns:
            df = df.merge(hand[[key, "affected_side"]], on=key, how="left")
            df["task_type"] = (df["trial_hand"].astype(str).str.strip()
                               == df["affected_side"].astype(str).str.strip()).astype(int)
            print(f"[UCPMP] task_type restored: {df.task_type.value_counts().to_dict()}")
    df["is_success"] = -1
    return df


def build_ee_tables(msl_path, ucpmp_path, out_results, hand_ability_csv=None):
    tables = {}
    for name, path in (("MSL", msl_path), ("UCPMP", ucpmp_path)):
        if not path or not os.path.exists(path):
            print(f"[{name}] skipped - {path} not found")
            continue
        df = load_embeddings(path, name)
        if name == "UCPMP" and hand_ability_csv:
            df = fix_ucpmp_metadata(df, hand_ability_csv)
        df["ee_ratio"] = pd.to_numeric(df[RATIO_SRC], errors="coerce")
        df["row_id"] = np.arange(len(df))

        # phases follow the sequences the E-E ratio was computed over:
        # MSL by participant x task_type, MP by participant (both hands pooled)
        group_cols = ("participant_id", "task_type") if name == "MSL" else ("participant_id",)
        df = add_phase_labels(df, group_cols=group_cols)

        out = os.path.join(out_results, f"{name}_ee_table_{TAG}.csv")
        df.to_csv(out, index=False)
        print(f"[{name}] wrote {out}  (ee_ratio non-null {df.ee_ratio.notna().mean():.1%})")
        tables[name] = df
    return tables


def m1_lmm(tables, out_results):
    stats, effect = run_stats(tables, os.path.join(out_results, "m1_lmm_contrasts.csv"))
    print(stats[["dataset", "term", "beta_std", "p", "p_fdr", "significant_fdr05"]].to_string(index=False))
    return stats, effect


def m2_clinical(tables, out_results, dataset_dir):
    frames = []
    for name, ee in tables.items():
        subj = subject_summaries(ee)
        res, merged = run_clinical_panel(subj, ee, name, dataset_dir, test_pids=None)
        if res.empty:
            continue
        frames.append(res)
        for measure, df in merged.items():
            if measure in REPORTED_MEASURES:
                df.to_csv(os.path.join(out_results, f"{name}_subject_level_{measure}.csv"), index=False)
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    out = out[out["measure"].isin(REPORTED_MEASURES)]
    out.to_csv(os.path.join(out_results, "m2_clinical_validity.csv"), index=False)
    print(out[out["type"] == "raw"][["dataset", "measure", "predictor", "rho", "p", "p_fdr", "n"]]
          .to_string(index=False))
    return out


def figure_clinical_validity(m2, out_figs):
    """Fig. 8: Spearman correlations with 95% CI against the two instruments."""
    raw = m2[(m2["type"] == "raw") & (m2["measure"].isin(("abilhand_logits", "nih_flanker")))]
    if raw.empty:
        return
    predictors = ["ee_mean", "ee_early", "ee_late", "ee_slope", "ee_decline"]
    titles = {"abilhand_logits": "ABILHAND-Kids", "nih_flanker": "NIH Toolbox Flanker"}
    measures = [m for m in ("abilhand_logits", "nih_flanker") if m in raw["measure"].unique()]

    fig, axes = plt.subplots(1, len(measures), figsize=(5.0 * len(measures), 3.4), sharey=True)
    for ax, measure in zip(np.atleast_1d(axes), measures):
        sub = raw[raw["measure"] == measure].set_index("predictor").reindex(predictors).dropna(subset=["rho"])
        y = np.arange(len(sub))
        sig = sub["p_fdr"] < 0.05
        ax.errorbar(sub["rho"], y,
                    xerr=[sub["rho"] - sub["ci_low"], sub["ci_high"] - sub["rho"]],
                    fmt="o", color="0.3", ecolor="0.6", capsize=3)
        ax.scatter(sub["rho"][sig], y[sig.values], color="crimson", zorder=3)
        ax.axvline(0, color="0.7", linestyle="--", linewidth=1)
        ax.set_yticks(y); ax.set_yticklabels(sub.index)
        ax.set_xlabel(r"Spearman $\rho$")
        ax.set_title(titles.get(measure, measure))
    fig.tight_layout()
    path = os.path.join(out_figs, "fig_m2_clinical_validity.png")
    fig.savefig(path, dpi=300)
    plt.close(fig)
    print("wrote", path)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--msl", default=None, help="MSL embedded-trajectory CSV")
    ap.add_argument("--ucpmp", default=None, help="MP embedded-trajectory CSV")
    ap.add_argument("--dataset-dir", default="Dataset", help="Directory holding the clinical score CSVs")
    ap.add_argument("--hand-ability", default=None, help="UCPMP_HandAbility.csv, for the task_type fix")
    ap.add_argument("--out", default=HERE)
    ap.add_argument("--skip", nargs="*", default=[], help="any of: m1 m2 fig")
    args = ap.parse_args()

    out_results = os.path.join(args.out, "results"); os.makedirs(out_results, exist_ok=True)
    out_figs = os.path.join(args.out, "figures"); os.makedirs(out_figs, exist_ok=True)

    tables = build_ee_tables(args.msl, args.ucpmp, out_results, args.hand_ability)
    if not tables:
        raise SystemExit("no embedding tables loaded - pass --msl and/or --ucpmp")

    if "m1" not in args.skip:
        m1_lmm(tables, out_results)
    m2 = m2_clinical(tables, out_results, args.dataset_dir) if "m2" not in args.skip else pd.DataFrame()
    if "fig" not in args.skip and len(m2):
        figure_clinical_validity(m2, out_figs)


if __name__ == "__main__":
    main()
