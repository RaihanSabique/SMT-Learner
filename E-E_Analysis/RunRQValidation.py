"""RQ validation experiments on the saved SMT-Learner E-E metrics (MSL + MP).

Reproduces the analyses of
  E-E_Analysis/RQ_validation/QR1_RQ3_micro_vs_macro_and_distribution.ipynb
  E-E_Analysis/RQ_validation/QR2_MSL_vs_PM_and_Case_Study.ipynb
against the clean repo's E-E tables, using the same figure style and
configuration.  Everything lands in RQ_Validation/.

  RQ1  micro (E-E ratio) vs macro (speed, accuracy) group separation
  RQ2  early vs late learning dynamics, MP vs MSL
  RQ3  group distributions and trial-progression slope distributions

MonkeyPR panels from the originals are absent: that dataset was removed from
the revision.  See analysis/rq_validation.py for the other porting notes.
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from RQValidation import (
    MICRO_COL, ROOT, MSL_MACRO_BIN, MSL_MICRO_BIN, MP_MACRO_BIN, MP_MICRO_BIN,
    EARLY_LATE_BIN, N_BOOT, BOOT_SEED, MSL_TRIALS_PER_SESSION, MSL_TOTAL_TRIALS,
    MP_TRIALS_PER_SESSION, MP_TOTAL_TRIALS, MP_PALETTE, MSL_PALETTE, MP_STYLES,
    MSL_STYLES, EARLY_LATE_COLORS, EARLY_LATE_STYLES, SLOPE_COLORS,
    apply_style, save_fig, ci95_t, fmt_mean_ci, welch_ttest, paired_ttest,
    load_ee_tables, binned_mean, binned_range, early_late_per_unit,
    bin_trials_by_unit, ols_slope, bootstrap_slopes,
)

OUT_DIR = os.path.join(ROOT, "RQ_Validation")
METRICS = ["speed", "rmsd_norm", MICRO_COL]
METRIC_LABELS = {"speed": "Speed", "rmsd_norm": "Accuracy (RMSD/dist)", MICRO_COL: "E–E Ratio"}


# --------------------------------------------------------------------------
# RQ1
# --------------------------------------------------------------------------
def _plot_metric(ax, df, y_label, palette, linestyles, add_macro_headroom=False, headroom_frac=0.2):
    for g, gdf in df.groupby("group"):
        gdf = gdf.sort_values("x")
        ax.plot(gdf["x"], gdf["mean"], label=str(g), color=palette.get(g, "black"),
                linestyle=linestyles.get(g, "-"), linewidth=2.2)
    if add_macro_headroom and df["mean"].notna().any():
        data_min, data_max = float(df["mean"].min()), float(df["mean"].max())
        yrange = data_max - data_min
        pad = (yrange * headroom_frac) if yrange > 0 else (abs(data_max) * headroom_frac if data_max else 1.0)
        cur_ymin, cur_ymax = ax.get_ylim()
        ax.set_ylim(cur_ymin, max(cur_ymax, data_max + pad))
    ax.set_ylabel(y_label)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="upper right", frameon=False)
    sns.despine(ax=ax)


def _unit_means(df, metrics=METRICS):
    return (df.groupby(["participant_id", "group"], as_index=False)[metrics].mean()
              .rename(columns={"participant_id": "unit_id"}))


def _ttest_table(units, dataset, group_a, group_b, subset=""):
    rows = []
    for m in METRICS:
        a = units.loc[units["group"] == group_a, m]
        b = units.loc[units["group"] == group_b, m]
        ma, la, ha, na = ci95_t(a)
        mb, lb, hb, nb = ci95_t(b)
        t = welch_ttest(a, b)
        rows.append({
            "dataset": dataset, "subset": subset, "metric": m,
            "group_A": group_a, "n_A": na, "mean_95CI_A": fmt_mean_ci(ma, la, ha),
            "group_B": group_b, "n_B": nb, "mean_95CI_B": fmt_mean_ci(mb, lb, hb),
            "t": t["t"], "p": t["p"], "cohen_d": t["cohen_d"],
            "significant_005": bool(np.isfinite(t["p"]) and t["p"] < 0.05),
        })
    return pd.DataFrame(rows)


def run_rq1(msl, mp, out_dir):
    print("\n=== RQ1: micro E-E vs macro speed/accuracy ===")
    msl_uni = msl[msl["task_type"] == 0]
    msl_bi = msl[msl["task_type"] == 1]

    tables = [
        _ttest_table(_unit_means(msl_uni), "MSL", "TERM", "PRETERM", "Unimanual (task_type=0)"),
        _ttest_table(_unit_means(msl_bi), "MSL", "TERM", "PRETERM", "Bimanual (task_type=1)"),
        _ttest_table(_unit_means(mp), "MP", "TD", "UCP", "all trials"),
    ]
    ttests = pd.concat(tables, ignore_index=True)
    ttests.to_csv(os.path.join(out_dir, "RQ1_ttests.csv"), index=False)
    print(ttests[["dataset", "subset", "metric", "t", "p", "cohen_d", "significant_005"]].to_string(index=False))

    # sequential binned plots, one figure per dataset
    for name, df, group_col, palette, styles, sess_trials, total, macro_bin, micro_bin, title in [
        ("MP", mp, "group", MP_PALETTE, MP_STYLES, MP_TRIALS_PER_SESSION, MP_TOTAL_TRIALS,
         MP_MACRO_BIN, MP_MICRO_BIN, "MP: TD vs UCP"),
        ("MSL", msl_uni, "group", MSL_PALETTE, MSL_STYLES, MSL_TRIALS_PER_SESSION, MSL_TOTAL_TRIALS,
         MSL_MACRO_BIN, MSL_MICRO_BIN, "MSL (Unimanual): Term vs Preterm"),
    ]:
        panels = []
        for m in METRICS:
            bin_size = micro_bin if m == MICRO_COL else macro_bin
            panels.append(binned_mean(df, group_col, "participant_id", "session_no", "trial_no",
                                      sess_trials, total, bin_size, m))
        fig, axes = plt.subplots(1, 3, figsize=(10.0, 2.9), sharex=False)
        for ax, m, panel in zip(axes, METRICS, panels):
            _plot_metric(ax, panel, METRIC_LABELS[m], palette, styles,
                         add_macro_headroom=(m != MICRO_COL))
            ax.set_xlabel("Trial")
        fig.suptitle(title, y=1.06, fontweight="bold")
        fig.tight_layout()
        save_fig(fig, out_dir, f"RQ1_{name}_macro{macro_bin}_micro{micro_bin}.png", dpi=300)
    return ttests


# --------------------------------------------------------------------------
# RQ2
# --------------------------------------------------------------------------
def run_rq2(msl, mp, out_dir):
    print("\n=== RQ2: early vs late learning ===")
    msl_uni = msl[msl["task_type"] == 0]
    fontsize = 16
    fig, axes = plt.subplots(1, 2, figsize=(8, 4), sharey=True)
    plt.subplots_adjust(wspace=0.15)

    for ax, which, xlabel, title in [(axes[0], "early", "Trial (Early 24)", "Early Learning"),
                                     (axes[1], "late", "Trial (Late 24)", "Late Learning")]:
        mp_b = binned_range(mp, "group", "participant_id", "session_no", "trial_no",
                            MP_TRIALS_PER_SESSION, MP_TOTAL_TRIALS, EARLY_LATE_BIN, MICRO_COL, which)
        ms_b = binned_range(msl_uni, "group", "participant_id", "session_no", "trial_no",
                            MSL_TRIALS_PER_SESSION, MSL_TOTAL_TRIALS, EARLY_LATE_BIN, MICRO_COL, which)
        for g, gdf in mp_b.groupby("group"):
            ax.plot(gdf["x"], gdf["mean"], label=f"MP {g}", color=EARLY_LATE_COLORS["mp"],
                    linestyle=EARLY_LATE_STYLES.get(g, "-"), linewidth=2)
        for g, gdf in ms_b.groupby("group"):
            ax.plot(gdf["x"], gdf["mean"], label=f"MSL {g}", color=EARLY_LATE_COLORS["msl_unimanual"],
                    linestyle=EARLY_LATE_STYLES.get(g, "-"), linewidth=2)
        ax.set_title(title, fontsize=fontsize, loc="left", fontweight="bold")
        ax.set_xlabel(xlabel, fontsize=fontsize)
        ax.set_xticks(np.unique(np.concatenate([mp_b["x"].unique(), ms_b["x"].unique()])))
        ax.tick_params(axis="both", labelsize=fontsize - 2)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    axes[0].set_ylabel("E–E Ratio", fontsize=fontsize)
    axes[1].set_ylabel("")
    axes[1].tick_params(axis="y", which="both", left=True, labelleft=True)
    handles, labels = axes[1].get_legend_handles_labels()
    axes[1].legend(handles, labels, frameon=False, fontsize=fontsize - 2, loc="upper right")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    save_fig(fig, out_dir, "RQ2_MP_vs_MSL_Early_Late.png", dpi=600)

    # statistics: per-unit early vs late (paired), and between-cohort deltas
    rows = []
    per_unit = {}
    for name, df, sess_trials, total in [("MP", mp, MP_TRIALS_PER_SESSION, MP_TOTAL_TRIALS),
                                         ("MSL", msl_uni, MSL_TRIALS_PER_SESSION, MSL_TOTAL_TRIALS)]:
        eu = early_late_per_unit(df, "participant_id", "session_no", "trial_no", sess_trials, total)
        per_unit[name] = eu
        for g, gdf in eu.groupby("group"):
            me, le, he, ne = ci95_t(gdf["early"])
            ml, ll, hl, nl = ci95_t(gdf["late"])
            pt = paired_ttest(gdf["late"], gdf["early"])
            rows.append({"dataset": name, "group": g, "n_units": ne,
                         "early_mean_95CI": fmt_mean_ci(me, le, he),
                         "late_mean_95CI": fmt_mean_ci(ml, ll, hl),
                         "delta_late_minus_early": float(gdf["delta"].mean()),
                         "paired_t": pt["t"], "paired_p": pt["p"], "cohen_dz": pt["cohen_dz"]})
        groups = sorted(eu["group"].unique())
        if len(groups) == 2:
            a = eu.loc[eu["group"] == groups[0], "delta"]
            b = eu.loc[eu["group"] == groups[1], "delta"]
            w = welch_ttest(a, b)
            rows.append({"dataset": name, "group": f"delta {groups[0]} vs {groups[1]}",
                         "n_units": w["n1"] + w["n2"], "early_mean_95CI": "", "late_mean_95CI": "",
                         "delta_late_minus_early": float(a.mean() - b.mean()),
                         "paired_t": w["t"], "paired_p": w["p"], "cohen_dz": w["cohen_d"]})
    stats_df = pd.DataFrame(rows)
    stats_df.to_csv(os.path.join(out_dir, "RQ2_early_late_stats.csv"), index=False)
    print(stats_df.to_string(index=False))
    return stats_df, per_unit


# --------------------------------------------------------------------------
# RQ3
# --------------------------------------------------------------------------
def run_rq3(msl, mp, out_dir):
    print("\n=== RQ3: group and slope distributions ===")
    msl_uni = msl[msl["task_type"] == 0]

    # --- group-wise E-E box plot (participant-level means) ---
    parts, summary_rows = [], []
    for label_map, df in [({"TD": "TD", "UCP": "UCP"}, mp),
                          ({"TERM": "Term", "PRETERM": "Preterm"}, msl_uni)]:
        units = df.groupby(["participant_id", "group"], as_index=False)[MICRO_COL].mean()
        for raw, label in label_map.items():
            vals = units.loc[units["group"] == raw, MICRO_COL].dropna()
            mean, lo, hi, n = ci95_t(vals)
            summary_rows.append({"group": label, "mean": mean, "ci_low": lo, "ci_high": hi, "n_units": n})
            if n:
                parts.append(pd.DataFrame({"group": label, MICRO_COL: vals.values}))
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(os.path.join(out_dir, "RQ3_group_EE_summary.csv"), index=False)
    print(summary_df.to_string(index=False))

    box_df = pd.concat(parts, ignore_index=True)
    box_order = ["Term", "Preterm", "UCP", "TD"]
    box_df["group"] = pd.Categorical(box_df["group"], categories=box_order, ordered=True)
    box_df = box_df.dropna(subset=[MICRO_COL]).sort_values("group")
    fig = plt.figure(figsize=(5.8, 3.0))
    sns.boxplot(data=box_df, x=MICRO_COL, y="group", color="0.5", showfliers=False, linewidth=1.0)
    plt.title("Group distribution of E-E Ratio")
    plt.xlabel("E-E Ratio")
    plt.ylabel("Group")
    sns.despine()
    plt.tight_layout()
    save_fig(fig, out_dir, "RQ3_group_EE_boxplot.png", dpi=600)

    # --- bootstrap slope distributions ---
    binned_by_cohort, binned_by_dataset = {}, {}
    for cohort, raw, df in [("UCP", "UCP", mp), ("TD", "TD", mp),
                            ("Term", "TERM", msl_uni), ("Preterm", "PRETERM", msl_uni)]:
        sub = df[df["group"] == raw]
        binned_by_cohort[cohort] = bin_trials_by_unit(sub, ["participant_id", "session_no"], "trial_no")
    for name, df in [("MP", mp), ("MSL", msl_uni)]:
        binned_by_dataset[name] = bin_trials_by_unit(df, ["participant_id", "session_no"], "trial_no")

    def _slopes(binned_map):
        rows, boots = [], []
        for key, bdf in binned_map.items():
            slope = ols_slope(bdf)
            boot = bootstrap_slopes(bdf, n_boot=N_BOOT, seed=BOOT_SEED)
            boot["cohort"] = key
            boots.append(boot)
            lo, hi = (np.percentile(boot["slope"], [2.5, 97.5]) if not boot.empty else (np.nan, np.nan))
            rows.append({"group": key, "slope": slope, "boot_mean": float(boot["slope"].mean()),
                         "boot_ci_low": lo, "boot_ci_high": hi,
                         "n_units": int(bdf["unit_id"].nunique()),
                         "excludes_zero": bool(np.isfinite(lo) and np.isfinite(hi) and (lo > 0 or hi < 0))})
        return pd.DataFrame(rows), pd.concat(boots, ignore_index=True)

    cohort_summary, cohort_boot = _slopes(binned_by_cohort)
    dataset_summary, dataset_boot = _slopes(binned_by_dataset)
    slope_summary = pd.concat([cohort_summary.assign(level="cohort"),
                               dataset_summary.assign(level="dataset")], ignore_index=True)
    slope_summary.to_csv(os.path.join(out_dir, "RQ3_slope_summary.csv"), index=False)
    print(slope_summary.to_string(index=False))

    def _density(boot_all, order, colors, filename, add_zero_line=False):
        fig = plt.figure(figsize=(4.8, 2.8))
        for key in [k for k in order if k in boot_all["cohort"].unique()]:
            b = boot_all[boot_all["cohort"] == key]
            if b.empty:
                continue
            mean_slope = b["slope"].mean()
            lo, hi = np.percentile(b["slope"], [2.5, 97.5])
            sns.kdeplot(b["slope"], label=f"{key} (Mean = {mean_slope:.4f})", linewidth=2.0,
                        fill=True, alpha=0.25, color=colors.get(key, "0.5"))
            plt.axvline(mean_slope, color=colors.get(key, "0.5"), linestyle="--", alpha=0.8, linewidth=2.2)
            plt.axvline(lo, color=colors.get(key, "0.5"), linestyle=":", alpha=0.6, linewidth=1.8)
            plt.axvline(hi, color=colors.get(key, "0.5"), linestyle=":", alpha=0.6, linewidth=1.8)
        if add_zero_line:
            plt.axvline(0, color="r", linestyle="-", alpha=0.5, linewidth=2.5, label="No Change")
        plt.xlabel("Slope (Change in E-E Ratio per Percentage of Trials)")
        plt.ylabel("Density")
        plt.legend()
        plt.xticks(rotation=20)
        plt.tight_layout()
        save_fig(fig, out_dir, filename, dpi=600)

    _density(cohort_boot, ["UCP", "TD", "Term", "Preterm"], SLOPE_COLORS,
             "RQ3_slope_distribution_cohortwise.png")
    ds_colors = dict(zip(["MP", "MSL"], sns.color_palette("Set2", n_colors=2)))
    _density(dataset_boot, ["MP", "MSL"], ds_colors,
             "RQ3_slope_distribution_datasetwise.png", add_zero_line=True)
    return summary_df, slope_summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="oldmodel",
                    help="E-E table tag, matching {MSL,UCPMP}_ee_table_<tag>.csv")
    ap.add_argument("--out", default=OUT_DIR)
    ap.add_argument("--results-dir", default=None,
                    help="directory holding {MSL,UCPMP}_ee_table_<tag>.csv (default: results/)")
    args = ap.parse_args()

    apply_style()
    os.makedirs(args.out, exist_ok=True)
    msl, mp = load_ee_tables(tag=args.tag, results_dir=args.results_dir)
    print(f"loaded MSL {msl.shape} ({msl.participant_id.nunique()} participants), "
          f"MP {mp.shape} ({mp.participant_id.nunique()} participants)")

    ttests = run_rq1(msl, mp, args.out)
    el_stats, _ = run_rq2(msl, mp, args.out)
    group_summary, slope_summary = run_rq3(msl, mp, args.out)

    res_dir = args.results_dir or "results"
    manifest = {
        "source_tag": args.tag,
        "ee_tables": [os.path.join(res_dir, f"MSL_ee_table_{args.tag}.csv"),
                      os.path.join(res_dir, f"UCPMP_ee_table_{args.tag}.csv")],
        "n_participants": {"MSL": int(msl.participant_id.nunique()), "MP": int(mp.participant_id.nunique())},
        "config": {
            "msl_macro_bin": MSL_MACRO_BIN, "msl_micro_bin": MSL_MICRO_BIN,
            "mp_macro_bin": MP_MACRO_BIN, "mp_micro_bin": MP_MICRO_BIN,
            "early_late_bin": EARLY_LATE_BIN, "n_boot": N_BOOT, "boot_seed": BOOT_SEED,
            "msl_trials_per_session": MSL_TRIALS_PER_SESSION, "msl_total_trials": MSL_TOTAL_TRIALS,
            "mp_trials_per_session": MP_TRIALS_PER_SESSION, "mp_total_trials": MP_TOTAL_TRIALS,
        },
        "rq1_significant": ttests.loc[ttests["significant_005"], ["dataset", "subset", "metric", "p"]]
                                 .to_dict("records"),
        "notes": [
            "MonkeyPR panels from the original notebooks are omitted (dataset removed).",
            "MSL uses Unimanual (task_type==0); the RQ1 table also reports Bimanual.",
            "actual_dist merged from the raw Dataset CSVs; speed=actual_dist/completion_time, "
            "accuracy=rmsd/actual_dist.",
        ],
    }
    with open(os.path.join(args.out, "RQ_validation_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nwrote {os.path.join(args.out, 'RQ_validation_manifest.json')}")
    print("RQ validation complete")


if __name__ == "__main__":
    main()
