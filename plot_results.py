import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def plot_metric(df: pd.DataFrame, metric: str, output_dir: str, title_suffix: str = "") -> None:
    subset = df.copy()
    if "algorithm" in subset.columns:
        subset = subset[subset["algorithm"].isin(["CH", "RH", "LoadAwareCH"])]
    if subset.empty:
        return
    pivot = subset.pivot_table(index="node_count", columns="algorithm", values=metric, aggfunc="mean")
    ax = pivot.plot(marker="o", linewidth=2)
    ax.set_title(f"{metric} {title_suffix}".strip())
    ax.set_xlabel("Node count")
    ax.set_ylabel(metric)
    ax.grid(True, alpha=0.3)
    ax.figure.tight_layout()
    output_path = os.path.join(output_dir, f"{metric}.png")
    ax.figure.savefig(output_path, dpi=180)
    plt.close(ax.figure)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate research plots from the distributed hashing CSV output.")
    parser.add_argument("csv_path", nargs="?", default="results.csv", help="Path to the experiment CSV file")
    parser.add_argument("--output-dir", default="plots", help="Folder to save figures in")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    df = pd.read_csv(args.csv_path)
    if df.empty:
        raise ValueError(f"No rows available in {args.csv_path}")

    metrics = [
        "mean_load",
        "max_load",
        "jain_fairness",
        "p95_node_load",
        "p99_node_load",
        "routing_time_sec",
        "routing_ops_sec",
        "membership_add_remap_pct",
        "membership_remove_remap_pct",
        "load_reassignment_pct",
    ]

    for metric in metrics:
        if metric in df.columns:
            plot_metric(df, metric, args.output_dir, title_suffix="by algorithm")

    grouped = df.groupby(["workload", "algorithm", "node_count"], as_index=False).mean(numeric_only=True)
    for workload_value in sorted(grouped["workload"].dropna().unique()):
        subset = grouped[grouped["workload"] == workload_value]
        if subset.empty:
            continue
        pivot = subset.pivot(index="node_count", columns="algorithm", values="jain_fairness")
        ax = pivot.plot(marker="o", title=f"Jain fairness for {workload_value}")
        ax.set_xlabel("Node count")
        ax.set_ylabel("Jain fairness")
        ax.grid(True, alpha=0.3)
        ax.figure.tight_layout()
        ax.figure.savefig(os.path.join(args.output_dir, f"jain_fairness_{workload_value}.png"), dpi=180)
        plt.close(ax.figure)

    alpha_df = df[df["sensitivity_type"] == "alpha"] if "sensitivity_type" in df.columns else pd.DataFrame()
    if not alpha_df.empty:
        pivot = alpha_df.pivot_table(index="alpha", columns="algorithm", values="load_reassignment_pct", aggfunc="mean")
        ax = pivot.plot(marker="o", title="Alpha sensitivity")
        ax.set_xlabel("Alpha")
        ax.set_ylabel("Load reassignment (%)")
        ax.grid(True, alpha=0.3)
        ax.figure.tight_layout()
        ax.figure.savefig(os.path.join(args.output_dir, "alpha_sensitivity.png"), dpi=180)
        plt.close(ax.figure)

    candidate_df = df[df["sensitivity_type"] == "candidate"] if "sensitivity_type" in df.columns else pd.DataFrame()
    if not candidate_df.empty:
        pivot = candidate_df.pivot_table(index="candidate_count", columns="algorithm", values="load_reassignment_pct", aggfunc="mean")
        ax = pivot.plot(marker="o", title="Candidate-count sensitivity")
        ax.set_xlabel("Candidate count")
        ax.set_ylabel("Load reassignment (%)")
        ax.grid(True, alpha=0.3)
        ax.figure.tight_layout()
        ax.figure.savefig(os.path.join(args.output_dir, "candidate_sensitivity.png"), dpi=180)
        plt.close(ax.figure)

    print(f"Generated plots in {args.output_dir}")


if __name__ == "__main__":
    main()
