import argparse
import os
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def analyze_and_plot(file_path: str, save_plot: bool = True):
    if not os.path.exists(file_path):
        print(f"[ERROR] Log file '{file_path}' not found.")
        return

    # Read tab-separated log file
    df = pd.read_csv(file_path, sep="\t")

    # Map boolean success to readable string categories
    df["Outcome"] = df["Success"].apply(
        lambda x: "Success" if str(x).strip().lower() == "true" else "Failure"
    )

    # Color scheme & markers
    palette = {"Success": "#2ca02c", "Failure": "#d62728"}
    markers = {"Success": "o", "Failure": "X"}

    # Set up side-by-side plots
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # -------------------------------------------------------------
    # Plot 1: Object Initial Positions (X vs Y)
    # -------------------------------------------------------------
    sns.scatterplot(
        data=df,
        x="Obj_X",
        y="Obj_Y",
        hue="Outcome",
        style="Outcome",
        palette=palette,
        markers=markers,
        s=120,
        alpha=0.85,
        ax=axes[0],
    )
    axes[0].set_title("Initial Object Positions (X vs Y)", fontsize=13, fontweight="bold")
    axes[0].set_xlabel("Object Position X (m)", fontsize=11)
    axes[0].set_ylabel("Object Position Y (m)", fontsize=11)
    axes[0].axis("equal")  # Keep 1:1 spatial aspect ratio

    # Add failure rate summary inside subplot 0
    fail_count = (df["Outcome"] == "Failure").sum()
    total = len(df)
    axes[0].text(
        0.03,
        0.03,
        f"Total: {total} | Failures: {fail_count}",
        transform=axes[0].transAxes,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
        fontsize=10,
    )

    # -------------------------------------------------------------
    # Plot 2: Target/Scale Initial Positions (X vs Y)
    # -------------------------------------------------------------
    sns.scatterplot(
        data=df,
        x="Target_X",
        y="Target_Y",
        hue="Outcome",
        style="Outcome",
        palette=palette,
        markers=markers,
        s=120,
        alpha=0.85,
        ax=axes[1],
    )
    axes[1].set_title("Target / Scale Positions (X vs Y)", fontsize=13, fontweight="bold")
    axes[1].set_xlabel("Target Position X (m)", fontsize=11)
    axes[1].set_ylabel("Target Position Y (m)", fontsize=11)
    axes[1].axis("equal")  # Keep 1:1 spatial aspect ratio

    plt.suptitle("Rollout Spatial Distribution & Policy Success Analysis", fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()

    if save_plot:
        out_filename = os.path.splitext(file_path)[0] + "_spatial_plot.png"
        plt.savefig(out_filename, dpi=300, bbox_inches="tight")
        print(f"[INFO] Plot saved to: {os.path.abspath(out_filename)}")

    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot rollout initial positions colored by task success.")
    parser.add_argument(
        "--file", type=str, default="docs/rollout_analysis.txt", help="Path to rollout analysis TSV file."
    )
    args = parser.parse_args()

    analyze_and_plot(args.file)