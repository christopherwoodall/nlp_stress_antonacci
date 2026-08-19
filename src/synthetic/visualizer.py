"""
visualizer.py

Blog-quality comparison plots for synthetic TESI evaluation results.

Produces:
    - Bar chart: average predicted severity per LLM model
    - Box plot: distribution of predicted severity per model
    - Heatmap: predicted severity by question × model
    - Scatter: confidence vs predicted severity

All plots saved as high-DPI PNGs.
"""

from pathlib import Path
from typing import Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


# --- Style setup ---

sns.set_theme(style="whitegrid", context="talk")
plt.rcParams["figure.dpi"] = 150
plt.rcParams["savefig.dpi"] = 300
plt.rcParams["figure.figsize"] = (12, 8)


# --- Plotting functions ---

def plot_average_severity(df: pd.DataFrame, output_dir: Path) -> None:
    """
    Bar chart of average predicted severity per model.
    """
    pred_cols = [c for c in df.columns if c.startswith("pred_")]
    if not pred_cols:
        print("No prediction columns found (pred_*). Skipping bar chart.")
        return

    # Melt to long form for seaborn
    melted = df.melt(
        id_vars=["model", "question_id"],
        value_vars=pred_cols,
        var_name="predictor",
        value_name="severity",
    )

    # Group by actual generation model (the 'model' column) and predictor
    grouped = melted.groupby(["model", "predictor"])["severity"].mean().reset_index()

    fig, ax = plt.subplots(figsize=(12, 7))
    sns.barplot(
        data=grouped,
        x="model",
        y="severity",
        hue="predictor",
        palette="viridis",
        ax=ax,
    )
    ax.set_title("Average Predicted Severity by LLM Model", fontsize=16, fontweight="bold")
    ax.set_xlabel("LLM Model (Response Generator)", fontsize=13)
    ax.set_ylabel("Mean Predicted Severity", fontsize=13)
    ax.legend(title="Predictor Model", loc="upper right")
    plt.tight_layout()
    out_path = output_dir / "average_severity.png"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_severity_distribution(df: pd.DataFrame, output_dir: Path) -> None:
    """
    Box plot showing distribution of predicted severity per model.
    """
    pred_cols = [c for c in df.columns if c.startswith("pred_")]
    if not pred_cols:
        print("No prediction columns found. Skipping box plot.")
        return

    # Use the first prediction column for the distribution
    pred_col = pred_cols[0]

    fig, ax = plt.subplots(figsize=(12, 7))
    sns.boxplot(
        data=df,
        x="model",
        y=pred_col,
        palette="Set2",
        ax=ax,
    )
    ax.set_title(f"Predicted Severity Distribution ({pred_col})", fontsize=16, fontweight="bold")
    ax.set_xlabel("LLM Model", fontsize=13)
    ax.set_ylabel("Predicted Severity", fontsize=13)
    plt.tight_layout()
    out_path = output_dir / "severity_distribution.png"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_question_heatmap(df: pd.DataFrame, output_dir: Path) -> None:
    """
    Heatmap of average predicted severity by question × model.
    """
    pred_cols = [c for c in df.columns if c.startswith("pred_")]
    if not pred_cols:
        print("No prediction columns found. Skipping heatmap.")
        return

    # Use first predictor for heatmap
    pred_col = pred_cols[0]

    pivot = df.groupby(["question_id", "model"])[pred_col].mean().unstack()

    if pivot.empty:
        print("Not enough data for heatmap. Skipping.")
        return

    fig, ax = plt.subplots(figsize=(14, 10))
    sns.heatmap(
        pivot,
        annot=True,
        fmt=".2f",
        cmap="RdYlBu_r",
        center=pivot.values.mean(),
        linewidths=0.5,
        ax=ax,
    )
    ax.set_title("Predicted Severity by Question × Model", fontsize=16, fontweight="bold")
    ax.set_xlabel("LLM Model", fontsize=13)
    ax.set_ylabel("TESI Question ID", fontsize=13)
    plt.tight_layout()
    out_path = output_dir / "question_heatmap.png"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_confidence_scatter(df: pd.DataFrame, output_dir: Path) -> None:
    """
    Scatter plot of prediction confidence (std across predictors) vs mean severity.
    """
    pred_cols = [c for c in df.columns if c.startswith("pred_")]
    if len(pred_cols) < 2:
        print("Need >=2 predictors for confidence scatter. Skipping.")
        return

    df["mean_severity"] = df[pred_cols].mean(axis=1)
    df["std_severity"] = df[pred_cols].std(axis=1)

    fig, ax = plt.subplots(figsize=(12, 7))
    for model_name, subdf in df.groupby("model"):
        ax.scatter(
            subdf["mean_severity"],
            subdf["std_severity"],
            label=model_name,
            alpha=0.6,
            s=80,
        )

    ax.set_title("Prediction Confidence vs Mean Severity", fontsize=16, fontweight="bold")
    ax.set_xlabel("Mean Predicted Severity", fontsize=13)
    ax.set_ylabel("Std Dev Across Predictors (Uncertainty)", fontsize=13)
    ax.legend(title="LLM Model")
    plt.tight_layout()
    out_path = output_dir / "confidence_scatter.png"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_model_comparison_radar(df: pd.DataFrame, output_dir: Path) -> None:
    """
    Radar chart comparing models across question categories.
    """
    pred_cols = [c for c in df.columns if c.startswith("pred_")]
    if not pred_cols:
        return

    pred_col = pred_cols[0]

    # Group questions into categories based on prefix
    def question_category(qid):
        if qid.startswith("tesi_1"):
            return "Accidents / Disasters"
        elif qid.startswith("tesi_2"):
            return "Assault / Violence"
        elif qid.startswith("tesi_3"):
            return "Family Conflict"
        elif qid.startswith("tesi_4"):
            return "Community Violence"
        elif qid.startswith("tesi_5"):
            return "Unwanted Touch"
        elif qid.startswith("tesi_6"):
            return "Other Worst Event"
        return "Other"

    df["category"] = df["question_id"].apply(question_category)

    grouped = df.groupby(["model", "category"])[pred_col].mean().reset_index()
    pivot = grouped.pivot(index="model", columns="category", values=pred_col).fillna(0)

    if pivot.empty or len(pivot.columns) < 3:
        print("Not enough categories for radar chart. Skipping.")
        return

    categories = list(pivot.columns)
    N = len(categories)

    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))

    colors = plt.cm.tab10(np.linspace(0, 1, len(pivot)))

    for idx, (model_name, row) in enumerate(pivot.iterrows()):
        values = row.values.tolist()
        values += values[:1]
        ax.plot(angles, values, "o-", linewidth=2, label=model_name, color=colors[idx])
        ax.fill(angles, values, alpha=0.15, color=colors[idx])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, size=11)
    ax.set_title("Model Comparison by TESI Category", fontsize=16, fontweight="bold", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
    plt.tight_layout()
    out_path = output_dir / "model_radar.png"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


# --- Main visualization pipeline ---

def visualize(
    evaluations_csv: Union[str, Path],
    output_dir: Union[str, Path],
) -> None:
    """
    Generate all comparison plots from an evaluations CSV.

    Parameters
    ----------
    evaluations_csv : str or Path
        Path to the evaluations CSV produced by evaluator.py.
    output_dir : str or Path
        Directory to save PNG plots.
    """
    evaluations_csv = Path(evaluations_csv)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(evaluations_csv)
    print(f"Loaded {len(df)} evaluations")

    plot_average_severity(df, output_dir)
    plot_severity_distribution(df, output_dir)
    plot_question_heatmap(df, output_dir)
    plot_confidence_scatter(df, output_dir)
    plot_model_comparison_radar(df, output_dir)

    print(f"\nAll plots saved to: {output_dir}")


# --- CLI entry ---

def main_visualize(args=None):
    """CLI entry point for visualize subcommand."""
    import argparse

    parser = argparse.ArgumentParser(description="Visualize synthetic evaluation results")
    parser.add_argument("--evaluations", required=True, help="Path to evaluations CSV")
    parser.add_argument("--output-dir", required=True, help="Directory to save plots")
    parsed = parser.parse_args(args)

    visualize(evaluations_csv=parsed.evaluations, output_dir=parsed.output_dir)


if __name__ == "__main__":
    main_visualize()
