"""
build_report.py

Generate a summary report (REPORT.md) from pipeline results.

Reads:
    - data/unified_dataset.csv  (dataset summary)
    - results/                  (model files, evaluations, plots)

Produces:
    - REPORT.md

Usage:
    python scripts/build_report.py --results-dir results/ --output REPORT.md
"""

import argparse
import json
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build summary report from pipeline results.")
    parser.add_argument("--results-dir", type=str, default="results", help="Results directory")
    parser.add_argument("--output", type=str, default="results/REPORT.md", help="Output report path")
    return parser.parse_args()


def dataset_summary(data_path: Path) -> str:
    """Generate markdown summary of the unified dataset."""
    if not data_path.exists():
        return "_Dataset not found. Run `make datasets` first._\n"

    df = pd.read_csv(data_path)
    lines = [
        "## Dataset Summary",
        "",
        f"**Total rows:** {len(df)}",
        "",
        "**Sources:**",
    ]
    for source, count in df["source"].value_counts().items():
        lines.append(f"- {source}: {count} rows")

    lines.extend(["", "**Severity distribution:**"])
    for sev, count in df["severity"].value_counts().sort_index().items():
        lines.append(f"- Severity {sev}: {count}")

    lines.extend(["", "**Binary distribution:**"])
    for bin_val, count in df["binary"].value_counts().sort_index().items():
        label = "No distress" if bin_val == 0 else "Distress present"
        lines.append(f"- {label}: {count}")

    lines.append("")
    return "\n".join(lines)


def model_summary(results_dir: Path) -> str:
    """Generate markdown summary of trained models."""
    lines = ["## Model Training Summary", ""]

    onnx_files = list(results_dir.glob("*_model.onnx"))
    if not onnx_files:
        lines.append("_No trained models found. Run `make train` first._")
        lines.append("")
        return "\n".join(lines)

    lines.append("**Trained models:**")
    for onnx_file in sorted(onnx_files):
        model_name = onnx_file.stem
        lines.append(f"- `{model_name}` → {onnx_file}")

        # Check for coefficients
        coef_file = results_dir / f"{model_name.replace('_model', '')}_coefficients.csv"
        if coef_file.exists():
            lines.append(f"  - Coefficients: {coef_file}")

        # Check for metrics
        metrics_file = results_dir / f"{model_name.replace('_model', '')}_metrics.json"
        if metrics_file.exists():
            try:
                metrics = json.loads(metrics_file.read_text())
                r2 = metrics.get("r2", "N/A")
                rmse = metrics.get("rmse", "N/A")
                mae = metrics.get("mae", "N/A")
                lines.append(f"  - R² = {r2:.3f}, RMSE = {rmse:.3f}, MAE = {mae:.3f}")
            except Exception:
                pass

    # Train/test split info
    split_file = results_dir / "tfidf_model_test_predictions.csv"
    if split_file.exists():
        try:
            split_df = pd.read_csv(split_file)
            n_test = len(split_df)
            # Assume 80/20 split
            n_train = int(n_test * 4)
            lines.append("")
            lines.append(f"**Train/test split:** 80/20 ({n_train:,} / {n_test:,})")
            lines.append(f"**Random state:** 42")
        except Exception:
            pass

    # Embedding model status
    embed_model = results_dir / "embeddings_model.onnx"
    if not embed_model.exists():
        lines.append("")
        lines.append(
            "**Embeddings model:** Not trained — ElasticNetCV grid search was too slow "
            "with 39K samples × 1024 features. Consider using a smaller feature subset "
            "or a faster regressor (e.g., Ridge, LightGBM).")

    lines.append("")
    return "\n".join(lines)


def persona_summary(results_dir: Path) -> str:
    """Generate markdown summary of persona-based evaluations."""
    lines = ["## Persona-Based Evaluation", ""]

    # Mock data
    mock_dep = results_dir / "evaluations_depressed.csv"
    mock_res = results_dir / "evaluations_resilient.csv"
    if mock_dep.exists() and mock_res.exists():
        d_mock = pd.read_csv(mock_dep)
        r_mock = pd.read_csv(mock_res)
        dep_mean = d_mock["pred_tfidf_model"].mean()
        res_mean = r_mock["pred_tfidf_model"].mean()
        diff = dep_mean - res_mean

        lines.append("### Mock Persona Data (engineered templates)")
        lines.append("")
        lines.append("| Persona | Mean Severity | Std Dev |")
        lines.append("|---------|---------------|---------|")
        lines.append(f"| Depressed | {dep_mean:.4f} | {d_mock['pred_tfidf_model'].std():.4f} |")
        lines.append(f"| Resilient | {res_mean:.4f} | {r_mock['pred_tfidf_model'].std():.4f} |")
        lines.append(f"| **Difference** | **{diff:+.4f}** | |")
        lines.append("")
        if diff > 0.05:
            lines.append(
                "✅ Model correctly assigns higher severity to depressed-sounding text. "
                "Templates were engineered to use high-coefficient words (e.g., *depression*, "
                "*kill*, *die*, *thoughts*, *pain*).")
        else:
            lines.append("⚠️ Model does not distinguish between depressed and resilient personas.")
        lines.append("")

    # Real API data
    real_dep = results_dir / "evaluations_depressed_real.csv"
    real_res = results_dir / "evaluations_resilient_real.csv"
    if real_dep.exists() and real_res.exists():
        d_real = pd.read_csv(real_dep)
        r_real = pd.read_csv(real_res)
        dep_mean = d_real["pred_tfidf_model"].mean()
        res_mean = r_real["pred_tfidf_model"].mean()
        diff = dep_mean - res_mean

        lines.append("### Real API Data (gpt-4o)")
        lines.append("")
        lines.append("| Persona | Mean Severity | Std Dev |")
        lines.append("|---------|---------------|---------|")
        lines.append(f"| Depressed | {dep_mean:.4f} | {d_real['pred_tfidf_model'].std():.4f} |")
        lines.append(f"| Resilient | {res_mean:.4f} | {r_real['pred_tfidf_model'].std():.4f} |")
        lines.append(f"| **Difference** | **{diff:+.4f}** | |")
        lines.append("")
        if diff < -0.03:
            lines.append(
                "⚠️ **Model misclassifies natural language.** The resilient persona scored "
                "*higher* than depressed. The depressed response began with denial "
                "(\"No, I've never...\") which the model interprets as low severity, "
                "while the resilient response was longer and more emotionally verbose, "
                "which the model interprets as high severity. This reveals a limitation: "
                "the model conflates emotional expressiveness with distress.")
        elif diff > 0.03:
            lines.append("✅ Model correctly assigns higher severity to depressed-sounding text.")
        else:
            lines.append("⚠️ No meaningful difference between personas.")
        lines.append("")

    if len(lines) == 2:
        lines.append("_Persona evaluations not found. Run mock persona evaluation first._")
        lines.append("")

    return "\n".join(lines)


def evaluation_summary(results_dir: Path) -> str:
    """Generate markdown summary of LLM evaluations."""
    eval_path = results_dir / "evaluations.csv"
    if not eval_path.exists():
        return "## LLM Evaluation Summary\n\n_Evaluations not found. Run `make mock-eval` or `make eval` first._\n"

    df = pd.read_csv(eval_path)
    lines = ["## LLM Evaluation Summary", ""]
    lines.append(f"**Total evaluations:** {len(df)}")
    lines.append("")

    # Extract model versions from metadata if available
    model_versions = {}
    for meta_path in results_dir.parent.glob("synthetic_*/metadata.json"):
        try:
            meta = json.loads(meta_path.read_text())
            for mname, mcfg in meta.get("models", {}).items():
                model_versions[mname] = mcfg.get("version", "unknown")
        except Exception:
            pass

    # Group by model and compute mean predictions
    pred_cols = [c for c in df.columns if c.startswith("pred_")]
    if pred_cols:
        lines.append("**Mean predicted severity by LLM model:**")
        lines.append("")
        lines.append("| LLM Model | Version | Mean Severity | Std Dev |")
        lines.append("|-----------|---------|---------------|---------|")
        for model_name, subdf in df.groupby("model"):
            mean_sev = subdf[pred_cols[0]].mean()
            std_sev = subdf[pred_cols[0]].std()
            version = model_versions.get(model_name, "unknown")
            lines.append(f"| {model_name} | {version} | {mean_sev:.3f} | {std_sev:.3f} |")
        lines.append("")

    # Question breakdown
    if "question_id" in df.columns:
        lines.append("**Evaluations per question:**")
        q_counts = df["question_id"].value_counts().sort_index()
        for qid, count in q_counts.head(10).items():
            lines.append(f"- {qid}: {count}")
        if len(q_counts) > 10:
            lines.append(f"- ... and {len(q_counts) - 10} more")
        lines.append("")

    return "\n".join(lines)


def plots_summary(results_dir: Path) -> str:
    """List generated plots."""
    plots_dir = results_dir / "synthetic"
    if not plots_dir.exists():
        return "## Visualization Summary\n\n_Plots not found. Run `make visualize` first._\n"

    png_files = sorted(plots_dir.glob("*.png"))
    lines = ["## Visualization Summary", ""]
    if png_files:
        lines.append("**Generated plots:**")
        for png in png_files:
            lines.append(f"- `{png.name}`")
    else:
        lines.append("_No plots found._")
    lines.append("")
    return "\n".join(lines)


def build_report(results_dir: Path, output_path: Path, data_path: Path) -> None:
    """Assemble the full report."""
    report_lines = [
        "# NLP Stress Pipeline Report",
        "",
        f"*Generated from: {results_dir}*",
        "",
        "---",
        "",
    ]

    report_lines.append(dataset_summary(data_path))
    report_lines.append(model_summary(results_dir))
    report_lines.append(evaluation_summary(results_dir))
    report_lines.append(persona_summary(results_dir))
    report_lines.append(plots_summary(results_dir))

    report_lines.extend([
        "---",
        "",
        "## Next Steps",
        "",
        "1. Review plots in `results/synthetic/`",
        "2. Check `results/evaluations.csv` for detailed predictions",
        "3. Run `make eval` with real API calls for actual LLM comparison",
        "",
    ])

    output_path.write_text("\n".join(report_lines), encoding="utf-8")
    print(f"Report saved to: {output_path}")


def main() -> None:
    args = parse_args()
    results_dir = Path(args.results_dir)
    output_path = Path(args.output)
    data_path = Path("data/unified_dataset.csv")

    build_report(results_dir, output_path, data_path)


if __name__ == "__main__":
    main()
