"""
lda.py

LDA Topic Components and Internalizing Outcomes —
translated from analysis/LDA_Analyses.Rmd

Replicates:
  1. PCA on LDA topic distributions (500 topics → 20 PCs)
  2. Elastic net on 20 PC scores (simple CV, alpha=0.5)
  3. Retain significant PCs and export coefficients
  4. Compute PC word loadings for retained components
  5. Predicted vs observed scatterplot

Expected data files (relative to --data-dir):
  - participant_topic_distribution.csv  (topic proportions)
  - ysr_internalizing_t3.csv            (outcomes)
  - topic_word_matrix.csv               (topic-word distributions for wordclouds)
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import ElasticNetCV

from ._base import (
    compute_metrics,
    save_metrics,
    save_coefficients_csv,
    save_model_onnx,
    load_model_onnx,
    predict_onnx,
)
from ._plots import (
    plot_predicted_vs_observed,
    plot_feature_importance,
    plot_scree,
)


OUTCOME_COL = "internalizing_t3"
ID_COL = "id"


def _load_data(data_dir: Path) -> tuple:
    """Load topic distributions, outcomes, and topic-word matrix."""
    topics_path = data_dir / "participant_topic_distribution.csv"
    ysr_path = data_dir / "ysr_internalizing_t3.csv"
    word_path = data_dir / "topic_word_matrix.csv"

    for p in (topics_path, ysr_path):
        if not p.exists():
            print(
                f"ERROR: Expected data file not found: {p}\n"
                "The LDA analysis requires clinical data files. "
                "Contact the Stanford NAPL (PI: Ian Gotlib) for data access.",
                file=sys.stderr,
            )
            raise FileNotFoundError(p)

    df_topics = pd.read_csv(topics_path)
    df_ysr = pd.read_csv(ysr_path)

    # merge
    df_merged = df_topics.merge(df_ysr, on=ID_COL, how="inner")

    df_words = None
    if word_path.exists():
        df_words = pd.read_csv(word_path)

    return df_merged, df_words


def _get_topic_cols(df: pd.DataFrame) -> list:
    """Identify topic_ columns."""
    return [c for c in df.columns if c.startswith("topic_")]


# --- 1. PCA on Topic Distributions ---

def run_pca_and_enet(df: pd.DataFrame, topic_cols: list, output_dir: Path, random_state: int = 1234) -> dict:
    """
    PCA on topic distributions, then elastic net on PC scores.

    Replicates the core LDA analysis from the R notebook.
    """
    print("\n--- PCA on Topic Distributions ---")

    X_topics = df[topic_cols].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_topics)

    pca = PCA()
    pc_scores = pca.fit_transform(X_scaled)

    # Retain first 20 PCs (as in the R notebook)
    n_pcs = min(20, pc_scores.shape[1])
    pc_scores_20 = pc_scores[:, :n_pcs]
    pc_labels = [f"PC{i+1}" for i in range(n_pcs)]

    pc_df = pd.DataFrame(pc_scores_20, columns=pc_labels)
    pc_df[ID_COL] = df[ID_COL].values
    pc_df.to_csv(output_dir / "topic_pcs_20.csv", index=False)

    print(f"  Retained {n_pcs} PCs explaining {pca.explained_variance_ratio_[:n_pcs].sum() * 100:.1f}% variance")

    plot_scree(
        pca,
        output_path=output_dir / "lda_pca_scree_plot.pdf",
        title="Scree Plot of Principal Components (LDA Topics)",
    )

    # --- Elastic Net on 20 PCs ---
    print("\n--- Elastic Net on PC Scores ---")

    y = df[OUTCOME_COL].values
    valid = ~np.isnan(y)
    X_pcs = pc_scores_20[valid]
    y_valid = y[valid]

    # Simple CV with alpha=0.5 (as in R notebook)
    model = ElasticNetCV(
        l1_ratio=0.5,
        cv=5,
        random_state=random_state,
        n_jobs=-1,
        max_iter=10000,
    )
    model.fit(X_pcs, y_valid)

    print(f"  Best lambda: {model.alpha_:.6f}")
    print(f"  CV R²: {model.score(X_pcs, y_valid):.4f}")

    # Extract and save coefficients
    coef_df = pd.DataFrame({
        "term": pc_labels,
        "estimate": model.coef_,
    })
    coef_df = coef_df[coef_df["estimate"] != 0].sort_values(by="estimate", key=abs, ascending=False)
    coef_df.to_csv(output_dir / "lda_elastic_net_coefficients.csv", index=False)
    print(f"  {len(coef_df)} non-zero PC coefficients retained.")

    # Predicted vs observed on training data (the R notebook does not do nested CV here)
    y_pred = model.predict(X_pcs)
    plot_predicted_vs_observed(
        y_valid, y_pred,
        output_path=output_dir / "lda_en_predicted_vs_observed.pdf",
        title="LDA Topic PC Model: Predicted vs Observed",
    )

    # Save model
    save_model_onnx(model, pc_labels, output_dir / "lda_model.onnx")
    save_coefficients_csv(model, pc_labels, output_dir / "lda_coefficients.csv")

    return {
        "pca": pca,
        "pc_scores": pc_scores_20,
        "model": model,
        "coef_df": coef_df,
    }


# --- 2. PC Word Loadings ---

def compute_pc_word_loadings(
    df_words: pd.DataFrame,
    topic_cols: list,
    pca,
    retained_pcs: list,
    output_dir: Path,
) -> dict:
    """
    Compute how each word loads onto retained PCs.

    Replicates the R wordcloud data preparation. Returns a dict of
    DataFrames (one per retained PC) that can be used for wordclouds
    or other visualizations.
    """
    if df_words is None or pca is None:
        print("  Skipping word loadings (topic_word_matrix.csv not found)")
        return {}

    print("\n--- PC Word Loadings ---")

    word_vec = df_words["word"].values
    topic_word_mat = df_words[topic_cols].values  # (n_words, n_topics)
    pc_rotation = pca.components_  # (n_components, n_topics)

    results = {}
    for pc_label in retained_pcs:
        pc_index = int(pc_label.replace("PC", "")) - 1
        loads = topic_word_mat @ pc_rotation[pc_index, :]  # (n_words,)
        load_df = pd.DataFrame({
            "word": word_vec,
            "loading": loads,
        })
        load_df["abs_loading"] = load_df["loading"].abs()
        load_df = load_df.sort_values("abs_loading", ascending=False)
        results[pc_label] = load_df
        load_df.head(100).to_csv(
            output_dir / f"lda_word_loadings_{pc_label}.csv", index=False
        )
        print(f"  {pc_label}: top loading word = '{load_df.iloc[0]['word']}' ({load_df.iloc[0]['loading']:.3f})")

    return results


# --- 3. Full CV Analysis Orchestrator ---

def run_cv_analysis(data_dir: Path, output_dir: Path, random_state: int = 1234) -> dict:
    """Run the complete LDA cross-validation analysis."""
    output_dir.mkdir(parents=True, exist_ok=True)
    print("=" * 60)
    print("LDA Analysis — Cross-Validation")
    print("=" * 60)

    df_merged, df_words = _load_data(data_dir)
    topic_cols = _get_topic_cols(df_merged)
    print(f"Loaded {len(df_merged)} participants with {len(topic_cols)} topic features.")

    en_results = run_pca_and_enet(df_merged, topic_cols, output_dir, random_state=random_state)

    # Compute word loadings for retained PCs
    retained_pcs = en_results["coef_df"]["term"].tolist()
    if df_words is not None:
        compute_pc_word_loadings(
            df_words, topic_cols, en_results["pca"], retained_pcs, output_dir
        )

    # Feature importance plot
    plot_feature_importance(
        en_results["coef_df"],
        output_path=output_dir / "lda_en_predictors.pdf",
        title="Retained LDA Topic PC Predictors",
    )

    print("\nLDA CV analysis complete.")
    return en_results


# --- 4. Train Final Model ---

def train_final_model(data_dir: Path, output_dir: Path, random_state: int = 1234):
    """Train final model on full data and save."""
    output_dir.mkdir(parents=True, exist_ok=True)
    print("=" * 60)
    print("LDA Analysis — Train Final Model")
    print("=" * 60)

    df_merged, _ = _load_data(data_dir)
    topic_cols = _get_topic_cols(df_merged)

    # PCA
    X_topics = df_merged[topic_cols].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_topics)
    pca = PCA(n_components=20)
    X_pcs = pca.fit_transform(X_scaled)

    y = df_merged[OUTCOME_COL].values
    valid = ~np.isnan(y)

    model = ElasticNetCV(
        l1_ratio=0.5,
        cv=5,
        random_state=random_state,
        n_jobs=-1,
        max_iter=10000,
    )
    model.fit(X_pcs[valid], y[valid])

    print(f"  Best lambda: {model.alpha_:.6f}")

    # Save PCA + scaler + model
    import joblib
    joblib.dump(pca, output_dir / "lda_pca.pkl")
    joblib.dump(scaler, output_dir / "lda_scaler.pkl")

    pc_labels = [f"PC{i+1}" for i in range(X_pcs.shape[1])]
    save_model_onnx(model, pc_labels, output_dir / "lda_model.onnx")
    save_coefficients_csv(model, pc_labels, output_dir / "lda_coefficients.csv")
    print("Final model saved.")
    return model


# --- 5. Test Model ---

def test_model(model_path: Path, test_data_path: Path, output_dir: Path) -> dict:
    """Evaluate saved ONNX model on test data."""
    output_dir.mkdir(parents=True, exist_ok=True)
    print("=" * 60)
    print("LDA Analysis — Test Model")
    print("=" * 60)

    df_test = pd.read_csv(test_data_path)
    if OUTCOME_COL not in df_test.columns:
        raise ValueError(f"Test data must contain '{OUTCOME_COL}' column")

    topic_cols = _get_topic_cols(df_test)
    X_topics = df_test[topic_cols].values

    # Load preprocessing
    scaler_path = model_path.parent / "lda_scaler.pkl"
    pca_path = model_path.parent / "lda_pca.pkl"
    if scaler_path.exists() and pca_path.exists():
        import joblib
        scaler = joblib.load(scaler_path)
        pca = joblib.load(pca_path)
        X_pcs = pca.transform(scaler.transform(X_topics))
    else:
        scaler = StandardScaler()
        pca = PCA(n_components=20)
        X_pcs = pca.fit_transform(scaler.fit_transform(X_topics))

    session = load_model_onnx(model_path)
    y_pred = predict_onnx(session, X_pcs)
    y_true = df_test[OUTCOME_COL].values

    metrics = compute_metrics(y_true, y_pred)
    print(f"  Test R² = {metrics['r2']:.4f}")
    save_metrics(metrics, output_dir / "lda_test_metrics.json")

    plot_predicted_vs_observed(
        y_true, y_pred,
        output_path=output_dir / "lda_test_predicted_vs_observed.pdf",
        title="Test Set Performance",
    )
    return metrics


# --- 6. Predict ---

def predict(model_path: Path, input_path: Path, output_path: Path) -> pd.DataFrame:
    """Make predictions on new data."""
    print("=" * 60)
    print("LDA Analysis — Predict")
    print("=" * 60)

    df_input = pd.read_csv(input_path)
    topic_cols = _get_topic_cols(df_input)
    X_topics = df_input[topic_cols].values

    scaler_path = model_path.parent / "lda_scaler.pkl"
    pca_path = model_path.parent / "lda_pca.pkl"
    if scaler_path.exists() and pca_path.exists():
        import joblib
        scaler = joblib.load(scaler_path)
        pca = joblib.load(pca_path)
        X_pcs = pca.transform(scaler.transform(X_topics))
    else:
        scaler = StandardScaler()
        pca = PCA(n_components=20)
        X_pcs = pca.fit_transform(scaler.fit_transform(X_topics))

    session = load_model_onnx(model_path)
    y_pred = predict_onnx(session, X_pcs)

    df_out = df_input.copy()
    df_out["Predicted_Internalizing"] = y_pred
    df_out.to_csv(output_path, index=False)
    print(f"Predictions saved to: {output_path}")
    return df_out


# --- CLI ---

def main():
    parser = argparse.ArgumentParser(description="LDA Analysis")
    parser.add_argument("--mode", choices=["cv", "train", "test", "predict"], required=True)
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--output-dir", type=str, default="results")
    parser.add_argument("--model-path", type=str)
    parser.add_argument("--test-data", type=str)
    parser.add_argument("--input", type=str)
    parser.add_argument("--output", type=str)
    parser.add_argument("--random-state", type=int, default=1234)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.mode == "cv":
        run_cv_analysis(data_dir, output_dir, random_state=args.random_state)
    elif args.mode == "train":
        train_final_model(data_dir, output_dir, random_state=args.random_state)
    elif args.mode == "test":
        if not args.model_path or not args.test_data:
            parser.error("--mode test requires --model-path and --test-data")
        test_model(Path(args.model_path), Path(args.test_data), output_dir)
    elif args.mode == "predict":
        if not args.model_path or not args.input or not args.output:
            parser.error("--mode predict requires --model-path, --input, and --output")
        predict(Path(args.model_path), Path(args.input), Path(args.output))


if __name__ == "__main__":
    main()
