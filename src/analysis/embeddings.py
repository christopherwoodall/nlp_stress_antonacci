"""
embeddings.py

Sentence Embedding Analyses — translated from analysis/SentenceEmbedding_Analysis.Rmd

Replicates:
  1. Elastic net nested CV on RoBERTa embeddings
  2. Export 768-dim coefficient vector for Python dot-product projection
  3. UMAP + HDBSCAN clustering on embeddings
  4. Cluster-wise comparisons (t-tests, violin plots)
  5. Linear / logistic models on UMAP dimensions
  6. Predicted vs observed scatterplot and feature importance plot

Expected data files (relative to --data-dir):
  - roberta_embeddings.csv       (768-dim embeddings + demographics + outcomes)
  - any_intDx_T1-TA.xlsx         (diagnosis outcomes)
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression, LogisticRegression

from ._base import (
    load_merged_data,
    nested_cv_elastic_net,
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
    plot_umap_scatter,
    plot_cluster_violin,
)


# --- Column definitions ---

DEMO_COLS = [
    "demo_Race.T1",
    "demo_Sex.T1",
    "demo_Age.T1",
    "demo_Parent_Edu.T1",
    "demo_INR.T1",
    "TESI_obj_sumsev.T1",
]

OUTCOME_COL = "YSR_Internalizing_Total.T3"
DX_COL = "T3_T4_any_intDx"
EMB_PREFIX = "emb_"


def _load_data(data_dir: Path) -> pd.DataFrame:
    """Load RoBERTa embeddings merged with diagnoses."""
    emb_path = data_dir / "roberta_embeddings.csv"
    dx_path = data_dir / "any_intDx_T1-TA.xlsx"

    for p in (emb_path, dx_path):
        if not p.exists():
            print(
                f"ERROR: Expected data file not found: {p}\n"
                "The embedding analysis requires clinical data files. "
                "Contact the Stanford NAPL (PI: Ian Gotlib) for data access.",
                file=sys.stderr,
            )
            raise FileNotFoundError(p)

    df_emb = pd.read_csv(emb_path)
    df_emb["ELS_ID"] = df_emb["ELS_ID"].astype(str)

    df_dx = pd.read_excel(dx_path)
    df_dx["ELS_ID"] = df_dx["ELS_ID"].astype(str)

    df_merged = df_emb.merge(
        df_dx[["ELS_ID", DX_COL]],
        on="ELS_ID",
        how="left",
    )
    return df_merged


def _get_emb_cols(df: pd.DataFrame) -> list:
    """Identify embedding columns (emb_0 ... emb_767)."""
    return [c for c in df.columns if c.startswith(EMB_PREFIX)]


# --- 1. Elastic Net Nested CV ---

def run_elastic_net_cv(df: pd.DataFrame, emb_cols: list, output_dir: Path, random_state: int = 300) -> dict:
    """Elastic net nested CV on RoBERTa embeddings."""
    print("\n--- Elastic Net Nested CV ---")

    y = df[OUTCOME_COL].values
    all_predictors = DEMO_COLS + emb_cols
    X_raw = df[all_predictors].copy()
    X = pd.get_dummies(X_raw, drop_first=True).values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    results = nested_cv_elastic_net(
        X_scaled, y, n_outer_folds=5, random_state=random_state
    )

    print(f"  Global R² = {results['global_r2']:.4f}")

    cv_df = pd.DataFrame({
        "ELS_ID": df["ELS_ID"].values,
        "Actual": y,
        "Predicted": results["cv_predictions"],
    })
    cv_df.to_csv(output_dir / "embeddings_cv_predictions.csv", index=False)

    save_metrics(
        {
            "global_r2": float(results["global_r2"]),
            "best_alpha_folds": results["best_alpha_folds"],
            "best_lambda_folds": results["best_lambda_folds"],
            "cv_mse": results["cv_mse"],
            "cv_rmse": results["cv_rmse"],
            "cv_mae": results["cv_mae"],
            "cv_r2": results["cv_r2"],
        },
        output_dir / "embeddings_elastic_net_metrics.json",
    )

    plot_predicted_vs_observed(
        y, results["cv_predictions"],
        output_path=output_dir / "embeddings_en_predicted_vs_observed.pdf",
        title="Model Performance: Contextualized Sentence Embeddings",
        subtitle="Predicted vs. Actual YSR Internalizing Scores from RoBERTa Features",
    )

    return results


# --- 2. Train Final Model + Export Coefficients ---

def train_final_model(data_dir: Path, output_dir: Path, random_state: int = 123):
    """Train final model on full data and export coefficients for dot product."""
    output_dir.mkdir(parents=True, exist_ok=True)
    print("=" * 60)
    print("Embeddings Analysis — Train Final Model")
    print("=" * 60)

    df = _load_data(data_dir)
    emb_cols = _get_emb_cols(df)

    y = df[OUTCOME_COL].values
    all_predictors = DEMO_COLS + emb_cols
    X_raw = df[all_predictors].copy()
    X = pd.get_dummies(X_raw, drop_first=True).values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    from ._base import train_final_elastic_net
    model = train_final_elastic_net(X_scaled, y, random_state=random_state)

    print(f"  Best alpha: {model.l1_ratio_:.3f}")
    print(f"  Best lambda: {model.alpha_:.6f}")

    import joblib
    joblib.dump(scaler, output_dir / "embeddings_scaler.pkl")

    feature_names = pd.get_dummies(X_raw, drop_first=True).columns.tolist()
    save_model_onnx(model, feature_names, output_dir / "embeddings_model.onnx")
    save_coefficients_csv(model, feature_names, output_dir / "embeddings_coefficients.csv")

    # --- Export embedding coefficients for Python dot product ---
    # The R notebook extracts only the emb_* coefficients and saves them
    # as roberta_embedding_coefficients.csv for the Python script to load.
    coef_series = pd.Series(model.coef_, index=feature_names)
    emb_coefs = coef_series[coef_series.index.str.startswith("emb_")]
    emb_coefs_df = pd.DataFrame({"estimate": emb_coefs.values}, index=emb_coefs.index)
    emb_coefs_df.to_csv(output_dir / "roberta_embedding_coefficients.csv")
    print(f"  Exported {len(emb_coefs)} embedding coefficients for dot-product projection.")

    print("Final model saved.")
    return model


# --- 3. UMAP + HDBSCAN Clustering ---

def run_umap_clustering(df: pd.DataFrame, emb_cols: list, output_dir: Path, random_state: int = 123) -> dict:
    """
    UMAP dimensionality reduction + HDBSCAN clustering on embeddings.

    Replicates the R UMAP analysis section.
    """
    print("\n--- UMAP + HDBSCAN Clustering ---")

    try:
        import umap
        from hdbscan import HDBSCAN
    except ImportError:
        print(
            "WARNING: umap-learn and/or hdbscan not installed. "
            "Skipping UMAP clustering. Install with: uv pip install umap-learn hdbscan",
            file=sys.stderr,
        )
        return {}

    X_emb = df[emb_cols].values

    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=15,
        min_dist=0.1,
        random_state=random_state,
    )
    umap_embedding = reducer.fit_transform(X_emb)

    clusterer = HDBSCAN(min_samples=24)
    clusters = clusterer.fit_predict(umap_embedding)

    # Build output dataframe
    umap_df = pd.DataFrame({
        "ELS_ID": df["ELS_ID"].values,
        "UMAP1": umap_embedding[:, 0],
        "UMAP2": umap_embedding[:, 1],
        "Cluster": clusters,
    })
    umap_df.to_csv(output_dir / "embeddings_umap_clusters.csv", index=False)

    # Merge back
    df_clusters = df.merge(umap_df, on="ELS_ID", how="left")

    print(f"  Found {len(set(clusters)) - (1 if -1 in clusters else 0)} clusters (+ noise points)")

    # Plot UMAP scatter
    plot_umap_scatter(
        umap_df,
        output_path=output_dir / "embeddings_umap_scatter.pdf",
        title="Semantic Map of Linguistic Styles",
        subtitle="UMAP projection revealing distinct clusters",
    )

    # Cluster violin plot
    if OUTCOME_COL in df_clusters.columns:
        plot_cluster_violin(
            df_clusters,
            outcome_col=OUTCOME_COL,
            cluster_col="Cluster",
            output_path=output_dir / "embeddings_cluster_violin.pdf",
        )

    return {"umap_df": umap_df, "df_clusters": df_clusters}


# --- 4. UMAP Dimension Models ---

def run_umap_models(df_clusters: pd.DataFrame, output_dir: Path) -> dict:
    """Linear and logistic models using UMAP dimensions as predictors."""
    print("\n--- UMAP Dimension Models ---")

    y = df_clusters[OUTCOME_COL].values
    covariates = [c for c in DEMO_COLS if c in df_clusters.columns]
    X_umap = pd.get_dummies(df_clusters[covariates + ["UMAP1"]], drop_first=True).values

    # Linear model with UMAP1
    model_lin = LinearRegression()
    model_lin.fit(X_umap, y)
    r2 = model_lin.score(X_umap, y)
    print(f"  Linear model with UMAP1: R² = {r2:.4f}")

    # Logistic model with UMAP1
    if DX_COL in df_clusters.columns and df_clusters[DX_COL].notna().any():
        y_dx = df_clusters[DX_COL].values
        valid = ~np.isnan(y_dx)
        if valid.sum() > 10:
            model_log = LogisticRegression(max_iter=1000)
            model_log.fit(X_umap[valid], y_dx[valid])
            print(f"  Logistic model with UMAP1: fitted on {valid.sum()} observations")

    return {}


# --- 5. Full CV Analysis Orchestrator ---

def run_cv_analysis(data_dir: Path, output_dir: Path, random_state: int = 300) -> dict:
    """Run the complete embedding cross-validation analysis."""
    output_dir.mkdir(parents=True, exist_ok=True)
    print("=" * 60)
    print("Embeddings Analysis — Cross-Validation")
    print("=" * 60)

    df = _load_data(data_dir)
    emb_cols = _get_emb_cols(df)
    print(f"Loaded {len(df)} participants with {len(emb_cols)} embedding dimensions.")

    en_results = run_elastic_net_cv(df, emb_cols, output_dir, random_state=random_state)

    # UMAP clustering
    cluster_results = run_umap_clustering(df, emb_cols, output_dir, random_state=123)
    if cluster_results:
        run_umap_models(cluster_results["df_clusters"], output_dir)

    # Feature importance from full-data model
    _train_and_plot_importance(df, emb_cols, output_dir, random_state=random_state)

    print("\nEmbeddings CV analysis complete.")
    return {
        "elastic_net": en_results,
        "clustering": cluster_results,
    }


def _train_and_plot_importance(df, emb_cols, output_dir, random_state=123):
    """Train on full data and plot feature importance."""
    from ._base import train_final_elastic_net

    y = df[OUTCOME_COL].values
    all_predictors = DEMO_COLS + emb_cols
    X_raw = df[all_predictors].copy()
    X = pd.get_dummies(X_raw, drop_first=True).values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = train_final_elastic_net(X_scaled, y, random_state=random_state)
    feature_names = pd.get_dummies(X_raw, drop_first=True).columns.tolist()

    coef_df = pd.DataFrame({
        "term": feature_names,
        "estimate": model.coef_,
    })
    plot_feature_importance(
        coef_df,
        output_path=output_dir / "embeddings_en_predictors.pdf",
        title="Key Predictors of Mental Health Problems",
        subtitle="Standardized Coefficients from Elastic Net Model",
    )


# --- 6. Test Model ---

def test_model(model_path: Path, test_data_path: Path, output_dir: Path) -> dict:
    """Evaluate saved ONNX model on test data."""
    output_dir.mkdir(parents=True, exist_ok=True)
    print("=" * 60)
    print("Embeddings Analysis — Test Model")
    print("=" * 60)

    df_test = pd.read_csv(test_data_path)
    if OUTCOME_COL not in df_test.columns:
        raise ValueError(f"Test data must contain '{OUTCOME_COL}' column")

    emb_cols = _get_emb_cols(df_test)
    all_predictors = DEMO_COLS + emb_cols
    X_raw = df_test[all_predictors].copy()
    X = pd.get_dummies(X_raw, drop_first=True).values

    scaler_path = model_path.parent / "embeddings_scaler.pkl"
    if scaler_path.exists():
        import joblib
        scaler = joblib.load(scaler_path)
        X = scaler.transform(X)
    else:
        X = StandardScaler().fit_transform(X)

    session = load_model_onnx(model_path)
    y_pred = predict_onnx(session, X)
    y_true = df_test[OUTCOME_COL].values

    metrics = compute_metrics(y_true, y_pred)
    print(f"  Test R² = {metrics['r2']:.4f}")
    save_metrics(metrics, output_dir / "embeddings_test_metrics.json")

    plot_predicted_vs_observed(
        y_true, y_pred,
        output_path=output_dir / "embeddings_test_predicted_vs_observed.pdf",
        title="Test Set Performance",
    )
    return metrics


# --- 7. Predict ---

def predict(model_path: Path, input_path: Path, output_path: Path) -> pd.DataFrame:
    """Make predictions on new data."""
    print("=" * 60)
    print("Embeddings Analysis — Predict")
    print("=" * 60)

    df_input = pd.read_csv(input_path)
    emb_cols = _get_emb_cols(df_input)
    all_predictors = DEMO_COLS + emb_cols
    X_raw = df_input[all_predictors].copy()
    X = pd.get_dummies(X_raw, drop_first=True).values

    scaler_path = model_path.parent / "embeddings_scaler.pkl"
    if scaler_path.exists():
        import joblib
        scaler = joblib.load(scaler_path)
        X = scaler.transform(X)
    else:
        X = StandardScaler().fit_transform(X)

    session = load_model_onnx(model_path)
    y_pred = predict_onnx(session, X)

    df_out = df_input.copy()
    df_out["Predicted_YSR_Internalizing"] = y_pred
    df_out.to_csv(output_path, index=False)
    print(f"Predictions saved to: {output_path}")
    return df_out


# --- CLI ---

def main():
    parser = argparse.ArgumentParser(description="Embeddings Analysis")
    parser.add_argument("--mode", choices=["cv", "train", "test", "predict"], required=True)
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--output-dir", type=str, default="results")
    parser.add_argument("--model-path", type=str)
    parser.add_argument("--test-data", type=str)
    parser.add_argument("--input", type=str)
    parser.add_argument("--output", type=str)
    parser.add_argument("--random-state", type=int, default=300)
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
