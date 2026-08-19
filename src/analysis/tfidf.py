"""
tfidf.py

TF-IDF Analyses — translated from analysis/TF-IDF_Analyses.Rmd

Replicates:
  1. Elastic net nested CV on TF-IDF features
  2. Bivariate correlations between TF-IDF words and YSR score
  3. PCA on TF-IDF scores → linear / logistic models on PCs
  4. PC loadings plot
  5. Predicted vs observed scatterplot and feature importance plot

Expected data files (relative to --data-dir):
  - tfidf_scores_Filtered_participantOnlySpeech.csv  (features)
  - any_intDx_T1-TA.xlsx                             (diagnoses)
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
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
    plot_scree,
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


def _load_data(data_dir: Path) -> pd.DataFrame:
    """Load TF-IDF features merged with outcomes and diagnoses."""
    feature_path = data_dir / "tfidf_scores_Filtered_participantOnlySpeech.csv"
    dx_path = data_dir / "any_intDx_T1-TA.xlsx"

    for p in (feature_path, dx_path):
        if not p.exists():
            print(
                f"ERROR: Expected data file not found: {p}\n"
                "The TF-IDF analysis requires clinical data files. "
                "Contact the Stanford NAPL (PI: Ian Gotlib) for data access.",
                file=sys.stderr,
            )
            raise FileNotFoundError(p)

    df_features = pd.read_csv(feature_path)
    df_features["ELS_ID"] = df_features["ELS_ID"].astype(str)

    df_dx = pd.read_excel(dx_path)
    df_dx["ELS_ID"] = df_dx["ELS_ID"].astype(str)

    # merge diagnoses
    df_merged = df_features.merge(
        df_dx[["ELS_ID", DX_COL]],
        on="ELS_ID",
        how="left",
    )
    return df_merged


def _get_feature_cols(df: pd.DataFrame) -> list:
    """Identify TF-IDF word columns (numeric, not ID/demographic/outcome)."""
    exclude = {"ELS_ID", OUTCOME_COL, DX_COL} | set(DEMO_COLS)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    feature_cols = [c for c in numeric_cols if c not in exclude]
    return feature_cols


# --- 1. Elastic Net Nested CV ---

def run_elastic_net_cv(df: pd.DataFrame, feature_cols: list, output_dir: Path, random_state: int = 123) -> dict:
    """Elastic net nested CV on TF-IDF features."""
    print("\n--- Elastic Net Nested CV ---")

    y = df[OUTCOME_COL].values
    all_predictors = DEMO_COLS + feature_cols
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
    cv_df.to_csv(output_dir / "tfidf_cv_predictions.csv", index=False)

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
        output_dir / "tfidf_elastic_net_metrics.json",
    )

    plot_predicted_vs_observed(
        y, results["cv_predictions"],
        output_path=output_dir / "tfidf_en_predicted_vs_observed.pdf",
        title="Predictive Performance of a Word-Level TF-IDF Model",
        subtitle="Predicted vs. Actual YSR Internalizing Scores",
    )

    return results


# --- 2. Bivariate Correlations ---

def run_bivariate_correlations(df: pd.DataFrame, feature_cols: list, output_dir: Path) -> pd.DataFrame:
    """
    Compute Pearson correlations between each TF-IDF word and YSR score.

    Replicates the R bivariate-correlations chunk.
    """
    print("\n--- Bivariate Correlations ---")

    y = df[OUTCOME_COL].values
    rows = []

    for word in feature_cols:
        x = df[word].values
        # skip if constant
        if np.std(x) == 0:
            continue
        r, p = stats.pearsonr(x, y)
        rows.append({"Word": word, "Correlation": r, "P_Value": p})

    corr_df = pd.DataFrame(rows)
    # FDR correction (Benjamini-Hochberg)
    from statsmodels.stats.multitest import multipletests
    if len(corr_df) > 0:
        _, corr_df["P_Value_FDR"], _, _ = multipletests(corr_df["P_Value"], method="fdr_bh")
        corr_df = corr_df.sort_values(by="Correlation", key=abs, ascending=False)

    corr_df.to_csv(output_dir / "tfidf_bivariate_correlations.csv", index=False)
    print(f"  Computed {len(corr_df)} word correlations.")
    return corr_df


# --- 3. PCA on TF-IDF ---

def run_pca_analysis(df: pd.DataFrame, feature_cols: list, output_dir: Path) -> dict:
    """PCA on TF-IDF features, then linear/logistic models on PCs."""
    print("\n--- PCA Analysis ---")

    X_tfidf = df[feature_cols].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_tfidf)

    pca = PCA()
    pc_scores = pca.fit_transform(X_scaled)

    pc_df = pd.DataFrame(pc_scores[:, :10], columns=[f"PC{i+1}" for i in range(10)])
    pc_df["ELS_ID"] = df["ELS_ID"].values
    pc_df.to_csv(output_dir / "tfidf_pc_scores.csv", index=False)

    plot_scree(
        pca,
        output_path=output_dir / "tfidf_pca_scree_plot.pdf",
        title="Scree Plot of Principal Components",
    )

    # PC loadings plot (top 10 per PC for first 3 PCs)
    _plot_pc_loadings(pca, feature_cols, output_dir)

    # Merge PCs back
    df_pcs = df.copy()
    df_pcs["PC1"] = pc_scores[:, 0]
    df_pcs["PC2"] = pc_scores[:, 1]
    df_pcs["PC3"] = pc_scores[:, 2]

    # Linear model with PC1+PC2+PC3
    y = df_pcs[OUTCOME_COL].values
    covariates = [c for c in DEMO_COLS if c in df_pcs.columns]
    X_pc = pd.get_dummies(df_pcs[covariates + ["PC1", "PC2", "PC3"]], drop_first=True).values
    model_pc = LinearRegression()
    model_pc.fit(X_pc, y)
    r2_pc = model_pc.score(X_pc, y)
    print(f"  Linear model with PC1+PC2+PC3: R² = {r2_pc:.4f}")

    # Logistic model with PC1
    if DX_COL in df_pcs.columns and df_pcs[DX_COL].notna().any():
        y_dx = df_pcs[DX_COL].values
        valid = ~np.isnan(y_dx)
        if valid.sum() > 10:
            X_pc1 = pd.get_dummies(df_pcs[covariates + ["PC1"]], drop_first=True).values
            model_logistic = LogisticRegression(max_iter=1000)
            model_logistic.fit(X_pc1[valid], y_dx[valid])
            print(f"  Logistic model with PC1: fitted on {valid.sum()} observations")

    return {"pca": pca, "pc_scores": pc_scores}


def _plot_pc_loadings(pca, feature_names, output_dir):
    """Plot top 10 loadings per PC for the first 3 PCs."""
    import matplotlib.pyplot as plt
    import seaborn as sns

    loadings = pca.components_[:3, :].T  # (n_features, 3)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    pc_labels = ["PC1: Function Words", "PC2: Interpersonal Focus", "PC3: Visceral Experience"]

    for i, ax in enumerate(axes):
        pc_loadings = pd.DataFrame({
            "Feature": feature_names,
            "Loading": loadings[:, i],
        })
        pc_loadings["AbsLoading"] = pc_loadings["Loading"].abs()
        top = pc_loadings.nlargest(10, "AbsLoading").sort_values("Loading")

        colors = top["Loading"].apply(lambda x: "#21908C" if x > 0 else "#FDE725")
        ax.barh(top["Feature"], top["Loading"], color=colors, alpha=0.8, edgecolor="black", linewidth=0.5)
        ax.set_title(pc_labels[i], fontsize=10, fontweight="bold")
        ax.set_xlabel("Loading Strength", fontsize=9)
        sns.despine(ax=ax)

    fig.suptitle("Top 10 Word Loadings for the First Three Principal Components", fontsize=12, fontweight="bold")
    fig.tight_layout()
    output_path = output_dir / "tfidf_pca_loadings.pdf"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot: {output_path}")


# --- 4. Full CV Analysis Orchestrator ---

def run_cv_analysis(data_dir: Path, output_dir: Path, random_state: int = 123) -> dict:
    """Run the complete TF-IDF cross-validation analysis."""
    output_dir.mkdir(parents=True, exist_ok=True)
    print("=" * 60)
    print("TF-IDF Analysis — Cross-Validation")
    print("=" * 60)

    df = _load_data(data_dir)
    feature_cols = _get_feature_cols(df)
    print(f"Loaded {len(df)} participants with {len(feature_cols)} TF-IDF features.")

    en_results = run_elastic_net_cv(df, feature_cols, output_dir, random_state=random_state)
    corr_results = run_bivariate_correlations(df, feature_cols, output_dir)
    pca_results = run_pca_analysis(df, feature_cols, output_dir)

    # Feature importance from full-data model
    _train_and_plot_importance(df, feature_cols, output_dir, random_state=random_state)

    print("\nTF-IDF CV analysis complete.")
    return {
        "elastic_net": en_results,
        "correlations": corr_results,
        "pca": pca_results,
    }


def _train_and_plot_importance(df, feature_cols, output_dir, random_state=123):
    """Train on full data and plot feature importance."""
    from ._base import train_final_elastic_net

    y = df[OUTCOME_COL].values
    all_predictors = DEMO_COLS + feature_cols
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
        output_path=output_dir / "tfidf_en_predictors.pdf",
        title="Top Word-Level Predictors of Internalizing Problems",
    )


# --- 5. Train Final Model ---

def train_final_model(data_dir: Path, output_dir: Path, random_state: int = 123):
    """Train final model on full data and save."""
    output_dir.mkdir(parents=True, exist_ok=True)
    print("=" * 60)
    print("TF-IDF Analysis — Train Final Model")
    print("=" * 60)

    df = _load_data(data_dir)
    feature_cols = _get_feature_cols(df)

    y = df[OUTCOME_COL].values
    all_predictors = DEMO_COLS + feature_cols
    X_raw = df[all_predictors].copy()
    X = pd.get_dummies(X_raw, drop_first=True).values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    from ._base import train_final_elastic_net
    model = train_final_elastic_net(X_scaled, y, random_state=random_state)

    print(f"  Best alpha: {model.l1_ratio_:.3f}")
    print(f"  Best lambda: {model.alpha_:.6f}")

    import joblib
    joblib.dump(scaler, output_dir / "tfidf_scaler.pkl")

    feature_names = pd.get_dummies(X_raw, drop_first=True).columns.tolist()
    save_model_onnx(model, feature_names, output_dir / "tfidf_model.onnx")
    save_coefficients_csv(model, feature_names, output_dir / "tfidf_coefficients.csv")
    print("Final model saved.")
    return model


# --- 6. Test Model ---

def test_model(model_path: Path, test_data_path: Path, output_dir: Path) -> dict:
    """Evaluate saved ONNX model on test data."""
    output_dir.mkdir(parents=True, exist_ok=True)
    print("=" * 60)
    print("TF-IDF Analysis — Test Model")
    print("=" * 60)

    df_test = pd.read_csv(test_data_path)
    if OUTCOME_COL not in df_test.columns:
        raise ValueError(f"Test data must contain '{OUTCOME_COL}' column")

    feature_cols = _get_feature_cols(df_test)
    all_predictors = DEMO_COLS + feature_cols
    X_raw = df_test[all_predictors].copy()
    X = pd.get_dummies(X_raw, drop_first=True).values

    scaler_path = model_path.parent / "tfidf_scaler.pkl"
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
    save_metrics(metrics, output_dir / "tfidf_test_metrics.json")

    plot_predicted_vs_observed(
        y_true, y_pred,
        output_path=output_dir / "tfidf_test_predicted_vs_observed.pdf",
        title="Test Set Performance",
    )
    return metrics


# --- 7. Predict ---

def predict(model_path: Path, input_path: Path, output_path: Path) -> pd.DataFrame:
    """Make predictions on new data."""
    print("=" * 60)
    print("TF-IDF Analysis — Predict")
    print("=" * 60)

    df_input = pd.read_csv(input_path)
    feature_cols = _get_feature_cols(df_input)
    all_predictors = DEMO_COLS + feature_cols
    X_raw = df_input[all_predictors].copy()
    X = pd.get_dummies(X_raw, drop_first=True).values

    scaler_path = model_path.parent / "tfidf_scaler.pkl"
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
    parser = argparse.ArgumentParser(description="TF-IDF Analysis")
    parser.add_argument("--mode", choices=["cv", "train", "test", "predict"], required=True)
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--output-dir", type=str, default="results")
    parser.add_argument("--model-path", type=str)
    parser.add_argument("--test-data", type=str)
    parser.add_argument("--input", type=str)
    parser.add_argument("--output", type=str)
    parser.add_argument("--random-state", type=int, default=123)
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
