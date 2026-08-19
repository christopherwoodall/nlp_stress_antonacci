"""
liwc.py

LIWC Analyses — translated from analysis/LIWC_Analyses.Rmd

Replicates:
  1. Descriptive heatmap of LIWC features
  2. Baseline linear models (demographics only, demographics + cumulative stress)
  3. Elastic net nested CV on LIWC features
  4. PCA of LIWC features → linear / logistic models on PCs
  5. Incremental variance explained figure
  6. Predicted vs observed scatterplot and feature importance plot

Expected data files (relative to --data-dir):
  - LIWC_Imputed_ChildOnly_Final_T1.xlsx  (features + demographics + outcomes)
  - any_intDx_T1-TA.xlsx                  (diagnosis outcomes)
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
    cv_linear_regression,
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
    plot_incremental_r2,
)


# --- Column definitions matching the R notebook ---

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


def _load_data(data_dir: Path) -> tuple:
    """
    Load LIWC features, outcomes, and diagnoses.

    Returns (df_merged, df_dx) or raises FileNotFoundError with guidance.
    """
    liwc_path = data_dir / "LIWC_Imputed_ChildOnly_Final_T1.xlsx"
    dx_path = data_dir / "any_intDx_T1-TA.xlsx"

    for p in (liwc_path, dx_path):
        if not p.exists():
            print(
                f"ERROR: Expected data file not found: {p}\n"
                "The LIWC analysis requires clinical data files that are not "
                "included in this repository. Contact the Stanford NAPL (PI: Ian Gotlib) "
                "for data access, or use the sample dataset to explore the pipeline.",
                file=sys.stderr,
            )
            raise FileNotFoundError(p)

    df_liwc = pd.read_excel(liwc_path)
    df_dx = pd.read_excel(dx_path)

    # ensure ID is string for safe merging
    df_liwc["ELS_ID"] = df_liwc["ELS_ID"].astype(str)
    df_dx["ELS_ID"] = df_dx["ELS_ID"].astype(str)

    return df_liwc, df_dx


def _get_feature_cols(df: pd.DataFrame) -> list:
    """Identify LIWC feature columns (all numeric cols after demographics)."""
    # In the R notebook, features start at column 22.
    # We dynamically select all numeric columns not in {ID, demographics, outcomes}.
    exclude = {"ELS_ID", OUTCOME_COL, DX_COL} | set(DEMO_COLS)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    feature_cols = [c for c in numeric_cols if c not in exclude]
    return feature_cols


# --- 1. Baseline Models ---

def run_baseline_models(df: pd.DataFrame, output_dir: Path, random_state: int = 123) -> dict:
    """
    Run baseline linear regression models with manual 5-fold CV.

    Replicates the R baseline model section:
      - Model 1: Demographics only
      - Model 2: Demographics + TESI cumulative stress severity
    """
    print("\n--- Baseline Models ---")

    y = df[OUTCOME_COL].values

    # --- Model 1: Demographics only ---
    demo_cols_only = [c for c in DEMO_COLS if c != "TESI_obj_sumsev.T1"]
    X_demo = pd.get_dummies(df[demo_cols_only], drop_first=True).values
    result_demo = cv_linear_regression(X_demo, y, n_folds=5, random_state=random_state)
    r2_demo = result_demo["r2"]
    print(f"  Demographics only: R² = {r2_demo:.4f}")

    # --- Model 2: Demographics + SumSev ---
    X_demo_sumsev = pd.get_dummies(df[DEMO_COLS], drop_first=True).values
    result_sumsev = cv_linear_regression(X_demo_sumsev, y, n_folds=5, random_state=random_state + 111)
    r2_sumsev = result_sumsev["r2"]
    print(f"  Demographics + SumSev: R² = {r2_sumsev:.4f}")

    # Save predictions for potential plotting
    baseline_results = {
        "r2_demo": float(r2_demo),
        "r2_sumsev": float(r2_sumsev),
        "predictions_demo": result_demo["predictions"].tolist(),
        "predictions_sumsev": result_sumsev["predictions"].tolist(),
    }
    save_metrics(baseline_results, output_dir / "liwc_baseline_metrics.json")
    return baseline_results


# --- 2. Elastic Net Nested CV ---

def run_elastic_net_cv(df: pd.DataFrame, feature_cols: list, output_dir: Path, random_state: int = 123) -> dict:
    """
    Run elastic net with nested CV on LIWC features.

    Replicates the R elastic-net-cv chunk.
    """
    print("\n--- Elastic Net Nested CV ---")

    y = df[OUTCOME_COL].values
    # Include demographics + LIWC features, as in the R notebook
    all_predictors = DEMO_COLS + feature_cols
    X_raw = df[all_predictors].copy()
    # One-hot encode categorical demographics
    X = pd.get_dummies(X_raw, drop_first=True).values
    # Scale (the R notebook calls scale(X))
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    results = nested_cv_elastic_net(
        X_scaled, y, n_outer_folds=5, random_state=random_state
    )

    print(f"  Global R² = {results['global_r2']:.4f}")
    print(f"  Best alphas per fold: {[round(a, 2) for a in results['best_alpha_folds']]}")

    # Save CV predictions
    cv_df = pd.DataFrame({
        "ELS_ID": df["ELS_ID"].values,
        "Actual": y,
        "Predicted": results["cv_predictions"],
    })
    cv_df.to_csv(output_dir / "liwc_cv_predictions.csv", index=False)

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
        output_dir / "liwc_elastic_net_metrics.json",
    )

    # Plot predicted vs observed
    plot_predicted_vs_observed(
        y, results["cv_predictions"],
        output_path=output_dir / "liwc_en_predicted_vs_observed.pdf",
        title="Model Performance on Held-Out Data",
        subtitle="Predicted vs. Actual YSR Internalizing Scores",
    )

    return results


# --- 3. PCA of LIWC Features ---

def run_pca_analysis(df: pd.DataFrame, feature_cols: list, output_dir: Path) -> dict:
    """
    PCA on LIWC features, then linear and logistic models on PC scores.

    Replicates the R PCA analysis section.
    """
    print("\n--- PCA Analysis ---")

    # Run PCA on LIWC features only (not demographics)
    X_liwc = df[feature_cols].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_liwc)

    pca = PCA()
    pc_scores = pca.fit_transform(X_scaled)

    # Save PC scores
    pc_df = pd.DataFrame(pc_scores[:, :10], columns=[f"PC{i+1}" for i in range(10)])
    pc_df["ELS_ID"] = df["ELS_ID"].values
    pc_df.to_csv(output_dir / "liwc_pc_scores.csv", index=False)

    # Scree plot
    plot_scree(
        pca,
        output_path=output_dir / "liwc_pca_scree_plot.pdf",
        title="Scree Plot of Principal Components",
    )

    # Merge PC1, PC2 back into main dataframe
    df_pcs = df.copy()
    df_pcs["PC1"] = pc_scores[:, 0]
    df_pcs["PC2"] = pc_scores[:, 1]

    # --- Linear model: YSR ~ covariates + PC1 + PC2 ---
    y = df_pcs[OUTCOME_COL].values
    covariates = [c for c in DEMO_COLS if c in df_pcs.columns]
    X_pc = pd.get_dummies(df_pcs[covariates + ["PC1", "PC2"]], drop_first=True).values

    model_pc = LinearRegression()
    model_pc.fit(X_pc, y)
    r2_pc = model_pc.score(X_pc, y)
    print(f"  Linear model with PC1+PC2: R² = {r2_pc:.4f}")

    # --- Logistic model: Diagnosis ~ covariates + PC1 + PC2 ---
    if DX_COL in df_pcs.columns and df_pcs[DX_COL].notna().any():
        y_dx = df_pcs[DX_COL].values
        # drop rows with missing diagnosis
        valid = ~np.isnan(y_dx)
        if valid.sum() > 10:
            model_logistic = LogisticRegression(max_iter=1000)
            model_logistic.fit(X_pc[valid], y_dx[valid])
            print(f"  Logistic model with PC1+PC2: fitted on {valid.sum()} observations")

    return {"pca": pca, "pc_scores": pc_scores}


# --- 4. Full CV Analysis Orchestrator ---

def run_cv_analysis(data_dir: Path, output_dir: Path, random_state: int = 123) -> dict:
    """
    Run the complete LIWC cross-validation analysis.

    This is the main entry point for --mode cv.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    print("=" * 60)
    print("LIWC Analysis — Cross-Validation")
    print("=" * 60)

    df_liwc, df_dx = _load_data(data_dir)
    feature_cols = _get_feature_cols(df_liwc)
    print(f"Loaded {len(df_liwc)} participants with {len(feature_cols)} LIWC features.")

    # Baseline models
    baseline = run_baseline_models(df_liwc, output_dir, random_state=random_state)

    # Elastic net nested CV
    en_results = run_elastic_net_cv(df_liwc, feature_cols, output_dir, random_state=random_state)

    # PCA analysis
    pca_results = run_pca_analysis(df_liwc, feature_cols, output_dir)

    # Incremental R² plot
    r2_values = {
        "Demographics Only": baseline["r2_demo"],
        "+ Cumulative Stress": max(0, baseline["r2_sumsev"] - baseline["r2_demo"]),
        "+ LIWC Features": max(0, en_results["global_r2"] - baseline["r2_sumsev"]),
    }
    plot_incremental_r2(
        r2_values,
        output_path=output_dir / "liwc_incremental_R2_plot.pdf",
    )

    print("\nLIWC CV analysis complete.")
    return {
        "baseline": baseline,
        "elastic_net": en_results,
        "pca": pca_results,
    }


# --- 5. Train Final Model ---

def train_final_model(data_dir: Path, output_dir: Path, random_state: int = 123):
    """
    Train the final elastic net model on the full dataset.

    Saves:
      - ONNX model file
      - Coefficient CSV
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    print("=" * 60)
    print("LIWC Analysis — Train Final Model")
    print("=" * 60)

    df_liwc, _ = _load_data(data_dir)
    feature_cols = _get_feature_cols(df_liwc)

    y = df_liwc[OUTCOME_COL].values
    all_predictors = DEMO_COLS + feature_cols
    X_raw = df_liwc[all_predictors].copy()
    X = pd.get_dummies(X_raw, drop_first=True).values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    from ._base import train_final_elastic_net
    model = train_final_elastic_net(X_scaled, y, random_state=random_state)

    print(f"  Best alpha (l1_ratio): {model.l1_ratio_:.3f}")
    print(f"  Best lambda (alpha):   {model.alpha_:.6f}")

    # Save scaler for future prediction
    import joblib
    joblib.dump(scaler, output_dir / "liwc_scaler.pkl")

    # Save ONNX model
    feature_names = pd.get_dummies(X_raw, drop_first=True).columns.tolist()
    save_model_onnx(model, feature_names, output_dir / "liwc_model.onnx")

    # Save coefficients CSV
    save_coefficients_csv(
        model,
        feature_names,
        output_dir / "liwc_coefficients.csv",
        intercept=True,
    )

    print("Final model saved.")
    return model


# --- 6. Test Model ---

def test_model(model_path: Path, test_data_path: Path, output_dir: Path) -> dict:
    """Evaluate a saved ONNX model on test data."""
    output_dir.mkdir(parents=True, exist_ok=True)
    print("=" * 60)
    print("LIWC Analysis — Test Model")
    print("=" * 60)

    df_test = pd.read_csv(test_data_path)
    if OUTCOME_COL not in df_test.columns:
        raise ValueError(f"Test data must contain '{OUTCOME_COL}' column")

    feature_cols = _get_feature_cols(df_test)
    all_predictors = DEMO_COLS + feature_cols
    X_raw = df_test[all_predictors].copy()
    X = pd.get_dummies(X_raw, drop_first=True).values

    # load scaler if available
    scaler_path = model_path.parent / "liwc_scaler.pkl"
    if scaler_path.exists():
        import joblib
        scaler = joblib.load(scaler_path)
        X = scaler.transform(X)
    else:
        # fallback: standardize in-place
        from sklearn.preprocessing import StandardScaler
        X = StandardScaler().fit_transform(X)

    session = load_model_onnx(model_path)
    y_pred = predict_onnx(session, X)
    y_true = df_test[OUTCOME_COL].values

    metrics = compute_metrics(y_true, y_pred)
    print(f"  Test R² = {metrics['r2']:.4f}")
    print(f"  Test RMSE = {metrics['rmse']:.4f}")

    save_metrics(metrics, output_dir / "liwc_test_metrics.json")

    plot_predicted_vs_observed(
        y_true, y_pred,
        output_path=output_dir / "liwc_test_predicted_vs_observed.pdf",
        title="Test Set Performance",
    )

    return metrics


# --- 7. Predict ---

def predict(model_path: Path, input_path: Path, output_path: Path) -> pd.DataFrame:
    """Make predictions on new data using a saved ONNX model."""
    print("=" * 60)
    print("LIWC Analysis — Predict")
    print("=" * 60)

    df_input = pd.read_csv(input_path)
    feature_cols = _get_feature_cols(df_input)
    all_predictors = DEMO_COLS + feature_cols
    X_raw = df_input[all_predictors].copy()
    X = pd.get_dummies(X_raw, drop_first=True).values

    # load scaler if available
    scaler_path = model_path.parent / "liwc_scaler.pkl"
    if scaler_path.exists():
        import joblib
        scaler = joblib.load(scaler_path)
        X = scaler.transform(X)
    else:
        from sklearn.preprocessing import StandardScaler
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
    parser = argparse.ArgumentParser(description="LIWC Analysis")
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
