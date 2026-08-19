"""
train_model.py

Train an elastic-net model on substitute dataset features and save to ONNX.

Reads:
    data/tfidf_features.csv      or  data/embedding_features.csv
    data/outcomes.csv

Saves:
    results/{feature_type}_model.onnx
    results/{feature_type}_scaler.pkl
    results/{feature_type}_metrics.json

Usage:
    python scripts/train_model.py --feature-type tfidf --data-dir data/ --output-dir results/
    python scripts/train_model.py --feature-type embeddings --data-dir data/ --output-dir results/
"""

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNetCV
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train model on substitute dataset features.")
    parser.add_argument(
        "--feature-type",
        required=True,
        choices=["tfidf", "embeddings"],
        help="Which feature file to use",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data",
        help="Directory containing feature and outcome files",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results",
        help="Directory to write model and metrics",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=123,
        help="Random seed",
    )
    parser.add_argument(
        "--target",
        type=str,
        default="severity",
        choices=["severity", "binary"],
        help="Which outcome to predict",
    )
    return parser.parse_args()


def load_features_and_outcomes(data_dir: Path, feature_type: str) -> tuple:
    """Load feature matrix and outcomes, merge on ELS_ID."""
    if feature_type == "tfidf":
        feature_path = data_dir / "tfidf_features.csv"
    elif feature_type == "embeddings":
        feature_path = data_dir / "embedding_features.csv"
    else:
        raise ValueError(f"Unknown feature type: {feature_type}")

    outcome_path = data_dir / "outcomes.csv"

    if not feature_path.exists():
        print(f"ERROR: Feature file not found: {feature_path}", file=sys.stderr)
        print(f"Run: python scripts/build_features.py --input data/unified_dataset.csv", file=sys.stderr)
        sys.exit(1)

    if not outcome_path.exists():
        print(f"ERROR: Outcome file not found: {outcome_path}", file=sys.stderr)
        sys.exit(1)

    df_features = pd.read_csv(feature_path)
    df_outcomes = pd.read_csv(outcome_path)

    # Merge on ELS_ID
    df_merged = df_features.merge(df_outcomes, on="ELS_ID", how="inner")

    # Identify feature columns (all numeric except ELS_ID and outcome cols)
    exclude = {"ELS_ID", "severity", "binary"}
    feature_cols = [c for c in df_merged.columns if c not in exclude]

    X = df_merged[feature_cols].values
    y = df_merged["severity"].values  # default to severity for regression

    return X, y, feature_cols, df_merged["ELS_ID"].values


def train_and_save(
    X: np.ndarray,
    y: np.ndarray,
    feature_cols: list,
    ids: np.ndarray,
    output_dir: Path,
    feature_type: str,
    random_state: int,
) -> Path:
    """Train elastic net, save model + scaler + metrics."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Split for validation metrics
    X_train, X_test, y_train, y_test, ids_train, ids_test = train_test_split(
        X, y, ids, test_size=0.2, random_state=random_state, stratify=None,
    )

    # Scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train
    print(f"Training elastic net on {len(feature_cols)} features...")
    # l1_ratio=0 (pure Ridge) is excluded because ElasticNetCV cannot auto-generate
    # an alpha grid when there is no L1 penalty. We start at 0.05.
    model = ElasticNetCV(
        l1_ratio=list(np.arange(0.05, 1.01, 0.05)),
        cv=5,
        random_state=random_state,
        n_jobs=-1,
        max_iter=10000,
    )
    model.fit(X_train_scaled, y_train)

    print(f"  Best alpha (l1_ratio): {model.l1_ratio_:.3f}")
    print(f"  Best lambda (alpha): {model.alpha_:.6f}")

    # Evaluate
    y_pred = model.predict(X_test_scaled)
    metrics = {
        "r2": float(r2_score(y_test, y_pred)),
        "mse": float(mean_squared_error(y_test, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_test, y_pred))),
        "mae": float(mean_absolute_error(y_test, y_pred)),
    }
    print(f"  Test R² = {metrics['r2']:.4f}")

    # Save
    prefix = f"{feature_type}_model"

    # Scaler
    scaler_path = output_dir / f"{prefix}_scaler.pkl"
    joblib.dump(scaler, scaler_path)

    # Metrics
    metrics_path = output_dir / f"{prefix}_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    # Coefficients
    coef_df = pd.DataFrame({
        "term": feature_cols,
        "estimate": model.coef_,
    })
    coef_path = output_dir / f"{prefix}_coefficients.csv"
    coef_df.to_csv(coef_path, index=False)

    # Predictions on test set
    pred_df = pd.DataFrame({
        "ELS_ID": ids_test,
        "actual": y_test,
        "predicted": y_pred,
    })
    pred_path = output_dir / f"{prefix}_test_predictions.csv"
    pred_df.to_csv(pred_path, index=False)

    # ONNX
    try:
        from skl2onnx import convert_sklearn
        from skl2onnx.common.data_types import FloatTensorType

        onnx_path = output_dir / f"{prefix}.onnx"
        initial_type = [("float_input", FloatTensorType([None, len(feature_cols)]))]
        onnx_model = convert_sklearn(model, initial_types=initial_type)
        with open(onnx_path, "wb") as f:
            f.write(onnx_model.SerializeToString())
        print(f"  ONNX model: {onnx_path}")
    except ImportError:
        print("  WARNING: skl2onnx not installed, skipping ONNX export.")
        onnx_path = None

    print(f"  Metrics: {metrics_path}")
    print(f"  Coefficients: {coef_path}")
    return onnx_path or scaler_path


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)

    X, y, feature_cols, ids = load_features_and_outcomes(data_dir, args.feature_type)
    print(f"Loaded {len(X)} samples, {len(feature_cols)} features")

    train_and_save(
        X, y, feature_cols, ids,
        output_dir=output_dir,
        feature_type=args.feature_type,
        random_state=args.random_state,
    )

    print("\nTraining complete.")


if __name__ == "__main__":
    main()
