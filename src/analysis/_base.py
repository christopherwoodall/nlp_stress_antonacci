"""
_base.py

Shared utilities for the analysis package.

Provides:
  - Data loading and merging helpers
  - Nested cross-validation for elastic net (replicates R glmnet pattern)
  - Baseline model runners (linear regression with demographics)
  - Metrics computation (R², MSE, RMSE, MAE)
  - ONNX model serialization / deserialization
"""

import json
from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNetCV, ElasticNet, LinearRegression
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error


def load_merged_data(
    feature_path: Union[str, Path],
    outcome_path: Union[str, Path],
    feature_id_col: str = "ELS_ID",
    outcome_id_col: str = "ELS_ID",
    outcome_cols: list = None,
) -> pd.DataFrame:
    """
    Load feature and outcome files and merge on a common ID column.

    Supports .csv and .xlsx file extensions.
    """
    feature_path = Path(feature_path)
    outcome_path = Path(outcome_path)

    if feature_path.suffix == ".xlsx":
        df_features = pd.read_excel(feature_path)
    else:
        df_features = pd.read_csv(feature_path)

    if outcome_path.suffix == ".xlsx":
        df_outcomes = pd.read_excel(outcome_path)
    else:
        df_outcomes = pd.read_csv(outcome_path)

    merge_cols = [outcome_id_col] + (outcome_cols or [])
    # deduplicate columns in case outcome_id_col == feature_id_col
    merge_cols = list(dict.fromkeys(merge_cols))

    df_merged = df_features.merge(
        df_outcomes[merge_cols],
        left_on=feature_id_col,
        right_on=outcome_id_col,
        how="left",
    )

    # drop duplicate ID column if names differ
    if feature_id_col != outcome_id_col and outcome_id_col in df_merged.columns:
        df_merged = df_merged.drop(columns=[outcome_id_col])

    return df_merged


def nested_cv_elastic_net(
    X: np.ndarray,
    y: np.ndarray,
    n_outer_folds: int = 5,
    alpha_grid: np.ndarray = None,
    random_state: int = 123,
) -> dict:
    """
    Run nested cross-validation for elastic net regression.

    Outer loop: KFold(n_splits=5) splits data into train/test.
    Inner loop: ElasticNetCV tunes alpha (l1_ratio) and lambda (alpha)
                via 5-fold CV on the outer training fold.

    Replicates the pattern in the R notebooks where they manually loop
    over an alpha grid and call cv.glmnet() for each value.

    Parameters
    ----------
    X : np.ndarray, shape (n_samples, n_features)
        Feature matrix. Should already be scaled if desired.
    y : np.ndarray, shape (n_samples,)
        Outcome vector.
    n_outer_folds : int
        Number of outer CV folds.
    alpha_grid : np.ndarray
        Grid of alpha (l1_ratio) values to search. Defaults to 0–1 by 0.05.
    random_state : int
        Seed for reproducible fold generation.

    Returns
    -------
    dict with keys:
        cv_predictions : np.ndarray — out-of-sample predictions
        global_r2 : float — R² computed on all held-out predictions
        best_alpha_folds : list — best alpha per outer fold
        best_lambda_folds : list — best lambda per outer fold
        cv_mse : list — per-fold MSE
        cv_rmse : list — per-fold RMSE
        cv_mae : list — per-fold MAE
        cv_r2 : list — per-fold R²
    """
    if alpha_grid is None:
        alpha_grid = np.arange(0, 1.01, 0.05)

    outer_cv = KFold(n_splits=n_outer_folds, shuffle=True, random_state=random_state)
    cv_predictions = np.full_like(y, np.nan, dtype=float)

    best_alpha_folds = []
    best_lambda_folds = []
    cv_mse = []
    cv_rmse = []
    cv_mae = []
    cv_r2 = []

    for train_idx, test_idx in outer_cv.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # --- Inner CV: tune alpha and lambda ---
        # ElasticNetCV with a list of l1_ratio values searches over the
        # alpha grid automatically, equivalent to the R manual loop.
        inner_model = ElasticNetCV(
            l1_ratio=list(alpha_grid),
            cv=5,
            random_state=random_state,
            n_jobs=-1,
            max_iter=10000,
        )
        inner_model.fit(X_train, y_train)

        best_alpha = inner_model.l1_ratio_
        best_lambda = inner_model.alpha_

        best_alpha_folds.append(best_alpha)
        best_lambda_folds.append(best_lambda)

        # --- Predict on outer test fold ---
        y_pred = inner_model.predict(X_test)
        cv_predictions[test_idx] = y_pred

        # --- Compute per-fold metrics ---
        mse = mean_squared_error(y_test, y_pred)
        cv_mse.append(mse)
        cv_rmse.append(np.sqrt(mse))
        cv_mae.append(mean_absolute_error(y_test, y_pred))
        cv_r2.append(r2_score(y_test, y_pred))

    # --- Aggregate global metrics ---
    global_r2 = r2_score(y, cv_predictions)

    return {
        "cv_predictions": cv_predictions,
        "global_r2": global_r2,
        "best_alpha_folds": best_alpha_folds,
        "best_lambda_folds": best_lambda_folds,
        "cv_mse": cv_mse,
        "cv_rmse": cv_rmse,
        "cv_mae": cv_mae,
        "cv_r2": cv_r2,
    }


def train_final_elastic_net(
    X: np.ndarray,
    y: np.ndarray,
    alpha_grid: np.ndarray = None,
    random_state: int = 123,
) -> ElasticNetCV:
    """
    Train a final elastic net model on the full dataset.

    Uses ElasticNetCV to tune alpha and lambda via 5-fold CV on the
    full data, then returns the fitted model.
    """
    if alpha_grid is None:
        alpha_grid = np.arange(0, 1.01, 0.05)

    model = ElasticNetCV(
        l1_ratio=list(alpha_grid),
        cv=5,
        random_state=random_state,
        n_jobs=-1,
        max_iter=10000,
    )
    model.fit(X, y)
    return model


def cv_linear_regression(
    X: np.ndarray,
    y: np.ndarray,
    n_folds: int = 5,
    random_state: int = 123,
) -> dict:
    """
    Run manual K-fold cross-validation for ordinary least squares.

    Replicates the R baseline model CV loops.
    """
    folds = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    predictions = np.full_like(y, np.nan, dtype=float)

    for train_idx, test_idx in folds.split(X):
        model = LinearRegression()
        model.fit(X[train_idx], y[train_idx])
        predictions[test_idx] = model.predict(X[test_idx])

    r2 = r2_score(y, predictions)
    return {"predictions": predictions, "r2": r2}


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Compute MSE, RMSE, MAE, and R²."""
    mse = mean_squared_error(y_true, y_pred)
    return {
        "mse": float(mse),
        "rmse": float(np.sqrt(mse)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def save_metrics(metrics: dict, path: Union[str, Path]) -> None:
    """Save metrics dict to JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2)


def save_coefficients_csv(
    model,
    feature_names: list,
    path: Union[str, Path],
    intercept: bool = True,
) -> None:
    """
    Save model coefficients to a CSV file.

    The output format matches what the R notebooks write, with columns:
        term, estimate
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    if intercept:
        rows.append({"term": "(Intercept)", "estimate": float(model.intercept_)})

    coefs = model.coef_
    if coefs.ndim > 1:
        coefs = coefs.ravel()

    for name, coef in zip(feature_names, coefs):
        rows.append({"term": name, "estimate": float(coef)})

    pd.DataFrame(rows).to_csv(path, index=False)


# --- ONNX serialization ---

def _try_import_skl2onnx():
    try:
        from skl2onnx import convert_sklearn
        from skl2onnx.common.data_types import FloatTensorType
        return convert_sklearn, FloatTensorType
    except ImportError:
        return None, None


def save_model_onnx(
    model,
    feature_names: list,
    path: Union[str, Path],
) -> None:
    """
    Serialize a fitted sklearn model to ONNX format.

    Requires skl2onnx and onnxruntime to be installed.
    """
    convert_sklearn, FloatTensorType = _try_import_skl2onnx()
    if convert_sklearn is None:
        raise ImportError(
            "skl2onnx is required for ONNX export. "
            "Install it with: uv pip install skl2onnx onnxruntime"
        )

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    initial_type = [("float_input", FloatTensorType([None, len(feature_names)]))]
    onnx_model = convert_sklearn(model, initial_types=initial_type)

    with open(path, "wb") as f:
        f.write(onnx_model.SerializeToString())

    print(f"ONNX model saved to: {path}")


def load_model_onnx(path: Union[str, Path]):
    """Load an ONNX model and return an onnxruntime InferenceSession."""
    try:
        import onnxruntime as ort
    except ImportError:
        raise ImportError(
            "onnxruntime is required for ONNX inference. "
            "Install it with: uv pip install onnxruntime"
        )

    path = Path(path)
    session = ort.InferenceSession(str(path))
    return session


def predict_onnx(session, X: np.ndarray) -> np.ndarray:
    """Run inference with an ONNX runtime session."""
    input_name = session.get_inputs()[0].name
    return session.run(None, {input_name: X.astype(np.float32)})[0].ravel()
