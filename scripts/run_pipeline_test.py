"""
run_pipeline_test.py

End-to-end integration test for the full pipeline.

Creates a minimal synthetic dataset, extracts features, trains a model,
saves to ONNX, loads ONNX, and verifies predictions are reasonable.

This test uses TF-IDF (faster than embeddings) to keep CI quick,
but exercises the same train→save→load→predict cycle as the real pipeline.

Usage:
    python scripts/run_pipeline_test.py
"""

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd


def create_test_dataset(output_dir: Path, n_samples: int = 100) -> Path:
    """Create a minimal synthetic dataset with text, severity, and binary labels."""
    print("[1/6] Creating test dataset...")

    # Mix of "depressed" and "not depressed" text
    depressed_texts = [
        "I feel so empty and hopeless every single day.",
        "Nothing brings me joy anymore. I just want to sleep.",
        "I can't concentrate and everything feels pointless.",
        "I hate myself and I don't see any reason to keep going.",
        "The sadness never goes away no matter what I try.",
    ]
    normal_texts = [
        "I had a pretty good day today. Things are going well.",
        "School was fine and I hung out with friends after.",
        "I'm excited about the weekend plans with my family.",
        "Life has been pretty normal lately, no complaints.",
        "I feel happy and things are looking up for me.",
    ]

    rows = []
    rng = np.random.default_rng(42)
    for i in range(n_samples):
        is_depressed = i % 2 == 0  # 50/50 split
        text = rng.choice(depressed_texts if is_depressed else normal_texts)
        rows.append({
            "text": text,
            "source": "test",
            "severity": 2 if is_depressed else 0,
            "binary": 1 if is_depressed else 0,
            "original_label": "depressed" if is_depressed else "normal",
        })

    df = pd.DataFrame(rows)
    out_path = output_dir / "unified_dataset.csv"
    df.to_csv(out_path, index=False)
    print(f"       Created {out_path} with {len(df)} rows")
    return out_path


def extract_features(data_path: Path, output_dir: Path, results_dir: Path) -> None:
    """Extract TF-IDF features only (skip slow embeddings for pipeline test)."""
    print("[2/6] Extracting TF-IDF features...")

    df = pd.read_csv(data_path)
    df = df.dropna(subset=["text"])

    # Build outcomes
    outcome_df = df[["severity", "binary"]].copy()
    outcome_df["ELS_ID"] = range(len(outcome_df))
    outcome_df.to_csv(output_dir / "outcomes.csv", index=False)

    # Extract TF-IDF
    from sklearn.feature_extraction.text import TfidfVectorizer
    import joblib

    vectorizer = TfidfVectorizer(
        token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z]+\b",
        min_df=0.05,
        max_features=5000,
    )
    X = vectorizer.fit_transform(df["text"].tolist())
    feature_df = pd.DataFrame(X.toarray(), columns=vectorizer.get_feature_names_out())
    feature_df["ELS_ID"] = range(len(feature_df))
    feature_df.to_csv(output_dir / "tfidf_features.csv", index=False)

    # Save vectorizer
    results_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(vectorizer, results_dir / "tfidf_vectorizer.joblib")

    print("       Features extracted.")


def train_model(data_dir: Path, output_dir: Path) -> Path:
    """Train a TF-IDF model and save to ONNX."""
    print("[3/6] Training model...")

    cmd = [
        sys.executable, "scripts/train_model.py",
        "--feature-type", "tfidf",
        "--data-dir", str(data_dir),
        "--output-dir", str(output_dir),
        "--random-state", "42",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(result.stderr, file=sys.stderr)
        raise RuntimeError("train_model.py failed")

    # Find the saved ONNX model
    onnx_path = output_dir / "tfidf_model.onnx"
    if not onnx_path.exists():
        raise FileNotFoundError(f"ONNX model not found at {onnx_path}")
    print(f"       Model saved: {onnx_path}")
    return onnx_path


def verify_onnx_model(model_path: Path) -> None:
    """Verify the ONNX model can be loaded."""
    print("[4/6] Verifying ONNX model...")

    import onnxruntime as ort
    session = ort.InferenceSession(str(model_path))
    assert session is not None
    print(f"       ONNX model loads OK. Inputs: {len(session.get_inputs())}")


def run_prediction(model_path: Path, test_data: Path, output_path: Path) -> None:
    """Run prediction on a small test sample using the saved model."""
    print("[5/6] Running prediction...")

    # Load the model and scaler manually for prediction
    import joblib
    import onnxruntime as ort

    scaler_path = model_path.parent / "tfidf_model_scaler.pkl"
    scaler = joblib.load(scaler_path)

    df_test = pd.read_csv(test_data)
    exclude = {"ELS_ID", "severity", "binary"}
    feature_cols = [c for c in df_test.columns if c not in exclude]
    X = df_test[feature_cols].values
    X_scaled = scaler.transform(X)

    session = ort.InferenceSession(str(model_path))
    input_name = session.get_inputs()[0].name
    preds = session.run(None, {input_name: X_scaled.astype(np.float32)})[0].ravel()

    pred_df = pd.DataFrame({
        "ELS_ID": df_test["ELS_ID"].values,
        "prediction": preds,
    })
    pred_df.to_csv(output_path, index=False)
    print(f"       Predictions saved: {output_path} ({len(pred_df)} rows)")


def verify_predictions(pred_path: Path) -> None:
    """Check that predictions are reasonable."""
    print("[6/6] Verifying predictions...")

    df = pd.read_csv(pred_path)
    preds = df["prediction"].values

    # Basic sanity checks
    assert np.all(np.isfinite(preds)), "Predictions contain NaN or Inf!"
    assert len(preds) > 0, "No predictions generated!"

    # For a 50/50 depressed/normal split, predictions should span a range
    pred_range = preds.max() - preds.min()
    assert pred_range > 0.01, f"Predictions are too uniform (range={pred_range})"

    print(f"       Predictions: min={preds.min():.3f}, max={preds.max():.3f}, mean={preds.mean():.3f}")
    print("       All checks passed!")


def main() -> int:
    print("=" * 60)
    print("Running end-to-end pipeline test")
    print("=" * 60)

    test_dir = Path(tempfile.mkdtemp(prefix="pipeline_test_"))
    data_dir = test_dir / "data"
    results_dir = test_dir / "results"
    data_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    print(f"Test directory: {test_dir}\n")

    try:
        # 1. Create dataset
        data_path = create_test_dataset(data_dir, n_samples=100)

        # 2. Extract features
        extract_features(data_path, data_dir, results_dir)

        # 3. Train model
        model_path = train_model(data_dir, results_dir)

        # 4. Verify ONNX
        verify_onnx_model(model_path)

        # 5. Run prediction
        test_data = data_dir / "tfidf_features.csv"
        pred_output = results_dir / "test_predictions.csv"
        run_prediction(model_path, test_data, pred_output)

        # 6. Verify predictions
        verify_predictions(pred_output)

        print("\n" + "=" * 60)
        print("PIPELINE TEST PASSED")
        print("=" * 60)
        return 0

    except Exception as e:
        print(f"\nPIPELINE TEST FAILED: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1

    finally:
        print(f"\nCleaning up: {test_dir}")
        shutil.rmtree(test_dir, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
