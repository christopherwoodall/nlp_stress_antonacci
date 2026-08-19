"""
test_integration_pipeline.py

End-to-end integration test: create data → extract features → train → predict.

Uses TF-IDF (faster) to keep test execution quick.
"""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts.build_features import build_outcomes, extract_tfidf
from scripts.train_model import load_features_and_outcomes, train_and_save


class TestIntegrationPipeline:
    def test_full_train_predict_cycle(self):
        """Full pipeline from dataset to predictions."""
        with tempfile.TemporaryDirectory() as tmp:
            data_dir = Path(tmp) / "data"
            results_dir = Path(tmp) / "results"
            data_dir.mkdir()
            results_dir.mkdir()

            # 1. Create synthetic dataset
            depressed = [
                "I feel empty and hopeless every single day",
                "Nothing brings me joy anymore",
                "I hate myself and see no reason to continue",
            ]
            normal = [
                "I had a great day today",
                "School was fun and I saw friends",
                "Everything is going well for me",
            ]
            texts = []
            severities = []
            for i in range(60):
                is_dep = i % 2 == 0
                texts.append(np.random.choice(depressed if is_dep else normal))
                severities.append(2 if is_dep else 0)

            df = pd.DataFrame({
                "text": texts,
                "severity": severities,
                "binary": [1 if s == 2 else 0 for s in severities],
            })
            df.to_csv(data_dir / "unified_dataset.csv", index=False)

            # 2. Extract features
            build_outcomes(df, data_dir)
            extract_tfidf(df, data_dir, results_dir)

            # 3. Load features
            X, y, feature_cols, ids = load_features_and_outcomes(data_dir, "tfidf")
            assert len(X) == 60
            assert len(feature_cols) > 0

            # 4. Train model
            model_path = train_and_save(
                X, y, feature_cols, ids,
                output_dir=results_dir,
                feature_type="tfidf",
                random_state=42,
            )
            assert model_path.exists()

            # 5. Verify ONNX loads
            import onnxruntime as ort
            session = ort.InferenceSession(str(model_path))
            assert len(session.get_inputs()) == 1

            # 6. Predict on same data (just to verify pipeline)
            import joblib
            scaler = joblib.load(results_dir / "tfidf_model_scaler.pkl")
            X_scaled = scaler.transform(X)
            input_name = session.get_inputs()[0].name
            preds = session.run(None, {input_name: X_scaled.astype(np.float32)})[0].ravel()

            # Sanity checks
            assert np.all(np.isfinite(preds))
            assert len(preds) == 60
            pred_range = preds.max() - preds.min()
            assert pred_range > 0.01, f"Predictions too uniform: range={pred_range}"

            print(f"Integration test passed: R² not computed, but predictions range={pred_range:.3f}")

    def test_train_model_cli(self):
        """Test train_model.py via CLI-like invocation."""
        with tempfile.TemporaryDirectory() as tmp:
            data_dir = Path(tmp) / "data"
            results_dir = Path(tmp) / "results"
            data_dir.mkdir()
            results_dir.mkdir()

            # Minimal dataset
            df = pd.DataFrame({
                "text": ["sad text"] * 20 + ["happy text"] * 20,
                "severity": [2] * 20 + [0] * 20,
                "binary": [1] * 20 + [0] * 20,
            })
            df.to_csv(data_dir / "unified_dataset.csv", index=False)

            # Extract features
            build_outcomes(df, data_dir)
            extract_tfidf(df, data_dir, results_dir)

            # Run train_model.py as subprocess
            import subprocess
            import sys
            result = subprocess.run([
                sys.executable, "scripts/train_model.py",
                "--feature-type", "tfidf",
                "--data-dir", str(data_dir),
                "--output-dir", str(results_dir),
                "--random-state", "42",
            ], capture_output=True, text=True)

            assert result.returncode == 0, f"train_model.py failed: {result.stderr}"
            assert (results_dir / "tfidf_model.onnx").exists()
            assert (results_dir / "tfidf_model_metrics.json").exists()
