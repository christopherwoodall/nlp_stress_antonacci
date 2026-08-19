"""
test_build_features.py

Tests for the feature extraction script.
"""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts.build_features import build_outcomes, extract_tfidf


class TestBuildOutcomes:
    def test_outcomes_columns(self):
        """Outcome file should have severity, binary, and ELS_ID."""
        df = pd.DataFrame({
            "text": ["a", "b", "c"],
            "severity": [0, 2, 1],
            "binary": [0, 1, 1],
        })
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp)
            build_outcomes(df, out_dir)
            result = pd.read_csv(out_dir / "outcomes.csv")
            assert set(result.columns) == {"severity", "binary", "ELS_ID"}
            assert len(result) == 3
            assert list(result["severity"]) == [0, 2, 1]

    def test_els_id_sequential(self):
        """ELS_ID should be sequential starting from 0."""
        df = pd.DataFrame({
            "text": ["x"],
            "severity": [2],
            "binary": [1],
        })
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp)
            build_outcomes(df, out_dir)
            result = pd.read_csv(out_dir / "outcomes.csv")
            assert list(result["ELS_ID"]) == [0]


class TestExtractTfidf:
    def test_tfidf_shape(self):
        """TF-IDF matrix should have correct shape and include ELS_ID."""
        texts = [
            "I feel sad and hopeless every day",
            "I had a good day and feel happy",
            "Nothing brings me joy anymore",
        ]
        df = pd.DataFrame({
            "text": texts,
            "severity": [2, 0, 2],
            "binary": [1, 0, 1],
        })
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp)
            results_dir = Path(tmp) / "results"
            extract_tfidf(df, out_dir, results_dir)

            feature_df = pd.read_csv(out_dir / "tfidf_features.csv")
            assert "ELS_ID" in feature_df.columns
            assert len(feature_df) == 3
            # Should have at least some features (words)
            assert feature_df.shape[1] > 1

    def test_vectorizer_saved(self):
        """Vectorizer should be saved to results dir."""
        df = pd.DataFrame({
            "text": ["happy day", "sad night", "good morning"],
            "severity": [0, 2, 0],
            "binary": [0, 1, 0],
        })
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp)
            results_dir = Path(tmp) / "results"
            extract_tfidf(df, out_dir, results_dir)

            vectorizer_path = results_dir / "tfidf_vectorizer.joblib"
            assert vectorizer_path.exists()


class TestFeatureExtractionIntegration:
    def test_features_and_outcomes_merge(self):
        """Feature and outcome files should merge correctly on ELS_ID."""
        texts = [
            "I feel sad and hopeless",
            "I had a good day today",
        ]
        df = pd.DataFrame({
            "text": texts,
            "severity": [2, 0],
            "binary": [1, 0],
        })
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp)
            results_dir = Path(tmp) / "results"
            build_outcomes(df, out_dir)
            extract_tfidf(df, out_dir, results_dir)

            features = pd.read_csv(out_dir / "tfidf_features.csv")
            outcomes = pd.read_csv(out_dir / "outcomes.csv")
            merged = features.merge(outcomes, on="ELS_ID")
            assert len(merged) == 2
            assert "severity" in merged.columns
            assert "binary" in merged.columns
