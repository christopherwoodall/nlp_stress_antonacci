"""
test_external_datasets_loader.py

Tests for the public dataset loaders.
"""

import pandas as pd
import pytest

from external_datasets.loader import (
    _extract_user_text,
    _standardize_df,
    _text_hash,
    load_mhdialog,
    load_zenodo_depression,
)


class TestHelpers:
    def test_text_hash_deterministic(self):
        """Same text should produce same hash."""
        h1 = _text_hash("Hello world")
        h2 = _text_hash("Hello world")
        assert h1 == h2

    def test_text_hash_normalization(self):
        """Whitespace differences should be normalized."""
        h1 = _text_hash("Hello   world")
        h2 = _text_hash("hello world")
        assert h1 == h2

    def test_standardize_df(self):
        """Standardize adds source and text_hash columns."""
        df = pd.DataFrame({
            "text": ["sample text"],
            "severity": [1],
            "binary": [1],
            "original_label": ["minor"],
        })
        result = _standardize_df(df, "test_source")
        assert "source" in result.columns
        assert "text_hash" in result.columns
        assert result["source"].iloc[0] == "test_source"


class TestMHDialogExtraction:
    def test_extract_user_text_json(self):
        """Parse JSON-formatted MHDialog turns."""
        dialogue = '[{"round": 1, "user": "I feel sad.", "supporter": "I hear you."}]'
        result = _extract_user_text(dialogue)
        assert result == "I feel sad."

    def test_extract_user_text_plain(self):
        """Plain text should pass through."""
        dialogue = "Just a plain string"
        result = _extract_user_text(dialogue)
        assert result == "Just a plain string"

    def test_extract_user_text_empty(self):
        """Empty string returns empty."""
        assert _extract_user_text("") == ""


class TestZenodoLoader:
    def test_label_mapping(self, tmp_path):
        """Zenodo labels map correctly to severity/binary."""
        csv_path = tmp_path / "zenodo.csv"
        csv_path.write_text("Tweets,Labels\nI feel down,major\nGood day,postpartum\n")
        df = load_zenodo_depression(str(csv_path))
        assert len(df) == 2
        assert df["severity"].iloc[0] == 2
        assert df["binary"].iloc[0] == 1
        assert df["source"].iloc[0] == "zenodo_depression"


class TestLoadAll:
    def test_load_all_with_mock(self, monkeypatch, tmp_path):
        """load_all should concatenate and deduplicate."""
        # Mock the individual loaders to avoid network calls
        def mock_load_mhdialog(**kwargs):
            return pd.DataFrame({
                "text": ["duplicate text", "unique mhdialog"],
                "source": ["mhdialog", "mhdialog"],
                "severity": [1, 2],
                "binary": [1, 1],
                "original_label": ["Minor", "Moderate"],
                "text_hash": ["hash1", "hash2"],
            })

        def mock_load_zenodo(**kwargs):
            return pd.DataFrame({
                "text": ["duplicate text", "unique zenodo"],
                "source": ["zenodo_depression", "zenodo_depression"],
                "severity": [2, 2],
                "binary": [1, 1],
                "original_label": ["major", "bipolar"],
                "text_hash": ["hash1", "hash3"],
            })

        monkeypatch.setattr(
            "external_datasets.loader.load_mhdialog", mock_load_mhdialog
        )
        monkeypatch.setattr(
            "external_datasets.loader.load_zenodo_depression", mock_load_zenodo
        )

        from external_datasets.loader import load_all

        df = load_all(
            sources=["mhdialog", "zenodo_depression"],
            deduplicate=True,
        )
        # 4 rows - 1 duplicate = 3
        assert len(df) == 3
        assert set(df["source"].unique()) == {"mhdialog", "zenodo_depression"}
