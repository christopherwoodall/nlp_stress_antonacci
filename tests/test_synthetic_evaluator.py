"""
test_synthetic_evaluator.py

Tests for the synthetic evaluator.
"""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from synthetic.evaluator import load_transcripts


class TestLoadTranscripts:
    def test_load_transcripts_basic(self, tmp_path):
        """Load transcripts from directory tree."""
        # Create fake transcript structure
        model_dir = tmp_path / "gpt-4o"
        model_dir.mkdir()
        (model_dir / "tesi_1_1_run1.txt").write_text("I felt scared during the accident.")
        (model_dir / "tesi_1_1_run2.txt").write_text("It was a bad storm.")
        (model_dir / "tesi_1_2_run1.txt").write_text("I saw it happen.")

        df = load_transcripts(str(tmp_path))
        assert len(df) == 3
        assert set(df["model"].unique()) == {"gpt-4o"}
        assert set(df["question_id"].unique()) == {"tesi_1_1", "tesi_1_2"}

    def test_load_transcripts_empty(self, tmp_path):
        """Empty directory should raise ValueError."""
        with pytest.raises(ValueError):
            load_transcripts(str(tmp_path))
