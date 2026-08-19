"""
test_mock_generator.py

Tests for the mock synthetic transcript generator.
"""

import tempfile
from pathlib import Path

import pytest
import yaml

from scripts.mock_generator import generate_mock_responses


class TestMockGenerator:
    def test_creates_directory_structure(self):
        """Should create model subdirectories with transcript files."""
        config = {
            "models": {
                "gpt-4o": {"provider": "openrouter", "model": "openai/gpt-4o"},
            },
            "questions": [
                {"id": "q1", "text": "Tell me about a stressful event."},
                {"id": "q2", "text": "How did you feel?"},
            ],
            "settings": {"responses_per_question": 3},
        }
        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "config.yaml"
            with open(config_path, "w") as f:
                yaml.dump(config, f)

            out_dir = Path(tmp) / "transcripts"
            generate_mock_responses(str(config_path), out_dir, seed=42)

            # Check directory structure
            model_dir = out_dir / "gpt-4o"
            assert model_dir.exists()

            # Check files exist
            files = list(model_dir.glob("*.txt"))
            assert len(files) == 6  # 2 questions × 3 runs

    def test_files_have_content(self):
        """Generated files should contain non-empty text."""
        config = {
            "models": {
                "test-model": {"provider": "mock", "model": "mock"},
            },
            "questions": [
                {"id": "tesi_1", "text": "Question one?"},
            ],
            "settings": {"responses_per_question": 2},
        }
        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "config.yaml"
            with open(config_path, "w") as f:
                yaml.dump(config, f)

            out_dir = Path(tmp) / "out"
            generate_mock_responses(str(config_path), out_dir, seed=123)

            for txt_file in (out_dir / "test-model").glob("*.txt"):
                content = txt_file.read_text()
                assert len(content) > 10

    def test_metadata_saved(self):
        """Metadata JSON should be created."""
        config = {
            "models": {"m": {"provider": "mock", "model": "mock"}},
            "questions": [{"id": "q", "text": "Q?"}],
            "settings": {"responses_per_question": 1},
        }
        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "config.yaml"
            with open(config_path, "w") as f:
                yaml.dump(config, f)

            out_dir = Path(tmp) / "out"
            generate_mock_responses(str(config_path), out_dir)

            meta_path = out_dir / "metadata.json"
            assert meta_path.exists()
            import json
            meta = json.loads(meta_path.read_text())
            assert meta["mock"] is True
            assert len(meta["responses"]) == 1

    def test_deterministic_with_same_seed(self):
        """Same seed should produce same files."""
        config = {
            "models": {"m": {"provider": "mock", "model": "mock"}},
            "questions": [{"id": "q", "text": "Q?"}],
            "settings": {"responses_per_question": 1},
        }
        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "config.yaml"
            with open(config_path, "w") as f:
                yaml.dump(config, f)

            out1 = Path(tmp) / "out1"
            out2 = Path(tmp) / "out2"
            generate_mock_responses(str(config_path), out1, seed=42)
            generate_mock_responses(str(config_path), out2, seed=42)

            f1 = list((out1 / "m").glob("*.txt"))[0].read_text()
            f2 = list((out2 / "m").glob("*.txt"))[0].read_text()
            assert f1 == f2
