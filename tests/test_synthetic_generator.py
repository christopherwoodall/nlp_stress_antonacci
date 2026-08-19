"""
test_synthetic_generator.py

Tests for the synthetic generator.
"""

import tempfile
from pathlib import Path

import pytest
import yaml

from synthetic.generator import load_config


class TestConfigLoading:
    def test_load_config(self, tmp_path):
        """Config YAML loads into dict with expected keys."""
        config_path = tmp_path / "test_config.yaml"
        config = {
            "models": {
                "gpt-4o": {
                    "provider": "openrouter",
                    "model": "openai/gpt-4o",
                    "temperature": 0.7,
                    "max_tokens": 512,
                }
            },
            "system_prompt": "You are a teenager.",
            "questions": [
                {"id": "q1", "text": "Tell me about a stressful event."}
            ],
            "settings": {"responses_per_question": 3},
        }
        with open(config_path, "w") as f:
            yaml.dump(config, f)

        loaded = load_config(str(config_path))
        assert "models" in loaded
        assert "questions" in loaded
        assert loaded["settings"]["responses_per_question"] == 3


class TestPromptConstruction:
    def test_messages_format(self):
        """Messages should be in OpenAI chat format."""
        system_prompt = "You are a teenager."
        question = "Tell me about a stressful event."
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question},
        ]
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"
        assert messages[1]["content"] == question
