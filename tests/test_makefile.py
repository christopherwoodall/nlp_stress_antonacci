"""
test_makefile.py

Verify that the Makefile contains expected targets.
"""

import re
from pathlib import Path

import pytest

MAKEFILE_PATH = Path(__file__).parent.parent / "Makefile"


class TestMakefileTargets:
    @pytest.fixture(scope="class")
    def makefile_content(self):
        if not MAKEFILE_PATH.exists():
            pytest.skip("Makefile not found")
        return MAKEFILE_PATH.read_text()

    @pytest.fixture(scope="class")
    def targets(self, makefile_content):
        """Extract all .PHONY targets from Makefile."""
        # Match lines like: target:  ## description
        pattern = re.compile(r'^[a-zA-Z0-9_-]+:', re.MULTILINE)
        return pattern.findall(makefile_content)

    def test_help_target_exists(self, targets):
        assert "help:" in targets

    def test_hydrate_target_exists(self, targets):
        assert "hydrate:" in targets

    def test_datasets_target_exists(self, targets):
        assert "datasets:" in targets

    def test_features_target_exists(self, targets):
        assert "features:" in targets

    def test_train_target_exists(self, targets):
        assert "train:" in targets

    def test_embeddings_target_exists(self, targets):
        assert "embeddings:" in targets

    def test_tfidf_target_exists(self, targets):
        assert "tfidf:" in targets

    def test_eval_target_exists(self, targets):
        assert "eval:" in targets

    def test_mock_eval_target_exists(self, targets):
        assert "mock-eval:" in targets

    def test_visualize_target_exists(self, targets):
        assert "visualize:" in targets

    def test_report_target_exists(self, targets):
        assert "report:" in targets

    def test_all_target_exists(self, targets):
        assert "all:" in targets

    def test_pytest_target_exists(self, targets):
        assert "pytest:" in targets

    def test_test_pipeline_target_exists(self, targets):
        assert "test-pipeline:" in targets

    def test_clean_target_exists(self, targets):
        assert "clean:" in targets

    def test_targets_have_descriptions(self, makefile_content):
        """All .PHONY targets should have ## descriptions for help."""
        # Count targets with ## comments
        lines = makefile_content.splitlines()
        target_lines = [l for l in lines if re.match(r'^[a-zA-Z0-9_-]+:', l)]
        described = [l for l in target_lines if "##" in l]
        # Most targets should have descriptions (allow a few exceptions)
        assert len(described) >= len(target_lines) - 2
