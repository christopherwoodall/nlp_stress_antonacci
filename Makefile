# NLP Stress Antonacci Pipeline Makefile
# Self-documenting: run `make` or `make help` to see all targets.

PYTHON := python
PIP := uv pip

# --- Help ---

.PHONY: help
help:  ## Show this help message
	@grep -E '^[a-zA-Z0-9_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

# --- Environment ---

.PHONY: install
install:  ## Install package in editable mode with all dependencies
	$(PIP) install -e .

# --- Data hydration ---

.PHONY: hydrate
hydrate:  ## Download TESI PDF, public datasets, and generate sample data
	$(PYTHON) scripts/hydrate.py

.PHONY: datasets
datasets:  ## Load and unify public substitute datasets
	datasets-load --sources all --output data/unified_dataset.csv

# --- Feature extraction ---

.PHONY: features
features:  ## Extract TF-IDF and embedding features from unified dataset
	$(PYTHON) scripts/build_features.py --input data/unified_dataset.csv --output-dir data/ --results-dir results/

.PHONY: tfidf-features
tfidf-features:  ## Extract TF-IDF features only
	$(PYTHON) scripts/build_features.py --input data/unified_dataset.csv --output-dir data/ --results-dir results/

.PHONY: embedding-features
embedding-features:  ## Extract RoBERTa embedding features only
	$(PYTHON) scripts/build_features.py --input data/unified_dataset.csv --output-dir data/ --results-dir results/

# --- Model training ---

.PHONY: train
train:  ## Train predictive model (default: embeddings)
	$(PYTHON) scripts/train_model.py --feature-type embeddings --data-dir data/ --output-dir results/

.PHONY: embeddings
embeddings:  ## Train model on RoBERTa embeddings
	$(PYTHON) scripts/train_model.py --feature-type embeddings --data-dir data/ --output-dir results/

.PHONY: tfidf
tfidf:  ## Train model on TF-IDF features
	$(PYTHON) scripts/train_model.py --feature-type tfidf --data-dir data/ --output-dir results/

# --- Evaluation ---

.PHONY: eval
eval:  ## Generate synthetic TESI responses and evaluate (requires OPENROUTER_API_KEY)
	synthetic generate --config config/example_generator.yaml --output-dir synthetic_transcripts/
	$(PYTHON) scripts/build_features.py --input data/unified_dataset.csv --output-dir data/ --results-dir results/
	synthetic evaluate --transcripts synthetic_transcripts/ --models-dir results/ --output results/evaluations.csv --feature-type embeddings

.PHONY: mock-eval
mock-eval:  ## Run evaluation with fake transcripts (no API needed)
	$(PYTHON) scripts/mock_generator.py --output-dir synthetic_transcripts/ --config config/example_generator.yaml
	$(PYTHON) scripts/build_features.py --input data/unified_dataset.csv --output-dir data/ --results-dir results/
	synthetic evaluate --transcripts synthetic_transcripts/ --models-dir results/ --output results/evaluations.csv --feature-type embeddings

# --- Visualization ---

.PHONY: visualize
visualize:  ## Create comparison plots from evaluations
	synthetic visualize --evaluations results/evaluations.csv --output-dir results/synthetic/

.PHONY: plot
plot: visualize  ## Alias for visualize

# --- Reporting ---

.PHONY: report
report:  ## Generate summary report of all results
	$(PYTHON) scripts/build_report.py --results-dir results/ --output REPORT.md

# --- Full pipeline ---

.PHONY: all
all:  ## Run full pipeline: hydrate → datasets → features → train → mock-eval → visualize → report
	$(MAKE) hydrate
	$(MAKE) features
	$(MAKE) train
	$(MAKE) mock-eval
	$(MAKE) visualize
	$(MAKE) report

# --- Testing ---

.PHONY: test
pytest:  ## Run unit tests
	$(PYTHON) -m pytest tests/ -v

.PHONY: test-pipeline
test-pipeline:  ## Run end-to-end pipeline integration test (no API needed)
	$(PYTHON) scripts/run_pipeline_test.py

.PHONY: test-all
test-all: pytest test-pipeline  ## Run all tests (unit + integration)

# --- Cleanup ---

.PHONY: clean
clean:  ## Remove generated data, results, and synthetic transcripts
	rm -rf data/unified_dataset.csv data/*_features.csv data/outcomes.csv
	rm -rf results/ synthetic_transcripts/ REPORT.md
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
