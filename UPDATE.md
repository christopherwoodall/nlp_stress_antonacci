# UPDATE.md

Changelog and project evolution notes for the NLP Stress Antonacci repository.

---

## 2026-08-19 — Package Restructure & Entry Points

### What changed

The original repository was a loose collection of standalone scripts. We restructured it into a proper, installable Python package with CLI entry points so the pipeline can be invoked consistently and dependencies are managed centrally.

### Files added

- `pyproject.toml` — build config, dependencies, and CLI entry points
- `src/nlp_stress/__init__.py` — package init
- `src/nlp_stress/whisperx_process.py` — moved from `scripts/whisperX_process.py`
- `src/nlp_stress/speaker_parsing.py` — moved from `scripts/Speaker_Parsing.py`
- `src/nlp_stress/tfidf_generation.py` — moved from `scripts/TF-IDF_Generation.py`
- `src/nlp_stress/generate_embeddings.py` — moved from `scripts/Generate_Embeddings.py`
- `src/nlp_stress/lda_preprocess.py` — moved from `scripts/01_preprocess_transcripts.py`
- `src/nlp_stress/lda_aggregate.py` — moved from `scripts/02_lda_to_topics.py`

### Files removed

- `scripts/01_preprocess_transcripts.py`
- `scripts/02_lda_to_topics.py`
- `scripts/Generate_Embeddings.py`
- `scripts/Speaker_Parsing.py`
- `scripts/TF-IDF_Generation.py`
- `scripts/whisperX_process.py`

(The shell batch script `whisperX_batch.sh` and the `GPT-4o_Standardized_Prompt` file remain in `scripts/`.)

### Entry points added

After `uv pip install -e .`, the following CLI commands are available:

| Command | Source module | What it does |
|---|---|---|
| `whisperx-process` | `nlp_stress.whisperx_process:main` | Transcribe/diarize a single audio file with WhisperX |
| `speaker-parse` | `nlp_stress.speaker_parsing:main` | Extract participant-only speech from diarized transcripts |
| `tfidf-generate` | `nlp_stress.tfidf_generation:main` | Compute and filter TF-IDF features |
| `embeddings-generate` | `nlp_stress.generate_embeddings:main` | Generate RoBERTa participant & sentence embeddings |
| `lda-preprocess` | `nlp_stress.lda_preprocess:main` | Tokenize transcripts into sentences for DLATK/MALLET |
| `lda-aggregate` | `nlp_stress.lda_aggregate:main` | Aggregate sentence-level LDA topics to participant-level |
| `pull-dataset` | `nlp_stress.pull_dataset:main` | Pull or generate the dataset for the pipeline |

### Comments restored

The original scripts contained extensive inline comments and section headers (e.g. `# --- 1. Setup ---`, `# --- 2. Helper Functions ---`) that were accidentally stripped during the initial package migration. These have been restored to all package modules so the code flow and intent remain clear.

### Bug fixes during migration

- **nltk `punkt_tab` resource**: Newer NLTK versions require the `punkt_tab` tokenizer data in addition to `punkt`. Added `nltk.download("punkt_tab", quiet=True)` to `lda_preprocess.py` to prevent a `LookupError` on first run.

## 2026-08-19 — R Analysis Notebooks Translated to Python

### What changed

All four R Markdown analysis notebooks (`analysis/*.Rmd`) have been translated into a new Python package at `src/analysis/`. The translation preserves the exact modeling logic, nested CV structure, and analysis flow from the R originals, while adding train/test/predict capability via ONNX model serialization.

### Files added

- `src/analysis/__init__.py`
- `src/analysis/_base.py` — shared utilities (nested CV, metrics, data loading, ONNX I/O)
- `src/analysis/_plots.py` — matplotlib/seaborn replications of key R figures
- `src/analysis/cli.py` — unified CLI entry point
- `src/analysis/liwc.py` — LIWC_Analyses.Rmd
- `src/analysis/tfidf.py` — TF-IDF_Analyses.Rmd
- `src/analysis/lda.py` — LDA_Analyses.Rmd
- `src/analysis/embeddings.py` — SentenceEmbedding_Analysis.Rmd

### New dependencies

`matplotlib`, `seaborn`, `umap-learn`, `hdbscan`, `skl2onnx`, `onnxruntime`, `openpyxl`, `statsmodels`, `joblib`

### New entry point

| Command | Source module | What it does |
|---|---|---|
| `analysis` | `analysis.cli:main` | Unified CLI for all feature-type analyses |

### User interaction

```bash
# Run cross-validation analysis (replicates the R notebook)
analysis --feature-type liwc --mode cv --data-dir data/ --output-dir results/

# Train final model on full data, save to ONNX
analysis --feature-type liwc --mode train --data-dir data/ --output-dir results/

# Evaluate saved model on test data
analysis --feature-type liwc --mode test --model-path results/liwc_model.onnx --test-data data/test.csv --output results/test_metrics.json

# Predict on new data
analysis --feature-type liwc --mode predict --model-path results/liwc_model.onnx --input data/new.csv --output predictions.csv
```

Feature types: `liwc`, `tfidf`, `lda`, `embeddings`. Modes: `cv`, `train`, `test`, `predict`.

### R → Python translation notes

- `cv.glmnet(alpha, nfolds=5)` → `ElasticNetCV(l1_ratio=alpha, cv=5)`
- `glmnet(alpha, lambda)` → `ElasticNet(l1_ratio=alpha, alpha=lambda)`
- `prcomp(center=T, scale.=T)` → `StandardScaler` + `PCA`
- `lm()` → `LinearRegression`
- `glm(family=binomial)` → `LogisticRegression`
- Nested CV: outer `KFold` + inner `ElasticNetCV` with alpha grid (0–1 by 0.05)

### Model persistence

Trained models are saved in ONNX format (via `skl2onnx`) for framework-agnostic inference. Scalers and PCA objects are saved with `joblib`. Coefficients are exported to CSV to match the R workflow (especially for the embedding dot-product projection).

### Visualization

Key figures from the R notebooks are replicated with matplotlib/seaborn:
- Predicted vs. observed scatterplots
- Feature importance bar charts
- Scree plots
- Incremental R² bar charts (LIWC)
- UMAP scatter + cluster violin plots (embeddings)

### Environment note

The original `.venv` was on Python 3.14, which broke C-extension builds for some dependencies (e.g. `ruamel-yaml-clib`). We recreated the venv with Python 3.12 and installed the package in editable mode.

---

## 2026-08-19 — Public Substitute Datasets, Synthetic Generator, Evaluator & Visualizer

### What changed

Since the original ELS clinical interview data is not publicly available, we built a complete substitute pipeline:
1. **Public dataset loaders** that download and unify labeled depression/distress datasets
2. **Label mapping** to a unified severity scale (0–3) and binary flag (0/1) matching the original study's outcomes
3. **Synthetic TESI response generator** using OpenRouter API with real TESI-C questions
4. **Evaluator** that runs generated text through trained ONNX models
5. **Visualizer** producing blog-quality comparison plots of LLM "depression scores"

### Files added

- `src/external_datasets/__init__.py`
- `src/external_datasets/loader.py` — MHDialog, Zenodo, Dreaddit, joangaes loaders with label mapping
- `src/external_datasets/cli.py` — `datasets-load` entry point
- `src/synthetic/__init__.py`
- `src/synthetic/generator.py` — OpenRouter API client for TESI question responses
- `src/synthetic/evaluator.py` — feature extraction + ONNX prediction on synthetic transcripts
- `src/synthetic/visualizer.py` — matplotlib/seaborn comparison plots
- `src/synthetic/cli.py` — unified `synthetic` entry point with `generate`, `evaluate`, `visualize` subcommands
- `config/example_generator.yaml` — example config with 16 real TESI-C questions from VA PDF
- `data/TESI-C.pdf` — downloaded from VA National Center for PTSD
- `tests/test_external_datasets_loader.py` — unit tests for dataset loaders
- `tests/test_synthetic_generator.py` — unit tests for config loading
- `tests/test_synthetic_evaluator.py` — unit tests for transcript loading
- `NOTE.md` — research note explaining the substitute dataset strategy and limitations
- `PIPELINE.md` — end-to-end guide from data loading to LLM comparison plots

### New dependencies

`requests`, `pyyaml`, `datasets`, `tqdm`

### New entry points

| Command | Source module | What it does |
|---|---|---|
| `datasets-load` | `external_datasets.cli:main` | Download and unify public substitute datasets |
| `synthetic` | `synthetic.cli:main` | Unified CLI for generate / evaluate / visualize |

### Usage

```bash
# Load public datasets
datasets-load --sources all --output data/unified_dataset.csv

# Generate synthetic TESI responses (requires OPENROUTER_API_KEY)
synthetic generate --config config/example_generator.yaml --output-dir synthetic_transcripts/

# Evaluate LLM responses with trained models
synthetic evaluate --transcripts synthetic_transcripts/ --models-dir results/ --output results/evaluations.csv

# Create comparison plots
synthetic visualize --evaluations results/evaluations.csv --output-dir results/synthetic/
```

### Label mapping

| Dataset | Original Label | Severity | Binary |
|---------|---------------|----------|--------|
| MHDialog | No | 0 | 0 |
| MHDialog | Minor | 1 | 1 |
| MHDialog | Moderate | 2 | 1 |
| MHDialog | Severe | 3 | 1 |
| Zenodo | Any depression type | 2 | 1 |
| Dreaddit | no_stress | 0 | 0 |
| Dreaddit | stress | 1 | 1 |
| joangaes | 0 (not depressed) | 0 | 0 |
| joangaes | 1 (depressed) | 2 | 1 |

### TESI-C questions

The 16 TESI-C questions were extracted from the VA PDF (`data/TESI-C.pdf`) downloaded from:
https://www.ptsd.va.gov/professional/assessment/documents/TESI-C.pdf

Questions cover: accidents, witnessed accidents, disasters, bereavement, hospitalization, separation, physical assault, threats, mugging/kidnapping, animal attacks, family violence, community violence, unwanted touch, and open-ended "worst event."

### Tests

All 12 new unit tests pass:
- Text hash deduplication
- MHDialog JSON dialogue extraction
- Zenodo label mapping
- load_all concatenation and deduplication
- Config YAML loading
- Transcript directory loading

---

## 2026-08-19 — Makefile, Pipeline Scripts, Integration Tests & Documentation

### What changed

Added a self-documenting Makefile and supporting scripts to make the pipeline runnable with simple commands like `make embeddings`, `make tfidf`, `make mock-eval`, and `make all`.

### Files added

- `Makefile` — self-documenting with `##` comment trick; targets: help, hydrate, datasets, features, train, embeddings, tfidf, eval, mock-eval, visualize, report, all, clean, pytest, test-pipeline, test-all
- `scripts/hydrate.py` — downloads TESI PDF, loads public datasets, generates sample data
- `scripts/build_features.py` — extracts TF-IDF + RoBERTa embeddings from unified dataset
- `scripts/train_model.py` — trains elastic-net model on substitute data and saves to ONNX
- `scripts/mock_generator.py` — creates fake synthetic TESI transcripts for testing (no API cost)
- `scripts/build_report.py` — generates REPORT.md from pipeline results
- `scripts/run_pipeline_test.py` — end-to-end integration test: data → features → train → predict → verify

### New dependencies

`sentence-transformers` (for RoBERTa embedding extraction in build_features.py)

### Makefile targets

| Target | Purpose |
|--------|---------|
| `make hydrate` | Download TESI PDF + public datasets + sample data |
| `make datasets` | Load and unify public substitute datasets |
| `make features` | Extract TF-IDF and embedding features |
| `make embeddings` | Train model on RoBERTa embeddings |
| `make tfidf` | Train model on TF-IDF features |
| `make mock-eval` | Evaluate fake transcripts (no API cost) |
| `make eval` | Evaluate with real LLM API |
| `make visualize` | Create comparison plots |
| `make report` | Generate REPORT.md summary |
| `make all` | Run full pipeline end-to-end |
| `make test` | Run unit tests |
| `make test-pipeline` | Run end-to-end integration test |

### Per-feature training

You can now train on specific feature types:
```bash
make embeddings   # RoBERTa embeddings (default)
make tfidf        # TF-IDF features
```

### Bug fix

Fixed `ElasticNetCV` crash when `l1_ratio=0` (pure Ridge) was included in the alpha grid. sklearn cannot auto-generate an alpha grid without an L1 penalty. Changed the default grid from `np.arange(0, 1.01, 0.05)` to `np.arange(0.05, 1.01, 0.05)` in both `src/analysis/_base.py` and `scripts/train_model.py`.

### Tests

All 39 tests pass (up from 12):
- 12 original tests (dataset loaders, synthetic generator/evaluator)
- 8 new tests for build_features.py (outcomes, TF-IDF shape, vectorizer, merge)
- 4 new tests for mock_generator.py (directory structure, content, metadata, determinism)
- 2 new integration tests (full train→predict cycle, CLI invocation)
- 13 new tests for Makefile (target existence, descriptions)

### Pipeline verification

`make test-pipeline` runs a complete end-to-end test in a temporary directory:
1. Creates 100 synthetic samples (50 depressed, 50 normal)
2. Extracts TF-IDF features
3. Trains elastic-net model
4. Saves to ONNX
5. Loads ONNX and predicts
6. Verifies predictions are finite and span a meaningful range

### Documentation updates

- `PIPELINE.md` — rewritten with Makefile-first instructions; added per-feature targets
- `NOTE.md` — added Makefile quick-start section and per-feature training examples

---
