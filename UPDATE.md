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
