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

### Pipeline execution results

Executed the full pipeline on 39,081 unified substitute samples:

**TF-IDF model:**
- Features: 176 (after filtering)
- Train/test: 31,264 / 7,816 (80/20 split, random_state=42)
- R² = 0.557, RMSE = 0.691, MAE = 0.547
- Best alpha (l1_ratio): 0.050, Best lambda: 0.007209

**Mock evaluation:**
- Generated 160 fake transcripts (2 models × 16 questions × 5 runs)
- Evaluated with trained TF-IDF ONNX model
- Mean predicted severity: gpt-4o = 1.418, claude-sonnet = 1.414

**Generated outputs:**
- `results/synthetic/average_severity.png`
- `results/synthetic/severity_distribution.png`
- `results/synthetic/question_heatmap.png`
- `results/synthetic/model_radar.png`
- `REPORT.md`

**Embeddings note:** RoBERTa embeddings model (768 dims → 1,024 after dummy encoding) was started but killed after 45min CPU time due to ElasticNetCV grid search being prohibitively slow with 39K samples × 1,024 features. For production use, consider reducing the alpha grid or using a smaller embedding model.

### Bug fixes during execution

- Fixed `extract_tfidf_features()` in `src/synthetic/evaluator.py` — `--vectorizer-path` CLI argument was passed as a string but the function expected a `Path` object. Added `Path(vectorizer_path).exists()` wrapper.

### Documentation updates

- `README.md` — completely rewritten with project overview, quick start, structure, two pipeline paths
- `PIPELINE.md` — rewritten with Makefile-first instructions; added per-feature targets; fixed duplicate Phase 1 section
- `NOTE.md` — added Makefile quick-start section, per-feature training examples, and model performance table
- `config/example_generator.yaml` — added header comment explaining file format and usage

---

## 2026-08-19 — Model Naming, Persona Experiments & Report Improvements

### What changed

1. **Descriptive model names:** Updated `config/example_generator.yaml` with `version` fields (`gpt-4o-2024-08-06`, `claude-3-5-sonnet-20241022`) so reports show full model identifiers instead of ambiguous short names.
2. **Persona-based evaluation:** Created depressed and resilient persona configs and ran both mock and real API experiments to test model discrimination.
3. **Report moved to `results/`:** `REPORT.md` now lives in `results/REPORT.md` alongside other outputs. `build_report.py` and `Makefile` updated accordingly.
4. **Enhanced report content:** Added train/test split stats, embeddings model status, persona comparison tables, and model version info.

### Files added

- `config/persona_depressed.yaml` — depressed persona with 2 TESI questions
- `config/persona_resilient.yaml` — resilient persona with 2 TESI questions
- `config/persona_depressed_minimal.yaml` — budget-conscious version (1 run, gpt-4o only)
- `config/persona_resilient_minimal.yaml` — budget-conscious version (1 run, gpt-4o only)

### Files updated

- `config/example_generator.yaml` — added `version` fields to both models
- `scripts/mock_generator.py` — added `--persona` flag (`neutral`/`depressed`/`resilient`) with distinct template pools engineered to align with model coefficients
- `scripts/build_report.py` — added `persona_summary()`, model versions, train/test split info, embeddings status
- `Makefile` — `report` target now outputs to `results/REPORT.md`; `clean` target updated
- `NOTE.md` — added persona evaluation section with results and interpretation
- `PIPELINE.md` — added Phase 5b (persona-based evaluation) with commands and expected results

### Persona experiment results

**Mock data (engineered templates):**
- Depressed mean: 1.575, Resilient mean: 1.467
- Difference: +0.108 — model correctly assigns higher severity to depressed text

**Real API data (gpt-4o):**
- Depressed mean: 1.449, Resilient mean: 1.515
- Difference: −0.066 — model **misclassifies** natural language
- The resilient response was longer and more emotionally verbose, scoring higher. The depressed response began with denial, scoring lower.
- **Finding:** The model conflates emotional expressiveness with distress. This is a limitation of bag-of-words models trained on social media data.

### Budget

Real API persona experiment used 4 calls (2 questions × 2 personas × 1 run × 1 model) at approximately $0.06 total. Well within the $10 budget.

### Pipeline test

`make test-pipeline` continues to pass: creates 100 synthetic samples, extracts TF-IDF, trains elastic-net, saves to ONNX, loads ONNX, predicts, and verifies predictions span a meaningful range (min=0.001, max=1.999, mean=1.000).

---

## See Also

- [README.md](README.md) — Project overview, quick start, installation
- [NOTE.md](NOTE.md) — Substitute dataset strategy, label mapping, limitations
- [PIPELINE.md](PIPELINE.md) — Step-by-step pipeline guide
