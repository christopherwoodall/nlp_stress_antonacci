# NLP Stress Project

This repository includes scripts to reproduce the analyses reported in:

Antonacci, C., Uy, J. P., Kwan, K., Giampetruzzi, E., Jones, S., Pennebaker, J. W., & Gotlib, I. H. (2026). Natural language processing of youth speech predicts psychopathology across adolescence. *Nature Mental Health*. https://doi.org/10.1038/s44220-026-00683-9

> **Update (2026-08-19):** This repository has been restructured into an installable Python package with CLI entry points, a Makefile for running the full pipeline, and a substitute dataset strategy for researchers who cannot access the original clinical data. See [NOTE.md](NOTE.md) for details on the substitute approach and [PIPELINE.md](PIPELINE.md) for the end-to-end guide.

---

## Table of Contents

- [Overview](#overview)
- [Quick Start](#quick-start)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [CLI Entry Points](#cli-entry-points)
- [Makefile Targets](#makefile-targets)
- [Two Pipeline Paths](#two-pipeline-paths)
  - [Path A: Original Clinical Pipeline](#path-a-original-clinical-pipeline)
  - [Path B: Substitute Dataset Pipeline](#path-b-substitute-dataset-pipeline)
- [Documentation](#documentation)
- [Citation](#citation)

---

## Overview

The workflow spans **raw audio transcription**, **speaker correction**, **NLP feature generation**, and **predictive modeling** of internalizing outcomes from **TESI (Traumatic Events Screening Inventory) stress interviews**.

The repository now supports two distinct usage paths:

1. **Original clinical pipeline** — for researchers with access to the ELS study data
2. **Substitute dataset pipeline** — for researchers using publicly available labeled depression/distress datasets

Both paths share the same analysis methodology (elastic-net nested CV, ONNX model export) and can evaluate synthetic LLM-generated TESI responses.

---

## Quick Start

```bash
# Install
uv pip install -e .

# See all available commands
make help

# Run the full substitute-dataset pipeline (no API costs)
make all

# Run tests
make test-all
```

---

## Installation

Requires Python 3.10+.

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate    # Windows

# Install in editable mode
uv pip install -e .
```

This installs all dependencies and registers CLI entry points.

---

## Project Structure

```
.
├── src/
│   ├── nlp_stress/          # Original pipeline scripts (transcription → features)
│   ├── analysis/            # R→Python translated analysis package
│   ├── external_datasets/   # Public substitute dataset loaders
│   └── synthetic/           # LLM generator, evaluator, visualizer
├── scripts/                 # Pipeline orchestration scripts
│   ├── hydrate.py           # Download TESI PDF + public datasets
│   ├── build_features.py    # Extract TF-IDF + embeddings
│   ├── train_model.py       # Train elastic-net + export ONNX
│   ├── mock_generator.py    # Fake transcripts for testing
│   ├── build_report.py      # Generate REPORT.md
│   └── run_pipeline_test.py # End-to-end integration test
├── config/
│   ├── example_generator.yaml      # TESI questions + LLM config
│   ├── persona_depressed.yaml      # Depressed persona for evaluation
│   ├── persona_resilient.yaml      # Resilient persona for evaluation
│   ├── persona_depressed_minimal.yaml  # Budget-conscious depressed config
│   └── persona_resilient_minimal.yaml  # Budget-conscious resilient config
├── tests/                   # Unit + integration tests
├── data/                    # Data directory (TESI PDF, transcripts, datasets)
├── results/                 # Model outputs, plots, reports
│   └── REPORT.md            # Auto-generated results summary
├── analysis/                # Original R Markdown notebooks
├── Makefile                 # Self-documenting build targets
├── pyproject.toml           # Package config + dependencies
├── README.md                # This file
├── NOTE.md                  # Substitute dataset strategy + limitations
├── PIPELINE.md              # Step-by-step pipeline guide
└── UPDATE.md                # Changelog
```

---

## CLI Entry Points

After installation, the following commands are available:

### Original Pipeline

| Command | What it does |
|---------|-------------|
| `whisperx-process` | Transcribe/diarize audio with WhisperX |
| `speaker-parse` | Extract participant-only speech |
| `tfidf-generate` | Compute filtered TF-IDF features |
| `embeddings-generate` | Generate RoBERTa embeddings |
| `lda-preprocess` | Tokenize transcripts for DLATK/MALLET |
| `lda-aggregate` | Aggregate LDA topics to participant-level |
| `pull-dataset` | Pull or generate sample dataset |

### Analysis & Modeling

| Command | What it does |
|---------|-------------|
| `analysis` | Unified CLI for LIWC/TF-IDF/LDA/embedding analyses |

### Substitute Dataset Pipeline

| Command | What it does |
|---------|-------------|
| `datasets-load` | Download and unify public substitute datasets |
| `synthetic` | Generate, evaluate, or visualize synthetic TESI responses |

---

## Makefile Targets

Run `make` or `make help` to see all targets.

| Target | Purpose |
|--------|---------|
| `make hydrate` | Download TESI PDF + public datasets + sample data |
| `make datasets` | Load and unify public substitute datasets |
| `make features` | Extract TF-IDF and embedding features |
| `make embeddings` | Train model on RoBERTa embeddings |
| `make tfidf` | Train model on TF-IDF features |
| `make mock-eval` | Evaluate fake transcripts (no API cost) |
| `make eval` | Evaluate with real LLM API (requires `OPENROUTER_API_KEY`) |
| `make visualize` | Create comparison plots |
| `make report` | Generate `results/REPORT.md` |
| `make all` | Run full pipeline end-to-end |
| `make test` | Run unit tests |
| `make test-pipeline` | Run end-to-end integration test |
| `make clean` | Remove generated files |

---

## Two Pipeline Paths

### Path A: Original Clinical Pipeline

For researchers with access to the ELS study data (stored on Stanford Box).

```bash
# 1. Transcription
whisperx-process --input audio.wav --output transcript.txt

# 2. Speaker parsing
speaker-parse --input transcript.txt --output participant.txt

# 3. Feature generation
tfidf-generate --input participant.txt --output features.csv
embeddings-generate --input participant.txt --output embeddings.csv

# 4. Analysis (requires clinical outcome files)
analysis --feature-type tfidf --mode cv --data-dir data/ --output-dir results/
analysis --feature-type tfidf --mode train --data-dir data/ --output-dir results/
```

See the `analysis/` directory for the original R Markdown notebooks.

### Path B: Substitute Dataset Pipeline

For researchers without access to the original clinical data. Uses publicly available labeled datasets.

```bash
# 1. Load public datasets
make datasets   # or: datasets-load --sources all --output data/unified_dataset.csv

# 2. Extract features
make features   # or: python scripts/build_features.py --input data/unified_dataset.csv ...

# 3. Train model
make tfidf      # or: python scripts/train_model.py --feature-type tfidf ...

# 4. Generate synthetic TESI responses (mock = no API cost)
make mock-eval  # or: python scripts/mock_generator.py ...

# 5. Evaluate and visualize
make visualize  # or: synthetic visualize --evaluations results/evaluations.csv ...

# 6. Generate report
make report     # or: python scripts/build_report.py --results-dir results/ ...
```

See [NOTE.md](NOTE.md) for the substitute dataset strategy, label mapping, and limitations.

---

## Documentation

| File | What it covers |
|------|-------------|
| [README.md](README.md) | This file — overview, quick start, structure |
| [NOTE.md](NOTE.md) | Substitute dataset strategy, label mapping, model performance, limitations, persona experiments |
| [PIPELINE.md](PIPELINE.md) | Step-by-step guide for both pipeline paths |
| [UPDATE.md](UPDATE.md) | Changelog with all changes and additions |
| [results/REPORT.md](results/REPORT.md) | Auto-generated summary of latest pipeline run |

---

## Citation

If you use the original pipeline:

- Antonacci, C., Uy, J. P., Kwan, K., Giampetruzzi, E., Jones, S., Pennebaker, J. W., & Gotlib, I. H. (2026). Natural language processing of youth speech predicts psychopathology across adolescence. *Nature Mental Health*. https://doi.org/10.1038/s44220-026-00683-9

If you use the substitute dataset pipeline, also cite:

- Zhang, Y., et al. (2026). MHDash: An Online Platform for Benchmarking Mental Health-Aware AI Assistants. arXiv:2602.00353.
- Nusrat, M. O., Shahzad, W., & Jamal, S. A. (2024). Multi Class Depression Detection Through Tweets using Artificial Intelligence. arXiv:2404.13104.

---

## See Also

| File | What it covers |
|------|-------------|
| [NOTE.md](NOTE.md) | Substitute dataset strategy, label mapping, limitations, model performance, persona experiments |
| [PIPELINE.md](PIPELINE.md) | Step-by-step guide for both pipeline paths |
| [UPDATE.md](UPDATE.md) | Changelog with all changes and additions |
| [results/REPORT.md](results/REPORT.md) | Auto-generated summary of latest pipeline run |

---

## Contact

For access to the original ELS clinical data, contact the Stanford Neurodevelopment, Affect, and Psychopathology Laboratory (PI: Ian Gotlib).
