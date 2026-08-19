# End-to-End Pipeline Guide

This document walks through the complete pipeline from loading substitute datasets to comparing LLMs on synthetic TESI interview responses.

**Quick start:** Run `make help` to see all available targets, or `make all` to run the full pipeline.

---

## Prerequisites

```bash
# Python 3.10+ required
python --version

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate    # Windows

# Install package in editable mode
uv pip install -e .
```

All CLI entry points should now be available:
- `datasets-load`
- `synthetic`
- `analysis`
- `whisperx-process`, `speaker-parse`, `tfidf-generate`, etc.

---

## Makefile Quick Reference

Run `make` or `make help` to see all targets. Key targets:

| Target | What it does |
|--------|-------------|
| `make hydrate` | Download TESI PDF, public datasets, sample data |
| `make datasets` | Load and unify public substitute datasets |
| `make features` | Extract TF-IDF + embedding features |
| `make embeddings` | Train model on RoBERTa embeddings |
| `make tfidf` | Train model on TF-IDF features |
| `make mock-eval` | Evaluate fake transcripts (no API cost) |
| `make eval` | Evaluate with real LLM API (requires OPENROUTER_API_KEY) |
| `make visualize` | Create comparison plots |
| `make report` | Generate REPORT.md summary |
| `make all` | Run full pipeline end-to-end |
| `make test` | Run unit tests |
| `make test-pipeline` | Run end-to-end integration test |
| `make clean` | Remove generated files |

---

## Phase 1: Load Public Substitute Datasets

Since the original ELS clinical data is not publicly available, we use labeled substitute datasets.

### Option A: Use the Makefile (recommended)

```bash
make hydrate   # Downloads TESI PDF + public datasets + sample data
```

### Option B: Manual CLI commands

---

## Phase 1: Load Public Substitute Datasets

Since the original ELS clinical data is not publicly available, we use labeled substitute datasets.

### Option A: Load all default sources (recommended)

```bash
datasets-load --sources all --output data/unified_dataset.csv --cache-dir ~/.cache/hf
```

This loads:
- **MHDialog** (HuggingFace) — synthetic mental-health dialogues with risk levels
- **Zenodo Multi-Class Depression** (~2,800 psychiatrist-verified tweets)
- **joangaes/depression** (HuggingFace) — 28K Reddit posts with binary labels

### Option B: Load specific sources

```bash
datasets-load \
  --sources mhdialog,zenodo_depression \
  --output data/unified_dataset.csv
```

### Option C: Include Dreaddit (requires manual Kaggle download)

Download `dreaddit` from Kaggle first, then:

```bash
datasets-load \
  --sources mhdialog,zenodo_depression,dreaddit \
  --dreaddit-csv /path/to/dreaddit.csv \
  --output data/unified_dataset.csv
```

### Verify output

```bash
head data/unified_dataset.csv
# Columns: text, source, severity, binary, original_label, text_hash
```

---

## Phase 2: Prepare Features for Model Training

The pipeline extracts two feature representations from the unified dataset:

### Option A: Use the Makefile

```bash
make features   # Extracts both TF-IDF and embedding features
```

This creates:
- `data/tfidf_features.csv`
- `data/embedding_features.csv`
- `data/outcomes.csv`
- `results/tfidf_vectorizer.joblib`

### Option B: Manual Python scripts

```bash
python scripts/build_features.py --input data/unified_dataset.csv --output-dir data/ --results-dir results/
```

### Option C: Manual feature extraction (advanced)

```python
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

# Load unified data
df = pd.read_csv("data/unified_dataset.csv")

# Extract TF-IDF
vectorizer = TfidfVectorizer(max_features=5000, min_df=0.05)
X = vectorizer.fit_transform(df["text"])

# Save features
feature_df = pd.DataFrame(X.toarray(), columns=vectorizer.get_feature_names_out())
feature_df["ELS_ID"] = range(len(feature_df))  # synthetic IDs
feature_df.to_csv("data/tfidf_features.csv", index=False)

# Save outcomes
outcome_df = df[["severity", "binary"]].copy()
outcome_df["ELS_ID"] = range(len(outcome_df))
outcome_df.to_csv("data/outcomes.csv", index=False)

# Save vectorizer for later evaluation
import joblib
joblib.dump(vectorizer, "results/tfidf_vectorizer.joblib")
```

---

## Phase 3: Train Models

### Option A: Use the Makefile (per-feature targets)

```bash
make embeddings   # Train on RoBERTa embeddings (default)
make tfidf        # Train on TF-IDF features
```

These save:
- `results/embeddings_model.onnx` / `results/tfidf_model.onnx`
- Corresponding metrics, coefficients, and scaler files

### Option B: Manual CLI

```bash
# Train embeddings model
python scripts/train_model.py --feature-type embeddings --data-dir data/ --output-dir results/

# Train TF-IDF model
python scripts/train_model.py --feature-type tfidf --data-dir data/ --output-dir results/
```

### Option C: Original analysis CLI (requires clinical data format)

```bash
analysis --feature-type tfidf --mode train --data-dir data/ --output-dir results/
analysis --feature-type embeddings --mode train --data-dir data/ --output-dir results/
```

**Note:** The `analysis` CLI expects the original clinical data file names. For the substitute dataset pipeline, use `scripts/train_model.py` or the Makefile targets instead.

---

## Phase 4: Generate Synthetic TESI Responses

### Configure the generator

Edit `config/example_generator.yaml` or create your own:

```yaml
models:
  gpt-4o:
    provider: openrouter
    model: openai/gpt-4o
    temperature: 0.7
    max_tokens: 512
  claude-sonnet:
    provider: openrouter
    model: anthropic/claude-3.5-sonnet
    temperature: 0.7
    max_tokens: 512

system_prompt: |
  You are a teenager being interviewed about stressful life events...

questions:
  - id: tesi_1_1
    text: "Have you ever been in a really bad accident..."
  # ... (16 questions total)

settings:
  responses_per_question: 5
  output_dir: synthetic_transcripts/
```

### Set API key

```bash
export OPENROUTER_API_KEY="your-key-here"
```

Get a key at: https://openrouter.ai/keys

### Generate responses

**Option A: Makefile (real API)**
```bash
make eval   # Generates responses + evaluates with trained models
```

**Option B: Makefile (mock/fake transcripts — no API cost)**
```bash
make mock-eval   # Uses fake transcripts for testing
```

**Option C: Manual CLI**
```bash
# Real API
synthetic generate --config config/example_generator.yaml --output-dir synthetic_transcripts/

# Mock (no API)
python scripts/mock_generator.py --output-dir synthetic_transcripts/ --config config/example_generator.yaml
```

Output structure:
```
synthetic_transcripts/
  gpt-4o/
    tesi_1_1_run1.txt
    tesi_1_1_run2.txt
    ...
  claude-sonnet/
    tesi_1_1_run1.txt
    ...
  metadata.json
```

---

## Phase 5: Evaluate LLMs

Run the synthetic transcripts through trained models.

### Option A: Makefile (includes mock-eval for testing)

```bash
make mock-eval   # Uses fake transcripts + evaluates (no API cost)
make eval        # Real LLM responses + evaluates (requires OPENROUTER_API_KEY)
```

### Option B: Manual CLI

```bash
synthetic evaluate \
  --transcripts synthetic_transcripts/ \
  --models-dir results/ \
  --output results/evaluations.csv \
  --feature-type embeddings
```

If using TF-IDF features, also pass the saved vectorizer:

```bash
synthetic evaluate \
  --transcripts synthetic_transcripts/ \
  --models-dir results/ \
  --output results/evaluations.csv \
  --feature-type tfidf \
  --vectorizer-path results/tfidf_vectorizer.joblib
```

Output CSV columns:
- `model` — which LLM generated the response
- `question_id` — which TESI question
- `run` — which repetition
- `text` — the generated text
- `pred_*` — predictions from each trained model

---

## Phase 6: Visualize Comparisons

Generate blog-quality comparison plots.

### Option A: Makefile

```bash
make visualize   # Create all comparison plots
make report      # Generate REPORT.md summary
```

### Option B: Manual CLI

```bash
synthetic visualize \
  --evaluations results/evaluations.csv \
  --output-dir results/synthetic/
```

Produces:
- `average_severity.png` — bar chart of mean predicted severity per LLM
- `severity_distribution.png` — box plot of severity distributions
- `question_heatmap.png` — heatmap of severity by question × model
- `confidence_scatter.png` — prediction uncertainty vs severity
- `model_radar.png` — radar chart by TESI category

---

## Summary of Commands

### Makefile targets (recommended)

| Target | Purpose |
|--------|---------|
| `make hydrate` | Download TESI PDF + public datasets + sample data |
| `make datasets` | Load and unify public datasets |
| `make features` | Extract TF-IDF + embedding features |
| `make embeddings` | Train model on RoBERTa embeddings |
| `make tfidf` | Train model on TF-IDF features |
| `make mock-eval` | Evaluate fake transcripts (no API cost) |
| `make eval` | Evaluate with real LLM API |
| `make visualize` | Create comparison plots |
| `make report` | Generate REPORT.md summary |
| `make all` | Run full pipeline end-to-end |
| `make test` | Run unit tests |
| `make test-pipeline` | Run end-to-end integration test |
| `make clean` | Remove generated files |

### CLI commands (manual)

| Command | Purpose |
|---------|---------|
| `datasets-load --sources all --output data/unified.csv` | Load and unify public datasets |
| `python scripts/train_model.py --feature-type embeddings ...` | Train predictive model |
| `synthetic generate --config config/example_generator.yaml` | Generate LLM responses |
| `synthetic evaluate --transcripts ... --models-dir ...` | Score responses with models |
| `synthetic visualize --evaluations ... --output-dir ...` | Create comparison plots |

---

## Troubleshooting

### `datasets` library not found
```bash
uv pip install datasets
```

### `onnxruntime` not found
```bash
uv pip install onnxruntime
```

### `sentence-transformers` not found (for embeddings evaluation)
```bash
uv pip install sentence-transformers
```

### Rate limiting from OpenRouter
The generator has built-in exponential backoff. If you hit limits, reduce `responses_per_question` or add delays.

### No ONNX models found
Make sure you ran `analysis --mode train` first. Models are saved as `*_model.onnx` in the output directory.
