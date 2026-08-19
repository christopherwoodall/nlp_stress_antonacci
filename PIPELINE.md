# End-to-End Pipeline Guide

This document walks through the complete pipeline from loading substitute datasets to comparing LLMs on synthetic TESI interview responses.

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

The `analysis` CLI expects feature matrices in a specific format. You need to:

1. **Extract features** from the unified dataset text column
2. **Create outcome files** with severity and binary labels
3. **Merge** into the format expected by the analysis modules

### Example: TF-IDF features

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

### Example: Embeddings features

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-roberta-large-v1")
embeddings = model.encode(df["text"].tolist(), show_progress_bar=True)

emb_df = pd.DataFrame(embeddings)
emb_df["ELS_ID"] = range(len(emb_df))
emb_df.to_csv("data/embedding_features.csv", index=False)
```

---

## Phase 3: Train Models

### Cross-validation analysis (replicates R notebook)

```bash
analysis --feature-type tfidf --mode cv --data-dir data/ --output-dir results/
```

### Train final model and save to ONNX

```bash
analysis --feature-type tfidf --mode train --data-dir data/ --output-dir results/
```

This saves:
- `results/tfidf_model.onnx` — ONNX model for inference
- `results/tfidf_coefficients.csv` — feature weights

Repeat for other feature types:

```bash
analysis --feature-type embeddings --mode train --data-dir data/ --output-dir results/
```

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

```bash
synthetic generate --config config/example_generator.yaml --output-dir synthetic_transcripts/
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

Run the synthetic transcripts through trained models:

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

Generate blog-quality comparison plots:

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

## Summary of CLI Commands

| Command | Purpose |
|---------|---------|
| `datasets-load --sources all --output data/unified.csv` | Load and unify public datasets |
| `analysis --feature-type tfidf --mode train ...` | Train predictive model |
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
