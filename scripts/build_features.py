"""
build_features.py

Extract TF-IDF and RoBERTa embedding features from the unified dataset
and save them in the format expected by the analysis CLI.

Input:
    data/unified_dataset.csv (from datasets-load)

Outputs:
    data/tfidf_features.csv      — TF-IDF document-term matrix
    data/embedding_features.csv  — RoBERTa pooled embeddings
    data/outcomes.csv            — severity + binary labels
    results/tfidf_vectorizer.joblib — saved vectorizer for eval

Usage:
    python scripts/build_features.py --input data/unified_dataset.csv --output-dir data/
"""

import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


# --- Constants ---

TOKEN_PATTERN = r"(?u)\b[a-zA-Z][a-zA-Z]+\b"
MIN_DOC_FREQUENCY = 0.05
MAX_FEATURES = 5000
EMBEDDING_MODEL = "all-roberta-large-v1"
EMBEDDING_DIM = 768


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract TF-IDF and embedding features from unified dataset."
    )
    parser.add_argument(
        "--input",
        type=str,
        default="data/unified_dataset.csv",
        help="Path to unified dataset CSV",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data",
        help="Directory to write feature files",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="results",
        help="Directory to save vectorizer and other artifacts",
    )
    return parser.parse_args()


def extract_tfidf(df: pd.DataFrame, output_dir: Path, results_dir: Path) -> None:
    """Extract TF-IDF features and save to CSV."""
    texts = df["text"].tolist()

    print("Extracting TF-IDF features...")
    vectorizer = TfidfVectorizer(
        token_pattern=TOKEN_PATTERN,
        min_df=MIN_DOC_FREQUENCY,
        max_features=MAX_FEATURES,
    )
    X = vectorizer.fit_transform(texts)
    feature_names = vectorizer.get_feature_names_out()

    feature_df = pd.DataFrame(X.toarray(), columns=feature_names)
    feature_df["ELS_ID"] = range(len(feature_df))

    out_path = output_dir / "tfidf_features.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    feature_df.to_csv(out_path, index=False)
    print(f"  Saved: {out_path}  shape={feature_df.shape}")

    # Save vectorizer for later evaluation
    results_dir.mkdir(parents=True, exist_ok=True)
    vectorizer_path = results_dir / "tfidf_vectorizer.joblib"
    joblib.dump(vectorizer, vectorizer_path)
    print(f"  Saved vectorizer: {vectorizer_path}")


def extract_embeddings(df: pd.DataFrame, output_dir: Path) -> None:
    """Extract RoBERTa sentence embeddings and save to CSV."""
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        print(
            "ERROR: sentence-transformers is required for embeddings. "
            "Install with: uv pip install sentence-transformers",
            file=sys.stderr,
        )
        sys.exit(1)

    texts = df["text"].tolist()

    print(f"Extracting embeddings with {EMBEDDING_MODEL}...")
    model = SentenceTransformer(EMBEDDING_MODEL)
    embeddings = model.encode(texts, show_progress_bar=True)

    feature_df = pd.DataFrame(embeddings)
    feature_df["ELS_ID"] = range(len(feature_df))

    out_path = output_dir / "embedding_features.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    feature_df.to_csv(out_path, index=False)
    print(f"  Saved: {out_path}  shape={feature_df.shape}")


def build_outcomes(df: pd.DataFrame, output_dir: Path) -> None:
    """Save outcome labels in analysis-ready format."""
    outcome_df = df[["severity", "binary"]].copy()
    outcome_df["ELS_ID"] = range(len(outcome_df))

    out_path = output_dir / "outcomes.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    outcome_df.to_csv(out_path, index=False)
    print(f"  Saved outcomes: {out_path}  rows={len(outcome_df)}")


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    results_dir = Path(args.results_dir)

    if not input_path.exists():
        print(f"ERROR: Input file not found: {input_path}", file=sys.stderr)
        print("Run: datasets-load --sources all --output data/unified_dataset.csv", file=sys.stderr)
        sys.exit(1)

    print(f"Loading unified dataset: {input_path}")
    df = pd.read_csv(input_path)
    print(f"  Loaded {len(df)} rows")

    # Drop rows with missing text
    df = df.dropna(subset=["text"])
    print(f"  After dropping NaN text: {len(df)} rows")

    print("\n--- Building outcomes ---")
    build_outcomes(df, output_dir)

    print("\n--- Extracting TF-IDF ---")
    extract_tfidf(df, output_dir, results_dir)

    print("\n--- Extracting embeddings ---")
    extract_embeddings(df, output_dir)

    print("\nDone. Feature files ready in:", output_dir)


if __name__ == "__main__":
    main()
