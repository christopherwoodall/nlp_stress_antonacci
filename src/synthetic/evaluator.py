"""
evaluator.py

Evaluate synthetic TESI transcripts by running them through the NLP
feature-extraction pipeline and trained ONNX models.

Steps:
    1. Load synthetic transcripts from directory
    2. Extract TF-IDF and/or embedding features (reuses nlp_stress modules)
    3. Load trained ONNX models from src/analysis/
    4. Predict severity and binary outcomes
    5. Output predictions CSV

Usage:
    synthetic evaluate --transcripts synthetic_transcripts/ --models-dir results/ --output results/evaluations.csv
"""

import json
import sys
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd


# --- Text loading ---

def load_transcripts(transcripts_dir: Union[str, Path]) -> pd.DataFrame:
    """
    Load all synthetic transcript .txt files from a directory tree.

    Expected structure:
        synthetic_transcripts/
            {model_name}/
                {question_id}_run{N}.txt

    Returns DataFrame with columns:
        model, question_id, run, text, file_path
    """
    transcripts_dir = Path(transcripts_dir)
    rows = []

    for model_dir in transcripts_dir.iterdir():
        if not model_dir.is_dir():
            continue
        model_name = model_dir.name

        for txt_file in model_dir.glob("*.txt"):
            # Parse filename: {question_id}_run{N}.txt
            stem = txt_file.stem  # e.g. "tesi_1_1_run1"
            parts = stem.rsplit("_run", 1)
            if len(parts) == 2:
                question_id = parts[0]
                run = int(parts[1])
            else:
                question_id = stem
                run = 1

            text = txt_file.read_text(encoding="utf-8").strip()
            if not text:
                continue

            rows.append({
                "model": model_name,
                "question_id": question_id,
                "run": run,
                "text": text,
                "file_path": str(txt_file),
            })

    if not rows:
        raise ValueError(f"No transcript .txt files found in {transcripts_dir}")

    df = pd.DataFrame(rows)
    print(f"Loaded {len(df)} transcripts from {len(df['model'].unique())} model(s)")
    return df


# --- Feature extraction ---

def extract_tfidf_features(texts: list, vectorizer_path: Optional[Path] = None) -> np.ndarray:
    """
    Extract TF-IDF features from a list of texts.

    If vectorizer_path is provided, loads a saved sklearn TfidfVectorizer.
    Otherwise, creates a new one (not recommended for evaluation — models
    were trained on a specific vocabulary).
    """
    from sklearn.feature_extraction.text import TfidfVectorizer

    if vectorizer_path and Path(vectorizer_path).exists():
        import joblib
        vectorizer = joblib.load(vectorizer_path)
        return vectorizer.transform(texts).toarray()

    # Fallback: create a new vectorizer (warns user)
    print("WARNING: No saved TF-IDF vectorizer found. Creating a new one. "
          "Predictions may be unreliable if vocabulary differs from training.",
          file=sys.stderr)
    vectorizer = TfidfVectorizer(max_features=5000)
    return vectorizer.fit_transform(texts).toarray()


def extract_embedding_features(texts: list) -> np.ndarray:
    """
    Extract RoBERTa sentence embeddings from a list of texts.

    Uses the sentence-transformers library with 'all-roberta-large-v1'.
    Falls back to mean-pooling if sentence-transformers is not available.
    """
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer("all-roberta-large-v1")
        return model.encode(texts, show_progress_bar=True)
    except ImportError:
        print("WARNING: sentence-transformers not installed. "
              "Install with: uv pip install sentence-transformers", file=sys.stderr)
        raise


# --- Model loading and prediction ---

def load_onnx_model(model_path: Union[str, Path]):
    """Load an ONNX model via onnxruntime."""
    try:
        import onnxruntime as ort
    except ImportError:
        raise ImportError(
            "onnxruntime is required for ONNX inference. "
            "Install with: uv pip install onnxruntime"
        )

    session = ort.InferenceSession(str(model_path))
    return session


def predict_with_onnx(session, X: np.ndarray) -> np.ndarray:
    """Run inference with an ONNX runtime session."""
    input_name = session.get_inputs()[0].name
    return session.run(None, {input_name: X.astype(np.float32)})[0].ravel()


# --- Evaluation pipeline ---

def evaluate_transcripts(
    transcripts_dir: Union[str, Path],
    models_dir: Union[str, Path],
    output_path: Union[str, Path],
    feature_type: str = "embeddings",
    vectorizer_path: Optional[Union[str, Path]] = None,
) -> pd.DataFrame:
    """
    Run the full evaluation pipeline on synthetic transcripts.

    Parameters
    ----------
    transcripts_dir : str or Path
        Directory containing synthetic transcripts.
    models_dir : str or Path
        Directory containing trained ONNX models (e.g. results/).
    output_path : str or Path
        Where to save the predictions CSV.
    feature_type : str
        Which features to extract: "tfidf" or "embeddings".
    vectorizer_path : str or Path or None
        Path to saved TF-IDF vectorizer (required if feature_type="tfidf").

    Returns
    -------
    pd.DataFrame with predictions
    """
    transcripts_dir = Path(transcripts_dir)
    models_dir = Path(models_dir)
    output_path = Path(output_path)

    # 1. Load transcripts
    df = load_transcripts(transcripts_dir)
    texts = df["text"].tolist()

    # 2. Extract features
    print(f"Extracting {feature_type} features...")
    if feature_type == "tfidf":
        X = extract_tfidf_features(texts, vectorizer_path=vectorizer_path)
    elif feature_type == "embeddings":
        X = extract_embedding_features(texts)
    else:
        raise ValueError(f"Unsupported feature_type: {feature_type}")

    print(f"Feature matrix shape: {X.shape}")

    # 3. Load ONNX models
    # Look for models in models_dir: *_model.onnx or similar
    model_files = list(models_dir.glob("*_model.onnx"))
    if not model_files:
        # Try broader search
        model_files = list(models_dir.glob("*.onnx"))

    if not model_files:
        raise FileNotFoundError(f"No ONNX models found in {models_dir}")

    print(f"Found {len(model_files)} ONNX model(s): {[m.name for m in model_files]}")

    # 4. Predict with each model
    for model_file in model_files:
        model_name = model_file.stem  # e.g. "liwc_model" or "embeddings_model"
        print(f"  Predicting with {model_name} ...")
        session = load_onnx_model(model_file)
        preds = predict_with_onnx(session, X)
        df[f"pred_{model_name}"] = preds

    # 5. Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"\nPredictions saved to: {output_path}")

    return df


# --- CLI entry ---

def main_evaluate(args=None):
    """CLI entry point for evaluate subcommand."""
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate synthetic transcripts")
    parser.add_argument("--transcripts", required=True, help="Path to synthetic transcripts directory")
    parser.add_argument("--models-dir", required=True, help="Path to trained ONNX models directory")
    parser.add_argument("--output", required=True, help="Output CSV path")
    parser.add_argument("--feature-type", default="embeddings", choices=["tfidf", "embeddings"])
    parser.add_argument("--vectorizer-path", default=None, help="Path to saved TF-IDF vectorizer")
    parsed = parser.parse_args(args)

    evaluate_transcripts(
        transcripts_dir=parsed.transcripts,
        models_dir=parsed.models_dir,
        output_path=parsed.output,
        feature_type=parsed.feature_type,
        vectorizer_path=parsed.vectorizer_path,
    )


if __name__ == "__main__":
    main_evaluate()
