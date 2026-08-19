"""
cli.py

Unified CLI entry point for all analysis modules.

Usage:
    analysis --feature-type liwc --mode cv --data-dir data/ --output-dir results/
    analysis --feature-type tfidf --mode train --data-dir data/ --output-dir results/
    analysis --feature-type lda --mode test --model-path results/lda_model.onnx --test-data data/test.csv --output results/test_metrics.json
    analysis --feature-type embeddings --mode predict --model-path results/emb_model.onnx --input data/new.csv --output predictions.csv
"""

import argparse
import sys
from pathlib import Path


# --- Dispatch table ---

MODULES = {}


def _import_modules():
    """Lazy import to avoid heavy imports at CLI startup."""
    global MODULES
    if MODULES:
        return
    from . import liwc, tfidf, lda, embeddings
    MODULES = {
        "liwc": liwc,
        "tfidf": tfidf,
        "lda": lda,
        "embeddings": embeddings,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Run analysis, training, testing, or prediction for NLP Stress models."
    )
    parser.add_argument(
        "--feature-type",
        required=True,
        choices=["liwc", "tfidf", "lda", "embeddings"],
        help="Which feature set to use",
    )
    parser.add_argument(
        "--mode",
        required=True,
        choices=["cv", "train", "test", "predict"],
        help="cv = cross-validation analysis; train = fit final model; test = evaluate; predict = infer",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data",
        help="Directory containing feature and outcome files",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results",
        help="Directory to write outputs",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        help="Path to saved ONNX model (required for test/predict)",
    )
    parser.add_argument(
        "--test-data",
        type=str,
        help="Path to test data CSV (required for test mode)",
    )
    parser.add_argument(
        "--input",
        type=str,
        help="Path to input CSV for prediction (required for predict mode)",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Path to write predictions (required for predict mode)",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=123,
        help="Random seed for reproducibility",
    )
    args = parser.parse_args()

    _import_modules()
    module = MODULES.get(args.feature_type)
    if module is None:
        print(f"Unknown feature type: {args.feature_type}", file=sys.stderr)
        sys.exit(1)

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.mode == "cv":
        module.run_cv_analysis(data_dir, output_dir, random_state=args.random_state)
    elif args.mode == "train":
        module.train_final_model(data_dir, output_dir, random_state=args.random_state)
    elif args.mode == "test":
        if not args.model_path or not args.test_data:
            parser.error("--mode test requires --model-path and --test-data")
        module.test_model(Path(args.model_path), Path(args.test_data), output_dir)
    elif args.mode == "predict":
        if not args.model_path or not args.input or not args.output:
            parser.error("--mode predict requires --model-path, --input, and --output")
        module.predict(Path(args.model_path), Path(args.input), Path(args.output))
    else:
        parser.error(f"Unknown mode: {args.mode}")


if __name__ == "__main__":
    main()
