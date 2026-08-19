"""
cli.py

Unified CLI entry point for synthetic generation, evaluation, and visualization.

Usage:
    synthetic generate --config config/example_generator.yaml --output-dir synthetic_transcripts/
    synthetic evaluate --transcripts synthetic_transcripts/ --models-dir results/ --output results/evaluations.csv
    synthetic visualize --evaluations results/evaluations.csv --output-dir results/synthetic/
"""

import argparse
import sys
from pathlib import Path

from .generator import generate_responses
from .evaluator import evaluate_transcripts
from .visualizer import visualize


def main():
    parser = argparse.ArgumentParser(
        description="Synthetic TESI generation, evaluation, and visualization."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # --- generate ---
    gen_parser = subparsers.add_parser("generate", help="Generate synthetic TESI responses")
    gen_parser.add_argument("--config", required=True, help="Path to generator YAML config")
    gen_parser.add_argument("--output-dir", default=None, help="Override output directory")
    gen_parser.set_defaults(func=lambda args: generate_responses(
        config_path=args.config,
        output_dir=args.output_dir,
    ))

    # --- evaluate ---
    eval_parser = subparsers.add_parser("evaluate", help="Evaluate synthetic transcripts")
    eval_parser.add_argument("--transcripts", required=True, help="Path to synthetic transcripts directory")
    eval_parser.add_argument("--models-dir", required=True, help="Path to trained ONNX models directory")
    eval_parser.add_argument("--output", required=True, help="Output CSV path")
    eval_parser.add_argument("--feature-type", default="embeddings", choices=["tfidf", "embeddings"])
    eval_parser.add_argument("--vectorizer-path", default=None, help="Path to saved TF-IDF vectorizer")
    eval_parser.set_defaults(func=lambda args: evaluate_transcripts(
        transcripts_dir=args.transcripts,
        models_dir=args.models_dir,
        output_path=args.output,
        feature_type=args.feature_type,
        vectorizer_path=args.vectorizer_path,
    ))

    # --- visualize ---
    viz_parser = subparsers.add_parser("visualize", help="Visualize evaluation results")
    viz_parser.add_argument("--evaluations", required=True, help="Path to evaluations CSV")
    viz_parser.add_argument("--output-dir", required=True, help="Directory to save plots")
    viz_parser.set_defaults(func=lambda args: visualize(
        evaluations_csv=args.evaluations,
        output_dir=args.output_dir,
    ))

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
