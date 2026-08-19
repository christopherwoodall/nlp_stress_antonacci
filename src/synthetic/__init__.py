"""
synthetic package

Tools for generating synthetic TESI interview responses via LLM APIs,
evaluating them with trained models, and visualizing comparisons.

Modules:
    generator.py  — OpenRouter API client for TESI question responses
    evaluator.py  — Feature extraction + ONNX prediction on synthetic transcripts
    visualizer.py — Matplotlib/seaborn comparison plots
    cli.py        — Unified CLI: generate / evaluate / visualize

Usage:
    synthetic generate --config config/example_generator.yaml --output-dir synthetic_transcripts/
    synthetic evaluate --transcripts synthetic_transcripts/ --models-dir results/ --output results/evaluations.csv
    synthetic visualize --evaluations results/evaluations.csv --output-dir results/synthetic/
"""
