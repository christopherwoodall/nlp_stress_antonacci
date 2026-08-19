"""
external_datasets package

Public dataset loaders for substitute depression/trauma text datasets.

Downloads and unifies labeled datasets from HuggingFace, Zenodo, and Kaggle,
with a consistent label mapping to severity (0–3) and binary (0/1) targets.

Modules:
    loader.py — MHDialog, Zenodo, Dreaddit, joangaes/depression loaders
    cli.py    — datasets-load entry point

Usage:
    datasets-load --sources all --output data/unified_dataset.csv
"""
