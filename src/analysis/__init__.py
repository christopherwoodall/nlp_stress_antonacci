"""
analysis package

Python translation of the R Markdown analysis notebooks for the NLP Stress pipeline.

Modules:
    liwc.py      — LIWC feature analysis (translated from LIWC_Analyses.Rmd)
    tfidf.py     — TF-IDF feature analysis (translated from TF-IDF_Analyses.Rmd)
    lda.py       — LDA topic analysis (translated from LDA_Analyses.Rmd)
    embeddings.py — Sentence embedding analysis (translated from SentenceEmbedding_Analysis.Rmd)
    _base.py     — Shared utilities: nested CV, metrics, ONNX I/O
    _plots.py    — Matplotlib/seaborn replications of key R figures
    cli.py       — Unified CLI entry point for all feature types

Usage:
    analysis --feature-type tfidf --mode cv --data-dir data/ --output-dir results/
    analysis --feature-type embeddings --mode train --data-dir data/ --output-dir results/
"""

__version__ = "0.1.0"
