"""
nlp_stress package

NLP pipeline for processing TESI stress interviews.

Modules:
    whisperx_process.py    — WhisperX transcription and diarization
    speaker_parsing.py     — Extract participant-only speech
    tfidf_generation.py    — Compute and filter TF-IDF features
    generate_embeddings.py — Generate RoBERTa participant & sentence embeddings
    lda_preprocess.py      — Tokenize transcripts for DLATK/MALLET
    lda_aggregate.py       — Aggregate LDA topics to participant-level
    pull_dataset.py        — Pull or generate the dataset for the pipeline

Usage:
    whisperx-process --input audio.wav --output transcript.txt
    speaker-parse --input transcript.txt --output participant.txt
    tfidf-generate --input participant.txt --output features.csv
"""

__version__ = "0.1.0"
