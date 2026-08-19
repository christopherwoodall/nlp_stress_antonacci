"""
loader.py

Download and load public substitute datasets for depression/distress text classification.

Each loader returns a standardized DataFrame with columns:
    text              — the raw text (dialogue, tweet, post)
    source            — which dataset it came from
    severity          — 0–3 unified severity scale
    binary            — 0/1 depression/distress flag
    original_label    — the dataset's native label

Supported datasets:
    mhdialog           — HuggingFace IkeZhang/MHDialog (risk levels → severity/binary)
    zenodo_depression  — Zenodo 14233292 (depression types → severity/binary)
    dreaddit           — Kaggle dreaddit stress dataset (stress/no-stress)
    joangaes_depression— HuggingFace joangaes/depression (binary)
"""

import hashlib
import io
import os
from pathlib import Path
from typing import Optional

import pandas as pd


# --- Label mapping tables ---

MHDIALOG_SEVERITY_MAP = {
    "No": 0,
    "Minor": 1,
    "Moderate": 2,
    "Severe": 3,
    "Not Related": 0,
    "Unsure": None,
}

MHDIALOG_BINARY_MAP = {
    "No": 0,
    "Minor": 1,
    "Moderate": 1,
    "Severe": 1,
    "Not Related": 0,
    "Unsure": None,
}

ZENODO_SEVERITY_MAP = {
    "bipolar": 2,
    "major": 2,
    "psychotic": 3,
    "atypical": 2,
    "postpartum": 2,
}

ZENODO_BINARY_MAP = {
    "bipolar": 1,
    "major": 1,
    "psychotic": 1,
    "atypical": 1,
    "postpartum": 1,
}

DREADDIT_SEVERITY_MAP = {
    "no_stress": 0,
    "stress": 1,
}

DREADDIT_BINARY_MAP = {
    "no_stress": 0,
    "stress": 1,
}

JOANGAES_SEVERITY_MAP = {
    0: 0,
    1: 2,
}

JOANGAES_BINARY_MAP = {
    0: 0,
    1: 1,
}


# --- Helpers ---

def _text_hash(text: str) -> str:
    """SHA-256 hash of normalized text for deduplication."""
    normalized = " ".join(text.lower().split())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:16]


def _standardize_df(df: pd.DataFrame, source: str) -> pd.DataFrame:
    """Add source column and text hash, ensure required columns exist."""
    df = df.copy()
    # Drop rows with missing text before hashing
    df = df.dropna(subset=["text"])
    df["source"] = source
    df["text_hash"] = df["text"].apply(_text_hash)
    required = ["text", "source", "severity", "binary", "original_label"]
    for col in required:
        if col not in df.columns:
            df[col] = None
    return df[required + ["text_hash"]]


# --- Individual dataset loaders ---

def load_mhdialog(split: str = "train", cache_dir: Optional[str] = None) -> pd.DataFrame:
    """
    Load MHDialog from HuggingFace.

    The dataset has 1,000 synthetic mental-health dialogues with:
        - Dialog Intent (8 categories)
        - Concern Type (7 categories)
        - Level (6 risk levels: No, Minor, Moderate, Severe, Not Related, Unsure)

    We use the 'Level' column for severity/binary mapping.
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError(
            "The 'datasets' library is required for MHDialog. "
            "Install it with: uv pip install datasets"
        )

    ds = load_dataset("IkeZhang/MHDialog", split=split, cache_dir=cache_dir)

    rows = []
    for example in ds:
        level = example.get("Level", "Unsure")
        severity = MHDIALOG_SEVERITY_MAP.get(level, None)
        binary = MHDIALOG_BINARY_MAP.get(level, None)
        if severity is None:
            continue  # skip "Unsure" labels

        # Extract user utterances from the JSON-formatted dialogue
        dialogue = example.get("Dialogue", "")
        text = _extract_user_text(dialogue)

        rows.append({
            "text": text,
            "severity": severity,
            "binary": binary,
            "original_label": level,
        })

    df = pd.DataFrame(rows)
    return _standardize_df(df, "mhdialog")


def _extract_user_text(dialogue: str) -> str:
    """
    Extract user utterances from MHDialog's JSON-formatted dialogue string.

    MHDialog stores dialogues as a JSON array of turn objects:
        [{"round": 1, "user": "...", "supporter": "..."}, ...]

    We concatenate all "user" turns to get the distressed person's speech.
    """
    import json

    text = dialogue.strip()
    if not text:
        return ""

    # Some rows are just a single string (e.g. "I'm sorry, I can't assist...")
    if text.startswith("["):
        try:
            turns = json.loads(text)
            user_utterances = [t.get("user", "") for t in turns if "user" in t]
            return " ".join(user_utterances)
        except json.JSONDecodeError:
            pass

    return text


def load_zenodo_depression(csv_path: Optional[str] = None) -> pd.DataFrame:
    """
    Load the Zenodo Multi-Class Depression Detection Dataset.

    If csv_path is provided, load from local file. Otherwise download from Zenodo.
    """
    if csv_path and Path(csv_path).exists():
        df_raw = pd.read_csv(csv_path)
    else:
        import requests

        url = "https://zenodo.org/records/14233292/files/dataset.csv?download=1"
        print(f"Downloading Zenodo dataset from {url} ...")
        response = requests.get(url, timeout=60)
        response.raise_for_status()
        df_raw = pd.read_csv(io.StringIO(response.text))

    # The CSV has columns: Tweets, Labels
    df_raw = df_raw.rename(columns={"Tweets": "text", "Labels": "original_label"})
    df_raw["severity"] = df_raw["original_label"].map(ZENODO_SEVERITY_MAP)
    df_raw["binary"] = df_raw["original_label"].map(ZENODO_BINARY_MAP)

    # Drop rows with unmapped labels or missing text
    df_raw = df_raw.dropna(subset=["severity", "binary", "text"])
    df_raw["severity"] = df_raw["severity"].astype(int)
    df_raw["binary"] = df_raw["binary"].astype(int)

    return _standardize_df(df_raw, "zenodo_depression")


def load_dreaddit(csv_path: Optional[str] = None) -> pd.DataFrame:
    """
    Load the Dreaddit stress dataset.

    Expected CSV columns: text, label (no_stress / stress)
    If csv_path is not provided, the user must download it from Kaggle first.
    """
    if not csv_path:
        raise FileNotFoundError(
            "Dreaddit must be downloaded from Kaggle manually. "
            "Please download it and pass --dreaddit-path <path>."
        )

    df_raw = pd.read_csv(csv_path)
    # Normalize column names
    df_raw = df_raw.rename(columns=str.lower)

    # Determine label column name
    label_col = None
    for candidate in ["label", "labels", "stress", "class"]:
        if candidate in df_raw.columns:
            label_col = candidate
            break

    if label_col is None:
        raise ValueError(
            f"Could not find label column in Dreaddit CSV. Columns: {list(df_raw.columns)}"
        )

    df_raw = df_raw.rename(columns={label_col: "original_label"})
    df_raw["severity"] = df_raw["original_label"].map(DREADDIT_SEVERITY_MAP)
    df_raw["binary"] = df_raw["original_label"].map(DREADDIT_BINARY_MAP)

    df_raw = df_raw.dropna(subset=["severity", "binary"])
    df_raw["severity"] = df_raw["severity"].astype(int)
    df_raw["binary"] = df_raw["binary"].astype(int)

    return _standardize_df(df_raw, "dreaddit")


def load_joangaes_depression(split: str = "train", cache_dir: Optional[str] = None) -> pd.DataFrame:
    """
    Load joangaes/depression from HuggingFace.

    This dataset has 28K Reddit posts with binary labels:
        0 = not depressed
        1 = depressed
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError(
            "The 'datasets' library is required. "
            "Install it with: uv pip install datasets"
        )

    ds = load_dataset("joangaes/depression", split=split, cache_dir=cache_dir)

    rows = []
    for example in ds:
        label = example.get("label", None)
        if label is None:
            continue

        severity = JOANGAES_SEVERITY_MAP.get(label, None)
        binary = JOANGAES_BINARY_MAP.get(label, None)
        if severity is None:
            continue

        rows.append({
            "text": example.get("text", ""),
            "severity": severity,
            "binary": binary,
            "original_label": label,
        })

    df = pd.DataFrame(rows)
    return _standardize_df(df, "joangaes_depression")


# --- Unified loader ---

def load_all(
    sources: Optional[list] = None,
    mhdialog_split: str = "train",
    zenodo_csv: Optional[str] = None,
    dreaddit_csv: Optional[str] = None,
    joangaes_split: str = "train",
    cache_dir: Optional[str] = None,
    deduplicate: bool = True,
) -> pd.DataFrame:
    """
    Load and combine multiple public datasets into a unified DataFrame.

    Parameters
    ----------
    sources : list of str or None
        Which datasets to load. If None, loads all available sources.
        Options: "mhdialog", "zenodo_depression", "dreaddit", "joangaes_depression"
    mhdialog_split : str
        HuggingFace split for MHDialog (default "train").
    zenodo_csv : str or None
        Local path to Zenodo CSV. If None, downloads from Zenodo.
    dreaddit_csv : str or None
        Local path to Dreaddit CSV. Required if "dreaddit" in sources.
    joangaes_split : str
        HuggingFace split for joangaes/depression (default "train").
    cache_dir : str or None
        Cache directory for HuggingFace datasets.
    deduplicate : bool
        If True, drop duplicate texts by SHA-256 hash (keeps first occurrence).

    Returns
    -------
    pd.DataFrame with columns: text, source, severity, binary, original_label, text_hash
    """
    if sources is None:
        sources = ["mhdialog", "zenodo_depression", "joangaes_depression"]

    dfs = []

    if "mhdialog" in sources:
        print("Loading MHDialog...")
        dfs.append(load_mhdialog(split=mhdialog_split, cache_dir=cache_dir))

    if "zenodo_depression" in sources:
        print("Loading Zenodo Multi-Class Depression...")
        dfs.append(load_zenodo_depression(csv_path=zenodo_csv))

    if "dreaddit" in sources:
        print("Loading Dreaddit...")
        dfs.append(load_dreaddit(csv_path=dreaddit_csv))

    if "joangaes_depression" in sources:
        print("Loading joangaes/depression...")
        dfs.append(load_joangaes_depression(split=joangaes_split, cache_dir=cache_dir))

    if not dfs:
        raise ValueError("No datasets were loaded. Check your --sources argument.")

    combined = pd.concat(dfs, ignore_index=True)

    if deduplicate:
        before = len(combined)
        combined = combined.drop_duplicates(subset=["text_hash"], keep="first")
        after = len(combined)
        print(f"Deduplicated: {before} → {after} rows ({before - after} removed)")

    return combined
