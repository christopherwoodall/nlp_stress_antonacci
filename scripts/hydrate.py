"""
hydrate.py

Hydrate the data directory with everything needed to run the pipeline.

Downloads:
    - TESI-C PDF from VA National Center for PTSD (if not present)
    - Public substitute datasets via datasets-load

Generates:
    - Sample synthetic participant transcripts for testing

Usage:
    python scripts/hydrate.py
"""

import os
import sys
from pathlib import Path

import requests

TESI_URL = "https://www.ptsd.va.gov/professional/assessment/documents/TESI-C.pdf"
DATA_DIR = Path("data")


def download_tesi_pdf() -> Path:
    """Download TESI-C PDF if not already present."""
    pdf_path = DATA_DIR / "TESI-C.pdf"
    if pdf_path.exists():
        print(f"[SKIP] TESI-C PDF already exists: {pdf_path}")
        return pdf_path

    print(f"[DOWNLOAD] TESI-C PDF from VA...")
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    response = requests.get(TESI_URL, timeout=30)
    response.raise_for_status()
    pdf_path.write_bytes(response.content)
    print(f"[OK] Saved: {pdf_path} ({pdf_path.stat().st_size} bytes)")
    return pdf_path


def generate_sample_data() -> None:
    """Generate sample transcripts using pull-dataset."""
    print("[GENERATE] Creating sample dataset...")
    # Use the existing pull-dataset entry point
    from nlp_stress.pull_dataset import generate_sample_dataset

    generate_sample_dataset(DATA_DIR, num_participants=5)
    print("[OK] Sample data generated.")


def load_public_datasets() -> None:
    """Load public substitute datasets."""
    print("[LOAD] Downloading public substitute datasets...")
    from external_datasets.cli import main as datasets_main

    # Temporarily override sys.argv to call the CLI
    old_argv = sys.argv
    sys.argv = [
        "datasets-load",
        "--sources", "all",
        "--output", str(DATA_DIR / "unified_dataset.csv"),
    ]
    try:
        datasets_main()
    except SystemExit as e:
        if e.code not in (0, None):
            raise
    finally:
        sys.argv = old_argv

    print("[OK] Public datasets loaded.")


def main() -> None:
    print("=" * 60)
    print("Hydrating data directory")
    print("=" * 60)

    download_tesi_pdf()
    generate_sample_data()
    load_public_datasets()

    print("\n" + "=" * 60)
    print("Hydration complete.")
    print("=" * 60)
    print(f"\nData directory: {DATA_DIR.resolve()}")
    print("  - TESI-C.pdf (interview protocol)")
    print("  - Sample transcripts (5 participants)")
    print("  - unified_dataset.csv (public substitute datasets)")


if __name__ == "__main__":
    main()
