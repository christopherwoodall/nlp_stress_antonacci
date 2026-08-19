"""
cli.py

Entry point: datasets-load

Usage:
    datasets-load --sources mhdialog,zenodo_depression --output data/unified_dataset.csv
    datasets-load --sources all --output data/unified.csv --cache-dir ~/.cache/hf
"""

import argparse
import sys
from pathlib import Path

from .loader import load_all


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Load and unify public substitute datasets for depression/distress NLP."
    )
    parser.add_argument(
        "--sources",
        type=str,
        default="all",
        help="Comma-separated list of sources: mhdialog, zenodo_depression, dreaddit, joangaes_depression, or 'all'",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output CSV path",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="Cache directory for HuggingFace datasets",
    )
    parser.add_argument(
        "--zenodo-csv",
        type=str,
        default=None,
        help="Local path to Zenodo dataset.csv (optional, will download if not provided)",
    )
    parser.add_argument(
        "--dreaddit-csv",
        type=str,
        default=None,
        help="Local path to Dreaddit CSV (required if dreaddit in sources)",
    )
    parser.add_argument(
        "--no-deduplicate",
        action="store_true",
        help="Disable text deduplication",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Parse sources
    if args.sources.lower() == "all":
        sources = None  # load_all defaults
    else:
        sources = [s.strip() for s in args.sources.split(",")]

    # Validate dreaddit path if needed
    if sources and "dreaddit" in sources and not args.dreaddit_csv:
        print("ERROR: --dreaddit-csv is required when 'dreaddit' is in --sources", file=sys.stderr)
        sys.exit(1)

    print(f"Loading sources: {sources or 'all default'}")

    df = load_all(
        sources=sources,
        zenodo_csv=args.zenodo_csv,
        dreaddit_csv=args.dreaddit_csv,
        cache_dir=args.cache_dir,
        deduplicate=not args.no_deduplicate,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    print(f"\nSaved unified dataset to: {output_path}")
    print(f"  Rows: {len(df)}")
    print(f"  Sources: {df['source'].value_counts().to_dict()}")
    print(f"  Severity distribution: {df['severity'].value_counts().sort_index().to_dict()}")
    print(f"  Binary distribution: {df['binary'].value_counts().sort_index().to_dict()}")


if __name__ == "__main__":
    main()
