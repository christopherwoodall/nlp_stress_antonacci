"""
pull_dataset.py

Flexible data acquisition tool for the NLP Stress pipeline.

Since the original ELS interview data is clinical youth data stored on
institutional Box storage, it is not publicly downloadable. This script
provides two modes:

  A. Pull from a configurable remote source (--source)
     Accepts a URL or local path to a ZIP/tarball/directory and extracts
     it into the expected data layout.

  B. Generate a synthetic sample dataset (--generate-sample)
     Creates a small set of fake transcripts in the correct directory
     layout so the pipeline can be run end-to-end for testing and to see
     what the data looks like.

The script also prints a data structure guide showing what directories
and files the pipeline expects.
"""

import argparse
import os
import random
import shutil
import sys
from pathlib import Path


# --- Data Structure Guide ---

EXPECTED_STRUCTURE = """
Expected data layout:

  data/
    WhisperX_Carina_Revised/   # diarized transcripts (output of WhisperX + GPT-4o)
                               #   Format per file:
                               #   start_time  end_time  Speaker: utterance
                               #   e.g. 0.42  3.15  Participant: I felt scared.
    Participant_Parse/         # participant-only speech (output of speaker-parse)
                               #   One .txt per participant, plain text paragraph.
    Transcripts/               # final cleaned transcripts (input to TF-IDF)
                               #   One .txt per participant, naming: ELS_XXX.txt
    participant_transcripts/   # alternative input for LDA preprocessing
                               #   Same content as Participant_Parse/ or Transcripts/

Note: The original data is clinical youth interview data and is not
publicly available. Contact the Stanford Neurodevelopment, Affect, and
Psychopathology Laboratory (PI: Ian Gotlib) for data access inquiries.
"""


# --- Sample Data Generator ---

SAMPLE_SENTENCES = [
    "I felt really scared when it happened.",
    "It was like everything just stopped making sense.",
    "I didn't know what to do, so I just froze.",
    "Sometimes I still think about it at night.",
    "My parents tried to help, but I couldn't explain it.",
    "I guess I just wanted to feel safe again.",
    "School was hard because I couldn't concentrate.",
    "I would get these headaches when I got stressed.",
    "It changed how I see people, I think.",
    "I don't really talk about it much anymore.",
    "I felt alone even when people were around.",
    "Sleeping was the hardest part for me.",
    "I would wake up and not remember where I was.",
    "Things got better slowly, but they did get better.",
    "I still have moments where it comes back.",
]


def generate_sample_dataset(output_dir: Path, num_participants: int = 5) -> None:
    """
    Create a synthetic sample dataset in the expected directory layout.

    Generates fake transcripts for `num_participants` participants with
    ELS-style IDs (e.g. ELS_001, ELS_002). Each transcript contains a
    handful of sentences randomly sampled from SAMPLE_SENTENCES.
    """
    # create all expected subdirectories
    whisperx_dir = output_dir / "WhisperX_Carina_Revised"
    participant_dir = output_dir / "Participant_Parse"
    transcripts_dir = output_dir / "Transcripts"
    participant_transcripts_dir = output_dir / "participant_transcripts"

    for d in (whisperx_dir, participant_dir, transcripts_dir, participant_transcripts_dir):
        d.mkdir(parents=True, exist_ok=True)

    for i in range(1, num_participants + 1):
        els_id = f"ELS_{i:03d}"

        # build a fake paragraph of 5-10 random sentences
        num_sentences = random.randint(5, 10)
        sentences = [random.choice(SAMPLE_SENTENCES) for _ in range(num_sentences)]
        paragraph = " ".join(sentences)

        # Participant_Parse: plain text paragraph
        pp_path = participant_dir / f"{els_id}_participant_parsing.txt"
        pp_path.write_text(paragraph + "\n", encoding="utf-8")

        # Transcripts: plain text with ELS_XXX.txt naming
        tx_path = transcripts_dir / f"{els_id}.txt"
        tx_path.write_text(paragraph + "\n", encoding="utf-8")

        # participant_transcripts: same content for LDA preprocessing
        pt_path = participant_transcripts_dir / f"{els_id}_participant_parsing.txt"
        pt_path.write_text(paragraph + "\n", encoding="utf-8")

        # WhisperX_Carina_Revised: diarized format with timestamps
        diarized_lines = []
        t = 0.0
        for sent in sentences:
            duration = round(random.uniform(2.0, 5.0), 2)
            speaker = random.choice(["Participant", "Interviewer"])
            diarized_lines.append(
                f"{t:.2f}\t{t + duration:.2f}\t{speaker}: {sent}"
            )
            t += duration + random.uniform(0.5, 1.5)
        wx_path = whisperx_dir / f"{els_id}_whisperx_transcript.txt"
        wx_path.write_text("\n".join(diarized_lines) + "\n", encoding="utf-8")

    print(f"Generated sample dataset with {num_participants} participants.")
    print(f"Output directory: {output_dir.resolve()}")
    print("\nGenerated directories:")
    for d in (whisperx_dir, participant_dir, transcripts_dir, participant_transcripts_dir):
        print(f"  {d}")


# --- Pull from Source ---

def pull_from_source(source: str, output_dir: Path) -> None:
    """
    Pull data from a configurable source (URL or local path).

    Supports:
      - Local directory path (copied/symlinked into output_dir)
      - Local ZIP file (extracted into output_dir)
      - HTTP(S) URL to a ZIP file (downloaded and extracted)
    """
    source_path = Path(source)

    if source_path.exists() and source_path.is_dir():
        print(f"Copying local directory: {source}")
        if output_dir.exists():
            shutil.rmtree(output_dir)
        shutil.copytree(source, output_dir)
        print(f"Copied to: {output_dir}")
        return

    if source_path.exists() and source_path.suffix.lower() == ".zip":
        print(f"Extracting local ZIP: {source}")
        shutil.unpack_archive(str(source_path), str(output_dir))
        print(f"Extracted to: {output_dir}")
        return

    # treat as URL
    if source.startswith("http://") or source.startswith("https://"):
        try:
            import requests
        except ImportError:
            print(
                "ERROR: 'requests' is required for URL downloads. "
                "Install it with: uv pip install requests",
                file=sys.stderr,
            )
            sys.exit(1)

        print(f"Downloading from URL: {source}")
        response = requests.get(source, stream=True)
        response.raise_for_status()

        archive_path = output_dir / "download.zip"
        output_dir.mkdir(parents=True, exist_ok=True)
        with open(archive_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        print(f"Downloaded to: {archive_path}")
        shutil.unpack_archive(str(archive_path), str(output_dir))
        archive_path.unlink()
        print(f"Extracted to: {output_dir}")
        return

    print(f"ERROR: Unsupported source: {source}", file=sys.stderr)
    print(
        "Expected a local directory, a local .zip file, or an HTTP(S) URL.",
        file=sys.stderr,
    )
    sys.exit(1)


# --- Validation ---

def validate_data_directory(data_dir: Path) -> bool:
    """
    Check whether the data directory contains the expected subdirectories
    and at least some .txt files.
    """
    expected = {
        "WhisperX_Carina_Revised": False,
        "Participant_Parse": False,
        "Transcripts": False,
        "participant_transcripts": False,
    }

    ok = True
    for subdir in expected:
        subdir_path = data_dir / subdir
        if subdir_path.exists() and subdir_path.is_dir():
            txt_files = list(subdir_path.glob("*.txt"))
            if txt_files:
                expected[subdir] = True
                print(f"  [OK] {subdir}/ — {len(txt_files)} .txt file(s)")
            else:
                print(f"  [WARN] {subdir}/ exists but contains no .txt files")
        else:
            print(f"  [MISSING] {subdir}/")
            ok = False

    return ok


# --- CLI ---

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pull or generate the dataset for the NLP Stress pipeline."
    )
    parser.add_argument(
        "--source",
        type=str,
        default=None,
        help="Source to pull data from: local dir, local .zip, or HTTP(S) URL",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data",
        help="Directory to write data into (default: data/)",
    )
    parser.add_argument(
        "--generate-sample",
        action="store_true",
        help="Generate a synthetic sample dataset instead of pulling from a source",
    )
    parser.add_argument(
        "--num-participants",
        type=int,
        default=5,
        help="Number of synthetic participants to generate (default: 5)",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Only validate the existing data directory, do not pull or generate",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)

    print("=" * 60)
    print("NLP Stress Pipeline — Dataset Pull")
    print("=" * 60)
    print(EXPECTED_STRUCTURE)

    if args.validate_only:
        print("Validating existing data directory...")
        ok = validate_data_directory(output_dir)
        sys.exit(0 if ok else 1)

    if args.generate_sample:
        generate_sample_dataset(output_dir, num_participants=args.num_participants)
    elif args.source:
        pull_from_source(args.source, output_dir)
    else:
        print("\nNo action specified. Use one of:")
        print("  --source <path_or_url>     Pull data from a source")
        print("  --generate-sample          Create a synthetic sample dataset")
        print("  --validate-only            Check existing data directory")
        sys.exit(1)

    print("\nValidating data directory...")
    validate_data_directory(output_dir)


if __name__ == "__main__":
    main()
