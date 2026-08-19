"""
mock_generator.py

Create fake synthetic TESI interview responses for pipeline testing.

This script generates deterministic fake transcripts in the same directory
structure as the real OpenRouter generator, but without any API calls.
Useful for testing the evaluation and visualization pipeline without
incurring API costs or needing an OPENROUTER_API_KEY.

Usage:
    python scripts/mock_generator.py --output-dir synthetic_transcripts/ --config config/example_generator.yaml
"""

import argparse
import json
import random
from pathlib import Path

import yaml


# --- Fake response templates ---

RESPONSE_TEMPLATES = [
    "I felt really scared when it happened. I didn't know what to do and I just froze. "
    "Sometimes I still think about it at night and it makes me anxious.",

    "Nothing like that ever happened to me. I've had a pretty normal life so far "
    "and my family has always been supportive.",

    "It was one of the worst days of my life. I remember every detail like it was yesterday. "
    "The sounds, the smells, everything. I still have nightmares about it sometimes.",

    "I guess it wasn't that bad compared to what some people go through. "
    "I was upset at the time but I'm mostly over it now.",

    "I don't really like talking about it. It happened a long time ago and I've tried to move on. "
    "Sometimes things trigger memories but I manage okay.",

    "Yeah, that happened to me. It was scary at first but I had people around me who helped. "
    "I think I'm stronger because of it now.",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate fake synthetic TESI transcripts for testing."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/example_generator.yaml",
        help="Path to generator YAML config",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="synthetic_transcripts",
        help="Directory to write fake transcripts",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic generation",
    )
    return parser.parse_args()


def generate_mock_responses(config_path: str, output_dir: Path, seed: int = 42) -> Path:
    """
    Generate fake TESI responses matching the real generator's output structure.
    """
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    models = config.get("models", {})
    questions = config.get("questions", [])
    settings = config.get("settings", {})
    responses_per_question = settings.get("responses_per_question", 5)

    random.seed(seed)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata = {
        "config_path": str(config_path),
        "output_dir": str(output_dir),
        "mock": True,
        "models": {},
        "responses": [],
    }

    for model_name, model_cfg in models.items():
        print(f"Mock generating for model: {model_name}")
        model_out = output_dir / model_name
        model_out.mkdir(parents=True, exist_ok=True)

        metadata["models"][model_name] = {
            "provider": model_cfg.get("provider", "mock"),
            "model_id": model_cfg.get("model", "mock"),
        }

        for q in questions:
            q_id = q["id"]

            for run in range(1, responses_per_question + 1):
                file_name = f"{q_id}_run{run}.txt"
                file_path = model_out / file_name

                # Pick a deterministic template based on question + run
                template_idx = (hash(q_id) + run + hash(model_name)) % len(RESPONSE_TEMPLATES)
                base_text = RESPONSE_TEMPLATES[template_idx]

                # Add slight variation so runs aren't identical
                variation = random.choice([
                    " That's all I want to say about it.",
                    " I don't want to go into more detail.",
                    " It still affects me sometimes.",
                    " I'm doing better now though.",
                    "",
                ])
                text = base_text + variation

                file_path.write_text(text, encoding="utf-8")

                metadata["responses"].append({
                    "model": model_name,
                    "question_id": q_id,
                    "question_text": q["text"],
                    "run": run,
                    "file": str(file_path.relative_to(output_dir)),
                })

    # Save metadata
    meta_path = output_dir / "metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nMock transcripts saved to: {output_dir}")
    print(f"  Models: {list(models.keys())}")
    print(f"  Questions: {len(questions)}")
    print(f"  Responses per question: {responses_per_question}")
    print(f"  Total files: {len(metadata['responses'])}")
    return output_dir


def main() -> None:
    args = parse_args()
    generate_mock_responses(
        config_path=args.config,
        output_dir=Path(args.output_dir),
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
