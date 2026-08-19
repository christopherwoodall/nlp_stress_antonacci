"""
mock_generator.py

Create fake synthetic TESI interview responses for pipeline testing.

Supports three personas:
    neutral    — generic mixed responses (default)
    depressed  — responses with hopelessness, nightmares, isolation
    resilient  — responses with coping, support, growth

This script generates deterministic fake transcripts in the same directory
structure as the real OpenRouter generator, but without any API calls.
Useful for testing the evaluation and visualization pipeline without
incurring API costs or needing an OPENROUTER_API_KEY.

Usage:
    python scripts/mock_generator.py --persona depressed --output-dir synthetic_depressed/
    python scripts/mock_generator.py --persona resilient --output-dir synthetic_resilient/
"""

import argparse
import json
import random
from pathlib import Path

import yaml


# --- Persona-specific response templates ---
# Templates are crafted to align with the trained TF-IDF model's learned coefficients.
# Depressed templates emphasize words with positive coefficients (high predicted severity).
# Resilient templates emphasize words with negative coefficients (low predicted severity).

DEPRESSED_TEMPLATES = [
    "I have depression and there is no end to the pain. I feel tired of living and have "
    "thoughts of killing myself. My life feels like a disorder that can never be fixed. "
    "I want to die and there is no one there to care. I am alone with my pain and the "
    "redflag of my own mind tells me it will never get better.",

    "Every night I have the same nightmares of death. I wake up shaking and can't go "
    "back to sleep. I lie there thinking about how worthless I am and how my life has "
    "no meaning anymore. The depression is killing me and I feel there is nothing I can "
    "do to stop the pain. I wish I could end it all.",

    "I stopped talking to everyone because I feel like a burden. My friends tried to "
    "reach out but I can't respond. I have thoughts that everyone would be better off "
    "without me. The depression makes me feel dead inside and there is no hope of getting "
    "better. I am alone with my pain and I want it to end.",

    "I can't concentrate on anything because my mind is full of dark thoughts. School "
    "feels pointless and I keep thinking about hurting myself. Sometimes I wonder if "
    "anyone would even notice if I was gone. The depression has taken everything from me "
    "and there is no end to the pain I feel.",

    "The emptiness never goes away no matter what I do. I fake smiles so people don't "
    "ask questions but inside I feel dead. I don't remember what happiness feels like and "
    "I have no hope of ever feeling it again. My life is a constant pain that I can't "
    "escape. I wish I could just end it all and make the pain stop.",

    "I hate myself so much that it hurts. Everything I do feels wrong and I deserve to "
    "feel this way. I have thoughts of ending my life every day and the depression makes "
    "me feel like there is no point in going on. I am alone with my pain and no one "
    "can help me. I want to die.",
]

RESILIENT_TEMPLATES = [
    "It was scary at first but I got through it with the help of my family and friends. "
    "We talked about what happened and I learned that asking for help is a good thing. "
    "My mom was really supportive and made sure I had people to talk to. I feel like "
    "I came out of it stronger and more confident than before.",

    "I went to therapy for a while and it really helped me process what happened. Now I "
    "have good coping strategies like journaling and exercise that keep me grounded. "
    "My school counselor also gave me resources that made a big difference. I would "
    "say that getting help was the best thing I ever did for myself.",

    "My parents were amazing through the whole thing. They never judged me and just "
    "listened when I needed to talk. Having people who believe in you makes all the "
    "difference. I also made new friends at school who helped me see that things can "
    "get better. I feel lucky to have such good people in my life.",

    "Honestly it was tough but I got through it one day at a time. I joined a club at "
    "school and met people who understood what I was going through. Now I try to help "
    "others going through similar things. I would tell anyone to just keep going and "
    "to reach out for help because it really does get better.",

    "I focused on the things I could control like my grades and my hobbies. Taking small "
    "steps every day helped me rebuild my confidence. I also started eating well and "
    "sleeping better which made a huge difference. I feel like a new person now and I "
    "am really proud of how far I have come.",

    "I talked to my school counselor and she connected me with resources that changed "
    "my life. It turns out a lot of people go through hard times and come out stronger. "
    "I would say that the support I got from my teachers and friends was the best part. "
    "Now I use what I learned to help others and that makes me feel really good about "
    "myself.",
]

NEUTRAL_TEMPLATES = [
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

PERSONA_MAP = {
    "depressed": DEPRESSED_TEMPLATES,
    "resilient": RESILIENT_TEMPLATES,
    "neutral": NEUTRAL_TEMPLATES,
}


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
        "--persona",
        type=str,
        default="neutral",
        choices=["neutral", "depressed", "resilient"],
        help="Persona for generated responses",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic generation",
    )
    return parser.parse_args()


def generate_mock_responses(
    config_path: str, output_dir: Path, persona: str = "neutral", seed: int = 42
) -> Path:
    """
    Generate fake TESI responses matching the real generator's output structure.

    Parameters
    ----------
    config_path : str
        Path to generator YAML config.
    output_dir : Path
        Directory to write fake transcripts.
    persona : str
        "neutral", "depressed", or "resilient" — determines response tone.
    seed : int
        Random seed for reproducibility.
    """
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    models = config.get("models", {})
    questions = config.get("questions", [])
    settings = config.get("settings", {})
    responses_per_question = settings.get("responses_per_question", 5)

    templates = PERSONA_MAP.get(persona, NEUTRAL_TEMPLATES)
    random.seed(seed)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata = {
        "config_path": str(config_path),
        "output_dir": str(output_dir),
        "mock": True,
        "persona": persona,
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
            "version": model_cfg.get("version", "unknown"),
        }

        for q in questions:
            q_id = q["id"]

            for run in range(1, responses_per_question + 1):
                file_name = f"{q_id}_run{run}.txt"
                file_path = model_out / file_name

                # Pick a deterministic template based on question + run
                template_idx = (hash(q_id) + run + hash(model_name)) % len(templates)
                base_text = templates[template_idx]

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
    print(f"  Persona: {persona}")
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
        persona=args.persona,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
