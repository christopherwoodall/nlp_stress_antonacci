"""
generator.py

Generate synthetic TESI interview responses using OpenRouter API.

Reads a YAML config specifying:
    - models (name, provider, model ID, temperature, max_tokens)
    - system prompt
    - TESI questions (id, text)
    - settings (responses_per_question, output_dir)

Outputs:
    - synthetic_transcripts/{model_name}/{question_id}_{run_id}.txt
    - synthetic_transcripts/metadata.json
"""

import json
import os
import random
import time
from pathlib import Path
from typing import Optional

import requests
import yaml


# --- API client ---

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1/chat/completions"


def _get_api_key() -> str:
    """Read OpenRouter API key from environment."""
    key = os.environ.get("OPENROUTER_API_KEY", "")
    if not key:
        raise RuntimeError(
            "OPENROUTER_API_KEY environment variable is not set. "
            "Get a key at https://openrouter.ai/keys"
        )
    return key


def _call_openrouter(
    model: str,
    messages: list,
    temperature: float = 0.7,
    max_tokens: int = 512,
    retries: int = 3,
    base_delay: float = 2.0,
) -> str:
    """
    Call OpenRouter chat completions endpoint with retry logic.

    Parameters
    ----------
    model : str
        OpenRouter model identifier, e.g. "openai/gpt-4o".
    messages : list of dict
        OpenAI-format messages: [{"role": "system", "content": "..."}, ...]
    temperature : float
        Sampling temperature.
    max_tokens : int
        Max tokens to generate.
    retries : int
        Number of retries on failure.
    base_delay : float
        Base delay in seconds for exponential backoff.

    Returns
    -------
    str — generated text content
    """
    api_key = _get_api_key()
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    for attempt in range(retries):
        try:
            response = requests.post(
                OPENROUTER_BASE_URL,
                headers=headers,
                json=payload,
                timeout=120,
            )
            response.raise_for_status()
            data = response.json()

            # Extract generated text
            choices = data.get("choices", [])
            if not choices:
                raise ValueError("No choices in API response")

            content = choices[0].get("message", {}).get("content", "").strip()
            if not content:
                raise ValueError("Empty content in API response")

            return content

        except requests.HTTPError as e:
            if response.status_code == 429:
                # Rate limited — wait longer
                delay = base_delay * (2 ** attempt) + random.uniform(0, 2)
                print(f"  Rate limited. Waiting {delay:.1f}s...")
                time.sleep(delay)
                continue
            if attempt < retries - 1:
                time.sleep(base_delay * (2 ** attempt))
                continue
            raise RuntimeError(f"OpenRouter API error: {e}") from e

        except Exception as e:
            if attempt < retries - 1:
                time.sleep(base_delay * (2 ** attempt))
                continue
            raise RuntimeError(f"OpenRouter call failed after {retries} retries: {e}") from e

    raise RuntimeError("Exhausted all retries")


# --- Config loading ---

def load_config(path: str) -> dict:
    """Load generator YAML config."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# --- Generation ---

def generate_responses(
    config_path: str,
    output_dir: Optional[str] = None,
    models_override: Optional[list] = None,
    questions_override: Optional[list] = None,
) -> Path:
    """
    Generate synthetic TESI responses for all configured models and questions.

    Parameters
    ----------
    config_path : str
        Path to generator YAML config.
    output_dir : str or None
        Override output directory from config.
    models_override : list or None
        Override which models to use (list of model config dicts).
    questions_override : list or None
        Override which questions to use.

    Returns
    -------
    Path to output directory
    """
    config = load_config(config_path)
    models = models_override or config.get("models", {})
    questions = questions_override or config.get("questions", [])
    system_prompt = config.get("system_prompt", "")
    settings = config.get("settings", {})

    out_dir = Path(output_dir or settings.get("output_dir", "synthetic_transcripts"))
    out_dir.mkdir(parents=True, exist_ok=True)

    responses_per_question = settings.get("responses_per_question", 5)

    metadata = {
        "config_path": str(config_path),
        "output_dir": str(out_dir),
        "models": {},
        "responses": [],
    }

    for model_name, model_cfg in models.items():
        print(f"\n=== Model: {model_name} ===")
        model_out = out_dir / model_name
        model_out.mkdir(parents=True, exist_ok=True)

        provider = model_cfg.get("provider", "openrouter")
        model_id = model_cfg.get("model", "")
        temperature = model_cfg.get("temperature", 0.7)
        max_tokens = model_cfg.get("max_tokens", 512)

        metadata["models"][model_name] = {
            "provider": provider,
            "model_id": model_id,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        for q in questions:
            q_id = q["id"]
            q_text = q["text"]

            for run in range(1, responses_per_question + 1):
                file_name = f"{q_id}_run{run}.txt"
                file_path = model_out / file_name

                if file_path.exists():
                    print(f"  Skipping existing: {file_name}")
                    continue

                print(f"  Generating {file_name} ...")

                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": q_text},
                ]

                try:
                    if provider == "openrouter":
                        content = _call_openrouter(
                            model=model_id,
                            messages=messages,
                            temperature=temperature,
                            max_tokens=max_tokens,
                        )
                    else:
                        raise ValueError(f"Unsupported provider: {provider}")

                    file_path.write_text(content, encoding="utf-8")

                    metadata["responses"].append({
                        "model": model_name,
                        "question_id": q_id,
                        "question_text": q_text,
                        "run": run,
                        "file": str(file_path.relative_to(out_dir)),
                    })

                    # Small delay to avoid rate limits
                    time.sleep(0.5)

                except Exception as e:
                    print(f"  ERROR generating {file_name}: {e}")
                    metadata["responses"].append({
                        "model": model_name,
                        "question_id": q_id,
                        "run": run,
                        "error": str(e),
                    })

    # Save metadata
    meta_path = out_dir / "metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nMetadata saved to: {meta_path}")
    return out_dir


# --- CLI entry ---

def main_generate(args=None):
    """CLI entry point for generate subcommand."""
    import argparse

    parser = argparse.ArgumentParser(description="Generate synthetic TESI responses")
    parser.add_argument("--config", required=True, help="Path to generator YAML config")
    parser.add_argument("--output-dir", default=None, help="Override output directory")
    parsed = parser.parse_args(args)

    generate_responses(config_path=parsed.config, output_dir=parsed.output_dir)


if __name__ == "__main__":
    main_generate()
