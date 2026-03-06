"""
llm_parser.py
-------------
Handles all communication with the locally hosted Ollama LLM.
Responsibilities:
  - Load the system prompt from disk
  - Send user commands to the Ollama API
  - Validate and return structured JSON intents
  - Apply a safe fallback state on any failure
"""

import json
import logging
import requests
from pathlib import Path

# ── Configuration ─────────────────────────────────────────────────────────────

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL_NAME = "llama3:8b"
SYSTEM_PROMPT_PATH = Path(__file__).parent.parent / "prompts" / "system_prompt.txt"

REQUEST_TIMEOUT = 120  # seconds — llama3:8b needs up to 60-90s on cold start

# ── Safe fallback state ────────────────────────────────────────────────────────
# Returned whenever the LLM output is invalid or a network error occurs.
SAFE_STATE = {
    "intent": "stop",
    "speed_target": 0,
    "urgency": "immediate",
}

# Valid values for schema enforcement
VALID_INTENTS = {"drive", "stop", "turn_left", "turn_right", "reverse", "unknown"}
VALID_URGENCIES = {"normal", "immediate"}

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("llm_parser")


# ── Internal helpers ───────────────────────────────────────────────────────────

def _load_system_prompt() -> str:
    """Load the system prompt from disk. Raises FileNotFoundError if missing."""
    if not SYSTEM_PROMPT_PATH.exists():
        raise FileNotFoundError(
            f"System prompt not found at: {SYSTEM_PROMPT_PATH}\n"
            "Ensure prompts/system_prompt.txt exists in the project root."
        )
    return SYSTEM_PROMPT_PATH.read_text(encoding="utf-8").strip()


def _validate_intent(data: dict) -> dict:
    """
    Enforce the JSON schema on the parsed LLM response.
    Returns the validated dict, or raises ValueError on schema mismatch.
    """
    # Check required keys
    required_keys = {"intent", "speed_target", "urgency"}
    missing = required_keys - data.keys()
    if missing:
        raise ValueError(f"Missing keys in LLM response: {missing}")

    # Validate intent
    if data["intent"] not in VALID_INTENTS:
        raise ValueError(f"Invalid intent '{data['intent']}'. Must be one of {VALID_INTENTS}")

    # Validate speed_target
    if not isinstance(data["speed_target"], int):
        # Attempt to coerce float → int gracefully
        try:
            data["speed_target"] = int(data["speed_target"])
        except (TypeError, ValueError):
            raise ValueError(f"speed_target must be an integer, got: {data['speed_target']!r}")

    if not (0 <= data["speed_target"] <= 120):
        raise ValueError(f"speed_target {data['speed_target']} out of range [0, 120]")

    # Validate urgency
    if data["urgency"] not in VALID_URGENCIES:
        raise ValueError(f"Invalid urgency '{data['urgency']}'. Must be one of {VALID_URGENCIES}")

    return data


# ── Public API ─────────────────────────────────────────────────────────────────

def parse_command(user_input: str) -> dict:
    """
    Translate a natural language driving command into a structured JSON intent.

    Args:
        user_input: Raw string from the Gradio text box.

    Returns:
        A validated dict with keys: intent, speed_target, urgency.
        Falls back to SAFE_STATE on any error.

    Example:
        >>> parse_command("Turn left slowly")
        {'intent': 'turn_left', 'speed_target': 10, 'urgency': 'normal'}
    """
    if not user_input or not user_input.strip():
        logger.warning("Empty input received — returning safe state.")
        return SAFE_STATE.copy()

    # 1. Load system prompt
    try:
        system_prompt = _load_system_prompt()
    except FileNotFoundError as e:
        logger.error(e)
        return SAFE_STATE.copy()

    # 2. Build request payload for Ollama /api/chat endpoint
    payload = {
        "model": MODEL_NAME,
        "stream": False,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_input.strip()},
        ],
    }

    # 3. Query Ollama
    try:
        logger.info(f"Querying Ollama ({MODEL_NAME}) with: '{user_input}'")
        response = requests.post(OLLAMA_URL, json=payload, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
    except requests.exceptions.ConnectionError:
        logger.error(
            "Cannot connect to Ollama. Is it running? Try: ollama serve"
        )
        return SAFE_STATE.copy()
    except requests.exceptions.Timeout:
        logger.error(f"Ollama request timed out after {REQUEST_TIMEOUT}s.")
        return SAFE_STATE.copy()
    except requests.exceptions.HTTPError as e:
        logger.error(f"Ollama HTTP error: {e}")
        return SAFE_STATE.copy()

    # 4. Extract the raw text from the response
    try:
        raw_text = response.json()["message"]["content"].strip()
        logger.info(f"Raw LLM output: {raw_text}")
    except (KeyError, ValueError) as e:
        logger.error(f"Unexpected Ollama response structure: {e}")
        return SAFE_STATE.copy()

    # 5. Strip markdown fences if the LLM wrapped output in ```json ... ```
    if "```" in raw_text:
        lines = raw_text.splitlines()
        raw_text = "\n".join(
            line for line in lines
            if not line.strip().startswith("```")
        ).strip()
        logger.warning(f"Stripped markdown fences. Clean text: {raw_text}")

    # 6. Parse JSON from the LLM output
    try:
        parsed = json.loads(raw_text)
    except json.JSONDecodeError as e:
        logger.error(f"JSONDecodeError — LLM returned non-JSON output: '{raw_text}' | Error: {e}")
        return SAFE_STATE.copy()

    # 6. Validate schema
    try:
        validated = _validate_intent(parsed)
        logger.info(f"Validated intent: {validated}")
        return validated
    except ValueError as e:
        logger.error(f"Schema validation failed: {e}")
        return SAFE_STATE.copy()


# ── Quick test harness ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    test_commands = [
        "Drive forward at 40 km/h",
        "Hit the brakes right now!",
        "Turn left slowly",
        "Go in reverse",
        "Speed up to 80",
        "Fly to the moon",          # Should return unknown
        "",                         # Should return safe state
    ]

    print("\n" + "=" * 55)
    print("  NLVC — LLM Parser Test Harness")
    print("=" * 55)

    for cmd in test_commands:
        result = parse_command(cmd)
        label = f"'{cmd}'" if cmd else "(empty string)"
        print(f"\nInput : {label}")
        print(f"Output: {result}")

    print("\n" + "=" * 55)