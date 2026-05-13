"""Reward functions for the fine-tuning task.

Two-stage reward:
1. `schema_validity_reward` — binary: does the model output parse as JSON
   AND validate against `PriorAuthExtraction`?
2. `field_match_reward` — graded: how close is the parsed output to the
   expected output, field by field?

`combined_reward` mixes them. Invalid JSON / invalid schema gets full
zero — there's no partial credit for malformed output, because malformed
output is unusable downstream.
"""

from __future__ import annotations

import json
import re
from typing import Any

from pydantic import ValidationError

from tunelab.schema import PriorAuthExtraction


def _extract_json(text: str) -> str | None:
    """Pull the first JSON object out of a model output.

    Models often emit JSON wrapped in prose ("Here's the extraction: { ... }").
    The trainer should learn to emit clean JSON, but the reward function
    is tolerant so a model that's *almost* right gets partial credit
    during early training.
    """
    if not text:
        return None
    text = text.strip()
    # Strip code-fence markers
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    # If it's already valid JSON, return as-is
    text = text.strip()
    if text.startswith("{") and text.endswith("}"):
        return text
    # Try to find the first {...} block
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end <= start:
        return None
    return text[start : end + 1]


def schema_validity_reward(output: str) -> float:
    """1.0 if the output is valid JSON AND validates as PriorAuthExtraction; else 0.0."""
    raw = _extract_json(output)
    if raw is None:
        return 0.0
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError:
        return 0.0
    try:
        PriorAuthExtraction.model_validate(obj)
    except ValidationError:
        return 0.0
    return 1.0


def _normalize(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip().lower())


def _fuzzy_string_match(predicted: str, expected: str) -> float:
    """1.0 if exact normalized match, 0.5 if either is substring of the other, else 0.0."""
    p, e = _normalize(predicted), _normalize(expected)
    if not e:
        return 1.0 if not p else 0.0
    if p == e:
        return 1.0
    if p and (p in e or e in p):
        return 0.5
    return 0.0


def _set_match(predicted: list, expected: list, key_fn=lambda x: x) -> float:
    """For list-of-things fields: precision × recall on normalized keys."""
    if not expected:
        return 1.0 if not predicted else 0.0
    pred_keys = {_normalize(str(key_fn(x))) for x in (predicted or [])}
    expected_keys = {_normalize(str(key_fn(x))) for x in expected}
    hits = len(pred_keys & expected_keys)
    if not pred_keys and not expected_keys:
        return 1.0
    precision = hits / max(1, len(pred_keys))
    recall = hits / max(1, len(expected_keys))
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)  # F1


def field_match_reward(predicted: dict[str, Any], expected: dict[str, Any]) -> float:
    """Graded reward across the expected fields. Mean of per-field scores.

    Each field type uses its own comparator:
    - strings → fuzzy match
    - numbers → exact match (with tolerance)
    - lists  → F1 over normalized keys
    """
    scores: list[float] = []

    # primary_diagnosis (string)
    scores.append(_fuzzy_string_match(predicted.get("primary_diagnosis", ""), expected.get("primary_diagnosis", "")))

    # requested_service (string)
    if expected.get("requested_service"):
        scores.append(_fuzzy_string_match(predicted.get("requested_service", ""), expected["requested_service"]))

    # diagnosis_duration_years (numeric, with tolerance)
    if expected.get("diagnosis_duration_years") is not None:
        try:
            p = float(predicted.get("diagnosis_duration_years") or -999)
            e = float(expected["diagnosis_duration_years"])
            scores.append(1.0 if abs(p - e) <= 1.0 else 0.5 if abs(p - e) <= 3.0 else 0.0)
        except (TypeError, ValueError):
            scores.append(0.0)

    # medications (list of dicts) — F1 over normalized names
    scores.append(
        _set_match(predicted.get("medications", []), expected.get("medications", []),
                   key_fn=lambda m: m.get("name", "") if isinstance(m, dict) else "")
    )

    # contraindications + red_flags — F1
    scores.append(_set_match(predicted.get("contraindications", []), expected.get("contraindications", [])))
    scores.append(_set_match(predicted.get("red_flags", []), expected.get("red_flags", [])))

    return sum(scores) / max(1, len(scores))


def combined_reward(output: str, expected: dict[str, Any]) -> dict[str, float]:
    """The reward signal handed to GRPO during training.

    Returns a dict with the components + the final scalar reward, so we
    can log each piece separately in W&B / TensorBoard.
    """
    schema_ok = schema_validity_reward(output)
    field_score = 0.0
    if schema_ok > 0:
        raw = _extract_json(output)
        try:
            predicted = json.loads(raw) if raw else {}
        except json.JSONDecodeError:
            predicted = {}
        field_score = field_match_reward(predicted, expected)
    final = 0.3 * schema_ok + 0.7 * field_score
    return {
        "reward": final,
        "schema_ok": schema_ok,
        "field_match": field_score,
    }
