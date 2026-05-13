"""Tests for the reward function + schema + eval harness.

These run without torch/trl/transformers — they exercise the deterministic
pieces only. The training script itself is integration-tested manually
on GPU.
"""

from __future__ import annotations

import json

from tunelab.dataset import load_eval, load_train
from tunelab.eval import mock_baseline_model, run_eval
from tunelab.reward import (
    _extract_json,
    combined_reward,
    field_match_reward,
    schema_validity_reward,
)
from tunelab.schema import PriorAuthExtraction


GOOD_OUTPUT = json.dumps({
    "primary_diagnosis": "Type 2 diabetes mellitus",
    "diagnosis_duration_years": 8,
    "medications": [
        {"name": "metformin", "outcome": "tried_intolerance", "note": ""},
        {"name": "glipizide", "outcome": "tried_hypoglycemia", "note": ""},
    ],
    "contraindications": [],
    "red_flags": [],
    "requested_service": "semaglutide",
})


def test_extract_json_handles_code_fences():
    text = "```json\n" + GOOD_OUTPUT + "\n```"
    assert _extract_json(text).strip().startswith("{")


def test_extract_json_handles_prose_wrapping():
    text = "Here's the extraction:\n\n" + GOOD_OUTPUT + "\n\nLet me know if you need more."
    assert _extract_json(text) is not None


def test_schema_validity_passes_for_valid_output():
    assert schema_validity_reward(GOOD_OUTPUT) == 1.0


def test_schema_validity_fails_for_invalid_outputs():
    assert schema_validity_reward("not json at all") == 0.0
    assert schema_validity_reward('{"bogus": "fields"}') == 0.0
    # Wrong enum value
    bad = json.dumps({
        "primary_diagnosis": "x", "medications": [{"name": "y", "outcome": "MADE_UP"}]
    })
    assert schema_validity_reward(bad) == 0.0


def test_field_match_perfect_when_predicted_equals_expected():
    expected = json.loads(GOOD_OUTPUT)
    predicted = json.loads(GOOD_OUTPUT)
    score = field_match_reward(predicted, expected)
    assert score >= 0.95  # allowing tiny float wobble


def test_field_match_partial_for_partial_match():
    expected = json.loads(GOOD_OUTPUT)
    predicted = json.loads(GOOD_OUTPUT)
    predicted["primary_diagnosis"] = "DIFFERENT thing"
    score = field_match_reward(predicted, expected)
    assert 0.0 < score < 0.95


def test_combined_reward_zero_on_invalid_schema():
    expected = json.loads(GOOD_OUTPUT)
    scores = combined_reward("not json", expected)
    assert scores["schema_ok"] == 0.0
    assert scores["reward"] == 0.0


def test_combined_reward_strong_on_valid_match():
    expected = json.loads(GOOD_OUTPUT)
    scores = combined_reward(GOOD_OUTPUT, expected)
    assert scores["schema_ok"] == 1.0
    assert scores["reward"] >= 0.9


def test_eval_loader_returns_at_least_5_examples():
    examples = load_eval()
    assert len(examples) >= 5
    assert all(e.input and e.expected for e in examples)


def test_train_loader_returns_examples():
    examples = load_train()
    assert len(examples) >= 5


def test_eval_harness_runs_against_mock_baseline():
    report = run_eval(mock_baseline_model)
    assert report.results
    # The mock baseline always returns the same JSON (a GLP-1 / T2D example).
    # It should match the first eval example well, and most others poorly.
    assert 0 < report.mean_reward < 1.0
    # Schema should always pass (mock returns valid JSON every time)
    assert report.schema_pass_rate == 1.0


def test_schema_is_strict_no_extra_fields():
    """The schema forbids extra fields — important for tight reward signal."""
    bad = json.dumps({**json.loads(GOOD_OUTPUT), "bogus_extra_field": 123})
    assert schema_validity_reward(bad) == 0.0


if __name__ == "__main__":
    import sys, pytest
    sys.exit(pytest.main([__file__, "-v"]))
