"""Tests for AutoFill — exercises the data models, target registry, and mock playback.

Playwright + real Computer Use are NOT tested in CI — they require a
browser binary and an API key. The mock playback is fully exercisable.
"""

from __future__ import annotations

from pathlib import Path

from autofill.mock import get_mock_run, mock_ca_doi_glp1_run
from autofill.models import StepAction, load_complaint
from autofill.targets import REGISTRY, get_target

ROOT = Path(__file__).resolve().parents[1]


def test_targets_registry_has_ca_doi():
    assert "ca_doi" in REGISTRY
    t = get_target("ca_doi")
    assert t.state == "CA"
    assert "insurance.ca.gov" in t.url
    assert "narrative" in t.field_hints
    assert "insurer_name" in t.field_hints


def test_glp1_complaint_loads():
    path = ROOT / "data" / "complaints" / "glp1_denial.yaml"
    c = load_complaint(path)
    assert c.id == "glp1_denial"
    assert c.insurer_name == "Cascade Health Plan"
    assert "step therapy" in c.denial_reason.lower() or "step-therapy" in c.denial_reason.lower()
    assert c.narrative.strip().startswith("I am filing this complaint")


def test_mock_playback_has_expected_shape():
    result = mock_ca_doi_glp1_run()
    assert result.target_id == "ca_doi"
    assert result.complaint_id == "glp1_denial"
    assert result.dry_run is True
    assert result.completed_to_review is True

    # Should have at least one of each major action type
    action_types = {s.action for s in result.steps}
    assert StepAction.NAVIGATE in action_types
    assert StepAction.TYPE in action_types
    assert StepAction.CLICK in action_types
    assert StepAction.HALT in action_types

    # Should end with HALT (no real submit in dry-run)
    assert result.steps[-1].action == StepAction.HALT


def test_mock_playback_includes_insurer_name_in_some_step():
    result = mock_ca_doi_glp1_run()
    typed_values = [s.value for s in result.steps if s.action == StepAction.TYPE]
    assert any("Cascade Health Plan" in v for v in typed_values), (
        "expected the insurer name to appear in at least one TYPE step's value"
    )


def test_get_mock_run_returns_none_for_unknown_pair():
    assert get_mock_run("unknown_target", "unknown_complaint") is None


if __name__ == "__main__":
    test_targets_registry_has_ca_doi()
    test_glp1_complaint_loads()
    test_mock_playback_has_expected_shape()
    test_mock_playback_includes_insurer_name_in_some_step()
    test_get_mock_run_returns_none_for_unknown_pair()
    print("ok")
