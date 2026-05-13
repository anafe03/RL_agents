"""Eval harness — runs any callable model_fn over the eval set and reports.

`model_fn: str -> str` — takes the input free-text, returns the model's
output. Works for HF transformers, API-based models, mocks — anything you
can wrap into a callable.

Reports per-example reward components + aggregate mean for the run.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from typing import Callable

from tunelab.dataset import Example, load_eval
from tunelab.reward import combined_reward


@dataclass
class ExampleResult:
    input: str
    output: str
    reward: float
    schema_ok: float
    field_match: float


@dataclass
class EvalReport:
    results: list[ExampleResult] = field(default_factory=list)

    @property
    def mean_reward(self) -> float:
        return statistics.mean(r.reward for r in self.results) if self.results else 0.0

    @property
    def schema_pass_rate(self) -> float:
        return statistics.mean(r.schema_ok for r in self.results) if self.results else 0.0

    @property
    def mean_field_match(self) -> float:
        return statistics.mean(r.field_match for r in self.results) if self.results else 0.0


def run_eval(
    model_fn: Callable[[str], str],
    examples: list[Example] | None = None,
) -> EvalReport:
    """Run model_fn against every example and produce a report."""
    examples = examples if examples is not None else load_eval()
    report = EvalReport()
    for ex in examples:
        output = model_fn(ex.input)
        scores = combined_reward(output, ex.expected)
        report.results.append(
            ExampleResult(
                input=ex.input,
                output=output,
                reward=scores["reward"],
                schema_ok=scores["schema_ok"],
                field_match=scores["field_match"],
            )
        )
    return report


def mock_baseline_model(prompt: str) -> str:
    """A "baseline" mock that returns vaguely-correct JSON but with errors.

    Useful for testing the eval harness end-to-end without a real model.
    Returns hardcoded plausible JSON that partially matches the eval set —
    the reward should land somewhere in the middle.
    """
    return (
        '{"primary_diagnosis": "Type 2 diabetes mellitus", '
        '"diagnosis_duration_years": 8, '
        '"medications": [{"name": "metformin", "outcome": "tried_intolerance"}], '
        '"contraindications": [], '
        '"red_flags": [], '
        '"requested_service": "semaglutide"}'
    )
