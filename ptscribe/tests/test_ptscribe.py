"""Tests for ptscribe — the scribe pipeline, the eval harness, and the
SQLite monitoring layer.

Every test uses the mock chat or hand-built models, so they run fast and
have no API-key dependency. The bundled transcripts double as the
golden test set: scribe → eval → assert clean extraction.
"""

from __future__ import annotations

from pathlib import Path

from ptscribe import llm
from ptscribe.eval import section_completeness, run_eval
from ptscribe.hallucination import find_hallucinations
from ptscribe.mock import make_mock_chat
from ptscribe.models import (
    Assessment,
    CheckStatus,
    EvalResult,
    Exercise,
    HallucinationFinding,  # noqa: F401  (kept for future tests)
    MMTGrade,
    Objective,
    PainScore,
    Plan,
    ROMMeasurement,
    RunRecord,
    SOAPNote,
    Side,
    Subjective,
)
from ptscribe.monitoring import (
    aggregate_stats,
    log_from_eval,
    log_run,
    recent_runs,
    reset_for_test,
)
from ptscribe.scribe import _scrub, extract_soap

ROOT = Path(__file__).resolve().parents[1]
TRANSCRIPTS = ROOT / "data" / "transcripts"


# -- the bundled transcripts each scribe + eval cleanly ----------------------

def _setup_mock():
    llm.set_chat_fn(make_mock_chat())


def _teardown_mock():
    llm.reset_chat_fn()


def test_every_bundled_transcript_extracts_into_a_full_soap_note():
    """Regression test on the golden set — what would run on every prompt change."""
    _setup_mock()
    try:
        files = sorted(TRANSCRIPTS.glob("*.txt"))
        assert files, "expected bundled transcripts under data/transcripts/"
        for path in files:
            transcript = path.read_text()
            note, _chat = extract_soap(transcript)
            assert note.patient_label, f"{path.stem}: no patient_label"
            assert note.discipline, f"{path.stem}: no discipline"
            assert note.assessment.summary, f"{path.stem}: empty assessment summary"
            # Plan must include something actionable.
            assert (note.plan.home_exercise_program
                    or note.plan.interventions_today
                    or note.plan.next_visit), f"{path.stem}: empty plan"
    finally:
        _teardown_mock()


def test_bundled_transcripts_produce_zero_hallucinations():
    """The canned mock SOAP notes are hand-grounded; the checker should find none.

    This is the regression invariant: if a future prompt change makes the
    scribe drift, hallucinations will appear here first.
    """
    _setup_mock()
    try:
        for path in sorted(TRANSCRIPTS.glob("*.txt")):
            transcript = path.read_text()
            note, _chat = extract_soap(transcript)
            findings = find_hallucinations(transcript, note)
            assert not findings, (
                f"{path.stem}: expected zero ungrounded claims, got "
                f"{[(f.field_path, f.claim) for f in findings]}"
            )
    finally:
        _teardown_mock()


def test_bundled_transcripts_pass_overall():
    _setup_mock()
    try:
        for path in sorted(TRANSCRIPTS.glob("*.txt")):
            transcript = path.read_text()
            note, _chat = extract_soap(transcript)
            ev, _cost = run_eval(transcript, note, transcript_id=path.stem)
            assert ev.overall == CheckStatus.PASS, (
                f"{path.stem}: overall = {ev.overall.value}, expected PASS"
            )
            assert ev.has_all_sections
            assert ev.completeness_score >= 0.7
    finally:
        _teardown_mock()


# -- scrub: null-numeric entries get dropped before pydantic validation -----

def test_scrub_drops_rom_with_null_degrees():
    payload = {
        "objective": {
            "rom": [
                {"joint": "knee flexion", "side": "right", "degrees": 110.0},
                {"joint": "shoulder flexion", "side": "right", "degrees": None},
            ],
        },
    }
    cleaned = _scrub(payload)
    assert len(cleaned["objective"]["rom"]) == 1
    assert cleaned["objective"]["rom"][0]["joint"] == "knee flexion"


def test_scrub_drops_mmt_with_null_grade_and_pain_with_null_score():
    payload = {
        "subjective": {
            "pain": [
                {"location": "knee", "score": 4},
                {"location": "back", "score": None},
            ],
        },
        "objective": {
            "strength": [
                {"muscle_group": "quad", "grade": 4.0},
                {"muscle_group": "deltoid", "grade": None},
            ],
        },
    }
    cleaned = _scrub(payload)
    assert len(cleaned["subjective"]["pain"]) == 1
    assert len(cleaned["objective"]["strength"]) == 1


def test_scrub_handles_missing_sections_without_error():
    assert _scrub({}) == {}
    assert _scrub({"foo": "bar"}) == {"foo": "bar"}


# -- hallucination checker -- targeted behaviour ----------------------------

def _note_with(rom: list = None, mmt: list = None, pain: list = None,
               exercises: list = None) -> SOAPNote:
    return SOAPNote(
        patient_label="x", visit_type="x", discipline="PT",
        subjective=Subjective(pain=pain or []),
        objective=Objective(rom=rom or [], strength=mmt or []),
        assessment=Assessment(),
        plan=Plan(home_exercise_program=exercises or []),
    )


def test_hallucination_flags_invented_rom():
    note = _note_with(rom=[ROMMeasurement(
        joint="shoulder flexion", side=Side.RIGHT, degrees=150.0
    )])
    transcript = "Patient is two weeks post left ankle sprain."  # nothing about shoulder
    findings = find_hallucinations(transcript, note)
    assert any("rom" in f.field_path for f in findings)


def test_hallucination_flags_invented_exercise():
    note = _note_with(exercises=[Exercise(name="single-leg deadlift")])
    transcript = "Patient prescribed walking and gentle stretching only."
    findings = find_hallucinations(transcript, note)
    assert any("home_exercise_program" in f.field_path for f in findings)


def test_hallucination_accepts_grounded_claims():
    note = _note_with(
        rom=[ROMMeasurement(joint="knee flexion", side=Side.RIGHT, degrees=110.0)],
        exercises=[Exercise(name="quad sets")],
    )
    transcript = "Right knee flexion is 110 degrees. Prescribed quad sets at home."
    findings = find_hallucinations(transcript, note)
    assert not findings, f"expected zero, got {[f.field_path for f in findings]}"


# -- section completeness ---------------------------------------------------

def test_completeness_empty_note():
    score, has_all = section_completeness(SOAPNote(discipline="PT"))
    assert score == 0.0
    assert has_all is False


def test_completeness_full_note():
    note = SOAPNote(
        discipline="PT", patient_label="x", visit_type="initial",
        subjective=Subjective(
            chief_complaint="x", history="y",
            pain=[PainScore(location="knee", score=4)],
            patient_goals=["walk"],
        ),
        objective=Objective(
            rom=[ROMMeasurement(joint="knee flexion", degrees=110.0)],
            strength=[MMTGrade(muscle_group="quad", grade=4.0)],
            observations=["limp"],
        ),
        assessment=Assessment(summary="x", progress="improving"),
        plan=Plan(
            interventions_today=["x"],
            home_exercise_program=[Exercise(name="x")],
            next_visit="weekly",
        ),
    )
    score, has_all = section_completeness(note)
    assert score == 1.0
    assert has_all is True


# -- monitoring -------------------------------------------------------------

def test_monitoring_logs_and_aggregates():
    reset_for_test()
    log_run(RunRecord(
        transcript_id="t1", model="m", mode="demo",
        cost_usd=0.001, latency_ms=300,
        input_chars=100, output_chars=400,
        completeness_score=0.9, hallucination_count=0,
    ))
    log_run(RunRecord(
        transcript_id="t2", model="m", mode="demo",
        cost_usd=0.002, latency_ms=500,
        input_chars=200, output_chars=600,
        completeness_score=0.8, hallucination_count=1,
    ))
    stats = aggregate_stats()
    assert stats["total_runs"] == 2
    assert stats["hallucinations_total"] == 1
    assert stats["hallucinations_per_run"] == 0.5
    assert stats["p50_latency_ms"] in (300, 500)

    runs = recent_runs(10)
    assert len(runs) == 2
    # most-recent-first
    assert runs[0].transcript_id == "t2"


def test_log_from_eval_writes_one_row():
    reset_for_test()
    er = EvalResult(transcript_id="x", has_all_sections=True,
                    completeness_score=0.85, hallucination_findings=[],
                    overall=CheckStatus.PASS)
    log_from_eval(
        transcript_id="x", model="m", mode="demo",
        cost_usd=0.0, latency_ms=200, input_chars=100, output_chars=400,
        eval_result=er,
    )
    assert aggregate_stats()["total_runs"] == 1


if __name__ == "__main__":
    import sys

    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"ok   {fn.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL {fn.__name__}: {e}")
    sys.exit(1 if failed else 0)
