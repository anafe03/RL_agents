"""Canned scribe + judge responses for demo mode.

Lets the whole app — scribe, eval, monitoring dashboard — run without
an API key on the bundled transcripts. Each canned SOAP note is hand-
authored to be *grounded in its transcript* so the hallucination checker
finds zero issues on it; this is what proves the eval harness is the
real check, not a vanity badge.
"""

from __future__ import annotations

import json
from typing import Any

from ptscribe import llm

# -- detection ---------------------------------------------------------------


def _detect_case(user_text: str) -> str:
    t = user_text.lower()
    if "total knee arthroplasty" in t or "right knee" in t and "post-op" in t:
        return "knee_post_op"
    if "low back" in t or "lumbar strain" in t:
        return "low_back_initial"
    if "stroke" in t or "hemiparesis" in t or "middle cerebral artery" in t:
        return "stroke_rehab_progress"
    return ""


def _detect_stage(system: str) -> str:
    if "clinical documentation assistant" in system:
        return "scribe"
    if "narrative sections faithfully capture" in system:
        return "judge"
    return "scribe"


# -- canned SOAP notes -------------------------------------------------------

_SCRIBE_RESPONSES: dict[str, dict[str, Any]] = {
    "knee_post_op": {
        "patient_label": "Patient A — right total knee arthroplasty, week 3 post-op",
        "visit_type": "post-op visit",
        "discipline": "PT",
        "subjective": {
            "chief_complaint": "post-op recovery from right TKA",
            "history": "67-year-old female three weeks post right total knee arthroplasty, working on regaining strength and motion.",
            "pain": [
                {"location": "right knee", "score": 4, "when": "at rest"},
                {"location": "right knee", "score": 6, "when": "with stairs"},
            ],
            "functional_limitations": ["difficulty with stair descent"],
            "patient_goals": ["return to walking her dog within six weeks",
                              "return to gardening within six weeks"],
        },
        "objective": {
            "rom": [
                {"joint": "knee flexion", "side": "right", "degrees": 110.0, "description": ""},
                {"joint": "knee extension", "side": "right", "degrees": 5.0,
                 "description": "lacks 5° of full extension"},
            ],
            "strength": [
                {"muscle_group": "quadriceps", "side": "right", "grade": 3.0, "note": ""},
                {"muscle_group": "hamstring", "side": "right", "grade": 4.0, "note": ""},
            ],
            "special_tests": [],
            "observations": [
                "mild effusion",
                "decreased right stance time during gait",
                "reduced terminal knee extension at toe-off",
            ],
        },
        "assessment": {
            "summary": "Patient is progressing as expected three weeks post right TKA. ROM gains are on track, but right quadriceps strength remains the primary limiter for stair function.",
            "progress": "improving — ROM on schedule, strength lagging",
            "impairments": [
                "right knee extension ROM deficit",
                "right quadriceps weakness",
                "antalgic gait pattern",
            ],
        },
        "plan": {
            "interventions_today": [],
            "home_exercise_program": [
                {"name": "terminal knee extensions", "sets": 3, "reps": 15,
                 "hold_seconds": None, "sessions_per_day": 2, "note": ""},
                {"name": "heel slides", "sets": 3, "reps": 10,
                 "hold_seconds": None, "sessions_per_day": None, "note": ""},
                {"name": "quad sets", "sets": 3, "reps": 10,
                 "hold_seconds": 5, "sessions_per_day": 2, "note": ""},
            ],
            "next_visit": "twice weekly for four weeks",
            "referrals": [],
        },
    },
    "low_back_initial": {
        "patient_label": "Patient B — acute low back pain, initial evaluation",
        "visit_type": "initial eval",
        "discipline": "PT",
        "subjective": {
            "chief_complaint": "low back pain after lifting a heavy box at work",
            "history": "42-year-old male with two-week history of right low back pain following a lifting incident at work. No leg symptoms.",
            "pain": [
                {"location": "right low back", "score": 7, "when": "with bending forward"},
                {"location": "right low back", "score": 3, "when": "at rest"},
            ],
            "functional_limitations": [
                "cannot sit at desk longer than 30 minutes",
                "cannot lift toddler without significant pain",
            ],
            "patient_goals": [
                "return to full lifting tolerance at work",
                "resume recreational golf",
            ],
        },
        "objective": {
            "rom": [
                {"joint": "lumbar flexion", "side": "unspecified", "degrees": 40.0,
                 "description": "pain at end-range"},
                {"joint": "lumbar extension", "side": "unspecified", "degrees": 15.0,
                 "description": "painful"},
            ],
            "strength": [
                {"muscle_group": "hip abductors", "side": "bilateral", "grade": 4.0, "note": ""},
            ],
            "special_tests": [
                "straight leg raise negative bilaterally",
                "slump test negative",
            ],
            "observations": [
                "forward flexed posture",
                "decreased lumbar lordosis",
                "breath-holding with lifting",
                "could not hold forearm plank longer than 20 seconds",
            ],
        },
        "assessment": {
            "summary": "Acute mechanical low back pain consistent with lumbar strain, no neurologic involvement. Good rehab candidate with expected resolution in 4-6 weeks under conservative management.",
            "progress": "initial visit",
            "impairments": [
                "lumbar flexion ROM deficit",
                "lumbar extension ROM deficit",
                "reduced core endurance",
                "altered lifting mechanics",
            ],
        },
        "plan": {
            "interventions_today": [
                "manual therapy with soft tissue work to paraspinals",
                "education on neutral spine and hip-hinge mechanics",
            ],
            "home_exercise_program": [
                {"name": "cat-cow", "sets": 2, "reps": 10,
                 "hold_seconds": None, "sessions_per_day": 3, "note": ""},
                {"name": "glute bridges", "sets": 3, "reps": 10,
                 "hold_seconds": None, "sessions_per_day": None, "note": ""},
                {"name": "walking", "sets": None, "reps": None,
                 "hold_seconds": 1200, "sessions_per_day": 1, "note": "20 minutes daily as tolerated"},
            ],
            "next_visit": "twice weekly for two weeks then reassess",
            "referrals": [],
        },
    },
    "stroke_rehab_progress": {
        "patient_label": "Patient C — left MCA stroke with right hemiparesis, week 8",
        "visit_type": "progress note",
        "discipline": "PT",
        "subjective": {
            "chief_complaint": "right-side weakness after left MCA stroke",
            "history": "71-year-old female eight weeks post left middle cerebral artery stroke. Walking more around the house this week with a quad cane.",
            "pain": [
                {"location": "right shoulder", "score": 5, "when": "with overhead reach"},
            ],
            "functional_limitations": [],
            "patient_goals": [],
        },
        "objective": {
            "rom": [
                {"joint": "shoulder flexion", "side": "right", "degrees": 130.0, "description": ""},
                {"joint": "shoulder abduction", "side": "right", "degrees": 100.0, "description": ""},
                {"joint": "elbow flexion", "side": "right", "degrees": 110.0,
                 "description": "mild tone"},
                {"joint": "ankle dorsiflexion", "side": "right", "degrees": 5.0,
                 "description": "with stretch"},
            ],
            "strength": [
                {"muscle_group": "deltoid", "side": "right", "grade": 3.0, "note": ""},
                {"muscle_group": "biceps", "side": "right", "grade": 3.0, "note": ""},
                {"muscle_group": "tibialis anterior", "side": "right", "grade": 2.0, "note": ""},
                {"muscle_group": "gastrocnemius", "side": "right", "grade": 3.0, "note": ""},
            ],
            "special_tests": [],
            "observations": [
                "slow gait with quad cane",
                "decreased right step length",
                "circumduction at right hip during swing",
            ],
        },
        "assessment": {
            "summary": "Continued steady progress in right-side strength and motor control 8 weeks post left MCA stroke. Right ankle dorsiflexion remains the primary limiter for gait quality. Patient demonstrates good motivation and consistency with her home program.",
            "progress": "improving — independent toilet transfers achieved this week",
            "impairments": [
                "right ankle dorsiflexion deficit",
                "right tibialis anterior weakness",
                "altered gait pattern with circumduction",
                "right shoulder pain limiting overhead reach",
            ],
        },
        "plan": {
            "interventions_today": [
                "task-specific gait training with rhythmic cues",
                "theraband dorsiflexion strengthening",
            ],
            "home_exercise_program": [
                {"name": "seated heel raises", "sets": 3, "reps": 15,
                 "hold_seconds": None, "sessions_per_day": None, "note": ""},
                {"name": "resisted dorsiflexion with theraband", "sets": 3, "reps": 15,
                 "hold_seconds": None, "sessions_per_day": 2, "note": ""},
                {"name": "step-ups onto a four-inch step", "sets": 2, "reps": 10,
                 "hold_seconds": None, "sessions_per_day": None, "note": ""},
            ],
            "next_visit": "twice weekly for four weeks",
            "referrals": [],
        },
    },
}


# Canned judge scores — calibrated to be a credible quality estimate per case.
_JUDGE_RESPONSES = {
    "knee_post_op": {
        "score": 0.88,
        "reasoning": "Narrative captures post-op timeline, ROM/strength pattern, and progression accurately; consistent with PT documentation voice.",
    },
    "low_back_initial": {
        "score": 0.86,
        "reasoning": "Initial-eval narrative covers mechanism, ruling-out of neurologic involvement, and conservative-care prognosis cleanly.",
    },
    "stroke_rehab_progress": {
        "score": 0.90,
        "reasoning": "Progress-note narrative ties functional gains to the impairment story and names the rate-limiter (right dorsiflexion) like a clinician would.",
    },
}


def _usage(input_tokens: int = 1400, output_tokens: int = 600) -> Any:
    import anthropic
    return anthropic.types.Usage(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cache_creation_input_tokens=0,
        cache_read_input_tokens=0,
        server_tool_use=None,
        service_tier=None,
    )


def make_mock_chat() -> Any:
    def fn(*, system: str, messages: list, **kwargs: Any) -> llm.ChatResult:
        user_text = ""
        if messages and isinstance(messages[-1].get("content"), str):
            user_text = messages[-1]["content"]
        stage = _detect_stage(system)
        case = _detect_case(user_text)

        if stage == "judge":
            payload = _JUDGE_RESPONSES.get(case, {"score": 0.55,
                                                  "reasoning": "(mock fallback — unrecognized case)"})
            return llm.ChatResult(
                text=json.dumps(payload),
                usage=_usage(700, 90),
                cost_usd=0.0,
                model="claude-sonnet-4-6",
                latency_ms=120,
            )

        # scribe stage
        payload = _SCRIBE_RESPONSES.get(case)
        if payload is None:
            payload = {
                "patient_label": "(demo) unrecognized transcript",
                "visit_type": "",
                "discipline": "PT",
                "subjective": {}, "objective": {},
                "assessment": {"summary": "Demo mode could not match this transcript — use a bundled one, or switch to Live mode."},
                "plan": {},
            }
        return llm.ChatResult(
            text=json.dumps(payload),
            usage=_usage(1400, 650),
            cost_usd=0.0,
            model="claude-sonnet-4-6",
            latency_ms=380,
        )

    return fn
