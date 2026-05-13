"""Multi-pass extraction.

Four independent LLM passes, each with its own focused system prompt:
1. metrics — financial line items
2. tone — per-speaker sentiment with evidence
3. surprises — guidance revisions / new initiatives / things not in prepared remarks
4. analyst_questions — ranked by sharpness

Each pass returns structured JSON parsed into Pydantic models. Citation
quotes are kept as-is for downstream verification (see `verifier.py`).
"""

from __future__ import annotations

import json
from typing import Any

from earningscall import llm
from earningscall.models import (
    AnalystQuestion,
    Metric,
    Quote,
    SpeakerRole,
    SpeakerTurn,
    Surprise,
    ToneAssessment,
    Transcript,
)

DEFAULT_MODEL = "claude-sonnet-4-6"


def _format_transcript(transcript: Transcript) -> str:
    lines: list[str] = [
        f"# {transcript.company} — {transcript.period} earnings call",
        "",
    ]
    for t in transcript.turns:
        role_tag = f" ({t.speaker_role.value})" if t.speaker_role != SpeakerRole.UNKNOWN else ""
        aff_tag = f" — {t.affiliation}" if t.affiliation else ""
        lines.append(f"[turn {t.turn_id}] {t.speaker_name}{role_tag}{aff_tag}:")
        lines.append(t.text)
        lines.append("")
    return "\n".join(lines)


# ---- pass 1: metrics ------------------------------------------------------

METRICS_SYSTEM = """You are a sell-side equity analyst extracting hard financial metrics from an earnings call transcript.

Pull out specific metrics the company REPORTED or GUIDED to. Examples: revenue, adjusted EPS, segment revenue, operating margin, free cash flow, full-year guidance, segment guidance, capital allocation announcements.

For each metric:
- "name": short name (e.g. "Q4 2025 Revenue", "Adjusted EPS", "FY26 Revenue Guidance")
- "value": the value as stated ("$3.42B", "$0.21", "8-10% growth")
- "vs_expectations": "beat" | "miss" | "in-line" | "raised" | "lowered" | "" if unknown
- "quote.speaker_name": who said it
- "quote.text": a SHORT verbatim quote (≤ 200 chars) from the transcript supporting the metric. Must be a literal substring of the transcript text.

Do not invent metrics. Do not paraphrase the quote — copy literally. Skip vague comments; only pull metrics with stated numbers or specific guidance.

Respond ONLY with JSON: {"metrics": [{...}, {...}, ...]}"""


def extract_metrics(transcript: Transcript, model: str = DEFAULT_MODEL) -> tuple[list[Metric], float]:
    transcript_text = _format_transcript(transcript)
    result = llm.chat(
        model=model,
        system=METRICS_SYSTEM,
        messages=[{"role": "user", "content": transcript_text}],
        max_tokens=2048,
    )
    obj = _parse_json(result.text)
    out: list[Metric] = []
    for m in obj.get("metrics", []):
        q = m.get("quote", {}) or {}
        out.append(
            Metric(
                name=m.get("name", ""),
                value=m.get("value", ""),
                vs_expectations=m.get("vs_expectations", ""),
                quote=Quote(speaker_name=q.get("speaker_name", ""), text=q.get("text", "")),
            )
        )
    return out, result.cost_usd


# ---- pass 2: tone ---------------------------------------------------------

TONE_SYSTEM = """You are an experienced sell-side analyst assessing the TONE of an earnings call.

Read the transcript and produce per-speaker tone assessments. Focus on the CEO and CFO; produce one assessment per speaker per coherent segment (e.g. "prepared remarks", "Q&A — guidance", "Q&A — segment results").

For each assessment:
- "speaker_name": who
- "speaker_role": "ceo" | "cfo" | "other_exec"
- "segment": short label
- "sentiment": one of "confident" | "cautiously optimistic" | "measured" | "defensive" | "evasive" | "celebratory"
- "evidence": 1-3 short verbatim quotes that drove your read (each ≤ 200 chars, literal substrings of the transcript)
- "note": one-line analyst-style observation

Be specific. Do not say "the CFO seemed cautious"; quote the specific phrase that shows it. If a speaker visibly shifts tone between prepared remarks and Q&A, write two separate assessments.

Respond ONLY with JSON: {"tone": [{...}, ...]}"""


def extract_tone(transcript: Transcript, model: str = DEFAULT_MODEL) -> tuple[list[ToneAssessment], float]:
    result = llm.chat(
        model=model,
        system=TONE_SYSTEM,
        messages=[{"role": "user", "content": _format_transcript(transcript)}],
        max_tokens=2048,
    )
    obj = _parse_json(result.text)
    out: list[ToneAssessment] = []
    for t in obj.get("tone", []):
        try:
            role = SpeakerRole(t.get("speaker_role", "other_exec"))
        except ValueError:
            role = SpeakerRole.OTHER_EXEC
        evidence = [
            Quote(speaker_name=t.get("speaker_name", ""), text=q.get("text", ""))
            if isinstance(q, dict)
            else Quote(speaker_name=t.get("speaker_name", ""), text=str(q))
            for q in (t.get("evidence") or [])
        ]
        out.append(
            ToneAssessment(
                speaker_name=t.get("speaker_name", ""),
                speaker_role=role,
                segment=t.get("segment", ""),
                sentiment=t.get("sentiment", ""),
                evidence=evidence,
                note=t.get("note", ""),
            )
        )
    return out, result.cost_usd


# ---- pass 3: surprises ----------------------------------------------------

SURPRISES_SYSTEM = """You are an analyst flagging SURPRISES in an earnings call — things that were not obviously expected from prior guidance or prepared remarks.

Candidates: guidance revisions (up or down), unexpected new initiatives or product launches, leadership changes, macro callouts that signal a shift, M&A hints, accounting changes, change in segment reporting.

For each surprise:
- "headline": one-line description
- "kind": "guidance_revision" | "new_initiative" | "leadership_change" | "macro_callout" | "m&a" | "other"
- "significance": "low" | "medium" | "high"
- "evidence": 1-2 short verbatim quotes (each ≤ 200 chars, literal substrings)
- "rationale": one sentence on why it matters

Be conservative: every quarter has prepared remarks; only flag things that genuinely shift the picture or weren't telegraphed.

Respond ONLY with JSON: {"surprises": [{...}, ...]}"""


def extract_surprises(transcript: Transcript, model: str = DEFAULT_MODEL) -> tuple[list[Surprise], float]:
    result = llm.chat(
        model=model,
        system=SURPRISES_SYSTEM,
        messages=[{"role": "user", "content": _format_transcript(transcript)}],
        max_tokens=2048,
    )
    obj = _parse_json(result.text)
    out: list[Surprise] = []
    for s in obj.get("surprises", []):
        evidence = [
            Quote(speaker_name="", text=q.get("text", "") if isinstance(q, dict) else str(q))
            for q in (s.get("evidence") or [])
        ]
        out.append(
            Surprise(
                headline=s.get("headline", ""),
                kind=s.get("kind", "other"),
                significance=s.get("significance", "medium"),
                evidence=evidence,
                rationale=s.get("rationale", ""),
            )
        )
    return out, result.cost_usd


# ---- pass 4: analyst Q&A ranking ------------------------------------------

QA_SYSTEM = """You are a senior analyst rating the QUALITY of analyst questions in an earnings call's Q&A section.

Rate each analyst question on a "sharpness" scale of 1-5, where:
- 5: actually pushed for a substantive, non-obvious answer; got past prepared deflections
- 4: real probing question on a real issue
- 3: standard professional question — fine, not memorable
- 2: softball; could have been asked of any company
- 1: filler; basically a polite hello

Also classify how management answered:
- "direct": addressed the actual question with specifics
- "hedged": acknowledged but qualified everything
- "dodged": pivoted to a different topic or to prepared messaging
- "promised follow-up": "we'll get back to you offline"

For each question:
- "turn_id": the turn index from the transcript header
- "analyst_name": who asked
- "affiliation": the bank/firm
- "question_summary": one-line restatement of what they asked
- "sharpness": 1-5
- "answer_quality": as above
- "quote.text": ≤ 200 chars verbatim substring of the question
- "rationale": one-sentence why it earned this score

Respond ONLY with JSON: {"questions": [{...}, ...]}"""


def extract_analyst_questions(
    transcript: Transcript, model: str = DEFAULT_MODEL
) -> tuple[list[AnalystQuestion], float]:
    result = llm.chat(
        model=model,
        system=QA_SYSTEM,
        messages=[{"role": "user", "content": _format_transcript(transcript)}],
        max_tokens=2048,
    )
    obj = _parse_json(result.text)
    out: list[AnalystQuestion] = []
    for q in obj.get("questions", []):
        quote_obj = q.get("quote", {}) or {}
        out.append(
            AnalystQuestion(
                turn_id=int(q.get("turn_id", 0)),
                analyst_name=q.get("analyst_name", ""),
                affiliation=q.get("affiliation", ""),
                question_summary=q.get("question_summary", ""),
                sharpness=int(q.get("sharpness", 3)),
                answer_quality=q.get("answer_quality", ""),
                quote=Quote(speaker_name=q.get("analyst_name", ""), text=quote_obj.get("text", "")),
                rationale=q.get("rationale", ""),
            )
        )
    return out, result.cost_usd


# ---- helpers --------------------------------------------------------------


def _parse_json(raw: str) -> dict[str, Any]:
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass
    start = raw.find("{")
    end = raw.rfind("}")
    if start == -1 or end <= start:
        return {}
    try:
        return json.loads(raw[start : end + 1])
    except json.JSONDecodeError:
        return {}
