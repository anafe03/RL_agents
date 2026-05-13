"""Deterministic mock chat for the public demo.

Returns canned, well-formed JSON keyed off (transcript, pipeline-pass) so
the full 4-pass pipeline produces a realistic report with no API key.
"""

from __future__ import annotations

import json
from typing import Any

import anthropic

from earningscall import llm


def _usage(input_tokens: int = 2000, output_tokens: int = 600) -> anthropic.types.Usage:
    return anthropic.types.Usage(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cache_creation_input_tokens=1200,
        cache_read_input_tokens=800,
        server_tool_use=None,
        service_tier=None,
    )


# ---- canned responses per transcript --------------------------------------

GLOW = {
    "metrics": {"metrics": [
        {"name": "Q4 2025 Revenue", "value": "$84.2 million", "vs_expectations": "miss",
         "quote": {"speaker_name": "Diane Park", "text": "Q4 revenue was $84.2 million, missing consensus by approximately $9 million"}},
        {"name": "Q4 2025 Adjusted Net Loss per share", "value": "$0.34", "vs_expectations": "",
         "quote": {"speaker_name": "Diane Park", "text": "Adjusted net loss was $0.34 per share"}},
        {"name": "Q4 2025 Gross Margin", "value": "71%", "vs_expectations": "miss",
         "quote": {"speaker_name": "Marcus Reeves", "text": "Gross margin came in at 71 percent, down 320 basis points year-over-year"}},
        {"name": "Cash and equivalents (Q4 end)", "value": "$612 million", "vs_expectations": "",
         "quote": {"speaker_name": "Marcus Reeves", "text": "We ended the quarter with $612 million in cash and equivalents"}},
        {"name": "FY2026 Revenue Guidance", "value": "$360M–$390M", "vs_expectations": "lowered",
         "quote": {"speaker_name": "Marcus Reeves", "text": "we are revising our revenue guidance to a range of $360 to $390 million, down from the prior $440 to $470 million range"}},
        {"name": "FY2026 Operating Cash Burn", "value": "$180M–$210M", "vs_expectations": "",
         "quote": {"speaker_name": "Marcus Reeves", "text": "We continue to expect operating cash burn of approximately $180 to $210 million for the year"}},
    ]},
    "tone": {"tone": [
        {"speaker_name": "Diane Park", "speaker_role": "ceo", "segment": "prepared remarks",
         "sentiment": "cautiously optimistic",
         "evidence": [
             {"text": "Q4 was a quarter of meaningful clinical progress for Glow, although our commercial results came in below where we and the Street had positioned us"},
             {"text": "our Phase 3 GLOW-301 readout in moderate-to-severe atopic dermatitis met its primary endpoint with a co-primary endpoint p-value of less than 0.001"}
         ],
         "note": "CEO frames the quarter around clinical wins to offset the commercial miss"},
        {"speaker_name": "Marcus Reeves", "speaker_role": "cfo", "segment": "prepared remarks — guidance",
         "sentiment": "defensive",
         "evidence": [
             {"text": "we are revising our revenue guidance to a range of $360 to $390 million, down from the prior $440 to $470 million range"},
             {"text": "We'd rather guide conservatively for the year and earn it back"}
         ],
         "note": "CFO leans into 'sandbag and beat' posture, signaling the cut may be larger than warranted"},
        {"speaker_name": "Marcus Reeves", "speaker_role": "cfo", "segment": "Q&A — capital allocation",
         "sentiment": "evasive",
         "evidence": [
             {"text": "We're not going to comment on specific financing scenarios on this call"}
         ],
         "note": "CFO declines to engage on Evercore's pointed financing question"},
    ]},
    "surprises": {"surprises": [
        {"headline": "FY2026 revenue guidance cut by ~17% at top end",
         "kind": "guidance_revision", "significance": "high",
         "evidence": [{"text": "down from the prior $440 to $470 million range we provided last quarter"}],
         "rationale": "Large reset of expectations the company set just last quarter — implies the dermatology channel weakness is more than a near-term blip"},
        {"headline": "One-time inventory write-down on legacy GlowDerm; more to come Q1",
         "kind": "other", "significance": "medium",
         "evidence": [
             {"text": "a one-time inventory write-down on legacy GlowDerm product"},
             {"text": "we do expect a modest additional charge in Q1 2026 as we complete the channel transition, in the range of $4 to $7 million"}
         ],
         "rationale": "Signals the formulation transition is messier than previously disclosed; gross margin won't normalize until exit 2026"},
        {"headline": "Cardiology indication launch delayed",
         "kind": "other", "significance": "medium",
         "evidence": [{"text": "a delayed launch of the cardiology indication"}],
         "rationale": "Buried in the guidance commentary — accounts for ~1/3 of the FY26 cut and was not previously disclosed as delayed"},
    ]},
    "questions": {"questions": [
        {"turn_id": 4, "analyst_name": "Priya Iyer", "affiliation": "Goldman Sachs",
         "question_summary": "Decompose the FY26 guide-cut: derm channel vs cardiology, and is the derm weakness structural or transitory?",
         "sharpness": 5, "answer_quality": "direct",
         "quote": {"text": "I'm trying to understand whether this is a one-quarter recalibration or a structural reset"},
         "rationale": "Asks two precise, hard-to-deflect questions, forces a split estimate (two-thirds / one-third), and gets the company to admit the answer is 'between' — best question on the call"},
        {"turn_id": 7, "analyst_name": "David Kim", "affiliation": "Evercore ISI",
         "question_summary": "Have you considered raising capital opportunistically given the stock price?",
         "sharpness": 4, "answer_quality": "dodged",
         "quote": {"text": "Have you and the board considered raising capital opportunistically given where the stock is, versus waiting until closer to a potential approval"},
         "rationale": "Sharp question about a real strategic option; CFO declined to engage — analyst gets useful info from the non-answer"},
        {"turn_id": 10, "analyst_name": "Sarah Mendoza", "affiliation": "Raymond James",
         "question_summary": "How are you thinking about derm-vs-PCP outreach in the AD launch?",
         "sharpness": 3, "answer_quality": "direct",
         "quote": {"text": "could you talk about how you're thinking about the commercial launch playbook for the AD indication"},
         "rationale": "Standard launch-playbook question; CEO answered cleanly but no new information"},
        {"turn_id": 13, "analyst_name": "Chen Wei", "affiliation": "Jefferies",
         "question_summary": "Is Q4 gross margin a clean number, or are there more inventory charges in our future?",
         "sharpness": 4, "answer_quality": "direct",
         "quote": {"text": "is the 71 percent gross margin in Q4 a clean number or are there more one-times in our future?"},
         "rationale": "Specifically extracted the $4-7M Q1 charge and the 73-75% exit-2026 margin target — concrete numbers the model could use"},
    ]},
}


HELIOS = {
    "metrics": {"metrics": [
        {"name": "Q3 2025 Revenue", "value": "$214 million", "vs_expectations": "beat",
         "quote": {"speaker_name": "Tomas Vidal", "text": "We reported revenue of $214 million, a 38 percent increase year-over-year and approximately $9 million ahead of consensus"}},
        {"name": "Q3 2025 Gross Margin", "value": "42.1%", "vs_expectations": "",
         "quote": {"speaker_name": "Annie Brooks", "text": "Gross margin was 42.1 percent, down 180 basis points year-over-year"}},
        {"name": "Q3 2025 Adjusted EPS", "value": "$0.31", "vs_expectations": "in-line",
         "quote": {"speaker_name": "Annie Brooks", "text": "Adjusted EPS came in at $0.31, in line with consensus"}},
        {"name": "Q4 2025 Revenue Guidance", "value": "$220M–$235M", "vs_expectations": "",
         "quote": {"speaker_name": "Annie Brooks", "text": "for Q4 we expect revenue in the range of $220 to $235 million"}},
        {"name": "FY2025 Revenue Guidance (raised)", "value": "$805M–$820M", "vs_expectations": "raised",
         "quote": {"speaker_name": "Annie Brooks", "text": "we are now guiding to revenue of $805 to $820 million, up from our prior $785 to $810 million range"}},
        {"name": "FY2025 Gross Margin Guidance (lowered)", "value": "41–42%", "vs_expectations": "lowered",
         "quote": {"speaker_name": "Annie Brooks", "text": "we are lowering our full year 2025 gross margin guidance by approximately 100 basis points to a range of 41 to 42 percent"}},
        {"name": "Cash + ST investments (Q3 end)", "value": "$1.04 billion", "vs_expectations": "",
         "quote": {"speaker_name": "Annie Brooks", "text": "We ended Q3 with $1.04 billion in cash, cash equivalents, and short-term investments"}},
    ]},
    "tone": {"tone": [
        {"speaker_name": "Tomas Vidal", "speaker_role": "ceo", "segment": "prepared remarks",
         "sentiment": "celebratory",
         "evidence": [
             {"text": "Q3 was our strongest revenue quarter in company history"},
             {"text": "we are deeply invested in the next generation of warehouse automation"}
         ],
         "note": "CEO leans hard into the revenue beat and teases a 2026 product line — confident, forward-looking"},
        {"speaker_name": "Annie Brooks", "speaker_role": "cfo", "segment": "prepared remarks — guidance",
         "sentiment": "measured",
         "evidence": [
             {"text": "we are lowering our full year 2025 gross margin guidance by approximately 100 basis points"},
             {"text": "reflecting the continued input-cost pressure we expect through the end of the year"}
         ],
         "note": "CFO doesn't try to sell the guide-down — flags it plainly and explains the drivers without spin"},
        {"speaker_name": "Annie Brooks", "speaker_role": "cfo", "segment": "Q&A — margins",
         "sentiment": "measured",
         "evidence": [
             {"text": "margin compression in 2025 is partly the cost of winning the deals we want to win for 2026 revenue"},
             {"text": "We're not, at this point, signaling sticky margin pressure into 2026, but we want to give you the data first"}
         ],
         "note": "CFO carefully avoids both committing to and dismissing 2026 margin pressure — gives the analyst the framework, not the answer"},
        {"speaker_name": "Tomas Vidal", "speaker_role": "ceo", "segment": "Q&A — new product",
         "sentiment": "confident",
         "evidence": [
             {"text": "the bulk of the development capex is already inside our 2025 OpEx and capitalized R&D"},
             {"text": "I appreciate you not pushing me harder than that today"}
         ],
         "note": "Deflects new-product detail to Q4 call but voluntarily defangs the capex concern — confident, prepared"},
    ]},
    "surprises": {"surprises": [
        {"headline": "FY2025 gross margin guidance lowered ~100 bps despite revenue raise",
         "kind": "guidance_revision", "significance": "medium",
         "evidence": [
             {"text": "we are lowering our full year 2025 gross margin guidance by approximately 100 basis points"}
         ],
         "rationale": "Unusual to raise revenue and lower margins on the same call — signals input-cost pressure is real, not transient mix"},
        {"headline": "Teased new product line announcement for Q4 call",
         "kind": "new_initiative", "significance": "medium",
         "evidence": [{"text": "We'll have more to say about a new product line on our Q4 call"}],
         "rationale": "Strategic frame coming next quarter — sets up Q4 as a guidance-plus-strategy event"},
        {"headline": "Backlog disclosure to begin with FY2025 10-K",
         "kind": "other", "significance": "low",
         "evidence": [{"text": "We expect to add backlog disclosure in our 10-K for fiscal 2025"}],
         "rationale": "Modeling signal: backlog will become a tracked metric, which the company is not yet ready to size on this call"},
    ]},
    "questions": {"questions": [
        {"turn_id": 4, "analyst_name": "Lou Patel", "affiliation": "Barclays",
         "question_summary": "Split the margin guide-down between input costs and mix; is input-cost pressure sticky into 2026?",
         "sharpness": 5, "answer_quality": "direct",
         "quote": {"text": "Walk us through whether this is primarily input cost or mix, and whether you think input costs are sticky into 2026"},
         "rationale": "Forces a 50/50 split answer and a specific 2026-stickiness framing — CFO gave the actual decomposition"},
        {"turn_id": 7, "analyst_name": "Yuki Tanaka", "affiliation": "Morgan Stanley",
         "question_summary": "On the teased new product — adjacency or new category, and does it require unforeseen capex?",
         "sharpness": 4, "answer_quality": "hedged",
         "quote": {"text": "is this an adjacency or a substantively new category, and does it require capex we haven't yet seen in the 2025 plan?"},
         "rationale": "Sharp two-part question. Got the capex piece answered (it's already in OpEx), got a soft 'adjacency' on the category question"},
        {"turn_id": 10, "analyst_name": "Karim El-Sayed", "affiliation": "Wedbush",
         "question_summary": "Will you give the backlog number?",
         "sharpness": 2, "answer_quality": "direct",
         "quote": {"text": "Are you giving us a number?"},
         "rationale": "Quick polite ask, declined cleanly; useful only because we learned backlog will be in the 10-K"},
    ]},
}


_RESPONSES = {
    "glow_q4_2025": GLOW,
    "helios_q3_2025": HELIOS,
}


def _detect_pass(system: str) -> str:
    if "extracting hard financial metrics" in system:
        return "metrics"
    if "assessing the TONE" in system or "tone of" in system.lower():
        return "tone"
    if "flagging SURPRISES" in system:
        return "surprises"
    if "rating the QUALITY of analyst questions" in system:
        return "questions"
    return "metrics"


def _detect_transcript(user_text: str) -> str:
    if "Glow Therapeutics" in user_text or "GlowDerm" in user_text:
        return "glow_q4_2025"
    if "Helios Robotics" in user_text or "palletizing" in user_text:
        return "helios_q3_2025"
    return ""


def make_mock_chat() -> Any:
    def fn(*, system: str, messages: list, **kwargs: Any) -> llm.ChatResult:
        user_text = ""
        if messages and isinstance(messages[0].get("content"), str):
            user_text = messages[0]["content"]
        transcript_id = _detect_transcript(user_text)
        pass_name = _detect_pass(system)
        bundle = _RESPONSES.get(transcript_id, GLOW)
        payload = bundle.get(pass_name, {})
        return llm.ChatResult(
            text=json.dumps(payload),
            usage=_usage(),
            cost_usd=0.015,
            model="claude-sonnet-4-6",
        )

    return fn
