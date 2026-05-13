"""Streamlit UI for Earnings Call Inspector."""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Force installed package to win over bare `earningscall/` namespace
# package at the repo root.
_SRC = Path(__file__).resolve().parents[2]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import streamlit as st

from earningscall import llm
from earningscall.extractor import (
    extract_analyst_questions,
    extract_metrics,
    extract_surprises,
    extract_tone,
)
from earningscall.mock import make_mock_chat
from earningscall.models import EarningsReport, load_transcript
from earningscall.verifier import verify_quotes


st.set_page_config(
    page_title="Earnings Call Inspector",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

GITHUB_URL = "https://github.com/anafe03/RL_agents/tree/main/earningscall"
TRANSCRIPTS_DIR = Path(__file__).resolve().parents[3] / "data" / "transcripts"


# --- sidebar ----------------------------------------------------------------

with st.sidebar:
    st.markdown("# 📊 Earnings Call Inspector")
    st.caption("Multi-pass extraction with citation enforcement.")
    st.markdown("---")

    mode = st.radio(
        "Mode",
        ["Demo (mock)", "Live (your API key)"],
        help="Demo runs the full 4-pass pipeline against canned responses in 2 seconds.",
    )
    api_key_input = ""
    selected_model = "claude-sonnet-4-6"
    if mode == "Live (your API key)":
        selected_model = st.selectbox(
            "Model",
            llm.SUPPORTED_MODELS,
            index=llm.SUPPORTED_MODELS.index("gpt-4o-mini"),
            help="gpt-4o-mini is ~100× cheaper for testing.",
        )
        if llm._is_openai_model(selected_model):
            api_key_input = st.text_input(
                "OPENAI_API_KEY", type="password", help="Used only in your browser session."
            )
            st.caption("~$0.0005 per transcript on gpt-4o-mini.")
        else:
            api_key_input = st.text_input(
                "ANTHROPIC_API_KEY", type="password", help="Used only in your browser session."
            )
            st.caption("~$0.05 per transcript on Sonnet 4.6.")

    st.markdown("---")
    transcript_files = sorted(TRANSCRIPTS_DIR.glob("*.yaml")) if TRANSCRIPTS_DIR.exists() else []
    if not transcript_files:
        st.error("No transcripts under data/transcripts/")
        st.stop()
    labels = {f"{load_transcript(p).company} {load_transcript(p).period}": p for p in transcript_files}
    selected_label = st.selectbox("Transcript", list(labels.keys()))
    transcript = load_transcript(labels[selected_label])

    st.markdown("---")
    st.warning(
        "⚠️ Bundled transcripts are synthetic. This is a portfolio project, not investment advice."
    )
    st.markdown(f"[GitHub repo]({GITHUB_URL})")


# --- header -----------------------------------------------------------------

st.markdown(f"# {transcript.company}")
st.caption(f"{transcript.period} · {transcript.call_date} · {transcript.sector} · {len(transcript.turns)} speaker turns")

with st.expander("Read transcript"):
    for t in transcript.turns:
        role_tag = f" *({t.speaker_role.value})*" if t.speaker_role.value != "unknown" else ""
        aff_tag = f" — {t.affiliation}" if t.affiliation else ""
        st.markdown(f"**{t.speaker_name}**{role_tag}{aff_tag}:")
        st.markdown(t.text)


# --- run --------------------------------------------------------------------

if "report" not in st.session_state:
    st.session_state.report = None
    st.session_state.report_transcript = None

can_run = True
if mode == "Live (your API key)" and not api_key_input:
    key_var = "OPENAI_API_KEY" if llm._is_openai_model(selected_model) else "ANTHROPIC_API_KEY"
    st.warning(f"Live mode needs an `{key_var}`. Enter one in the sidebar, or switch to Demo mode.")
    can_run = False

if st.button("📊 Inspect this call", type="primary", disabled=not can_run, use_container_width=True):
    if mode == "Demo (mock)":
        llm.set_chat_fn(make_mock_chat())
    else:
        if llm._is_openai_model(selected_model):
            os.environ["OPENAI_API_KEY"] = api_key_input
        else:
            os.environ["ANTHROPIC_API_KEY"] = api_key_input
        llm.reset_chat_fn()

    progress = st.progress(0.0, text="Pass 1/4 — extracting metrics...")
    try:
        metrics, c1 = extract_metrics(transcript)
        progress.progress(0.25, text="Pass 2/4 — assessing tone...")
        tone, c2 = extract_tone(transcript)
        progress.progress(0.5, text="Pass 3/4 — flagging surprises...")
        surprises, c3 = extract_surprises(transcript)
        progress.progress(0.75, text="Pass 4/4 — scoring analyst Q&A...")
        questions, c4 = extract_analyst_questions(transcript)
        progress.progress(0.9, text="Verifying citations...")
        citations = verify_quotes(transcript, metrics, tone, surprises, questions)
        progress.empty()
        report = EarningsReport(
            transcript_id=transcript.id,
            company=transcript.company,
            period=transcript.period,
            metrics=metrics, tone=tone, surprises=surprises, analyst_questions=questions,
            citation_results=citations, cost_usd=c1 + c2 + c3 + c4,
        )
        st.session_state.report = report
        st.session_state.report_transcript = transcript.id
        verified = sum(1 for c in citations if c.found)
        st.success(
            f"Done · {len(metrics)} metrics, {len(tone)} tone assessments, "
            f"{len(surprises)} surprises, {len(questions)} questions analyzed · "
            f"citations {verified}/{len(citations)} verified · ${report.cost_usd:.4f}"
        )
    finally:
        llm.reset_chat_fn()


# --- results ----------------------------------------------------------------

report: EarningsReport | None = st.session_state.report
if report is None or st.session_state.report_transcript != transcript.id:
    if report is not None:
        st.info("You switched transcripts — run a new inspection for this one.")
else:
    if not report.all_citations_verified:
        st.warning(
            f"⚠️ {report.n_unverified} quote(s) could not be verified as substrings of the transcript. "
            "These are flagged below — the model may have paraphrased rather than copied."
        )

    tab_metrics, tab_tone, tab_surprises, tab_qa, tab_citations = st.tabs(
        ["📈 Metrics", "🎭 Tone", "⚠️ Surprises", "💬 Analyst Q&A", "🔍 Citations"]
    )

    with tab_metrics:
        if not report.metrics:
            st.info("No metrics extracted.")
        for m in report.metrics:
            with st.container(border=True):
                cols = st.columns([2, 1, 1])
                cols[0].markdown(f"### {m.name}")
                cols[0].markdown(f"_{m.quote.speaker_name}_")
                cols[1].metric("Value", m.value)
                cols[2].metric("vs Expectations", m.vs_expectations or "—")
                st.markdown(f"> \"{m.quote.text}\"")

    with tab_tone:
        for t in report.tone:
            with st.container(border=True):
                st.markdown(f"### {t.speaker_name} — {t.segment}")
                st.markdown(f"**{t.sentiment.capitalize()}.** {t.note}")
                for q in t.evidence:
                    st.markdown(f"> \"{q.text}\"")

    with tab_surprises:
        if not report.surprises:
            st.info("No surprises flagged.")
        for s in report.surprises:
            border_color = {"high": "🔴", "medium": "🟡", "low": "🟢"}.get(s.significance, "⚪")
            with st.container(border=True):
                st.markdown(f"### {border_color} {s.headline}")
                st.caption(f"_{s.kind}_ · significance: **{s.significance.upper()}**")
                st.markdown(s.rationale)
                for q in s.evidence:
                    st.markdown(f"> \"{q.text}\"")

    with tab_qa:
        # ranked best-first
        ranked = sorted(report.analyst_questions, key=lambda x: -x.sharpness)
        for q in ranked:
            stars = "★" * q.sharpness + "☆" * (5 - q.sharpness)
            with st.container(border=True):
                st.markdown(f"### [{stars}] {q.analyst_name} ({q.affiliation})")
                st.caption(f"Answer: **{q.answer_quality}**")
                st.markdown(q.question_summary)
                st.markdown(f"> \"{q.quote.text}\"")
                st.markdown(f"_{q.rationale}_")

    with tab_citations:
        verified = sum(1 for c in report.citation_results if c.found)
        st.markdown(
            f"**{verified}/{len(report.citation_results)}** quotes verified as substrings of the transcript."
        )
        st.caption("Every quote in the report is checked against the transcript text. Hallucinated quotes are flagged here.")
        for c in report.citation_results:
            mark = "✅" if c.found else "❌"
            st.markdown(f"{mark} `{c.where_used}` — \"{c.quote_text[:120]}\"")
