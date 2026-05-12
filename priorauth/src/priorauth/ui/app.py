"""Streamlit UI for PriorAuth Assist.

Run locally:
    cd priorauth
    uv run streamlit run src/priorauth/ui/app.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Force installed package to win over bare `priorauth/` namespace package
# at the repo root (same gotcha simulacrum/festival had).
_SRC = Path(__file__).resolve().parents[2]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import streamlit as st

from priorauth import llm
from priorauth.assessor import assess_appeal
from priorauth.drafter import draft_appeal
from priorauth.mock import make_mock_chat
from priorauth.models import RubricVerdict, load_case, load_guideline_corpus
from priorauth.retriever import retrieve_relevant


st.set_page_config(
    page_title="PriorAuth Assist",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

GITHUB_URL = "https://github.com/anafe03/RL_agents/tree/main/priorauth"
CASES_DIR = Path(__file__).resolve().parents[3] / "data" / "cases"
GUIDELINES_DIR = Path(__file__).resolve().parents[3] / "data" / "guidelines"


# --- sidebar ----------------------------------------------------------------

with st.sidebar:
    st.markdown("# 🏥 PriorAuth Assist")
    st.caption("Cited appeal drafter for prior-auth denials.")
    st.markdown("---")

    mode = st.radio(
        "Mode",
        ["Demo (mock)", "Live (your API key)"],
        help="Demo runs the full retriever → drafter → assessor pipeline against canned responses in 2 seconds.",
    )
    api_key_input = ""
    if mode == "Live (your API key)":
        api_key_input = st.text_input(
            "ANTHROPIC_API_KEY", type="password", help="Used only in your browser session."
        )
        st.caption("A full case costs ~$0.06 (Sonnet 4.6 retriever+drafter + Opus 4.7 assessor).")

    st.markdown("---")
    case_files = sorted(CASES_DIR.glob("*.yaml")) if CASES_DIR.exists() else []
    if not case_files:
        st.error("No cases found under data/cases/")
        st.stop()
    case_labels = {load_case(p).title: p for p in case_files}
    selected_label = st.selectbox("Case", list(case_labels.keys()))
    case = load_case(case_labels[selected_label])

    st.markdown("---")
    st.warning(
        "⚠️ All cases are synthetic. This is a portfolio project, not a medical device or "
        "substitute for clinician judgment."
    )
    st.markdown(f"[GitHub repo]({GITHUB_URL})")


# --- header -----------------------------------------------------------------

st.markdown(f"# {case.title}")
st.caption(f"Payer: {case.denial.payer} · Service requested: {case.requested_service}")

# Side-by-side: patient context vs denial
col_patient, col_denial = st.columns(2)
with col_patient:
    with st.container(border=True):
        st.markdown("### Patient context")
        st.markdown(f"**Demographics:** {case.patient.demographics}")
        st.markdown("**Diagnoses:**")
        for d in case.patient.diagnoses:
            st.markdown(f"- {d}")
        st.markdown("**Medications tried:**")
        for m in case.patient.medications_tried:
            st.markdown(f"- {m}")
        if case.patient.relevant_labs:
            st.markdown("**Relevant labs:**")
            for k, v in case.patient.relevant_labs.items():
                st.markdown(f"- {k}: {v}")
        if case.patient.contraindications:
            st.markdown("**Contraindications:** " + "; ".join(case.patient.contraindications))
        if case.patient.red_flags:
            st.markdown("**Red flags:** " + "; ".join(case.patient.red_flags))
        with st.expander("Clinical history"):
            st.markdown(case.patient.clinical_history)

with col_denial:
    with st.container(border=True):
        st.markdown("### Denial letter")
        st.markdown(f"**Member ID:** `{case.denial.member_id}`")
        st.markdown(f"**Reason:** {case.denial.denial_reason}")
        st.markdown(f"**Cited policy:** {case.denial.cited_policy}")
        with st.expander("Letter text"):
            st.text(case.denial.raw_text)


# --- generate ---------------------------------------------------------------

st.markdown("---")

if "result" not in st.session_state:
    st.session_state.result = None
    st.session_state.result_case_id = None

can_run = True
if mode == "Live (your API key)" and not api_key_input:
    st.warning("Live mode needs an `ANTHROPIC_API_KEY`. Enter one in the sidebar, or switch to Demo mode.")
    can_run = False

if st.button("🏥 Draft cited appeal", type="primary", disabled=not can_run, use_container_width=True):
    if mode == "Demo (mock)":
        llm.set_chat_fn(make_mock_chat())
    else:
        os.environ["ANTHROPIC_API_KEY"] = api_key_input
        llm.reset_chat_fn()

    progress = st.progress(0.0, text="Retrieving relevant guidelines...")
    try:
        corpus = load_guideline_corpus(GUIDELINES_DIR)
        selected, retr_cost = retrieve_relevant(case, corpus)
        progress.progress(0.35, text=f"Selected {len(selected)} guidelines. Drafting appeal...")
        appeal, draft_cost = draft_appeal(case, selected)
        progress.progress(0.75, text="Independent reviewer assessing the draft...")
        assessment = assess_appeal(case, appeal, selected)
        progress.empty()
        st.session_state.result = {
            "appeal": appeal,
            "assessment": assessment,
            "guidelines": selected,
            "cost": retr_cost + draft_cost + assessment.cost_usd,
        }
        st.session_state.result_case_id = case.id
        st.success(
            f"Verdict: **{assessment.verdict.value.upper()}** · "
            f"cost ${st.session_state.result['cost']:.4f}"
        )
    finally:
        llm.reset_chat_fn()


# --- show result -----------------------------------------------------------

result = st.session_state.result
if result is None or st.session_state.result_case_id != case.id:
    if result is not None:
        st.info("You switched cases — draft a new appeal for this one.")
else:
    appeal = result["appeal"]
    assessment = result["assessment"]
    guidelines = result["guidelines"]

    st.markdown("---")
    st.markdown("## Independent assessment")

    verdict = assessment.verdict
    badge_color = {
        RubricVerdict.EXCELLENT: "🟢",
        RubricVerdict.STRONG: "🟢",
        RubricVerdict.MODERATE: "🟡",
        RubricVerdict.WEAK: "🔴",
    }.get(verdict, "⚪")
    st.markdown(f"### {badge_color} **{verdict.value.upper()}**")
    cols = st.columns(4)
    cols[0].metric("Addressed denial", "✓" if assessment.addressed_all_denial_criteria else "✗")
    cols[1].metric("Claims cited", "✓" if assessment.all_claims_cited else "✗")
    cols[2].metric("Facts accurate", "✓" if assessment.patient_facts_accurate else "✗")
    cols[3].metric("Clear ask", "✓" if assessment.has_clear_ask else "✗")
    st.markdown(f"> {assessment.reasoning}")
    if assessment.weak_points:
        st.warning("**Weak points to review before sending:**")
        for wp in assessment.weak_points:
            st.markdown(f"- {wp}")

    st.markdown("---")
    st.markdown("## Drafted appeal letter")

    with st.container(border=True):
        st.markdown(f"**Opening.** {appeal.opening}")
        st.markdown("")
        st.markdown("**Clinical rationale:**")
        for i, p in enumerate(appeal.clinical_rationale, 1):
            st.markdown(f"{i}. {p}")
        st.markdown("**Citations:**")
        for c in appeal.citations:
            with st.expander(f"`{c.guideline_id}` — {c.claim}"):
                st.markdown(f"> \"{c.quoted_excerpt}\"")
        st.markdown("")
        st.markdown(f"**Closing.** {appeal.closing}")
