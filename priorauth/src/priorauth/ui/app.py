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
from priorauth.benchmark import load_golden, run_benchmark
from priorauth.deidentify import Deidentifier
from priorauth.drafter import draft_appeal
from priorauth.mock import make_mock_chat
from priorauth.models import RubricVerdict, load_case, load_guideline_corpus
from priorauth.retrievers import REGISTRY, get_retriever


st.set_page_config(
    page_title="PriorAuth Assist",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Clinical "case file" aesthetic — clean whites, medical-blue accents,
# manila-folder header strip. Different from the festival/earnings/octagon
# vibes; reads like a clinic workflow tool, not a SaaS dashboard.
st.markdown(
    """
    <style>
    .stApp {
        background-color: #f8fafc;
    }
    .case-folder {
        background: #fbf6e8;
        border: 1px solid #e0d5b0;
        border-top: 4px solid #1e6091;
        padding: 1.2rem 1.5rem;
        margin-bottom: 1.5rem;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
    }
    .case-folder .case-label {
        color: #1e6091;
        font-size: 0.7rem;
        letter-spacing: 0.25em;
        font-weight: 700;
        text-transform: uppercase;
    }
    .case-folder .case-title {
        color: #0f172a;
        font-size: 1.6rem;
        font-weight: 700;
        margin-top: 0.25rem;
    }
    .case-folder .case-meta {
        color: #475569;
        font-size: 0.88rem;
        margin-top: 0.4rem;
    }
    /* Section dividers feel like chart sections */
    h2 {
        color: #1e6091 !important;
        border-bottom: 1px solid #cbd5e1;
        padding-bottom: 0.3rem;
    }
    /* Letterhead — formal physician appeal letter aesthetic. The drafted
       appeal renders inside a cream paper frame with a navy top stripe,
       Georgia serif body, and a hand-style signature block at the bottom.
       Citations stay outside the letterhead as expandable evidence. */
    .letterhead {
        background: #fdfcf7 !important;
        border: 1px solid #c9c2b0;
        border-top: 5px solid #1e6091;
        padding: 2.5rem 3rem 2rem 3rem;
        margin: 0.5rem 0 1rem 0;
        font-family: Georgia, 'Times New Roman', serif !important;
        color: #1a1a1a !important;
        line-height: 1.6;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.06);
    }
    .letterhead * {
        color: #1a1a1a !important;
        font-family: Georgia, 'Times New Roman', serif !important;
    }
    .letterhead .lh-header {
        border-bottom: 1px solid #d4cdb8;
        padding-bottom: 1rem;
        margin-bottom: 1.4rem;
    }
    .letterhead .lh-firm {
        font-size: 1.15rem;
        font-weight: 700;
        color: #1e6091 !important;
        letter-spacing: 0.04em;
    }
    .letterhead .lh-firm-sub {
        font-size: 0.82rem;
        color: #6b6354 !important;
        font-style: italic;
        margin-top: 0.1rem;
    }
    .letterhead .lh-meta {
        font-size: 0.9rem;
        color: #333 !important;
        margin-top: 0.9rem;
    }
    .letterhead .lh-meta b {
        color: #1e6091 !important;
    }
    .letterhead .lh-re {
        font-weight: 700;
        margin: 1.1rem 0 0.7rem 0;
        text-decoration: underline;
    }
    .letterhead .lh-salutation {
        margin-bottom: 0.8rem;
    }
    .letterhead .lh-body p {
        text-indent: 1.5em;
        margin-bottom: 0.9rem;
        text-align: justify;
        color: #1a1a1a !important;
    }
    .letterhead .lh-closing {
        margin-top: 1.4rem;
    }
    .letterhead .lh-signature {
        margin-top: 0.4rem;
        font-family: 'Brush Script MT', 'Lucida Handwriting', cursive !important;
        font-size: 1.8rem;
        color: #1e3a5f !important;
        letter-spacing: 0.02em;
    }
    .letterhead .lh-signature-line {
        border-top: 1px solid #2a2a2a;
        width: 220px;
        margin-top: 0.3rem;
        padding-top: 0.3rem;
        font-size: 0.85rem;
        color: #555 !important;
        font-style: italic;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

GITHUB_URL = "https://github.com/anafe03/RL_agents/tree/main/priorauth"
CASES_DIR = Path(__file__).resolve().parents[3] / "data" / "cases"
GUIDELINES_DIR = Path(__file__).resolve().parents[3] / "data" / "guidelines"
GOLDEN_PATH = Path(__file__).resolve().parents[3] / "data" / "golden.yaml"


@st.cache_resource
def get_cached_retriever(name: str, corpus_version: int = 1):
    """Cache retriever instances across runs — esp. Chroma which downloads a model on first init."""
    r = get_retriever(name)
    corpus = load_guideline_corpus(GUIDELINES_DIR)
    r.index(corpus)
    return r


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
    selected_model = "claude-sonnet-4-6"  # default
    if mode == "Live (your API key)":
        selected_model = st.selectbox(
            "Model",
            llm.SUPPORTED_MODELS,
            index=llm.SUPPORTED_MODELS.index("gpt-4o-mini"),
            help=(
                "Provider auto-detected by prefix. **gpt-4o-mini** is the cheapest "
                "(~$0.0005 per case). **claude-sonnet-4-6** is the default for "
                "documentation. Claude-only features (prompt caching, tool use) "
                "are skipped on GPT models."
            ),
        )
        if llm._is_openai_model(selected_model):
            api_key_input = st.text_input(
                "OPENAI_API_KEY", type="password", help="Used only in your browser session."
            )
            st.caption("A full case costs ~$0.0005 on gpt-4o-mini (≈100× cheaper than Sonnet).")
        else:
            api_key_input = st.text_input(
                "ANTHROPIC_API_KEY", type="password", help="Used only in your browser session."
            )
            st.caption("A full case costs ~$0.06 on Sonnet 4.6 + Opus 4.7 assessor.")

    st.markdown("---")
    case_files = sorted(CASES_DIR.glob("*.yaml")) if CASES_DIR.exists() else []
    if not case_files:
        st.error("No cases found under data/cases/")
        st.stop()
    case_labels = {load_case(p).title: p for p in case_files}
    selected_label = st.selectbox("Case", list(case_labels.keys()))
    case = load_case(case_labels[selected_label])

    st.markdown("---")
    retriever_choice = st.selectbox(
        "Retriever",
        sorted(REGISTRY.keys()),
        index=sorted(REGISTRY.keys()).index("llm_judged"),
        help=(
            "Which retrieval backend feeds the drafter. "
            "**bm25** = keyword search (fastest, no API). "
            "**chroma_minilm** = dense vector (ONNX-MiniLM, no API). "
            "**llm_judged** = zero-shot LLM picks (best precision, requires API key for Live mode)."
        ),
    )

    st.markdown("---")
    deidentify_enabled = st.checkbox(
        "De-identify PHI before LLM calls",
        value=False,
        help=(
            "Scrub member IDs, payer names, phone numbers, emails, addresses, dates, and SSNs "
            "from the case before sending to the LLM API. PHI is replaced with stable tokens "
            "(e.g. `[MEMBER_ID_1]`) and a reverse mapping is kept locally. HIPAA-shaped feature; "
            "not by itself a compliance guarantee."
        ),
    )

    st.markdown("---")
    st.warning(
        "⚠️ All cases are synthetic. This is a portfolio project, not a medical device or "
        "substitute for clinician judgment."
    )
    st.markdown(f"[GitHub repo]({GITHUB_URL})")


# --- header -----------------------------------------------------------------

st.markdown(
    f"""
    <div class="case-folder">
        <div class="case-label">CASE FILE  ·  {case.id.upper()}</div>
        <div class="case-title">{case.title}</div>
        <div class="case-meta"><b>Payer:</b> {case.denial.payer}  ·  <b>Service requested:</b> {case.requested_service}</div>
    </div>
    """,
    unsafe_allow_html=True,
)

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
    key_var = "OPENAI_API_KEY" if llm._is_openai_model(selected_model) else "ANTHROPIC_API_KEY"
    st.warning(f"Live mode needs an `{key_var}`. Enter one in the sidebar, or switch to Demo mode.")
    can_run = False

if st.button("🏥 Draft cited appeal", type="primary", disabled=not can_run, use_container_width=True):
    if mode == "Demo (mock)":
        llm.set_chat_fn(make_mock_chat())
    else:
        if llm._is_openai_model(selected_model):
            os.environ["OPENAI_API_KEY"] = api_key_input
        else:
            os.environ["ANTHROPIC_API_KEY"] = api_key_input
        llm.reset_chat_fn()

    progress = st.progress(0.0, text="Retrieving relevant guidelines...")
    deid_mapping = None
    try:
        corpus = load_guideline_corpus(GUIDELINES_DIR)
        # If de-id is on, scrub the case before any LLM call sees it.
        if deidentify_enabled:
            case_for_llm, deid_mapping = Deidentifier().deidentify_case(case)
        else:
            case_for_llm = case
        # Use selected retriever. For Chroma, the cache_resource decorator
        # avoids re-downloading the ONNX model across runs.
        if retriever_choice == "llm_judged":
            r = get_retriever("llm_judged")
            r.index(corpus)
        else:
            r = get_cached_retriever(retriever_choice)
        selected = r.retrieve(case_for_llm, k=5)
        retr_cost = r.cost_usd
        progress.progress(0.35, text=f"Selected {len(selected)} guidelines. Drafting appeal...")
        appeal, draft_cost = draft_appeal(case_for_llm, selected)
        progress.progress(0.75, text="Independent reviewer assessing the draft...")
        assessment = assess_appeal(case_for_llm, appeal, selected)
        progress.empty()
        st.session_state.result = {
            "appeal": appeal,
            "assessment": assessment,
            "guidelines": selected,
            "deid_mapping": deid_mapping,
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
    deid_mapping = result.get("deid_mapping")

    # If the run used de-identification, show what was scrubbed before LLM calls.
    if deid_mapping is not None and deid_mapping.tokens:
        st.markdown("---")
        st.markdown("## 🔒 De-identification — what got scrubbed before any LLM saw the data")
        st.caption(
            f"{len(deid_mapping.tokens)} PHI tokens were replaced. The LLM was sent only the "
            "tokenized version. Mapping below stays local — it does not leave your browser session."
        )
        with st.expander(f"Mapping ({len(deid_mapping.tokens)} entries)"):
            mapping_rows = []
            for tok, orig in deid_mapping.tokens.items():
                category = tok.split("_")[0].lstrip("[")
                # Truncate originals for display so addresses / narrative don't explode the UI
                display = orig if len(orig) <= 80 else orig[:77] + "..."
                mapping_rows.append({"Token": tok, "Category": category, "Original (masked)": display})
            try:
                import pandas as pd
                st.dataframe(pd.DataFrame(mapping_rows), hide_index=True, use_container_width=True)
            except Exception:
                st.table(mapping_rows)

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

    import html as _html
    from datetime import date as _date

    today_str = _date.today().strftime("%B %d, %Y")
    body_paragraphs = "".join(
        f"<p>{_html.escape(p)}</p>" for p in appeal.clinical_rationale
    )
    st.markdown(
        f"""
        <div class="letterhead">
            <div class="lh-header">
                <div class="lh-firm">OFFICE OF THE TREATING PHYSICIAN</div>
                <div class="lh-firm-sub">Medical necessity appeal · prepared with PriorAuth Assist</div>
                <div class="lh-meta">
                    {today_str}<br>
                    <b>To:</b> Appeals Department, {_html.escape(case.denial.payer)}<br>
                    <b>Member ID:</b> {_html.escape(case.denial.member_id)}
                </div>
                <div class="lh-re">RE: Appeal of denial — {_html.escape(case.requested_service)}</div>
            </div>
            <div class="lh-salutation">To the Appeals Reviewer,</div>
            <div class="lh-body">
                <p>{_html.escape(appeal.opening)}</p>
                {body_paragraphs}
                <p>{_html.escape(appeal.closing)}</p>
            </div>
            <div class="lh-closing">
                Respectfully,
                <div class="lh-signature">Treating Physician</div>
                <div class="lh-signature-line">Signature on file</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("**Source citations** — each clinical claim above traces to a verifiable quote:")
    for c in appeal.citations:
        with st.expander(f"`{c.guideline_id}` — {c.claim}"):
            st.markdown(f"> \"{c.quoted_excerpt}\"")


# --- benchmark section ------------------------------------------------------

st.markdown("---")
st.markdown("## 🏁 Retriever benchmark")
st.caption(
    "Compare the three retrievers on the bundled cases. Same corpus, same queries — "
    "different strategies (keyword vs dense vector vs zero-shot LLM)."
)

bench_col1, bench_col2 = st.columns([3, 1])
with bench_col1:
    bench_options = [n for n in sorted(REGISTRY.keys()) if n != "llm_judged"]
    if mode == "Live (your API key)" and api_key_input and not llm._is_openai_model(selected_model):
        bench_options.append("llm_judged")
    elif mode == "Demo (mock)":
        bench_options.append("llm_judged")
    bench_choices = st.multiselect(
        "Retrievers to benchmark",
        bench_options,
        default=bench_options,
        help="`llm_judged` requires Demo mode or a Live API key.",
    )
with bench_col2:
    st.markdown("")
    st.markdown("")
    run_bench = st.button("Run benchmark", use_container_width=True)

if run_bench and bench_choices:
    # Set chat fn based on mode for any llm_judged-style retrievers
    if mode == "Demo (mock)":
        llm.set_chat_fn(make_mock_chat())
    else:
        if api_key_input:
            os.environ["ANTHROPIC_API_KEY"] = api_key_input
        llm.reset_chat_fn()

    try:
        corpus = load_guideline_corpus(GUIDELINES_DIR)
        cases = [load_case(p) for p in sorted(CASES_DIR.glob("*.yaml"))]
        golden = load_golden(GOLDEN_PATH)
        # Build retrievers, using cached for the local ones (avoids re-loading ONNX)
        bench_retrievers = []
        for name in bench_choices:
            if name == "llm_judged":
                r = get_retriever("llm_judged")
                r.index(corpus)
            else:
                r = get_cached_retriever(name)
            bench_retrievers.append(r)
        with st.spinner("Running benchmark..."):
            report = run_benchmark(bench_retrievers, cases, corpus, golden, k=5)
        st.session_state.bench_report = report
    finally:
        llm.reset_chat_fn()

if st.session_state.get("bench_report") is not None:
    report = st.session_state.bench_report
    # Aggregate table
    rows = []
    for name, agg in report.by_retriever().items():
        rows.append({
            "Retriever": name,
            "Precision@5": round(agg["precision_at_k"], 2),
            "Recall@5": round(agg["recall_at_k"], 2),
            "Avg latency (ms)": round(agg["latency_ms"], 1),
            "Total cost (USD)": round(agg["cost_usd"], 4),
        })
    try:
        import pandas as pd
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
    except Exception:
        st.table(rows)

    with st.expander("Per-case detail"):
        for r in report.results:
            mark = "✅" if r.recall_at_k == 1.0 else ("⚠️" if r.recall_at_k > 0 else "❌")
            st.markdown(
                f"{mark} **{r.retriever_name}** · `{r.case_id}` — "
                f"P={r.precision_at_k:.2f}, R={r.recall_at_k:.2f}, "
                f"lat={r.latency_ms:.0f}ms · retrieved: `{', '.join(r.retrieved_ids[:5])}`"
            )
