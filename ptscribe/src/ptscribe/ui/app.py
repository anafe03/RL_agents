"""Streamlit UI for ptscribe.

Two views in the same app:
  1. Scribe + Eval — paste a transcript, see the structured SOAP note, the
     hallucination findings, the completeness score, and the judge result.
  2. Monitoring — every run is logged to SQLite; this tab shows the
     aggregate stats (p50/p95 latency, avg cost, hallucination rate) and
     a table of recent runs.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[2]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import pandas as pd
import streamlit as st

from ptscribe import llm
from ptscribe.eval import run_eval
from ptscribe.mock import make_mock_chat
from ptscribe.models import CheckStatus
from ptscribe.monitoring import aggregate_stats, log_from_eval, recent_runs
from ptscribe.scribe import extract_soap

st.set_page_config(page_title="ptscribe", page_icon="🩺", layout="wide",
                   initial_sidebar_state="expanded")

ACCENT = "#FF5757"  # Prompt Health brand coral
st.markdown(
    f"""
    <style>
    /* Force LIGHT theme — repo-wide .streamlit/config.toml defaults to
       dark (for octagon). ptscribe brand-themes to Prompt's coral palette
       on white, so the Streamlit container colors are hard-overridden. */
    .stApp,
    [data-testid="stAppViewContainer"],
    [data-testid="stMain"],
    [data-testid="stMainBlockContainer"],
    [data-testid="stHeader"] {{
        background-color: #fafafa !important;
        color: #1a1a1a !important;
    }}
    [data-testid="stSidebar"],
    [data-testid="stSidebar"] > div {{
        background-color: #f1f1f1 !important;
    }}
    [data-testid="stSidebar"] *,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] span,
    [data-testid="stSidebar"] div {{
        color: #1a1a1a !important;
    }}
    .stApp p, .stApp li, .stApp label,
    .stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5, .stApp h6,
    [data-testid="stMarkdownContainer"] p,
    [data-testid="stMarkdownContainer"] li,
    [data-testid="stMarkdownContainer"] h1,
    [data-testid="stMarkdownContainer"] h2,
    [data-testid="stMarkdownContainer"] h3 {{
        color: #1a1a1a !important;
    }}
    [data-testid="stVerticalBlockBorderWrapper"] {{
        background-color: #ffffff !important;
        border-color: #e5e5e5 !important;
    }}
    code {{
        background-color: #f3f3f3 !important;
        color: #1a1a1a !important;
    }}

    /* ptscribe — Prompt coral accent */
    .pt-header {{
        background: linear-gradient(135deg, #ffffff 0%, #fff2f2 100%);
        border: 1px solid #ffd6d6;
        border-left: 4px solid {ACCENT};
        border-radius: 8px;
        padding: 1.05rem 1.4rem;
        margin-bottom: 0.6rem;
    }}
    .pt-header .title {{
        font-size: 1.7rem; font-weight: 800; color: {ACCENT};
        font-family: -apple-system, "Segoe UI", sans-serif; letter-spacing: 0.03em;
    }}
    .pt-header .sub {{ color: #6b6b6b; font-size: 0.87rem; margin-top: 0.2rem; }}
    .pt-card {{
        background: #ffffff; border: 1px solid #e5e5e5; border-radius: 8px;
        padding: 0.8rem 1.05rem; margin-bottom: 0.55rem;
        color: #1a1a1a; font-family: "SF Mono", Menlo, monospace; font-size: 0.84rem;
    }}
    .pt-chip {{
        display: inline-block; font-family: "SF Mono", Menlo, monospace;
        font-size: 0.7rem; padding: 0.15rem 0.5rem; border-radius: 4px;
        background: #f3f3f3; color: #4b4b4b; margin-right: 0.3rem;
        border: 1px solid #e5e5e5;
    }}
    /* Primary buttons (Streamlit) tinted with the brand coral. */
    .stButton button[kind="primary"], .stButton button[data-baseweb="button"][kind="primary"] {{
        background-color: {ACCENT} !important;
        color: #ffffff !important;
        border: 1px solid {ACCENT} !important;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

GITHUB_URL = "https://github.com/anafe03/RL_agents/tree/main/ptscribe"
TRANSCRIPTS_DIR = Path(__file__).resolve().parents[3] / "data" / "transcripts"

_STATUS_COLOR = {
    CheckStatus.PASS: ("#16a34a", "PASS"),
    CheckStatus.WARN: ("#d97706", "WARN"),
    CheckStatus.FAIL: ("#dc2626", "FAIL"),
}


# --- session state ----------------------------------------------------------

if "result" not in st.session_state:
    st.session_state.result = None  # latest scribe+eval+chat tuple
if "transcript_text" not in st.session_state:
    # Default to the first bundled transcript
    files = sorted(TRANSCRIPTS_DIR.glob("*.txt"))
    st.session_state.transcript_text = files[0].read_text() if files else ""
    st.session_state.transcript_id = files[0].stem if files else "pasted"


# --- sidebar ----------------------------------------------------------------

with st.sidebar:
    st.markdown("# 🩺 ptscribe")
    st.caption("SOAP-note scribe for PT / OT / ST with a full eval harness.")
    st.markdown("---")

    mode = st.radio("Mode", ["Demo", "Live (your API key)"],
                    help="Demo uses canned responses for the bundled transcripts. "
                         "Live uses a real LLM and logs every run.")
    model = "claude-sonnet-4-6"
    api_key = ""
    if mode.startswith("Live"):
        model = st.selectbox(
            "Model", llm.SUPPORTED_MODELS,
            index=llm.SUPPORTED_MODELS.index("gpt-4o-mini"),
        )
        key_var = "OPENAI_API_KEY" if llm._is_openai_model(model) else "ANTHROPIC_API_KEY"
        api_key = st.text_input(key_var, type="password")

    use_judge = st.checkbox(
        "Run LLM-as-judge on narrative",
        value=False,
        help="A separate LLM call scores faithfulness + completeness + clinical voice "
             "of the narrative sections.",
    )

    st.markdown("---")
    transcript_files = sorted(TRANSCRIPTS_DIR.glob("*.txt"))
    labels = {p.stem: p for p in transcript_files}
    pick = st.selectbox("Bundled transcript", list(labels.keys()))
    if st.button("Load bundled transcript", use_container_width=True):
        st.session_state.transcript_text = labels[pick].read_text()
        st.session_state.transcript_id = pick
        st.session_state.result = None
        st.rerun()

    st.markdown("---")
    st.warning(
        "⚠️ Synthetic transcripts only. A portfolio project, not a clinical tool — "
        "thresholds are illustrative, not validated."
    )
    st.markdown(f"[GitHub repo]({GITHUB_URL})")


# --- header -----------------------------------------------------------------

st.markdown(
    """
    <div class="pt-header">
        <div class="title">ptscribe</div>
        <div class="sub">ambient SOAP-note scribe for PT/OT/ST — structured
        extraction, hallucination cross-check against the transcript, and a
        run-by-run monitoring dashboard</div>
    </div>
    """,
    unsafe_allow_html=True,
)


# --- tabs -------------------------------------------------------------------

tab_scribe, tab_monitoring = st.tabs(["📝 Scribe + Eval", "📊 Monitoring"])


with tab_scribe:
    st.markdown("##### Transcript")
    transcript = st.text_area(
        "Clinician transcript",
        value=st.session_state.transcript_text,
        key="transcript_text",
        height=240,
        label_visibility="collapsed",
    )

    can_run = True
    if mode.startswith("Live") and not api_key:
        st.warning("Enter an API key in the sidebar, or switch to Demo mode.")
        can_run = False

    if st.button("▶ Scribe + run eval", type="primary",
                 use_container_width=True, disabled=not can_run):
        if mode == "Demo":
            llm.set_chat_fn(make_mock_chat())
        else:
            key_var = "OPENAI_API_KEY" if llm._is_openai_model(model) else "ANTHROPIC_API_KEY"
            os.environ[key_var] = api_key
            llm.reset_chat_fn()
        try:
            note, chat_result = extract_soap(transcript, model=model)
            eval_result, judge_cost = run_eval(
                transcript, note,
                transcript_id=st.session_state.transcript_id,
                use_judge=use_judge and mode.startswith("Live"),
                judge_model=model,
            )
            log_from_eval(
                transcript_id=st.session_state.transcript_id,
                model=chat_result.model,
                mode="live" if mode.startswith("Live") else "demo",
                cost_usd=chat_result.cost_usd + judge_cost,
                latency_ms=chat_result.latency_ms,
                input_chars=len(transcript),
                output_chars=len(chat_result.text),
                eval_result=eval_result,
            )
            st.session_state.result = {
                "note": note,
                "eval": eval_result,
                "chat": chat_result,
                "judge_cost": judge_cost,
            }
        except Exception as exc:  # noqa: BLE001
            st.error(f"Run failed — {type(exc).__name__}: {exc}")
            st.session_state.result = None
        finally:
            llm.reset_chat_fn()

    result = st.session_state.result
    if result is None:
        st.info("Pick a transcript and click **Scribe + run eval** to begin.")
    else:
        note = result["note"]
        ev = result["eval"]
        chat_res = result["chat"]
        total_cost = chat_res.cost_usd + result["judge_cost"]

        color, label = _STATUS_COLOR[ev.overall]
        st.markdown(
            f"<div class='pt-card' style='border-left:4px solid {color};"
            f"font-family:-apple-system,sans-serif'>"
            f"<span class='pt-chip' style='background:{color};color:#ffffff;"
            f"border:1px solid {color};font-weight:700'>{label}</span> "
            f"<b>{note.patient_label}</b><br>"
            f"<span class='pt-chip'>visit: {note.visit_type or '—'}</span>"
            f"<span class='pt-chip'>discipline: {note.discipline}</span>"
            f"<span class='pt-chip'>completeness {ev.completeness_score:.2f}</span>"
            f"<span class='pt-chip'>hallucinations {len(ev.hallucination_findings)}</span>"
            + (f"<span class='pt-chip'>judge {ev.judge_score:.2f}</span>"
               if ev.judge_score is not None else "")
            + f"<span class='pt-chip'>${total_cost:.5f}</span>"
            f"<span class='pt-chip'>{chat_res.latency_ms}ms</span>"
            f"</div>",
            unsafe_allow_html=True,
        )

        soap_col, eval_col = st.columns([3, 2])

        with soap_col:
            st.markdown("##### Structured SOAP")

            st.markdown("**S — Subjective**")
            s = note.subjective
            if s.chief_complaint:
                st.markdown(f"_Chief complaint:_ {s.chief_complaint}")
            if s.history:
                st.markdown(f"_History:_ {s.history}")
            if s.pain:
                st.markdown("_Pain:_")
                for p in s.pain:
                    st.markdown(f"- {p.location}: **{p.score}/10** ({p.when or 'unspecified'})")
            if s.functional_limitations:
                st.markdown("_Functional limitations:_")
                for f in s.functional_limitations:
                    st.markdown(f"- {f}")
            if s.patient_goals:
                st.markdown("_Patient goals:_")
                for g in s.patient_goals:
                    st.markdown(f"- {g}")

            st.markdown("**O — Objective**")
            o = note.objective
            if o.rom:
                rom_df = pd.DataFrame([
                    {"Joint": r.joint, "Side": r.side.value, "Degrees": r.degrees,
                     "Note": r.description}
                    for r in o.rom
                ])
                st.dataframe(rom_df, hide_index=True, use_container_width=True)
            if o.strength:
                mmt_df = pd.DataFrame([
                    {"Muscle": m.muscle_group, "Side": m.side.value,
                     "Grade": f"{m.grade:g}/5", "Note": m.note}
                    for m in o.strength
                ])
                st.dataframe(mmt_df, hide_index=True, use_container_width=True)
            if o.special_tests:
                st.markdown("_Special tests:_")
                for t in o.special_tests:
                    st.markdown(f"- {t}")
            if o.observations:
                st.markdown("_Observations:_")
                for obs in o.observations:
                    st.markdown(f"- {obs}")

            st.markdown("**A — Assessment**")
            a = note.assessment
            if a.summary:
                st.markdown(a.summary)
            if a.progress:
                st.markdown(f"_Progress:_ {a.progress}")
            if a.impairments:
                st.markdown("_Impairments:_ " + "; ".join(a.impairments))

            st.markdown("**P — Plan**")
            p = note.plan
            if p.interventions_today:
                st.markdown("_Today:_")
                for i in p.interventions_today:
                    st.markdown(f"- {i}")
            if p.home_exercise_program:
                st.markdown("_Home exercise program:_")
                for ex in p.home_exercise_program:
                    pieces = [f"**{ex.name}**"]
                    if ex.sets and ex.reps:
                        pieces.append(f"{ex.sets} × {ex.reps}")
                    elif ex.sets:
                        pieces.append(f"{ex.sets} sets")
                    if ex.hold_seconds:
                        pieces.append(f"hold {ex.hold_seconds}s")
                    if ex.sessions_per_day:
                        pieces.append(f"{ex.sessions_per_day}/day")
                    if ex.note:
                        pieces.append(f"({ex.note})")
                    st.markdown(f"- " + " · ".join(pieces))
            if p.next_visit:
                st.markdown(f"_Next visit:_ {p.next_visit}")

        with eval_col:
            st.markdown("##### Eval — every claim cross-checked")
            st.caption(
                "Each ROM, MMT, pain score, exercise, and special test in the "
                "note is searched for in the transcript. Anything that doesn't "
                "ground is flagged for human review."
            )

            if not ev.hallucination_findings:
                st.success("✓ No ungrounded claims found — every datum in the note traces to the transcript.")
            else:
                st.warning(f"⚠ {len(ev.hallucination_findings)} ungrounded "
                           f"claim{'' if len(ev.hallucination_findings) == 1 else 's'}.")
                for f in ev.hallucination_findings:
                    severity = "#dc2626" if f.confidence >= 0.8 else "#d97706"
                    st.markdown(
                        f"<div class='pt-card' style='border-left:3px solid {severity}'>"
                        f"<span class='pt-chip'>{f.field_path}</span><br>"
                        f"<b style='color:#1a1a1a'>{f.claim}</b><br>"
                        f"<span style='color:#6b6b6b;font-size:0.78rem'>"
                        f"confidence {f.confidence:.2f} — {f.note}</span></div>",
                        unsafe_allow_html=True,
                    )

            if ev.judge_score is not None:
                judge_color = "#16a34a" if ev.judge_score >= 0.7 else "#d97706"
                st.markdown("##### LLM-as-judge")
                st.markdown(
                    f"<div class='pt-card' style='border-left:3px solid {judge_color}'>"
                    f"<b style='color:{judge_color};font-size:1.3rem'>"
                    f"{ev.judge_score:.2f}</b><br>"
                    f"<span style='color:#1a1a1a;font-size:0.84rem'>"
                    f"{ev.judge_reasoning}</span></div>",
                    unsafe_allow_html=True,
                )


with tab_monitoring:
    st.markdown("##### Run monitoring")
    st.caption(
        "Every scribe call — demo or live — is logged to the local SQLite DB. "
        "Aggregate stats and recent runs below. Pointing at Postgres for "
        "production is a one-line URL change."
    )

    stats = aggregate_stats()
    if stats.get("total_runs", 0) == 0:
        st.info("No runs logged yet — run a scribe in the other tab to populate this.")
    else:
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total runs", stats["total_runs"])
        m2.metric("p50 / p95 latency",
                  f"{stats['p50_latency_ms']} / {stats['p95_latency_ms']} ms")
        m3.metric("Avg cost / run", f"${stats['avg_cost_usd']:.5f}")
        m4.metric("Hallucinations / run", stats["hallucinations_per_run"])

        st.markdown("##### Recent runs")
        runs = recent_runs(limit=30)
        if runs:
            df = pd.DataFrame([
                {
                    "when": r.timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                    "transcript": r.transcript_id,
                    "model": r.model,
                    "mode": r.mode,
                    "cost $": round(r.cost_usd, 5),
                    "latency ms": r.latency_ms,
                    "complete": round(r.completeness_score, 2),
                    "halluc.": r.hallucination_count,
                    "judge": (None if r.judge_score is None
                              else round(r.judge_score, 2)),
                }
                for r in runs
            ])
            st.dataframe(df, hide_index=True, use_container_width=True)

st.divider()
st.caption(
    "ptscribe — designed and built as a portfolio project. Synthetic transcripts; "
    "illustrative thresholds; not a validated clinical tool."
)
