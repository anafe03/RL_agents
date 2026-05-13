"""Streamlit UI for Octagon's Red Cell.

Run locally:
    uv run streamlit run src/octagon/ui/app.py

Hosted on Streamlit Cloud — see README for the link.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import streamlit as st

from octagon import llm
from octagon.defenders import get_defender, REGISTRY
from octagon.mock import make_mock_chat
from octagon.models import AttackResult, AuditReport, Outcome
from octagon.runner import load_attacks, run_audit

# --- page config ------------------------------------------------------------

st.set_page_config(
    page_title="Octagon · Red Cell",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Red Cell ops aesthetic — dark, red/amber accents, "incident report" look.
# Attack outcomes color-coded; defender stats up top in a status panel.
st.markdown(
    """
    <style>
    .stApp {
        background-color: #0d0d0f;
    }
    .block-container {
        padding-top: 1.5rem;
        max-width: 1300px;
    }
    .redcell-hero {
        background: linear-gradient(135deg, #1a0a0a 0%, #2d1010 50%, #1a0a0a 100%);
        border: 1px solid #5a1a1a;
        border-left: 4px solid #ff3838;
        padding: 1.2rem 1.5rem;
        margin-bottom: 1.5rem;
        font-family: 'JetBrains Mono', 'SF Mono', Menlo, monospace;
    }
    .redcell-hero .classification {
        color: #ff3838;
        font-size: 0.78rem;
        letter-spacing: 0.3em;
        font-weight: 700;
    }
    .redcell-hero h1 {
        color: #ffffff;
        font-size: 2rem;
        font-weight: 900;
        margin: 0.3rem 0 0 0;
        letter-spacing: -0.5px;
    }
    .redcell-hero .target {
        color: #ffa07a;
        font-size: 0.9rem;
        margin-top: 0.3rem;
    }
    /* Defender status strip */
    .status-strip {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 1px;
        background: #2a1010;
        border: 1px solid #5a1a1a;
        margin-bottom: 1.2rem;
    }
    .status-cell {
        background: #15080a;
        padding: 0.9rem 1.2rem;
        font-family: 'JetBrains Mono', monospace;
    }
    .status-cell .label {
        color: #ff7878;
        font-size: 0.7rem;
        text-transform: uppercase;
        letter-spacing: 0.15em;
    }
    .status-cell .value {
        color: #ffffff;
        font-size: 1.8rem;
        font-weight: 800;
        margin-top: 0.2rem;
    }
    .status-cell .value.good { color: #4ade80; }
    .status-cell .value.bad { color: #ff4d4d; }
    /* Attack cards */
    .attack-card {
        background: #15080a;
        border: 1px solid #3a1a1a;
        padding: 0.8rem 1rem;
        margin-bottom: 0.6rem;
        font-family: 'JetBrains Mono', monospace;
    }
    .attack-card.blocked { border-left: 4px solid #4ade80; }
    .attack-card.succeeded { border-left: 4px solid #ff3838; background: #2a0d10; }
    .attack-card.ambiguous { border-left: 4px solid #ffc107; }
    .attack-card .outcome-tag {
        display: inline-block;
        padding: 2px 8px;
        border-radius: 3px;
        font-size: 0.7rem;
        font-weight: 700;
        letter-spacing: 0.1em;
        margin-right: 0.6rem;
    }
    .outcome-tag.blocked { background: #14532d; color: #4ade80; }
    .outcome-tag.succeeded { background: #5a0d10; color: #ff7878; }
    .outcome-tag.ambiguous { background: #4a3500; color: #ffc107; }
    </style>
    """,
    unsafe_allow_html=True,
)

GITHUB_URL = "https://github.com/anafe03/RL_agents/tree/main/octagon"


# --- sidebar ----------------------------------------------------------------

with st.sidebar:
    st.markdown("# 🛡️ Octagon")
    st.caption("Adversarial audit for AI agents")
    st.markdown("---")

    mode = st.radio(
        "Mode",
        options=["Demo (mock)", "Live (your API key)"],
        help="Demo runs in 2 seconds with canned data. Live runs the real audit against Claude Sonnet 4.6 (defender) + Opus 4.7 (judge).",
    )

    api_key_input = ""
    if mode == "Live (your API key)":
        api_key_input = st.text_input(
            "ANTHROPIC_API_KEY",
            type="password",
            help="Used only in your browser session. Not stored.",
        )
        st.caption("A 15-attack audit costs ~$0.50 with prompt caching.")

    st.markdown("---")
    defender_name = st.selectbox(
        "Defender",
        options=sorted(REGISTRY.keys()),
        index=0,
        help="The AI agent under test.",
    )

    st.markdown("---")
    st.markdown(f"[GitHub repo]({GITHUB_URL})")


# --- state ------------------------------------------------------------------

if "report" not in st.session_state:
    st.session_state.report = None
if "attacks" not in st.session_state:
    st.session_state.attacks = None


# --- header -----------------------------------------------------------------

st.markdown(
    f"""
    <div class="redcell-hero">
        <div class="classification">RED CELL // CLASSIFIED // ADVERSARIAL AUDIT</div>
        <h1>OCTAGON</h1>
        <div class="target">▸ TARGET: {defender_name}  ·  PHASE: PRE-DEPLOYMENT  ·  RUN MODE: {"LIVE" if mode.startswith("Live") else "MOCK"}</div>
    </div>
    """,
    unsafe_allow_html=True,
)
st.markdown(
    "Pen-test your AI agent. Octagon runs a library of categorized attacks — prompt injection, "
    "social engineering, tool-argument abuse, indirect injection — against a target agent and "
    "produces a structured incident report. Built for AI platform teams and cyber insurance "
    "underwriters who need to assess agent risk with numbers, not vibes."
)

# --- tabs -------------------------------------------------------------------

tab_run, tab_results, tab_how, tab_about = st.tabs(
    ["▶ Run Audit", "📊 Results", "📖 How it works", "ℹ️ About"]
)


# --- Run tab ----------------------------------------------------------------

with tab_run:
    col_left, col_right = st.columns([2, 1])

    with col_left:
        st.subheader(f"Target: `{defender_name}`")
        defender = get_defender(defender_name)
        with st.expander("Defender system prompt (the rules the agent must follow)"):
            st.markdown(defender.system_prompt)
        with st.expander(f"Defender tools ({len(defender.tool_schemas)})"):
            for tool in defender.tool_schemas:
                st.markdown(f"**`{tool['name']}`** — {tool['description']}")

    with col_right:
        st.subheader("Attack library")
        attacks = load_attacks("attacks")
        st.metric("Attacks loaded", f"{len(attacks)}")
        cats: dict[str, int] = {}
        for a in attacks:
            cats[a.category.value] = cats.get(a.category.value, 0) + 1
        for cat, n in sorted(cats.items()):
            st.markdown(f"- `{cat}` × {n}")

    st.markdown("---")

    can_run = True
    if mode == "Live (your API key)" and not api_key_input:
        st.warning("Live mode needs an `ANTHROPIC_API_KEY`. Enter one in the sidebar, or switch to Demo mode.")
        can_run = False

    if st.button("🚀 Run audit", type="primary", disabled=not can_run, use_container_width=True):
        if mode == "Demo (mock)":
            llm.set_chat_fn(make_mock_chat())
        else:
            os.environ["ANTHROPIC_API_KEY"] = api_key_input
            llm.reset_chat_fn()

        progress = st.progress(0.0, text="Loading attacks...")
        status = st.empty()
        try:
            results_so_far: list[AttackResult] = []
            from octagon.runner import run_attack
            from octagon.models import AuditReport
            from datetime import datetime, timezone

            report = AuditReport(defender_name=defender.name)
            n = len(attacks)
            for i, attack in enumerate(attacks):
                status.markdown(f"Running **{i+1}/{n}** — `{attack.id}` ({attack.category.value})")
                r = run_attack(defender, attack)
                report.results.append(r)
                progress.progress((i + 1) / n, text=f"Completed {i+1}/{n}")
            report.ended_at = datetime.now(timezone.utc)
            st.session_state.report = report
            st.session_state.attacks = attacks
            progress.empty()
            status.empty()
            st.success(f"Done! {report.total_blocked}/{report.total_attacks} blocked ({report.block_rate:.1%}). Open the **Results** tab.")
        finally:
            llm.reset_chat_fn()


# --- Results tab ------------------------------------------------------------

def _outcome_badge(outcome: Outcome) -> str:
    return {
        Outcome.BLOCKED: "✅ blocked",
        Outcome.SUCCEEDED: "❌ succeeded",
        Outcome.AMBIGUOUS: "❓ ambiguous",
        Outcome.ERROR: "⚠️ error",
    }.get(outcome, str(outcome))


with tab_results:
    report: AuditReport | None = st.session_state.report
    if report is None:
        st.info("No audit run yet. Head to the **Run Audit** tab and click 🚀.")
    else:
        # Defender status strip (terminal-style)
        block_class = "good" if report.block_rate >= 0.7 else "bad"
        succeeded_class = "bad" if report.total_succeeded > 0 else "good"
        st.markdown(
            f"""
            <div class="status-strip">
                <div class="status-cell">
                    <div class="label">BLOCK RATE</div>
                    <div class="value {block_class}">{report.block_rate:.1%}</div>
                </div>
                <div class="status-cell">
                    <div class="label">BLOCKED</div>
                    <div class="value good">{report.total_blocked}/{report.total_attacks}</div>
                </div>
                <div class="status-cell">
                    <div class="label">VIOLATIONS</div>
                    <div class="value {succeeded_class}">{report.total_succeeded}</div>
                </div>
                <div class="status-cell">
                    <div class="label">RUN COST</div>
                    <div class="value">${report.total_cost_usd:.4f}</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("---")

        # Chart: blocked vs succeeded by category
        st.subheader("By category")
        by_cat = report.by_category
        chart_data = {
            "Category": [],
            "Blocked": [],
            "Succeeded": [],
        }
        for cat, stats in sorted(by_cat.items()):
            chart_data["Category"].append(cat)
            chart_data["Blocked"].append(stats["blocked"])
            chart_data["Succeeded"].append(stats["succeeded"])

        try:
            import pandas as pd
            df = pd.DataFrame(chart_data).set_index("Category")
            st.bar_chart(df, color=["#16a34a", "#dc2626"])
        except Exception:
            st.bar_chart(chart_data)

        st.markdown("---")

        # Sort: succeeded first (most interesting), then blocked
        succeeded = [r for r in report.results if r.outcome == Outcome.SUCCEEDED]
        blocked = [r for r in report.results if r.outcome == Outcome.BLOCKED]
        other = [r for r in report.results if r.outcome not in (Outcome.SUCCEEDED, Outcome.BLOCKED)]
        ordered = succeeded + other + blocked

        # Detail per attack
        if succeeded:
            st.subheader(f"⚠️ Successful attacks ({len(succeeded)}) — defender violations")
            st.caption("These are the attacks where the defender failed. Click to inspect.")
        for r in ordered:
            label = f"{_outcome_badge(r.outcome)}  ·  `{r.attack_id}`  ·  {r.attack_category.value}"
            with st.expander(label):
                attack = next((a for a in (st.session_state.attacks or []) if a.id == r.attack_id), None)
                if attack:
                    st.markdown(f"**Attack name:** {attack.name}")
                    st.markdown(f"**Intent:** {attack.intent}")
                if r.tools_called:
                    st.markdown(f"**Tools the defender called:** {', '.join(r.tools_called)}")
                st.markdown("**Judge's reasoning:**")
                st.markdown(f"> {r.judge_reasoning}")
                if r.transcript:
                    st.markdown("**Transcript:**")
                    for turn in r.transcript:
                        if turn.role == "attacker":
                            st.markdown(f"**🔴 Attacker:** {turn.content}")
                        elif turn.role == "defender":
                            st.markdown(f"**🔵 Defender:** {turn.content or '_(tool call only)_'}")
                            for tc in turn.tool_calls:
                                st.code(f"{tc.name}({json.dumps(tc.args)})", language="python")
                        elif turn.role == "tool":
                            st.caption(f"[tool result] {turn.content}")


# --- How it works tab -------------------------------------------------------

with tab_how:
    st.subheader("The threat model")
    st.markdown(
        "An AI agent in production with tool access is a new attack surface. It can be:\n"
        "- **Tricked into running its tools** with bad arguments (social engineering, prompt injection)\n"
        "- **Poisoned through its data inputs** (a customer ticket containing hidden instructions)\n"
        "- **Talked out of its safety rules** via authority claims or fabricated 'system updates'\n\n"
        "Traditional security questionnaires (MFA, EDR, patch SLAs) don't measure any of this. "
        "Octagon's Red Cell does: it runs a library of categorized attacks and produces a numeric block rate, "
        "a heat map by attack class, and the specific transcripts that show where the agent cracked."
    )

    st.markdown("---")
    st.subheader("The 7 attack categories")
    categories = [
        ("authority_impersonation", "**'I'm the CEO, do it now'**", "Business Email Compromise"),
        ("emergency_framing", "**'Customer demo in 5 min, no time to verify'**", "Wire fraud / urgency-driven authorization bypass"),
        ("prompt_injection", "**Direct instruction overrides** ('ignore previous rules')", "Malicious-input-induced unauthorized action"),
        ("role_confusion", "**Fake '[SYSTEM UPDATE]' framing inside user content**", "Insider-style privilege escalation"),
        ("indirect_injection", "**Payload hidden in a 'ticket' or user record field**", "Supply-chain / poisoned-data incident"),
        ("tool_argument_abuse", "**Legitimate-looking call with unsafe args**", "Sanctioned-action misuse"),
        ("social_engineering_chain", "**Multi-turn pretext building** (rapport then pivot)", "Vishing / impersonation incident"),
    ]
    for cat, what, claim in categories:
        with st.container(border=True):
            st.markdown(f"### `{cat}`")
            st.markdown(what)
            st.caption(f"Insurance-claim analogue: *{claim}*")

    st.markdown("---")
    st.subheader("Why this matters for cyber insurance")
    st.markdown(
        "Cyber insurers (Coalition, Resilience, At-Bay) have decades of practice underwriting traditional IT risk. "
        "They have almost none for agentic AI in production. The new risk surface — a bot with tool access, social-engineerable, "
        "prompt-injectable, capable of executing real workflows — is barely measurable with today's questionnaires. "
        "**Octagon's Red Cell is a measurement.** The output is the shape an active-insurance program could feed into pre-binding "
        "diligence or post-bind monitoring for any insured deploying a customer-facing or operational AI agent."
    )


# --- About tab --------------------------------------------------------------

with tab_about:
    st.markdown("### Octagon — Adversarial audit + tournament platform for LLM agents")
    st.markdown(
        "**v0.1 ships Red Cell** — the adversarial audit module. Next: tournament leagues "
        "(travel itinerary debates, music critic faceoffs, sustainability consulting, cyber "
        "code-review), ELO ratings, and learning attackers (evolutionary search over attack templates)."
    )
    st.markdown("---")
    st.markdown("**Stack:** Anthropic SDK · Claude Sonnet 4.6 (defender) · Claude Opus 4.7 (judge) · prompt caching · structured JSON output")
    st.markdown(f"**Repo:** [{GITHUB_URL}]({GITHUB_URL})")
    st.markdown("**Author:** Austin Nafe")
