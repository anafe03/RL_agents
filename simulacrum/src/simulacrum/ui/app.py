"""Streamlit UI for Simulacrum.

Run locally:
    cd simulacrum
    uv run streamlit run src/simulacrum/ui/app.py

Hosted on Streamlit Cloud — see README for the link.
"""

from __future__ import annotations

import os
from pathlib import Path

import streamlit as st

from simulacrum import llm
from simulacrum.engine import run_scenario
from simulacrum.mock import make_mock_chat
from simulacrum.models import Scenario, Transcript, load_scenario


st.set_page_config(
    page_title="Simulacrum",
    page_icon="🎭",
    layout="wide",
    initial_sidebar_state="expanded",
)

GITHUB_URL = "https://github.com/anafe03/RL_agents/tree/main/simulacrum"


# --- discover scenarios -----------------------------------------------------

SCENARIOS_DIR = Path(__file__).resolve().parents[3] / "scenarios"


def discover_scenarios() -> list[tuple[str, Path]]:
    """Return (display_name, path) for every scenario folder."""
    out: list[tuple[str, Path]] = []
    if not SCENARIOS_DIR.exists():
        return out
    for child in sorted(SCENARIOS_DIR.iterdir()):
        if not child.is_dir():
            continue
        if (child / "scenario.yaml").exists():
            try:
                sc = load_scenario(child)
                out.append((sc.title or sc.name, child))
            except Exception:
                continue
    return out


# --- sidebar ----------------------------------------------------------------

with st.sidebar:
    st.markdown("# 🎭 Simulacrum")
    st.caption("Cast personas. Set a stage. Press play.")
    st.markdown("---")

    mode = st.radio(
        "Mode",
        options=["Demo (mock)", "Live (your API key)"],
        help="Demo plays a canned dialogue in 2 seconds. Live runs the real LLMs (Claude Sonnet 4.6).",
    )

    api_key_input = ""
    if mode == "Live (your API key)":
        api_key_input = st.text_input(
            "ANTHROPIC_API_KEY",
            type="password",
            help="Used only in your browser session. Not stored.",
        )
        st.caption("A 5-tick scenario costs ~$0.05.")

    st.markdown("---")
    scenarios = discover_scenarios()
    if not scenarios:
        st.error("No scenarios found.")
        st.stop()

    scenario_labels = [label for label, _ in scenarios]
    selected_label = st.selectbox("Scenario", scenario_labels)
    selected_path = next(p for label, p in scenarios if label == selected_label)
    scenario: Scenario = load_scenario(selected_path)

    st.markdown("---")
    ticks_override = st.slider(
        "Max ticks",
        min_value=1,
        max_value=12,
        value=min(6, scenario.max_ticks),
        help="How many rounds of conversation. Stops early if everyone passes.",
    )

    st.markdown("---")
    st.markdown(f"[GitHub repo]({GITHUB_URL})")


# --- header -----------------------------------------------------------------

st.markdown(f"# {scenario.title or scenario.name}")
if scenario.setting:
    st.markdown(f"*{scenario.setting.strip()}*")

if scenario.shared_goal:
    with st.container(border=True):
        st.markdown("**🎯 Shared goal of the room**")
        st.markdown(scenario.shared_goal.strip())


# --- cast cards -------------------------------------------------------------

st.markdown("---")
st.markdown("### Cast")
cols = st.columns(len(scenario.agents))
for col, agent in zip(cols, scenario.agents):
    with col, st.container(border=True):
        st.markdown(f"**{agent.name}**")
        if agent.role:
            st.caption(agent.role)
        if agent.private_goal:
            with st.expander("Private goal"):
                st.markdown(agent.private_goal)
        with st.expander("Persona"):
            st.markdown(agent.persona)


# --- run --------------------------------------------------------------------

st.markdown("---")

if "transcript" not in st.session_state:
    st.session_state.transcript = None
    st.session_state.transcript_scenario = None

can_run = True
if mode == "Live (your API key)" and not api_key_input:
    st.warning("Live mode needs an `ANTHROPIC_API_KEY`. Enter one in the sidebar, or switch to Demo mode.")
    can_run = False

if st.button("▶ Play scenario", type="primary", disabled=not can_run, use_container_width=True):
    if mode == "Demo (mock)":
        llm.set_chat_fn(make_mock_chat(scenario_name=scenario.name))
    else:
        os.environ["ANTHROPIC_API_KEY"] = api_key_input
        llm.reset_chat_fn()

    progress = st.progress(0.0, text="Running tick 1...")
    try:
        transcript = run_scenario(scenario, max_ticks=ticks_override)
        st.session_state.transcript = transcript
        st.session_state.transcript_scenario = scenario.name
        progress.empty()
        st.success(f"Done — {len(transcript.ticks)} ticks. Cost: ${transcript.cost_usd:.4f}")
    finally:
        llm.reset_chat_fn()


# --- transcript -------------------------------------------------------------

transcript: Transcript | None = st.session_state.transcript
if transcript and st.session_state.transcript_scenario == scenario.name:
    st.markdown("### Transcript")
    names = {a.id: a.name for a in scenario.agents}
    # Per-agent colors via emoji prefix so it works in vanilla streamlit
    palette = ["🟦", "🟩", "🟧", "🟪", "🟥", "🟨"]
    agent_emoji = {a.id: palette[i % len(palette)] for i, a in enumerate(scenario.agents)}

    for tick in transcript.ticks:
        st.markdown(f"#### Tick {tick.number}")
        for action in tick.actions:
            name = names.get(action.actor_id, action.actor_id)
            emoji = agent_emoji.get(action.actor_id, "·")
            if action.type.value == "pass":
                st.caption(f"_{emoji} {name} passes._")
            else:
                st.markdown(f"{emoji} **{name}:** {action.content}")
        st.markdown("")

    st.caption(f"Total LLM cost: ${transcript.cost_usd:.4f}")
elif st.session_state.transcript and st.session_state.transcript_scenario != scenario.name:
    st.info("You switched scenarios. Click ▶ Play again to run this one.")
