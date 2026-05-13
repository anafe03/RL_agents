"""Streamlit UI for Simulacrum.

Run locally:
    cd simulacrum
    uv run streamlit run src/simulacrum/ui/app.py

Hosted on Streamlit Cloud — see README for the link.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Streamlit Cloud runs from the repo root, which contains a bare `simulacrum/`
# directory. Without this, Python finds that directory as a namespace package
# and crashes on `from simulacrum import llm` because the namespace has no
# `llm` submodule. Prepending the package's actual `src/` dir to sys.path
# makes the installed `simulacrum` package win the import.
_SRC = Path(__file__).resolve().parents[2]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

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

# Theatrical / screenplay aesthetic — serif display type, warm cream/sepia,
# dialogue formatted like a play.
st.markdown(
    """
    <style>
    .stApp {
        background-color: #f5efe6;
    }
    .block-container {
        max-width: 1100px;
    }
    h1, h2, h3 {
        font-family: 'Georgia', 'Playfair Display', serif !important;
        color: #2a2118 !important;
    }
    .scene-header {
        border-top: 2px solid #2a2118;
        border-bottom: 1px solid #2a2118;
        padding: 1rem 0;
        margin-bottom: 1.5rem;
        font-family: 'Georgia', serif;
        text-align: center;
    }
    .scene-header .act {
        color: #8b6f47;
        font-size: 0.85rem;
        letter-spacing: 0.4em;
        font-weight: 700;
        text-transform: uppercase;
    }
    .scene-header .title {
        color: #2a2118;
        font-size: 2.2rem;
        font-weight: 700;
        font-style: italic;
        margin-top: 0.3rem;
    }
    /* Each agent line styled like a play — name in caps, then dialogue */
    .agent-line {
        margin: 0.7rem 0;
        font-family: 'Georgia', serif;
        line-height: 1.5;
    }
    .agent-line .name {
        color: #8b1538;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        font-size: 0.92rem;
    }
    .agent-line .dialogue {
        color: #2a2118;
        font-size: 1.05rem;
        margin-left: 1rem;
        margin-top: 0.1rem;
    }
    .agent-line.pass .dialogue {
        font-style: italic;
        color: #8b6f47;
    }
    </style>
    """,
    unsafe_allow_html=True,
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
    selected_model = "claude-sonnet-4-6"
    if mode == "Live (your API key)":
        selected_model = st.selectbox(
            "Model",
            llm.SUPPORTED_MODELS,
            index=llm.SUPPORTED_MODELS.index("gpt-4o-mini"),
            help="gpt-4o-mini is ~100× cheaper than Sonnet for testing.",
        )
        if llm._is_openai_model(selected_model):
            api_key_input = st.text_input(
                "OPENAI_API_KEY", type="password", help="Used only in your browser session."
            )
            st.caption("A 5-tick scenario costs ~$0.001 on gpt-4o-mini.")
        else:
            api_key_input = st.text_input(
                "ANTHROPIC_API_KEY", type="password", help="Used only in your browser session."
            )
            st.caption("A 5-tick scenario costs ~$0.05 on Sonnet 4.6.")

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
    key_var = "OPENAI_API_KEY" if llm._is_openai_model(selected_model) else "ANTHROPIC_API_KEY"
    st.warning(f"Live mode needs an `{key_var}`. Enter one in the sidebar, or switch to Demo mode.")
    can_run = False

if st.button("▶ Play scenario", type="primary", disabled=not can_run, use_container_width=True):
    if mode == "Demo (mock)":
        llm.set_chat_fn(make_mock_chat(scenario_name=scenario.name))
    else:
        if llm._is_openai_model(selected_model):
            os.environ["OPENAI_API_KEY"] = api_key_input
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
    names = {a.id: a.name for a in scenario.agents}

    # Render as a script: scene headers between ticks, dialogue formatted
    # with uppercase speaker name followed by their line, italicized passes.
    # Uses the .agent-line / .name / .dialogue CSS classes defined at the top.

    import html as _html

    def _esc(s: str) -> str:
        return _html.escape(s).replace("\n", "<br>")

    transcript_html_parts: list[str] = []
    n_acts = len(transcript.ticks)
    roman = ["I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X", "XI", "XII"]

    for i, tick in enumerate(transcript.ticks):
        act_label = roman[i] if i < len(roman) else str(i + 1)
        transcript_html_parts.append(
            f'<div class="scene-header">'
            f'<div class="act">ACT {act_label}</div>'
            f'<div class="title">Tick {tick.number}</div>'
            f'</div>'
        )
        for action in tick.actions:
            name = names.get(action.actor_id, action.actor_id)
            if action.type.value == "pass":
                transcript_html_parts.append(
                    f'<div class="agent-line pass">'
                    f'<div class="name">{_esc(name)}</div>'
                    f'<div class="dialogue">(beat — passes)</div>'
                    f'</div>'
                )
            else:
                transcript_html_parts.append(
                    f'<div class="agent-line">'
                    f'<div class="name">{_esc(name)}</div>'
                    f'<div class="dialogue">{_esc(action.content)}</div>'
                    f'</div>'
                )
    st.markdown("\n".join(transcript_html_parts), unsafe_allow_html=True)

    st.markdown(
        f'<div style="text-align:center;font-family:Georgia,serif;font-style:italic;color:#8b6f47;margin-top:2rem;border-top:1px solid #8b6f47;padding-top:1rem;">— END —  ·  total LLM cost: ${transcript.cost_usd:.4f}</div>',
        unsafe_allow_html=True,
    )
elif st.session_state.transcript and st.session_state.transcript_scenario != scenario.name:
    st.info("You switched scenarios. Click ▶ Play again to run this one.")
