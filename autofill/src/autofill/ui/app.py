"""Streamlit UI for AutoFill.

Run locally:
    cd autofill
    uv sync
    uv run streamlit run src/autofill/ui/app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Force installed package to win over any bare `autofill/` namespace package
# at the repo root.
_SRC = Path(__file__).resolve().parents[2]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import streamlit as st

from autofill.mock import get_mock_run
from autofill.models import StepAction, load_complaint
from autofill.targets import REGISTRY


st.set_page_config(
    page_title="AutoFill",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

GITHUB_URL = "https://github.com/anafe03/RL_agents/tree/main/autofill"
COMPLAINTS_DIR = Path(__file__).resolve().parents[3] / "data" / "complaints"


# --- sidebar ----------------------------------------------------------------

with st.sidebar:
    st.markdown("# 🤖 AutoFill")
    st.caption("Computer Use agent for public complaint forms.")
    st.markdown("---")

    mode = st.radio(
        "Mode",
        ["Demo (mock playback)", "Live (your API key, runs locally)"],
        help="Mock playback replays a recorded session. Live mode requires Playwright installed locally and an API key.",
    )

    if mode.startswith("Live"):
        st.info(
            "Live mode requires `uv sync --extra browser` + "
            "`uv run playwright install chromium` + `ANTHROPIC_API_KEY`. "
            "It opens a real Chromium window and drives it. "
            "**Not available in this hosted demo.**"
        )

    st.markdown("---")
    target_choices = list(REGISTRY.keys())
    target_id = st.selectbox("Target form", target_choices)
    target = REGISTRY[target_id]

    complaint_files = sorted(COMPLAINTS_DIR.glob("*.yaml")) if COMPLAINTS_DIR.exists() else []
    if not complaint_files:
        st.error("No complaints under data/complaints/")
        st.stop()
    complaint_labels = {load_complaint(p).id: p for p in complaint_files}
    complaint_id = st.selectbox("Complaint", list(complaint_labels.keys()))
    complaint = load_complaint(complaint_labels[complaint_id])

    st.markdown("---")
    st.warning(
        "⚠️ All complaints are synthetic. The agent halts before clicking Submit; "
        "no real DOI complaints are filed by this demo."
    )
    st.markdown(f"[GitHub repo]({GITHUB_URL})")


# --- header -----------------------------------------------------------------

st.markdown(f"# {target.name}")
st.caption(f"Target URL: {target.url}")

# Show the complaint that's about to be filed
col_who, col_what = st.columns(2)
with col_who:
    with st.container(border=True):
        st.markdown("### Complainant")
        st.markdown(f"**Name:** {complaint.complainant_name}")
        st.markdown(f"**Email:** {complaint.complainant_email}")
        st.markdown(f"**Phone:** {complaint.complainant_phone}")
        st.markdown(f"**Address:** {complaint.complainant_address}")
with col_what:
    with st.container(border=True):
        st.markdown("### Complaint")
        st.markdown(f"**Insurer:** {complaint.insurer_name}")
        st.markdown(f"**Member ID:** `{complaint.insurer_member_id}`")
        st.markdown(f"**Claim #:** `{complaint.claim_number}`")
        st.markdown(f"**Service requested:** {complaint.requested_service}")
        with st.expander("Narrative"):
            st.markdown(complaint.narrative)


# --- run --------------------------------------------------------------------

st.markdown("---")

if "playback" not in st.session_state:
    st.session_state.playback = None
    st.session_state.playback_position = 0
    st.session_state.playback_key = None

can_play = True
if mode.startswith("Live"):
    st.error(
        "Live submission is not available in this hosted demo. Run AutoFill locally "
        "with `uv sync --extra browser` to enable it."
    )
    can_play = False

if st.button("▶ Play the agent run", type="primary", disabled=not can_play, use_container_width=True):
    playback = get_mock_run(target_id, complaint.id)
    if playback is None:
        st.error(
            f"No mock playback recorded for target={target_id}, complaint={complaint.id}. "
            "Try the bundled CA DOI + glp1_denial pair."
        )
    else:
        st.session_state.playback = playback
        st.session_state.playback_position = 0
        st.session_state.playback_key = (target_id, complaint.id)


playback = st.session_state.playback
if playback and st.session_state.playback_key == (target_id, complaint.id):
    n = len(playback.steps)
    pos = st.slider("Step", 0, n - 1, st.session_state.playback_position, key="step_slider")
    st.session_state.playback_position = pos

    step = playback.steps[pos]

    col_step, col_meta = st.columns([3, 1])
    with col_step:
        action_color = {
            StepAction.NAVIGATE: "🌐",
            StepAction.SCREENSHOT: "📷",
            StepAction.CLICK: "🖱️",
            StepAction.TYPE: "⌨️",
            StepAction.KEY: "⌨️",
            StepAction.SCROLL: "🔽",
            StepAction.OBSERVE: "👁️",
            StepAction.HALT: "🛑",
            StepAction.SUBMIT_ATTEMPTED: "📤",
        }.get(step.action, "•")

        st.markdown(f"### {action_color} Step {step.step_id:02d} — `{step.action.value}`")
        if step.target_label:
            st.markdown(f"**Target:** {step.target_label}")
        if step.value:
            st.markdown(f"**Value typed:** `{step.value[:200]}`")
        if step.narration:
            with st.container(border=True):
                st.markdown(f"_{step.narration}_")

    with col_meta:
        st.metric("Step", f"{pos + 1} / {n}")
        if playback.dry_run:
            st.success("DRY RUN")
        else:
            st.warning("LIVE SUBMIT")
        st.caption(f"Cost: ${playback.cost_usd:.4f}")

    # Action-summary timeline
    with st.expander("Full step timeline"):
        for s in playback.steps:
            mark = "✅" if s.step_id < pos else ("⏵" if s.step_id == pos else "·")
            st.markdown(f"{mark} `[{s.step_id:02d}]` **{s.action.value}** — {s.target_label or '(no label)'}")

    if step.action == StepAction.HALT:
        st.info(
            "The agent stopped before clicking Submit (dry-run mode). In a real workflow a human "
            "reviewer would now scan the filled form and click Submit themselves."
        )
