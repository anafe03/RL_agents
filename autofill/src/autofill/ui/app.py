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

# Browser-chrome aesthetic — each step renders inside a faux Chrome window
# frame with a URL bar, traffic-light dots, and an action-indicator strip.
# Feels like you're watching a screen recording instead of reading cards.
st.markdown(
    """
    <style>
    .stApp {
        background-color: #f1f3f4;
    }
    /* Browser chrome window */
    .browser-window {
        background: white;
        border: 1px solid #d0d7de;
        border-radius: 8px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
        overflow: hidden;
        margin: 0.5rem 0 1rem 0;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }
    .browser-chrome {
        background: #f1f3f4;
        border-bottom: 1px solid #d0d7de;
        padding: 0.7rem 1rem;
        display: flex;
        align-items: center;
        gap: 0.7rem;
    }
    .browser-dots {
        display: flex;
        gap: 6px;
    }
    .browser-dots span {
        width: 12px;
        height: 12px;
        border-radius: 50%;
        display: inline-block;
    }
    .browser-dots .red { background: #ff5f57; }
    .browser-dots .yellow { background: #febc2e; }
    .browser-dots .green { background: #28c840; }
    .browser-url-bar {
        flex: 1;
        background: white;
        border: 1px solid #d0d7de;
        border-radius: 16px;
        padding: 0.35rem 0.9rem;
        font-size: 0.85rem;
        color: #444;
        font-family: -apple-system, "SF Mono", Menlo, monospace;
        overflow: hidden;
        text-overflow: ellipsis;
        white-space: nowrap;
    }
    .browser-url-bar .lock { color: #28c840; margin-right: 0.4rem; }
    .browser-body {
        background: white;
        padding: 1.5rem 2rem;
        min-height: 220px;
        position: relative;
    }
    /* Action-indicator strip — shows which action the agent took on this frame */
    .action-strip {
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        background: #1f2328;
        color: white;
        padding: 0.45rem 0.9rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
        margin-bottom: 1rem;
    }
    .action-strip.click { background: #fb8500; }
    .action-strip.type { background: #023e8a; }
    .action-strip.scroll { background: #6c757d; }
    .action-strip.navigate { background: #2d6a4f; }
    .action-strip.screenshot { background: #495057; }
    .action-strip.observe { background: #6f42c1; }
    .action-strip.halt { background: #c1121f; }
    .frame-content h3 {
        margin: 0 0 0.4rem 0;
        font-size: 1rem;
        color: #1f2328;
    }
    .frame-content .typed-value {
        background: #f6f8fa;
        border: 1px solid #d0d7de;
        border-radius: 4px;
        padding: 0.4rem 0.7rem;
        font-family: "SF Mono", Menlo, Consolas, monospace;
        font-size: 0.85rem;
        color: #0969da;
        display: inline-block;
        margin: 0.3rem 0;
    }
    .frame-content .narration {
        color: #57606a;
        font-style: italic;
        margin-top: 0.7rem;
        line-height: 1.5;
    }
    </style>
    """,
    unsafe_allow_html=True,
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

    # Pick the URL displayed in the chrome bar based on the current step
    # (NAVIGATE steps carry the URL in `value`; later steps stay on the
    # last-known URL).
    current_url = target.url
    for prior in playback.steps[: pos + 1]:
        if prior.action == StepAction.NAVIGATE and prior.value:
            current_url = prior.value

    action_class = {
        StepAction.NAVIGATE: "navigate",
        StepAction.SCREENSHOT: "screenshot",
        StepAction.CLICK: "click",
        StepAction.TYPE: "type",
        StepAction.KEY: "type",
        StepAction.SCROLL: "scroll",
        StepAction.OBSERVE: "observe",
        StepAction.HALT: "halt",
        StepAction.SUBMIT_ATTEMPTED: "halt",
    }.get(step.action, "observe")

    action_label = {
        StepAction.NAVIGATE: "🌐 NAVIGATE",
        StepAction.SCREENSHOT: "📷 SCREENSHOT",
        StepAction.CLICK: "🖱️ CLICK",
        StepAction.TYPE: "⌨️ TYPE",
        StepAction.KEY: "⌨️ KEY",
        StepAction.SCROLL: "🔽 SCROLL",
        StepAction.OBSERVE: "👁️ OBSERVE",
        StepAction.HALT: "🛑 HALT — BEFORE SUBMIT",
        StepAction.SUBMIT_ATTEMPTED: "📤 SUBMIT",
    }.get(step.action, str(step.action.value))

    import html as _html
    target_label_html = _html.escape(step.target_label) if step.target_label else ""
    value_html = _html.escape(step.value)[:200] if step.value else ""
    narration_html = _html.escape(step.narration).replace("\n", "<br>") if step.narration else ""

    typed_block = f'<div><span class="typed-value">{value_html}</span></div>' if value_html else ""
    target_block = f'<h3>{action_label.split(" ", 1)[-1]}: {target_label_html}</h3>' if target_label_html else f'<h3>{action_label}</h3>'
    narration_block = f'<div class="narration">{narration_html}</div>' if narration_html else ""

    col_step, col_meta = st.columns([3, 1])
    with col_step:
        st.markdown(
            f"""
            <div class="browser-window">
                <div class="browser-chrome">
                    <div class="browser-dots">
                        <span class="red"></span><span class="yellow"></span><span class="green"></span>
                    </div>
                    <div class="browser-url-bar"><span class="lock">🔒</span>{_html.escape(current_url)}</div>
                </div>
                <div class="browser-body">
                    <div class="action-strip {action_class}">{action_label}  ·  step {step.step_id:02d}</div>
                    <div class="frame-content">
                        {target_block}
                        {typed_block}
                        {narration_block}
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

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
