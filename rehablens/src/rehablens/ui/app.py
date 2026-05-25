"""Streamlit UI for rehablens.

Run locally:
    cd rehablens
    uv sync
    uv run streamlit run src/rehablens/ui/app.py

Upload a photo of yourself (or the patient) at peak position for the
chosen exercise — bottom of a squat, overhead reach, mid-balance — and
rehablens overlays the pose skeleton and grades the form.
"""

from __future__ import annotations

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[2]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import streamlit as st

from rehablens.analyzer import analyze
from rehablens.exercises import EXERCISES, get_exercise
from rehablens.models import FormStatus
from rehablens.pose import detect_pose
from rehablens.render import overlay_pose

st.set_page_config(page_title="rehablens", page_icon="🦴", layout="wide",
                   initial_sidebar_state="expanded")

ACCENT = "#22d3ee"
st.markdown(
    f"""
    <style>
    .stApp {{ background-color: #0a0f14; }}
    .rl-header {{
        background: linear-gradient(135deg, #11181f 0%, #0a0f14 100%);
        border: 1px solid #1b2630;
        border-left: 4px solid {ACCENT};
        border-radius: 8px;
        padding: 1.1rem 1.4rem;
        margin-bottom: 0.6rem;
    }}
    .rl-header .title {{
        font-size: 1.7rem; font-weight: 800; color: {ACCENT};
        font-family: -apple-system, "Segoe UI", sans-serif; letter-spacing: 0.03em;
    }}
    .rl-header .sub {{ color: #8b97a3; font-size: 0.88rem; margin-top: 0.2rem; }}
    .rl-card {{
        background: #11181f; border: 1px solid #1b2630;
        border-radius: 8px; padding: 0.8rem 1rem; margin-bottom: 0.6rem;
        font-family: "SF Mono", Menlo, monospace; color: #cbd4dd;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

_STATUS = {
    FormStatus.OK: ("#22c55e", "✓"),
    FormStatus.WARN: ("#eab308", "⚠"),
    FormStatus.FAIL: ("#ef4444", "✗"),
    FormStatus.UNKNOWN: ("#5d6b7a", "·"),
}

GITHUB_URL = "https://github.com/anafe03/RL_agents/tree/main/rehablens"


# --- sidebar ----------------------------------------------------------------

with st.sidebar:
    st.markdown("# 🦴 rehablens")
    st.caption("Vision-based motion analysis for PM&R rehab.")
    st.markdown("---")

    exercise_id = st.selectbox(
        "Exercise",
        list(EXERCISES.keys()),
        format_func=lambda k: EXERCISES[k].name,
    )
    exercise = get_exercise(exercise_id)
    st.caption(exercise.description)

    st.markdown("---")
    st.caption(
        "Upload a clear, full-body photo at peak position. Side view for "
        "squat and shoulder reach; front view for single-leg stand."
    )
    st.markdown(f"[GitHub repo]({GITHUB_URL})")


# --- header -----------------------------------------------------------------

st.markdown(
    """
    <div class="rl-header">
        <div class="title">rehablens</div>
        <div class="sub">upload a photo, get a pose skeleton + form checks
        for the chosen rehab exercise</div>
    </div>
    """,
    unsafe_allow_html=True,
)


# --- upload + analyze -------------------------------------------------------

upload = st.file_uploader(
    "Patient photo", type=["jpg", "jpeg", "png"],
    help="A clear photo of one person, full-body, at peak position for the exercise.",
)

if upload is None:
    st.info("Pick an exercise in the sidebar and upload a photo above to begin.")
    st.stop()

image_bytes = upload.getvalue()

with st.spinner("Detecting pose..."):
    frame = detect_pose(image_bytes)

if not frame.detected:
    st.error(
        "No pose detected in that image. Try a clearer, full-body photo with "
        "good lighting — MediaPipe needs to see the whole person."
    )
    st.image(image_bytes, caption="Uploaded image")
    st.stop()

result = analyze(frame, exercise)
overlaid = overlay_pose(image_bytes, frame)


# --- results ---------------------------------------------------------------

img_col, info_col = st.columns([3, 2])

with img_col:
    st.image(overlaid, caption=f"{exercise.name} — pose overlay")

with info_col:
    color, icon = _STATUS[result.overall]
    st.markdown(
        f"<div style='font-family:-apple-system,sans-serif;"
        f"background:#11181f;border:1px solid #1b2630;"
        f"border-left:4px solid {color};border-radius:8px;"
        f"padding:0.9rem 1.1rem;margin-bottom:0.7rem'>"
        f"<div style='color:#8b97a3;font-size:0.78rem;font-family:monospace;"
        f"letter-spacing:0.12em'>OVERALL</div>"
        f"<div style='color:{color};font-size:1.5rem;font-weight:800'>"
        f"{icon} {result.overall.value.upper()}</div>"
        f"<div style='color:#cbd4dd;font-size:0.86rem;margin-top:0.25rem'>"
        f"{result.summary}</div></div>",
        unsafe_allow_html=True,
    )

    st.markdown("##### Form checks")
    for check in result.checks:
        c, icon = _STATUS[check.status]
        value = "—" if check.value is None else f"{check.value:g} {check.unit}"
        st.markdown(
            f"<div class='rl-card'>"
            f"<span style='color:{c};font-weight:800;font-size:1.1rem'>{icon}</span>"
            f" &nbsp;<b style='color:#dde4ea'>{check.name}</b><br>"
            f"<span style='color:#8b97a3;font-size:0.78rem'>"
            f"measured <b style='color:{c}'>{value}</b> &nbsp;·&nbsp; "
            f"target {check.target or '—'}</span></div>",
            unsafe_allow_html=True,
        )

st.divider()
st.caption(
    "Illustrative thresholds — not validated clinical criteria, and not a "
    "substitute for a physiatrist or physical therapist."
)
