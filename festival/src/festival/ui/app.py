"""Streamlit UI for Festival Companion.

Run locally:
    cd festival
    uv run streamlit run src/festival/ui/app.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Force the installed festival package to win over any bare `festival/`
# namespace directory that Streamlit Cloud's CWD might expose (same gotcha
# we hit with simulacrum).
_SRC = Path(__file__).resolve().parents[2]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import streamlit as st

from festival import llm
from festival.matcher import score_lineup
from festival.mock import make_mock_chat
from festival.models import TasteProfile, list_lineups, load_lineup
from festival.scheduler import build_schedule


st.set_page_config(
    page_title="Festival Companion",
    page_icon="🎟️",
    layout="wide",
    initial_sidebar_state="expanded",
)

GITHUB_URL = "https://github.com/anafe03/RL_agents/tree/main/festival"

LINEUPS_DIR = Path(__file__).resolve().parents[3] / "data" / "lineups"


# --- sidebar ----------------------------------------------------------------

with st.sidebar:
    st.markdown("# 🎟️ Festival Companion")
    st.caption("Paste your taste, get a schedule.")
    st.markdown("---")

    mode = st.radio(
        "Mode",
        ["Demo (mock)", "Live (your API key)"],
        help="Demo plans the schedule in 2 seconds with deterministic scoring. Live uses Claude Sonnet 4.6 per artist.",
    )
    api_key_input = ""
    if mode == "Live (your API key)":
        api_key_input = st.text_input(
            "ANTHROPIC_API_KEY", type="password", help="Used only in your browser session."
        )
        st.caption("A 25-artist lineup costs ~$0.08 with prompt caching.")

    st.markdown("---")
    lineups = list_lineups(LINEUPS_DIR)
    if not lineups:
        st.error("No lineups found under data/lineups/")
        st.stop()
    lineup_labels = [f"{name}" for name, _ in lineups]
    selected_label = st.selectbox("Festival", lineup_labels)
    selected_path = next(p for name, p in lineups if name == selected_label)
    festival = load_lineup(selected_path)

    st.markdown("---")
    st.markdown(f"[GitHub repo]({GITHUB_URL})")


# --- header -----------------------------------------------------------------

st.markdown(f"# {festival.name}")
st.caption(f"{festival.city} · {festival.year} · {len(festival.days)} days · {len(festival.sets)} sets")

# Glance at the lineup before generating
with st.expander("Full lineup"):
    for day in festival.days:
        st.markdown(f"**{day}**")
        for s in sorted(festival.sets_on(day), key=lambda x: x.start):
            head = " ★" if s.headliner else ""
            st.markdown(f"- `{s.start}–{s.end}` {s.artist}{head} *({s.stage})*")


# --- taste form -------------------------------------------------------------

st.markdown("---")
st.markdown("## Your taste")

col1, col2 = st.columns([3, 2])
with col1:
    description = st.text_area(
        "Describe what you listen to",
        placeholder="e.g. indie rock, dream pop, some electronic; love sad songwriters, mid-tempo dance, weird jazz; not into pop punk",
        height=120,
    )
    favorites_str = st.text_input(
        "Favorite artists (comma-separated)",
        placeholder="Phoebe Bridgers, Beach House, Caribou",
    )

with col2:
    artist_options = sorted({s.artist for s in festival.sets})
    must_see = st.multiselect("Must-see (will be locked into the schedule)", artist_options)
    avoid = st.multiselect("Never schedule these", artist_options)


# --- generate ---------------------------------------------------------------

if "schedule" not in st.session_state:
    st.session_state.schedule = None
    st.session_state.schedule_festival = None

can_run = True
if mode == "Live (your API key)" and not api_key_input:
    st.warning("Live mode needs an `ANTHROPIC_API_KEY`. Enter one in the sidebar, or switch to Demo mode.")
    can_run = False

if st.button("🎟️ Generate my schedule", type="primary", disabled=not can_run, use_container_width=True):
    if mode == "Demo (mock)":
        llm.set_chat_fn(make_mock_chat())
    else:
        os.environ["ANTHROPIC_API_KEY"] = api_key_input
        llm.reset_chat_fn()

    favorites = [a.strip() for a in favorites_str.split(",") if a.strip()]
    profile = TasteProfile(
        description=description,
        favorite_artists=favorites,
        must_see=list(must_see),
        avoid=list(avoid),
    )

    progress = st.progress(0.0, text="Scoring artists...")
    try:
        recs, cost = score_lineup(festival, profile)
        progress.progress(0.7, text="Resolving conflicts...")
        schedule = build_schedule(festival, profile, recs)
        schedule.cost_usd = cost
        progress.empty()
        st.session_state.schedule = schedule
        st.session_state.schedule_festival = festival.name
        st.success(
            f"{schedule.total_picks} picks across {len(schedule.days)} days · "
            f"avg fit {schedule.average_score:.2f} · cost ${schedule.cost_usd:.4f}"
        )
    finally:
        llm.reset_chat_fn()


# --- schedule view ----------------------------------------------------------

schedule = st.session_state.schedule
if schedule is None or st.session_state.schedule_festival != festival.name:
    if schedule is not None:
        st.info("You switched festivals — generate a new schedule for this one.")
else:
    st.markdown("---")
    st.markdown("## Your schedule")
    day_tabs = st.tabs([d.day for d in schedule.days])
    for tab, day in zip(day_tabs, schedule.days):
        with tab:
            if not day.picks:
                st.info("No picks for this day.")
                continue
            for p in day.picks:
                container = st.container(border=True)
                with container:
                    cols = st.columns([1, 2, 1])
                    star = "  ★" if p.must_see else ""
                    cols[0].markdown(f"**{p.start} – {p.end}**\n\n*{p.stage}*")
                    cols[1].markdown(f"### {p.artist}{star}")
                    cols[1].markdown(p.reasoning)
                    cols[2].metric("Fit", f"{p.score:.2f}")
            if day.skipped_due_to_conflict:
                with st.expander(f"Skipped due to conflicts ({len(day.skipped_due_to_conflict)})"):
                    for s in day.skipped_due_to_conflict[:8]:
                        st.markdown(
                            f"- `{s.start}–{s.end}` **{s.artist}** ({s.stage}) — fit {s.score:.2f}\n  *{s.reasoning}*"
                        )
