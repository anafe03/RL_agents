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

# Festival-poster aesthetic: warm gradient header, condensed display type,
# stage-color accents. The schedule view is a Plotly Gantt-style timeline
# (one row per stage, bars across time, bordered + starred for picks).
st.markdown(
    """
    <style>
    /* Festival-poster header */
    .festival-hero {
        background: linear-gradient(135deg, #ff6b35 0%, #f7931e 30%, #ffc107 70%, #ff6b35 100%);
        padding: 2rem 2rem;
        border-radius: 8px;
        color: white;
        margin-bottom: 1.5rem;
        text-align: center;
        box-shadow: 0 8px 24px rgba(255, 107, 53, 0.25);
    }
    .festival-hero h1 {
        font-family: 'Georgia', 'Playfair Display', serif;
        font-size: 3.5rem;
        font-weight: 900;
        letter-spacing: -1px;
        margin: 0;
        text-shadow: 2px 2px 0 rgba(0, 0, 0, 0.2);
    }
    .festival-hero .subtitle {
        font-family: 'Georgia', serif;
        font-size: 1.05rem;
        margin-top: 0.5rem;
        letter-spacing: 3px;
        text-transform: uppercase;
        opacity: 0.9;
    }
    .pick-pill {
        display: inline-block;
        background: #1a1a1a;
        color: #ffc107;
        padding: 2px 10px;
        border-radius: 12px;
        font-size: 0.78rem;
        font-weight: 600;
        margin-right: 6px;
    }
    </style>
    """,
    unsafe_allow_html=True,
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
    selected_model = "claude-sonnet-4-6"
    if mode == "Live (your API key)":
        selected_model = st.selectbox(
            "Model",
            llm.SUPPORTED_MODELS,
            index=llm.SUPPORTED_MODELS.index("gpt-4o-mini"),
            help="gpt-4o-mini is ~100× cheaper for testing.",
        )
        if llm._is_openai_model(selected_model):
            api_key_input = st.text_input(
                "OPENAI_API_KEY", type="password", help="Used only in your browser session."
            )
            st.caption("A 25-artist lineup costs ~$0.001 on gpt-4o-mini.")
        else:
            api_key_input = st.text_input(
                "ANTHROPIC_API_KEY", type="password", help="Used only in your browser session."
            )
            st.caption("A 25-artist lineup costs ~$0.08 on Sonnet 4.6.")

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

st.markdown(
    f"""
    <div class="festival-hero">
        <h1>{festival.name}</h1>
        <div class="subtitle">{festival.city} · {festival.year} · {len(festival.days)} days · {len(festival.sets)} sets</div>
    </div>
    """,
    unsafe_allow_html=True,
)

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
    key_var = "OPENAI_API_KEY" if llm._is_openai_model(selected_model) else "ANTHROPIC_API_KEY"
    st.warning(f"Live mode needs an `{key_var}`. Enter one in the sidebar, or switch to Demo mode.")
    can_run = False

if st.button("🎟️ Generate my schedule", type="primary", disabled=not can_run, use_container_width=True):
    if mode == "Demo (mock)":
        llm.set_chat_fn(make_mock_chat())
    else:
        if llm._is_openai_model(selected_model):
            os.environ["OPENAI_API_KEY"] = api_key_input
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
        recs, cost = score_lineup(festival, profile, model=selected_model)
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
    st.markdown("## Your schedule")

    # Build a single Gantt-style timeline per day with Plotly.
    import plotly.graph_objects as go
    from datetime import datetime

    # Use a fixed reference date so HH:MM strings can be parsed as datetimes
    # for Plotly. The reference date itself doesn't matter — it's only
    # there so Plotly has datetime objects to draw bars between.
    def _to_dt(day_idx: int, hhmm: str) -> datetime:
        h, m = hhmm.split(":")
        return datetime(2025, 1, day_idx + 1, int(h), int(m))

    day_tabs = st.tabs([d.day for d in schedule.days])
    for tab_idx, (tab, day) in enumerate(zip(day_tabs, schedule.days)):
        with tab:
            if not day.picks and not day.skipped_due_to_conflict:
                st.info("No picks or skipped sets for this day.")
                continue

            # Combine picks + skipped so the full day's sets are on the chart.
            all_recs = list(day.picks) + list(day.skipped_due_to_conflict)
            stages = sorted({r.stage for r in all_recs})

            picks_set = {r.set_id for r in day.picks}
            fig = go.Figure()
            for r in all_recs:
                is_pick = r.set_id in picks_set
                # Color picks by fit score (warm), skipped in gray
                if is_pick:
                    score = max(0.0, min(1.0, r.score))
                    # Warm gradient: low fit → orange, high fit → gold
                    color = f"rgb({int(255)}, {int(107 + 148 * score)}, {int(53 * (1 - score))})"
                    border_color = "#1a1a1a"
                    border_width = 2
                else:
                    color = "rgba(180, 180, 180, 0.35)"
                    border_color = "rgba(120, 120, 120, 0.5)"
                    border_width = 1

                start_dt = _to_dt(tab_idx, r.start)
                end_dt = _to_dt(tab_idx, r.end)
                star = " ★" if r.must_see else ""
                # Use a Bar trace with custom hover info
                fig.add_trace(
                    go.Bar(
                        base=start_dt,
                        x=[end_dt - start_dt],
                        y=[r.stage],
                        orientation="h",
                        marker={"color": color, "line": {"color": border_color, "width": border_width}},
                        text=f"{r.artist}{star}",
                        textposition="inside",
                        insidetextanchor="middle",
                        textfont={"size": 11, "color": "#1a1a1a" if is_pick else "#666"},
                        hovertemplate=(
                            f"<b>{r.artist}</b>{star}<br>"
                            f"{r.start}–{r.end} · {r.stage}<br>"
                            f"Fit: {r.score:.2f}<br>"
                            f"<i>{r.reasoning}</i>"
                            "<extra></extra>"
                        ),
                        showlegend=False,
                    )
                )
            fig.update_layout(
                barmode="overlay",
                height=120 + len(stages) * 60,
                margin={"l": 0, "r": 0, "t": 24, "b": 12},
                xaxis={"type": "date", "tickformat": "%H:%M", "title": None, "showgrid": True, "gridcolor": "rgba(0,0,0,0.06)"},
                yaxis={"title": None, "categoryorder": "array", "categoryarray": list(reversed(stages))},
                # Solid backgrounds — fixes the "chart floats above the cards
                # below" glitch where the transparent chart visually overlaps
                # the picks expander.
                paper_bgcolor="#fffbf3",
                plot_bgcolor="#fffbf3",
                font={"family": "Georgia, serif", "color": "#2a2118"},
            )
            st.plotly_chart(
                fig,
                use_container_width=True,
                key=f"timeline_{day.day}_{tab_idx}",
                # Hide the Plotly modebar (zoom/pan/etc) — it floats in the
                # top-right corner and looks like garbage on top of the chart.
                config={"displayModeBar": False, "responsive": True},
            )
            st.divider()

            # Mini stats strip under the timeline
            cols = st.columns([1, 1, 1, 2])
            cols[0].metric("Picks", len(day.picks))
            cols[1].metric("Conflicts skipped", len(day.skipped_due_to_conflict))
            if day.picks:
                avg = sum(p.score for p in day.picks) / len(day.picks)
                cols[2].metric("Avg fit", f"{avg:.2f}")
            cols[3].markdown(" ".join(f'<span class="pick-pill">{p.start} {p.artist}</span>' for p in day.picks), unsafe_allow_html=True)

            # Detailed picks below the chart
            with st.expander("Why each pick", expanded=False):
                for p in day.picks:
                    star = "  ★" if p.must_see else ""
                    st.markdown(f"**{p.start}–{p.end}  ·  {p.artist}{star}**  *({p.stage})*  · fit `{p.score:.2f}`")
                    st.markdown(f"_{p.reasoning}_")
                    st.markdown("")
            if day.skipped_due_to_conflict:
                with st.expander(f"Skipped due to conflicts ({len(day.skipped_due_to_conflict)})"):
                    for s in day.skipped_due_to_conflict[:8]:
                        st.markdown(
                            f"- `{s.start}–{s.end}` **{s.artist}** ({s.stage}) — fit `{s.score:.2f}`\n  *{s.reasoning}*"
                        )
