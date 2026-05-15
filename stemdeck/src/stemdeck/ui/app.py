"""Streamlit UI for stemdeck — the live mixer board.

Run locally:
    cd stemdeck
    uv sync
    uv run streamlit run src/stemdeck/ui/app.py

The board is a grid: rows are songs, columns are canonical channels. Tap a
cell to bring that instrument in or out. Cells of queued songs are tinted
by harmonic safety against whatever is currently anchoring the mix.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Force the installed package to win over any bare `stemdeck/` namespace
# package at the repo root.
_SRC = Path(__file__).resolve().parents[2]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import streamlit as st

from stemdeck.compat import channel_safety, rank_next
from stemdeck.mixer import MixBoard
from stemdeck.mock import demo_catalog
from stemdeck.models import CHANNEL_ORDER, Channel

st.set_page_config(
    page_title="stemdeck",
    page_icon="🎛️",
    layout="wide",
    initial_sidebar_state="expanded",
)

GITHUB_URL = "https://github.com/anafe03/RL_agents/tree/main/stemdeck"

# DJ-deck aesthetic — near-black rack panel, neon-green accent, monospace
# technical chips. Leans into the repo's dark theme rather than fighting it.
ACCENT = "#22e58a"
st.markdown(
    f"""
    <style>
    .stApp {{ background-color: #0a0c0e; }}
    .deck-header {{
        background: linear-gradient(135deg, #11151a 0%, #0a0c0e 100%);
        border: 1px solid #1d2530;
        border-left: 4px solid {ACCENT};
        border-radius: 8px;
        padding: 1.1rem 1.4rem;
        margin-bottom: 1rem;
        font-family: "SF Mono", "JetBrains Mono", Menlo, monospace;
    }}
    .deck-header .title {{
        font-size: 1.7rem;
        font-weight: 800;
        color: {ACCENT};
        letter-spacing: 0.04em;
    }}
    .deck-header .sub {{ color: #7d8a99; font-size: 0.85rem; margin-top: 0.2rem; }}
    .chip {{
        display: inline-block;
        font-family: "SF Mono", Menlo, monospace;
        font-size: 0.72rem;
        padding: 0.12rem 0.5rem;
        border-radius: 4px;
        margin-right: 0.3rem;
        background: #1d2530;
        color: #aeb9c6;
    }}
    .chip.live {{ background: {ACCENT}; color: #06140d; font-weight: 700; }}
    .now-strip {{
        background: #11151a;
        border: 1px solid #1d2530;
        border-radius: 6px;
        padding: 0.6rem 0.9rem;
        font-family: "SF Mono", Menlo, monospace;
        font-size: 0.82rem;
        color: #aeb9c6;
        margin-bottom: 0.6rem;
    }}
    /* Channel-cell buttons are small + monospace. */
    div[class*="st-key-cell_"] button {{
        font-family: "SF Mono", Menlo, monospace !important;
        font-size: 0.7rem !important;
        padding: 0.2rem 0 !important;
        border-radius: 4px !important;
    }}
    div[class*="st-key-rowlabel_"] button {{
        font-family: "SF Mono", Menlo, monospace !important;
        font-size: 0.78rem !important;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)


# --- state ------------------------------------------------------------------

catalog = demo_catalog()

if "active_cells" not in st.session_state:
    st.session_state.active_cells = set()  # set of (song_id, channel_value)
if "log" not in st.session_state:
    st.session_state.log = []

board = MixBoard(catalog)
for song_id, ch_value in st.session_state.active_cells:
    board._active.add((song_id, Channel(ch_value)))


def _persist() -> None:
    st.session_state.active_cells = {(sid, ch.value) for sid, ch in board.active_cells()}


# --- sidebar ----------------------------------------------------------------

with st.sidebar:
    st.markdown("# 🎛️ stemdeck")
    st.caption("Live element-level mashup tool for an Ableton catalog.")
    st.markdown("---")
    st.radio(
        "Mode",
        ["Demo (synthetic catalog)", "Live (AbletonOSC, local rig)"],
        help="Demo mode simulates the mix. Live mode drives a running Ableton "
        "instance over OSC — needs `uv sync --extra live` and a local rig.",
    )
    st.info(
        "Live mode is not available in this hosted demo. It opens an OSC "
        "connection to Ableton on your own machine."
    )
    st.markdown("---")
    st.markdown(f"**Catalog:** {len(catalog.songs)} songs")
    for s in catalog.songs:
        st.caption(f"`{s.camelot:>3s}` · {s.bpm:g} BPM · {s.title}")
    st.markdown("---")
    if st.button("Reset mix", use_container_width=True):
        board.clear()
        st.session_state.log = []
        _persist()
        st.rerun()
    st.markdown(f"[GitHub repo]({GITHUB_URL})")


# --- header -----------------------------------------------------------------

st.markdown(
    """
    <div class="deck-header">
        <div class="title">STEMDECK</div>
        <div class="sub">tap a cell to bring an instrument in / out · harmonic
        channels of queued songs are tinted by key safety · drums layer freely</div>
    </div>
    """,
    unsafe_allow_html=True,
)


# --- determine the anchor song + per-cell state -----------------------------

playing_ids = board.now_playing()
# The anchor = the playing song with the most active channels.
anchor = None
if playing_ids:
    anchor = max(
        (catalog.get(sid) for sid in playing_ids),
        key=lambda s: len(board.active_channels(s.id)),
    )

# Pre-compute every cell's visual state so the CSS can be injected up front.
# state ∈ {"active", "neutral", "safe", "caution", "clash", "absent"}
cell_state: dict[tuple[str, Channel], str] = {}
for song in catalog.songs:
    safety = {}
    if anchor is not None and song.id != anchor.id:
        safety = channel_safety(anchor, song)
    for ch in CHANNEL_ORDER:
        if song.track_for(ch) is None:
            cell_state[(song.id, ch)] = "absent"
        elif board.is_active(song.id, ch):
            cell_state[(song.id, ch)] = "active"
        elif anchor is None or song.id == anchor.id:
            cell_state[(song.id, ch)] = "neutral"
        else:
            cell_state[(song.id, ch)] = safety.get(ch, "neutral")

_CELL_COLORS = {
    "active": (ACCENT, "#06140d"),
    "neutral": ("#1d2530", "#aeb9c6"),
    "safe": ("#163a2a", "#46e0a0"),
    "caution": ("#3d3416", "#e8c352"),
    "clash": ("#3d1a1a", "#ef6b6b"),
    "absent": ("#0e1115", "#3a4350"),
}

# Inject per-cell button colors keyed by the widget key.
css_rules = []
for (song_id, ch), state in cell_state.items():
    bg, fg = _CELL_COLORS[state]
    key = f"cell_{song_id}_{ch.value}"
    css_rules.append(
        f'div.st-key-{key} button {{ background-color: {bg} !important; '
        f'color: {fg} !important; border: 1px solid {bg} !important; }}'
    )
st.markdown("<style>" + "\n".join(css_rules) + "</style>", unsafe_allow_html=True)


# --- now-playing strip ------------------------------------------------------

if playing_ids:
    parts = []
    for sid in playing_ids:
        s = catalog.get(sid)
        n = len(board.active_channels(sid))
        tag = "anchor" if anchor and sid == anchor.id else f"{n} ch"
        parts.append(f"<b style='color:{ACCENT}'>{s.title}</b> ({s.camelot} · {s.bpm:g}) · {tag}")
    st.markdown(f'<div class="now-strip">▶ LIVE: {"  &nbsp;|&nbsp;  ".join(parts)}</div>',
                unsafe_allow_html=True)
else:
    st.markdown('<div class="now-strip">▶ LIVE: — nothing playing — tap a cell to start</div>',
                unsafe_allow_html=True)


# --- the mixer board --------------------------------------------------------

# Header row: blank label cell + channel names.
header_cols = st.columns([3] + [1] * len(CHANNEL_ORDER))
header_cols[0].markdown("&nbsp;")
for i, ch in enumerate(CHANNEL_ORDER):
    header_cols[i + 1].markdown(
        f"<div style='text-align:center;font-family:monospace;font-size:0.7rem;"
        f"color:#7d8a99'>{ch.value.upper()}</div>",
        unsafe_allow_html=True,
    )

clicked: tuple[str, Channel] | None = None
for song in catalog.songs:
    row = st.columns([3] + [1] * len(CHANNEL_ORDER))
    live = song.id in playing_ids
    chip_class = "chip live" if live else "chip"
    row[0].markdown(
        f"<div style='font-family:monospace;font-size:0.82rem;color:#dde4ea;"
        f"padding-top:0.35rem'><b>{song.title}</b><br>"
        f"<span class='{chip_class}'>{song.camelot}</span>"
        f"<span class='chip'>{song.bpm:g}</span>"
        f"<span class='chip'>E{song.energy}</span></div>",
        unsafe_allow_html=True,
    )
    for i, ch in enumerate(CHANNEL_ORDER):
        state = cell_state[(song.id, ch)]
        key = f"cell_{song.id}_{ch.value}"
        if state == "absent":
            row[i + 1].button("·", key=key, disabled=True, use_container_width=True)
            continue
        label = "●" if state == "active" else ("✕" if state == "clash" else "○")
        if row[i + 1].button(label, key=key, use_container_width=True):
            clicked = (song.id, ch)

if clicked is not None:
    song_id, ch = clicked
    events = board.toggle(song_id, ch)
    for ev in events:
        verb = "▲ in " if ev.action == "in" else "▼ out"
        st.session_state.log.insert(
            0, f"{verb}  {catalog.get(ev.song_id).title} · {ev.channel.value} — {ev.reason}"
        )
    st.session_state.log = st.session_state.log[:12]
    _persist()
    st.rerun()


# --- next options + transition log -----------------------------------------

col_next, col_log = st.columns(2)

with col_next:
    st.markdown("##### Next options")
    if anchor is None:
        st.caption("Start a mix to see compatible next songs.")
    else:
        st.caption(f"Ranked transitions out of **{anchor.title}**:")
        for p in rank_next(anchor, catalog.songs)[:3]:
            target = catalog.get(p.song_b)
            stars = "★" * p.stars + "☆" * (5 - p.stars)
            color = ACCENT if p.stars >= 4 else "#e8c352" if p.stars >= 3 else "#ef6b6b"
            st.markdown(
                f"<div style='font-family:monospace;font-size:0.8rem;color:#cbd4dd;"
                f"padding:0.25rem 0'>"
                f"<span style='color:{color}'>{stars}</span> &nbsp;"
                f"<b>{target.title}</b> &nbsp;"
                f"<span class='chip'>{target.camelot}</span>"
                f"<span class='chip'>{target.bpm:g}</span> — {p.note}</div>",
                unsafe_allow_html=True,
            )

with col_log:
    st.markdown("##### Transition log")
    if not st.session_state.log:
        st.caption("Cell actions and auto-ducks show up here.")
    else:
        for line in st.session_state.log:
            st.markdown(
                f"<div style='font-family:monospace;font-size:0.76rem;"
                f"color:#8e9aa8'>{line}</div>",
                unsafe_allow_html=True,
            )

st.divider()
st.caption(
    "Synthetic catalog — a portfolio project, not a released product. "
    "Drums (kick/snare/hats) layer freely; harmonic channels follow the "
    "Camelot wheel. A clash auto-ducks the conflicting channel of the anchor song."
)
