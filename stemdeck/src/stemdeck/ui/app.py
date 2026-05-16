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

from stemdeck.analyzer import analyze
from stemdeck.compat import channel_safety, rank_next
from stemdeck.match import track_match
from stemdeck.mixer import MixBoard
from stemdeck.mock import demo_catalog
from stemdeck.models import CHANNEL_ORDER, Catalog, Channel, Song
from stemdeck.parser import parse_als_bytes

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

if "active_cells" not in st.session_state:
    st.session_state.active_cells = set()  # set of (song_id, channel_value)
if "log" not in st.session_state:
    st.session_state.log = []


@st.cache_data(show_spinner=False)
def _parse_upload(name: str, data: bytes) -> Song | None:
    """Parse + analyze one uploaded .als. Cached on (name, bytes) so cell
    clicks don't re-parse. Returns None if the file can't be read."""
    try:
        return analyze(parse_als_bytes(data, name))
    except Exception:  # noqa: BLE001 - a bad upload shouldn't crash the app
        return None


# Camelot wheel colors — 12 hues around the wheel. Adjacent Camelot
# numbers get adjacent colors, so visually-similar = key-compatible.
_CAMELOT_HUES = {
    1: "#1bb5a0", 2: "#1b9fc4", 3: "#3d7fd6", 4: "#6a5fd0",
    5: "#9b4fc9", 6: "#c94fae", 7: "#d65478", 8: "#db6f4f",
    9: "#d69a3f", 10: "#c2be3f", 11: "#7fc24a", 12: "#3fc270",
}


def _camelot_color(camelot: str) -> str:
    code = (camelot or "").strip().upper()
    if len(code) >= 2 and code[:-1].isdigit():
        return _CAMELOT_HUES.get(int(code[:-1]), "#5d6b7a")
    return "#5d6b7a"


def _rhythm_strip(rhythm: list[int], color: str = ACCENT) -> str:
    """Render a 16-step onset grid as a row of small blocks (HTML)."""
    grid = list(rhythm or []) + [0] * (16 - len(rhythm or []))
    cells = []
    for i in range(16):
        if grid[i]:
            bg = color
        else:
            bg = "#222d39" if i % 4 == 0 else "#161b22"  # mark downbeats
        cells.append(
            f"<span style='display:inline-block;width:8px;height:13px;"
            f"background:{bg};margin-right:2px;border-radius:1px'></span>"
        )
    return "".join(cells)


# --- sidebar ----------------------------------------------------------------

with st.sidebar:
    st.markdown("# 🎛️ stemdeck")
    st.caption("Live element-level mashup tool for an Ableton catalog.")
    st.markdown("---")

    source = st.radio(
        "Catalog source",
        ["Demo catalog", "Upload my .als files"],
        help="Upload your own Ableton projects — parsing is pure Python, so "
        "it works right here in the hosted app. No Ableton needed for analysis.",
    )
    if source == "Upload my .als files":
        uploads = st.file_uploader(
            "Drop Ableton .als project files",
            type=["als"],
            accept_multiple_files=True,
        )
        songs = []
        failed = []
        for uf in uploads or []:
            parsed = _parse_upload(uf.name, uf.getvalue())
            (songs.append(parsed) if parsed is not None else failed.append(uf.name))
        catalog = Catalog(songs=songs)
        if failed:
            st.warning("Couldn't parse: " + ", ".join(failed))
    else:
        catalog = demo_catalog()

    st.markdown("---")
    st.radio(
        "Mode",
        ["Demo (simulate the mix)", "Live (AbletonOSC, local rig)"],
        help="Demo mode simulates the mix. Live mode drives a running Ableton "
        "instance over OSC — needs `uv sync --extra live` and a local rig.",
    )
    st.info(
        "Live OSC control is not available in this hosted demo. Catalog "
        "analysis + mix simulation run fully here."
    )
    st.markdown("---")
    if catalog.songs:
        st.markdown(f"**Catalog:** {len(catalog.songs)} songs")
        for s in catalog.songs:
            dot = _camelot_color(s.camelot)
            st.markdown(
                f"<div style='font-family:monospace;font-size:0.74rem;color:#8e9aa8'>"
                f"<span style='color:{dot}'>●</span> "
                f"<b style='color:#cbd4dd'>{s.camelot or '?'}</b> "
                f"{s.key or 'key ?'} · {s.bpm:g} BPM · {s.title}</div>",
                unsafe_allow_html=True,
            )
    if st.button("Reset mix", use_container_width=True):
        st.session_state.active_cells = set()
        st.session_state.log = []
        st.rerun()
    st.markdown(f"[GitHub repo]({GITHUB_URL})")


if not catalog.songs:
    st.info(
        "👈 Upload one or more Ableton `.als` files in the sidebar to build "
        "your catalog — or switch back to the demo catalog."
    )
    st.stop()

board = MixBoard(catalog)
# restore() drops cells pointing at songs/tracks no longer in the catalog.
board.restore(st.session_state.active_cells)


def _persist() -> None:
    st.session_state.active_cells = board.active_cells()


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
st.caption(
    "stemdeck controls the **mix plan** — which tracks are live and how they "
    "layer. It doesn't synthesize audio: tapping a cell sets mix state, it "
    "doesn't make sound here. In Live mode it drives Ableton over OSC and "
    "Ableton plays the audio."
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

# Pre-compute every track cell's visual state so the CSS can be injected
# up front. state ∈ {"active", "neutral", "safe", "caution", "clash"}.
# Cells are keyed per individual track (song_id, track index).
cell_state: dict[tuple[str, int], str] = {}
for song in catalog.songs:
    safety = {}
    if anchor is not None and song.id != anchor.id:
        safety = channel_safety(anchor, song)
    for idx, track in enumerate(song.tracks):
        if board.is_active(song.id, idx):
            cell_state[(song.id, idx)] = "active"
        elif anchor is None or song.id == anchor.id:
            cell_state[(song.id, idx)] = "neutral"
        else:
            cell_state[(song.id, idx)] = safety.get(track.channel, "neutral")

_CELL_COLORS = {
    "active": (ACCENT, "#06140d"),
    "neutral": ("#1d2530", "#aeb9c6"),
    "safe": ("#163a2a", "#46e0a0"),
    "caution": ("#3d3416", "#e8c352"),
    "clash": ("#3d1a1a", "#ef6b6b"),
}

# Inject per-cell button colors keyed by the widget key.
css_rules = []
for (song_id, idx), state in cell_state.items():
    bg, fg = _CELL_COLORS[state]
    key = f"cell_{song_id}_{idx}"
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
        n = len(board.active_tracks(sid))
        tag = "anchor" if anchor and sid == anchor.id else f"{n} trk"
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

clicked: tuple[str, int] | None = None
for song in catalog.songs:
    row = st.columns([3] + [1] * len(CHANNEL_ORDER))
    cam_color = _camelot_color(song.camelot)
    row[0].markdown(
        f"<div style='font-family:monospace;font-size:0.82rem;color:#dde4ea;"
        f"padding-top:0.3rem'><b>{song.title}</b><br>"
        f"<span style='display:inline-block;background:{cam_color};color:#0a0c0e;"
        f"font-weight:700;font-size:0.7rem;padding:0.12rem 0.45rem;border-radius:4px;"
        f"margin-right:0.3rem'>{song.camelot or '?'}</span>"
        f"<span class='chip'>{song.key or 'key ?'}</span>"
        f"<span class='chip'>{song.bpm:g}</span>"
        f"<span class='chip'>E{song.energy}</span></div>",
        unsafe_allow_html=True,
    )
    for i, ch in enumerate(CHANNEL_ORDER):
        # Every track of this song mapped to this channel — stacked.
        tracks_in_channel = [
            (idx, t) for idx, t in enumerate(song.tracks) if t.channel == ch
        ]
        if not tracks_in_channel:
            row[i + 1].button("·", key=f"empty_{song.id}_{ch.value}",
                              disabled=True, use_container_width=True)
            continue
        for idx, track in tracks_in_channel:
            state = cell_state[(song.id, idx)]
            short = track.name[:11] if track.name else ch.value
            label = ("● " + short) if state == "active" else short
            if row[i + 1].button(label, key=f"cell_{song.id}_{idx}",
                                 use_container_width=True,
                                 help=f"{track.name} · {ch.value}"):
                clicked = (song.id, idx)

if clicked is not None:
    song_id, track_idx = clicked
    events = board.toggle(song_id, track_idx)
    for ev in events:
        verb = "▲ in " if ev.action == "in" else "▼ out"
        st.session_state.log.insert(
            0,
            f"{verb}  {catalog.get(ev.song_id).title} · {ev.track_name} "
            f"({ev.channel.value}) — {ev.reason}",
        )
    st.session_state.log = st.session_state.log[:12]
    _persist()
    st.rerun()


# --- layer analysis ---------------------------------------------------------
# Whenever two different songs have a track active on the same channel at
# once, that is a real layer the performer is hearing — score the match.

_by_channel: dict[Channel, list[tuple[str, int]]] = {}
for sid, idx in board.active_cells():
    song = catalog.get(sid)
    if song is None or idx >= len(song.tracks):
        continue
    _by_channel.setdefault(song.tracks[idx].channel, []).append((sid, idx))

layer_rows: list[tuple[Channel, object, object, object, object, object]] = []
for ch in CHANNEL_ORDER:
    cells = sorted(_by_channel.get(ch, []))
    for a in range(len(cells)):
        for b in range(a + 1, len(cells)):
            sid_a, idx_a = cells[a]
            sid_b, idx_b = cells[b]
            if sid_a == sid_b:
                continue  # two tracks of one song playing together is just the song
            song_a = catalog.get(sid_a)
            song_b = catalog.get(sid_b)
            track_a = song_a.tracks[idx_a]
            track_b = song_b.tracks[idx_b]
            layer_rows.append(
                (ch, song_a, track_a, song_b, track_b, track_match(track_a, track_b))
            )

_VERDICT_COLOR = {
    "locked": ACCENT,
    "blends": "#46e0a0",
    "loose": "#e8c352",
    "clash": "#ef6b6b",
}

st.markdown("##### Layer analysis")
if not layer_rows:
    st.caption("Layer two tracks on the same channel (across songs) to measure "
               "their rhythmic + harmonic match.")
else:
    for ch, song_a, track_a, song_b, track_b, score in layer_rows:
        rhy = "n/a" if score.rhythmic is None else f"{score.rhythmic:.2f}"
        har = "n/a" if score.harmonic is None else f"{score.harmonic:.2f}"
        color = _VERDICT_COLOR[score.verdict]
        st.markdown(
            f"<div style='font-family:monospace;font-size:0.8rem;color:#cbd4dd;"
            f"background:#11151a;border:1px solid #1d2530;border-left:3px solid {color};"
            f"border-radius:5px;padding:0.45rem 0.7rem;margin-bottom:0.35rem'>"
            f"<span class='chip'>{ch.value.upper()}</span> "
            f"<b>{track_a.name}</b> <span style='color:#5d6b7a'>({song_a.title})</span> "
            f"&nbsp;⊕&nbsp; <b>{track_b.name}</b> "
            f"<span style='color:#5d6b7a'>({song_b.title})</span>"
            f"<div style='margin:0.4rem 0 0.45rem 0'>"
            f"<div style='margin-bottom:3px'>{_rhythm_strip(track_a.rhythm)}"
            f"<span style='color:#5d6b7a;margin-left:6px'>{track_a.name}</span></div>"
            f"<div>{_rhythm_strip(track_b.rhythm, color='#46a0e0')}"
            f"<span style='color:#5d6b7a;margin-left:6px'>{track_b.name}</span></div>"
            f"</div>"
            f"<span style='color:#7d8a99'>rhythmic</span> {rhy} &nbsp; "
            f"<span style='color:#7d8a99'>harmonic</span> {har} &nbsp; "
            f"<span style='color:#7d8a99'>overall</span> {score.overall:.2f} &nbsp; "
            f"<span style='color:{color};font-weight:700'>{score.verdict.upper()}</span>"
            f"</div>",
            unsafe_allow_html=True,
        )

st.markdown("")


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

# --- catalog detail ---------------------------------------------------------
# Every parsed track, so you can verify nothing was dropped on import.

_total_tracks = sum(len(s.tracks) for s in catalog.songs)
with st.expander(f"🎚 All parsed tracks — {_total_tracks} across {len(catalog.songs)} songs"):
    for song in catalog.songs:
        st.markdown(
            f"**{song.title}** — {song.key or 'key ?'} "
            f"(`{song.camelot or '?'}`) · {song.bpm:g} BPM · "
            f"energy {song.energy} · {len(song.tracks)} tracks"
        )
        for track in song.tracks:
            kind = "MIDI" if track.is_midi else "audio"
            pitch = f"{len(track.notes)} notes" if track.notes else "no pitch"
            st.markdown(
                f"<div style='font-family:monospace;font-size:0.74rem;color:#8e9aa8;"
                f"padding:0.1rem 0 0.25rem 0'>"
                f"<span class='chip'>{track.channel.value}</span> "
                f"<b style='color:#cbd4dd'>{track.name}</b> "
                f"<span style='color:#5d6b7a'>· {kind} · {pitch} · "
                f"{track.clip_count} clips</span><br>"
                f"<span style='display:inline-block;margin-top:0.2rem'>"
                f"{_rhythm_strip(track.rhythm)}</span></div>",
                unsafe_allow_html=True,
            )
        st.markdown("")

st.caption(
    "Drums (kick/snare/hats) layer freely; harmonic channels follow the "
    "Camelot wheel. A clash auto-ducks the conflicting track of the anchor "
    "song. A portfolio project — not a released product."
)
