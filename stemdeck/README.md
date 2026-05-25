# stemdeck

**Live element-level mashup tool for an Ableton catalog.**

Point stemdeck at a folder of Ableton `.als` projects. It parses each one,
detects key and BPM, and gives you a live mixer board where every *channel*
is one instrument across *all* your songs — so you can keep song A's drums
and vocals playing while you bring in song B's bassline, beat-quantized and
key-matched, mid-performance.

## Why this exists

DJ software (Traktor, Rekordbox, Serato) mixes two *stereo audio decks*.
Once a song is bounced to a stereo file, "just the bassline" is gone — you
can EQ and filter, but you cannot isolate stems. Algorithmic stem-separation
modes fake four stems from finished audio.

Ableton's Session View *can* do true multitrack mashups — but it has no live
UI for it. Switching which song's elements are exposed is manual, slow, and
has to be rehearsed.

stemdeck is the missing piece: a mixer board that knows about song
boundaries and shows you, in real time, which element combinations are
musically safe.

```
                 Kick   Snare  Hats   Bass   Lead   Pad    Vocal  FX
Drift   (8A 124) [ ON ] [ ON ] [ ON ] [ ON ] [    ] [ ON ] [ ON ] [   ]   playing
Hollow  (9A 122) [ -- ] [ -- ] [ -- ] [ ok ] [ ok ] [ ok ] [ ok ] [-- ]   queued, key-safe
Glass  (11A 125) [ -- ] [ -- ] [ -- ] [!!! ] [!!! ] [!!! ] [!!! ] [-- ]   queued, key clash
```

Tap a cell → that element fades in on the next bar; clashing harmonic
channels of other songs auto-duck. Drums layer freely (BPM handles them);
harmonic channels respect the Camelot wheel.

## Quickstart

```bash
cd stemdeck
uv sync
uv run streamlit run src/stemdeck/ui/app.py    # the demo UI
uv run stemdeck demo                            # compatibility matrix in the terminal
```

## Demo vs. live

- **Demo mode** (hosted, no Ableton): loads a bundled catalog of synthetic
  songs, shows the compatibility graph, the mixer board, and a *simulated*
  beat-quantized transition timeline. Fully exercisable without Ableton.
- **Live mode** (local rig only): `uv sync --extra live` adds `python-osc`.
  With Ableton + the AbletonOSC remote script running, stemdeck drives the
  real instance — tempo ramps, scene launches, per-track crossfades.

## Connecting stemdeck to Ableton (Live mode)

stemdeck talks to Ableton over OSC using the open-source
[**AbletonOSC**](https://github.com/ideoforms/AbletonOSC) remote script.
End-to-end setup:

### 1. Install AbletonOSC in Ableton

Download or `git clone` AbletonOSC, then drop the `AbletonOSC` folder into
Ableton's User Library `Remote Scripts` directory:

- **macOS:** `~/Music/Ableton/User Library/Remote Scripts/AbletonOSC/`
- **Windows:** `C:\Users\<you>\Documents\Ableton\User Library\Remote Scripts\AbletonOSC\`

(Putting it under the User Library — rather than the Ableton.app bundle —
means it survives Ableton updates.)

### 2. Enable it as a Control Surface

Restart Ableton. Then:

- Open **Preferences → Link, Tempo & MIDI**
- Under **Control Surface**, pick `AbletonOSC` from the dropdown
- Leave Input/Output set to None

Ableton is now listening for OSC on **port 11000** and replying on 11001.

### 3. Install stemdeck's live extra

```bash
cd stemdeck
uv sync --extra live              # adds python-osc
uv run streamlit run src/stemdeck/ui/app.py
```

### 4. Connect from stemdeck

In the sidebar:

- **Mode:** `Live (AbletonOSC)`
- **Host:** `127.0.0.1`
- **Port:** `11000`
- Click **Connect** — you should see `● OSC ready → 127.0.0.1:11000`.

Verify the link: set a tempo in the sidebar and click **Send tempo** — if
Ableton's master tempo changes, you're talking to the DAW.

### 5. Load your songs in Ableton

The `.als` projects you upload to stemdeck only feed the *analysis* —
stemdeck doesn't play audio itself. To hear the mix, your Ableton Set
needs the same audio/MIDI tracks loaded (open the songs, or build a
performance template that contains them).

### Known limitation — track mapping

Cell taps currently send `set_track_mute` using each track's index in
stemdeck's catalog. That maps cleanly only when your Ableton Set's track
order matches the catalog. A per-project mapping screen
(stemdeck-track → Ableton-track) is the next planned upgrade.

## Architecture

| Module | Responsibility |
|---|---|
| `parser.py`   | Decompress + parse `.als` (gzipped XML) into a `Song` |
| `mapping.py`  | Map raw track names → canonical channels (Kick, Bass, …) |
| `analyzer.py` | Key detection (Krumhansl-Schmuckler), energy estimate |
| `compat.py`   | Camelot-wheel key relations, BPM relations, pair scoring |
| `mixer.py`    | The live mix board — bring channels in/out, auto-duck clashes |
| `osc_bridge.py` | Live AbletonOSC control (optional, local only) |
| `mock.py`     | Bundled demo catalog loader |

## Status

v0.1 — alpha. Core analysis + compatibility + mix simulation work and are
tested. The `.als` parser is best-effort against the modern Ableton schema
(the format is gzipped XML and shifts between Live versions). LLM-backed
fuzzy track mapping and the live OSC bridge are scaffolded next.

All catalog songs in `data/catalog/` are synthetic. This is a portfolio
project, not a released product.
