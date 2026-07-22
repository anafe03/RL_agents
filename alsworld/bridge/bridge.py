"""ALS World — Ableton → browser live bridge.

Ableton speaks OSC (via the AbletonOSC remote script) over UDP. Browsers
cannot receive UDP, so this process sits in the middle:

    Ableton Live ──OSC/UDP──▶ this bridge ──WebSocket──▶ ALS World (browser)

It subscribes to Ableton's transport (tempo, play state, song position,
beat) and polls each track's output meter, then pushes a compact JSON state
to any connected browser at ~30 Hz. The browser uses song position to move
the wanderer through the world parsed from your .als, and the per-track
meter levels to make each stem's region react in real time.

Run:  python bridge.py            (defaults match AbletonOSC + ws :8765)
Deps: pip install python-osc websockets   (or: uv pip install ...)

Prereq: install AbletonOSC (https://github.com/ideoforms/AbletonOSC) as a
Control Surface in Ableton → Preferences → Link/MIDI. Then start this
bridge, then open alsworld/index.html and click "connect live".
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time

try:
    from pythonosc.dispatcher import Dispatcher
    from pythonosc.osc_server import AsyncIOOSCUDPServer
    from pythonosc.udp_client import SimpleUDPClient
except ImportError:  # pragma: no cover
    raise SystemExit(
        "python-osc is required:  pip install python-osc websockets"
    )

try:
    import websockets
except ImportError:  # pragma: no cover
    raise SystemExit(
        "websockets is required:  pip install python-osc websockets"
    )


# Shared, mutable snapshot the WebSocket side serialises each tick.
STATE: dict = {
    "connected": False,      # have we heard anything back from Ableton yet
    "playing": False,
    "tempo": 120.0,
    "beat": 0.0,             # song position in beats (drives the playhead)
    "num_tracks": 0,
    "track_names": {},       # index -> name
    "levels": {},            # index -> 0..1 output meter
    "last_rx": 0.0,          # monotonic time of last OSC reply
}


class Ableton:
    """Thin AbletonOSC driver: registers listeners and polls meters."""

    def __init__(self, host: str, send_port: int) -> None:
        self.client = SimpleUDPClient(host, send_port)

    def send(self, address: str, *args) -> None:
        self.client.send_message(address, list(args))

    def subscribe_transport(self) -> None:
        # Push-based listeners — Ableton emits on change.
        self.send("/live/song/start_listen/tempo")
        self.send("/live/song/start_listen/is_playing")
        self.send("/live/song/start_listen/current_song_time")
        self.send("/live/song/start_listen/beat")
        # One-shot queries to seed track list.
        self.send("/live/song/get/num_tracks")

    def request_track_names(self, n: int) -> None:
        for i in range(n):
            self.send("/live/track/get/name", i)

    def poll_meters(self, n: int) -> None:
        for i in range(n):
            self.send("/live/track/get/output_meter_level", i)


# ----- OSC reply handlers (Ableton -> bridge) -------------------------------

def _mark_rx() -> None:
    STATE["connected"] = True
    STATE["last_rx"] = time.monotonic()


def on_tempo(_addr, *args):
    if args:
        STATE["tempo"] = float(args[0])
    _mark_rx()


def on_is_playing(_addr, *args):
    if args:
        STATE["playing"] = bool(int(args[0]))
    _mark_rx()


def on_song_time(_addr, *args):
    if args:
        STATE["beat"] = float(args[0])
    _mark_rx()


def on_beat(_addr, *args):
    # integer beat counter — handy for a discrete on-beat pulse
    if args:
        STATE["beat_int"] = int(args[0])
    _mark_rx()


def on_num_tracks(_addr, *args):
    if args:
        STATE["num_tracks"] = int(args[0])
    _mark_rx()


def on_track_name(_addr, *args):
    # AbletonOSC replies: [track_index, name]
    if len(args) >= 2:
        STATE["track_names"][int(args[0])] = str(args[1])
    _mark_rx()


def on_meter(_addr, *args):
    # [track_index, level] — level is 0..1 (peak of the track output)
    if len(args) >= 2:
        STATE["levels"][int(args[0])] = float(args[1])
    _mark_rx()


def build_dispatcher() -> "Dispatcher":
    d = Dispatcher()
    d.map("/live/song/get/tempo", on_tempo)
    d.map("/live/song/get/is_playing", on_is_playing)
    d.map("/live/song/get/current_song_time", on_song_time)
    d.map("/live/song/get/song_time", on_song_time)
    d.map("/live/song/beat", on_beat)
    d.map("/live/song/get/num_tracks", on_num_tracks)
    d.map("/live/track/get/name", on_track_name)
    d.map("/live/track/get/output_meter_level", on_meter)
    d.set_default_handler(lambda *a: _mark_rx())
    return d


# ----- main async wiring ----------------------------------------------------

async def poll_loop(ab: Ableton, hz: float) -> None:
    """Seed the track list once we know the count, then poll meters forever."""
    seeded = False
    while True:
        n = STATE["num_tracks"]
        if n and not seeded:
            ab.request_track_names(n)
            seeded = True
        if n:
            ab.poll_meters(n)
        # if we lost Ableton, keep re-subscribing so a restart reconnects
        if time.monotonic() - STATE["last_rx"] > 2.0:
            STATE["connected"] = False
            ab.subscribe_transport()
        await asyncio.sleep(1.0 / hz)


async def ws_handler(conn) -> None:
    print("● browser connected")
    try:
        while True:
            payload = {
                "connected": STATE["connected"],
                "playing": STATE["playing"],
                "tempo": STATE["tempo"],
                "beat": STATE["beat"],
                "numTracks": STATE["num_tracks"],
                "tracks": [
                    {
                        "index": i,
                        "name": STATE["track_names"].get(i, f"Track {i+1}"),
                        "level": STATE["levels"].get(i, 0.0),
                    }
                    for i in range(STATE["num_tracks"])
                ],
            }
            await conn.send(json.dumps(payload))
            await asyncio.sleep(1.0 / 30.0)
    except Exception:
        print("○ browser disconnected")


async def main() -> None:
    p = argparse.ArgumentParser(description="ALS World Ableton→browser bridge")
    p.add_argument("--ableton-host", default="127.0.0.1")
    p.add_argument("--send-port", type=int, default=11000, help="AbletonOSC receive port")
    p.add_argument("--recv-port", type=int, default=11001, help="AbletonOSC reply port (we listen here)")
    p.add_argument("--ws-port", type=int, default=8765)
    p.add_argument("--poll-hz", type=float, default=30.0)
    args = p.parse_args()

    ab = Ableton(args.ableton_host, args.send_port)
    ab.subscribe_transport()

    dispatcher = build_dispatcher()
    loop = asyncio.get_running_loop()
    osc = AsyncIOOSCUDPServer((args.ableton_host, args.recv_port), dispatcher, loop)
    transport, _protocol = await osc.create_serve_endpoint()

    print(f"▸ listening for Ableton on {args.ableton_host}:{args.recv_port}")
    print(f"▸ sending to AbletonOSC on {args.ableton_host}:{args.send_port}")
    print(f"▸ browser WebSocket on ws://localhost:{args.ws_port}")
    print("  (open alsworld/index.html and click 'connect live')")

    try:
        async with websockets.serve(ws_handler, "localhost", args.ws_port):
            await poll_loop(ab, args.poll_hz)
    finally:
        transport.close()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nbye")
