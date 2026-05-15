"""Live Ableton control over OSC (optional — local performance rig only).

This module talks to a running Ableton instance via the AbletonOSC remote
script. It is intentionally a thin wrapper: the musical decisions live in
`compat` and `mixer`; this just executes them on the real DAW.

`python-osc` is an optional dependency (`uv sync --extra live`). Importing
this module without it raises a clear error only when you actually try to
connect — so the hosted demo, which never calls it, stays clean.
"""

from __future__ import annotations


class AbletonOSC:
    """Minimal AbletonOSC client.

    Defaults match AbletonOSC's out-of-the-box ports. The host runs Ableton
    on localhost in a real rig, so there is no network latency to manage.
    """

    def __init__(self, host: str = "127.0.0.1", send_port: int = 11000) -> None:
        self.host = host
        self.send_port = send_port
        self._client = None

    def connect(self) -> None:
        try:
            from pythonosc.udp_client import SimpleUDPClient
        except ImportError as exc:  # pragma: no cover - exercised only on a live rig
            raise RuntimeError(
                "Live mode needs python-osc. Install it with "
                "`uv sync --extra live`."
            ) from exc
        self._client = SimpleUDPClient(self.host, self.send_port)

    def _send(self, address: str, *args: object) -> None:
        if self._client is None:
            raise RuntimeError("Not connected — call connect() first.")
        self._client.send_message(address, list(args))

    # -- transport ----------------------------------------------------------

    def set_tempo(self, bpm: float) -> None:
        """Set master tempo. A smooth ramp is the caller's job — step it."""
        self._send("/live/song/set/tempo", bpm)

    def fire_scene(self, scene_index: int) -> None:
        """Launch a scene (a row). Ableton quantizes the launch to the bar."""
        self._send("/live/scene/fire", scene_index)

    # -- per-channel mixing -------------------------------------------------

    def set_track_volume(self, track_index: int, volume: float) -> None:
        """Set a track's mixer volume, 0.0..1.0 — used for crossfades."""
        self._send("/live/track/set/volume", track_index, volume)

    def set_track_mute(self, track_index: int, muted: bool) -> None:
        self._send("/live/track/set/mute", track_index, int(muted))
