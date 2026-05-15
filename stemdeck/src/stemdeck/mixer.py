"""The live mix board — the in-memory state of what is playing right now.

A `MixBoard` tracks the set of active (song, channel) cells. Bringing a
harmonic channel in from one song auto-ducks the clashing harmonic channels
of any *other* song that is currently playing, so whatever the performer
taps stays musically safe.
"""

from __future__ import annotations

from pydantic import BaseModel

from stemdeck.compat import camelot_relation
from stemdeck.models import HARMONIC, Catalog, Channel


class MixEvent(BaseModel):
    """One thing that happened as a result of a board action."""

    song_id: str
    channel: Channel
    action: str  # "in" or "out"
    reason: str  # human-readable why — for the transition log / UI


# Only a true Camelot clash auto-ducks. "energy"/"risky" moves stay the
# performer's call — `channel_safety` flags them "caution", not "clash".
_CLASHING_RELATIONS = {"clash"}


class MixBoard:
    """Mutable mix state. Not a pydantic model — it is live performance state."""

    def __init__(self, catalog: Catalog) -> None:
        self.catalog = catalog
        self._active: set[tuple[str, Channel]] = set()

    # -- queries ------------------------------------------------------------

    def is_active(self, song_id: str, channel: Channel) -> bool:
        return (song_id, channel) in self._active

    def active_cells(self) -> set[tuple[str, Channel]]:
        return set(self._active)

    def now_playing(self) -> list[str]:
        """Song ids that have at least one active channel, catalog order."""
        live = {song_id for song_id, _ in self._active}
        return [s.id for s in self.catalog.songs if s.id in live]

    def active_channels(self, song_id: str) -> set[Channel]:
        return {ch for sid, ch in self._active if sid == song_id}

    # -- mutations ----------------------------------------------------------

    def bring_in(self, song_id: str, channel: Channel) -> list[MixEvent]:
        """Activate a cell. Auto-ducks clashing harmonic channels of other songs."""
        if self.catalog.get(song_id) is None:
            raise ValueError(f"Unknown song: {song_id!r}")
        events: list[MixEvent] = []
        if (song_id, channel) not in self._active:
            self._active.add((song_id, channel))
            events.append(MixEvent(song_id=song_id, channel=channel, action="in",
                                   reason="brought in"))

        # A harmonic channel can clash with the same-type channel of another
        # playing song — duck those automatically.
        if channel in HARMONIC:
            incoming = self.catalog.get(song_id)
            for other_id, other_ch in list(self._active):
                if other_id == song_id or other_ch != channel:
                    continue
                other = self.catalog.get(other_id)
                if other is None:
                    continue
                relation = camelot_relation(other.camelot, incoming.camelot)
                if relation in _CLASHING_RELATIONS:
                    self._active.discard((other_id, other_ch))
                    events.append(MixEvent(
                        song_id=other_id, channel=other_ch, action="out",
                        reason=f"auto-ducked — {relation} against {song_id}",
                    ))
        return events

    def bring_out(self, song_id: str, channel: Channel) -> list[MixEvent]:
        """Deactivate a cell."""
        if (song_id, channel) in self._active:
            self._active.discard((song_id, channel))
            return [MixEvent(song_id=song_id, channel=channel, action="out",
                             reason="brought out")]
        return []

    def toggle(self, song_id: str, channel: Channel) -> list[MixEvent]:
        if self.is_active(song_id, channel):
            return self.bring_out(song_id, channel)
        return self.bring_in(song_id, channel)

    def clear(self) -> None:
        self._active.clear()
