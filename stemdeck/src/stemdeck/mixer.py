"""The live mix board — the in-memory state of what is playing right now.

State is keyed per *individual track* (`song_id`, index into `song.tracks`),
not per channel — a song can have several tracks in one channel (three
lead synths, two vocal layers) and every one is independently mixable.

Bringing a harmonic track in from one song auto-ducks the same-channel
harmonic tracks of any *other* song that clashes on the Camelot wheel, so
whatever the performer taps stays musically safe.
"""

from __future__ import annotations

from pydantic import BaseModel

from stemdeck.compat import camelot_relation
from stemdeck.models import HARMONIC, Catalog, Channel, Track


class MixEvent(BaseModel):
    """One thing that happened as a result of a board action."""

    song_id: str
    track_index: int  # index into the song's `tracks` list
    channel: Channel
    track_name: str
    action: str  # "in" or "out"
    reason: str  # human-readable why — for the transition log / UI


# Only a true Camelot clash auto-ducks. "energy"/"risky" moves stay the
# performer's call — `channel_safety` flags them "caution", not "clash".
_CLASHING_RELATIONS = {"clash"}


class MixBoard:
    """Mutable mix state. Not a pydantic model — it is live performance state."""

    def __init__(self, catalog: Catalog) -> None:
        self.catalog = catalog
        self._active: set[tuple[str, int]] = set()  # (song_id, track_index)

    # -- internal -----------------------------------------------------------

    def _track(self, song_id: str, track_index: int) -> tuple[object, Track | None]:
        song = self.catalog.get(song_id)
        if song is None or not (0 <= track_index < len(song.tracks)):
            return song, None
        return song, song.tracks[track_index]

    # -- queries ------------------------------------------------------------

    def is_active(self, song_id: str, track_index: int) -> bool:
        return (song_id, track_index) in self._active

    def active_cells(self) -> set[tuple[str, int]]:
        return set(self._active)

    def now_playing(self) -> list[str]:
        """Song ids with at least one active track, in catalog order."""
        live = {song_id for song_id, _ in self._active}
        return [s.id for s in self.catalog.songs if s.id in live]

    def active_tracks(self, song_id: str) -> set[int]:
        return {idx for sid, idx in self._active if sid == song_id}

    def active_channels(self, song_id: str) -> set[Channel]:
        """Distinct channels a song currently has at least one active track in."""
        song = self.catalog.get(song_id)
        if song is None:
            return set()
        return {
            song.tracks[idx].channel
            for idx in self.active_tracks(song_id)
            if idx < len(song.tracks)
        }

    # -- mutations ----------------------------------------------------------

    def bring_in(self, song_id: str, track_index: int) -> list[MixEvent]:
        """Activate a track. Auto-ducks clashing harmonic tracks of other songs."""
        song, track = self._track(song_id, track_index)
        if track is None:
            raise ValueError(f"Unknown track: {song_id!r}[{track_index}]")
        events: list[MixEvent] = []
        if (song_id, track_index) not in self._active:
            self._active.add((song_id, track_index))
            events.append(MixEvent(
                song_id=song_id, track_index=track_index, channel=track.channel,
                track_name=track.name, action="in", reason="brought in",
            ))

        # A harmonic track clashes with the same-channel harmonic tracks of
        # any other playing song whose key is incompatible — duck those.
        if track.channel in HARMONIC:
            for other_id, other_idx in list(self._active):
                if other_id == song_id:
                    continue
                other_song, other_track = self._track(other_id, other_idx)
                if other_track is None or other_track.channel != track.channel:
                    continue
                relation = camelot_relation(other_song.camelot, song.camelot)
                if relation in _CLASHING_RELATIONS:
                    self._active.discard((other_id, other_idx))
                    events.append(MixEvent(
                        song_id=other_id, track_index=other_idx,
                        channel=other_track.channel, track_name=other_track.name,
                        action="out",
                        reason=f"auto-ducked — {relation} against {song.title}",
                    ))
        return events

    def bring_out(self, song_id: str, track_index: int) -> list[MixEvent]:
        """Deactivate a track."""
        if (song_id, track_index) not in self._active:
            return []
        self._active.discard((song_id, track_index))
        _, track = self._track(song_id, track_index)
        return [MixEvent(
            song_id=song_id, track_index=track_index,
            channel=track.channel if track else Channel.FX,
            track_name=track.name if track else "",
            action="out", reason="brought out",
        )]

    def toggle(self, song_id: str, track_index: int) -> list[MixEvent]:
        if self.is_active(song_id, track_index):
            return self.bring_out(song_id, track_index)
        return self.bring_in(song_id, track_index)

    def restore(self, cells: set[tuple[str, int]]) -> None:
        """Re-activate cells from saved state, skipping any that no longer exist.

        Tolerates malformed entries (e.g. state saved by an older build that
        keyed cells differently) — those are simply dropped.
        """
        for cell in cells:
            if not isinstance(cell, tuple) or len(cell) != 2:
                continue
            song_id, track_index = cell
            if not isinstance(track_index, int):
                continue
            _, track = self._track(song_id, track_index)
            if track is not None:
                self._active.add((song_id, track_index))

    def clear(self) -> None:
        self._active.clear()
