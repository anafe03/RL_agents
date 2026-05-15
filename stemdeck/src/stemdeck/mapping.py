"""Map raw Ableton track names to canonical channels.

Your "Operator Lead 2" and "Serum Lead Bright" both need to become the
single `LEAD` channel so they line up in the mixer board. v1 uses keyword
rules; the planned enhancement is an LLM-backed matcher for names the
heuristic misses (see README).
"""

from __future__ import annotations

from stemdeck.models import Channel

# Ordered most-specific-first. The first channel with a matching keyword wins.
_RULES: list[tuple[Channel, tuple[str, ...]]] = [
    (Channel.KICK, ("kick", "bassdrum", "bd ", " bd", "kik")),
    (Channel.SNARE, ("snare", "clap", "rimshot", "rim ", "sd ", " sd")),
    (Channel.HATS, ("hat", "hh", "cymbal", "ride", "shaker", "perc", "tamb")),
    (Channel.BASS, ("bass", "sub", "808", "reese", "wobble")),
    (Channel.VOCAL, ("vocal", "vox", "voice", "acapella", "choir", "adlib")),
    (Channel.PAD, ("pad", "string", "atmos", "drone", "texture", "chord", "ambient")),
    (Channel.LEAD, ("lead", "arp", "pluck", "synth", "melody", "stab", "key", "piano")),
    (Channel.FX, ("fx", "riser", "sweep", "impact", "noise", "downlifter", "uplifter")),
]


def map_channel(track_name: str) -> Channel:
    """Best-effort map a track name to a canonical channel.

    Defaults to `LEAD` for unrecognized names — most stray synth tracks are
    melodic, and LEAD is the safest harmonic bucket to fall back to.
    """
    name = track_name.lower()
    for channel, keywords in _RULES:
        if any(kw in name for kw in keywords):
            return channel
    return Channel.LEAD
