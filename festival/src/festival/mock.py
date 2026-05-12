"""Deterministic mock chat — gives Festival Companion a realistic-feeling
demo run without calling Anthropic.

The mock looks at (a) the user's taste profile in the system prompt context
and (b) the artist's name and genres in the user payload, and returns a
score keyed to genre keywords. This produces consistent, debuggable scores
that mirror what a real model would do.
"""

from __future__ import annotations

import json
import re
from typing import Any

import anthropic

from festival import llm

# Genre-keyword → base score table. Matching is substring on the artist's
# genres list AND on the user's taste description.
# Higher = "user with this taste should like this genre family."
_GENRE_AFFINITY: dict[str, dict[str, float]] = {
    # taste_keyword: {artist_genre_keyword: score_delta}
    "indie": {"indie": 0.9, "dream pop": 0.8, "post-punk": 0.7, "folk": 0.7},
    "dream pop": {"dream pop": 0.95, "shoegaze": 0.85, "indie": 0.65, "slowcore": 0.7},
    "folk": {"folk": 0.9, "americana": 0.85, "chamber pop": 0.7, "indie folk": 0.95},
    "electronic": {"electronic": 0.9, "idm": 0.85, "house": 0.8, "ambient": 0.75, "dubstep": 0.7},
    "dance": {"electronic": 0.85, "house": 0.9, "uk garage": 0.85, "dance": 0.95},
    "house": {"house": 0.95, "electronic": 0.8, "uk garage": 0.85},
    "ambient": {"ambient": 0.95, "idm": 0.7, "post-dubstep": 0.7},
    "hip hop": {"hip hop": 0.9, "rap": 0.9, "alternative hip hop": 0.85},
    "rap": {"hip hop": 0.9, "rap": 0.95},
    "pop": {"pop": 0.85, "hyperpop": 0.7, "rnb": 0.7, "pop punk": 0.7},
    "punk": {"post-punk": 0.85, "hardcore": 0.85, "pop punk": 0.7},
    "post-punk": {"post-punk": 0.95, "hardcore": 0.7, "irish rock": 0.8},
    "jazz": {"jazz": 0.95, "afrobeat": 0.8, "beat science": 0.85, "brass": 0.8},
    "sad": {"sad indie": 0.95, "indie folk": 0.85, "slowcore": 0.85, "dream pop": 0.7},
    "art pop": {"art pop": 0.95, "indie rock": 0.7, "chamber pop": 0.7},
}


_REASONS_HIGH = [
    "Direct hit for your stated taste — go.",
    "This is exactly the lane you described.",
    "Strong genre match and a reliably great live set.",
    "Slots right into your favorite-artist orbit.",
]
_REASONS_MID = [
    "Adjacent to your taste — could be a nice surprise.",
    "Genre-overlap is partial; the live show is worth a maybe.",
    "Not your usual lane, but the production may pull you in.",
    "Decent fit on vibe even if the genre tag is off.",
]
_REASONS_LOW = [
    "Outside your stated lane; skip unless nothing else is on.",
    "Different aesthetic register from your favorites.",
    "Wouldn't expect this to land for you given the profile.",
    "Probably not your night — skip with low regret.",
]


def _usage() -> anthropic.types.Usage:
    return anthropic.types.Usage(
        input_tokens=500,
        output_tokens=80,
        cache_creation_input_tokens=300,
        cache_read_input_tokens=200,
        server_tool_use=None,
        service_tier=None,
    )


def _score_for(genre_text: str, artist_genres: list[str]) -> float:
    """Compute a deterministic score from taste keywords × artist genres.

    `genre_text` should only contain the user's genre/description signals —
    NOT favorite-artist names. Artist names containing genre-keywords (e.g.
    "Beach House" containing "house") would otherwise poison the matcher.
    """
    taste_lower = genre_text.lower()
    genres_lower = [g.lower() for g in artist_genres]
    best = 0.0
    for taste_keyword, genre_map in _GENRE_AFFINITY.items():
        if taste_keyword not in taste_lower:
            continue
        for ag in genres_lower:
            for g, weight in genre_map.items():
                if g in ag:
                    best = max(best, weight)
    return min(1.0, best)


def _pick_reason(score: float, artist_name: str) -> str:
    bucket = _REASONS_HIGH if score >= 0.75 else _REASONS_MID if score >= 0.5 else _REASONS_LOW
    # Deterministic pick — index by hash of artist name (stable across runs).
    return bucket[abs(hash(artist_name)) % len(bucket)]


def make_mock_chat() -> Any:
    """Return a chat fn that emulates the matcher's scoring behavior."""

    def fn(*, messages: list, **kwargs: Any) -> llm.ChatResult:
        user_text = ""
        if messages and isinstance(messages[0].get("content"), str):
            user_text = messages[0]["content"]

        # Pull the delimited sections written by matcher._taste_blurb.
        genres_desc = _extract_field(user_text, r"GENRES_DESC: (.+)")
        favorite_genres = _extract_field(user_text, r"FAVORITE_GENRES: (.+)")
        favorite_artists = _extract_field(user_text, r"FAVORITE_ARTISTS: (.+)")
        artist = _extract_field(user_text, r"Artist to score: \*\*(.+?)\*\*")
        genres_str = _extract_field(user_text, r"Genres: (.+)")
        genres = [g.strip() for g in genres_str.split(",")] if genres_str else []

        # Genre-keyword text excludes artist names — strip any favorite-artist
        # name from the description so e.g. "love Beach House" doesn't trigger
        # the "house" keyword and falsely match electronic acts.
        genre_text = f"{genres_desc} {favorite_genres}"
        for fav in favorite_artists.split(","):
            fav = fav.strip()
            if fav:
                genre_text = re.sub(re.escape(fav), " ", genre_text, flags=re.IGNORECASE)
        score = _score_for(genre_text, genres)

        # Named-favorite override: artist appears in user's favorite_artists.
        if artist and artist.lower() in favorite_artists.lower():
            score = max(score, 0.97)

        reasoning = _pick_reason(score, artist or "?")
        payload = {"score": round(score, 2), "reasoning": reasoning}
        return llm.ChatResult(
            text=json.dumps(payload),
            usage=_usage(),
            cost_usd=0.0009,
            model="claude-sonnet-4-6",
        )

    return fn


def _extract_block(text: str, start_marker: str, end_marker: str) -> str:
    si = text.find(start_marker)
    if si == -1:
        return ""
    ei = text.find(end_marker, si)
    if ei == -1:
        return text[si + len(start_marker):]
    return text[si + len(start_marker): ei]


def _extract_field(text: str, pattern: str) -> str:
    m = re.search(pattern, text)
    return m.group(1).strip() if m else ""
