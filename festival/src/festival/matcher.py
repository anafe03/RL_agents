"""Taste matcher — uses the LLM to score every artist in the lineup against
the user's taste profile, and to write the "why this fits you" annotation.

Returns one `SetRecommendation` per set in the festival. The scheduler
takes those scores and runs weighted interval scheduling on top.
"""

from __future__ import annotations

import json

from festival import llm
from festival.models import (
    Festival,
    SetRecommendation,
    TasteProfile,
)

DEFAULT_MODEL = "claude-sonnet-4-6"

SYSTEM_PROMPT = """You are a music taste consultant scoring artists at a festival against a single user's stated taste.

You will receive:
1. The user's taste profile (freeform description, favorite artists, favorite genres).
2. ONE artist at a time, with their genres and a short blurb.

Your job: produce a score from 0.00 to 1.00 of how well this artist fits the user's taste, AND a one-sentence reasoning the user would actually find useful at a festival ("Why catch this set?" / "Why skip?").

Score calibration:
- 0.90-1.00: a perfect or near-perfect match. The user almost certainly already loves this artist or one virtually indistinguishable from it.
- 0.70-0.89: a strong match in vibe / genre / sensibility. Worth catching.
- 0.50-0.69: adjacent or partial match. Could be a nice surprise; might miss.
- 0.30-0.49: notable but probably outside the user's lane. Only catch if nothing better is happening.
- 0.00-0.29: clearly not their thing.

Be honest. Do not inflate scores. A 0.4 with a thoughtful one-line reason is more useful than a 0.7 hedge.

Reasoning style: concrete, second-person, addresses what specifically about THIS artist will land or not given the user's profile. No generic praise. No "if you like X, you'll like Y" filler.

Respond ONLY in JSON: {"score": <float 0..1>, "reasoning": "<one sentence>"}"""


def _taste_blurb(taste: TasteProfile) -> str:
    """Format the taste profile into clearly-delimited sections.

    The mock matcher parses these section headers — keep them stable.
    """
    parts: list[str] = []
    desc = taste.description or "(none)"
    genres = ", ".join(taste.favorite_genres) if taste.favorite_genres else "(none)"
    artists = ", ".join(taste.favorite_artists) if taste.favorite_artists else "(none)"
    parts.append(f"GENRES_DESC: {desc}")
    parts.append(f"FAVORITE_GENRES: {genres}")
    parts.append(f"FAVORITE_ARTISTS: {artists}")
    return "\n".join(parts)


def _score_one_artist(
    artist_name: str,
    artist_genres: list[str],
    artist_blurb: str,
    taste_text: str,
    model: str,
) -> tuple[float, str, float]:
    """Score one artist; return (score, reasoning, cost_usd)."""
    user_payload = f"""User taste profile:

{taste_text}

Artist to score: **{artist_name}**
Genres: {', '.join(artist_genres) if artist_genres else '(unknown)'}
Blurb: {artist_blurb or '(no description)'}

Return JSON only."""

    result = llm.chat(
        model=model,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_payload}],
        max_tokens=400,
    )
    score, reasoning = _parse_score(result.text)
    return score, reasoning, result.cost_usd


def _parse_score(raw: str) -> tuple[float, str]:
    """Tolerant JSON parse — fall back to (0.5, raw) if anything's wrong."""
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError:
        start = raw.find("{")
        end = raw.rfind("}")
        if start == -1 or end <= start:
            return 0.5, f"(judge response was not JSON: {raw[:120]})"
        try:
            obj = json.loads(raw[start : end + 1])
        except json.JSONDecodeError:
            return 0.5, f"(judge response was not JSON: {raw[:120]})"
    score = float(obj.get("score", 0.5))
    score = max(0.0, min(1.0, score))
    reasoning = str(obj.get("reasoning", ""))
    return score, reasoning


def score_lineup(
    festival: Festival,
    taste: TasteProfile,
    model: str = DEFAULT_MODEL,
) -> tuple[list[SetRecommendation], float]:
    """Score every set in the festival. Returns (recommendations, total_cost_usd).

    Each artist is scored once — multiple sets by the same artist would reuse
    the score (festival lineups typically have one set per artist, so this
    only matters for edge cases).
    """
    taste_text = _taste_blurb(taste)
    artist_scores: dict[str, tuple[float, str]] = {}
    total_cost = 0.0

    for s in festival.sets:
        if s.artist in artist_scores:
            continue
        a = festival.artist(s.artist)
        genres = a.genres if a else []
        blurb = a.blurb if a else ""
        score, reasoning, cost = _score_one_artist(
            s.artist, genres, blurb, taste_text, model
        )
        artist_scores[s.artist] = (score, reasoning)
        total_cost += cost

    recs: list[SetRecommendation] = []
    for s in festival.sets:
        score, reasoning = artist_scores[s.artist]
        recs.append(
            SetRecommendation(
                set_id=s.id,
                artist=s.artist,
                day=s.day,
                stage=s.stage,
                start=s.start,
                end=s.end,
                score=score,
                reasoning=reasoning,
            )
        )
    return recs, total_cost
