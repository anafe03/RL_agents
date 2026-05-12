"""Weighted interval scheduling for conflict resolution.

Classic CS problem: given N intervals, each with a weight (a taste-match
score), pick a non-overlapping subset that maximizes total weight. Solvable
in O(n log n) with DP after sorting by end time. Reference: CLRS / Kleinberg-
Tardos chapter 6.

This module is pure — no LLM, no I/O. The matcher (`matcher.py`) hands it
a fully-scored list of recommendations; the scheduler picks the best
non-overlapping subset *per day* while honoring hard `must_see` locks and
`avoid` exclusions.
"""

from __future__ import annotations

from bisect import bisect_right

from festival.models import (
    DaySchedule,
    Festival,
    Schedule,
    Set,
    SetRecommendation,
    TasteProfile,
)


# ---- core algorithm --------------------------------------------------------


def _weighted_interval_schedule(
    items: list[tuple[int, int, float]],
) -> list[int]:
    """Return indices into `items` of the chosen non-overlapping subset.

    Each item is (start_min, end_min, weight). The chosen subset has no
    pair of overlapping intervals and maximizes total weight.

    O(n log n) — sort by end time, compute `p[i]` (last item ending at or
    before `items[i]`'s start) via binary search, run DP.
    """
    if not items:
        return []

    # Sort by end time, keeping original indices so callers can map back.
    indexed = sorted(enumerate(items), key=lambda x: x[1][1])
    order = [orig for orig, _ in indexed]
    intervals = [iv for _, iv in indexed]

    n = len(intervals)
    starts = [iv[0] for iv in intervals]
    ends = [iv[1] for iv in intervals]
    weights = [iv[2] for iv in intervals]

    # p[i] = index of the latest interval whose end <= intervals[i].start, else -1.
    # We binary-search on `ends` (already sorted) for starts[i].
    p: list[int] = []
    for i in range(n):
        # find rightmost end <= starts[i]
        idx = bisect_right(ends, starts[i]) - 1
        p.append(idx)

    # DP: opt[i] = max weight using items[0..i] (in sorted-by-end order).
    opt = [0.0] * (n + 1)
    take = [False] * n
    for i in range(n):
        include = weights[i] + (opt[p[i] + 1] if p[i] >= 0 else 0.0)
        exclude = opt[i]
        if include >= exclude:
            opt[i + 1] = include
            take[i] = True
        else:
            opt[i + 1] = exclude
            take[i] = False

    # Backtrack to recover which sorted-indices we took.
    chosen_sorted: list[int] = []
    i = n - 1
    while i >= 0:
        if take[i]:
            include = weights[i] + (opt[p[i] + 1] if p[i] >= 0 else 0.0)
            exclude = opt[i]
            if include >= exclude:
                chosen_sorted.append(i)
                i = p[i]
                continue
        i -= 1

    # Map sorted indices back to original.
    return sorted(order[i] for i in chosen_sorted)


# ---- adapter to festival types --------------------------------------------


def _filter_by_must_avoid(
    recs: list[SetRecommendation],
    taste: TasteProfile,
) -> list[SetRecommendation]:
    """Drop `avoid` artists; tag `must_see` and inflate their scores."""
    out: list[SetRecommendation] = []
    avoid_lower = {a.lower() for a in taste.avoid}
    must_see_lower = {a.lower() for a in taste.must_see}
    for r in recs:
        if r.artist.lower() in avoid_lower:
            continue
        if r.artist.lower() in must_see_lower:
            r = r.model_copy(update={"must_see": True, "score": max(r.score, 0.99)})
        out.append(r)
    return out


def _schedule_one_day(
    sets_for_day: list[Set],
    recs_by_set_id: dict[str, SetRecommendation],
    day: str,
) -> DaySchedule:
    """Run weighted interval scheduling on one day."""
    intervals: list[tuple[int, int, float]] = []
    set_index: list[Set] = []
    for s in sets_for_day:
        rec = recs_by_set_id.get(s.id)
        if rec is None:
            continue
        intervals.append((s.start_min, s.end_min, rec.score))
        set_index.append(s)

    chosen_idx = _weighted_interval_schedule(intervals)
    chosen_set_ids = {set_index[i].id for i in chosen_idx}

    picks: list[SetRecommendation] = []
    skipped: list[SetRecommendation] = []
    for s in sets_for_day:
        rec = recs_by_set_id.get(s.id)
        if rec is None:
            continue
        if s.id in chosen_set_ids:
            picks.append(rec)
        else:
            # Only count something as "skipped due to conflict" if it had a
            # meaningful score — otherwise it's noise.
            if rec.score >= 0.5:
                skipped.append(rec)

    picks.sort(key=lambda r: _hhmm_to_min(r.start))
    skipped.sort(key=lambda r: r.score, reverse=True)
    return DaySchedule(day=day, picks=picks, skipped_due_to_conflict=skipped)


def _hhmm_to_min(s: str) -> int:
    h, m = s.split(":")
    return int(h) * 60 + int(m)


def build_schedule(
    festival: Festival,
    taste: TasteProfile,
    recommendations: list[SetRecommendation],
) -> Schedule:
    """Build the full multi-day schedule from per-set recommendations."""
    recs = _filter_by_must_avoid(recommendations, taste)
    by_id = {r.set_id: r for r in recs}
    days = [_schedule_one_day(festival.sets_on(d), by_id, d) for d in festival.days]
    return Schedule(festival_name=festival.name, taste=taste, days=days)
