"""Detects recurring weekday/hour access patterns per user — the "working context"
signal: if it's Monday 10am and this user has opened the same files at Monday 10am in
several previous weeks, that's a real, detectable pattern (data/synth/access_log.py
injects exactly this structure), not a guess.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime

from sqlmodel import select

from app.db import get_session
from app.models import AccessEvent

WEEKDAY_NAMES = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


@dataclass
class RecurringPattern:
    weekday: int  # 0=Monday
    hour: int
    file_ids: list[int]
    occurrences: int       # distinct weeks this file set showed up at this weekday/hour
    weeks_observed: int    # distinct weeks this weekday/hour bucket has any activity at all
    confidence: float      # occurrences / weeks_observed

    @property
    def weekday_name(self) -> str:
        return WEEKDAY_NAMES[self.weekday]


def detect_recurring_patterns(
    user_id: int, min_occurrences: int = 3, min_confidence: float = 0.4
) -> list[RecurringPattern]:
    with get_session() as session:
        events = session.exec(select(AccessEvent).where(AccessEvent.user_id == user_id)).all()

    if not events:
        return []

    # bucket: (weekday, hour) -> iso_week -> set[file_id]
    buckets: dict[tuple[int, int], dict[tuple[int, int], set[int]]] = defaultdict(lambda: defaultdict(set))
    for e in events:
        weekday, hour = e.timestamp.weekday(), e.timestamp.hour
        iso_year, iso_week, _ = e.timestamp.isocalendar()
        buckets[(weekday, hour)][(iso_year, iso_week)].add(e.file_id)

    patterns: list[RecurringPattern] = []
    for (weekday, hour), weeks in buckets.items():
        weeks_observed = len(weeks)
        if weeks_observed < min_occurrences:
            continue

        file_week_counts: dict[int, int] = defaultdict(int)
        for file_ids in weeks.values():
            for file_id in file_ids:
                file_week_counts[file_id] += 1

        recurring_file_ids = [
            file_id for file_id, count in file_week_counts.items() if count >= min_occurrences
        ]
        if not recurring_file_ids:
            continue

        occurrences = max(file_week_counts[f] for f in recurring_file_ids)
        confidence = occurrences / weeks_observed
        if confidence < min_confidence:
            continue

        patterns.append(
            RecurringPattern(
                weekday=weekday,
                hour=hour,
                file_ids=recurring_file_ids,
                occurrences=occurrences,
                weeks_observed=weeks_observed,
                confidence=confidence,
            )
        )

    patterns.sort(key=lambda p: p.confidence, reverse=True)
    return patterns


def current_context_boost(
    user_id: int, now: datetime | None = None, hour_tolerance: int = 1
) -> dict[int, float]:
    """Returns {file_id: confidence} for any recurring pattern matching the current
    weekday and (within tolerance) hour. Empty dict if nothing matches — most of the
    time, for most users, this correctly returns nothing."""
    now = now or datetime.now()
    patterns = detect_recurring_patterns(user_id)

    boost: dict[int, float] = {}
    for p in patterns:
        if p.weekday != now.weekday():
            continue
        if abs(p.hour - now.hour) > hour_tolerance:
            continue
        for file_id in p.file_ids:
            boost[file_id] = max(boost.get(file_id, 0.0), p.confidence)
    return boost
