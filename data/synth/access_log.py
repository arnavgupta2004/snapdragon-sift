"""Simulates a per-user file access event stream with deliberately injected structure.

Two sources of structure, both load-bearing for Objective 2 (personalization):
  1. A fixed weekly recurring session (same ~3 files, same weekday, same hour) per
     persona — this is what temporal_patterns.py should be able to detect and what the
     UI's "it's Monday morning, you usually open X" insight is built on.
  2. Topically-coherent "working sessions" drawn from each persona's topic/file-type
     weights and active-hours distribution, rather than uniform random access — this is
     what frequency/topic-affinity scoring in profile_builder.py has to recover.
"""

from __future__ import annotations

import random
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta

from data.synth.personas import Persona
from app.models import AccessEvent, FileRecord, UserRecord

ACTION_WEIGHTS = {"open": 0.7, "edit": 0.2, "search": 0.1}


def _weighted_choice(rng: random.Random, weights: dict) -> object:
    keys = list(weights.keys())
    vals = list(weights.values())
    return rng.choices(keys, weights=vals, k=1)[0]


@dataclass
class AccessLogResult:
    events: list[AccessEvent]
    recurring_files_by_user: dict[int, dict[int, list[int]]]  # user_id -> weekday -> file_ids


def simulate_access_log(
    files: list[FileRecord],
    user_personas: list[tuple[UserRecord, Persona]],
    weeks: int,
    seed: int,
) -> AccessLogResult:
    rng = random.Random(seed)

    files_by_topic: dict[str, list[FileRecord]] = defaultdict(list)
    for f in files:
        files_by_topic[f.topic_cluster].append(f)

    end_date = datetime.now()
    start_date = end_date - timedelta(weeks=weeks)

    events: list[AccessEvent] = []
    recurring_files_by_user: dict[int, dict[int, list[int]]] = {}

    for user, persona in user_personas:
        assert user.id is not None
        recurring_files: dict[int, list[FileRecord]] = {}
        for weekday, topic_key in persona.recurring_weekday_topic.items():
            pool = files_by_topic.get(topic_key, [])
            if pool:
                recurring_files[weekday] = rng.sample(pool, k=min(3, len(pool)))
        recurring_files_by_user[user.id] = {
            wd: [f.id for f in fl if f.id is not None] for wd, fl in recurring_files.items()
        }

        for week_idx in range(weeks):
            week_start = start_date + timedelta(weeks=week_idx)

            for weekday, fixed_files in recurring_files.items():
                day = week_start + timedelta(days=weekday)
                if day > end_date or rng.random() > 0.85:
                    continue  # occasionally skipped, so the pattern is strong but not robotic
                base_ts = day.replace(
                    hour=persona.recurring_hour, minute=rng.randint(0, 20), second=0, microsecond=0
                )
                session_id = f"u{user.id}-rec-w{week_idx}-d{weekday}"
                for i, f in enumerate(fixed_files):
                    if f.id is None:
                        continue
                    events.append(
                        AccessEvent(
                            user_id=user.id,
                            file_id=f.id,
                            action=_weighted_choice(rng, ACTION_WEIGHTS),
                            timestamp=base_ts + timedelta(minutes=rng.randint(0, 3) + i * 2),
                            session_id=session_id,
                        )
                    )

            n_sessions = max(1, round(persona.sessions_per_week))
            for s in range(n_sessions):
                day_offset = rng.randint(0, 6)
                day = week_start + timedelta(days=day_offset)
                if day > end_date:
                    continue
                hour = _weighted_choice(rng, persona.active_hours)
                base_ts = day.replace(hour=hour, minute=rng.randint(0, 59), second=0, microsecond=0)

                topic_key = _weighted_choice(rng, persona.topic_weights)
                pool = files_by_topic.get(topic_key, [])
                if not pool:
                    continue
                session_size = max(1, round(rng.gauss(persona.avg_session_size, 1.2)))
                session_size = min(session_size, len(pool))
                chosen = rng.sample(pool, k=session_size)

                if rng.random() < 0.15:
                    secondary_topic = _weighted_choice(rng, persona.topic_weights)
                    secondary_pool = files_by_topic.get(secondary_topic, [])
                    if secondary_pool:
                        chosen.append(rng.choice(secondary_pool))

                session_id = f"u{user.id}-w{week_idx}-s{s}"
                ts = base_ts
                for f in chosen:
                    if f.id is None:
                        continue
                    ts = ts + timedelta(minutes=rng.randint(1, 15))
                    events.append(
                        AccessEvent(
                            user_id=user.id,
                            file_id=f.id,
                            action=_weighted_choice(rng, ACTION_WEIGHTS),
                            timestamp=ts,
                            session_id=session_id,
                        )
                    )

    events.sort(key=lambda e: e.timestamp)
    return AccessLogResult(events=events, recurring_files_by_user=recurring_files_by_user)
