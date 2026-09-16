"""Writes feedback signals (thumbs up/down, implicit "opened") back into the
feedback log. Called by the API (POST /api/feedback) and the dev UI's feedback
buttons — this is what closes the loop: app.personalization.training_data reads these
rows back out when retrain.py is run again.
"""

from __future__ import annotations

from datetime import datetime

from app.db import get_session
from app.models import FeedbackEvent

VALID_SIGNALS = {"thumbs_up", "thumbs_down", "opened"}


def record_feedback(user_id: int, file_id: int, query: str, signal: str) -> FeedbackEvent:
    if signal not in VALID_SIGNALS:
        raise ValueError(f"signal must be one of {VALID_SIGNALS}, got {signal!r}")

    event = FeedbackEvent(
        user_id=user_id, file_id=file_id, query=query, signal=signal, timestamp=datetime.now()
    )
    with get_session() as session:
        session.add(event)
        session.commit()
        session.refresh(event)
    return event
