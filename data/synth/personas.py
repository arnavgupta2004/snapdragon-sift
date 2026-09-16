"""Simulated user personas with distinct behavioral signatures.

Each persona defines *why* its access log looks the way it does, so the personalization
layer (app/personalization/) has real structure to recover: topic affinity, active hours,
file-type preference, and an explicit weekly recurring pattern (not left to chance —
injected deliberately so temporal_patterns.py has something genuine to detect).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Persona:
    key: str
    name: str
    description: str
    # topic_key -> relative weight (probability mass) this persona draws files from
    topic_weights: dict[str, float]
    # file extension -> relative preference multiplier
    file_type_affinity: dict[str, float]
    # hours of day (0-23) this persona is typically active, with relative weight
    active_hours: dict[int, float]
    # weekday (0=Mon..6=Sun) -> topic_key this persona has a fixed recurring session for
    recurring_weekday_topic: dict[int, str]
    # hour of day the recurring session happens at (the "it's Monday morning" signal)
    recurring_hour: int
    # average number of files opened together in one session
    avg_session_size: float
    # sessions per week
    sessions_per_week: float


PERSONAS: list[Persona] = [
    Persona(
        key="priya_grad_student",
        name="Priya Iyer",
        description=(
            "ML grad student and part-time flight-dynamics research intern. Heavy on code "
            "and research drafts, works in long late-night bursts, has a standing Monday "
            "10am advisor-prep ritual of reopening the same research-draft files."
        ),
        topic_weights={
            "ml_coursework": 0.30,
            "sim_scripts": 0.22,
            "research_drafts": 0.25,
            "personal_notes": 0.13,
            "conference_decks": 0.10,
        },
        file_type_affinity={"py": 1.6, "md": 1.3, "pdf": 1.1, "docx": 0.9, "xlsx": 0.5, "pptx": 0.8, "png": 0.6, "txt": 1.0},
        active_hours={21: 1.2, 22: 1.5, 23: 1.5, 0: 1.3, 1: 0.8, 14: 0.8, 15: 1.0, 16: 1.0},
        recurring_weekday_topic={0: "research_drafts"},  # Monday
        recurring_hour=10,
        avg_session_size=4.5,
        sessions_per_week=9,
    ),
    Persona(
        key="david_analyst",
        name="David Chen",
        description=(
            "Business analyst on a client engagement. Heavy on reports and spreadsheets, "
            "strict 9-to-6 business-hours user, has a standing Monday 9am status-meeting "
            "session reopening the same three client-report files."
        ),
        topic_weights={
            "client_project": 0.35,
            "finance_reports": 0.28,
            "admin_hr": 0.15,
            "conference_decks": 0.07,
            "personal_notes": 0.15,
        },
        file_type_affinity={"xlsx": 1.7, "docx": 1.3, "pdf": 1.2, "pptx": 1.1, "py": 0.4, "md": 0.6, "png": 0.5, "txt": 0.7},
        active_hours={9: 1.4, 10: 1.3, 11: 1.1, 13: 1.0, 14: 1.1, 15: 1.0, 16: 0.9, 17: 0.8},
        recurring_weekday_topic={0: "client_project"},  # Monday
        recurring_hour=9,
        avg_session_size=3.2,
        sessions_per_week=12,
    ),
    Persona(
        key="maria_freelancer",
        name="Maria Santos",
        description=(
            "Freelance PM/designer juggling multiple small engagements. Mixed file-type use, "
            "business hours with occasional evening catch-up bursts, standing Friday-afternoon "
            "wrap-up session reopening design-asset and admin files before weekly invoicing."
        ),
        topic_weights={
            "design_assets": 0.28,
            "client_project": 0.20,
            "admin_hr": 0.20,
            "personal_notes": 0.20,
            "conference_decks": 0.12,
        },
        file_type_affinity={"png": 1.6, "pdf": 1.2, "docx": 1.1, "xlsx": 1.0, "pptx": 0.9, "md": 0.9, "py": 0.3, "txt": 0.8},
        active_hours={10: 1.1, 11: 1.2, 12: 0.9, 15: 1.0, 16: 1.1, 19: 1.0, 20: 0.9},
        recurring_weekday_topic={4: "admin_hr"},  # Friday
        recurring_hour=15,
        avg_session_size=3.8,
        sessions_per_week=10,
    ),
]

PERSONA_BY_KEY: dict[str, Persona] = {p.key: p for p in PERSONAS}
