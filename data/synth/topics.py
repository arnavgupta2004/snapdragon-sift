"""Topic clusters for the synthetic file corpus.

Ten clusters chosen to mimic what an actual grad student / early-career engineer's
filesystem looks like, spanning coursework, freelance/client work, research, and admin
life. Each cluster carries a file-type bias (weights, not hard rules) so the corpus
isn't uniform, and a vocabulary the content generator draws on so topic clusters are
genuinely separable by keyword/semantic search, not just by folder name.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Topic:
    key: str
    name: str
    slug: str
    keywords: tuple[str, ...]
    file_type_weights: dict[str, float] = field(default_factory=dict)
    description: str = ""


TOPICS: list[Topic] = [
    Topic(
        key="ml_coursework",
        name="ML Coursework",
        slug="ml-coursework",
        keywords=(
            "gradient descent", "backpropagation", "convolutional neural network",
            "loss function", "regularization", "cross-validation", "hyperparameter",
            "PyTorch", "training loop", "overfitting", "learning rate", "assignment",
        ),
        file_type_weights={"py": 0.45, "md": 0.2, "pdf": 0.15, "pptx": 0.1, "xlsx": 0.1},
        description="Machine learning course assignments, lecture notes, and homework scripts.",
    ),
    Topic(
        key="sim_scripts",
        name="Aerospace Simulation Scripts",
        slug="aerospace-sim",
        keywords=(
            "flight dynamics", "trajectory simulation", "aerodynamic coefficient",
            "control loop", "sensor fusion", "Kalman filter", "actuator", "telemetry",
            "wind tunnel", "structural load", "simulation run",
        ),
        file_type_weights={"py": 0.5, "xlsx": 0.2, "md": 0.15, "pdf": 0.15},
        description="Internship/research scripts simulating flight dynamics and control systems.",
    ),
    Topic(
        key="client_project",
        name="Client Project — Retail Dashboard",
        slug="client-retail-dashboard",
        keywords=(
            "client requirements", "dashboard", "KPI", "stakeholder", "sprint",
            "deliverable", "invoice", "scope", "revenue by region", "churn rate",
            "SLA", "milestone",
        ),
        file_type_weights={"docx": 0.25, "xlsx": 0.25, "pptx": 0.2, "py": 0.15, "pdf": 0.15},
        description="Freelance client engagement building an analytics dashboard for a retail chain.",
    ),
    Topic(
        key="personal_notes",
        name="Personal Notes",
        slug="personal-notes",
        keywords=(
            "grocery list", "apartment", "budget", "travel itinerary", "book notes",
            "journal", "reminder", "goals", "habit tracker", "recipe",
        ),
        file_type_weights={"md": 0.5, "txt": 0.35, "xlsx": 0.15},
        description="Day-to-day personal notes, lists, and journaling.",
    ),
    Topic(
        key="research_drafts",
        name="Research Paper Drafts",
        slug="research-drafts",
        keywords=(
            "literature review", "related work", "ablation study", "baseline",
            "evaluation metric", "experiment", "abstract", "citation", "reviewer comment",
            "audio deepfake", "spoofing detection", "advisor feedback",
        ),
        file_type_weights={"docx": 0.35, "pdf": 0.3, "md": 0.2, "pptx": 0.15},
        description="Drafts and notes for a research paper on audio deepfake detection with an advisor.",
    ),
    Topic(
        key="data_eng",
        name="Data Engineering / ETL",
        slug="data-eng",
        keywords=(
            "ETL pipeline", "data warehouse", "schema migration", "Airflow DAG",
            "partitioning", "batch job", "data quality check", "ingestion", "backfill",
            "cron schedule",
        ),
        file_type_weights={"py": 0.55, "md": 0.2, "xlsx": 0.15, "txt": 0.1},
        description="ETL pipeline scripts and data infrastructure notes from a part-time role.",
    ),
    Topic(
        key="finance_reports",
        name="Financial / Business Reports",
        slug="finance-reports",
        keywords=(
            "quarterly report", "revenue", "expense breakdown", "budget forecast",
            "profit margin", "balance sheet", "Q3", "Q4", "fiscal year", "cost center",
        ),
        file_type_weights={"xlsx": 0.4, "pdf": 0.3, "docx": 0.2, "pptx": 0.1},
        description="Quarterly and annual financial reports and budget spreadsheets.",
    ),
    Topic(
        key="conference_decks",
        name="Conference Presentation Decks",
        slug="conference-decks",
        keywords=(
            "keynote", "poster session", "talk abstract", "slide deck", "conference",
            "workshop", "panel", "submission deadline", "camera-ready", "Q&A",
        ),
        file_type_weights={"pptx": 0.55, "pdf": 0.25, "docx": 0.2},
        description="Slide decks and materials prepared for academic/industry conferences.",
    ),
    Topic(
        key="design_assets",
        name="Design Assets",
        slug="design-assets",
        keywords=(
            "mockup", "wireframe", "color palette", "typography", "logo",
            "brand guideline", "figma export", "icon set", "hero image",
        ),
        file_type_weights={"png": 0.7, "pdf": 0.15, "docx": 0.15},
        description="UI mockups, icons, and branding assets exported as images.",
    ),
    Topic(
        key="admin_hr",
        name="Admin / HR Documents",
        slug="admin-hr",
        keywords=(
            "offer letter", "reimbursement", "tax form", "insurance", "onboarding",
            "leave request", "performance review", "policy", "timesheet",
        ),
        file_type_weights={"pdf": 0.4, "docx": 0.35, "xlsx": 0.25},
        description="Administrative, HR, and paperwork-adjacent documents.",
    ),
]

TOPIC_BY_KEY: dict[str, Topic] = {t.key: t for t in TOPICS}
