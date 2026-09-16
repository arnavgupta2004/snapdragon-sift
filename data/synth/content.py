"""Content generation for synthetic corpus files.

Two backends:
  - LLM backend (app.llm_client): used when GEMINI_API_KEY is set, for realistic,
    non-repetitive prose per file.
  - Template backend: deterministic, seeded, keyword-driven generation that is
    topically coherent (draws on each Topic's keyword bank) without being lorem-ipsum.
    Always available, used as the default and as the fallback when no API key is set.

Every generator returns a `GeneratedContent` with a `display_text` (what would be
written into the actual file) and `extracted_text` (what a real text-extraction step
would pull out of that file type for indexing — identical to display_text for
text-native formats, a flattened version for tables/slides, a caption for images).
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from data.synth.topics import Topic

CONNECTORS = [
    "Note that", "Importantly,", "As a result,", "In this section,", "Next,",
    "For reference,", "Following up on the previous point,", "To summarize,",
    "One open question is whether", "It's worth double-checking that",
]


@dataclass
class GeneratedContent:
    display_text: str
    extracted_text: str


def _sentence(rng: random.Random, topic: Topic, extra_vocab: tuple[str, ...] = ()) -> str:
    vocab = list(topic.keywords) + list(extra_vocab)
    kw = rng.choice(vocab)
    connector = rng.choice(CONNECTORS)
    templates = [
        f"{connector} the {kw} needs another pass before this is finalized.",
        f"{connector} we should revisit the {kw} approach used here.",
        f"This section covers {kw} and how it affects the overall result.",
        f"The current numbers for {kw} look reasonable but are not yet verified.",
        f"{connector} {kw} was the main blocker this week.",
        f"A quick summary of {kw}: still in progress, revisiting after feedback.",
    ]
    return rng.choice(templates)


def generate_prose(topic: Topic, filename: str, rng: random.Random, paragraphs: int = 3) -> GeneratedContent:
    title = filename.rsplit(".", 1)[0].replace("_", " ").replace("-", " ").title()
    lines = [f"# {title}", ""]
    for _ in range(paragraphs):
        sentence_count = rng.randint(3, 6)
        para = " ".join(_sentence(rng, topic) for _ in range(sentence_count))
        lines.append(para)
        lines.append("")
    text = "\n".join(lines).strip()
    return GeneratedContent(display_text=text, extracted_text=text)


def generate_notes(topic: Topic, filename: str, rng: random.Random) -> GeneratedContent:
    title = filename.rsplit(".", 1)[0].replace("_", " ").replace("-", " ").title()
    n_items = rng.randint(4, 9)
    bullets = []
    for _ in range(n_items):
        kw = rng.choice(topic.keywords)
        verb = rng.choice(["Check", "Follow up on", "Revisit", "Finish", "Ask about", "Review"])
        bullets.append(f"- [ ] {verb} {kw}")
    text = f"# {title}\n\n" + "\n".join(bullets)
    return GeneratedContent(display_text=text, extracted_text=text)


_CODE_SNIPPETS: dict[str, list[str]] = {
    "ml_coursework": [
        '''import torch
import torch.nn as nn

class {cls}(nn.Module):
    """Training assignment: {kw1} with {kw2} for regularization."""

    def __init__(self, in_dim: int, hidden: int = {hidden}, out_dim: int = 10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(0.{drop}),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x):
        return self.net(x)


def train_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0.0
    for xb, yb in loader:
        optimizer.zero_grad()
        loss = criterion(model(xb), yb)
        loss.backward()  # {kw3} happens here
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)
''',
    ],
    "sim_scripts": [
        '''import numpy as np

class {cls}:
    """{kw1} model with a simple {kw2} update."""

    def __init__(self, dt: float = 0.0{dt}):
        self.dt = dt
        self.state = np.zeros(6)  # position, velocity, attitude

    def step(self, control_input: np.ndarray) -> np.ndarray:
        # Very simplified {kw3} integration step.
        self.state[:3] += self.state[3:] * self.dt
        self.state[3:] += control_input * self.dt
        return self.state.copy()


def run_simulation(steps: int = {steps}):
    sim = {cls}()
    trajectory = []
    for _ in range(steps):
        u = np.random.randn(3) * 0.01
        trajectory.append(sim.step(u))
    return np.array(trajectory)
''',
    ],
    "data_eng": [
        '''from datetime import datetime, timedelta

def extract(source_table: str, since: datetime):
    """Pulls incremental rows for the {kw1} job."""
    print(f"extracting {{source_table}} since {{since.isoformat()}}")
    return []  # placeholder for the actual query


def transform(rows: list[dict]) -> list[dict]:
    # {kw2} normalization pass
    for r in rows:
        r["ingested_at"] = datetime.utcnow().isoformat()
    return rows


def load(rows: list[dict], target_table: str):
    """Writes rows into {kw3}, handles the {backfill} case."""
    print(f"loading {{len(rows)}} rows into {{target_table}}")


def run_daily_job():
    since = datetime.utcnow() - timedelta(days=1)
    rows = extract("raw.events", since)
    rows = transform(rows)
    load(rows, "warehouse.events_clean")
''',
    ],
    "client_project": [
        '''import pandas as pd

def load_revenue_by_region(path: str) -> pd.DataFrame:
    """Loads the {kw1} extract used for the {kw2} dashboard."""
    df = pd.read_csv(path)
    return df.groupby("region")["revenue"].sum().reset_index()


def compute_churn_rate(active_start: int, active_end: int, churned: int) -> float:
    """{kw3} calculation for the weekly stakeholder update."""
    if active_start == 0:
        return 0.0
    return churned / active_start
''',
    ],
}


def generate_code(topic: Topic, filename: str, rng: random.Random) -> GeneratedContent:
    snippets = _CODE_SNIPPETS.get(topic.key)
    if not snippets:
        snippets = _CODE_SNIPPETS["data_eng"]
    template = rng.choice(snippets)
    kws = rng.sample(list(topic.keywords), k=min(3, len(topic.keywords)))
    cls_base = filename.rsplit(".", 1)[0].replace("_", " ").replace("-", " ").title().replace(" ", "")
    code = template.format(
        cls=cls_base or "Model",
        kw1=kws[0] if len(kws) > 0 else "signal",
        kw2=kws[1] if len(kws) > 1 else "update",
        kw3=kws[2] if len(kws) > 2 else "processing",
        hidden=rng.choice([64, 128, 256]),
        drop=rng.randint(1, 5),
        dt=rng.randint(1, 9),
        steps=rng.choice([100, 200, 500]),
        backfill=rng.choice(["late-arriving rows", "duplicate keys", "null timestamps"]),
    )
    return GeneratedContent(display_text=code, extracted_text=code)


def generate_table(topic: Topic, filename: str, rng: random.Random) -> tuple[list[list[str]], str]:
    """Returns (rows_including_header, flattened_extracted_text) for spreadsheet files."""
    n_rows = rng.randint(6, 14)
    if topic.key in ("finance_reports", "client_project"):
        header = ["Category", "Q1", "Q2", "Q3", "Q4"]
        rows = [header]
        for _ in range(n_rows):
            cat = rng.choice(topic.keywords).title()
            rows.append([cat] + [str(rng.randint(1000, 90000)) for _ in range(4)])
    elif topic.key == "sim_scripts":
        header = ["Run ID", "Metric", "Value", "Notes"]
        rows = [header]
        for i in range(n_rows):
            rows.append([f"RUN-{i:03d}", rng.choice(topic.keywords), f"{rng.uniform(0, 1):.3f}", "ok"])
    else:
        header = ["Item", "Status", "Owner", "Notes"]
        rows = [header]
        for _ in range(n_rows):
            rows.append([rng.choice(topic.keywords).title(), rng.choice(["done", "in progress", "blocked"]), "TBD", ""])
    flattened = "\n".join(", ".join(row) for row in rows)
    return rows, flattened


def generate_slides(topic: Topic, filename: str, rng: random.Random) -> tuple[list[dict], str]:
    n_slides = rng.randint(5, 10)
    title = filename.rsplit(".", 1)[0].replace("_", " ").replace("-", " ").title()
    slides = [{"title": title, "bullets": [topic.description]}]
    for _ in range(n_slides - 1):
        kw = rng.choice(topic.keywords)
        slide_title = kw.title()
        bullets = [_sentence(rng, topic) for _ in range(rng.randint(2, 4))]
        slides.append({"title": slide_title, "bullets": bullets})
    flattened = "\n".join(s["title"] + ": " + " ".join(s["bullets"]) for s in slides)
    return slides, flattened


def generate_image_caption(topic: Topic, filename: str, rng: random.Random) -> str:
    kw = rng.choice(topic.keywords)
    title = filename.rsplit(".", 1)[0].replace("_", " ").replace("-", " ").title()
    return f"{title} — {topic.name} asset related to {kw}."


def generate_llm_prose(topic: Topic, filename: str, file_type: str) -> GeneratedContent | None:
    """Best-effort real-content generation via Gemini. Returns None on any failure so the
    caller can fall back to the template generator without the whole run failing."""
    from app.llm_client import get_client

    client = get_client()
    if not client.is_available:
        return None
    try:
        prompt = (
            f"Write realistic {file_type} file content for a file named '{filename}' that would "
            f"plausibly exist on a real person's laptop under the topic '{topic.name}' "
            f"({topic.description}). Use vocabulary like: {', '.join(topic.keywords[:6])}. "
            f"Keep it under 250 words, no preamble, just the content itself."
        )
        result = client.complete(prompt, max_tokens=500)
        return GeneratedContent(display_text=result.text.strip(), extracted_text=result.text.strip())
    except Exception:
        return None
