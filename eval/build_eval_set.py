#!/usr/bin/env python3
"""Builds the labeled eval set: 50 queries (15 easy, 20 medium, 15 hard) with
hand-specified ground truth, computed against whatever corpus currently exists in
data/db.sqlite so the set stays valid across regenerations.

Difficulty design (hand-authored, grounded programmatically):
  - easy:   exact filename lookup. Ground truth is unambiguous: the one file.
  - medium: file-type + topic (+ implicit recency framing) filter. Ground truth is
            every file in the corpus actually matching that topic+type combination.
  - hard:   a hand-written vague/conversational paraphrase of a specific real topic
            keyword (e.g. "audio deepfakes" -> "that thing with my advisor a few weeks
            ago"), so it requires genuine semantic understanding, not keyword luck.
            Ground truth is every file whose filename was generated from that keyword.

    python eval/build_eval_set.py
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path
from random import Random

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from sqlmodel import select  # noqa: E402

from app.db import get_session  # noqa: E402
from app.models import FileRecord  # noqa: E402
from data.generate_synthetic_data import slugify  # noqa: E402
from data.synth.topics import TOPIC_BY_KEY  # noqa: E402

OUT_PATH = REPO_ROOT / "eval" / "eval_set.json"
SEED = 7

# (topic_key, keyword, hand-written vague/conversational query text)
HARD_TEMPLATES = [
    ("research_drafts", "audio deepfake", "find that thing I was working on with my advisor about audio deepfakes a few weeks ago"),
    ("research_drafts", "spoofing detection", "wasn't there something about spoofing detection I was drafting recently"),
    ("research_drafts", "ablation study", "the experiment where I removed pieces one at a time to see what mattered"),
    ("sim_scripts", "kalman filter", "that sensor fusion thing I was messing with a while back, some kind of filter"),
    ("sim_scripts", "trajectory simulation", "the flight path thing I was running simulations on"),
    ("data_eng", "backfill", "the script that handled those late data issues, forget what it's called"),
    ("data_eng", "airflow dag", "that scheduled pipeline thing that runs every night"),
    ("client_project", "churn rate", "that number the client cares about, the one about people leaving"),
    ("client_project", "dashboard", "that visual thing I built for the client to see their numbers"),
    ("ml_coursework", "overfitting", "my notes about the model doing too well on training but not testing"),
    ("finance_reports", "profit margin", "the numbers about how much we're actually making after costs"),
    ("finance_reports", "balance sheet", "the document with all the assets and liabilities laid out"),
    ("conference_decks", "camera-ready", "the deck I had to finalize before the deadline for that talk"),
    ("design_assets", "wireframe", "that rough sketch of the app layout I made"),
    ("admin_hr", "reimbursement", "the paperwork for getting my money back from that trip"),
]


def build_easy(files: list[FileRecord], rng: Random, n: int = 15) -> list[dict]:
    sample = rng.sample(files, k=min(n, len(files)))
    return [
        {
            "id": f"easy-{f.id}",
            "query": f"open {f.filename}",
            "difficulty": "easy",
            "ground_truth_file_ids": [f.id],
        }
        for f in sample
    ]


def build_medium(files: list[FileRecord], rng: Random, n: int = 20) -> list[dict]:
    by_topic_type: dict[tuple[str, str], list[FileRecord]] = defaultdict(list)
    for f in files:
        by_topic_type[(f.topic_cluster, f.file_type)].append(f)

    combos = [(k, v) for k, v in by_topic_type.items() if len(v) >= 3]
    rng.shuffle(combos)

    queries = []
    for (topic_key, file_type), matched in combos[:n]:
        topic = TOPIC_BY_KEY[topic_key]
        queries.append(
            {
                "id": f"medium-{topic_key}-{file_type}",
                "query": f"find my {file_type} files related to {topic.name.lower()}",
                "difficulty": "medium",
                "ground_truth_file_ids": sorted(f.id for f in matched),
            }
        )
    return queries


def build_hard(files: list[FileRecord]) -> list[dict]:
    queries = []
    for topic_key, keyword, query_text in HARD_TEMPLATES:
        kw_slug = slugify(keyword)
        matched = [f.id for f in files if f.topic_cluster == topic_key and kw_slug in f.filename]
        if not matched:
            continue
        queries.append(
            {
                "id": f"hard-{topic_key}-{kw_slug}",
                "query": query_text,
                "difficulty": "hard",
                "ground_truth_file_ids": sorted(matched),
            }
        )
    return queries


def main() -> None:
    with get_session() as session:
        files = session.exec(select(FileRecord)).all()

    if not files:
        raise SystemExit("data/db.sqlite has no files. Run data/generate_synthetic_data.py first.")

    rng = Random(SEED)
    queries = build_easy(files, rng) + build_medium(files, rng) + build_hard(files)

    OUT_PATH.write_text(json.dumps(queries, indent=2))

    by_diff: dict[str, int] = defaultdict(int)
    for q in queries:
        by_diff[q["difficulty"]] += 1
    print(f"wrote {len(queries)} eval queries to {OUT_PATH}")
    print(f"  by difficulty: {dict(by_diff)}")


if __name__ == "__main__":
    main()
