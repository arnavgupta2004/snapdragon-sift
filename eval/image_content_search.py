#!/usr/bin/env python3
"""Tests image CONTENT search: queries that describe what's visually IN an image
(a shape + color), with zero keyword/filename overlap — a hit can only come from CLIP
actually looking at the pixels (app/retrieval/image_search.py), not from matching
text. This is the direct answer to the motivating example for this work: "searching an
image is very difficult... convert this image into some embedding and the query...
will get matched."

Kept as a separate, standalone eval script — NOT part of eval/eval_set.json /
build_eval_set.py — so it's purely additive and doesn't touch the existing harness's
reproducibility of the Phase 1-16 numbers already in REPORT.md.

Two things are measured:
  1. image_search.search() in isolation — does CLIP content-matching work at all.
  2. The full agent pipeline (run_query) — do these content-only queries route
     correctly (no filename/exact-match signal, so they should never hit the fast
     tier) and does the final ranked/personalized output still surface the right
     images after RRF fusion with the (irrelevant, for these queries) text retrievers.

Requires data/image_subjects.json (written by data/generate_synthetic_data.py) for
ground truth — the shape+color assigned to each generated image.

    python eval/image_content_search.py
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from sqlmodel import select  # noqa: E402

from app.agent.graph import run_query  # noqa: E402
from app.db import get_session  # noqa: E402
from app.models import FileRecord, UserRecord  # noqa: E402
from app.retrieval import image_search  # noqa: E402
from eval.metrics import mrr, ndcg_at_k, precision_at_k, recall_at_k  # noqa: E402
from eval.run_benchmark import RESULTS_DIR  # noqa: E402

SUBJECTS_PATH = REPO_ROOT / "data" / "image_subjects.json"
MIN_SUPPORT = 2  # only build a query for a shape+color combo with >= this many images
MAX_QUERIES = 6
K_PRECISION = 5
K_NDCG = 10


def _query_text(shape: str, color: str) -> str:
    # Deliberately avoids the word "image" — it happens to literally appear in a few
    # corpus filenames (hero_image_*.png), which made BM25 keyword_search match on
    # lexical coincidence rather than testing pure visual-content matching. "photo" /
    # "picture" don't collide with anything in the corpus (checked against the
    # generated filenames), so a hit here can only come from CLIP looking at pixels.
    article = "an" if color[0] in "aeiou" else "a"
    return f"there's a photo somewhere with {article} large {color} {shape} in it, not sure what it's called"


def build_queries() -> list[dict]:
    if not SUBJECTS_PATH.exists():
        raise SystemExit(
            f"{SUBJECTS_PATH} not found. Run data/generate_synthetic_data.py first "
            "(it writes this sidecar alongside the corpus)."
        )
    subjects: dict[str, dict[str, str]] = json.loads(SUBJECTS_PATH.read_text())

    with get_session() as session:
        files_by_filename = {f.filename: f for f in session.exec(select(FileRecord)).all()}

    by_shape_color: dict[tuple[str, str], list[int]] = defaultdict(list)
    for filename, info in subjects.items():
        record = files_by_filename.get(filename)
        if record is not None and record.id is not None:
            by_shape_color[(info["shape"], info["color"])].append(record.id)

    combos = sorted(by_shape_color.items(), key=lambda kv: len(kv[1]), reverse=True)
    queries = []
    for (shape, color), file_ids in combos:
        if len(file_ids) < MIN_SUPPORT:
            continue
        queries.append(
            {
                "id": f"image-content-{shape}-{color}",
                "query": _query_text(shape, color),
                "shape": shape,
                "color": color,
                "ground_truth_file_ids": sorted(file_ids),
            }
        )
        if len(queries) >= MAX_QUERIES:
            break
    return queries


def bench_retriever_in_isolation(queries: list[dict]) -> dict:
    rows = []
    for q in queries:
        relevant = set(q["ground_truth_file_ids"])
        results = image_search.search(q["query"], limit=10)
        ranked_ids = [r.file_id for r in results]
        rows.append(
            {
                "id": q["id"],
                "precision_at_5": precision_at_k(ranked_ids, relevant, K_PRECISION),
                "recall_at_5": recall_at_k(ranked_ids, relevant, K_PRECISION),
                "ndcg_at_10": ndcg_at_k(ranked_ids, relevant, K_NDCG),
                "mrr": mrr(ranked_ids, relevant),
            }
        )
    return {
        "n_queries": len(rows),
        "precision_at_5": mean(r["precision_at_5"] for r in rows) if rows else None,
        "recall_at_5": mean(r["recall_at_5"] for r in rows) if rows else None,
        "ndcg_at_10": mean(r["ndcg_at_10"] for r in rows) if rows else None,
        "mrr": mean(r["mrr"] for r in rows) if rows else None,
        "per_query": rows,
    }


def bench_full_pipeline(queries: list[dict], user_ids: list[int]) -> dict:
    rows = []
    tiers_hit = defaultdict(int)
    for q in queries:
        relevant = set(q["ground_truth_file_ids"])
        for user_id in user_ids:
            out = run_query(q["query"], user_id)
            tiers_hit[out["routing_trace"]["tier"]] += 1
            ranked_ids = [r["file_id"] for r in out["results"]]
            rows.append(
                {
                    "id": q["id"],
                    "user_id": user_id,
                    "tier": out["routing_trace"]["tier"],
                    "precision_at_5": precision_at_k(ranked_ids, relevant, K_PRECISION),
                    "ndcg_at_10": ndcg_at_k(ranked_ids, relevant, K_NDCG),
                    "mrr": mrr(ranked_ids, relevant),
                }
            )
    return {
        "n_runs": len(rows),
        "precision_at_5": mean(r["precision_at_5"] for r in rows) if rows else None,
        "ndcg_at_10": mean(r["ndcg_at_10"] for r in rows) if rows else None,
        "mrr": mean(r["mrr"] for r in rows) if rows else None,
        "tiers_hit": dict(tiers_hit),
        "per_run": rows,
    }


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("[image_content_search] building content-description queries from data/image_subjects.json...")
    queries = build_queries()
    if not queries:
        raise SystemExit(
            "No shape+color combo has >= MIN_SUPPORT images. Regenerate the corpus "
            "(data/generate_synthetic_data.py) — the current one may predate the "
            "image-subject feature."
        )
    print(f"[image_content_search] built {len(queries)} queries: {[q['query'] for q in queries]}")

    print("[image_content_search] benchmarking image_search.search() in isolation...")
    isolated = bench_retriever_in_isolation(queries)
    print(f"    {({k: v for k, v in isolated.items() if k != 'per_query'})}")

    with get_session() as session:
        user_ids = [u.id for u in session.exec(select(UserRecord)).all()]

    print(f"[image_content_search] benchmarking the full pipeline ({len(queries)} queries x {len(user_ids)} users)...")
    full = bench_full_pipeline(queries, user_ids)
    print(f"    {({k: v for k, v in full.items() if k != 'per_run'})}")

    result = {"queries": queries, "isolated_retriever": isolated, "full_pipeline": full}
    summary_path = RESULTS_DIR / "image_content_search_summary.json"
    summary_path.write_text(json.dumps(result, indent=2))
    print(f"\nwrote {summary_path}")


if __name__ == "__main__":
    main()
