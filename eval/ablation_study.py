#!/usr/bin/env python3
"""Component-by-component ablation: metadata-only -> +keyword -> +semantic ->
+hybrid fusion -> +reranker -> +personalization. Each stage is evaluated against the
full labeled eval set with the same metrics, so the marginal contribution of each
component is explicit rather than asserted.

Stages 1-3 use naive list concatenation (not real fusion) deliberately, so the jump at
stage 4 ("+hybrid fusion") isolates the value of RRF itself, not just "more signals".

    python eval/ablation_study.py
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from statistics import mean

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from sqlmodel import select  # noqa: E402

from app.agent.query_understanding import QueryIntent, extract_entities  # noqa: E402
from app.db import get_session  # noqa: E402
from app.models import UserRecord  # noqa: E402
from app.personalization.personalized_ranker import WeightedSumPersonalizer  # noqa: E402
from app.retrieval import keyword_search, metadata_search, semantic_search  # noqa: E402
from app.retrieval.base import ScoredFile  # noqa: E402
from app.retrieval.hybrid_fusion import reciprocal_rank_fusion  # noqa: E402
from app.retrieval.reranker import rerank  # noqa: E402
from eval.metrics import mrr, ndcg_at_k, precision_at_k, recall_at_k  # noqa: E402
from eval.run_benchmark import EVAL_SET_PATH, RESULTS_DIR, load_eval_set  # noqa: E402

POOL = 20
K_PRECISION, K_RECALL, K_NDCG = 5, 5, 10
METRIC_NAMES = ["precision_at_5", "recall_at_5", "ndcg_at_10", "mrr"]

STAGE_ORDER = [
    "1_metadata_only",
    "2_plus_keyword",
    "3_plus_semantic",
    "4_hybrid_fusion",
    "5_plus_reranker",
    "6_plus_personalization",
]


def _has_filters(intent: QueryIntent) -> bool:
    f = intent.filters
    return bool(f.file_types or f.modified_after or f.topic_cluster or f.filename_contains)


def _dedupe_concat(*id_lists: list[int]) -> list[int]:
    seen: set[int] = set()
    out: list[int] = []
    for ids in id_lists:
        for i in ids:
            if i not in seen:
                seen.add(i)
                out.append(i)
    return out


def _ids(results: list[ScoredFile]) -> list[int]:
    return [r.file_id for r in results]


def stage_metadata_only(intent: QueryIntent) -> list[int]:
    if not _has_filters(intent):
        return []
    return _ids(metadata_search.search(intent.filters, limit=POOL))


def stage_plus_keyword(intent: QueryIntent) -> list[int]:
    meta_ids = stage_metadata_only(intent)
    kw_ids = _ids(keyword_search.search(intent.search_query, limit=POOL))
    return _dedupe_concat(meta_ids, kw_ids)


def stage_plus_semantic(intent: QueryIntent) -> list[int]:
    prev_ids = stage_plus_keyword(intent)
    sem_ids = _ids(semantic_search.search(intent.search_query, limit=POOL))
    return _dedupe_concat(prev_ids, sem_ids)


def _retrieve_lists(intent: QueryIntent) -> tuple[list[ScoredFile], list[ScoredFile], list[ScoredFile]]:
    meta = metadata_search.search(intent.filters, limit=POOL) if _has_filters(intent) else []
    kw = keyword_search.search(intent.search_query, limit=POOL)
    sem = semantic_search.search(intent.search_query, limit=POOL)
    return meta, kw, sem


def stage_hybrid_fusion(intent: QueryIntent) -> list[ScoredFile]:
    meta, kw, sem = _retrieve_lists(intent)
    return reciprocal_rank_fusion([lst for lst in (meta, kw, sem) if lst], limit=POOL)


def stage_plus_reranker(intent: QueryIntent) -> list[ScoredFile]:
    fused = stage_hybrid_fusion(intent)
    return rerank(intent.search_query, fused, top_k=POOL)


def stage_plus_personalization(intent: QueryIntent, user_id: int) -> list[ScoredFile]:
    reranked = stage_plus_reranker(intent)
    return WeightedSumPersonalizer().rank(user_id, reranked)


def _metrics_for(retrieved_ids: list[int], relevant: set[int]) -> dict[str, float]:
    return {
        "precision_at_5": precision_at_k(retrieved_ids, relevant, K_PRECISION),
        "recall_at_5": recall_at_k(retrieved_ids, relevant, K_RECALL),
        "ndcg_at_10": ndcg_at_k(retrieved_ids, relevant, K_NDCG),
        "mrr": mrr(retrieved_ids, relevant),
    }


def run_ablation(eval_set: list[dict] | None = None, user_ids: list[int] | None = None) -> dict[str, list[dict]]:
    eval_set = eval_set if eval_set is not None else load_eval_set()
    if user_ids is None:
        with get_session() as session:
            user_ids = [u.id for u in session.exec(select(UserRecord)).all()]

    per_stage_rows: dict[str, list[dict]] = {s: [] for s in STAGE_ORDER}

    for q in eval_set:
        relevant = set(q["ground_truth_file_ids"])
        intent = extract_entities(q["query"])

        stage1 = stage_metadata_only(intent)
        stage2 = stage_plus_keyword(intent)
        stage3 = stage_plus_semantic(intent)
        stage4 = _ids(stage_hybrid_fusion(intent))
        stage5 = _ids(stage_plus_reranker(intent))

        row_base = {"id": q["id"], "difficulty": q["difficulty"], "query": q["query"]}
        per_stage_rows["1_metadata_only"].append({**row_base, **_metrics_for(stage1, relevant)})
        per_stage_rows["2_plus_keyword"].append({**row_base, **_metrics_for(stage2, relevant)})
        per_stage_rows["3_plus_semantic"].append({**row_base, **_metrics_for(stage3, relevant)})
        per_stage_rows["4_hybrid_fusion"].append({**row_base, **_metrics_for(stage4, relevant)})
        per_stage_rows["5_plus_reranker"].append({**row_base, **_metrics_for(stage5, relevant)})

        stage6_per_metric = {m: [] for m in METRIC_NAMES}
        for uid in user_ids:
            stage6 = _ids(stage_plus_personalization(intent, uid))
            m = _metrics_for(stage6, relevant)
            for k, v in m.items():
                stage6_per_metric[k].append(v)
        per_stage_rows["6_plus_personalization"].append(
            {**row_base, **{k: mean(v) for k, v in stage6_per_metric.items()}}
        )

    return per_stage_rows


def summarize(per_stage_rows: dict[str, list[dict]]) -> dict[str, dict]:
    return {
        stage: {m: mean(r[m] for r in rows) for m in METRIC_NAMES}
        for stage, rows in per_stage_rows.items()
    }


def plot_ladder(summary: dict[str, dict], out_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    stages = STAGE_ORDER
    x = range(len(stages))
    width = 0.2

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, m in enumerate(METRIC_NAMES):
        vals = [summary[s][m] for s in stages]
        ax.bar([xi + i * width for xi in x], vals, width, label=m)

    labels = [s.split("_", 1)[1].replace("_", " ") for s in stages]
    ax.set_xticks([xi + 1.5 * width for xi in x])
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("score")
    ax.set_title("Ablation: marginal contribution of each retrieval component")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    if not EVAL_SET_PATH.exists():
        raise SystemExit("eval/eval_set.json not found. Run eval/build_eval_set.py first.")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("[ablation_study] running the component ladder across the eval set...")
    per_stage_rows = run_ablation()
    summary = summarize(per_stage_rows)

    csv_path = RESULTS_DIR / "ablation_study_per_query.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["stage", "id", "difficulty", "query", *METRIC_NAMES])
        for stage, rows in per_stage_rows.items():
            for r in rows:
                writer.writerow([stage, r["id"], r["difficulty"], r["query"], *[r[m] for m in METRIC_NAMES]])

    summary_path = RESULTS_DIR / "ablation_study_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))

    plot_path = RESULTS_DIR / "ablation_study.png"
    plot_ladder(summary, plot_path)

    print(json.dumps(summary, indent=2))
    print(f"\nwrote {csv_path}\nwrote {summary_path}\nwrote {plot_path}")


if __name__ == "__main__":
    main()
