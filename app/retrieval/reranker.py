"""Cross-encoder reranking — deep route only.

A cross-encoder scores (query, document) pairs jointly (unlike the bi-encoder used for
semantic_search, which embeds them independently), which is more accurate but too slow
to run over the whole corpus — so it only ever runs over the top candidates a cheaper
retriever already narrowed down to. This is why it's gated to the deep route.

Image-sourced candidates (app.retrieval.image_search, CLIP content matching) are
deliberately excluded from cross-encoder scoring and keep their pre-rerank fused
score instead. This isn't an oversight: the cross-encoder only ever sees
`filename + extracted_text` — for an image file, that's just its caption, never the
actual pixels — so text-reranking a CLIP-matched image would judge it on caption
text it was never selected for, discarding the real (visual) relevance signal that
got it into the candidate pool. Found via eval/image_content_search.py: without this
split, full-pipeline NDCG@10 on image-content queries was ~0.30 despite the isolated
CLIP retriever scoring ~0.83 on the same queries — the reranker was overriding a
correct visual match with an irrelevant text judgment.
"""

from __future__ import annotations

from functools import lru_cache

from sqlmodel import select

from app.db import get_session
from app.models import FileRecord
from app.retrieval.base import ScoredFile

MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
MAX_CHARS = 1000


@lru_cache(maxsize=1)
def _get_cross_encoder():
    from sentence_transformers import CrossEncoder

    return CrossEncoder(MODEL_NAME)


def _is_image_sourced(candidate: ScoredFile) -> bool:
    return "image_semantic" in candidate.explanation


def rerank(query: str, candidates: list[ScoredFile], top_k: int | None = None) -> list[ScoredFile]:
    if not candidates:
        return []

    image_sourced = [c for c in candidates if _is_image_sourced(c)]
    text_candidates = [c for c in candidates if not _is_image_sourced(c)]

    reranked_text: list[ScoredFile] = []
    if text_candidates:
        file_ids = [c.file_id for c in text_candidates]
        with get_session() as session:
            records = session.exec(select(FileRecord).where(FileRecord.id.in_(file_ids))).all()
        text_by_id = {r.id: f"{r.filename}\n{r.extracted_text[:MAX_CHARS]}" for r in records}

        pairs = [(query, text_by_id.get(c.file_id, "")) for c in text_candidates]
        encoder = _get_cross_encoder()
        raw_scores = encoder.predict(pairs)

        lo, hi = float(min(raw_scores)), float(max(raw_scores))
        span = (hi - lo) or 1.0

        reranked_text = [
            ScoredFile(
                file_id=c.file_id,
                score=(float(s) - lo) / span,  # min-max normalize to [0, 1] within this batch
                source="reranker",
                explanation=f"cross-encoder relevance score {float(s):.3f} ({MODEL_NAME})",
            )
            for c, s in zip(text_candidates, raw_scores)
        ]

    combined = reranked_text + image_sourced
    combined.sort(key=lambda r: r.score, reverse=True)
    return combined[:top_k] if top_k else combined
