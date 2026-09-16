"""CLIP-based image *content* search — embeds actual pixels, not filename/caption
text, into a shared text-image embedding space so a query like "find that photo from
the internship" can match on what's visually in an image, not just what it's named.

This is the direct answer to the motivating gap: app.retrieval.semantic_search (the
text embedder, all-MiniLM-L6-v2) never looks at image bytes at all — a .png/.jpg file
was previously only findable by filename/metadata search. This module is what makes
it findable by content.

Runs entirely locally via sentence-transformers' clip-ViT-B-32 wrapper (no cloud
call) — consistent with this project's on-device-by-default LLM backend, and it's a
genuinely separate embedding space from the text embedder (512-dim CLIP vs. 384-dim
MiniLM), so it lives in its own Chroma collection rather than sharing
semantic_search.py's.
"""

from __future__ import annotations

import argparse
from functools import lru_cache
from pathlib import Path

from sqlmodel import select

from app.config import IMAGE_CLIP_MODEL_NAME as MODEL_NAME
from app.db import get_session
from app.models import FileRecord
from app.retrieval.base import ScoredFile

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
CHROMA_DIR = REPO_ROOT / "data" / "chroma"
COLLECTION_NAME = "images_clip"
IMAGE_EXTENSIONS = {"png", "jpg", "jpeg"}


@lru_cache(maxsize=1)
def _get_clip_model():
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(MODEL_NAME)


@lru_cache(maxsize=1)
def _get_collection():
    import chromadb

    # anonymized_telemetry=False: chromadb otherwise phones home to PostHog on every
    # client init — silently contradicting "runs entirely on-device, nothing leaves
    # the machine" and, when that network call stalls, hanging the whole process (this
    # is exactly what happened during development of eval/local_vs_cloud.py: a script
    # got stuck for 11+ minutes at 0% CPU on a telemetry connection, not on any LLM
    # backend at all).
    client = chromadb.PersistentClient(
        path=str(CHROMA_DIR), settings=chromadb.Settings(anonymized_telemetry=False)
    )
    return client.get_or_create_collection(name=COLLECTION_NAME, metadata={"hnsw:space": "cosine"})


def resolve_image_path(record: FileRecord) -> Path | None:
    """Real files ingested via FilesystemDataSource store an absolute, self-locating
    path; synthetic files store a path relative to data/files_corpus/. Either way,
    this is how we find the actual pixels to embed."""
    p = Path(record.path)
    candidate = p if p.is_absolute() else REPO_ROOT / "data" / "files_corpus" / record.path
    return candidate if candidate.exists() else None


def build_index(force: bool = False) -> int:
    """Embeds every image file's actual pixel content and upserts into Chroma. Skips
    work if the index already covers every image file, unless force=True — same
    pattern as semantic_search.build_index."""
    collection = _get_collection()

    with get_session() as session:
        records = session.exec(
            select(FileRecord).where(FileRecord.file_type.in_(IMAGE_EXTENSIONS))
        ).all()
    records = [r for r in records if r.id is not None]

    if not force and collection.count() == len(records):
        return collection.count()

    if force and collection.count() > 0:
        existing_ids = collection.get()["ids"]
        if existing_ids:
            collection.delete(ids=existing_ids)

    from PIL import Image

    model = _get_clip_model()
    ids: list[str] = []
    embeddings: list[list[float]] = []
    docs: list[str] = []

    for r in records:
        path = resolve_image_path(r)
        if path is None:
            continue
        try:
            img = Image.open(path).convert("RGB")
        except Exception:
            continue
        vec = model.encode([img], normalize_embeddings=True, show_progress_bar=False)[0]
        ids.append(str(r.id))
        embeddings.append(vec.tolist())
        docs.append(r.filename)

    if ids:
        collection.upsert(ids=ids, embeddings=embeddings, documents=docs)
    return len(ids)


def search(query: str, limit: int = 10) -> list[ScoredFile]:
    build_index(force=False)
    collection = _get_collection()
    if collection.count() == 0:
        return []

    model = _get_clip_model()
    query_vec = model.encode([query], normalize_embeddings=True, show_progress_bar=False)[0].tolist()

    result = collection.query(query_embeddings=[query_vec], n_results=min(limit, collection.count()))
    ids = result["ids"][0]
    distances = result["distances"][0]

    results = []
    for file_id_str, distance in zip(ids, distances):
        similarity = max(0.0, 1.0 - distance)
        results.append(
            ScoredFile(
                file_id=int(file_id_str),
                score=similarity,
                source="image_semantic",
                explanation=f"image content match {similarity:.3f} (CLIP, {MODEL_NAME})",
            )
        )
    return results


def _main() -> None:
    parser = argparse.ArgumentParser(description="CLIP image-content search over the corpus DB.")
    parser.add_argument("query", nargs="?")
    parser.add_argument("--build", action="store_true", help="force-rebuild the image index")
    parser.add_argument("--limit", type=int, default=10)
    args = parser.parse_args()

    if args.build:
        n = build_index(force=True)
        print(f"indexed {n} images into Chroma at {CHROMA_DIR}")
        if not args.query:
            return

    if not args.query:
        parser.error("query is required unless --build is passed alone")

    with get_session() as session:
        id_to_name = {f.id: f.filename for f in session.exec(select(FileRecord)).all()}

    for r in search(args.query, limit=args.limit):
        print(f"{r.score:.3f}  {id_to_name.get(r.file_id, '?')}  ({r.explanation})")


if __name__ == "__main__":
    _main()
