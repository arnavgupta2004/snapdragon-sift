"""Semantic search: local sentence-transformer embeddings persisted in ChromaDB.

Local and free (no embedding API calls), per the tech-stack constraint. Model defaults
to all-MiniLM-L6-v2 (384-dim, fast); override with SEMANTIC_MODEL=BAAI/bge-small-en-v1.5
if you want the other model named in the spec.

Usage:
    python -m app.retrieval.semantic_search --build   # (re)build the index from the DB
    python -m app.retrieval.semantic_search "find that thing about audio deepfakes"
"""

from __future__ import annotations

import argparse
import os
from functools import lru_cache
from pathlib import Path

from sqlmodel import select

from app.db import get_session
from app.models import FileRecord
from app.retrieval.base import ScoredFile

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
CHROMA_DIR = REPO_ROOT / "data" / "chroma"
COLLECTION_NAME = "files"
MODEL_NAME = os.environ.get("SEMANTIC_MODEL", "all-MiniLM-L6-v2")
MAX_CHARS = 2000  # per-file text truncation before embedding; corpus docs are short


@lru_cache(maxsize=1)
def _get_embedder():
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(MODEL_NAME)


@lru_cache(maxsize=1)
def _get_collection():
    import chromadb

    # anonymized_telemetry=False: chromadb otherwise phones home to PostHog on every
    # client init — see app/retrieval/image_search.py's _get_collection for why this
    # matters beyond just privacy (it caused an 11+ minute hang during development).
    client = chromadb.PersistentClient(
        path=str(CHROMA_DIR), settings=chromadb.Settings(anonymized_telemetry=False)
    )
    return client.get_or_create_collection(name=COLLECTION_NAME, metadata={"hnsw:space": "cosine"})


def build_index(force: bool = False) -> int:
    """Embeds every file's extracted_text and upserts into Chroma. Returns count indexed.
    Cheap to call repeatedly: skips work if the index already has the right file count,
    unless force=True."""
    collection = _get_collection()

    with get_session() as session:
        files = session.exec(select(FileRecord)).all()
    files = [f for f in files if f.id is not None]

    if not force and collection.count() == len(files):
        return collection.count()

    if force and collection.count() > 0:
        existing_ids = collection.get()["ids"]
        if existing_ids:
            collection.delete(ids=existing_ids)

    embedder = _get_embedder()
    ids = [str(f.id) for f in files]
    docs = [f"{f.filename}\n{f.extracted_text[:MAX_CHARS]}" for f in files]
    embeddings = embedder.encode(docs, show_progress_bar=False, normalize_embeddings=True)

    collection.upsert(ids=ids, embeddings=embeddings.tolist(), documents=docs)
    return len(files)


def search(query: str, limit: int = 10) -> list[ScoredFile]:
    build_index(force=False)  # no-op if already current
    collection = _get_collection()
    if collection.count() == 0:
        return []

    embedder = _get_embedder()
    query_vec = embedder.encode([query], normalize_embeddings=True)[0].tolist()

    result = collection.query(query_embeddings=[query_vec], n_results=min(limit, collection.count()))
    ids = result["ids"][0]
    distances = result["distances"][0]  # cosine distance in [0, 2], 0 = identical

    results = []
    for file_id_str, distance in zip(ids, distances):
        similarity = max(0.0, 1.0 - distance)
        results.append(
            ScoredFile(
                file_id=int(file_id_str),
                score=similarity,
                source="semantic",
                explanation=f"semantic similarity {similarity:.3f} ({MODEL_NAME})",
            )
        )
    return results


def _main() -> None:
    parser = argparse.ArgumentParser(description="Semantic (embedding) search over the corpus DB.")
    parser.add_argument("query", nargs="?")
    parser.add_argument("--build", action="store_true", help="force-rebuild the index")
    parser.add_argument("--limit", type=int, default=10)
    args = parser.parse_args()

    if args.build:
        n = build_index(force=True)
        print(f"indexed {n} files into Chroma at {CHROMA_DIR}")
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
