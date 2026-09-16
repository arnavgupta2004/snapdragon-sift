"""Shared types for retrieval components.

Every retriever (filename, metadata, keyword, semantic) returns `list[ScoredFile]` so
hybrid_fusion.py can combine ranked lists from heterogeneous scoring functions without
caring which retriever produced them.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ScoredFile:
    file_id: int
    score: float
    source: str  # which retriever produced this: "filename" | "metadata" | "keyword" | "semantic" | "hybrid" | "reranker"
    explanation: str = ""

    def __repr__(self) -> str:
        return f"ScoredFile(file_id={self.file_id}, score={self.score:.4f}, source={self.source!r})"
