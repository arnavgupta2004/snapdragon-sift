"""Query understanding: entity/filter extraction and intent signals.

Split deliberately into a cheap, always-run rule-based pass (extract_entities) and an
optional LLM enrichment pass (enrich_with_llm) that the graph only invokes for the deep
route. Every query pays the rule-based cost; only ambiguous, deep-routed queries pay
for an LLM call here — that's what keeps the fast route's "zero LLM calls" claim true.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta

from app.retrieval.metadata_search import MetadataFilters

FILENAME_PATTERN = re.compile(
    r"\b[\w][\w\-]*\.(py|md|txt|docx|pdf|xlsx|pptx|png|jpg|jpeg|csv)\b", re.IGNORECASE
)

EXTENSION_KEYWORDS: dict[str, str] = {
    "pptx": "pptx", "ppt": "pptx", "powerpoint": "pptx", "slides": "pptx", "slide deck": "pptx",
    "xlsx": "xlsx", "excel": "xlsx", "spreadsheet": "xlsx",
    "docx": "docx", "word doc": "docx", "word document": "docx",
    "pdf": "pdf", "pdfs": "pdf",
    "python": "py", "py file": "py", "script": "py", "code file": "py",
    "markdown": "md", "md file": "md",
    "image": "png", "images": "png", "photo": "png", "picture": "png", "screenshot": "png",
}

TIME_KEYWORDS: dict[str, int] = {
    "today": 1, "yesterday": 2, "this week": 7, "last week": 14,
    "this month": 30, "last month": 60, "recent": 14, "recently": 14,
}

VAGUE_MARKERS = [
    "that thing", "something about", "a while back", "a few weeks ago", "can't remember",
    "cant remember", "i think", "sort of", "kind of", "whatever it was", "not sure",
    "the thing i was working on", "some file", "somewhere", "vaguely",
]

# Words that carry no topical content — stripped before judging whether a query is
# "just" a filter (fast route) vs. has real topical content needing keyword/semantic
# search (standard/deep route). Without this, "find my recent notes about transformers"
# would wrongly fast-route on the word "recent" alone, ignoring "transformers".
FILLER_WORDS = {
    "show", "me", "my", "file", "files", "from", "the", "a", "an", "of", "for",
    "please", "can", "you", "find", "get", "open", "list", "about", "on", "in",
    "to", "is", "are", "was", "were", "and", "or",
}


@dataclass
class QueryIntent:
    raw_query: str
    exact_filename: str | None = None
    filters: MetadataFilters = field(default_factory=MetadataFilters)
    vague_marker_hit: bool = False
    word_count: int = 0
    residual_content_word_count: int = 0  # word count after stripping filter/filler terms
    search_query: str = ""  # what gets fed to keyword/semantic search; may be LLM-rewritten
    used_llm: bool = False


def find_exact_filename(query: str) -> str | None:
    match = FILENAME_PATTERN.search(query)
    return match.group(0) if match else None


def extract_filters(query: str) -> MetadataFilters:
    q = query.lower()

    file_types = sorted({ext for kw, ext in EXTENSION_KEYWORDS.items() if kw in q})

    since_days = None
    for kw, days in TIME_KEYWORDS.items():
        if kw in q:
            since_days = days if since_days is None else min(since_days, days)

    return MetadataFilters(
        file_types=file_types or None,
        modified_after=(datetime.now() - timedelta(days=since_days)) if since_days else None,
    )


def has_vague_marker(query: str) -> bool:
    q = query.lower()
    return any(marker in q for marker in VAGUE_MARKERS)


def count_residual_content_words(query: str) -> int:
    """Word count remaining after stripping filter phrases (extension/time keywords)
    and generic filler words — what's left is the actual topical content of the query."""
    q = query.lower()
    for kw in list(EXTENSION_KEYWORDS) + list(TIME_KEYWORDS):
        q = q.replace(kw, " ")
    tokens = re.findall(r"[a-z0-9]+", q)
    return len([t for t in tokens if t not in FILLER_WORDS])


def extract_entities(query: str) -> QueryIntent:
    """Cheap, rule-based, always run — this is the part of 'query understanding' every
    tier pays for, fast route included."""
    return QueryIntent(
        raw_query=query,
        exact_filename=find_exact_filename(query),
        filters=extract_filters(query),
        vague_marker_hit=has_vague_marker(query),
        word_count=len(query.split()),
        residual_content_word_count=count_residual_content_words(query),
        search_query=query,
    )


def enrich_with_llm(intent: QueryIntent) -> QueryIntent:
    """Deep-route-only enrichment: asks the LLM to rewrite a vague/conversational query
    into a cleaner phrase for semantic search. Best-effort — falls back to the raw query
    on any failure or if no API key is configured, so the graph never breaks on this."""
    from app.llm_client import get_client

    client = get_client()
    if not client.is_available:
        return intent

    try:
        prompt = (
            "Rewrite this vague file-search query into a short, concrete search phrase "
            "(under 12 words) capturing what the person is actually looking for. "
            f"Query: \"{intent.raw_query}\"\nRespond with only the rewritten phrase, "
            "no quotes, no preamble."
        )
        result = client.complete(prompt, max_tokens=60, temperature=0.3)
        rewritten = result.text.strip().strip('"')
        if rewritten:
            intent.search_query = rewritten
            intent.used_llm = True
    except Exception:
        pass
    return intent
