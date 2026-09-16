"""Per-result natural-language explanations — standard/deep routes only.

Batches every candidate into a single LLM call (not one call per file) so the
explanation stage's cost is O(1) LLM calls regardless of how many results are shown.
Falls back to each retriever's own rule-based `.explanation` string (e.g. "BM25 keyword
match", "semantic similarity 0.72") when no API key is configured — so results are
always explained, just less fluently without an LLM.
"""

from __future__ import annotations

import re

from app.models import FileRecord
from app.retrieval.base import ScoredFile

_LINE_RE = re.compile(r"id=(\d+):\s*(.+)")


def generate_explanations(
    query: str,
    results: list[ScoredFile],
    records_by_id: dict[int, FileRecord],
    max_files: int = 8,
) -> dict[int, str]:
    items = results[:max_files]
    fallback = {r.file_id: r.explanation for r in items}

    from app.llm_client import get_client

    client = get_client()
    if not client.is_available or not items:
        return fallback

    try:
        listing_lines = []
        for i, r in enumerate(items, start=1):
            rec = records_by_id.get(r.file_id)
            if rec is None:
                continue
            snippet = rec.extracted_text[:150].replace("\n", " ")
            listing_lines.append(f"{i}. [id={r.file_id}] {rec.filename} — snippet: \"{snippet}\"")

        prompt = (
            f'A user searched for: "{query}"\n\n'
            f"Candidate files:\n" + "\n".join(listing_lines) + "\n\n"
            "For each file, write ONE short sentence (under 20 words) explaining why it "
            "is or isn't relevant to the query. Respond with exactly one line per file, "
            "formatted as:\nid=<id>: <explanation>"
        )
        result = client.complete(prompt, max_tokens=500, temperature=0.4)

        explanations = dict(fallback)
        for line in result.text.strip().splitlines():
            match = _LINE_RE.match(line.strip())
            if match:
                file_id = int(match.group(1))
                if file_id in fallback:
                    explanations[file_id] = match.group(2).strip()
        return explanations
    except Exception:
        return fallback
