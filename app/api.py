"""FastAPI endpoints.

/api/query is the simple synchronous entry point (full result in one response).
/api/query/stream is the SSE entry point the live trace panel uses: it streams a
"route" event as soon as the tier is decided, a "stage" event as each pipeline stage
finishes (or is marked skipped), and a final "done" event with results — driven by
LangGraph's own `.stream()`, not a fake progress bar.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from sqlmodel import select

from app.agent.graph import GraphState, get_compiled_graph, run_query
from app.db import get_session
from app.models import FileRecord, UserRecord
from app.personalization.feedback import record_feedback
from app.personalization.profile_builder import build_user_profile
from app.personalization.temporal_patterns import current_context_boost, detect_recurring_patterns
from app.tracing import RoutingTrace

REPO_ROOT = Path(__file__).resolve().parent.parent
EVAL_RESULTS_DIR = REPO_ROOT / "eval" / "results"

app = FastAPI(title="Agentic File Recommendation and Retrieval API")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)


class QueryRequest(BaseModel):
    query: str
    user_id: int


@app.get("/api/users")
def list_users():
    with get_session() as session:
        users = session.exec(select(UserRecord)).all()
    return [{"id": u.id, "name": u.name, "persona_key": u.persona_key} for u in users]


class IngestRequest(BaseModel):
    root: str
    max_files: int = 300
    clear_existing: bool = False


@app.post("/api/ingest")
def ingest(req: IngestRequest):
    """Crawls a real local directory (the backend process's own filesystem — this is a
    local-first app, not a browser upload) and ingests it into the same FileRecord
    table + semantic index every other endpoint reads. This is what the UI's "index a
    real folder" flow calls; see app/datasources/filesystem_source.py."""
    from app.datasources.filesystem_source import FilesystemDataSource
    from app.retrieval.semantic_search import build_index
    from data.ingest_datasource import ingest_files

    try:
        source = FilesystemDataSource(root=req.root, max_files=req.max_files)
        raw_files = source.list_files()
    except FileNotFoundError as e:
        raise HTTPException(status_code=400, detail=str(e))

    if not raw_files:
        raise HTTPException(status_code=400, detail=f"no readable files found under {source.root}")

    with get_session() as session:
        by_type = ingest_files(raw_files, session, clear_existing=req.clear_existing)

    n_indexed = build_index(force=True)

    return {
        "root": str(source.root),
        "n_files_crawled": len(raw_files),
        "by_type": by_type,
        "n_indexed_total": n_indexed,
        "cleared_existing": req.clear_existing,
    }


@app.post("/api/query")
def query(req: QueryRequest):
    return run_query(req.query, req.user_id)


class FeedbackRequest(BaseModel):
    user_id: int
    file_id: int
    query: str
    signal: str  # "thumbs_up" | "thumbs_down" | "opened"


@app.post("/api/feedback")
def feedback(req: FeedbackRequest):
    try:
        event = record_feedback(req.user_id, req.file_id, req.query, req.signal)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    return {"id": event.id, "recorded": True}


@app.get("/api/query/stream")
def query_stream(query: str, user_id: int):
    def event_generator():
        trace = RoutingTrace()
        initial_state: GraphState = {
            "query": query,
            "user_id": user_id,
            "now": datetime.now(),
            "trace": trace,
        }

        seen_stages = 0
        tier_emitted = False
        final_results = None

        for step in get_compiled_graph().stream(initial_state):
            node_name, partial = next(iter(step.items()))

            if not tier_emitted and trace.tier:
                yield _sse({"type": "route", "tier": trace.tier, "rationale": trace.rationale})
                tier_emitted = True

            while seen_stages < len(trace.stages):
                s = trace.stages[seen_stages]
                yield _sse({
                    "type": "stage",
                    "name": s.name,
                    "duration_ms": round(s.duration_ms, 2),
                    "skipped": s.skipped,
                    "detail": s.detail,
                })
                seen_stages += 1

            if node_name == "finalize":
                final_results = partial.get("final_results")

        yield _sse({"type": "done", "results": final_results, "routing_trace": trace.to_dict()})

    return StreamingResponse(event_generator(), media_type="text/event-stream")


def _sse(payload: dict) -> str:
    return f"data: {json.dumps(payload)}\n\n"


@app.get("/api/personalization/{user_id}")
def personalization_insights(user_id: int):
    now = datetime.now()
    profile = build_user_profile(user_id, now=now)
    patterns = detect_recurring_patterns(user_id)
    active_boost = current_context_boost(user_id, now=now)

    top_files = sorted(profile.file_frequency.items(), key=lambda x: x[1], reverse=True)[:5]
    all_file_ids = {fid for fid, _ in top_files} | {fid for p in patterns for fid in p.file_ids}
    with get_session() as session:
        records = session.exec(select(FileRecord).where(FileRecord.id.in_(all_file_ids))).all()
    name_by_id = {r.id: r.filename for r in records}

    return {
        "preferred_file_types": profile.preferred_file_types[:5],
        "top_files_by_frequency": [
            {"file_id": fid, "filename": name_by_id.get(fid, "?"), "frequency": round(freq, 3)}
            for fid, freq in top_files
        ],
        "recurring_patterns": [
            {
                "weekday": p.weekday_name,
                "hour": p.hour,
                "file_ids": p.file_ids,
                "filenames": [name_by_id.get(fid, "?") for fid in p.file_ids],
                "confidence": round(p.confidence, 2),
            }
            for p in patterns
        ],
        "active_context_boost_now": bool(active_boost),
        "active_context_files": [name_by_id.get(fid, "?") for fid in active_boost] if active_boost else [],
    }


@app.get("/api/eval/baseline-comparison")
def baseline_comparison():
    path = EVAL_RESULTS_DIR / "baseline_comparison_summary.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail="run eval/baseline_comparison.py first")
    return json.loads(path.read_text())
