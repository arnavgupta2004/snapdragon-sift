"""The LangGraph state graph: query -> entities -> route -> [fast|standard|deep]
retrieval -> personalization -> [explanation] -> results.

Explicit nodes and edges (not a hidden agent loop) so the pipeline is inspectable and
diagrammable, and so the routing trace can be threaded through every node without
guessing what ran. This is the file the UI's live trace panel and
eval/latency_comparison.py are both ultimately reading the behavior of.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph
from sqlmodel import select

from app.agent.explain import generate_explanations
from app.agent.query_understanding import QueryIntent, enrich_with_llm, extract_entities
from app.agent.router import RouteDecision, route
from app.config import CANDIDATE_POOL_SIZE, FINAL_RESULT_COUNT
from app.db import get_session
from app.models import FileRecord
from app.personalization.personalized_ranker import WeightedSumPersonalizer
from app.retrieval import filename_search, image_search, keyword_search, metadata_search, semantic_search
from app.retrieval.base import ScoredFile
from app.retrieval.hybrid_fusion import reciprocal_rank_fusion
from app.retrieval.metadata_search import MetadataFilters
from app.retrieval.reranker import rerank
from app.tracing import RoutingTrace


class GraphState(TypedDict, total=False):
    query: str
    user_id: int
    now: datetime
    intent: QueryIntent
    decision: RouteDecision
    candidates: list[ScoredFile]
    personalized: list[ScoredFile]
    explanations: dict[int, str]
    final_results: list[dict[str, Any]]
    trace: RoutingTrace


def _has_filters(filters: MetadataFilters) -> bool:
    return bool(filters.file_types or filters.modified_after or filters.topic_cluster or filters.filename_contains)


# --- nodes ---------------------------------------------------------------

def node_extract_entities(state: GraphState) -> dict:
    trace = state["trace"]
    with trace.stage("entity_extraction"):
        intent = extract_entities(state["query"])
    return {"intent": intent}


def node_route(state: GraphState) -> dict:
    trace = state["trace"]
    with trace.stage("routing"):
        decision = route(state["intent"])
    trace.tier = decision.tier
    trace.rationale = decision.rationale
    trace.used_llm_fallback_classification = decision.used_llm_fallback
    trace.used_learned_router = decision.used_learned_router
    if decision.used_llm_fallback:
        trace.record_llm_call()
    return {"decision": decision}


def _tier_after_route(state: GraphState) -> str:
    tier = state["decision"].tier
    return {"fast": "fast_retrieve", "standard": "standard_retrieve", "deep": "enrich_query"}[tier]


def node_enrich_query(state: GraphState) -> dict:
    trace = state["trace"]
    with trace.stage("query_enrichment"):
        intent = enrich_with_llm(state["intent"])
    if intent.used_llm:
        trace.record_llm_call()
    return {"intent": intent}


def node_fast_retrieve(state: GraphState) -> dict:
    trace = state["trace"]
    intent = state["intent"]

    with trace.stage("filename_search"):
        fname_results = filename_search.search(intent.raw_query, limit=CANDIDATE_POOL_SIZE)
        if intent.exact_filename:
            exact = filename_search.exact_match(intent.exact_filename)
            if exact is not None:
                fname_results = [exact] + [r for r in fname_results if r.file_id != exact.file_id]

    if intent.exact_filename:
        # The filename's own extension (e.g. ".pptx") can spuriously trigger a
        # file-type filter match in extract_filters — running metadata_search on top
        # of an already-resolved exact match would just flood the fusion with
        # unrelated same-type files and dilute the one file actually named.
        trace.skip("metadata_search", "exact filename already resolved deterministically")
        meta_results = []
    elif _has_filters(intent.filters):
        with trace.stage("metadata_search"):
            meta_results = metadata_search.search(intent.filters, limit=CANDIDATE_POOL_SIZE)
    else:
        trace.skip("metadata_search", "no file-type/date filter extracted from query")
        meta_results = []

    trace.skip("keyword_search", "fast route: filename+metadata only")
    trace.skip("semantic_search", "fast route: filename+metadata only")
    trace.skip("image_search", "fast route: filename+metadata only")
    trace.skip("reranker", "fast route: skips the cross-encoder entirely")

    with trace.stage("hybrid_fusion", detail="filename+metadata"):
        candidates = reciprocal_rank_fusion(
            [lst for lst in (fname_results, meta_results) if lst], limit=CANDIDATE_POOL_SIZE
        )
    if not candidates:
        candidates = fname_results or meta_results

    return {"candidates": candidates}


def node_standard_retrieve(state: GraphState) -> dict:
    trace = state["trace"]
    intent = state["intent"]

    trace.skip("filename_search", "standard route: metadata+keyword+semantic+image only")

    if _has_filters(intent.filters):
        with trace.stage("metadata_search"):
            meta_results = metadata_search.search(intent.filters, limit=CANDIDATE_POOL_SIZE)
    else:
        trace.skip("metadata_search", "no file-type/date filter extracted from query")
        meta_results = []

    with trace.stage("keyword_search"):
        kw_results = keyword_search.search(intent.search_query, limit=CANDIDATE_POOL_SIZE)

    with trace.stage("semantic_search"):
        sem_results = semantic_search.search(intent.search_query, limit=CANDIDATE_POOL_SIZE)

    with trace.stage("image_search", detail="CLIP content match over indexed images"):
        img_results = image_search.search(intent.search_query, limit=CANDIDATE_POOL_SIZE)

    trace.skip("reranker", "standard route: no cross-encoder pass")

    with trace.stage("hybrid_fusion", detail="metadata+keyword+semantic+image"):
        candidates = reciprocal_rank_fusion(
            [lst for lst in (meta_results, kw_results, sem_results, img_results) if lst],
            limit=CANDIDATE_POOL_SIZE,
        )

    return {"candidates": candidates}


def node_deep_retrieve(state: GraphState) -> dict:
    trace = state["trace"]
    intent = state["intent"]

    trace.skip("filename_search", "deep route: relies on hybrid semantic+keyword instead")

    if _has_filters(intent.filters):
        with trace.stage("metadata_search"):
            meta_results = metadata_search.search(intent.filters, limit=CANDIDATE_POOL_SIZE)
    else:
        trace.skip("metadata_search", "no file-type/date filter extracted from query")
        meta_results = []

    with trace.stage("keyword_search"):
        kw_results = keyword_search.search(intent.search_query, limit=CANDIDATE_POOL_SIZE)

    with trace.stage("semantic_search"):
        sem_results = semantic_search.search(intent.search_query, limit=CANDIDATE_POOL_SIZE)

    with trace.stage("image_search", detail="CLIP content match over indexed images"):
        img_results = image_search.search(intent.search_query, limit=CANDIDATE_POOL_SIZE)

    with trace.stage("hybrid_fusion", detail="metadata+keyword+semantic+image"):
        fused = reciprocal_rank_fusion(
            [lst for lst in (meta_results, kw_results, sem_results, img_results) if lst],
            limit=CANDIDATE_POOL_SIZE,
        )

    with trace.stage("reranker"):
        candidates = rerank(intent.search_query, fused, top_k=CANDIDATE_POOL_SIZE)

    return {"candidates": candidates}


def node_personalize(state: GraphState) -> dict:
    trace = state["trace"]

    if state["intent"].exact_filename:
        # An exact filename in the query is a deterministic ask for *that* file —
        # personalization re-ranking would only risk bumping it out of the top slot
        # for a user who happens to favor other files, which defeats the point of
        # naming it. Skip re-ranking, keep retrieval order as-is.
        trace.skip(
            "personalization",
            "query names an exact filename — skipped so the named file isn't outranked by unrelated user history",
        )
        return {"personalized": state["candidates"][:FINAL_RESULT_COUNT]}

    with trace.stage("personalization"):
        personalized = WeightedSumPersonalizer().rank(
            state["user_id"], state["candidates"], now=state["now"]
        )
    return {"personalized": personalized[:FINAL_RESULT_COUNT]}


def _tier_after_personalize(state: GraphState) -> str:
    return "explain_llm" if state["decision"].tier == "deep" else "explain_skip"


def node_explain_llm(state: GraphState) -> dict:
    trace = state["trace"]
    from app.llm_client import get_client

    with get_session() as session:
        ids = [r.file_id for r in state["personalized"]]
        records = session.exec(select(FileRecord).where(FileRecord.id.in_(ids))).all()
    records_by_id = {r.id: r for r in records}

    with trace.stage("llm_explanation"):
        explanations = generate_explanations(state["query"], state["personalized"], records_by_id)
    if get_client().is_available:
        trace.record_llm_call()

    return {"explanations": explanations}


def node_explain_skip(state: GraphState) -> dict:
    trace = state["trace"]
    trace.skip("llm_explanation", f"{state['decision'].tier} route uses rule-based explanations only")
    explanations = {r.file_id: r.explanation for r in state["personalized"]}
    return {"explanations": explanations}


def node_finalize(state: GraphState) -> dict:
    trace = state["trace"]
    with trace.stage("finalize"):
        with get_session() as session:
            ids = [r.file_id for r in state["personalized"]]
            records = session.exec(select(FileRecord).where(FileRecord.id.in_(ids))).all()
        records_by_id = {r.id: r for r in records}

        final_results = []
        for r in state["personalized"]:
            rec = records_by_id.get(r.file_id)
            if rec is None:
                continue
            final_results.append(
                {
                    "file_id": r.file_id,
                    "filename": rec.filename,
                    "path": rec.path,
                    "file_type": rec.file_type,
                    "topic_cluster": rec.topic_cluster,
                    "modified_at": rec.modified_at.isoformat(),
                    "score": round(r.score, 4),
                    "explanation": state["explanations"].get(r.file_id, r.explanation),
                }
            )
    return {"final_results": final_results}


# --- graph assembly --------------------------------------------------------

def build_graph():
    graph = StateGraph(GraphState)
    graph.add_node("extract_entities", node_extract_entities)
    graph.add_node("route", node_route)
    graph.add_node("enrich_query", node_enrich_query)
    graph.add_node("fast_retrieve", node_fast_retrieve)
    graph.add_node("standard_retrieve", node_standard_retrieve)
    graph.add_node("deep_retrieve", node_deep_retrieve)
    graph.add_node("personalize", node_personalize)
    graph.add_node("explain_llm", node_explain_llm)
    graph.add_node("explain_skip", node_explain_skip)
    graph.add_node("finalize", node_finalize)

    graph.add_edge(START, "extract_entities")
    graph.add_edge("extract_entities", "route")
    graph.add_conditional_edges(
        "route", _tier_after_route,
        {"fast_retrieve": "fast_retrieve", "standard_retrieve": "standard_retrieve", "enrich_query": "enrich_query"},
    )
    graph.add_edge("enrich_query", "deep_retrieve")
    graph.add_edge("fast_retrieve", "personalize")
    graph.add_edge("standard_retrieve", "personalize")
    graph.add_edge("deep_retrieve", "personalize")
    graph.add_conditional_edges(
        "personalize", _tier_after_personalize,
        {"explain_llm": "explain_llm", "explain_skip": "explain_skip"},
    )
    graph.add_edge("explain_llm", "finalize")
    graph.add_edge("explain_skip", "finalize")
    graph.add_edge("finalize", END)

    return graph.compile()


_compiled_graph = None


def get_compiled_graph():
    global _compiled_graph
    if _compiled_graph is None:
        _compiled_graph = build_graph()
    return _compiled_graph


def run_query(query: str, user_id: int, now: datetime | None = None) -> dict[str, Any]:
    trace = RoutingTrace()
    initial_state: GraphState = {
        "query": query,
        "user_id": user_id,
        "now": now or datetime.now(),
        "trace": trace,
    }
    final_state = get_compiled_graph().invoke(initial_state)
    return {
        "query": query,
        "results": final_state["final_results"],
        "routing_trace": final_state["trace"].to_dict(),
    }
