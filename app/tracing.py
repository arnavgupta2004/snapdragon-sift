"""Per-stage timing and the routing trace object.

The routing trace is a first-class output, not a debug log: it's what the UI's live
trace panel renders, and what eval/latency_comparison.py reads to compute per-component
latency and LLM-call counts. Every node in app/agent/graph.py records into the same
trace object as it runs (or explicitly marks itself skipped).
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass, field


@dataclass
class StageTiming:
    name: str
    duration_ms: float
    skipped: bool = False
    detail: str = ""


@dataclass
class RoutingTrace:
    tier: str = ""
    rationale: str = ""
    used_llm_fallback_classification: bool = False
    used_learned_router: bool = False
    stages: list[StageTiming] = field(default_factory=list)
    llm_call_count: int = 0

    @contextmanager
    def stage(self, name: str, detail: str = ""):
        start = time.perf_counter()
        try:
            yield self
        finally:
            duration_ms = (time.perf_counter() - start) * 1000.0
            self.stages.append(StageTiming(name=name, duration_ms=duration_ms, detail=detail))

    def skip(self, name: str, reason: str = "") -> None:
        self.stages.append(StageTiming(name=name, duration_ms=0.0, skipped=True, detail=reason))

    def record_llm_call(self) -> None:
        self.llm_call_count += 1

    @property
    def total_duration_ms(self) -> float:
        return sum(s.duration_ms for s in self.stages if not s.skipped)

    def to_dict(self) -> dict:
        return {
            "tier": self.tier,
            "rationale": self.rationale,
            "used_llm_fallback_classification": self.used_llm_fallback_classification,
            "used_learned_router": self.used_learned_router,
            "llm_call_count": self.llm_call_count,
            "total_duration_ms": round(self.total_duration_ms, 2),
            "stages": [
                {
                    "name": s.name,
                    "duration_ms": round(s.duration_ms, 2),
                    "skipped": s.skipped,
                    "detail": s.detail,
                }
                for s in self.stages
            ],
        }
