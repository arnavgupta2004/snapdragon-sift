#!/usr/bin/env python3
"""NPU (QNN) vs. CPU/GPU fallback: inference latency for both the instruction LLM
(app/llm_client.py) and the CLIP embedding model (app/embedding_client.py), on a
small set of representative queries drawn from eval/eval_set.json.

    python eval/benchmark_qnn_vs_cpu.py

Writes eval/results/qnn_vs_cpu_benchmark_summary.json and
eval/results/qnn_vs_cpu_benchmark.md (a deck-ready table).

This does NOT fabricate numbers for a backend it can't reach: if QNN_LLM_MODEL_DIR /
QNN_CLIP_MODEL_DIR aren't set, the QNN execution provider isn't installed, or no
Snapdragon NPU is present, that backend's rows are written as "unavailable" with a
reason, not a made-up latency. On non-Snapdragon dev hardware (e.g. this benchmark
run on a Mac or x86 box) the QNN rows will always show unavailable — that's expected;
re-run this script on the target Snapdragon device to fill them in.

Power draw: this harness does not attempt to measure it. There is no cross-platform
Python API for per-process power draw, and estimating it from CPU/NPU utilization
without real instrumentation would itself be a fabricated number — the JSON/Markdown
output says so explicitly rather than omitting the field silently.
"""

from __future__ import annotations

import json
import os
import statistics
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

RESULTS_DIR = REPO_ROOT / "eval" / "results"
EVAL_SET_PATH = REPO_ROOT / "eval" / "eval_set.json"

N_TIMED_RUNS = 5
N_WARMUP_RUNS = 1

POWER_DRAW_NOTE = (
    "not measurable in this environment (no per-process power instrumentation "
    "available) -- not fabricated"
)


def _representative_queries(n: int = 5) -> list[str]:
    if EVAL_SET_PATH.exists():
        eval_set = json.loads(EVAL_SET_PATH.read_text())
        queries = [q["query"] for q in eval_set][:n]
        if queries:
            return queries
    return [
        "quarterly finance report from last spring",
        "the slide deck I presented at the client kickoff",
        "python script that simulates the access log",
        "notes about the research draft on retrieval quality",
        "spreadsheet with admin HR numbers",
    ]


def _representative_image_query_texts(n: int = 5) -> list[str]:
    return [
        "a large red circle",
        "a small blue square",
        "a photo with a triangle shape",
        "a picture that is mostly green",
        "an image with a yellow shape in the corner",
    ][:n]


def _timeit(fn, n_runs: int, n_warmup: int) -> dict:
    for _ in range(n_warmup):
        fn()
    latencies_ms = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        fn()
        latencies_ms.append((time.perf_counter() - t0) * 1000)
    return {
        "n_runs": n_runs,
        "mean_ms": statistics.mean(latencies_ms),
        "p95_ms": latencies_ms[-1] if n_runs == 1 else statistics.quantiles(latencies_ms, n=100)[min(94, n_runs - 2)],
        "min_ms": min(latencies_ms),
        "max_ms": max(latencies_ms),
    }


def _bench_llm_backend(backend: str, queries: list[str]) -> dict:
    from app.llm_client import LLMClient

    try:
        client = LLMClient(backend=backend)
    except Exception as e:
        return {"backend": backend, "available": False, "reason": f"backend init failed: {e}"}

    if not client.is_available:
        reason = {
            "local": "Ollama not running / model not pulled",
            "qnn": "QNN_LLM_MODEL_DIR not set, model missing, or QNN execution provider not installed "
            "(expected on non-Snapdragon hardware)",
            "cloud": "GEMINI_API_KEY not set",
        }.get(backend, "backend reported unavailable")
        return {"backend": backend, "available": False, "reason": reason}

    prompt_template = "In one short sentence, explain why this file might match the query: {q!r}"

    def run_one():
        client.complete(prompt_template.format(q=queries[0]), max_tokens=60, temperature=0.0)

    timing = _timeit(run_one, N_TIMED_RUNS, N_WARMUP_RUNS)
    return {"backend": backend, "available": True, "model": client.model, **timing}


def _bench_embedding_backend(backend: str, image_query_texts: list[str]) -> dict:
    from app.embedding_client import EmbeddingClient

    try:
        client = EmbeddingClient(backend=backend)
    except Exception as e:
        return {"backend": backend, "available": False, "reason": f"backend init failed: {e}"}

    if not client.is_available:
        reason = {
            "cpu": "sentence-transformers model failed to load",
            "qnn": "QNN_CLIP_MODEL_DIR not set, encoder .onnx files missing, or QNN execution provider "
            "not installed (expected on non-Snapdragon hardware)",
        }.get(backend, "backend reported unavailable")
        return {"backend": backend, "available": False, "reason": reason}

    def run_one():
        client.embed_text(image_query_texts[0])

    timing = _timeit(run_one, N_TIMED_RUNS, N_WARMUP_RUNS)
    return {"backend": backend, "available": True, "model": client.model_name, **timing}


def run_benchmark() -> dict:
    queries = _representative_queries()
    image_queries = _representative_image_query_texts()

    print("[benchmark_qnn_vs_cpu] LLM: local (Ollama)...")
    llm_local = _bench_llm_backend("local", queries)
    print("[benchmark_qnn_vs_cpu] LLM: qnn (Hexagon NPU)...")
    llm_qnn = _bench_llm_backend("qnn", queries)

    print("[benchmark_qnn_vs_cpu] embedding: cpu (sentence-transformers CLIP)...")
    embed_cpu = _bench_embedding_backend("cpu", image_queries)
    print("[benchmark_qnn_vs_cpu] embedding: qnn (Hexagon NPU CLIP)...")
    embed_qnn = _bench_embedding_backend("qnn", image_queries)

    return {
        "llm": {"local": llm_local, "qnn": llm_qnn},
        "embedding": {"cpu": embed_cpu, "qnn": embed_qnn},
        "power_draw": POWER_DRAW_NOTE,
        "host_platform": f"{sys.platform} ({os.uname().machine if hasattr(os, 'uname') else 'unknown'})",
    }


def _fmt_row(label: str, result: dict) -> str:
    if not result.get("available"):
        return f"| {label} | unavailable | — | {result.get('reason', 'n/a')} |"
    return f"| {label} | {result['model']} | {result['mean_ms']:.1f} ms (p95 {result['p95_ms']:.1f} ms) | {result['n_runs']} runs |"


def to_markdown(summary: dict) -> str:
    lines = [
        "# QNN (Hexagon NPU) vs. CPU/GPU fallback — inference latency",
        "",
        f"Host: `{summary['host_platform']}`",
        "",
        "## LLM (instruction model)",
        "",
        "| Backend | Model | Mean latency | Notes |",
        "|---|---|---|---|",
        _fmt_row("local (CPU/GPU, Ollama)", summary["llm"]["local"]),
        _fmt_row("qnn (Hexagon NPU)", summary["llm"]["qnn"]),
        "",
        "## Embedding model (CLIP text encode)",
        "",
        "| Backend | Model | Mean latency | Notes |",
        "|---|---|---|---|",
        _fmt_row("cpu (sentence-transformers)", summary["embedding"]["cpu"]),
        _fmt_row("qnn (Hexagon NPU)", summary["embedding"]["qnn"]),
        "",
        f"**Power draw**: {summary['power_draw']}",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    summary = run_benchmark()

    json_path = RESULTS_DIR / "qnn_vs_cpu_benchmark_summary.json"
    json_path.write_text(json.dumps(summary, indent=2))

    md_path = RESULTS_DIR / "qnn_vs_cpu_benchmark.md"
    md_path.write_text(to_markdown(summary))

    print(json.dumps(summary, indent=2))
    print(f"\nwrote {json_path}\nwrote {md_path}")


if __name__ == "__main__":
    main()
