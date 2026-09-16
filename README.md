# Snapdragon Sift: NPU-Accelerated Agentic File Search

**Snapdragon AI Lab Build & Present Challenge 2026 submission.** Built on top of an existing on-device
file-search capstone project (an agentic file-recommendation and retrieval system — see below), now
extended with Qualcomm AI Hub models running on the Hexagon NPU via ONNX Runtime's QNN execution
provider.

An agentic file-search system that understands a natural-language query, decides *which* retrieval
strategies it actually needs (filename/fuzzy, metadata, keyword BM25, semantic embeddings, CLIP image
content, or a hybrid of these), personalizes results to the requesting user's behavioral history, and
explains its own ranking decisions — instead of always running one fixed RAG pipeline. **Runs entirely
on-device by default** (local LLM, local embedding/rerank/CLIP models, local vector store) — no API key
required, nothing leaves the machine.

## NPU backend (this fork's contribution)

The base project's LLM and image-embedding paths ran on CPU/GPU only (Ollama for the LLM, a
sentence-transformers CLIP model for image content search). This fork adds a swappable NPU-accelerated
backend, selected entirely by config, alongside the original CPU/GPU path — nothing was ripped out:

| Component | CPU/GPU path (default, unchanged) | NPU path (new) |
|---|---|---|
| Instruction LLM | Ollama, `qwen2.5:1.5b` | **Llama-v3.2-1B-Instruct**, quantized + compiled for the Hexagon NPU via **Qualcomm AI Hub**, run through **ONNX Runtime GenAI + the QNN execution provider** |
| Image/text embedding (CLIP) | `clip-ViT-B-32` via `sentence-transformers` | **OpenAI CLIP ViT-B/32**, exported to ONNX and compiled for the Hexagon NPU via **Qualcomm AI Hub**, run through **ONNX Runtime + the QNN execution provider** |

Backend selection is one env var each — `LLM_BACKEND=local|qnn|cloud` (`app/llm_client.py`) and
`EMBEDDING_BACKEND=cpu|qnn` (`app/embedding_client.py`) — never hardcoded. See **Running on Snapdragon
hardware** below for setup, and `eval/benchmark_qnn_vs_cpu.py` for the latency comparison harness.

Image content search ("find that photo by describing what's in it") was already a first-class query type
in the agent's plan → retrieve → rank → explain loop before this fork (`app/retrieval/image_search.py`,
wired into `app/agent/graph.py`'s standard/deep routes); this fork's job there was making its embedding
model swappable to the NPU path, and wiring index-time embedding into the filesystem-scan/ingest step
(`data/ingest_datasource.py`, `POST /api/ingest`) rather than only lazily at first query.

> **Status: complete**, core system (build-order phases 1-9) and extended scope (LightGBM personalization
> + closed feedback loop, learned router, real filesystem connector as the UI's primary data path,
> production React/TS UI, Docker + CI, full written report) — see `REPORT.md`. Everything below is real,
> run, and verified — not aspirational, with the exception of the NPU (QNN) backend itself: it was
> developed and code-reviewed off Snapdragon hardware, so it has not been execution-verified on a real
> Hexagon NPU — see **Running on Snapdragon hardware** for what to check first on real hardware.

## Headline numbers

Full methodology and every chart: **[`eval/RESULTS.md`](eval/RESULTS.md)**. All reproducible via the
commands in that file.

- **Routing saves 5.5x mean latency overall** (23.8x on simple/exact queries) for a stated **−0.033
  NDCG@10** quality cost — the router is a real, quantified tradeoff, not a free lunch.
- **The full system beats naive keyword-only and semantic-only baselines** on NDCG@10 (0.557 vs. 0.470
  / 0.493) and MRR (0.644 vs. 0.514 / 0.556), while running 5.7x faster than the always-full-pipeline
  agent it's built on.
- **Hybrid RRF fusion nearly doubles NDCG@10** over naively concatenating retriever results (0.277 →
  0.481) — genuine rank fusion, not just "more retrievers," is what earns the gain.
- **Personalization's hand-tuned baseline shows a small negative lift** (−0.016 NDCG@10 against users'
  own access history) — reported honestly, not adjusted to look better, and used as the motivation for
  the learned (LightGBM) ranker in the extended-scope phase.
- **Image content search actually works**: a CLIP-based retriever finds images by what's visually *in*
  them — zero filename/keyword overlap with the query — at NDCG@10 0.83 in isolation. Found (and
  partially fixed) a real bug doing this: the cross-encoder reranker was discarding CLIP's correct
  matches because it only ever sees caption text, never pixels.
- **Runs fully on-device by default** (Ollama, `qwen2.5:1.5b`, picked via a 3-model benchmark) — no API
  key, nothing leaves the machine. The local model trails a cloud model on both routing accuracy and
  retrieval-quality-relevant reasoning, reported honestly rather than hidden — see `REPORT.md` §5.6.

## Why this exists (the three objectives)

| Objective | What it means | Where it lives |
|---|---|---|
| **1. Intelligent Discovery** | Understand intent, choose retrieval strategy/strategies (filename/metadata/keyword/semantic text/**image content via CLIP**), retrieve, rank, explain each result in natural language. | `app/agent/query_understanding.py`, `app/retrieval/` |
| **2. Personalization** | Build a per-user behavioral profile (frequency, recency, topic affinity, temporal/session patterns) from an access log and use it to re-rank results. | `app/personalization/` |
| **3. Agentic Routing** | Classify query complexity and route through the cheapest sufficient pipeline (fast / standard / deep), with measured latency savings vs. an always-full-pipeline baseline. | `app/agent/router.py`, `eval/latency_comparison.py` |

Every response the system returns includes a `routing_trace`: which tier was chosen, which components
ran (and which were skipped), per-component latency, and a one-line rationale for the routing decision.
This isn't a debug log — it's a first-class API field (`app/tracing.py`) and it's what the live SSE
stream (`GET /api/query/stream`) and the dev UI's trace panel render in real time.

## Architecture

```
                        Query (NL)
                            |
                 Query Understanding (rule-based,
                 always; LLM enrichment on deep only)
                            |
                    Complexity Router
              (rule-based first, LLM fallback
               only for genuinely ambiguous queries)
                            |
        +-------------------+-------------------+
        |                   |                   |
     FAST                STANDARD              DEEP
  filename +       metadata + keyword    metadata + keyword
  metadata only    + semantic + image    + semantic + image
  (RRF fusion)        (RRF fusion)          (RRF fusion)
        |                   |                   |
        |                   |         Cross-encoder rerank (text
        |                   |        candidates only — image-CLIP
        |                   |         matches keep their score)
        |                   |                   |
        +-------------------+-------------------+
                            |
              Personalization re-rank (skipped for
              exact-filename queries — see below)
                            |
                 [LLM explanation — deep only]
                            |
                  Results + routing_trace
```

Explicit LangGraph nodes and edges (`app/agent/graph.py`) — inspectable, not a hidden agent loop.
Two things worth knowing if you read the code:

- **Personalization is skipped when the query names an exact filename.** Typing a literal filename is a
  deterministic ask; letting personalization reorder it based on unrelated user history was a real bug
  we found via the eval harness (see `eval/RESULTS.md` §1) and fixed with a regression test.
- **Every tier pays for rule-based entity extraction; only the deep tier can pay for LLM query
  enrichment; only borderline-complexity queries pay for an LLM routing-classification call.** This is
  what makes the fast route's "zero LLM calls" claim in the eval results literally true, not just
  approximately true.

## Repository layout

```
snapdragon-sift/
├── data/
│   ├── generate_synthetic_data.py  # single reproducible entry point (corpus + DB + access log)
│   ├── synth/                      # topics, personas, content generation, access-log simulation
│   ├── ingest_datasource.py        # real-data connector ingestion; also rebuilds the CLIP image index
│   └── files_corpus/, db.sqlite    # generated, gitignored — not checked in
├── app/
│   ├── agent/       # graph.py (LangGraph), router.py + learned_router.py, query_understanding.py, explain.py
│   ├── retrieval/   # filename, metadata, keyword, semantic, image_search (CLIP), hybrid_fusion, reranker
│   ├── personalization/  # profile_builder, personalized_ranker, learned_ranker.py (LightGBM), retrain.py
│   ├── datasources/  # DataSource interface + filesystem/synthetic implementations
│   ├── api.py       # FastAPI: /api/query, /api/query/stream (SSE), /api/feedback, /api/personalization
│   ├── llm_client.py        # single chokepoint for all LLM calls — backend picked via LLM_BACKEND
│   ├── embedding_client.py  # single chokepoint for all CLIP embedding calls — via EMBEDDING_BACKEND
│   ├── qnn_runtime.py        # shared ONNX Runtime + QNN execution-provider session plumbing
│   ├── models.py    # shared SQLModel schema
│   └── tracing.py   # RoutingTrace — the first-class per-stage timing object
├── ui/
│   ├── streamlit_app.py  # internal-iteration dev UI (throwaway — see build order)
│   └── frontend/          # production React + TypeScript UI (Vite) — the graded deliverable
├── eval/
│   ├── build_eval_set.py, metrics.py, run_benchmark.py
│   ├── ablation_study.py, latency_comparison.py, baseline_comparison.py, personalization_lift.py
│   ├── feedback_loop_demo.py, learned_ranker_comparison.py, build_router_labels.py, router_agreement.py
│   ├── benchmark_qnn_vs_cpu.py  # NPU vs. CPU/GPU latency for the LLM and embedding model
│   ├── eval_set.json, router_labels.json   # labeled data, checked in
│   ├── RESULTS.md        # full write-up, every number sourced from results/
│   └── results/           # CSVs, JSON summaries, PNG charts — all regeneratable, checked in
├── tests/            # pytest cases across every component
├── requirements.txt      # everything needed for the CPU/GPU path, installable anywhere
├── requirements-qnn.txt  # Snapdragon-only additions (onnxruntime-qnn, onnxruntime-genai) for the NPU path
├── Dockerfile, docker-compose.yml, docker-entrypoint.sh, ui/frontend/Dockerfile
└── .github/workflows/ci.yml
```

## Running it

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# on Snapdragon hardware, also: pip install -r requirements-qnn.txt

python data/generate_synthetic_data.py   # builds the corpus + DB + access log from scratch
python eval/build_eval_set.py            # builds the labeled eval set against that corpus

pytest                                    # full test suite

streamlit run ui/streamlit_app.py         # internal dev UI
# or:
uvicorn app.api:app --reload              # API (POST /api/query, GET /api/query/stream)
cd ui/frontend && npm install && npm run dev   # production React/TS UI (proxies to the API above)
```

### Docker

```bash
docker-compose up
```

Brings up the API (`localhost:8000`) and the production UI (`localhost:5173`) together.
First boot generates the synthetic corpus and builds the embedding index inside the `api`
container (a few minutes on a cold Hugging Face model cache; the healthcheck's `start_period`
accounts for this) — no manual setup. See `Dockerfile`, `docker-entrypoint.sh`,
`ui/frontend/Dockerfile`, and `docker-compose.yml`.

### CI

`.github/workflows/ci.yml` runs on every push/PR: regenerates the corpus, runs the full test
suite, runs the entire eval harness, diffs the freshly generated `eval/results/` against what's
committed (reported in the job summary), and separately type-checks + builds the React UI. No
`GEMINI_API_KEY` secret is configured for CI, so it exercises the rule-based fallback paths —
consistent with what the system is designed to degrade to.

### LLM backend — runs fully on-device by default

**Sift runs entirely offline by default.** Every LLM call — query enrichment, routing-classification
fallback, explanation generation, synthetic-content generation — goes through `app/llm_client.py`,
which defaults to **Ollama running locally** (`LLM_BACKEND=local`, model `qwen2.5:1.5b`). No API key,
no network call ever leaves the machine, no per-request cost. Install and pull the model once:

```bash
brew install ollama          # or your platform's equivalent
brew services start ollama   # or: ollama serve
ollama pull qwen2.5:1.5b
```

`qwen2.5:1.5b` was picked empirically, not by default assumption — it benchmarked best of 3 candidates
(qwen2.5:1.5b, llama3.2:1b, phi3:mini) on this project's actual routing/explanation tasks. See
`REPORT.md` §5.6 and `eval/local_vs_cloud.py` for the full 3-model comparison and the on-device-vs-cloud
tradeoff table.

**The system runs completely end-to-end without Ollama running at all**, too — every LLM-gated step
has a rule-based fallback, same as before.

A third backend, `LLM_BACKEND=qnn`, runs the NPU-accelerated path described below — see **Running on
Snapdragon hardware**.

An optional cloud comparison arm (Gemini) is still available — set `LLM_BACKEND=cloud` and
`GEMINI_API_KEY` in `.env` — but it is never the default and the system never silently falls back to
it. Most of this repo's eval numbers were generated with the local backend (the required default for
grading); the retrieval-quality and router-agreement numbers were generated with the cloud backend
specifically to establish real-LLM ground truth to compare the local model against — see `REPORT.md`
§8.1 and §5.6 for exactly which numbers are which and why.

## Running on Snapdragon hardware

This section is for testing the NPU-accelerated path on a Snapdragon-powered device (e.g. an HP OmniBook
or EliteBook on Snapdragon X Elite/Plus). Everything in **Running it** above still applies — the NPU
path is additive, selected by config, and the CPU/GPU path keeps working the same way it does on any
other machine.

1. **Install the QNN SDK** for your platform (Windows-on-Snapdragon or Android/embedded Linux) from the
   Qualcomm Package Manager / Qualcomm AI Hub, and note the path to the Hexagon HTP backend library
   (`QnnHtp.dll` on Windows-on-Snapdragon, `libQnnHtp.so` on Android/Linux).
2. **Install the NPU-only Python dependencies**, in addition to the base requirements:
   ```bash
   pip install -r requirements.txt -r requirements-qnn.txt
   ```
3. **Get the compiled models from Qualcomm AI Hub** (requires a free AI Hub account):
   - **LLM**: `Llama-v3.2-1B-Instruct` from the AI Hub model zoo, exported/quantized/compiled for your
     target Hexagon chipset via AI Hub's export pipeline (produces a directory with `genai_config.json` +
     weights + tokenizer, consumed by `onnxruntime-genai`).
   - **CLIP embedding model**: `OpenAI-Clip` (ViT-B/32) from the AI Hub model zoo, with the image and text
     towers exported/compiled separately to ONNX (consumed directly via `onnxruntime` + the QNN execution
     provider).
4. **Point the app at them** in `.env`:
   ```bash
   LLM_BACKEND=qnn
   QNN_LLM_MODEL_DIR=/path/to/llama-v3-2-1b-instruct-qnn
   EMBEDDING_BACKEND=qnn
   QNN_CLIP_MODEL_DIR=/path/to/clip-qnn
   QNN_BACKEND_LIB=/path/to/QnnHtp.dll   # or libQnnHtp.so
   ```
5. **Sanity-check the provider is actually visible to ONNX Runtime**:
   ```bash
   python -c "import onnxruntime as ort; print('QNNExecutionProvider' in ort.get_available_providers())"
   ```
   If this prints `False`, `app/qnn_runtime.py`'s `qnn_provider_available()` will too, and both QNN
   backends will report `is_available=False` and fall back to being unusable (not to CPU automatically —
   switch `LLM_BACKEND`/`EMBEDDING_BACKEND` back to `local`/`cpu` in `.env` if you need the app usable
   while debugging the QNN setup).
6. **Run the benchmark harness** to compare NPU vs. CPU/GPU latency directly:
   ```bash
   python eval/benchmark_qnn_vs_cpu.py
   ```
   Writes `eval/results/qnn_vs_cpu_benchmark_summary.json` and `.md`. If a QNN backend isn't reachable,
   its row is written as `unavailable` with the reason — the script never fabricates a latency number for
   a backend it couldn't actually call.

> **Note on verification**: `app/qnn_runtime.py`, `_QnnLlmBackend` (`app/llm_client.py`), and
> `_ClipQnnBackend` (`app/embedding_client.py`) were written against the documented ONNX Runtime QNN
> execution provider and `onnxruntime-genai` APIs, but developed on non-Snapdragon hardware — there is no
> Hexagon NPU in the dev environment this fork was built in. Treat the exact provider-option names and
> the `onnxruntime_genai` call sequence as the first things to check if the NPU path doesn't come up
> cleanly on real hardware; the CPU/GPU fallback path (`LLM_BACKEND=local`, `EMBEDDING_BACKEND=cpu`) is
> the one that's been actually run end-to-end.

**Free-tier cloud rate limits** (only relevant if you explicitly opt into `LLM_BACKEND=cloud`): on
`gemini-3.5-flash-lite`'s free tier (15 requests/minute), running `pytest` or the eval scripts against
the cloud backend can be noticeably slower — the retry/backoff in `app/llm_client.py` can compound with
the Gemini SDK's own internal retries under sustained load. The local backend has no such limit.

## Reproducing every claim

Every quantitative claim in this README and in `eval/RESULTS.md` is produced by a script under `eval/`
that can be re-run from scratch — no numbers here are hand-typed without a corresponding script.
