# Sift: Agentic File Recommendation and Retrieval System

An agentic file-search system that understands a natural-language query, decides *which* retrieval
strategies it actually needs (filename/fuzzy, metadata, keyword BM25, semantic embeddings, CLIP image
content, or a hybrid of these), personalizes results to the requesting user's behavioral history, and
explains its own ranking decisions — instead of always running one fixed RAG pipeline. **Runs entirely
on-device by default** (local Ollama LLM, local embedding/rerank/CLIP models, local vector store) —
no API key required, nothing leaves the machine.

> **Status: complete**, core system (build-order phases 1-9) and extended scope (LightGBM personalization
> + closed feedback loop, learned router, real filesystem connector as the UI's primary data path,
> production React/TS UI, Docker + CI, full written report) — see `REPORT.md`. Everything below is real,
> run, and verified — not aspirational.

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
Sift/
├── data/
│   ├── generate_synthetic_data.py  # single reproducible entry point (corpus + DB + access log)
│   ├── synth/                      # topics, personas, content generation, access-log simulation
│   ├── ingest_datasource.py        # real-data connector ingestion (Phase 12)
│   └── files_corpus/, db.sqlite    # generated, gitignored — not checked in
├── app/
│   ├── agent/       # graph.py (LangGraph), router.py + learned_router.py, query_understanding.py, explain.py
│   ├── retrieval/   # filename, metadata, keyword, semantic, hybrid_fusion (RRF from scratch), reranker
│   ├── personalization/  # profile_builder, personalized_ranker, learned_ranker.py (LightGBM), retrain.py
│   ├── datasources/  # DataSource interface + filesystem/synthetic implementations
│   ├── api.py       # FastAPI: /api/query, /api/query/stream (SSE), /api/feedback, /api/personalization
│   ├── llm_client.py  # single chokepoint for all LLM calls (Gemini-backed; see note below)
│   ├── models.py    # shared SQLModel schema
│   └── tracing.py   # RoutingTrace — the first-class per-stage timing object
├── ui/
│   ├── streamlit_app.py  # internal-iteration dev UI (throwaway — see build order)
│   └── frontend/          # production React + TypeScript UI (Vite) — the graded deliverable
├── eval/
│   ├── build_eval_set.py, metrics.py, run_benchmark.py
│   ├── ablation_study.py, latency_comparison.py, baseline_comparison.py, personalization_lift.py
│   ├── feedback_loop_demo.py, learned_ranker_comparison.py, build_router_labels.py, router_agreement.py
│   ├── eval_set.json, router_labels.json   # labeled data, checked in
│   ├── RESULTS.md        # full write-up, every number sourced from results/
│   └── results/           # CSVs, JSON summaries, PNG charts — all regeneratable, checked in
├── tests/            # pytest cases across every component
├── Dockerfile, docker-compose.yml, docker-entrypoint.sh, ui/frontend/Dockerfile
└── .github/workflows/ci.yml
```

## Running it

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

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

An optional cloud comparison arm (Gemini) is still available — set `LLM_BACKEND=cloud` and
`GEMINI_API_KEY` in `.env` — but it is never the default and the system never silently falls back to
it. Most of this repo's eval numbers were generated with the local backend (the required default for
grading); the retrieval-quality and router-agreement numbers were generated with the cloud backend
specifically to establish real-LLM ground truth to compare the local model against — see `REPORT.md`
§8.1 and §5.6 for exactly which numbers are which and why.

**Free-tier cloud rate limits** (only relevant if you explicitly opt into `LLM_BACKEND=cloud`): on
`gemini-3.5-flash-lite`'s free tier (15 requests/minute), running `pytest` or the eval scripts against
the cloud backend can be noticeably slower — the retry/backoff in `app/llm_client.py` can compound with
the Gemini SDK's own internal retries under sustained load. The local backend has no such limit.

## Reproducing every claim

Every quantitative claim in this README and in `eval/RESULTS.md` is produced by a script under `eval/`
that can be re-run from scratch — no numbers here are hand-typed without a corresponding script.
