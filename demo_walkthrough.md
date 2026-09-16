# Demo walkthrough

A scripted run-through for presenting Sift live, covering all three routing tiers, the
image-content search capability, and the real-data connector. ~10-12 minutes.

## Setup (before the audience arrives)

```bash
# on-device LLM (the default backend — no API key, nothing leaves the machine)
brew install ollama
brew services start ollama       # or: ollama serve
ollama pull qwen2.5:1.5b

# from repo root
pip install -r requirements.txt
python data/generate_synthetic_data.py
python eval/build_eval_set.py
python -m app.retrieval.semantic_search --build   # warm the text embedding index
python -m app.retrieval.image_search --build      # warm the CLIP image index

# two terminals:
uvicorn app.api:app --reload            # terminal 1
cd ui/frontend && npm install && npm run dev   # terminal 2
```

Open `http://localhost:5173`. With Ollama running, deep-route queries below show real
LLM-generated explanations and query enrichment — entirely on-device. Without Ollama
running, they fall back to rule-based explanations — say so explicitly rather than
letting it look broken. (An optional cloud comparison arm exists — `LLM_BACKEND=cloud`
plus `GEMINI_API_KEY` — but it is not the default and isn't needed for this walkthrough.)

Run each of the three text queries below **once** before presenting, so the sentence-
transformer, CLIP, and cross-encoder models are warm (cold start is ~15-25s on first
call; warm is under a second for fast/standard, a couple seconds for deep).

## 1. Fast route — "this should feel instant"

Pick any real filename from your corpus and type:

```
open <some_filename>.docx
```

Point out on the trace panel: **filename search** and **metadata search** run,
**everything else is greyed out** — keyword, semantic, image, reranker,
personalization, explanation all skipped. Total latency should be single-digit
milliseconds. Say the number out loud; it's the whole point of Objective 3.

## 2. Standard route — "moderately specific, no reranker"

```
find my recent notes about transformers
```

Point out: metadata + keyword + semantic + image all ran, RRF fusion combined them,
personalization re-ranked — but the reranker and LLM explanation are still greyed
out. This is the tier that's "pretty sure but doesn't need the expensive pass."

## 3. Deep route — "this is where the agent actually reasons"

```
find that thing I was working on with my advisor about audio deepfakes a few weeks ago
```

Every stage lights up, including the cross-encoder reranker and the on-device LLM
explanation. Read one of the explanations out loud — it should be a genuine sentence
about *why* that file matches, not a template, and it came from a 1.5B model running
locally, not an API call. Switch the user dropdown (top of the query bar) and re-run
the same query: the ranking changes because personalization is now blending in a
different user's access history — point at the score breakdown to show the
base-vs-personalization split.

## 4. Image content search — "searching a photo by what's in it"

This is the single most memorable moment if the audience includes anyone skeptical
that file search can go beyond filenames: describe what's *visually in* an image, not
its name or any text near it.

```
there's a photo somewhere with a large blue triangle in it, not sure what it's called
```

(Substitute the actual shape/color combination present in your generated corpus —
check `data/image_subjects.json` after running `generate_synthetic_data.py`, or just
try a few colors/shapes; the corpus has ~24 generated images with genuinely distinct
drawn subjects, not just captions.) Point out the **image search** stage in the trace
— this is CLIP (`clip-ViT-B-32`), running locally, embedding actual pixels into the
same space as the text query. No filename or caption text overlaps with the query at
all; a match can only come from the model actually looking at the image.

## 5. Real data, live — index a real folder from the UI

Expand **📁 Index a real folder** at the top of the page, click one of the suggested
paths (`~/Downloads`, `~/Documents`, `~/Desktop`) or type any local path, and click
**Index this folder**. This crawls a directory on *this machine* — works identically
on Windows, macOS, or Linux, since it's a plain filesystem walk, not a browser upload
— and indexes real, messy files (PDFs, images, code, whatever's actually there)
alongside the synthetic corpus. Point out the result banner: real file-type counts,
real total.

Then run a natural-language query about something you know is in that folder — this
is not a pre-baked demo file, it's the same pipeline finding a file that didn't exist
in the system a few seconds ago. If a real image happens to match a content query,
even better: it proves image-content search works on genuine photos, not just the
synthetic corpus's drawn shapes.

(CLI equivalent, if you'd rather not click through the UI:
`python data/ingest_datasource.py --root ~/Downloads --max-files 30`)

## 6. The receipts

Scroll to the **baseline comparison** panel and narrate the four bars: naive
keyword-only, naive semantic-only ("what a lazy RAG wrapper looks like"),
always-full-pipeline, and the full system — this project beats both naive baselines
on quality while running a fraction of the always-full-pipeline's latency. If asked
"how do you know the personalization/routing/on-device-LLM tradeoff actually hold
up," this panel and `eval/RESULTS.md` / `REPORT.md` are the answer — every number on
screen came from a script in `eval/`, not a slide, including the local-vs-cloud model
comparison (`eval/local_vs_cloud.py`) and the image-content-search benchmark
(`eval/image_content_search.py`).

## If something goes wrong

- **First deep query takes 20+ seconds and looks stuck**: that's model cold-start,
  not a hang. Mention it, or run one warm-up query before presenting (see Setup).
- **No LLM explanations, just short template strings**: Ollama isn't running —
  `brew services start ollama` (or `ollama serve`) and confirm `ollama list` shows
  `qwen2.5:1.5b`. This is a documented fallback, not a bug.
- **Folder indexing takes a while on a large folder**: it's doing real work (crawling
  + text extraction + re-embedding the whole corpus). Cap `max_files` lower (the UI
  defaults to 300) for a snappier live demo.
- **Baseline comparison panel says "not available yet"**: `eval/results/` wasn't
  generated — run `python eval/baseline_comparison.py` once beforehand.
