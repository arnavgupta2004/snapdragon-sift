# Sift — production UI

React + TypeScript (Vite). Replaces `ui/streamlit_app.py` as the graded deliverable per
the build order — the Streamlit app stays for internal iteration only.

## Run it

```bash
# from repo root: start the API first
uvicorn app.api:app --reload

# in this directory:
npm install
npm run dev
```

Vite proxies `/api/*` to `http://localhost:8000` (see `vite.config.ts`) so the app
talks to the FastAPI backend with no CORS configuration needed in dev.

## What's here

- `src/App.tsx` — layout, user switcher, query bar, SSE streaming state
- `src/components/TracePanel.tsx` — the live animated routing trace: fixed canonical
  stage order so the panel doesn't jump around between tiers, each stage transitions
  pending → done/skipped as SSE events arrive from `GET /api/query/stream`
- `src/components/ResultsList.tsx` — ranked results, score bars, thumbs up/down wired
  to `POST /api/feedback` (closes the feedback loop from the browser, not just eval
  scripts)
- `src/components/PersonalizationPanel.tsx` — preferred file types, recurring
  patterns, active context boost, from `GET /api/personalization/{user_id}`
- `src/components/BaselineChart.tsx` — the four-way baseline comparison table (§8b)
  rendered as Recharts bar charts, from `GET /api/eval/baseline-comparison`

## Build

```bash
npm run build
```

Type-checks (`tsc -b`) then builds to `dist/`.
