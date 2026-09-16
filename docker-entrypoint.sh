#!/bin/sh
set -e

DB_PATH="data/db.sqlite"

if [ ! -f "$DB_PATH" ]; then
    echo "[docker-entrypoint] no data/db.sqlite found — generating the synthetic corpus (first boot only)..."
    python data/generate_synthetic_data.py
    python eval/build_eval_set.py
else
    echo "[docker-entrypoint] data/db.sqlite already present, skipping generation."
fi

echo "[docker-entrypoint] warming the semantic search index..."
python -m app.retrieval.semantic_search --build

echo "[docker-entrypoint] warming the image content (CLIP) index..."
python -m app.retrieval.image_search --build

exec "$@"
