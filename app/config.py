"""Central home for tunable thresholds and weights that were previously scattered as
module-level constants across the codebase. Nothing here changes behavior — every
value matches what was already the default; this just makes the knobs findable and
editable in one place instead of hunting through app/agent/router.py,
app/agent/graph.py, app/personalization/*.py, etc.

Values that are genuinely per-instance configuration (e.g. PersonalizationWeights,
which the eval harness sweeps to A/B different blends) stay as dataclasses in their
own modules rather than flat constants here — this file is for the thresholds that
are effectively global policy, not per-call parameters.
"""

from __future__ import annotations

# --- Router (app/agent/router.py) ------------------------------------------------
# Below this word count, a query with a clean file-type/date filter and no leftover
# topical content routes fast with zero embedding/LLM calls.
FAST_MAX_WORDS_FOR_FILTER_ONLY = 9
# At or above this word count (or with a vague-language marker), a query routes deep
# without needing a classifier call at all.
DEEP_MIN_WORDS = 14
# The learned router's classification is trusted only at or above this confidence;
# below it, the real LLM (or a rule-based default) is used instead.
LOW_CONFIDENCE_THRESHOLD = 0.6

# --- Agent graph (app/agent/graph.py) ---------------------------------------------
# How many candidates each retrieval stage pulls before fusion/reranking.
CANDIDATE_POOL_SIZE = 20
# How many results are returned to the caller after personalization.
FINAL_RESULT_COUNT = 10

# --- Hybrid fusion (app/retrieval/hybrid_fusion.py) -------------------------------
# The RRF damping constant — the standard default from the original RRF paper.
RRF_K = 60

# --- Personalization (app/personalization/profile_builder.py) --------------------
# Number of KMeans clusters discovered over semantic embeddings for topic affinity.
# Matches the synthetic corpus's 10 hand-designed topics for eval comparability, but
# this is a discovered count, not read from ground truth.
N_CLUSTERS = 10
# Exponential decay half-life (days) for the recency signal.
RECENCY_HALF_LIFE_DAYS = 14.0

# --- Learned ranker training data (app/personalization/training_data.py) ---------
# How many negative examples (same-topic / different-topic) accompany each positive
# access-log event when building synthetic training groups.
N_NEGATIVES_SAME_TOPIC = 3
N_NEGATIVES_OTHER_TOPIC = 3
MAX_EVENTS_PER_USER = 60
# Feedback rows are explicit, query-specific signal; weighted higher than the coarse
# access-log bootstrap so the feedback loop's effect is visible after a realistic
# number of rounds (see eval/feedback_loop_demo.py).
ACCESS_LOG_ROW_WEIGHT = 1.0
FEEDBACK_ROW_WEIGHT = 3.0

# --- Filesystem data source (app/datasources/filesystem_source.py) ---------------
# Per-file text extraction cap, so one huge PDF/spreadsheet doesn't blow up the
# embedding/keyword index with an outlier-sized document.
MAX_EXTRACTED_CONTENT_CHARS = 20_000

# --- Image content search (app/retrieval/image_search.py) ------------------------
# CPU/GPU CLIP variant via sentence-transformers — embeds actual pixels, not
# filename/caption text, into a shared text-image space for content-based matching.
# This is app/embedding_client.py's "cpu" backend (the default, works everywhere).
IMAGE_CLIP_MODEL_NAME = "clip-ViT-B-32"

# --- QNN / Qualcomm AI Hub model backend (app/qnn_runtime.py, app/llm_client.py,
# app/embedding_client.py) --------------------------------------------------------
# These name the specific Qualcomm AI Hub model-zoo listings this project's NPU
# backend is built against, and the on-disk layout it expects inside
# QNN_LLM_MODEL_DIR / QNN_CLIP_MODEL_DIR (see .env.example). Getting the actual
# compiled artifacts requires a Qualcomm AI Hub account and its export/compile
# pipeline, which happens outside this repo — these constants just document what
# this repo's code expects to find once that's done.
QNN_LLM_HF_SOURCE = "Llama-v3.2-1B-Instruct"  # AI Hub model-zoo listing, quantized + compiled for Hexagon
QNN_CLIP_HF_SOURCE = "openai/clip-vit-base-patch32"  # same weights family as IMAGE_CLIP_MODEL_NAME, so both
# backends embed into a compatible space
QNN_CLIP_IMAGE_ENCODER_FILENAME = "clip_image_encoder.onnx"
QNN_CLIP_TEXT_ENCODER_FILENAME = "clip_text_encoder.onnx"
