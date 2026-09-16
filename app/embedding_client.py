"""Single chokepoint for every image/text embedding call this system makes into the
shared CLIP space — app/retrieval/image_search.py's image-content index build and
query encoding both go through here, and nowhere else. Mirrors app/llm_client.py's
backend-selection pattern so the embedding model is swappable via config exactly the
same way the LLM backend is.

Two backends, selected by EMBEDDING_BACKEND (default "cpu"):
  - "cpu" (default): sentence-transformers' clip-ViT-B-32 wrapper, running on
    CPU/GPU via PyTorch. Works everywhere, no setup — the pre-existing path, and the
    fallback on non-Snapdragon hardware during development.
  - "qnn": a CLIP-family vision-language model from Qualcomm AI Hub (OpenAI CLIP
    ViT-B/32's image and text towers, exported separately to ONNX and compiled for
    the Hexagon NPU), run through ONNX Runtime's QNN execution provider. Only usable
    on Snapdragon silicon with the QNN SDK installed (see QNN_CLIP_MODEL_DIR in
    .env.example).

Like llm_client.py, this never silently swaps backends: if EMBEDDING_BACKEND=qnn but
the NPU model/provider isn't actually available, is_available is False and calling
embed_image/embed_text raises rather than quietly falling back to CPU — pick the
backend you want via config, the same contract as LLM_BACKEND.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from PIL import Image

from app.config import (
    IMAGE_CLIP_MODEL_NAME,
    QNN_CLIP_HF_SOURCE,
    QNN_CLIP_IMAGE_ENCODER_FILENAME,
    QNN_CLIP_TEXT_ENCODER_FILENAME,
)
from app.qnn_runtime import create_qnn_session, qnn_provider_available

load_dotenv()

EMBEDDING_BACKEND = os.environ.get("EMBEDDING_BACKEND", "cpu").strip().lower()
QNN_CLIP_MODEL_DIR = os.environ.get("QNN_CLIP_MODEL_DIR", "").strip()


def _l2_normalize(vec: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 0 else vec


class _ClipCpuBackend:
    """CPU/GPU CLIP via sentence-transformers — the pre-existing image_search.py
    path, unchanged behavior, kept as the always-available default. Model loads
    lazily on first use (same reason the old module-level lru_cache did: importing
    torch/sentence-transformers at module import time would slow down every process
    that merely imports this module, even ones that never embed anything)."""

    def __init__(self, model_name: str = IMAGE_CLIP_MODEL_NAME) -> None:
        self.model_name = model_name
        self._model = None
        self.is_available = True

    def _get_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name)
        return self._model

    def embed_image(self, image: Image.Image) -> np.ndarray:
        return self._get_model().encode([image], normalize_embeddings=True, show_progress_bar=False)[0]

    def embed_text(self, text: str) -> np.ndarray:
        return self._get_model().encode([text], normalize_embeddings=True, show_progress_bar=False)[0]


class _ClipQnnBackend:
    """CLIP image/text embeddings via a Qualcomm AI Hub-compiled model: the image and
    text towers exported separately to ONNX and run on the Hexagon NPU through ONNX
    Runtime's QNN execution provider. Pre/post-processing (image resize/crop/
    normalize, BPE tokenization) stays on CPU via the original model's Hugging Face
    processor/tokenizer — AI Hub compiles the transformer graph itself, not text/
    image preprocessing, so this mirrors how AI Hub's own CLIP sample app is
    structured.

    Input/output tensor names are read from the ONNX session's own metadata rather
    than hardcoded, since the exact names an AI Hub export uses can vary by export
    version — this hasn't been run against a real compiled model on this machine (no
    Snapdragon hardware in this dev environment), so treat that as unverified until
    checked against the actual exported graph."""

    def __init__(self, model_dir: str = QNN_CLIP_MODEL_DIR) -> None:
        self.model_dir = Path(model_dir) if model_dir else None
        self.model_name = QNN_CLIP_HF_SOURCE
        self._image_session = None
        self._text_session = None
        self._processor = None
        self._tokenizer = None
        self.is_available = self._load()

    def _load(self) -> bool:
        if self.model_dir is None or not qnn_provider_available():
            return False
        image_path = self.model_dir / QNN_CLIP_IMAGE_ENCODER_FILENAME
        text_path = self.model_dir / QNN_CLIP_TEXT_ENCODER_FILENAME
        if not image_path.exists() or not text_path.exists():
            return False
        try:
            self._image_session = create_qnn_session(image_path)
            self._text_session = create_qnn_session(text_path)
            from transformers import CLIPImageProcessor, CLIPTokenizerFast

            self._processor = CLIPImageProcessor.from_pretrained(QNN_CLIP_HF_SOURCE)
            self._tokenizer = CLIPTokenizerFast.from_pretrained(QNN_CLIP_HF_SOURCE)
        except Exception:
            return False
        return True

    def _require_loaded(self) -> None:
        if not self.is_available:
            raise RuntimeError(
                "QNN CLIP backend called without a loaded NPU session. Check "
                "is_available before calling, or point QNN_CLIP_MODEL_DIR at "
                "compiled image/text encoder .onnx files."
            )

    def embed_image(self, image: Image.Image) -> np.ndarray:
        self._require_loaded()
        pixel_values = self._processor(images=image, return_tensors="np")["pixel_values"].astype(np.float32)
        input_name = self._image_session.get_inputs()[0].name
        output_name = self._image_session.get_outputs()[0].name
        (embeds,) = self._image_session.run([output_name], {input_name: pixel_values})
        return _l2_normalize(np.asarray(embeds[0], dtype=np.float32))

    def embed_text(self, text: str) -> np.ndarray:
        self._require_loaded()
        tokens = self._tokenizer([text], padding="max_length", truncation=True, max_length=77, return_tensors="np")
        input_feed = {
            inp.name: tokens[inp.name].astype(np.int64)
            for inp in self._text_session.get_inputs()
            if inp.name in tokens
        }
        output_name = self._text_session.get_outputs()[0].name
        (embeds,) = self._text_session.run([output_name], input_feed)
        return _l2_normalize(np.asarray(embeds[0], dtype=np.float32))


class EmbeddingClient:
    """Provider-agnostic facade. Picks a backend from EMBEDDING_BACKEND ("cpu" by
    default, "qnn" for the Hexagon-NPU path) and exposes the same embed_image()/
    embed_text()/is_available surface regardless of which one is active."""

    def __init__(self, backend: str = EMBEDDING_BACKEND) -> None:
        self.backend = backend
        if backend == "cpu":
            self._impl = _ClipCpuBackend()
        elif backend == "qnn":
            self._impl = _ClipQnnBackend()
        else:
            raise ValueError(f"unknown EMBEDDING_BACKEND {backend!r}, expected 'cpu' or 'qnn'")
        self.model_name = self._impl.model_name

    @property
    def is_available(self) -> bool:
        return self._impl.is_available

    def embed_image(self, image: Image.Image) -> np.ndarray:
        return self._impl.embed_image(image)

    def embed_text(self, text: str) -> np.ndarray:
        return self._impl.embed_text(text)


def get_embedding_client(backend: str | None = None) -> EmbeddingClient:
    """Resolves EMBEDDING_BACKEND from the environment fresh on every call (not
    frozen at import time), matching app.llm_client.get_client()'s contract — so a
    benchmark script comparing backends within one process can flip
    os.environ["EMBEDDING_BACKEND"] between calls."""
    resolved = (backend or os.environ.get("EMBEDDING_BACKEND", "cpu")).strip().lower()
    return _get_cached_embedding_client(resolved)


@lru_cache(maxsize=2)
def _get_cached_embedding_client(backend: str) -> EmbeddingClient:
    return EmbeddingClient(backend=backend)
