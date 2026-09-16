"""Shared ONNX Runtime + QNN execution-provider plumbing, used by both the LLM
backend (app/llm_client.py's _QnnLlmBackend, via onnxruntime-genai) and the CLIP
embedding backend (app/embedding_client.py's _ClipQnnBackend, via a raw
onnxruntime.InferenceSession).

QNNExecutionProvider is Qualcomm's ONNX Runtime execution provider for the Hexagon
NPU on Snapdragon silicon (Windows-on-Snapdragon, or Android/embedded Linux with the
QNN SDK installed). It does not exist on any other platform -- the `onnxruntime-qnn`
distribution that ships it is a different wheel than the plain `onnxruntime` CPU
package and isn't installable on x86/Apple Silicon (see requirements-qnn.txt) -- so
every call in this module is written to degrade to "unavailable" rather than raise,
the same is_available contract app/llm_client.py already uses.
"""

from __future__ import annotations

import os
from pathlib import Path

QNN_BACKEND_LIB = os.environ.get("QNN_BACKEND_LIB", "").strip()
QNN_HTP_PERFORMANCE_MODE = os.environ.get("QNN_HTP_PERFORMANCE_MODE", "burst").strip()


def qnn_provider_available() -> bool:
    """True only if onnxruntime is importable, was built with QNNExecutionProvider,
    and a Hexagon backend library path has been configured (QNN_BACKEND_LIB). Cheap
    and side-effect-free -- safe to call before deciding whether to even attempt
    loading a QNN model."""
    if not QNN_BACKEND_LIB:
        return False
    try:
        import onnxruntime as ort
    except ImportError:
        return False
    return "QNNExecutionProvider" in ort.get_available_providers()


def qnn_provider_options() -> dict:
    return {
        "backend_path": QNN_BACKEND_LIB,
        "htp_performance_mode": QNN_HTP_PERFORMANCE_MODE,
        "htp_graph_finalization_optimization_mode": "3",
    }


def create_qnn_session(model_path: str | Path):
    """Builds an ORT InferenceSession for `model_path` targeting the QNN execution
    provider, with CPUExecutionProvider listed as a per-node fallback (standard ORT
    behavior if a given op isn't supported on the NPU). Callers are expected to have
    already checked qnn_provider_available() -- this raises if session creation
    itself fails rather than silently downgrading, so a misconfigured
    QNN_BACKEND_LIB/model pair fails loudly instead of pretending to run on NPU."""
    import onnxruntime as ort

    return ort.InferenceSession(
        str(model_path),
        providers=[("QNNExecutionProvider", qnn_provider_options()), "CPUExecutionProvider"],
    )
