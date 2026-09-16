"""Single chokepoint for every LLM call in this system.

Every LLM call in the agent (query understanding, complexity-classification fallback,
explanation generation) and in the synthetic data generator goes through this module.
That is deliberate: the router's latency/compute claims ("fast route makes zero LLM
calls") are only honest if there is exactly one place an LLM call could originate from,
and it is easy to audit.

Three backends, selected by LLM_BACKEND (default "local"):
  - "local" (default): Ollama, running entirely on-device via CPU/GPU. No API key, no
    network call leaving the machine, no per-request cost — the safe default on any
    machine, and the fallback path on non-Snapdragon hardware during development.
  - "qnn": a quantized instruction LLM from Qualcomm AI Hub, compiled for the Hexagon
    NPU and run via ONNX Runtime GenAI + the QNN execution provider. Only usable on
    Snapdragon silicon with the QNN SDK installed (see QNN_LLM_MODEL_DIR in
    .env.example) — this is the NPU-accelerated path this project targets for the
    Snapdragon AI Lab Build & Present Challenge.
  - "cloud": Google Gemini, kept as an optional comparison arm (set LLM_BACKEND=cloud
    and GEMINI_API_KEY) for eval/local_vs_cloud.py and for anyone who wants to see the
    tradeoff directly. Not the default, and never silently falls back to it — if a
    backend isn't usable (missing key, missing QNN hardware/model, Ollama not
    running), is_available is simply False and the rest of the system uses its
    rule-based fallback path.

Model choice for the local backend (qwen2.5:1.5b) was picked empirically, not by
default assumption — see eval/local_vs_cloud.py and REPORT.md for the 3-model
benchmark (qwen2.5:1.5b vs. llama3.2:1b vs. phi3:mini) this default is based on.
"""

from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

LLM_BACKEND = os.environ.get("LLM_BACKEND", "local").strip().lower()

OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "qwen2.5:1.5b")
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
QNN_LLM_MODEL_DIR = os.environ.get("QNN_LLM_MODEL_DIR", "").strip()

MAX_RATE_LIMIT_RETRIES = 3
_RETRY_DELAY_RE = re.compile(r"retryDelay['\"]?\s*:\s*['\"]?(\d+(?:\.\d+)?)s")


@dataclass
class LLMCallResult:
    text: str
    input_tokens: int
    output_tokens: int
    model: str
    backend: str


class _OllamaBackend:
    """Local, on-device inference via a running `ollama serve` (default port 11434).
    Availability is checked with a fast, short-timeout ping so the rest of the system
    can degrade to its rule-based fallback immediately if Ollama isn't running, rather
    than hanging on a dead connection."""

    def __init__(self, model: str, host: str = OLLAMA_HOST) -> None:
        self.model = model
        self.host = host
        self.is_available = self._ping()

    def _ping(self) -> bool:
        try:
            req = urllib.request.Request(f"{self.host}/api/version")
            with urllib.request.urlopen(req, timeout=1.0) as resp:
                return resp.status == 200
        except Exception:
            return False

    def complete(self, prompt: str, system: str | None, max_tokens: int, temperature: float) -> LLMCallResult:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        payload = json.dumps(
            {
                "model": self.model,
                "messages": messages,
                "stream": False,
                "options": {"num_predict": max_tokens, "temperature": temperature},
            }
        ).encode()
        req = urllib.request.Request(
            f"{self.host}/api/chat", data=payload, headers={"Content-Type": "application/json"}
        )
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                data = json.load(resp)
        except urllib.error.URLError as e:
            raise RuntimeError(f"Ollama request failed ({self.host}, model={self.model}): {e}") from e

        return LLMCallResult(
            text=(data.get("message") or {}).get("content", ""),
            input_tokens=data.get("prompt_eval_count", 0),
            output_tokens=data.get("eval_count", 0),
            model=self.model,
            backend="local",
        )


class _QnnLlmBackend:
    """On-device NPU inference via a quantized instruction LLM from Qualcomm AI Hub,
    run through ONNX Runtime GenAI (`onnxruntime_genai`) with the QNN execution
    provider targeting the Hexagon NPU.

    AI Hub's export/compile step for genai-class models produces a model directory
    (weights + tokenizer + a genai_config.json) where the execution provider and the
    Hexagon backend library path are already named in genai_config.json — so unlike
    app/embedding_client.py's QNN backend, this class doesn't select the provider
    itself; it just points onnxruntime_genai at QNN_LLM_MODEL_DIR and the config file
    picks the EP up. Availability requires: onnxruntime_genai importable, and that
    directory actually present with a genai_config.json in it — anything else and
    is_available is False, same degrade-to-rule-based-fallback contract as every
    other backend in this module.

    The onnxruntime_genai call surface below (Model/Tokenizer/GeneratorParams/
    Generator) matches the library's documented API as of this writing, but hasn't
    been exercised against a real QNN-compiled model on this machine (no Snapdragon
    hardware in this dev environment) — verify against whatever onnxruntime_genai
    version ships with the exported model before relying on it."""

    def __init__(self, model_dir: str = QNN_LLM_MODEL_DIR) -> None:
        self.model_dir = model_dir
        self.model = Path(model_dir).name if model_dir else "qnn-llm"
        self._og = None
        self._model = None
        self._tokenizer = None
        self.is_available = self._load()

    def _load(self) -> bool:
        if not self.model_dir or not (Path(self.model_dir) / "genai_config.json").exists():
            return False
        try:
            import onnxruntime_genai as og
        except ImportError:
            return False
        try:
            self._og = og
            self._model = og.Model(self.model_dir)
            self._tokenizer = og.Tokenizer(self._model)
        except Exception:
            return False
        return True

    def complete(self, prompt: str, system: str | None, max_tokens: int, temperature: float) -> LLMCallResult:
        if not self.is_available:
            raise RuntimeError(
                "QNN LLM backend called without a loaded NPU model. Check is_available "
                "before calling, or point QNN_LLM_MODEL_DIR at a compiled model directory."
            )
        full_prompt = _llama_instruct_prompt(system, prompt)
        input_tokens = self._tokenizer.encode(full_prompt)

        params = self._og.GeneratorParams(self._model)
        params.set_search_options(
            max_length=len(input_tokens) + max_tokens,
            temperature=max(temperature, 1e-4),
            do_sample=temperature > 0,
        )
        generator = self._og.Generator(self._model, params)
        generator.append_tokens(input_tokens)

        output_tokens: list[int] = []
        while not generator.is_done():
            generator.generate_next_token()
            output_tokens.append(generator.get_next_tokens()[0])

        return LLMCallResult(
            text=self._tokenizer.decode(output_tokens).strip(),
            input_tokens=len(input_tokens),
            output_tokens=len(output_tokens),
            model=self.model,
            backend="qnn",
        )


def _llama_instruct_prompt(system: str | None, user: str) -> str:
    """Llama-3.2-Instruct's chat template, built by hand rather than via a
    tokenizer.apply_chat_template call so this has no dependency on exactly which
    tokenizer config files AI Hub ships inside the exported model directory."""
    system = system or "You are a helpful assistant."
    return (
        "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
        f"{system}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"
        f"{user}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    )


class _GeminiBackend:
    """Cloud comparison arm. No-ops gracefully when no API key is set."""

    def __init__(self, model: str) -> None:
        self.model = model
        self._client = None
        api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        self.is_available = bool(api_key)
        if self.is_available:
            from google import genai

            self._client = genai.Client(api_key=api_key)

    def complete(self, prompt: str, system: str | None, max_tokens: int, temperature: float) -> LLMCallResult:
        if not self.is_available or self._client is None:
            raise RuntimeError(
                "Gemini backend called without GEMINI_API_KEY set. "
                "Check client.is_available before calling, or set the env var."
            )
        from google.genai import errors, types

        config = types.GenerateContentConfig(
            system_instruction=system or None,
            max_output_tokens=max_tokens,
            temperature=temperature,
        )

        attempt = 0
        while True:
            try:
                response = self._client.models.generate_content(
                    model=self.model, contents=prompt, config=config
                )
                break
            except errors.ClientError as e:
                # Free-tier quota (429 RESOURCE_EXHAUSTED) is routine, not exceptional,
                # on the model this project runs against — retry with the delay the API
                # itself suggests rather than failing the whole call on a transient limit.
                if getattr(e, "code", None) != 429 or attempt >= MAX_RATE_LIMIT_RETRIES:
                    raise
                delay = _extract_retry_delay(str(e))
                attempt += 1
                time.sleep(delay)

        text = response.text or ""
        usage = response.usage_metadata
        return LLMCallResult(
            text=text,
            input_tokens=getattr(usage, "prompt_token_count", 0) or 0,
            output_tokens=getattr(usage, "candidates_token_count", 0) or 0,
            model=self.model,
            backend="cloud",
        )


def _extract_retry_delay(error_text: str, default: float = 15.0) -> float:
    match = _RETRY_DELAY_RE.search(error_text)
    return float(match.group(1)) + 1.0 if match else default


class LLMClient:
    """Provider-agnostic facade. Picks a backend from LLM_BACKEND ("local" by default,
    "cloud" for the optional Gemini comparison arm) and exposes the same
    `complete()`/`is_available` surface regardless of which one is active."""

    def __init__(self, backend: str = LLM_BACKEND, model: str | None = None) -> None:
        self.backend = backend
        if backend == "local":
            self._impl = _OllamaBackend(model or OLLAMA_MODEL)
        elif backend == "qnn":
            self._impl = _QnnLlmBackend(model or QNN_LLM_MODEL_DIR)
        elif backend == "cloud":
            self._impl = _GeminiBackend(model or GEMINI_MODEL)
        else:
            raise ValueError(f"unknown LLM_BACKEND {backend!r}, expected 'local', 'qnn', or 'cloud'")
        self.model = self._impl.model

    @property
    def is_available(self) -> bool:
        return self._impl.is_available

    def complete(
        self,
        prompt: str,
        system: str | None = None,
        max_tokens: int = 1024,
        temperature: float = 1.0,
    ) -> LLMCallResult:
        return self._impl.complete(prompt, system, max_tokens, temperature)


def get_client(backend: str | None = None) -> LLMClient:
    """Resolves LLM_BACKEND from the environment fresh on every call (not frozen at
    import time), so eval scripts that need to compare backends within one process
    (eval/local_vs_cloud.py) can flip os.environ["LLM_BACKEND"] between calls and have
    every call site — router.py, explain.py, query_understanding.py — pick it up
    without needing to be told explicitly."""
    resolved = (backend or os.environ.get("LLM_BACKEND", "local")).strip().lower()
    return _get_cached_client(resolved)


@lru_cache(maxsize=4)
def _get_cached_client(backend: str) -> LLMClient:
    return LLMClient(backend=backend)
