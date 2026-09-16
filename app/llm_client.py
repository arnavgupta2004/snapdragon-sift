"""Single chokepoint for every LLM call in this system.

Every LLM call in the agent (query understanding, complexity-classification fallback,
explanation generation) and in the synthetic data generator goes through this module.
That is deliberate: the router's latency/compute claims ("fast route makes zero LLM
calls") are only honest if there is exactly one place an LLM call could originate from,
and it is easy to audit.

Two backends, selected by LLM_BACKEND (default "local"):
  - "local" (default): Ollama, running entirely on-device. No API key, no network call
    leaving the machine, no per-request cost — this is the required default per the
    course brief (grading is against a fully offline, on-device system; a cloud LLM
    API is explicitly disallowed as the primary path).
  - "cloud": Google Gemini, kept as an optional comparison arm (set LLM_BACKEND=cloud
    and GEMINI_API_KEY) for eval/local_vs_cloud.py and for anyone who wants to see the
    tradeoff directly. Not the default, and never silently falls back to it — if a key
    is missing for whichever backend is selected, is_available is simply False and the
    rest of the system uses its rule-based fallback path.

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

from dotenv import load_dotenv

load_dotenv()

LLM_BACKEND = os.environ.get("LLM_BACKEND", "local").strip().lower()

OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "qwen2.5:1.5b")
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")

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
        elif backend == "cloud":
            self._impl = _GeminiBackend(model or GEMINI_MODEL)
        else:
            raise ValueError(f"unknown LLM_BACKEND {backend!r}, expected 'local' or 'cloud'")
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
