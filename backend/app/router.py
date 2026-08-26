"""Resolves a task_type ("reasoning" / "coding" / "vision" / "embedding")
to a live model client, entirely driven by config/models.yaml — this is the
file the Phase 1 acceptance check exercises: adding a model to models.yaml
and switching a task type's `active` candidate must never require editing
anything here.
"""

import ollama as ollama_pkg
from langchain_ollama import ChatOllama

from app.config import get_config
from app.load_monitor import select_candidate
from app.ollama_client import ensure_pulled
from app.model_warmup import warm_up

_model_cache: dict[tuple[str, str], ChatOllama] = {}


def _is_unsupported_thinking_error(exc: Exception) -> bool:
    return isinstance(exc, ollama_pkg.ResponseError) and "does not support thinking" in str(exc)


def _build_model(model_id: str, endpoint: str, context_window: int) -> ChatOllama:
    """`reasoning=True` is what makes Ollama's separate "thinking" stream
    show up at all (langchain_ollama otherwise drops it entirely — verified
    live: without this, a reasoning model's whole thinking phase, which can
    run minutes on a complex prompt, streams nothing, looking hung) — but
    Ollama 400s outright if the model doesn't support thinking at all, so
    this is detected once per model here (not hardcoded per candidate,
    since whether a model thinks isn't visible from its name) and cached,
    rather than assumed.

    `context_window` (models.yaml's own per-candidate field) is passed
    through as `num_ctx` — until this was wired up, that field was purely
    documentation, never actually applied, so Ollama fell back to each
    model's own built-in default context length regardless of what
    models.yaml said. Caught live on this project's actual dev hardware (a
    memory-constrained 18GB machine): qwen3:8b defaulted to a 40960-token
    context and 11GB resident just from that default, real evidence from
    this project's own repeated multi-minute hangs under memory pressure.
    Explicitly setting num_ctx to a size that matches what this project's
    tasks actually need (chat + tool results + a handful of retrieved SOP
    excerpts, not a 40K-token document) dropped that to 6.6GB with no
    observed behavior change — verified via `ollama ps` before/after."""
    kwargs = {"model": model_id, "base_url": endpoint, "temperature": 0, "keep_alive": "30m", "num_ctx": context_window}
    model = ChatOllama(reasoning=True, **kwargs)
    try:
        model.invoke("hi")
    except Exception as exc:
        if _is_unsupported_thinking_error(exc):
            model = ChatOllama(**kwargs)
        # any other exception here (e.g. a transient empty response) isn't
        # this function's concern — warm_up() below retries past those.
    return model


def get_chat_model(task_type: str, tier: str | None = None) -> ChatOllama:
    """Return a cached ChatOllama for `task_type`. Keyed on (task_type,
    model_id) so switching the active candidate (via set_active_model), or
    passing a different `tier` (Phase 11 — app/complexity.py picks "fast"
    vs "strong" per request), transparently resolves to the right cached
    client rather than serving a stale one — each distinct model this
    process has ever needed stays warm in `_model_cache`, so alternating
    tiers across requests only pays a cold-start cost once per model, not
    per request."""
    candidate = select_candidate(task_type, tier=tier)
    cache_key = (task_type, candidate.model_id)
    if cache_key not in _model_cache:
        ensure_pulled(candidate.model_id)
        endpoint = get_config().ollama_endpoint
        model = _build_model(candidate.model_id, endpoint, candidate.context_window)
        warm_up(model)
        _model_cache[cache_key] = model
    return _model_cache[cache_key]
