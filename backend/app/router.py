"""Resolves a task_type ("reasoning" / "coding" / "vision" / "embedding")
to a live model client, entirely driven by config/models.yaml — this is the
file the Phase 1 acceptance check exercises: adding a model to models.yaml
and switching a task type's `active` candidate must never require editing
anything here.
"""

import ollama as ollama_pkg
from langchain_ollama import ChatOllama

from app.config import get_active_candidate, get_config
from app.ollama_client import ensure_pulled
from app.model_warmup import warm_up

_model_cache: dict[tuple[str, str], ChatOllama] = {}


def _is_unsupported_thinking_error(exc: Exception) -> bool:
    return isinstance(exc, ollama_pkg.ResponseError) and "does not support thinking" in str(exc)


def _build_model(model_id: str, endpoint: str) -> ChatOllama:
    """`reasoning=True` is what makes Ollama's separate "thinking" stream
    show up at all (langchain_ollama otherwise drops it entirely — verified
    live: without this, a reasoning model's whole thinking phase, which can
    run minutes on a complex prompt, streams nothing, looking hung) — but
    Ollama 400s outright if the model doesn't support thinking at all, so
    this is detected once per model here (not hardcoded per candidate,
    since whether a model thinks isn't visible from its name) and cached,
    rather than assumed."""
    kwargs = {"model": model_id, "base_url": endpoint, "temperature": 0, "keep_alive": "30m"}
    model = ChatOllama(reasoning=True, **kwargs)
    try:
        model.invoke("hi")
    except Exception as exc:
        if _is_unsupported_thinking_error(exc):
            model = ChatOllama(**kwargs)
        # any other exception here (e.g. a transient empty response) isn't
        # this function's concern — warm_up() below retries past those.
    return model


def get_chat_model(task_type: str) -> ChatOllama:
    """Return a cached ChatOllama for the model currently active for
    `task_type`. Keyed on (task_type, model_id) so switching the active
    candidate (via set_active_model) transparently resolves to a fresh
    client on the next call instead of serving a stale cached model."""
    candidate = get_active_candidate(task_type)
    cache_key = (task_type, candidate.model_id)
    if cache_key not in _model_cache:
        ensure_pulled(candidate.model_id)
        endpoint = get_config().ollama_endpoint
        model = _build_model(candidate.model_id, endpoint)
        warm_up(model)
        _model_cache[cache_key] = model
    return _model_cache[cache_key]
