"""Thin wrapper around the local Ollama daemon: what's actually pulled on
this machine right now (for the Model Registry UI's local/available
distinction) and pulling a model on demand if it's missing.

Adapted from hi/hi/models.py's `_is_pulled`/`ensure_pulled`, generalized
from tier names to arbitrary model ids since SAGE routes by task type, not
by a fixed haiku/sonnet/opus tier set.
"""

import ollama

_pulled_cache: set[str] = set()


def list_local_models() -> set[str]:
    """Every model tag `ollama list` currently reports — includes both
    fully-local weights and cloud tags that have been referenced before
    (Ollama still lists those, even though inference for them leaves the
    machine)."""
    response = ollama.list()
    return {model.model for model in response.models}


def is_pulled(model_id: str) -> bool:
    if model_id in _pulled_cache:
        return True
    if model_id in list_local_models():
        _pulled_cache.add(model_id)
        return True
    return False


def ensure_pulled(model_id: str) -> None:
    """Pull `model_id` if it isn't already available, streaming progress —
    a genuinely new local model can take minutes, so this must not hang
    with no feedback."""
    if is_pulled(model_id):
        return
    print(f"Pulling model '{model_id}'... this may take a while.")
    for update in ollama.pull(model_id, stream=True):
        status = getattr(update, "status", None) or update.get("status", "")
        print(f"\r{status}", end="", flush=True)
    print()
    _pulled_cache.add(model_id)
