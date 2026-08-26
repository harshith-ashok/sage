"""Loads and validates config/models.yaml — the single source of truth for
task_type -> model routing (see that file's header comment for the schema
and the config-only add/switch contract).

Reads are served from an in-memory cache; `set_active_model` writes the
change back to disk and refreshes the cache, so a model switch (e.g. from
the Model Registry UI) persists across restarts without touching any
orchestrator code.
"""

import threading
from pathlib import Path

import yaml
from pydantic import BaseModel, RootModel

CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "models.yaml"

_lock = threading.Lock()
_cache: "ModelsConfig | None" = None


class ModelCandidate(BaseModel):
    model_id: str
    location: str  # "local" | "cloud"
    context_window: int
    engine: str = "ollama"  # "ollama" (default, served via app/router.py) | "transformers" (loaded directly — Phase 8's IndicTrans2)
    tier: str | None = None  # "fast" | "strong" | None (Phase 11: complexity-based routing, app/complexity.py) — None means this candidate isn't part of that scheme, same as before it existed


class TaskTypeConfig(BaseModel):
    active: str
    candidates: dict[str, ModelCandidate]

    def active_candidate(self) -> ModelCandidate:
        return self.candidates[self.active]


class ModelsConfig(RootModel):
    root: dict

    @property
    def ollama_endpoint(self) -> str:
        return self.root["ollama_endpoint"]

    @property
    def task_types(self) -> dict[str, TaskTypeConfig]:
        return {name: TaskTypeConfig(**cfg) for name, cfg in self.root["task_types"].items()}


def _load_from_disk() -> ModelsConfig:
    with open(CONFIG_PATH) as f:
        raw = yaml.safe_load(f)
    return ModelsConfig(root=raw)


def get_config(force_reload: bool = False) -> ModelsConfig:
    global _cache
    with _lock:
        if _cache is None or force_reload:
            _cache = _load_from_disk()
        return _cache


def list_task_types() -> list[str]:
    return list(get_config().task_types.keys())


def get_task_type_config(task_type: str) -> TaskTypeConfig:
    task_types = get_config().task_types
    if task_type not in task_types:
        raise ValueError(f"Unknown task type '{task_type}'. Expected one of: {sorted(task_types)}")
    return task_types[task_type]


def get_active_candidate(task_type: str) -> ModelCandidate:
    return get_task_type_config(task_type).active_candidate()


def set_active_model(task_type: str, candidate_key: str) -> ModelCandidate:
    """Point `task_type` at a different already-declared candidate and
    persist it to models.yaml. Raises ValueError if either the task type or
    the candidate key doesn't exist — callers (the API layer) turn that into
    a 404, not a silent no-op."""
    with _lock:
        with open(CONFIG_PATH) as f:
            raw = yaml.safe_load(f)

        if task_type not in raw["task_types"]:
            raise ValueError(f"Unknown task type '{task_type}'. Expected one of: {sorted(raw['task_types'])}")
        candidates = raw["task_types"][task_type]["candidates"]
        if candidate_key not in candidates:
            raise ValueError(f"Unknown candidate '{candidate_key}' for task type '{task_type}'. Expected one of: {sorted(candidates)}")

        raw["task_types"][task_type]["active"] = candidate_key
        with open(CONFIG_PATH, "w") as f:
            yaml.safe_dump(raw, f, sort_keys=False)

        global _cache
        _cache = ModelsConfig(root=raw)
        return _cache.task_types[task_type].active_candidate()
