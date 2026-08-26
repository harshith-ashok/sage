"""Phase 11: load-aware routing. Polls real, local, no-network signals of
how loaded this machine currently is, and lets app/router.py fall back to a
lighter candidate under pressure instead of piling onto an already-strained
system.

**What "GPU memory/utilization" means on this hardware, honestly**: this
project's actual dev/demo machine is a single Mac, not a multi-GPU server —
there is no `nvidia-smi`, and Ollama's own accelerator (Metal) shares
unified memory with the rest of the OS rather than exposing separate "GPU
memory" the way a discrete card would. The real, observable constraint here
is **system memory pressure**: `ollama.ps()` (below) already confirms that a
locally-loaded model's `size_vram` on this hardware equals its full
resident size, i.e. the model's memory footprint *is* system memory
pressure on this box. So this monitor reads `psutil.virtual_memory()`
(local system call, no network) as the load signal, and `ollama.ps()` (also
purely local — it's a call to the same already-running Ollama daemon every
other model call in this app already talks to) for what's actually loaded
right now. On real GPU-server hardware this would poll `nvidia-smi`/NVML
instead; the threshold-based classification and fallback logic below don't
care which signal feeds them.

**Fallback never crosses `location` automatically.** This project's one
non-negotiable rule (top of CLAUDE.md) is that nothing makes a network call
after the initial model download, ever, unless a human explicitly chooses
that — so an automatic switch from a local candidate to a cloud one under
load would itself be exactly the kind of unattended network call that rule
exists to prevent, and at the worst possible moment (silently, under
stress, exactly when a user is least likely to be watching closely). So
this module's fallback only ever picks among candidates that share the
*same* `location` as the one currently active — real degradation, not a
quiet zero-egress violation. When no safe same-location fallback exists
(true today for every task type except `embedding`, which is the only one
with two local candidates declared), it says so on the Model Registry view
instead of silently reaching for cloud.
"""

import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone

import ollama
import psutil

from app.config import ModelCandidate, get_task_type_config
from app.ollama_client import is_pulled

# Tunable, not magic: "elevated" is a heads-up (some other local model
# could plausibly cause pressure soon); "high" is where this module starts
# actually preferring an alternate same-location candidate over the
# configured active one.
_ELEVATED_AVAILABLE_PERCENT = 30.0
_HIGH_AVAILABLE_PERCENT = 15.0

_MAX_FALLBACK_LOG = 20
_log_lock = threading.Lock()
_fallback_log: list["FallbackEvent"] = []


@dataclass
class FallbackEvent:
    task_type: str
    from_model_id: str
    to_model_id: str | None  # None: load was high but no safe same-location fallback existed
    reason: str
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


@dataclass
class LoadedModel:
    model_id: str
    size_bytes: int
    size_vram_bytes: int


@dataclass
class LoadSnapshot:
    level: str  # "normal" | "elevated" | "high"
    available_percent: float
    loaded_models: list[LoadedModel]


def get_system_load() -> LoadSnapshot:
    vm = psutil.virtual_memory()
    available_percent = 100.0 - vm.percent

    try:
        ps_response = ollama.ps()
        loaded_models = [
            LoadedModel(model_id=m.model, size_bytes=m.size or 0, size_vram_bytes=m.size_vram or 0)
            for m in ps_response.models
        ]
    except Exception:
        # Ollama daemon unreachable is a real, separate failure mode this
        # module shouldn't mask — but it also shouldn't crash a load check
        # over it; report an empty loaded-models list rather than raising.
        loaded_models = []

    if available_percent < _HIGH_AVAILABLE_PERCENT:
        level = "high"
    elif available_percent < _ELEVATED_AVAILABLE_PERCENT:
        level = "elevated"
    else:
        level = "normal"

    return LoadSnapshot(level=level, available_percent=available_percent, loaded_models=loaded_models)


def record_fallback(task_type: str, from_model_id: str, to_model_id: str | None, reason: str) -> None:
    with _log_lock:
        _fallback_log.append(FallbackEvent(task_type=task_type, from_model_id=from_model_id, to_model_id=to_model_id, reason=reason))
        del _fallback_log[:-_MAX_FALLBACK_LOG]


def get_fallback_log() -> list[FallbackEvent]:
    with _log_lock:
        return list(_fallback_log)


def select_candidate(task_type: str, tier: str | None = None) -> ModelCandidate:
    """The one shared entry point every task-type caller (app/router.py's
    chat models, app/knowledge/embeddings.py's dense embeddings) should use
    instead of `get_active_candidate` directly, so load-aware fallback
    applies uniformly rather than only where someone remembered to wire it
    in.

    Two independent mechanisms compose here, in order:

    1. **Tier selection** (Phase 11, app/complexity.py): if `tier` is given
       and some candidate for this task type declares that tier, it becomes
       the starting point instead of the configured `active` candidate — a
       cheap prompt-complexity heuristic picking "fast" vs "strong" per
       request. If no candidate declares that tier (most task types don't —
       see complexity.py's docstring for which ones do), this is a no-op
       and `active` wins, same as before tiering existed.
    2. **Load-aware fallback** (also Phase 11, the memory-pressure signal
       this module's own docstring describes): applied on top of whichever
       candidate step 1 produced — if it's local and the system is under
       real memory pressure, prefer another already-pulled local candidate
       for this task type instead. See this module's docstring for why this
       never crosses from local to cloud automatically.

    `embedding` is exempt from both — caught live, not hypothetically: the
    very first real test of load-aware fallback swapped bge-m3 (1024-dim)
    for nomic-embed-text (768-dim) mid-session and Qdrant immediately
    rejected the query with a vector dimension mismatch. Chat models can
    freely substitute for each other (quality varies, correctness doesn't);
    embedding models can't — the whole knowledge base is indexed in one
    specific model's vector space, so switching which model embeds a
    *query* without re-embedding the entire corpus doesn't degrade
    retrieval, it breaks it outright (or worse, if dimensions happened to
    coincide, would silently return nearest-neighbors in the wrong space
    instead of erroring loudly). The same reasoning applies to tiering: an
    embedding task type simply never declares a `tier` on its candidates.
    """
    config = get_task_type_config(task_type)
    active = config.active_candidate()
    if task_type == "embedding":
        return active

    starting = active
    if tier is not None:
        for candidate in config.candidates.values():
            # Phase 12 fix, caught live by the zero-egress integration test:
            # this used to match on tier alone, so a prompt classified
            # "strong" could silently resolve to a *cloud* candidate even
            # when a human had explicitly set `active` to a local one for a
            # zero-egress run — an unattended network call this project's
            # one non-negotiable rule never allows. Tiering may only move
            # *toward* local (lighter/cheaper, always safe) or stay within
            # the active candidate's own location; it can never move away
            # from local into cloud on its own. Going cloud -> local from a
            # cloud `active` is unaffected (that direction was never the
            # risk) and still works exactly as before.
            if candidate.location == "cloud" and active.location == "local":
                continue
            if candidate.tier == tier:
                starting = candidate
                break

    if starting.location != "local":
        return starting

    load = get_system_load()
    if load.level != "high":
        return starting

    for candidate in config.candidates.values():
        if candidate.model_id == starting.model_id:
            continue
        if candidate.location == "local" and is_pulled(candidate.model_id):
            record_fallback(
                task_type,
                starting.model_id,
                candidate.model_id,
                reason=f"high memory load ({load.available_percent:.0f}% free)",
            )
            return candidate

    record_fallback(
        task_type,
        starting.model_id,
        None,
        reason=f"high memory load ({load.available_percent:.0f}% free) — no other local candidate pulled",
    )
    return starting
