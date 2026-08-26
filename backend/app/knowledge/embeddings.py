"""Dense embeddings, routed through the Phase 1 model registry's "embedding"
task type (app/config.py) rather than a hardcoded sentence-transformers
model name — switching the active embedding candidate in models.yaml (or via
the Model Registry UI) changes what ingest/query embed with, no code touched
here. Re-run ingestion after switching: Qdrant's collection is recreated
fresh each ingest (pipeline.py), so vector dimensions never go stale, but a
mid-session switch without re-ingesting would.

Candidate resolution goes through app.load_monitor.select_candidate rather
than get_active_candidate directly (Phase 11) — `embedding` is the only
task type with two declared *local* candidates today, so it's the one place
load-aware fallback actually has somewhere to fall back to on this hardware.
"""

import ollama

from app.config import get_config
from app.load_monitor import select_candidate
from app.ollama_client import ensure_pulled


def _endpoint_client() -> ollama.Client:
    return ollama.Client(host=get_config().ollama_endpoint)


def embed_text(text: str) -> list[float]:
    return embed_texts([text])[0]


def embed_texts(texts: list[str]) -> list[list[float]]:
    candidate = select_candidate("embedding")
    ensure_pulled(candidate.model_id)
    response = _endpoint_client().embed(model=candidate.model_id, input=texts)
    return [list(vec) for vec in response.embeddings]


def embedding_dimension() -> int:
    return len(embed_text("dimension probe"))
