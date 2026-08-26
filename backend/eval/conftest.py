"""Adapted from insurance_claim_agent/eval/conftest.py — real Qdrant + real
Ollama, no mocks; skip cleanly with a clear reason if either isn't up
rather than a wall of connection-refused tracebacks.
"""

import json
import os

import pytest
from qdrant_client import QdrantClient

from app.knowledge.config import COLLECTION_NAME, QDRANT_URL
from app.knowledge.query import run_query
from eval.dataset import load_golden_set

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


@pytest.fixture(scope="session", autouse=True)
def _require_live_services():
    client = QdrantClient(url=QDRANT_URL)
    try:
        if not client.collection_exists(COLLECTION_NAME):
            pytest.skip(
                f"Qdrant collection '{COLLECTION_NAME}' does not exist — "
                "run `uv run python -m scripts.ingest_knowledge` first."
            )
    except Exception as exc:
        pytest.skip(f"Qdrant is not reachable at {QDRANT_URL}: {exc}")

    try:
        import ollama

        ollama.list()
    except Exception as exc:
        pytest.skip(f"Ollama is not reachable: {exc}")


@pytest.fixture(scope="session")
def golden_set():
    return load_golden_set()


@pytest.fixture(scope="session")
def pipeline_results(golden_set):
    """Runs the pipeline once per golden-set question and caches results —
    retrieval metrics only need `candidates`, but this also exercises
    generation/citation-verification/confidence-flagging for the same run
    a future test module could reuse without re-paying LLM latency."""
    results = {}
    for entry in golden_set:
        results[entry["id"]] = run_query(entry["question"], target_source=entry.get("target_source"))

    with open(os.path.join(RESULTS_DIR, "pipeline_results.json"), "w") as f:
        json.dump(
            {
                eid: {
                    "answer": r["answer"],
                    "error": r["error"],
                    "gated": r["gated"],
                    "top_score": r["top_score"],
                    "confidence": r["confidence"],
                    "context_candidates": [
                        {"meta": c["meta"], "cross_score": c["cross_score"]} for c in r["context_candidates"]
                    ],
                }
                for eid, r in results.items()
            },
            f,
            indent=1,
        )
    return results
