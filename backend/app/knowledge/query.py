"""Adapted from insurance_claim_agent/query.py: retrieve -> rerank -> dedup
-> optionally gate -> generate -> verify citations -> flag confidence.
Drops the optional Langfuse tracing hook (not needed here, avoids an unused
dependency) — everything else is the same pipeline shape.
"""

import logging
import time

from app.knowledge.config import COLLECTION_NAME, CONFIDENCE_THRESHOLD, TOP_K_CONTEXT
from app.knowledge.dependencies import PipelineDependencies
from app.knowledge.embeddings import embed_text
from app.knowledge.exceptions import LLMInferenceError, QdrantConnectionError
from app.knowledge.guardrail import flag_confidence, verify_citations
from app.knowledge.llm import ABSTENTION_MESSAGE, generate
from app.knowledge.search import build_filter, deduplicate, hybrid_search, rerank

logger = logging.getLogger(__name__)

_default_deps = PipelineDependencies.build()


def format_context(candidates: list[dict]) -> str:
    """Renders retrieved candidates into the citable context block both
    run_query() and callers of retrieve_context() (Phase 4's document task)
    feed to a generation prompt."""
    return "\n\n".join(
        f"[Location -> Source: {c['meta']['source']}, Page: {c['meta']['page']}, Section: {c['meta']['section']}]\n{c['text']}"
        for c in candidates
    )


def retrieve_context(
    query: str,
    target_section: str | None = None,
    target_source: str | None = None,
    top_k: int = TOP_K_CONTEXT,
    deps: PipelineDependencies | None = None,
) -> list[dict]:
    """Retrieval only (embed -> hybrid_search -> rerank -> dedupe), no LLM
    generation — for callers that need grounded context to build their own
    prompt (Phase 4's document task) instead of a direct natural-language
    answer, without paying for a throwaway generation call. Shares the same
    retrieval pipeline as run_query() rather than duplicating it."""
    deps = deps or _default_deps
    if not deps.client.collection_exists(COLLECTION_NAME):
        return []

    query_dense = embed_text(query)
    query_filter = build_filter(target_section, target_source)
    try:
        candidates = hybrid_search(deps.client, COLLECTION_NAME, query, query_dense, query_filter)
    except QdrantConnectionError:
        logger.exception("Retrieval failed for query %r", query)
        return []
    if not candidates:
        return []

    sorted_candidates = rerank(deps.reranker, query, candidates)
    sorted_candidates = deduplicate(sorted_candidates)
    return [c.model_dump() for c in sorted_candidates[:top_k]]


def run_query(
    query: str,
    target_section: str | None = None,
    target_source: str | None = None,
    top_k: int = TOP_K_CONTEXT,
    confidence_threshold: float | None = CONFIDENCE_THRESHOLD,
    deps: PipelineDependencies | None = None,
    on_retrieved=None,
    on_generation_start=None,
    on_token=None,
):
    """Programmatic pipeline entry point: retrieve, rerank, dedup, optionally
    gate, generate, verify citations, flag confidence.

    Returns a result dict (not an exception) on pipeline failure, so eval/
    and the API layer can handle it uniformly. `deps` lets a caller inject an
    alternate PipelineDependencies; defaults to the module-level singleton
    built at import time.
    """
    deps = deps or _default_deps

    result = {
        "query": query,
        "target_section": target_section,
        "target_source": target_source,
        "error": None,
        "candidates": [],
        "context_candidates": [],
        "top_score": None,
        "gated": False,
        "answer": None,
        "citation_verifications": [],
        "confidence": None,
        "timings": {"embed_ms": 0.0, "hybrid_ms": 0.0, "rerank_ms": 0.0, "gen_ms": 0.0},
    }

    if not deps.client.collection_exists(COLLECTION_NAME):
        result["error"] = f"Collection '{COLLECTION_NAME}' does not exist. Run scripts/ingest_knowledge.py first."
        return result

    t0 = time.perf_counter()
    query_dense = embed_text(query)
    result["timings"]["embed_ms"] = (time.perf_counter() - t0) * 1000

    query_filter = build_filter(target_section, target_source)

    t1 = time.perf_counter()
    try:
        candidates = hybrid_search(deps.client, COLLECTION_NAME, query, query_dense, query_filter)
    except QdrantConnectionError as exc:
        logger.exception("Retrieval failed for query %r", query)
        result["error"] = str(exc)
        return result
    result["timings"]["hybrid_ms"] = (time.perf_counter() - t1) * 1000

    if not candidates:
        result["gated"] = True
        result["answer"] = ABSTENTION_MESSAGE
        result["confidence"] = flag_confidence([], None).model_dump()
        return result

    t2 = time.perf_counter()
    sorted_candidates = rerank(deps.reranker, query, candidates)
    sorted_candidates = deduplicate(sorted_candidates)
    result["timings"]["rerank_ms"] = (time.perf_counter() - t2) * 1000

    result["candidates"] = [c.model_dump() for c in sorted_candidates]
    result["top_score"] = sorted_candidates[0].cross_score
    result["context_candidates"] = [c.model_dump() for c in sorted_candidates[:top_k]]

    if on_retrieved:
        on_retrieved(result)

    if confidence_threshold is not None and result["top_score"] < confidence_threshold:
        result["gated"] = True
        result["answer"] = ABSTENTION_MESSAGE
        result["confidence"] = flag_confidence([], result["top_score"]).model_dump()
        return result

    if on_generation_start:
        on_generation_start()

    context_str = format_context(result["context_candidates"])

    t3 = time.perf_counter()
    try:
        result["answer"] = generate(query, context_str, on_token=on_token)
    except LLMInferenceError as exc:
        result["error"] = str(exc)
        return result
    result["timings"]["gen_ms"] = (time.perf_counter() - t3) * 1000

    result["citation_verifications"] = verify_citations(result["answer"], result["context_candidates"])
    result["confidence"] = flag_confidence(result["citation_verifications"], result["top_score"]).model_dump()

    unverified = [v for v in result["citation_verifications"] if not v["verified"]]
    if unverified:
        logger.warning(
            "%d/%d citations failed verification for query %r",
            len(unverified),
            len(result["citation_verifications"]),
            query,
        )

    return result
