"""Retrieval metrics, ported from insurance_claim_agent/eval/metrics.py.

Only recall_at_k/precision_at_k/mrr are ported — Phase 3 asks for "a real
recall/precision number," not the reference repo's full abstention/
consistency/RAGAS suite (citation_accuracy, classify_abstention,
answer_similarity, citation_agreement); those can be added later against
this same golden set if a later phase needs them.
"""

__all__ = ["recall_at_k", "precision_at_k", "mrr"]


def _matches_ground_truth(candidate_meta: dict, expected_citation: dict) -> bool:
    """Ground truth is matched on (source, page) only — section labels are a
    known-noisy forward-carrying heuristic (see parsing.py), not a hard
    match key."""
    if candidate_meta.get("source") != expected_citation.get("source"):
        return False
    try:
        return int(candidate_meta.get("page")) == int(expected_citation.get("page"))
    except (TypeError, ValueError):
        return False


def recall_at_k(ranked_candidates: list[dict], expected_citations: list[dict], k: int) -> float | None:
    """Fraction of expected ground-truth chunks that appear in the top-k
    retrieved candidates. None when the question has no ground-truth chunk
    (unanswerable)."""
    if not expected_citations:
        return None
    top_k = ranked_candidates[:k]
    found = sum(
        1 for exp in expected_citations if any(_matches_ground_truth(c["meta"], exp) for c in top_k)
    )
    return found / len(expected_citations)


def precision_at_k(ranked_candidates: list[dict], expected_citations: list[dict], k: int) -> float | None:
    if not expected_citations:
        return None
    top_k = ranked_candidates[:k]
    if not top_k:
        return 0.0
    relevant = sum(
        1 for c in top_k if any(_matches_ground_truth(c["meta"], exp) for exp in expected_citations)
    )
    return relevant / len(top_k)


def mrr(ranked_candidates: list[dict], expected_citations: list[dict]) -> float | None:
    """Reciprocal rank of the first retrieved candidate that matches any
    expected ground-truth chunk, over the full ranked candidate list (not
    just top-k)."""
    if not expected_citations:
        return None
    for i, c in enumerate(ranked_candidates, start=1):
        if any(_matches_ground_truth(c["meta"], exp) for exp in expected_citations):
            return 1.0 / i
    return 0.0
