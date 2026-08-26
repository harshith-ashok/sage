"""Ported from insurance_claim_agent/inference/guardrail.py, with
verify_citations rewired onto app.knowledge.embeddings (Ollama, routed via
the model registry) instead of a directly-held sentence-transformers
dense_model — cosine similarity is computed with numpy on plain float lists
instead of sentence_transformers.util.cos_sim on tensors, since embeddings
are no longer tensor objects.

flag_confidence() is new: Phase 3 asks to extend this citation check into a
general confidence flag, not just per-claim verified/unverified booleans
buried in a list — this aggregates citation_verifications + the top
retrieval score into one signal the caller (API/UI) can act on directly,
without blocking generation the way the (disabled-by-default)
CONFIDENCE_THRESHOLD hard gate in query.py does.
"""

import re

import numpy as np

from app.knowledge.config import LOW_CONFIDENCE_THRESHOLD
from app.knowledge.embeddings import embed_text
from app.knowledge.schemas import ConfidenceFlag

CITATION_RE = re.compile(r"\(Page\s+(\d+),\s*Section\s+([^)]*)\)", re.IGNORECASE)


def parse_citations(answer_text: str) -> list[dict]:
    """Extract (Page X, Section Y) tokens from a generated answer."""
    if not answer_text:
        return []
    return [
        {"page": int(m.group(1)), "section": m.group(2).strip()}
        for m in CITATION_RE.finditer(answer_text)
    ]


def citation_claim_spans(answer_text: str) -> list[tuple[str, dict]]:
    """Pair each citation token with the claim text immediately preceding it."""
    if not answer_text:
        return []
    spans = []
    last_end = 0
    for m in CITATION_RE.finditer(answer_text):
        preceding = answer_text[last_end : m.start()]
        claim_text = preceding.splitlines()[-1].strip(" -*•\t") if preceding.strip() else ""
        spans.append((claim_text, {"page": int(m.group(1)), "section": m.group(2).strip()}))
        last_end = m.end()
    return spans


def _cos_sim(a: list[float], b: list[float]) -> float:
    a_arr, b_arr = np.array(a), np.array(b)
    denom = np.linalg.norm(a_arr) * np.linalg.norm(b_arr)
    return float(np.dot(a_arr, b_arr) / denom) if denom else 0.0


def verify_citations(
    answer_text: str, context_candidates: list[dict], similarity_threshold: float = 0.35
) -> list[dict]:
    """Citation *verification*, not just presence: re-embed the claim text
    next to each citation and check it against the specific chunk(s) on that
    cited page, rather than trusting a plausible-looking (Page X, Section Y)
    token. Runtime guardrail (query.py) and the eval suite both call this
    same function."""
    spans = citation_claim_spans(answer_text)
    if not spans:
        return []

    page_to_candidates: dict[int, list[dict]] = {}
    for c in context_candidates:
        page_to_candidates.setdefault(int(c["meta"]["page"]), []).append(c)

    results = []
    for claim_text, citation in spans:
        candidates_for_page = page_to_candidates.get(citation["page"], [])
        if not candidates_for_page or not claim_text:
            results.append(
                {
                    "claim": claim_text,
                    "citation": citation,
                    "verified": False,
                    "similarity": None,
                    "reason": "no_matching_chunk_in_context" if not candidates_for_page else "empty_claim",
                }
            )
            continue
        claim_emb = embed_text(claim_text)
        best_sim = max(_cos_sim(claim_emb, embed_text(c["text"])) for c in candidates_for_page)
        results.append(
            {
                "claim": claim_text,
                "citation": citation,
                "verified": best_sim >= similarity_threshold,
                "similarity": best_sim,
            }
        )
    return results


def flag_confidence(citation_verifications: list[dict], top_score: float | None) -> ConfidenceFlag:
    """Aggregate confidence signal for a generated answer:
    - "unverifiable": the answer made no citations to check at all
    - "low": any citation failed verification, or the top retrieval score
      is below LOW_CONFIDENCE_THRESHOLD
    - "high": otherwise
    """
    total = len(citation_verifications)
    unverified = sum(1 for v in citation_verifications if not v["verified"])

    if total == 0:
        level = "unverifiable"
    elif unverified > 0 or (top_score is not None and top_score < LOW_CONFIDENCE_THRESHOLD):
        level = "low"
    else:
        level = "high"

    return ConfidenceFlag(
        level=level,
        unverified_claims=unverified,
        total_claims=total,
        top_retrieval_score=top_score,
    )
