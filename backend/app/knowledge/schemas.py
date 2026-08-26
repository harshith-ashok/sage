"""Pydantic models for the knowledge pipeline. Ported from
insurance_claim_agent/models.py (renamed to schemas.py here since "models"
in SAGE means the LLM registry — app/config.py's ModelCandidate etc. —
and reusing that name for pydantic schemas would be confusing side by side).
"""

from pydantic import BaseModel


class ChunkMetadata(BaseModel):
    page: int
    chunk_id: str
    section: str
    source: str


class Chunk(BaseModel):
    """A single ingested unit of text, produced by parsing.py."""

    text: str
    metadata: ChunkMetadata


class RetrievalCandidate(BaseModel):
    """A chunk retrieved for a query, carrying its fusion/rerank scores."""

    text: str
    rrf_score: float
    meta: ChunkMetadata
    cross_score: float | None = None


class CitationVerification(BaseModel):
    claim: str
    citation: dict
    verified: bool
    similarity: float | None = None
    reason: str | None = None


class ConfidenceFlag(BaseModel):
    """Phase 3's confidence flagging: an aggregate signal over
    citation_verifications + the top retrieval score, distinct from the
    (disabled-by-default) hard abstention gate — this never blocks
    generation, it just marks the result for the caller/UI."""

    level: str  # "high" | "low" | "unverifiable"
    unverified_claims: int
    total_claims: int
    top_retrieval_score: float | None


class QueryResult(BaseModel):
    """Internal representation of a run_query() call; converted to a plain
    dict via model_dump() at the run_query() boundary."""

    query: str
    target_section: str | None = None
    target_source: str | None = None
    error: str | None = None
    candidates: list[RetrievalCandidate] = []
    context_candidates: list[RetrievalCandidate] = []
    top_score: float | None = None
    gated: bool = False
    answer: str | None = None
    citation_verifications: list[CitationVerification] = []
    confidence: ConfidenceFlag | None = None
