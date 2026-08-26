"""Ported verbatim from insurance_claim_agent/exceptions.py."""


class PipelineError(Exception):
    """Base class for errors raised by the ingestion/retrieval/inference pipeline."""


class QdrantConnectionError(PipelineError):
    """Qdrant was unreachable or returned an error during a query/upload."""


class LLMInferenceError(PipelineError):
    """The chat model call failed or returned an unusable response."""
