"""Adapted from insurance_claim_agent/retrieval/dependencies.py. Drops
dense_model — dense embeddings go through app/knowledge/embeddings.py
(routed via the model registry), not a client held here — this only holds
what still needs an eagerly-constructed live client: Qdrant and the
reranker.
"""

from dataclasses import dataclass

from qdrant_client import QdrantClient
from sentence_transformers import CrossEncoder

from app.knowledge.config import QDRANT_URL, RERANKER_MODEL_NAME


@dataclass
class PipelineDependencies:
    client: QdrantClient
    reranker: CrossEncoder

    @classmethod
    def build(cls) -> "PipelineDependencies":
        return cls(
            client=QdrantClient(url=QDRANT_URL),
            reranker=CrossEncoder(RERANKER_MODEL_NAME),
        )
