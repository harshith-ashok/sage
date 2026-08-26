"""Knowledge-base constants, ported from insurance_claim_agent/config.py.
Kept separate from app/config.py (the model registry) — this is retrieval
tuning, not model routing.
"""

QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "sage_sop_docs"

DOCS_DIR = "data/sop_docs"

# Dense embeddings go through the Phase 1 router's "embedding" task type
# (app/router.py's get_embedding_model), not a hardcoded model name — see
# app/knowledge/embeddings.py. RERANKER_MODEL_NAME is a small (~80MB)
# sentence-transformers cross-encoder, downloaded once and run fully
# in-process after that — same one-time-download-then-air-gapped pattern as
# the Ollama model weights, so it isn't routed through models.yaml.
RERANKER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# Chunking (parsing.py) — sliding-window RecursiveCharacterTextSplitter.
CHUNK_SIZE = 1500
CHUNK_OVERLAP = 300

# Retrieval (search.py). PREFETCH_LIMIT is the candidate pool pulled per
# vector type (dense, sparse) before RRF fusion; RRF_LIMIT is how many fused
# candidates go to the reranker; TOP_K_CONTEXT is how many reranked chunks
# are actually sent to the LLM as context.
PREFETCH_LIMIT = 30
RRF_LIMIT = 15
TOP_K_CONTEXT = 4

# Near-duplicate suppression: chunks whose text similarity (difflib ratio) to
# an already-kept, higher-ranked, same-(source,page) candidate meets or
# exceeds this are dropped before taking TOP_K_CONTEXT.
DEDUP_SIMILARITY_THRESHOLD = 0.9

# Hard abstention gate: if set, run_query() skips the LLM call entirely when
# the top reranked score is below this. Disabled by default, matching
# insurance_claim_agent's finding that a threshold tight enough to catch
# hallucinations also wrongly refuses correct answers on a small corpus.
CONFIDENCE_THRESHOLD = None

# Softer signal (Phase 3's "confidence flagging"): unlike CONFIDENCE_THRESHOLD,
# this never blocks generation — app/knowledge/guardrail.py's flag_confidence()
# uses it to mark an answer "low" confidence (surfaced to the caller/UI) when
# the top retrieval score falls below it, even though an answer was still
# generated.
LOW_CONFIDENCE_THRESHOLD = 0.15
