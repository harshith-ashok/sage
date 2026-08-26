"""Adapted from insurance_claim_agent/inference/llm.py: generation goes
through app.router.get_chat_model("reasoning") — the Phase 1 model
registry — instead of a hardcoded `ollama.chat(model=...)` call, so
switching the active reasoning model (models.yaml or the Model Registry UI)
changes what drafts knowledge-base answers with zero code touched here.
"""

import logging

from langchain_core.messages import HumanMessage, SystemMessage

from app.knowledge.exceptions import LLMInferenceError
from app.router import get_chat_model

logger = logging.getLogger(__name__)

ABSTENTION_MESSAGE = "This information is not available in the indexed SOP documents."

SYSTEM_PROMPT = (
    "You are a technical assistant for industrial standard operating procedures (SOPs). "
    "Provide highly precise answers derived ONLY from the provided text context.\n"
    "Rules:\n"
    "1. Every single assertion must include an inline location token explicitly showing page "
    "numbers and sections, matching this exact format: (Page X, Section Y).\n"
    f"2. If the context does not explicitly contain the details needed to form a direct answer, "
    f"state exactly: '{ABSTENTION_MESSAGE}' Do not extrapolate.\n"
    "3. Explicitly enumerate all steps, requirements, or thresholds mentioned in the context "
    "without truncation."
)


def generate(query: str, context_str: str, on_token=None) -> str:
    """Streams the routed reasoning model's answer, optionally firing
    on_token(str) per chunk, and returns the full concatenated text."""
    try:
        model = get_chat_model("reasoning")
        messages = [
            SystemMessage(SYSTEM_PROMPT),
            HumanMessage(f"Context:\n{context_str}\n\nQuery: {query}"),
        ]
        tokens = []
        for chunk in model.stream(messages):
            token = chunk.content
            if token:
                tokens.append(token)
                if on_token:
                    on_token(token)
        return "".join(tokens)
    except Exception as exc:
        logger.exception("LLM inference failed")
        raise LLMInferenceError(str(exc)) from exc
