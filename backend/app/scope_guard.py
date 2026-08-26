"""Topic guardrail: refuses a request before the agent ever runs, rather
than just instructing the model to decline in its own system prompt. A
prompt instruction is soft — a smaller/faster local model in particular can
just as easily answer an off-topic question anyway, especially once
conversation history makes the instruction less salient a few turns in.
This is a hard gate: an out-of-scope prompt never reaches `create_react_agent`
at all, so it's not a matter of the model "choosing" to follow a rule.

**Why an LLM call, not a keyword/regex filter**: topic relevance is a
judgment call — "what's the weather" is obviously out of scope but "why did
the reading fail" needs real understanding of the surrounding context to
place. This project already has a strong bias against freeform LLM
judgment where a deterministic tool exists (Phase 5: calculations always go
through sympy, never the model's own arithmetic) — but there IS no
deterministic ground truth for "is this on-topic," so a small classification
call is the right tool here, not a shortcut around one. Kept cheap the same
way Phase 11's complexity tiering is: always the "fast" local candidate for
`reasoning`, never the heavier "strong" one, and a single short yes/no
completion, not a full agent turn.
"""

from langchain_core.messages import HumanMessage

from app.router import get_chat_model

SCOPE_SYSTEM_PROMPT = (
    "You are a strict topic classifier for SAGE, an air-gapped industrial/technical "
    "assistant. SAGE's domain is: industrial and engineering operations, safety "
    "procedures (SOPs), equipment inspection/maintenance/compliance, engineering "
    "correspondence and reports, and general professional/technical work SAGE can "
    "help perform — writing or running code, doing calculations or statistical "
    "fitting, drafting or exporting documents/spreadsheets/presentations, "
    "translating text, transcribing audio, or reading an attached image/document — "
    "even when that request doesn't name a specific procedure or piece of equipment. "
    "Attachments (an image, a document, audio) always count as in scope — someone "
    "attaching a file is presumptively doing real work with it.\n\n"
    "Answer with exactly one word: IN_SCOPE or OUT_OF_SCOPE. Nothing else.\n\n"
    "IN_SCOPE examples: \"How often shall critical service valves be visually "
    "inspected?\", \"Write code that prints the first 10 primes\", \"What is 5000 N "
    "over 0.02 m² in pascals?\", \"Fit a regression on this data and give me the "
    "slope\", \"Translate this into Hindi\", \"Summarize the attached report\", "
    "\"What's in the knowledge base about lockout/tagout?\".\n"
    "OUT_OF_SCOPE examples: \"What's the weather today?\", \"Tell me a joke\", "
    "\"Who won the game last night?\", \"What's a good recipe for pasta?\", \"Write "
    "me a poem about the ocean\", \"What's the meaning of life?\"."
)

REFUSAL_MESSAGE = (
    "I'm SAGE, an assistant for industrial/engineering operations, safety procedures, "
    "and this site's knowledge base — along with the technical tasks that support that "
    "work (code, calculations, documents, translation). That question is outside what "
    "I'm built to help with, so I can't answer it here. Try asking about an SOP, an "
    "inspection or compliance question, or a technical task tied to your work."
)


def is_in_scope(prompt: str, has_attachment: bool) -> bool:
    if has_attachment:
        return True
    if not prompt.strip():
        return True

    try:
        return _classify(prompt)
    except Exception:
        # A guardrail that can't run is not a reason to refuse every
        # message until it's fixed — if the classifier call fails outright
        # (e.g. Ollama unreachable), let the request through; the main
        # agent turn will hit and surface the same underlying problem on
        # its own next, with a real error instead of a silent, confusing
        # blanket refusal.
        return True


def _classify(prompt: str) -> bool:
    # reasoning=False here is load-bearing, not an optimization to skip:
    # Ollama's reasoning-capable local models (qwen3 and friends) "think"
    # through even a one-word classification by default, which measured at
    # 11-22s per call — an unacceptable delay added to every single message,
    # not just the ones that get refused. Disabling it for this specific
    # call (via .bind(), not touching the shared cached model app.router
    # returns) dropped that to consistently under half a second with no
    # change in classification accuracy in testing.
    model = get_chat_model("reasoning", tier="fast")
    try:
        response = model.bind(reasoning=False).invoke(
            [HumanMessage(SCOPE_SYSTEM_PROMPT), HumanMessage(f"Classify this request:\n\n{prompt}")]
        )
    except Exception:
        # `reasoning=False` is verified safe on the thinking-capable local
        # candidates this project actually ships (qwen3 and friends) — but
        # app/router.py's own get_chat_model() already shows a
        # non-thinking model can 400 on an unexpected reasoning kwarg
        # (that's exactly why it detects support per-model there before
        # deciding whether to pass reasoning=True at all). A guardrail
        # failing shouldn't be the reason a legitimate request never gets
        # answered, so retry once without forcing the kwarg at all rather
        # than assume every future "fast" candidate handles it the same way.
        response = model.invoke([HumanMessage(SCOPE_SYSTEM_PROMPT), HumanMessage(f"Classify this request:\n\n{prompt}")])
    verdict = str(response.content).strip().upper()
    # Defaults to allowing the request through on anything but a clear,
    # exact "OUT_OF_SCOPE" — a malformed/unexpected classifier response
    # should never be the reason a legitimate work question gets refused.
    return "OUT_OF_SCOPE" not in verdict
