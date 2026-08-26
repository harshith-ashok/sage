"""Phase 11: complexity-based model tiering — route a simple subtask to a
faster/lighter model and a harder one to a stronger model, the way Claude's
own haiku/sonnet/opus tiers work, applied to this project's task types.

**Deliberately a cheap heuristic, not a classifier model.** The phase's own
instruction is to implement this only if it doesn't add meaningful overhead
or latency — running a real classification model (even a small one) before
every request would itself add a model call and a real latency cost to
*every single turn*, working against the very thing this feature is for.
Word-count-and-keyword heuristics on the prompt text are effectively free
(microseconds, no I/O) and, more importantly, err in the *safe* direction:
this project's outputs are grounded documents about industrial/safety
procedures (SOPs, inspection reports, contradiction checks), so a wrong
"simple" classification that should have been "strong" is a real quality
risk, while a wrong "strong" that should have been "fast" only costs a
little latency. The thresholds below are tuned to be conservative about
that asymmetry — defaulting to "strong" whenever a signal is ambiguous.

**Scope**: only `reasoning` (app/agent.py's own ReAct loop, the model
behind nearly every user-facing turn) and `coding` (app/tasks/code.py's
code-generation subtask, invoked *within* a reasoning turn via the
`run_sandboxed_code` tool — a real second example of the "different tasks
and subtasks" the phase asks for, not just one). Vision/embedding/
translation aren't tiered: each is a single fixed-purpose call already
scoped by its own task, not something whose complexity varies turn to turn
the way a free-form chat prompt or a generated script does.
"""

import re

# A prompt shorter than this, with none of the complexity signals below, is
# treated as "fast" — simple factual lookups, short confirmations, trivial
# one-liners ("what's 2+2", "say hello", "list the SOPs you have").
_SHORT_PROMPT_WORD_COUNT = 12

# Signals that a request needs real reasoning, synthesis across sources, or
# careful/safety-relevant judgment — any one of these is enough to route to
# the "strong" tier, matching the conservative-by-default design above.
_STRONG_SIGNAL_PATTERN = re.compile(
    r"\b("
    r"why|explain|analy[sz]e|compare|derive|prove|justify|evaluate|assess|"
    r"summar(?:y|ize)|contradiction|conflict|regression|correlat|statistic|"
    r"confidence interval|step[- ]by[- ]step|calculate|equation|"
    r"draft|report|approval|sop|procedure|inspect|escalat|compliance|"
    r"safety|hazard"
    r")\b",
    re.IGNORECASE,
)


def classify_complexity(text: str, has_attachment: bool = False) -> str:
    """Returns "fast" or "strong". `has_attachment` covers image/document/
    audio/P&ID input — multimodal or grounded synthesis tasks skew complex
    almost by definition (something needs to be read and reasoned about,
    not just answered from a short prompt), so an attachment alone routes
    to "strong" regardless of prompt length."""
    if has_attachment:
        return "strong"

    stripped = text.strip()
    if not stripped:
        return "fast"

    if _STRONG_SIGNAL_PATTERN.search(stripped):
        return "strong"

    word_count = len(stripped.split())
    if word_count > _SHORT_PROMPT_WORD_COUNT:
        return "strong"

    # Multiple distinct asks in one short message (e.g. "do X and Y and Z")
    # add real complexity a raw word count alone would miss.
    if stripped.count("?") > 1 or len(re.findall(r"\band\b", stripped, re.IGNORECASE)) > 1:
        return "strong"

    return "fast"
