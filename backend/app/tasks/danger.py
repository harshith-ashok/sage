"""Ported verbatim from hi/hi/agents/term/danger.py. Flags shell/code that
looks destructive or hard to reverse — used here as a pre-flight gate before
sandboxed execution (app/tasks/code.py): the Docker sandbox (--network none,
throwaway mount, deleted after every run) already bounds the blast radius,
but refusing an obviously destructive command outright is still cheaper and
clearer than "ran it in a disposable box and it did something pointless."
This is a heuristic, not a sandbox or a process inspector — it only gates
behavior, not what a command is capable of doing.
"""

import re

_DESTRUCTIVE_PATTERNS = [
    r"\brm\b",
    r"\bmv\b",
    r"\bsudo\b",
    r"\bchmod\b",
    r"\bchown\b",
    r"\bdd\b",
    r"\bmkfs(\.\w+)?\b",
    r"\bkill(all)?\b",
    r"\bpkill\b",
    r"\bshutdown\b",
    r"\breboot\b",
    r">>?",
    r"\bgit\s+push\s+.*--force",
    r"\bgit\s+reset\s+--hard\b",
    r"\bgit\s+clean\s+-[a-z]*f",
    r"\|\s*(sh|bash|zsh)\b",
]

_DESTRUCTIVE_RE = re.compile("|".join(_DESTRUCTIVE_PATTERNS))


def looks_destructive(command: str) -> bool:
    return bool(_DESTRUCTIVE_RE.search(command))
