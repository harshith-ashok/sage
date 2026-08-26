"""Phase 5: sandboxed coding task. generate -> execute -> verify loop
(same plan/act/verify shape as Phase 1's orchestrator and Phase 4's
document task): a coding-model-written Python script is run in the
Docker sandbox (app/tasks/sandbox.py) and its stdout is checked against an
expected output — not "does the LLM think it's right," an actual run.
A failing attempt feeds its real stdout/stderr back into the next
generation, bounded by MAX_ATTEMPTS.

`expected_output` is optional: when given, verification is Phase 5's
original exact-string match. When omitted, there's no single correct
string for genuinely open-ended/analytical code (e.g. "fit a regression and
report the coefficients") to match, so verification instead just confirms
the script actually ran (exit 0) and printed something — real execution,
not a guess, without forcing a fixed-answer shape onto exploratory tasks.
Caught live: without this, a stats-heavy request kept "failing" and
retrying correct, working code for 3 straight attempts every time, purely
because nothing it could print would ever equal a guessed expected string.
"""

from typing import Callable, TypedDict

from langchain_core.messages import HumanMessage
from langgraph.graph import END, StateGraph

from app.complexity import classify_complexity
from app.model_warmup import invoke_with_retry
from app.router import get_chat_model
from app.tasks.sandbox import run_python

MAX_ATTEMPTS = 3

GENERATE_SYSTEM_PROMPT = (
    "You write a single self-contained Python script that solves the given task and "
    "prints its result to stdout. Rules:\n"
    "1. Output ONLY the Python code, no explanation, no markdown code fences.\n"
    "2. The script must run standalone — numpy and scipy are pre-installed in the "
    "sandbox and may be imported; no other third-party packages and no pip installs "
    "are possible (the sandbox has no network access).\n"
    "3. If an exact expected output is given, the script's stdout must match it "
    "exactly. If none is given, there's no fixed string to match — just make sure the "
    "script actually runs without error and prints whatever real, computed values the "
    "task calls for."
)


class CodeTaskState(TypedDict):
    task: str
    expected_output: str | None
    tier: str
    code: str
    stdout: str
    stderr: str
    exit_code: int | None
    refused: str | None
    passed: bool
    feedback: str
    attempts: int


def _strip_code_fence(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines[-1].strip() == "```":
            lines = lines[1:-1]
        else:
            lines = lines[1:]
        if lines and lines[0].strip().lower() in ("python", "py"):
            lines = lines[1:]
        text = "\n".join(lines)
    return text.strip()


def generate_node(state: CodeTaskState) -> dict:
    model = get_chat_model("coding", tier=state.get("tier"))
    retry_note = (
        f"\n\nThe previous attempt was rejected:\n```\n{state['code']}\n```\n"
        f"stdout was:\n{state['stdout']!r}\nstderr was:\n{state['stderr']!r}\n"
        f"Expected stdout:\n{state['expected_output']!r}\n"
        f"Reason for rejection: {state['feedback']}\nFix it."
        if state["feedback"]
        else ""
    )
    expected_line = (
        f"Expected stdout (must match exactly):\n{state['expected_output']}"
        if state["expected_output"]
        else "No exact expected output given — print the real computed result(s)."
    )
    prompt = f"Task:\n{state['task']}\n\n{expected_line}{retry_note}"
    code = invoke_with_retry(model, [HumanMessage(GENERATE_SYSTEM_PROMPT + "\n\n" + prompt)])
    return {"code": _strip_code_fence(code), "attempts": state["attempts"] + 1}


def execute_node(state: CodeTaskState) -> dict:
    result = run_python(state["code"])
    return {
        "stdout": result["stdout"],
        "stderr": result["stderr"],
        "exit_code": result["exit_code"],
        "refused": result["refused"],
    }


def verify_node(state: CodeTaskState) -> dict:
    if state["refused"]:
        return {"passed": False, "feedback": state["refused"]}
    if state["exit_code"] != 0:
        return {"passed": False, "feedback": f"Script exited with code {state['exit_code']}. stderr:\n{state['stderr']}"}
    if not state["expected_output"]:
        # No exact string to check against (open-ended/analytical code) —
        # exit_code == 0 above already confirmed it actually ran; a blank
        # stdout on a task that was clearly supposed to print something is
        # still worth catching, otherwise this is as far as verification
        # can go without a fixed expected answer.
        if not state["stdout"].strip():
            return {"passed": False, "feedback": "Script ran without error but printed nothing."}
        return {"passed": True, "feedback": ""}
    actual = state["stdout"].strip()
    expected = state["expected_output"].strip()
    if actual == expected:
        return {"passed": True, "feedback": ""}
    return {"passed": False, "feedback": f"stdout did not match. Expected:\n{expected!r}\nGot:\n{actual!r}"}


def route_after_verify(state: CodeTaskState) -> str:
    if state["passed"] or state["attempts"] >= MAX_ATTEMPTS:
        return "report"
    return "generate"


def report_node(state: CodeTaskState) -> dict:
    return {
        "passed": state["passed"],
        "code": state["code"],
        "stdout": state["stdout"],
        "attempts": state["attempts"],
    }


def _build_graph():
    graph = StateGraph(CodeTaskState)
    graph.add_node("generate", generate_node)
    graph.add_node("execute", execute_node)
    graph.add_node("verify", verify_node)
    graph.add_node("report", report_node)
    graph.set_entry_point("generate")
    graph.add_edge("generate", "execute")
    graph.add_edge("execute", "verify")
    graph.add_conditional_edges("verify", route_after_verify, {"generate": "generate", "report": "report"})
    graph.add_edge("report", END)
    return graph.compile()


_graph = _build_graph()

_NODE_EVENTS = {
    "generate": "generate_done",
    "execute": "execute_done",
    "verify": "verify_done",
    "report": "done",
}


def run_code_task(task: str, expected_output: str | None, emit: Callable[[str, dict], None]) -> None:
    # Phase 11: complexity-based tiering for the coding subtask, same
    # heuristic and same reasoning as app/agent.py's own tiering of the
    # outer reasoning turn (see app/complexity.py) — a second real
    # "subtask" example, not just the top-level chat model.
    tier = classify_complexity(task)
    emit("tier_selected", {"task_type": "coding", "tier": tier})
    initial: CodeTaskState = {
        "task": task,
        "expected_output": expected_output,
        "tier": tier,
        "code": "",
        "stdout": "",
        "stderr": "",
        "exit_code": None,
        "refused": None,
        "passed": False,
        "feedback": "",
        "attempts": 0,
    }
    for step in _graph.stream(initial, stream_mode="updates"):
        for node_name, update in step.items():
            # LangGraph reports a node that returned {} (report_node, a
            # deliberate no-op) as None here, not {} — dict(None) would raise.
            event = _NODE_EVENTS.get(node_name, node_name)
            emit(event, dict(update or {}))
