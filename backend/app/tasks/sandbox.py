"""Sandboxed Python execution: `docker run --network none`, a throwaway
temp directory as the only mount, deleted after every run.

hi/hi/agents/term/shell_pane.py (also named in Phase 5's plan) runs
commands in a tmux pane the user can see and interact with — that model is
architecturally the opposite of a sandbox: it deliberately gives the
command full access to the user's own visible shell, and it requires an
interactive tmux session, which a headless FastAPI backend doesn't have.
Nothing there was reusable for actual sandboxing, so this module is built
fresh against the Stack's real "Docker sandbox (no network)" requirement —
danger.py (also named in the plan) is the piece that *did* port directly,
as the pre-flight gate below.
"""

import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import TypedDict

from app.tasks.danger import looks_destructive

SANDBOX_IMAGE = "sage-sandbox:latest"  # docker/sandbox/Dockerfile — python:3.12-slim + numpy/scipy pre-installed


class SandboxResult(TypedDict):
    stdout: str
    stderr: str
    exit_code: int | None
    timed_out: bool
    refused: str | None  # set (to a reason) instead of running, if danger.looks_destructive() flagged the code


def run_python(code: str, timeout: float = 20.0) -> SandboxResult:
    if looks_destructive(code):
        return {
            "stdout": "",
            "stderr": "",
            "exit_code": None,
            "timed_out": False,
            "refused": "Code matched a destructive-pattern heuristic (app/tasks/danger.py) and was not executed.",
        }

    workdir = Path(tempfile.mkdtemp(prefix="sage_sandbox_"))
    try:
        (workdir / "script.py").write_text(code)
        try:
            proc = subprocess.run(
                [
                    "docker",
                    "run",
                    "--rm",
                    "--network",
                    "none",
                    "--memory",
                    "256m",
                    "--cpus",
                    "0.5",
                    "--pids-limit",
                    "128",
                    "-v",
                    f"{workdir}:/sandbox:ro",
                    "-w",
                    "/sandbox",
                    SANDBOX_IMAGE,
                    "python",
                    "script.py",
                ],
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            return {
                "stdout": proc.stdout,
                "stderr": proc.stderr,
                "exit_code": proc.returncode,
                "timed_out": False,
                "refused": None,
            }
        except subprocess.TimeoutExpired as exc:
            return {
                "stdout": (exc.stdout or b"").decode() if isinstance(exc.stdout, bytes) else (exc.stdout or ""),
                "stderr": ((exc.stderr or b"").decode() if isinstance(exc.stderr, bytes) else (exc.stderr or "")) + "\n(timed out)",
                "exit_code": None,
                "timed_out": True,
                "refused": None,
            }
    finally:
        shutil.rmtree(workdir, ignore_errors=True)
