"""Background task + SSE bridge: runs a function on a thread, forwards each
event it emits through a queue as SSE frames — adapted from
insurance_claim_agent/api/streaming.py's queue-bridging pattern, generalized
from one hardcoded query flow into a reusable runner every task endpoint can
use (Phase 4's /tasks/document today; Phase 5/7's /tasks/code and
/tasks/calculate reuse this same module rather than each reimplementing the
thread+queue plumbing).

A POST endpoint can't be consumed by the browser's EventSource (GET-only),
so the pattern here is: POST starts the task and returns a task_id
immediately, then a companion GET /.../{task_id}/stream endpoint streams its
progress — the frontend's streamSSE() (lib/api.ts) EventSource-based client
works against that GET endpoint unmodified.
"""

import json
import queue
import threading
import uuid
from typing import Callable, Iterator

EmitFn = Callable[[str, dict], None]

_tasks: dict[str, "queue.Queue[tuple[str, dict] | None]"] = {}


def start_task(work: Callable[[EmitFn], None]) -> str:
    """Runs `work(emit)` on a background thread and returns a task_id the
    caller can pass to stream_task() to watch its progress. `work` should
    call emit(event_name, data) as it progresses; an uncaught exception is
    turned into an `error` event instead of killing the thread silently."""
    task_id = str(uuid.uuid4())
    q: "queue.Queue[tuple[str, dict] | None]" = queue.Queue()
    _tasks[task_id] = q

    def emit(event: str, data: dict) -> None:
        q.put((event, data))

    def run() -> None:
        try:
            work(emit)
        except Exception as exc:
            emit("error", {"error": str(exc)})
        finally:
            q.put(None)  # sentinel: stream done

    threading.Thread(target=run, daemon=True).start()
    return task_id


def stream_task(task_id: str) -> Iterator[str]:
    """SSE frame generator for a task started with start_task(). Removes the
    task's queue once the stream ends, so a given task_id can only be
    streamed once."""
    q = _tasks.get(task_id)
    if q is None:
        yield f"event: error\ndata: {json.dumps({'error': 'unknown task_id'})}\n\n"
        return
    try:
        while True:
            item = q.get()
            if item is None:
                break
            event, data = item
            yield f"event: {event}\ndata: {json.dumps(data)}\n\n"
    finally:
        _tasks.pop(task_id, None)
