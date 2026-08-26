"""Phase 12: zero-egress proof. Not `nethogs`/`iptables` — neither exists on
this dev machine (Darwin, no Linux-only net-namespace tooling), so this
polls the same real signal via macOS-native tools already installed:
`lsof -i` for per-process open sockets. This is detection/visibility, not
enforcement — the actual blocking already exists where it matters (Phase 5's
`docker run --network none` sandbox has no route out at all, proven with a
real DNS-failure test). The point here is to make any egress *visible*, so
a demo can show "zero external calls" happening live, and so a
misconfigured cloud-model default gets caught instead of silently leaking.

**Which processes get watched, and why both matter**: the SAGE backend's
own process (`os.getpid()`) is the obvious one, but it isn't the only place
a network call can actually originate from. When an Ollama *-cloud* model
is active, the Python backend only ever talks to the local Ollama daemon on
127.0.0.1:11434 — the daemon itself is what opens the real outbound
connection to Ollama's cloud API. Watching only the backend's own PID would
show a false "zero calls" even while a cloud model is mid-request, so the
Ollama daemon's PID(s) (`pgrep ollama`) are polled too.

**A real violation this module caught while being built, immediately
fixed**: `lsof` on the backend's own PID showed a genuine outbound HTTPS
connection to a CloudFront edge (reverse-DNS confirmed `*.cloudfront.net`)
on every single startup — traced to `huggingface_hub`'s default etag/
rate-limit check on `from_pretrained()`, which reaches out even when every
file it needs is already cached locally. Fixed at the root in
app/hf_cache.py (skip that check once a model is confirmed fully cached),
not by hiding it from this monitor.
"""

import re
import subprocess
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone

# Private/loopback ranges plus this machine's own LAN prefix count as
# "local" — a connection to another private-network host (e.g. the Docker
# daemon's own bridge, or this box's own LAN IP) is not egress to the
# public internet, which is what zero-egress actually means. Anything else
# — a public IP, a hostname resolving off-box — is external.
_LOCAL_IP_PATTERNS = [
    re.compile(r"^127\."),
    re.compile(r"^10\."),
    re.compile(r"^172\.(1[6-9]|2\d|3[01])\."),
    re.compile(r"^192\.168\."),
    re.compile(r"^::1$"),
    re.compile(r"^\[?::1\]?$"),
    re.compile(r"^fe80:", re.IGNORECASE),
]

_MAX_EXTERNAL_LOG = 50
_log_lock = threading.Lock()
_external_log: list["ExternalConnection"] = []
_seen_keys: set[str] = set()  # dedupe: (pid, remote) already logged once


@dataclass
class ExternalConnection:
    pid: int
    process: str
    remote: str
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


@dataclass
class NetworkSnapshot:
    external_count: int
    external_connections: list[ExternalConnection]
    local_connection_count: int
    watched_processes: list[str]
    external_log: list[ExternalConnection]


def _is_local_host(host: str) -> bool:
    host = host.strip("[]")
    if host == "localhost":
        return True
    return any(p.match(host) for p in _LOCAL_IP_PATTERNS)


def _find_pids(process_name: str) -> list[tuple[int, str]]:
    try:
        result = subprocess.run(["pgrep", "-x", process_name], capture_output=True, text=True, timeout=3)
    except (OSError, subprocess.TimeoutExpired):
        return []
    pids = [int(p) for p in result.stdout.split() if p.strip()]
    return [(pid, process_name) for pid in pids]


def _watched_pids() -> list[tuple[int, str]]:
    import os

    watched = [(os.getpid(), "sage-backend")]
    watched += _find_pids("ollama")
    return watched


# Matches lsof -F output's connection field, e.g.
# "192.168.20.101:54244->13.33.45.84:443" or "127.0.0.1:8000" (listening).
_CONN_LINE = re.compile(r"^n(?:\*|(?P<lhost>[^:]+):(?P<lport>\d+))(?:->(?P<rhost>[^:]+):(?P<rport>\d+))?")


def _connections_for_pid(pid: int) -> list[str]:
    """Returns raw "host:port->host:port" strings for established
    connections owned by `pid`, via `lsof -F n` (machine-readable field
    output — no header/column parsing needed)."""
    try:
        result = subprocess.run(
            ["lsof", "-i", "-n", "-P", "-a", "-p", str(pid), "-F", "n"],
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.TimeoutExpired):
        return []
    conns = []
    for line in result.stdout.splitlines():
        m = _CONN_LINE.match(line)
        if m and m.group("rhost"):
            conns.append(f"{m.group('rhost')}:{m.group('rport')}")
    return conns


def snapshot() -> NetworkSnapshot:
    watched = _watched_pids()
    external: list[ExternalConnection] = []
    local_count = 0

    for pid, name in watched:
        for remote in _connections_for_pid(pid):
            host = remote.rsplit(":", 1)[0]
            if _is_local_host(host):
                local_count += 1
                continue
            conn = ExternalConnection(pid=pid, process=name, remote=remote)
            external.append(conn)
            key = f"{pid}:{remote}"
            with _log_lock:
                if key not in _seen_keys:
                    _seen_keys.add(key)
                    _external_log.append(conn)
                    del _external_log[:-_MAX_EXTERNAL_LOG]

    with _log_lock:
        log_copy = list(_external_log)

    return NetworkSnapshot(
        external_count=len(external),
        external_connections=external,
        local_connection_count=local_count,
        watched_processes=[f"{name} (pid {pid})" for pid, name in watched],
        external_log=log_copy,
    )
