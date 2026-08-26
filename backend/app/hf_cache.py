"""Shared helper for zero-egress-safe HuggingFace model loading (Phase 3's
reranker, Phase 8's IndicTrans2 translation models): once a model's weights
are cached locally, every subsequent process start should load them
entirely offline — no network call at all, not even the etag/rate-limit
check `huggingface_hub` makes by default on every `from_pretrained()` call
even when nothing actually needs downloading.

Caught live while building Phase 12's network monitor: a real outbound
HTTPS connection to a CloudFront edge (HuggingFace's CDN backend) showed up
on every single backend startup, purely from that check — reverse-DNS
confirmed `*.cloudfront.net`, and it traced directly to `translate.py`'s
`from_pretrained()` calls (the same "unauthenticated requests to the HF
Hub" warning printed at every startup was the visible symptom, just never
connected to an actual network call before this). Exactly the kind of
after-the-fact egress this project's zero-egress guarantee exists to catch,
not something to leave once found.

Only ever goes offline once the repo is already fully present in the local
HF cache — `snapshot_download(..., local_files_only=True)` succeeding
proves that. The very first download for a model that's genuinely never
been pulled before is deliberately left online: this project's own
top-level rule allows exactly one category of network call, the initial
model download — every call after that shouldn't happen, but that first
one still has to.
"""

from huggingface_hub import snapshot_download
from huggingface_hub.utils import LocalEntryNotFoundError


def is_fully_cached(repo_id: str) -> bool:
    try:
        snapshot_download(repo_id, local_files_only=True)
        return True
    except LocalEntryNotFoundError:
        return False
    except Exception:
        # Any other local-only failure (e.g. a corrupted cache entry)
        # shouldn't silently force offline mode over a real problem — let
        # the caller's normal (online) path run and surface the real error.
        return False


def offline_kwargs(repo_id: str) -> dict:
    """`local_files_only=True` if `repo_id` is already fully cached, else
    `{}` — falls through to the library's normal online behavior for a
    genuine first pull."""
    return {"local_files_only": True} if is_fully_cached(repo_id) else {}
