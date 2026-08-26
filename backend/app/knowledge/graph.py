"""Knowledge Base graph view: the actual ingested corpus structure (which
documents, which sections, how many chunks each), not a search result — the
Knowledge Base view's new graph panel renders this once on load, then
highlights the relevant nodes/edges after a search rather than re-fetching.

Reads directly from Qdrant via a scroll (the same collection app/knowledge's
retrieval pipeline already queries) rather than adding a parallel metadata
store — the corpus is small enough (tens of documents, low hundreds of
chunks at most for this project's scope) that scrolling the whole
collection on each request is simpler and always-correct, not a
performance concern worth caching around.
"""

import re

from qdrant_client import QdrantClient

from app.knowledge.config import COLLECTION_NAME, QDRANT_URL

_TITLE_STRIP_RE = re.compile(r"^[#*\s]+|[#*\s]+$")


_MAX_TITLE_LEN = 90
_HEADING_RE = re.compile(r"^(#{1,6})\s")


def _derive_title(source: str, chunks: list[tuple[int, str]]) -> str:
    """Prefer the document's own top-level markdown heading (e.g. "#
    **SOP-LOTO-400: Lockout/Tagout Procedure**" -> "SOP-LOTO-400:
    Lockout/Tagout Procedure") over the raw filename, which is usually just
    a slugified version of the same title. `chunks` is every (page, text)
    chunk for this document — chunking is by character window, not by
    page, so several chunks can share a page number and only one of them
    is actually the title (the other candidates start mid-section); the
    real title is the chunk whose first line is the *highest-level*
    heading (fewest leading '#'), earliest page breaking ties. A chunk
    boundary landing mid-paragraph with no heading at all is rejected
    outright rather than mistaken for a title."""
    best: tuple[int, int, str] | None = None  # (heading_level, page, cleaned_title)
    for page, text in chunks:
        first_line = text.strip().splitlines()[0] if text.strip() else ""
        m = _HEADING_RE.match(first_line.lstrip())
        if not m:
            continue
        cleaned = _TITLE_STRIP_RE.sub("", first_line).strip()
        if not cleaned or len(cleaned) > _MAX_TITLE_LEN:
            continue
        level = len(m.group(1))
        if best is None or (level, page) < (best[0], best[1]):
            best = (level, page, cleaned)

    if best:
        return best[2]
    stem = re.sub(r"\.(pdf|md|docx)$", "", source, flags=re.IGNORECASE)
    return stem.replace("-", " ").replace("_", " ").title()


def build_corpus_graph() -> dict:
    client = QdrantClient(url=QDRANT_URL)
    if not client.collection_exists(COLLECTION_NAME):
        return {"documents": [], "total_chunks": 0}

    points, _ = client.scroll(collection_name=COLLECTION_NAME, limit=10000, with_payload=True, with_vectors=False)

    # source -> section -> list of {chunk_id, page, preview}
    by_source: dict[str, dict] = {}
    for point in points:
        payload = point.payload or {}
        meta = payload.get("metadata", {})
        source = meta.get("source", "unknown")
        section = meta.get("section") or "(no section)"
        text = payload.get("text", "")
        page = meta.get("page", 0)
        preview = text.strip().replace("\n", " ")[:160]

        doc = by_source.setdefault(source, {"chunks": [], "sections": {}})
        doc["chunks"].append((page, text))
        sec = doc["sections"].setdefault(section, [])
        sec.append({"chunk_id": meta.get("chunk_id", str(point.id)), "page": page, "preview": preview})

    documents = []
    for source, doc in sorted(by_source.items()):
        sections = [
            {"section": section_name, "chunk_count": len(chunks), "chunks": sorted(chunks, key=lambda c: c["page"])}
            for section_name, chunks in doc["sections"].items()
        ]
        # Sections sort naturally where possible (numeric sections first, in order).
        sections.sort(key=lambda s: (0, int(s["section"])) if s["section"].isdigit() else (1, s["section"]))
        documents.append(
            {
                "source": source,
                "title": _derive_title(source, doc["chunks"]),
                "chunk_count": len(doc["chunks"]),
                "sections": sections,
            }
        )

    return {"documents": documents, "total_chunks": len(points)}
