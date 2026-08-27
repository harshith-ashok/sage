"""FastAPI app exposing the model router/registry (Phase 1) so it can be
proven live and, later, driven from the frontend's Model Registry view
(Phase 6). CORS is scoped to Vite's default dev origin, matching
insurance_claim_agent/api/main.py's pattern.
"""

import base64
import io
import os
import tempfile
import zipfile

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel

from app import config as model_config
from app.documents.extract import UnsupportedDocumentType, extract_text, render_pdf_pages
from app.knowledge.graph import build_corpus_graph
from app.knowledge.query import run_query
from app.language.pipeline import run_agent_multilingual
from app.language.translate import SUPPORTED_LANGUAGES, translate_from_english
from app.load_monitor import get_fallback_log, get_system_load
from app.network_monitor import snapshot as network_snapshot
from app.ollama_client import list_local_models
from app.speech.stt import transcribe as transcribe_audio
from app.task_stream import start_task, stream_task
from app.tasks.document import DELIVERABLES_DIR, run_document_task

app = FastAPI(title="SAGE backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class SetActiveModelRequest(BaseModel):
    candidate_key: str


class TranslateRequest(BaseModel):
    text: str
    target_language: str


class DeliverableFilenamesRequest(BaseModel):
    filenames: list[str]


@app.get("/models")
def list_models():
    """The full model registry: every task type, its declared candidates,
    which one is active, and whether each candidate is actually pulled on
    this machine right now — the data the Model Registry UI renders and
    switches from."""
    local_models = list_local_models()
    task_types = model_config.get_config().task_types
    return {
        task_type: {
            "active": cfg.active,
            "candidates": {
                key: {
                    **candidate.model_dump(),
                    "locally_available": candidate.model_id in local_models,
                }
                for key, candidate in cfg.candidates.items()
            },
        }
        for task_type, cfg in task_types.items()
    }


@app.post("/models/{task_type}/active")
def set_active_model(task_type: str, body: SetActiveModelRequest):
    try:
        candidate = model_config.set_active_model(task_type, body.candidate_key)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    return {"task_type": task_type, "active": body.candidate_key, "candidate": candidate.model_dump()}


@app.get("/system/load")
def get_system_load_endpoint():
    """Phase 11: the live signal app/router.py's load-aware routing acts on
    (see app/load_monitor.py for what "load" means on this hardware and
    why), plus a recent log of any fallbacks it's actually taken — the
    Model Registry view polls this to show current pressure and whether
    routing has quietly degraded anything."""
    load = get_system_load()
    return {
        "level": load.level,
        "available_percent": load.available_percent,
        "loaded_models": [
            {"model_id": m.model_id, "size_bytes": m.size_bytes, "size_vram_bytes": m.size_vram_bytes}
            for m in load.loaded_models
        ],
        "fallback_log": [
            {
                "task_type": e.task_type,
                "from_model_id": e.from_model_id,
                "to_model_id": e.to_model_id,
                "reason": e.reason,
                "timestamp": e.timestamp,
            }
            for e in reversed(get_fallback_log())
        ],
    }


@app.get("/system/network")
def get_system_network_endpoint():
    """Phase 12: zero-egress proof — see app/network_monitor.py for what
    "external" means and which processes are actually watched. The Network
    Monitor view polls this to render the live "N external calls"
    indicator plus a running log of anything ever caught."""
    snap = network_snapshot()
    return {
        "external_count": snap.external_count,
        "external_connections": [
            {"pid": c.pid, "process": c.process, "remote": c.remote, "timestamp": c.timestamp}
            for c in snap.external_connections
        ],
        "local_connection_count": snap.local_connection_count,
        "watched_processes": snap.watched_processes,
        "external_log": [
            {"pid": c.pid, "process": c.process, "remote": c.remote, "timestamp": c.timestamp}
            for c in reversed(snap.external_log)
        ],
    }


@app.post("/agent/chat")
async def create_agent_chat(
    prompt: str = Form(...),
    file: UploadFile | None = File(None),
    audio: UploadFile | None = File(None),
    document: UploadFile | None = File(None),
    conversation_id: str | None = Form(None),
    output_language: str | None = Form(None),
    use_knowledge_base: bool = Form(True),
):
    """The Console's single entry point: one free-form prompt, one optional
    attached image, audio clip, and/or text document, no task-type picker.
    The agent (app/agent.py) decides on its own which tool(s) the request
    needs. The prompt's language is detected first (Phase 8): a supported
    non-English prompt is transparently translated to/from English around
    the (unchanged) agent — see app/language/pipeline.py. Returns a task_id
    immediately; watch it at GET /agent/chat/{task_id}/stream.

    `conversation_id` is the frontend's per-chat id (one per "New chat"),
    passed straight through as the LangGraph thread_id (app/agent.py) so a
    follow-up like "convert it into a word doc" can see and act on what the
    agent said earlier in the same chat rather than starting cold. Omitted
    entirely, each request is its own memoryless turn, same as before this
    existed — direct API callers aren't required to adopt it.

    `output_language` (one of app.language.translate.SUPPORTED_LANGUAGES, or
    "en") requests the answer be translated into that language regardless of
    what language the prompt itself was written in — omitted, the answer
    mirrors the input language exactly as before this existed (English in,
    English out; a supported non-English language in, the same language
    back).

    `use_knowledge_base` (default True): the acceptance checklist's
    "Knowledge base grounding: turn it off, confirm the answer changes" —
    False removes the knowledge-base tools from the model entirely for
    this turn, so an answer that would otherwise cite the SOP corpus comes
    from the model's own general knowledge instead."""
    image_b64 = None
    if file is not None:
        image_bytes = await file.read()
        image_b64 = base64.b64encode(image_bytes).decode()

    audio_path = None
    if audio is not None:
        suffix = os.path.splitext(audio.filename or "")[1] or ".wav"
        fd, audio_path = tempfile.mkstemp(suffix=suffix)
        with os.fdopen(fd, "wb") as f:
            f.write(await audio.read())

    document_text = None
    document_page_images = None
    if document is not None:
        fd, doc_path = tempfile.mkstemp(suffix=os.path.splitext(document.filename or "")[1])
        try:
            with os.fdopen(fd, "wb") as f:
                f.write(await document.read())
            document_text = extract_text(doc_path, document.filename or "")
            # pymupdf4llm's own OCR pass (extract_text, above) picks up any
            # text *inside* a PDF's embedded figures, but has no shape/symbol
            # understanding at all — a P&ID's bowtie-valve symbols become
            # unrecognizable character soup, just tags like "PT-2203" read
            # correctly. Rendering each page as a real image lets the vision
            # model (read_document_page_as_image, app/agent.py) actually see
            # and classify what a figure shows, not just OCR its text.
            if os.path.splitext(document.filename or "")[1].lower() == ".pdf":
                page_images = render_pdf_pages(doc_path)
                document_page_images = [base64.b64encode(img).decode() for img in page_images]
        except UnsupportedDocumentType as e:
            raise HTTPException(status_code=400, detail=str(e))
        finally:
            os.remove(doc_path)

    def work(emit):
        try:
            run_agent_multilingual(
                prompt, image_b64, emit, audio_path, document_text, conversation_id, output_language, use_knowledge_base, document_page_images
            )
        finally:
            if audio_path:
                os.remove(audio_path)

    task_id = start_task(work)
    return {"task_id": task_id}


@app.get("/agent/chat/{task_id}/stream")
def stream_agent_chat(task_id: str):
    return StreamingResponse(stream_task(task_id), media_type="text/event-stream")


@app.post("/translate")
def translate_text(body: TranslateRequest):
    """Directly translates already-generated English text into
    `target_language` — the Console's language toggle bar uses this to
    re-translate every answer already on screen the instant you click a
    different language, rather than only affecting the next message you
    send. Synchronous: IndicTrans2 inference on one short answer is fast
    enough not to need the task_id/SSE machinery every other endpoint with
    real background work uses."""
    if body.target_language not in SUPPORTED_LANGUAGES:
        raise HTTPException(status_code=400, detail=f"Unsupported target_language '{body.target_language}'. Expected one of: {sorted(SUPPORTED_LANGUAGES)}")
    try:
        translated = translate_from_english(body.text, body.target_language)
    except Exception as e:
        raise HTTPException(status_code=502, detail=str(e))
    return {"translated": translated}


@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...)):
    """Standalone speech-to-text (app/speech/stt.py, faster-whisper — local,
    on-device): the Console's mic button and unified upload button both use
    this to turn spoken audio directly into text in the prompt box before
    sending, rather than attaching the recording itself as a chat attachment.
    Synchronous (a single short clip, nothing worth streaming segment-by-
    segment for this use) — returns the transcript, detected language, and
    its confidence."""
    suffix = os.path.splitext(file.filename or "")[1] or ".wav"
    fd, audio_path = tempfile.mkstemp(suffix=suffix)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(await file.read())
        result = transcribe_audio(audio_path)
    finally:
        os.remove(audio_path)
    return {"text": result["text"], "language": result["language"], "language_probability": result["language_probability"]}


@app.get("/knowledge/search")
def knowledge_search(
    query: str,
    target_section: str | None = None,
    target_source: str | None = None,
    top_k: int = 4,
):
    """Runs the Phase 3 hybrid-RAG pipeline (retrieve -> rerank -> dedup ->
    generate -> verify citations -> flag confidence) against the ingested
    SOP corpus (scripts/ingest_knowledge.py)."""
    return run_query(query, target_section=target_section, target_source=target_source, top_k=top_k)


@app.get("/knowledge/graph")
def knowledge_graph():
    """The actual ingested corpus structure — documents, their sections,
    chunk counts — for the Knowledge Base view's graph panel. Not a search
    result: this is what's *in* the knowledge base, rendered once and
    highlighted against afterward as searches come in."""
    return build_corpus_graph()


@app.post("/tasks/document")
async def create_document_task(file: UploadFile = File(...)):
    """Starts the Phase 4 agentic document task (OCR -> grounded draft ->
    reviewer/critic -> .docx) on a scanned/photographed inspection report
    image. Returns a task_id immediately; watch its progress at
    GET /tasks/document/{task_id}/stream.

    NOTE: kept as a standalone endpoint deliberately, unlike the other
    per-task-type endpoints removed alongside this one (they were fully
    redundant with equivalent app.agent tools the Console already calls).
    This one is different: it's the only place the Phase 4 reviewer/critic
    loop (a hard, bounded re-check against SOPs before finalizing) actually
    runs — the Console's free-form agent does the same job today by
    chaining read_uploaded_image + search_knowledge_base +
    draft_docx_approval_note itself, prompted rather than mechanically
    enforced. The Console UI doesn't call this endpoint, so that stronger
    path isn't reachable from the product right now; flagged rather than
    silently deleted or silently left as-is."""
    image_bytes = await file.read()
    image_b64 = base64.b64encode(image_bytes).decode()
    task_id = start_task(lambda emit: run_document_task(image_b64, emit))
    return {"task_id": task_id}


@app.get("/tasks/document/{task_id}/stream")
def stream_document_task(task_id: str):
    return StreamingResponse(stream_task(task_id), media_type="text/event-stream")


# MS Office file types the agent can produce (docx: app.agent's
# draft_docx_approval_note / Phase 4's finalize_node; xlsx/pptx: the new
# export_to_excel/export_to_powerpoint tools). Both list_deliverables and
# download_deliverable need this — the former to not silently hide non-docx
# files (this was ".docx"-only until the xlsx/pptx writers existed, which
# would have made them invisible in the Deliverables view despite saving
# correctly), the latter to serve the right Content-Type instead of
# mislabeling every file as a Word document.
_DELIVERABLE_MEDIA_TYPES = {
    ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    ".pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
    # Phase 9 follow-up: annotate_pid_drawing (app/tasks/pid_annotate.py)
    # saves a real annotated image, not an Office file — same deliverable
    # list/download path, just a different media type.
    ".png": "image/png",
}


@app.get("/deliverables")
def list_deliverables():
    """Every finished .docx/.xlsx/.pptx the agent has produced — the
    Deliverables view's (Phase 6) data source."""
    if not os.path.isdir(DELIVERABLES_DIR):
        return []
    entries = []
    for filename in sorted(os.listdir(DELIVERABLES_DIR)):
        if os.path.splitext(filename)[1].lower() not in _DELIVERABLE_MEDIA_TYPES:
            continue
        path = os.path.join(DELIVERABLES_DIR, filename)
        stat = os.stat(path)
        entries.append({"filename": filename, "size_bytes": stat.st_size, "created": stat.st_mtime})
    return sorted(entries, key=lambda e: e["created"], reverse=True)


@app.get("/deliverables/{filename}")
def download_deliverable(filename: str):
    path = os.path.join(DELIVERABLES_DIR, filename)
    if "/" in filename or not os.path.isfile(path):
        raise HTTPException(status_code=404, detail="No such deliverable")
    ext = os.path.splitext(filename)[1].lower()
    media_type = _DELIVERABLE_MEDIA_TYPES.get(ext, "application/octet-stream")
    return FileResponse(path, media_type=media_type, filename=filename)


def _safe_deliverable_path(filename: str) -> str:
    """Rejects any filename that isn't a plain name inside DELIVERABLES_DIR
    — same guard as download_deliverable's `"/" in filename` check, shared
    here since the batch endpoints below (Phase 13) touch multiple
    filenames from one request body instead of a single URL path segment."""
    path = os.path.join(DELIVERABLES_DIR, filename)
    if "/" in filename or ".." in filename or not os.path.isfile(path):
        raise HTTPException(status_code=404, detail=f"No such deliverable: {filename}")
    return path


@app.post("/deliverables/export")
def export_deliverables(body: DeliverableFilenamesRequest):
    """Phase 13: bulk export — zips the selected deliverables in-memory
    (these are individual small office documents, not large media, so
    holding the zip in memory rather than streaming to a temp file is a
    reasonable trade for simplicity) and returns it as one download."""
    if not body.filenames:
        raise HTTPException(status_code=400, detail="No filenames given")
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        for filename in body.filenames:
            path = _safe_deliverable_path(filename)
            zf.write(path, arcname=filename)
    buffer.seek(0)
    return StreamingResponse(
        buffer,
        media_type="application/zip",
        headers={"Content-Disposition": "attachment; filename=deliverables.zip"},
    )


@app.post("/deliverables/delete")
def delete_deliverables(body: DeliverableFilenamesRequest):
    """Phase 13: bulk delete — validates every filename before deleting any
    of them, so a single bad name in the batch fails the whole request
    instead of silently deleting some and skipping others."""
    if not body.filenames:
        raise HTTPException(status_code=400, detail="No filenames given")
    paths = [_safe_deliverable_path(filename) for filename in body.filenames]
    for path in paths:
        os.remove(path)
    return {"deleted": len(paths)}
