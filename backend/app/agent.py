"""Unified agent orchestrator — the Console's single free-form entry point,
styled after Claude Code/Codex: one prompt (plus an optional attached
image), no task-type picker. The model itself decides which tool(s) a
request needs by calling them, the same "orchestrator picks tool + args
from natural-language request, model interprets the result, doesn't compute
it" principle Phase 7 names for the ML tool, generalized here to every
capability built so far (Phases 3/4/5): searching the knowledge base,
reading an attached image, running sandboxed code, calculating, and writing
finished Office files (.docx/.xlsx/.pptx).

Built on LangGraph's prebuilt ReAct agent (bind-tools loop: model call ->
run any requested tools -> feed results back -> repeat until the model
answers without calling anything) rather than a hand-rolled loop — it's a
well-tested implementation already in the installed dependency tree, not
something to reinvent.
"""

import base64
import os
import threading
import uuid
from typing import Callable

from docx import Document
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent

from app.complexity import classify_complexity
from app.scope_guard import REFUSAL_MESSAGE, is_in_scope
from app.knowledge.query import format_context, retrieve_context
from app.router import get_chat_model
from app.tasks.calculate import CalculationError
from app.tasks.calculate import calculate as calculate_fn
from app.tasks.code import run_code_task
from app.tasks.document import DELIVERABLES_DIR
from app.tasks.ml import MLToolError
from app.tasks.ml import fit_linear_regression as fit_linear_regression_fn
from app.tasks.ml import fit_logistic_regression as fit_logistic_regression_fn
from app.tasks.ml import kmeans_cluster as kmeans_cluster_fn
from app.tasks.ml import pca_reduce as pca_reduce_fn
from app.tasks.docx_writer import write_markdown
from app.tasks.excel_writer import write_excel
from app.tasks.pid_annotate import PidAnnotationError, annotate_pid
from app.tasks.pptx_writer import write_pptx
from app.speech.stt import transcribe as transcribe_audio_fn

MAX_DOCUMENT_CHARS = 20000

# One shared checkpointer for every conversation, keyed by thread_id (the
# frontend's per-chat conversation_id — see main.py's /agent/chat). This is
# what makes a follow-up like "convert it into a word doc" work: LangGraph
# reloads that thread's prior messages (including the agent's own previous
# answer) before adding the new one, so the model can see and act on what it
# already said instead of starting cold every request. In-memory only —
# scoped to this backend process, so it doesn't survive a restart (the
# frontend's own localStorage transcript is a separate, longer-lived copy of
# the same conversation for display, but a restarted backend can't resume
# reasoning over it — a follow-up after a restart starts a fresh thread).
_checkpointer = MemorySaver()

SYSTEM_PROMPT = (
    "You are SAGE, an air-gapped industrial/technical assistant. You have tools for "
    "searching the site's SOP knowledge base, reading an image the user attached, "
    "running Python code in a sandbox, doing exact calculations, and writing finished "
    "Office files — a .docx note (draft_docx_approval_note), a .xlsx spreadsheet for "
    "tabular data (export_to_excel), or a .pptx slide deck (export_to_powerpoint). "
    "Use a tool whenever it would give a more grounded or verified answer than "
    "reasoning alone:\n"
    "- Pick the Office format that actually matches the request: a document/note/report "
    "is docx, a data table/spreadsheet is xlsx, a presentation/slide summary is pptx — "
    "don't default to docx for everything just because it's the most common one.\n"
    "- Never compute arithmetic yourself when `calculate` is available.\n"
    "- Never state what an SOP/procedure requires from memory — search for it, and "
    "cite the specific (Page X, Section Y) each claim came from.\n"
    "- Never claim code works without actually running it via `run_sandboxed_code`.\n"
    "- If the user reports a new finding, measurement, or reading (a thickness, a pressure, "
    "a recorded interval, a compliance status) tied to specific equipment, call "
    "`check_cross_document_contradictions` before treating it as fine — it retrieves related "
    "SOPs and engineering correspondence on that equipment so you can check for a numeric or "
    "policy conflict, not just answer from the finding alone. If it finds one, mark that part "
    "of your answer with the literal text '⚠️ CONFLICT:' (the Console highlights this marker) "
    "and state exactly what conflicts and its source — do this in a drafted document too if "
    "you're writing one, as its own clearly labeled section.\n"
    "- Never eyeball or guess a statistical result (a trend, a correlation, a "
    "cluster grouping, a dimensionality reduction) — fit a real model with the "
    "appropriate ML tool (fit_linear_regression/fit_logistic_regression/kmeans_cluster/"
    "pca_reduce) and report its actual R²/accuracy/silhouette score, confidence interval, "
    "etc. Prefer these over writing your own fitting code in `run_sandboxed_code` — they "
    "already compute the standard statistics correctly (including confidence intervals "
    "for regression coefficients) and the sandbox otherwise has no way to install a "
    "stats package if it turns out to need one. If the user's request doesn't actually "
    "include usable numeric data to fit against, say so instead of calling an ML tool "
    "with fabricated numbers.\n"
    "- If the user attached an image and wants a formal note/approval document from "
    "it, read the image first, then search the knowledge base MULTIPLE times before "
    "drafting: once for the specific finding itself, and at least once more "
    "specifically for escalation, reporting-timeframe, or urgent-action requirements "
    "for that kind of finding — a single narrow search can retrieve the general "
    "inspection procedure while missing a separate escalation clause elsewhere in the "
    "same SOP. If a technician's own notes recommend an action, verify it against "
    "what the SOP actually requires rather than repeating it — do not approve a "
    "finding that should have been escalated. Only call `draft_docx_approval_note` "
    "once you've checked for escalation requirements specifically.\n"
    "- If the user attached a P&ID/process drawing (not a scanned form or photo), use "
    "`read_pid_drawing` instead of `read_uploaded_image` — it identifies symbols against a "
    "fixed legend instead of guessing. If a spec sheet/equipment list was also attached or is "
    "in the knowledge base, cross-reference each tagged symbol's actual drawn type against what "
    "the spec requires for that tag, and flag any mismatch explicitly — don't just describe the "
    "drawing and the spec side by side and leave the comparison to the user. If the user asks to "
    "see/verify/export/download the annotated drawing (not just a text description), use "
    "`annotate_pid_drawing` instead — it draws a labeled box around each symbol directly onto the "
    "image and saves it as a real file, so a human can visually check the read.\n"
    "- If the user attached audio, transcribe it first via `transcribe_audio` to find "
    "out what they actually said/asked before doing anything else with it.\n"
    "- If the user attached a document (PDF/DOCX/XLSX/PPTX/TXT/etc.), read it via "
    "`read_uploaded_document` before answering — don't ask them to paste the content.\n"
    "- Write your answers in Markdown (**bold**, `code`, fenced ```code blocks``` with a "
    "language tag, bullet lists) — it's rendered properly, not shown as raw text. For "
    "equations, formulas, or unit expressions with exponents/fractions/subscripts, write "
    "real LaTeX (either $...$/$$...$$ or \\(...\\)/\\[...\\] delimiters are fine) instead "
    "of plain-text math — it's rendered as a real equation, not shown as raw source.\n"
    "Briefly say what you're doing as you go, in plain language."
)


def _make_tools(
    image_b64: str | None,
    audio_path: str | None = None,
    document_text: str | None = None,
    emit: Callable[[str, dict], None] | None = None,
    include_knowledge_base: bool = True,
    document_page_images: list[str] | None = None,
) -> list:
    @tool
    def search_knowledge_base(query: str) -> str:
        """Search the site's SOP/procedure knowledge base. Returns the raw retrieved excerpts with their (Page, Section) location, NOT a pre-written answer — read them yourself and cite the specific ones you rely on. A single narrow query can miss a relevant clause elsewhere in the same document, so search again with different wording if the first result doesn't seem to cover the full picture (e.g. it mentions a procedure but not what to do if that procedure's finding is severe)."""
        candidates = retrieve_context(query, top_k=6)
        if not candidates:
            return "No relevant excerpts found in the knowledge base for this query."
        return format_context(candidates)

    @tool
    def check_cross_document_contradictions(finding: str) -> str:
        """Check a new finding/measurement/claim against the knowledge base for numeric or policy conflicts with related SOPs or engineering correspondence on the same equipment/topic. Pass a short, specific factual statement (e.g. "Vessel V-2201 current wall thickness reading: 9.8 mm" or "PSV-2201 recertification interval logged as 24 months") — retrieves the related excerpts (same tag/topic) for you to compare it against. Use this before finalizing any report, note, or answer that states a measurement, setpoint, interval, or compliance status, not only once something already looks suspicious — a real conflict is exactly the kind of thing that doesn't look suspicious until it's checked against the record."""
        candidates = retrieve_context(finding, top_k=6)
        if not candidates:
            return "No related SOP or correspondence found in the knowledge base to cross-check this finding against."
        return (
            "Related excerpts — read them and explicitly decide whether there's a numeric/policy "
            f"conflict with the finding (\"{finding}\") or it's consistent with the record. If there IS a "
            "conflict, start that part of your answer with the exact literal marker '⚠️ CONFLICT:' "
            "followed by what conflicts and the specific source it conflicts with — this marker is what "
            "makes the Console visually flag it, so don't paraphrase it away.\n\n" + format_context(candidates)
        )

    @tool
    def read_uploaded_image(instructions: str = "Transcribe everything in the image in full, preserving field/value pairs.") -> str:
        """Read the image the user attached to this message (a scan/photo of a document, form, or drawing). Only works if an image was actually attached this turn."""
        if not image_b64:
            return "No image was attached to this message."
        model = get_chat_model("vision")
        message = HumanMessage(
            content=[
                {"type": "text", "text": instructions},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_b64}"}},
            ]
        )
        return model.invoke([message]).content

    @tool
    def read_pid_drawing() -> str:
        """Read the P&ID (piping & instrumentation diagram) drawing the user attached to this message, identifying symbols against a fixed, documented legend rather than guessing freely — use this instead of the generic read_uploaded_image whenever the attached image is a P&ID/process drawing (valves, instrument bubbles, process lines), not a scanned form or a photo. Only works if an image was actually attached this turn."""
        if not image_b64:
            return "No image was attached to this message."
        model = get_chat_model("vision")
        instructions = (
            "This is an excerpt from a P&ID (piping & instrumentation diagram). Identify every "
            "tagged symbol using ONLY this legend — do not guess at conventions outside it:\n"
            "- A 'bowtie' (two triangles meeting at a point) with NOTHING else attached = a manual "
            "GATE VALVE.\n"
            "- A bowtie with a small box/rectangle on a stem above it = a CONTROL VALVE (has a "
            "powered actuator).\n"
            "- A bowtie with a diagonal line ending in an arrow/flag off to the side = a PRESSURE "
            "SAFETY/RELIEF VALVE (PSV).\n"
            "- An open circle with letters inside (e.g. 'PT', 'FT', 'TT') = a field-mounted "
            "instrument bubble; the tag below it is its full loop number (e.g. 'PT-2203').\n"
            "- A thick solid line = the process pipe; a thinner line off of it to an instrument "
            "bubble = an instrument signal line, not process piping.\n\n"
            "For each tagged symbol you find, report: the tag, which legend category it matches, "
            "and a one-line description of exactly what's drawn (e.g. 'bowtie only, no actuator "
            "box attached') so someone can verify your read without looking at the image again. "
            "List the process line's own line number/tag too. Don't infer what a tag 'should' be "
            "from its name (e.g. a tag starting with FV doesn't mean you should assume it's a "
            "control valve) — report only what the symbol actually shows."
        )
        message = HumanMessage(
            content=[
                {"type": "text", "text": instructions},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_b64}"}},
            ]
        )
        return model.invoke([message]).content

    @tool
    def annotate_pid_drawing() -> str:
        """Like read_pid_drawing, but draws the result back onto the actual P&ID image — a bounding box and tag around each identified symbol, color-coded to a legend printed on the image itself — and saves it as a real, downloadable file so a human can visually verify the read against the real drawing. Use this instead of read_pid_drawing whenever the user wants to see/check/export the annotated result, not just a text description. Not a trained detector (this project has no labeled P&ID training data) — it's the same vision model used elsewhere, asked to also report each symbol's location, so treat the boxes as a starting point for human review, not ground truth. Only works if an image was actually attached this turn."""
        if not image_b64:
            return "No image was attached to this message."
        model = get_chat_model("vision")
        try:
            image_bytes = base64.b64decode(image_b64)
            filename, symbols, boxes_drawn = annotate_pid(model, image_b64, image_bytes)
        except PidAnnotationError as exc:
            return f"Could not annotate the drawing: {exc}"
        summary = "\n".join(f"- {s.get('tag', '?')}: {s.get('category', 'unknown')} — {s.get('description', '')}" for s in symbols)
        warning = ""
        if boxes_drawn < len(symbols):
            missing = len(symbols) - boxes_drawn
            warning = (
                f"\n\nNote: the active vision model didn't return a usable location for "
                f"{missing} of {len(symbols)} symbol(s), so no box was drawn for those — only the "
                "legend and text list below cover them. Switching the vision candidate (Model "
                "Registry) to a model with reliable spatial grounding, e.g. the cloud default, "
                "gets real boxes for every symbol."
            )
        return f"Saved as {filename}\n\nDetected {len(symbols)} symbol(s):\n{summary}{warning}\n\nOpen the file to visually verify each box against the real drawing — this is a starting point for human review, not a certified read."

    @tool
    def run_sandboxed_code(task: str, expected_output: str | int | float | bool | list | None = None) -> str:
        """Write and run Python code in a network-isolated Docker sandbox (numpy/scipy pre-installed; no other packages, no network). Use for anything that needs real code execution, not a guessed answer. Pass expected_output when you know the exact correct stdout (e.g. a specific number or list) — it's verified as an exact match against the script's printed output and retried on mismatch. Leave it unset for open-ended/analytical code (e.g. fitting a model, exploring data) where there's no single correct printed string — it's instead just verified as having actually run without error."""
        # The model naturally reaches for a real Python value here (e.g. a
        # list for "the first 10 primes"), not always a string — caught
        # live: `expected_output=[2, 3, 5, ...]` crashed with a Pydantic
        # validation error before this function body ever ran, the same
        # class of bug as Phase 7's ML-tool flat-list issue. Verification
        # is always a stdout *string* match (run_code_task/verify_node), so
        # normalize here via str() — for a plain list/number this produces
        # exactly what Python's own print() would output (e.g.
        # "[2, 3, 5, 7]"), matching what a script doing `print(the_value)`
        # actually prints.
        expected_str = expected_output if expected_output is None or isinstance(expected_output, str) else str(expected_output)
        events: list[tuple[str, dict]] = []

        def _collect(event: str, data: dict) -> None:
            events.append((event, data))
            # Forward just the tier choice out to the outer trace — the
            # coding subtask's own generate/execute/verify steps are
            # already summarized in this tool's own return value, so only
            # the tiering decision (Phase 11) is worth surfacing separately.
            if event == "tier_selected" and emit is not None:
                emit(event, data)

        run_code_task(task, expected_str, _collect)
        done = next((data for event, data in reversed(events) if event == "done"), {})
        status = "PASSED" if done.get("passed") else "FAILED after retries"
        return f"{status}\n\n```python\n{done.get('code', '')}\n```\n\nOutput:\n```\n{done.get('stdout', '')}\n```"

    @tool
    def calculate(expression: str, variables: dict[str, float | list[float]] | None = None, convert_to_unit: str | None = None) -> str:
        """Exact calculation via sympy/numpy, with units tracked — use for ANY arithmetic, algebra, or unit conversion instead of computing it yourself. IMPORTANT: always parenthesize a multi-token denominator, e.g. "120 months / (12 months)" not "120 months / 12 months" — without parentheses this parses left-to-right as ordinary math notation does ((120 months / 12) * months), not as the ratio you likely mean."""
        try:
            result = calculate_fn(expression, variables=variables, convert_to_unit=convert_to_unit)
        except CalculationError as exc:
            return f"Calculation error: {exc}"
        return f"Result: {result['result']}\nSteps:\n" + "\n".join(result["steps"])

    @tool
    def draft_docx_approval_note(title: str, content: str) -> str:
        """Write finished note/report text to a real, downloadable .docx file (Deliverables). `content` is Markdown (**bold**, *italic*, "- " bullet lines, "# "/"## " headings) — it's rendered as real Word formatting, not literal asterisks. Only call this once the content is final — it is not reviewed after this."""
        os.makedirs(DELIVERABLES_DIR, exist_ok=True)
        filename = f"note-{uuid.uuid4().hex[:8]}.docx"
        path = os.path.join(DELIVERABLES_DIR, filename)
        doc = Document()
        doc.add_heading(title, level=1)
        write_markdown(doc, content)
        doc.save(path)
        return f"Saved as {filename} (visible in the Deliverables view)."

    @tool
    def export_to_excel(title: str, headers: list[str], rows: list[list[str | int | float]], sheet_name: str = "Sheet1") -> str:
        """Write tabular data to a real, downloadable .xlsx spreadsheet (Deliverables) — use whenever the user wants data/results as Excel or "a spreadsheet" rather than a Word document, e.g. inspection readings, calculation results, or a data table too large/numeric to be a good fit for a docx table. `headers` is the column header row; each entry in `rows` is one data row with the same number of values as `headers`, in the same order. Pass numbers as actual numbers (not strings) so the sheet is usable in Excel formulas/sorting, not just readable."""
        os.makedirs(DELIVERABLES_DIR, exist_ok=True)
        filename = f"sheet-{uuid.uuid4().hex[:8]}.xlsx"
        path = os.path.join(DELIVERABLES_DIR, filename)
        write_excel(path, sheet_name, headers, rows)
        return f"Saved as {filename} (visible in the Deliverables view)."

    @tool
    def export_to_powerpoint(title: str, subtitle: str, slides: list[dict]) -> str:
        """Write a real, downloadable .pptx slide deck (Deliverables) — use whenever the user wants a presentation/slide summary rather than a document. `slides` is a list of content slides after the title slide, each a dict with "heading" (str) and "bullets" (list of str) — keep bullets short, this is a slide not a report; use draft_docx_approval_note instead for anything that needs full paragraphs."""
        os.makedirs(DELIVERABLES_DIR, exist_ok=True)
        filename = f"deck-{uuid.uuid4().hex[:8]}.pptx"
        path = os.path.join(DELIVERABLES_DIR, filename)
        write_pptx(path, title, subtitle, slides)
        return f"Saved as {filename} (visible in the Deliverables view)."

    @tool
    def fit_linear_regression(X: list[float] | list[list[float]], y: list[float], confidence_level: float = 0.95) -> str:
        """Fit a real linear regression (scikit-learn/scipy) on numeric data — use whenever the user wants a numeric trend/relationship, coefficients, a confidence interval, or a prediction from data. Never estimate any of this yourself. X is a list of samples; for the common single-variable case (one x per data point, e.g. fitting y against time) just pass a flat list like [0, 3, 6, ...] — it's treated as one feature per sample automatically. For multiple features per sample, pass a list of lists instead, e.g. [[x1, x2], [x1, x2], ...]. Returns coefficients + a real confidence interval for each (standard OLS t-statistic), plus r2_full (fit quality on all the data) and r2_test (a held-out generalization check, NOT the same as r2_full — report both if asked how reliable the fit is)."""
        try:
            result = fit_linear_regression_fn(X, y, confidence_level=confidence_level)
        except MLToolError as exc:
            return f"Could not fit: {exc}"
        note = f" WARNING: {result['sanity_note']}" if not result["sanity_ok"] else ""
        ci = (
            f"{int(result['confidence_level'] * 100)}% CI={list(zip(result['coefficient_ci_lower'], result['coefficient_ci_upper']))}, "
            f"standard_error={result['coefficient_standard_error']}"
            if result["coefficient_ci_lower"] is not None
            else "CI unavailable (not enough degrees of freedom for this many features/samples)"
        )
        return (
            f"coefficients={result['coefficients']}, intercept={result['intercept']:.4f}\n"
            f"{ci}\n"
            f"r2_full={result['r2_full']:.4f} (fit on all {result['n_total']} points), "
            f"r2_train={result['r2_train']:.4f}, r2_test={result['r2_test']:.4f} "
            f"(n_train={result['n_train']}, n_test={result['n_test']}){note}"
        )

    @tool
    def fit_logistic_regression(X: list[float] | list[list[float]], y: list[float]) -> str:
        """Fit a real logistic regression classifier (scikit-learn) on labeled numeric data, held-out test split included — use whenever the user wants a classification/prediction from data. Never guess the accuracy yourself. X is a list of samples; a flat list like [0, 3, 6, ...] means one feature per sample, a list of lists means multiple features per sample."""
        try:
            result = fit_logistic_regression_fn(X, y)
        except MLToolError as exc:
            return f"Could not fit: {exc}"
        note = f" WARNING: {result['sanity_note']}" if not result["sanity_ok"] else ""
        return (
            f"accuracy_train={result['accuracy_train']:.4f}, accuracy_test={result['accuracy_test']:.4f} "
            f"(majority-class baseline={result['baseline_accuracy']:.4f}, n_train={result['n_train']}, "
            f"n_test={result['n_test']}){note}"
        )

    @tool
    def kmeans_cluster(X: list[float] | list[list[float]], n_clusters: int) -> str:
        """Cluster numeric rows into n_clusters groups with real k-means (scikit-learn), reporting a silhouette score. Use whenever the user wants to group/segment data. X is a list of samples; a flat list like [0, 3, 6, ...] means one feature per sample, a list of lists means multiple features per sample."""
        try:
            result = kmeans_cluster_fn(X, n_clusters)
        except MLToolError as exc:
            return f"Could not cluster: {exc}"
        note = f" WARNING: {result['sanity_note']}" if not result["sanity_ok"] else ""
        return f"labels={result['labels']}\nsilhouette_score={result['silhouette_score']:.4f}{note}"

    @tool
    def pca_reduce(X: list[float] | list[list[float]], n_components: int) -> str:
        """Reduce numeric data to n_components dimensions with real PCA (scikit-learn), reporting the actual variance retained. Use whenever the user wants to reduce dimensionality/find the main axes of variation in data. X is a list of samples, each a list of that sample's features (a flat list is treated as one feature per sample, but PCA only makes sense with multiple features)."""
        try:
            result = pca_reduce_fn(X, n_components)
        except MLToolError as exc:
            return f"Could not reduce: {exc}"
        note = f" WARNING: {result['sanity_note']}" if not result["sanity_ok"] else ""
        return (
            f"transformed={result['transformed']}\n"
            f"explained_variance_ratio={result['explained_variance_ratio']}\n"
            f"cumulative_explained_variance={result['cumulative_explained_variance']:.4f}{note}"
        )

    @tool
    def transcribe_audio() -> str:
        """Transcribe the audio the user attached to this message (local speech-to-text, auto-detects the spoken language). Only works if audio was actually attached this turn."""
        if not audio_path:
            return "No audio was attached to this message."
        result = transcribe_audio_fn(audio_path)
        return f"[detected language: {result['language']} ({result['language_probability']:.2f} confidence)]\n{result['text']}"

    @tool
    def read_uploaded_document() -> str:
        """Read the document (PDF/DOCX/XLSX/PPTX/TXT/MD/CSV/JSON/YAML) the user attached to this message — already extracted, returns the full text directly. Only works if a document was actually attached this turn."""
        if not document_text:
            return "No document was attached to this message."
        if len(document_text) > MAX_DOCUMENT_CHARS:
            return document_text[:MAX_DOCUMENT_CHARS] + f"\n\n[...truncated, {len(document_text) - MAX_DOCUMENT_CHARS} more characters not shown]"
        return document_text

    @tool
    def read_document_page_as_image(page_number: int = 1) -> str:
        """See an attached PDF's page as a real image, not just its extracted text — use this when a PDF might contain a figure/diagram/photo (a P&ID excerpt, a drawing, a scanned form) that read_uploaded_document's text extraction wouldn't meaningfully capture. read_uploaded_document DOES run OCR on embedded figures, but has no shape/symbol understanding at all — it can read a tag like "PT-2203" printed near a symbol, but can't tell a gate valve from a control valve from a pressure safety valve, or describe a photo. This tool sees the actual page image, the same way an attached image would be read. page_number is 1-indexed; only the PDF's first few pages are available this way. Only works if a PDF was actually attached this turn."""
        if not document_page_images:
            return "No PDF page images are available — either no document was attached this turn, or it wasn't a PDF."
        if page_number < 1 or page_number > len(document_page_images):
            return f"Page {page_number} isn't available — this PDF has {len(document_page_images)} page(s) rendered (its first {len(document_page_images)})."
        model = get_chat_model("vision")
        message = HumanMessage(
            content=[
                {"type": "text", "text": "Describe everything on this document page in full — both the text and any figure/diagram/photo, preserving field/value pairs and describing what any drawing actually shows rather than just noting one is present."},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{document_page_images[page_number - 1]}"}},
            ]
        )
        return model.invoke([message]).content

    # Acceptance checklist: "Knowledge base grounding: turn it off, confirm
    # the answer changes" — the tools simply aren't offered to the model at
    # all when disabled, not hidden behind a prompt instruction it could
    # ignore. Without them, an answer that depended on the SOP corpus comes
    # from the model's own general knowledge instead — a real, checkable
    # difference, not a cosmetic one.
    knowledge_base_tools = [search_knowledge_base, check_cross_document_contradictions] if include_knowledge_base else []

    return [
        *knowledge_base_tools,
        read_uploaded_image,
        read_pid_drawing,
        annotate_pid_drawing,
        run_sandboxed_code,
        calculate,
        draft_docx_approval_note,
        export_to_excel,
        export_to_powerpoint,
        fit_linear_regression,
        fit_logistic_regression,
        kmeans_cluster,
        pca_reduce,
        transcribe_audio,
        read_uploaded_document,
        read_document_page_as_image,
    ]


def _emit_update(node_name: str, update: dict, emit: Callable[[str, dict], None]) -> str:
    """Handles the "updates" stream (tool calls + each complete message).
    Returns the node's final text content, if it produced any (used by
    run_agent to track the overall answer for the `done` event)."""
    final_text = ""
    for msg in (update or {}).get("messages", []):
        if node_name == "tools":
            emit("tool_result", {"tool": getattr(msg, "name", ""), "content": msg.content})
            continue
        tool_calls = getattr(msg, "tool_calls", None) or []
        for call in tool_calls:
            emit("tool_call", {"tool": call["name"], "args": call["args"]})
        if msg.content:
            # The complete text for this message, once fully generated —
            # `token` events (from the "messages" stream, below) already
            # streamed it incrementally; this lets the frontend swap from
            # raw incremental text to a clean final markdown render.
            emit("message", {"id": msg.id, "content": msg.content})
            final_text = msg.content
    return final_text


# Per-thread_id locks — reported live as LangGraph's INVALID_CHAT_HISTORY
# error, recurring across unrelated tool calls. Root cause: create_react_agent
# re-validates the *entire* checkpointed thread history on every "agent" node
# call (_validate_chat_history in langgraph's chat_agent_executor.py), and
# the most plausible way an AIMessage tool_call ends up with no matching
# ToolMessage is two overlapping graph.stream() calls on the same thread_id
# (e.g. a second message sent before the first finished) interleaving writes
# to the shared MemorySaver checkpoint — one run's "agent" step (tool_call
# checkpointed) racing another run's steps, so the first run's "tools" step
# never resolves it. A lock per thread_id serializes runs on the same
# conversation without blocking unrelated conversations. Module-level dict is
# fine here — single backend process, never persisted, so it can't itself
# leak across restarts or processes.
_thread_locks: dict[str, threading.Lock] = {}
_thread_locks_guard = threading.Lock()


def _lock_for_thread(thread_id: str) -> threading.Lock:
    with _thread_locks_guard:
        lock = _thread_locks.get(thread_id)
        if lock is None:
            lock = threading.Lock()
            _thread_locks[thread_id] = lock
        return lock


def _repair_dangling_tool_calls(graph, config) -> bool:
    """Self-healing for a thread already corrupted (by the race above, or a
    backend crash/restart mid-turn) before this lock existed. Once
    _validate_chat_history rejects a thread, every future call on that
    thread_id fails identically forever — there's no way to "skip" the bad
    turn, since the checkpointed history is what's rejected, not the new
    message. Finds any tool_call with no matching ToolMessage anywhere in the
    thread and injects a synthetic one explaining what happened, via
    graph.update_state (the supported way to patch a checkpoint). Returns
    True if a repair was made, so the caller knows a retry is worth it."""
    snapshot = graph.get_state(config)
    messages = snapshot.values.get("messages", []) if snapshot.values else []
    resolved_ids = {msg.tool_call_id for msg in messages if isinstance(msg, ToolMessage)}
    dangling = [
        call
        for msg in messages
        if isinstance(msg, AIMessage)
        for call in msg.tool_calls
        if call["id"] not in resolved_ids
    ]
    if not dangling:
        return False
    graph.update_state(
        config,
        {
            "messages": [
                ToolMessage(
                    content=(
                        "This tool call was interrupted by a prior request that did not "
                        "finish and never produced a result. Disregard it; if it's still "
                        "relevant, call the tool again."
                    ),
                    tool_call_id=call["id"],
                    name=call.get("name", ""),
                )
                for call in dangling
            ]
        },
    )
    return True


def _stream_graph(graph, initial: dict, config: dict, emit: Callable[[str, dict], None]) -> str:
    final_content = ""
    for mode, payload in graph.stream(initial, config=config, stream_mode=["updates", "messages"]):
        if mode == "messages":
            chunk, metadata = payload
            if metadata.get("langgraph_node") == "agent":
                # Ollama's reasoning models (gpt-oss, qwen3, ...) emit a
                # separate "thinking" stream before any real content — this
                # is the fix for a genuinely long-running request (a
                # multi-part statistical/analysis prompt) looking completely
                # hung: without surfacing this, nothing streams at all for
                # however long the model spends reasoning.
                reasoning = chunk.additional_kwargs.get("reasoning_content")
                if reasoning:
                    emit("thinking", {"id": chunk.id, "content": reasoning})
                if chunk.content:
                    emit("token", {"id": chunk.id, "content": chunk.content})
        elif mode == "updates":
            for node_name, update in payload.items():
                text = _emit_update(node_name, update, emit)
                if node_name == "agent" and text:
                    final_content = text
    return final_content


def run_agent(
    prompt: str,
    image_b64: str | None,
    emit: Callable[[str, dict], None],
    audio_path: str | None = None,
    document_text: str | None = None,
    thread_id: str | None = None,
    use_knowledge_base: bool = True,
    document_page_images: list[str] | None = None,
) -> None:
    """Drives the agent via .stream() on two combined stream modes, so the
    reply streams token-by-token instead of waiting for each full message:
    `token` (incremental text, {id, content} — id groups chunks belonging to
    one message so the frontend can tell messages apart), `tool_call` (the
    model chose a tool), `tool_result` (it ran), `message` (a message's full
    text once complete, for the frontend to re-render as clean markdown),
    then `done` once the loop ends with no further tool calls.

    `thread_id` ties this call to a conversation in `_checkpointer`: passing
    the same id as a prior call resumes that conversation (the model sees
    its own earlier messages, not just this one), omitting it starts a
    stateless one-off turn with no memory, same as before this existed."""
    has_attachment = bool(image_b64 or audio_path or document_text)

    # Topic guardrail (app/scope_guard.py): a hard gate, not a prompt
    # instruction the model could just ignore — an out-of-scope request
    # never reaches create_react_agent, so no tool ever runs and nothing
    # gets generated for it. Checked before tiering/model warm-up on
    # purpose: if this is going to refuse, there's no reason to do that
    # work first. A refused turn is never added to `thread_id`'s LangGraph
    # message history (the graph is never invoked for it) — there's no
    # real reasoning trace to persist, just a canned refusal.
    if not is_in_scope(prompt, has_attachment):
        emit("scope_blocked", {})
        message_id = str(uuid.uuid4())
        emit("message", {"id": message_id, "content": REFUSAL_MESSAGE})
        emit("done", {"content": REFUSAL_MESSAGE})
        return

    # Phase 11: complexity-based tiering — a cheap heuristic on the prompt
    # (app/complexity.py), not an extra model call, picks "fast" vs
    # "strong" once per turn before the model is even resolved, so this
    # adds no latency of its own on top of the model call that was always
    # going to happen.
    tier = classify_complexity(prompt, has_attachment=has_attachment)
    model = get_chat_model("reasoning", tier=tier)
    emit("tier_selected", {"task_type": "reasoning", "tier": tier, "model_id": model.model})
    tools = _make_tools(
        image_b64, audio_path, document_text, emit=emit, include_knowledge_base=use_knowledge_base, document_page_images=document_page_images
    )
    graph = create_react_agent(model, tools, prompt=SYSTEM_PROMPT, checkpointer=_checkpointer)
    config = {"configurable": {"thread_id": thread_id or str(uuid.uuid4())}}

    # Ground truth about what's actually attached *this* turn, instead of
    # leaving the model to infer it from phrasing alone. Caught live: with a
    # neutral prompt ("here's a field log entry, take a look") the model
    # called read_uploaded_document — the wrong one of four attachment
    # tools now available (image/PID/document/audio) — got back "no
    # document attached", and gave up without ever trying read_uploaded_image,
    # even though an image genuinely was attached. Once conversation memory
    # (thread_id, above) exists, this compounds: a later turn correctly has
    # no image_b64 of its own, but by then the image was never read into the
    # conversation's history at all, so nothing in the whole thread ever
    # saw it. Only noted when something IS attached — the tools already
    # degrade gracefully ("No image was attached...") when nothing is, so a
    # note isn't needed to avoid confusion in the common no-attachment case.
    attached = []
    if image_b64:
        attached.append("an image (read_uploaded_image or read_pid_drawing)")
    if audio_path:
        attached.append("audio (transcribe_audio)")
    if document_text:
        note = "a document (read_uploaded_document"
        if document_page_images:
            note += ", or read_document_page_as_image if it might contain a figure/diagram/photo the text extraction wouldn't meaningfully capture"
        attached.append(note + ")")
    attachment_note = f"[Attached to this message: {', '.join(attached)}.]\n\n" if attached else ""

    initial = {"messages": [HumanMessage(attachment_note + prompt)]}

    # Serializes runs on this thread_id (see _lock_for_thread above) so a
    # second message on the same conversation can't interleave graph.stream()
    # calls against the same checkpoint. If the thread was already corrupted
    # (by that race before this lock existed, or a backend crash mid-turn),
    # self-heal once via _repair_dangling_tool_calls and retry — otherwise
    # this exact thread_id would fail identically on every future message.
    with _lock_for_thread(config["configurable"]["thread_id"]):
        try:
            final_content = _stream_graph(graph, initial, config, emit)
        except ValueError as exc:
            if "INVALID_CHAT_HISTORY" in str(exc) and _repair_dangling_tool_calls(graph, config):
                emit(
                    "thinking",
                    {
                        "id": str(uuid.uuid4()),
                        "content": "[Recovered from an interrupted previous turn — retrying.]\n",
                    },
                )
                final_content = _stream_graph(graph, initial, config, emit)
            else:
                raise

    emit("done", {"content": final_content})
