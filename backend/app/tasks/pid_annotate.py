"""P&ID symbol annotation: draws the vision model's own symbol reads back
onto a copy of the actual drawing (bounding box + tag + category, color-
coded to a legend baked into the image itself) and saves it as a real,
downloadable file — so a human can visually check the read against the
real drawing instead of trusting a text description on faith. This is
exactly what app/agent.py's existing `read_pid_drawing` tool doesn't give
you: a description with no way to verify *where* each symbol was found.

**Not a trained detector — flagged, not hidden.** A real YOLO-style model
needs labeled P&ID training data this project doesn't have (same
conclusion Phase 9 already reached for symbol *classification*). What's
built here instead: the same vision-language model already used for
reading the drawing is asked to report each symbol's *location* too (as a
bounding box, in percentages of image width/height so it's resolution-
independent), not just its type. That's a real, useful capability — but a
VLM's spatial grounding is measurably less precise than a purpose-trained
detector, which is exactly why the output here is a file for a human to
check, not a claim of ground truth.
"""

import io
import json
import re
import uuid
from pathlib import Path

from langchain_core.messages import HumanMessage
from PIL import Image, ImageDraw, ImageFont

from app.tasks.document import DELIVERABLES_DIR

# One fixed color per legend category (app/agent.py's read_pid_drawing
# documents the same five categories) — used for both the box drawn around
# a detected symbol and its row in the legend strip, so the mapping between
# the two is immediate rather than requiring a lookup.
CATEGORY_COLORS: dict[str, tuple[int, int, int]] = {
    "gate valve": (37, 99, 235),  # blue
    "control valve": (16, 150, 72),  # green
    "pressure safety valve": (217, 45, 32),  # red
    "instrument bubble": (147, 51, 191),  # purple
    "process line": (90, 98, 110),  # gray
    "instrument signal line": (150, 156, 166),  # light gray
    "unknown": (200, 140, 20),  # amber — model reported something outside the fixed legend
}

DETECTION_PROMPT = (
    "This is an excerpt from a P&ID (piping & instrumentation diagram). Identify every "
    "tagged symbol using ONLY this legend — do not guess at conventions outside it:\n"
    "- A 'bowtie' (two triangles meeting at a point) with NOTHING else attached = a manual "
    "GATE VALVE.\n"
    "- A bowtie with a small box/rectangle on a stem above it = a CONTROL VALVE (has a "
    "powered actuator).\n"
    "- A bowtie with a diagonal line ending in an arrow/flag off to the side = a PRESSURE "
    "SAFETY/RELIEF VALVE.\n"
    "- An open circle with letters inside (e.g. 'PT', 'FT', 'TT') = an INSTRUMENT BUBBLE; the "
    "tag below it is its full loop number (e.g. 'PT-2203').\n"
    "- A thick solid line = a PROCESS LINE; a thinner line off of it to an instrument bubble = "
    "an INSTRUMENT SIGNAL LINE, not process piping.\n\n"
    "Report every tagged symbol AND its approximate location in the image, as strict JSON — "
    "no markdown fences, no commentary before or after, just the JSON array. Example output "
    "format (this is only a format example, not real data — do not copy these values):\n"
    '[{"tag": "EXAMPLE-1", "category": "gate valve", "description": "bowtie only, no actuator '
    'box attached", "bbox_pct": [42, 18, 49, 24]}]\n\n'
    "Field meanings:\n"
    "- tag: the symbol's own tag (e.g. FV-2202), or a short label if untagged.\n"
    "- category: one of gate valve, control valve, pressure safety valve, instrument bubble, "
    "process line, instrument signal line.\n"
    "- description: one line describing exactly what's drawn.\n"
    "- bbox_pct: [x1, y1, x2, y2], the symbol's bounding box as PERCENTAGES of the full image "
    "width/height — every value from 0 to 100, x1<x2, y1<y2. Not pixel coordinates. Estimate "
    "generously around the symbol itself, not the whole surrounding area. Every entry MUST "
    "include a real bbox_pct estimate in this 0-100 percentage form — never omit it and never "
    "give raw pixel coordinates.\n"
    "Don't infer a category from a tag name (e.g. a tag starting with FV doesn't mean assume "
    "control valve) — report only what the symbol actually shows."
)

_JSON_ARRAY_RE = re.compile(r"\[.*\]", re.DOTALL)


class PidAnnotationError(Exception):
    pass


def _parse_symbols(raw: str) -> list[dict]:
    match = _JSON_ARRAY_RE.search(raw)
    if not match:
        raise PidAnnotationError(f"Model response didn't contain a JSON array to parse:\n{raw[:400]}")
    try:
        parsed = json.loads(match.group(0))
    except json.JSONDecodeError as exc:
        raise PidAnnotationError(f"Model's JSON was malformed: {exc}\n{match.group(0)[:400]}") from exc
    if not isinstance(parsed, list):
        raise PidAnnotationError("Model's JSON was valid but not a list of symbols.")
    return parsed


def detect_pid_symbols(model, image_b64: str) -> list[dict]:
    message = HumanMessage(
        content=[
            {"type": "text", "text": DETECTION_PROMPT},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_b64}"}},
        ]
    )
    response = model.invoke([message])
    return _parse_symbols(str(response.content))


def _draw_legend(draw: ImageDraw.ImageDraw, x: int, y: int, font: ImageFont.FreeTypeFont, categories_present: set[str]) -> None:
    draw.text((x, y), "Legend", fill=(20, 20, 20), font=font)
    y += 22
    for category, color in CATEGORY_COLORS.items():
        if category == "unknown" and "unknown" not in categories_present:
            continue
        if category != "unknown" and category not in categories_present:
            continue
        draw.rectangle([x, y, x + 16, y + 16], outline=color, width=3)
        draw.text((x + 24, y), category, fill=(20, 20, 20), font=font)
        y += 22


def _has_valid_bbox(symbol: dict) -> bool:
    """`bbox_pct` is only trustworthy as *percentages of the actual image*
    if every value is actually in [0, 100] with x1<x2, y1<y2 — caught live
    comparing model output against a known 1000x500 test image: the local
    qwen2.5-vl:7b ignored the percentage instruction entirely and reported
    values in its own internal resize space instead (y-values over 500 on
    a 500px-tall image is not a rounding error, it's a different coordinate
    system this code has no reliable way to invert — Qwen-VL's own
    "smart resize" preprocessing dimensions aren't exposed through the
    Ollama API this project talks to). Silently guessing a scale factor
    risks drawing a *confidently wrong* box, which is worse for a feature
    whose whole purpose is human verification than drawing no box at all —
    so this rejects anything out of range rather than trying to normalize it.
    """
    bbox = symbol.get("bbox_pct")
    if not bbox or len(bbox) != 4:
        return False
    x1, y1, x2, y2 = bbox
    return all(0 <= v <= 100 for v in bbox) and x1 < x2 and y1 < y2


def render_annotated_image(image_bytes: bytes, symbols: list[dict]) -> tuple[Image.Image, int]:
    """Returns (image, boxes_drawn) — the caller uses boxes_drawn to decide
    whether to warn that localization wasn't available for this model."""
    base = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    width, height = base.size
    font_size = max(12, width // 60)
    font = ImageFont.load_default(size=font_size)

    # A legend strip appended below the drawing, not overlaid on top of
    # it — overlaying risks covering exactly the symbols being verified.
    legend_categories = {s.get("category", "unknown") if s.get("category") in CATEGORY_COLORS else "unknown" for s in symbols}
    legend_height = 30 + 22 * max(len(legend_categories), 1)
    canvas = Image.new("RGB", (width, height + legend_height), (255, 255, 255))
    canvas.paste(base, (0, 0))
    draw = ImageDraw.Draw(canvas)

    boxes_drawn = 0
    for symbol in symbols:
        if not _has_valid_bbox(symbol):
            continue
        x1, y1, x2, y2 = symbol["bbox_pct"]
        box = (x1 / 100 * width, y1 / 100 * height, x2 / 100 * width, y2 / 100 * height)
        category = symbol.get("category") if symbol.get("category") in CATEGORY_COLORS else "unknown"
        color = CATEGORY_COLORS[category]
        draw.rectangle(box, outline=color, width=3)
        label = symbol.get("tag", "?")
        text_y = max(box[1] - font_size - 4, 0)
        draw.rectangle([box[0], text_y, box[0] + font.getlength(label) + 6, text_y + font_size + 4], fill=color)
        draw.text((box[0] + 3, text_y + 1), label, fill=(255, 255, 255), font=font)
        boxes_drawn += 1

    _draw_legend(draw, 12, height + 10, font, legend_categories)
    return canvas, boxes_drawn


def annotate_pid(model, image_b64: str, image_bytes: bytes) -> tuple[str, list[dict], int]:
    """Runs detection + drawing + save, returns (filename, symbols,
    boxes_drawn) — the filename is a real file in DELIVERABLES_DIR,
    immediately downloadable via the existing GET /deliverables/{filename}
    endpoint. boxes_drawn < len(symbols) means the active vision model
    didn't return usable locations for every symbol it found (see
    _has_valid_bbox) — the caller surfaces this honestly rather than
    silently shipping a legend with no boxes on it."""
    symbols = detect_pid_symbols(model, image_b64)
    annotated, boxes_drawn = render_annotated_image(image_bytes, symbols)

    Path(DELIVERABLES_DIR).mkdir(parents=True, exist_ok=True)
    filename = f"pid-annotated-{uuid.uuid4().hex[:8]}.png"
    annotated.save(Path(DELIVERABLES_DIR) / filename)
    return filename, symbols, boxes_drawn
