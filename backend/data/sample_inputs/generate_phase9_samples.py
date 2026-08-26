"""Generates Phase 9's two sample artifacts (kept as source for
reproducibility, same reason inspection-report-source.html was kept for
Phase 4's sample):

1. handwritten-note.png — a second, genuinely different artifact type from
   Phase 4's scanned inspection FORM (this is a free-form handwritten field
   note), for the "apply the vision pipeline to a second artifact type"
   acceptance item.
2. pid-drawing.png + pid-line-6hc2201-spec.md — a small synthetic P&ID
   drawing (three tagged symbols drawn from a documented, deliberately small
   legend: gate valve / control valve / pressure-safety valve / a field
   instrument bubble) plus a companion equipment spec sheet, with one
   symbol/spec mismatch deliberately planted (FV-2202 is drawn as a plain
   gate valve; the spec sheet calls for a control valve) to test whether
   cross-referencing the two actually catches it — the same "ground it,
   don't trust it" pattern as Phase 4's technician-note mismatch, applied to
   drawings instead of text.

Run from backend/: `uv run python data/sample_inputs/generate_phase9_samples.py`
"""

import os

from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(__file__)

FONT_DIR = "/System/Library/Fonts/Supplemental"
HAND_FONT = os.path.join(FONT_DIR, "Bradley Hand Bold.ttf")
SANS_FONT = os.path.join(FONT_DIR, "Arial.ttf")
MONO_FONT = os.path.join(FONT_DIR, "Courier New.ttf")


def _font(path: str, size: int) -> ImageFont.FreeTypeFont:
    try:
        return ImageFont.truetype(path, size)
    except OSError:
        return ImageFont.load_default(size=size)


# ---------------------------------------------------------------------------
# 1. Handwritten field note — a LOTO log entry with a real, catchable gap
#    against SOP-LOTO-400: Step 3.2 permits a tag alone (no lock) only when
#    the device genuinely can't be locked AND only with the Step 2.4
#    zero-energy verification also done and documented. The note's own
#    wording ("felt safe to me") is an informal visual impression, not the
#    documented actuation-based verification the SOP actually requires.
# ---------------------------------------------------------------------------
def generate_handwritten_note() -> None:
    w, h = 850, 1100
    img = Image.new("RGB", (w, h), "#fdfaf0")
    draw = ImageDraw.Draw(img)

    # Ruled paper lines
    for y in range(120, h - 60, 42):
        draw.line([(50, y), (w - 50, y)], fill="#cfd6e4", width=1)
    draw.line([(90, 0), (90, h)], fill="#e3a8a8", width=2)  # margin rule

    hand = _font(HAND_FONT, 30)
    hand_small = _font(HAND_FONT, 26)

    lines = [
        (150, "LOTO -- Pump P-204 discharge valve"),
        (234, "Valve handle wouldn't take my lock so I"),
        (276, "just tagged it & started work."),
        (360, "Line looked drained, felt safe to me."),
        (444, "Discharge press. gauge read 0 psi."),
        (528, "Job took about 40 min, no issues."),
        (612, ""),
        (654, "-- R. Tan, 8/24, day shift"),
    ]
    for y, text in lines:
        if text:
            draw.text((120, y), text, font=hand if y != 654 else hand_small, fill="#1a3a8a")

    draw.text((120, 60), "Maintenance Field Log", font=_font(SANS_FONT, 26), fill="#111")
    draw.text((120, 92), "(loose-leaf note, not the formal LOTO sheet)", font=_font(SANS_FONT, 15), fill="#555")

    out = os.path.join(HERE, "handwritten-note.png")
    img.save(out)
    print(f"wrote {out}")


# ---------------------------------------------------------------------------
# 2. P&ID drawing — three tagged symbols on one process line, drawn from a
#    small legend (documented in app/agent.py's read_pid_drawing tool
#    prompt, not on the image itself, so reading it is a real visual task
#    rather than the model just re-OCR'ing a legend):
#      - bowtie only            -> gate valve
#      - bowtie + actuator box  -> control valve
#      - bowtie + relief flag   -> pressure safety valve (PSV)
#      - open circle, tag below -> field-mounted instrument
#      - dashed line            -> instrument signal line (vs. solid process line)
# ---------------------------------------------------------------------------
def _bowtie(draw: ImageDraw.ImageDraw, cx: int, cy: int, size: int = 26) -> None:
    left = [(cx - size, cy - size // 2), (cx - size, cy + size // 2), (cx, cy)]
    right = [(cx + size, cy - size // 2), (cx + size, cy + size // 2), (cx, cy)]
    draw.polygon(left, outline="black", fill="white", width=3)
    draw.polygon(right, outline="black", fill="white", width=3)


def generate_pid_drawing() -> None:
    w, h = 1000, 500
    img = Image.new("RGB", (w, h), "white")
    draw = ImageDraw.Draw(img)
    sans = _font(SANS_FONT, 18)
    sans_bold = _font(SANS_FONT, 20)
    mono = _font(MONO_FONT, 16)

    draw.text((30, 20), "P&ID-14-B -- Line 6\"-HC-2201 (excerpt)", font=sans_bold, fill="black")

    line_y = 260
    draw.line([(60, line_y), (w - 60, line_y)], fill="black", width=5)  # process line
    draw.text((60, line_y + 12), '6"-HC-2201', font=mono, fill="black")

    # PSV-2201: pressure safety valve (bowtie + relief flag) -- matches spec
    psv_x = 220
    _bowtie(draw, psv_x, line_y, size=24)
    draw.line([(psv_x, line_y - 12), (psv_x + 35, line_y - 55)], fill="black", width=3)
    draw.polygon(
        [(psv_x + 35, line_y - 55), (psv_x + 55, line_y - 48), (psv_x + 45, line_y - 68)],
        fill="black",
    )
    draw.text((psv_x - 35, line_y + 40), "PSV-2201", font=mono, fill="black")

    # FV-2202: drawn as a plain gate valve (bowtie only) -- spec sheet calls
    # for a control valve with pneumatic actuator; this is the deliberate
    # mismatch to catch.
    fv_x = 500
    _bowtie(draw, fv_x, line_y, size=24)
    draw.text((fv_x - 30, line_y + 40), "FV-2202", font=mono, fill="black")

    # PT-2203: field-mounted pressure instrument (open circle, no bar) off a
    # dashed signal line -- matches spec.
    pt_x, pt_y = 760, 140
    draw.line([(pt_x, line_y), (pt_x, pt_y + 25)], fill="black", width=2)
    for y in range(line_y - 10, pt_y + 25, 12):
        draw.line([(pt_x, y), (pt_x, min(y + 6, line_y))], fill="black", width=2)  # dashed signal line
    r = 30
    draw.ellipse([pt_x - r, pt_y - r, pt_x + r, pt_y + r], outline="black", width=3, fill="white")
    draw.text((pt_x - 12, pt_y - 12), "PT", font=sans, fill="black")
    draw.text((pt_x - 30, pt_y + r + 8), "PT-2203", font=mono, fill="black")

    out = os.path.join(HERE, "pid-drawing.png")
    img.save(out)
    print(f"wrote {out}")


def generate_spec_sheet() -> None:
    content = """# Equipment Spec Sheet -- Line 6"-HC-2201 (P&ID-14-B)

**Document owner:** Process Engineering
**Applies to:** P&ID-14-B, Line 6"-HC-2201

| Tag | Equipment | Required type | Notes |
|-----|-----------|----------------|-------|
| PSV-2201 | Pressure safety valve | Spring-loaded relief valve | Set pressure per relief study RS-2201, separator overpressure protection |
| FV-2202 | Flow control valve | Control valve, pneumatic actuator, fail-closed on air loss | Modulates flow to downstream separator; must fail closed per HAZOP action item HZ-2201-07 |
| PT-2203 | Pressure transmitter | Field-mounted pressure transmitter | 4-20mA signal to DCS, alarm high at 18 barg |

Any deviation from the required type above (e.g. a manual valve installed
where a control valve is specified) must be raised to Process Engineering
before the line is returned to service -- a manual gate valve cannot provide
the fail-closed, remotely-modulated action the HAZOP action item requires.
"""
    out = os.path.join(HERE, "pid-line-6hc2201-spec.md")
    with open(out, "w") as f:
        f.write(content)
    print(f"wrote {out}")


if __name__ == "__main__":
    generate_handwritten_note()
    generate_pid_drawing()
    generate_spec_sheet()
