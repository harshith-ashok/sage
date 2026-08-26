"""Renders LaTeX math to a small transparent PNG via matplotlib's built-in
mathtext engine — no system LaTeX/TeX distribution needed (matplotlib ships
its own math renderer), works fully offline, no runtime network call.

Used by docx_writer.py so an equation in the model's Markdown output (e.g.
"$$E = mc^2$$" or "\\[...\\]") becomes an actual rendered image in the
generated .docx instead of showing up as raw, unrendered LaTeX source —
mirroring what the frontend already does with KaTeX for the Console, since
the model writes the same LaTeX either way.

Not full LaTeX: mathtext supports a large common subset (fractions, roots,
sub/superscripts, greek letters, sums/integrals, \\text{}, comparison
operators, ...) but not the complete macro system — confirmed live that
\\boxed{} isn't supported (unwrapped below rather than failing) and there
will be other gaps. Callers must treat render_latex_png as fallible
(LatexRenderError) and degrade to showing the raw LaTeX text rather than
letting one bad equation abort the whole document.
"""

import io
import re

import matplotlib

matplotlib.use("Agg")  # headless — no display server needed/available here
import matplotlib.pyplot as plt

DPI = 220


class LatexRenderError(Exception):
    """Raised when mathtext can't parse the given LaTeX."""


def _unwrap_boxed(latex: str) -> str:
    """\\boxed{...} isn't a mathtext command; the box itself is purely
    decorative, so its content is kept and the wrapper dropped rather than
    failing the whole equation over it. Brace-matched by hand (not regex)
    since the content commonly contains its own nested braces, e.g.
    \\boxed{\\text{rate} \\approx 0.02}."""
    marker = "\\boxed{"
    out: list[str] = []
    i = 0
    while True:
        idx = latex.find(marker, i)
        if idx == -1:
            out.append(latex[i:])
            return "".join(out)
        out.append(latex[i:idx])
        depth = 1
        j = idx + len(marker)
        while j < len(latex) and depth > 0:
            if latex[j] == "{":
                depth += 1
            elif latex[j] == "}":
                depth -= 1
            j += 1
        out.append(latex[idx + len(marker) : j - 1])
        i = j


# Pure style/sizing directives with no rendering effect mathtext needs to
# know about — safe to strip outright rather than fail the equation, unlike
# an unknown *content* command (which genuinely can't be dropped without
# changing the math). Confirmed live: the model reaches for \displaystyle
# fairly often (to force a fraction/sum to render at full size inline).
_STRIP_STYLE_COMMANDS_RE = re.compile(r"\\(?:displaystyle|textstyle|scriptstyle|scriptscriptstyle)\b\s*")

# mathtext only recognizes the long forms (\leq/\geq); confirmed live in a
# confidence-interval expression ("-0.020 \le \beta_1 \le -0.019") that the
# model reaches for the common shorthand instead, which otherwise fails the
# whole inequality. `\b` after the shorthand keeps this from matching inside
# \leq/\geq themselves (no word-boundary between "e" and "q").
_ALIAS_COMMANDS = {r"\\le\b": r"\leq", r"\\ge\b": r"\geq"}


def render_latex_png(latex: str, fontsize: int = 13) -> tuple[bytes, float]:
    """Renders `latex` (no surrounding $ delimiters) to a tightly-cropped
    transparent PNG. Returns (png_bytes, width/height aspect ratio) so the
    caller can size the inserted image proportionally instead of guessing."""
    latex = _unwrap_boxed(latex.strip())
    latex = _STRIP_STYLE_COMMANDS_RE.sub("", latex)
    for pattern, replacement in _ALIAS_COMMANDS.items():
        # A lambda, not the bare replacement string: re.sub treats a string
        # repl as its own escape-sequence mini-language (\1, \g<name>, ...),
        # which choked on the literal "\l" in "\leq" — a callable repl is
        # used verbatim instead, no escape processing.
        latex = re.sub(pattern, lambda _m, r=replacement: r, latex)
    if not latex:
        raise LatexRenderError("empty expression")
    fig = plt.figure(figsize=(0.01, 0.01))
    try:
        text_obj = fig.text(0, 0, f"${latex}$", fontsize=fontsize)
        fig.canvas.draw()
    except Exception as exc:
        plt.close(fig)
        raise LatexRenderError(str(exc)) from exc
    bbox = text_obj.get_window_extent()
    width_in = max(bbox.width / fig.dpi, 0.01)
    height_in = max(bbox.height / fig.dpi, 0.01)
    fig.set_size_inches(width_in, height_in)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=DPI, transparent=True, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue(), width_in / height_in
