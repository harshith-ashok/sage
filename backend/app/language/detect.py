"""Phase 8: language ID on incoming text queries via fastText's lid.176
model — fully local after the one-time download
(scripts/download_lid_model.py), no runtime network calls. Used by
app/language/pipeline.py to decide whether a Console prompt needs
translation before reaching the (English-internal) agent.
"""

import os

import fasttext
import numpy as np

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "data", "models", "lid.176.bin")

_model = None


class LanguageDetectionUnavailable(Exception):
    """Raised when the lid.176 model hasn't been downloaded yet."""


def _get_model():
    global _model
    if _model is None:
        if not os.path.isfile(MODEL_PATH):
            raise LanguageDetectionUnavailable(
                f"Language ID model not found at {MODEL_PATH} — run "
                "`uv run python -m scripts.download_lid_model` once first."
            )
        fasttext.FastText.eprint = lambda *args, **kwargs: None  # suppress a harmless load-time warning
        _model = fasttext.load_model(MODEL_PATH)
    return _model


def detect_language(text: str) -> dict:
    """Returns {"language": ISO 639-1 code, "confidence": 0-1}. fastText
    expects single-line input, so newlines are flattened first.

    Calls the underlying `model.f.predict()` binding directly instead of
    `model.predict()` — the installed fasttext-wheel's Python wrapper does
    `np.array(probs, copy=False)` on a plain tuple, which numpy>=2.0 raises
    on (that exact call now requires a possible copy); duplicating the
    wrapper's small amount of logic here with `np.asarray` avoids depending
    on a fix to the third-party package.
    """
    model = _get_model()
    cleaned = " ".join(text.split())
    if not cleaned:
        return {"language": "en", "confidence": 0.0}
    predictions = model.f.predict(cleaned + "\n", 1, 0.0, "strict")
    if not predictions:
        return {"language": "en", "confidence": 0.0}
    probs, labels = zip(*predictions)
    language = labels[0].replace("__label__", "")
    return {"language": language, "confidence": float(np.asarray(probs)[0])}
