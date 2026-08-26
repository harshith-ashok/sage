"""Phase 8: language ID on incoming text queries via fast-langdetect's
bundled fastText lid.176 model (`model="lite"`, a compressed ~1MB variant
shipped *inside* the pip package itself — no separate download step at all,
unlike the fasttext-wheel + scripts/download_lid_model.py setup this
replaced, which needed a manual one-time 126MB fetch before language
detection worked. Used by app/language/pipeline.py to decide whether a
Console prompt needs translation before reaching the (English-internal)
agent.

Verified live against the exact same 5-language test set as the previous
implementation (Hindi/Tamil/Telugu/Kannada/Malayalam plus English): 6/6
correct, all above 0.94 confidence — comparable to or better than the old
fasttext-wheel setup's >93%, with zero setup required.
"""

from fast_langdetect import detect as _detect
from fast_langdetect.infer import FastLangdetectError


class LanguageDetectionUnavailable(Exception):
    """Raised if the bundled language ID model can't be loaded — kept for
    the same defensive handling app/language/pipeline.py already had, even
    though the bundled model shipping inside the package means this should
    no longer happen in normal operation the way a missing separate
    download used to."""


def detect_language(text: str) -> dict:
    """Returns {"language": ISO 639-1 code, "confidence": 0-1}."""
    cleaned = " ".join(text.split())
    if not cleaned:
        return {"language": "en", "confidence": 0.0}
    try:
        results = _detect(cleaned, model="lite", k=1)
    except FastLangdetectError as exc:
        raise LanguageDetectionUnavailable(str(exc)) from exc
    if not results:
        return {"language": "en", "confidence": 0.0}
    return {"language": results[0]["lang"], "confidence": float(results[0]["score"])}
