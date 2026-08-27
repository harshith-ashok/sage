"""Phase 8: transparent multilingual wrapper around the Console's agent.
Detects the prompt's language; a supported non-English Indic prompt gets
translated to English before reaching app.agent.run_agent (unchanged —
Phases 1-5/6 stay English-internal, per this phase's explicit requirement),
and the final answer gets translated back before being emitted.

English prompts — the overwhelming common case, and everything tested
through Phase 7 — never touch the translation path at all:
detect_language() is cheap (fastText, <10ms) and runs first; IndicTrans2
only loads lazily, on the first actual non-English request. This is also
why a non-authenticated IndicTrans2 (see app/language/translate.py) doesn't
block ordinary English usage — it only surfaces as an error on the specific
request that needed it.

`output_language`, added later per explicit request ("have the output also
be translatable into the regional languages"): originally the *only* way to
get a translated answer was to ask the question in that language — an
English question always got an English answer, with no way to request
"answer in Hindi" independently of what language you typed in. An explicit
`output_language` decouples the two: it always wins when set (including
"en", which means "answer in English" even if the input was Hindi — an
explicit request to suppress the mirror-the-input-language default below),
falling back to that mirroring default only when unset.
"""

from typing import Callable

from app.agent import run_agent
from app.language.detect import LanguageDetectionUnavailable, detect_language
from app.language.translate import SUPPORTED_LANGUAGES, translate_from_english, translate_to_english

CONFIDENCE_THRESHOLD = 0.5


def run_agent_multilingual(
    prompt: str,
    image_b64: str | None,
    emit: Callable[[str, dict], None],
    audio_path: str | None = None,
    document_text: str | None = None,
    thread_id: str | None = None,
    output_language: str | None = None,
    use_knowledge_base: bool = True,
    document_page_images: list[str] | None = None,
) -> None:
    try:
        detected = detect_language(prompt)
    except LanguageDetectionUnavailable as exc:
        emit("language_warning", {"warning": str(exc)})
        detected = {"language": "en", "confidence": 0.0}

    input_needs_translation = detected["language"] in SUPPORTED_LANGUAGES and detected["confidence"] >= CONFIDENCE_THRESHOLD

    if output_language is not None:
        # An explicit request always wins, even "en" (answer in English
        # regardless of input language) — that's a real request to
        # *suppress* the mirror-the-input-language default, not "unset".
        target_language = output_language if output_language in SUPPORTED_LANGUAGES else None
    else:
        target_language = detected["language"] if input_needs_translation else None

    if not input_needs_translation:
        english_prompt = prompt
    else:
        emit("language_detected", detected)
        try:
            english_prompt = translate_to_english(prompt, detected["language"])
        except Exception as exc:
            emit("error", {"error": f"Translation to English failed: {exc}"})
            return
        emit("translated_prompt", {"original": prompt, "english": english_prompt})

    if not target_language:
        run_agent(english_prompt, image_b64, emit, audio_path, document_text, thread_id, use_knowledge_base, document_page_images)
        return

    final_content = ""

    def capture(event: str, data: dict) -> None:
        nonlocal final_content
        if event == "done":
            # swallowed here — this wrapper emits its own "done" below,
            # once the answer has been translated back
            final_content = data.get("content", "")
            return
        emit(event, data)

    run_agent(english_prompt, image_b64, capture, audio_path, document_text, thread_id, use_knowledge_base, document_page_images)

    if final_content:
        try:
            translated_back = translate_from_english(final_content, target_language)
            emit("translated_response", {"english": final_content, "translated": translated_back})
            emit("done", {"content": translated_back})
            return
        except Exception as exc:
            emit("language_warning", {"warning": f"Translating the answer back failed: {exc}"})

    emit("done", {"content": final_content})
