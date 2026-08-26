"""Phase 8: local speech-to-text via faster-whisper (a CTranslate2
reimplementation of OpenAI Whisper) — the model weights download once on
first use and every transcription after that runs fully on-device, no
runtime network calls.
"""

from typing import Callable

from faster_whisper import WhisperModel

MODEL_SIZE = "small"

_model: WhisperModel | None = None


def _get_model() -> WhisperModel:
    global _model
    if _model is None:
        _model = WhisperModel(MODEL_SIZE, device="cpu", compute_type="int8")
    return _model


def transcribe(audio_path: str, emit: Callable[[str, dict], None] | None = None) -> dict:
    """Transcribes `audio_path`, auto-detecting the spoken language. If
    `emit` is given, streams a `language_detected` event as soon as it's
    known and one `segment` event per decoded chunk (with start/end
    timestamps) as they're produced — faster-whisper's segment iterator is
    lazy, so this is genuine incremental progress, not a fake replay."""
    model = _get_model()
    segments_iter, info = model.transcribe(audio_path, task="transcribe")

    if emit:
        emit("language_detected", {"language": info.language, "probability": float(info.language_probability)})

    segments = []
    for seg in segments_iter:
        entry = {"start": float(seg.start), "end": float(seg.end), "text": seg.text.strip()}
        segments.append(entry)
        if emit:
            emit("segment", entry)

    return {
        "text": " ".join(s["text"] for s in segments).strip(),
        "language": info.language,
        "language_probability": float(info.language_probability),
        "segments": segments,
    }
