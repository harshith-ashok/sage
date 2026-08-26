# ToDo

## Phase 7 — Speech input

- [ ] Add local `faster-whisper` speech-to-text for uploaded audio and microphone input, including model configuration, language detection, timestamps, and transcript streaming into the Console

## Phase 8 — Regional language layer

- [ ] Add Google Translate after `faster-whisper` transcription for Hindi, Tamil, Malayalam, and Telugu support, with source/target language selection and translated text returned alongside the original transcript
- [ ] Verify the Google Translate integration against SAGE's air-gapped requirement: no runtime network calls are permitted, so use an offline-compatible/local translation model or a pre-downloaded translation service before marking this complete
- [ ] Connect regional-language transcripts and translations to the existing Phase 6 agent/chat pipeline without changing the English-internal core task pipeline
