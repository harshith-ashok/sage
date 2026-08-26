# SAGE — Sovereign Agentic (Grounded) Engine

SAGE is a self-hosted AI workbench for confidential industrial and government work.
Everything — the models, the search index, the code sandbox — runs on your own
machine. **No component ever calls out to the internet after the initial model
download.** You can literally unplug the network and keep using it.

Think of it as an internal ChatGPT-style assistant that can also: search your own
documents, draft real Word documents grounded in your procedures, write and safely
run code, and do exact math — instead of guessing.

---

## What it can do today

- 💬 **One chat box for everything** — no menus, no task-type dropdown. Type what
  you need (optionally attach an image or document) into the Console and the
  agent itself figures out which of the capabilities below it needs, in what
  order — search, then calculate, then write a file, all in one request if
  that's what it takes. The reply streams in token-by-token as it's generated
  (not a wait-then-dump), rendered as real formatted text — headings, **bold**,
  bullet lists, and syntax-highlighted code blocks (click one to pop it open
  full-size) — not raw `**markdown**` characters. Think Claude Code/Codex, but
  for procedures, documents, and numbers instead of source code.
- 📎 **Attach any text document** — PDF, `.docx`, `.xlsx`, `.pptx`, `.txt`, `.md`,
  `.csv`, `.json`, `.yaml` — the agent reads it directly (tables and slide text
  included) and answers questions about it, no separate upload flow needed.
- 🧵 **It remembers the conversation** — ask a follow-up like "convert it into a
  word doc" right after a long analysis and it knows what "it" is; each chat
  keeps its own message history until you start a "New chat".
- ∑ **Real math rendering** — equations and formulas come back as actual typeset
  math (via KaTeX, bundled locally), not raw `$...$` LaTeX source.
- 🗂️ **Chat history survives a refresh** — the transcript is saved to this
  browser only (nothing leaves the machine) and restored on reload; "New chat"
  in the Console header starts a clean one.
- 🧠 **See it actually thinking, not just a spinner** — a complex multi-part
  request can spend a while in an internal reasoning pass before it says
  anything visible; that reasoning streams live in a collapsible "Thinking…"
  block instead of the UI going quiet for minutes with no sign anything is
  happening.
- 🧠 **Switch AI models on the fly** — pick which model handles reasoning, coding,
  vision, and search from the Model Registry page, no restart, no code changes.
  Mix local models (fully private) and cloud models (faster, but leaves the
  building) freely.
- 📚 **Answer questions from your own documents** — ask a question, get an answer
  with citations back to the exact page and section it came from, plus a
  confidence flag when it's not sure.
- 📄 **Turn a scanned report — or an analysis — into a real Word document** — attach
  a photo/scan of a filled-in inspection form, or just say "convert it into a word
  doc" after a chat answer, and SAGE produces a real, downloadable `.docx` —
  properly formatted (real headings, bold, bullet lists, **actual tables**, not a
  row of pipe characters dumped as text, and equations rendered as real typeset
  math, not raw `\[...\]` source) — catching it if a technician's own
  recommendation doesn't actually match what the procedure requires.
- 📊 **Or Excel / PowerPoint, if that's the better fit** — "export this as a
  spreadsheet" produces a real `.xlsx` with actual numeric cells (usable in
  formulas, not just readable); "make a slide summary" produces a real `.pptx`
  deck. The agent picks the format that matches what you asked for.
- 📐 **Read a P&ID drawing against a spec sheet** — attach a piping &
  instrumentation diagram excerpt and its equipment spec, and SAGE identifies
  each tagged symbol (valve type, instrument bubble, line type) against a
  fixed, documented legend — not a guess — then flags where the drawing
  doesn't actually match what the spec requires.
- 💻 **Write and run code safely** — SAGE writes code (numpy/scipy available,
  nothing else — no network to install anything), runs it in a locked-down
  sandbox, checks the actual output against what was expected when there's a
  known answer, and tries again if it got it wrong. Open-ended analysis code
  (no single "correct" output to match) is verified by actually running
  cleanly instead of forcing a fixed-answer shape onto it.
- 🔢 **Do exact math, not guessed math** — calculations (with units, like
  `force / area` → pascals) are computed by a real math engine, not "vibes" from
  the language model.
- 📈 **Fit real statistical models, not eyeballed trends** — give it numbers and
  ask for a trend, a classification, a clustering, or a dimensionality
  reduction, and SAGE actually fits it with scikit-learn (held-out test score,
  and for regression a real confidence interval on the coefficients — not a
  language model guessing what the answer "looks like"). Guardrailed against
  nonsense input (too little data, mismatched shapes, impossible parameters)
  so it fails clearly instead of fitting garbage.
- 🎤 **Talk to it** — hit record (or attach an audio file) in the Console; local
  speech-to-text (faster-whisper) transcribes it straight into the text box,
  auto-detecting the language, so you see exactly what it heard and can edit
  it before sending — one unified upload button handles images, documents,
  and audio alike.
- 🌐 **Ask in Hindi, Tamil, Telugu, Kannada, or Malayalam — or just ask for the
  answer in one** — SAGE detects the language automatically and translates
  transparently around its (unchanged, English-internal) core pipeline, so a
  question in your language gets an answer back in your language. The two are
  also independent: the "Answer in" toggle buttons in the Console header let
  you type in English and get the answer translated into any of those
  languages anyway (or the reverse — ask in Hindi, pin the answer to
  English). Clicking one re-translates every answer already on screen right
  away, not just the next thing you send. *(The
  translation model, IndicTrans2, needs a one-time authenticated download
  before this works — see
  [Setup](#5-optional-enable-regional-language-translation). Everything else
  on this page works without it.)*

## What's coming next

The project is being built in phases (see [CLAUDE.md](CLAUDE.md) for the full,
up-to-date checklist). Not yet built: spoken (text-to-speech) output for the
regional-language layer, extra multimodal/diagram-reading features,
cross-document contradiction detection, load-aware model routing, and a
visible "zero external calls" network monitor.

---

## How it's put together

```
┌─────────────┐      HTTP + SSE       ┌──────────────┐
│  Frontend    │ ───────────────────▶ │   Backend     │
│  Vue 3 UI    │ ◀─────────────────── │   FastAPI     │
└─────────────┘                       └───────┬──────┘
                                               │
                        ┌──────────────────────┼────────────────────────┐
                        ▼                      ▼                        ▼
                 ┌─────────────┐       ┌───────────────┐        ┌───────────────┐
                 │   Ollama     │       │    Qdrant      │        │    Docker      │
                 │ (the models) │       │ (search index) │        │ (code sandbox) │
                 └─────────────┘       └───────────────┘        └───────────────┘
```

- **Frontend** (`frontend/`) — a Vue 3 + Tailwind single-page app. Sidebar
  sections: Console (the chat — everything happens here), Knowledge Base
  (search your documents directly, without going through the agent), Deliverables
  (download finished files), Model Registry (see and switch which model does
  what), Network Monitor and Settings (placeholders for later phases).
- **Backend** (`backend/`) — a Python FastAPI service, deliberately kept to the
  handful of HTTP endpoints the product actually calls (no separate REST
  endpoint per capability sitting alongside the agent tool that already does
  the same job), built around two ideas:
  - A **model router**: every AI call is tagged with a "task type"
    (`reasoning` / `coding` / `vision` / `embedding`), and a single config file
    (`backend/config/models.yaml`) decides which actual model handles each one.
    Changing that file — or clicking a button in the Model Registry page — is the
    *only* thing needed to swap models. No code ever hardcodes a model name.
  - An **agent** (`app/agent.py`) that the Console talks to: it hands your prompt
    to the reasoning model along with a toolbox (search the knowledge base, read
    an attached image, transcribe attached audio, run sandboxed code, calculate,
    fit a statistical model, write a `.docx`), and the model decides for itself
    which tools to call, in what order, based on what you actually asked for.
  - A **language layer** (`app/language/`) that wraps the agent from the
    outside, transparently: detect the prompt's language (fastText), and if
    it's a supported non-English language, translate to English (IndicTrans2),
    run the *same, unmodified* agent, then translate the answer back. English
    requests never touch this layer at all.
- **Ollama** serves the actual language models, both models running fully on your
  machine and (optionally, for speed during development) models proxied through
  Ollama's cloud service.
- **Qdrant** is the search database behind the Knowledge Base — it stores your
  documents as searchable chunks and finds the most relevant ones for a question.
- **Docker** provides the sandbox the coding task runs generated code inside —
  network access is switched off for that container, so generated code can't
  reach the internet even if it tried.

### A request, step by step

Say you type into the Console: *"How often shall critical service valves be
visually inspected?"*

1. The agent reads your prompt and decides it needs the `search_knowledge_base`
   tool — it doesn't answer from memory.
2. That tool searches Qdrant, re-ranks the results for relevance, and hands the
   agent the actual excerpts (with page/section) instead of a guess.
3. The agent reads those excerpts and writes an answer that cites the specific
   page and section it came from — you see this happen live, step by step, not
   after a long spinner.

A more involved request — *"I attached a scanned inspection report; check the
finding against our SOPs and draft an approval note"* — plays out the same way,
just with more steps: read the image → search the procedures (more than once, if
the first search doesn't cover escalation/urgency requirements) → write the
`.docx`. Nothing is hardcoded about which steps a request needs; the agent works
that out from the prompt, the same way it would pick `calculate` for a maths
question or `run_sandboxed_code` for a coding one.

---

## Project layout

```
backend/                 Python backend (FastAPI + LangGraph), uv-managed
  app/
    config.py             loads/writes models.yaml (the model registry)
    router.py              resolves a task type to a live model client
    agent.py                the Console's brain: one chat loop, picks its own tools
    orchestrator.py           generic plan -> act -> verify loop (used directly by /route)
    knowledge/               Phase 3: document search (RAG) pipeline
    tasks/                    Phase 4/5/7: document task, code task, calc tool, ML tool
                                (also used as tools by agent.py)
    speech/                    Phase 8: local speech-to-text (faster-whisper)
    language/                   Phase 8: language ID + translation, wraps agent.py
  config/models.yaml       *the* file that decides which model does what
  data/sop_docs/            sample documents the Knowledge Base searches
  data/deliverables/         finished .docx/.xlsx/.pptx files land here
  eval/                      recall/precision test suite for the search pipeline
  scripts/ingest_knowledge.py    (re)index the documents into Qdrant

frontend/                Vue 3 + Tailwind + Vite single-page app
  src/views/               one file per sidebar page
  src/components/base/      shared UI building blocks (Button, Card, ...)
  src/stores/                shared app state (Pinia)
  src/lib/api.ts              talks to the backend (fetch + live streaming)

insurance_claim_agent/   reference project this one borrowed patterns from
hi/                       reference project this one borrowed patterns from
CLAUDE.md                 the full build plan, phase by phase, with status
```

---

## Setup (local development)

### Prerequisites

Install these once:

- **[uv](https://docs.astral.sh/uv/)** — manages the Python backend and its own
  Python install, so you don't need Python pre-installed.
- **[Node.js](https://nodejs.org/)** (20+) — for the frontend.
- **[Ollama](https://ollama.com/)** — runs the AI models locally.
- **Docker** (or Rancher Desktop / OrbStack / any Docker-compatible engine) — runs
  the search database and the code sandbox.

### 1. Pull the models

```bash
ollama pull qwen3:8b            # reasoning (fully local)
ollama pull qwen2.5-coder:7b    # coding (fully local)
ollama pull qwen2.5vl:7b        # vision (fully local)
ollama pull bge-m3              # embeddings, for document search
```

These are the fully-local defaults. `backend/config/models.yaml` also lists
faster cloud-proxied alternatives (needs `ollama signin`) if you'd rather trade a
little privacy for speed during development — see that file's comments.

### 2. Start Qdrant (the search database) and build the sandbox image

```bash
docker run -d --name sage-qdrant -p 6333:6333 -p 6334:6334 \
  -v sage_qdrant_storage:/qdrant/storage qdrant/qdrant:latest

docker build -t sage-sandbox:latest backend/docker/sandbox/
```

Both are one-time setup — Qdrant keeps running in the background afterward
(`docker start sage-qdrant` to bring it back after a reboot), and the sandbox
image (a plain Python image with numpy/scipy pre-installed, since the sandbox
itself has no network to install anything at run time) doesn't need rebuilding
unless you change `backend/docker/sandbox/Dockerfile`.

### 3. Set up and run the backend

```bash
cd backend
uv sync                                  # installs everything, no system Python needed

# index the sample documents into Qdrant, so the Knowledge Base has something
# to search (re-run this any time you add/change documents in data/sop_docs/)
uv run python -m scripts.ingest_knowledge

# start the API server
uv run uvicorn app.main:app --reload --port 8000
```

The backend is now running at `http://localhost:8000`.

### 4. Set up and run the frontend

In a separate terminal:

```bash
cd frontend
npm install
npm run dev
```

Open **http://localhost:5173** — that's the app.

### 5. (Optional) enable regional-language translation

Everything above works without this step — it's only needed to ask the
Console a question in Hindi, Tamil, Telugu, Kannada, or Malayalam and get a
translated answer back (English works regardless, and speech-to-text already
auto-detects the spoken language on its own without this).

IndicTrans2 (the translation model) is **gated** on HuggingFace — a real
account and a one-time click-through are unavoidable, there's no way around
it programmatically:

1. Create a free account at [huggingface.co](https://huggingface.co).
2. Visit
   [`ai4bharat/indictrans2-en-indic-dist-200M`](https://huggingface.co/ai4bharat/indictrans2-en-indic-dist-200M)
   and
   [`ai4bharat/indictrans2-indic-en-dist-200M`](https://huggingface.co/ai4bharat/indictrans2-indic-en-dist-200M),
   and accept each model's terms (the button on the model page).
3. Create an access token: Settings → Access Tokens → New token (read access
   is enough).
4. Set it before starting the backend: `export HF_TOKEN=hf_...` (or add it to
   your shell profile). The two ~200M-parameter models download automatically
   the first time a non-English prompt actually needs them.

### 6. Try it out

Open the **Console** and just type — see [Examples](#examples) below for things
to try. Or peek at the other pages first:

- **Model Registry** — see every model SAGE knows about, and which one is
  currently active for each task. Try switching one.
- **Knowledge Base** — search your documents directly (same search the agent
  itself uses), without going through the chat.
- **Deliverables** — anything the Console writes to a file (`.docx` notes) shows
  up here, downloadable.

---

## Examples

Everything below happens in the **Console** — one input, nothing to configure
first. These use the sample SOP documents SAGE ships with (`backend/data/sop_docs/`:
valve inspection, confined space entry, pressure vessel inspection,
lockout/tagout) — try them as-is, or swap in your own documents and adapt the
questions.

### Simple — one capability at a time

| What you type | What happens |
| --- | --- |
| *"How often shall critical service valves be visually inspected?"* | Searches the knowledge base, answers **"every 6 months"** with a citation back to the exact page/section. |
| *"What is 5000 newtons over 0.02 square meters, in pascals?"* | Computes it with a real math engine (not a guess): **250,000 Pa**. |
| *"Write and run code to check whether 97 is a prime number."* | Writes a real Python script, runs it in the network-isolated sandbox, and reports the actual output: **True**. |
| *(attach a scanned/photographed form)* + *"Read this and summarize the finding."* | Reads the image with the vision model and describes what's on it. |
| *"I have wall-thickness readings from 12 vessel inspections over time: months=[0,3,6,...], thickness_mm=[12.0,11.8,...]. Fit a model and tell me the corrosion rate and how reliable the trend is."* | Actually fits a linear regression (scikit-learn, held-out test split) instead of guessing a trend — reports the real slope (**≈ ‑0.057 mm/month**) and test-set R² (**0.9996**). |
| *(hit 🎤 and ask your question out loud, or attach a voice memo)* | Local speech-to-text (faster-whisper) transcribes it — auto-detecting the language — and the agent answers what you actually asked, same as if you'd typed it. |
| *"Export this data as an Excel spreadsheet: months=[0,3,6,9], thickness_mm=[12.0,11.9,11.8,11.7]."* | Produces a real, downloadable `.xlsx` — actual numeric cells, sortable/usable in formulas, not text pasted into a grid. |
| *(attach a handwritten field note)* + *"Did this technician follow procedure?"* | Reads the handwriting, searches the relevant SOP, and checks the actual documented steps against it — not just a scanned form this time, a free-form note. |

### Multi-step / complex — the agent chains several tools on its own

These are the ones worth watching in the trace panel — each is **one prompt**;
the agent decides on its own how many steps and which tools it needs.

**1. Search, then calculate**

> *"What is the minimum oxygen percentage required before confined space entry?
> If a space currently reads 18.2 percent, how many percentage points short is
> that, precisely?"*

It searches the knowledge base for the requirement (19.5%), *then* hands the
numbers to the calculation tool rather than doing the subtraction itself —
**1.3 percentage points short**, with the SOP citation for where 19.5% came from.

**2. Search, calculate, and file a real document — three tools, one prompt**

> *"Search for the recertification interval for pressure relief valves in
> dirty/fouling service, calculate exactly how many recertifications occur over
> a 10-year service life, and save the answer as a short docx note titled 'PRV
> Recertification Schedule'."*

Search finds the rule (recertify every 12 months in fouling service) → calculate
works out 10 years ÷ 1 year = **10 recertifications** → a real `.docx` gets
written and shows up in Deliverables — all from one message.

**3. The flagship example — read an image, cross-check it against procedure, and
catch a mistake a human made**

> *(attach a scanned, filled-in inspection form)* +
> *"I attached a scanned inspection report. Read it, check the finding against
> our SOPs, and draft a formal approval note as a docx."*

This is the one that shows why grounding matters. The sample form describes a
**severe** valve leak, and the technician's own handwritten note recommends just
logging it through the standard 48-hour report. The agent reads the image,
searches the procedures — including a *second, more specific* search for
escalation requirements, since the first search alone doesn't surface the
right clause — and finds that a severe leak actually requires an **immediate
shutdown request**, not the technician's own recommendation. The final `.docx`
correctly overrides the human's mistake and cites exactly which clause requires
it, instead of quietly rubber-stamping the wrong call.

**4. Ask it out loud — voice in, grounded answer out**

> *(record) "What is the minimum oxygen percentage required before confined
> space entry?"*

Hit record, and the transcript appears in the text box as soon as you stop —
review or edit it, then send it like anything else you'd type. From there
it's the same grounded pipeline: the agent searches the knowledge base with
your (now-text) question and answers with a citation, just starting from
your voice instead of the keyboard.

**5. A genuinely hard analysis question — regression, confidence interval, and
engineering judgment in one go**

> *"I have wall-thickness readings from 12 vessel inspections over time:
> months = [0, 3, 6, ..., 33] and thickness_mm = [12.00, 11.94, ..., 11.35].
> Fit an appropriate regression model, report the corrosion rate in mm/month
> and mm/year, the fitted equation, R², and a confidence interval for the
> rate. Assess how reliable the trend is, whether a linear model is
> appropriate, the total thickness loss, how to use the rate to predict
> remaining service life, and the limitations (measurement uncertainty, UT
> repeatability, localized/pitting corrosion)."*

This is the one worth watching the "Thinking…" block for — it's a lot to ask
in one message. The agent fits a real regression (`fit_linear_regression`:
slope, intercept, R², **and** a proper 95% confidence interval on the slope
via `scipy.stats`, not a guess), then reasons through reliability, model
appropriateness, and the engineering limitations directly. Answers correctly
and completely in well under 15 seconds.

Then, as a **follow-up in the same chat**, just:

> *"convert it into a word doc"*

No need to repeat the data or the results — the agent remembers its own
answer from the message above and drafts a real `.docx` straight from it,
with the actual fitted numbers (not re-estimated, not generic placeholder
text). This only works within one chat; starting a "New chat" clears that
memory on purpose, the same as closing a conversation would. Any table in
the answer becomes a real Word table, and the equations (fitted model,
confidence interval) render as actual typeset math, not raw `\[...\]` text.

**6. Catch a drawing that doesn't match the spec — P&ID cross-referencing**

> *(attach a P&ID drawing excerpt and its equipment spec sheet)* +
> *"Check whether each tagged symbol on the drawing matches what the spec
> requires and flag any mismatches."*

The agent reads the drawing with `read_pid_drawing` — a vision call
constrained to a small, explicit symbol legend (gate valve, control valve,
pressure-safety valve, instrument bubble, process vs. signal line) rather
than guessing freely — reports exactly what's drawn at each tag, then reads
the spec sheet and compares them. In the sample data, one tag (`FV-2202`) is
drawn as a plain manual gate valve while the spec calls for a pneumatic
control valve — the agent catches it, explains the real consequence (it
can't provide the required fail-closed action), and flags it for escalation,
the same "ground it, don't quietly trust either source" pattern as the
inspection-report example above, applied to a drawing instead of text.

### Common issues

- **"Couldn't reach the task stream" / connection errors** — make sure the
  backend (`uvicorn`, port 8000) is actually running; the frontend expects it at
  `http://localhost:8000` by default (override with `VITE_API_BASE_URL`).
- **Knowledge Base returns "collection does not exist"** — you skipped step 3's
  ingestion command; run `uv run python -m scripts.ingest_knowledge` from `backend/`.
- **A cloud model (`-cloud` suffix) fails with a subscription error** — that
  model needs an Ollama Cloud plan; switch to the local alternative on the Model
  Registry page instead.
- **First request to a new page feels slow** — the first call to a given model
  or the citation-checking model both trigger a one-time download/warm-up; it's
  fast after that.
- **A non-English question fails with "gated repo" / 401 or 403** — you
  skipped [step 5](#5-optional-enable-regional-language-translation)
  (IndicTrans2 needs a one-time HuggingFace login, accepting the terms on
  *both* model pages, and `HF_TOKEN`); English works regardless, and
  speech-to-text's own language detection isn't affected by this.
- **Translation still doesn't work even with a valid, authorized `HF_TOKEN`**
  — known, currently unresolved: past the auth wall, IndicTrans2's official
  HuggingFace repo hits several transformers-5.x compatibility breaks in a
  row (`app/language/translate.py` documents each one it hit and fixed —
  import paths, tokenizer init order, a removed weight-tying method), and
  gets as far as actually starting real inference before failing inside the
  model's own custom attention code on a KV-cache shape mismatch that
  hasn't been resolved yet. Everything else in this app is unaffected —
  this is isolated to IndicTrans2 translation specifically.
- **A complex, multi-part request sits on "Thinking…" for a while before
  anything else shows up** — that's expected, not a hang: reasoning models
  stream their thinking separately from their answer, and the Console now
  shows that stream live instead of a blank spinner. If a request truly never
  progresses past thinking after several minutes, check the backend logs for
  an actual error (e.g. an Ollama Cloud outage) rather than assuming it's
  still working.
