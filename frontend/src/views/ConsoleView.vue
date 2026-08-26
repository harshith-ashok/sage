<script setup lang="ts">
import { nextTick, onMounted, onUnmounted, ref } from "vue";
import Button from "../components/base/Button.vue";
import CodeModal from "../components/base/CodeModal.vue";
import Markdown from "../components/base/Markdown.vue";
import NetworkStatusBadge from "../components/base/NetworkStatusBadge.vue";
import { API_BASE, apiPost, apiPostForm, streamSSE } from "../lib/api";

type Segment =
  | { kind: "thinking"; id: string; content: string }
  | { kind: "text"; id: string; content: string; streaming: boolean; originalEnglish?: string; showOriginal?: boolean }
  | { kind: "tool_call"; tool: string; args: any }
  | { kind: "tool_result"; tool: string; content: string }
  | { kind: "tier_selected"; taskType: string; tier: string; modelId?: string }
  | { kind: "language_detected"; language: string; confidence?: number; probability?: number }
  | { kind: "segment_timing"; start: number; end: number; text: string }
  | { kind: "translated_prompt"; english: string }
  | { kind: "translating" }
  | { kind: "language_warning"; warning: string }
  | { kind: "scope_blocked" };

interface Turn {
  prompt: string;
  imageName: string | null;
  imagePreviewUrl: string | null;
  documentName: string | null;
  segments: Segment[];
  status: "running" | "done" | "error";
  errorMessage: string;
  // What this turn actually requested (captured at send time, not read live
  // off the dropdown later — the dropdown can change while this turn is
  // still streaming). "" means auto/no explicit request.
  requestedOutputLanguage: string;
}

const TOOL_LABELS: Record<string, string> = {
  search_knowledge_base: "Searching the knowledge base",
  check_cross_document_contradictions: "Checking for conflicts against records",
  read_uploaded_image: "Reading the attached image",
  read_pid_drawing: "Reading the P&ID drawing",
  run_sandboxed_code: "Running code in the sandbox",
  calculate: "Calculating",
  draft_docx_approval_note: "Writing the .docx file",
  export_to_excel: "Writing the .xlsx file",
  export_to_powerpoint: "Writing the .pptx file",
  transcribe_audio: "Transcribing the attached audio",
  read_uploaded_document: "Reading the attached document",
  fit_linear_regression: "Fitting a linear regression",
  fit_logistic_regression: "Fitting a logistic regression",
  kmeans_cluster: "Clustering the data",
  pca_reduce: "Reducing dimensionality",
};

// One unified upload input, dispatched by file type on change (see
// onUploadChange) rather than a separate button per attachment kind.
const UPLOAD_ACCEPT = "image/*,audio/*,.pdf,.docx,.xlsx,.pptx,.txt,.md,.csv,.json,.yaml,.yml,.log";

// "" = auto (mirror whatever language the prompt itself is written in —
// the pre-existing default). Any other value forces the answer into that
// language regardless of the prompt's language, sent as output_language on
// every /agent/chat request (app/language/pipeline.py). Toggle buttons, not
// a dropdown — clicking one takes effect immediately (see onLanguageToggle
// below), re-translating every answer already on screen, not just the next
// message sent.
const LANGUAGE_TOGGLES = [
  { value: "", label: "Auto" },
  { value: "en", label: "EN" },
  { value: "hi", label: "HI" },
  { value: "ta", label: "TA" },
  { value: "te", label: "TE" },
  { value: "kn", label: "KN" },
  { value: "ml", label: "ML" },
];
const outputLanguage = ref("");
const retranslating = ref(false);
const retranslateError = ref<string | null>(null);

const turns = ref<Turn[]>([]);
const prompt = ref("");
const attachedFile = ref<File | null>(null);
const attachedPreviewUrl = ref<string | null>(null);
const attachedDocument = ref<File | null>(null);
const recording = ref(false);
const transcribing = ref(false);
const transcribeError = ref<string | null>(null);
const running = ref(false);
const transcriptEl = ref<HTMLDivElement | null>(null);
const textareaEl = ref<HTMLTextAreaElement | null>(null);
const codeModal = ref<{ code: string; language: string } | null>(null);

let activeSource: EventSource | null = null;
let mediaRecorder: MediaRecorder | null = null;
let recordedChunks: Blob[] = [];

// Session persistence: the transcript survives a page refresh/tab close via
// localStorage, private to this browser (air-gapped — nothing leaves the
// machine). Only completed turns are written (see the `done`/`error`
// handlers below), not on every streaming token, so this stays cheap.
// Object URLs (image/audio previews) can't survive a reload — they're
// dropped on save and the name label is kept so the attachment is still
// visible in history, just not re-previewable.
const STORAGE_KEY = "sage-console-turns-v1";
const CONVERSATION_ID_KEY = "sage-console-conversation-id-v1";

// One id per chat, sent with every /agent/chat request (see submit()) so
// the backend's LangGraph checkpointer (app/agent.py) can resume that
// conversation's message history — without this, every message was its own
// stateless turn and a follow-up like "convert it into a word doc" had
// nothing to refer back to. Persisted alongside the transcript so a page
// refresh continues the same backend thread; reset in newChat() so
// starting over doesn't drag old context into a new conversation.
const conversationId = ref<string>(crypto.randomUUID());

function persistTurns() {
  try {
    const serializable = turns.value.map((t) => ({ ...t, imagePreviewUrl: null }));
    localStorage.setItem(STORAGE_KEY, JSON.stringify(serializable));
    localStorage.setItem(CONVERSATION_ID_KEY, conversationId.value);
  } catch {
    // localStorage full or unavailable (private browsing, etc.) — history
    // just won't survive a reload this session, nothing else depends on it.
  }
}

function loadTurns() {
  try {
    const savedId = localStorage.getItem(CONVERSATION_ID_KEY);
    if (savedId) conversationId.value = savedId;
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return;
    const restored = JSON.parse(raw) as Turn[];
    // Never restore a turn stuck "running" from a tab that was closed
    // mid-stream — there's no live connection to resume it.
    for (const t of restored) {
      if (t.status === "running") {
        t.status = "error";
        t.errorMessage = "Interrupted (page was closed or reloaded while this was running).";
      }
    }
    turns.value = restored;
  } catch {
    // Corrupt/unreadable saved state — start with a clean transcript rather
    // than crashing the view.
  }
}

function newChat() {
  activeSource?.close();
  running.value = false;
  turns.value = [];
  conversationId.value = crypto.randomUUID();
  localStorage.removeItem(STORAGE_KEY);
  localStorage.setItem(CONVERSATION_ID_KEY, conversationId.value);
}

/** A turn's own natural language — English unless its input triggered
 * translation (a `language_detected` segment exists), in which case it
 * mirrors that. Used for "Auto": unlike every other toggle, "Auto" has no
 * single fixed target — different turns in the same conversation can each
 * have their own natural language if you switched languages mid-chat. */
function turnNaturalLanguage(turn: Turn): string {
  const seg = turn.segments.find((s): s is Segment & { kind: "language_detected" } => s.kind === "language_detected");
  return seg?.language ?? "en";
}

/** Re-translates one text segment to `target`, always working from its
 * stable English source (never re-translating an already-translated string,
 * which IndicTrans2 doesn't support indic-to-indic anyway — only via
 * English). `originalEnglish` doubles as both the "Show original" toggle's
 * source and this cache, set the first time a segment is ever translated. */
async function translateSegmentTo(seg: Segment & { kind: "text" }, target: string): Promise<void> {
  const englishSource = seg.originalEnglish ?? seg.content;
  if (target === "en") {
    seg.content = englishSource;
    return;
  }
  const result = await apiPost<{ translated: string }>("/translate", { text: englishSource, target_language: target });
  seg.content = result.translated;
  seg.originalEnglish = englishSource;
}

/** Fired the instant a language toggle is clicked — re-translates every
 * answer already on screen right away, not just messages sent from here on
 * (which still also get the same output_language, via submit() below). */
async function onLanguageToggle(value: string) {
  if (value === outputLanguage.value || retranslating.value) return;
  outputLanguage.value = value;
  retranslateError.value = null;

  const jobs: Promise<void>[] = [];
  for (const turn of turns.value) {
    const target = value === "" ? turnNaturalLanguage(turn) : value;
    for (const seg of turn.segments) {
      if (seg.kind === "text" && seg.content) jobs.push(translateSegmentTo(seg, target));
    }
  }
  if (jobs.length === 0) return;

  retranslating.value = true;
  try {
    await Promise.all(jobs);
    persistTurns();
  } catch (e) {
    retranslateError.value = e instanceof Error ? e.message : "Translation failed.";
  } finally {
    retranslating.value = false;
  }
}

onMounted(() => {
  loadTurns();
  scrollToBottom();
});

function autoResize() {
  const el = textareaEl.value;
  if (!el) return;
  el.style.height = "auto";
  el.style.height = `${Math.min(el.scrollHeight, 220)}px`;
}

// One button, any supported file type — dispatched here instead of asking
// the user to pick the right one of three separate attach buttons. Audio
// doesn't become a pending "attachment" at all: like the mic button below,
// it's transcribed immediately and dropped straight into the prompt text,
// not sent to the agent as a raw clip.
function onUploadChange(e: Event) {
  const input = e.target as HTMLInputElement;
  const file = input.files?.[0] ?? null;
  input.value = ""; // allow picking the same file again later
  if (!file) return;
  if (file.type.startsWith("image/")) {
    attachedFile.value = file;
    attachedPreviewUrl.value = URL.createObjectURL(file);
  } else if (file.type.startsWith("audio/")) {
    transcribeAndInsert(file);
  } else {
    attachedDocument.value = file;
  }
}

function clearAttachment() {
  attachedFile.value = null;
  attachedPreviewUrl.value = null;
}
function clearDocument() {
  attachedDocument.value = null;
}

/** Sends a clip to /transcribe and drops the resulting text into the prompt
 * box (appended after anything already typed) — used by both the mic button
 * and an uploaded audio file via the unified upload button. Nothing is sent
 * to the agent yet; the user sees the transcript and can edit it before
 * hitting Send, same as if they'd typed it themselves. */
async function transcribeAndInsert(file: File) {
  transcribing.value = true;
  transcribeError.value = null;
  try {
    const form = new FormData();
    form.append("file", file);
    const result = await apiPostForm<{ text: string; language: string; language_probability: number }>("/transcribe", form);
    const text = result.text.trim();
    if (text) {
      prompt.value = prompt.value.trim() ? `${prompt.value.trim()} ${text}` : text;
      await nextTick();
      autoResize();
      textareaEl.value?.focus();
    } else {
      transcribeError.value = "Didn't catch any speech in that clip.";
    }
  } catch (e) {
    transcribeError.value = e instanceof Error ? e.message : "Transcription failed.";
  } finally {
    transcribing.value = false;
  }
}

async function toggleRecording() {
  if (recording.value) {
    mediaRecorder?.stop();
    recording.value = false;
    return;
  }
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    recordedChunks = [];
    mediaRecorder = new MediaRecorder(stream);
    mediaRecorder.ondataavailable = (e) => {
      if (e.data.size > 0) recordedChunks.push(e.data);
    };
    mediaRecorder.onstop = () => {
      const blob = new Blob(recordedChunks, { type: "audio/webm" });
      stream.getTracks().forEach((t) => t.stop());
      transcribeAndInsert(new File([blob], "recording.webm", { type: "audio/webm" }));
    };
    mediaRecorder.start();
    recording.value = true;
  } catch {
    recording.value = false;
  }
}

function deliverableLink(content: string): { filename: string } | null {
  const match = /Saved as ([\w.-]+\.(?:docx|xlsx|pptx))/.exec(content);
  return match ? { filename: match[1] } : null;
}

// check_cross_document_contradictions (app/agent.py) instructs the model to
// mark a real conflict with this exact literal string — a plain substring
// check here is enough to surface it as a distinct, hard-to-miss banner
// instead of it just being one more paragraph in the answer.
function hasContradictionFlag(content: string): boolean {
  return content.includes("⚠️ CONFLICT:") || content.includes("⚠️ CONFLICT :");
}

function onViewCode(code: string, language: string) {
  codeModal.value = { code, language };
}

// Streaming fires token events dozens of times a second; calling
// scrollTo({behavior: "smooth"}) on every single one queues a fresh smooth-
// scroll animation before the last one finishes, which fights the browser's
// compositor for every frame and starves it of paint time for the actual
// text updates — the reactive state updates instantly, but visually nothing
// appears to move until the flood of scroll calls stops, making a genuinely
// streaming response look like it "just appears" at the end. Coalescing to
// one scroll per animation frame (instant, not smooth, while more tokens are
// still arriving) fixes that; only used the isolated single completed-turn
// case (`done`) with `smooth` for the final settle.
let scrollFrame: number | null = null;
function scrollToBottom(behavior: ScrollBehavior = "auto") {
  if (scrollFrame !== null) return;
  scrollFrame = requestAnimationFrame(() => {
    scrollFrame = null;
    const el = transcriptEl.value;
    if (!el) return;
    // Don't yank the view back down if the user has scrolled up to reread
    // something earlier in the transcript.
    const nearBottom = el.scrollHeight - el.scrollTop - el.clientHeight < 120;
    if (nearBottom) el.scrollTo({ top: el.scrollHeight, behavior });
  });
}

function findTextSegment(turn: Turn, id: string): Segment & { kind: "text" } {
  const existing = turn.segments.find((s) => s.kind === "text" && s.id === id);
  if (existing) return existing as Segment & { kind: "text" };
  const created: Segment = { kind: "text", id, content: "", streaming: true };
  turn.segments.push(created);
  return created as Segment & { kind: "text" };
}

function findThinkingSegment(turn: Turn, id: string): Segment & { kind: "thinking" } {
  const existing = turn.segments.find((s) => s.kind === "thinking" && s.id === id);
  if (existing) return existing as Segment & { kind: "thinking" };
  const created: Segment = { kind: "thinking", id, content: "" };
  turn.segments.push(created);
  return created as Segment & { kind: "thinking" };
}

async function submit() {
  const trimmed = prompt.value.trim();
  if (!trimmed || running.value || transcribing.value) return;

  const turn: Turn = {
    prompt: trimmed,
    imageName: attachedFile.value?.name ?? null,
    imagePreviewUrl: attachedPreviewUrl.value,
    documentName: attachedDocument.value?.name ?? null,
    segments: [],
    status: "running",
    errorMessage: "",
    requestedOutputLanguage: outputLanguage.value,
  };
  turns.value.push(turn);
  running.value = true;
  scrollToBottom("smooth");

  const form = new FormData();
  form.append("prompt", trimmed);
  form.append("conversation_id", conversationId.value);
  if (outputLanguage.value) form.append("output_language", outputLanguage.value);
  if (attachedFile.value) form.append("file", attachedFile.value);
  if (attachedDocument.value) form.append("document", attachedDocument.value);

  prompt.value = "";
  attachedFile.value = null;
  attachedPreviewUrl.value = null;
  attachedDocument.value = null;
  await nextTick();
  autoResize();

  try {
    const { task_id } = await apiPostForm<{ task_id: string }>("/agent/chat", form);
    const streamPath = `/agent/chat/${task_id}/stream`;
    activeSource = streamSSE(
      streamPath,
      {
        thinking: (d) => {
          const seg = findThinkingSegment(turn, d.id);
          seg.content += d.content;
          scrollToBottom();
        },
        token: (d) => {
          const seg = findTextSegment(turn, d.id);
          seg.content += d.content;
          scrollToBottom();
        },
        message: (d) => {
          const seg = findTextSegment(turn, d.id);
          seg.content = d.content;
          seg.streaming = false;
          // This is the English draft finishing — translate_from_english()
          // runs synchronously after this with no event of its own, so
          // without a placeholder the UI would sit on the (wrong-language)
          // English answer for a few seconds before it suddenly flips to
          // the translated one when `translated_response` lands. A visible
          // "Translating…" row fills that gap honestly. Mirrors
          // app/language/pipeline.py's own target-language logic exactly:
          // an explicit request (including "en", meaning "answer in
          // English regardless of input language") always wins outright;
          // only with no explicit request does it fall back to "auto",
          // where a detected non-English input means the answer mirrors it.
          const translationExpected = turn.requestedOutputLanguage
            ? turn.requestedOutputLanguage !== "en"
            : turn.segments.some((s) => s.kind === "language_detected");
          if (translationExpected && !turn.segments.some((s) => s.kind === "translating")) {
            turn.segments.push({ kind: "translating" });
          }
        },
        tool_call: (d) => {
          turn.segments.push({ kind: "tool_call", tool: d.tool, args: d.args });
          scrollToBottom();
        },
        tool_result: (d) => {
          turn.segments.push({ kind: "tool_result", tool: d.tool, content: d.content });
          scrollToBottom();
        },
        tier_selected: (d) => {
          turn.segments.push({ kind: "tier_selected", taskType: d.task_type, tier: d.tier, modelId: d.model_id });
          scrollToBottom();
        },
        language_detected: (d) => {
          turn.segments.push({ kind: "language_detected", ...d });
          scrollToBottom();
        },
        segment: (d) => {
          turn.segments.push({ kind: "segment_timing", ...d });
          scrollToBottom();
        },
        translated_prompt: (d) => {
          turn.segments.push({ kind: "translated_prompt", english: d.english });
        },
        translated_response: (d) => {
          // The translated answer becomes the primary, visible response
          // (not a secondary footnote below the English draft) — that's
          // the actual point of asking in your own language. The English
          // draft is kept for an optional "Show original" toggle rather
          // than shown as the main text.
          turn.segments = turn.segments.filter((s) => s.kind !== "translating");
          const textSegs = turn.segments.filter((s): s is Segment & { kind: "text" } => s.kind === "text");
          const target = textSegs[textSegs.length - 1];
          if (target) {
            target.originalEnglish = d.english;
            target.content = d.translated;
            target.streaming = false;
          } else {
            turn.segments.push({ kind: "text", id: "translated", content: d.translated, streaming: false, originalEnglish: d.english });
          }
          scrollToBottom();
        },
        language_warning: (d) => {
          turn.segments.push({ kind: "language_warning", warning: d.warning });
        },
        scope_blocked: () => {
          // The topic guardrail (app/scope_guard.py) refused this turn before
          // the agent ever ran — no tool_call/tier_selected trace exists for
          // it, so this is the only signal marking it as a refusal rather
          // than a normal short answer.
          turn.segments.push({ kind: "scope_blocked" });
        },
        done: (d) => {
          // Any segment still marked "streaming" (no matching `message` arrived,
          // e.g. the very last chunk) gets a final, clean render too.
          for (const seg of turn.segments) {
            if (seg.kind === "text") seg.streaming = false;
          }
          // If translation failed, pipeline.py falls back to emitting the
          // English content directly here without ever sending
          // `translated_response` — drop the now-stale "Translating…" row
          // rather than leaving it stuck forever.
          turn.segments = turn.segments.filter((s) => s.kind !== "translating");
          if (d?.content && !turn.segments.some((s) => s.kind === "text")) {
            turn.segments.push({ kind: "text", id: "final", content: d.content, streaming: false });
          }
          turn.status = "done";
          running.value = false;
          scrollToBottom("smooth");
          persistTurns();
        },
        error: (d) => {
          turn.status = "error";
          turn.errorMessage = d?.error ?? "The agent failed.";
          running.value = false;
          persistTurns();
        },
      },
      {
        onConnectionError: () => {
          turn.status = "error";
          turn.errorMessage = `Couldn't reach the agent stream at ${API_BASE}${streamPath}`;
          running.value = false;
          persistTurns();
        },
        terminalEvents: ["done", "error"],
      },
    );
  } catch (e) {
    turn.status = "error";
    turn.errorMessage = e instanceof Error ? e.message : String(e);
    running.value = false;
    persistTurns();
  }
}

function onKeydown(e: KeyboardEvent) {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    submit();
  }
}

onUnmounted(() => {
  activeSource?.close();
  mediaRecorder?.stop();
  if (scrollFrame !== null) cancelAnimationFrame(scrollFrame);
});
</script>

<template>
  <div class="flex h-full flex-col">
    <div class="flex items-start justify-between gap-4 border-b border-border px-8 py-4">
      <div>
        <h1 class="font-display text-lg font-semibold text-text">Console</h1>
        <p class="mt-0.5 text-[12px] text-dim">
          One place for everything — ask a question (in English, Hindi, Tamil, Telugu, Kannada, or Malayalam), attach
          a scanned report, a document, or record a voice question, or hand it a coding/math problem. The agent
          decides what to do.
        </p>
      </div>
      <div class="flex shrink-0 flex-col items-end gap-1.5">
        <NetworkStatusBadge />
        <div class="flex items-center gap-2">
          <span class="text-[11px] text-dim" title="Every answer already on screen re-translates the instant you click one">Answer in</span>
          <div class="flex overflow-hidden rounded-md border border-border">
            <button
              v-for="(opt, i) in LANGUAGE_TOGGLES"
              :key="opt.value"
              type="button"
              :disabled="retranslating"
              class="px-2 py-1 text-[11px] font-medium transition-colors disabled:cursor-not-allowed disabled:opacity-50"
              :class="[
                outputLanguage === opt.value ? 'bg-accent text-accent-ink' : 'bg-panel-2 text-dim hover:text-text',
                i > 0 ? 'border-l border-border' : '',
              ]"
              @click="onLanguageToggle(opt.value)"
            >
              {{ opt.label }}
            </button>
          </div>
          <span v-if="retranslating" class="h-1.5 w-1.5 animate-pulse rounded-full bg-accent" title="Translating everything on screen…" />
          <Button v-if="turns.length > 0" variant="ghost" @click="newChat">New chat</Button>
        </div>
        <p v-if="retranslateError" class="text-[11px] text-danger">{{ retranslateError }}</p>
      </div>
    </div>

    <div ref="transcriptEl" class="flex-1 overflow-y-auto px-8 py-6">
      <div class="mx-auto flex max-w-3xl flex-col gap-6">
        <p v-if="turns.length === 0" class="text-[12.5px] text-dim-2">
          Try: "How often shall critical service valves be visually inspected?", "What is 5000 N over 0.02 m² in
          pascals?", "Write code that prints the first 10 primes", attach a scanned inspection report and ask for an
          approval note, attach a document and ask about it, or record a voice question.
        </p>

        <div v-for="(turn, i) in turns" :key="i" class="flex flex-col gap-3">
          <!-- user message -->
          <div class="self-end max-w-[85%] rounded-lg bg-accent/10 border border-accent/30 px-3.5 py-2.5">
            <img v-if="turn.imagePreviewUrl" :src="turn.imagePreviewUrl" class="mb-2 max-h-40 rounded-md border border-border-soft" />
            <div v-if="turn.documentName" class="mb-2 inline-flex items-center gap-1.5 rounded-md border border-border-soft bg-panel-2 px-2 py-1 text-[11px] text-dim">
              📄 {{ turn.documentName }}
            </div>
            <p class="whitespace-pre-wrap text-[13px] text-text">{{ turn.prompt }}</p>
          </div>

          <!-- agent trace -->
          <TransitionGroup tag="div" name="sage-trace" class="flex flex-col gap-1.5">
            <div v-for="(seg, j) in turn.segments" :key="j">
              <details v-if="seg.kind === 'thinking' && seg.content" class="group rounded-md border border-border-soft bg-panel-2/50" open>
                <summary class="flex cursor-pointer list-none items-center gap-2 px-3 py-1.5 text-[11.5px] text-dim-2 select-none">
                  <span class="h-1.5 w-1.5 rounded-full bg-dim-2 animate-pulse" />
                  <span class="italic">Thinking…</span>
                  <span class="ml-auto text-[10px] transition-transform group-open:rotate-180">▾</span>
                </summary>
                <div class="max-h-28 overflow-y-auto border-t border-border-soft px-3 py-2 font-mono text-[10.5px] italic leading-relaxed text-dim-2 whitespace-pre-wrap">
                  {{ seg.content }}
                </div>
              </details>

              <div v-else-if="seg.kind === 'tier_selected'" class="flex items-center gap-2 text-[11px] text-dim-2">
                <span
                  class="rounded-full border px-1.5 py-0.5 font-mono text-[9.5px] uppercase tracking-wide"
                  :class="seg.tier === 'strong' ? 'border-accent/30 text-accent' : 'border-border-soft text-dim-2'"
                >
                  {{ seg.tier }}
                </span>
                <span>{{ seg.taskType }}{{ seg.modelId ? ` · ${seg.modelId}` : "" }}</span>
              </div>

              <div v-else-if="seg.kind === 'tool_call'" class="flex items-center gap-2 text-[12px] text-dim">
                <span class="h-1.5 w-1.5 rounded-full bg-accent" :class="{ 'animate-pulse': turn.status === 'running' }" />
                <span>{{ TOOL_LABELS[seg.tool] ?? seg.tool }}…</span>
                <span class="font-mono text-[10.5px] text-dim-2 truncate">{{ JSON.stringify(seg.args) }}</span>
              </div>

              <details
                v-else-if="seg.kind === 'tool_result' && seg.tool === 'search_knowledge_base'"
                class="ml-3.5 rounded-md border border-border-soft bg-panel-2"
              >
                <summary class="cursor-pointer select-none px-3 py-2 text-[11.5px] text-dim">
                  Knowledge base results ({{ seg.content.length }} chars — click to expand)
                </summary>
                <div class="border-t border-border-soft px-3 py-2">
                  <Markdown :content="seg.content" @view-code="onViewCode" />
                </div>
              </details>

              <div v-else-if="seg.kind === 'tool_result'" class="ml-3.5 rounded-md border border-border-soft bg-panel-2 px-3 py-2">
                <Markdown :content="seg.content" @view-code="onViewCode" />
                <a
                  v-if="deliverableLink(seg.content)"
                  :href="`${API_BASE}/deliverables/${deliverableLink(seg.content)!.filename}`"
                  class="mt-1 inline-block text-[11.5px] font-medium text-accent hover:underline"
                >
                  Download {{ deliverableLink(seg.content)!.filename }}
                </a>
              </div>

              <div v-else-if="seg.kind === 'text' && seg.content" class="text-[13px] leading-relaxed text-text">
                <div
                  v-if="!seg.streaming && hasContradictionFlag(seg.content)"
                  class="mb-2 flex items-center gap-2 rounded-md border border-danger/30 bg-danger/10 px-3 py-2 text-[12px] font-medium text-danger"
                >
                  <span class="h-1.5 w-1.5 shrink-0 rounded-full bg-danger" />
                  Contradiction flagged against site records — see below
                </div>
                <Markdown v-if="!seg.streaming" :content="seg.content" @view-code="onViewCode" />
                <p v-else class="whitespace-pre-wrap">{{ seg.content }}<span class="sage-caret text-accent">▍</span></p>
                <button
                  v-if="seg.originalEnglish && seg.originalEnglish !== seg.content && !seg.streaming"
                  class="mt-1 text-[11px] text-dim-2 hover:text-dim"
                  @click="seg.showOriginal = !seg.showOriginal"
                >
                  {{ seg.showOriginal ? "Hide" : "Show" }} original (English)
                </button>
                <div v-if="seg.showOriginal" class="mt-1 rounded-md border border-border-soft bg-panel-2 px-3 py-2 text-[12px] text-dim">
                  <Markdown :content="seg.originalEnglish!" @view-code="onViewCode" />
                </div>
              </div>

              <div v-else-if="seg.kind === 'translating'" class="flex items-center gap-2 text-[12px] text-dim">
                <span class="h-1.5 w-1.5 animate-pulse rounded-full bg-accent" />
                <span class="italic">Translating…</span>
              </div>
              <div v-else-if="seg.kind === 'language_detected'" class="flex items-center gap-2 text-[12px] text-dim">
                <span class="h-1.5 w-1.5 rounded-full bg-accent" />
                <span>Detected language: {{ seg.language }} ({{ Math.round((seg.confidence ?? seg.probability ?? 0) * 100) }}%)</span>
              </div>
              <div v-else-if="seg.kind === 'segment_timing'" class="ml-3.5 font-mono text-[11px] text-dim">
                [{{ seg.start.toFixed(1) }}s–{{ seg.end.toFixed(1) }}s] {{ seg.text }}
              </div>
              <div v-else-if="seg.kind === 'translated_prompt'" class="ml-3.5 rounded-md border border-border-soft bg-panel-2 px-3 py-2 text-[11.5px] text-dim">
                Translated to English: <span class="text-text">{{ seg.english }}</span>
              </div>
              <p v-else-if="seg.kind === 'language_warning'" class="rounded-md border border-warn/30 bg-warn/10 px-3 py-2 text-[12px] text-warn">
                {{ seg.warning }}
              </p>
              <div v-else-if="seg.kind === 'scope_blocked'" class="flex items-center gap-2 text-[11px] text-dim-2">
                <span class="rounded-full border border-border-soft px-1.5 py-0.5 font-mono text-[9.5px] uppercase tracking-wide">Off-topic</span>
                <span>Refused before running — outside SAGE's scope</span>
              </div>
            </div>
          </TransitionGroup>

          <div v-if="turn.status === 'running'" class="flex items-center gap-2 text-[12px] text-dim-2">
            <span class="h-1.5 w-1.5 animate-pulse rounded-full bg-dim-2" />
            working…
          </div>
          <p v-if="turn.status === 'error'" class="rounded-md border border-danger/30 bg-danger/10 px-3 py-2 text-[12.5px] text-danger">
            {{ turn.errorMessage }}
          </p>
        </div>
      </div>
    </div>

    <div class="border-t border-border px-8 py-4">
      <div class="mx-auto flex max-w-3xl flex-col gap-2">
        <div v-if="attachedPreviewUrl || attachedDocument || transcribing || transcribeError" class="flex flex-wrap items-center gap-2">
          <div v-if="attachedPreviewUrl" class="flex items-center gap-2">
            <img :src="attachedPreviewUrl" class="h-12 w-12 rounded-md border border-border object-cover" />
            <span class="text-[11.5px] text-dim">{{ attachedFile?.name }}</span>
            <button class="text-[11.5px] text-dim-2 hover:text-danger" @click="clearAttachment">Remove</button>
          </div>
          <div v-if="attachedDocument" class="flex items-center gap-2 rounded-md border border-border bg-panel-2 px-2 py-1">
            <span class="text-[11.5px] text-dim">📄 {{ attachedDocument.name }}</span>
            <button class="text-[11.5px] text-dim-2 hover:text-danger" @click="clearDocument">Remove</button>
          </div>
          <div v-if="transcribing" class="flex items-center gap-1.5 text-[11.5px] text-dim">
            <span class="h-1.5 w-1.5 animate-pulse rounded-full bg-accent" />
            Transcribing…
          </div>
          <p v-if="transcribeError" class="text-[11.5px] text-danger">{{ transcribeError }}</p>
        </div>
        <div class="flex items-end gap-2">
          <label
            class="cursor-pointer rounded-md border border-border bg-panel-2 px-2.5 py-2 text-[12px] text-dim hover:text-text"
            title="Attach an image, document, or audio file"
          >
            📎
            <input type="file" :accept="UPLOAD_ACCEPT" class="hidden" @change="onUploadChange" />
          </label>
          <button
            type="button"
            class="rounded-md border px-2.5 py-2 text-[12px] disabled:cursor-not-allowed disabled:opacity-40"
            :class="recording ? 'border-danger/40 bg-danger/10 text-danger' : 'border-border bg-panel-2 text-dim hover:text-text'"
            title="Speak your question — transcribed into the box, not sent as audio"
            :disabled="transcribing"
            @click="toggleRecording"
          >
            {{ recording ? "⏹ stop" : "🎤" }}
          </button>
          <textarea
            ref="textareaEl"
            v-model="prompt"
            rows="1"
            placeholder="Ask anything, attach a file, or record a voice question…"
            class="max-h-[220px] flex-1 resize-none overflow-y-auto rounded-md border border-border bg-panel-2 px-3 py-2 text-[13px] text-text placeholder:text-dim-2 focus:border-accent focus:outline-none"
            @keydown="onKeydown"
            @input="autoResize"
          />
          <Button variant="primary" :disabled="running || transcribing || !prompt.trim()" @click="submit">
            {{ running ? "Running…" : "Send" }}
          </Button>
        </div>
      </div>
    </div>

    <CodeModal v-if="codeModal" :code="codeModal.code" :language="codeModal.language" @close="codeModal = null" />
  </div>
</template>
