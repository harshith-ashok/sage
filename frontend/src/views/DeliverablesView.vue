<script setup lang="ts">
import { computed, onMounted, ref } from "vue";
import Button from "../components/base/Button.vue";
import Card from "../components/base/Card.vue";
import StatusPill from "../components/base/StatusPill.vue";
import VirtualDataTable from "../components/base/VirtualDataTable.vue";
import { API_BASE, apiGet, apiPost, apiPostBlob, apiPostForm, streamSSE } from "../lib/api";

interface Deliverable {
  filename: string;
  size_bytes: number;
  created: number;
}

const items = ref<Deliverable[]>([]);
const loading = ref(false);
const errorMessage = ref("");
const selected = ref<Set<string>>(new Set());
const bulkBusy = ref(false);

const columns = [
  { key: "filename", label: "Name", width: "2fr" },
  { key: "size", label: "Size", width: "1fr" },
  { key: "created", label: "Created", width: "1fr" },
  { key: "download", label: "", width: "0.6fr" },
];

function formatSize(bytes: number) {
  return bytes < 1024 * 1024 ? `${(bytes / 1024).toFixed(1)} KB` : `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

function formatDate(epochSeconds: number) {
  return new Date(epochSeconds * 1000).toLocaleString();
}

async function load() {
  loading.value = true;
  errorMessage.value = "";
  try {
    items.value = await apiGet<Deliverable[]>("/deliverables");
    // Drop selections for files that no longer exist (e.g. after a delete
    // elsewhere) rather than silently keeping stale keys around.
    const known = new Set(items.value.map((i) => i.filename));
    selected.value = new Set([...selected.value].filter((f) => known.has(f)));
  } catch (e) {
    errorMessage.value = e instanceof Error ? e.message : String(e);
  } finally {
    loading.value = false;
  }
}

async function exportSelected() {
  bulkBusy.value = true;
  errorMessage.value = "";
  try {
    const blob = await apiPostBlob("/deliverables/export", { filenames: [...selected.value] });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "deliverables.zip";
    a.click();
    URL.revokeObjectURL(url);
  } catch (e) {
    errorMessage.value = e instanceof Error ? e.message : String(e);
  } finally {
    bulkBusy.value = false;
  }
}

async function deleteSelected() {
  if (!confirm(`Delete ${selected.value.size} deliverable${selected.value.size === 1 ? "" : "s"}? This can't be undone.`)) return;
  bulkBusy.value = true;
  errorMessage.value = "";
  try {
    await apiPost("/deliverables/delete", { filenames: [...selected.value] });
    selected.value = new Set();
    await load();
  } catch (e) {
    errorMessage.value = e instanceof Error ? e.message : String(e);
  } finally {
    bulkBusy.value = false;
  }
}

// --- Batch upload / queue mode (Phase 13) ---
// Reuses Phase 4's still-live single-file /tasks/document endpoint + its
// SSE stream (kept alive specifically because it's the only place the
// hard, bounded reviewer/critic loop runs — see Phase 6's writeup) rather
// than adding a new bulk backend endpoint: the "batch" part is entirely a
// frontend queue driving N independent existing task runs, capped at a
// small concurrency so this doesn't try to warm/run everything at once.
type QueueStatus = "queued" | "ocr" | "draft" | "review" | "done" | "error";
interface QueueItem {
  id: string;
  name: string;
  file: File;
  status: QueueStatus;
  errorMessage: string;
}
const QUEUE_CONCURRENCY = 2;
const queue = ref<QueueItem[]>([]);
const queueRunning = ref(false);
const uploadInput = ref<HTMLInputElement | null>(null);

const queueDone = computed(() => queue.value.length > 0 && queue.value.every((q) => q.status === "done" || q.status === "error"));

function onQueueFilesChosen(e: Event) {
  const files = Array.from((e.target as HTMLInputElement).files ?? []);
  if (files.length === 0) return;
  queue.value = files.map((file) => ({ id: crypto.randomUUID(), name: file.name, file, status: "queued", errorMessage: "" }));
  if (uploadInput.value) uploadInput.value.value = "";
  runQueue();
}

async function runOne(item: QueueItem): Promise<void> {
  return new Promise((resolve) => {
    (async () => {
      try {
        const form = new FormData();
        form.append("file", item.file);
        const { task_id } = await apiPostForm<{ task_id: string }>("/tasks/document", form);
        streamSSE(
          `/tasks/document/${task_id}/stream`,
          {
            ocr_done: () => (item.status = "ocr"),
            draft_done: () => (item.status = "draft"),
            review_done: () => (item.status = "review"),
            done: () => {
              item.status = "done";
              resolve();
            },
            error: (d) => {
              item.status = "error";
              item.errorMessage = d?.error ?? "Failed";
              resolve();
            },
          },
          { onConnectionError: () => { item.status = "error"; item.errorMessage = "Connection lost"; resolve(); } },
        );
      } catch (e) {
        item.status = "error";
        item.errorMessage = e instanceof Error ? e.message : String(e);
        resolve();
      }
    })();
  });
}

async function runQueue() {
  queueRunning.value = true;
  const pending = [...queue.value];
  const workers = Array.from({ length: QUEUE_CONCURRENCY }, async () => {
    while (pending.length > 0) {
      const item = pending.shift();
      if (item) await runOne(item);
    }
  });
  await Promise.all(workers);
  queueRunning.value = false;
  await load();
}

function clearQueue() {
  queue.value = [];
}

const QUEUE_STATUS_LABEL: Record<QueueStatus, string> = {
  queued: "Queued",
  ocr: "Reading…",
  draft: "Drafting…",
  review: "Reviewing…",
  done: "Done",
  error: "Failed",
};

onMounted(load);
</script>

<template>
  <div class="mx-auto max-w-3xl px-8 py-8">
    <div class="mb-5 flex items-center justify-between">
      <div>
        <h1 class="font-display text-lg font-semibold text-text">Deliverables</h1>
        <p class="mt-1 text-[12px] text-dim">Finished .docx/.xlsx/.pptx files the agent has produced.</p>
      </div>
      <Button variant="ghost" :disabled="loading" @click="load">Refresh</Button>
    </div>

    <p v-if="errorMessage" class="mb-4 rounded-md border border-danger/30 bg-danger/10 px-3 py-2 text-[12.5px] text-danger">
      {{ errorMessage }}
    </p>

    <Card title="Batch upload (scanned inspection reports)" class="mb-4">
      <p class="mb-3 text-[12px] text-dim">
        Queue multiple scanned/photographed reports through Phase 4's OCR → draft → review pipeline at once — each
        runs the full grounded, reviewed .docx flow independently.
      </p>
      <div class="flex items-center gap-2">
        <input ref="uploadInput" type="file" accept="image/*" multiple class="hidden" @change="onQueueFilesChosen" />
        <Button variant="secondary" :disabled="queueRunning" @click="uploadInput?.click()">Choose images…</Button>
        <Button v-if="queue.length > 0 && queueDone" variant="ghost" @click="clearQueue">Clear</Button>
      </div>
      <div v-if="queue.length > 0" class="mt-3 flex flex-col gap-1.5">
        <div
          v-for="item in queue"
          :key="item.id"
          class="flex items-center justify-between gap-2 rounded-md border border-border-soft bg-panel-2 px-3 py-1.5 text-[11.5px]"
        >
          <span class="truncate font-mono text-dim">{{ item.name }}</span>
          <div class="flex items-center gap-2">
            <span v-if="item.status === 'error'" class="text-danger">{{ item.errorMessage || "Failed" }}</span>
            <StatusPill :status="item.status === 'done' ? 'ok' : item.status === 'error' ? 'danger' : 'warn'">
              {{ QUEUE_STATUS_LABEL[item.status] }}
            </StatusPill>
          </div>
        </div>
      </div>
    </Card>

    <Card title="Outputs">
      <template #actions>
        <div class="flex items-center gap-2">
          <template v-if="selected.size > 0">
            <span class="text-[11.5px] text-dim">{{ selected.size }} selected</span>
            <Button variant="secondary" :disabled="bulkBusy" @click="exportSelected">Export selected</Button>
            <Button variant="ghost" :disabled="bulkBusy" @click="deleteSelected">Delete selected</Button>
          </template>
          <StatusPill status="neutral">{{ items.length }} file{{ items.length === 1 ? "" : "s" }}</StatusPill>
        </div>
      </template>
      <VirtualDataTable
        :items="items"
        :columns="columns"
        height="360px"
        :row-height="44"
        selectable
        row-key="filename"
        v-model:selected="selected"
      >
        <template #cell-filename="{ item }">
          <span class="truncate font-mono text-[11.5px]">{{ item.filename }}</span>
        </template>
        <template #cell-size="{ item }">{{ formatSize(item.size_bytes) }}</template>
        <template #cell-created="{ item }">{{ formatDate(item.created) }}</template>
        <template #cell-download="{ item }">
          <a
            :href="`${API_BASE}/deliverables/${item.filename}`"
            class="text-[11.5px] font-medium text-accent hover:underline"
          >
            Download
          </a>
        </template>
      </VirtualDataTable>
    </Card>
  </div>
</template>
