<script setup lang="ts">
import { onMounted, ref } from "vue";
import Button from "../components/base/Button.vue";
import Card from "../components/base/Card.vue";
import StatusPill from "../components/base/StatusPill.vue";
import VirtualDataTable from "../components/base/VirtualDataTable.vue";
import { API_BASE, apiGet } from "../lib/api";

interface Deliverable {
  filename: string;
  size_bytes: number;
  created: number;
}

const items = ref<Deliverable[]>([]);
const loading = ref(false);
const errorMessage = ref("");

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
  } catch (e) {
    errorMessage.value = e instanceof Error ? e.message : String(e);
  } finally {
    loading.value = false;
  }
}

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

    <Card title="Outputs">
      <template #actions>
        <StatusPill status="neutral">{{ items.length }} file{{ items.length === 1 ? "" : "s" }}</StatusPill>
      </template>
      <VirtualDataTable :items="items" :columns="columns" height="360px" :row-height="44">
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
