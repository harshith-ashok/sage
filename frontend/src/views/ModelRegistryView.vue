<script setup lang="ts">
import { onMounted, onUnmounted } from "vue";
import Button from "../components/base/Button.vue";
import Card from "../components/base/Card.vue";
import StatusPill from "../components/base/StatusPill.vue";
import { useModelsStore } from "../stores/models";

const store = useModelsStore();
onMounted(() => {
  store.fetchRegistry();
  store.startLoadPolling();
});
onUnmounted(() => store.stopLoadPolling());

function loadPillStatus(level: string | undefined): "ok" | "warn" | "danger" | "neutral" {
  if (level === "high") return "danger";
  if (level === "elevated") return "warn";
  if (level === "normal") return "ok";
  return "neutral";
}

function formatBytes(bytes: number): string {
  const gb = bytes / 1024 ** 3;
  return gb >= 0.1 ? `${gb.toFixed(1)} GB` : `${(bytes / 1024 ** 2).toFixed(0)} MB`;
}

function formatTime(iso: string): string {
  return new Date(iso).toLocaleTimeString();
}
</script>

<template>
  <div class="mx-auto max-w-3xl px-8 py-8">
    <div class="mb-5 flex items-center justify-between">
      <div>
        <h1 class="font-display text-lg font-semibold text-text">Model Registry</h1>
        <p class="mt-1 text-[12px] text-dim">
          Switching here edits <code class="font-mono text-dim-2">models.yaml</code> live — no restart, no code
          touched.
        </p>
      </div>
      <Button variant="ghost" :disabled="store.loading" @click="store.fetchRegistry()">Refresh</Button>
    </div>

    <p v-if="store.error" class="mb-4 rounded-md border border-danger/30 bg-danger/10 px-3 py-2 text-[12.5px] text-danger">
      {{ store.error }}
    </p>

    <Card title="System load" class="mb-4">
      <div class="flex flex-col gap-3">
        <div class="flex items-center gap-2">
          <StatusPill :status="loadPillStatus(store.load?.level)">
            {{ store.load ? store.load.level : "checking…" }}
          </StatusPill>
          <span v-if="store.load" class="font-mono text-[11px] text-dim-2">
            {{ store.load.available_percent.toFixed(0) }}% memory free
          </span>
          <span class="text-[10.5px] text-dim-2">
            — polled every 5s; local candidates only fall back under "high" load (see Model Registry docs in
            CLAUDE.md Phase 11)
          </span>
        </div>

        <p v-if="store.loadError" class="text-[12px] text-danger">{{ store.loadError }}</p>

        <div v-if="store.load && store.load.loaded_models.length > 0">
          <div class="mb-1 text-[10.5px] uppercase tracking-wide text-dim-2">Currently loaded (Ollama)</div>
          <div class="flex flex-wrap gap-2">
            <span
              v-for="m in store.load.loaded_models"
              :key="m.model_id"
              class="rounded-md border border-border-soft bg-panel-2 px-2 py-1 font-mono text-[10.5px] text-dim"
            >
              {{ m.model_id }} · {{ formatBytes(m.size_vram_bytes) }}
            </span>
          </div>
        </div>

        <div v-if="store.load && store.load.fallback_log.length > 0">
          <div class="mb-1 text-[10.5px] uppercase tracking-wide text-dim-2">Recent fallbacks</div>
          <div class="flex flex-col gap-1.5">
            <div
              v-for="(e, i) in store.load.fallback_log"
              :key="i"
              class="rounded-md border px-2 py-1.5 text-[11.5px]"
              :class="e.to_model_id ? 'border-warn/30 bg-warn/5' : 'border-danger/30 bg-danger/5'"
            >
              <span class="font-mono text-dim-2">{{ formatTime(e.timestamp) }}</span>
              <span class="mx-1.5 text-dim">·</span>
              <span class="font-mono text-text">{{ e.task_type }}</span>
              <template v-if="e.to_model_id">
                <span class="mx-1.5 text-dim">fell back</span>
                <span class="font-mono text-dim-2">{{ e.from_model_id }}</span>
                <span class="mx-1 text-dim">→</span>
                <span class="font-mono text-text">{{ e.to_model_id }}</span>
              </template>
              <template v-else>
                <span class="mx-1.5 text-danger">no safe fallback for</span>
                <span class="font-mono text-dim-2">{{ e.from_model_id }}</span>
              </template>
              <div class="mt-0.5 text-[10.5px] text-dim-2">{{ e.reason }}</div>
            </div>
          </div>
        </div>
      </div>
    </Card>

    <div class="flex flex-col gap-4">
      <Card v-for="(cfg, taskType) in store.registry" :key="taskType" :title="String(taskType)">
        <div class="flex flex-col gap-2">
          <div
            v-for="(candidate, key) in cfg.candidates"
            :key="key"
            class="flex items-center justify-between gap-3 rounded-md border px-3 py-2"
            :class="key === cfg.active ? 'border-accent/40 bg-accent/5' : 'border-border-soft'"
          >
            <div class="min-w-0">
              <div class="flex items-center gap-2">
                <span class="truncate font-mono text-[12px] text-text">{{ candidate.model_id }}</span>
                <StatusPill :status="candidate.location === 'local' ? 'ok' : 'warn'">
                  {{ candidate.location }}
                </StatusPill>
                <StatusPill :status="candidate.locally_available ? 'ok' : 'neutral'">
                  {{ candidate.locally_available ? "pulled" : "not pulled" }}
                </StatusPill>
                <StatusPill v-if="key === cfg.active" status="ok">active</StatusPill>
              </div>
              <div class="mt-0.5 font-mono text-[10.5px] text-dim-2">
                context window: {{ candidate.context_window.toLocaleString() }}
              </div>
            </div>
            <Button
              v-if="key !== cfg.active"
              variant="secondary"
              :disabled="store.switching === taskType"
              @click="store.setActive(String(taskType), String(key))"
            >
              {{ store.switching === taskType ? "Switching…" : "Set active" }}
            </Button>
          </div>
        </div>
      </Card>

      <p v-if="!store.loading && Object.keys(store.registry).length === 0 && !store.error" class="text-[12.5px] text-dim">
        No task types found.
      </p>
    </div>
  </div>
</template>
