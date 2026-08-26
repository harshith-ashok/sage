<script setup lang="ts">
import { onMounted } from "vue";
import Button from "../components/base/Button.vue";
import Card from "../components/base/Card.vue";
import StatusPill from "../components/base/StatusPill.vue";
import { useModelsStore } from "../stores/models";

const store = useModelsStore();
onMounted(() => store.fetchRegistry());
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
