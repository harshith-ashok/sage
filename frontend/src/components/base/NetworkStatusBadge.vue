<script setup lang="ts">
import { onMounted, onUnmounted, ref } from "vue";
import { apiGet } from "../../lib/api";

interface ExternalConnection {
  pid: number;
  process: string;
  remote: string;
  timestamp: string;
}
interface NetworkSnapshot {
  external_count: number;
  external_connections: ExternalConnection[];
  local_connection_count: number;
  watched_processes: string[];
  external_log: ExternalConnection[];
}

const snapshot = ref<NetworkSnapshot | null>(null);
const open = ref(false);
const rootEl = ref<HTMLDivElement | null>(null);
let pollHandle: ReturnType<typeof setInterval> | null = null;

async function poll() {
  try {
    snapshot.value = await apiGet<NetworkSnapshot>("/system/network");
  } catch {
    // Silent — a small header badge shouldn't throw an error banner over a
    // transient poll failure; it just shows "…" until the next tick succeeds.
    snapshot.value = null;
  }
}

function onDocClick(e: MouseEvent) {
  if (open.value && rootEl.value && !rootEl.value.contains(e.target as Node)) open.value = false;
}

function formatTime(iso: string): string {
  return new Date(iso).toLocaleTimeString();
}

onMounted(() => {
  poll();
  pollHandle = setInterval(poll, 3000);
  document.addEventListener("click", onDocClick);
});
onUnmounted(() => {
  if (pollHandle !== null) clearInterval(pollHandle);
  document.removeEventListener("click", onDocClick);
});
</script>

<template>
  <div ref="rootEl" class="relative">
    <button
      type="button"
      class="flex items-center gap-1.5 rounded-full border px-2.5 py-1 text-[11px] font-medium transition-colors"
      :class="
        snapshot && snapshot.external_count > 0
          ? 'border-danger/30 bg-danger/10 text-danger hover:bg-danger/15'
          : 'border-ok/30 bg-ok/10 text-ok hover:bg-ok/15'
      "
      :title="snapshot ? `${snapshot.external_count} external network call(s) right now` : 'Checking network…'"
      @click="open = !open"
    >
      <span
        class="h-1.5 w-1.5 rounded-full"
        :class="snapshot && snapshot.external_count > 0 ? 'bg-danger' : 'bg-ok'"
      />
      {{ snapshot ? snapshot.external_count : "…" }} external
    </button>

    <div
      v-if="open"
      class="absolute right-0 z-20 mt-2 w-72 rounded-lg border border-border bg-panel p-3 text-[11.5px]"
      style="box-shadow: var(--shadow-pop)"
    >
      <div class="mb-2 flex items-center justify-between">
        <span class="font-display text-[12px] font-semibold text-text">Network</span>
        <RouterLink to="/network" class="text-[10.5px] text-accent hover:underline" @click="open = false">Full view →</RouterLink>
      </div>

      <p v-if="!snapshot" class="text-dim-2">Checking…</p>
      <template v-else>
        <div class="mb-2 flex items-center gap-2">
          <span
            class="rounded-full px-2 py-0.5 font-mono text-[10px] uppercase tracking-wide"
            :class="snapshot.external_count > 0 ? 'bg-danger/10 text-danger' : 'bg-ok/10 text-ok'"
          >
            {{ snapshot.external_count > 0 ? "active egress" : "zero egress" }}
          </span>
          <span class="text-dim-2">{{ snapshot.local_connection_count }} local connections</span>
        </div>

        <div v-if="snapshot.external_connections.length > 0" class="mb-2 flex flex-col gap-1">
          <div
            v-for="c in snapshot.external_connections"
            :key="`${c.pid}-${c.remote}`"
            class="rounded-md border border-danger/30 bg-danger/5 px-2 py-1 font-mono text-[10.5px] text-text"
          >
            {{ c.process }} → {{ c.remote }}
          </div>
        </div>

        <div class="mb-1 text-[10px] uppercase tracking-wide text-dim-2">Watched</div>
        <div class="mb-2 flex flex-wrap gap-1">
          <span v-for="p in snapshot.watched_processes" :key="p" class="rounded border border-border-soft bg-panel-2 px-1.5 py-0.5 font-mono text-[9.5px] text-dim">
            {{ p }}
          </span>
        </div>

        <template v-if="snapshot.external_log.length > 0">
          <div class="mb-1 text-[10px] uppercase tracking-wide text-dim-2">Recent</div>
          <div class="flex max-h-24 flex-col gap-1 overflow-y-auto">
            <div v-for="(c, i) in snapshot.external_log.slice(0, 5)" :key="i" class="text-[10.5px] text-dim-2">
              <span class="font-mono">{{ formatTime(c.timestamp) }}</span> · {{ c.process }} → {{ c.remote }}
            </div>
          </div>
        </template>
      </template>
    </div>
  </div>
</template>
