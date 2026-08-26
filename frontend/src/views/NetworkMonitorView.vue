<script setup lang="ts">
import { onMounted, onUnmounted, ref } from "vue";
import Card from "../components/base/Card.vue";
import StatusPill from "../components/base/StatusPill.vue";
import { apiGet } from "../lib/api";

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
const errorMessage = ref("");
let pollHandle: ReturnType<typeof setInterval> | null = null;

async function poll() {
  try {
    snapshot.value = await apiGet<NetworkSnapshot>("/system/network");
    errorMessage.value = "";
  } catch (e) {
    errorMessage.value = e instanceof Error ? e.message : String(e);
  }
}

function formatTime(iso: string): string {
  return new Date(iso).toLocaleTimeString();
}

onMounted(() => {
  poll();
  pollHandle = setInterval(poll, 3000);
});
onUnmounted(() => {
  if (pollHandle !== null) clearInterval(pollHandle);
});
</script>

<template>
  <div class="mx-auto max-w-3xl px-8 py-8">
    <div class="mb-5 flex items-center justify-between">
      <div>
        <h1 class="font-display text-lg font-semibold text-text">Network Monitor</h1>
        <p class="mt-1 text-[12px] text-dim">
          Live proof of zero egress — polls open sockets on the backend and the local Ollama daemon every 3s.
        </p>
      </div>
      <StatusPill status="neutral">polling</StatusPill>
    </div>

    <p v-if="errorMessage" class="mb-4 rounded-md border border-danger/30 bg-danger/10 px-3 py-2 text-[12.5px] text-danger">
      {{ errorMessage }}
    </p>

    <div class="mb-5 flex flex-col items-center justify-center rounded-lg border py-10" :class="snapshot && snapshot.external_count > 0 ? 'border-danger/40 bg-danger/5' : 'border-ok/30 bg-ok/5'">
      <div class="font-display text-5xl font-bold" :class="snapshot && snapshot.external_count > 0 ? 'text-danger' : 'text-ok'">
        {{ snapshot ? snapshot.external_count : "…" }}
      </div>
      <div class="mt-2 text-[12.5px] uppercase tracking-wide" :class="snapshot && snapshot.external_count > 0 ? 'text-danger' : 'text-ok'">
        external call{{ snapshot?.external_count === 1 ? "" : "s" }} right now
      </div>
      <div class="mt-1 text-[11px] text-dim-2">{{ snapshot?.local_connection_count ?? 0 }} local connections (loopback/LAN — not egress)</div>
    </div>

    <Card title="Watched processes" class="mb-4">
      <div class="flex flex-wrap gap-2">
        <span
          v-for="p in snapshot?.watched_processes ?? []"
          :key="p"
          class="rounded-md border border-border-soft bg-panel-2 px-2 py-1 font-mono text-[10.5px] text-dim"
        >
          {{ p }}
        </span>
      </div>
      <p class="mt-2 text-[10.5px] text-dim-2">
        A cloud model's own inference call happens inside the Ollama daemon's process, not this backend's — both are
        watched so an active cloud candidate can't hide from this count.
      </p>
    </Card>

    <Card v-if="snapshot && snapshot.external_connections.length > 0" title="Active external connections" class="mb-4">
      <div class="flex flex-col gap-1.5">
        <div
          v-for="c in snapshot.external_connections"
          :key="`${c.pid}-${c.remote}`"
          class="rounded-md border border-danger/30 bg-danger/5 px-2 py-1.5 font-mono text-[11.5px] text-text"
        >
          {{ c.process }} (pid {{ c.pid }}) → {{ c.remote }}
        </div>
      </div>
    </Card>

    <Card title="External connection log">
      <p v-if="!snapshot || snapshot.external_log.length === 0" class="text-[12.5px] text-dim">
        Nothing logged yet — no external connection has been observed since this backend started.
      </p>
      <div v-else class="flex flex-col gap-1.5">
        <div
          v-for="(c, i) in snapshot.external_log"
          :key="i"
          class="rounded-md border border-border-soft bg-panel-2 px-2 py-1.5 text-[11.5px]"
        >
          <span class="font-mono text-dim-2">{{ formatTime(c.timestamp) }}</span>
          <span class="mx-1.5 text-dim">·</span>
          <span class="font-mono text-text">{{ c.process }} (pid {{ c.pid }})</span>
          <span class="mx-1.5 text-dim">→</span>
          <span class="font-mono text-danger">{{ c.remote }}</span>
        </div>
      </div>
    </Card>
  </div>
</template>
