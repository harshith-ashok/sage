<script setup lang="ts">
import Card from "../components/base/Card.vue";
import StatusPill from "../components/base/StatusPill.vue";
</script>

<template>
  <div class="mx-auto max-w-3xl px-8 py-8">
    <div class="mb-5 flex items-center justify-between">
      <h1 class="font-display text-lg font-semibold text-text">Network Monitor</h1>
      <StatusPill status="neutral">Phase 12</StatusPill>
    </div>
    <Card title="Zero-egress proof">
      <p class="text-[12.5px] text-dim">
        The large "0 external calls" live indicator and connection log land in Phase 12, backed by
        <code class="font-mono text-dim-2">GET /system/network</code>.
      </p>
    </Card>
  </div>
</template>
