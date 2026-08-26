<script setup lang="ts">
withDefaults(defineProps<{ status?: "ok" | "warn" | "danger" | "neutral" }>(), { status: "neutral" });
</script>

<template>
  <span
    class="inline-flex items-center gap-1.5 rounded-full border px-2 py-0.5 font-mono text-[10.5px] uppercase tracking-wide"
    :class="{
      'border-ok/30 bg-ok/10 text-ok': status === 'ok',
      'border-warn/30 bg-warn/10 text-warn': status === 'warn',
      'border-danger/30 bg-danger/10 text-danger': status === 'danger',
      'border-border bg-panel-2 text-dim': status === 'neutral',
    }"
  >
    <span
      class="h-1.5 w-1.5 rounded-full"
      :class="{
        'bg-ok': status === 'ok',
        'bg-warn': status === 'warn',
        'bg-danger': status === 'danger',
        'bg-dim-2': status === 'neutral',
      }"
    />
    <slot />
  </span>
</template>
