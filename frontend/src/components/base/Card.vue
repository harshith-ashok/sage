<script setup lang="ts">
defineProps<{ title?: string }>();
</script>

<template>
  <div class="rounded-xl border border-border bg-panel" style="box-shadow: var(--shadow-card)">
    <div v-if="title || $slots.actions" class="flex items-center justify-between border-b border-border-soft px-4 py-3">
      <h3 v-if="title" class="font-display text-[13px] font-semibold text-text">{{ title }}</h3>
      <div v-if="$slots.actions" class="flex items-center gap-2">
        <slot name="actions" />
      </div>
    </div>
    <div class="p-4">
      <slot />
    </div>
  </div>
</template>
