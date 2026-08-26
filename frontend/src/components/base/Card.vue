<script setup lang="ts">
defineProps<{ title?: string }>();
</script>

<template>
  <div class="rounded-lg border border-border bg-panel">
    <div v-if="title || $slots.actions" class="flex items-center justify-between border-b border-border-soft px-4 py-2.5">
      <h3 v-if="title" class="font-display text-[13px] font-medium text-text">{{ title }}</h3>
      <div v-if="$slots.actions" class="flex items-center gap-2">
        <slot name="actions" />
      </div>
    </div>
    <div class="p-4">
      <slot />
    </div>
  </div>
</template>
