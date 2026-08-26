<script setup lang="ts">
// Renders text as it accumulates from an SSE token stream (see lib/api.ts's
// streamSSE) — the caller owns the accumulation, this just shows a caret
// while `streaming` is true so partial output doesn't read as "finished".
withDefaults(defineProps<{ text: string; streaming?: boolean }>(), { streaming: false });
</script>

<template>
  <span class="whitespace-pre-wrap font-sans text-[13px] leading-relaxed text-text">{{ text
    }}<span v-if="streaming" class="animate-pulse text-accent">▍</span></span>
</template>
