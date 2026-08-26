<script setup lang="ts">
import { computed, onMounted, onUnmounted } from "vue";
import { highlightCode } from "../../lib/markdown";

const props = defineProps<{ code: string; language: string }>();
const emit = defineEmits<{ close: [] }>();

const highlighted = computed(() => highlightCode(props.code, props.language));

function onKeydown(e: KeyboardEvent) {
  if (e.key === "Escape") emit("close");
}
onMounted(() => window.addEventListener("keydown", onKeydown));
onUnmounted(() => window.removeEventListener("keydown", onKeydown));
</script>

<template>
  <div class="fixed inset-0 z-50 flex items-center justify-center bg-black/60 p-6" @click.self="$emit('close')">
    <div class="flex max-h-[80vh] w-full max-w-3xl flex-col overflow-hidden rounded-lg border border-border bg-panel shadow-2xl">
      <div class="flex shrink-0 items-center justify-between border-b border-border-soft px-4 py-2.5">
        <span class="font-mono text-[11px] uppercase tracking-wide text-dim">{{ language }}</span>
        <button class="text-dim hover:text-text" @click="$emit('close')">✕</button>
      </div>
      <div class="overflow-auto p-4">
        <pre class="!m-0"><code class="hljs font-mono text-[12.5px] leading-relaxed" v-html="highlighted" /></pre>
      </div>
    </div>
  </div>
</template>
