<script setup lang="ts">
import { computed } from "vue";
import { renderMarkdown } from "../../lib/markdown";

const props = defineProps<{ content: string }>();
const emit = defineEmits<{ viewCode: [code: string, language: string] }>();

const html = computed(() => renderMarkdown(props.content));

// v-html content isn't part of Vue's reactive DOM tree, so code-block clicks
// are caught here via delegation rather than a per-block @click binding.
function onClick(e: MouseEvent) {
  const pre = (e.target as HTMLElement).closest("pre");
  if (!pre) return;
  const codeEl = pre.querySelector("code");
  if (!codeEl) return;
  const lang = /language-(\S+)/.exec(codeEl.className)?.[1] ?? "plaintext";
  emit("viewCode", codeEl.textContent ?? "", lang);
}
</script>

<template>
  <div class="markdown-body" v-html="html" @click="onClick" />
</template>
