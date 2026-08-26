<script setup lang="ts">
import { ref } from "vue";
import Button from "../components/base/Button.vue";
import Card from "../components/base/Card.vue";
import StatusPill from "../components/base/StatusPill.vue";
import { apiGet } from "../lib/api";

interface Candidate {
  text: string;
  rrf_score: number;
  cross_score: number | null;
  meta: { source: string; page: number; section: string };
}

interface Confidence {
  level: "high" | "low" | "unverifiable";
  unverified_claims: number;
  total_claims: number;
  top_retrieval_score: number | null;
}

interface SearchResult {
  query: string;
  error: string | null;
  answer: string | null;
  gated: boolean;
  context_candidates: Candidate[];
  confidence: Confidence | null;
}

const query = ref("");
const loading = ref(false);
const result = ref<SearchResult | null>(null);
const errorMessage = ref("");

async function search() {
  const trimmed = query.value.trim();
  if (!trimmed) return;
  loading.value = true;
  errorMessage.value = "";
  result.value = null;
  try {
    result.value = await apiGet<SearchResult>(`/knowledge/search?query=${encodeURIComponent(trimmed)}`);
  } catch (e) {
    errorMessage.value = e instanceof Error ? e.message : String(e);
  } finally {
    loading.value = false;
  }
}

function confidenceStatus(level: Confidence["level"] | undefined) {
  if (level === "high") return "ok";
  if (level === "low") return "warn";
  return "neutral";
}
</script>

<template>
  <div class="mx-auto max-w-3xl px-8 py-8">
    <h1 class="mb-1 font-display text-lg font-semibold text-text">Knowledge Base</h1>
    <p class="mb-5 text-[12px] text-dim">Hybrid RAG over the ingested SOP corpus (Phase 3).</p>

    <form class="mb-5 flex gap-2" @submit.prevent="search">
      <input
        v-model="query"
        type="text"
        placeholder="Ask about an SOP…"
        class="flex-1 rounded-md border border-border bg-panel-2 px-3 py-2 text-[12.5px] text-text placeholder:text-dim-2 focus:border-accent focus:outline-none"
      />
      <Button type="submit" variant="primary" :disabled="loading || !query.trim()">
        {{ loading ? "Searching…" : "Search" }}
      </Button>
    </form>

    <p v-if="errorMessage" class="mb-4 rounded-md border border-danger/30 bg-danger/10 px-3 py-2 text-[12.5px] text-danger">
      {{ errorMessage }}
    </p>

    <div v-if="result" class="flex flex-col gap-4">
      <p v-if="result.error" class="rounded-md border border-danger/30 bg-danger/10 px-3 py-2 text-[12.5px] text-danger">
        {{ result.error }}
      </p>

      <Card v-else title="Answer">
        <template #actions>
          <StatusPill :status="confidenceStatus(result.confidence?.level)">
            {{ result.confidence?.level ?? "unknown" }} confidence
          </StatusPill>
        </template>
        <p class="whitespace-pre-wrap text-[13px] leading-relaxed text-text">{{ result.answer }}</p>
      </Card>

      <Card v-if="result.context_candidates.length" title="Grounding">
        <div class="flex flex-col gap-3">
          <div v-for="(c, i) in result.context_candidates" :key="i" class="rounded-md border border-border-soft p-3">
            <div class="mb-1.5 flex items-center gap-2 font-mono text-[10.5px] text-dim-2">
              <span>{{ c.meta.source }}</span>
              <span>·</span>
              <span>page {{ c.meta.page }}</span>
              <span>·</span>
              <span>section {{ c.meta.section }}</span>
              <span v-if="c.cross_score !== null" class="ml-auto">score {{ c.cross_score.toFixed(2) }}</span>
            </div>
            <p class="line-clamp-4 text-[12px] leading-relaxed text-dim">{{ c.text }}</p>
          </div>
        </div>
      </Card>
    </div>
  </div>
</template>
