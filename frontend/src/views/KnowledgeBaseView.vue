<script setup lang="ts">
import { computed, onMounted, ref } from "vue";
import Button from "../components/base/Button.vue";
import Card from "../components/base/Card.vue";
import StatusPill from "../components/base/StatusPill.vue";
import KnowledgeGraph, { type GraphData, type Selection } from "../components/knowledge/KnowledgeGraph.vue";
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

const graph = ref<GraphData | null>(null);
const graphLoading = ref(false);
const graphError = ref("");
const selection = ref<Selection>(null);

async function loadGraph() {
  graphLoading.value = true;
  graphError.value = "";
  try {
    graph.value = await apiGet<GraphData>("/knowledge/graph");
  } catch (e) {
    graphError.value = e instanceof Error ? e.message : String(e);
  } finally {
    graphLoading.value = false;
  }
}

async function search() {
  const trimmed = query.value.trim();
  if (!trimmed) return;
  loading.value = true;
  errorMessage.value = "";
  result.value = null;
  selection.value = null;
  try {
    result.value = await apiGet<SearchResult>(`/knowledge/search?query=${encodeURIComponent(trimmed)}`);
  } catch (e) {
    errorMessage.value = e instanceof Error ? e.message : String(e);
  } finally {
    loading.value = false;
  }
}

// "source" alone highlights the whole document node; "source::section"
// highlights that specific section — see KnowledgeGraph.vue's isHighlighted().
const highlightedSections = computed(() => {
  const set = new Set<string>();
  for (const c of result.value?.context_candidates ?? []) {
    set.add(`${c.meta.source}::${c.meta.section}`);
  }
  return set;
});

function confidenceStatus(level: Confidence["level"] | undefined) {
  if (level === "high") return "ok";
  if (level === "low") return "warn";
  return "neutral";
}

function onSelect(sel: Selection) {
  selection.value = sel;
}

onMounted(loadGraph);
</script>

<template>
  <div class="mx-auto max-w-6xl px-8 py-8">
    <div class="mb-5 flex items-center justify-between">
      <div>
        <h1 class="font-display text-lg font-semibold text-text">Knowledge Base</h1>
        <p class="mt-1 text-[12px] text-dim">Hybrid RAG over the ingested SOP corpus — search, or explore the corpus graph below.</p>
      </div>
      <Button variant="ghost" :disabled="graphLoading" @click="loadGraph">Refresh graph</Button>
    </div>

    <form class="mb-5 flex gap-2" @submit.prevent="search">
      <input
        v-model="query"
        type="text"
        placeholder="Ask about an SOP…"
        class="flex-1 rounded-md border border-border bg-panel px-3 py-2 text-[12.5px] text-text placeholder:text-dim-2 focus:border-accent focus:outline-none"
      />
      <Button type="submit" variant="primary" :disabled="loading || !query.trim()">
        {{ loading ? "Searching…" : "Search" }}
      </Button>
    </form>

    <p v-if="errorMessage" class="mb-4 rounded-md border border-danger/30 bg-danger/10 px-3 py-2 text-[12.5px] text-danger">
      {{ errorMessage }}
    </p>
    <p v-if="graphError" class="mb-4 rounded-md border border-danger/30 bg-danger/10 px-3 py-2 text-[12.5px] text-danger">
      {{ graphError }}
    </p>

    <div class="mb-6 grid grid-cols-1 gap-4 lg:grid-cols-[1.6fr_1fr]">
      <Card title="Corpus graph">
        <template #actions>
          <StatusPill v-if="graph" status="neutral">{{ graph.documents.length }} docs · {{ graph.total_chunks }} chunks</StatusPill>
        </template>
        <div class="flex aspect-square w-full items-center justify-center">
          <p v-if="graphLoading" class="text-[12.5px] text-dim-2">Loading corpus…</p>
          <p v-else-if="graph && graph.documents.length === 0" class="text-[12.5px] text-dim-2">Nothing ingested yet.</p>
          <KnowledgeGraph
            v-else-if="graph"
            :data="graph"
            :highlighted-sections="highlightedSections"
            :query-label="result?.query ?? null"
            @select="onSelect"
          />
        </div>
      </Card>

      <Card :title="selection ? (selection.kind === 'document' ? 'Document' : 'Section') : 'Corpus overview'">
        <div v-if="!selection" class="flex flex-col gap-2">
          <p class="text-[12px] text-dim">Click a node in the graph to inspect it. Search results highlight where an answer's grounding came from.</p>
          <div v-if="graph" class="mt-1 flex flex-col gap-1.5">
            <div v-for="d in graph.documents" :key="d.source" class="flex items-center justify-between rounded-md border border-border-soft px-2.5 py-1.5 text-[11.5px]">
              <span class="truncate text-text">{{ d.title }}</span>
              <span class="ml-2 shrink-0 font-mono text-dim-2">{{ d.chunk_count }}</span>
            </div>
          </div>
        </div>

        <div v-else-if="selection.kind === 'document'" class="flex flex-col gap-2">
          <div class="font-medium text-text">{{ selection.document.title }}</div>
          <div class="font-mono text-[10.5px] text-dim-2">{{ selection.document.source }}</div>
          <div class="text-[11.5px] text-dim">{{ selection.document.sections.length }} sections · {{ selection.document.chunk_count }} chunks</div>
          <div class="mt-1 flex flex-col gap-1">
            <button
              v-for="s in selection.document.sections"
              :key="s.section"
              class="flex items-center justify-between rounded-md border border-border-soft px-2.5 py-1.5 text-left text-[11.5px] hover:border-accent/40 hover:bg-accent/5"
              @click="onSelect({ kind: 'section', document: selection.document, section: s })"
            >
              <span class="text-text">Section {{ s.section }}</span>
              <span class="font-mono text-dim-2">{{ s.chunk_count }}</span>
            </button>
          </div>
        </div>

        <div v-else class="flex flex-col gap-2">
          <div class="font-medium text-text">{{ selection.document.title }}</div>
          <div class="text-[11.5px] text-dim">Section {{ selection.section.section }} · {{ selection.section.chunk_count }} chunk{{ selection.section.chunk_count === 1 ? "" : "s" }}</div>
          <div class="mt-1 flex flex-col gap-2">
            <div v-for="c in selection.section.chunks" :key="c.chunk_id" class="rounded-md border border-border-soft p-2.5">
              <div class="mb-1 font-mono text-[10px] text-dim-2">page {{ c.page }}</div>
              <p class="text-[11.5px] leading-relaxed text-dim">{{ c.preview }}…</p>
            </div>
          </div>
        </div>
      </Card>
    </div>

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
