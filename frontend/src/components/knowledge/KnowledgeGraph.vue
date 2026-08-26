<script setup lang="ts">
import { computed } from "vue";

interface ChunkNode {
  chunk_id: string;
  page: number;
  preview: string;
}
interface SectionNode {
  section: string;
  chunk_count: number;
  chunks: ChunkNode[];
}
interface DocumentNode {
  source: string;
  title: string;
  chunk_count: number;
  sections: SectionNode[];
}
export interface GraphData {
  documents: DocumentNode[];
  total_chunks: number;
}

export type Selection =
  | { kind: "document"; document: DocumentNode }
  | { kind: "section"; document: DocumentNode; section: SectionNode }
  | null;

const props = defineProps<{
  data: GraphData;
  // Sources/sections a live search matched — "source" alone highlights the
  // whole document node, "source::section" highlights that section node
  // specifically. A plain SVG line is drawn from the query pseudo-node to
  // every highlighted section so it's visually obvious *where* an answer's
  // grounding actually came from in the corpus, not just that it matched.
  highlightedSections?: Set<string>;
  queryLabel?: string | null;
}>();
const emit = defineEmits<{ select: [Selection] }>();

const SIZE = 640;
const CENTER = SIZE / 2;
const DOC_RADIUS = 150;
const SECTION_RADIUS = 235;
const QUERY_Y = 26;

const layout = computed(() => {
  const docs = props.data.documents;
  const n = Math.max(docs.length, 1);
  return docs.map((doc, i) => {
    const angle = (i / n) * Math.PI * 2 - Math.PI / 2;
    const x = CENTER + DOC_RADIUS * Math.cos(angle);
    const y = CENTER + DOC_RADIUS * Math.sin(angle);
    const r = Math.min(10 + doc.chunk_count * 1.8, 26);

    const sectionCount = Math.max(doc.sections.length, 1);
    // Sections fan out in a small arc centered on the document's own
    // angle, rather than a full circle — keeps each document's sections
    // visually grouped near their parent instead of scattered.
    const arc = Math.min(0.16 * sectionCount, Math.PI * 1.8) / n;
    const sections = doc.sections.map((section, j) => {
      const sAngle = angle - arc / 2 + (sectionCount === 1 ? arc / 2 : (arc * j) / (sectionCount - 1));
      return {
        section,
        x: CENTER + SECTION_RADIUS * Math.cos(sAngle),
        y: CENTER + SECTION_RADIUS * Math.sin(sAngle),
        r: Math.min(5 + section.chunk_count * 1.3, 11),
      };
    });

    return { doc, x, y, r, angle, sections };
  });
});

function isHighlighted(source: string, section?: string): boolean {
  if (!props.highlightedSections) return false;
  if (section && props.highlightedSections.has(`${source}::${section}`)) return true;
  return props.highlightedSections.has(source);
}

function docIsHighlighted(source: string): boolean {
  if (!props.highlightedSections) return false;
  for (const key of props.highlightedSections) {
    if (key === source || key.startsWith(`${source}::`)) return true;
  }
  return false;
}
</script>

<template>
  <svg :viewBox="`0 0 ${SIZE} ${SIZE}`" class="h-full w-full select-none" role="img" aria-label="Knowledge base corpus graph">
    <!-- center -> document edges -->
    <g stroke="var(--color-border)" stroke-width="1">
      <line v-for="l in layout" :key="`e-${l.doc.source}`" :x1="CENTER" :y1="CENTER" :x2="l.x" :y2="l.y" />
    </g>
    <!-- document -> section edges -->
    <g stroke="var(--color-border-soft)" stroke-width="1">
      <template v-for="l in layout" :key="`se-${l.doc.source}`">
        <line v-for="s in l.sections" :key="`se-${l.doc.source}-${s.section.section}`" :x1="l.x" :y1="l.y" :x2="s.x" :y2="s.y" />
      </template>
    </g>

    <!-- query pseudo-node + highlight edges, only while a search's results are showing -->
    <template v-if="queryLabel && highlightedSections && highlightedSections.size > 0">
      <g stroke="var(--color-accent)" stroke-width="1.4" stroke-dasharray="3 3" opacity="0.7">
        <template v-for="l in layout" :key="`h-${l.doc.source}`">
          <template v-for="s in l.sections" :key="`h-${l.doc.source}-${s.section.section}`">
            <line v-if="isHighlighted(l.doc.source, s.section.section)" :x1="CENTER" :y1="QUERY_Y" :x2="s.x" :y2="s.y" />
          </template>
        </template>
      </g>
      <circle :cx="CENTER" :cy="QUERY_Y" r="7" fill="var(--color-accent)" />
      <text :x="CENTER" :y="QUERY_Y - 12" text-anchor="middle" font-size="10.5" fill="var(--color-accent)" font-family="var(--font-mono)">
        query
      </text>
    </template>

    <!-- center node -->
    <circle :cx="CENTER" :cy="CENTER" r="20" fill="var(--color-panel-2)" stroke="var(--color-border)" stroke-width="1.5" />
    <text :x="CENTER" :y="CENTER + 4" text-anchor="middle" font-size="9" font-family="var(--font-mono)" fill="var(--color-dim)">KB</text>

    <!-- document + section nodes -->
    <g v-for="l in layout" :key="l.doc.source">
      <g v-for="s in l.sections" :key="s.section.section" class="cursor-pointer" @click="emit('select', { kind: 'section', document: l.doc, section: s.section })">
        <circle
          :cx="s.x"
          :cy="s.y"
          :r="s.r"
          :fill="isHighlighted(l.doc.source, s.section.section) ? 'var(--color-accent)' : 'var(--color-panel-2)'"
          :stroke="isHighlighted(l.doc.source, s.section.section) ? 'var(--color-accent)' : 'var(--color-border)'"
          stroke-width="1.3"
        >
          <title>{{ l.doc.title }} · Section {{ s.section.section }} ({{ s.section.chunk_count }} chunk{{ s.section.chunk_count === 1 ? "" : "s" }})</title>
        </circle>
      </g>

      <circle
        :cx="l.x"
        :cy="l.y"
        :r="l.r"
        :fill="docIsHighlighted(l.doc.source) ? 'var(--color-accent)' : 'var(--color-panel)'"
        :stroke="docIsHighlighted(l.doc.source) ? 'var(--color-accent)' : 'var(--color-border)'"
        stroke-width="1.6"
        class="cursor-pointer transition-colors"
        @click="emit('select', { kind: 'document', document: l.doc })"
      >
        <title>{{ l.doc.title }} ({{ l.doc.chunk_count }} chunks)</title>
      </circle>
      <text
        :x="l.x"
        :y="l.y + l.r + 13"
        text-anchor="middle"
        font-size="9.5"
        font-family="var(--font-sans)"
        :fill="docIsHighlighted(l.doc.source) ? 'var(--color-accent)' : 'var(--color-dim)'"
        class="pointer-events-none"
      >
        {{ l.doc.title.length > 22 ? l.doc.title.slice(0, 21) + "…" : l.doc.title }}
      </text>
    </g>
  </svg>
</template>
