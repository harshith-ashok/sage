<script setup lang="ts">
import { onBeforeUnmount, onMounted, ref, watch } from "vue";
import { DataSet } from "vis-data";
import { Network } from "vis-network";
import "vis-network/styles/vis-network.css";

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
  // "source" alone highlights the whole document node, "source::section"
  // highlights that specific section — see updateHighlights() below.
  highlightedSections?: Set<string>;
  queryLabel?: string | null;
}>();
const emit = defineEmits<{ select: [Selection] }>();

const container = ref<HTMLDivElement | null>(null);
let network: Network | null = null;
let nodes: DataSet<any> | null = null;
let edges: DataSet<any> | null = null;

// Theme tokens are CSS custom properties (see style.css) — read them at
// build time so node/edge colors follow whichever theme is active instead
// of hardcoding light- or dark-mode-only colors.
function cssVar(name: string): string {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim() || "#888";
}

const CATEGORY_ROOT = "__root__";

function buildGraph() {
  if (!container.value) return;

  const nodeItems: any[] = [
    { id: CATEGORY_ROOT, label: "Knowledge Base", shape: "dot", size: 26, color: cssVar("--color-panel-2"), font: { color: cssVar("--color-text"), size: 13 }, fixed: false },
  ];
  const edgeItems: any[] = [];

  for (const doc of props.data.documents) {
    const docId = `doc::${doc.source}`;
    nodeItems.push({
      id: docId,
      label: doc.title.length > 28 ? doc.title.slice(0, 27) + "…" : doc.title,
      shape: "dot",
      size: Math.min(14 + doc.chunk_count * 1.6, 30),
      color: { background: cssVar("--color-panel"), border: cssVar("--color-border"), highlight: { background: cssVar("--color-accent"), border: cssVar("--color-accent") } },
      font: { color: cssVar("--color-dim"), size: 11 },
      borderWidth: 2,
    });
    edgeItems.push({ id: `e::${docId}`, from: CATEGORY_ROOT, to: docId, color: cssVar("--color-border") });

    for (const section of doc.sections) {
      const secId = `sec::${doc.source}::${section.section}`;
      nodeItems.push({
        id: secId,
        label: `§${section.section}`,
        shape: "dot",
        size: Math.min(6 + section.chunk_count * 1.4, 14),
        color: { background: cssVar("--color-panel-2"), border: cssVar("--color-border") },
        font: { color: cssVar("--color-dim-2"), size: 9 },
        borderWidth: 1.5,
      });
      edgeItems.push({ id: `e::${secId}`, from: docId, to: secId, color: cssVar("--color-border-soft"), length: 60 });
    }
  }

  nodes = new DataSet(nodeItems);
  edges = new DataSet(edgeItems);

  network = new Network(
    container.value,
    { nodes, edges },
    {
      physics: {
        enabled: true,
        solver: "barnesHut",
        barnesHut: { gravitationalConstant: -3000, springLength: 90, springConstant: 0.04, damping: 0.4, avoidOverlap: 0.4 },
        stabilization: { iterations: 200 },
      },
      interaction: { hover: true, tooltipDelay: 150, dragNodes: true, zoomView: true },
      nodes: { shadow: false },
      edges: { smooth: { enabled: true, type: "continuous", roundness: 0.5 } },
    },
  );

  network.on("click", (params) => {
    if (params.nodes.length === 0) return;
    const id = params.nodes[0] as string;
    emitSelectionFor(id);
  });

  updateHighlights();
}

function emitSelectionFor(id: string) {
  if (id.startsWith("doc::")) {
    const source = id.slice("doc::".length);
    const doc = props.data.documents.find((d) => d.source === source);
    if (doc) emit("select", { kind: "document", document: doc });
  } else if (id.startsWith("sec::")) {
    const rest = id.slice("sec::".length);
    const lastSep = rest.lastIndexOf("::");
    const source = rest.slice(0, lastSep);
    const sectionKey = rest.slice(lastSep + 2);
    const doc = props.data.documents.find((d) => d.source === source);
    const section = doc?.sections.find((s) => s.section === sectionKey);
    if (doc && section) emit("select", { kind: "section", document: doc, section });
  }
}

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

const QUERY_NODE_ID = "__query__";

function updateHighlights() {
  if (!nodes || !edges) return;
  const accent = cssVar("--color-accent");
  const accentInk = cssVar("--color-accent-ink");
  const updates: any[] = [];
  const highlightedSecIds: string[] = [];
  for (const doc of props.data.documents) {
    const docId = `doc::${doc.source}`;
    const docHi = docIsHighlighted(doc.source);
    updates.push({
      id: docId,
      color: docHi
        ? { background: accent, border: accent }
        : { background: cssVar("--color-panel"), border: cssVar("--color-border") },
      font: { color: docHi ? accentInk : cssVar("--color-dim"), size: 11 },
    });
    for (const section of doc.sections) {
      const secId = `sec::${doc.source}::${section.section}`;
      const secHi = isHighlighted(doc.source, section.section);
      if (secHi) highlightedSecIds.push(secId);
      updates.push({
        id: secId,
        color: secHi
          ? { background: accent, border: accent }
          : { background: cssVar("--color-panel-2"), border: cssVar("--color-border") },
        font: { color: secHi ? accentInk : cssVar("--color-dim-2"), size: 9 },
      });
    }
  }
  nodes.update(updates);

  // A transient "query" node connected to whatever a live search actually
  // matched — added/removed dynamically rather than always present, so the
  // graph's default (no search yet) state stays a clean corpus overview.
  const queryEdgeIds = edges.getIds({ filter: (e: any) => typeof e.id === "string" && e.id.startsWith("qe::") });
  edges.remove(queryEdgeIds);
  nodes.remove(nodes.getIds({ filter: (n: any) => n.id === QUERY_NODE_ID }));

  if (props.queryLabel && highlightedSecIds.length > 0) {
    nodes.add({
      id: QUERY_NODE_ID,
      label: "query",
      shape: "diamond",
      size: 16,
      color: { background: accent, border: accent },
      font: { color: cssVar("--color-text"), size: 10 },
    });
    edges.add(
      highlightedSecIds.map((secId) => ({
        id: `qe::${secId}`,
        from: QUERY_NODE_ID,
        to: secId,
        dashes: true,
        color: accent,
        width: 1.5,
      })),
    );
  }
}

watch(() => props.highlightedSections, updateHighlights, { deep: true });
watch(
  () => props.data,
  () => {
    network?.destroy();
    buildGraph();
  },
);

onMounted(buildGraph);
onBeforeUnmount(() => network?.destroy());
</script>

<template>
  <div ref="container" class="h-full w-full" />
</template>
