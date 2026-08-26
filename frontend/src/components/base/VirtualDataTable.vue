<script setup lang="ts" generic="T extends Record<string, any>">
import { computed, ref } from "vue";
import { useVirtualizer } from "@tanstack/vue-virtual";

const props = withDefaults(
  defineProps<{
    items: T[];
    columns: { key: string; label: string; width?: string }[];
    rowHeight?: number;
    height?: string;
    // Phase 13: optional multi-select — pass rowKey (a field that uniquely
    // identifies a row, e.g. "filename") to enable a checkbox column.
    // Selection state is a plain Set of that key's values, v-model'd via
    // selected/update:selected so the parent (bulk-action buttons) reads
    // and clears it directly rather than this component owning the state.
    selectable?: boolean;
    rowKey?: string;
    selected?: Set<string>;
  }>(),
  { rowHeight: 36, height: "480px", selectable: false, rowKey: "", selected: () => new Set() },
);

const emit = defineEmits<{ "update:selected": [Set<string>] }>();

const scrollRef = ref<HTMLDivElement | null>(null);

const virtualizer = useVirtualizer(
  computed(() => ({
    count: props.items.length,
    getScrollElement: () => scrollRef.value,
    estimateSize: () => props.rowHeight,
    overscan: 12,
  })),
);

const virtualRows = computed(() => virtualizer.value.getVirtualItems());
const totalHeight = computed(() => virtualizer.value.getTotalSize());
const gridTemplate = computed(() => {
  const widths = props.columns.map((c) => c.width ?? "1fr");
  return props.selectable ? `2.5rem ${widths.join(" ")}` : widths.join(" ");
});

const allSelected = computed(() => props.items.length > 0 && props.items.every((item) => props.selected.has(item[props.rowKey])));
const someSelected = computed(() => props.selected.size > 0 && !allSelected.value);

function toggleAll() {
  if (allSelected.value) {
    emit("update:selected", new Set());
  } else {
    emit("update:selected", new Set(props.items.map((item) => item[props.rowKey])));
  }
}

function toggleRow(key: string) {
  const next = new Set(props.selected);
  if (next.has(key)) next.delete(key);
  else next.add(key);
  emit("update:selected", next);
}
</script>

<template>
  <div class="overflow-hidden rounded-md border border-border">
    <div
      class="grid border-b border-border-soft bg-panel-2 font-mono text-[10.5px] uppercase tracking-wide text-dim"
      :style="{ gridTemplateColumns: gridTemplate }"
    >
      <div v-if="selectable" class="flex items-center justify-center px-3 py-2">
        <input
          type="checkbox"
          :checked="allSelected"
          :indeterminate="someSelected"
          class="h-3.5 w-3.5 cursor-pointer accent-accent"
          @change="toggleAll"
        />
      </div>
      <div v-for="col in columns" :key="col.key" class="truncate px-3 py-2">{{ col.label }}</div>
    </div>
    <div v-if="items.length === 0" class="px-3 py-8 text-center text-[12.5px] text-dim-2">No rows</div>
    <div v-else ref="scrollRef" class="overflow-y-auto" :style="{ height }">
      <div :style="{ height: `${totalHeight}px`, position: 'relative', width: '100%' }">
        <div
          v-for="row in virtualRows"
          :key="row.index"
          class="absolute top-0 left-0 grid w-full items-center border-b border-border-soft text-[12.5px] text-text hover:bg-panel-2"
          :style="{ height: `${row.size}px`, transform: `translateY(${row.start}px)`, gridTemplateColumns: gridTemplate }"
        >
          <div v-if="selectable" class="flex items-center justify-center px-3">
            <input
              type="checkbox"
              :checked="selected.has(items[row.index]?.[rowKey])"
              class="h-3.5 w-3.5 cursor-pointer accent-accent"
              @change="toggleRow(items[row.index]?.[rowKey])"
            />
          </div>
          <div v-for="col in columns" :key="col.key" class="truncate px-3">
            <slot :name="`cell-${col.key}`" :item="items[row.index]" :index="row.index">
              {{ items[row.index]?.[col.key] }}
            </slot>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>
