<script setup lang="ts" generic="T extends Record<string, any>">
import { computed, ref } from "vue";
import { useVirtualizer } from "@tanstack/vue-virtual";

const props = withDefaults(
  defineProps<{
    items: T[];
    columns: { key: string; label: string; width?: string }[];
    rowHeight?: number;
    height?: string;
  }>(),
  { rowHeight: 36, height: "480px" },
);

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
const gridTemplate = computed(() => props.columns.map((c) => c.width ?? "1fr").join(" "));
</script>

<template>
  <div class="overflow-hidden rounded-md border border-border">
    <div
      class="grid border-b border-border-soft bg-panel-2 font-mono text-[10.5px] uppercase tracking-wide text-dim"
      :style="{ gridTemplateColumns: gridTemplate }"
    >
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
