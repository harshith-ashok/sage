<script setup lang="ts">
import { useRoute } from "vue-router";
import { routes } from "../../router";

const route = useRoute();
</script>

<template>
  <div class="flex h-screen bg-bg">
    <aside class="flex w-56 shrink-0 flex-col overflow-y-auto border-r border-border bg-panel px-3 py-5">
      <div class="mb-6 flex items-center gap-2 px-2">
        <svg width="20" height="20" viewBox="0 0 24 24" fill="none">
          <rect x="2" y="2" width="20" height="20" rx="4" stroke="var(--color-accent)" stroke-width="1.6" />
          <path d="M7 12h10M12 7v10" stroke="var(--color-accent)" stroke-width="1.6" stroke-linecap="round" />
        </svg>
        <div>
          <div class="font-display text-[14.5px] font-semibold tracking-tight text-text">SAGE</div>
          <div class="font-mono text-[9px] uppercase tracking-wider text-dim-2">air-gapped</div>
        </div>
      </div>

      <nav class="flex flex-col gap-0.5">
        <RouterLink
          v-for="r in routes"
          :key="r.path"
          :to="r.path"
          class="flex items-center gap-2.5 rounded-md border-l-2 border-transparent px-2.5 py-2 text-[12.5px]"
          :class="
            route.name === r.name
              ? 'border-l-accent bg-accent/10 text-text'
              : 'text-dim hover:bg-panel-2 hover:text-text'
          "
        >
          {{ r.label }}
        </RouterLink>
      </nav>
    </aside>

    <main class="min-w-0 flex-1 overflow-y-auto">
      <RouterView />
    </main>
  </div>
</template>
