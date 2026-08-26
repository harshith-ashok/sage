<script setup lang="ts">
import { useRoute } from "vue-router";
import { routes } from "../../router";
import { useTheme } from "../../lib/theme";

const route = useRoute();
const { theme, toggleTheme } = useTheme();
</script>

<template>
  <div class="flex h-screen bg-bg">
    <aside class="flex w-56 shrink-0 flex-col overflow-y-auto border-r border-border bg-panel px-3 py-5">
      <div class="mb-6 flex items-center gap-2 px-2">
        <svg width="20" height="20" viewBox="0 0 24 24" fill="none">
          <rect x="2" y="2" width="20" height="20" rx="5" stroke="var(--color-accent)" stroke-width="1.6" />
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
          class="rounded-md px-2.5 py-2 text-[12.5px] font-medium transition-colors"
          :class="
            route.name === r.name
              ? 'bg-accent/10 text-accent'
              : 'text-dim hover:bg-panel-2 hover:text-text'
          "
        >
          {{ r.label }}
        </RouterLink>
      </nav>

      <div class="mt-auto flex flex-col gap-1 pt-4">
        <button
          type="button"
          class="flex items-center gap-2.5 rounded-md px-2.5 py-2 text-[12.5px] font-medium text-dim transition-colors hover:bg-panel-2 hover:text-text"
          :title="theme === 'light' ? 'Switch to dark theme' : 'Switch to light theme'"
          @click="toggleTheme"
        >
          <svg v-if="theme === 'light'" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8">
            <circle cx="12" cy="12" r="4.5" />
            <path d="M12 2v2.5M12 19.5V22M4.2 4.2l1.8 1.8M18 18l1.8 1.8M2 12h2.5M19.5 12H22M4.2 19.8L6 18M18 6l1.8-1.8" stroke-linecap="round" />
          </svg>
          <svg v-else width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8">
            <path d="M20 14.5A8.5 8.5 0 1 1 9.5 4a7 7 0 0 0 10.5 10.5Z" stroke-linejoin="round" />
          </svg>
          {{ theme === "light" ? "Light" : "Dark" }} theme
        </button>
      </div>
    </aside>

    <main class="min-w-0 flex-1 overflow-y-auto">
      <RouterView />
    </main>
  </div>
</template>
