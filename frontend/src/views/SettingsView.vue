<script setup lang="ts">
import Card from "../components/base/Card.vue";
import { useTheme } from "../lib/theme";

const { theme, setTheme } = useTheme();
</script>

<template>
  <div class="mx-auto max-w-3xl px-8 py-8">
    <h1 class="mb-5 font-display text-lg font-semibold text-text">Settings</h1>
    <Card title="Appearance">
      <div class="flex items-center justify-between">
        <div>
          <div class="text-[12.5px] font-medium text-text">Theme</div>
          <p class="mt-0.5 text-[11.5px] text-dim">Applies everywhere, saved on this device only.</p>
        </div>
        <div class="flex overflow-hidden rounded-md border border-border">
          <button
            type="button"
            class="px-3 py-1.5 text-[12px] font-medium transition-colors"
            :class="theme === 'light' ? 'bg-accent text-accent-ink' : 'bg-panel text-dim hover:text-text'"
            @click="setTheme('light')"
          >
            Light
          </button>
          <button
            type="button"
            class="border-l border-border px-3 py-1.5 text-[12px] font-medium transition-colors"
            :class="theme === 'dark' ? 'bg-accent text-accent-ink' : 'bg-panel text-dim hover:text-text'"
            @click="setTheme('dark')"
          >
            Dark
          </button>
        </div>
      </div>
    </Card>
  </div>
</template>
