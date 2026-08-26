<script setup lang="ts">
withDefaults(
  defineProps<{
    variant?: "primary" | "secondary" | "ghost" | "danger";
    disabled?: boolean;
    type?: "button" | "submit";
  }>(),
  { variant: "secondary", disabled: false, type: "button" },
);
</script>

<template>
  <button
    :type="type"
    :disabled="disabled"
    class="inline-flex items-center gap-1.5 rounded-md px-3 py-1.5 text-[12.5px] font-medium transition-colors duration-150 disabled:cursor-not-allowed disabled:opacity-40"
    :class="{
      'bg-accent text-accent-ink hover:bg-accent/90 shadow-sm': variant === 'primary',
      'bg-panel text-text border border-border hover:border-dim-2 hover:bg-panel-2': variant === 'secondary',
      'text-dim hover:text-text hover:bg-panel-2': variant === 'ghost',
      'bg-danger/10 text-danger border border-danger/30 hover:bg-danger/20': variant === 'danger',
    }"
  >
    <slot />
  </button>
</template>
