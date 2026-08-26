import { ref } from "vue";

const STORAGE_KEY = "sage-theme";

function readInitialTheme(): "light" | "dark" {
  const saved = localStorage.getItem(STORAGE_KEY);
  if (saved === "light" || saved === "dark") return saved;
  return window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
}

// Module-level singleton (not a Pinia store — this has nothing to persist
// beyond localStorage and no other state depends on it) so every component
// that calls useTheme() shares the same reactive value; index.html's inline
// script already applied this exact value to <html data-theme> before Vue
// even mounted, so this just needs to agree with it, not re-apply on init.
const theme = ref<"light" | "dark">(readInitialTheme());

function apply(value: "light" | "dark") {
  theme.value = value;
  document.documentElement.setAttribute("data-theme", value);
  try {
    localStorage.setItem(STORAGE_KEY, value);
  } catch {
    // Private browsing / storage disabled — the toggle still works for
    // this page load, it just won't be remembered next time.
  }
}

export function useTheme() {
  return {
    theme,
    setTheme: apply,
    toggleTheme: () => apply(theme.value === "light" ? "dark" : "light"),
  };
}
