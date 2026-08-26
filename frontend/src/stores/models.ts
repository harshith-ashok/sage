import { defineStore } from "pinia";
import { apiGet, apiPost } from "../lib/api";

export interface ModelCandidate {
  model_id: string;
  location: "local" | "cloud";
  context_window: number;
  locally_available: boolean;
}

export interface TaskTypeRegistry {
  active: string;
  candidates: Record<string, ModelCandidate>;
}

export type ModelsRegistry = Record<string, TaskTypeRegistry>;

// Phase 11: load-aware routing. Mirrors backend/app/load_monitor.py's
// LoadSnapshot/FallbackEvent shapes, as returned by GET /system/load.
export interface LoadedModel {
  model_id: string;
  size_bytes: number;
  size_vram_bytes: number;
}

export interface FallbackEvent {
  task_type: string;
  from_model_id: string;
  to_model_id: string | null;
  reason: string;
  timestamp: string;
}

export interface SystemLoad {
  level: "normal" | "elevated" | "high";
  available_percent: number;
  loaded_models: LoadedModel[];
  fallback_log: FallbackEvent[];
}

export const useModelsStore = defineStore("models", {
  state: () => ({
    registry: {} as ModelsRegistry,
    loading: false,
    error: "",
    switching: "" as string, // task_type currently being switched, for per-row disabling
    load: null as SystemLoad | null,
    loadError: "",
    loadPollHandle: null as ReturnType<typeof setInterval> | null,
  }),
  actions: {
    async fetchRegistry() {
      this.loading = true;
      this.error = "";
      try {
        this.registry = await apiGet<ModelsRegistry>("/models");
      } catch (e) {
        this.error = e instanceof Error ? e.message : String(e);
      } finally {
        this.loading = false;
      }
    },
    async setActive(taskType: string, candidateKey: string) {
      this.switching = taskType;
      this.error = "";
      try {
        await apiPost(`/models/${taskType}/active`, { candidate_key: candidateKey });
        await this.fetchRegistry();
      } catch (e) {
        this.error = e instanceof Error ? e.message : String(e);
      } finally {
        this.switching = "";
      }
    },
    async fetchLoad() {
      try {
        this.load = await apiGet<SystemLoad>("/system/load");
        this.loadError = "";
      } catch (e) {
        this.loadError = e instanceof Error ? e.message : String(e);
      }
    },
    startLoadPolling() {
      if (this.loadPollHandle !== null) return;
      this.fetchLoad();
      this.loadPollHandle = setInterval(() => this.fetchLoad(), 5000);
    },
    stopLoadPolling() {
      if (this.loadPollHandle !== null) {
        clearInterval(this.loadPollHandle);
        this.loadPollHandle = null;
      }
    },
  },
});
