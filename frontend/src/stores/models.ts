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

export const useModelsStore = defineStore("models", {
  state: () => ({
    registry: {} as ModelsRegistry,
    loading: false,
    error: "",
    switching: "" as string, // task_type currently being switched, for per-row disabling
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
  },
});
