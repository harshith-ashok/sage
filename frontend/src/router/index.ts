import { createRouter, createWebHistory } from "vue-router";

export const routes = [
  { path: "/", name: "console", label: "Console", component: () => import("../views/ConsoleView.vue") },
  { path: "/knowledge", name: "knowledge", label: "Knowledge Base", component: () => import("../views/KnowledgeBaseView.vue") },
  { path: "/deliverables", name: "deliverables", label: "Deliverables", component: () => import("../views/DeliverablesView.vue") },
  { path: "/models", name: "models", label: "Model Registry", component: () => import("../views/ModelRegistryView.vue") },
  { path: "/network", name: "network", label: "Network Monitor", component: () => import("../views/NetworkMonitorView.vue") },
  { path: "/settings", name: "settings", label: "Settings", component: () => import("../views/SettingsView.vue") },
];

export const router = createRouter({
  history: createWebHistory(),
  routes,
});
