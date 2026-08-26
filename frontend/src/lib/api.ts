// Thin fetch + EventSource wrapper for the FastAPI backend (backend/app/main.py).
// Generalizes insurance_claim_agent/frontend/src/composables/useQueryDemo.ts's
// SSE handling (named events, a reserved `error` for connection failures) into
// a reusable client, since SAGE has many streaming task endpoints (Phases 4/5/7),
// not just one query flow.

export const API_BASE = (import.meta.env.VITE_API_BASE_URL as string | undefined) ?? "http://localhost:8000";

export class ApiError extends Error {
  status: number;

  constructor(message: string, status: number) {
    super(message);
    this.name = "ApiError";
    this.status = status;
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, {
    headers: { "Content-Type": "application/json" },
    ...init,
  });
  if (!res.ok) {
    const detail = await res.json().catch(() => null);
    throw new ApiError(detail?.detail ?? res.statusText, res.status);
  }
  return res.json() as Promise<T>;
}

export function apiGet<T>(path: string): Promise<T> {
  return request<T>(path);
}

export function apiPost<T>(path: string, body?: unknown): Promise<T> {
  return request<T>(path, { method: "POST", body: body === undefined ? undefined : JSON.stringify(body) });
}

/** POST a multipart form (file uploads) — no Content-Type header set here so
 * the browser fills in the multipart boundary itself. */
export async function apiPostForm<T>(path: string, form: FormData): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, { method: "POST", body: form });
  if (!res.ok) {
    const detail = await res.json().catch(() => null);
    throw new ApiError(detail?.detail ?? res.statusText, res.status);
  }
  return res.json() as Promise<T>;
}

/**
 * Opens an SSE stream at `path`, dispatching each named event in `handlers`
 * (event name -> parsed-JSON-data callback). Returns the EventSource so the
 * caller can `.close()` it (e.g. on unmount or when starting a new stream).
 *
 * `onConnectionError` fires only for a genuine transport failure — a stream
 * that already reached one of `terminalEvents` closing itself is not an
 * error, matching useQueryDemo.ts's `status === "done"` guard.
 */
export function streamSSE(
  path: string,
  handlers: Record<string, (data: any) => void>,
  options?: { onConnectionError?: (err: Event) => void; terminalEvents?: string[] },
): EventSource {
  const es = new EventSource(`${API_BASE}${path}`);
  const terminal = new Set(options?.terminalEvents ?? ["done", "error"]);
  let closed = false;

  for (const [event, handler] of Object.entries(handlers)) {
    es.addEventListener(event, (e: MessageEvent) => {
      handler(e.data ? JSON.parse(e.data) : null);
      if (terminal.has(event)) {
        closed = true;
        es.close();
      }
    });
  }

  es.addEventListener("error", (e) => {
    if (closed) return; // stream already finished cleanly
    options?.onConnectionError?.(e);
    es.close();
  });

  return es;
}
