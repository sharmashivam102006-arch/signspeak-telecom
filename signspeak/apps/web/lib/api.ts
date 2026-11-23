/** Thin typed client for the FastAPI backend. */

const BASE = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${BASE}${path}`, {
    ...init,
    headers: { "content-type": "application/json", ...(init?.headers ?? {}) },
  });
  if (!res.ok) {
    const body = await res.text();
    throw new Error(`${res.status} ${res.statusText}: ${body.slice(0, 200)}`);
  }
  return res.status === 204 ? (undefined as T) : res.json();
}

export interface RoomToken {
  room_name: string;
  call_id: string;
  identity: string;
  token: string;
  livekit_url: string;
  transcript_enabled: boolean;
  server_time_ms: number;
}

export const api = {
  createRoom: (body: { display_name: string; modality: string; transcript_enabled?: boolean }) =>
    request<RoomToken>("/rooms", { method: "POST", body: JSON.stringify(body) }),

  joinRoom: (room: string, body: { display_name: string; modality: string; consent_transcript?: boolean }) =>
    request<RoomToken>(`/rooms/${room}/join`, { method: "POST", body: JSON.stringify(body) }),

  /** Fire-and-forget: a failed telemetry write must never interrupt a call. */
  logEvent: (callId: string, identity: string, body: Record<string, unknown>) =>
    request<void>(`/calls/${callId}/events?identity=${encodeURIComponent(identity)}`, {
      method: "POST",
      body: JSON.stringify(body),
    }).catch(() => undefined),

  addTranscriptEntry: (callId: string, identity: string, body: Record<string, unknown>) =>
    request<unknown>(`/calls/${callId}/transcript?identity=${encodeURIComponent(identity)}`, {
      method: "POST",
      body: JSON.stringify(body),
    }).catch(() => undefined),

  getTranscript: (callId: string) => request<unknown[]>(`/calls/${callId}/transcript`),
  eventsSummary: (callId: string) => request<Record<string, unknown>>(`/calls/${callId}/events/summary`),
};
