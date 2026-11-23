// Typed client for the real AI pipelines served by the backend:
//   sign model  — GET /api/ai/status, /api/ai/sign/status|samples|collect|train|predict
//   Whisper ASR — POST /api/ai/transcribe/audio (see lib/captions.ts for the WS stream)
//   Piper TTS   — POST /api/ai/tts (audio file), /api/ai/tts/voices
//   voice agent — /api/ai/rooms/{id}/voice/* (speaks into the LiveKit room)
//
// Every function degrades honestly: a 404 means the backend does not
// expose that endpoint yet (returns null / "unavailable"), a 503 means
// the model/service is not installed. Nothing here invents a success.
import { ApiError, api } from "./api";

export type SignModelStatus = {
  mode: string;
  trained: boolean;
  torch_available: boolean;
  mediapipe_available: boolean;
  model_path: string;
  samples_per_sign: Record<string, number>;
  vocabulary: string[];
};

export type WhisperStatus = { available: boolean; model: string | null; engine: string | null };
export type TTSStatus = { available: boolean; engine: string | null; voices: string[] };

export type AIStatus = {
  sign: SignModelStatus | null;
  whisper: WhisperStatus | null;
  tts: TTSStatus | null;
  voice_agent: { rooms_active: number } | null;
};

export type TrainJobStatus = "queued" | "running" | "done" | "failed";
export type TrainJob = {
  job_id: string;
  status: TrainJobStatus;
  progress: number;
  epochs_done: number;
  val_accuracy: number | null;
  message: string;
};

export type VoiceStatus = { active: boolean };

function isNotFound(e: unknown): boolean {
  return e instanceof ApiError && e.status === 404;
}

/** GET /api/ai/status — whole-pipeline status. null when the backend does not expose it yet. */
export async function fetchAIStatus(): Promise<AIStatus | null> {
  try {
    const d = await api<{
      sign?: SignModelStatus;
      whisper?: WhisperStatus;
      tts?: TTSStatus;
      voice_agent?: { rooms_active: number | unknown[] };
    }>("/api/ai/status");
    // rooms_active is a count; tolerate a legacy list response by
    // converting it to its length instead of rendering "[object Object]".
    const va = d.voice_agent ?? null;
    return {
      sign: d.sign ?? null,
      whisper: d.whisper ?? null,
      tts: d.tts ?? null,
      voice_agent: va
        ? { rooms_active: Array.isArray(va.rooms_active) ? va.rooms_active.length : Number(va.rooms_active) || 0 }
        : null,
    };
  } catch (e) {
    if (isNotFound(e)) return null;
    throw e;
  }
}

// --- Server transcripts (consent-gated; creator-only on the backend) ---

export type ServerTranscriptEntry = {
  id: string;
  room_id: string;
  speaker_name: string;
  direction: string;
  text: string;
  confidence: number | null;
  created_at: string | null;
};

/** GET /api/rooms/{roomId}/transcripts — server-stored entries.
 *  Throws ApiError 403 for non-creators (backend restriction). */
export async function fetchServerTranscripts(roomId: string): Promise<ServerTranscriptEntry[]> {
  return api<ServerTranscriptEntry[]>(`/api/rooms/${encodeURIComponent(roomId)}/transcripts`);
}

export type TranscriptSaveItem = {
  direction: string;
  text: string;
  speaker_name?: string;
  /** 0–1 float; converted to the int 0–100 the API schema requires. */
  confidence?: number | null;
};

/** POST each entry to /api/rooms/{roomId}/transcripts with consent=true.
 *  Returns how many were stored. Throws on the first failure (e.g. 403
 *  "Only the room creator can access this room's transcripts") — the
 *  caller surfaces the error instead of claiming the save worked. */
export async function saveTranscriptsToServer(
  roomId: string,
  entries: TranscriptSaveItem[],
): Promise<{ saved: number }> {
  let saved = 0;
  for (const e of entries) {
    if (!e.text.trim()) continue;
    await api(`/api/rooms/${encodeURIComponent(roomId)}/transcripts`, {
      method: "POST",
      body: JSON.stringify({
        direction: e.direction,
        text: e.text,
        speaker_name: e.speaker_name || "",
        confidence:
          typeof e.confidence === "number"
            ? Math.max(0, Math.min(100, Math.round(e.confidence * 100)))
            : null,
        consent: true,
      }),
    });
    saved += 1;
  }
  return { saved };
}

/** GET /api/ai/sign/status — null when the endpoint is not deployed yet. */
export async function fetchSignStatus(): Promise<SignModelStatus | null> {
  try {
    return await api<SignModelStatus>("/api/ai/sign/status");
  } catch (e) {
    if (isNotFound(e)) return null;
    throw e;
  }
}

/** GET /api/ai/sign/samples — per-sign recorded sample counts. */
export async function fetchSignSamples(): Promise<{ samples_per_sign: Record<string, number>; total: number }> {
  const d = await api<{ samples_per_sign?: Record<string, number>; total?: number }>("/api/ai/sign/samples");
  return { samples_per_sign: d.samples_per_sign ?? {}, total: d.total ?? 0 };
}

/** POST /api/ai/sign/collect — store one recorded sequence under a label. */
export async function collectSignSample(
  label: string,
  sequence: number[][][],
): Promise<{ label: string; count: number }> {
  return api<{ label: string; count: number }>("/api/ai/sign/collect", {
    method: "POST",
    body: JSON.stringify({ label, sequence }),
  });
}

/** POST /api/ai/sign/train — queue a training job. 409 when one is already running. */
export async function startSignTraining(): Promise<TrainJob> {
  const d = await api<{ job_id: string; status: string }>("/api/ai/sign/train", {
    method: "POST",
    body: JSON.stringify({}),
  });
  return { job_id: d.job_id, status: (d.status as TrainJobStatus) || "queued", progress: 0, epochs_done: 0, val_accuracy: null, message: "" };
}

/** GET /api/ai/sign/train/{job_id} — poll a training job. */
export async function fetchTrainJob(jobId: string): Promise<TrainJob> {
  const d = await api<{
    job_id: string;
    status: string;
    progress?: number;
    epochs_done?: number;
    val_accuracy?: number | null;
    message?: string;
  }>(`/api/ai/sign/train/${encodeURIComponent(jobId)}`);
  return {
    job_id: d.job_id,
    status: (d.status as TrainJobStatus) || "queued",
    progress: d.progress ?? 0,
    epochs_done: d.epochs_done ?? 0,
    val_accuracy: d.val_accuracy ?? null,
    message: d.message ?? "",
  };
}

// --- Voice agent (room audio) ---

function normaliseVoiceStatus(d: unknown): VoiceStatus {
  const o = (d ?? {}) as Record<string, unknown>;
  const active = o.active ?? o.running ?? o.started ?? o.enabled ?? false;
  return { active: active === true };
}

/** GET /api/ai/rooms/{roomId}/voice/status — null when the endpoint is not deployed yet
 *  (unknown), so the UI can distinguish "unknown" from "off". */
export async function fetchVoiceStatus(roomId: string): Promise<VoiceStatus | null> {
  try {
    return normaliseVoiceStatus(await api<unknown>(`/api/ai/rooms/${encodeURIComponent(roomId)}/voice/status`));
  } catch (e) {
    if (isNotFound(e)) return null;
    throw e;
  }
}

/** POST /api/ai/rooms/{roomId}/voice/speak — agent speaks text into the LiveKit room. */
export async function speakViaVoiceAgent(roomId: string, text: string): Promise<{ spoken: boolean; duration_s: number }> {
  const d = await api<{ spoken?: boolean; duration_s?: number }>(
    `/api/ai/rooms/${encodeURIComponent(roomId)}/voice/speak`,
    { method: "POST", body: JSON.stringify({ text }) },
  );
  return { spoken: d.spoken ?? false, duration_s: d.duration_s ?? 0 };
}

/** POST /api/ai/rooms/{roomId}/voice/start */
export async function startVoiceAgent(roomId: string): Promise<void> {
  await api(`/api/ai/rooms/${encodeURIComponent(roomId)}/voice/start`, { method: "POST", body: JSON.stringify({}) });
}

/** POST /api/ai/rooms/{roomId}/voice/stop */
export async function stopVoiceAgent(roomId: string): Promise<void> {
  await api(`/api/ai/rooms/${encodeURIComponent(roomId)}/voice/stop`, { method: "POST", body: JSON.stringify({}) });
}

// --- Server TTS (Piper) ---
//
// NOTE: the old `synthesizeTTS()` helper lived here; it was deleted
// 2026-11-18 (Priority 8 dead-code cleanup) because no component ever
// imported it. TTS from the UI goes through the voice agent instead:
// `speakViaVoiceAgent()` above, which publishes Piper audio into the
// LiveKit room. The raw endpoint is still POST /api/ai/tts — see docs/API.md.
