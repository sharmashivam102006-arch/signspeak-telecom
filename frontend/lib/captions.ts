// Live captions via the real Whisper path: microphone audio is chunked
// with MediaRecorder and streamed over WS /ws/captions; the server
// replies with FINAL captions only (engine "whisper-tiny") — it never
// emits partial/interim messages. Interim text exists only in the
// browser Web Speech fallback, which is then labelled "Browser captions
// (not Whisper)", never as Whisper.
//
// Cross-participant delivery: final captions are ALSO sent over the
// room relay WS /ws/call/{room_id} (CaptionRelayClient below) so the
// other participant sees them. The server broadcasts relay messages to
// the other peers in the room; it never echoes a message back to its
// sender. If the relay is down, captions stay local and the UI says so.
import { WS_URL } from "./api";

export type CaptionEvent =
  | { type: "partial"; text: string }
  | { type: "final"; text: string; confidence: number; engine: string }
  | { type: "error"; message: string };

type CaptionHandlers = {
  onEvent: (e: CaptionEvent) => void;
  onClose: () => void;
  onOpen?: () => void;
};

type ServerCaptionMessage = {
  type?: string;
  text?: string;
  confidence?: number;
  is_final?: boolean;
  engine?: string;
  message?: string;
};

export class WhisperCaptionClient {
  private ws: WebSocket | null = null;
  private closed = false;

  constructor(private token: string | null, private handlers: CaptionHandlers) {}

  get isOpen(): boolean {
    return !!this.ws && this.ws.readyState === WebSocket.OPEN;
  }

  connect(): void {
    this.closed = false;
    const url = `${WS_URL}/ws/captions${this.token ? `?token=${encodeURIComponent(this.token)}` : ""}`;
    const ws = new WebSocket(url);
    this.ws = ws;
    ws.onopen = () => {
      this.handlers.onOpen?.();
    };
    ws.onmessage = (ev) => {
      let data: ServerCaptionMessage;
      try {
        data = JSON.parse(String(ev.data)) as ServerCaptionMessage;
      } catch {
        return; // Non-JSON frame: ignore.
      }
      if (data.type === "caption" || data.type === "final") {
        if (data.text?.trim()) {
          this.handlers.onEvent({
            type: "final",
            text: data.text.trim(),
            confidence: typeof data.confidence === "number" ? data.confidence : 0,
            engine: data.engine ?? "whisper-tiny",
          });
        }
      } else if (data.type === "partial") {
        if (data.text?.trim()) this.handlers.onEvent({ type: "partial", text: data.text.trim() });
      } else if (data.type === "error") {
        this.handlers.onEvent({ type: "error", message: data.message || "Whisper caption error." });
      }
    };
    ws.onerror = () => {
      this.handlers.onEvent({ type: "error", message: "Could not connect to the Whisper caption stream." });
    };
    ws.onclose = () => {
      this.ws = null;
      if (!this.closed) this.handlers.onClose();
    };
  }

  sendAudioChunk(data: ArrayBuffer): void {
    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      try { this.ws.send(data); } catch { /* chunk dropped */ }
    }
  }

  close(): void {
    this.closed = true;
    try { this.ws?.close(); } catch { /* ignore */ }
    this.ws = null;
  }
}

/** One caption as it travels over the room relay (/ws/call). The
 *  server relays the message verbatim to the other peers (overwriting
 *  only `sender` with the authenticated user id), so the receiver can
 *  trust `speaker_id`/`speaker_name` as sent by the speaker's client
 *  and dedupe on `id`. */
export type RelayCaption = {
  id: string;
  text: string;
  confidence: number;
  speaker_id: string;
  speaker_name: string;
  is_final: boolean;
  timestamp: string;
  engine?: "whisper" | "browser";
};

export type RelayState = "connected" | "reconnecting" | "disconnected";

type RelayHandlers = {
  onCaption: (c: RelayCaption) => void;
  onState?: (s: RelayState) => void;
};

const RELAY_MAX_BACKOFF_MS = 8000;

/** Client for the room caption relay: WS /ws/call/{room_id}?token=<jwt>.
 *
 *  Same socket the (legacy) signalling code used; the server accepts
 *  any JSON message whose `type` is in its allowlist ("caption" is one)
 *  and broadcasts it to the other peers in the room. Reconnects with
 *  exponential backoff when the socket drops unexpectedly, and reports
 *  its state so the UI can show "caption relay reconnecting" instead
 *  of silently dropping remote captions. */
export class CaptionRelayClient {
  private ws: WebSocket | null = null;
  private closed = false;
  private reconnectTimer: number | null = null;
  private attempts = 0;
  private state: RelayState = "disconnected";

  constructor(
    private roomId: string,
    private token: string | null,
    private handlers: RelayHandlers,
  ) {}

  get relayState(): RelayState {
    return this.state;
  }

  get isOpen(): boolean {
    return !!this.ws && this.ws.readyState === WebSocket.OPEN;
  }

  connect(): void {
    this.closed = false;
    this.attempts = 0;
    this.openSocket();
  }

  private setState(s: RelayState): void {
    if (this.state === s) return;
    this.state = s;
    this.handlers.onState?.(s);
  }

  private openSocket(): void {
    const url = `${WS_URL}/ws/call/${encodeURIComponent(this.roomId)}${this.token ? `?token=${encodeURIComponent(this.token)}` : ""}`;
    const ws = new WebSocket(url);
    this.ws = ws;
    ws.onopen = () => {
      this.attempts = 0;
      this.setState("connected");
    };
    ws.onmessage = (ev) => {
      let data: Record<string, unknown>;
      try {
        data = JSON.parse(String(ev.data)) as Record<string, unknown>;
      } catch {
        return; // Non-JSON frame: ignore.
      }
      // Join/leave/signalling frames are not captions.
      if (data.type !== "caption") return;
      const text = typeof data.text === "string" ? data.text.trim() : "";
      if (!text) return;
      this.handlers.onCaption({
        id: typeof data.id === "string" && data.id ? data.id : `relay-${Date.now()}-${Math.random().toString(36).slice(2, 6)}`,
        text,
        confidence: typeof data.confidence === "number" ? data.confidence : 0,
        speaker_id: typeof data.speaker_id === "string" ? data.speaker_id : String(data.sender ?? ""),
        speaker_name: typeof data.speaker_name === "string" && data.speaker_name ? data.speaker_name : "Remote participant",
        is_final: data.is_final !== false,
        timestamp: typeof data.timestamp === "string" ? data.timestamp : new Date().toISOString(),
        engine: data.engine === "browser" ? "browser" : "whisper",
      });
    };
    ws.onerror = () => {
      // onclose follows; reconnect is handled there.
    };
    ws.onclose = () => {
      this.ws = null;
      if (this.closed) {
        this.setState("disconnected");
        return;
      }
      // Unexpected drop: reconnect with backoff and tell the UI.
      this.setState("reconnecting");
      const delay = Math.min(1000 * 2 ** this.attempts, RELAY_MAX_BACKOFF_MS);
      this.attempts += 1;
      this.reconnectTimer = window.setTimeout(() => this.openSocket(), delay);
    };
  }

  /** Send one final caption to the other participants. Returns false
   *  when the socket is not open (caller keeps the local caption; the
   *  relay state badge explains the gap). */
  sendCaption(c: Omit<RelayCaption, "id" | "timestamp"> & { id?: string; timestamp?: string }): boolean {
    if (!this.ws || this.ws.readyState !== WebSocket.OPEN) return false;
    const msg: RelayCaption = {
      id: c.id ?? `${Date.now()}-${Math.random().toString(36).slice(2, 8)}`,
      timestamp: c.timestamp ?? new Date().toISOString(),
      ...c,
    };
    try {
      this.ws.send(JSON.stringify({ type: "caption", ...msg }));
      return true;
    } catch {
      return false;
    }
  }

  close(): void {
    this.closed = true;
    if (this.reconnectTimer) window.clearTimeout(this.reconnectTimer);
    this.reconnectTimer = null;
    try { this.ws?.close(); } catch { /* ignore */ }
    this.ws = null;
    this.setState("disconnected");
  }
}

/** Best audio MIME type this browser can record (Opus in WebM preferred). */
export function pickAudioMimeType(): string {
  if (typeof MediaRecorder === "undefined") return "";
  for (const mt of ["audio/webm;codecs=opus", "audio/webm", "audio/mp4", "audio/ogg;codecs=opus"]) {
    try {
      if (MediaRecorder.isTypeSupported(mt)) return mt;
    } catch { /* ignore */ }
  }
  return "";
}

export function mediaRecorderSupported(): boolean {
  return typeof window !== "undefined" && typeof MediaRecorder !== "undefined";
}

/** Records a MediaStream in ~2s chunks and hands each Blob to the callback.
 *  The caller sends the chunks to the Whisper socket. */
export class MicChunkStreamer {
  private recorder: MediaRecorder | null = null;
  private stream: MediaStream;

  constructor(stream: MediaStream, private onChunk: (blob: Blob) => void, private chunkMs = 2000) {
    this.stream = stream;
  }

  start(): boolean {
    try {
      const mimeType = pickAudioMimeType();
      this.recorder = mimeType ? new MediaRecorder(this.stream, { mimeType }) : new MediaRecorder(this.stream);
      this.recorder.ondataavailable = (e) => {
        if (e.data && e.data.size > 0) this.onChunk(e.data);
      };
      this.recorder.start(this.chunkMs);
      return true;
    } catch {
      return false;
    }
  }

  stop(): void {
    try {
      if (this.recorder && this.recorder.state !== "inactive") this.recorder.stop();
    } catch { /* ignore */ }
    this.recorder = null;
  }
}
