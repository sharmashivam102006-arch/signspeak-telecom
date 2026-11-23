// Live captions via the real Whisper path: microphone audio is converted
// to 16 kHz mono PCM16 in the browser (PcmMicStreamer) and streamed over
// WS /ws/captions after a {"type":"config","format":"pcm16"} message. The
// server cuts the stream into utterances at pauses and replies with one
// FINAL caption per utterance (engine "whisper-tiny") — it never emits
// partial/interim messages. Interim text exists only in the browser Web
// Speech fallback, which is then labelled "Browser captions (not
// Whisper)", never as Whisper.
//
// If the server cannot keep up (too little CPU for real-time Whisper) or
// the model is unavailable, it sends {"type":"error","code":...} and the
// call page switches to browser captions instead of showing nothing.
//
// Cross-participant delivery: final captions are ALSO sent over the
// room relay WS /ws/call/{room_id} (CaptionRelayClient below) so the
// other participant sees them. The server broadcasts relay messages to
// the other peers in the room; it never echoes a message back to its
// sender. If the relay is down, captions stay local and the UI says so.
import { WS_URL } from "./api";

export type CaptionEvent =
  | { type: "partial"; text: string }
  | { type: "final"; text: string; confidence: number; engine: string; latencyS?: number }
  | { type: "status"; state: "loading" | "ready" }
  | { type: "error"; message: string; code?: string };

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
  code?: string;
  state?: string;
  latency_s?: number;
};

export const CAPTION_SAMPLE_RATE = 16000;

export class WhisperCaptionClient {
  private ws: WebSocket | null = null;
  private closed = false;

  constructor(private token: string | null, private handlers: CaptionHandlers, private language = "en") {}

  get isOpen(): boolean {
    return !!this.ws && this.ws.readyState === WebSocket.OPEN;
  }

  connect(): void {
    this.closed = false;
    const url = `${WS_URL}/ws/captions${this.token ? `?token=${encodeURIComponent(this.token)}` : ""}`;
    const ws = new WebSocket(url);
    ws.binaryType = "arraybuffer";
    this.ws = ws;
    ws.onopen = () => {
      // Tell the server what the binary frames are before any audio.
      try {
        ws.send(JSON.stringify({
          type: "config", format: "pcm16", sample_rate: CAPTION_SAMPLE_RATE,
          language: (this.language || "en").slice(0, 2).toLowerCase(),
        }));
      } catch { /* onclose follows */ }
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
            latencyS: typeof data.latency_s === "number" ? data.latency_s : undefined,
          });
        }
      } else if (data.type === "partial") {
        if (data.text?.trim()) this.handlers.onEvent({ type: "partial", text: data.text.trim() });
      } else if (data.type === "status") {
        if (data.state === "loading" || data.state === "ready") this.handlers.onEvent({ type: "status", state: data.state });
      } else if (data.type === "error") {
        this.handlers.onEvent({ type: "error", message: data.message || "Whisper caption error.", code: data.code });
      }
    };
    ws.onerror = () => {
      this.handlers.onEvent({ type: "error", message: "Could not connect to the Whisper caption stream.", code: "connect" });
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

/** Streaming resampler: Float32 mic samples at the AudioContext rate ->
 *  16 kHz PCM16. Each output sample is the average of the input samples
 *  it covers (a box filter), which also removes most content above 8 kHz
 *  before decimating. Keeps state across calls so chunk boundaries do
 *  not click or drop samples. */
export class Pcm16Resampler {
  private carry: Float32Array = new Float32Array(0);
  // Absolute positions (samples since start): carry[0] is input sample
  // `carryBase`; `outCount` samples have been produced. Output k covers
  // input [k*in/out, (k+1)*in/out) — computed from these integers every
  // time, so how the input is chunked never changes the result.
  private carryBase = 0;
  private outCount = 0;

  constructor(private inRate: number, private outRate = CAPTION_SAMPLE_RATE) {
    if (!(inRate > 0) || !(outRate > 0) || outRate > inRate) throw new Error("unsupported sample rates");
  }

  private edge(k: number): number {
    return Math.floor((k * this.inRate) / this.outRate);
  }

  push(input: Float32Array): Int16Array {
    const x = new Float32Array(this.carry.length + input.length);
    x.set(this.carry, 0);
    x.set(input, this.carry.length);
    const available = this.carryBase + x.length; // absolute end of input
    const out: number[] = [];
    while (this.edge(this.outCount + 1) <= available) {
      const a = this.edge(this.outCount) - this.carryBase;
      const b = this.edge(this.outCount + 1) - this.carryBase;
      let sum = 0;
      for (let j = a; j < b; j++) sum += x[j];
      const v = Math.max(-1, Math.min(1, sum / Math.max(1, b - a)));
      out.push(v < 0 ? Math.round(v * 32768) : Math.round(v * 32767));
      this.outCount += 1;
    }
    const keepFrom = this.edge(this.outCount) - this.carryBase;
    this.carry = x.slice(keepFrom);
    this.carryBase += keepFrom;
    return Int16Array.from(out);
  }
}

type AudioCtxCtor = new (opts?: AudioContextOptions) => AudioContext;

function audioContextCtor(): AudioCtxCtor | null {
  if (typeof window === "undefined") return null;
  const w = window as unknown as { AudioContext?: AudioCtxCtor; webkitAudioContext?: AudioCtxCtor };
  return w.AudioContext ?? w.webkitAudioContext ?? null;
}

/** True when this browser can capture raw mic samples for Whisper. */
export function pcmStreamingSupported(): boolean {
  return audioContextCtor() !== null && typeof WebSocket !== "undefined";
}

// AudioWorklet that forwards each 128-sample render quantum of channel 0.
const WORKLET_SRC = `
class SignSpeakPcmTap extends AudioWorkletProcessor {
  process(inputs) {
    const ch = inputs[0] && inputs[0][0];
    if (ch && ch.length) this.port.postMessage(ch.slice(0));
    return true;
  }
}
registerProcessor("signspeak-pcm-tap", SignSpeakPcmTap);
`;

/** Captures a mic MediaStream as 16 kHz mono PCM16 and hands it to the
 *  callback in ~frameMs frames (ArrayBuffer, little-endian). Uses an
 *  AudioWorklet, or a ScriptProcessorNode where worklets are missing. */
export class PcmMicStreamer {
  private ctx: AudioContext | null = null;
  private nodes: AudioNode[] = [];
  private resampler: Pcm16Resampler | null = null;
  private pending: Int16Array[] = [];
  private pendingLen = 0;
  private stopped = false;
  private readonly frameSamples: number;

  constructor(private stream: MediaStream, private onFrame: (buf: ArrayBuffer) => void, frameMs = 250) {
    this.frameSamples = Math.round((CAPTION_SAMPLE_RATE * frameMs) / 1000);
  }

  async start(): Promise<boolean> {
    const Ctor = audioContextCtor();
    if (!Ctor) return false;
    try {
      const ctx = new Ctor();
      this.ctx = ctx;
      if (ctx.state === "suspended") await ctx.resume().catch(() => undefined);
      this.resampler = new Pcm16Resampler(ctx.sampleRate);
      const source = ctx.createMediaStreamSource(this.stream);
      // Pull the graph without playing the mic back to the speakers.
      const mute = ctx.createGain();
      mute.gain.value = 0;
      mute.connect(ctx.destination);
      let tap: AudioNode;
      if (ctx.audioWorklet && typeof AudioWorkletNode !== "undefined") {
        const url = URL.createObjectURL(new Blob([WORKLET_SRC], { type: "application/javascript" }));
        try {
          await ctx.audioWorklet.addModule(url);
        } finally {
          URL.revokeObjectURL(url);
        }
        const node = new AudioWorkletNode(ctx, "signspeak-pcm-tap", { numberOfInputs: 1, numberOfOutputs: 1, channelCount: 1 });
        node.port.onmessage = (e: MessageEvent<Float32Array>) => this.handle(e.data);
        tap = node;
      } else {
        const node = ctx.createScriptProcessor(4096, 1, 1);
        node.onaudioprocess = (e) => this.handle(new Float32Array(e.inputBuffer.getChannelData(0)));
        tap = node;
      }
      source.connect(tap);
      tap.connect(mute);
      this.nodes = [source, tap, mute];
      if (this.stopped) { this.stop(); return false; }
      return true;
    } catch {
      this.stop();
      return false;
    }
  }

  private handle(samples: Float32Array): void {
    if (this.stopped || !this.resampler) return;
    const pcm = this.resampler.push(samples);
    if (!pcm.length) return;
    this.pending.push(pcm);
    this.pendingLen += pcm.length;
    if (this.pendingLen < this.frameSamples) return;
    const frame = new Int16Array(this.pendingLen);
    let off = 0;
    for (const p of this.pending) { frame.set(p, off); off += p.length; }
    this.pending = [];
    this.pendingLen = 0;
    this.onFrame(frame.buffer);
  }

  stop(): void {
    this.stopped = true;
    for (const n of this.nodes) { try { n.disconnect(); } catch { /* ignore */ } }
    this.nodes = [];
    const ctx = this.ctx;
    this.ctx = null;
    if (ctx && ctx.state !== "closed") void ctx.close().catch(() => undefined);
  }
}
