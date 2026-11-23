// Unit tests for frontend/lib/captions.ts — the parts that run without a
// microphone: the 16 kHz PCM16 resampler, capability detection, and the
// Whisper caption client's protocol handling (against a fake WebSocket).
// PcmMicStreamer needs a real AudioContext + mic, so it is exercised in
// manual two-browser testing instead.
import { describe, expect, it, vi, afterEach } from "vitest";
import { CAPTION_SAMPLE_RATE, Pcm16Resampler, WhisperCaptionClient, pcmStreamingSupported, type CaptionEvent } from "../captions";

afterEach(() => {
  vi.unstubAllGlobals();
  vi.restoreAllMocks();
});

function sine(n: number, rate: number, freq: number, amp = 0.5): Float32Array {
  const x = new Float32Array(n);
  for (let i = 0; i < n; i++) x[i] = amp * Math.sin((2 * Math.PI * freq * i) / rate);
  return x;
}

describe("Pcm16Resampler", () => {
  it("turns 1 s at 48 kHz into 1 s at 16 kHz", () => {
    const r = new Pcm16Resampler(48000);
    const out = r.push(sine(48000, 48000, 440));
    expect(out.length).toBe(CAPTION_SAMPLE_RATE);
  });

  it("gives the same samples whether fed in one piece or in odd-sized chunks", () => {
    const input = sine(44100, 44100, 300);
    const whole = new Pcm16Resampler(44100).push(input);
    const r = new Pcm16Resampler(44100);
    const parts: number[] = [];
    for (let i = 0; i < input.length; i += 128) parts.push(...r.push(input.subarray(i, i + 128)));
    expect(parts).toEqual(Array.from(whole));
  });

  it("keeps a low tone and clips out-of-range input to PCM16 limits", () => {
    const out = new Pcm16Resampler(48000).push(sine(4800, 48000, 200, 2.0));
    expect(Math.max(...out)).toBe(32767);
    expect(Math.min(...out)).toBe(-32768);
    const quiet = new Pcm16Resampler(48000).push(sine(4800, 48000, 200, 0.25));
    const peak = Math.max(...quiet);
    expect(peak).toBeGreaterThan(0.2 * 32767);
    expect(peak).toBeLessThan(0.26 * 32767);
  });

  it("rejects upsampling", () => {
    expect(() => new Pcm16Resampler(8000)).toThrow();
  });
});

describe("pcmStreamingSupported", () => {
  it("is false in Node (no window / AudioContext)", () => {
    expect(pcmStreamingSupported()).toBe(false);
  });
});

class FakeWebSocket {
  static OPEN = 1;
  static last: FakeWebSocket | null = null;
  readyState = 0;
  binaryType = "blob";
  sent: unknown[] = [];
  onopen: (() => void) | null = null;
  onmessage: ((ev: { data: string }) => void) | null = null;
  onerror: (() => void) | null = null;
  onclose: (() => void) | null = null;
  constructor(public url: string) { FakeWebSocket.last = this; }
  send(d: unknown) { this.sent.push(d); }
  close() { this.readyState = 3; }
  open() { this.readyState = 1; this.onopen?.(); }
  receive(msg: object) { this.onmessage?.({ data: JSON.stringify(msg) }); }
}

describe("WhisperCaptionClient", () => {
  function connect(language = "en-IN") {
    vi.stubGlobal("WebSocket", FakeWebSocket);
    const events: CaptionEvent[] = [];
    const onClose = vi.fn();
    const c = new WhisperCaptionClient("tok", { onEvent: (e) => events.push(e), onClose }, language);
    c.connect();
    const ws = FakeWebSocket.last!;
    return { c, ws, events, onClose };
  }

  it("sends the pcm16 config first, then binary audio", () => {
    const { c, ws } = connect();
    expect(ws.url).toContain("/ws/captions?token=tok");
    expect(ws.binaryType).toBe("arraybuffer");
    ws.open();
    expect(JSON.parse(ws.sent[0] as string)).toEqual({ type: "config", format: "pcm16", sample_rate: 16000, language: "en" });
    const buf = new Int16Array(4000).buffer;
    c.sendAudioChunk(buf);
    expect(ws.sent[1]).toBe(buf);
  });

  it("maps captions, status and coded errors", () => {
    const { ws, events } = connect();
    ws.open();
    ws.receive({ type: "status", state: "loading" });
    ws.receive({ type: "caption", text: " Hello there ", confidence: 0.8, engine: "whisper-tiny", latency_s: 1.2 });
    ws.receive({ type: "caption", text: "   " });
    ws.receive({ type: "error", code: "too-slow", message: "behind" });
    expect(events).toEqual([
      { type: "status", state: "loading" },
      { type: "final", text: "Hello there", confidence: 0.8, engine: "whisper-tiny", latencyS: 1.2 },
      { type: "error", message: "behind", code: "too-slow" },
    ]);
  });

  it("reports an unexpected close but not one it asked for", () => {
    const a = connect();
    a.ws.open();
    a.ws.onclose?.();
    expect(a.onClose).toHaveBeenCalledTimes(1);
    const b = connect();
    b.ws.open();
    b.c.close();
    b.ws.onclose?.();
    expect(b.onClose).not.toHaveBeenCalled();
  });
});
