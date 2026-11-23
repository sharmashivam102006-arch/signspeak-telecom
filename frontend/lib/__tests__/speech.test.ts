// Unit tests for frontend/lib/speech.ts — the browser (Web Speech API)
// caption fallback, against a fake SpeechRecognition.
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { createSpeechRecognizer } from "../speech";

type Result = { isFinal: boolean; 0: { transcript: string } };

class FakeRecognition {
  static instances: FakeRecognition[] = [];
  continuous = false;
  interimResults = false;
  lang = "";
  starts = 0;
  running = false;
  onresult: ((e: { resultIndex: number; results: Result[] }) => void) | null = null;
  onerror: ((e: { error: string }) => void) | null = null;
  onend: (() => void) | null = null;
  constructor() { FakeRecognition.instances.push(this); }
  start() {
    if (this.running) throw new Error("InvalidStateError");
    this.running = true;
    this.starts += 1;
  }
  stop() { if (this.running) { this.running = false; this.onend?.(); } }
  abort() { this.stop(); }
  // Test helpers: what Chrome does.
  end() { this.running = false; this.onend?.(); }
  results(resultIndex: number, rs: [string, boolean][]) {
    this.onresult?.({ resultIndex, results: rs.map(([t, f]) => ({ isFinal: f, 0: { transcript: t } })) });
  }
}

function handlers() {
  return { onInterim: vi.fn(), onFinal: vi.fn(), onError: vi.fn(), onEnd: vi.fn() };
}

beforeEach(() => {
  FakeRecognition.instances = [];
  vi.useFakeTimers();
  vi.stubGlobal("window", { webkitSpeechRecognition: FakeRecognition });
});

afterEach(() => {
  vi.useRealTimers();
  vi.unstubAllGlobals();
});

describe("createSpeechRecognizer", () => {
  it("restarts by itself when the browser ends the session after a pause", () => {
    const h = handlers();
    const r = createSpeechRecognizer("en-US", h)!;
    r.start();
    const rec = FakeRecognition.instances[0];
    expect(rec.starts).toBe(1);
    vi.advanceTimersByTime(10_000); // session ran a while
    rec.end();
    vi.advanceTimersByTime(300);
    expect(rec.starts).toBe(2);
    expect(h.onEnd).not.toHaveBeenCalled();
    expect(h.onError).not.toHaveBeenCalled();
  });

  it("treats no-speech as normal, not as an error", () => {
    const h = handlers();
    createSpeechRecognizer("en-US", h)!.start();
    const rec = FakeRecognition.instances[0];
    rec.onerror?.({ error: "no-speech" });
    rec.end();
    vi.advanceTimersByTime(300);
    expect(h.onError).not.toHaveBeenCalled();
    expect(rec.starts).toBe(2);
  });

  it("backs off when sessions die immediately", () => {
    const h = handlers();
    createSpeechRecognizer("en-US", h)!.start();
    const rec = FakeRecognition.instances[0];
    rec.onerror?.({ error: "network" });
    rec.end();
    vi.advanceTimersByTime(250);
    expect(rec.starts).toBe(2);
    rec.end();
    vi.advanceTimersByTime(250);
    expect(rec.starts).toBe(2); // second retry waits 500 ms
    vi.advanceTimersByTime(250);
    expect(rec.starts).toBe(3);
  });

  it("stops for good on a fatal error and reports it", () => {
    const h = handlers();
    createSpeechRecognizer("en-US", h)!.start();
    const rec = FakeRecognition.instances[0];
    rec.onerror?.({ error: "not-allowed" });
    rec.end();
    vi.advanceTimersByTime(10_000);
    expect(h.onError).toHaveBeenCalledWith("Microphone permission denied for speech recognition.");
    expect(h.onEnd).toHaveBeenCalledTimes(1);
    expect(rec.starts).toBe(1);
  });

  it("does not restart after stop()", () => {
    const h = handlers();
    const r = createSpeechRecognizer("en-US", h)!;
    r.start();
    r.stop();
    vi.advanceTimersByTime(10_000);
    expect(FakeRecognition.instances[0].starts).toBe(1);
    expect(h.onEnd).toHaveBeenCalledTimes(1);
  });

  it("emits each final result once, not the whole session so far", () => {
    const h = handlers();
    createSpeechRecognizer("en-US", h)!.start();
    const rec = FakeRecognition.instances[0];
    rec.results(0, [["hello there", false]]);
    rec.results(0, [["hello there", true]]);
    rec.results(1, [["hello there", true], ["how are you", true]]);
    expect(h.onFinal.mock.calls.map(c => c[0])).toEqual(["hello there", "how are you"]);
    expect(h.onInterim).toHaveBeenCalledWith("hello there");
  });
});
