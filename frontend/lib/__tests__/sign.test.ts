// Unit tests for frontend/lib/sign.ts — the pure logic that guards the
// live sign path: sequence buffering, prediction smoothing / duplicate
// suppression, movement detection and landmark normalisation.
// (createHandTracker / predictSign need a browser + backend, so they are
// covered by the backend test suite and manual two-browser testing instead.)
import { describe, expect, it, vi, beforeEach, afterEach } from "vitest";
import {
  SequenceBuffer,
  PredictionSmoother,
  sequenceMoved,
  normaliseLandmarks,
  type Landmark,
} from "../sign";

function frame(x = 0.1, y = 0.2, z = 0): Landmark[] {
  return Array.from({ length: 21 }, (_, i) => ({
    x: x + i * 0.001,
    y: y + i * 0.001,
    z,
  }));
}

describe("SequenceBuffer", () => {
  it("ignores frames with fewer than 21 landmarks", () => {
    const buf = new SequenceBuffer();
    buf.push([{ x: 0, y: 0, z: 0 }]);
    expect(buf.length).toBe(0);
    expect(buf.ready).toBe(false);
  });

  it("becomes ready at 30 frames and caps at 30 (oldest dropped)", () => {
    const buf = new SequenceBuffer();
    for (let i = 0; i < 35; i++) buf.push(frame(0.1 + i * 0.01));
    expect(buf.length).toBe(30);
    expect(buf.ready).toBe(true);
    // First frame (x=0.1) was dropped; the oldest kept frame is i=5.
    expect(buf.snapshot()[0][0].x).toBeCloseTo(0.15, 6);
  });

  it("snapshot copies the frame list (replacing a frame does not touch the buffer)", () => {
    const buf = new SequenceBuffer();
    buf.push(frame());
    const snap = buf.snapshot();
    snap[0] = frame(9, 9);
    expect(buf.lastFrame()![0].x).toBeCloseTo(0.1, 6);
    buf.clear();
    expect(buf.length).toBe(0);
    expect(buf.lastFrame()).toBeNull();
  });
});

describe("PredictionSmoother (duplicate suppression)", () => {
  beforeEach(() => vi.useFakeTimers());
  afterEach(() => {
    vi.useRealTimers();
    vi.restoreAllMocks();
  });

  it("rejects predictions below the confidence threshold", () => {
    const s = new PredictionSmoother();
    expect(s.accept("Hello", 0.5, 0.6)).toBeNull();
    expect(s.accept("Hello", 0.6, 0.6)).not.toBeNull();
  });

  it("suppresses the same sign repeated within 5 seconds", () => {
    const s = new PredictionSmoother();
    expect(s.accept("Hello", 0.9, 0.6)).toEqual({ sign: "Hello", confidence: 0.9 });
    vi.advanceTimersByTime(4000);
    expect(s.accept("Hello", 0.9, 0.6)).toBeNull();
  });

  it("allows the same sign again after 5 seconds, and a different sign immediately", () => {
    const s = new PredictionSmoother();
    s.accept("Hello", 0.9, 0.6);
    expect(s.accept("Yes", 0.9, 0.6)).toEqual({ sign: "Yes", confidence: 0.9 });
    vi.advanceTimersByTime(5000);
    expect(s.accept("Yes", 0.9, 0.6)).toEqual({ sign: "Yes", confidence: 0.9 });
  });

  it("reset clears the suppression window", () => {
    const s = new PredictionSmoother();
    s.accept("Hello", 0.9, 0.6);
    s.reset();
    expect(s.accept("Hello", 0.9, 0.6)).not.toBeNull();
  });
});

describe("sequenceMoved", () => {
  it("returns false for fewer than 2 frames and for a held-still hand", () => {
    expect(sequenceMoved([])).toBe(false);
    expect(sequenceMoved([frame()])).toBe(false);
    expect(sequenceMoved([frame(), frame()])).toBe(false);
  });

  it("returns true when the hand moved across the sequence", () => {
    expect(sequenceMoved([frame(0.1, 0.2), frame(0.5, 0.6)])).toBe(true);
  });
});

describe("normaliseLandmarks", () => {
  it("returns [] for fewer than 21 landmarks", () => {
    expect(normaliseLandmarks([{ x: 0, y: 0, z: 0 }])).toEqual([]);
  });

  it("translates the wrist to the origin and scales by wrist→middle-MCP", () => {
    const lm = frame(0.2, 0.3, 0);
    const out = normaliseLandmarks(lm);
    expect(out).toHaveLength(63);
    // Wrist (landmark 0) lands on the origin.
    expect(out[0]).toBe(0);
    expect(out[1]).toBe(0);
    expect(out[2]).toBe(0);
    // Middle-MCP (landmark 9) is at unit distance from the wrist.
    const d = Math.hypot(out[9 * 3], out[9 * 3 + 1], out[9 * 3 + 2]);
    expect(d).toBeCloseTo(1, 6);
  });
});
