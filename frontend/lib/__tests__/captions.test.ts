// Unit tests for frontend/lib/captions.ts — the parts that run without a
// microphone or a WebSocket: audio MIME-type selection and capability
// detection. (WhisperCaptionClient / MicChunkStreamer need a live browser
// MediaRecorder + WS server, so they are exercised in manual two-browser
// testing instead.)
//
// NOTE: captions.ts itself contains no de-duplication function — caption
// de-dup lives in the call page / PredictionSmoother (see sign.test.ts).
// These tests cover what the module actually exports.
import { describe, expect, it, vi, afterEach } from "vitest";
import { pickAudioMimeType, mediaRecorderSupported } from "../captions";

afterEach(() => {
  vi.unstubAllGlobals();
  vi.restoreAllMocks();
});

describe("pickAudioMimeType", () => {
  it("returns '' when MediaRecorder is unavailable (e.g. Node / old browser)", () => {
    expect(typeof MediaRecorder).toBe("undefined");
    expect(pickAudioMimeType()).toBe("");
  });

  it("prefers Opus-in-WebM when the browser supports it", () => {
    vi.stubGlobal(
      "MediaRecorder",
      class {
        static isTypeSupported(mt: string) {
          return mt === "audio/webm;codecs=opus";
        }
      },
    );
    expect(pickAudioMimeType()).toBe("audio/webm;codecs=opus");
  });

  it("falls through the preference list to the first supported type", () => {
    vi.stubGlobal(
      "MediaRecorder",
      class {
        static isTypeSupported(mt: string) {
          return mt === "audio/mp4";
        }
      },
    );
    expect(pickAudioMimeType()).toBe("audio/mp4");
  });

  it("returns '' when the browser supports no listed type", () => {
    vi.stubGlobal("MediaRecorder", class {
      static isTypeSupported() {
        return false;
      }
    });
    expect(pickAudioMimeType()).toBe("");
  });
});

describe("mediaRecorderSupported", () => {
  it("is false in Node (no window / MediaRecorder)", () => {
    expect(mediaRecorderSupported()).toBe(false);
  });
});
