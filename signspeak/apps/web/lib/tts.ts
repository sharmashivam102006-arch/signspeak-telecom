/**
 * Text -> speech for the sign user's side of the conversation.
 *
 * Two problems that look small and are not.
 *
 * 1. Autoplay policy. `speechSynthesis.speak()` is silently ignored until the
 *    page has had a real user gesture. If the first thing the app does is
 *    speak, nothing comes out, no error is thrown, and you spend an afternoon
 *    debugging the classifier. `unlock()` must be called from a click - the
 *    Join button is the natural place.
 *
 * 2. The voice does not reach the other person. `speechSynthesis` plays
 *    through the local speakers, which is the wrong end of the call: it is
 *    audible to the signer, who cannot hear it, and inaudible to the hearing
 *    participant, who needs it. The fix is to render TTS into a MediaStream
 *    and publish that as an extra LiveKit audio track - `synthesizeToTrack()`
 *    below. The naive version demos fine in one room with two laptops and
 *    fails the moment the call is actually remote.
 */

export interface TtsOptions {
  voiceURI?: string | null;
  rate?: number;
  pitch?: number;
  volume?: number;
}

let unlocked = false;

/** Call from a click handler, once. */
export function unlock(): void {
  if (unlocked || typeof window === "undefined" || !window.speechSynthesis) return;
  const u = new SpeechSynthesisUtterance("");
  u.volume = 0;
  window.speechSynthesis.speak(u);
  unlocked = true;
}

export function isUnlocked(): boolean {
  return unlocked;
}

export async function listVoices(): Promise<SpeechSynthesisVoice[]> {
  if (typeof window === "undefined" || !window.speechSynthesis) return [];
  const existing = window.speechSynthesis.getVoices();
  if (existing.length) return existing;
  // Chrome populates the list asynchronously and fires voiceschanged once.
  return new Promise((resolve) => {
    const t = setTimeout(() => resolve(window.speechSynthesis.getVoices()), 1200);
    window.speechSynthesis.addEventListener(
      "voiceschanged",
      () => {
        clearTimeout(t);
        resolve(window.speechSynthesis.getVoices());
      },
      { once: true },
    );
  });
}

export interface SpeakResult {
  /** Request -> audio actually starts. Part of the end-to-end budget and
   * usually 50-150ms locally; cloud TTS is 200-500ms. Measure, don't assume. */
  latencyMs: number;
  spoken: boolean;
}

export function speak(text: string, opts: TtsOptions = {}): Promise<SpeakResult> {
  return new Promise((resolve) => {
    if (typeof window === "undefined" || !window.speechSynthesis) {
      resolve({ latencyMs: 0, spoken: false });
      return;
    }
    if (!unlocked) {
      console.warn("[tts] not unlocked - speak() will be silently dropped by the browser");
    }

    const t0 = performance.now();
    const u = new SpeechSynthesisUtterance(text);
    u.rate = opts.rate ?? 1.0;
    u.pitch = opts.pitch ?? 1.0;
    u.volume = opts.volume ?? 1.0;

    if (opts.voiceURI) {
      const v = window.speechSynthesis.getVoices().find((x) => x.voiceURI === opts.voiceURI);
      if (v) u.voice = v;
    }

    let settled = false;
    const done = (spoken: boolean) => {
      if (settled) return;
      settled = true;
      resolve({ latencyMs: performance.now() - t0, spoken });
    };

    u.onstart = () => done(true);
    u.onerror = () => done(false);
    // Very short utterances sometimes skip onstart entirely.
    setTimeout(() => done(true), 1500);

    window.speechSynthesis.speak(u);
  });
}

export function cancel(): void {
  if (typeof window !== "undefined" && window.speechSynthesis) {
    window.speechSynthesis.cancel();
  }
}

/**
 * Render TTS into a MediaStreamTrack so it can be published to the call.
 *
 * `speechSynthesis` has no route into a MediaStream, so this needs a TTS
 * endpoint that returns audio (any server-side engine - Piper, Coqui, a cloud
 * API). The returned track is published as a second audio track alongside the
 * microphone; the hearing participant's client mixes both.
 *
 * Publish it as a separate track rather than mixing into the mic: it keeps the
 * synthesised voice out of the signer's own echo cancellation, and it lets the
 * receiver label it ("spoken by SignSpeak") instead of passing it off as a
 * human voice - which matters, because the hearing participant should know
 * when they are listening to a machine's guess.
 */
export async function synthesizeToTrack(
  text: string,
  endpoint = "/api/tts",
): Promise<MediaStreamTrack | null> {
  try {
    const res = await fetch(endpoint, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ text }),
    });
    if (!res.ok) return null;

    const buf = await res.arrayBuffer();
    const ctx = new AudioContext();
    const audio = await ctx.decodeAudioData(buf);
    const dest = ctx.createMediaStreamDestination();
    const src = ctx.createBufferSource();
    src.buffer = audio;
    src.connect(dest);
    src.start();
    return dest.stream.getAudioTracks()[0] ?? null;
  } catch {
    return null;
  }
}
