// Web Speech API (SpeechRecognition) for speech → text live captions.
// Interim + final results, deduped. Falls back honestly when unsupported.
//
// Chrome ends a recognition session on its own after a pause in speech,
// after ~60 s, or on a transient network error — even with
// continuous=true. The recognizer below restarts itself (with backoff)
// until stop() is called, so captions keep running for the whole call.
// Only fatal errors (permission denied, no microphone, unsupported
// language) are reported through onError; "no-speech" and "aborted" are
// normal and handled by the restart.
export type SpeechHandlers = {
  onInterim: (text: string) => void;
  onFinal: (text: string) => void;
  /** Fatal: recognition has stopped and will not restart. */
  onError: (msg: string) => void;
  /** Recognition stopped for good (stop() was called, or a fatal error). */
  onEnd: () => void;
};
type RecogResult = { isFinal: boolean; 0: { transcript: string } };
type Recog = {
  continuous: boolean; interimResults: boolean; lang: string;
  start: () => void; stop: () => void; abort: () => void;
  onresult: ((e: { resultIndex?: number; results: ArrayLike<RecogResult> }) => void) | null;
  onerror: ((e: { error: string }) => void) | null;
  onend: (() => void) | null;
};

const FATAL_ERRORS: Record<string, string> = {
  "not-allowed": "Microphone permission denied for speech recognition.",
  "service-not-allowed": "Speech recognition is blocked in this browser.",
  "audio-capture": "No microphone is available for speech recognition.",
  "language-not-supported": "This caption language is not supported by the browser's speech recognition.",
};
const RESTART_BASE_MS = 250;
const RESTART_MAX_MS = 5000;

export function speechSupported(): boolean {
  if (typeof window === "undefined") return false;
  return Boolean((window as unknown as { SpeechRecognition?: unknown; webkitSpeechRecognition?: unknown }).SpeechRecognition
    || (window as unknown as { webkitSpeechRecognition?: unknown }).webkitSpeechRecognition);
}

export function createSpeechRecognizer(lang: string, h: SpeechHandlers): { start: () => void; stop: () => void } | null {
  if (!speechSupported()) { h.onError("Speech recognition is not supported in this browser. Use Chrome or Edge for live captions."); return null; }
  const Ctor = (window as unknown as { SpeechRecognition?: new () => Recog; webkitSpeechRecognition?: new () => Recog }).SpeechRecognition
    || (window as unknown as { webkitSpeechRecognition: new () => Recog }).webkitSpeechRecognition;
  const rec = new Ctor();
  rec.continuous = true; rec.interimResults = true; rec.lang = lang || "en-IN";

  let active = false;
  let failures = 0; // consecutive sessions that ended almost immediately
  let startedAt = 0;
  let timer: ReturnType<typeof setTimeout> | null = null;
  let lastFinal = "";

  const tryStart = () => {
    timer = null;
    if (!active) return;
    startedAt = Date.now();
    try { rec.start(); } catch { /* already started */ }
  };

  rec.onresult = (e) => {
    // Only look at results that changed in this event: earlier final
    // results stay in e.results for the whole session and must not be
    // re-emitted (that made each caption repeat everything said so far).
    let interim = "";
    for (let i = e.resultIndex ?? 0; i < e.results.length; i++) {
      const r = e.results[i];
      const t = r[0].transcript.trim();
      if (!t) continue;
      if (r.isFinal) {
        if (t !== lastFinal) { lastFinal = t; h.onFinal(t); }
      } else {
        interim += (interim ? " " : "") + t;
      }
    }
    h.onInterim(interim);
  };
  rec.onerror = (e) => {
    const fatal = FATAL_ERRORS[e.error];
    if (fatal) {
      active = false;
      h.onError(fatal);
    }
    // Anything else (no-speech, aborted, network): onend follows and restarts.
  };
  rec.onend = () => {
    h.onInterim("");
    if (!active) { h.onEnd(); return; }
    // A session that ran a while (normal end after a pause) restarts at
    // once; only sessions that die immediately back off, so a broken
    // service is not hammered.
    if (Date.now() - startedAt > 3000) failures = 0;
    const delay = Math.min(RESTART_BASE_MS * 2 ** failures, RESTART_MAX_MS);
    failures += 1;
    timer = setTimeout(tryStart, delay);
  };
  return {
    start: () => { active = true; failures = 0; tryStart(); },
    stop: () => {
      active = false;
      if (timer) { clearTimeout(timer); timer = null; }
      try { rec.stop(); } catch { /* ignore */ }
    },
  };
}
