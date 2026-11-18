// Web Speech API (SpeechRecognition) for speech → text live captions.
// Interim + final results, deduped. Falls back honestly when unsupported.
export type SpeechHandlers = {
  onInterim: (text: string) => void;
  onFinal: (text: string) => void;
  onError: (msg: string) => void;
  onEnd: () => void;
};
type Recog = {
  continuous: boolean; interimResults: boolean; lang: string;
  start: () => void; stop: () => void; abort: () => void;
  onresult: ((e: { results: ArrayLike<{ isFinal: boolean; 0: { transcript: string } }> }) => void) | null;
  onerror: ((e: { error: string }) => void) | null;
  onend: (() => void) | null;
};
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
  let lastFinal = "";
  rec.onresult = (e) => {
    let interim = "", finalText = "";
    for (let i = 0; i < e.results.length; i++) {
      const r = e.results[i];
      if (r.isFinal) finalText += r[0].transcript;
      else interim += r[0].transcript;
    }
    if (finalText.trim() && finalText.trim() !== lastFinal) { lastFinal = finalText.trim(); h.onFinal(finalText.trim()); }
    if (interim.trim()) h.onInterim(interim.trim());
  };
  rec.onerror = (e) => h.onError(e.error === "not-allowed" ? "Microphone permission denied for speech recognition." : `Speech recognition error: ${e.error}`);
  rec.onend = () => h.onEnd();
  return {
    start: () => { try { rec.start(); } catch { /* already started */ } },
    stop: () => { try { rec.stop(); } catch { /* ignore */ } },
  };
}
