// Browser SpeechSynthesis TTS for sign → speech.
// Speaks only recognised text, dedupes identical utterances, never autoplays duplicates.
let lastSpoken = { text: "", at: 0 };

export function ttsSupported(): boolean {
  return typeof window !== "undefined" && "speechSynthesis" in window;
}
export function listVoices(): SpeechSynthesisVoice[] {
  return ttsSupported() ? window.speechSynthesis.getVoices() : [];
}
export function speakText(text: string, opts: { muted?: boolean; volume?: number; voiceName?: string; lang?: string } = {}): boolean {
  if (!ttsSupported() || opts.muted || !text.trim()) return false;
  const now = Date.now();
  if (lastSpoken.text === text.trim() && now - lastSpoken.at < 4000) return false; // duplicate suppression
  window.speechSynthesis.cancel();
  const u = new SpeechSynthesisUtterance(text);
  u.volume = opts.volume ?? 0.9;
  u.lang = opts.lang || "en-IN";
  const voice = listVoices().find(v => v.name === opts.voiceName);
  if (voice) u.voice = voice;
  lastSpoken = { text: text.trim(), at: now };
  window.speechSynthesis.speak(u);
  return true;
}
export function stopSpeaking() { if (ttsSupported()) window.speechSynthesis.cancel(); }
