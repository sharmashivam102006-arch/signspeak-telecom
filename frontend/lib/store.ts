export type Direction = "sign-to-speech" | "speech-to-text";
export type TranscriptEntry = {
  id: string; text: string; direction: Direction;
  source: "sign" | "speech" | "system";
  confidence?: number; demo?: boolean;
  timestamp: string; // ISO
};
export type CallRecord = { roomId: string; startedAt: string; endedAt?: string; durationSec?: number; participants?: number };

export type Settings = {
  captionSize: "small" | "medium" | "large";
  highContrast: boolean;
  ttsMuted: boolean; ttsVolume: number; ttsVoice: string;
  signThresholdHigh: number; signThresholdLow: number;
  language: string;
};
export const DEFAULT_SETTINGS: Settings = {
  captionSize: "medium", highContrast: false,
  ttsMuted: false, ttsVolume: 0.9, ttsVoice: "",
  signThresholdHigh: 85, signThresholdLow: 60, language: "en-IN",
};
const SETTINGS_KEY = "signspeak_settings";
export function loadSettings(): Settings {
  if (typeof window === "undefined") return DEFAULT_SETTINGS;
  try { return { ...DEFAULT_SETTINGS, ...JSON.parse(localStorage.getItem(SETTINGS_KEY) || "{}") }; }
  catch { return DEFAULT_SETTINGS; }
}
export function saveSettings(s: Settings) { localStorage.setItem(SETTINGS_KEY, JSON.stringify(s)); }

export function loadTranscript(roomId: string): TranscriptEntry[] {
  try { return JSON.parse(localStorage.getItem(`signspeak_transcript_${roomId}`) || "[]"); } catch { return []; }
}
export function saveTranscript(roomId: string, entries: TranscriptEntry[]) {
  localStorage.setItem(`signspeak_transcript_${roomId}`, JSON.stringify(entries.slice(-300)));
}
export function loadRecentTranscripts(): { roomId: string; entries: TranscriptEntry[] }[] {
  if (typeof window === "undefined") return [];
  const out: { roomId: string; entries: TranscriptEntry[] }[] = [];
  for (let i = 0; i < localStorage.length; i++) {
    const k = localStorage.key(i);
    if (k?.startsWith("signspeak_transcript_")) {
      try { out.push({ roomId: k.replace("signspeak_transcript_", ""), entries: JSON.parse(localStorage.getItem(k) || "[]") }); } catch { /* ignore */ }
    }
  }
  return out.sort((a, b) => (b.entries.at(-1)?.timestamp || "").localeCompare(a.entries.at(-1)?.timestamp || ""));
}
export function loadCalls(): CallRecord[] {
  try { return JSON.parse(localStorage.getItem("signspeak_calls") || "[]"); } catch { return []; }
}
export function saveCall(rec: CallRecord) {
  const all = loadCalls(); all.unshift(rec);
  localStorage.setItem("signspeak_calls", JSON.stringify(all.slice(0, 20)));
}
export function transcriptToText(entries: TranscriptEntry[]): string {
  return entries.map(e => `[${new Date(e.timestamp).toLocaleTimeString()}] ${e.direction === "sign-to-speech" ? "SIGN → SPEECH" : "SPEECH → TEXT"}${e.demo ? " (Demo)" : ""}: ${e.text}`).join("\n");
}
export const SIGN_VOCABULARY = ["Hello", "Yes", "No", "Help", "Thank You", "Emergency"] as const;
