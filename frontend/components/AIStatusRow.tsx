// Compact AI status strip on the room page: what is available right now,
// in plain words. Renders from GET /api/ai/status. When the backend does
// not expose that endpoint (null), it says so instead of guessing.
"use client";
import { Hand, Mic, Volume2 } from "lucide-react";
import type { ReactNode } from "react";
import type { AIStatus } from "@/lib/ai";

type Tone = "green" | "amber" | "red" | "slate";
const DOT: Record<Tone, string> = {
  green: "bg-success", amber: "bg-amber-400", red: "bg-red-400", slate: "bg-slate-500",
};

function Item({ icon, label, state, tone, title }: { icon: ReactNode; label: string; state: string; tone: Tone; title?: string }) {
  return (
    <span className="flex items-center gap-1.5" title={title}>
      <span className="text-muted">{icon}</span>
      <span className="text-slate-300">{label}</span>
      <span className={`h-1.5 w-1.5 rounded-full ${DOT[tone]}`} aria-hidden />
      <span className="text-muted">{state}</span>
    </span>
  );
}

export function AIStatusRow({ aiStatus, unavailableNote }: { aiStatus: AIStatus | null; unavailableNote: string }) {
  if (!aiStatus) {
    return (
      <p className="text-xs text-muted">
        {unavailableNote || "AI status is unavailable — the backend does not expose /api/ai/status yet."}
      </p>
    );
  }
  const sign = aiStatus.sign;
  const whisper = aiStatus.whisper;
  const tts = aiStatus.tts;
  return (
    <div className="flex flex-wrap items-center gap-x-5 gap-y-1.5 text-xs">
      <Item
        icon={<Hand size={13} />} label="Signs"
        state={sign ? (sign.trained ? "Trained model" : "Not trained") : "Unknown"}
        tone={sign ? (sign.trained ? "green" : "amber") : "slate"}
      />
      <Item
        icon={<Mic size={13} />} label="Captions"
        state={whisper ? (whisper.available ? "Server (Whisper)" : "Browser only") : "Unknown"}
        tone={whisper ? (whisper.available ? "green" : "amber") : "slate"}
        title={whisper?.available ? `Whisper ${whisper.model ?? ""}`.trim() : "Server Whisper is unavailable; captions use the browser's speech recognition."}
      />
      <Item
        icon={<Volume2 size={13} />} label="Voice"
        state={tts ? (tts.available ? "Server voice" : "Unavailable") : "Unknown"}
        tone={tts ? (tts.available ? "green" : "red") : "slate"}
        title={tts?.engine ?? undefined}
      />
    </div>
  );
}
