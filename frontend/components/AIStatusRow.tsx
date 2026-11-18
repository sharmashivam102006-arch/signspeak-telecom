// Compact AI status row shown on the room page: what is real right now.
// Renders from GET /api/ai/status. When the backend does not expose that
// endpoint yet (null), it says so plainly instead of guessing.
"use client";
import { Hand, Mic, Volume2, Radio } from "lucide-react";
import type { AIStatus } from "@/lib/ai";
import { Badge } from "./ui/card";

function Pill({ label, state, tone }: { label: string; state: string; tone: "green" | "amber" | "red" | "slate" | "cyan" }) {
  return (
    <span className="flex items-center gap-1.5 rounded-full border border-slate-800 bg-panel/70 px-2.5 py-1 text-xs">
      <span className="text-muted">{label}:</span>
      <Badge tone={tone}>{state}</Badge>
    </span>
  );
}

export function AIStatusRow({ aiStatus, unavailableNote }: { aiStatus: AIStatus | null; unavailableNote: string }) {
  if (!aiStatus) {
    return (
      <div className="flex flex-wrap items-center gap-2 rounded-xl border border-slate-800 bg-panel/50 px-3 py-2 text-xs text-muted">
        <span className="font-medium text-slate-300">AI status</span>
        <span>{unavailableNote || "AI status is unavailable — the backend does not expose /api/ai/status yet."}</span>
      </div>
    );
  }
  const sign = aiStatus.sign;
  const whisper = aiStatus.whisper;
  const tts = aiStatus.tts;
  // rooms_active is an integer count from the backend; tolerate a legacy
  // list value (length) so the pill never renders an empty string.
  const rawRoomsActive: unknown = aiStatus.voice_agent?.rooms_active;
  const roomsActiveCount =
    typeof rawRoomsActive === "number"
      ? rawRoomsActive
      : Array.isArray(rawRoomsActive)
        ? rawRoomsActive.length
        : 0;
  return (
    <div className="flex flex-wrap items-center gap-2 rounded-xl border border-slate-800 bg-panel/50 px-3 py-2 text-xs">
      <span className="flex items-center gap-1 font-medium text-slate-300"><Radio size={13} /> AI status</span>
      <Pill
        label="Sign model"
        state={sign ? (sign.trained ? "Trained" : "Not trained") : "Unknown"}
        tone={sign ? (sign.trained ? "green" : "amber") : "slate"}
      />
      <Pill
        label="Whisper"
        state={whisper ? (whisper.available ? `Available${whisper.model ? ` (${whisper.model})` : ""}` : "Unavailable") : "Unknown"}
        tone={whisper ? (whisper.available ? "green" : "red") : "slate"}
      />
      <Pill
        label="TTS"
        state={tts ? (tts.available ? (tts.engine ?? "Available") : "Unavailable") : "Unknown"}
        tone={tts ? (tts.available ? "green" : "red") : "slate"}
      />
      <Pill
        label="Voice agent"
        state={aiStatus.voice_agent ? `${roomsActiveCount} room(s) active` : "Unknown"}
        tone={aiStatus.voice_agent ? (roomsActiveCount > 0 ? "cyan" : "slate") : "slate"}
      />
      <span className="ml-auto hidden items-center gap-1 text-muted sm:flex">
        <Hand size={12} /> <Mic size={12} /> <Volume2 size={12} /> Sign → speech · Speech → captions · Voice agent
      </span>
    </div>
  );
}
