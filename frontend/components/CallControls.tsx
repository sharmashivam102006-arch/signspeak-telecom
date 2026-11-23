"use client";
import type { ReactNode } from "react";
import { Mic, MicOff, Video, VideoOff, PhoneOff, Captions, PanelRight } from "lucide-react";

function Control({ label, onClick, tone, children }: {
  label: string; onClick: () => void; tone: "default" | "off" | "on-cyan" | "on-violet" | "danger"; children: ReactNode;
}) {
  const tones: Record<typeof tone, string> = {
    default: "bg-slate-700/80 text-slate-100 hover:bg-slate-600",
    off: "bg-red-500/90 text-white hover:bg-red-500",
    "on-cyan": "bg-cyan text-ink hover:bg-cyan-300",
    "on-violet": "bg-violet text-white hover:bg-violet-500",
    danger: "bg-red-500 text-white hover:bg-red-400",
  };
  return (
    <button onClick={onClick} aria-label={label} title={label} className="group flex flex-col items-center gap-1">
      <span className={`flex h-11 w-11 items-center justify-center rounded-full transition ${tones[tone]}`}>{children}</span>
      <span className="hidden text-[11px] text-muted group-hover:text-slate-200 sm:block">{label}</span>
    </button>
  );
}

export function CallControls({ micOn, camOn, captionsOn, panelOpen, onToggleMic, onToggleCam, onToggleCaptions, onTogglePanel, onEnd }: {
  micOn: boolean; camOn: boolean; captionsOn: boolean; panelOpen: boolean;
  onToggleMic: () => void; onToggleCam: () => void; onToggleCaptions: () => void; onTogglePanel: () => void; onEnd: () => void;
}) {
  return (
    <div className="mx-auto flex w-fit items-start gap-3 rounded-2xl border border-slate-800 bg-panel/80 px-4 py-2.5 sm:gap-4">
      <Control label={micOn ? "Mute" : "Unmute"} onClick={onToggleMic} tone={micOn ? "default" : "off"}>
        {micOn ? <Mic size={19} /> : <MicOff size={19} />}
      </Control>
      <Control label={camOn ? "Stop video" : "Start video"} onClick={onToggleCam} tone={camOn ? "default" : "off"}>
        {camOn ? <Video size={19} /> : <VideoOff size={19} />}
      </Control>
      <Control label="Captions" onClick={onToggleCaptions} tone={captionsOn ? "on-cyan" : "default"}>
        <Captions size={19} />
      </Control>
      <Control label="AI panel" onClick={onTogglePanel} tone={panelOpen ? "on-violet" : "default"}>
        <PanelRight size={19} />
      </Control>
      <span className="mx-1 h-11 w-px self-start bg-slate-700" aria-hidden />
      <Control label="End call" onClick={onEnd} tone="danger">
        <PhoneOff size={19} />
      </Control>
    </div>
  );
}
