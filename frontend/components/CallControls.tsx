"use client";
import { Mic, MicOff, Video, VideoOff, PhoneOff, Captions, PanelRight } from "lucide-react";
export function CallControls({ micOn, camOn, captionsOn, panelOpen, onToggleMic, onToggleCam, onToggleCaptions, onTogglePanel, onEnd }: {
  micOn: boolean; camOn: boolean; captionsOn: boolean; panelOpen: boolean;
  onToggleMic: () => void; onToggleCam: () => void; onToggleCaptions: () => void; onTogglePanel: () => void; onEnd: () => void;
}) {
  const btn = "flex h-12 w-12 items-center justify-center rounded-full transition";
  return (
    <div className="flex items-center justify-center gap-3">
      <button onClick={onToggleMic} aria-label={micOn ? "Mute microphone" : "Unmute microphone"} title={micOn ? "Mute microphone" : "Unmute microphone"} className={`${btn} ${micOn ? "bg-slate-700 hover:bg-slate-600" : "bg-red-500"}`}>{micOn ? <Mic size={20} /> : <MicOff size={20} />}</button>
      <button onClick={onToggleCam} aria-label={camOn ? "Turn camera off" : "Turn camera on"} title={camOn ? "Turn camera off" : "Turn camera on"} className={`${btn} ${camOn ? "bg-slate-700 hover:bg-slate-600" : "bg-red-500"}`}>{camOn ? <Video size={20} /> : <VideoOff size={20} />}</button>
      <button onClick={onEnd} aria-label="End call" title="End call" className={`${btn} h-14 w-14 bg-red-500 hover:bg-red-400`}><PhoneOff size={22} /></button>
      <button onClick={onToggleCaptions} aria-label="Toggle captions" title="Toggle live captions" className={`${btn} ${captionsOn ? "bg-cyan text-ink" : "bg-slate-700 hover:bg-slate-600"}`}><Captions size={20} /></button>
      <button onClick={onTogglePanel} aria-label="Toggle AI panel" title="Toggle AI panel" className={`${btn} ${panelOpen ? "bg-violet text-white" : "bg-slate-700 hover:bg-slate-600"}`}><PanelRight size={20} /></button>
    </div>
  );
}
