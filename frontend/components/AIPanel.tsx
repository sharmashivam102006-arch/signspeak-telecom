// AI communication panel (room sidebar).
//
// Honest states only:
//  * Sign recognition — "Trained model" only when the backend reports a
//    trained model; otherwise a "Sign model not trained yet" card with the
//    per-sign sample counts and a button into the trainer. Demo buttons
//    call POST /predict with demo=true and their results carry a DEMO badge.
//  * Speech → captions — badge says "Whisper tiny" or "Browser captions
//    (not Whisper)" depending on what actually produced the captions.
//  * Sign → speech — accepted signs go to the remote participant through
//    the room voice agent (LiveKit audio). Browser speechSynthesis is
//    only a local preview, labelled as such, and only when the voice
//    agent is off. The "Say this" box speaks typed text the same way.
"use client";
import { useState } from "react";
import { Hand, Mic, Volume2, VolumeX, Sparkles, GraduationCap, Megaphone, Loader2, AlertTriangle } from "lucide-react";
import { SIGN_VOCABULARY } from "@/lib/store";
import { Badge } from "./ui/card";
import { Button } from "./ui/button";

export function AIPanel({ signStatus, speechStatus, ttsMuted, onToggleTts, onDemoSign, signEnabled, onToggleSign,
  signTrained, sampleCounts, onOpenTrainer, captionEngine, voiceAgentOn, onToggleVoiceAgent, sayText, onSayTextChange, onSay, speaking, speakError, ttsEngine,
  voiceVolume, onVoiceVolumeChange, voiceMuted, onToggleVoiceMute, relayState }: {
  signStatus: "real" | "demo" | "unavailable";
  speechStatus: "ready" | "unsupported" | "listening" | "error";
  ttsMuted: boolean; onToggleTts: () => void;
  onDemoSign: (sign: string) => void;
  signEnabled: boolean; onToggleSign: () => void;
  /** null = unknown (backend did not report); false = not trained yet. */
  signTrained: boolean | null;
  sampleCounts: Record<string, number>;
  onOpenTrainer: () => void;
  captionEngine: "whisper" | "browser" | null;
  /** null = unknown/unavailable; the voice agent speaks into the LiveKit room. */
  voiceAgentOn: boolean | null;
  onToggleVoiceAgent: () => void;
  sayText: string; onSayTextChange: (v: string) => void; onSay: () => void;
  speaking: boolean; speakError: string | null;
  /** Server TTS engine label, e.g. "piper-lessac-medium" — null when unknown. */
  ttsEngine: string | null;
  /** Voice-agent (remote) audio level 0–1 and mute, applied to the agent tile. */
  voiceVolume: number; onVoiceVolumeChange: (v: number) => void;
  voiceMuted: boolean; onToggleVoiceMute: () => void;
  /** Caption relay (/ws/call) state: remote captions only arrive while connected. */
  relayState: "connected" | "reconnecting" | "disconnected";
}) {
  const [sayOpen, setSayOpen] = useState(false);
  return (
    <div className="space-y-4">
      <div>
        <div className="mb-2 flex items-center justify-between">
          <h3 className="flex items-center gap-1.5 text-sm font-semibold"><Hand size={15} className="text-violet-300" /> Sign recognition</h3>
          <Badge tone={signStatus === "real" ? "green" : signStatus === "demo" ? "amber" : "red"}>
            {signStatus === "real" ? "Trained model" : signStatus === "demo" ? "Demo mode" : "Unavailable"}
          </Badge>
        </div>
        <p className="text-xs text-muted">
          {signStatus === "real"
            ? "A trained sign model is serving predictions. Sign in front of your camera; predictions are sent to the backend model."
            : signStatus === "demo"
              ? "Demo mode: no trained sign model is available, so live predictions use the labelled demo rule-based path (or MediaPipe could not be loaded). Use the demo buttons below — they are clearly labelled Demo and are not real model predictions."
              : "Sign recognition is unavailable right now. Demo buttons below still work for testing the pipeline."}
        </p>
        {signTrained === false && (
          <div className="mt-3 rounded-xl border border-amber-500/40 bg-amber-500/5 p-3">
            <p className="text-sm font-semibold text-amber-200">Sign model not trained yet</p>
            <p className="mt-1 text-xs text-muted">
              Live recognition needs a trained model. Record samples for each sign, then train — no predictions
              are shown until training finishes.
            </p>
            <div className="mt-2 grid grid-cols-2 gap-1 text-xs">
              {SIGN_VOCABULARY.map((s) => (
                <span key={s} className="flex justify-between rounded bg-ink/60 px-2 py-1">
                  <span>{s}</span><span className="text-muted">{sampleCounts[s] ?? 0}/10</span>
                </span>
              ))}
            </div>
            <Button size="sm" className="mt-2.5 w-full" onClick={onOpenTrainer}>
              <GraduationCap size={14} /> Train signs
            </Button>
          </div>
        )}
        <div className="mt-2 flex gap-2">
          <Button size="sm" variant={signEnabled ? "primary" : "outline"} onClick={onToggleSign}>{signEnabled ? "Sign detection on" : "Sign detection off"}</Button>
          {signTrained !== false && (
            <Button size="sm" variant="outline" onClick={onOpenTrainer}><GraduationCap size={14} /> Train signs</Button>
          )}
        </div>
        <div className="mt-3">
          <p className="mb-1.5 flex items-center gap-1 text-xs font-medium text-muted"><Sparkles size={12} /> Demo signs — labelled Demo in the transcript</p>
          <div className="flex flex-wrap gap-1.5">
            {SIGN_VOCABULARY.map(s => (
              <Button key={s} size="sm" variant="outline" onClick={() => onDemoSign(s)}>{s}</Button>
            ))}
          </div>
        </div>
      </div>
      <div className="border-t border-slate-800 pt-4">
        <div className="mb-2 flex items-center justify-between">
          <h3 className="flex items-center gap-1.5 text-sm font-semibold"><Mic size={15} className="text-cyan" /> Speech → captions</h3>
          <Badge tone={speechStatus === "listening" ? "green" : speechStatus === "ready" ? "cyan" : "red"}>
            {speechStatus === "listening" ? "Listening" : speechStatus === "ready" ? "Ready" : speechStatus === "unsupported" ? "Unsupported" : "Error"}
          </Badge>
        </div>
        <p className="text-xs text-muted">
          {captionEngine === "whisper"
            ? "Live captions from the server Whisper model (whisper-tiny) over the caption stream. The server sends final captions only (no interim results); each final caption carries a confidence score."
            : captionEngine === "browser"
              ? "Whisper is unavailable, so captions use the browser Web Speech API — labelled “Browser captions (not Whisper)”. Interim captions appear live; final captions are added to the transcript. Requires Chrome or Edge."
              : "Speech captions start when your microphone is live. The engine (Whisper or browser fallback) is shown here and on the caption bar."}
        </p>
        <p className="mt-1.5 text-xs text-muted">
          Caption relay:{" "}
          <span className={relayState === "connected" ? "text-success" : relayState === "reconnecting" ? "text-amber-300" : "text-red-300"}>
            {relayState === "connected"
              ? "connected — your captions are sent to the remote participant."
              : relayState === "reconnecting"
                ? "reconnecting — remote captions may be delayed."
                : "not connected — captions are shown only to you right now."}
          </span>
        </p>
      </div>
      <div className="border-t border-slate-800 pt-4">
        <div className="mb-2 flex items-center justify-between">
          <h3 className="flex items-center gap-1.5 text-sm font-semibold"><Volume2 size={15} className="text-violet-300" /> Sign → speech (voice agent)</h3>
          <Badge tone={voiceAgentOn === true ? "green" : voiceAgentOn === false ? "slate" : "amber"}>
            {voiceAgentOn === true ? "Voice agent on" : voiceAgentOn === false ? "Voice agent off" : "Unknown"}
          </Badge>
        </div>
        <p className="text-xs text-muted">
          {voiceAgentOn === true
            ? "Accepted signs are spoken by the room voice agent into the LiveKit call, so the remote participant hears them."
            : "The voice agent is off, so accepted signs are only previewed locally in this browser — the remote participant cannot hear them. Turn the voice agent on to speak into the call."}
          {ttsEngine && <span className="block mt-1">Server TTS engine: {ttsEngine}.</span>}
        </p>
        <div className="mt-2 flex gap-2">
          <Button size="sm" variant="outline" onClick={onToggleVoiceAgent}>
            <Megaphone size={14} /> {voiceAgentOn === true ? "Stop voice agent" : "Start voice agent"}
          </Button>
          <Button size="sm" variant="outline" onClick={() => setSayOpen(v => !v)}>Say this…</Button>
        </div>
        {voiceAgentOn === true && (
          <div className="mt-2.5 rounded-xl border border-slate-800 bg-ink/60 p-2.5">
            <label htmlFor="voice-agent-volume" className="mb-1.5 flex items-center justify-between text-xs font-medium text-slate-300">
              <span>Voice agent volume</span><span className="text-muted">{Math.round(voiceVolume * 100)}%</span>
            </label>
            <div className="flex items-center gap-2">
              <input
                id="voice-agent-volume"
                type="range" min={0} max={100} step={5}
                value={Math.round(voiceVolume * 100)}
                onChange={(e) => onVoiceVolumeChange(Number(e.target.value) / 100)}
                className="min-w-0 flex-1 accent-cyan"
                aria-label="Voice agent volume"
              />
              <Button size="sm" variant="outline" onClick={onToggleVoiceMute} aria-label={voiceMuted ? "Unmute voice agent" : "Mute voice agent"}>
                {voiceMuted ? <><VolumeX size={14} /> Muted</> : <><Volume2 size={14} /> Mute</>}
              </Button>
            </div>
            <p className="mt-1 text-[11px] text-muted">Controls the voice agent&apos;s audio in this browser only.</p>
          </div>
        )}
        {sayOpen && (
          <div className="mt-2.5 flex gap-1.5">
            <input
              className="min-w-0 flex-1 rounded-lg border border-slate-700 bg-ink px-2.5 py-1.5 text-sm"
              placeholder="Type text to speak into the call…"
              value={sayText}
              onChange={(e) => onSayTextChange(e.target.value)}
              onKeyDown={(e) => { if (e.key === "Enter") onSay(); }}
              aria-label="Text to speak"
            />
            <Button size="sm" onClick={onSay} disabled={speaking || !sayText.trim()}>
              {speaking ? <><Loader2 size={14} className="animate-spin" /> Speaking…</> : "Speak"}
            </Button>
          </div>
        )}
        {speaking && <p className="mt-2 text-xs text-cyan">Speaking…</p>}
        {speakError && (
          <p role="alert" className="mt-2 flex items-start gap-1.5 rounded-lg border border-amber-500/40 bg-amber-500/10 px-2.5 py-1.5 text-xs text-amber-200">
            <AlertTriangle size={13} className="mt-0.5 shrink-0" /> {speakError}
          </p>
        )}
        <p className="mt-2 text-xs text-muted">Local preview (browser speech) is {ttsMuted ? "muted" : "on"} — it only plays in this browser.</p>
        <Button size="sm" variant="outline" className="mt-2" onClick={onToggleTts}>
          {ttsMuted ? <><VolumeX size={14} /> Local preview muted — unmute</> : <><Volume2 size={14} /> Local preview on — mute</>}
        </Button>
      </div>
    </div>
  );
}
