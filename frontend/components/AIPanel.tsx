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
//
// Until a trained model exists, the sign section first asks whether the
// user wants to set up sign recognition (record + train). Training,
// detection and demo controls appear only after they say yes; "Not now"
// collapses the section to one line and turns sign detection off. The
// choice is remembered in this browser (localStorage).
"use client";
import { useEffect, useState } from "react";
import { Hand, Mic, Volume2, VolumeX, Sparkles, GraduationCap, Megaphone, Loader2, AlertTriangle } from "lucide-react";
import { SIGN_VOCABULARY } from "@/lib/store";
import { Badge } from "./ui/card";
import { Button } from "./ui/button";

const SIGN_SETUP_KEY = "signspeak.signSetup";
type SignSetupChoice = "yes" | "no" | null;

function readSignSetupChoice(): SignSetupChoice {
  try {
    const v = window.localStorage.getItem(SIGN_SETUP_KEY);
    return v === "yes" || v === "no" ? v : null;
  } catch {
    return null;
  }
}

function saveSignSetupChoice(v: Exclude<SignSetupChoice, null>): void {
  try { window.localStorage.setItem(SIGN_SETUP_KEY, v); } catch { /* private mode: not remembered */ }
}

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
  // Read after mount so server and client render the same first frame.
  const [setupChoice, setSetupChoice] = useState<SignSetupChoice>(null);
  const [choiceLoaded, setChoiceLoaded] = useState(false);
  useEffect(() => {
    setSetupChoice(readSignSetupChoice());
    setChoiceLoaded(true);
  }, []);
  const modelReady = signStatus === "real" || signTrained === true;
  const showSignTools = modelReady || setupChoice === "yes";

  const chooseSetup = (v: "yes" | "no") => {
    setSetupChoice(v);
    saveSignSetupChoice(v);
    // Detection does nothing useful without a model: turn it off on "Not now".
    if (v === "no" && signEnabled) onToggleSign();
  };

  return (
    <div className="space-y-4">
      <div>
        <div className="mb-2 flex items-center justify-between">
          <h3 className="flex items-center gap-1.5 text-sm font-semibold"><Hand size={15} className="text-violet-300" /> Sign recognition</h3>
          {showSignTools ? (
            <Badge tone={signStatus === "real" ? "green" : signStatus === "demo" ? "amber" : "red"}>
              {signStatus === "real" ? "Trained model" : signStatus === "demo" ? "Demo mode" : "Unavailable"}
            </Badge>
          ) : (
            <Badge tone="slate">{setupChoice === "no" ? "Off" : "Not set up"}</Badge>
          )}
        </div>
        {!showSignTools && choiceLoaded && setupChoice === null && (
          <div className="rounded-xl border border-violet-500/40 bg-violet-500/5 p-3">
            <p className="text-sm font-semibold text-violet-200">Do you want to use sign recognition?</p>
            <p className="mt-1 text-xs text-muted">
              It needs a short training session first: you record a few samples of each sign with your camera,
              then train the model. You can skip this and still use captions and the voice agent.
            </p>
            <div className="mt-2.5 flex gap-2">
              <Button size="sm" className="flex-1" onClick={() => chooseSetup("yes")}>
                <GraduationCap size={14} /> Yes, set it up
              </Button>
              <Button size="sm" variant="outline" className="flex-1" onClick={() => chooseSetup("no")}>Not now</Button>
            </div>
          </div>
        )}
        {!showSignTools && setupChoice === "no" && (
          <p className="text-xs text-muted">
            Sign recognition is off.{" "}
            <button type="button" className="text-cyan underline-offset-2 hover:underline" onClick={() => chooseSetup("yes")}>
              Set up sign recognition
            </button>
          </p>
        )}
        {showSignTools && (<>
        <p className="text-xs text-muted">
          {signStatus === "real"
            ? "Sign in front of your camera — the trained model recognises your signs."
            : signStatus === "demo"
              ? "No trained model yet. Live predictions use the labelled demo path."
              : "Live recognition is not available yet. The demo buttons still work."}
        </p>
        {signTrained === false && (
          <div className="mt-3 rounded-xl border border-amber-500/40 bg-amber-500/5 p-3">
            <p className="text-sm font-semibold text-amber-200">Train your signs</p>
            <p className="mt-1 text-xs text-muted">
              Record about 10 samples of each sign, then train. Recognition starts when training finishes.
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
          <p className="mb-1.5 flex items-center gap-1 text-xs font-medium text-muted"><Sparkles size={12} /> Try a demo sign</p>
          <div className="flex flex-wrap gap-1.5">
            {SIGN_VOCABULARY.map(s => (
              <Button key={s} size="sm" variant="outline" onClick={() => onDemoSign(s)}>{s}</Button>
            ))}
          </div>
        </div>
        {!modelReady && (
          <button type="button" className="mt-2 text-xs text-muted underline-offset-2 hover:text-slate-300 hover:underline" onClick={() => chooseSetup("no")}>
            Hide sign recognition
          </button>
        )}
        </>)}
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
            ? "Captioned by Whisper on the server, one sentence at a time."
            : captionEngine === "browser"
              ? "Captioned by your browser (Chrome or Edge), shown as you speak."
              : "Captions start when your microphone is on."}
        </p>
        <p className="mt-1.5 flex items-center gap-1.5 text-xs text-muted">
          <span className={`h-1.5 w-1.5 rounded-full ${relayState === "connected" ? "bg-success" : relayState === "reconnecting" ? "bg-amber-400" : "bg-red-400"}`} aria-hidden />
          {relayState === "connected"
            ? "Shared with the other participant"
            : relayState === "reconnecting"
              ? "Reconnecting — the other participant may see captions late"
              : "Only shown to you right now"}
        </p>
      </div>
      <div className="border-t border-slate-800 pt-4">
        <div className="mb-2 flex items-center justify-between">
          <h3 className="flex items-center gap-1.5 text-sm font-semibold"><Volume2 size={15} className="text-violet-300" /> Sign → speech (voice agent)</h3>
          <Badge tone={voiceAgentOn === true ? "green" : voiceAgentOn === false ? "slate" : "amber"}>
            {voiceAgentOn === true ? "Voice agent on" : voiceAgentOn === false ? "Voice agent off" : "Unknown"}
          </Badge>
        </div>
        <p className="text-xs text-muted" title={ttsEngine ? `Server voice: ${ttsEngine}` : undefined}>
          {voiceAgentOn === true
            ? "Accepted signs are spoken into the call for the other participant."
            : "Start the voice agent so the other participant hears your accepted signs."}
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
        <button type="button" onClick={onToggleTts} className="mt-2.5 flex items-center gap-1.5 text-xs text-muted hover:text-slate-200">
          {ttsMuted ? <VolumeX size={13} /> : <Volume2 size={13} />}
          Preview in this browser: {ttsMuted ? "muted" : "on"}
        </button>
      </div>
    </div>
  );
}
