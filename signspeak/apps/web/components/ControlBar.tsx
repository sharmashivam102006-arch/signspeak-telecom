"use client";

/**
 * Call controls.
 *
 * Every control has a visible text label, not just an icon. Icon-only toolbars
 * assume a shared visual vocabulary, and "the crossed-out microphone means
 * muted" is learned, not obvious - a bad assumption in a product whose whole
 * premise is that people communicate differently.
 *
 * "Leave" is separated and coloured differently because it is the one
 * irreversible action on the bar.
 */

interface Props {
  micOn: boolean;
  camOn: boolean;
  onToggleMic: () => void;
  onToggleCam: () => void;
  onOpenTranscript: () => void;
  onLeave: () => void;
  onUnlockAudio: () => void;
}

export default function ControlBar({
  micOn, camOn, onToggleMic, onToggleCam, onOpenTranscript, onLeave, onUnlockAudio,
}: Props) {
  return (
    <footer className="flex flex-wrap items-center justify-center gap-2 border-t border-border bg-surface px-4 py-3">
      <Control
        label={micOn ? "Mute" : "Unmute"}
        icon={micOn ? "🎙" : "🔇"}
        pressed={micOn}
        onClick={() => { onUnlockAudio(); onToggleMic(); }}
      />
      <Control
        label={camOn ? "Camera off" : "Camera on"}
        icon={camOn ? "📷" : "🚫"}
        pressed={camOn}
        onClick={onToggleCam}
        hint={camOn ? "Turning the camera off also pauses sign recognition" : undefined}
      />
      <Control label="Transcript" icon="📝" onClick={onOpenTranscript} />
      <button
        onClick={onLeave}
        className="ml-4 rounded-lg bg-low px-4 py-2 text-sm font-semibold text-black hover:opacity-90"
      >
        Leave call
      </button>
    </footer>
  );
}

function Control({
  label, icon, onClick, pressed, hint,
}: {
  label: string; icon: string; onClick: () => void; pressed?: boolean; hint?: string;
}) {
  return (
    <button
      onClick={onClick}
      aria-pressed={pressed}
      title={hint ?? label}
      className={`flex items-center gap-2 rounded-lg border px-3 py-2 text-sm font-medium transition
        ${pressed === false ? "border-low/60 text-low" : "border-border text-fg hover:border-accent"}`}
    >
      <span aria-hidden>{icon}</span>
      {label}
    </button>
  );
}
