"use client";
import { useEffect, useRef } from "react";
import { MicOff, VideoOff, User } from "lucide-react";
import { Badge } from "./ui/card";
import type { Track } from "livekit-client";

/**
 * A real video tile. It renders ONLY a live LiveKit track (or, for the
 * legacy local preview, a MediaStream). When there is no track, or the
 * camera is off, it shows an honest "Camera off" / "Waiting for video…"
 * state — never a fake video frame or a placeholder pretending to be video.
 *
 * - videoTrack: a livekit-client VideoTrack (local or remote). Attached
 *   with track.attach(videoEl) and detached on change/unmount.
 * - audioTrack: a livekit-client AudioTrack. Attached to a hidden <audio>
 *   element. Remote audio elements are NOT muted (that is how you hear
 *   the other participant); the local preview video IS muted (no feedback).
 * - stream: legacy MediaStream fallback (local preview). Prefer tracks.
 */
export function VideoTile({
  stream,
  videoTrack,
  audioTrack,
  name,
  label,
  muted = false,
  mirrored = false,
  micOn = true,
  camOn = true,
  active = false,
  speaking = false,
  /** Remote audio level 0–1 (voice-agent tile). Ignored for local tiles. */
  volume = 1,
  /** Mute the remote audio element (voice-agent tile). */
  audioMuted = false,
}: {
  stream?: MediaStream | null;
  videoTrack?: Track | null;
  audioTrack?: Track | null;
  name: string;
  label: string;
  muted?: boolean;
  mirrored?: boolean;
  micOn?: boolean;
  camOn?: boolean;
  active?: boolean;
  speaking?: boolean;
  volume?: number;
  audioMuted?: boolean;
}) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const audioRef = useRef<HTMLAudioElement>(null);

  // Attach the LiveKit video track (preferred) or fall back to a MediaStream.
  useEffect(() => {
    const el = videoRef.current;
    if (!el) return;
    if (videoTrack) {
      videoTrack.attach(el);
      return () => {
        videoTrack.detach(el);
      };
    }
    if (stream) {
      el.srcObject = stream;
      return () => {
        el.srcObject = null;
      };
    }
    return undefined;
  }, [videoTrack, stream]);

  // Attach the LiveKit audio track (remote participants). Never muted here:
  // muting is controlled by the tile's `muted` prop only for local preview,
  // and audio elements have no `muted` set at all.
  useEffect(() => {
    const el = audioRef.current;
    if (!el) return;
    if (audioTrack) {
      audioTrack.attach(el);
      return () => {
        audioTrack.detach(el);
      };
    }
    return undefined;
  }, [audioTrack]);

  // Remote audio level/mute (voice-agent tile): applied to the <audio>
  // element whenever the track or the controls change. Runs again when
  // audioTrack arrives, so the element exists by then.
  useEffect(() => {
    const el = audioRef.current;
    if (!el) return;
    el.volume = Math.min(1, Math.max(0, volume));
    el.muted = audioMuted;
  }, [volume, audioMuted, audioTrack]);

  const hasVideo = Boolean(videoTrack || stream) && camOn;
  return (
    <div
      className={`relative aspect-video w-full overflow-hidden rounded-2xl border bg-black ${
        speaking ? "border-success ring-2 ring-success/50" : active ? "border-cyan/60" : "border-slate-800"
      }`}
    >
      {/* The <video> element is always rendered so track.attach() survives
          camera on/off toggles (the element is never unmounted, only covered). */}
      <video
        ref={videoRef}
        autoPlay
        playsInline
        muted={muted}
        className={`h-full w-full object-cover ${mirrored ? "scale-x-[-1]" : ""} ${hasVideo ? "" : "invisible"}`}
      />
      {!hasVideo && (
        <div className="absolute inset-0 flex h-full w-full flex-col items-center justify-center gap-2 bg-panel text-muted">
          <User size={40} />
          <p className="text-sm">{camOn ? "Waiting for video…" : "Camera off"}</p>
        </div>
      )}
      {/* Remote audio: attached, unmuted, invisible. */}
      {audioTrack && <audio ref={audioRef} autoPlay playsInline />}
      <div className="absolute left-3 top-3 flex items-center gap-2">
        <span className="rounded-lg bg-black/60 px-2.5 py-1 text-xs font-medium backdrop-blur">{name}</span>
        <Badge tone={label.includes("Sign") ? "violet" : "cyan"}>{label}</Badge>
        {speaking && <Badge tone="green">Speaking</Badge>}
      </div>
      <div className="absolute bottom-3 right-3 flex gap-1.5">
        {!micOn && (
          <span className="rounded-full bg-red-500/90 p-1.5" title="Microphone muted">
            <MicOff size={13} />
          </span>
        )}
        {!camOn && (
          <span className="rounded-full bg-red-500/90 p-1.5" title="Camera off">
            <VideoOff size={13} />
          </span>
        )}
      </div>
    </div>
  );
}
