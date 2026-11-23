"use client";

/**
 * Video area.
 *
 * The self-view is deliberately large for the signer. In a normal video call
 * the self-view is a thumbnail, because you already know what you look like.
 * Here the signer needs to see whether their hands are inside the frame the
 * recogniser is reading - so the self-view is a working instrument, not vanity.
 * The framing guide draws that region explicitly.
 */

import { useEffect, useRef, useState, type RefObject } from "react";
import type { RoomClient } from "@/lib/room";

interface Props {
  videoRef: RefObject<HTMLVideoElement>;
  room: RoomClient | null;
  camOn: boolean;
  signing: boolean;
  tracked: boolean;
  isSigner: boolean;
}

export default function CallStage({ videoRef, camOn, signing, tracked, isSigner }: Props) {
  const remoteRef = useRef<HTMLVideoElement>(null);
  const [hasRemote, setHasRemote] = useState(false);

  useEffect(() => {
    const el = remoteRef.current;
    if (!el) return;
    const onLoaded = () => setHasRemote(true);
    el.addEventListener("loadeddata", onLoaded);
    return () => el.removeEventListener("loadeddata", onLoaded);
  }, []);

  return (
    <div className={`grid gap-3 ${isSigner ? "lg:grid-cols-[3fr_2fr]" : "lg:grid-cols-[2fr_3fr]"}`}>
      <figure className="relative overflow-hidden rounded-xl border border-border bg-black">
        <video
          ref={videoRef}
          autoPlay
          playsInline
          muted
          className="aspect-video w-full -scale-x-100 object-cover"
        />
        {isSigner && camOn && <FramingGuide tracked={tracked} signing={signing} />}
        {!camOn && (
          <div className="absolute inset-0 grid place-items-center bg-surface text-center text-sm text-muted">
            <div>
              <p className="mb-1 font-semibold text-fg">Camera off</p>
              {isSigner && <p>Sign recognition is paused while the camera is off.</p>}
            </div>
          </div>
        )}
        <figcaption className="absolute bottom-2 left-2 rounded bg-black/70 px-2 py-1 text-xs text-white">
          You {isSigner ? "(signing)" : "(speaking)"}
        </figcaption>
      </figure>

      <figure className="relative overflow-hidden rounded-xl border border-border bg-black">
        <video ref={remoteRef} autoPlay playsInline className="aspect-video w-full object-cover" />
        {!hasRemote && (
          <div className="absolute inset-0 grid place-items-center text-sm text-muted">
            Waiting for the other person to join…
          </div>
        )}
      </figure>
    </div>
  );
}

/**
 * The region the recogniser actually reads.
 *
 * Without this, users discover the frame boundary by being misunderstood,
 * which is the worst possible feedback channel. Showing it turns an invisible
 * constraint into something a person can simply step into.
 */
function FramingGuide({ tracked, signing }: { tracked: boolean; signing: boolean }) {
  const color = !tracked ? "rgb(var(--low))" : signing ? "rgb(var(--accent))" : "rgb(var(--high))";
  return (
    <svg className="pointer-events-none absolute inset-0 h-full w-full" aria-hidden>
      <rect
        x="12%" y="6%" width="76%" height="88%"
        fill="none" stroke={color} strokeWidth="2" strokeDasharray="10 8" rx="14" opacity="0.65"
      />
    </svg>
  );
}
