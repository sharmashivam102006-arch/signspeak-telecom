"use client";

/**
 * Live captions.
 *
 * Design rules, all of them load-bearing for the user who cannot hear:
 *
 *  - Captions render the instant they arrive, NOT through the transcript's
 *    reorder buffer. Being briefly out of order on screen is far less
 *    confusing than being a second behind the conversation. The stored
 *    transcript uses the buffer; the live rail does not. Same events, two
 *    consumers, two correct-but-different policies.
 *
 *  - Speaker identity is never carried by colour alone. A name label and an
 *    icon do the work; colour is reinforcement.
 *
 *  - Sign-derived lines carry a visible confidence marker. The hearing
 *    participant hears a fluent synthetic voice with no way to judge it; this
 *    is the only signal they get that the machine was unsure.
 *
 *  - aria-live="polite" on the container, not on each line, so a screen reader
 *    announces new text without interrupting itself. "assertive" here would
 *    make the page unusable with a screen reader - every caption would cut off
 *    the last one.
 */

import { useEffect, useRef } from "react";
import type { CaptionLine } from "@/app/call/[room]/page";

interface Props {
  lines: CaptionLine[];
  /** From the accessibility menu; persisted per user. */
  size?: "small" | "medium" | "large" | "xlarge";
}

const SIZE_CLASS = {
  small: "text-caption-sm",
  medium: "text-caption-md",
  large: "text-caption-lg",
  xlarge: "text-caption-xl",
} as const;

export default function CaptionRail({ lines, size = "medium" }: Props) {
  const endRef = useRef<HTMLDivElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const pinnedRef = useRef(true);

  // Auto-scroll, but only while the user is already at the bottom. Yanking
  // someone back down while they are reading an earlier line is the single
  // most irritating caption bug there is.
  useEffect(() => {
    const el = containerRef.current;
    if (!el) return;
    const onScroll = () => {
      pinnedRef.current = el.scrollHeight - el.scrollTop - el.clientHeight < 48;
    };
    el.addEventListener("scroll", onScroll, { passive: true });
    return () => el.removeEventListener("scroll", onScroll);
  }, []);

  useEffect(() => {
    if (pinnedRef.current) endRef.current?.scrollIntoView({ block: "end" });
  }, [lines]);

  return (
    <div
      ref={containerRef}
      className="caption-rail h-full overflow-y-auto rounded-xl border border-border bg-surface p-4"
      role="log"
      aria-live="polite"
      aria-relevant="additions text"
      aria-label="Live captions"
    >
      {lines.length === 0 && (
        <p className="text-muted">Captions appear here as soon as anyone speaks or signs.</p>
      )}

      <ul className="space-y-3">
        {lines.map((line) => (
          <li key={line.id} className="flex flex-col gap-1">
            <div className="flex items-center gap-2 text-xs uppercase tracking-wide text-muted">
              <SourceIcon source={line.source} />
              <span className="font-semibold">{line.displayName}</span>
              {line.source === "sign" && <span className="text-muted">· signed</span>}
              {!line.isFinal && <span className="italic opacity-70">· still speaking</span>}
              {line.confidence !== undefined && (
                <ConfidenceChip value={line.confidence} />
              )}
            </div>
            <p
              className={`${SIZE_CLASS[size]} ${
                line.isFinal ? "" : "opacity-70"
              } ${line.source === "system" ? "italic text-muted" : ""}`}
            >
              {line.text}
            </p>
          </li>
        ))}
      </ul>
      <div ref={endRef} />
    </div>
  );
}

function SourceIcon({ source }: { source: CaptionLine["source"] }) {
  const map = { sign: "✋", speech: "🔊", text: "⌨", system: "ⓘ" } as const;
  return (
    <span aria-hidden className="text-sm">
      {map[source]}
    </span>
  );
}

/**
 * Confidence, visible to the person who did NOT sign it.
 *
 * Tempting to hide this once confidence is high - resist it. A hearing
 * participant who only ever sees a marker when something is wrong learns to
 * treat its absence as certainty, which is exactly the false impression the
 * calibration work exists to prevent.
 */
function ConfidenceChip({ value }: { value: number }) {
  const pct = Math.round(value * 100);
  const tone = value >= 0.9 ? "high" : value >= 0.7 ? "medium" : "low";
  const label = tone === "high" ? "high" : tone === "medium" ? "moderate" : "low";
  const cls =
    tone === "high"
      ? "border-high/50 text-high"
      : tone === "medium"
        ? "border-medium/50 text-medium"
        : "border-low/50 text-low";
  return (
    <span
      className={`rounded border px-1.5 py-0.5 text-[0.65rem] tabular-nums ${cls}`}
      title={`Recognition confidence: ${pct}% (${label})`}
    >
      {pct}% {label}
    </span>
  );
}
