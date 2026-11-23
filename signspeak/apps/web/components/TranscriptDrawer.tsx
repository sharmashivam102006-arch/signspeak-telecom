"use client";

/**
 * Transcript.
 *
 * Ordered by when each utterance STARTED, not when it arrived. Sign events are
 * inherently later than speech events (the segmenter must wait for stillness
 * to know a sign ended), so arrival order produces a transcript where the
 * answer precedes the question. See apps/api/.../services/ordering.py.
 *
 * The download is plain text on purpose: it has to open on a hospital
 * reception desk, a bank terminal, and a ten-year-old laptop.
 */

import { useEffect, useRef } from "react";
import type { CaptionLine } from "@/app/call/[room]/page";

interface Props {
  open: boolean;
  onClose: () => void;
  lines: CaptionLine[];
  roomName: string;
}

export default function TranscriptDrawer({ open, onClose, lines, roomName }: Props) {
  const ref = useRef<HTMLDivElement>(null);

  // Escape closes, and focus moves into the drawer on open - basic dialog
  // behaviour that keyboard and screen-reader users depend on.
  useEffect(() => {
    if (!open) return;
    ref.current?.focus();
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && onClose();
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [open, onClose]);

  if (!open) return null;

  const ordered = [...lines]
    .filter((l) => l.isFinal && l.source !== "system")
    .sort((a, b) => a.startedAtMs - b.startedAtMs);

  const asText = ordered
    .map((l) => {
      const t = new Date(Date.now() - (performance.now() - l.startedAtMs));
      const stamp = t.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
      const conf = l.confidence !== undefined ? ` [${Math.round(l.confidence * 100)}%]` : "";
      const via = l.source === "sign" ? " (signed)" : "";
      return `${stamp} — ${l.displayName}${via}${conf}\n${l.text}\n`;
    })
    .join("\n");

  return (
    <div className="fixed inset-0 z-30 flex justify-end bg-black/50" onClick={onClose}>
      <div
        ref={ref}
        tabIndex={-1}
        role="dialog"
        aria-modal="true"
        aria-label="Call transcript"
        onClick={(e) => e.stopPropagation()}
        className="flex h-full w-full max-w-lg flex-col border-l border-border bg-bg"
      >
        <header className="flex items-center justify-between border-b border-border px-4 py-3">
          <h2 className="text-lg font-semibold">Transcript</h2>
          <button onClick={onClose} className="rounded border border-border px-3 py-1 text-sm">
            Close
          </button>
        </header>

        <div className="flex-1 overflow-y-auto p-4">
          {ordered.length === 0 && <p className="text-muted">Nothing yet.</p>}
          <ol className="space-y-4">
            {ordered.map((l) => (
              <li key={l.id}>
                <div className="mb-1 flex items-center gap-2 text-xs uppercase tracking-wide text-muted">
                  <span className="font-semibold">{l.displayName}</span>
                  {l.source === "sign" && <span>· signed</span>}
                  {l.confidence !== undefined && (
                    <span className="tabular-nums">· {Math.round(l.confidence * 100)}%</span>
                  )}
                </div>
                <p className="text-base">{l.text}</p>
              </li>
            ))}
          </ol>
        </div>

        <footer className="flex items-center justify-between gap-2 border-t border-border px-4 py-3">
          <p className="text-xs text-muted">
            Stored only if everyone in the call agreed. Either person can delete it.
          </p>
          <a
            download={`signspeak-${roomName}.txt`}
            href={`data:text/plain;charset=utf-8,${encodeURIComponent(asText)}`}
            className="rounded border border-border px-3 py-1 text-sm hover:border-accent"
          >
            Download
          </a>
        </footer>
      </div>
    </div>
  );
}
