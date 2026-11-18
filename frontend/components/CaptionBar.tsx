// Live caption line. The engine label is honest: "Whisper tiny" only
// when the caption really came from the server Whisper stream;
// browser Web Speech results are labelled "Browser captions (not Whisper)".
// Interim (partial) text is grey + italic; final text shows its
// confidence when Whisper reported one.
"use client";
import { Captions } from "lucide-react";
export function CaptionBar({ text, interim, size = "medium", engine, confidence }: {
  text: string; interim?: string; size?: "small" | "medium" | "large";
  engine?: "whisper" | "browser" | null;
  confidence?: number | null;
}) {
  const cls = size === "large" ? "text-xl" : size === "small" ? "text-sm" : "text-base";
  if (!text && !interim) return null;
  return (
    <div className="caption-in mx-auto max-w-3xl rounded-xl bg-black/75 px-4 py-2.5 text-center backdrop-blur border border-slate-700/60">
      <div className="mb-0.5 flex items-center justify-center gap-1.5 text-[11px] uppercase tracking-wider text-cyan">
        <Captions size={12} /> Live captions · {engine === "whisper" ? "Whisper tiny" : engine === "browser" ? "Browser captions (not Whisper)" : "Speech → Text"}
        {engine === "whisper" && confidence != null && <span className="normal-case tracking-normal text-muted">· {Math.round(confidence * 100)}% confidence</span>}
      </div>
      <p className={`font-medium leading-snug ${cls}`}>{text}{interim ? <span className="text-muted italic"> {interim}</span> : ""}</p>
    </div>
  );
}
