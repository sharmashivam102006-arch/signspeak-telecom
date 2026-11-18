// A recognised sign awaiting confirmation, with the four actions from the
// 85%/60% thresholds: Accept, Correct (pick the right sign from the
// vocabulary — the corrected sequence is then stored for training),
// Repeat sign, Ignore.
// `mode` is the backend's own label: "trained-model" for a real model
// prediction, "demo" for a labelled demo result. Never shown as real AI
// when it is a demo.
"use client";
import { useState } from "react";
import { Check, Pencil, RotateCcw, X } from "lucide-react";
import { SIGN_VOCABULARY } from "@/lib/store";
import { Button } from "./ui/button";
import { Badge } from "./ui/card";

export function ConfidenceCard({ sign, confidence, demo, mode, vocabulary, onAccept, onCorrect, onRepeat, onIgnore }: {
  sign: string; confidence: number; demo?: boolean;
  mode?: "trained-model" | "demo";
  vocabulary?: string[];
  onAccept: () => void; onCorrect: (text: string) => void; onRepeat: () => void; onIgnore: () => void;
}) {
  const [correcting, setCorrecting] = useState(false);
  const [picked, setPicked] = useState(sign);
  const pct = Math.round(confidence * 100);
  const level = pct >= 85 ? "high" : pct >= 60 ? "medium" : "low";
  const options = vocabulary && vocabulary.length > 0 ? vocabulary : [...SIGN_VOCABULARY];
  return (
    <div className={`rounded-xl border p-3 ${level === "high" ? "border-success/40 bg-success/5" : level === "medium" ? "border-amber-500/40 bg-amber-500/5" : "border-red-500/40 bg-red-500/5"}`}>
      <div className="flex items-center justify-between gap-2">
        <p className="text-sm font-semibold">Recognised: “{sign}”</p>
        <span className="flex gap-1.5">
          <Badge tone={mode === "trained-model" ? "green" : "amber"}>{mode === "trained-model" ? "Trained model" : "Demo"}</Badge>
          <Badge tone={level === "high" ? "green" : level === "medium" ? "amber" : "red"}>{pct}% confidence</Badge>
        </span>
      </div>
      <div className="mt-2 h-2 overflow-hidden rounded-full bg-slate-800">
        <div className={`h-full ${level === "high" ? "bg-success" : level === "medium" ? "bg-amber-400" : "bg-red-400"}`} style={{ width: `${pct}%` }} />
      </div>
      <p className="mt-2 text-xs text-muted">
        {level === "high" ? "High confidence — will be spoken and added to the transcript."
          : level === "medium" ? "Uncertain — please confirm before it is spoken."
          : "Low confidence — please repeat the sign or correct it manually."}
        {demo ? " (Demo prediction — not a real model result.)" : ""}
      </p>
      {!correcting ? (
        <div className="mt-3 flex flex-wrap gap-1.5">
          <Button size="sm" onClick={onAccept}><Check size={14} /> Accept</Button>
          <Button size="sm" variant="outline" onClick={() => { setPicked(sign); setCorrecting(true); }}><Pencil size={14} /> Correct</Button>
          <Button size="sm" variant="outline" onClick={onRepeat}><RotateCcw size={14} /> Repeat sign</Button>
          <Button size="sm" variant="ghost" onClick={onIgnore}><X size={14} /> Ignore</Button>
        </div>
      ) : (
        <div className="mt-3 rounded-lg border border-slate-800 bg-ink/60 p-2.5">
          <p className="mb-1.5 text-xs font-medium text-slate-300">What did you actually sign?</p>
          <div className="flex flex-wrap gap-1.5">
            <select
              className="rounded-lg border border-slate-700 bg-ink px-2 py-1.5 text-sm"
              value={options.includes(picked) ? picked : options[0]}
              onChange={(e) => setPicked(e.target.value)}
              aria-label="Corrected sign"
            >
              {options.map((o) => <option key={o} value={o}>{o}</option>)}
            </select>
            <Button size="sm" onClick={() => { setCorrecting(false); if (picked.trim()) onCorrect(picked.trim()); }}>
              <Check size={14} /> Save correction
            </Button>
            <Button size="sm" variant="ghost" onClick={() => setCorrecting(false)}>Cancel</Button>
          </div>
          <p className="mt-1.5 text-[11px] text-muted">The corrected sequence is stored as a training sample for that sign.</p>
        </div>
      )}
    </div>
  );
}
