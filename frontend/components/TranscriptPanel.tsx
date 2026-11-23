"use client";
import { useState } from "react";
import { Copy, Download, Trash2, Check, CloudUpload, AlertTriangle } from "lucide-react";
import { TranscriptEntry, transcriptToText } from "@/lib/store";
import { Badge } from "./ui/card";
import { Button } from "./ui/button";

/** A transcript entry as displayed: the stored entry plus the speaker
 *  name when known (remote captions carry the speaker's name; local
 *  entries use "You"). The speaker is kept alongside the entry — not
 *  inside the shared store type — so the store schema stays untouched. */
export type TranscriptDisplayEntry = TranscriptEntry & { speaker?: string };

export function TranscriptPanel({ entries, onClear, roomId, isCreator, onSaveToServer }: {
  entries: TranscriptDisplayEntry[];
  onClear: () => void;
  roomId: string;
  /** Only the room creator may save to the server (backend restriction). */
  isCreator: boolean;
  /** Saves the current entries server-side (consent=true). Resolves
   *  with how many were stored; rejects with the backend's error. */
  onSaveToServer: (() => Promise<{ saved: number }>) | null;
}) {
  const [copied, setCopied] = useState(false);
  const [saving, setSaving] = useState(false);
  const [saveNote, setSaveNote] = useState("");
  const [saveError, setSaveError] = useState("");
  const copy = async () => {
    await navigator.clipboard.writeText(transcriptToText(entries));
    setCopied(true); setTimeout(() => setCopied(false), 1500);
  };
  const download = () => {
    const blob = new Blob([transcriptToText(entries)], { type: "text/plain" });
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob); a.download = `signspeak-transcript-${roomId}.txt`; a.click();
    URL.revokeObjectURL(a.href);
  };
  const saveToServer = async () => {
    if (!onSaveToServer || saving) return;
    setSaving(true); setSaveNote(""); setSaveError("");
    try {
      const r = await onSaveToServer();
      setSaveNote(r.saved === 0 ? "Nothing new to save." : `Saved ${r.saved} entr${r.saved === 1 ? "y" : "ies"} to the server.`);
    } catch (e) {
      // Surface the backend's reason verbatim (e.g. creator-only) —
      // never claim the save worked when it did not.
      setSaveError(e instanceof Error ? e.message : String(e));
    } finally {
      setSaving(false);
    }
  };
  return (
    <div className="flex h-full flex-col">
      <div className="mb-2 flex items-center justify-between">
        <h3 className="text-sm font-semibold">Conversation transcript</h3>
        <div className="flex gap-1">
          <Button variant="ghost" size="sm" onClick={copy} title="Copy transcript">{copied ? <Check size={14} className="text-success" /> : <Copy size={14} />}</Button>
          <Button variant="ghost" size="sm" onClick={download} title="Export as .txt"><Download size={14} /></Button>
          <Button variant="ghost" size="sm" onClick={onClear} title="Clear transcript"><Trash2 size={14} /></Button>
        </div>
      </div>
      <div className="scrollbar-thin flex-1 space-y-2 overflow-y-auto rounded-xl border border-slate-800 bg-ink/60 p-3">
        {entries.length === 0 && <p className="text-sm text-muted">No messages yet. Recognised signs and speech captions will appear here.</p>}
        {entries.map(e => (
          <div key={e.id} className={`rounded-lg border-l-2 px-3 py-2 text-sm ${e.direction === "sign-to-speech" ? "border-violet bg-violet/5" : "border-cyan bg-cyan/5"}`}>
            <div className="mb-0.5 flex flex-wrap items-center gap-2 text-[11px] text-muted">
              <Badge tone={e.direction === "sign-to-speech" ? "violet" : "cyan"}>{e.direction === "sign-to-speech" ? "Sign → Speech" : "Speech → Text"}</Badge>
              {e.demo && <Badge tone="amber">Demo</Badge>}
              {e.speaker && <span className="font-medium text-slate-300">{e.speaker}</span>}
              <span>{new Date(e.timestamp).toLocaleTimeString()}</span>
              {e.confidence != null && <span>· {Math.round(e.confidence * 100)}% confidence</span>}
            </div>
            <p>{e.text}</p>
          </div>
        ))}
      </div>
      <div className="mt-2 flex flex-wrap items-center gap-2">
        {isCreator && onSaveToServer ? (
          <Button variant="outline" size="sm" onClick={saveToServer} disabled={saving || entries.length === 0}>
            <CloudUpload size={14} /> {saving ? "Saving…" : "Save to server"}
          </Button>
        ) : (
          <p className="text-[11px] text-muted">Server saving is available to the room creator.</p>
        )}
        {saveNote && <p className="text-xs text-success">{saveNote}</p>}
        {saveError && (
          <p role="alert" className="flex items-start gap-1.5 text-xs text-amber-200">
            <AlertTriangle size={13} className="mt-0.5 shrink-0" /> {saveError}
          </p>
        )}
      </div>
    </div>
  );
}
