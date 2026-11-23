// "Train signs" panel: record 2-second landmark sequences per sign,
// store them with POST /api/ai/sign/collect, then train the PyTorch
// SignLSTM with POST /api/ai/sign/train and poll the job.
// Every backend error is shown verbatim — never replaced with a fake success.
// Recording needs the live hand tracker: the parent passes refs that the
// room page's tracker loop keeps current (latest landmarks + hand-visible).
"use client";
import { useCallback, useEffect, useRef, useState } from "react";
import type { RefObject } from "react";
import { ArrowLeft, Circle, Check, AlertTriangle, Loader2, GraduationCap, Hand } from "lucide-react";
import { ApiError } from "@/lib/api";
import { collectSignSample, fetchSignSamples, fetchSignStatus, fetchTrainJob, startSignTraining, type TrainJob } from "@/lib/ai";
import { SIGN_VOCABULARY } from "@/lib/store";
import type { Landmark } from "@/lib/sign";
import { Badge } from "./ui/card";
import { Button } from "./ui/button";

const TARGET_SAMPLES = 10;
const MIN_TO_TRAIN = 5;
const RECORD_MS = 2000;
const MIN_VISIBLE_FRACTION = 0.7;

function errText(e: unknown): string {
  return e instanceof Error ? e.message : String(e);
}

export function SignTrainer({ latestLandmarksRef, handVisibleRef, onTrainingDone, onClose }: {
  latestLandmarksRef: RefObject<Landmark[] | null>;
  handVisibleRef: RefObject<boolean>;
  onTrainingDone: () => void;
  onClose: () => void;
}) {
  const [counts, setCounts] = useState<Record<string, number>>({});
  const [totalSigners, setTotalSigners] = useState<number | null>(null);
  const [loadError, setLoadError] = useState("");
  const [loading, setLoading] = useState(true);
  const [recording, setRecording] = useState<string | null>(null);
  const [recordProgress, setRecordProgress] = useState(0);
  const [recordNote, setRecordNote] = useState("");
  const [recordError, setRecordError] = useState("");
  const [job, setJob] = useState<TrainJob | null>(null);
  const [trainError, setTrainError] = useState("");
  const [training, setTraining] = useState(false);
  const [savedNote, setSavedNote] = useState("");
  const pollRef = useRef<number | null>(null);

  const refreshCounts = useCallback(async () => {
    try {
      const d = await fetchSignSamples();
      setCounts(d.samples_per_sign);
      // Per-signer counts come from the status endpoint (the backend
      // reports samples_per_signer / total_signers there). The shared
      // SignModelStatus type predates those fields, so read them via a
      // cast; a missing field just leaves the contributor line hidden.
      const st = await fetchSignStatus() as (Record<string, unknown> | null);
      if (st && typeof st.total_signers === "number") setTotalSigners(st.total_signers);
      setLoadError("");
    } catch (e) {
      setLoadError(
        e instanceof ApiError && e.status === 404
          ? "Sample collection is not available on this backend yet (GET /api/ai/sign/samples returned 404)."
          : `Could not load sample counts: ${errText(e)}`,
      );
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => { void refreshCounts(); }, [refreshCounts]);
  useEffect(() => () => { if (pollRef.current) window.clearInterval(pollRef.current); }, []);

  const allReady = SIGN_VOCABULARY.every((s) => (counts[s] ?? 0) >= MIN_TO_TRAIN);

  const record = (label: string) => {
    if (recording) return;
    setRecording(label);
    setRecordError("");
    setRecordNote("");
    setSavedNote("");
    setRecordProgress(0);
    const frames: Landmark[][] = [];
    let visibleFrames = 0;
    let ticks = 0;
    const totalTicks = RECORD_MS / 100;
    const iv = window.setInterval(() => {
      ticks += 1;
      setRecordProgress(Math.min(100, Math.round((ticks / totalTicks) * 100)));
      const lm = latestLandmarksRef.current;
      if (handVisibleRef.current && lm && lm.length >= 21) {
        visibleFrames += 1;
        frames.push(lm.map((p) => ({ ...p })));
      }
      if (ticks >= totalTicks) {
        window.clearInterval(iv);
        void finishRecording(label, frames, visibleFrames, ticks);
      }
    }, 100);
  };

  const finishRecording = async (label: string, frames: Landmark[][], visibleFrames: number, ticks: number) => {
    const visibleFraction = ticks > 0 ? visibleFrames / ticks : 0;
    if (visibleFraction < MIN_VISIBLE_FRACTION || frames.length < 5) {
      setRecording(null);
      setRecordProgress(0);
      setRecordError(
        `Sample discarded — your hand was visible for only ${Math.round(visibleFraction * 100)}% of the 2 seconds (need ≥${Math.round(MIN_VISIBLE_FRACTION * 100)}%). Hold your hand in view of the camera and try again.`,
      );
      return;
    }
    const sequence = frames.slice(0, 30).map((f) => f.slice(0, 21).map((p) => [p.x, p.y, p.z]));
    try {
      const r = await collectSignSample(label, sequence);
      setCounts((prev) => ({ ...prev, [label]: r.count }));
      setSavedNote(`Saved “${label}” sample ${r.count}/${TARGET_SAMPLES}.`);
      setRecordNote("");
    } catch (e) {
      setRecordError(
        e instanceof ApiError && e.status === 404
          ? "Sample collection is not available on this backend yet (POST /api/ai/sign/collect returned 404)."
          : `Could not save the sample: ${errText(e)}`,
      );
    } finally {
      setRecording(null);
      setRecordProgress(0);
    }
  };

  const train = async () => {
    setTrainError("");
    setJob(null);
    setTraining(true);
    try {
      const started = await startSignTraining();
      setJob(started);
      pollRef.current = window.setInterval(async () => {
        try {
          const j = await fetchTrainJob(started.job_id);
          setJob(j);
          if (j.status === "done" || j.status === "failed") {
            if (pollRef.current) window.clearInterval(pollRef.current);
            pollRef.current = null;
            setTraining(false);
            if (j.status === "done") {
              onTrainingDone();
              void refreshCounts();
            }
          }
        } catch (e) {
          if (pollRef.current) window.clearInterval(pollRef.current);
          pollRef.current = null;
          setTraining(false);
          setTrainError(`Could not poll the training job: ${errText(e)}`);
        }
      }, 2000);
    } catch (e) {
      setTraining(false);
      setTrainError(
        e instanceof ApiError && e.status === 409
          ? `Training could not start: ${errText(e)}`
          : e instanceof ApiError && e.status === 404
            ? "Training is not available on this backend yet (POST /api/ai/sign/train returned 404)."
            : `Training could not start: ${errText(e)}`,
      );
    }
  };

  const trainingDone = job?.status === "done";
  const trainingFailed = job?.status === "failed";

  return (
    <div className="space-y-3">
      <div className="flex items-center justify-between">
        <h3 className="flex items-center gap-1.5 text-sm font-semibold"><GraduationCap size={15} className="text-violet-300" /> Train signs</h3>
        <Button variant="ghost" size="sm" onClick={onClose}><ArrowLeft size={14} /> Back</Button>
      </div>
      <p className="text-xs text-muted">
        Record 2-second samples of each sign in front of your camera. Target {TARGET_SAMPLES} samples per sign
        (training unlocks at {MIN_TO_TRAIN} per sign). The model trains on the server — no weights exist until
        training finishes.
      </p>
      {totalSigners !== null && (
        <p className="text-xs text-muted">
          Contributors so far: {totalSigners} signer{totalSigners === 1 ? "" : "s"}.
          {totalSigners < 2
            ? " Ask a second person to record the same signs — validation holds out a whole signer, so one signer alone cannot prove the model generalises."
            : " Validation will hold out one whole signer."}
        </p>
      )}
      {loading && <p className="flex items-center gap-2 text-xs text-muted"><Loader2 size={13} className="animate-spin" /> Loading sample counts…</p>}
      {loadError && (
        <p role="alert" className="flex items-start gap-2 rounded-xl border border-amber-500/40 bg-amber-500/10 px-3 py-2 text-xs text-amber-200">
          <AlertTriangle size={14} className="mt-0.5 shrink-0" /> {loadError}
        </p>
      )}
      {!loading && !loadError && (
        <ul className="space-y-2">
          {SIGN_VOCABULARY.map((s) => {
            const n = counts[s] ?? 0;
            const done = n >= TARGET_SAMPLES;
            return (
              <li key={s} className="rounded-lg border border-slate-800 px-2.5 py-2">
                <div className="flex items-center justify-between gap-2">
                  <span className="flex items-center gap-2 text-sm">
                    <Hand size={14} className="text-violet-300" /> {s}
                    {done && <Badge tone="green">Ready</Badge>}
                  </span>
                  <span className="text-xs text-muted">{n}/{TARGET_SAMPLES}</span>
                </div>
                <div className="mt-1.5 h-1.5 overflow-hidden rounded-full bg-slate-800">
                  <div className="h-full bg-violet" style={{ width: `${Math.min(100, (n / TARGET_SAMPLES) * 100)}%` }} />
                </div>
                <Button
                  size="sm" variant="outline" className="mt-2"
                  disabled={recording !== null}
                  onClick={() => record(s)}
                >
                  <Circle size={13} /> {recording === s ? `Recording… ${recordProgress}%` : `Record “${s}” (2s)`}
                </Button>
                {recording === s && (
                  <div className="mt-2 h-2 overflow-hidden rounded-full bg-slate-800">
                    <div className="h-full bg-cyan transition-all" style={{ width: `${recordProgress}%` }} />
                  </div>
                )}
              </li>
            );
          })}
        </ul>
      )}
      {recordNote && <p className="text-xs text-muted">{recordNote}</p>}
      {savedNote && <p className="text-xs text-success">{savedNote}</p>}
      {recordError && (
        <p role="alert" className="flex items-start gap-2 rounded-xl border border-amber-500/40 bg-amber-500/10 px-3 py-2 text-xs text-amber-200">
          <AlertTriangle size={14} className="mt-0.5 shrink-0" /> {recordError}
        </p>
      )}
      <div className="border-t border-slate-800 pt-3">
        <Button className="w-full" disabled={!allReady || training || loading} onClick={() => void train()}>
          {training ? <><Loader2 size={14} className="animate-spin" /> Training…</> : <><GraduationCap size={14} /> Train model</>}
        </Button>
        {!allReady && !loading && (
          <p className="mt-1.5 text-center text-[11px] text-muted">
            Training unlocks when every sign has at least {MIN_TO_TRAIN} samples.
          </p>
        )}
      </div>
      {job && (
        <div className="rounded-xl border border-slate-800 bg-ink/60 p-3 text-xs">
          <p className="flex items-center justify-between font-medium text-slate-300">
            <span>Training job {job.job_id.slice(0, 8)}…</span>
            <Badge tone={trainingDone ? "green" : trainingFailed ? "red" : "cyan"}>{job.status}</Badge>
          </p>
          <div className="mt-2 h-2 overflow-hidden rounded-full bg-slate-800">
            <div className="h-full bg-violet transition-all" style={{ width: `${Math.round(job.progress * 100)}%` }} />
          </div>
          <p className="mt-1.5 text-muted">
            {Math.round(job.progress * 100)}% · epochs {job.epochs_done}
            {job.val_accuracy != null && <> · val accuracy {Math.round(job.val_accuracy * 100)}%</>}
          </p>
          {job.message && <p className="mt-1 text-muted">{job.message}</p>}
          {trainingDone && (
            <p className="mt-2 font-medium text-success">
              <Check size={13} className="mr-1 inline" />
              Model trained{job.val_accuracy != null ? ` — val accuracy ${Math.round(job.val_accuracy * 100)}%` : ""} — live recognition is now on.
            </p>
          )}
          {trainingFailed && (
            <p role="alert" className="mt-2 flex items-start gap-1.5 text-amber-200">
              <AlertTriangle size={13} className="mt-0.5 shrink-0" /> Training failed: {job.message || "no details from the server."}
            </p>
          )}
        </div>
      )}
      {trainError && (
        <p role="alert" className="flex items-start gap-2 rounded-xl border border-amber-500/40 bg-amber-500/10 px-3 py-2 text-xs text-amber-200">
          <AlertTriangle size={14} className="mt-0.5 shrink-0" /> {trainError}
        </p>
      )}
    </div>
  );
}
