"use client";

/**
 * The call.
 *
 * This is where the two directions meet, and the layout encodes a decision:
 * captions are not an overlay on the video, they are a peer of it. For the
 * Deaf participant the caption rail IS the conversation; video is context.
 * Most video-call UIs get this backwards because most of their users can hear.
 */

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useParams, useSearchParams } from "next/navigation";

import CallStage from "@/components/CallStage";
import CaptionRail from "@/components/CaptionRail";
import ControlBar from "@/components/ControlBar";
import SignPanel from "@/components/SignPanel";
import TranscriptDrawer from "@/components/TranscriptDrawer";
import AccessibilityMenu from "@/components/AccessibilityMenu";

import { loadRuntimeConfig, signerMessage, type Decision, type RuntimeConfig } from "@/lib/decision";
import { MultiLandmarker } from "@/lib/landmarker";
import { SignEngine, type EngineEvent } from "@/lib/signEngine";
import { WebSpeechAsr, isWebSpeechSupported } from "@/lib/asr";
import { speak, unlock as unlockTts } from "@/lib/tts";
import { RoomClient, type RoomEvent } from "@/lib/room";
import { api } from "@/lib/api";

export interface CaptionLine {
  id: string;
  identity: string;
  displayName: string;
  source: "sign" | "speech" | "text" | "system";
  text: string;
  confidence?: number;
  action?: "speak" | "confirm" | "reject";
  startedAtMs: number;
  isFinal: boolean;
  utteranceId?: string;
  mine: boolean;
}

export default function CallPage() {
  const params = useParams<{ room: string }>();
  const search = useSearchParams();
  const roomName = (params.room ?? "").toUpperCase();
  const modality = (search.get("modality") ?? "speech") as "sign" | "speech";
  const displayName = search.get("name") ?? "Guest";
  const dominant = (search.get("hand") ?? "right") as "left" | "right";

  const videoRef = useRef<HTMLVideoElement>(null);
  const engineRef = useRef<SignEngine | null>(null);
  const asrRef = useRef<WebSpeechAsr | null>(null);
  const roomRef = useRef<RoomClient | null>(null);

  const [cfg, setCfg] = useState<RuntimeConfig | null>(null);
  const [status, setStatus] = useState<string>("Connecting…");
  const [error, setError] = useState<string | null>(null);
  const [captions, setCaptions] = useState<CaptionLine[]>([]);
  const [lastDecision, setLastDecision] = useState<Decision | null>(null);
  const [signing, setSigning] = useState(false);
  const [tracked, setTracked] = useState(true);
  const [micOn, setMicOn] = useState(modality === "speech");
  const [camOn, setCamOn] = useState(true);
  const [drawerOpen, setDrawerOpen] = useState(false);
  const [callId, setCallId] = useState<string | null>(null);
  const [latency, setLatency] = useState<{ inference: number; pipeline: number } | null>(null);

  const isSigner = modality === "sign";

  /* --------------------------------------------------------------- */
  /* caption plumbing                                                 */
  /* --------------------------------------------------------------- */
  const pushCaption = useCallback((line: CaptionLine) => {
    setCaptions((prev) => {
      // Streaming ASR emits a growing prefix. Replace the live partial for an
      // utterance rather than appending each one, or a single sentence fills
      // the rail three times over.
      if (!line.isFinal && line.utteranceId) {
        const i = prev.findIndex(
          (c) => c.utteranceId === line.utteranceId && !c.isFinal && c.identity === line.identity,
        );
        if (i >= 0) {
          const next = prev.slice();
          next[i] = line;
          return next;
        }
      }
      if (line.isFinal && line.utteranceId) {
        const filtered = prev.filter(
          (c) => !(c.utteranceId === line.utteranceId && !c.isFinal && c.identity === line.identity),
        );
        return [...filtered, line].slice(-80);
      }
      return [...prev, line].slice(-80);
    });
  }, []);

  /* --------------------------------------------------------------- */
  /* join                                                             */
  /* --------------------------------------------------------------- */
  useEffect(() => {
    let cancelled = false;

    (async () => {
      try {
        setStatus("Joining room…");
        const join = await api.joinRoom(roomName, {
          display_name: displayName,
          modality,
          consent_transcript: false,
        });
        if (cancelled) return;
        setCallId(join.call_id);

        const room = new RoomClient({
          url: join.livekit_url,
          token: join.token,
          identity: join.identity,
          onEvent: handleRoomEvent,
        });
        roomRef.current = room;
        await room.connect({ audio: modality === "speech", video: true });
        if (cancelled) return;

        if (videoRef.current) {
          const local = room.localVideoStream();
          if (local) videoRef.current.srcObject = local;
        }

        setStatus("Connected");
      } catch (e) {
        setError((e as Error).message);
        setStatus("Failed to join");
      }
    })();

    return () => {
      cancelled = true;
      roomRef.current?.disconnect();
      engineRef.current?.stop();
      asrRef.current?.stop();
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [roomName]);

  function handleRoomEvent(ev: RoomEvent) {
    if (ev.type === "data") {
      const msg = ev.payload;
      if (msg.kind === "caption") {
        pushCaption({
          id: `${ev.identity}-${msg.utteranceId ?? Date.now()}-${msg.isFinal}`,
          identity: ev.identity,
          displayName: msg.displayName ?? ev.identity,
          source: msg.source,
          text: msg.text,
          confidence: msg.confidence,
          action: msg.action,
          startedAtMs: msg.startedAtMs,
          isFinal: msg.isFinal,
          utteranceId: msg.utteranceId,
          mine: false,
        });
      }
    } else if (ev.type === "participant") {
      pushCaption({
        id: `sys-${Date.now()}`,
        identity: "system",
        displayName: "System",
        source: "system",
        text: `${ev.identity} ${ev.joined ? "joined" : "left"} the call`,
        startedAtMs: performance.now(),
        isFinal: true,
        mine: false,
      });
    } else if (ev.type === "remoteVideo") {
      // handled inside CallStage via the room client
    }
  }

  /* --------------------------------------------------------------- */
  /* sign engine (signer side)                                        */
  /* --------------------------------------------------------------- */
  useEffect(() => {
    if (!isSigner || !videoRef.current) return;
    let engine: SignEngine | null = null;
    let cancelled = false;

    (async () => {
      try {
        setStatus("Loading sign model…");
        const runtime = await loadRuntimeConfig();
        if (cancelled) return;
        setCfg(runtime);

        engine = new SignEngine(runtime, `/models/${runtime.onnx}`, dominant, {
          onEvent: handleSignEvent,
          onActivity: setSigning,
          onTrackingChange: setTracked,
          onError: (e) => setError(e.message),
        });
        await engine.load(() => MultiLandmarker.create({ useFace: false }));
        if (cancelled) return;

        engineRef.current = engine;
        engine.start(videoRef.current!);
        setStatus("Ready — start signing");
      } catch (e) {
        setError(
          `Sign recognition unavailable: ${(e as Error).message}. ` +
            `The call still works; captions and speech are unaffected.`,
        );
        setStatus("Connected (sign recognition off)");
      }
    })();

    return () => {
      cancelled = true;
      engine?.stop();
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [isSigner, dominant]);

  const handleSignEvent = useCallback(
    async (ev: EngineEvent) => {
      const d = ev.decision;
      setLastDecision(d);
      setLatency({ inference: ev.inferenceMs, pipeline: ev.pipelineMs });

      // Rejections are logged for evaluation but never shown to the other
      // person. Broadcasting "the machine saw something it could not read" is
      // noise to them and embarrassing to the signer.
      if (callId) {
        void api.logEvent(callId, roomRef.current?.identity ?? "", {
          predicted_label: d.label,
          confidence: d.confidence,
          energy_score: d.energy,
          action:
            d.action === "speak" ? "spoken" : d.action === "confirm" ? "confirmed" : "rejected",
          segment_frames: ev.segmentFrames,
          inference_ms: ev.inferenceMs,
          end_to_end_ms: ev.pipelineMs,
          model_version: cfg?.model,
        });
      }

      if (d.action !== "speak") return;

      const utteranceId = `sign-${Math.round(ev.startedAtMs)}`;
      const line: CaptionLine = {
        id: utteranceId,
        identity: roomRef.current?.identity ?? "me",
        displayName,
        source: "sign",
        text: d.spoken,
        confidence: d.confidence,
        action: d.action,
        startedAtMs: ev.startedAtMs,
        isFinal: true,
        utteranceId,
        mine: true,
      };
      pushCaption(line);

      roomRef.current?.publishCaption({
        kind: "caption",
        source: "sign",
        text: d.spoken,
        displayName,
        confidence: d.confidence,
        action: d.action,
        startedAtMs: ev.startedAtMs,
        isFinal: true,
        utteranceId,
      });

      // Local playback. In a genuinely remote call this is the wrong speaker -
      // see lib/tts.ts synthesizeToTrack() for publishing the voice into the
      // call so the other person actually hears it.
      void speak(d.spoken, { voiceURI: null });
    },
    [callId, cfg, displayName, pushCaption],
  );

  /* --------------------------------------------------------------- */
  /* ASR (hearing side)                                               */
  /* --------------------------------------------------------------- */
  useEffect(() => {
    if (isSigner || !micOn) return;
    if (!isWebSpeechSupported()) {
      setError("This browser has no speech recognition — try Chrome or Edge.");
      return;
    }
    const asr = new WebSpeechAsr({
      lang: "en-IN",
      onHypothesis: (h) => {
        const line: CaptionLine = {
          id: `${h.utteranceId}-${h.isFinal}`,
          identity: roomRef.current?.identity ?? "me",
          displayName,
          source: "speech",
          text: h.text,
          startedAtMs: h.startedAtMs,
          isFinal: h.isFinal,
          utteranceId: h.utteranceId,
          mine: true,
        };
        pushCaption(line);
        roomRef.current?.publishCaption({
          kind: "caption",
          source: "speech",
          text: h.text,
          displayName,
          startedAtMs: h.startedAtMs,
          isFinal: h.isFinal,
          utteranceId: h.utteranceId,
        });
      },
      onError: (e) => setError(`Speech recognition: ${e}`),
    });
    asr.start();
    asrRef.current = asr;
    return () => asr.stop();
  }, [isSigner, micOn, displayName, pushCaption]);

  /* --------------------------------------------------------------- */
  const signerHint = useMemo(
    () => (lastDecision ? signerMessage(lastDecision) : ""),
    [lastDecision],
  );

  return (
    <div className="flex h-dvh flex-col bg-bg text-fg">
      <a href="#captions" className="skip-link">
        Skip to captions
      </a>

      <header className="flex items-center justify-between border-b border-border px-4 py-3">
        <div className="flex items-baseline gap-3">
          <h1 className="text-lg font-semibold tracking-tight">SignSpeak</h1>
          <code className="rounded bg-surface px-2 py-1 text-sm tracking-widest">{roomName}</code>
        </div>
        <div className="flex items-center gap-3">
          <span className="text-sm text-muted" role="status" aria-live="polite">
            {status}
          </span>
          <AccessibilityMenu />
        </div>
      </header>

      {error && (
        <div
          role="alert"
          className="border-b border-low/40 bg-low/10 px-4 py-2 text-sm text-low"
        >
          {error}
        </div>
      )}

      <main className="grid flex-1 grid-cols-1 gap-4 overflow-hidden p-4 lg:grid-cols-[1fr_22rem]">
        <div className="flex min-h-0 flex-col gap-4">
          <CallStage
            videoRef={videoRef}
            room={roomRef.current}
            camOn={camOn}
            signing={signing}
            tracked={tracked}
            isSigner={isSigner}
          />
          <section id="captions" aria-label="Live captions" className="min-h-0 flex-1">
            <CaptionRail lines={captions} />
          </section>
        </div>

        <aside className="flex min-h-0 flex-col gap-4 overflow-y-auto">
          {isSigner && (
            <SignPanel
              decision={lastDecision}
              signing={signing}
              tracked={tracked}
              latency={latency}
              vocab={cfg?.vocab ?? []}
              onCorrect={(actual) => {
                if (!callId || !lastDecision) return;
                void api.logEvent(callId, roomRef.current?.identity ?? "", {
                  predicted_label: lastDecision.label,
                  confidence: lastDecision.confidence,
                  action: "corrected",
                  corrected_label: actual,
                  segment_frames: 0,
                  model_version: cfg?.model,
                });
              }}
            />
          )}
          <p className="rounded-lg border border-border bg-surface p-3 text-sm text-muted">
            {signerHint || (isSigner ? "Sign within the frame, then pause briefly." : "Speak normally.")}
          </p>
        </aside>
      </main>

      <TranscriptDrawer
        open={drawerOpen}
        onClose={() => setDrawerOpen(false)}
        lines={captions}
        roomName={roomName}
      />

      <ControlBar
        micOn={micOn}
        camOn={camOn}
        onToggleMic={() => {
          roomRef.current?.setMicrophone(!micOn);
          setMicOn((v) => !v);
        }}
        onToggleCam={() => {
          roomRef.current?.setCamera(!camOn);
          setCamOn((v) => !v);
        }}
        onOpenTranscript={() => setDrawerOpen(true)}
        onLeave={() => {
          roomRef.current?.disconnect();
          window.location.href = "/";
        }}
        onUnlockAudio={unlockTts}
      />
    </div>
  );
}
