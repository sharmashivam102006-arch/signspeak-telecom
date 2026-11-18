"use client";
import { useCallback, useEffect, useRef, useState } from "react";
import { useParams, useRouter } from "next/navigation";
import { Copy, Check, Wifi, Users, AlertTriangle, PhoneOff, Mic, MicOff, Video, VideoOff, X } from "lucide-react";
import { Navbar } from "@/components/Navbar";
import { VideoTile } from "@/components/VideoTile";
import { CaptionBar } from "@/components/CaptionBar";
import { TranscriptPanel, type TranscriptDisplayEntry } from "@/components/TranscriptPanel";
import { ConfidenceCard } from "@/components/ConfidenceCard";
import { AIPanel } from "@/components/AIPanel";
import { AIStatusRow } from "@/components/AIStatusRow";
import { SignTrainer } from "@/components/SignTrainer";
import { CallControls } from "@/components/CallControls";
import { Badge } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { API_URL, ApiError, api, getToken, getUser } from "@/lib/api";
import { LiveKitCall, type ParticipantInfo, type RemoteTrackInfo } from "@/lib/livekit";
import type { LocalAudioTrack, LocalVideoTrack } from "livekit-client";
import { createHandTracker, SequenceBuffer, PredictionSmoother, predictSign, sequenceMoved, type HandTracker, type Landmark } from "@/lib/sign";
import { CaptionRelayClient, MicChunkStreamer, WhisperCaptionClient, mediaRecorderSupported, type RelayCaption, type RelayState } from "@/lib/captions";
import { createSpeechRecognizer, speechSupported } from "@/lib/speech";
import { speakText, stopSpeaking, ttsSupported } from "@/lib/tts";
import { TranscriptEntry, Settings, loadSettings, loadTranscript, saveTranscript, saveCall, SIGN_VOCABULARY } from "@/lib/store";
import { collectSignSample, fetchAIStatus, fetchServerTranscripts, fetchSignSamples, fetchSignStatus, fetchVoiceStatus, saveTranscriptsToServer, speakViaVoiceAgent, startVoiceAgent, stopVoiceAgent, type AIStatus } from "@/lib/ai";

type Pending = { sign: string; confidence: number; demo: boolean; mode: "trained-model" | "demo"; sequence?: Landmark[][] };
type RoomInfo = { id: string; code: string; name: string; creator_id?: string | null; is_active: boolean };
type TokenResponse = { room_id: string; code: string; token: string | null; url: string | null; mode: string; note?: string };

export default function RoomPage() {
  const params = useParams();
  const roomId = String(params.id);
  const router = useRouter();
  const [settings, setSettings] = useState<Settings>(loadSettings());
  const settingsRef = useRef(settings);
  settingsRef.current = settings;

  const [room, setRoom] = useState<RoomInfo | null>(null);
  const [isCreator, setIsCreator] = useState(false);
  const [roomMissing, setRoomMissing] = useState("");
  const [status, setStatus] = useState("Connecting…");
  const [error, setError] = useState("");
  const [micOn, setMicOn] = useState(true);
  const [camOn, setCamOn] = useState(true);
  const [captionsOn, setCaptionsOn] = useState(true);
  const [panelOpen, setPanelOpen] = useState(true);
  const [participants, setParticipants] = useState<ParticipantInfo[]>([]);
  const [remotes, setRemotes] = useState<RemoteTrackInfo[]>([]);
  const [serverParticipants, setServerParticipants] = useState<{ identity: string; name?: string }[]>([]);
  const [localVideoTrack, setLocalVideoTrack] = useState<LocalVideoTrack | null>(null);
  const [localAudioTrack, setLocalAudioTrack] = useState<LocalAudioTrack | null>(null);
  const [localStream, setLocalStream] = useState<MediaStream | null>(null);
  const [caption, setCaption] = useState("");
  const [captionConfidence, setCaptionConfidence] = useState<number | null>(null);
  const [captionEngine, setCaptionEngine] = useState<"whisper" | "browser" | null>(null);
  const [interim, setInterim] = useState("");
  // Caption relay (/ws/call): carries final captions to the other
  // participant. `remoteCaption` is the latest caption received from
  // them, shown with their name.
  const [relayState, setRelayState] = useState<RelayState>("disconnected");
  const [remoteCaption, setRemoteCaption] = useState<RelayCaption | null>(null);
  const [entries, setEntries] = useState<TranscriptDisplayEntry[]>(() => loadTranscript(roomId) as TranscriptDisplayEntry[]);
  const [pending, setPending] = useState<Pending | null>(null);
  const [signStatus, setSignStatus] = useState<"real" | "demo" | "unavailable">("unavailable");
  const [signEnabled, setSignEnabled] = useState(true);
  const [speechStatus, setSpeechStatus] = useState<"ready" | "unsupported" | "listening" | "error">("ready");
  const [copied, setCopied] = useState(false);
  const [elapsed, setElapsed] = useState(0);
  const [leaving, setLeaving] = useState(false);

  // --- Real AI pipeline state ---
  const [aiStatus, setAiStatus] = useState<AIStatus | null>(null);
  const [aiStatusNote, setAiStatusNote] = useState("");
  const [signTrained, setSignTrained] = useState<boolean | null>(null);
  const [sampleCounts, setSampleCounts] = useState<Record<string, number>>({});
  const [trainerOpen, setTrainerOpen] = useState(false);
  const [voiceAgentOn, setVoiceAgentOn] = useState<boolean | null>(null);
  const [speaking, setSpeaking] = useState(false);
  const [speakError, setSpeakError] = useState("");
  const [sayText, setSayText] = useState("");
  // Voice-agent (remote) audio level/mute for this browser only.
  const [voiceVolume, setVoiceVolume] = useState(0.9);
  const [voiceMuted, setVoiceMuted] = useState(false);
  const [ttsEngine, setTtsEngine] = useState<string | null>(null);
  const [aiNote, setAiNote] = useState("");
  const [correctNote, setCorrectNote] = useState("");

  const localVideoRef = useRef<HTMLVideoElement>(null);
  const callRef = useRef<LiveKitCall | null>(null);
  const trackerRef = useRef<HandTracker | null>(null);
  const smootherRef = useRef(new PredictionSmoother());
  const bufferRef = useRef(new SequenceBuffer(30));
  const latestLandmarksRef = useRef<Landmark[] | null>(null);
  const handVisibleRef = useRef(false);
  const modelNotTrainedRef = useRef(false);
  const predictInFlightRef = useRef(false);
  const startedAtRef = useRef(Date.now());
  const leftRef = useRef(false);
  // Caption relay + dedup state (refs so socket callbacks stay fresh).
  const relayRef = useRef<CaptionRelayClient | null>(null);
  const seenRelayIdsRef = useRef<Set<string>>(new Set());
  const lastRelayTextRef = useRef<{ text: string; at: number }>({ text: "", at: 0 });
  const lastSpokenRef = useRef<{ text: string; at: number }>({ text: "", at: 0 });
  const serverTranscriptsLoadedRef = useRef(false);
  const savedServerIdsRef = useRef<Set<string>>(new Set());

  const addEntry = useCallback((e: Omit<TranscriptEntry, "id" | "timestamp"> & { speaker?: string }) => {
    const entry: TranscriptDisplayEntry = { ...e, id: `${Date.now()}-${Math.random().toString(36).slice(2, 6)}`, timestamp: new Date().toISOString() };
    setEntries(prev => {
      const last = prev[prev.length - 1];
      if (last && last.text === entry.text && last.direction === entry.direction && Date.now() - new Date(last.timestamp).getTime() < 4000) return prev;
      const next = [...prev, entry].slice(-300);
      saveTranscript(roomId, next);
      return next;
    });
  }, [roomId]);

  // --- Caption relay: send a local final caption to the other side ---
  const sendLocalCaption = useCallback((text: string, confidence: number | null, engine: "whisper" | "browser") => {
    const user = getUser();
    const ok = relayRef.current?.sendCaption({
      text,
      confidence: confidence ?? 0,
      speaker_id: user?.id ?? "",
      speaker_name: user?.name ?? "You",
      is_final: true,
      engine,
    });
    if (ok === false && relayRef.current) {
      // Relay down: the caption still shows locally; the relay badge
      // explains it is not reaching the other participant.
    }
  }, []);

  // --- Caption relay: a caption arrived from the remote participant ---
  const handleRelayCaption = useCallback((c: RelayCaption) => {
    // Dedupe: same relay id twice, or identical text within 2s.
    if (seenRelayIdsRef.current.has(c.id)) return;
    seenRelayIdsRef.current.add(c.id);
    const now = Date.now();
    if (lastRelayTextRef.current.text === c.text && now - lastRelayTextRef.current.at < 2000) return;
    lastRelayTextRef.current = { text: c.text, at: now };
    setRemoteCaption(c);
    addEntry({ text: c.text, direction: "speech-to-text", source: "speech", confidence: c.confidence, speaker: c.speaker_name });
  }, [addEntry]);
  const handleRelayCaptionRef = useRef(handleRelayCaption);
  handleRelayCaptionRef.current = handleRelayCaption;
  const sendLocalCaptionRef = useRef(sendLocalCaption);
  sendLocalCaptionRef.current = sendLocalCaption;

  // --- Voice agent: speak accepted text into the LiveKit room ---
  const speakAccepted = useCallback(async (text: string): Promise<boolean> => {
    // Prevent repeated speech: identical text within the 4s duplicate
    // window is already spoken (or already in the transcript) — do not
    // queue it again.
    const now = Date.now();
    if (lastSpokenRef.current.text === text && now - lastSpokenRef.current.at < 4000) return true;
    setSpeaking(true);
    setSpeakError("");
    try {
      await speakViaVoiceAgent(roomId, text);
      lastSpokenRef.current = { text, at: Date.now() };
      return true;
    } catch (e) {
      // Show the backend's error verbatim; do not pretend it was spoken.
      setSpeakError(e instanceof Error ? e.message : String(e));
      return false;
    } finally {
      setSpeaking(false);
    }
  }, [roomId]);

  const localPreview = useCallback((text: string) => {
    const s = settingsRef.current;
    if (ttsSupported() && !s.ttsMuted) {
      speakText(text, { muted: false, volume: s.ttsVolume, voiceName: s.ttsVoice || undefined, lang: s.language });
    }
  }, []);

  const acceptPrediction = useCallback((p: Pending, correctedText?: string) => {
    const text = correctedText || p.sign;
    addEntry({ text, direction: "sign-to-speech", source: "sign", confidence: p.confidence, demo: p.demo, speaker: getUser()?.name ?? "You" });
    if (p.demo) {
      // Demo results are a labelled test aid: local preview only, never
      // the voice agent — a demo sign must not be spoken to the remote side.
      localPreview(text);
    } else if (voiceAgentOn) {
      void speakAccepted(text);
    } else {
      // Voice agent off: local preview only; the panel explains that the
      // remote participant cannot hear it.
      localPreview(text);
    }
    setPending(null);
  }, [addEntry, localPreview, speakAccepted, voiceAgentOn]);

  const handlePrediction = useCallback((sign: string, confidence: number, demo: boolean, sequence: Landmark[][]) => {
    const s = settingsRef.current;
    // Surface every non-duplicate prediction in the confirm card; only
    // >= high threshold auto-accepts. Low confidence is shown, not hidden,
    // so the user can Repeat or Correct it.
    const accepted = smootherRef.current.accept(sign, confidence, 0);
    if (!accepted) return;
    const mode = demo ? "demo" : "trained-model";
    if (confidence >= s.signThresholdHigh / 100) acceptPrediction({ sign, confidence, demo, mode, sequence });
    else setPending({ sign, confidence, demo, mode, sequence });
  }, [acceptPrediction]);

  // The tracker loop runs outside React's render cycle; keep the latest
  // handler in a ref so it always sees current voice-agent state.
  const handlePredictionRef = useRef(handlePrediction);
  handlePredictionRef.current = handlePrediction;

  const setModelNotTrained = useCallback((message: string) => {
    modelNotTrainedRef.current = true;
    setSignTrained(false);
    setSignStatus("unavailable");
    setAiNote(message);
  }, []);

  // --- AI status on room entry (once) ---
  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const st = await fetchAIStatus();
        if (cancelled) return;
        if (st) {
          setAiStatus(st);
          if (st.sign) {
            setSignTrained(st.sign.trained);
            setSampleCounts(st.sign.samples_per_sign ?? {});
            if (st.sign.trained) modelNotTrainedRef.current = false;
            else modelNotTrainedRef.current = true;
          }
          if (st.tts?.engine) setTtsEngine(st.tts.engine);
        } else {
          setAiStatusNote("AI status is unavailable — the backend does not expose /api/ai/status yet.");
          try {
            const ss = await fetchSignStatus();
            if (!cancelled && ss) {
              setSignTrained(ss.trained);
              setSampleCounts(ss.samples_per_sign ?? {});
              modelNotTrainedRef.current = !ss.trained;
            }
          } catch { /* sign status probe is best-effort */ }
          try {
            const smp = await fetchSignSamples();
            if (!cancelled) setSampleCounts(smp.samples_per_sign);
          } catch { /* samples probe is best-effort */ }
        }
      } catch (e) {
        if (!cancelled) setAiStatusNote(`Could not load AI status: ${e instanceof Error ? e.message : String(e)}`);
      }
      // Voice status is NOT fetched here: it requires room membership,
      // so it 403s on mount before the join runs. It is fetched after
      // the join succeeds (see the join effect below).
    })();
    return () => { cancelled = true; };
  }, [roomId]);

  // --- Server transcripts: load the room's stored transcript on open
  // (creator only — the backend 403s for everyone else). Merged into
  // the local transcript, deduped against local entries by text +
  // timestamp so a reloaded call does not show the same line twice.
  useEffect(() => {
    if (!isCreator || !room || serverTranscriptsLoadedRef.current) return;
    serverTranscriptsLoadedRef.current = true;
    (async () => {
      try {
        const server = await fetchServerTranscripts(roomId);
        if (!server.length) return;
        setEntries(prev => {
          const incoming: TranscriptDisplayEntry[] = server
            .filter(s => s.text.trim())
            .map(s => ({
              id: `server-${s.id}`,
              text: s.text,
              direction: (s.direction === "sign-to-speech" ? "sign-to-speech" : "speech-to-text") as TranscriptDisplayEntry["direction"],
              source: (s.direction === "sign-to-speech" ? "sign" : "speech") as TranscriptDisplayEntry["source"],
              confidence: typeof s.confidence === "number" ? s.confidence / 100 : undefined,
              speaker: s.speaker_name || undefined,
              timestamp: s.created_at || new Date().toISOString(),
            }))
            .filter(s => !prev.some(p =>
              p.text === s.text && p.direction === s.direction &&
              Math.abs(new Date(p.timestamp).getTime() - new Date(s.timestamp).getTime()) < 2000));
          if (!incoming.length) return prev;
          const merged = [...prev, ...incoming].sort((a, b) => new Date(a.timestamp).getTime() - new Date(b.timestamp).getTime()).slice(-300);
          saveTranscript(roomId, merged);
          return merged;
        });
      } catch {
        // 403 for non-creators is expected; anything else just means
        // the server copy stays unavailable — local transcript still works.
      }
    })();
  }, [isCreator, room, roomId]);

  // --- Save the local transcript to the server (creator only). Entries
  // already stored (or loaded from the server) are skipped so repeated
  // saves do not duplicate rows server-side.
  const saveToServer = useCallback(async (): Promise<{ saved: number }> => {
    const user = getUser();
    const toSave = entries.filter(e => !e.id.startsWith("server-") && !savedServerIdsRef.current.has(e.id));
    const r = await saveTranscriptsToServer(roomId, toSave.map(e => ({
      direction: e.direction,
      text: e.text,
      speaker_name: e.speaker ?? user?.name ?? "",
      confidence: e.confidence ?? null,
    })));
    toSave.forEach(e => savedServerIdsRef.current.add(e.id));
    return r;
  }, [entries, roomId]);

  // --- LiveKit call: validate room -> join -> token -> connect ---
  useEffect(() => {
    if (!getToken()) { router.replace(`/login?next=/room/${encodeURIComponent(roomId)}`); return; }
    let cancelled = false;
    const joinCall = async () => {
      try {
        // 1. Validate the room exists before touching any media.
        // Room lookup is member-only: an invitee who has never joined
        // gets 403 here, so join first (the invite entry point) and
        // then re-read the room instead of failing the invite.
        let info: RoomInfo;
        try {
          info = await api<RoomInfo>(`/api/rooms/${encodeURIComponent(roomId)}`);
        } catch (e) {
          if (e instanceof ApiError && e.status === 403) {
            const isCode = /^[A-Za-z0-9]{6}$/.test(roomId);
            await api(`/api/rooms/${encodeURIComponent(roomId)}/join`, {
              method: "POST",
              body: JSON.stringify(isCode ? { code: roomId } : {}),
            });
            info = await api<RoomInfo>(`/api/rooms/${encodeURIComponent(roomId)}`);
          } else {
            throw e;
          }
        }
        if (cancelled) return;
        if (!info.is_active) {
          setRoomMissing("This room has ended.");
          return;
        }
        setRoom(info);
        setIsCreator(info.creator_id != null && info.creator_id === getUser()?.id);

        // 2. Record the join (server-side presence; idempotent).
        await api(`/api/rooms/${encodeURIComponent(roomId)}/join`, { method: "POST", body: JSON.stringify({}) });

        // 2b. Open the caption relay (/ws/call) now — independent of
        // LiveKit, so captions can still cross to the other participant
        // even if media connect fails. Closed on unmount.
        const relay = new CaptionRelayClient(roomId, getToken(), {
          onCaption: (c) => handleRelayCaptionRef.current(c),
          onState: (s) => { if (!cancelled) setRelayState(s); },
        });
        relayRef.current = relay;
        relay.connect();
        // Test hook (used by the Playwright caption-relay check): the
        // page's own relay client, so a test can send a real caption
        // through the production send path.
        (window as unknown as { __captionRelay?: CaptionRelayClient }).__captionRelay = relay;

        // 3. Server-issued LiveKit token. No token / not LiveKit mode means
        //    the call cannot start — show it honestly, never fall back to
        //    P2P or a fake call.
        let tok: TokenResponse;
        try {
          tok = await api<TokenResponse>(`/api/rooms/${encodeURIComponent(roomId)}/token`);
        } catch (e) {
          const msg = e instanceof ApiError ? e.message : "LiveKit token request failed.";
          setError(`LiveKit is not available: ${msg} (HTTP ${e instanceof ApiError ? e.status : "?"}) — the backend must be configured with LiveKit credentials.`);
          setStatus("Failed");
          return;
        }
        if (cancelled) return;
        if (tok.mode !== "livekit" || !tok.token || !tok.url) {
          setError(
            `LiveKit is not configured on the server${tok.note ? `: ${tok.note}` : "."} Media calling is unavailable — ask the server operator to set LIVEKIT_URL / LIVEKIT_API_KEY / LIVEKIT_API_SECRET.`,
          );
          setStatus("Failed");
          return;
        }

        // 3b. Seed the participant panel from the server presence list
        // BEFORE connect(): connect() throws on failure, so a seed placed
        // after it never runs and the panel shows a wrong "1 participant".
        // (LiveKit presence remains the source of truth once connected.)
        try {
          const list = await api<unknown>(`/api/rooms/${encodeURIComponent(roomId)}/participants`);
          const arr = Array.isArray(list) ? list : (list as { participants?: unknown[] })?.participants;
          if (Array.isArray(arr)) {
            setServerParticipants(
              arr
                .filter((p): p is Record<string, unknown> => typeof p === "object" && p !== null)
                .map((p) => ({ identity: String(p.identity ?? p.user_id ?? ""), name: typeof p.name === "string" ? p.name : undefined }))
                .filter((p) => p.identity),
            );
          }
        } catch {
          // Participants endpoint not available yet — LiveKit events still
          // keep the panel live. Not a call failure.
        }
        if (cancelled) return;

        // 4. Connect to the LiveKit room and publish camera + mic.
        const call = new LiveKitCall({
          onLocalTracks: (video, audio) => {
            setLocalVideoTrack(video);
            setLocalAudioTrack(audio);
          },
          onRemoteTracks: (info) => {
            setRemotes((prev) => {
              const next = prev.filter((r) => r.participant.identity !== info.participant.identity);
              return [...next, info];
            });
          },
          onRemoteLeft: (identity) => {
            setRemotes((prev) => prev.filter((r) => r.participant.identity !== identity));
          },
          onParticipantsChanged: (list) => setParticipants(list),
          onConnectionState: (s) => setStatus(s),
          onError: (m) => setError(m),
        });
        callRef.current = call;
        await call.connect(tok.url, tok.token);
        if (cancelled) {
          await call.disconnect();
          return;
        }
        setLocalStream(call.getLocalMediaStream());
        setMicOn(call.getLocalParticipant()?.isMicrophoneEnabled ?? true);
        setCamOn(call.getLocalParticipant()?.isCameraEnabled ?? true);

        // 5. Voice-agent status — only now, after the join, when the
        // membership check passes (fetching on mount 403s).
        try {
          const vs = await fetchVoiceStatus(roomId);
          if (!cancelled && vs) setVoiceAgentOn(vs.active);
        } catch { /* voice status unknown — badge shows Unknown */ }
      } catch (e) {
        if (cancelled) return;
        if (e instanceof ApiError && e.status === 404) setRoomMissing("Room not found — check the invite link or room code.");
        else if (e instanceof ApiError && e.status === 410) setRoomMissing("This room has ended.");
        else setError(`Could not join the room: ${e instanceof Error ? e.message : String(e)}`);
        setStatus("Failed");
        // A failed join leaves mic/cam actually off (no tracks were
        // published), so sync the buttons to the real state instead of
        // the optimistic `true` defaults — otherwise the first tap
        // enables the device instead of muting it.
        const failedCall = callRef.current;
        if (failedCall) {
          setMicOn(failedCall.getLocalParticipant()?.isMicrophoneEnabled ?? false);
          setCamOn(failedCall.getLocalParticipant()?.isCameraEnabled ?? false);
        }
      }
    };
    joinCall();
    const timer = setInterval(() => setElapsed(Math.floor((Date.now() - startedAtRef.current) / 1000)), 1000);
    return () => {
      cancelled = true;
      clearInterval(timer);
      // Leaving via unmount (e.g. back navigation): disconnect media and
      // close the caption relay. The explicit End-call buttons also POST
      // /leave first.
      relayRef.current?.close();
      relayRef.current = null;
      void callRef.current?.disconnect();
      callRef.current = null;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [roomId]);

  // Keep the MediaPipe input stream in sync when local tracks change
  // (e.g. camera re-acquired after being toggled back on).
  useEffect(() => {
    const call = callRef.current;
    if (call) setLocalStream(call.getLocalMediaStream());
  }, [localVideoTrack, localAudioTrack]);

  // Attach the LiveKit local camera stream to the hidden processing video.
  useEffect(() => {
    const el = localVideoRef.current;
    if (!el || !localStream) return;
    el.srcObject = localStream;
    el.play().catch(() => undefined);
  }, [localStream]);

  // --- Live captions: Whisper WS first, browser Web Speech fallback ---
  useEffect(() => {
    if (!captionsOn) {
      setCaptionEngine(null);
      setInterim("");
      return;
    }
    if (!localAudioTrack) return; // Wait for the mic track.
    let cancelled = false;
    let streamer: MicChunkStreamer | null = null;
    let wsClient: WhisperCaptionClient | null = null;
    let browserRec: { stop: () => void } | null = null;
    let fallbackStarted = false;
    let fallbackTimer: number | null = null;

    const startBrowser = () => {
      if (cancelled || fallbackStarted) return;
      fallbackStarted = true;
      setCaptionEngine("browser");
      if (!speechSupported()) {
        setSpeechStatus("unsupported");
        setAiNote("Live captions are unavailable: Whisper could not be reached and this browser has no Web Speech API.");
        return;
      }
      const rec = createSpeechRecognizer(settingsRef.current.language, {
        onInterim: setInterim,
        onFinal: (text) => {
          setInterim("");
          setCaption(text);
          setCaptionConfidence(null);
          addEntry({ text, direction: "speech-to-text", source: "speech", speaker: getUser()?.name ?? "You" });
          // Browser captions are real transcriptions too: relay them so
          // the remote participant sees them (labelled by engine).
          sendLocalCaptionRef.current(text, null, "browser");
        },
        onError: (msg) => { setSpeechStatus("error"); setError(msg); },
        onEnd: () => setSpeechStatus(prev => (prev === "listening" ? "ready" : prev)),
      });
      if (!rec) return;
      browserRec = rec;
      rec.start();
      setSpeechStatus("listening");
    };

    const startWhisper = () => {
      if (!mediaRecorderSupported()) { startBrowser(); return; }
      const track = localAudioTrack.mediaStreamTrack;
      if (!track) { startBrowser(); return; }
      setSpeechStatus("listening");
      wsClient = new WhisperCaptionClient(getToken(), {
        onOpen: () => {
          if (!cancelled) setCaptionEngine("whisper");
        },
        onEvent: (e) => {
          if (cancelled) return;
          if (e.type === "partial") {
            setCaptionEngine("whisper");
            setInterim(e.text);
          } else if (e.type === "final") {
            setInterim("");
            setCaption(e.text);
            setCaptionConfidence(e.confidence);
            setCaptionEngine("whisper");
            addEntry({ text: e.text, direction: "speech-to-text", source: "speech", confidence: e.confidence, speaker: getUser()?.name ?? "You" });
            // Cross-participant delivery: send the final Whisper caption
            // over the room relay so the other participant sees it.
            sendLocalCaptionRef.current(e.text, e.confidence, "whisper");
          } else if (e.type === "error") {
            // Whisper reported an error (e.g. model unavailable): fall back.
            setAiNote(`Whisper captions: ${e.message} Falling back to browser captions.`);
            wsClient?.close();
            startBrowser();
          }
        },
        onClose: () => { if (!cancelled) startBrowser(); },
      });
      wsClient.connect();
      streamer = new MicChunkStreamer(new MediaStream([track]), (blob) => {
        void blob.arrayBuffer().then((buf) => wsClient?.sendAudioChunk(buf));
      });
      if (!streamer.start()) startBrowser();
      // If the socket is not open within 6s, fall back to the browser.
      fallbackTimer = window.setTimeout(() => {
        if (!cancelled && wsClient && !wsClient.isOpen) {
          wsClient.close();
          startBrowser();
        }
      }, 6000);
    };

    startWhisper();
    return () => {
      cancelled = true;
      if (fallbackTimer) window.clearTimeout(fallbackTimer);
      streamer?.stop();
      wsClient?.close();
      browserRec?.stop();
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [captionsOn, localAudioTrack]);

  // --- MediaPipe hand tracking (sign → trained model) over the LiveKit camera ---
  useEffect(() => {
    if (!signEnabled || !localStream) return;
    let tracker: HandTracker | null = null;
    let stopped = false;
    let lastPredict = 0;
    let lastErr = "";
    createHandTracker().then((res) => {
      if (stopped) { res.tracker?.stop(); return; }
      if (!res.tracker) {
        setSignStatus("unavailable");
        setAiNote(res.error);
        return;
      }
      tracker = res.tracker;
      trackerRef.current = tracker;
      const video = localVideoRef.current;
      if (!video) return;
      tracker.start(video, (lm) => {
        latestLandmarksRef.current = lm;
        handVisibleRef.current = lm !== null;
        if (!lm) return;
        // Raw 21x3 landmarks — normalisation happens server-side.
        bufferRef.current.push(lm);
        const now = Date.now();
        if (!bufferRef.current.ready || now - lastPredict < 800) return;
        if (modelNotTrainedRef.current || predictInFlightRef.current) return;
        const seq = bufferRef.current.snapshot();
        if (!sequenceMoved(seq)) return;
        lastPredict = now;
        predictInFlightRef.current = true;
        bufferRef.current.clear();
        predictSign(seq, API_URL, getToken())
          .then((out) => {
            predictInFlightRef.current = false;
            if (out.kind === "trained") {
              setSignStatus("real");
              setAiNote("");
              handlePredictionRef.current(out.prediction, out.confidence, false, seq);
            } else if (out.kind === "demo") {
              // A plain live call returned demo heuristics (old backend):
              // not real recognition — report the model as not trained.
              setModelNotTrained("Sign model is not trained yet — the backend only offers demo heuristics.");
            } else if (out.kind === "not-trained") {
              setModelNotTrained(out.message);
            } else if (out.kind === "error") {
              if (out.message !== lastErr) { lastErr = out.message; setAiNote(out.message); }
            }
            // no-hand: ignore silently.
          })
          .catch(() => { predictInFlightRef.current = false; });
      });
    });
    return () => {
      stopped = true;
      tracker?.stop();
      trackerRef.current = null;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [signEnabled, localStream]);

  // Demo sign buttons: labelled test aid. They call POST /predict with
  // demo=true on the current landmark sequence and show the result with a
  // DEMO badge — never as a real model prediction. Needs a hand in view.
  const demoSign = async (sign: string) => {
    const seq = bufferRef.current.snapshot();
    if (seq.length < 5) {
      setAiNote("Demo needs a hand in view — hold a sign pose in front of the camera, then tap a demo sign.");
      return;
    }
    const out = await predictSign(seq, API_URL, getToken(), { demo: true });
    if (out.kind === "demo") {
      setSignStatus("demo");
      handlePrediction(out.prediction, out.confidence, true, seq);
    } else if (out.kind === "no-hand") {
      setAiNote("Demo: no hand pose was recognised from the current frames.");
    } else if (out.kind === "not-trained") {
      setAiNote(out.message);
    } else if (out.kind === "trained") {
      // A demo call returning a trained prediction is still a real result.
      setSignStatus("real");
      handlePrediction(out.prediction, out.confidence, false, seq);
    } else {
      setAiNote(out.message);
    }
  };

  // Correct: accept the corrected text AND store the sequence that
  // produced the prediction as a training sample for the corrected label.
  const correctPrediction = (correctedText: string) => {
    const p = pending;
    if (!p) return;
    if (p.sequence && p.sequence.length >= 5 && (SIGN_VOCABULARY as readonly string[]).includes(correctedText)) {
      const frames = p.sequence.map((f) => f.slice(0, 21).map((pt) => [pt.x, pt.y, pt.z]));
      collectSignSample(correctedText, frames)
        .then((r) => {
          setSampleCounts((prev) => ({ ...prev, [correctedText]: r.count }));
          setCorrectNote(`Saved as a “${correctedText}” training sample (${r.count}/10).`);
        })
        .catch((e: unknown) => {
          setCorrectNote(`Correction accepted, but it could not be saved as a training sample: ${e instanceof Error ? e.message : String(e)}`);
        });
    }
    acceptPrediction(p, correctedText);
  };

  // --- Voice agent controls ---
  const toggleVoiceAgent = async () => {
    setSpeakError("");
    try {
      if (voiceAgentOn) {
        await stopVoiceAgent(roomId);
        setVoiceAgentOn(false);
      } else {
        await startVoiceAgent(roomId);
        setVoiceAgentOn(true);
        // Refresh the room-wide status row after the change.
        const st = await fetchAIStatus();
        if (st) setAiStatus(st);
      }
    } catch (e) {
      setSpeakError(e instanceof Error ? e.message : String(e));
    }
  };

  const sayThis = async () => {
    const text = sayText.trim();
    if (!text) return;
    if (voiceAgentOn) {
      // Only record the line as spoken into the call when the voice
      // agent actually spoke it; on failure the error is shown and no
      // transcript line pretends it was heard.
      const ok = await speakAccepted(text);
      if (ok) addEntry({ text, direction: "sign-to-speech", source: "system", speaker: getUser()?.name ?? "You" });
    } else {
      addEntry({ text, direction: "sign-to-speech", source: "system", speaker: getUser()?.name ?? "You" });
      localPreview(text);
    }
    setSayText("");
  };

  const onTrainingDone = useCallback(async () => {
    modelNotTrainedRef.current = false;
    setSignTrained(true);
    setTrainerOpen(false);
    try {
      const st = await fetchAIStatus();
      if (st) {
        setAiStatus(st);
        if (st.sign) setSampleCounts(st.sign.samples_per_sign ?? {});
      }
    } catch { /* status refresh is best-effort */ }
    setAiNote("Model trained — live sign recognition is now on.");
  }, []);

  // --- Controls: these drive LiveKit, not just local state ---
  // Mic/cam only work while connected; otherwise the buttons are
  // disabled (see the CallControls wrapper below) and a tap that still
  // gets through explains why instead of silently doing nothing.
  const connected = status === "Connected";
  const notConnectedNote = "You are not connected to the call, so microphone and camera controls are unavailable.";
  const toggleMic = async () => {
    const call = callRef.current;
    if (!call || !connected) { setError(notConnectedNote); return; }
    try {
      setMicOn(await call.toggleMicrophone());
    } catch (e) {
      setError(`Could not change microphone: ${e instanceof Error ? e.message : String(e)}`);
    }
  };
  const toggleCam = async () => {
    const call = callRef.current;
    if (!call || !connected) { setError(notConnectedNote); return; }
    try {
      setCamOn(await call.toggleCamera());
    } catch (e) {
      setError(`Could not change camera: ${e instanceof Error ? e.message : String(e)}`);
    }
  };

  const leaveAndGoHome = useCallback(async (endRoom: boolean) => {
    if (leftRef.current || leaving) return;
    leftRef.current = true;
    setLeaving(true);
    stopSpeaking();
    saveCall({ roomId, startedAt: new Date(startedAtRef.current).toISOString(), endedAt: new Date().toISOString(), durationSec: elapsed, participants: participants.length });
    try {
      if (endRoom) await api(`/api/rooms/${encodeURIComponent(roomId)}/end`, { method: "POST", body: JSON.stringify({}) });
      else await api(`/api/rooms/${encodeURIComponent(roomId)}/leave`, { method: "POST", body: JSON.stringify({}) });
    } catch {
      // Leave/end are best-effort (the endpoint may not be deployed yet);
      // disconnecting locally still ends this user's call.
    }
    await callRef.current?.disconnect();
    callRef.current = null;
    router.push("/dashboard");
  }, [roomId, elapsed, participants.length, leaving, router]);

  const endCall = () => void leaveAndGoHome(false);
  const endRoomForEveryone = () => void leaveAndGoHome(true);

  const copyLink = async () => {
    await navigator.clipboard.writeText(`${window.location.origin}/room/${roomId}`);
    setCopied(true); setTimeout(() => setCopied(false), 1500);
  };
  const mm = `${String(Math.floor(elapsed / 60)).padStart(2, "0")}:${String(elapsed % 60).padStart(2, "0")}`;
  const user = getUser();
  const localInfo = participants.find((p) => p.isLocal);
  const remoteInfos = participants.filter((p) => !p.isLocal);
  // Panel = LiveKit presence, plus any server-known participants not yet
  // connected over LiveKit (shown as "joining…").
  const panelParticipants: ParticipantInfo[] = [
    ...participants,
    ...serverParticipants
      .filter((s) => !participants.some((p) => p.identity === s.identity))
      .map((s) => ({ identity: s.identity, name: s.name || s.identity, isLocal: false, isCameraEnabled: false, isMicrophoneEnabled: false, isSpeaking: false, hasVideo: false, hasAudio: false })),
  ];
  const totalTiles = 1 + remotes.length;
  const gridClass = totalTiles <= 2 ? "md:grid-cols-2" : "md:grid-cols-2 lg:grid-cols-3";
  const vocabulary = aiStatus?.sign?.vocabulary?.length ? aiStatus.sign.vocabulary : [...SIGN_VOCABULARY];
  const captionNote =
    captionEngine === "whisper"
      ? relayState === "connected"
        ? "Live captions by Whisper (server) from your microphone — sent to the remote participant over the caption relay."
        : "Live captions by Whisper (server) from your microphone — the caption relay is not connected, so they are shown only to you right now."
      : captionEngine === "browser"
        ? relayState === "connected"
          ? "Browser captions (not Whisper) from your microphone — sent to the remote participant over the caption relay."
          : "Browser captions (not Whisper) from your microphone — the caption relay is not connected, so they are shown only to you right now."
        : "Captions are generated from your microphone. When the caption relay is connected, the other participant's speech is captioned for you too.";

  if (roomMissing) {
    return (
      <div className="flex min-h-screen flex-col bg-ink">
        <Navbar />
        <main className="mx-auto flex w-full max-w-3xl flex-1 flex-col items-center justify-center px-4 text-center">
          <AlertTriangle size={40} className="text-amber-400" />
          <h1 className="mt-4 text-2xl font-bold">{roomMissing}</h1>
          <p className="mt-2 text-sm text-muted">The room may have been deleted, ended, or the link may be mistyped.</p>
          <Button className="mt-6" onClick={() => router.push("/dashboard")}>Back to dashboard</Button>
        </main>
      </div>
    );
  }

  return (
    <div className="flex min-h-screen flex-col bg-ink">
      <style>{`button:focus-visible, a:focus-visible, input:focus-visible { outline: 2px solid #22D3EE; outline-offset: 2px; }`}</style>
      <Navbar />
      {/* hidden video element for MediaPipe processing (LiveKit camera stream) */}
      <video ref={localVideoRef} className="hidden" playsInline muted />
      <main className="mx-auto flex w-full max-w-7xl flex-1 flex-col px-4 py-4 sm:px-6">
        <a href="#transcript" className="sr-only focus:not-sr-only focus:absolute focus:left-4 focus:top-4 focus:z-50 focus:rounded-lg focus:bg-cyan focus:px-3 focus:py-2 focus:text-sm focus:font-semibold focus:text-ink">Skip to transcript</a>
        <div className="mb-3 flex flex-wrap items-center justify-between gap-2">
          <div className="flex items-center gap-2">
            <h1 className="text-lg font-bold">Room {room?.code || roomId}</h1>
            <Badge tone={status === "Connected" ? "green" : status === "Failed" ? "red" : "slate"}><Wifi size={12} /> {status}</Badge>
            <Badge tone="slate"><Users size={12} /> {participants.length || 1} participant{(participants.length || 1) === 1 ? "" : "s"}</Badge>
            <span className="text-sm text-muted">{mm}</span>
            <Badge tone="cyan">LiveKit</Badge>
          </div>
          <div className="flex gap-2">
            {isCreator && (
              <Button variant="outline" size="sm" onClick={endRoomForEveryone} disabled={leaving}>End room for everyone</Button>
            )}
            <Button variant="outline" size="sm" onClick={copyLink}>{copied ? <><Check size={14} /> Link copied</> : <><Copy size={14} /> Copy invite link</>}</Button>
          </div>
        </div>
        <div className="mb-3">
          <AIStatusRow aiStatus={aiStatus} unavailableNote={aiStatusNote} />
        </div>
        {error && (
          <p role="alert" className="mb-3 flex items-start gap-2 rounded-xl border border-amber-500/40 bg-amber-500/10 px-3 py-2 text-sm text-amber-200">
            <AlertTriangle size={16} className="mt-0.5 shrink-0" /> {error}
          </p>
        )}
        {aiNote && (
          <p className="mb-3 flex items-start gap-2 rounded-xl border border-slate-800 bg-panel/70 px-3 py-2 text-sm text-muted">
            <AlertTriangle size={16} className="mt-0.5 shrink-0 text-amber-300" />
            <span>{aiNote}</span>
            <button className="ml-auto shrink-0 text-muted hover:text-white" onClick={() => setAiNote("")} aria-label="Dismiss AI note"><X size={14} /></button>
          </p>
        )}
        <div className={`grid flex-1 gap-4 ${panelOpen ? "lg:grid-cols-[1fr_340px]" : "lg:grid-cols-1"}`}>
          <div className="flex flex-col gap-3">
            <div className={`grid gap-3 ${gridClass}`}>
              <VideoTile
                videoTrack={localVideoTrack} stream={localStream}
                name={user?.name || "You"} label="You · Sign" muted mirrored
                micOn={micOn} camOn={camOn} active
                speaking={localInfo?.isSpeaking ?? false}
              />
              {remotes.length === 0 ? (
                <VideoTile name="Waiting for participant…" label="Remote" micOn={false} camOn={false} active={false} />
              ) : (
                remotes.map((r) => {
                  const info = remoteInfos.find((p) => p.identity === r.participant.identity);
                  // The voice agent is a remote participant: its audio
                  // gets the volume/mute controls from the AI panel.
                  const isVoiceAgent = r.participant.identity === "signspeak-voice";
                  return (
                    <VideoTile
                      key={r.participant.identity}
                      videoTrack={r.videoTrack}
                      audioTrack={r.audioTrack}
                      name={info?.name || r.participant.identity}
                      label={isVoiceAgent ? "Remote · Voice agent" : "Remote"}
                      micOn={info?.isMicrophoneEnabled ?? true}
                      camOn={info?.isCameraEnabled ?? true}
                      active={!!r.videoTrack}
                      speaking={info?.isSpeaking ?? false}
                      volume={isVoiceAgent ? voiceVolume : 1}
                      audioMuted={isVoiceAgent ? voiceMuted : false}
                    />
                  );
                })
              )}
            </div>
            {captionsOn && (
              <div className={settings.highContrast ? "rounded-xl border-2 border-white bg-black px-2 py-1" : ""}>
                <div aria-live="polite">
                  <CaptionBar text={caption} interim={interim} size={settings.captionSize} engine={captionEngine} confidence={captionConfidence} />
                </div>
                {remoteCaption && (
                  <div aria-live="polite" className="mx-auto mt-2 max-w-3xl rounded-xl border border-cyan/40 bg-cyan/10 px-4 py-2.5 text-center backdrop-blur">
                    <div className="mb-0.5 text-[11px] uppercase tracking-wider text-cyan">
                      Remote caption · {remoteCaption.speaker_name} · {new Date(remoteCaption.timestamp).toLocaleTimeString()}
                    </div>
                    <p className="font-medium leading-snug">{remoteCaption.text}</p>
                    {remoteCaption.confidence > 0 && (
                      <p className="mt-0.5 text-[11px] text-muted">{Math.round(remoteCaption.confidence * 100)}% confidence · {remoteCaption.engine === "browser" ? "Browser captions" : "Whisper tiny"}</p>
                    )}
                  </div>
                )}
                <p className="mt-1 text-center text-[11px] text-muted">{captionNote}</p>
                {relayState !== "connected" && (
                  <p className="mt-0.5 text-center text-[11px] text-amber-300">
                    {relayState === "reconnecting" ? "Caption relay reconnecting — remote captions may be delayed." : "Caption relay not connected — remote captions are unavailable right now."}
                  </p>
                )}
              </div>
            )}
            {pending && (
              <ConfidenceCard
                sign={pending.sign} confidence={pending.confidence} demo={pending.demo} mode={pending.mode} vocabulary={vocabulary}
                onAccept={() => acceptPrediction(pending)}
                onCorrect={correctPrediction}
                onRepeat={() => { setPending(null); smootherRef.current.reset(); bufferRef.current.clear(); }}
                onIgnore={() => setPending(null)}
              />
            )}
            {correctNote && (
              <p className="rounded-xl border border-slate-800 bg-panel/70 px-3 py-2 text-xs text-muted">{correctNote}</p>
            )}
            <div id="transcript" className="rounded-2xl border border-slate-800 bg-panel/70 p-3">
              <TranscriptPanel entries={entries} roomId={roomId} isCreator={isCreator} onSaveToServer={saveToServer} onClear={() => { setEntries([]); saveTranscript(roomId, []); }} />
            </div>
            <div
              aria-disabled={!connected}
              className={connected ? undefined : "pointer-events-none opacity-50 select-none"}
              ref={(el) => {
                // `inert` is not rendered by React 18, so set it
                // imperatively: it removes the disconnected controls from
                // keyboard tab order as well as from pointer input.
                if (!el) return;
                if (connected) el.removeAttribute("inert");
                else el.setAttribute("inert", "");
              }}
            >
              <CallControls
                micOn={micOn} camOn={camOn} captionsOn={captionsOn} panelOpen={panelOpen}
                onToggleMic={() => void toggleMic()} onToggleCam={() => void toggleCam()}
                onToggleCaptions={() => setCaptionsOn(v => !v)}
                onTogglePanel={() => setPanelOpen(v => !v)}
                onEnd={endCall}
              />
            </div>
            {!connected && (
              <p className="text-center text-xs text-amber-300">Microphone and camera controls are unavailable until you are connected to the call.</p>
            )}
            <p className="text-center text-xs text-muted">No video or audio is recorded. Captions are sent to the other participant over the caption relay when it is connected; demo predictions are always labelled Demo.</p>
          </div>
          {panelOpen && (
            <aside className="rounded-2xl border border-slate-800 bg-panel/80 p-4">
              {trainerOpen ? (
                <SignTrainer
                  latestLandmarksRef={latestLandmarksRef}
                  handVisibleRef={handVisibleRef}
                  onTrainingDone={() => void onTrainingDone()}
                  onClose={() => setTrainerOpen(false)}
                />
              ) : (
                <>
                  <div className="mb-3 flex items-center justify-between">
                    <h2 className="font-semibold">AI communication</h2>
                    <Button variant="ghost" size="sm" onClick={() => setPanelOpen(false)}><PhoneOff size={14} className="hidden" /> Hide</Button>
                  </div>
                  <div className="mb-4">
                    <h3 className="mb-2 flex items-center gap-1.5 text-sm font-semibold"><Users size={15} className="text-cyan" /> Participants ({panelParticipants.length})</h3>
                    {panelParticipants.length === 0 ? (
                      <p className="text-xs text-muted">Connecting…</p>
                    ) : (
                      <ul className="space-y-1.5">
                        {panelParticipants.map((p) => (
                          <li key={p.identity} className="flex items-center justify-between rounded-lg border border-slate-800 px-2.5 py-2 text-sm">
                            <span className="flex items-center gap-2 truncate">
                              <span className="truncate">{p.name}{p.isLocal ? " (you)" : ""}</span>
                              {p.isSpeaking && <Badge tone="green">Speaking</Badge>}
                            </span>
                            <span className="flex shrink-0 gap-1.5 text-muted">
                              {p.isMicrophoneEnabled ? <Mic size={14} className="text-success" /> : <MicOff size={14} className="text-red-400" />}
                              {p.isCameraEnabled ? <Video size={14} className="text-success" /> : <VideoOff size={14} className="text-red-400" />}
                            </span>
                          </li>
                        ))}
                      </ul>
                    )}
                  </div>
                  <AIPanel
                    signStatus={signStatus} speechStatus={speechStatus}
                    ttsMuted={settings.ttsMuted}
                    onToggleTts={() => { const next = !settings.ttsMuted; setSettings({ ...settings, ttsMuted: next }); settingsRef.current = { ...settings, ttsMuted: next }; if (next) stopSpeaking(); }}
                    onDemoSign={(s) => void demoSign(s)}
                    signEnabled={signEnabled} onToggleSign={() => setSignEnabled(v => !v)}
                    signTrained={signTrained} sampleCounts={sampleCounts} onOpenTrainer={() => setTrainerOpen(true)}
                    captionEngine={captionEngine}
                    voiceAgentOn={voiceAgentOn} onToggleVoiceAgent={() => void toggleVoiceAgent()}
                    sayText={sayText} onSayTextChange={setSayText} onSay={() => void sayThis()}
                    speaking={speaking} speakError={speakError}
                    ttsEngine={ttsEngine}
                    voiceVolume={voiceVolume} onVoiceVolumeChange={setVoiceVolume}
                    voiceMuted={voiceMuted} onToggleVoiceMute={() => setVoiceMuted(v => !v)}
                    relayState={relayState}
                  />
                  <div className="mt-4 rounded-xl border border-slate-800 bg-ink/60 p-3 text-xs text-muted">
                    <p className="mb-1 font-medium text-slate-300">Thresholds (from your settings)</p>
                    <p>Auto-accept ≥ {settings.signThresholdHigh}% · Confirm {settings.signThresholdLow}–{settings.signThresholdHigh}% · Repeat below {settings.signThresholdLow}%.</p>
                    <p className="mt-1">Change them in <button className="text-cyan underline" onClick={() => router.push("/dashboard")}>Dashboard → Accessibility settings</button>.</p>
                  </div>
                </>
              )}
            </aside>
          )}
        </div>
      </main>
    </div>
  );
}
