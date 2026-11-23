/**
 * Speech -> text for the hearing participant.
 *
 * Default is the browser's Web Speech API: genuinely streaming, free, no
 * infrastructure. Whisper is the fallback, and the spec's plan to reach for it
 * first is worth reconsidering - Whisper is not a streaming model. Every
 * "streaming Whisper" re-runs the model on a growing buffer, so text is
 * unstable (words change after they appear) and cost grows with utterance
 * length. Unstable captions are a specific problem here: the person reading
 * them cannot hear the audio to check what changed.
 *
 * Known limits of Web Speech, to state in the report rather than discover in
 * the viva:
 *   - Chrome and Edge send audio to Google's servers. That breaks the
 *     "nothing leaves the device" story for the SPEECH direction, though not
 *     for signs. Say so plainly; do not let the privacy claim quietly cover
 *     both.
 *   - Firefox has no support. The Whisper path exists for that.
 *   - It stops on its own after a pause and needs restarting, which is what
 *     the restart logic below is for.
 */

export interface AsrHypothesis {
  text: string;
  isFinal: boolean;
  startedAtMs: number;
  endedAtMs: number;
  utteranceId: string;
  confidence?: number;
}

export interface AsrOptions {
  lang?: string;
  onHypothesis: (h: AsrHypothesis) => void;
  onError?: (e: string) => void;
  onStateChange?: (listening: boolean) => void;
}

type SR = any;

function getSpeechRecognition(): SR | null {
  if (typeof window === "undefined") return null;
  return (window as any).SpeechRecognition ?? (window as any).webkitSpeechRecognition ?? null;
}

export function isWebSpeechSupported(): boolean {
  return getSpeechRecognition() !== null;
}

export class WebSpeechAsr {
  private rec: any = null;
  private running = false;
  private utteranceId = "";
  private startedAt = 0;
  private restartTimer: any = null;

  constructor(private opts: AsrOptions) {}

  start() {
    const Ctor = getSpeechRecognition();
    if (!Ctor) {
      this.opts.onError?.(
        "This browser has no speech recognition. Use Chrome or Edge, or enable the Whisper fallback.",
      );
      return;
    }
    if (this.running) return;

    const rec = new Ctor();
    rec.lang = this.opts.lang ?? "en-IN";
    rec.continuous = true;
    rec.interimResults = true;      // partials are the point; without them captions lag by a sentence
    rec.maxAlternatives = 1;

    rec.onstart = () => {
      this.running = true;
      this.opts.onStateChange?.(true);
    };

    rec.onresult = (event: any) => {
      for (let i = event.resultIndex; i < event.results.length; i++) {
        const result = event.results[i];
        const alt = result[0];
        if (!this.utteranceId) {
          this.utteranceId = `u${Date.now()}`;
          this.startedAt = performance.now();
        }
        this.opts.onHypothesis({
          text: alt.transcript.trim(),
          isFinal: result.isFinal,
          startedAtMs: this.startedAt,
          endedAtMs: performance.now(),
          utteranceId: this.utteranceId,
          // Present on finals in Chrome, absent on partials. Not comparable
          // with the sign classifier's calibrated confidence - different
          // model, different scale - so never render them side by side as if
          // they were.
          confidence: result.isFinal ? alt.confidence : undefined,
        });
        if (result.isFinal) this.utteranceId = "";
      }
    };

    rec.onerror = (e: any) => {
      // "no-speech" and "aborted" are routine, not failures.
      if (e.error !== "no-speech" && e.error !== "aborted") {
        this.opts.onError?.(String(e.error));
      }
    };

    rec.onend = () => {
      this.running = false;
      this.opts.onStateChange?.(false);
      // The API stops itself after silence. Restart with a short delay, or a
      // failing mic turns into a busy loop.
      if (this.rec === rec) {
        this.restartTimer = setTimeout(() => this.start(), 250);
      }
    };

    this.rec = rec;
    try {
      rec.start();
    } catch {
      /* already started */
    }
  }

  stop() {
    if (this.restartTimer) clearTimeout(this.restartTimer);
    const rec = this.rec;
    this.rec = null;
    this.running = false;
    try {
      rec?.stop();
    } catch {
      /* not running */
    }
    this.opts.onStateChange?.(false);
  }

  get listening() {
    return this.running;
  }
}

/**
 * Server Whisper fallback: streams 16kHz PCM over a WebSocket.
 *
 * Used when Web Speech is unavailable, when the target language is not well
 * supported by the browser (Hindi, for the ISL direction), or when sending
 * audio to a third party is not acceptable for the deployment. Downsampling
 * happens here rather than server-side to cut bandwidth ~3x.
 */
export class WhisperAsr {
  private ws: WebSocket | null = null;
  private ctx: AudioContext | null = null;
  private node: ScriptProcessorNode | AudioWorkletNode | null = null;

  constructor(private url: string, private opts: AsrOptions) {}

  async start(stream: MediaStream) {
    this.ws = new WebSocket(this.url);
    this.ws.binaryType = "arraybuffer";
    this.ws.onmessage = (ev) => {
      const msg = JSON.parse(ev.data);
      if (msg.type === "hypothesis") this.opts.onHypothesis(msg.data);
    };
    this.ws.onerror = () => this.opts.onError?.("ASR socket failed");

    this.ctx = new AudioContext({ sampleRate: 16000 });
    const src = this.ctx.createMediaStreamSource(stream);
    const proc = this.ctx.createScriptProcessor(4096, 1, 1);
    proc.onaudioprocess = (e) => {
      if (this.ws?.readyState !== WebSocket.OPEN) return;
      const f32 = e.inputBuffer.getChannelData(0);
      const i16 = new Int16Array(f32.length);
      for (let i = 0; i < f32.length; i++) {
        i16[i] = Math.max(-1, Math.min(1, f32[i])) * 0x7fff;
      }
      this.ws.send(i16.buffer);
    };
    src.connect(proc);
    proc.connect(this.ctx.destination);
    this.node = proc;
    this.opts.onStateChange?.(true);
  }

  stop() {
    this.node?.disconnect();
    this.ctx?.close();
    this.ws?.close();
    this.ws = null;
    this.opts.onStateChange?.(false);
  }
}
