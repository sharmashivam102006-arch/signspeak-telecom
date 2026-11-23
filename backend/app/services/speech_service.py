"""Speech-to-text service — REAL faster-whisper ASR.

The Whisper model is a lazy singleton: loaded once on first use (never
at import/startup), guarded by an asyncio lock so concurrent
transcriptions serialise through the one model instance, and all
inference runs in a threadpool (transcribe_audio_bytes is sync; the
async wrapper runs it via asyncio.to_thread) so the FastAPI event loop
is never blocked by the model.

Model: faster-whisper, WHISPER_MODEL (default tiny), int8, CPU, loaded
from WHISPER_MODEL_DIR (backend/models/whisper-<model>). Download with:
    python -m app.services.speech_service --download
(no network at request time — if the model directory is missing or the
model cannot load, transcribe_audio_bytes raises RuntimeError and the
route answers HTTP 503; it never invents a transcript).

Confidence: 0-1 derived from the average log-probability of the
segments (avg_logprob is <= 0; confidence = exp(avg_logprob) clamped
to [0, 1]). Empty / no-speech audio returns text "" with low
confidence — never invented words.

Client-supplied text is NEVER labelled Whisper: transcribe_text()
returns engine "client-text-passthrough" with a note that the text came
from the client (e.g. browser Web Speech API).
"""
import asyncio
import io
import logging
import math
import os
import threading

from app import config

logger = logging.getLogger(__name__)

ENGINE_NAME = f"whisper-{config.WHISPER_MODEL}"

_model = None
_model_lock = threading.Lock()
_model_dir = None


def whisper_available() -> bool:
    """faster-whisper importable AND a model directory present."""
    try:
        import faster_whisper  # type: ignore  # noqa: F401
    except ImportError:
        return False
    return bool(_model_path())


def _model_path() -> str | None:
    """Directory holding the ctranslate2 model, if it exists on disk."""
    d = os.path.abspath(config.WHISPER_MODEL_DIR)
    if os.path.isdir(d) and any(os.scandir(d)):
        return d
    return None


def _load_model():
    """Load (once) and return the faster-whisper model, or raise."""
    global _model, _model_dir
    with _model_lock:
        if _model is not None:
            return _model
        try:
            from faster_whisper import WhisperModel  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "faster-whisper is not installed. Install it "
                "(pip install faster-whisper) to enable server-side Whisper ASR.") from exc
        path = _model_path()
        if path is None:
            raise RuntimeError(
                f"Whisper model '{config.WHISPER_MODEL}' is not downloaded. "
                f"Expected a ctranslate2 model in {os.path.abspath(config.WHISPER_MODEL_DIR)}. "
                "Download it with: python -m app.services.speech_service --download")
        threads = cpu_threads()
        try:
            _model = WhisperModel(path, device="cpu", compute_type="int8",
                                  cpu_threads=threads, num_workers=1)
        except Exception as exc:
            raise RuntimeError(f"Whisper model at {path} could not be loaded: {exc}") from exc
        _model_dir = path
        logger.info("Loaded faster-whisper model from %s (cpu_threads=%d, beam_size=%d)",
                    path, threads, config.WHISPER_BEAM_SIZE)
        return _model


def model_loaded() -> bool:
    return _model is not None


def load_model() -> None:
    """Load the model now (e.g. when a caption socket opens) so the first
    utterance is not delayed by model loading. Raises RuntimeError."""
    _load_model()


def cpu_threads() -> int:
    """Threads for ctranslate2: WHISPER_CPU_THREADS, or (0 = auto) the
    container CPU quota from cgroup v2 cpu.max / v1 cfs files, else
    os.cpu_count(); always 1..4. os.cpu_count() alone reports the *host*
    cores, which on a quota-limited container oversubscribes the CPU."""
    if config.WHISPER_CPU_THREADS > 0:
        return config.WHISPER_CPU_THREADS
    quota = None
    try:
        with open("/sys/fs/cgroup/cpu.max") as f:
            q, period = f.read().split()[:2]
            if q != "max":
                quota = int(q) / int(period)
    except (OSError, ValueError):
        try:
            with open("/sys/fs/cgroup/cpu/cpu.cfs_quota_us") as f:
                q = int(f.read())
            with open("/sys/fs/cgroup/cpu/cpu.cfs_period_us") as f:
                period = int(f.read())
            if q > 0 and period > 0:
                quota = q / period
        except (OSError, ValueError):
            pass
    n = math.ceil(quota) if quota else (os.cpu_count() or 1)
    return max(1, min(4, n))


def release_model() -> None:
    """Drop the cached model (tests / low-memory teardown)."""
    global _model, _model_dir
    with _model_lock:
        _model = None
        _model_dir = None


def _confidence(avg_logprob: float | None) -> float:
    """avg_logprob (<= 0) -> 0..1 confidence. None -> 0.0."""
    if avg_logprob is None:
        return 0.0
    try:
        return round(min(1.0, max(0.0, math.exp(float(avg_logprob)))), 3)
    except (TypeError, ValueError, OverflowError):
        return 0.0


def _result(segments, info, language: str, drop_no_speech: bool = False) -> dict:
    out_segments = []
    total_logprob, total_dur = 0.0, 0.0
    text_parts = []
    for seg in segments:
        avg_logprob = getattr(seg, "avg_logprob", None)
        # Whisper "hears" words in noise/silence; drop segments it itself
        # marks as probably-not-speech with low confidence (the standard
        # faster-whisper/openai-whisper no-speech rule).
        if drop_no_speech and getattr(seg, "no_speech_prob", 0.0) > 0.6                 and (avg_logprob is None or avg_logprob < -1.0):
            continue
        dur = max(seg.end - seg.start, 0.0)
        conf = _confidence(avg_logprob)
        out_segments.append({
            "start": round(seg.start, 2), "end": round(seg.end, 2),
            "text": seg.text.strip(), "confidence": conf,
        })
        text_parts.append(seg.text.strip())
        if avg_logprob is not None:
            total_logprob += avg_logprob * dur
            total_dur += dur
    text = " ".join(p for p in text_parts if p).strip()
    confidence = _confidence(total_logprob / total_dur) if total_dur > 0 else 0.0
    return {
        "text": text,
        "language": info.language or language,
        "engine": ENGINE_NAME,
        "confidence": confidence,
        "segments": out_segments,
        "is_final": True,
    }


def transcribe_audio_bytes(audio_bytes: bytes, language: str = "en") -> dict:
    """Transcribe raw audio bytes with faster-whisper (sync; run via
    transcribe_audio_bytes_async from async code).

    Accepts wav / webm / opus / mp3 — faster-whisper decodes via av.
    Returns {text, language, engine, confidence, segments, is_final}.
    Raises RuntimeError when the model is unavailable.
    """
    if not audio_bytes:
        raise ValueError("Empty audio")
    model = _load_model()
    segments, info = model.transcribe(
        io.BytesIO(audio_bytes), language=language or None,
        beam_size=config.WHISPER_BEAM_SIZE,
        vad_filter=True, vad_parameters={"min_silence_duration_ms": 300})
    return _result(segments, info, language)


def transcribe_pcm(samples, language: str = "en") -> dict:
    """Transcribe one live-caption utterance: float32 mono samples at
    16 kHz in [-1, 1] (already cut at pauses by caption_stream, so no
    VAD pass and no decoding of a container format). Greedy decoding,
    no timestamps and no conditioning on earlier text keep each call
    short and independent. Raises RuntimeError when the model is
    unavailable."""
    if samples is None or len(samples) == 0:
        raise ValueError("Empty audio")
    model = _load_model()
    segments, info = model.transcribe(
        samples, language=language or None,
        beam_size=config.WHISPER_BEAM_SIZE,
        condition_on_previous_text=False, without_timestamps=True,
        vad_filter=False)
    return _result(segments, info, language, drop_no_speech=True)


async def transcribe_pcm_async(samples, language: str = "en") -> dict:
    return await asyncio.to_thread(transcribe_pcm, samples, language)


async def transcribe_audio_bytes_async(audio_bytes: bytes, language: str = "en") -> dict:
    """Async wrapper: Whisper inference runs in a threadpool."""
    return await asyncio.to_thread(transcribe_audio_bytes, audio_bytes, language)


def transcribe_text(text: str, language: str = "en") -> dict:
    """Client-supplied text (e.g. browser Web Speech API).

    Labelled honestly: engine "client-text-passthrough" — this text was
    NOT produced by Whisper on the server.
    """
    return {
        "text": text.strip(), "language": language,
        "engine": "client-text-passthrough", "is_final": True,
        "note": "Text was supplied by the client (e.g. browser Web Speech API); "
                "it was not transcribed by Whisper on the server.",
    }


def download_model() -> str:
    """Download the faster-whisper model from Hugging Face into
    WHISPER_MODEL_DIR. Returns the directory. Requires network + the
    huggingface_hub package (a faster-whisper dependency)."""
    from huggingface_hub import snapshot_download  # type: ignore

    repo = f"Systran/faster-whisper-{config.WHISPER_MODEL}"
    dest = os.path.abspath(config.WHISPER_MODEL_DIR)
    os.makedirs(dest, exist_ok=True)
    snapshot_download(repo_id=repo, local_dir=dest,
                      allow_patterns=["*.json", "model.bin", "vocabulary.*",
                                      "tokenizer.json", "preprocessor_config.json"])
    return dest


if __name__ == "__main__":
    import sys
    if "--download" in sys.argv:
        print("Downloading", f"Systran/faster-whisper-{config.WHISPER_MODEL}",
              "->", os.path.abspath(config.WHISPER_MODEL_DIR))
        print("Saved to", download_model())
    else:
        print("Whisper model dir:", os.path.abspath(config.WHISPER_MODEL_DIR))
        print("Available:", whisper_available())
