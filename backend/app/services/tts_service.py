"""Text-to-speech service — REAL Piper neural TTS (local, no API key).

Piper voices are lazy singletons: a voice (.onnx + .onnx.json) is
loaded once on first use, guarded by a lock, and all synthesis runs in
a threadpool (synthesize_async) so the FastAPI event loop is never
blocked. Synthesis writes a real WAV file under TTS_AUDIO_DIR and
returns its /api/ai/tts/audio/<id> URL.

Voice files live in PIPER_VOICE_DIR (backend/models/piper/). Download
the default voice with:
    python -m app.services.tts_service --download
(no network at request time — if Piper or the voice is unavailable,
synthesize() raises RuntimeError and the route answers HTTP 503; it
never pretends audio was produced).
"""
import asyncio
import io
import logging
import os
import threading
import uuid
import wave

from app import config

logger = logging.getLogger(__name__)

_voices: dict = {}
_voices_lock = threading.Lock()


def piper_available() -> bool:
    try:
        import piper  # type: ignore  # noqa: F401
        return True
    except ImportError:
        return False


def _voice_files(voice_id: str):
    """(onnx_path, config_path) for a voice id, or None if not installed."""
    base = os.path.join(os.path.abspath(config.PIPER_VOICE_DIR), voice_id)
    onnx = base + ".onnx"
    cfg = base + ".onnx.json"
    if os.path.isfile(onnx) and os.path.isfile(cfg):
        return onnx, cfg
    return None


def _parse_voice_id(filename: str):
    """en_US-lessac-medium.onnx -> {id, name, language}."""
    voice_id = filename[:-5]  # strip .onnx
    parts = voice_id.split("-", 1)
    lang = parts[0].replace("_", "-") if parts else "en"
    return {"id": voice_id, "name": voice_id, "language": lang}


def list_voices() -> list:
    """Piper voices actually installed on disk (never a hardcoded list)."""
    d = os.path.abspath(config.PIPER_VOICE_DIR)
    if not os.path.isdir(d):
        return []
    return [_parse_voice_id(fn) for fn in sorted(os.listdir(d))
            if fn.endswith(".onnx")]


def voices_available() -> bool:
    return bool(list_voices())


def _load_voice(voice_id: str):
    """Load (once) and return the PiperVoice for voice_id, or raise."""
    with _voices_lock:
        if voice_id in _voices:
            return _voices[voice_id]
        if not piper_available():
            raise RuntimeError(
                "Piper TTS is not installed. Install it (pip install piper-tts) "
                "and download a voice to enable server-side TTS.")
        files = _voice_files(voice_id)
        if files is None:
            raise RuntimeError(
                f"Piper voice '{voice_id}' is not installed in "
                f"{os.path.abspath(config.PIPER_VOICE_DIR)}. Download it with: "
                "python -m app.services.tts_service --download "
                f"[--voice {voice_id}]")
        from piper import PiperVoice  # type: ignore

        try:
            voice = PiperVoice.load(files[0], config_path=files[1])
        except Exception as exc:
            raise RuntimeError(f"Piper voice '{voice_id}' could not be loaded: {exc}") from exc
        _voices[voice_id] = voice
        logger.info("Loaded Piper voice %s", voice_id)
        return voice


def release_voices() -> None:
    """Drop cached voices (tests / low-memory teardown)."""
    with _voices_lock:
        _voices.clear()


def synthesize(text: str, voice: str | None = None, language: str = "en") -> dict:
    """Synthesise text to a real WAV file. Returns
    {text, audio_url, engine, voice, duration_s}.

    Raises RuntimeError when Piper/voice is unavailable; ValueError for
    invalid input (empty text, > TTS_MAX_CHARS).
    """
    if not text or not text.strip():
        raise ValueError("Text is empty")
    text = text.strip()
    if len(text) > config.TTS_MAX_CHARS:
        raise ValueError(f"Text exceeds the {config.TTS_MAX_CHARS}-character limit "
                         f"({len(text)} characters)")
    voice_id = (voice or config.PIPER_VOICE or "en_US-lessac-medium").strip()
    piper_voice = _load_voice(voice_id)
    audio_id = uuid.uuid4().hex
    out_dir = os.path.abspath(config.TTS_AUDIO_DIR)
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{audio_id}.wav")
    with _voices_lock:
        with wave.open(path, "wb") as wav_file:
            wav_file.setframerate(piper_voice.config.sample_rate)
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            piper_voice.synthesize_wav(text, wav_file)
    with wave.open(path, "rb") as wav_file:
        frames = wav_file.getnframes()
        rate = wav_file.getframerate()
    duration_s = round(frames / rate, 2) if rate else 0.0
    return {
        "text": text,
        "audio_url": f"/api/ai/tts/audio/{audio_id}",
        "engine": f"piper-{voice_id}",
        "voice": voice_id,
        "duration_s": duration_s,
    }


async def synthesize_async(text: str, voice: str | None = None,
                           language: str = "en") -> dict:
    """Async wrapper: Piper synthesis runs in a threadpool."""
    return await asyncio.to_thread(synthesize, text, voice, language)


def audio_path(audio_id: str):
    """Filesystem path for a served WAV id, or None (unknown id / unsafe)."""
    if not audio_id or not all(c in "0123456789abcdef" for c in audio_id.lower()):
        return None
    path = os.path.join(os.path.abspath(config.TTS_AUDIO_DIR), f"{audio_id}.wav")
    return path if os.path.isfile(path) else None


def download_voice(voice_id: str | None = None) -> str:
    """Download a Piper voice (.onnx + .onnx.json) from Hugging Face
    (rhasspy/piper-voices) into PIPER_VOICE_DIR. Returns the .onnx path."""
    from huggingface_hub import hf_hub_download  # type: ignore

    voice_id = (voice_id or config.PIPER_VOICE or "en_US-lessac-medium").strip()
    # en_US-lessac-medium -> lang en, region en_US, name lessac, quality medium
    lang_region, _, rest = voice_id.partition("-")
    name, _, quality = rest.partition("-")
    lang = lang_region.split("_")[0]
    repo_path = f"{lang}/{lang_region}/{name}/{quality}/{voice_id}"
    dest = os.path.abspath(config.PIPER_VOICE_DIR)
    os.makedirs(dest, exist_ok=True)
    # hf_hub_download preserves the repo path under local_dir; flatten
    # the files to <dest>/<voice_id>.onnx[.json] afterwards so
    # _voice_files()/list_voices() find them.
    import shutil

    for suffix in (".onnx", ".onnx.json"):
        downloaded = hf_hub_download(repo_id="rhasspy/piper-voices",
                                     filename=repo_path + suffix, local_dir=dest,
                                     local_dir_use_symlinks=False)
        flat = os.path.join(dest, voice_id + suffix)
        if os.path.abspath(downloaded) != os.path.abspath(flat):
            shutil.move(downloaded, flat)
    # Remove the now-empty nested repo path (and any hf cache dir).
    nested = os.path.join(dest, lang)
    if os.path.isdir(nested):
        shutil.rmtree(nested, ignore_errors=True)
    cache_dir = os.path.join(dest, ".cache")
    if os.path.isdir(cache_dir):
        shutil.rmtree(cache_dir, ignore_errors=True)
    return os.path.join(dest, voice_id + ".onnx")


if __name__ == "__main__":
    import sys
    if "--download" in sys.argv:
        vid = None
        if "--voice" in sys.argv:
            vid = sys.argv[sys.argv.index("--voice") + 1]
        print("Downloading Piper voice ->", download_voice(vid))
    else:
        print("Piper installed:", piper_available())
        print("Installed voices:", list_voices())
