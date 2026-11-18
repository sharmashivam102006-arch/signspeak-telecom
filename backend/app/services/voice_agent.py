"""LiveKit voice agent — "SignSpeak Voice" speaks TTS into the room.

The agent joins a LiveKit room as participant identity
``signspeak-voice`` (display name "SignSpeak Voice") using a token
minted by livekit_service, publishes a REAL audio track, and speaks
text by synthesising it with Piper and streaming 48 kHz PCM frames.

Threading model (deliberate): each agent owns one daemon thread
running its own asyncio event loop. The FastAPI process never shares
its event loop with the agent, and the agent never blocks FastAPI —
speak requests cross a thread-safe queue and completion is signalled
with a threading.Event. One agent per room, max; overlapping speaks
queue instead of overlapping audio.

Audio path: Piper synthesises a WAV at the voice's native sample rate
(22050 Hz for en_US-lessac-medium). LiveKit publishes 48 kHz PCM, so
the samples are resampled with numpy linear interpolation
(np.interp over the sample index) — documented here because it is a
real resample, not a rate label change. Frames are 10 ms (480 samples
at 48 kHz) and are paced with asyncio.sleep(0.01) so speech plays in
real time.

Nothing here is simulated: if LiveKit is unconfigured, the token
cannot be minted, Piper is missing, or the voice is not installed,
start()/speak() raise RuntimeError and the route answers HTTP 503.
"""
import asyncio
import contextlib
import logging
import os
import queue
import threading
import uuid
import wave

from app import config
from app.services import livekit_service, tts_service

logger = logging.getLogger(__name__)

# Proxy env vars the LiveKit Rust client honours when opening its signal
# connection. It does NOT honour NO_PROXY: in this sandbox it routes even
# ws://localhost:7880 through the egress proxy, which closes the
# connection ("Proxy connection closed while reading response") and the
# agent can never join. Neutralising them for the duration of the
# connect (only) lets the signal connection reach the local server.
_PROXY_ENV_VARS = (
    "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY",
    "http_proxy", "https_proxy", "all_proxy",
)


@contextlib.contextmanager
def _no_proxy_env():
    """Temporarily clear proxy env vars (and force NO_PROXY=*) around the
    LiveKit signal connect, then restore them. Scoped to the connect
    call only — the rest of the backend keeps its normal proxy config."""
    saved = {k: os.environ.pop(k) for k in _PROXY_ENV_VARS if k in os.environ}
    prev_no_proxy = os.environ.get("NO_PROXY")
    prev_no_proxy_lower = os.environ.get("no_proxy")
    os.environ["NO_PROXY"] = "*"
    os.environ["no_proxy"] = "*"
    try:
        yield
    finally:
        for k in _PROXY_ENV_VARS:
            if k in saved:
                os.environ[k] = saved[k]
            else:
                os.environ.pop(k, None)
        if prev_no_proxy is None:
            os.environ.pop("NO_PROXY", None)
        else:
            os.environ["NO_PROXY"] = prev_no_proxy
        if prev_no_proxy_lower is None:
            os.environ.pop("no_proxy", None)
        else:
            os.environ["no_proxy"] = prev_no_proxy_lower


def _loopback_url(url: str) -> str:
    """Prefer the 127.0.0.1 loopback address over the "localhost" hostname.

    The LiveKit Rust client sends "localhost" URLs to the configured
    egress proxy even when NO_PROXY lists localhost; the numeric
    loopback address is not proxied. Behaviour is identical on a normal
    network (both resolve to the same server)."""
    return url.replace("://localhost", "://127.0.0.1")

AGENT_IDENTITY = "signspeak-voice"
AGENT_NAME = "SignSpeak Voice"
TARGET_SAMPLE_RATE = 48000
FRAME_SAMPLES = 480  # 10 ms at 48 kHz

_agents: dict = {}
_agents_lock = threading.Lock()


def _resample_to_48k(pcm: bytes, src_rate: int) -> bytes:
    """Resample mono int16 PCM from src_rate to 48 kHz.

    Linear interpolation over the sample index (numpy.interp). Good
    enough for speech; documented as a real resample, not a relabel.
    """
    import numpy as np  # type: ignore

    if src_rate == TARGET_SAMPLE_RATE:
        return pcm
    x = np.frombuffer(pcm, dtype=np.int16).astype(np.float32)
    if len(x) == 0:
        return b""
    n_out = int(round(len(x) * TARGET_SAMPLE_RATE / src_rate))
    idx = np.linspace(0, len(x) - 1, n_out)
    y = np.interp(idx, np.arange(len(x)), x)
    return np.clip(y, -32768, 32767).astype(np.int16).tobytes()


def _read_wav_pcm(path: str):
    """Read a mono WAV file -> (pcm int16 bytes, sample_rate)."""
    with wave.open(path, "rb") as w:
        if w.getnchannels() != 1:
            raise ValueError("Expected a mono WAV file from Piper")
        if w.getsampwidth() != 2:
            raise ValueError("Expected 16-bit PCM WAV from Piper")
        return w.readframes(w.getnframes()), w.getframerate()


def _make_audio_frame(chunk: bytes):
    """Build a livekit-rtc AudioFrame from 10 ms of int16 mono PCM."""
    from livekit import rtc  # type: ignore

    samples = len(chunk) // 2
    try:
        return rtc.AudioFrame(chunk, TARGET_SAMPLE_RATE, 1, samples)
    except TypeError:
        frame = rtc.AudioFrame.create(TARGET_SAMPLE_RATE, 1, samples)
        frame.data[:] = chunk
        return frame


class VoiceAgent:
    """One agent for one room. Owns a thread + event loop + LiveKit Room."""

    def __init__(self, room_id: str, room_code: str):
        self.room_id = room_id
        self.room_code = room_code
        self.identity = AGENT_IDENTITY
        self._queue: asyncio.Queue | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._ready = threading.Event()
        self._stopped = threading.Event()
        self._error: str | None = None
        self._queue_depth = 0
        self._lock = threading.Lock()

    # -- lifecycle (called from FastAPI threads) ----------------------
    def start(self, timeout: float = 30.0) -> None:
        """Start the agent thread and wait until it has joined LiveKit.

        Raises RuntimeError on timeout or on join failure (LiveKit
        unconfigured, token failure, connection failure).
        """
        if not config.VOICE_AGENT_ENABLED:
            raise RuntimeError(
                "Voice agent is disabled (VOICE_AGENT_ENABLED=false).")
        if not livekit_service.livekit_configured():
            raise RuntimeError(
                "LiveKit is not configured, so the voice agent cannot join a room.")
        self._thread = threading.Thread(target=self._run, daemon=True,
                                        name=f"voice-agent-{self.room_id[:8]}")
        self._thread.start()
        if not self._ready.wait(timeout=timeout):
            raise RuntimeError("Voice agent did not join the LiveKit room in time.")
        if self._error:
            raise RuntimeError(f"Voice agent failed to join the room: {self._error}")

    def speak(self, text: str, timeout: float = 180.0) -> dict:
        """Queue text to be spoken; block until it finishes (or errors).

        Returns {"spoken": bool, "duration_s": float, "error": str|None}.
        Raises RuntimeError if the agent is not running.
        """
        if self._stopped.is_set() or self._loop is None or self._queue is None:
            raise RuntimeError("Voice agent is not running for this room.")
        done = threading.Event()
        result: dict = {"spoken": False, "duration_s": 0.0, "error": None}
        fut = asyncio.run_coroutine_threadsafe(
            self._queue.put((text, done, result)), self._loop)
        fut.result(timeout=10)
        with self._lock:
            self._queue_depth += 1
        if not done.wait(timeout=timeout):
            return {"spoken": False, "duration_s": 0.0,
                    "error": "Timed out waiting for the voice agent to speak."}
        return result

    def stop(self, timeout: float = 15.0) -> bool:
        """Ask the agent to leave LiveKit and shut down. Returns True if
        the agent thread exited cleanly."""
        if self._stopped.is_set():
            return True
        self._stopped.set()
        if self._loop is not None and self._queue is not None:
            try:
                fut = asyncio.run_coroutine_threadsafe(
                    self._queue.put(None), self._loop)
                fut.result(timeout=5)
            except Exception:
                pass
        if self._thread is not None:
            self._thread.join(timeout=timeout)
            return not self._thread.is_alive()
        return True

    def status(self) -> dict:
        return {
            "room_id": self.room_id,
            "identity": self.identity,
            "running": not self._stopped.is_set() and self._error is None,
            "queue_depth": self._queue_depth,
            "error": self._error,
        }

    # -- agent thread ---------------------------------------------------
    def _run(self):
        try:
            asyncio.run(self._main())
        except Exception as exc:  # join failure etc.
            self._error = str(exc)
            logger.exception("Voice agent for room %s failed: %s", self.room_id, exc)
        finally:
            self._ready.set()
            self._stopped.set()

    async def _main(self):
        from livekit import rtc  # type: ignore

        token = livekit_service.generate_token(
            self.room_code, identity=self.identity, name=AGENT_NAME)
        if token["mode"] != livekit_service.MODE_LIVEKIT or not token["token"]:
            raise RuntimeError(token.get("note") or "LiveKit is not configured.")
        self._loop = asyncio.get_running_loop()
        self._queue = asyncio.Queue()
        room = rtc.Room()
        # Connect with proxies neutralised and the loopback IP (see
        # _no_proxy_env / _loopback_url): without this the Rust client
        # routes ws://localhost:7880 to the egress proxy and the signal
        # connection fails before LiveKit is even reached. NOTE: this
        # fixes the *signal* connection only. The PeerConnection (ICE)
        # still needs working UDP; in a sandbox that blocks UDP the
        # connect fails later with "wait_pc_connection timed out" and
        # speak honestly answers 503 — remote audio delivery then stays
        # unverified until run on a normal network.
        url = _loopback_url(token["url"])
        try:
            with _no_proxy_env():
                await asyncio.wait_for(
                    room.connect(url, token["token"]), timeout=20)
        except asyncio.TimeoutError:
            # wait_for cut the client's internal retry loop short: the
            # signal connection and/or the media (ICE) setup did not
            # finish in 20s. Name the stage instead of surfacing the
            # bare "TimeoutError".
            raise RuntimeError(
                "Timed out (20s) connecting to the LiveKit room — the "
                "signal connection or the media (ICE) setup did not "
                "complete.") from None
        except Exception as exc:
            detail = str(exc) or type(exc).__name__
            raise RuntimeError(f"Could not connect to the LiveKit room: {detail}") from exc
        try:
            source = rtc.AudioSource(TARGET_SAMPLE_RATE, 1)
            track = rtc.LocalAudioTrack.create_audio_track(
                "signspeak-voice", source)
            await room.local_participant.publish_track(track)
        except Exception as exc:
            await room.disconnect()
            raise RuntimeError(f"Could not publish the voice audio track: {exc}") from exc
        self._ready.set()
        logger.info("Voice agent joined LiveKit room %s as %s",
                    self.room_code, self.identity)
        try:
            while True:
                item = await self._queue.get()
                if item is None:
                    break
                text, done, result = item
                with self._lock:
                    self._queue_depth = max(0, self._queue_depth - 1)
                try:
                    await self._speak_one(source, text, result)
                except Exception as exc:
                    result.update({"spoken": False, "duration_s": 0.0,
                                   "error": str(exc)})
                    logger.exception("Voice agent speak failed: %s", exc)
                finally:
                    done.set()
        finally:
            try:
                await room.disconnect()
            except Exception:
                pass
            logger.info("Voice agent left LiveKit room %s", self.room_code)

    async def _speak_one(self, source, text: str, result: dict) -> None:
        """Synthesise one text with Piper and stream it as 48 kHz frames."""
        synth = await tts_service.synthesize_async(text)
        pcm_path = tts_service.audio_path(
            synth["audio_url"].rsplit("/", 1)[-1])
        if pcm_path is None:
            raise RuntimeError("Synthesised audio file was not found on disk.")
        pcm, src_rate = await asyncio.to_thread(_read_wav_pcm, pcm_path)
        pcm48 = await asyncio.to_thread(_resample_to_48k, pcm, src_rate)
        total_samples = len(pcm48) // 2
        result["duration_s"] = round(total_samples / TARGET_SAMPLE_RATE, 2)
        frame_bytes = FRAME_SAMPLES * 2
        for off in range(0, len(pcm48), frame_bytes):
            if self._stopped.is_set():
                break
            chunk = pcm48[off:off + frame_bytes]
            if len(chunk) < frame_bytes:
                chunk = chunk + b"\x00" * (frame_bytes - len(chunk))
            frame = _make_audio_frame(chunk)
            await source.capture_frame(frame)
            await asyncio.sleep(0.01)
        result["spoken"] = True


# -- registry (sync helpers for the routes) ----------------------------

def start_agent(room_id: str, room_code: str) -> dict:
    """Start the voice agent for a room. 409-style RuntimeError if one is
    already running; RuntimeError with a clear reason otherwise."""
    with _agents_lock:
        existing = _agents.get(room_id)
        if existing is not None and not existing._stopped.is_set():
            raise RuntimeError("Voice agent is already running for this room.")
        agent = VoiceAgent(room_id, room_code)
        _agents[room_id] = agent
    try:
        agent.start()
    except Exception:
        with _agents_lock:
            _agents.pop(room_id, None)
        raise
    return agent.status()


def speak_in_room(room_id: str, room_code: str, text: str) -> dict:
    """Speak text in a room, auto-starting the agent if needed."""
    with _agents_lock:
        agent = _agents.get(room_id)
    if agent is None or agent._stopped.is_set():
        start_agent(room_id, room_code)
        with _agents_lock:
            agent = _agents[room_id]
    return agent.speak(text)


def stop_agent(room_id: str) -> bool:
    """Stop the agent for a room. True if one was running (and stopped)."""
    with _agents_lock:
        agent = _agents.pop(room_id, None)
    if agent is None:
        return False
    try:
        return agent.stop()
    except Exception:
        logger.exception("Voice agent stop failed for room %s", room_id)
        return False


def agent_status(room_id: str) -> dict:
    with _agents_lock:
        agent = _agents.get(room_id)
    if agent is None:
        return {"room_id": room_id, "identity": AGENT_IDENTITY,
                "running": False, "queue_depth": 0, "error": None}
    return agent.status()


def rooms_active() -> int:
    """Number of rooms with a live voice agent.

    Returns a count, not the room-id list: GET /api/ai/status exposes
    this as `rooms_active` and the frontend (and the API contract)
    treat it as a number. (Previously returned a list, which the
    frontend rendered as an empty string.)
    """
    with _agents_lock:
        return sum(1 for a in _agents.values() if not a._stopped.is_set())
