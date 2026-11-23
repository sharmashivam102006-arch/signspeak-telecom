"""Unit tests for app/services/caption_stream.py — utterance segmentation
and the caption worker's lag handling. No Whisper model needed: the
transcriber is a fake coroutine."""
import asyncio
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.services.caption_stream import (  # noqa: E402
    SAMPLE_RATE, CaptionSession, SpeechSegmenter, parse_config)


def pcm(seconds: float, amp: float = 0.0, freq: float = 220.0) -> bytes:
    """PCM16 bytes: a tone of the given amplitude (0 = silence) plus a
    little noise, like a real mic."""
    n = int(seconds * SAMPLE_RATE)
    t = np.arange(n) / SAMPLE_RATE
    rng = np.random.default_rng(0)
    x = amp * np.sin(2 * np.pi * freq * t) + rng.normal(0, 0.001, n)
    return (np.clip(x, -1, 1) * 32767).astype("<i2").tobytes()


def feed(seg: SpeechSegmenter, data: bytes, chunk: int = 8000) -> list:
    out = []
    for i in range(0, len(data), chunk):
        out += seg.push(data[i:i + chunk])
    return out


def test_silence_produces_nothing():
    s = SpeechSegmenter()
    assert feed(s, pcm(5.0)) == []
    assert s.flush() is None


def test_two_utterances_split_at_pause():
    s = SpeechSegmenter()
    audio = pcm(0.5) + pcm(1.5, 0.3) + pcm(1.0) + pcm(2.0, 0.3) + pcm(1.0)
    segs = feed(s, audio)
    assert len(segs) == 2
    # Each utterance ~ speech + pre-roll + the trailing pause that ended it.
    assert 1.5 <= len(segs[0]) / SAMPLE_RATE <= 2.6
    assert 2.0 <= len(segs[1]) / SAMPLE_RATE <= 3.1


def test_odd_chunk_sizes_are_reassembled():
    s1, s2 = SpeechSegmenter(), SpeechSegmenter()
    audio = pcm(0.5) + pcm(1.5, 0.3) + pcm(1.0)
    a = feed(s1, audio, chunk=8000)
    b = feed(s2, audio, chunk=777)  # odd byte count, splits samples
    assert len(a) == len(b) == 1
    assert abs(len(a[0]) - len(b[0])) <= 320


def test_long_speech_is_capped():
    s = SpeechSegmenter(max_segment_s=4.0)
    segs = feed(s, pcm(10.0, 0.3))
    # Cut at the quietest frame of the last 1.5 s before each 4 s cap.
    assert 2 <= len(segs) <= 3
    for seg in segs:
        assert 2.5 - 1e-6 <= len(seg) / SAMPLE_RATE <= 4.0 + 1e-6
    rest = s.flush()
    assert rest is not None and len(rest) > 0
    total = sum(len(x) for x in segs) + len(rest)
    assert abs(total / SAMPLE_RATE - 10.0) < 0.5  # nothing lost or duplicated


def test_flush_ends_open_utterance_and_tiny_blips_are_dropped():
    s = SpeechSegmenter()
    feed(s, pcm(1.0, 0.3))
    seg = s.flush()
    assert seg is not None and 0.9 <= len(seg) / SAMPLE_RATE <= 1.4
    s2 = SpeechSegmenter()
    feed(s2, pcm(0.5) + pcm(0.06, 0.3) + pcm(1.0))  # 60 ms click
    assert s2.flush() is None


def test_parse_config():
    assert parse_config({"format": "pcm16", "sample_rate": 16000}) is None
    assert parse_config({"format": "webm"})
    assert parse_config({"format": "pcm16", "sample_rate": 48000})


class FakeClock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


def run(coro):
    return asyncio.run(coro)


def test_session_sends_captions_in_order():
    async def main():
        sent = []

        async def send(m):
            sent.append(m)

        async def transcribe(seg, lang):
            return {"text": f"utt {len(sent)}", "confidence": 0.9, "engine": "whisper-tiny"}

        s = CaptionSession(send, transcribe)
        await s.push_audio(pcm(0.5) + pcm(1.0, 0.3) + pcm(1.0) + pcm(1.0, 0.3) + pcm(1.0))
        for _ in range(20):
            await asyncio.sleep(0)
        await s.close()
        return sent

    sent = run(main())
    caps = [m for m in sent if m["type"] == "caption"]
    assert [c["text"] for c in caps] == ["utt 0", "utt 1"]
    assert all(c["is_final"] and c["engine"] == "whisper-tiny" for c in caps)


def test_session_reports_too_slow_once():
    async def main():
        sent = []
        clock = FakeClock()
        gate = asyncio.Event()

        async def send(m):
            sent.append(m)

        async def transcribe(seg, lang):
            await gate.wait()  # server "busy" until released
            return {"text": "late", "confidence": 0.5}

        s = CaptionSession(send, transcribe, max_lag_s=12.0, clock=clock)
        await s.push_audio(pcm(0.5) + pcm(1.0, 0.3) + pcm(1.0))  # one utterance queued
        await asyncio.sleep(0)
        clock.t = 5.0
        await s.push_audio(pcm(0.25))
        assert not s.failed
        clock.t = 20.0
        await s.push_audio(pcm(0.25))
        await s.push_audio(pcm(0.25))
        assert s.failed
        gate.set()
        for _ in range(10):
            await asyncio.sleep(0)
        await s.close()
        return sent

    sent = run(main())
    errors = [m for m in sent if m["type"] == "error"]
    assert len(errors) == 1 and errors[0]["code"] == "too-slow"
    assert not [m for m in sent if m["type"] == "caption"]  # nothing after failing


def test_session_waits_for_model_then_times_out():
    async def main():
        sent = []
        clock = FakeClock()

        async def send(m):
            sent.append(m)

        async def transcribe(seg, lang):
            await asyncio.sleep(3600)

        s = CaptionSession(send, transcribe, model_ready=lambda: False,
                           load_timeout_s=45.0, clock=clock)
        clock.t = 30.0
        await s.push_audio(pcm(0.25))
        assert not s.failed  # still within the load timeout
        clock.t = 50.0
        await s.push_audio(pcm(0.25))
        await s.close()
        return s, sent

    s, sent = run(main())
    assert s.failed and sent[-1]["code"] == "too-slow"


def test_session_transcriber_error_is_reported():
    async def main():
        sent = []

        async def send(m):
            sent.append(m)

        async def transcribe(seg, lang):
            raise RuntimeError("Whisper model is not downloaded")

        s = CaptionSession(send, transcribe)
        await s.push_audio(pcm(0.5) + pcm(1.0, 0.3) + pcm(1.0))
        for _ in range(10):
            await asyncio.sleep(0)
        await s.close()
        return sent

    sent = run(main())
    assert sent[-1]["type"] == "error" and sent[-1]["code"] == "whisper-error"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
