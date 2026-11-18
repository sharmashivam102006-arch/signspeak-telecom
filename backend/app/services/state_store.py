"""State store: process-local vs Redis-backed ephemeral state.

SignSpeak keeps some state outside the SQL database because it is
ephemeral and per-room/per-job, not durable records:

* per-room REST signal queues (``app/routes/rooms.py`` ``_signal_store``)
* training-job status dicts (``app/services/sign_training.py`` ``_jobs``)

Today both are plain module-level dicts: they are lost on restart and
are not shared across workers. This module provides one small store
abstraction with two backends:

* :class:`InMemoryStateStore` — default, current behaviour. Used when
  ``REDIS_URL`` is empty.
* :class:`RedisStateStore` — used when ``REDIS_URL`` is set
  (e.g. ``redis://localhost:6379/0``). State survives restarts and is
  shared across workers.

Values are plain strings. Callers serialise structured values (dicts)
to JSON themselves — see the wiring snippet in ``backend/docs/DATABASE.md``.

NOT stored here: voice-agent sessions. A voice agent holds a live
LiveKit RTC connection (``app/services/voice_agent.py`` ``_agents``),
which cannot be serialised or shared across processes. Only its status
metadata (a small JSON dict) could move to this store; the live agent
objects must stay in-process.

This module is deliberately NOT wired into any route or service yet —
integration is a separate, pending step (see ``backend/docs/DATABASE.md``).
Importing it has no side effects on the running app.
"""
import json
import threading
import time
from typing import Optional

# ---------------------------------------------------------------------------
# Key helpers — keep key naming in one place so the in-memory and Redis
# backends (and future callers) agree on it.
# ---------------------------------------------------------------------------

SIGNALS_PREFIX = "signals"
TRAIN_JOB_PREFIX = "train_job"


def signal_queue_key(room_id: str) -> str:
    """Redis/list key for a room's pending signal messages."""
    return f"{SIGNALS_PREFIX}:{room_id}"


def train_job_key(job_id: str) -> str:
    """Key for a training job's status dict (stored as a JSON string)."""
    return f"{TRAIN_JOB_PREFIX}:{job_id}"


class StateStore:
    """Minimal store interface shared by both backends.

    ``append``/``list``/``drain`` treat a key as an ordered list of
    strings (FIFO queue) — the shape the REST signal fallback needs.
    ``get``/``set`` treat a key as a single string value — the shape a
    training-job status dict (JSON-encoded) needs.
    """

    def get(self, key: str) -> Optional[str]:
        raise NotImplementedError

    def set(self, key: str, value: str, ttl: Optional[int] = None) -> None:
        """Store ``value`` under ``key``; ``ttl`` is seconds (None = no expiry)."""
        raise NotImplementedError

    def delete(self, key: str) -> bool:
        """Delete ``key``. Returns True if the key existed."""
        raise NotImplementedError

    def append(self, key: str, value: str, max_len: Optional[int] = None) -> int:
        """Append ``value`` to the list at ``key``; returns the new length.

        When ``max_len`` is given, only the newest ``max_len`` items are
        kept (oldest dropped) — mirrors the 100-message cap in rooms.py.
        """
        raise NotImplementedError

    def list(self, key: str, limit: Optional[int] = None) -> list:
        """Return list items oldest-first (up to ``limit`` if given)."""
        raise NotImplementedError

    def drain(self, key: str) -> list:
        """Return all list items oldest-first and remove the key.

        This is the read side of the signal queue: GET /signal returns
        the pending messages and clears the queue.
        """
        raise NotImplementedError


class InMemoryStateStore(StateStore):
    """Thread-safe in-process store. Default backend; current behaviour."""

    def __init__(self) -> None:
        self._values: dict = {}
        self._lists: dict = {}
        self._expiry: dict = {}  # key -> monotonic deadline
        self._lock = threading.RLock()

    def _expired(self, key: str) -> bool:
        deadline = self._expiry.get(key)
        if deadline is None:
            return False
        if time.monotonic() >= deadline:
            self._values.pop(key, None)
            self._lists.pop(key, None)
            self._expiry.pop(key, None)
            return True
        return False

    def get(self, key: str) -> Optional[str]:
        with self._lock:
            if self._expired(key):
                return None
            return self._values.get(key)

    def set(self, key: str, value: str, ttl: Optional[int] = None) -> None:
        with self._lock:
            self._values[key] = value
            self._lists.pop(key, None)  # a key is either a value or a list
            if ttl is not None:
                self._expiry[key] = time.monotonic() + ttl
            else:
                self._expiry.pop(key, None)

    def delete(self, key: str) -> bool:
        with self._lock:
            existed = key in self._values or key in self._lists
            self._values.pop(key, None)
            self._lists.pop(key, None)
            self._expiry.pop(key, None)
            return existed

    def append(self, key: str, value: str, max_len: Optional[int] = None) -> int:
        with self._lock:
            self._expired(key)  # clears the key if its TTL passed
            self._values.pop(key, None)  # a key is either a value or a list
            queue = self._lists.setdefault(key, [])
            queue.append(value)
            if max_len is not None and len(queue) > max_len:
                del queue[:-max_len]
            return len(queue)

    def list(self, key: str, limit: Optional[int] = None) -> list:
        with self._lock:
            if self._expired(key):
                return []
            items = list(self._lists.get(key, []))
            return items[:limit] if limit is not None else items

    def drain(self, key: str) -> list:
        with self._lock:
            if self._expired(key):
                return []
            return self._lists.pop(key, [])


class RedisStateStore(StateStore):
    """Redis-backed store. ``redis_url`` e.g. ``redis://localhost:6379/0``.

    The ``redis`` package is imported lazily so importing this module
    never fails on a machine without redis installed. The client
    connects lazily too; a bad/unreachable URL raises
    ``redis.exceptions.ConnectionError`` on first use, not at construction.
    """

    def __init__(self, redis_url: str) -> None:
        self.redis_url = redis_url
        self._client = None

    @property
    def client(self):
        if self._client is None:
            import redis

            self._client = redis.Redis.from_url(
                self.redis_url, decode_responses=True)
        return self._client

    def get(self, key: str) -> Optional[str]:
        return self.client.get(key)

    def set(self, key: str, value: str, ttl: Optional[int] = None) -> None:
        self.client.set(key, value, ex=ttl)

    def delete(self, key: str) -> bool:
        return bool(self.client.delete(key))

    def append(self, key: str, value: str, max_len: Optional[int] = None) -> int:
        pipe = self.client.pipeline()
        pipe.rpush(key, value)
        if max_len is not None:
            # Keep only the newest max_len items.
            pipe.ltrim(key, -max_len, -1)
        pipe.llen(key)
        _, _, length = pipe.execute()
        return int(length)

    def list(self, key: str, limit: Optional[int] = None) -> list:
        end = limit - 1 if limit is not None else -1
        return list(self.client.lrange(key, 0, end))

    def drain(self, key: str) -> list:
        pipe = self.client.pipeline()
        pipe.lrange(key, 0, -1)
        pipe.delete(key)
        items, _ = pipe.execute()
        return list(items)


# ---------------------------------------------------------------------------
# Factory / process-wide singleton
# ---------------------------------------------------------------------------

_store: Optional[StateStore] = None
_store_lock = threading.Lock()


def create_state_store(redis_url: Optional[str] = None) -> StateStore:
    """Build a store. ``redis_url`` None/empty -> in-memory (default).

    When ``redis_url`` is omitted the value is read from
    ``app.config.REDIS_URL`` at call time (not import time), so tests can
    monkeypatch the config.
    """
    if redis_url is None:
        from app.config import REDIS_URL

        redis_url = REDIS_URL
    redis_url = (redis_url or "").strip()
    if redis_url:
        return RedisStateStore(redis_url)
    return InMemoryStateStore()


def get_state_store() -> StateStore:
    """Process-wide store instance (created lazily, once)."""
    global _store
    if _store is None:
        with _store_lock:
            if _store is None:
                _store = create_state_store()
    return _store


def reset_state_store() -> None:
    """Drop the cached singleton (tests only)."""
    global _store
    with _store_lock:
        _store = None


def store_json(store: StateStore, key: str, value: dict,
               ttl: Optional[int] = None) -> None:
    """Convenience: store a dict as a JSON string."""
    store.set(key, json.dumps(value), ttl=ttl)


def load_json(store: StateStore, key: str) -> Optional[dict]:
    """Convenience: read a JSON-encoded dict, or None if absent/invalid."""
    raw = store.get(key)
    if raw is None:
        return None
    try:
        parsed = json.loads(raw)
    except (ValueError, TypeError):
        return None
    return parsed if isinstance(parsed, dict) else None
