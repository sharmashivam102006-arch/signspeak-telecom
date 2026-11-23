"""Tests for app.services.state_store.

The in-memory backend is tested fully. The Redis backend is tested
only when a Redis server is actually reachable (``$REDIS_URL`` or
``redis://localhost:6379/0``); otherwise those tests are SKIPPED with an
explicit reason — never faked. Run: python -m pytest tests/test_state_store.py -v
"""
import os
import sys
import threading
import time
import uuid

os.environ.setdefault("DATABASE_URL", "sqlite:///./test_signspeak.db")
os.environ.setdefault("SECRET_KEY", "test-secret")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

from app.services.state_store import (
    InMemoryStateStore,
    RedisStateStore,
    create_state_store,
    get_state_store,
    load_json,
    reset_state_store,
    signal_queue_key,
    store_json,
    train_job_key,
)


def _redis_available() -> bool:
    """True only if a real Redis server answers a PING right now."""
    try:
        import redis
    except ImportError:
        return False
    url = os.environ.get("REDIS_URL", "redis://localhost:6379/0")
    try:
        client = redis.Redis.from_url(
            url, socket_connect_timeout=2, decode_responses=True)
        return bool(client.ping())
    except Exception:
        return False


REDIS_AVAILABLE = _redis_available()
REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379/0")

requires_redis = pytest.mark.skipif(
    not REDIS_AVAILABLE,
    reason="no Redis server reachable at $REDIS_URL or "
           "redis://localhost:6379/0 — Redis backend tests skipped, not faked",
)


def _key(prefix: str) -> str:
    return f"test:{prefix}:{uuid.uuid4().hex[:8]}"


def _check_store_behaviour(store) -> None:
    """Full interface contract, run against both backends."""
    # --- get/set/delete ---
    k = _key("kv")
    assert store.get(k) is None
    store.set(k, "hello")
    assert store.get(k) == "hello"
    store.set(k, "world")  # overwrite
    assert store.get(k) == "world"
    assert store.delete(k) is True
    assert store.get(k) is None
    assert store.delete(k) is False  # deleting a missing key

    # --- TTL expiry ---
    k2 = _key("ttl")
    store.set(k2, "short-lived", ttl=1)
    assert store.get(k2) == "short-lived"
    store.set(k2, "shorter", ttl=0.05)
    time.sleep(0.15)
    assert store.get(k2) is None

    # --- append / list (FIFO, oldest first) ---
    q = _key("queue")
    assert store.list(q) == []
    assert store.append(q, "a") == 1
    assert store.append(q, "b") == 2
    assert store.append(q, "c") == 3
    assert store.list(q) == ["a", "b", "c"]
    assert store.list(q, limit=2) == ["a", "b"]
    # list() does not consume
    assert store.list(q) == ["a", "b", "c"]

    # --- max_len keeps only the newest items ---
    q2 = _key("capped")
    for i in range(5):
        store.append(q2, str(i), max_len=3)
    assert store.list(q2) == ["2", "3", "4"]

    # --- drain returns all and empties ---
    assert store.drain(q) == ["a", "b", "c"]
    assert store.list(q) == []
    assert store.drain(q) == []
    assert store.get(q) is None


def test_inmemory_full_behaviour():
    _check_store_behaviour(InMemoryStateStore())


@requires_redis
def test_redis_full_behaviour():
    store = RedisStateStore(REDIS_URL)
    try:
        _check_store_behaviour(store)
    finally:
        # Best-effort cleanup of any keys this test created.
        for pattern_key in ("test:kv:*", "test:ttl:*", "test:queue:*",
                            "test:capped:*"):
            for key in store.client.keys(pattern_key):
                store.client.delete(key)


def test_inmemory_ttl_on_missing_key_is_noop():
    store = InMemoryStateStore()
    assert store.get(_key("nope")) is None


def test_inmemory_concurrent_appends():
    store = InMemoryStateStore()
    q = _key("concurrent")
    threads = [
        threading.Thread(target=lambda: [store.append(q, "x") for _ in range(20)])
        for _ in range(5)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(store.list(q)) == 100


def test_key_helpers():
    assert signal_queue_key("room-123") == "signals:room-123"
    assert train_job_key("abc123") == "train_job:abc123"
    assert signal_queue_key("a") != signal_queue_key("b")
    assert train_job_key("a") != signal_queue_key("a")


def test_store_json_round_trip():
    store = InMemoryStateStore()
    k = _key("json")
    assert load_json(store, k) is None
    store_json(store, k, {"job_id": "j1", "status": "running", "progress": 0.5})
    assert load_json(store, k) == {"job_id": "j1", "status": "running", "progress": 0.5}
    # Invalid JSON / non-dict JSON read back as None, not an exception.
    store.set(k, "not-json{")
    assert load_json(store, k) is None
    store.set(k, "[1, 2, 3]")
    assert load_json(store, k) is None


def test_factory_returns_inmemory_when_no_redis_url(monkeypatch):
    # create_state_store reads app.config.REDIS_URL at call time.
    import app.config as app_config
    monkeypatch.setattr(app_config, "REDIS_URL", "")
    assert isinstance(create_state_store(), InMemoryStateStore)
    assert isinstance(create_state_store(redis_url=""), InMemoryStateStore)
    assert isinstance(create_state_store(redis_url=None), InMemoryStateStore)


def test_factory_returns_redis_store_when_url_set():
    store = create_state_store(redis_url="redis://localhost:6379/0")
    assert isinstance(store, RedisStateStore)
    assert store.redis_url == "redis://localhost:6379/0"
    # Construction must not connect: a bad URL only fails on first use.
    bad = create_state_store(redis_url="redis://127.0.0.1:6399/0")
    assert isinstance(bad, RedisStateStore)


def test_get_state_store_singleton_and_reset(monkeypatch):
    import app.config as app_config
    monkeypatch.setattr(app_config, "REDIS_URL", "")
    reset_state_store()
    first = get_state_store()
    second = get_state_store()
    assert first is second
    assert isinstance(first, InMemoryStateStore)
    reset_state_store()
    third = get_state_store()
    assert third is not first
    reset_state_store()
