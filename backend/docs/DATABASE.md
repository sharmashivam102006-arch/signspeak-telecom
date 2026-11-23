# Database & State — SignSpeak Telecom

## Current state

| Concern | Status |
|---|---|
| Local development DB | **SQLite** (`signspeak.db` in `backend/`), created by SQLAlchemy `create_all` historically |
| Migrations | **Alembic, added 2026-11-17.** Initial migration `761b2074d828` creates all 5 tables (see below). `create_all` is still used by the test suite only |
| PostgreSQL | Supported via `DATABASE_URL` + `psycopg[binary]` driver; **not verified live** — no Postgres server is available in this sandbox (see § Postgres) |
| Redis | Optional state-store backend; **not verified live** — no Redis server is available in this sandbox (see § Redis) |
| Redis wiring | **Not integrated yet** — the store module exists and is tested, but no route/service uses it (see § Pending integration) |

### Tables (SQLAlchemy models in `app/models.py`)

| Table | Model | Purpose |
|---|---|---|
| `users` | `User` | Accounts (email + bcrypt hash, or guest) |
| `rooms` | `Room` | Call rooms (6-char code, creator, active flag) |
| `room_participants` | `RoomParticipant` | Membership of a user in a room, incl. join/leave |
| `transcript_entries` | `TranscriptEntry` | Stored transcript lines (consent-gated, creator-only reads) |
| `user_preferences` | `UserPreferences` | Per-user role, captions/TTS settings, thresholds |

---

## Migrations (Alembic)

Alembic lives in `backend/alembic/` with config in `backend/alembic.ini`.
`alembic/env.py` reads the database URL from the app config
(`app/config.py` → `DATABASE_URL` env var / `.env`), so migrations always
target the same database the app uses. The URL in `alembic.ini` is a
placeholder and is overridden by `env.py`.

### Commands (run from `backend/`)

```bash
# Apply all migrations (fresh DB or existing DB at an older revision)
alembic upgrade head

# Show current revision
alembic current

# Create a new migration after changing app/models.py
alembic revision --autogenerate -m "describe the change"

# Roll back one migration / everything
alembic downgrade -1
alembic downgrade base

# Print the SQL a migration would run, without touching the DB
alembic upgrade head --sql
```

### Existing databases created by `create_all`

Databases created before Alembic (the dev `signspeak.db`, the test
`test_signspeak.db`) already have the schema but no `alembic_version`
row. Running `alembic upgrade head` against one fails with "table
already exists". Mark such a database as current instead — no DDL runs:

```bash
DATABASE_URL="sqlite:///./signspeak.db" alembic stamp head
```

Only use `stamp` on a database whose schema provably matches the
migration (same 5 tables/columns). A fresh database should always use
`alembic upgrade head`.

### Verification (2026-11-17, this sandbox)

```bash
rm -f /tmp/alembic-test.db
DATABASE_URL="sqlite:////tmp/alembic-test.db" alembic revision --autogenerate \
  -m "initial schema: users, rooms, room_participants, transcript_entries, user_preferences"
# -> revision 761b2074d828, detected all 5 tables + indexes
DATABASE_URL="sqlite:////tmp/alembic-test.db" alembic upgrade head
# -> created users, rooms, room_participants, transcript_entries,
#    user_preferences + alembic_version stamped at 761b2074d828 (head)
DATABASE_URL="sqlite:////tmp/alembic-test.db" alembic downgrade base   # drops all 5
DATABASE_URL="sqlite:////tmp/alembic-test.db" alembic upgrade head     # re-creates them
```

The live `signspeak.db` was **not** migrated or stamped during this work;
it is untouched (still `create_all` schema, no `alembic_version` row).

---

## PostgreSQL

### Switching (deployment)

```bash
# 1. Run Postgres (example: local Docker)
docker run -d --name signspeak-pg -e POSTGRES_USER=signspeak \
  -e POSTGRES_PASSWORD='<long-random-password>' -e POSTGRES_DB=signspeak \
  -p 5432:5432 postgres:16

# 2. Point the app at it (in .env or the environment)
DATABASE_URL=postgresql://signspeak:<password>@localhost:5432/signspeak

# 3. Create the schema with Alembic (never create_all in production)
alembic upgrade head

# 4. Start the backend as usual
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

### Driver note

`requirements.txt` installs `psycopg[binary]` (psycopg v3). SQLAlchemy's
bare `postgresql://` scheme defaults to the psycopg2 driver, which is
**not** installed, so `app/database.py` rewrites a bare `postgresql://`
URL to `postgresql+psycopg://` at startup. Either form works:

```text
DATABASE_URL=postgresql://signspeak:<password>@localhost:5432/signspeak
DATABASE_URL=postgresql+psycopg://signspeak:<password>@localhost:5432/signspeak
```

### Verification status

**NOT VERIFIED live.** No PostgreSQL server is available in this sandbox
(no `psql`/`pg_isready` binaries; TCP connect to `127.0.0.1:5432`
refused). What was verified instead:

* `psycopg[binary]` 3.3.9 installs and imports in `backend/.venv`.
* The SQLite path still works end to end — the full backend test suite
  passes (see § Tests).
* The URL-rewrite logic in `app/database.py` is unit-covered by the fact
  that the app boots and the test suite runs against SQLite; the
  Postgres branch itself has **not** been exercised against a live
  server. Before deploying to Postgres, run `alembic upgrade head`
  against a real server and the test suite with
  `DATABASE_URL=postgresql://...` pointed at a scratch database.

### Local development stays SQLite

The default when `DATABASE_URL` is unset is `sqlite:///./signspeak.db`.
Do not change the default: local dev, tests and the sandbox all rely on
it. Postgres is for deployment (and for staging/CI if a server exists).

---

## Redis (optional state store)

### What it is for

Some state is ephemeral and does not belong in SQL: per-room REST
signal queues and training-job status. Today those are module-level
dicts (lost on restart, invisible to other workers). The store in
`app/services/state_store.py` abstracts that state behind two backends:

| Backend | When | Behaviour |
|---|---|---|
| `InMemoryStateStore` | `REDIS_URL` empty (default) | Current behaviour; process-local, thread-safe, TTL via monotonic deadlines |
| `RedisStateStore` | `REDIS_URL` set | Same interface; state survives restarts and is shared across workers |

Enable it with:

```bash
# Example: local Docker
docker run -d --name signspeak-redis -p 6379:6379 redis:7
REDIS_URL=redis://localhost:6379/0
```

Values are plain strings; callers JSON-encode dicts (helpers
`store_json` / `load_json` are provided). List keys (`append`/`list`/
`drain`) are FIFO queues with an optional newest-N cap — the shape the
signal fallback needs. Voice-agent sessions are **not** stored here:
a voice agent holds a live LiveKit RTC connection, which cannot be
serialised; only its status metadata could move to the store, the live
agent objects must stay in-process.

### Verification status

**NOT VERIFIED live.** No Redis server is available in this sandbox
(no `redis-cli` binary; TCP connect to `127.0.0.1:6379` refused).
What was verified instead:

* `redis` 8.1.0 installs and imports in `backend/.venv`.
* The in-memory backend is fully tested (see § Tests).
* The Redis backend test (`test_redis_full_behaviour`) runs the same
  contract against a real server when one is reachable and **skips with
  an explicit reason** when it is not — it is never faked. Before
  relying on Redis, start a server and run
  `pytest tests/test_state_store.py -v` with `REDIS_URL` set; the Redis
  test should pass instead of skip.

---

## Pending integration (not done in this step)

The store module is delivered and tested, but **no route or service
uses it yet** — the existing in-memory dicts are untouched. The
integration step (a separate change, touching `rooms.py` and
`sign_training.py`) is:

### 1. Signal queue — `app/routes/rooms.py`

Replace the module-level `_signal_store` dict:

```python
from app.services.state_store import get_state_store, signal_queue_key
import json

# remove: _signal_store: dict[str, list[dict]] = {}

@router.post("/{room_id}/signal")
def post_signal(room_id: str, msg: SignalMessage, ...):
    _find(db, room_id)
    store = get_state_store()
    count = store.append(
        signal_queue_key(room_id), json.dumps(msg.model_dump()), max_len=100)
    return {"queued": True, "count": count}

@router.get("/{room_id}/signal")
def get_signal(room_id: str, ...):
    _find(db, room_id)
    store = get_state_store()
    msgs = [json.loads(raw) for raw in store.drain(signal_queue_key(room_id))]
    return {"messages": msgs}
```

Behaviour is preserved exactly: 100-message cap (newest kept), GET
drains the queue. With `REDIS_URL` set, queues survive restarts and are
shared across workers; without it, behaviour is identical to today.

### 2. Training-job status — `app/services/sign_training.py`

Job status is a small dict mutated in place by the training thread, so
it needs a read-modify-write through the store rather than a shared
reference. Sketch (status only; the training thread itself stays local):

```python
from app.services.state_store import get_state_store, store_json, load_json, train_job_key

# _jobs[job_id] = job  ->  store_json(get_state_store(), train_job_key(job_id), job)
# job = _jobs[job_id]  ->  job = load_json(get_state_store(), train_job_key(job_id))
# _public(job)         ->  _public(load_json(get_state_store(), train_job_key(job_id)))
```

Progress updates inside `_run_job` / `_progress` must re-store the dict
after each mutation. The `_active_job_id` guard (one active job) can
stay in-process — it is a local scheduling guard, not shared state.
Do **not** store the thread or any non-JSON value in the store.

---

## Migration workflow (future schema changes)

1. Change the model in `app/models.py`.
2. `cd backend && alembic revision --autogenerate -m "<what changed>"`.
3. **Read the generated file** — autogenerate misses some things
   (server defaults, some renames, data migrations). Fix it by hand.
4. Test on a scratch DB first:
   `DATABASE_URL="sqlite:////tmp/mig-test.db" alembic upgrade head`
   (and `downgrade` if the migration has one), then run the test suite.
5. Deploy: `alembic upgrade head` against the real `DATABASE_URL`
   before starting the new backend code.
6. Never edit a migration that has already run in production — write a
   new one.

---

## Tests

```bash
cd backend
.venv/bin/python -m pytest tests/test_api.py tests/test_state_store.py -v
```

* `tests/test_api.py` — 34 tests (existing suite; uses
  `DATABASE_URL=sqlite:///./test_signspeak.db`). Known flaky under
  load: `test_sign_train_job_lifecycle` can time out when torch +
  Whisper + MediaPipe + Piper are all loaded (see audit report).
* `tests/test_state_store.py` — 9 tests: in-memory backend contract
  (get/set/delete/TTL/append/list/drain/max_len), concurrent appends,
  JSON helpers, key helpers, factory + singleton. The Redis contract
  test runs only when a server is reachable and skips otherwise.

Full-suite result for this change (2026-11-17, this sandbox):
**41 passed, 2 skipped in 55.18 s.** The 2 skips are pre-existing /
environmental, not failures: `test_voice_agent_lifecycle` (voice agent
cannot join LiveKit in this sandbox) and `test_redis_full_behaviour`
(no Redis server reachable). See the implementation report for the
exact run.
