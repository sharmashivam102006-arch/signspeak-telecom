-- SignSpeak Telecom schema (mirrors SQLAlchemy create_all in app/main.py).
-- For production use Alembic migrations; this file documents the tables.
CREATE TABLE IF NOT EXISTS users (
  id TEXT PRIMARY KEY, email TEXT UNIQUE NOT NULL, name TEXT NOT NULL DEFAULT '',
  hashed_password TEXT NOT NULL DEFAULT '', is_guest BOOLEAN DEFAULT FALSE,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS rooms (
  id TEXT PRIMARY KEY, code TEXT UNIQUE NOT NULL, name TEXT NOT NULL DEFAULT 'SignSpeak Call',
  creator_id TEXT REFERENCES users(id), created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  is_active BOOLEAN DEFAULT TRUE
);
CREATE TABLE IF NOT EXISTS transcript_entries (
  id TEXT PRIMARY KEY, room_id TEXT NOT NULL REFERENCES rooms(id),
  user_id TEXT REFERENCES users(id), speaker_name TEXT DEFAULT '',
  direction TEXT NOT NULL DEFAULT 'speech-to-text', text TEXT NOT NULL,
  confidence INTEGER, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS room_participants (
  id TEXT PRIMARY KEY, room_id TEXT NOT NULL REFERENCES rooms(id),
  user_id TEXT REFERENCES users(id), identity TEXT NOT NULL DEFAULT '',
  name TEXT NOT NULL DEFAULT '', joined_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  left_at TIMESTAMP, is_active BOOLEAN DEFAULT TRUE
);
CREATE INDEX IF NOT EXISTS idx_room_participants_room ON room_participants(room_id);
CREATE INDEX IF NOT EXISTS idx_room_participants_user ON room_participants(user_id);
CREATE TABLE IF NOT EXISTS user_preferences (
  id TEXT PRIMARY KEY, user_id TEXT UNIQUE NOT NULL REFERENCES users(id),
  role TEXT DEFAULT 'hearing', captions_enabled BOOLEAN DEFAULT TRUE,
  tts_enabled BOOLEAN DEFAULT TRUE, tts_muted BOOLEAN DEFAULT FALSE,
  tts_volume INTEGER DEFAULT 80, caption_language TEXT DEFAULT 'en',
  sign_threshold_high INTEGER DEFAULT 85, sign_threshold_low INTEGER DEFAULT 60,
  store_transcripts BOOLEAN DEFAULT FALSE
);
