from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    app_name: str = "SignSpeak Telecom API"
    environment: str = "development"
    debug: bool = True

    database_url: str = "postgresql+asyncpg://signspeak:signspeak@localhost:5432/signspeak"
    redis_url: str = "redis://localhost:6379/0"

    # Auth. In production this must come from the environment; the default is
    # here so `docker compose up` works on a laptop and nowhere else.
    jwt_secret: str = "dev-only-change-me"
    jwt_algorithm: str = "HS256"
    access_token_minutes: int = 60 * 12

    # LiveKit
    livekit_url: str = "ws://localhost:7880"
    livekit_api_key: str = "devkey"
    livekit_api_secret: str = "devsecretdevsecretdevsecret12345"

    cors_origins: list[str] = ["http://localhost:3000"]

    # Transcript reorder window. Sign events and speech events reach the server
    # with very different latencies (segmentation lag vs streaming ASR), so
    # appending in arrival order produces a transcript where the reply lands
    # before the question. We buffer briefly and sort by client start time.
    # See services/ordering.py.
    transcript_reorder_ms: int = 900

    # Data minimisation. Raw media never touches this service by design; these
    # control how long the derived text lives.
    transcript_retention_days: int = 30
    store_transcripts_by_default: bool = False

    whisper_model: str = "base.en"
    whisper_device: str = "cpu"
    whisper_compute_type: str = "int8"

    @property
    def is_production(self) -> bool:
        return self.environment.lower() in {"production", "prod"}


@lru_cache
def get_settings() -> Settings:
    s = Settings()
    if s.is_production and s.jwt_secret == "dev-only-change-me":
        raise RuntimeError("JWT_SECRET must be set in production")
    return s


settings = get_settings()
