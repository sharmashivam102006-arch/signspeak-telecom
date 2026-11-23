from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .config import settings
from .db import create_all, engine
from .routers import auth, rooms, transcripts, ws

logging.basicConfig(level=logging.INFO if not settings.debug else logging.DEBUG)
log = logging.getLogger("signspeak")


@asynccontextmanager
async def lifespan(app: FastAPI):
    if not settings.is_production:
        await create_all()
        log.info("dev schema ensured (production uses alembic upgrade head)")
    yield
    await engine.dispose()


app = FastAPI(
    title=settings.app_name,
    version="0.1.0",
    lifespan=lifespan,
    description=(
        "Accessibility layer for real-time calls. This service handles identity, "
        "rooms, LiveKit tokens and text. It never receives video, audio or landmarks: "
        "sign recognition runs in the browser and publishes only the recognised word."
    ),
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth.router)
app.include_router(rooms.router)
app.include_router(transcripts.router)
app.include_router(ws.router)


@app.get("/health", tags=["ops"])
async def health() -> dict:
    return {"status": "ok", "environment": settings.environment}
