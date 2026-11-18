"""Transcript persistence helpers.

Transcripts are stored ONLY when the user consents (store_transcripts
preference / explicit save). Nothing is recorded by default.
"""
from sqlalchemy.orm import Session

from app.models import TranscriptEntry, UserPreferences


def add_entry(db: Session, room_id: str, direction: str, text: str,
              speaker_name: str = "", user_id=None, confidence=None,
              consent: bool = False) -> TranscriptEntry:
    # Consent-gated storage: store only with explicit per-request consent
    # OR the user's stored store_transcripts preference. Otherwise refuse
    # (the route turns this into a 400) — never store silently.
    if not consent:
        prefs = (
            db.query(UserPreferences).filter(UserPreferences.user_id == user_id).first()
            if user_id else None
        )
        if not prefs or not prefs.store_transcripts:
            raise ValueError(
                "Transcript storage requires consent: pass consent=true or "
                "enable 'store transcripts' in your preferences."
            )
    entry = TranscriptEntry(room_id=room_id, direction=direction, text=text,
                            speaker_name=speaker_name, user_id=user_id,
                            confidence=confidence)
    db.add(entry)
    db.commit()
    db.refresh(entry)
    return entry


def list_entries(db: Session, room_id: str):
    return (db.query(TranscriptEntry).filter(TranscriptEntry.room_id == room_id)
            .order_by(TranscriptEntry.created_at.asc()).all())


def export_txt(entries) -> str:
    lines = ["SignSpeak Telecom — Call Transcript", ""]
    for e in entries:
        ts = e.created_at.isoformat() if e.created_at else ""
        lines.append(f"[{ts}] ({e.direction}) {e.speaker_name or 'Unknown'}: {e.text}")
    return "\n".join(lines) + "\n"
