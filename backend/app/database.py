"""SQLAlchemy setup. SQLite by default; DATABASE_URL can point to Postgres.

Postgres URLs: SQLAlchemy's bare ``postgresql://`` scheme defaults to the
psycopg2 driver, which this project does not install — the installed
driver is psycopg v3 (``psycopg[binary]``). A bare ``postgresql://`` URL
is therefore rewritten to ``postgresql+psycopg://`` below, so the
``DATABASE_URL=postgresql://...`` form documented in .env.example works.
"""
from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

from app.config import DATABASE_URL

if DATABASE_URL.startswith("postgresql://"):
    DATABASE_URL = "postgresql+psycopg://" + DATABASE_URL[len("postgresql://"):]

connect_args = {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}
engine = create_engine(DATABASE_URL, connect_args=connect_args)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
