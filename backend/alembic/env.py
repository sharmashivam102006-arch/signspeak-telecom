"""Alembic environment for SignSpeak Telecom.

The database URL comes from the app config (``app/config.py``), which
reads ``DATABASE_URL`` from the environment / .env — the same source the
running app uses. Default is local SQLite (``sqlite:///./signspeak.db``);
set ``DATABASE_URL=postgresql://...`` for Postgres. The URL in
alembic.ini is only a placeholder and is overridden here.
"""
import os
import sys
from logging.config import fileConfig

from alembic import context

# Make the `app` package importable when alembic runs from backend/.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# this is the Alembic Config object, which provides
# access to the values within the .ini file in use.
config = context.config

# Interpret the config file for Python logging.
# This line sets up loggers basically.
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

from app.config import DATABASE_URL  # noqa: E402
from app.models import Base  # noqa: E402 — importing registers all models

# The single source of truth for the DB URL; overrides alembic.ini.
config.set_main_option("sqlalchemy.url", DATABASE_URL)

# for 'autogenerate' support
target_metadata = Base.metadata

# other values from the config, defined by the needs of env.py,
# can be acquired:
# my_important_option = config.get_main_option("my_important_option")
# ... etc.


def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode.

    This configures the context with just a URL
    and not an Engine, though an Engine is acceptable
    here as well.  By skipping the Engine creation
    we don't even need a DBAPI to be available.

    Calls to context.execute() here emit the given string to the
    script output.

    """
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Run migrations in 'online' mode.

    In this scenario we need to create an Engine
    and associate a connection with the context.

    """
    # Reuse the app's engine: it already applies the SQLite
    # check_same_thread flag and the postgresql:// -> postgresql+psycopg://
    # driver normalisation from app/database.py.
    from app.database import engine as connectable

    with connectable.connect() as connection:
        context.configure(
            connection=connection, target_metadata=target_metadata
        )

        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
