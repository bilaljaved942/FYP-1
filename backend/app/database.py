"""
Database Configuration & Session Management
=============================================
Configures the async PostgreSQL connection using SQLAlchemy 2.0+.

- Uses `asyncpg` as the async database driver.
- DATABASE_URL is read from environment variables (for deployment flexibility),
  falling back to a local development default.
- Provides:
    - `engine`:             The async database engine (connection pool).
    - `async_session_maker`: Factory for creating new async database sessions.
    - `Base`:               Declarative base class that all ORM models inherit from.
    - `get_db()`:           FastAPI dependency that yields a scoped async session.
    - `init_db()`:          Creates all tables on application startup.
"""

import os

from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase

# ── Database connection URL ──────────────────────────────────────────
# Reads from the DATABASE_URL environment variable when deployed (e.g., Docker/EC2).
# Falls back to the local PostgreSQL development database if not set.
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql+asyncpg://postgres:pakistan942@localhost:5432/fyp_db"
)

# ── SQLAlchemy Async Engine ──────────────────────────────────────────
# `echo=False` suppresses SQL query logging in production.
# Set to True during debugging to see raw SQL in the console.
engine = create_async_engine(DATABASE_URL, echo=False)

# ── Session Factory ──────────────────────────────────────────────────
# Creates async sessions that are used in every API endpoint via `get_db()`.
# `expire_on_commit=False` allows accessing attributes after commit without
# triggering lazy loads (important for async contexts).
async_session_maker = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False
)


# ── Declarative Base ─────────────────────────────────────────────────
# All ORM models (User, AnalysisJob) inherit from this Base class.
class Base(DeclarativeBase):
    pass


# ── FastAPI Dependency ───────────────────────────────────────────────
async def get_db() -> AsyncSession:
    """
    Yields an async database session for use in FastAPI route handlers.
    The session is automatically closed when the request finishes.

    Usage in endpoints:
        async def my_endpoint(db: AsyncSession = Depends(get_db)):
    """
    async with async_session_maker() as session:
        yield session


# ── Table Initialization ────────────────────────────────────────────
async def init_db() -> None:
    """
    Creates all database tables defined by ORM models if they don't exist.
    Called once during FastAPI's startup event.
    """
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
