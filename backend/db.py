"""
backend/db.py

Async PostgreSQL connection pool lifecycle for the Attention-Locator backend.
Uses asyncpg with pgvector codec registration.
"""
from __future__ import annotations

import os

import asyncpg
from pgvector.asyncpg import register_vector


# ── Configuration ─────────────────────────────────────────────────────────────
#
# Override any of these via environment variables before starting uvicorn, e.g.:
#   export ATTN_DB_HOST=localhost
#   export ATTN_DB_PASSWORD=<password>
#
DB_HOST     = os.environ.get("ATTN_DB_HOST",     "localhost")
DB_PORT     = int(os.environ.get("ATTN_DB_PORT", "5432"))
DB_NAME     = os.environ.get("ATTN_DB_NAME",     "attention_locator")
DB_USER     = os.environ.get("ATTN_DB_USER",     "postgres")
DB_PASSWORD = os.environ.get("ATTN_DB_PASSWORD", "postgres")
DB_MIN_CONN = int(os.environ.get("ATTN_DB_MIN_CONN", "1"))
DB_MAX_CONN = int(os.environ.get("ATTN_DB_MAX_CONN", "5"))


async def _init_connection(conn: asyncpg.Connection) -> None:
    """Register the pgvector codec on every new connection in the pool."""
    await register_vector(conn)


async def create_pool() -> asyncpg.Pool:
    return await asyncpg.create_pool(
        host=DB_HOST,
        port=DB_PORT,
        database=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD,
        min_size=DB_MIN_CONN,
        max_size=DB_MAX_CONN,
        init=_init_connection,
    )
