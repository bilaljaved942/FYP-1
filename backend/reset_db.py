"""
Database Reset Utility
=======================
Drops ALL existing tables and recreates them with the latest schema.

Usage:
    python reset_db.py

WARNING: This permanently deletes all data (users, analysis jobs, results).
Only use when the schema has changed (new columns, modified types, etc.)
and you need a clean slate. Not for production use.
"""

import asyncio
import logging
from app.database import engine, Base

# Import all models so their table definitions are registered with Base.metadata.
# Without these imports, SQLAlchemy wouldn't know which tables to create.
from app.models import User, AnalysisJob 

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def reset_database():
    """
    Perform a full database reset:
        1. Connect to PostgreSQL
        2. DROP all tables (users, analysis_jobs, enums)
        3. CREATE all tables fresh with the current schema
    """
    logger.info("Connecting to database...")
    async with engine.begin() as conn:
        logger.info("Dropping existing tables (wiping old schema)...")
        await conn.run_sync(Base.metadata.drop_all)
        
        logger.info("Creating new tables with updated schema (roles, names, etc)...")
        await conn.run_sync(Base.metadata.create_all)
        
    logger.info("Database reset complete! You are ready to go.")


if __name__ == "__main__":
    asyncio.run(reset_database())
