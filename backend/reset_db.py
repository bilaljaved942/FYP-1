import asyncio
import logging
from app.database import engine, Base
# Import all models so they get registered with Base.metadata
from app.models import User, AnalysisJob 

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def reset_database():
    logger.info("Connecting to database...")
    async with engine.begin() as conn:
        logger.info("Dropping existing tables (wiping old schema)...")
        await conn.run_sync(Base.metadata.drop_all)
        
        logger.info("Creating new tables with updated schema (roles, names, etc)...")
        await conn.run_sync(Base.metadata.create_all)
        
    logger.info("Database reset complete! You are ready to go.")

if __name__ == "__main__":
    asyncio.run(reset_database())
