#!/usr/bin/env python3
"""
Database Migration Runner

Applies all SQL migrations in the correct order to ensure database schema is up to date.
"""

import asyncio
import logging
import sys
from pathlib import Path

# Add the app directory to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.core.config import get_settings
from app.core.database import get_database

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Migration files in order
MIGRATION_FILES = [
    'migrations/init.sql',
    'migrations/001_add_crawl_cache.sql',
    'migrations/002_enhance_crawl_cache.sql', 
    'migrations/003_optimize_cache_indexes.sql',
    'migrations/004_hybrid_cache_tables.sql',
    'migrations/005_add_metadata_to_crawl_cache.sql'
]

# Expected tables after migrations
EXPECTED_TABLES = [
    'images',
    'faces', 
    'audit_logs',
    'search_audit_logs',
    'crawl_cache',
    'crawl_cache_metadata'
]

async def run_migration_file(migration_path: str) -> bool:
    """Run a single migration file."""
    try:
        # Read migration file
        with open(migration_path, 'r', encoding='utf-8') as f:
            sql_content = f.read()
        
        if not sql_content.strip():
            logger.warning(f"Migration file {migration_path} is empty, skipping")
            return True
        
        # Execute migration
        db = get_database()
        await db.execute(sql_content)
        
        logger.info(f"✓ Applied migration: {migration_path}")
        return True
        
    except Exception as e:
        logger.error(f"✗ Failed to apply migration {migration_path}: {e}")
        return False

async def run_all_migrations():
    """Run all migration files in order."""
    logger.info("Starting database migrations...")
    
    for migration_file in MIGRATION_FILES:
        migration_path = Path(migration_file)
        
        if not migration_path.exists():
            logger.warning(f"Migration file not found: {migration_file}, skipping")
            continue
        
        success = await run_migration_file(migration_file)
        if not success:
            logger.error(f"Migration failed: {migration_file}")
            return False
    
    logger.info("✓ All migrations completed successfully")
    return True

async def verify_tables():
    """Verify all expected tables exist."""
    logger.info("Verifying database tables...")
    
    try:
        db = get_database()
        
        # Check each expected table
        for table_name in EXPECTED_TABLES:
            result = await db.fetch_one(
                """
                SELECT EXISTS (
                    SELECT FROM information_schema.tables 
                    WHERE table_schema = 'public' 
                    AND table_name = $1
                )
                """,
                table_name
            )
            
            if result and result[0]:
                logger.info(f"✓ Table '{table_name}' exists")
            else:
                logger.error(f"✗ Table '{table_name}' does not exist")
                return False
        
        logger.info("✓ All required tables exist")
        return True
        
    except Exception as e:
        logger.error(f"✗ Error verifying tables: {e}")
        return False

async def get_table_info():
    """Get information about existing tables."""
    try:
        db = get_database()
        
        result = await db.fetch_all(
            """
            SELECT table_name, 
                   (SELECT COUNT(*) FROM information_schema.columns 
                    WHERE table_schema = 'public' AND table_name = t.table_name) as column_count
            FROM information_schema.tables t
            WHERE table_schema = 'public' 
            AND table_type = 'BASE TABLE'
            ORDER BY table_name
            """
        )
        
        logger.info("Database tables:")
        for row in result:
            logger.info(f"  - {row[0]} ({row[1]} columns)")
            
    except Exception as e:
        logger.error(f"Error getting table info: {e}")

async def main():
    """Main function to run migrations and verify database."""
    logger.info("Starting database migration process...")
    
    # Run all migrations
    success = await run_all_migrations()
    if not success:
        logger.error("Migration process failed")
        sys.exit(1)
    
    # Verify tables exist
    success = await verify_tables()
    if not success:
        logger.error("Table verification failed")
        sys.exit(1)
    
    # Show table information
    await get_table_info()
    
    logger.info("Database migration process completed successfully")

if __name__ == "__main__":
    asyncio.run(main())
