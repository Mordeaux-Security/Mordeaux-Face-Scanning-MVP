import asyncio
import time
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import logging
import psycopg
from ..core.config import get_settings

logger = logging.getLogger(__name__)

class DatabaseOptimizationService:
    """Service for database optimization and maintenance tasks."""
    
    def __init__(self):
        self.settings = get_settings()
    
    async def analyze_query_performance(self) -> Dict[str, Any]:
        """Analyze query performance and identify slow queries."""
        try:
            connection_string = f"postgresql://{self.settings.postgres_user}:{self.settings.postgres_password}@{self.settings.postgres_host}:{self.settings.postgres_port}/{self.settings.postgres_db}"
            
            async with psycopg.AsyncConnectionPool(connection_string, min_size=1, max_size=2) as pool:
                async with pool.connection() as conn:
                    async with conn.cursor() as cur:
                        # Get slow queries from pg_stat_statements (if available)
                        slow_queries = await self._get_slow_queries(cur)
                        
                        # Get index usage statistics
                        index_stats = await self._get_index_usage_stats(cur)
                        
                        # Get table statistics
                        table_stats = await self._get_table_statistics(cur)
                        
                        # Get connection statistics
                        connection_stats = await self._get_connection_stats(cur)
                        
                        return {
                            "timestamp": time.time(),
                            "slow_queries": slow_queries,
                            "index_usage": index_stats,
                            "table_statistics": table_stats,
                            "connection_stats": connection_stats
                        }
        except Exception as e:
            logger.error(f"Error analyzing query performance: {e}")
            return {"error": str(e)}
    
    async def optimize_audit_logs_table(self) -> Dict[str, Any]:
        """Optimize the audit logs table for better performance."""
        try:
            connection_string = f"postgresql://{self.settings.postgres_user}:{self.settings.postgres_password}@{self.settings.postgres_host}:{self.settings.postgres_port}/{self.settings.postgres_db}"
            
            async with psycopg.AsyncConnectionPool(connection_string, min_size=1, max_size=2) as pool:
                async with pool.connection() as conn:
                    async with conn.cursor() as cur:
                        # Analyze the audit_logs table
                        await cur.execute("ANALYZE audit_logs;")
                        
                        # Update table statistics
                        await cur.execute("""
                            SELECT 
                                schemaname, 
                                tablename, 
                                n_tup_ins as inserts,
                                n_tup_upd as updates,
                                n_tup_del as deletes,
                                n_live_tup as live_tuples,
                                n_dead_tup as dead_tuples,
                                last_vacuum,
                                last_autovacuum,
                                last_analyze,
                                last_autoanalyze
                            FROM pg_stat_user_tables 
                            WHERE tablename = 'audit_logs'
                        """)
                        
                        table_stats = await cur.fetchone()
                        
                        # Get index statistics
                        await cur.execute("""
                            SELECT 
                                indexname,
                                idx_tup_read,
                                idx_tup_fetch,
                                idx_scan,
                                idx_tup_read / NULLIF(idx_scan, 0) as avg_tuples_per_scan
                            FROM pg_stat_user_indexes 
                            WHERE schemaname = 'public' AND tablename = 'audit_logs'
                            ORDER BY idx_scan DESC
                        """)
                        
                        index_stats = await cur.fetchall()
                        
                        return {
                            "timestamp": time.time(),
                            "table_stats": {
                                "inserts": table_stats[2] if table_stats else 0,
                                "updates": table_stats[3] if table_stats else 0,
                                "deletes": table_stats[4] if table_stats else 0,
                                "live_tuples": table_stats[5] if table_stats else 0,
                                "dead_tuples": table_stats[6] if table_stats else 0,
                                "last_vacuum": table_stats[7] if table_stats else None,
                                "last_analyze": table_stats[9] if table_stats else None
                            },
                            "index_stats": [
                                {
                                    "index_name": row[0],
                                    "tuples_read": row[1],
                                    "tuples_fetched": row[2],
                                    "scans": row[3],
                                    "avg_tuples_per_scan": float(row[4]) if row[4] else 0.0
                                }
                                for row in index_stats
                            ]
                        }
        except Exception as e:
            logger.error(f"Error optimizing audit logs table: {e}")
            return {"error": str(e)}
    
    async def cleanup_old_audit_logs(self, retention_days: int = 30) -> Dict[str, Any]:
        """Clean up old audit logs based on retention policy."""
        try:
            connection_string = f"postgresql://{self.settings.postgres_user}:{self.settings.postgres_password}@{self.settings.postgres_host}:{self.settings.postgres_port}/{self.settings.postgres_db}"
            
            cutoff_date = datetime.now() - timedelta(days=retention_days)
            cutoff_timestamp = cutoff_date.timestamp()
            
            async with psycopg.AsyncConnectionPool(connection_string, min_size=1, max_size=2) as pool:
                async with pool.connection() as conn:
                    async with conn.cursor() as cur:
                        # Delete old audit logs
                        await cur.execute(
                            "DELETE FROM audit_logs WHERE created_at < %s",
                            (cutoff_timestamp,)
                        )
                        audit_deleted = cur.rowcount
                        
                        # Delete old search audit logs
                        await cur.execute(
                            "DELETE FROM search_audit_logs WHERE created_at < %s",
                            (cutoff_timestamp,)
                        )
                        search_audit_deleted = cur.rowcount
                        
                        await conn.commit()
                        
                        # Vacuum the tables to reclaim space
                        await cur.execute("VACUUM audit_logs;")
                        await cur.execute("VACUUM search_audit_logs;")
                        
                        return {
                            "timestamp": time.time(),
                            "retention_days": retention_days,
                            "cutoff_date": cutoff_date.isoformat(),
                            "audit_logs_deleted": audit_deleted,
                            "search_audit_logs_deleted": search_audit_deleted,
                            "total_deleted": audit_deleted + search_audit_deleted,
                            "vacuum_completed": True
                        }
        except Exception as e:
            logger.error(f"Error cleaning up old audit logs: {e}")
            return {"error": str(e)}
    
    async def get_database_health_metrics(self) -> Dict[str, Any]:
        """Get comprehensive database health metrics."""
        try:
            connection_string = f"postgresql://{self.settings.postgres_user}:{self.settings.postgres_password}@{self.settings.postgres_host}:{self.settings.postgres_port}/{self.settings.postgres_db}"
            
            async with psycopg.AsyncConnectionPool(connection_string, min_size=1, max_size=2) as pool:
                async with pool.connection() as conn:
                    async with conn.cursor() as cur:
                        # Get database size
                        await cur.execute("SELECT pg_database_size(current_database())")
                        db_size = await cur.fetchone()
                        
                        # Get table sizes
                        await cur.execute("""
                            SELECT 
                                schemaname,
                                tablename,
                                pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) as size,
                                pg_total_relation_size(schemaname||'.'||tablename) as size_bytes
                            FROM pg_tables 
                            WHERE schemaname = 'public'
                            ORDER BY pg_total_relation_size(schemaname||'.'||tablename) DESC
                        """)
                        
                        table_sizes = await cur.fetchall()
                        
                        # Get connection statistics
                        await cur.execute("""
                            SELECT 
                                count(*) as total_connections,
                                count(*) FILTER (WHERE state = 'active') as active_connections,
                                count(*) FILTER (WHERE state = 'idle') as idle_connections,
                                count(*) FILTER (WHERE state = 'idle in transaction') as idle_in_transaction
                            FROM pg_stat_activity
                        """)
                        
                        connection_stats = await cur.fetchone()
                        
                        # Get lock statistics
                        await cur.execute("""
                            SELECT 
                                count(*) as total_locks,
                                count(*) FILTER (WHERE granted = true) as granted_locks,
                                count(*) FILTER (WHERE granted = false) as waiting_locks
                            FROM pg_locks
                        """)
                        
                        lock_stats = await cur.fetchone()
                        
                        return {
                            "timestamp": time.time(),
                            "database_size_bytes": db_size[0] if db_size else 0,
                            "database_size_human": self._format_bytes(db_size[0] if db_size else 0),
                            "table_sizes": [
                                {
                                    "schema": row[0],
                                    "table": row[1],
                                    "size": row[2],
                                    "size_bytes": row[3]
                                }
                                for row in table_sizes
                            ],
                            "connections": {
                                "total": connection_stats[0] if connection_stats else 0,
                                "active": connection_stats[1] if connection_stats else 0,
                                "idle": connection_stats[2] if connection_stats else 0,
                                "idle_in_transaction": connection_stats[3] if connection_stats else 0
                            },
                            "locks": {
                                "total": lock_stats[0] if lock_stats else 0,
                                "granted": lock_stats[1] if lock_stats else 0,
                                "waiting": lock_stats[2] if lock_stats else 0
                            }
                        }
        except Exception as e:
            logger.error(f"Error getting database health metrics: {e}")
            return {"error": str(e)}
    
    async def _get_slow_queries(self, cur) -> List[Dict[str, Any]]:
        """Get slow queries from pg_stat_statements."""
        try:
            # Check if pg_stat_statements extension is available
            await cur.execute("""
                SELECT EXISTS(
                    SELECT 1 FROM pg_extension WHERE extname = 'pg_stat_statements'
                )
            """)
            
            extension_exists = await cur.fetchone()
            
            if not extension_exists[0]:
                return []
            
            # Get slow queries
            await cur.execute("""
                SELECT 
                    query,
                    calls,
                    total_time,
                    mean_time,
                    rows,
                    100.0 * shared_blks_hit / nullif(shared_blks_hit + shared_blks_read, 0) AS hit_percent
                FROM pg_stat_statements 
                WHERE mean_time > 100  -- Queries taking more than 100ms on average
                ORDER BY mean_time DESC 
                LIMIT 10
            """)
            
            slow_queries = await cur.fetchall()
            
            return [
                {
                    "query": row[0][:200] + "..." if len(row[0]) > 200 else row[0],  # Truncate long queries
                    "calls": row[1],
                    "total_time": float(row[2]),
                    "mean_time": float(row[3]),
                    "rows": row[4],
                    "hit_percent": float(row[5]) if row[5] else 0.0
                }
                for row in slow_queries
            ]
        except Exception as e:
            logger.warning(f"Could not get slow queries: {e}")
            return []
    
    async def _get_index_usage_stats(self, cur) -> List[Dict[str, Any]]:
        """Get index usage statistics."""
        try:
            await cur.execute("""
                SELECT 
                    schemaname,
                    tablename,
                    indexname,
                    idx_scan,
                    idx_tup_read,
                    idx_tup_fetch,
                    pg_size_pretty(pg_relation_size(indexrelid)) as index_size
                FROM pg_stat_user_indexes 
                WHERE schemaname = 'public'
                ORDER BY idx_scan DESC
                LIMIT 20
            """)
            
            index_stats = await cur.fetchall()
            
            return [
                {
                    "schema": row[0],
                    "table": row[1],
                    "index": row[2],
                    "scans": row[3],
                    "tuples_read": row[4],
                    "tuples_fetched": row[5],
                    "size": row[6]
                }
                for row in index_stats
            ]
        except Exception as e:
            logger.error(f"Error getting index usage stats: {e}")
            return []
    
    async def _get_table_statistics(self, cur) -> List[Dict[str, Any]]:
        """Get table statistics."""
        try:
            await cur.execute("""
                SELECT 
                    schemaname,
                    tablename,
                    n_tup_ins,
                    n_tup_upd,
                    n_tup_del,
                    n_live_tup,
                    n_dead_tup,
                    last_vacuum,
                    last_autovacuum,
                    last_analyze,
                    last_autoanalyze
                FROM pg_stat_user_tables 
                WHERE schemaname = 'public'
                ORDER BY n_live_tup DESC
            """)
            
            table_stats = await cur.fetchall()
            
            return [
                {
                    "schema": row[0],
                    "table": row[1],
                    "inserts": row[2],
                    "updates": row[3],
                    "deletes": row[4],
                    "live_tuples": row[5],
                    "dead_tuples": row[6],
                    "last_vacuum": row[7].isoformat() if row[7] else None,
                    "last_autovacuum": row[8].isoformat() if row[8] else None,
                    "last_analyze": row[9].isoformat() if row[9] else None,
                    "last_autoanalyze": row[10].isoformat() if row[10] else None
                }
                for row in table_stats
            ]
        except Exception as e:
            logger.error(f"Error getting table statistics: {e}")
            return []
    
    async def _get_connection_stats(self, cur) -> Dict[str, Any]:
        """Get connection statistics."""
        try:
            await cur.execute("""
                SELECT 
                    count(*) as total_connections,
                    count(*) FILTER (WHERE state = 'active') as active_connections,
                    count(*) FILTER (WHERE state = 'idle') as idle_connections,
                    count(*) FILTER (WHERE state = 'idle in transaction') as idle_in_transaction,
                    count(*) FILTER (WHERE state = 'idle in transaction (aborted)') as idle_in_transaction_aborted
                FROM pg_stat_activity
            """)
            
            connection_stats = await cur.fetchone()
            
            return {
                "total": connection_stats[0] if connection_stats else 0,
                "active": connection_stats[1] if connection_stats else 0,
                "idle": connection_stats[2] if connection_stats else 0,
                "idle_in_transaction": connection_stats[3] if connection_stats else 0,
                "idle_in_transaction_aborted": connection_stats[4] if connection_stats else 0
            }
        except Exception as e:
            logger.error(f"Error getting connection stats: {e}")
            return {}
    
    def _format_bytes(self, bytes_value: int) -> str:
        """Format bytes into human readable format."""
        for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
            if bytes_value < 1024.0:
                return f"{bytes_value:.1f} {unit}"
            bytes_value /= 1024.0
        return f"{bytes_value:.1f} PB"

# Global database optimization service instance
_db_optimization_service = None

def get_db_optimization_service() -> DatabaseOptimizationService:
    """Get database optimization service instance."""
    global _db_optimization_service
    if _db_optimization_service is None:
        _db_optimization_service = DatabaseOptimizationService()
    return _db_optimization_service
