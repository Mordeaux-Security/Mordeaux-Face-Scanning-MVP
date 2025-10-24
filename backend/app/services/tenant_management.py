import time
import uuid
import json
from typing import Dict, List, Any, Optional
from datetime import datetime
import logging
import psycopg


from ..core.config import get_settings

logger = logging.getLogger(__name__)

class TenantInfo:
    """Represents tenant information."""

    def __init__(self, tenant_id: str, name: str, description: str = "",
                 created_at: float = None, updated_at: float = None,
                 status: str = "active", metadata: Dict[str, Any] = None):
        self.tenant_id = tenant_id
        self.name = name
        self.description = description
        self.created_at = created_at or time.time()
        self.updated_at = updated_at or time.time()
        self.status = status  # active, suspended, deleted
        self.metadata = metadata or {}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tenant_id": self.tenant_id,
            "name": self.name,
            "description": self.description,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "status": self.status,
            "metadata": self.metadata
        }

class TenantManagementService:
    """Service for managing tenants and their configurations."""

    def __init__(self):
        self.settings = get_settings()
        self._ensure_tenant_table()

    def _ensure_tenant_table(self):
        """Ensure the tenants table exists."""
        try:
            connection_string = f"postgresql://{self.settings.postgres_user}:{self.settings.postgres_password}@{self.settings.postgres_host}:{self.settings.postgres_port}/{self.settings.postgres_db}"

            with psycopg.connect(connection_string) as conn:
                with conn.cursor() as cur:
                    # Create tenants table if it doesn't exist
                    cur.execute("""
                        CREATE TABLE IF NOT EXISTS tenants (
                            tenant_id TEXT PRIMARY KEY,
                            name TEXT NOT NULL,
                            description TEXT DEFAULT '',
                            status TEXT DEFAULT 'active' CHECK (status IN ('active', 'suspended', 'deleted')),
                            metadata JSONB DEFAULT '{}',
                            created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                            updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
                        )
                    """)

                    # Create indexes
                    cur.execute("CREATE INDEX IF NOT EXISTS idx_tenants_status ON tenants(status)")
                    cur.execute("CREATE INDEX IF NOT EXISTS idx_tenants_created_at ON tenants(created_at)")
                    cur.execute("CREATE INDEX IF NOT EXISTS idx_tenants_name ON tenants(name)")

                    conn.commit()
                    logger.info("Tenants table and indexes created successfully")
        except Exception as e:
            logger.error(f"Error creating tenants table: {e}")
            raise

    async def create_tenant(self, name: str, description: str = "",
                           metadata: Dict[str, Any] = None) -> TenantInfo:
        """Create a new tenant."""
        try:
            tenant_id = f"tenant_{uuid.uuid4().hex[:12]}"
            current_time = time.time()

            connection_string = f"postgresql://{self.settings.postgres_user}:{self.settings.postgres_password}@{self.settings.postgres_host}:{self.settings.postgres_port}/{self.settings.postgres_db}"

            with psycopg.connect(connection_string) as conn:
                with conn.cursor() as cur:
                    # Insert new tenant
                    cur.execute("""
                        INSERT INTO tenants (tenant_id, name, description, metadata, created_at, updated_at)
                        VALUES (%s, %s, %s, %s, %s, %s)
                    """, (tenant_id, name, description, json.dumps(metadata or {}),
                          datetime.fromtimestamp(current_time), datetime.fromtimestamp(current_time)))

                    conn.commit()

                    logger.info(f"Created new tenant: {tenant_id} ({name})")

                    return TenantInfo(
                        tenant_id=tenant_id,
                        name=name,
                        description=description,
                        created_at=current_time,
                        updated_at=current_time,
                        metadata=metadata or {}
                    )
        except Exception as e:
            logger.error(f"Error creating tenant: {e}")
            raise

    async def get_tenant(self, tenant_id: str) -> Optional[TenantInfo]:
        """Get tenant information by ID."""
        try:
            connection_string = f"postgresql://{self.settings.postgres_user}:{self.settings.postgres_password}@{self.settings.postgres_host}:{self.settings.postgres_port}/{self.settings.postgres_db}"

            with psycopg.connect(connection_string) as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        SELECT tenant_id, name, description, status, metadata,
                               EXTRACT(EPOCH FROM created_at) as created_at,
                               EXTRACT(EPOCH FROM updated_at) as updated_at
                        FROM tenants
                        WHERE tenant_id = %s AND status != 'deleted'
                    """, (tenant_id,))

                    row = cur.fetchone()

                    if row:
                        return TenantInfo(
                            tenant_id=row[0],
                            name=row[1],
                            description=row[2],
                            status=row[3],
                            metadata=row[4] or {},
                            created_at=float(row[5]),
                            updated_at=float(row[6])
                        )

                    return None
        except Exception as e:
            logger.error(f"Error getting tenant {tenant_id}: {e}")
            raise

    async def list_tenants(self, status: str = None, limit: int = 100, offset: int = 0) -> List[TenantInfo]:
        """List tenants with optional filtering."""
        try:
            connection_string = f"postgresql://{self.settings.postgres_user}:{self.settings.postgres_password}@{self.settings.postgres_host}:{self.settings.postgres_port}/{self.settings.postgres_db}"

            async with psycopg.AsyncConnectionPool(connection_string, min_size=1, max_size=2) as pool:
                async with pool.connection() as conn:
                    async with conn.cursor() as cur:
                        query = """
                            SELECT tenant_id, name, description, status, metadata,
                                   EXTRACT(EPOCH FROM created_at) as created_at,
                                   EXTRACT(EPOCH FROM updated_at) as updated_at
                            FROM tenants
                            WHERE status != 'deleted'
                        """
                        params = []

                        if status:
                            query += " AND status = %s"
                            params.append(status)

                        query += " ORDER BY created_at DESC LIMIT %s OFFSET %s"
                        params.extend([limit, offset])

                        await cur.execute(query, params)
                        rows = await cur.fetchall()

                        return [
                            TenantInfo(
                                tenant_id=row[0],
                                name=row[1],
                                description=row[2],
                                status=row[3],
                                metadata=row[4] or {},
                                created_at=float(row[5]),
                                updated_at=float(row[6])
                            )
                            for row in rows
                        ]
        except Exception as e:
            logger.error(f"Error listing tenants: {e}")
            raise

    async def update_tenant(self, tenant_id: str, name: str = None,
                           description: str = None, metadata: Dict[str, Any] = None) -> Optional[TenantInfo]:
        """Update tenant information."""
        try:
            current_time = time.time()

            connection_string = f"postgresql://{self.settings.postgres_user}:{self.settings.postgres_password}@{self.settings.postgres_host}:{self.settings.postgres_port}/{self.settings.postgres_db}"

            async with psycopg.AsyncConnectionPool(connection_string, min_size=1, max_size=2) as pool:
                async with pool.connection() as conn:
                    async with conn.cursor() as cur:
                        # Build update query dynamically
                        update_fields = []
                        params = []

                        if name is not None:
                            update_fields.append("name = %s")
                            params.append(name)

                        if description is not None:
                            update_fields.append("description = %s")
                            params.append(description)

                        if metadata is not None:
                            update_fields.append("metadata = %s")
                            params.append(metadata)

                        if not update_fields:
                            # No fields to update, just return current tenant
                            return await self.get_tenant(tenant_id)

                        update_fields.append("updated_at = %s")
                        params.append(current_time)
                        params.append(tenant_id)

                        query = f"""
                            UPDATE tenants
                            SET {', '.join(update_fields)}
                            WHERE tenant_id = %s AND status != 'deleted'
                        """

                        await cur.execute(query, params)

                        if cur.rowcount == 0:
                            return None

                        await conn.commit()

                        logger.info(f"Updated tenant: {tenant_id}")

                        return await self.get_tenant(tenant_id)
        except Exception as e:
            logger.error(f"Error updating tenant {tenant_id}: {e}")
            raise

    async def suspend_tenant(self, tenant_id: str) -> bool:
        """Suspend a tenant."""
        try:
            connection_string = f"postgresql://{self.settings.postgres_user}:{self.settings.postgres_password}@{self.settings.postgres_host}:{self.settings.postgres_port}/{self.settings.postgres_db}"

            async with psycopg.AsyncConnectionPool(connection_string, min_size=1, max_size=2) as pool:
                async with pool.connection() as conn:
                    async with conn.cursor() as cur:
                        await cur.execute("""
                            UPDATE tenants
                            SET status = 'suspended', updated_at = %s
                            WHERE tenant_id = %s AND status != 'deleted'
                        """, (time.time(), tenant_id))

                        success = cur.rowcount > 0
                        await conn.commit()

                        if success:
                            logger.info(f"Suspended tenant: {tenant_id}")

                        return success
        except Exception as e:
            logger.error(f"Error suspending tenant {tenant_id}: {e}")
            raise

    async def activate_tenant(self, tenant_id: str) -> bool:
        """Activate a suspended tenant."""
        try:
            connection_string = f"postgresql://{self.settings.postgres_user}:{self.settings.postgres_password}@{self.settings.postgres_host}:{self.settings.postgres_port}/{self.settings.postgres_db}"

            async with psycopg.AsyncConnectionPool(connection_string, min_size=1, max_size=2) as pool:
                async with pool.connection() as conn:
                    async with conn.cursor() as cur:
                        await cur.execute("""
                            UPDATE tenants
                            SET status = 'active', updated_at = %s
                            WHERE tenant_id = %s AND status != 'deleted'
                        """, (time.time(), tenant_id))

                        success = cur.rowcount > 0
                        await conn.commit()

                        if success:
                            logger.info(f"Activated tenant: {tenant_id}")

                        return success
        except Exception as e:
            logger.error(f"Error activating tenant {tenant_id}: {e}")
            raise

    async def delete_tenant(self, tenant_id: str, hard_delete: bool = False) -> bool:
        """Delete a tenant (soft delete by default)."""
        try:
            connection_string = f"postgresql://{self.settings.postgres_user}:{self.settings.postgres_password}@{self.settings.postgres_host}:{self.settings.postgres_port}/{self.settings.postgres_db}"

            async with psycopg.AsyncConnectionPool(connection_string, min_size=1, max_size=2) as pool:
                async with pool.connection() as conn:
                    async with conn.cursor() as cur:
                        if hard_delete:
                            # Hard delete - remove from database
                            await cur.execute("DELETE FROM tenants WHERE tenant_id = %s", (tenant_id,))
                        else:
                            # Soft delete - mark as deleted
                            await cur.execute("""
                                UPDATE tenants
                                SET status = 'deleted', updated_at = %s
                                WHERE tenant_id = %s
                            """, (time.time(), tenant_id))

                        success = cur.rowcount > 0
                        await conn.commit()

                        if success:
                            logger.info(f"Deleted tenant: {tenant_id} (hard_delete={hard_delete})")

                        return success
        except Exception as e:
            logger.error(f"Error deleting tenant {tenant_id}: {e}")
            raise

    async def get_tenant_stats(self, tenant_id: str) -> Dict[str, Any]:
        """Get statistics for a specific tenant."""
        try:
            connection_string = f"postgresql://{self.settings.postgres_user}:{self.settings.postgres_password}@{self.settings.postgres_host}:{self.settings.postgres_port}/{self.settings.postgres_db}"

            async with psycopg.AsyncConnectionPool(connection_string, min_size=1, max_size=2) as pool:
                async with pool.connection() as conn:
                    async with conn.cursor() as cur:
                        # Get audit log statistics
                        await cur.execute("""
                            SELECT
                                COUNT(*) as total_requests,
                                COUNT(CASE WHEN status_code >= 400 THEN 1 END) as error_requests,
                                AVG(process_time) as avg_process_time,
                                MIN(created_at) as first_request,
                                MAX(created_at) as last_request
                            FROM audit_logs
                            WHERE tenant_id = %s
                        """, (tenant_id,))

                        audit_stats = await cur.fetchone()

                        # Get search audit statistics
                        await cur.execute("""
                            SELECT
                                operation_type,
                                COUNT(*) as count,
                                SUM(face_count) as total_faces,
                                SUM(result_count) as total_results
                            FROM search_audit_logs
                            WHERE tenant_id = %s
                            GROUP BY operation_type
                        """, (tenant_id,))

                        search_stats = await cur.fetchall()

                        return {
                            "tenant_id": tenant_id,
                            "timestamp": time.time(),
                            "audit_logs": {
                                "total_requests": audit_stats[0] if audit_stats else 0,
                                "error_requests": audit_stats[1] if audit_stats else 0,
                                "avg_process_time": float(audit_stats[2]) if audit_stats and audit_stats[2] else 0.0,
                                "first_request": audit_stats[3].isoformat() if audit_stats and audit_stats[3] else None,
                                "last_request": audit_stats[4].isoformat() if audit_stats and audit_stats[4] else None
                            },
                            "search_operations": {
                                row[0]: {
                                    "count": row[1],
                                    "total_faces": row[2],
                                    "total_results": row[3]
                                }
                                for row in search_stats
                            }
                        }
        except Exception as e:
            logger.error(f"Error getting tenant stats for {tenant_id}: {e}")
            return {"error": str(e)}

    async def get_tenant_usage_summary(self) -> Dict[str, Any]:
        """Get usage summary across all tenants."""
        try:
            connection_string = f"postgresql://{self.settings.postgres_user}:{self.settings.postgres_password}@{self.settings.postgres_host}:{self.settings.postgres_port}/{self.settings.postgres_db}"

            async with psycopg.AsyncConnectionPool(connection_string, min_size=1, max_size=2) as pool:
                async with pool.connection() as conn:
                    async with conn.cursor() as cur:
                        # Get tenant counts by status
                        await cur.execute("""
                            SELECT status, COUNT(*)
                            FROM tenants
                            GROUP BY status
                        """)

                        status_counts = dict(await cur.fetchall())

                        # Get total requests by tenant
                        await cur.execute("""
                            SELECT
                                t.tenant_id,
                                t.name,
                                t.status,
                                COUNT(al.id) as total_requests,
                                COUNT(CASE WHEN al.status_code >= 400 THEN 1 END) as error_requests
                            FROM tenants t
                            LEFT JOIN audit_logs al ON t.tenant_id = al.tenant_id
                            WHERE t.status != 'deleted'
                            GROUP BY t.tenant_id, t.name, t.status
                            ORDER BY total_requests DESC
                            LIMIT 20
                        """)

                        tenant_usage = await cur.fetchall()

                        return {
                            "timestamp": time.time(),
                            "tenant_counts": status_counts,
                            "top_tenants_by_usage": [
                                {
                                    "tenant_id": row[0],
                                    "name": row[1],
                                    "status": row[2],
                                    "total_requests": row[3],
                                    "error_requests": row[4]
                                }
                                for row in tenant_usage
                            ]
                        }
        except Exception as e:
            logger.error(f"Error getting tenant usage summary: {e}")
            return {"error": str(e)}

# Global tenant management service instance
_tenant_management_service = None

def get_tenant_management_service() -> TenantManagementService:
    """Get tenant management service instance."""
    global _tenant_management_service
    if _tenant_management_service is None:
        _tenant_management_service = TenantManagementService()
    return _tenant_management_service
