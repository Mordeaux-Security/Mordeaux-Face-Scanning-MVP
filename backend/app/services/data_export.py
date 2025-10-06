import csv
import json
import time
import asyncio
from typing import Dict, List, Any, Optional, Union
from datetime import datetime, timedelta
import logging
import psycopg
from io import StringIO, BytesIO
from ..core.config import get_settings

logger = logging.getLogger(__name__)

class DataExportService:
    """Service for exporting data in various formats."""
    
    def __init__(self):
        self.settings = get_settings()
    
    async def export_audit_logs(self, tenant_id: str = None, start_date: datetime = None, 
                               end_date: datetime = None, format: str = "json") -> Dict[str, Any]:
        """Export audit logs in specified format."""
        try:
            connection_string = f"postgresql://{self.settings.postgres_user}:{self.settings.postgres_password}@{self.settings.postgres_host}:{self.settings.postgres_port}/{self.settings.postgres_db}"
            
            async with psycopg.AsyncConnectionPool(connection_string, min_size=1, max_size=2) as pool:
                async with pool.connection() as conn:
                    async with conn.cursor() as cur:
                        # Build query
                        query = """
                            SELECT 
                                id,
                                request_id,
                                tenant_id,
                                method,
                                path,
                                status_code,
                                process_time,
                                user_agent,
                                ip_address,
                                response_size,
                                created_at
                            FROM audit_logs
                            WHERE 1=1
                        """
                        params = []
                        
                        if tenant_id:
                            query += " AND tenant_id = %s"
                            params.append(tenant_id)
                        
                        if start_date:
                            query += " AND created_at >= %s"
                            params.append(start_date.timestamp())
                        
                        if end_date:
                            query += " AND created_at <= %s"
                            params.append(end_date.timestamp())
                        
                        query += " ORDER BY created_at DESC"
                        
                        await cur.execute(query, params)
                        rows = await cur.fetchall()
                        
                        # Convert to list of dictionaries
                        audit_logs = []
                        for row in rows:
                            audit_logs.append({
                                "id": str(row[0]),
                                "request_id": row[1],
                                "tenant_id": row[2],
                                "method": row[3],
                                "path": row[4],
                                "status_code": row[5],
                                "process_time": float(row[6]),
                                "user_agent": row[7],
                                "ip_address": str(row[8]) if row[8] else None,
                                "response_size": row[9],
                                "created_at": datetime.fromtimestamp(row[10]).isoformat()
                            })
                        
                        # Export in requested format
                        if format.lower() == "csv":
                            return await self._export_to_csv(audit_logs, "audit_logs")
                        elif format.lower() == "json":
                            return await self._export_to_json(audit_logs, "audit_logs")
                        else:
                            raise ValueError(f"Unsupported format: {format}")
        
        except Exception as e:
            logger.error(f"Error exporting audit logs: {e}")
            return {"error": str(e)}
    
    async def export_search_audit_logs(self, tenant_id: str = None, start_date: datetime = None,
                                      end_date: datetime = None, format: str = "json") -> Dict[str, Any]:
        """Export search audit logs in specified format."""
        try:
            connection_string = f"postgresql://{self.settings.postgres_user}:{self.settings.postgres_password}@{self.settings.postgres_host}:{self.settings.postgres_port}/{self.settings.postgres_db}"
            
            async with psycopg.AsyncConnectionPool(connection_string, min_size=1, max_size=2) as pool:
                async with pool.connection() as conn:
                    async with conn.cursor() as cur:
                        # Build query
                        query = """
                            SELECT 
                                id,
                                request_id,
                                tenant_id,
                                operation_type,
                                face_count,
                                result_count,
                                vector_backend,
                                created_at
                            FROM search_audit_logs
                            WHERE 1=1
                        """
                        params = []
                        
                        if tenant_id:
                            query += " AND tenant_id = %s"
                            params.append(tenant_id)
                        
                        if start_date:
                            query += " AND created_at >= %s"
                            params.append(start_date.timestamp())
                        
                        if end_date:
                            query += " AND created_at <= %s"
                            params.append(end_date.timestamp())
                        
                        query += " ORDER BY created_at DESC"
                        
                        await cur.execute(query, params)
                        rows = await cur.fetchall()
                        
                        # Convert to list of dictionaries
                        search_logs = []
                        for row in rows:
                            search_logs.append({
                                "id": str(row[0]),
                                "request_id": row[1],
                                "tenant_id": row[2],
                                "operation_type": row[3],
                                "face_count": row[4],
                                "result_count": row[5],
                                "vector_backend": row[6],
                                "created_at": datetime.fromtimestamp(row[7]).isoformat()
                            })
                        
                        # Export in requested format
                        if format.lower() == "csv":
                            return await self._export_to_csv(search_logs, "search_audit_logs")
                        elif format.lower() == "json":
                            return await self._export_to_json(search_logs, "search_audit_logs")
                        else:
                            raise ValueError(f"Unsupported format: {format}")
        
        except Exception as e:
            logger.error(f"Error exporting search audit logs: {e}")
            return {"error": str(e)}
    
    async def export_tenant_data(self, tenant_id: str, format: str = "json") -> Dict[str, Any]:
        """Export all data for a specific tenant."""
        try:
            connection_string = f"postgresql://{self.settings.postgres_user}:{self.settings.postgres_password}@{self.settings.postgres_host}:{self.settings.postgres_port}/{self.settings.postgres_db}"
            
            async with psycopg.AsyncConnectionPool(connection_string, min_size=1, max_size=2) as pool:
                async with pool.connection() as conn:
                    async with conn.cursor() as cur:
                        # Get tenant information
                        await cur.execute("""
                            SELECT tenant_id, name, description, status, metadata, 
                                   EXTRACT(EPOCH FROM created_at) as created_at,
                                   EXTRACT(EPOCH FROM updated_at) as updated_at
                            FROM tenants 
                            WHERE tenant_id = %s
                        """, (tenant_id,))
                        
                        tenant_row = await cur.fetchone()
                        if not tenant_row:
                            return {"error": "Tenant not found"}
                        
                        tenant_info = {
                            "tenant_id": tenant_row[0],
                            "name": tenant_row[1],
                            "description": tenant_row[2],
                            "status": tenant_row[3],
                            "metadata": tenant_row[4] or {},
                            "created_at": datetime.fromtimestamp(tenant_row[5]).isoformat(),
                            "updated_at": datetime.fromtimestamp(tenant_row[6]).isoformat()
                        }
                        
                        # Get images
                        await cur.execute("""
                            SELECT id, tenant_id, raw_key, thumb_key, phash, 
                                   EXTRACT(EPOCH FROM created_at) as created_at
                            FROM images 
                            WHERE tenant_id = %s
                            ORDER BY created_at DESC
                        """, (tenant_id,))
                        
                        image_rows = await cur.fetchall()
                        images = []
                        for row in image_rows:
                            images.append({
                                "id": str(row[0]),
                                "tenant_id": row[1],
                                "raw_key": row[2],
                                "thumb_key": row[3],
                                "phash": row[4],
                                "created_at": datetime.fromtimestamp(row[5]).isoformat()
                            })
                        
                        # Get faces
                        await cur.execute("""
                            SELECT f.id, f.tenant_id, f.image_id, f.face_key, f.embedding_id,
                                   EXTRACT(EPOCH FROM f.created_at) as created_at,
                                   i.raw_key, i.thumb_key
                            FROM faces f
                            JOIN images i ON f.image_id = i.id
                            WHERE f.tenant_id = %s
                            ORDER BY f.created_at DESC
                        """, (tenant_id,))
                        
                        face_rows = await cur.fetchall()
                        faces = []
                        for row in face_rows:
                            faces.append({
                                "id": str(row[0]),
                                "tenant_id": row[1],
                                "image_id": str(row[2]),
                                "face_key": row[3],
                                "embedding_id": row[4],
                                "created_at": datetime.fromtimestamp(row[5]).isoformat(),
                                "image_raw_key": row[6],
                                "image_thumb_key": row[7]
                            })
                        
                        # Get audit logs
                        await cur.execute("""
                            SELECT id, request_id, method, path, status_code, process_time,
                                   user_agent, ip_address, response_size,
                                   EXTRACT(EPOCH FROM created_at) as created_at
                            FROM audit_logs 
                            WHERE tenant_id = %s
                            ORDER BY created_at DESC
                        """, (tenant_id,))
                        
                        audit_rows = await cur.fetchall()
                        audit_logs = []
                        for row in audit_rows:
                            audit_logs.append({
                                "id": str(row[0]),
                                "request_id": row[1],
                                "method": row[2],
                                "path": row[3],
                                "status_code": row[4],
                                "process_time": float(row[5]),
                                "user_agent": row[6],
                                "ip_address": str(row[7]) if row[7] else None,
                                "response_size": row[8],
                                "created_at": datetime.fromtimestamp(row[9]).isoformat()
                            })
                        
                        # Get search audit logs
                        await cur.execute("""
                            SELECT id, request_id, operation_type, face_count, result_count,
                                   vector_backend, EXTRACT(EPOCH FROM created_at) as created_at
                            FROM search_audit_logs 
                            WHERE tenant_id = %s
                            ORDER BY created_at DESC
                        """, (tenant_id,))
                        
                        search_rows = await cur.fetchall()
                        search_logs = []
                        for row in search_rows:
                            search_logs.append({
                                "id": str(row[0]),
                                "request_id": row[1],
                                "operation_type": row[2],
                                "face_count": row[3],
                                "result_count": row[4],
                                "vector_backend": row[5],
                                "created_at": datetime.fromtimestamp(row[6]).isoformat()
                            })
                        
                        # Combine all data
                        export_data = {
                            "export_info": {
                                "tenant_id": tenant_id,
                                "export_timestamp": datetime.now().isoformat(),
                                "format": format,
                                "total_images": len(images),
                                "total_faces": len(faces),
                                "total_audit_logs": len(audit_logs),
                                "total_search_logs": len(search_logs)
                            },
                            "tenant_info": tenant_info,
                            "images": images,
                            "faces": faces,
                            "audit_logs": audit_logs,
                            "search_audit_logs": search_logs
                        }
                        
                        # Export in requested format
                        if format.lower() == "csv":
                            return await self._export_tenant_to_csv(export_data)
                        elif format.lower() == "json":
                            return await self._export_to_json(export_data, f"tenant_{tenant_id}_export")
                        else:
                            raise ValueError(f"Unsupported format: {format}")
        
        except Exception as e:
            logger.error(f"Error exporting tenant data for {tenant_id}: {e}")
            return {"error": str(e)}
    
    async def export_system_metrics(self, start_date: datetime = None, end_date: datetime = None,
                                   format: str = "json") -> Dict[str, Any]:
        """Export system metrics and statistics."""
        try:
            connection_string = f"postgresql://{self.settings.postgres_user}:{self.settings.postgres_password}@{self.settings.postgres_host}:{self.settings.postgres_port}/{self.settings.postgres_db}"
            
            async with psycopg.AsyncConnectionPool(connection_string, min_size=1, max_size=2) as pool:
                async with pool.connection() as conn:
                    async with conn.cursor() as cur:
                        # Build date filter
                        date_filter = ""
                        params = []
                        if start_date:
                            date_filter += " AND created_at >= %s"
                            params.append(start_date.timestamp())
                        if end_date:
                            date_filter += " AND created_at <= %s"
                            params.append(end_date.timestamp())
                        
                        # Get system-wide audit log statistics
                        await cur.execute(f"""
                            SELECT 
                                COUNT(*) as total_requests,
                                COUNT(CASE WHEN status_code >= 400 THEN 1 END) as error_requests,
                                AVG(process_time) as avg_process_time,
                                MIN(process_time) as min_process_time,
                                MAX(process_time) as max_process_time,
                                COUNT(DISTINCT tenant_id) as unique_tenants,
                                COUNT(DISTINCT method) as unique_methods
                            FROM audit_logs
                            WHERE 1=1 {date_filter}
                        """, params)
                        
                        audit_stats = await cur.fetchone()
                        
                        # Get search operation statistics
                        await cur.execute(f"""
                            SELECT 
                                operation_type,
                                COUNT(*) as count,
                                SUM(face_count) as total_faces,
                                SUM(result_count) as total_results,
                                AVG(face_count) as avg_faces_per_operation,
                                AVG(result_count) as avg_results_per_operation
                            FROM search_audit_logs
                            WHERE 1=1 {date_filter}
                            GROUP BY operation_type
                        """, params)
                        
                        search_stats = await cur.fetchall()
                        
                        # Get tenant usage statistics
                        await cur.execute(f"""
                            SELECT 
                                tenant_id,
                                COUNT(*) as request_count,
                                COUNT(CASE WHEN status_code >= 400 THEN 1 END) as error_count,
                                AVG(process_time) as avg_process_time
                            FROM audit_logs
                            WHERE 1=1 {date_filter}
                            GROUP BY tenant_id
                            ORDER BY request_count DESC
                        """, params)
                        
                        tenant_stats = await cur.fetchall()
                        
                        # Get hourly request distribution
                        await cur.execute(f"""
                            SELECT 
                                EXTRACT(HOUR FROM to_timestamp(created_at)) as hour,
                                COUNT(*) as request_count
                            FROM audit_logs
                            WHERE 1=1 {date_filter}
                            GROUP BY EXTRACT(HOUR FROM to_timestamp(created_at))
                            ORDER BY hour
                        """, params)
                        
                        hourly_stats = await cur.fetchall()
                        
                        # Combine metrics
                        metrics_data = {
                            "export_info": {
                                "export_timestamp": datetime.now().isoformat(),
                                "start_date": start_date.isoformat() if start_date else None,
                                "end_date": end_date.isoformat() if end_date else None,
                                "format": format
                            },
                            "audit_statistics": {
                                "total_requests": audit_stats[0] if audit_stats else 0,
                                "error_requests": audit_stats[1] if audit_stats else 0,
                                "error_rate": (audit_stats[1] / max(audit_stats[0], 1)) if audit_stats else 0.0,
                                "avg_process_time": float(audit_stats[2]) if audit_stats and audit_stats[2] else 0.0,
                                "min_process_time": float(audit_stats[3]) if audit_stats and audit_stats[3] else 0.0,
                                "max_process_time": float(audit_stats[4]) if audit_stats and audit_stats[4] else 0.0,
                                "unique_tenants": audit_stats[5] if audit_stats else 0,
                                "unique_methods": audit_stats[6] if audit_stats else 0
                            },
                            "search_operations": {
                                row[0]: {
                                    "count": row[1],
                                    "total_faces": row[2],
                                    "total_results": row[3],
                                    "avg_faces_per_operation": float(row[4]) if row[4] else 0.0,
                                    "avg_results_per_operation": float(row[5]) if row[5] else 0.0
                                }
                                for row in search_stats
                            },
                            "tenant_usage": [
                                {
                                    "tenant_id": row[0],
                                    "request_count": row[1],
                                    "error_count": row[2],
                                    "error_rate": row[2] / max(row[1], 1),
                                    "avg_process_time": float(row[3]) if row[3] else 0.0
                                }
                                for row in tenant_stats
                            ],
                            "hourly_distribution": [
                                {
                                    "hour": int(row[0]),
                                    "request_count": row[1]
                                }
                                for row in hourly_stats
                            ]
                        }
                        
                        # Export in requested format
                        if format.lower() == "csv":
                            return await self._export_metrics_to_csv(metrics_data)
                        elif format.lower() == "json":
                            return await self._export_to_json(metrics_data, "system_metrics")
                        else:
                            raise ValueError(f"Unsupported format: {format}")
        
        except Exception as e:
            logger.error(f"Error exporting system metrics: {e}")
            return {"error": str(e)}
    
    async def _export_to_json(self, data: Union[List[Dict], Dict], filename: str) -> Dict[str, Any]:
        """Export data to JSON format."""
        try:
            json_content = json.dumps(data, indent=2, ensure_ascii=False)
            
            return {
                "filename": f"{filename}_{int(time.time())}.json",
                "content_type": "application/json",
                "content": json_content,
                "size_bytes": len(json_content.encode('utf-8')),
                "format": "json",
                "export_timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Error exporting to JSON: {e}")
            return {"error": str(e)}
    
    async def _export_to_csv(self, data: List[Dict], filename: str) -> Dict[str, Any]:
        """Export data to CSV format."""
        try:
            if not data:
                return {
                    "filename": f"{filename}_{int(time.time())}.csv",
                    "content_type": "text/csv",
                    "content": "",
                    "size_bytes": 0,
                    "format": "csv",
                    "export_timestamp": datetime.now().isoformat()
                }
            
            # Create CSV content
            output = StringIO()
            fieldnames = data[0].keys()
            writer = csv.DictWriter(output, fieldnames=fieldnames)
            
            writer.writeheader()
            for row in data:
                writer.writerow(row)
            
            csv_content = output.getvalue()
            output.close()
            
            return {
                "filename": f"{filename}_{int(time.time())}.csv",
                "content_type": "text/csv",
                "content": csv_content,
                "size_bytes": len(csv_content.encode('utf-8')),
                "format": "csv",
                "export_timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Error exporting to CSV: {e}")
            return {"error": str(e)}
    
    async def _export_tenant_to_csv(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Export tenant data to CSV format with multiple sheets."""
        try:
            # Create a ZIP-like structure with multiple CSV files
            csv_files = {}
            
            # Export each section as separate CSV
            for section_name, section_data in data.items():
                if isinstance(section_data, list) and section_data:
                    csv_files[f"{section_name}.csv"] = await self._export_to_csv(section_data, section_name)
                elif isinstance(section_data, dict):
                    # Convert single dict to list for CSV export
                    csv_files[f"{section_name}.csv"] = await self._export_to_csv([section_data], section_name)
            
            # For now, return the first CSV file (in a real implementation, you'd create a ZIP)
            if csv_files:
                first_file = list(csv_files.values())[0]
                first_file["additional_files"] = list(csv_files.keys())
                return first_file
            else:
                return {
                    "filename": f"tenant_export_{int(time.time())}.csv",
                    "content_type": "text/csv",
                    "content": "",
                    "size_bytes": 0,
                    "format": "csv",
                    "export_timestamp": datetime.now().isoformat()
                }
        except Exception as e:
            logger.error(f"Error exporting tenant data to CSV: {e}")
            return {"error": str(e)}
    
    async def _export_metrics_to_csv(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Export metrics data to CSV format."""
        try:
            # Flatten the metrics data for CSV export
            flattened_data = []
            
            # Add export info
            flattened_data.append({
                "section": "export_info",
                "key": "export_timestamp",
                "value": data["export_info"]["export_timestamp"]
            })
            
            # Add audit statistics
            for key, value in data["audit_statistics"].items():
                flattened_data.append({
                    "section": "audit_statistics",
                    "key": key,
                    "value": value
                })
            
            # Add search operations
            for operation, stats in data["search_operations"].items():
                for key, value in stats.items():
                    flattened_data.append({
                        "section": f"search_operations_{operation}",
                        "key": key,
                        "value": value
                    })
            
            # Add tenant usage
            for i, tenant in enumerate(data["tenant_usage"]):
                for key, value in tenant.items():
                    flattened_data.append({
                        "section": f"tenant_usage_{i}",
                        "key": key,
                        "value": value
                    })
            
            # Add hourly distribution
            for hour_data in data["hourly_distribution"]:
                for key, value in hour_data.items():
                    flattened_data.append({
                        "section": "hourly_distribution",
                        "key": f"hour_{hour_data['hour']}_{key}",
                        "value": value
                    })
            
            return await self._export_to_csv(flattened_data, "system_metrics")
        except Exception as e:
            logger.error(f"Error exporting metrics to CSV: {e}")
            return {"error": str(e)}

# Global data export service instance
_data_export_service = None

def get_data_export_service() -> DataExportService:
    """Get data export service instance."""
    global _data_export_service
    if _data_export_service is None:
        _data_export_service = DataExportService()
    return _data_export_service
