"""
Migration script to update tenant_id for crawled faces in Qdrant.

This script updates all points with tenant_id="crawler" to tenant_id="demo-tenant"
so they appear in website search results.

Usage:
    cd face-pipeline
    python -m scripts.migrate_tenant_id --dry-run    # Preview changes
    python -m scripts.migrate_tenant_id              # Apply changes
"""

import argparse
import time
from typing import List

from pipeline.indexer import get_client
from config.settings import settings


FACES_COLLECTION = settings.QDRANT_COLLECTION
OLD_TENANT = "crawler"
NEW_TENANT = "demo-tenant"
BATCH_SIZE = 100


def count_points_by_tenant(client, tenant_id: str) -> int:
    """Count points with a specific tenant_id."""
    from qdrant_client.http.models import Filter, FieldCondition, MatchValue
    
    result = client.count(
        collection_name=FACES_COLLECTION,
        count_filter=Filter(
            must=[FieldCondition(key="tenant_id", match=MatchValue(value=tenant_id))]
        ),
        exact=True,
    )
    return result.count


def get_points_by_tenant(client, tenant_id: str, limit: int = 100, offset=None) -> tuple:
    """Get points with a specific tenant_id."""
    from qdrant_client.http.models import Filter, FieldCondition, MatchValue
    
    points, next_offset = client.scroll(
        collection_name=FACES_COLLECTION,
        scroll_filter=Filter(
            must=[FieldCondition(key="tenant_id", match=MatchValue(value=tenant_id))]
        ),
        limit=limit,
        offset=offset,
        with_payload=True,
        with_vectors=False,
    )
    return points, next_offset


def migrate_tenant_ids(dry_run: bool = True) -> dict:
    """
    Migrate all points from OLD_TENANT to NEW_TENANT.
    
    Returns stats dict with counts.
    """
    client = get_client()
    
    # Count points to migrate
    old_count = count_points_by_tenant(client, OLD_TENANT)
    new_count = count_points_by_tenant(client, NEW_TENANT)
    
    print(f"\n=== Tenant ID Migration ===")
    print(f"Collection: {FACES_COLLECTION}")
    print(f"Points with tenant_id='{OLD_TENANT}': {old_count}")
    print(f"Points with tenant_id='{NEW_TENANT}': {new_count}")
    print()
    
    if old_count == 0:
        print("No points to migrate. Done!")
        return {"migrated": 0, "errors": 0}
    
    if dry_run:
        print(f"DRY RUN: Would migrate {old_count} points from '{OLD_TENANT}' to '{NEW_TENANT}'")
        print("Run without --dry-run to apply changes.")
        return {"migrated": 0, "errors": 0, "would_migrate": old_count}
    
    print(f"Migrating {old_count} points from '{OLD_TENANT}' to '{NEW_TENANT}'...")
    
    migrated = 0
    errors = 0
    offset = None
    start_time = time.time()
    
    while True:
        points, offset = get_points_by_tenant(client, OLD_TENANT, limit=BATCH_SIZE, offset=offset)
        
        if not points:
            break
        
        # Update payload for each point
        point_ids = [p.id for p in points]
        
        try:
            # Use set_payload to update tenant_id
            client.set_payload(
                collection_name=FACES_COLLECTION,
                payload={"tenant_id": NEW_TENANT},
                points=point_ids,
            )
            migrated += len(point_ids)
            
            # Progress update
            elapsed = time.time() - start_time
            rate = migrated / elapsed if elapsed > 0 else 0
            print(f"  Migrated {migrated}/{old_count} points ({rate:.1f}/sec)")
            
        except Exception as e:
            print(f"  ERROR updating batch: {e}")
            errors += len(point_ids)
        
        if offset is None:
            break
    
    # Verify migration
    final_old = count_points_by_tenant(client, OLD_TENANT)
    final_new = count_points_by_tenant(client, NEW_TENANT)
    
    print()
    print(f"=== Migration Complete ===")
    print(f"Migrated: {migrated} points")
    print(f"Errors: {errors}")
    print(f"Points now with tenant_id='{OLD_TENANT}': {final_old}")
    print(f"Points now with tenant_id='{NEW_TENANT}': {final_new}")
    
    return {"migrated": migrated, "errors": errors}


def main():
    parser = argparse.ArgumentParser(
        description="Migrate tenant_id from 'crawler' to 'demo-tenant' in Qdrant"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview changes without applying them"
    )
    args = parser.parse_args()
    
    migrate_tenant_ids(dry_run=args.dry_run)


if __name__ == "__main__":
    main()

