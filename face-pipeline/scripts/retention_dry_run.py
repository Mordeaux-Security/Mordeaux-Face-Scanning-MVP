#!/usr/bin/env python3
"""
Retention Dry-Run Script

Identifies MinIO artifacts (crops, thumbs, metadata) older than N days for potential cleanup.
Runs in dry-run mode only - no actual deletions are performed.

Usage:
    python retention_dry_run.py --days 30 --preview-limit 25
    python retention_dry_run.py --days 7  # 7 days retention
    python retention_dry_run.py --help

Output: JSON report with candidate counts and preview of oldest artifacts.
"""

import argparse
import json
import sys
import os
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Any

# Add the parent directory to the Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline.storage import get_client
from config.settings import settings


def format_timestamp(dt: datetime) -> str:
    """Format datetime as ISO string with timezone."""
    return dt.isoformat().replace('+00:00', 'Z')


def analyze_bucket(client, bucket_name: str, cutoff_date: datetime) -> Dict[str, Any]:
    """
    Analyze a single MinIO bucket for old artifacts.
    
    Args:
        client: MinIO client
        bucket_name: Name of the bucket to analyze
        cutoff_date: Objects older than this date are candidates
        
    Returns:
        Dictionary with count, size, and candidate objects
    """
    candidates = []
    total_size = 0
    
    try:
        # List all objects in the bucket
        objects = client.list_objects(bucket_name, recursive=True)
        
        for obj in objects:
            # Check if object is older than cutoff
            if obj.last_modified < cutoff_date:
                candidate = {
                    "key": obj.object_name,
                    "size": obj.size,
                    "last_modified": format_timestamp(obj.last_modified)
                }
                candidates.append(candidate)
                total_size += obj.size
                
    except Exception as e:
        print(f"Error analyzing bucket {bucket_name}: {e}", file=sys.stderr)
        return {"count": 0, "size_bytes": 0, "candidates": []}
    
    # Sort by last_modified (oldest first)
    candidates.sort(key=lambda x: x["last_modified"])
    
    return {
        "count": len(candidates),
        "size_bytes": total_size,
        "candidates": candidates
    }


def main():
    """Main entry point for retention dry-run script."""
    parser = argparse.ArgumentParser(
        description="Identify old MinIO artifacts for potential cleanup (dry-run only)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python retention_dry_run.py --days 30
  python retention_dry_run.py --days 7 --preview-limit 10
  python retention_dry_run.py --days 90 --preview-limit 50
        """
    )
    
    parser.add_argument(
        "--days",
        type=int,
        default=30,
        help="Number of days for retention cutoff (default: 30)"
    )
    
    parser.add_argument(
        "--preview-limit",
        type=int,
        default=25,
        help="Number of candidates to include in preview (default: 25)"
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.days < 0:
        print("Error: --days must be non-negative", file=sys.stderr)
        sys.exit(1)
    
    if args.preview_limit < 0:
        print("Error: --preview-limit must be non-negative", file=sys.stderr)
        sys.exit(1)
    
    # Calculate cutoff date
    cutoff_date = datetime.now(timezone.utc) - timedelta(days=args.days)
    
    # Get MinIO client
    try:
        client = get_client()
    except Exception as e:
        print(f"Error connecting to MinIO: {e}", file=sys.stderr)
        sys.exit(1)
    
    # Define buckets to analyze
    buckets = {
        "face-crops": settings.MINIO_BUCKET_CROPS,
        "thumbnails": settings.MINIO_BUCKET_THUMBS,
        "face-metadata": settings.MINIO_BUCKET_METADATA
    }
    
    # Analyze each bucket
    results = {}
    total_candidates = 0
    total_size = 0
    all_candidates = []
    
    for bucket_display_name, bucket_name in buckets.items():
        print(f"Analyzing bucket: {bucket_name}", file=sys.stderr)
        
        bucket_result = analyze_bucket(client, bucket_name, cutoff_date)
        
        # Add bucket name to each candidate for preview
        for candidate in bucket_result["candidates"]:
            candidate["bucket"] = bucket_display_name
            all_candidates.append(candidate)
        
        results[bucket_display_name] = {
            "count": bucket_result["count"],
            "size_bytes": bucket_result["size_bytes"]
        }
        
        total_candidates += bucket_result["count"]
        total_size += bucket_result["size_bytes"]
    
    # Sort all candidates by last_modified (oldest first)
    all_candidates.sort(key=lambda x: x["last_modified"])
    
    # Create preview (first N candidates)
    preview = all_candidates[:args.preview_limit]
    
    # Build final report
    report = {
        "cutoff_date": format_timestamp(cutoff_date),
        "days": args.days,
        "total_candidates": total_candidates,
        "total_size_bytes": total_size,
        "by_bucket": results,
        "preview": preview
    }
    
    # Output JSON report
    print(json.dumps(report, indent=2))
    
    # Print summary to stderr for visibility
    print(f"\nSummary:", file=sys.stderr)
    print(f"  Cutoff date: {format_timestamp(cutoff_date)}", file=sys.stderr)
    print(f"  Total candidates: {total_candidates}", file=sys.stderr)
    print(f"  Total size: {total_size:,} bytes ({total_size / (1024*1024):.1f} MB)", file=sys.stderr)
    print(f"  Preview: {len(preview)} of {total_candidates} candidates", file=sys.stderr)


if __name__ == "__main__":
    main()
