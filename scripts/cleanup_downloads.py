#!/usr/bin/env python3
"""
Cleanup script for removing download directories created by download_images.sh

This script removes the 'flat' and 'zips' directories that are created when
running 'make download-both' or the download_images.sh script.

Usage:
    python scripts/cleanup_downloads.py [--dry-run] [--verbose]

Options:
    --dry-run    Show what would be deleted without actually deleting
    --verbose    Show detailed output
"""

import argparse
import os
import shutil
import sys
from pathlib import Path


def cleanup_downloads(dry_run=False, verbose=False):
    """Remove flat and zips directories."""
    base_dir = Path.cwd()
    directories_to_remove = ['flat', 'zips']
    
    if verbose:
        print("=== Download Directory Cleanup ===")
        print(f"Working directory: {base_dir}")
        print(f"Dry run mode: {dry_run}")
        print()
    
    total_size = 0
    files_removed = 0
    dirs_removed = 0
    
    for dir_name in directories_to_remove:
        dir_path = base_dir / dir_name
        
        if not dir_path.exists():
            if verbose:
                print(f"  - {dir_name}/ directory not found")
            continue
        
        # Calculate size and file count
        dir_size = 0
        dir_files = 0
        
        if dir_path.is_dir():
            for root, dirs, files in os.walk(dir_path):
                for file in files:
                    file_path = Path(root) / file
                    try:
                        dir_size += file_path.stat().st_size
                        dir_files += 1
                    except (OSError, FileNotFoundError):
                        # File might have been deleted between listing and stat
                        pass
        
        total_size += dir_size
        files_removed += dir_files
        
        if verbose:
            size_mb = dir_size / (1024 * 1024)
            print(f"  - {dir_name}/ directory found:")
            print(f"    - Files: {dir_files}")
            print(f"    - Size: {size_mb:.2f} MB")
        
        if dry_run:
            if verbose:
                print(f"    - Would remove: {dir_path}")
            dirs_removed += 1
        else:
            try:
                shutil.rmtree(dir_path)
                if verbose:
                    print(f"    ✓ Removed: {dir_path}")
                dirs_removed += 1
            except Exception as e:
                print(f"    ✗ Error removing {dir_path}: {e}")
                return False
    
    # Summary
    if verbose or dry_run:
        print()
        size_mb = total_size / (1024 * 1024)
        if dry_run:
            print("=== Dry Run Summary ===")
            print(f"Would remove {dirs_removed} directories")
            print(f"Would remove {files_removed} files")
            print(f"Would free {size_mb:.2f} MB of space")
        else:
            print("=== Cleanup Complete ===")
            print(f"Removed {dirs_removed} directories")
            print(f"Removed {files_removed} files")
            print(f"Freed {size_mb:.2f} MB of space")
    else:
        if dirs_removed > 0:
            print("Download directories cleaned.")
        else:
            print("No download directories found to clean.")
    
    return True


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Clean up download directories (flat/ and zips/)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python scripts/cleanup_downloads.py
    python scripts/cleanup_downloads.py --dry-run
    python scripts/cleanup_downloads.py --verbose
        """
    )
    
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be deleted without actually deleting'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Show detailed output'
    )
    
    args = parser.parse_args()
    
    try:
        success = cleanup_downloads(dry_run=args.dry_run, verbose=args.verbose)
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\nOperation cancelled by user.")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
