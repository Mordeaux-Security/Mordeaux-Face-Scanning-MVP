#!/usr/bin/env python3
"""
Iterative Crawler Test Runner

Automates running the crawler multiple times with debug cleanup.
Deletes debug folder before each run to ensure fresh debug files.
Enables rapid iteration and early failure detection.
"""

import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

def clean_debug_folder():
    """Delete the debug folder to ensure fresh debug files."""
    debug_dir = Path("backend/crawl_output/debug")
    if debug_dir.exists():
        print(f"Cleaning debug folder: {debug_dir}")
        shutil.rmtree(debug_dir)
        print("Debug folder cleaned")
    else:
        print(f"Debug folder does not exist: {debug_dir}")

def run_crawler():
    """Run the crawler and capture output."""
    print("\n" + "="*80)
    print("Starting crawler run...")
    print("="*80 + "\n")
    
    # Run docker compose and tee to debugamd.txt
    cmd = ["docker", "compose", "up", "new-crawler"]
    
    try:
        # Open debugamd.txt for writing
        with open("debugamd.txt", "w", encoding="utf-8") as log_file:
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True
            )
            
            # Stream output to both file and console
            for line in process.stdout:
                log_file.write(line)
                log_file.flush()
                print(line, end='')
            
            process.wait()
            return process.returncode
            
    except KeyboardInterrupt:
        print("\n\nCrawler interrupted by user")
        return 130
    except Exception as e:
        print(f"\n\nError running crawler: {e}")
        return 1

def main():
    """Main entry point for iterative crawler runs."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Iteratively run crawler with debug cleanup"
    )
    parser.add_argument(
        "-n", "--iterations",
        type=int,
        default=1,
        help="Number of iterations to run (default: 1)"
    )
    parser.add_argument(
        "-w", "--wait",
        type=float,
        default=0,
        help="Wait time in seconds between iterations (default: 0)"
    )
    parser.add_argument(
        "--no-clean",
        action="store_true",
        help="Skip cleaning debug folder before run"
    )
    
    args = parser.parse_args()
    
    print("="*80)
    print("Iterative Crawler Test Runner")
    print("="*80)
    print(f"Iterations: {args.iterations}")
    print(f"Wait between runs: {args.wait}s")
    print(f"Clean debug folder: {not args.no_clean}")
    print("="*80)
    
    results = []
    
    for i in range(args.iterations):
        print(f"\n{'='*80}")
        print(f"ITERATION {i+1}/{args.iterations}")
        print(f"{'='*80}\n")
        
        # Clean debug folder before each run (unless disabled)
        if not args.no_clean:
            clean_debug_folder()
        
        # Run crawler
        start_time = time.time()
        return_code = run_crawler()
        duration = time.time() - start_time
        
        results.append({
            'iteration': i+1,
            'return_code': return_code,
            'duration': duration,
            'success': return_code == 0
        })
        
        print(f"\n{'='*80}")
        print(f"Iteration {i+1} completed: {'SUCCESS' if return_code == 0 else 'FAILED'} (exit code: {return_code}, duration: {duration:.1f}s)")
        print(f"{'='*80}\n")
        
        # Wait between iterations (except after last one)
        if i < args.iterations - 1 and args.wait > 0:
            print(f"Waiting {args.wait}s before next iteration...")
            time.sleep(args.wait)
    
    # Print summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    for result in results:
        status = "SUCCESS" if result['success'] else "FAILED"
        print(f"Iteration {result['iteration']}: {status} (exit: {result['return_code']}, duration: {result['duration']:.1f}s)")
    
    successful = sum(1 for r in results if r['success'])
    print(f"\nTotal: {successful}/{len(results)} successful")
    print("="*80)
    
    # Exit with error code if any iteration failed
    if not all(r['success'] for r in results):
        sys.exit(1)

if __name__ == "__main__":
    main()

