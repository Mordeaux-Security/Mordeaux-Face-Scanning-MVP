#!/usr/bin/env python3
"""
Simple logging cleanup script - removes emojis from debug output
"""

import re
import sys
from pathlib import Path

def clean_emojis_in_file(file_path):
    """Remove emojis from logging statements in a file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Remove emojis from logger statements
        emoji_pattern = re.compile(r'(logger\.(?:info|debug|warning|error)\(.*?)[🔍✅❌📊🚀📈📄💾⏭️👤🖼️📁🎯⚡]+(.*?\))')
        new_content = emoji_pattern.sub(r'\1\2', content)
        
        if new_content != content:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(new_content)
            print(f"Cleaned emojis from {file_path}")
            return True
        return False
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return False

def main():
    crawler_dir = Path(__file__).parent.parent / "app" / "crawler"
    
    if len(sys.argv) > 1:
        files = [Path(f) for f in sys.argv[1:]]
    else:
        files = list(crawler_dir.rglob("*.py"))
    
    cleaned_count = 0
    for file_path in files:
        if clean_emojis_in_file(file_path):
            cleaned_count += 1
    
    print(f"Cleaned emojis from {cleaned_count} files")

if __name__ == "__main__":
    main()
