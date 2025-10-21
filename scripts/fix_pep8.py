#!/usr/bin/env python3
"""
PEP8 Auto-Fixer for Mordeaux Face Scanning MVP
Automatically fixes common PEP8 violations
"""

import os
import re
import sys
from pathlib import Path

def fix_trailing_whitespace(file_path):
    """Remove trailing whitespace from all lines."""
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    fixed_lines = []
    for line in lines:
        # Remove trailing whitespace but preserve line endings
        fixed_line = line.rstrip() + '\n' if line.endswith('\n') else line.rstrip()
        fixed_lines.append(fixed_line)

    with open(file_path, 'w', encoding='utf-8') as f:
        f.writelines(fixed_lines)

def fix_line_length(file_path, max_length=120):
    """Fix lines that are too long by breaking them appropriately."""
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    fixed_lines = []
    for line in lines:
        if len(line.rstrip()) > max_length:
            # Try to break at logical points
            stripped = line.rstrip()
            if ' ' in stripped:
                # Try to break at the last space before max_length
                break_point = stripped.rfind(' ', 0, max_length)
                if break_point > max_length * 0.7:  # Only break if we can break reasonably
                    line1 = stripped[:break_point] + '\n'
                    line2 = '    ' + stripped[break_point+1:] + '\n'
                    fixed_lines.append(line1)
                    fixed_lines.append(line2)
                else:
                    fixed_lines.append(line)
            else:
                fixed_lines.append(line)
        else:
            fixed_lines.append(line)

    with open(file_path, 'w', encoding='utf-8') as f:
        f.writelines(fixed_lines)

def fix_imports(file_path):
    """Fix import-related PEP8 violations."""
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    fixed_lines = []
    in_imports = False
    import_lines = []
    other_lines = []

    for line in lines:
        stripped = line.strip()

        if stripped.startswith(('import ', 'from ')):
            in_imports = True
            import_lines.append(line)
        elif stripped and not stripped.startswith('#') and in_imports:
            # End of import section
            in_imports = False
            other_lines.append(line)
        elif in_imports:
            import_lines.append(line)
        else:
            other_lines.append(line)

    # Sort imports
    if import_lines:
        # Separate standard library, third-party, and local imports
        std_imports = []
        third_party_imports = []
        local_imports = []

        for line in import_lines:
            stripped = line.strip()
            if stripped.startswith('from ') and '.' in stripped:
                # Check if it's a local import
                module = stripped.split()[1].split('.')[0]
                if module in ['app', 'backend', 'face-pipeline', 'worker']:
                    local_imports.append(line)
                else:
                    third_party_imports.append(line)
            else:
                std_imports.append(line)

        # Combine sorted imports
        all_imports = std_imports + third_party_imports + local_imports
        if all_imports and not all_imports[-1].strip():
            # Remove empty line at end of imports
            all_imports = all_imports[:-1]

        fixed_lines.extend(all_imports)
        if all_imports and other_lines:
            fixed_lines.append('\n')  # Add blank line after imports

    fixed_lines.extend(other_lines)

    with open(file_path, 'w', encoding='utf-8') as f:
        f.writelines(fixed_lines)

def fix_indentation(file_path):
    """Fix indentation issues (tabs vs spaces)."""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Replace tabs with 4 spaces
    content = content.replace('\t', '    ')

    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)

def fix_pep8_file(file_path):
    """Fix PEP8 violations in a single file."""
    print(f"Fixing {file_path}...")

    try:
        # Fix trailing whitespace
        fix_trailing_whitespace(file_path)

        # Fix indentation
        fix_indentation(file_path)

        # Fix imports
        fix_imports(file_path)

        # Fix line length (be careful with this one)
        # fix_line_length(file_path)

        return True
    except Exception as e:
        print(f"Error fixing {file_path}: {e}")
        return False

def main():
    """Main function to fix all Python files."""
    print("🔧 PEP8 Auto-Fixer for Mordeaux Face Scanning MVP")
    print("=" * 60)

    # Directories to check
    directories = [
        'backend/app',
        'face-pipeline',
        'worker'
    ]

    files_fixed = 0
    files_failed = 0

    for directory in directories:
        if not os.path.exists(directory):
            print(f"⚠️  Directory {directory} not found, skipping...")
            continue

        print(f"\n📁 Fixing directory: {directory}")

        for root, dirs, files in os.walk(directory):
            # Skip __pycache__ directories
            dirs[:] = [d for d in dirs if d != '__pycache__']

            for file in files:
                if file.endswith('.py'):
                    file_path = os.path.join(root, file)

                    if fix_pep8_file(file_path):
                        files_fixed += 1
                        print(f"  ✅ Fixed")
                    else:
                        files_failed += 1
                        print(f"  ❌ Failed")

    print(f"\n📊 Summary:")
    print(f"  Files fixed: {files_fixed}")
    print(f"  Files failed: {files_failed}")

    if files_failed == 0:
        print("🎉 All files fixed successfully!")
        return 0
    else:
        print(f"❌ {files_failed} files failed to fix.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
