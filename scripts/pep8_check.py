#!/usr/bin/env python3
"""
PEP8 Compliance Checker for Mordeaux Face Scanning MVP
Checks Python files for common PEP8 violations
"""

import os
import re
import sys
from pathlib import Path

def check_line_length(file_path, max_length=120):
    """Check for lines exceeding maximum length."""
    violations = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            if len(line.rstrip()) > max_length:
                violations.append({
                    'line': line_num,
                    'length': len(line.rstrip()),
                    'content': line.rstrip()[:100] + '...' if len(line.rstrip()) > 100 else line.rstrip()
                })
    return violations

def check_imports(file_path):
    """Check for import-related PEP8 violations."""
    violations = []
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    in_imports = False
    for line_num, line in enumerate(lines, 1):
        stripped = line.strip()

        # Check for imports at module level
        if stripped.startswith(('import ', 'from ')):
            in_imports = True
            # Check for multiple imports on same line
            if ',' in stripped and not stripped.startswith('from '):
                violations.append({
                    'line': line_num,
                    'type': 'E401',
                    'message': 'Multiple imports on one line'
                })
        elif stripped and not stripped.startswith('#') and in_imports:
            # End of import section
            in_imports = False

        # Check for trailing whitespace
        if line.rstrip() != line.rstrip(' \t'):
            violations.append({
                'line': line_num,
                'type': 'W291',
                'message': 'Trailing whitespace'
            })

        # Check for tabs vs spaces
        if '\t' in line:
            violations.append({
                'line': line_num,
                'type': 'W191',
                'message': 'Indentation contains tabs'
            })

    return violations

def check_function_definitions(file_path):
    """Check for function definition PEP8 violations."""
    violations = []
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    for line_num, line in enumerate(lines, 1):
        stripped = line.strip()

        # Check for function definitions
        if stripped.startswith('def '):
            # Check for space after function name
            if not re.match(r'def \w+\(', stripped):
                violations.append({
                    'line': line_num,
                    'type': 'E211',
                    'message': 'Missing space after function name'
                })

        # Check for class definitions
        if stripped.startswith('class '):
            # Check for space after class name
            if not re.match(r'class \w+', stripped):
                violations.append({
                    'line': line_num,
                    'type': 'E211',
                    'message': 'Missing space after class name'
                })

    return violations

def check_pep8_file(file_path):
    """Check a single file for PEP8 violations."""
    print(f"Checking {file_path}...")

    all_violations = []

    # Check line length
    length_violations = check_line_length(file_path)
    for violation in length_violations:
        all_violations.append({
            'line': violation['line'],
            'type': 'E501',
            'message': f"Line too long ({violation['length']} > 120 characters)",
            'content': violation['content']
        })

    # Check imports
    import_violations = check_imports(file_path)
    all_violations.extend(import_violations)

    # Check function definitions
    func_violations = check_function_definitions(file_path)
    all_violations.extend(func_violations)

    return all_violations

def main():
    """Main function to check all Python files."""
    print("🔍 PEP8 Compliance Check for Mordeaux Face Scanning MVP")
    print("=" * 60)

    # Directories to check
    directories = [
        'backend/app',
        'face-pipeline',
        'worker'
    ]

    total_violations = 0
    files_checked = 0

    for directory in directories:
        if not os.path.exists(directory):
            print(f"⚠️  Directory {directory} not found, skipping...")
            continue

        print(f"\n📁 Checking directory: {directory}")

        for root, dirs, files in os.walk(directory):
            # Skip __pycache__ directories
            dirs[:] = [d for d in dirs if d != '__pycache__']

            for file in files:
                if file.endswith('.py'):
                    file_path = os.path.join(root, file)
                    files_checked += 1

                    violations = check_pep8_file(file_path)
                    total_violations += len(violations)

                    if violations:
                        print(f"  ❌ {len(violations)} violations found:")
                        for violation in violations:
                            print(f"    Line {violation['line']}: {violation['type']} - {violation['message']}")
                            if 'content' in violation:
                                print(f"      Content: {violation['content']}")
                    else:
                        print(f"  ✅ No violations found")

    print(f"\n📊 Summary:")
    print(f"  Files checked: {files_checked}")
    print(f"  Total violations: {total_violations}")

    if total_violations == 0:
        print("🎉 All files are PEP8 compliant!")
        return 0
    else:
        print(f"❌ Found {total_violations} PEP8 violations that need to be fixed.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
