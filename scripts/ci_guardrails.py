#!/usr/bin/env python3
"""
CI Guardrails for Data Scraper Suite

Prevents reintroduction of deleted abstractions and enforces architectural constraints.
Run this script as part of CI/CD pipeline to catch violations early.

Usage:
    python scripts/ci_guardrails.py
    python scripts/ci_guardrails.py --fix  # Auto-fix some issues (future)

Exit codes:
    0 - All checks passed
    1 - Violations found
"""

import os
import re
import sys
from pathlib import Path
from typing import List, Tuple, Set

# Forbidden patterns that should never appear in the codebase
FORBIDDEN_PATTERNS = [
    # Deleted abstractions
    (r'\bEnhancedBaseScraper\b', 'EnhancedBaseScraper was removed - use BaseScraper from core.base_scraper'),
    (r'\bai_precheck\b', 'ai_precheck was removed - governance is handled by enforce() method'),
    (r'from\s+scrapers\.base_scraper\s+import', 'scrapers.base_scraper was deleted - import from core.base_scraper'),
    (r'from\s+scrapers\s+import\s+base_scraper', 'scrapers.base_scraper was deleted - import from core.base_scraper'),
]

# Files/directories to exclude from scanning
EXCLUDE_PATTERNS = [
    '__pycache__',
    '.git',
    '.venv',
    'venv',
    'node_modules',
    '*.pyc',
    '*.pyo',
    '.eggs',
    '*.egg-info',
    'dist',
    'build',
    'scripts/ci_guardrails.py',  # Exclude self
]

# Required patterns that should exist somewhere in the codebase
REQUIRED_PATTERNS = [
    (r'class\s+BaseScraper\s*\(', 'core/base_scraper.py', 'BaseScraper class must exist'),
    (r'@abstractmethod', 'core/base_scraper.py', 'BaseScraper must have abstract methods'),
    (r'__init_subclass__', 'core/base_scraper.py', 'Contract enforcement via __init_subclass__ required'),
]


def should_exclude(path: Path) -> bool:
    """Check if path should be excluded from scanning."""
    path_str = str(path)
    for pattern in EXCLUDE_PATTERNS:
        if pattern in path_str:
            return True
    return False


def scan_file(filepath: Path) -> List[Tuple[int, str, str]]:
    """
    Scan a single file for forbidden patterns.
    
    Returns:
        List of (line_number, line_content, violation_message)
    """
    violations = []
    
    try:
        content = filepath.read_text(encoding='utf-8')
        lines = content.split('\n')
        
        for line_num, line in enumerate(lines, 1):
            for pattern, message in FORBIDDEN_PATTERNS:
                if re.search(pattern, line):
                    violations.append((line_num, line.strip(), message))
    except (UnicodeDecodeError, PermissionError):
        pass  # Skip binary or unreadable files
    
    return violations


def scan_codebase(root_dir: Path) -> dict:
    """
    Scan entire codebase for violations.
    
    Returns:
        Dict mapping filepath to list of violations
    """
    all_violations = {}
    
    for filepath in root_dir.rglob('*.py'):
        if should_exclude(filepath):
            continue
        
        violations = scan_file(filepath)
        if violations:
            all_violations[filepath] = violations
    
    return all_violations


def check_required_patterns(root_dir: Path) -> List[str]:
    """
    Check that required patterns exist in expected files.
    
    Returns:
        List of missing requirement messages
    """
    missing = []
    
    for pattern, expected_file, message in REQUIRED_PATTERNS:
        filepath = root_dir / expected_file
        
        if not filepath.exists():
            missing.append(f"Missing file: {expected_file} - {message}")
            continue
        
        content = filepath.read_text(encoding='utf-8')
        if not re.search(pattern, content):
            missing.append(f"Missing in {expected_file}: {message}")
    
    return missing


def print_violations(violations: dict) -> None:
    """Print violations in a readable format."""
    for filepath, file_violations in violations.items():
        print(f"\n❌ {filepath}")
        for line_num, line_content, message in file_violations:
            print(f"   Line {line_num}: {message}")
            print(f"   > {line_content[:80]}{'...' if len(line_content) > 80 else ''}")


def main() -> int:
    """Main entry point."""
    # Find repository root
    script_dir = Path(__file__).parent
    root_dir = script_dir.parent
    
    # Verify we're in the right place
    if not (root_dir / 'core' / 'base_scraper.py').exists():
        print("❌ Error: Cannot find core/base_scraper.py - are you in the right directory?")
        return 1
    
    print("🔍 Running CI Guardrails...")
    print(f"   Scanning: {root_dir}")
    
    # Scan for forbidden patterns
    violations = scan_codebase(root_dir)
    
    # Check required patterns
    missing_requirements = check_required_patterns(root_dir)
    
    # Report results
    if violations:
        print("\n" + "=" * 60)
        print("FORBIDDEN PATTERN VIOLATIONS FOUND")
        print("=" * 60)
        print_violations(violations)
    
    if missing_requirements:
        print("\n" + "=" * 60)
        print("MISSING REQUIRED PATTERNS")
        print("=" * 60)
        for msg in missing_requirements:
            print(f"   ❌ {msg}")
    
    # Summary
    total_violations = sum(len(v) for v in violations.values())
    
    if total_violations == 0 and not missing_requirements:
        print("\n✅ All CI guardrail checks passed!")
        return 0
    else:
        print(f"\n❌ CI guardrail checks failed:")
        print(f"   - {total_violations} forbidden pattern violation(s)")
        print(f"   - {len(missing_requirements)} missing requirement(s)")
        return 1


if __name__ == '__main__':
    sys.exit(main())
