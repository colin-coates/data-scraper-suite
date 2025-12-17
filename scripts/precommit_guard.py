import sys
from pathlib import Path

FORBIDDEN = [
    "EnhancedBaseScraper",
    "ai_precheck",
    "scrapers/base_scraper.py",
]

EXCLUDE = [
    "scripts/precommit_guard.py",
    "scripts/ci_guardrails.py",
]

violations = []

for path in Path(".").rglob("*.py"):
    if any(exc in str(path) for exc in EXCLUDE):
        continue
    text = path.read_text(errors="ignore")
    for token in FORBIDDEN:
        if token in text:
            violations.append(f"{path}: contains {token}")

if violations:
    print("Pre-commit guard failed:")
    for v in violations:
        print(v)
    sys.exit(1)

sys.exit(0)
