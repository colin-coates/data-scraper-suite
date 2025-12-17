from dataclasses import dataclass
from typing import Dict, Any

@dataclass(frozen=True)
class ScrapeContext:
    scraper_name: str
    target: str
    metadata: Dict[str, Any]
    dry_run: bool
