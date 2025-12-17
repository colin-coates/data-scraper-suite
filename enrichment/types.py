from dataclasses import dataclass
from typing import Dict, Any, Optional

@dataclass(frozen=True)
class ScrapeResult:
    scraper_name: str
    raw_data: Dict[str, Any]
    metadata: Dict[str, Any]

@dataclass(frozen=True)
class EnrichedResult:
    scraper_name: str
    enriched_data: Dict[str, Any]
    confidence: float
    notes: Optional[str] = None
