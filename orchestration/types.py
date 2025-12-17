from dataclasses import dataclass
from typing import Optional

@dataclass(frozen=True)
class ExecutionPlan:
    scraper_name: str
    execution_mode: str
    retries: int
    dry_run: bool
    reason: Optional[str] = None
