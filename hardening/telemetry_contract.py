from dataclasses import dataclass
from typing import Optional
from hardening.failure_codes import FailureCode

@dataclass(frozen=True)
class TelemetryEvent:
    trace_id: str
    scraper_name: str
    success: bool
    failure_code: Optional[FailureCode] = None
    message: Optional[str] = None
