from dataclasses import dataclass
from typing import Optional

@dataclass(frozen=True)
class IntelligenceResult:
    risk_score: float
    cost_estimate: float
    priority: int
    recommended_execution_mode: str
    reason: Optional[str] = None
