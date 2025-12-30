# Copyright (c) 2024 Mountain Jewels Intelligence. All rights reserved.
#
# This software is proprietary and confidential. Unauthorized copying,
# modification, distribution, or use is strictly prohibited.

"""
Machine Learning Adaptive Scraper for MJ Data Scraper Suite

Provides ML-based adaptive scraping:
- Success pattern learning
- Failure prediction
- Optimal timing detection
- Strategy recommendation
- Anti-detection adaptation
"""

import logging
import json
import os
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
from enum import Enum
import random
import math

logger = logging.getLogger(__name__)


class ScrapingStrategy(Enum):
    """Available scraping strategies."""
    STEALTH = "stealth"  # Slow, careful, human-like
    BALANCED = "balanced"  # Default approach
    AGGRESSIVE = "aggressive"  # Fast, higher risk
    RETRY = "retry"  # After failure, more cautious
    BURST = "burst"  # Quick burst then pause


@dataclass
class ScrapeAttempt:
    """Record of a scrape attempt."""
    timestamp: datetime
    domain: str
    scraper_type: str
    strategy: str
    success: bool
    response_time_ms: int
    status_code: Optional[int]
    records_found: int
    error_type: Optional[str]
    
    # Context
    hour_of_day: int
    day_of_week: int
    proxy_used: bool
    user_agent: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        d = asdict(self)
        d["timestamp"] = self.timestamp.isoformat()
        return d
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ScrapeAttempt":
        """Create from dictionary."""
        d["timestamp"] = datetime.fromisoformat(d["timestamp"])
        return cls(**d)


@dataclass
class DomainProfile:
    """Learned profile for a domain."""
    domain: str
    total_attempts: int = 0
    successful_attempts: int = 0
    
    # Success rates by strategy
    strategy_success: Dict[str, Tuple[int, int]] = field(default_factory=dict)  # strategy -> (success, total)
    
    # Success rates by hour
    hourly_success: Dict[int, Tuple[int, int]] = field(default_factory=dict)  # hour -> (success, total)
    
    # Success rates by day
    daily_success: Dict[int, Tuple[int, int]] = field(default_factory=dict)  # day -> (success, total)
    
    # Average response times
    avg_response_time: float = 0
    
    # Common error types
    error_counts: Dict[str, int] = field(default_factory=dict)
    
    # Recommended settings
    recommended_strategy: str = "balanced"
    recommended_delay_ms: int = 1000
    requires_proxy: bool = False
    requires_js: bool = False
    
    # Last updated
    last_updated: datetime = field(default_factory=datetime.utcnow)
    
    @property
    def success_rate(self) -> float:
        """Overall success rate."""
        if self.total_attempts == 0:
            return 0.5  # Unknown, assume 50%
        return self.successful_attempts / self.total_attempts
    
    def get_best_hour(self) -> Optional[int]:
        """Get hour with highest success rate."""
        if not self.hourly_success:
            return None
        
        best_hour = None
        best_rate = 0
        
        for hour, (success, total) in self.hourly_success.items():
            if total >= 3:  # Need minimum samples
                rate = success / total
                if rate > best_rate:
                    best_rate = rate
                    best_hour = hour
        
        return best_hour
    
    def get_best_strategy(self) -> str:
        """Get strategy with highest success rate."""
        if not self.strategy_success:
            return "balanced"
        
        best_strategy = "balanced"
        best_rate = 0
        
        for strategy, (success, total) in self.strategy_success.items():
            if total >= 3:
                rate = success / total
                if rate > best_rate:
                    best_rate = rate
                    best_strategy = strategy
        
        return best_strategy


class AdaptiveScraper:
    """
    ML-based adaptive scraping system.
    
    Features:
    - Learns from success/failure patterns
    - Recommends optimal strategies
    - Predicts best times to scrape
    - Adapts to anti-bot measures
    """

    def __init__(self, data_path: Optional[str] = None):
        """
        Initialize adaptive scraper.

        Args:
            data_path: Path to persist learning data
        """
        self.data_path = data_path or "./data/adaptive_learning.json"
        self.attempts: List[ScrapeAttempt] = []
        self.domain_profiles: Dict[str, DomainProfile] = {}
        
        # Learning parameters
        self.min_samples_for_recommendation = 5
        self.decay_factor = 0.95  # Weight recent data more
        self.exploration_rate = 0.1  # Try non-optimal strategies sometimes
        
        # Load existing data
        self._load_data()

    def record_attempt(
        self,
        domain: str,
        scraper_type: str,
        strategy: str,
        success: bool,
        response_time_ms: int,
        status_code: Optional[int] = None,
        records_found: int = 0,
        error_type: Optional[str] = None,
        proxy_used: bool = False,
        user_agent: str = ""
    ) -> None:
        """
        Record a scrape attempt for learning.

        Args:
            domain: Target domain
            scraper_type: Type of scraper used
            strategy: Strategy used
            success: Whether attempt succeeded
            response_time_ms: Response time in milliseconds
            status_code: HTTP status code
            records_found: Number of records found
            error_type: Type of error if failed
            proxy_used: Whether proxy was used
            user_agent: User agent string used
        """
        now = datetime.utcnow()
        
        attempt = ScrapeAttempt(
            timestamp=now,
            domain=domain,
            scraper_type=scraper_type,
            strategy=strategy,
            success=success,
            response_time_ms=response_time_ms,
            status_code=status_code,
            records_found=records_found,
            error_type=error_type,
            hour_of_day=now.hour,
            day_of_week=now.weekday(),
            proxy_used=proxy_used,
            user_agent=user_agent
        )
        
        self.attempts.append(attempt)
        self._update_domain_profile(attempt)
        
        # Persist periodically
        if len(self.attempts) % 10 == 0:
            self._save_data()

    def _update_domain_profile(self, attempt: ScrapeAttempt) -> None:
        """Update domain profile with new attempt."""
        domain = attempt.domain
        
        if domain not in self.domain_profiles:
            self.domain_profiles[domain] = DomainProfile(domain=domain)
        
        profile = self.domain_profiles[domain]
        profile.total_attempts += 1
        
        if attempt.success:
            profile.successful_attempts += 1
        
        # Update strategy stats
        if attempt.strategy not in profile.strategy_success:
            profile.strategy_success[attempt.strategy] = (0, 0)
        success, total = profile.strategy_success[attempt.strategy]
        profile.strategy_success[attempt.strategy] = (
            success + (1 if attempt.success else 0),
            total + 1
        )
        
        # Update hourly stats
        hour = attempt.hour_of_day
        if hour not in profile.hourly_success:
            profile.hourly_success[hour] = (0, 0)
        success, total = profile.hourly_success[hour]
        profile.hourly_success[hour] = (
            success + (1 if attempt.success else 0),
            total + 1
        )
        
        # Update daily stats
        day = attempt.day_of_week
        if day not in profile.daily_success:
            profile.daily_success[day] = (0, 0)
        success, total = profile.daily_success[day]
        profile.daily_success[day] = (
            success + (1 if attempt.success else 0),
            total + 1
        )
        
        # Update response time (exponential moving average)
        alpha = 0.2
        profile.avg_response_time = (
            alpha * attempt.response_time_ms +
            (1 - alpha) * profile.avg_response_time
        )
        
        # Update error counts
        if attempt.error_type:
            profile.error_counts[attempt.error_type] = (
                profile.error_counts.get(attempt.error_type, 0) + 1
            )
        
        # Update recommendations
        profile.recommended_strategy = profile.get_best_strategy()
        profile.last_updated = datetime.utcnow()
        
        # Detect if proxy is needed
        rate_limit_errors = profile.error_counts.get("rate_limit", 0)
        blocked_errors = profile.error_counts.get("blocked", 0)
        if rate_limit_errors + blocked_errors > profile.total_attempts * 0.2:
            profile.requires_proxy = True
        
        # Detect if JS rendering is needed
        empty_errors = profile.error_counts.get("empty_content", 0)
        if empty_errors > profile.total_attempts * 0.3:
            profile.requires_js = True

    def get_recommendation(self, domain: str) -> Dict[str, Any]:
        """
        Get scraping recommendations for a domain.

        Args:
            domain: Target domain

        Returns:
            Recommendation dictionary
        """
        profile = self.domain_profiles.get(domain)
        
        if not profile or profile.total_attempts < self.min_samples_for_recommendation:
            # Not enough data, return defaults with exploration
            return {
                "strategy": self._explore_strategy(),
                "delay_ms": 1000 + random.randint(0, 500),
                "use_proxy": False,
                "use_js": False,
                "best_hour": None,
                "confidence": "low",
                "reason": "Insufficient data for domain"
            }
        
        # Decide whether to explore or exploit
        if random.random() < self.exploration_rate:
            strategy = self._explore_strategy()
            reason = "Exploration mode"
        else:
            strategy = profile.recommended_strategy
            reason = f"Best performing strategy ({profile.success_rate:.1%} success rate)"
        
        # Calculate recommended delay based on response time
        base_delay = max(500, int(profile.avg_response_time * 1.5))
        
        # Adjust delay based on strategy
        delay_multipliers = {
            "stealth": 3.0,
            "balanced": 1.5,
            "aggressive": 0.5,
            "retry": 2.0,
            "burst": 0.3
        }
        delay_ms = int(base_delay * delay_multipliers.get(strategy, 1.5))
        
        return {
            "strategy": strategy,
            "delay_ms": delay_ms,
            "use_proxy": profile.requires_proxy,
            "use_js": profile.requires_js,
            "best_hour": profile.get_best_hour(),
            "confidence": self._calculate_confidence(profile),
            "success_rate": profile.success_rate,
            "total_attempts": profile.total_attempts,
            "reason": reason
        }

    def _explore_strategy(self) -> str:
        """Select a strategy for exploration."""
        strategies = list(ScrapingStrategy)
        weights = [1, 3, 1, 1, 1]  # Favor balanced
        return random.choices(strategies, weights=weights)[0].value

    def _calculate_confidence(self, profile: DomainProfile) -> str:
        """Calculate confidence level for recommendations."""
        if profile.total_attempts < 10:
            return "low"
        elif profile.total_attempts < 50:
            return "medium"
        else:
            return "high"

    def predict_success(
        self,
        domain: str,
        strategy: str,
        hour: Optional[int] = None
    ) -> float:
        """
        Predict success probability for a scrape attempt.

        Args:
            domain: Target domain
            strategy: Planned strategy
            hour: Hour of day (uses current if not specified)

        Returns:
            Predicted success probability (0.0 to 1.0)
        """
        profile = self.domain_profiles.get(domain)
        
        if not profile:
            return 0.5  # Unknown domain
        
        # Base probability from overall success rate
        base_prob = profile.success_rate
        
        # Adjust for strategy
        if strategy in profile.strategy_success:
            success, total = profile.strategy_success[strategy]
            if total >= 3:
                strategy_rate = success / total
                base_prob = 0.7 * strategy_rate + 0.3 * base_prob
        
        # Adjust for hour
        hour = hour if hour is not None else datetime.utcnow().hour
        if hour in profile.hourly_success:
            success, total = profile.hourly_success[hour]
            if total >= 3:
                hour_rate = success / total
                base_prob = 0.8 * base_prob + 0.2 * hour_rate
        
        return min(1.0, max(0.0, base_prob))

    def get_optimal_schedule(
        self,
        domain: str,
        num_scrapes: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Get optimal schedule for multiple scrapes.

        Args:
            domain: Target domain
            num_scrapes: Number of scrapes to schedule

        Returns:
            List of scheduled scrape configurations
        """
        profile = self.domain_profiles.get(domain)
        recommendation = self.get_recommendation(domain)
        
        schedule = []
        
        # Find best hours
        best_hours = []
        if profile and profile.hourly_success:
            sorted_hours = sorted(
                profile.hourly_success.items(),
                key=lambda x: x[1][0] / max(1, x[1][1]),
                reverse=True
            )
            best_hours = [h for h, _ in sorted_hours[:6]]
        
        if not best_hours:
            best_hours = [9, 10, 11, 14, 15, 16]  # Business hours default
        
        for i in range(num_scrapes):
            hour = best_hours[i % len(best_hours)]
            
            # Vary strategy slightly
            if i % 5 == 0:
                strategy = self._explore_strategy()
            else:
                strategy = recommendation["strategy"]
            
            schedule.append({
                "index": i,
                "hour": hour,
                "strategy": strategy,
                "delay_ms": recommendation["delay_ms"] + random.randint(-200, 200),
                "use_proxy": recommendation["use_proxy"],
                "predicted_success": self.predict_success(domain, strategy, hour)
            })
        
        return schedule

    def get_domain_insights(self, domain: str) -> Dict[str, Any]:
        """
        Get detailed insights for a domain.

        Args:
            domain: Target domain

        Returns:
            Insights dictionary
        """
        profile = self.domain_profiles.get(domain)
        
        if not profile:
            return {"error": "No data for domain"}
        
        # Find patterns
        best_hour = profile.get_best_hour()
        worst_hour = None
        if profile.hourly_success:
            sorted_hours = sorted(
                profile.hourly_success.items(),
                key=lambda x: x[1][0] / max(1, x[1][1])
            )
            if sorted_hours:
                worst_hour = sorted_hours[0][0]
        
        return {
            "domain": domain,
            "total_attempts": profile.total_attempts,
            "success_rate": profile.success_rate,
            "avg_response_time_ms": profile.avg_response_time,
            "best_strategy": profile.recommended_strategy,
            "best_hour": best_hour,
            "worst_hour": worst_hour,
            "requires_proxy": profile.requires_proxy,
            "requires_js": profile.requires_js,
            "common_errors": dict(sorted(
                profile.error_counts.items(),
                key=lambda x: x[1],
                reverse=True
            )[:5]),
            "strategy_performance": {
                strategy: {
                    "success_rate": success / max(1, total),
                    "attempts": total
                }
                for strategy, (success, total) in profile.strategy_success.items()
            },
            "hourly_performance": {
                hour: {
                    "success_rate": success / max(1, total),
                    "attempts": total
                }
                for hour, (success, total) in profile.hourly_success.items()
            }
        }

    def get_global_insights(self) -> Dict[str, Any]:
        """Get insights across all domains."""
        if not self.domain_profiles:
            return {"error": "No data available"}
        
        total_attempts = sum(p.total_attempts for p in self.domain_profiles.values())
        total_success = sum(p.successful_attempts for p in self.domain_profiles.values())
        
        # Best performing domains
        sorted_domains = sorted(
            self.domain_profiles.items(),
            key=lambda x: x[1].success_rate,
            reverse=True
        )
        
        # Strategy performance across all domains
        strategy_totals: Dict[str, Tuple[int, int]] = defaultdict(lambda: (0, 0))
        for profile in self.domain_profiles.values():
            for strategy, (success, total) in profile.strategy_success.items():
                s, t = strategy_totals[strategy]
                strategy_totals[strategy] = (s + success, t + total)
        
        return {
            "total_domains": len(self.domain_profiles),
            "total_attempts": total_attempts,
            "overall_success_rate": total_success / max(1, total_attempts),
            "best_domains": [
                {"domain": d, "success_rate": p.success_rate, "attempts": p.total_attempts}
                for d, p in sorted_domains[:5]
                if p.total_attempts >= 5
            ],
            "worst_domains": [
                {"domain": d, "success_rate": p.success_rate, "attempts": p.total_attempts}
                for d, p in reversed(sorted_domains[-5:])
                if p.total_attempts >= 5
            ],
            "strategy_performance": {
                strategy: {
                    "success_rate": success / max(1, total),
                    "attempts": total
                }
                for strategy, (success, total) in strategy_totals.items()
            }
        }

    def _save_data(self) -> None:
        """Persist learning data to disk."""
        try:
            os.makedirs(os.path.dirname(self.data_path), exist_ok=True)
            
            data = {
                "attempts": [a.to_dict() for a in self.attempts[-10000:]],  # Keep last 10k
                "profiles": {
                    domain: {
                        "domain": p.domain,
                        "total_attempts": p.total_attempts,
                        "successful_attempts": p.successful_attempts,
                        "strategy_success": p.strategy_success,
                        "hourly_success": {str(k): v for k, v in p.hourly_success.items()},
                        "daily_success": {str(k): v for k, v in p.daily_success.items()},
                        "avg_response_time": p.avg_response_time,
                        "error_counts": p.error_counts,
                        "recommended_strategy": p.recommended_strategy,
                        "recommended_delay_ms": p.recommended_delay_ms,
                        "requires_proxy": p.requires_proxy,
                        "requires_js": p.requires_js,
                        "last_updated": p.last_updated.isoformat()
                    }
                    for domain, p in self.domain_profiles.items()
                }
            }
            
            with open(self.data_path, "w") as f:
                json.dump(data, f)
                
        except Exception as e:
            logger.error(f"Failed to save adaptive learning data: {e}")

    def _load_data(self) -> None:
        """Load learning data from disk."""
        try:
            if not os.path.exists(self.data_path):
                return
            
            with open(self.data_path, "r") as f:
                data = json.load(f)
            
            self.attempts = [ScrapeAttempt.from_dict(a) for a in data.get("attempts", [])]
            
            for domain, p_data in data.get("profiles", {}).items():
                profile = DomainProfile(
                    domain=p_data["domain"],
                    total_attempts=p_data["total_attempts"],
                    successful_attempts=p_data["successful_attempts"],
                    strategy_success=p_data["strategy_success"],
                    hourly_success={int(k): tuple(v) for k, v in p_data["hourly_success"].items()},
                    daily_success={int(k): tuple(v) for k, v in p_data["daily_success"].items()},
                    avg_response_time=p_data["avg_response_time"],
                    error_counts=p_data["error_counts"],
                    recommended_strategy=p_data["recommended_strategy"],
                    recommended_delay_ms=p_data["recommended_delay_ms"],
                    requires_proxy=p_data["requires_proxy"],
                    requires_js=p_data["requires_js"],
                    last_updated=datetime.fromisoformat(p_data["last_updated"])
                )
                self.domain_profiles[domain] = profile
            
            logger.info(f"Loaded {len(self.attempts)} attempts and {len(self.domain_profiles)} domain profiles")
            
        except Exception as e:
            logger.error(f"Failed to load adaptive learning data: {e}")


# Global instance
adaptive_scraper = AdaptiveScraper()
