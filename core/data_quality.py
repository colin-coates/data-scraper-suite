# Copyright (c) 2024 Mountain Jewels Intelligence. All rights reserved.
#
# This software is proprietary and confidential. Unauthorized copying,
# modification, distribution, or use is strictly prohibited.

"""
Data Quality Scoring for MJ Data Scraper Suite

Provides automated quality assessment for scraped data:
- Completeness scoring
- Accuracy validation
- Freshness tracking
- Consistency checks
- Field-level quality metrics
"""

import re
import logging
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)


class QualityDimension(Enum):
    """Quality dimensions for scoring."""
    COMPLETENESS = "completeness"
    ACCURACY = "accuracy"
    FRESHNESS = "freshness"
    CONSISTENCY = "consistency"
    VALIDITY = "validity"


@dataclass
class FieldQuality:
    """Quality metrics for a single field."""
    field_name: str
    present: bool
    valid: bool
    score: float  # 0.0 to 1.0
    issues: List[str] = field(default_factory=list)


@dataclass
class QualityReport:
    """Complete quality report for scraped data."""
    overall_score: float  # 0.0 to 100.0
    grade: str  # A, B, C, D, F
    dimensions: Dict[str, float]
    field_scores: List[FieldQuality]
    issues: List[str]
    recommendations: List[str]
    scraped_at: datetime
    assessed_at: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "overall_score": self.overall_score,
            "grade": self.grade,
            "dimensions": self.dimensions,
            "field_scores": [
                {
                    "field_name": f.field_name,
                    "present": f.present,
                    "valid": f.valid,
                    "score": f.score,
                    "issues": f.issues
                }
                for f in self.field_scores
            ],
            "issues": self.issues,
            "recommendations": self.recommendations,
            "scraped_at": self.scraped_at.isoformat(),
            "assessed_at": self.assessed_at.isoformat()
        }


class DataQualityScorer:
    """
    Scores scraped data quality across multiple dimensions.
    
    Features:
    - Field-level validation
    - Pattern matching for common data types
    - Configurable scoring weights
    - Quality recommendations
    """

    # Common validation patterns
    PATTERNS = {
        "email": r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$',
        "phone": r'^[\+]?[(]?[0-9]{3}[)]?[-\s\.]?[0-9]{3}[-\s\.]?[0-9]{4,6}$',
        "url": r'^https?://[^\s/$.?#].[^\s]*$',
        "linkedin_url": r'^https?://(www\.)?linkedin\.com/in/[a-zA-Z0-9_-]+/?$',
        "twitter_handle": r'^@?[a-zA-Z0-9_]{1,15}$',
        "zip_code": r'^\d{5}(-\d{4})?$',
        "date_iso": r'^\d{4}-\d{2}-\d{2}',
    }

    # Default field configurations
    DEFAULT_FIELD_CONFIG = {
        "email": {"required": True, "pattern": "email", "weight": 1.5},
        "name": {"required": True, "min_length": 2, "weight": 1.2},
        "first_name": {"required": False, "min_length": 2, "weight": 1.0},
        "last_name": {"required": False, "min_length": 2, "weight": 1.0},
        "phone": {"required": False, "pattern": "phone", "weight": 1.0},
        "company": {"required": False, "min_length": 2, "weight": 0.8},
        "title": {"required": False, "min_length": 2, "weight": 0.8},
        "linkedin_url": {"required": False, "pattern": "linkedin_url", "weight": 0.7},
        "twitter": {"required": False, "pattern": "twitter_handle", "weight": 0.5},
        "website": {"required": False, "pattern": "url", "weight": 0.6},
        "address": {"required": False, "min_length": 5, "weight": 0.5},
        "city": {"required": False, "min_length": 2, "weight": 0.5},
        "state": {"required": False, "min_length": 2, "weight": 0.4},
        "zip_code": {"required": False, "pattern": "zip_code", "weight": 0.4},
        "country": {"required": False, "min_length": 2, "weight": 0.4},
    }

    # Dimension weights
    DIMENSION_WEIGHTS = {
        QualityDimension.COMPLETENESS: 0.30,
        QualityDimension.ACCURACY: 0.25,
        QualityDimension.VALIDITY: 0.25,
        QualityDimension.FRESHNESS: 0.10,
        QualityDimension.CONSISTENCY: 0.10,
    }

    def __init__(
        self,
        field_config: Optional[Dict[str, Dict]] = None,
        dimension_weights: Optional[Dict[QualityDimension, float]] = None
    ):
        """
        Initialize the quality scorer.

        Args:
            field_config: Custom field configurations
            dimension_weights: Custom dimension weights
        """
        self.field_config = {**self.DEFAULT_FIELD_CONFIG, **(field_config or {})}
        self.dimension_weights = dimension_weights or self.DIMENSION_WEIGHTS

    def score(
        self,
        data: Dict[str, Any],
        scraped_at: Optional[datetime] = None,
        expected_fields: Optional[List[str]] = None
    ) -> QualityReport:
        """
        Score the quality of scraped data.

        Args:
            data: The scraped data dictionary
            scraped_at: When the data was scraped
            expected_fields: List of expected field names

        Returns:
            QualityReport with detailed scoring
        """
        scraped_at = scraped_at or datetime.utcnow()
        expected_fields = expected_fields or list(self.field_config.keys())

        # Score each dimension
        completeness, comp_fields = self._score_completeness(data, expected_fields)
        accuracy, acc_issues = self._score_accuracy(data)
        validity, val_fields = self._score_validity(data)
        freshness = self._score_freshness(scraped_at)
        consistency, cons_issues = self._score_consistency(data)

        dimensions = {
            QualityDimension.COMPLETENESS.value: completeness,
            QualityDimension.ACCURACY.value: accuracy,
            QualityDimension.VALIDITY.value: validity,
            QualityDimension.FRESHNESS.value: freshness,
            QualityDimension.CONSISTENCY.value: consistency,
        }

        # Calculate weighted overall score
        overall_score = sum(
            dimensions[dim.value] * weight
            for dim, weight in self.dimension_weights.items()
        ) * 100

        # Combine field scores
        field_scores = self._merge_field_scores(comp_fields, val_fields)

        # Collect all issues
        issues = acc_issues + cons_issues
        for fs in field_scores:
            issues.extend(fs.issues)

        # Generate recommendations
        recommendations = self._generate_recommendations(dimensions, field_scores)

        # Determine grade
        grade = self._calculate_grade(overall_score)

        return QualityReport(
            overall_score=round(overall_score, 2),
            grade=grade,
            dimensions={k: round(v * 100, 2) for k, v in dimensions.items()},
            field_scores=field_scores,
            issues=issues,
            recommendations=recommendations,
            scraped_at=scraped_at
        )

    def _score_completeness(
        self,
        data: Dict[str, Any],
        expected_fields: List[str]
    ) -> Tuple[float, List[FieldQuality]]:
        """Score data completeness."""
        field_scores = []
        total_weight = 0
        weighted_score = 0

        for field_name in expected_fields:
            config = self.field_config.get(field_name, {"weight": 1.0})
            weight = config.get("weight", 1.0)
            total_weight += weight

            value = data.get(field_name)
            present = value is not None and str(value).strip() != ""

            if present:
                weighted_score += weight
                field_scores.append(FieldQuality(
                    field_name=field_name,
                    present=True,
                    valid=True,
                    score=1.0
                ))
            else:
                issues = []
                if config.get("required"):
                    issues.append(f"Required field '{field_name}' is missing")
                field_scores.append(FieldQuality(
                    field_name=field_name,
                    present=False,
                    valid=False,
                    score=0.0,
                    issues=issues
                ))

        completeness = weighted_score / total_weight if total_weight > 0 else 0
        return completeness, field_scores

    def _score_accuracy(self, data: Dict[str, Any]) -> Tuple[float, List[str]]:
        """Score data accuracy based on heuristics."""
        issues = []
        checks_passed = 0
        total_checks = 0

        # Check for placeholder/fake data
        fake_patterns = [
            r'^test',
            r'^fake',
            r'^sample',
            r'^example',
            r'@example\.com$',
            r'@test\.com$',
            r'^123',
            r'^xxx',
            r'^aaa',
        ]

        for field_name, value in data.items():
            if value is None:
                continue

            str_value = str(value).lower()
            total_checks += 1
            is_fake = False

            for pattern in fake_patterns:
                if re.search(pattern, str_value, re.IGNORECASE):
                    is_fake = True
                    issues.append(f"Field '{field_name}' appears to contain placeholder data")
                    break

            if not is_fake:
                checks_passed += 1

        accuracy = checks_passed / total_checks if total_checks > 0 else 1.0
        return accuracy, issues

    def _score_validity(self, data: Dict[str, Any]) -> Tuple[float, List[FieldQuality]]:
        """Score data validity based on patterns and rules."""
        field_scores = []
        total_weight = 0
        weighted_score = 0

        for field_name, value in data.items():
            if value is None or str(value).strip() == "":
                continue

            config = self.field_config.get(field_name, {})
            weight = config.get("weight", 1.0)
            total_weight += weight

            issues = []
            valid = True
            str_value = str(value)

            # Check pattern
            pattern_name = config.get("pattern")
            if pattern_name and pattern_name in self.PATTERNS:
                pattern = self.PATTERNS[pattern_name]
                if not re.match(pattern, str_value):
                    valid = False
                    issues.append(f"Field '{field_name}' does not match expected {pattern_name} format")

            # Check minimum length
            min_length = config.get("min_length")
            if min_length and len(str_value) < min_length:
                valid = False
                issues.append(f"Field '{field_name}' is too short (min {min_length} chars)")

            # Check maximum length
            max_length = config.get("max_length")
            if max_length and len(str_value) > max_length:
                valid = False
                issues.append(f"Field '{field_name}' is too long (max {max_length} chars)")

            if valid:
                weighted_score += weight

            field_scores.append(FieldQuality(
                field_name=field_name,
                present=True,
                valid=valid,
                score=1.0 if valid else 0.5,
                issues=issues
            ))

        validity = weighted_score / total_weight if total_weight > 0 else 1.0
        return validity, field_scores

    def _score_freshness(self, scraped_at: datetime) -> float:
        """Score data freshness based on age."""
        age_hours = (datetime.utcnow() - scraped_at).total_seconds() / 3600

        if age_hours < 1:
            return 1.0
        elif age_hours < 24:
            return 0.95
        elif age_hours < 168:  # 1 week
            return 0.85
        elif age_hours < 720:  # 30 days
            return 0.70
        elif age_hours < 2160:  # 90 days
            return 0.50
        else:
            return 0.30

    def _score_consistency(self, data: Dict[str, Any]) -> Tuple[float, List[str]]:
        """Score data consistency."""
        issues = []
        checks_passed = 0
        total_checks = 0

        # Check name consistency
        if "name" in data and ("first_name" in data or "last_name" in data):
            total_checks += 1
            full_name = str(data.get("name", "")).lower()
            first_name = str(data.get("first_name", "")).lower()
            last_name = str(data.get("last_name", "")).lower()

            if first_name and first_name not in full_name:
                issues.append("First name doesn't match full name")
            elif last_name and last_name not in full_name:
                issues.append("Last name doesn't match full name")
            else:
                checks_passed += 1

        # Check email domain matches company website
        if "email" in data and "website" in data:
            total_checks += 1
            email = str(data.get("email", ""))
            website = str(data.get("website", ""))

            if "@" in email:
                email_domain = email.split("@")[1].lower()
                # Extract domain from website
                website_domain = re.sub(r'^https?://(www\.)?', '', website.lower()).split('/')[0]

                if email_domain in website_domain or website_domain in email_domain:
                    checks_passed += 1
                else:
                    issues.append("Email domain doesn't match website domain")

        consistency = checks_passed / total_checks if total_checks > 0 else 1.0
        return consistency, issues

    def _merge_field_scores(
        self,
        comp_fields: List[FieldQuality],
        val_fields: List[FieldQuality]
    ) -> List[FieldQuality]:
        """Merge completeness and validity field scores."""
        merged = {}

        for fs in comp_fields:
            merged[fs.field_name] = fs

        for fs in val_fields:
            if fs.field_name in merged:
                existing = merged[fs.field_name]
                merged[fs.field_name] = FieldQuality(
                    field_name=fs.field_name,
                    present=existing.present,
                    valid=fs.valid,
                    score=(existing.score + fs.score) / 2,
                    issues=existing.issues + fs.issues
                )
            else:
                merged[fs.field_name] = fs

        return list(merged.values())

    def _generate_recommendations(
        self,
        dimensions: Dict[str, float],
        field_scores: List[FieldQuality]
    ) -> List[str]:
        """Generate quality improvement recommendations."""
        recommendations = []

        # Completeness recommendations
        if dimensions[QualityDimension.COMPLETENESS.value] < 0.7:
            missing_required = [
                fs.field_name for fs in field_scores
                if not fs.present and self.field_config.get(fs.field_name, {}).get("required")
            ]
            if missing_required:
                recommendations.append(
                    f"Add missing required fields: {', '.join(missing_required)}"
                )

        # Validity recommendations
        if dimensions[QualityDimension.VALIDITY.value] < 0.8:
            invalid_fields = [fs.field_name for fs in field_scores if not fs.valid]
            if invalid_fields:
                recommendations.append(
                    f"Fix invalid field formats: {', '.join(invalid_fields)}"
                )

        # Accuracy recommendations
        if dimensions[QualityDimension.ACCURACY.value] < 0.9:
            recommendations.append("Review data for placeholder or test values")

        # Freshness recommendations
        if dimensions[QualityDimension.FRESHNESS.value] < 0.7:
            recommendations.append("Consider re-scraping for fresher data")

        return recommendations

    def _calculate_grade(self, score: float) -> str:
        """Calculate letter grade from score."""
        if score >= 90:
            return "A"
        elif score >= 80:
            return "B"
        elif score >= 70:
            return "C"
        elif score >= 60:
            return "D"
        else:
            return "F"

    def score_batch(
        self,
        records: List[Dict[str, Any]],
        scraped_at: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """
        Score a batch of records.

        Args:
            records: List of data records
            scraped_at: When the data was scraped

        Returns:
            Batch quality summary
        """
        if not records:
            return {
                "total_records": 0,
                "average_score": 0,
                "grade_distribution": {},
                "common_issues": []
            }

        reports = [self.score(record, scraped_at) for record in records]

        # Calculate averages
        avg_score = sum(r.overall_score for r in reports) / len(reports)

        # Grade distribution
        grade_dist = {}
        for r in reports:
            grade_dist[r.grade] = grade_dist.get(r.grade, 0) + 1

        # Common issues
        issue_counts = {}
        for r in reports:
            for issue in r.issues:
                issue_counts[issue] = issue_counts.get(issue, 0) + 1

        common_issues = sorted(
            issue_counts.items(),
            key=lambda x: x[1],
            reverse=True
        )[:10]

        return {
            "total_records": len(records),
            "average_score": round(avg_score, 2),
            "average_grade": self._calculate_grade(avg_score),
            "grade_distribution": grade_dist,
            "common_issues": [{"issue": i, "count": c} for i, c in common_issues],
            "dimension_averages": {
                dim: round(sum(r.dimensions[dim] for r in reports) / len(reports), 2)
                for dim in reports[0].dimensions.keys()
            }
        }


# Global instance
quality_scorer = DataQualityScorer()
