# Copyright (c) 2024 Mountain Jewels Intelligence. All rights reserved.
#
# This software is proprietary and confidential. Unauthorized copying,
# modification, distribution, or use is strictly prohibited.

"""
Asset Signal Model for MJ Data Scraper Suite

Comprehensive data model for representing intelligence signals about assets
(people, companies, organizations) collected through web scraping operations.
Provides enterprise-grade signal processing, validation, and intelligence fusion.
"""

import uuid
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Union
from enum import Enum

try:
    from pydantic import BaseModel, Field, validator
except ImportError:
    # Fallback for environments without pydantic
    class BaseModel:
        def __init__(self, **data):
            for key, value in data.items():
                setattr(self, key, value)

        def dict(self):
            return self.__dict__

    def Field(default=None, description=None, **kwargs):
        return default

    def validator(method_name):
        def decorator(func):
            return func
        return decorator


class SignalType(Enum):
    """Types of intelligence signals."""
    IDENTITY = "identity"                    # Identity confirmation
    LOCATION = "location"                    # Location/Geography signal
    CONTACT = "contact"                      # Contact information
    PROFESSIONAL = "professional"            # Professional/career signal
    SOCIAL = "social"                       # Social network signal
    FINANCIAL = "financial"                 # Financial signal
    LEGAL = "legal"                         # Legal/regulatory signal
    BEHAVIORAL = "behavioral"               # Behavioral pattern
    RELATIONSHIP = "relationship"           # Relationship/connection signal
    EVENT = "event"                        # Event participation signal
    ANOMALY = "anomaly"                    # Anomalous activity signal

    # Real Estate & Property Signals
    BIRTHDAY = "birthday"                   # Birth date information
    ENGAGEMENT = "engagement"               # Engagement announcements
    WEDDING = "wedding"                     # Marriage information
    LIEN = "lien"                          # Property liens and encumbrances
    MORTGAGE = "mortgage"                  # Mortgage information
    DEED = "deed"                          # Property deed transfers
    FORECLOSURE = "foreclosure"            # Foreclosure proceedings
    COURT_CASE = "court_case"              # Legal court cases
    JUDGMENT = "judgment"                  # Court judgments and rulings
    TAX_ISSUE = "tax_issue"                # Tax-related issues and liens


class SignalSource(Enum):
    """Sources of intelligence signals."""
    WEB_SCRAPING = "web_scraping"
    API_COLLECTION = "api_collection"
    SOCIAL_MEDIA = "social_media"
    PUBLIC_RECORDS = "public_records"
    NEWS_FEEDS = "news_feeds"
    BUSINESS_DIRECTORIES = "business_directories"
    COURT_RECORDS = "court_records"
    FINANCIAL_REPORTS = "financial_reports"
    HUMAN_INTELLIGENCE = "human_intelligence"
    SIGNAL_FUSION = "signal_fusion"


class AssetType(Enum):
    """Types of assets that can be signaled."""
    PERSON = "person"
    COMPANY = "company"
    ORGANIZATION = "organization"
    LOCATION = "location"
    EVENT = "event"
    ASSET = "asset"  # Physical or financial asset

    # Real Estate Property Types
    SINGLE_FAMILY_HOME = "single_family_home"
    MULTI_FAMILY_SMALL = "multi_family_small"
    APARTMENT_BUILDING = "apartment_building"
    COMMERCIAL_PROPERTY = "commercial_property"


class SignalConfidence(Enum):
    """Confidence levels for signals."""
    VERY_LOW = "very_low"      # < 20% confidence
    LOW = "low"               # 20-40% confidence
    MEDIUM = "medium"         # 40-70% confidence
    HIGH = "high"            # 70-90% confidence
    VERY_HIGH = "very_high"  # > 90% confidence


class SignalValidationStatus(Enum):
    """Validation status of signals."""
    UNVALIDATED = "unvalidated"
    PENDING_VALIDATION = "pending_validation"
    VALIDATED = "validated"
    CONTRADICTED = "contradicted"
    EXPIRED = "expired"
    REVOKED = "revoked"


class Asset(BaseModel):
    """
    Asset model representing real estate properties and persons.

    Enhanced asset representation with comprehensive property and ownership information.
    """
    asset_type: AssetType = Field(..., description="Type of asset")
    address: Optional[str] = Field(None, description="Street address")
    city: Optional[str] = Field(None, description="City")
    county: Optional[str] = Field(None, description="County")
    state: Optional[str] = Field(None, description="State")
    zip_code: Optional[str] = Field(None, description="ZIP/postal code")
    owner_name: Optional[str] = Field(None, description="Property owner name")

    # Enhanced property information
    parcel_id: Optional[str] = Field(None, description="Property parcel/tax ID")
    property_value: Optional[float] = Field(None, description="Assessed property value")
    square_footage: Optional[int] = Field(None, description="Property square footage")
    year_built: Optional[int] = Field(None, description="Year property was built")
    bedrooms: Optional[int] = Field(None, description="Number of bedrooms")
    bathrooms: Optional[float] = Field(None, description="Number of bathrooms")

    # Legal information
    legal_description: Optional[str] = Field(None, description="Legal property description")
    zoning_code: Optional[str] = Field(None, description="Property zoning classification")
    tax_assessment: Optional[float] = Field(None, description="Annual property tax assessment")

    # Additional metadata
    last_updated: Optional[datetime] = Field(None, description="Last time asset info was updated")
    data_source: Optional[str] = Field(None, description="Source of asset information")
    confidence_score: Optional[float] = Field(None, description="Confidence in asset information")

    def get_full_address(self) -> str:
        """Get formatted full address."""
        components = []
        if self.address:
            components.append(self.address)
        if self.city:
            components.append(self.city)
        if self.state:
            components.append(self.state)
        if self.zip_code:
            components.append(self.zip_code)
        return ", ".join(components) if components else "Unknown Address"

    def get_location_string(self) -> str:
        """Get location string for mapping/searches."""
        components = []
        if self.city:
            components.append(self.city)
        if self.county:
            components.append(self.county)
        if self.state:
            components.append(self.state)
        return ", ".join(components) if components else "Unknown Location"

    def is_property(self) -> bool:
        """Check if this asset is a property (not a person)."""
        return self.asset_type in [
            AssetType.SINGLE_FAMILY_HOME,
            AssetType.MULTI_FAMILY_SMALL,
            AssetType.APARTMENT_BUILDING,
            AssetType.COMMERCIAL_PROPERTY
        ]

    def get_property_type_category(self) -> str:
        """Get property type category for analysis."""
        if self.asset_type == AssetType.SINGLE_FAMILY_HOME:
            return "residential_single"
        elif self.asset_type == AssetType.MULTI_FAMILY_SMALL:
            return "residential_multi_small"
        elif self.asset_type == AssetType.APARTMENT_BUILDING:
            return "residential_multi_large"
        elif self.asset_type == AssetType.COMMERCIAL_PROPERTY:
            return "commercial"
        else:
            return "non_property"


class SignalRequest(BaseModel):
    """
    Request model for signal collection and analysis.

    Defines which signals to collect and the time window for analysis.
    """
    signals: List[SignalType] = Field(..., description="Types of signals to collect/analyze")
    time_window_days: int = Field(30, gt=0, description="Time window in days for signal analysis")

    # Enhanced request parameters
    asset_filter: Optional[Dict[str, Any]] = Field(None, description="Asset filtering criteria")
    priority_signals: Optional[List[SignalType]] = Field(None, description="High-priority signal types")
    min_confidence: Optional[float] = Field(None, description="Minimum confidence threshold")
    max_results: Optional[int] = Field(None, description="Maximum number of results to return")

    # Request metadata
    request_id: Optional[str] = Field(None, description="Unique request identifier")
    requested_by: Optional[str] = Field(None, description="Entity making the request")
    request_timestamp: Optional[datetime] = Field(None, description="When request was made")

    def get_signal_priority(self, signal_type: SignalType) -> str:
        """Get priority level for a signal type."""
        if self.priority_signals and signal_type in self.priority_signals:
            return "high"
        else:
            return "normal"

    def should_include_signal(self, signal_type: SignalType, confidence: float = 0.0) -> bool:
        """Check if a signal should be included based on request criteria."""
        if signal_type not in self.signals:
            return False

        if self.min_confidence is not None and confidence < self.min_confidence:
            return False

        return True

    def get_request_summary(self) -> Dict[str, Any]:
        """Get request summary for logging/reporting."""
        return {
            "signal_count": len(self.signals),
            "time_window_days": self.time_window_days,
            "has_asset_filter": self.asset_filter is not None,
            "has_priority_signals": self.priority_signals is not None,
            "min_confidence": self.min_confidence,
            "max_results": self.max_results,
            "request_id": self.request_id
        }


class AssetSignal(BaseModel):
    """
    Enterprise-grade asset signal model.

    Represents intelligence signals about assets collected through various
    sources with comprehensive metadata, validation, and intelligence fusion.
    """

    # Core signal identification
    signal_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique signal identifier")
    asset_id: str = Field(..., description="Unique identifier for the asset this signal relates to")
    asset_type: AssetType = Field(..., description="Type of asset being signaled")

    # Signal characteristics
    signal_type: SignalType = Field(..., description="Type of intelligence signal")
    signal_source: SignalSource = Field(..., description="Source of the signal")
    signal_value: Any = Field(..., description="The actual signal data/payload")

    # Temporal information
    signal_timestamp: datetime = Field(default_factory=datetime.utcnow, description="When the signal was detected")
    collection_timestamp: datetime = Field(default_factory=datetime.utcnow, description="When the signal was collected")
    validity_start: Optional[datetime] = Field(None, description="When the signal becomes valid")
    validity_end: Optional[datetime] = Field(None, description="When the signal expires")
    last_updated: datetime = Field(default_factory=datetime.utcnow, description="Last time signal was updated")

    # Quality and confidence
    confidence_score: float = Field(0.5, description="Confidence score (0.0-1.0)")
    confidence_level: SignalConfidence = Field(SignalConfidence.MEDIUM, description="Categorical confidence level")
    signal_strength: float = Field(0.5, description="Signal strength indicator (0.0-1.0)")
    reliability_score: float = Field(0.5, description="Source reliability score (0.0-1.0)")

    # Validation and verification
    validation_status: SignalValidationStatus = Field(SignalValidationStatus.UNVALIDATED, description="Current validation status")
    validation_attempts: int = Field(0, description="Number of validation attempts")
    last_validated: Optional[datetime] = Field(None, description="Last validation timestamp")
    validation_method: Optional[str] = Field(None, description="Method used for validation")

    # Source attribution
    source_url: Optional[str] = Field(None, description="URL where signal was found")
    source_domain: Optional[str] = Field(None, description="Domain of the source")
    source_scraper: Optional[str] = Field(None, description="Scraper that collected the signal")
    source_api: Optional[str] = Field(None, description="API that provided the signal")
    source_human: Optional[str] = Field(None, description="Human source identifier")

    # Geographic context
    location_country: Optional[str] = Field(None, description="Country associated with signal")
    location_region: Optional[str] = Field(None, description="Region/state associated with signal")
    location_city: Optional[str] = Field(None, description="City associated with signal")
    location_coordinates: Optional[Dict[str, float]] = Field(None, description="GPS coordinates if available")

    # Asset identification
    asset_name: Optional[str] = Field(None, description="Human-readable asset name")
    asset_identifiers: Dict[str, Any] = Field(default_factory=dict, description="Additional asset identifiers")
    asset_metadata: Dict[str, Any] = Field(default_factory=dict, description="Asset-specific metadata")

    # Property-specific information (for real estate signals)
    property_address: Optional[str] = Field(None, description="Property street address")
    property_city: Optional[str] = Field(None, description="Property city")
    property_county: Optional[str] = Field(None, description="Property county")
    property_state: Optional[str] = Field(None, description="Property state")
    property_zip: Optional[str] = Field(None, description="Property ZIP code")
    parcel_id: Optional[str] = Field(None, description="Property parcel/tax ID")
    property_value: Optional[float] = Field(None, description="Property assessed value")

    # Legal/Financial information (for liens, mortgages, judgments)
    case_number: Optional[str] = Field(None, description="Legal case number")
    court_name: Optional[str] = Field(None, description="Court name/jurisdiction")
    filing_date: Optional[datetime] = Field(None, description="Legal filing date")
    judgment_amount: Optional[float] = Field(None, description="Judgment/monetary amount")
    plaintiff_name: Optional[str] = Field(None, description="Plaintiff/legal complainant")
    defendant_name: Optional[str] = Field(None, description="Defendant/legal respondent")
    attorney_name: Optional[str] = Field(None, description="Attorney name")

    # Real estate transaction information (for deeds, mortgages)
    transaction_amount: Optional[float] = Field(None, description="Transaction dollar amount")
    lender_name: Optional[str] = Field(None, description="Lender/financial institution")
    mortgage_type: Optional[str] = Field(None, description="Type of mortgage/loan")
    interest_rate: Optional[float] = Field(None, description="Mortgage interest rate")
    loan_term_years: Optional[int] = Field(None, description="Loan term in years")

    # Tax information (for tax issues/lien)
    tax_year: Optional[int] = Field(None, description="Tax year")
    tax_amount: Optional[float] = Field(None, description="Tax amount due")
    tax_type: Optional[str] = Field(None, description="Type of tax (property, income, etc.)")

    # Event information (for birthdays, weddings, engagements)
    event_date: Optional[datetime] = Field(None, description="Date of the event")
    spouse_name: Optional[str] = Field(None, description="Spouse name (for weddings)")
    venue_name: Optional[str] = Field(None, description="Event venue name")
    event_description: Optional[str] = Field(None, description="Event description/details")

    # Signal context and relationships
    related_signals: List[str] = Field(default_factory=list, description="IDs of related signals")
    parent_signal: Optional[str] = Field(None, description="Parent signal ID if this is derived")
    child_signals: List[str] = Field(default_factory=list, description="Derived child signal IDs")
    signal_tags: List[str] = Field(default_factory=list, description="Classification tags")

    # Intelligence fusion
    fusion_sources: List[str] = Field(default_factory=list, description="Sources used in signal fusion")
    fusion_confidence: Optional[float] = Field(None, description="Fusion confidence score")
    fusion_method: Optional[str] = Field(None, description="Fusion algorithm used")

    # Quality metrics
    data_quality_score: float = Field(0.5, description="Overall data quality (0.0-1.0)")
    completeness_score: float = Field(0.5, description="Data completeness (0.0-1.0)")
    timeliness_score: float = Field(0.5, description="Data timeliness (0.0-1.0)")
    consistency_score: float = Field(0.5, description="Data consistency (0.0-1.0)")

    # Privacy and compliance
    privacy_level: str = Field("public", description="Privacy classification level")
    compliance_flags: List[str] = Field(default_factory=list, description="Compliance requirements")
    retention_policy: str = Field("standard", description="Data retention policy")
    data_sensitivity: str = Field("low", description="Data sensitivity level")

    # Operational metadata
    processing_pipeline: List[str] = Field(default_factory=list, description="Processing steps applied")
    processing_duration: float = Field(0.0, description="Total processing time in seconds")
    processing_errors: List[str] = Field(default_factory=list, description="Processing errors encountered")

    # Audit trail
    audit_trail: List[Dict[str, Any]] = Field(default_factory=list, description="Complete audit history")
    created_by: str = Field("system", description="Entity that created the signal")
    updated_by: str = Field("system", description="Entity that last updated the signal")

    # Business intelligence
    business_value: float = Field(0.0, description="Business value score (0.0-1.0)")
    intelligence_priority: str = Field("medium", description="Intelligence priority level")
    action_required: bool = Field(False, description="Whether action is required")
    action_type: Optional[str] = Field(None, description="Type of action required")

    # External references
    external_references: Dict[str, str] = Field(default_factory=dict, description="External system references")
    correlation_ids: List[str] = Field(default_factory=list, description="Correlation IDs for tracking")

    # Custom metadata
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional signal metadata")

    # Validation
    # Validation temporarily disabled for import compatibility
    # @validator("confidence_score", "signal_strength", "reliability_score", "data_quality_score", "completeness_score", "timeliness_score", "consistency_score", "business_value")
    # def validate_scores(cls, v):
    #     if v is not None and not (0.0 <= v <= 1.0):
    #         raise ValueError("Score values must be between 0.0 and 1.0")
    #     return v

    # @validator("validity_end")
    # def validate_validity_period(cls, v, values):
    #     if v and "validity_start" in values and values["validity_start"] and v <= values["validity_start"]:
    #         raise ValueError("validity_end must be after validity_start")
    #     return v

    def is_valid(self) -> bool:
        """Check if the signal is currently valid."""
        now = datetime.utcnow()
        if self.validity_start and now < self.validity_start:
            return False
        if self.validity_end and now > self.validity_end:
            return False
        return self.validation_status in [SignalValidationStatus.VALIDATED, SignalValidationStatus.UNVALIDATED]

    def is_expired(self) -> bool:
        """Check if the signal has expired."""
        if self.validity_end and datetime.utcnow() > self.validity_end:
            return True
        return self.validation_status == SignalValidationStatus.EXPIRED

    def get_confidence_category(self) -> str:
        """Get human-readable confidence category."""
        if self.confidence_score >= 0.9:
            return "very_high"
        elif self.confidence_score >= 0.7:
            return "high"
        elif self.confidence_score >= 0.4:
            return "medium"
        elif self.confidence_score >= 0.2:
            return "low"
        else:
            return "very_low"

    def get_quality_score(self) -> float:
        """Calculate overall quality score from components."""
        return (self.data_quality_score + self.completeness_score +
                self.timeliness_score + self.consistency_score) / 4

    def get_signal_age_days(self) -> float:
        """Get signal age in days."""
        return (datetime.utcnow() - self.signal_timestamp).total_seconds() / (24 * 3600)

    def requires_validation(self) -> bool:
        """Check if signal requires validation."""
        return (self.validation_status == SignalValidationStatus.UNVALIDATED or
                self.validation_status == SignalValidationStatus.PENDING_VALIDATION)

    def get_audit_summary(self) -> Dict[str, Any]:
        """Get audit trail summary."""
        return {
            "total_events": len(self.audit_trail),
            "created": self.audit_trail[0]["timestamp"] if self.audit_trail else None,
            "last_updated": self.audit_trail[-1]["timestamp"] if self.audit_trail else None,
            "update_count": len(self.audit_trail),
            "created_by": self.created_by,
            "updated_by": self.updated_by
        }

    def get_compliance_summary(self) -> Dict[str, Any]:
        """Get compliance requirements summary."""
        return {
            "privacy_level": self.privacy_level,
            "compliance_flags": self.compliance_flags,
            "retention_policy": self.retention_policy,
            "data_sensitivity": self.data_sensitivity,
            "compliant": len(self.compliance_flags) == 0 or all(flag.endswith("_compliant") for flag in self.compliance_flags)
        }

    def get_processing_summary(self) -> Dict[str, Any]:
        """Get signal processing summary."""
        return {
            "processing_steps": self.processing_pipeline,
            "processing_duration_seconds": self.processing_duration,
            "processing_errors": self.processing_errors,
            "processing_success": len(self.processing_errors) == 0,
            "pipeline_length": len(self.processing_pipeline)
        }

    def to_summary_dict(self) -> Dict[str, Any]:
        """Convert to summary dictionary for API responses."""
        return {
            "signal_id": self.signal_id,
            "asset_id": self.asset_id,
            "asset_type": self.asset_type.value,
            "signal_type": self.signal_type.value,
            "signal_source": self.signal_source.value,
            "confidence_score": self.confidence_score,
            "confidence_level": self.confidence_level.value,
            "validation_status": self.validation_status.value,
            "signal_timestamp": self.signal_timestamp.isoformat(),
            "is_valid": self.is_valid(),
            "is_expired": self.is_expired(),
            "business_value": self.business_value,
            "action_required": self.action_required,
            "quality_score": self.get_quality_score(),
            "signal_age_days": self.get_signal_age_days()
        }

    def add_audit_entry(self, action: str, details: Dict[str, Any] = None, user: str = "system") -> None:
        """Add an entry to the audit trail."""
        entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "action": action,
            "user": user,
            "details": details or {}
        }
        self.audit_trail.append(entry)
        self.last_updated = datetime.utcnow()
        self.updated_by = user

    def update_validation_status(self, status: SignalValidationStatus, method: str = None, user: str = "system") -> None:
        """Update signal validation status."""
        old_status = self.validation_status
        self.validation_status = status
        self.validation_attempts += 1
        self.last_validated = datetime.utcnow()
        if method:
            self.validation_method = method

        self.add_audit_entry(
            "validation_status_changed",
            {
                "old_status": old_status.value,
                "new_status": status.value,
                "method": method,
                "attempt_number": self.validation_attempts
            },
            user
        )

    def update_confidence(self, new_score: float, reason: str = None, user: str = "system") -> None:
        """Update signal confidence score."""
        old_score = self.confidence_score
        old_level = self.confidence_level

        self.confidence_score = max(0.0, min(1.0, new_score))

        # Update confidence level
        if self.confidence_score >= 0.9:
            self.confidence_level = SignalConfidence.VERY_HIGH
        elif self.confidence_score >= 0.7:
            self.confidence_level = SignalConfidence.HIGH
        elif self.confidence_score >= 0.4:
            self.confidence_level = SignalConfidence.MEDIUM
        elif self.confidence_score >= 0.2:
            self.confidence_level = SignalConfidence.LOW
        else:
            self.confidence_level = SignalConfidence.VERY_LOW

        self.add_audit_entry(
            "confidence_updated",
            {
                "old_score": old_score,
                "new_score": self.confidence_score,
                "old_level": old_level.value,
                "new_level": self.confidence_level.value,
                "reason": reason
            },
            user
        )

    def merge_signal(self, other_signal: 'AssetSignal', merge_strategy: str = "conservative") -> 'AssetSignal':
        """Merge this signal with another signal."""
        # This is a simplified merge - in practice, this would be more sophisticated
        if merge_strategy == "conservative":
            # Take the higher confidence signal
            if other_signal.confidence_score > self.confidence_score:
                self.signal_value = other_signal.signal_value
                self.confidence_score = other_signal.confidence_score
                self.confidence_level = other_signal.confidence_level
        elif merge_strategy == "average":
            # Average the confidence scores
            self.confidence_score = (self.confidence_score + other_signal.confidence_score) / 2
            self.update_confidence(self.confidence_score, "signal_merge_average")

        # Add to related signals
        if other_signal.signal_id not in self.related_signals:
            self.related_signals.append(other_signal.signal_id)

        self.add_audit_entry("signal_merged", {
            "merged_signal_id": other_signal.signal_id,
            "merge_strategy": merge_strategy,
            "resulting_confidence": self.confidence_score
        })

        return self

    # Real Estate & Legal Intelligence Methods

    def is_property_signal(self) -> bool:
        """Check if this signal is property-related."""
        property_signals = [
            SignalType.LIEN, SignalType.MORTGAGE, SignalType.DEED,
            SignalType.FORECLOSURE, SignalType.TAX_ISSUE
        ]
        return self.signal_type in property_signals

    def is_legal_signal(self) -> bool:
        """Check if this signal is legal-related."""
        legal_signals = [
            SignalType.COURT_CASE, SignalType.JUDGMENT, SignalType.LIEN,
            SignalType.FORECLOSURE, SignalType.TAX_ISSUE
        ]
        return self.signal_type in legal_signals

    def is_personal_event_signal(self) -> bool:
        """Check if this signal is a personal life event."""
        personal_signals = [
            SignalType.BIRTHDAY, SignalType.ENGAGEMENT, SignalType.WEDDING
        ]
        return self.signal_type in personal_signals

    def get_property_address(self) -> Optional[str]:
        """Get the property address from signal data."""
        # Try different sources for property address
        if self.property_address:
            return self.property_address

        # Check signal value for address information
        signal_data = self.signal_value if isinstance(self.signal_value, dict) else {}

        address_components = []
        for field in ['address', 'property_address', 'street_address']:
            if field in signal_data and signal_data[field]:
                address_components.append(str(signal_data[field]))

        if address_components:
            return " ".join(address_components)

        return None

    def get_financial_impact(self) -> Optional[float]:
        """Get the financial impact amount from the signal."""
        if self.signal_type in [SignalType.JUDGMENT, SignalType.LIEN, SignalType.TAX_ISSUE]:
            return self.judgment_amount or self.transaction_amount or self.tax_amount

        elif self.signal_type == SignalType.MORTGAGE:
            return self.transaction_amount

        elif self.signal_type == SignalType.DEED:
            return self.transaction_amount

        # Check signal value for amount information
        signal_data = self.signal_value if isinstance(self.signal_value, dict) else {}
        for field in ['amount', 'value', 'judgment_amount', 'lien_amount', 'tax_amount']:
            if field in signal_data and isinstance(signal_data[field], (int, float)):
                return float(signal_data[field])

        return None

    def get_legal_parties(self) -> Dict[str, Optional[str]]:
        """Get legal parties involved in the signal."""
        return {
            "plaintiff": self.plaintiff_name,
            "defendant": self.defendant_name,
            "owner": self.asset_name or self.owner_name,
            "attorney": self.attorney_name
        }

    def get_risk_assessment(self) -> Dict[str, Any]:
        """Get comprehensive risk assessment for this signal."""
        risk_factors = []

        # Signal type risk weights
        signal_risks = {
            SignalType.FORECLOSURE: {"level": "critical", "weight": 1.0, "description": "Active foreclosure proceeding"},
            SignalType.JUDGMENT: {"level": "high", "weight": 0.8, "description": "Court judgment filed"},
            SignalType.LIEN: {"level": "high", "weight": 0.7, "description": "Property lien recorded"},
            SignalType.COURT_CASE: {"level": "medium", "weight": 0.6, "description": "Legal case filed"},
            SignalType.TAX_ISSUE: {"level": "medium", "weight": 0.5, "description": "Tax issue identified"},
            SignalType.MORTGAGE: {"level": "low", "weight": 0.2, "description": "Mortgage recorded"},
            SignalType.DEED: {"level": "low", "weight": 0.1, "description": "Property deed transfer"}
        }

        base_risk = signal_risks.get(self.signal_type, {"level": "low", "weight": 0.1, "description": "General signal"})

        # Financial impact risk adjustment
        financial_impact = self.get_financial_impact()
        if financial_impact:
            if financial_impact > 100000:
                base_risk["weight"] *= 1.5
                risk_factors.append("High financial impact")
            elif financial_impact > 50000:
                base_risk["weight"] *= 1.2
                risk_factors.append("Moderate financial impact")

        # Confidence adjustment
        confidence_penalty = (1.0 - self.confidence_score) * 0.3
        final_risk_score = min(1.0, base_risk["weight"] + confidence_penalty)

        return {
            "risk_level": base_risk["level"],
            "risk_score": final_risk_score,
            "base_risk_weight": base_risk["weight"],
            "confidence_penalty": confidence_penalty,
            "risk_factors": risk_factors,
            "description": base_risk["description"],
            "financial_impact": financial_impact,
            "recommendations": self._get_risk_recommendations(final_risk_score, base_risk["level"])
        }

    def _get_risk_recommendations(self, risk_score: float, risk_level: str) -> List[str]:
        """Generate risk-based recommendations."""
        recommendations = []

        if risk_level == "critical":
            recommendations.extend([
                "🚨 IMMEDIATE ACTION REQUIRED: Schedule emergency legal review",
                "📋 Conduct comprehensive due diligence investigation",
                "⚖️ Consult with legal counsel immediately",
                "📊 Perform detailed financial impact assessment",
                "🚫 Consider pausing all related transactions"
            ])
        elif risk_level == "high":
            recommendations.extend([
                "⚠️ HIGH PRIORITY: Schedule legal review within 48 hours",
                "🔍 Conduct targeted due diligence investigation",
                "📋 Review all related documentation thoroughly",
                "💰 Assess financial exposure and mitigation options",
                "📞 Consult with subject matter experts"
            ])
        elif risk_level == "medium":
            recommendations.extend([
                "📋 MEDIUM PRIORITY: Include in next legal review cycle",
                "🔍 Perform standard due diligence checks",
                "📊 Monitor situation for changes",
                "📝 Document findings and considerations"
            ])

        # Financial impact recommendations
        financial_impact = self.get_financial_impact()
        if financial_impact and financial_impact > 25000:
            recommendations.append(f"💰 HIGH FINANCIAL IMPACT: ${financial_impact:,.2f} exposure detected")
        elif financial_impact and financial_impact > 10000:
            recommendations.append(f"💰 SIGNIFICANT FINANCIAL IMPACT: ${financial_impact:,.2f} exposure detected")
        return recommendations[:5]  # Limit to top 5

    def get_property_intelligence(self) -> Dict[str, Any]:
        """Get property-specific intelligence from the signal."""
        if not self.is_property_signal():
            return {"error": "Not a property-related signal"}

        intelligence = {
            "property_address": self.get_property_address(),
            "parcel_id": self.parcel_id,
            "property_value": self.property_value,
            "owner_name": self.asset_name or self.owner_name,
            "signal_type": self.signal_type.value,
            "financial_impact": self.get_financial_impact(),
            "legal_parties": self.get_legal_parties(),
            "risk_assessment": self.get_risk_assessment()
        }

        # Add signal-specific intelligence
        if self.signal_type == SignalType.LIEN:
            intelligence.update({
                "lien_type": "property_lien",
                "title_implication": "title_search_required",
                "due_diligence_priority": "high"
            })
        elif self.signal_type == SignalType.MORTGAGE:
            intelligence.update({
                "lender": self.lender_name,
                "mortgage_type": self.mortgage_type,
                "loan_amount": self.transaction_amount,
                "title_implication": "mortgage_search_required"
            })
        elif self.signal_type == SignalType.DEED:
            intelligence.update({
                "transfer_amount": self.transaction_amount,
                "ownership_change": True,
                "title_implication": "ownership_verification_required",
                "due_diligence_priority": "critical"
            })
        elif self.signal_type == SignalType.FORECLOSURE:
            intelligence.update({
                "distressed_property": True,
                "market_impact": "negative",
                "due_diligence_priority": "critical",
                "investment_risk": "high"
            })

        return intelligence

    def get_compliance_requirements(self) -> List[str]:
        """Get compliance requirements for this signal type."""
        base_requirements = []

        # Signal type specific requirements
        if self.signal_type in [SignalType.JUDGMENT, SignalType.LIEN, SignalType.FORECLOSURE]:
            base_requirements.extend([
                "title_search_required",
                "financial_due_diligence_required",
                "legal_review_recommended"
            ])

        if self.signal_type == SignalType.TAX_ISSUE:
            base_requirements.extend([
                "tax_liability_assessment",
                "county_records_review",
                "payment_verification_required"
            ])

        if self.is_personal_event_signal():
            base_requirements.extend([
                "privacy_compliance_check",
                "data_retention_policy_review"
            ])

        # Financial impact based requirements
        financial_impact = self.get_financial_impact()
        if financial_impact and financial_impact > 50000:
            base_requirements.append("senior_management_review_required")

        return list(set(base_requirements + self.compliance_flags))


# Signal processing utilities
def create_identity_signal(
    asset_id: str,
    signal_value: Dict[str, Any],
    confidence_score: float = 0.8,
    **kwargs
) -> AssetSignal:
    """Create an identity confirmation signal."""
    return AssetSignal(
        asset_id=asset_id,
        asset_type=AssetType.PERSON,
        signal_type=SignalType.IDENTITY,
        signal_value=signal_value,
        confidence_score=confidence_score,
        **kwargs
    )


def create_location_signal(
    asset_id: str,
    location_data: Dict[str, Any],
    confidence_score: float = 0.7,
    **kwargs
) -> AssetSignal:
    """Create a location/geography signal."""
    return AssetSignal(
        asset_id=asset_id,
        signal_type=SignalType.LOCATION,
        signal_value=location_data,
        confidence_score=confidence_score,
        **kwargs
    )


def create_contact_signal(
    asset_id: str,
    contact_data: Dict[str, Any],
    confidence_score: float = 0.6,
    **kwargs
) -> AssetSignal:
    """Create a contact information signal."""
    return AssetSignal(
        asset_id=asset_id,
        signal_type=SignalType.CONTACT,
        signal_value=contact_data,
        confidence_score=confidence_score,
        **kwargs
    )


def create_professional_signal(
    asset_id: str,
    professional_data: Dict[str, Any],
    confidence_score: float = 0.75,
    **kwargs
) -> AssetSignal:
    """Create a professional/career signal."""
    return AssetSignal(
        asset_id=asset_id,
        signal_type=SignalType.PROFESSIONAL,
        signal_value=professional_data,
        confidence_score=confidence_score,
        **kwargs
    )


def create_relationship_signal(
    asset_id: str,
    relationship_data: Dict[str, Any],
    confidence_score: float = 0.65,
    **kwargs
) -> AssetSignal:
    """Create a relationship/connection signal."""
    return AssetSignal(
        asset_id=asset_id,
        signal_type=SignalType.RELATIONSHIP,
        signal_value=relationship_data,
        confidence_score=confidence_score,
        **kwargs
    )


def create_anomaly_signal(
    asset_id: str,
    anomaly_data: Dict[str, Any],
    confidence_score: float = 0.9,
    **kwargs
) -> AssetSignal:
    """Create an anomalous activity signal."""
    return AssetSignal(
        asset_id=asset_id,
        signal_type=SignalType.ANOMALY,
        signal_value=anomaly_data,
        confidence_score=confidence_score,
        action_required=True,
        action_type="investigation",
        **kwargs
    )


# Real Estate & Legal Signal Creation Helpers

def create_lien_signal(
    asset_id: str,
    lien_data: Dict[str, Any],
    confidence_score: float = 0.9,
    **kwargs
) -> AssetSignal:
    """Create a property lien signal."""
    return AssetSignal(
        asset_id=asset_id,
        asset_type=AssetType.SINGLE_FAMILY_HOME,  # Default, can be overridden
        signal_type=SignalType.LIEN,
        signal_value=lien_data,
        confidence_score=confidence_score,
        action_required=True,
        action_type="title_search_review",
        intelligence_priority="high",
        business_value=0.8,
        **kwargs
    )


def create_mortgage_signal(
    asset_id: str,
    mortgage_data: Dict[str, Any],
    confidence_score: float = 0.85,
    **kwargs
) -> AssetSignal:
    """Create a mortgage information signal."""
    return AssetSignal(
        asset_id=asset_id,
        signal_type=SignalType.MORTGAGE,
        signal_value=mortgage_data,
        confidence_score=confidence_score,
        intelligence_priority="medium",
        business_value=0.6,
        **kwargs
    )


def create_deed_signal(
    asset_id: str,
    deed_data: Dict[str, Any],
    confidence_score: float = 0.95,
    **kwargs
) -> AssetSignal:
    """Create a property deed transfer signal."""
    return AssetSignal(
        asset_id=asset_id,
        signal_type=SignalType.DEED,
        signal_value=deed_data,
        confidence_score=confidence_score,
        action_required=True,
        action_type="ownership_verification",
        intelligence_priority="high",
        business_value=0.9,
        **kwargs
    )


def create_foreclosure_signal(
    asset_id: str,
    foreclosure_data: Dict[str, Any],
    confidence_score: float = 0.95,
    **kwargs
) -> AssetSignal:
    """Create a foreclosure proceeding signal."""
    return AssetSignal(
        asset_id=asset_id,
        signal_type=SignalType.FORECLOSURE,
        signal_value=foreclosure_data,
        confidence_score=confidence_score,
        action_required=True,
        action_type="immediate_investigation",
        intelligence_priority="critical",
        business_value=0.95,
        compliance_flags=["distressed_property_review"],
        **kwargs
    )


def create_court_case_signal(
    asset_id: str,
    court_data: Dict[str, Any],
    confidence_score: float = 0.85,
    **kwargs
) -> AssetSignal:
    """Create a court case signal."""
    return AssetSignal(
        asset_id=asset_id,
        signal_type=SignalType.COURT_CASE,
        signal_value=court_data,
        confidence_score=confidence_score,
        action_required=True,
        action_type="legal_review",
        intelligence_priority="high",
        business_value=0.75,
        data_sensitivity="high",
        **kwargs
    )


def create_judgment_signal(
    asset_id: str,
    judgment_data: Dict[str, Any],
    confidence_score: float = 0.9,
    **kwargs
) -> AssetSignal:
    """Create a court judgment signal."""
    return AssetSignal(
        asset_id=asset_id,
        signal_type=SignalType.JUDGMENT,
        signal_value=judgment_data,
        confidence_score=confidence_score,
        action_required=True,
        action_type="judgment_analysis",
        intelligence_priority="high",
        business_value=0.85,
        data_sensitivity="high",
        compliance_flags=["financial_liability_review"],
        **kwargs
    )


def create_tax_issue_signal(
    asset_id: str,
    tax_data: Dict[str, Any],
    confidence_score: float = 0.8,
    **kwargs
) -> AssetSignal:
    """Create a tax issue signal."""
    return AssetSignal(
        asset_id=asset_id,
        signal_type=SignalType.TAX_ISSUE,
        signal_value=tax_data,
        confidence_score=confidence_score,
        action_required=True,
        action_type="tax_liability_review",
        intelligence_priority="medium",
        business_value=0.7,
        compliance_flags=["tax_compliance_review"],
        **kwargs
    )


def create_birthday_signal(
    asset_id: str,
    birthday_data: Dict[str, Any],
    confidence_score: float = 0.75,
    **kwargs
) -> AssetSignal:
    """Create a birthday information signal."""
    return AssetSignal(
        asset_id=asset_id,
        asset_type=AssetType.PERSON,
        signal_type=SignalType.BIRTHDAY,
        signal_value=birthday_data,
        confidence_score=confidence_score,
        intelligence_priority="low",
        business_value=0.3,
        privacy_level="personal",
        **kwargs
    )


def create_engagement_signal(
    asset_id: str,
    engagement_data: Dict[str, Any],
    confidence_score: float = 0.7,
    **kwargs
) -> AssetSignal:
    """Create an engagement announcement signal."""
    return AssetSignal(
        asset_id=asset_id,
        asset_type=AssetType.PERSON,
        signal_type=SignalType.ENGAGEMENT,
        signal_value=engagement_data,
        confidence_score=confidence_score,
        intelligence_priority="low",
        business_value=0.4,
        privacy_level="personal",
        **kwargs
    )


def create_wedding_signal(
    asset_id: str,
    wedding_data: Dict[str, Any],
    confidence_score: float = 0.8,
    **kwargs
) -> AssetSignal:
    """Create a wedding information signal."""
    return AssetSignal(
        asset_id=asset_id,
        asset_type=AssetType.PERSON,
        signal_type=SignalType.WEDDING,
        signal_value=wedding_data,
        confidence_score=confidence_score,
        intelligence_priority="medium",
        business_value=0.5,
        privacy_level="personal",
        **kwargs
    )


# Signal quality assessment
def assess_signal_quality(signal: AssetSignal) -> Dict[str, Any]:
    """Assess the overall quality of a signal."""
    quality_factors = {
        "confidence": signal.confidence_score,
        "completeness": signal.completeness_score,
        "timeliness": signal.timeliness_score,
        "consistency": signal.consistency_score,
        "source_reliability": signal.reliability_score,
        "validation_status": 1.0 if signal.validation_status == SignalValidationStatus.VALIDATED else 0.5,
        "data_freshness": max(0.0, 1.0 - (signal.get_signal_age_days() / 365))  # Degrade over year
    }

    overall_quality = sum(quality_factors.values()) / len(quality_factors)

    return {
        "overall_quality": overall_quality,
        "quality_factors": quality_factors,
        "quality_category": "high" if overall_quality >= 0.8 else "medium" if overall_quality >= 0.6 else "low",
        "recommendations": _get_quality_recommendations(quality_factors)
    }


def _get_quality_recommendations(quality_factors: Dict[str, float]) -> List[str]:
    """Generate quality improvement recommendations."""
    recommendations = []

    if quality_factors["confidence"] < 0.7:
        recommendations.append("Increase signal confidence through validation")
    if quality_factors["completeness"] < 0.7:
        recommendations.append("Improve data completeness")
    if quality_factors["timeliness"] < 0.7:
        recommendations.append("Refresh signal data for better timeliness")
    if quality_factors["consistency"] < 0.7:
        recommendations.append("Address data consistency issues")
    if quality_factors["source_reliability"] < 0.7:
        recommendations.append("Verify source reliability")
    if quality_factors["validation_status"] < 1.0:
        recommendations.append("Complete signal validation")
    if quality_factors["data_freshness"] < 0.7:
        recommendations.append("Update stale signal data")

    return recommendations if recommendations else ["Signal quality is acceptable"]


# Signal validation utilities
def validate_signal_completeness(signal: AssetSignal) -> float:
    """Validate signal data completeness."""
    required_fields = {
        SignalType.IDENTITY: ["asset_name", "asset_identifiers"],
        SignalType.LOCATION: ["location_country"],
        SignalType.CONTACT: ["signal_value"],
        SignalType.PROFESSIONAL: ["signal_value"],
        SignalType.SOCIAL: ["signal_value"],
        SignalType.FINANCIAL: ["signal_value"],
        SignalType.LEGAL: ["signal_value"],
        SignalType.BEHAVIORAL: ["signal_value"],
        SignalType.RELATIONSHIP: ["signal_value"],
        SignalType.EVENT: ["signal_value"],
        SignalType.ANOMALY: ["signal_value"]
    }

    required = required_fields.get(signal.signal_type, [])
    present = sum(1 for field in required if getattr(signal, field, None) is not None)
    return present / len(required) if required else 1.0


def validate_signal_consistency(signal: AssetSignal) -> float:
    """Validate signal data consistency."""
    # Basic consistency checks
    consistency_score = 1.0

    # Check timestamp consistency
    if signal.validity_start and signal.validity_end:
        if signal.validity_end <= signal.validity_start:
            consistency_score *= 0.5

    # Check confidence bounds
    if not (0.0 <= signal.confidence_score <= 1.0):
        consistency_score *= 0.5

    # Check signal age vs validity
    age_days = signal.get_signal_age_days()
    if signal.validity_end:
        validity_days = (signal.validity_end - signal.signal_timestamp).days
        if age_days > validity_days:
            consistency_score *= 0.7

    return consistency_score


def validate_signal_timeliness(signal: AssetSignal) -> float:
    """Validate signal timeliness."""
    age_days = signal.get_signal_age_days()

    # Different timeliness expectations by signal type
    timeliness_thresholds = {
        SignalType.IDENTITY: 365,      # 1 year
        SignalType.LOCATION: 180,      # 6 months
        SignalType.CONTACT: 90,        # 3 months
        SignalType.PROFESSIONAL: 180,  # 6 months
        SignalType.SOCIAL: 30,         # 1 month
        SignalType.FINANCIAL: 90,      # 3 months
        SignalType.LEGAL: 365,         # 1 year
        SignalType.BEHAVIORAL: 7,      # 1 week
        SignalType.RELATIONSHIP: 180,  # 6 months
        SignalType.EVENT: 1,           # 1 day
        SignalType.ANOMALY: 1          # 1 day
    }

    threshold = timeliness_thresholds.get(signal.signal_type, 180)
    return max(0.0, 1.0 - (age_days / threshold))


# Signal aggregation and fusion
def fuse_signals(signals: List[AssetSignal], fusion_method: str = "weighted_average") -> AssetSignal:
    """
    Fuse multiple signals about the same asset into a single high-confidence signal.
    """
    if not signals:
        raise ValueError("Cannot fuse empty signal list")

    if len(signals) == 1:
        return signals[0]

    # Use the first signal as base
    fused_signal = signals[0].copy()

    # Apply fusion method
    if fusion_method == "weighted_average":
        # Weight by confidence score
        total_weight = sum(s.confidence_score for s in signals)
        if total_weight > 0:
            fused_signal.confidence_score = sum(s.confidence_score * s.confidence_score for s in signals) / total_weight
        else:
            fused_signal.confidence_score = sum(s.confidence_score for s in signals) / len(signals)

    elif fusion_method == "highest_confidence":
        # Take the highest confidence signal
        best_signal = max(signals, key=lambda s: s.confidence_score)
        fused_signal = best_signal.copy()

    elif fusion_method == "most_recent":
        # Take the most recent signal
        best_signal = max(signals, key=lambda s: s.signal_timestamp)
        fused_signal = best_signal.copy()

    # Update fusion metadata
    fused_signal.fusion_sources = [s.signal_id for s in signals]
    fused_signal.fusion_confidence = fused_signal.confidence_score
    fused_signal.fusion_method = fusion_method
    fused_signal.related_signals = [s.signal_id for s in signals[1:]]

    # Update validation status
    fused_signal.validation_status = SignalValidationStatus.VALIDATED
    fused_signal.update_validation_status(SignalValidationStatus.VALIDATED, f"fused_{len(signals)}_signals")

    # Add audit entry
    fused_signal.add_audit_entry("signal_fusion", {
        "fusion_method": fusion_method,
        "source_signals": len(signals),
        "resulting_confidence": fused_signal.confidence_score
    })

    return fused_signal


def deduplicate_signals(signals: List[AssetSignal], similarity_threshold: float = 0.8) -> List[AssetSignal]:
    """
    Remove duplicate or highly similar signals.
    """
    if not signals:
        return []

    unique_signals = []

    for signal in signals:
        is_duplicate = False

        for unique_signal in unique_signals:
            # Simple similarity check based on signal value and confidence
            if (signal.signal_type == unique_signal.signal_type and
                signal.asset_id == unique_signal.asset_id and
                abs(signal.confidence_score - unique_signal.confidence_score) < (1 - similarity_threshold)):
                is_duplicate = True
                break

        if not is_duplicate:
            unique_signals.append(signal)

    return unique_signals
