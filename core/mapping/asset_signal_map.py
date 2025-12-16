# Copyright (c) 2024 Mountain Jewels Intelligence. All rights reserved.
#
# This software is proprietary and confidential. Unauthorized copying,
# modification, distribution, or use is strictly prohibited.

"""
Asset Signal Mapping Engine for MJ Data Scraper Suite

Comprehensive mapping and transformation layer for converting raw scraped data
into standardized asset signals with intelligent identification, deduplication,
enrichment, and cross-referencing capabilities.
"""

import asyncio
import logging
import re
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Set, Union
from collections import defaultdict, Counter
from dataclasses import dataclass, field

try:
    from fuzzywuzzy import fuzz
    from fuzzywuzzy.process import extractOne
    FUZZY_AVAILABLE = True
except ImportError:
    FUZZY_AVAILABLE = False
    # Fallback implementations
    def fuzz_ratio(a, b): return 100 if a.lower() == b.lower() else 0
    def extractOne(query, choices): return (query, 100) if query in choices else (None, 0)

from core.models.asset_signal import (
    AssetSignal,
    Asset,
    SignalType,
    SignalSource,
    AssetType,
    SignalRequest,
    create_identity_signal,
    create_location_signal,
    create_contact_signal,
    create_professional_signal,
    create_lien_signal,
    create_mortgage_signal,
    create_deed_signal,
    create_foreclosure_signal,
    create_court_case_signal,
    create_judgment_signal,
    create_tax_issue_signal,
    create_birthday_signal,
    create_engagement_signal,
    create_wedding_signal
)

# Enhanced Asset Signal Source Configuration
# Maps asset types to signal types and their optimal data sources
ASSET_SIGNAL_SOURCES = {
    AssetType.SINGLE_FAMILY_HOME: {
        SignalType.LIEN: ["county_clerk", "tax_assessor", "municipal_records"],
        SignalType.MORTGAGE: ["county_recorder", "title_company", "financial_records"],
        SignalType.DEED: ["county_recorder", "title_company", "property_records"],
        SignalType.FORECLOSURE: ["county_clerk", "court_records", "state_registry"],
        SignalType.COURT_CASE: ["local_court", "county_court", "municipal_court"],
        SignalType.JUDGMENT: ["county_clerk", "court_records", "judgment_lien_records"],
        SignalType.TAX_ISSUE: ["county_assessor", "tax_collector", "municipal_finance"]
    },
    AssetType.MULTI_FAMILY_SMALL: {
        SignalType.LIEN: ["county_clerk", "tax_assessor", "state_registry"],
        SignalType.MORTGAGE: ["county_recorder", "commercial_lenders", "title_company"],
        SignalType.DEED: ["county_recorder", "commercial_title", "property_records"],
        SignalType.COURT_CASE: ["state_court", "county_court", "commercial_court"],
        SignalType.FORECLOSURE: ["state_registry", "commercial_foreclosure_records"],
        SignalType.JUDGMENT: ["county_clerk", "commercial_judgments", "state_records"],
        SignalType.TAX_ISSUE: ["county_assessor", "commercial_tax_records", "state_tax"]
    },
    AssetType.APARTMENT_BUILDING: {
        SignalType.LIEN: ["county_clerk", "state_registry", "commercial_liens"],
        SignalType.MORTGAGE: ["commercial_lenders", "investment_banks", "title_company"],
        SignalType.DEED: ["county_recorder", "commercial_title", "property_records"],
        SignalType.COURT_CASE: ["state_court", "federal_bankruptcy", "commercial_court"],
        SignalType.FORECLOSURE: ["state_registry", "commercial_foreclosure", "federal_records"],
        SignalType.JUDGMENT: ["commercial_judgments", "state_court_records", "federal_liens"],
        SignalType.TAX_ISSUE: ["county_assessor", "commercial_tax", "state_tax_authority"]
    },
    AssetType.COMMERCIAL_PROPERTY: {
        SignalType.LIEN: ["county_clerk", "state_registry", "commercial_liens", "federal_liens"],
        SignalType.MORTGAGE: ["commercial_banks", "investment_banks", "title_company"],
        SignalType.DEED: ["county_recorder", "commercial_title", "property_records"],
        SignalType.COURT_CASE: ["federal_court", "state_court", "commercial_litigation"],
        SignalType.FORECLOSURE: ["federal_registry", "state_registry", "commercial_foreclosure"],
        SignalType.JUDGMENT: ["federal_court_records", "state_judgments", "commercial_liens"],
        SignalType.TAX_ISSUE: ["federal_tax_records", "state_tax", "county_assessor"]
    },
    AssetType.PERSON: {
        SignalType.BIRTHDAY: ["local_newspapers", "social_media", "public_records", "family_announcements"],
        SignalType.ENGAGEMENT: ["newspapers", "event_announcements", "social_media", "wedding_sites"],
        SignalType.WEDDING: ["newspapers", "marriage_records", "wedding_announcements", "social_media"],
        SignalType.COURT_CASE: ["county_court", "state_court", "federal_court", "civil_records"],
        SignalType.JUDGMENT: ["county_clerk", "court_records", "judgment_database", "civil_liens"],
        SignalType.IDENTITY: ["dmv_records", "social_security", "passport_records", "voter_registration"],
        SignalType.CONTACT: ["phone_directories", "social_media", "professional_networks", "public_records"],
        SignalType.PROFESSIONAL: ["linkedin", "company_websites", "professional_associations", "news_articles"],
        SignalType.FINANCIAL: ["credit_reports", "property_records", "court_records", "business_registrations"]
    },
    AssetType.COMPANY: {
        SignalType.COURT_CASE: ["federal_court", "state_court", "commercial_court", "sec_filings"],
        SignalType.JUDGMENT: ["federal_liens", "state_judgments", "commercial_liens", "court_records"],
        SignalType.FINANCIAL: ["sec_filings", "financial_reports", "credit_reports", "business_registrations"],
        SignalType.LEGAL: ["state_secretary", "federal_registrations", "court_records", "legal_filings"],
        SignalType.PROFESSIONAL: ["company_websites", "business_news", "industry_reports", "executive_profiles"]
    }
}

# Enhanced Signal Cost Weight Configuration
# Weights represent relative cost/complexity of acquiring each signal type
SIGNAL_COST_WEIGHT = {
    # Personal Events (Lowest cost, easiest to find)
    SignalType.BIRTHDAY: 1.0,
    SignalType.ENGAGEMENT: 1.3,
    SignalType.WEDDING: 1.6,

    # Property Records (Moderate cost, public records)
    SignalType.DEED: 1.8,
    SignalType.MORTGAGE: 2.0,
    SignalType.LIEN: 2.0,
    SignalType.TAX_ISSUE: 2.2,

    # Legal Records (Higher cost, court access required)
    SignalType.COURT_CASE: 3.5,
    SignalType.JUDGMENT: 3.8,

    # High-Risk Legal Proceedings (Highest cost, complex research)
    SignalType.FORECLOSURE: 4.0,

    # Identity & Professional (Variable cost based on source)
    SignalType.IDENTITY: 2.5,
    SignalType.CONTACT: 1.5,
    SignalType.PROFESSIONAL: 2.8,
    SignalType.FINANCIAL: 3.2,
    SignalType.LEGAL: 3.0,

    # Behavioral & Relationship (Social media/web scraping)
    SignalType.BEHAVIORAL: 2.5,
    SignalType.RELATIONSHIP: 2.2,
    SignalType.EVENT: 1.8,

    # Anomaly Detection (Advanced analysis required)
    SignalType.ANOMALY: 4.5
}

# Source Quality and Reliability Scores
# Higher scores indicate more reliable, authoritative sources
SOURCE_RELIABILITY = {
    # Government Records (Highest reliability)
    "county_clerk": 0.95,
    "county_recorder": 0.95,
    "state_registry": 0.90,
    "federal_court": 0.95,
    "state_court": 0.90,
    "county_court": 0.88,
    "federal_registry": 0.95,
    "dmv_records": 0.92,
    "passport_records": 0.95,
    "voter_registration": 0.85,
    "social_security": 0.98,

    # Financial Institutions
    "title_company": 0.85,
    "commercial_banks": 0.80,
    "investment_banks": 0.82,
    "credit_reports": 0.75,

    # Public Records
    "tax_assessor": 0.88,
    "county_assessor": 0.88,
    "tax_collector": 0.85,
    "municipal_records": 0.82,
    "property_records": 0.85,
    "court_records": 0.88,
    "judgment_lien_records": 0.86,
    "civil_records": 0.80,

    # Commercial/Business Records
    "sec_filings": 0.90,
    "financial_reports": 0.82,
    "business_registrations": 0.85,
    "commercial_liens": 0.80,
    "commercial_judgments": 0.78,
    "state_secretary": 0.88,

    # News and Media (Lower reliability due to potential bias/errors)
    "newspapers": 0.60,
    "local_newspapers": 0.65,
    "business_news": 0.65,
    "news_articles": 0.62,

    # Social and Web Sources (Variable reliability)
    "social_media": 0.45,
    "linkedin": 0.55,
    "company_websites": 0.70,
    "wedding_sites": 0.50,
    "event_announcements": 0.55,
    "wedding_announcements": 0.58,
    "professional_networks": 0.60,
    "family_announcements": 0.52,

    # Specialized Sources
    "professional_associations": 0.75,
    "industry_reports": 0.72,
    "executive_profiles": 0.68,
    "phone_directories": 0.55,
    "federal_tax_records": 0.92,
    "state_tax": 0.85,
    "commercial_tax": 0.78,
    "federal_bankruptcy": 0.90,
    "commercial_court": 0.82,
    "commercial_litigation": 0.80,
    "municipal_court": 0.78,
    "local_court": 0.76,
    "federal_liens": 0.93,
    "commercial_title": 0.82,
    "commercial_lenders": 0.78,
    "commercial_foreclosure": 0.80,
    "federal_records": 0.90
}

# Data Freshness Requirements (in days)
# How often signal data should be refreshed
SIGNAL_FRESHNESS_REQUIREMENTS = {
    SignalType.BIRTHDAY: 365 * 10,  # Birthdays don't change
    SignalType.ENGAGEMENT: 365 * 2,  # Engagements are short-term
    SignalType.WEDDING: 365 * 5,     # Wedding info relatively stable
    SignalType.DEED: 365 * 2,        # Property transfers happen occasionally
    SignalType.MORTGAGE: 180,        # Mortgages can change frequently
    SignalType.LIEN: 90,             # Liens can be added/removed quickly
    SignalType.COURT_CASE: 30,       # Active cases change rapidly
    SignalType.JUDGMENT: 180,        # Judgments may have appeal periods
    SignalType.FORECLOSURE: 7,       # Foreclosure proceedings are time-critical
    SignalType.TAX_ISSUE: 30,        # Tax issues can change with payments
    SignalType.IDENTITY: 365,        # Identity info relatively stable
    SignalType.CONTACT: 90,          # Contact info changes moderately
    SignalType.PROFESSIONAL: 180,    # Career changes happen
    SignalType.FINANCIAL: 30,        # Financial status can change quickly
    SignalType.LEGAL: 90,            # Legal status can change
    SignalType.BEHAVIORAL: 7,        # Behavioral data is very current
    SignalType.RELATIONSHIP: 30,     # Relationships change
    SignalType.EVENT: 1,             # Events are immediate
    SignalType.ANOMALY: 1            # Anomalies require immediate attention
}

logger = logging.getLogger(__name__)


@dataclass
class MappingRule:
    """Rule for mapping raw data to standardized signals."""
    source_pattern: str
    signal_type: SignalType
    field_mappings: Dict[str, str] = field(default_factory=dict)
    validation_rules: Dict[str, Any] = field(default_factory=dict)
    confidence_boost: float = 0.0
    priority: int = 1
    description: str = ""


@dataclass
class AssetIdentifier:
    """Standardized asset identifier with multiple reference types."""
    primary_id: str
    asset_type: AssetType
    identifiers: Dict[str, str] = field(default_factory=dict)
    confidence_score: float = 1.0
    last_updated: datetime = field(default_factory=datetime.utcnow)
    source_signals: List[str] = field(default_factory=list)

    def get_canonical_id(self) -> str:
        """Get the canonical identifier for this asset."""
        return self.primary_id

    def add_identifier(self, id_type: str, value: str, confidence: float = 1.0):
        """Add an additional identifier."""
        self.identifiers[id_type] = value
        self.confidence_score = min(self.confidence_score, confidence)
        self.last_updated = datetime.utcnow()

    def matches_identifier(self, id_type: str, value: str, fuzzy: bool = False) -> bool:
        """Check if this asset matches the given identifier."""
        if id_type in self.identifiers:
            existing_value = self.identifiers[id_type]
            if fuzzy and FUZZY_AVAILABLE:
                return fuzz_ratio(existing_value, value) >= 80
            else:
                return existing_value.lower() == value.lower()
        return False


class AssetSignalMapper:
    """
    Enterprise-grade mapping engine for asset signal transformation.

    Converts raw scraped data into standardized asset signals with intelligent
    identification, deduplication, enrichment, and cross-referencing.
    """

    def __init__(self):
        self.mapping_rules: List[MappingRule] = []
        self.asset_identifiers: Dict[str, AssetIdentifier] = {}
        self.signal_cache: Dict[str, AssetSignal] = {}
        self.duplicate_map: Dict[str, List[str]] = defaultdict(list)

        # Fuzzy matching settings
        self.fuzzy_threshold = 80
        self.name_similarity_threshold = 85
        self.address_similarity_threshold = 90

        # Performance tracking
        self.mapping_stats = defaultdict(int)
        self.cache_hits = 0
        self.cache_misses = 0

        # Source intelligence tracking
        self.source_performance = defaultdict(lambda: {"success": 0, "failure": 0, "avg_confidence": 0.0})
        self.signal_cost_tracking = defaultdict(float)

        # Initialize default mapping rules
        self._initialize_default_rules()

        logger.info("AssetSignalMapper initialized with intelligent source configuration")

    def _initialize_default_rules(self):
        """Initialize default mapping rules for common data sources."""

        # Real estate records mapping
        self.add_mapping_rule(MappingRule(
            source_pattern=r".*lien.*|.*encumbrance.*",
            signal_type=SignalType.LIEN,
            field_mappings={
                "amount": "judgment_amount",
                "recorded_date": "filing_date",
                "property_address": "property_address",
                "parcel_id": "parcel_id"
            },
            confidence_boost=0.1,
            priority=2,
            description="Property lien and encumbrance records"
        ))

        self.add_mapping_rule(MappingRule(
            source_pattern=r".*mortgage.*|.*loan.*",
            signal_type=SignalType.MORTGAGE,
            field_mappings={
                "loan_amount": "transaction_amount",
                "lender": "lender_name",
                "interest_rate": "interest_rate",
                "loan_term": "loan_term_years"
            },
            confidence_boost=0.1,
            priority=2,
            description="Mortgage and loan records"
        ))

        self.add_mapping_rule(MappingRule(
            source_pattern=r".*deed.*|.*transfer.*",
            signal_type=SignalType.DEED,
            field_mappings={
                "sale_price": "transaction_amount",
                "grantor": "plaintiff_name",
                "grantee": "defendant_name",
                "property_address": "property_address"
            },
            confidence_boost=0.15,
            priority=3,
            description="Property deed transfers"
        ))

        self.add_mapping_rule(MappingRule(
            source_pattern=r".*foreclosure.*|.*notice.*sale.*",
            signal_type=SignalType.FORECLOSURE,
            field_mappings={
                "case_number": "case_number",
                "property_address": "property_address",
                "amount": "transaction_amount",
                "trustee": "attorney_name"
            },
            confidence_boost=0.2,
            priority=4,
            description="Foreclosure proceedings"
        ))

        # Legal records mapping
        self.add_mapping_rule(MappingRule(
            source_pattern=r".*judgment.*|.*ruling.*",
            signal_type=SignalType.JUDGMENT,
            field_mappings={
                "case_number": "case_number",
                "amount": "judgment_amount",
                "plaintiff": "plaintiff_name",
                "defendant": "defendant_name",
                "court": "court_name"
            },
            confidence_boost=0.1,
            priority=3,
            description="Court judgments and rulings"
        ))

        self.add_mapping_rule(MappingRule(
            source_pattern=r".*court.*case.*|.*lawsuit.*",
            signal_type=SignalType.COURT_CASE,
            field_mappings={
                "case_number": "case_number",
                "plaintiff": "plaintiff_name",
                "defendant": "defendant_name",
                "court": "court_name",
                "filing_date": "filing_date"
            },
            confidence_boost=0.1,
            priority=2,
            description="Court case filings"
        ))

        # Tax records mapping
        self.add_mapping_rule(MappingRule(
            source_pattern=r".*tax.*lien.*|.*delinquent.*tax.*",
            signal_type=SignalType.TAX_ISSUE,
            field_mappings={
                "amount": "tax_amount",
                "tax_year": "tax_year",
                "due_date": "filing_date",
                "property_address": "property_address"
            },
            confidence_boost=0.1,
            priority=2,
            description="Tax liens and issues"
        ))

        # Personal event mapping
        self.add_mapping_rule(MappingRule(
            source_pattern=r".*birthday.*|.*birth.*date.*",
            signal_type=SignalType.BIRTHDAY,
            field_mappings={
                "birth_date": "event_date",
                "full_name": "asset_name"
            },
            confidence_boost=0.05,
            priority=1,
            description="Birthday information"
        ))

        self.add_mapping_rule(MappingRule(
            source_pattern=r".*engagement.*|.*betrothal.*",
            signal_type=SignalType.ENGAGEMENT,
            field_mappings={
                "partner_name": "spouse_name",
                "announcement_date": "event_date",
                "venue": "venue_name"
            },
            confidence_boost=0.05,
            priority=1,
            description="Engagement announcements"
        ))

        self.add_mapping_rule(MappingRule(
            source_pattern=r".*wedding.*|.*marriage.*",
            signal_type=SignalType.WEDDING,
            field_mappings={
                "spouse_name": "spouse_name",
                "wedding_date": "event_date",
                "venue": "venue_name",
                "description": "event_description"
            },
            confidence_boost=0.05,
            priority=1,
            description="Wedding information"
        ))

    def add_mapping_rule(self, rule: MappingRule):
        """Add a custom mapping rule."""
        self.mapping_rules.append(rule)
        self.mapping_rules.sort(key=lambda x: x.priority, reverse=True)
        logger.info(f"Added mapping rule: {rule.description} (priority: {rule.priority})")

    async def map_raw_data_to_signal(
        self,
        raw_data: Dict[str, Any],
        source: SignalSource,
        context: Optional[Dict[str, Any]] = None
    ) -> Optional[AssetSignal]:
        """
        Map raw scraped data to a standardized asset signal.

        Args:
            raw_data: Raw data from scraping source
            source: Source of the data
            context: Additional context information

        Returns:
            Standardized AssetSignal or None if mapping fails
        """
        try:
            # Find matching mapping rule
            rule = self._find_matching_rule(raw_data)
            if not rule:
                logger.debug(f"No mapping rule found for data: {list(raw_data.keys())}")
                return None

            # Apply mapping rule
            signal = await self._apply_mapping_rule(raw_data, rule, source, context)

            if signal:
                # Enhance with additional intelligence
                await self._enhance_signal(signal, raw_data, context)

                # Cache the signal
                self.signal_cache[signal.signal_id] = signal
                self.mapping_stats['signals_mapped'] += 1

                logger.debug(f"✅ Mapped raw data to signal: {signal.signal_type.value} for asset {signal.asset_id}")
                return signal

        except Exception as e:
            logger.error(f"❌ Failed to map raw data to signal: {e}")
            self.mapping_stats['mapping_errors'] += 1

        return None

    def _find_matching_rule(self, raw_data: Dict[str, Any]) -> Optional[MappingRule]:
        """Find the best matching mapping rule for raw data."""
        # Convert data to searchable text
        data_text = " ".join(str(v) for v in raw_data.values() if isinstance(v, str)).lower()

        best_match = None
        best_score = 0

        for rule in self.mapping_rules:
            # Check if pattern matches any key or value
            pattern_match = False

            # Check keys
            for key in raw_data.keys():
                if re.search(rule.source_pattern, key, re.IGNORECASE):
                    pattern_match = True
                    break

            # Check values if keys didn't match
            if not pattern_match:
                if re.search(rule.source_pattern, data_text):
                    pattern_match = True

            if pattern_match:
                # Calculate match score based on priority and data completeness
                score = rule.priority * 10
                required_fields = set(rule.field_mappings.values())
                available_fields = set(raw_data.keys())
                field_coverage = len(required_fields.intersection(available_fields)) / len(required_fields) if required_fields else 1.0
                score += field_coverage * 5

                if score > best_score:
                    best_score = score
                    best_match = rule

        return best_match

    async def _apply_mapping_rule(
        self,
        raw_data: Dict[str, Any],
        rule: MappingRule,
        source: SignalSource,
        context: Optional[Dict[str, Any]] = None
    ) -> Optional[AssetSignal]:
        """Apply a mapping rule to create a signal."""

        # Extract mapped fields
        signal_data = {}
        for raw_field, signal_field in rule.field_mappings.items():
            if raw_field in raw_data:
                signal_data[signal_field] = raw_data[raw_field]

        # Determine asset ID
        asset_id = await self._identify_asset(raw_data, context)

        # Create signal based on type
        signal = await self._create_signal_from_rule(
            rule.signal_type,
            asset_id,
            signal_data,
            raw_data,
            source,
            rule.confidence_boost
        )

        if signal:
            # Apply validation rules
            await self._apply_validation_rules(signal, rule.validation_rules)

            # Set signal source and metadata
            signal.signal_source = source
            signal.source_scraper = context.get('scraper') if context else None
            signal.source_url = context.get('url') if context else None
            signal.processing_pipeline.append('asset_signal_mapper')

        return signal

    async def _identify_asset(self, raw_data: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> str:
        """Identify or create asset identifier from raw data."""

        # Try different identification strategies
        asset_id = None

        # Strategy 1: Direct asset identifier
        for field in ['asset_id', 'property_id', 'parcel_id', 'person_id', 'company_id']:
            if field in raw_data and raw_data[field]:
                asset_id = str(raw_data[field])
                break

        # Strategy 2: Property-based identification
        if not asset_id:
            property_fields = ['address', 'property_address', 'street_address']
            for field in property_fields:
                if field in raw_data and raw_data[field]:
                    # Create hash-based ID from property address
                    address_hash = hashlib.md5(str(raw_data[field]).encode()).hexdigest()[:12]
                    asset_id = f"property_{address_hash}"
                    break

        # Strategy 3: Person-based identification
        if not asset_id:
            name_fields = ['name', 'full_name', 'owner_name', 'person_name']
            for field in name_fields:
                if field in raw_data and raw_data[field]:
                    # Create hash-based ID from person name
                    name_hash = hashlib.md5(str(raw_data[field]).encode()).hexdigest()[:12]
                    asset_id = f"person_{name_hash}"
                    break

        # Strategy 4: Fallback to data hash
        if not asset_id:
            data_hash = hashlib.md5(str(sorted(raw_data.items())).encode()).hexdigest()[:16]
            asset_id = f"unknown_{data_hash}"

        # Check for existing asset identifier and merge if found
        existing_asset = await self._find_existing_asset(asset_id, raw_data)
        if existing_asset:
            asset_id = existing_asset.get_canonical_id()

            # Add additional identifiers if found
            await self._update_asset_identifiers(existing_asset, raw_data)

        return asset_id

    async def _find_existing_asset(self, candidate_id: str, raw_data: Dict[str, Any]) -> Optional[AssetIdentifier]:
        """Find existing asset that matches the candidate."""

        # Direct ID match
        if candidate_id in self.asset_identifiers:
            return self.asset_identifiers[candidate_id]

        # Fuzzy name matching for persons
        name_fields = ['name', 'full_name', 'owner_name', 'person_name']
        for field in name_fields:
            if field in raw_data:
                name = str(raw_data[field])
                for asset_id, asset in self.asset_identifiers.items():
                    if asset.asset_type == AssetType.PERSON and asset.matches_identifier('name', name, fuzzy=True):
                        return asset

        # Address matching for properties
        address_fields = ['address', 'property_address', 'street_address']
        for field in address_fields:
            if field in raw_data:
                address = str(raw_data[field])
                for asset_id, asset in self.asset_identifiers.items():
                    if asset.asset_type in [AssetType.SINGLE_FAMILY_HOME, AssetType.COMMERCIAL_PROPERTY]:
                        if asset.matches_identifier('address', address, fuzzy=True):
                            return asset

        return None

    async def _update_asset_identifiers(self, asset: AssetIdentifier, raw_data: Dict[str, Any]):
        """Update asset with additional identifiers from raw data."""
        # Add various identifier types
        identifier_mappings = {
            'name': ['name', 'full_name', 'owner_name', 'person_name', 'company_name'],
            'address': ['address', 'property_address', 'street_address', 'mailing_address'],
            'phone': ['phone', 'phone_number', 'mobile', 'home_phone'],
            'email': ['email', 'email_address', 'contact_email'],
            'parcel_id': ['parcel_id', 'property_id', 'tax_id'],
            'ssn': ['ssn', 'social_security'],
            'license': ['license_number', 'drivers_license', 'business_license']
        }

        for id_type, fields in identifier_mappings.items():
            for field in fields:
                if field in raw_data and raw_data[field]:
                    asset.add_identifier(id_type, str(raw_data[field]))
                    break

    async def _create_signal_from_rule(
        self,
        signal_type: SignalType,
        asset_id: str,
        signal_data: Dict[str, Any],
        raw_data: Dict[str, Any],
        source: SignalSource,
        confidence_boost: float = 0.0
    ) -> Optional[AssetSignal]:
        """Create a signal using the appropriate factory function."""

        # Determine asset type from signal type and data
        asset_type = self._infer_asset_type(signal_type, raw_data)

        # Set basic signal parameters
        base_params = {
            'asset_id': asset_id,
            'asset_type': asset_type,
            'signal_source': source,
            'confidence_score': min(1.0, 0.7 + confidence_boost),  # Base confidence + boost
        }

        # Add signal-specific data
        signal_params = {**base_params, **signal_data}

        # Route to appropriate factory function
        try:
            if signal_type == SignalType.LIEN:
                return create_lien_signal(**signal_params)
            elif signal_type == SignalType.MORTGAGE:
                return create_mortgage_signal(**signal_params)
            elif signal_type == SignalType.DEED:
                return create_deed_signal(**signal_params)
            elif signal_type == SignalType.FORECLOSURE:
                return create_foreclosure_signal(**signal_params)
            elif signal_type == SignalType.COURT_CASE:
                return create_court_case_signal(**signal_params)
            elif signal_type == SignalType.JUDGMENT:
                return create_judgment_signal(**signal_params)
            elif signal_type == SignalType.TAX_ISSUE:
                return create_tax_issue_signal(**signal_params)
            elif signal_type == SignalType.BIRTHDAY:
                return create_birthday_signal(**signal_params)
            elif signal_type == SignalType.ENGAGEMENT:
                return create_engagement_signal(**signal_params)
            elif signal_type == SignalType.WEDDING:
                return create_wedding_signal(**signal_params)
            elif signal_type == SignalType.IDENTITY:
                return create_identity_signal(asset_id, raw_data, **base_params)
            elif signal_type == SignalType.LOCATION:
                return create_location_signal(asset_id, raw_data, **base_params)
            elif signal_type == SignalType.CONTACT:
                return create_contact_signal(asset_id, raw_data, **base_params)
            elif signal_type == SignalType.PROFESSIONAL:
                return create_professional_signal(asset_id, raw_data, **base_params)
            else:
                logger.warning(f"No factory function for signal type: {signal_type}")
                return None

        except Exception as e:
            logger.error(f"Failed to create signal of type {signal_type}: {e}")
            return None

    def _infer_asset_type(self, signal_type: SignalType, raw_data: Dict[str, Any]) -> AssetType:
        """Infer asset type from signal type and data."""
        # Property-related signals
        property_signals = [SignalType.LIEN, SignalType.MORTGAGE, SignalType.DEED, SignalType.FORECLOSURE, SignalType.TAX_ISSUE]
        if signal_type in property_signals:
            # Check for property indicators
            if any(field in raw_data for field in ['parcel_id', 'property_address', 'square_footage']):
                return AssetType.SINGLE_FAMILY_HOME  # Default property type
            return AssetType.COMMERCIAL_PROPERTY  # Fallback

        # Person-related signals
        person_signals = [SignalType.BIRTHDAY, SignalType.ENGAGEMENT, SignalType.WEDDING, SignalType.IDENTITY]
        if signal_type in person_signals:
            return AssetType.PERSON

        # Legal signals might be person or property
        legal_signals = [SignalType.COURT_CASE, SignalType.JUDGMENT]
        if signal_type in legal_signals:
            if any(field in raw_data for field in ['property_address', 'parcel_id']):
                return AssetType.SINGLE_FAMILY_HOME
            return AssetType.PERSON

        # Default fallback
        return AssetType.ASSET

    async def _enhance_signal(
        self,
        signal: AssetSignal,
        raw_data: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None
    ):
        """Enhance signal with additional intelligence."""
        # Add geographic information
        await self._add_geographic_enrichment(signal, raw_data)

        # Add temporal context
        await self._add_temporal_enrichment(signal, raw_data)

        # Add cross-references
        await self._add_cross_references(signal, raw_data)

        # Validate and clean data
        await self._validate_and_clean_signal(signal)

    async def _add_geographic_enrichment(self, signal: AssetSignal, raw_data: Dict[str, Any]):
        """Add geographic enrichment to signal."""
        # Extract geographic information from raw data
        geo_fields = {
            'city': ['city', 'property_city', 'mailing_city'],
            'state': ['state', 'property_state', 'mailing_state'],
            'zip_code': ['zip', 'zip_code', 'postal_code', 'property_zip'],
            'county': ['county', 'property_county']
        }

        for signal_field, raw_fields in geo_fields.items():
            for raw_field in raw_fields:
                if raw_field in raw_data and raw_data[raw_field]:
                    setattr(signal, f'property_{signal_field}' if signal.is_property_signal() else signal_field,
                           str(raw_data[raw_field]))
                    break

    async def _add_temporal_enrichment(self, signal: AssetSignal, raw_data: Dict[str, Any]):
        """Add temporal enrichment to signal."""
        # Parse dates from various formats
        date_fields = {
            'filing_date': ['filing_date', 'recorded_date', 'filed_date', 'date_filed'],
            'event_date': ['event_date', 'date', 'wedding_date', 'birth_date'],
        }

        for signal_field, raw_fields in date_fields.items():
            for raw_field in raw_fields:
                if raw_field in raw_data and raw_data[raw_field]:
                    try:
                        date_value = self._parse_date(raw_data[raw_field])
                        if date_value:
                            setattr(signal, signal_field, date_value)
                    except Exception as e:
                        logger.debug(f"Failed to parse date {raw_data[raw_field]}: {e}")
                    break

    def _parse_date(self, date_str: Any) -> Optional[datetime]:
        """Parse date from various formats."""
        if isinstance(date_str, datetime):
            return date_str
        if isinstance(date_str, str):
            # Try common date formats
            formats = [
                '%Y-%m-%d',
                '%m/%d/%Y',
                '%d/%m/%Y',
                '%Y/%m/%d',
                '%B %d, %Y',
                '%b %d, %Y'
            ]
            for fmt in formats:
                try:
                    return datetime.strptime(date_str, fmt)
                except ValueError:
                    continue
        return None

    async def _add_cross_references(self, signal: AssetSignal, raw_data: Dict[str, Any]):
        """Add cross-references to related signals."""
        # Find related signals based on asset ID, addresses, names, etc.
        related_signals = []

        # Same asset
        for sig_id, cached_signal in self.signal_cache.items():
            if (cached_signal.asset_id == signal.asset_id and
                cached_signal.signal_id != signal.signal_id):
                related_signals.append(sig_id)

        # Same address (for property signals)
        if signal.is_property_signal() and signal.property_address:
            for sig_id, cached_signal in self.signal_cache.items():
                if (cached_signal.is_property_signal() and
                    cached_signal.property_address == signal.property_address and
                    cached_signal.signal_id != signal.signal_id):
                    related_signals.append(sig_id)

        # Limit to most recent related signals
        signal.related_signals = list(set(related_signals))[:10]

    async def _validate_and_clean_signal(self, signal: AssetSignal):
        """Validate and clean signal data."""
        # Ensure required fields are present
        if not signal.asset_id:
            logger.warning("Signal missing asset_id, generating fallback")
            signal.asset_id = f"unknown_{hash(str(signal.signal_value)) % 10000}"

        # Normalize text fields
        text_fields = ['property_address', 'asset_name', 'court_name', 'attorney_name']
        for field in text_fields:
            value = getattr(signal, field, None)
            if value and isinstance(value, str):
                # Basic cleaning
                cleaned = value.strip().title()
                setattr(signal, field, cleaned)

        # Validate confidence score
        if signal.confidence_score < 0 or signal.confidence_score > 1:
            logger.warning(f"Invalid confidence score {signal.confidence_score}, clamping to [0,1]")
            signal.confidence_score = max(0.0, min(1.0, signal.confidence_score))

    async def _apply_validation_rules(self, signal: AssetSignal, validation_rules: Dict[str, Any]):
        """Apply validation rules to signal."""
        for rule_name, rule_config in validation_rules.items():
            if rule_name == 'required_fields':
                required = rule_config.get('fields', [])
                missing = [field for field in required if getattr(signal, field, None) is None]
                if missing:
                    logger.warning(f"Signal missing required fields: {missing}")
                    # Could set validation status or reduce confidence

            elif rule_name == 'value_range':
                field = rule_config.get('field')
                min_val = rule_config.get('min')
                max_val = rule_config.get('max')
                if field and min_val is not None and max_val is not None:
                    value = getattr(signal, field, None)
                    if value is not None and not (min_val <= value <= max_val):
                        logger.warning(f"Field {field} value {value} outside range [{min_val}, {max_val}]")

    async def deduplicate_signals(self, signals: List[AssetSignal]) -> List[AssetSignal]:
        """
        Deduplicate signals based on content similarity and asset relationships.

        Args:
            signals: List of signals to deduplicate

        Returns:
            List of unique signals with duplicates merged
        """
        if not signals:
            return []

        unique_signals = []
        processed_ids = set()

        for signal in signals:
            if signal.signal_id in processed_ids:
                continue

            # Find duplicates
            duplicates = []
            for other in signals:
                if (other.signal_id != signal.signal_id and
                    other.signal_id not in processed_ids and
                    self._are_signals_duplicates(signal, other)):
                    duplicates.append(other)

            if duplicates:
                # Merge duplicates into primary signal
                for duplicate in duplicates:
                    signal.merge_signal(duplicate, merge_strategy="highest_confidence")
                    processed_ids.add(duplicate.signal_id)

                    # Track duplication
                    self.duplicate_map[signal.signal_id].append(duplicate.signal_id)

            unique_signals.append(signal)
            processed_ids.add(signal.signal_id)

        logger.info(f"Deduplicated {len(signals)} signals into {len(unique_signals)} unique signals")
        return unique_signals

    def get_optimal_sources_for_signal(
        self,
        asset_type: AssetType,
        signal_type: SignalType,
        current_sources: Optional[List[str]] = None
    ) -> List[str]:
        """
        Get optimal data sources for a specific asset type and signal type.

        Uses configuration intelligence to recommend best sources based on:
        - Known reliable sources for the asset/signal combination
        - Source performance history
        - Cost efficiency
        - Data freshness requirements

        Args:
            asset_type: Type of asset
            signal_type: Type of signal
            current_sources: Currently available sources to consider

        Returns:
            Ordered list of optimal source names
        """
        # Get configured sources for this asset/signal combination
        configured_sources = ASSET_SIGNAL_SOURCES.get(asset_type, {}).get(signal_type, [])

        if not configured_sources:
            # Fallback to general sources based on signal type
            configured_sources = self._get_fallback_sources(signal_type)

        # Score and rank sources
        scored_sources = []
        for source in configured_sources:
            score = self._calculate_source_score(source, signal_type, current_sources)
            scored_sources.append((source, score))

        # Sort by score (highest first)
        scored_sources.sort(key=lambda x: x[1], reverse=True)

        optimal_sources = [source for source, score in scored_sources]

        logger.debug(f"Optimal sources for {asset_type.value}/{signal_type.value}: {optimal_sources}")
        return optimal_sources

    def _calculate_source_score(
        self,
        source: str,
        signal_type: SignalType,
        current_sources: Optional[List[str]] = None
    ) -> float:
        """Calculate overall score for a data source."""
        base_score = 0.0

        # Base reliability score
        reliability = SOURCE_RELIABILITY.get(source, 0.5)
        base_score += reliability * 0.4  # 40% weight on reliability

        # Performance history score
        performance = self.source_performance.get(source, {"success": 0, "failure": 0})
        total_attempts = performance["success"] + performance["failure"]
        if total_attempts > 0:
            success_rate = performance["success"] / total_attempts
            avg_confidence = performance["avg_confidence"]
            performance_score = (success_rate * 0.6) + (avg_confidence * 0.4)
            base_score += performance_score * 0.3  # 30% weight on performance

        # Cost efficiency score (inverse of cost weight)
        cost_weight = SIGNAL_COST_WEIGHT.get(signal_type, 2.5)
        cost_efficiency = 1.0 / cost_weight  # Lower cost = higher efficiency
        base_score += cost_efficiency * 0.3  # 30% weight on cost efficiency

        # Diversity bonus (prefer different sources if current_sources provided)
        if current_sources and source not in current_sources:
            base_score += 0.1  # Small bonus for source diversity

        return min(1.0, base_score)  # Cap at 1.0

    def _get_fallback_sources(self, signal_type: SignalType) -> List[str]:
        """Get fallback sources when specific asset/signal combination not configured."""
        fallbacks = {
            # Personal signals
            SignalType.BIRTHDAY: ["local_newspapers", "social_media", "public_records"],
            SignalType.ENGAGEMENT: ["newspapers", "social_media", "event_sites"],
            SignalType.WEDDING: ["newspapers", "marriage_records", "social_media"],

            # Property signals
            SignalType.LIEN: ["county_clerk", "tax_assessor", "court_records"],
            SignalType.MORTGAGE: ["county_recorder", "title_company"],
            SignalType.DEED: ["county_recorder", "property_records"],
            SignalType.FORECLOSURE: ["court_records", "state_registry"],
            SignalType.TAX_ISSUE: ["county_assessor", "tax_records"],

            # Legal signals
            SignalType.COURT_CASE: ["county_court", "court_records"],
            SignalType.JUDGMENT: ["court_records", "county_clerk"],

            # General signals
            SignalType.IDENTITY: ["public_records", "dmv_records"],
            SignalType.CONTACT: ["phone_directories", "public_records"],
            SignalType.PROFESSIONAL: ["company_websites", "linkedin"],
            SignalType.FINANCIAL: ["public_records", "court_records"],
            SignalType.LEGAL: ["state_records", "court_records"],
            SignalType.BEHAVIORAL: ["social_media", "web_tracking"],
            SignalType.RELATIONSHIP: ["social_media", "public_records"],
            SignalType.EVENT: ["news_sources", "event_sites"],
            SignalType.ANOMALY: ["monitoring_systems", "analysis_tools"]
        }

        return fallbacks.get(signal_type, ["web_scraping", "public_records"])

    def calculate_signal_cost_estimate(
        self,
        asset_type: AssetType,
        signal_type: SignalType,
        sources: Optional[List[str]] = None,
        data_quality: str = "standard"
    ) -> Dict[str, Any]:
        """
        Calculate comprehensive cost estimate for acquiring a signal.

        Considers:
        - Base signal cost weight
        - Source-specific costs
        - Data quality requirements
        - Asset type complexity
        - Historical performance data

        Args:
            asset_type: Type of asset
            signal_type: Type of signal
            sources: Specific sources to use (optional)
            data_quality: Required data quality level

        Returns:
            Dict with cost breakdown and estimates
        """
        base_cost = SIGNAL_COST_WEIGHT.get(signal_type, 2.5)

        # Adjust for asset type complexity
        asset_complexity = self._get_asset_complexity_multiplier(asset_type)
        adjusted_base_cost = base_cost * asset_complexity

        # Source-specific costs
        if sources:
            source_costs = [self._calculate_source_cost(source, signal_type) for source in sources]
            avg_source_cost = sum(source_costs) / len(source_costs) if source_costs else adjusted_base_cost
        else:
            # Use optimal sources
            optimal_sources = self.get_optimal_sources_for_signal(asset_type, signal_type)
            if optimal_sources:
                source_costs = [self._calculate_source_cost(source, signal_type) for source in optimal_sources[:2]]
                avg_source_cost = sum(source_costs) / len(source_costs) if source_costs else adjusted_base_cost
            else:
                avg_source_cost = adjusted_base_cost

        # Quality adjustment
        quality_multiplier = self._get_quality_multiplier(data_quality)

        # Historical adjustment based on past performance
        historical_adjustment = self._get_historical_cost_adjustment(signal_type)

        final_cost = (adjusted_base_cost + avg_source_cost) * quality_multiplier * historical_adjustment

        # Cost breakdown
        breakdown = {
            "base_signal_cost": base_cost,
            "asset_complexity_multiplier": asset_complexity,
            "adjusted_base_cost": adjusted_base_cost,
            "average_source_cost": avg_source_cost,
            "quality_multiplier": quality_multiplier,
            "historical_adjustment": historical_adjustment,
            "total_estimated_cost": round(final_cost, 2)
        }

        # Confidence in estimate
        confidence_sources = ["cost_weights", "asset_complexity"]
        if sources:
            confidence_sources.append("specific_sources")
        if self.signal_cost_tracking[signal_type] > 0:
            confidence_sources.append("historical_data")

        breakdown["estimate_confidence"] = len(confidence_sources) / 5.0  # Max 5 factors
        breakdown["confidence_sources"] = confidence_sources

        return breakdown

    def _get_asset_complexity_multiplier(self, asset_type: AssetType) -> float:
        """Get complexity multiplier based on asset type."""
        complexity_map = {
            AssetType.PERSON: 1.0,           # Baseline
            AssetType.SINGLE_FAMILY_HOME: 1.2,  # Property records add complexity
            AssetType.MULTI_FAMILY_SMALL: 1.4,  # More complex ownership
            AssetType.APARTMENT_BUILDING: 1.6,  # Commercial complexity
            AssetType.COMMERCIAL_PROPERTY: 1.8, # Highest complexity
            AssetType.COMPANY: 2.0,          # Business entity complexity
            AssetType.ORGANIZATION: 1.7,      # Organizational complexity
            AssetType.LOCATION: 1.1,          # Geographic complexity
            AssetType.EVENT: 1.3,             # Event-based complexity
            AssetType.ASSET: 1.5              # General asset complexity
        }
        return complexity_map.get(asset_type, 1.0)

    def _calculate_source_cost(self, source: str, signal_type: SignalType) -> float:
        """Calculate cost for a specific source."""
        base_source_cost = 1.0  # Default cost

        # Reliability affects cost (more reliable sources may be more expensive)
        reliability = SOURCE_RELIABILITY.get(source, 0.5)
        reliability_cost = reliability * 0.5  # Premium for high reliability

        # Source type cost variations
        if "federal" in source:
            base_source_cost = 2.5  # Federal sources more expensive
        elif "court" in source or "clerk" in source:
            base_source_cost = 1.8  # Court records moderately expensive
        elif "social_media" in source or "web" in source:
            base_source_cost = 0.8  # Web sources cheaper
        elif "newspaper" in source:
            base_source_cost = 1.2  # News sources moderate cost

        return base_source_cost + reliability_cost

    def _get_quality_multiplier(self, data_quality: str) -> float:
        """Get cost multiplier based on required data quality."""
        quality_multipliers = {
            "basic": 0.8,      # Minimal verification
            "standard": 1.0,   # Normal quality requirements
            "verified": 1.3,   # Additional verification required
            "premium": 1.6,    # High-quality, multi-source verification
            "enterprise": 2.0  # Maximum quality, legal-grade verification
        }
        return quality_multipliers.get(data_quality.lower(), 1.0)

    def _get_historical_cost_adjustment(self, signal_type: SignalType) -> float:
        """Get cost adjustment based on historical performance."""
        historical_cost = self.signal_cost_tracking.get(signal_type, 0)
        if historical_cost > 0:
            # Adjust towards historical average (with smoothing)
            return 0.7 + (historical_cost * 0.3)
        return 1.0  # No historical data

    def validate_source_for_signal(
        self,
        source: str,
        asset_type: AssetType,
        signal_type: SignalType
    ) -> Dict[str, Any]:
        """
        Validate if a source is appropriate and optimal for a signal.

        Returns validation results with recommendations.

        Args:
            source: Source name to validate
            asset_type: Asset type
            signal_type: Signal type

        Returns:
            Dict with validation results and recommendations
        """
        validation = {
            "source": source,
            "asset_type": asset_type.value,
            "signal_type": signal_type.value,
            "is_configured": False,
            "is_optimal": False,
            "reliability_score": 0.0,
            "performance_score": 0.0,
            "cost_efficiency": 0.0,
            "recommendations": [],
            "alternatives": []
        }

        # Check if source is configured for this asset/signal combination
        configured_sources = ASSET_SIGNAL_SOURCES.get(asset_type, {}).get(signal_type, [])
        validation["is_configured"] = source in configured_sources

        # Get optimal sources for comparison
        optimal_sources = self.get_optimal_sources_for_signal(asset_type, signal_type)

        # Calculate scores
        validation["reliability_score"] = SOURCE_RELIABILITY.get(source, 0.0)
        performance = self.source_performance.get(source, {"success": 0, "failure": 0})
        total_attempts = performance["success"] + performance["failure"]
        if total_attempts > 0:
            validation["performance_score"] = performance["success"] / total_attempts

        cost_weight = SIGNAL_COST_WEIGHT.get(signal_type, 2.5)
        validation["cost_efficiency"] = 1.0 / cost_weight

        # Overall assessment
        validation["is_optimal"] = source in optimal_sources[:3]  # Top 3 optimal sources

        # Generate recommendations
        if not validation["is_configured"]:
            validation["recommendations"].append(
                f"Source '{source}' is not in the configured sources for {asset_type.value}/{signal_type.value}"
            )

        if not validation["is_optimal"]:
            validation["recommendations"].append(
                f"Consider using optimal sources: {', '.join(optimal_sources[:3])}"
            )

        if validation["reliability_score"] < 0.7:
            validation["recommendations"].append(
                f"Low reliability source ({validation['reliability_score']:.2f}). Consider more authoritative sources."
            )

        if validation["performance_score"] < 0.8 and total_attempts > 5:
            validation["recommendations"].append(
                f"Poor historical performance ({validation['performance_score']:.1%} success rate)"
            )

        # Suggest alternatives
        if not validation["is_optimal"]:
            validation["alternatives"] = optimal_sources[:3]

        return validation

    def get_data_freshness_requirement(self, signal_type: SignalType) -> int:
        """
        Get the recommended data freshness requirement in days.

        Args:
            signal_type: Type of signal

        Returns:
            Days within which data should be considered fresh
        """
        return SIGNAL_FRESHNESS_REQUIREMENTS.get(signal_type, 365)  # Default 1 year

    def is_signal_data_fresh(self, signal: AssetSignal) -> Dict[str, Any]:
        """
        Check if signal data is fresh based on its timestamp and type.

        Args:
            signal: Signal to check

        Returns:
            Dict with freshness assessment
        """
        if not signal.created_at:
            return {
                "is_fresh": False,
                "days_old": None,
                "max_age_days": self.get_data_freshness_requirement(signal.signal_type),
                "needs_refresh": True,
                "reason": "No timestamp available"
            }

        days_old = (datetime.utcnow() - signal.created_at).days
        max_age = self.get_data_freshness_requirement(signal.signal_type)

        assessment = {
            "is_fresh": days_old <= max_age,
            "days_old": days_old,
            "max_age_days": max_age,
            "needs_refresh": days_old > max_age,
            "freshness_score": max(0, 1.0 - (days_old / max_age)) if days_old <= max_age else 0.0
        }

        if assessment["needs_refresh"]:
            assessment["reason"] = f"Data is {days_old} days old, exceeds {max_age} day limit"
        else:
            assessment["reason"] = f"Data is {days_old} days old, within {max_age} day limit"

        return assessment

    def _are_signals_duplicates(self, signal1: AssetSignal, signal2: AssetSignal) -> bool:
        """Check if two signals are duplicates."""
        # Must be same type and asset
        if signal1.signal_type != signal2.signal_type or signal1.asset_id != signal2.asset_id:
            return False

        # Check content similarity
        similarity_score = self._calculate_signal_similarity(signal1, signal2)

        # Thresholds based on signal type
        thresholds = {
            SignalType.IDENTITY: 95,      # Very strict for identity
            SignalType.LIEN: 90,          # Strict for legal documents
            SignalType.MORTGAGE: 85,      # Moderate for financial
            SignalType.DEED: 95,          # Very strict for ownership
            SignalType.FORECLOSURE: 90,   # Strict for legal
            SignalType.BIRTHDAY: 80,      # More lenient for personal
            SignalType.WEDDING: 85,       # Moderate for events
        }

        threshold = thresholds.get(signal1.signal_type, 85)
        return similarity_score >= threshold

    def _calculate_signal_similarity(self, signal1: AssetSignal, signal2: AssetSignal) -> float:
        """Calculate similarity score between two signals."""
        similarity_scores = []

        # Address similarity (for property signals)
        if signal1.is_property_signal():
            addr1 = signal1.get_property_address() or ""
            addr2 = signal2.get_property_address() or ""
            if addr1 and addr2:
                similarity_scores.append(fuzz_ratio(addr1, addr2))

        # Name similarity
        name1 = signal1.asset_name or ""
        name2 = signal2.asset_name or ""
        if name1 and name2:
            similarity_scores.append(fuzz_ratio(name1, name2))

        # Date similarity
        dates1 = [signal1.filing_date, signal1.event_date]
        dates2 = [signal2.filing_date, signal2.event_date]
        date_matches = 0
        for d1 in dates1:
            if d1:
                for d2 in dates2:
                    if d2 and abs((d1 - d2).days) <= 1:  # Within 1 day
                        date_matches += 1
                        break
        if dates1 or dates2:
            date_similarity = (date_matches / max(len([d for d in dates1 if d]), len([d for d in dates2 if d]), 1)) * 100
            similarity_scores.append(date_similarity)

        # Amount similarity (for financial signals)
        amt1 = signal1.get_financial_impact()
        amt2 = signal2.get_financial_impact()
        if amt1 and amt2 and amt1 > 0:
            amount_diff = abs(amt1 - amt2) / max(amt1, amt2)
            amount_similarity = (1 - amount_diff) * 100
            similarity_scores.append(amount_similarity)

        return sum(similarity_scores) / len(similarity_scores) if similarity_scores else 0.0

    async def batch_map_signals(
        self,
        raw_data_batch: List[Dict[str, Any]],
        source: SignalSource,
        context: Optional[Dict[str, Any]] = None,
        max_concurrent: int = 10
    ) -> List[AssetSignal]:
        """
        Batch map multiple raw data items to signals concurrently.

        Args:
            raw_data_batch: List of raw data items
            source: Source of all data items
            context: Context shared across all mappings
            max_concurrent: Maximum concurrent mappings

        Returns:
            List of mapped signals
        """
        semaphore = asyncio.Semaphore(max_concurrent)

        async def map_with_semaphore(data):
            async with semaphore:
                return await self.map_raw_data_to_signal(data, source, context)

        # Map concurrently
        tasks = [map_with_semaphore(data) for data in raw_data_batch]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Filter out None results and exceptions
        signals = []
        for result in results:
            if isinstance(result, AssetSignal):
                signals.append(result)
            elif isinstance(result, Exception):
                logger.error(f"Batch mapping error: {result}")

        # Deduplicate the results
        unique_signals = await self.deduplicate_signals(signals)

        logger.info(f"✅ Batch mapped {len(raw_data_batch)} raw items to {len(unique_signals)} unique signals")
        return unique_signals

    def get_mapping_stats(self) -> Dict[str, Any]:
        """Get comprehensive mapping statistics."""
        # Calculate source performance summary
        source_stats = {}
        for source, perf in self.source_performance.items():
            total = perf["success"] + perf["failure"]
            source_stats[source] = {
                "total_attempts": total,
                "success_rate": perf["success"] / max(total, 1),
                "avg_confidence": perf["avg_confidence"],
                "reliability_score": SOURCE_RELIABILITY.get(source, 0.0)
            }

        # Signal cost tracking summary
        cost_stats = {}
        for signal_type, cost in self.signal_cost_tracking.items():
            cost_stats[signal_type.value] = cost

        return {
            "mapping_rules_count": len(self.mapping_rules),
            "asset_identifiers_count": len(self.asset_identifiers),
            "cached_signals_count": len(self.signal_cache),
            "duplicate_groups_count": len(self.duplicate_map),
            "mapping_stats": dict(self.mapping_stats),
            "cache_performance": {
                "hits": self.cache_hits,
                "misses": self.cache_misses,
                "hit_rate": self.cache_hits / max(self.cache_hits + self.cache_misses, 1)
            },
            "source_performance": source_stats,
            "signal_cost_tracking": cost_stats,
            "configuration": {
                "asset_signal_sources_count": len(ASSET_SIGNAL_SOURCES),
                "signal_cost_weights_count": len(SIGNAL_COST_WEIGHT),
                "source_reliability_scores_count": len(SOURCE_RELIABILITY),
                "signal_freshness_requirements_count": len(SIGNAL_FRESHNESS_REQUIREMENTS)
            }
        }

    def clear_cache(self):
        """Clear the signal cache."""
        cache_size = len(self.signal_cache)
        self.signal_cache.clear()
        self.cache_hits = 0
        self.cache_misses = 0
        logger.info(f"Cleared signal cache ({cache_size} entries)")

    def export_mapping_rules(self) -> List[Dict[str, Any]]:
        """Export mapping rules for backup or sharing."""
        return [
            {
                "source_pattern": rule.source_pattern,
                "signal_type": rule.signal_type.value,
                "field_mappings": rule.field_mappings,
                "validation_rules": rule.validation_rules,
                "confidence_boost": rule.confidence_boost,
                "priority": rule.priority,
                "description": rule.description
            }
            for rule in self.mapping_rules
        ]

    def import_mapping_rules(self, rules_data: List[Dict[str, Any]]):
        """Import mapping rules from exported data."""
        for rule_data in rules_data:
            rule = MappingRule(
                source_pattern=rule_data["source_pattern"],
                signal_type=SignalType(rule_data["signal_type"]),
                field_mappings=rule_data.get("field_mappings", {}),
                validation_rules=rule_data.get("validation_rules", {}),
                confidence_boost=rule_data.get("confidence_boost", 0.0),
                priority=rule_data.get("priority", 1),
                description=rule_data.get("description", "")
            )
            self.add_mapping_rule(rule)


# Global mapper instance
_global_mapper = AssetSignalMapper()


# Convenience functions
async def map_to_signal(
    raw_data: Dict[str, Any],
    source: SignalSource,
    context: Optional[Dict[str, Any]] = None
) -> Optional[AssetSignal]:
    """
    Map raw data to a standardized asset signal.

    This is the main entry point for signal mapping in the MJ Data Scraper Suite.
    """
    return await _global_mapper.map_raw_data_to_signal(raw_data, source, context)


async def batch_map_to_signals(
    raw_data_batch: List[Dict[str, Any]],
    source: SignalSource,
    context: Optional[Dict[str, Any]] = None,
    max_concurrent: int = 10
) -> List[AssetSignal]:
    """
    Batch map multiple raw data items to signals.

    Efficiently processes large volumes of scraped data with concurrent mapping
    and automatic deduplication.
    """
    return await _global_mapper.batch_map_signals(raw_data_batch, source, context, max_concurrent)


async def deduplicate_signal_batch(signals: List[AssetSignal]) -> List[AssetSignal]:
    """
    Deduplicate a batch of signals.

    Removes duplicates based on content similarity and merges related signals
    for data quality and consistency.
    """
    return await _global_mapper.deduplicate_signals(signals)


def get_mapping_statistics() -> Dict[str, Any]:
    """
    Get comprehensive mapping statistics.

    Returns operational metrics for monitoring mapping performance and health.
    """
    return _global_mapper.get_mapping_stats()


def clear_signal_cache():
    """
    Clear the signal mapping cache.

    Useful for memory management or when data patterns change significantly.
    """
    _global_mapper.clear_cache()


def export_mapping_configuration() -> List[Dict[str, Any]]:
    """
    Export mapping rules for backup or distribution.

    Returns the current mapping rule configuration that can be imported
    into other instances or saved for backup.
    """
    return _global_mapper.export_mapping_rules()


def import_mapping_configuration(rules_data: List[Dict[str, Any]]):
    """
    Import mapping rules from configuration data.

    Allows loading pre-configured mapping rules for different data sources
    or use cases.
    """
    _global_mapper.import_mapping_rules(rules_data)


# Enhanced Intelligence Functions

def get_optimal_sources_for_signal(
    asset_type: AssetType,
    signal_type: SignalType,
    current_sources: Optional[List[str]] = None
) -> List[str]:
    """
    Get optimal data sources for acquiring a specific signal type.

    Uses intelligent source selection based on reliability, performance,
    and cost efficiency for the asset/signal combination.

    Args:
        asset_type: Type of asset
        signal_type: Type of signal to acquire
        current_sources: Currently available sources (for diversity)

    Returns:
        Ordered list of optimal source names
    """
    return _global_mapper.get_optimal_sources_for_signal(asset_type, signal_type, current_sources)


def calculate_signal_cost_estimate(
    asset_type: AssetType,
    signal_type: SignalType,
    sources: Optional[List[str]] = None,
    data_quality: str = "standard"
) -> Dict[str, Any]:
    """
    Calculate comprehensive cost estimate for signal acquisition.

    Provides detailed cost breakdown including base costs, source costs,
    quality adjustments, and historical performance factors.

    Args:
        asset_type: Type of asset
        signal_type: Type of signal
        sources: Specific sources to use (optional)
        data_quality: Required quality level ("basic", "standard", "verified", "premium", "enterprise")

    Returns:
        Dict with detailed cost breakdown and confidence metrics
    """
    return _global_mapper.calculate_signal_cost_estimate(asset_type, signal_type, sources, data_quality)


def validate_source_for_signal(
    source: str,
    asset_type: AssetType,
    signal_type: SignalType
) -> Dict[str, Any]:
    """
    Validate if a data source is appropriate for a signal type.

    Provides comprehensive assessment including reliability, performance,
    configuration status, and optimization recommendations.

    Args:
        source: Source name to validate
        asset_type: Asset type
        signal_type: Signal type

    Returns:
        Dict with validation results, scores, and recommendations
    """
    return _global_mapper.validate_source_for_signal(source, asset_type, signal_type)


def get_data_freshness_requirement(signal_type: SignalType) -> int:
    """
    Get the recommended data freshness requirement in days.

    Different signal types have different freshness requirements based on
    how quickly the data becomes stale.

    Args:
        signal_type: Type of signal

    Returns:
        Maximum age in days for data to be considered fresh
    """
    return _global_mapper.get_data_freshness_requirement(signal_type)


def is_signal_data_fresh(signal: AssetSignal) -> Dict[str, Any]:
    """
    Assess if signal data is fresh and current.

    Evaluates signal age against freshness requirements and provides
    refresh recommendations.

    Args:
        signal: AssetSignal to assess

    Returns:
        Dict with freshness assessment and recommendations
    """
    return _global_mapper.is_signal_data_fresh(signal)


def get_asset_signal_source_configuration() -> Dict[str, Any]:
    """
    Get the complete asset signal source configuration.

    Returns the full configuration mapping assets to signals to sources,
    along with cost weights and reliability scores.

    Returns:
        Dict containing complete source intelligence configuration
    """
    return {
        "asset_signal_sources": ASSET_SIGNAL_SOURCES,
        "signal_cost_weights": SIGNAL_COST_WEIGHT,
        "source_reliability": SOURCE_RELIABILITY,
        "signal_freshness_requirements": SIGNAL_FRESHNESS_REQUIREMENTS,
        "configuration_stats": {
            "total_asset_types": len(ASSET_SIGNAL_SOURCES),
            "total_signal_types": len(set(
                signal_type for asset_sources in ASSET_SIGNAL_SOURCES.values()
                for signal_type in asset_sources.keys()
            )),
            "total_sources": len(set(
                source for asset_sources in ASSET_SIGNAL_SOURCES.values()
                for signal_sources in asset_sources.values()
                for source in signal_sources
            )),
            "cost_weighted_signals": len(SIGNAL_COST_WEIGHT),
            "reliability_scored_sources": len(SOURCE_RELIABILITY),
            "freshness_tracked_signals": len(SIGNAL_FRESHNESS_REQUIREMENTS)
        }
    }


# Configuration Access Functions

def get_signal_cost_weight(signal_type: SignalType) -> float:
    """
    Get the cost weight for a signal type.

    Args:
        signal_type: Type of signal

    Returns:
        Cost weight (higher = more expensive)
    """
    return SIGNAL_COST_WEIGHT.get(signal_type, 2.5)


def get_source_reliability_score(source: str) -> float:
    """
    Get the reliability score for a data source.

    Args:
        source: Source name

    Returns:
        Reliability score (0.0 to 1.0, higher = more reliable)
    """
    return SOURCE_RELIABILITY.get(source, 0.5)


def get_sources_for_asset_signal(asset_type: AssetType, signal_type: SignalType) -> List[str]:
    """
    Get configured sources for a specific asset/signal combination.

    Args:
        asset_type: Type of asset
        signal_type: Type of signal

    Returns:
        List of configured source names
    """
    return ASSET_SIGNAL_SOURCES.get(asset_type, {}).get(signal_type, [])
