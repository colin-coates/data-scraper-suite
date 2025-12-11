# Copyright (c) 2024 Mountain Jewels Intelligence. All rights reserved.
#
# This software is proprietary and confidential. Unauthorized copying,
# modification, distribution, or use is strictly prohibited.

"""
Public Records Scraper Plugin for MJ Data Scraper Suite

Scrapes public records from government databases, court records, property records,
business registrations, and other public data sources with proper legal compliance.
"""

import asyncio
import logging
import re
import json
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from urllib.parse import urljoin, urlparse, urlencode

from core.base_scraper import BaseScraper, ScraperConfig

logger = logging.getLogger(__name__)

__version__ = "1.0.0"
__description__ = "Public records scraper with legal compliance and data validation"
__author__ = "MJ Intelligence"
__dependencies__ = ["requests", "beautifulsoup4", "lxml", "usaddress"]


class PublicRecordsConfig(ScraperConfig):
    """Configuration specific to public records scraping."""
    record_types: List[str] = ['property', 'business', 'court', 'voter', 'marriage', 'death']
    jurisdictions: List[str] = []  # Specific states/counties to target
    max_records_per_type: int = 100
    lookback_years: int = 5
    extract_property_records: bool = True
    extract_business_licenses: bool = True
    extract_court_records: bool = False  # Requires careful legal review
    extract_voter_records: bool = False  # Privacy sensitive
    extract_vital_records: bool = True   # Birth, marriage, death records
    respect_rate_limits: bool = True
    validate_data: bool = True
    anonymize_sensitive_data: bool = True
    compliance_logging: bool = True


class PublicRecordsScraper(BaseScraper):
    """
    Public records scraper with legal compliance and data validation.
    Scrapes government databases while respecting privacy laws and rate limits.
    """

    def __init__(self, config: PublicRecordsConfig):
        super().__init__(config)
        self.public_config = config

        # Public records state
        self.record_sources = self._load_record_sources()
        self.jurisdiction_data: Dict[str, Dict[str, Any]] = {}
        self.compliance_log: List[Dict[str, Any]] = []

        # Data validation patterns
        self.address_parser = None
        self.validation_patterns = self._load_validation_patterns()

        # Rate limiting per source
        self.source_rate_limits: Dict[str, Dict[str, Any]] = {}

        self.records_scraped = 0

    def _load_record_sources(self) -> Dict[str, Dict[str, Any]]:
        """Load configuration for public record sources."""
        return {
            "property_records": {
                "name": "County Property Records",
                "description": "Real estate ownership and property tax records",
                "rate_limit": 10,  # requests per minute
                "requires_jurisdiction": True,
                "data_types": ["ownership", "assessments", "transfers"]
            },
            "business_registrations": {
                "name": "Secretary of State Business Records",
                "description": "Business entity registrations and filings",
                "rate_limit": 20,
                "requires_jurisdiction": True,
                "data_types": ["corporations", "llcs", "partnerships", "trademarks"]
            },
            "court_records": {
                "name": "County Court Records",
                "description": "Civil and criminal court case records",
                "rate_limit": 5,  # Very conservative for legal data
                "requires_jurisdiction": True,
                "data_types": ["civil_cases", "criminal_cases", "divorces"]
            },
            "vital_records": {
                "name": "Vital Statistics Records",
                "description": "Birth, marriage, and death records",
                "rate_limit": 10,
                "requires_jurisdiction": True,
                "data_types": ["births", "marriages", "deaths", "divorces"]
            },
            "professional_licenses": {
                "name": "Professional Licensing Boards",
                "description": "Professional certifications and licenses",
                "rate_limit": 15,
                "requires_jurisdiction": True,
                "data_types": ["medical", "legal", "real_estate", "financial"]
            },
            "campaign_finance": {
                "name": "Campaign Finance Records",
                "description": "Political campaign contributions and expenditures",
                "rate_limit": 25,
                "requires_jurisdiction": True,
                "data_types": ["contributions", "expenditures", "filings"]
            }
        }

    def _load_validation_patterns(self) -> Dict[str, Any]:
        """Load data validation patterns."""
        return {
            "ssn_pattern": re.compile(r'\b\d{3}-?\d{2}-?\d{4}\b'),
            "phone_pattern": re.compile(r'\b(?:\+?1[-.\s]?)?\(?([0-9]{3})\)?[-.\s]?([0-9]{3})[-.\s]?([0-9]{4})\b'),
            "email_pattern": re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'),
            "zip_pattern": re.compile(r'\b\d{5}(?:-\d{4})?\b'),
            "date_patterns": [
                re.compile(r'\b\d{1,2}/\d{1,2}/\d{4}\b'),
                re.compile(r'\b\d{4}-\d{2}-\d{2}\b'),
                re.compile(r'\b\d{1,2} \w{3} \d{4}\b')
            ]
        }

    async def _execute_scrape(self, target: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute public records scraping with legal compliance.

        Args:
            target: Contains 'record_types', 'jurisdictions', 'search_terms', etc.

        Returns:
            Structured public records data
        """
        record_types = target.get('record_types', self.public_config.record_types)
        jurisdictions = target.get('jurisdictions', self.public_config.jurisdictions)
        search_terms = target.get('search_terms', {})

        if not jurisdictions and self.public_config.jurisdictions:
            jurisdictions = self.public_config.jurisdictions

        if not jurisdictions:
            raise ValueError("Public records scraper requires 'jurisdictions' to specify geographic scope")

        logger.info(f"Scraping public records for {len(record_types)} types across {len(jurisdictions)} jurisdictions")

        # Log compliance information
        await self._log_compliance_event({
            "action": "scrape_start",
            "record_types": record_types,
            "jurisdictions": jurisdictions,
            "search_terms": search_terms,
            "timestamp": datetime.utcnow().isoformat()
        })

        # Scrape records by type and jurisdiction
        all_records = []
        source_stats = {}

        for record_type in record_types:
            for jurisdiction in jurisdictions:
                try:
                    records = await self._scrape_records_by_type(record_type, jurisdiction, search_terms)
                    all_records.extend(records)

                    source_key = f"{record_type}_{jurisdiction}"
                    source_stats[source_key] = {
                        "records_found": len(records),
                        "success": True,
                        "last_updated": datetime.utcnow()
                    }

                except Exception as e:
                    logger.error(f"Failed to scrape {record_type} records for {jurisdiction}: {e}")
                    source_key = f"{record_type}_{jurisdiction}"
                    source_stats[source_key] = {
                        "error": str(e),
                        "success": False,
                        "last_updated": datetime.utcnow()
                    }

        # Validate and clean data
        if self.public_config.validate_data:
            all_records = await self._validate_records(all_records)

        # Anonymize sensitive data if configured
        if self.public_config.anonymize_sensitive_data:
            all_records = self._anonymize_sensitive_data(all_records)

        # Structure results
        result = {
            "records": all_records,
            "total_records": len(all_records),
            "record_types_processed": record_types,
            "jurisdictions_processed": jurisdictions,
            "source_stats": source_stats,
            "scraped_at": datetime.utcnow().isoformat(),
            "compliance_info": {
                "data_anonymized": self.public_config.anonymize_sensitive_data,
                "data_validated": self.public_config.validate_data,
                "rate_limits_respected": self.public_config.respect_rate_limits
            }
        }

        self.records_scraped += len(all_records)

        # Log completion
        await self._log_compliance_event({
            "action": "scrape_complete",
            "total_records": len(all_records),
            "timestamp": datetime.utcnow().isoformat()
        })

        return result

    async def _scrape_records_by_type(self, record_type: str, jurisdiction: str,
                                    search_terms: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Scrape records of a specific type for a jurisdiction."""
        if record_type not in self.record_sources:
            raise ValueError(f"Unknown record type: {record_type}")

        source_config = self.record_sources[record_type]

        # Check rate limits
        if self.public_config.respect_rate_limits:
            await self._check_source_rate_limit(record_type, jurisdiction)

        # Get jurisdiction-specific endpoints
        endpoints = await self._get_jurisdiction_endpoints(record_type, jurisdiction)

        records = []

        for endpoint in endpoints:
            try:
                batch_records = await self._scrape_endpoint(endpoint, search_terms)
                records.extend(batch_records)

                # Limit records per type/jurisdiction
                if len(records) >= self.public_config.max_records_per_type:
                    break

            except Exception as e:
                logger.warning(f"Failed to scrape endpoint {endpoint}: {e}")
                continue

        # Add metadata
        for record in records:
            record.update({
                "record_type": record_type,
                "jurisdiction": jurisdiction,
                "source": source_config["name"],
                "scraped_at": datetime.utcnow().isoformat(),
                "data_classification": self._classify_record_sensitivity(record)
            })

        return records

    async def _get_jurisdiction_endpoints(self, record_type: str, jurisdiction: str) -> List[Dict[str, Any]]:
        """Get endpoints for a specific record type and jurisdiction."""
        # This would contain mappings of jurisdictions to their data sources
        # In production, this would be a comprehensive database

        endpoints = {
            "property_records": {
                "california": [
                    {"url": f"https://www.county-clerk.ca.gov/{jurisdiction}/property-search", "method": "api"}
                ],
                "new_york": [
                    {"url": f"https://www.nycourts.gov/{jurisdiction}/property-records", "method": "web"}
                ]
            },
            "business_registrations": {
                "california": [
                    {"url": "https://www.sos.ca.gov/corps/search", "method": "web"}
                ],
                "delaware": [
                    {"url": "https://icis.corp.delaware.gov/Ecorp/EntitySearch/NameSearch.aspx", "method": "web"}
                ]
            },
            "vital_records": {
                "california": [
                    {"url": "https://www.cdph.ca.gov/vitals", "method": "web"}
                ]
            }
        }

        # Normalize jurisdiction (state name to lowercase)
        juris_key = jurisdiction.lower().replace(' ', '_')

        type_endpoints = endpoints.get(record_type, {})
        jurisdiction_endpoints = type_endpoints.get(juris_key, [])

        if not jurisdiction_endpoints:
            logger.warning(f"No endpoints configured for {record_type} in {jurisdiction}")
            return []

        return jurisdiction_endpoints

    async def _scrape_endpoint(self, endpoint: Dict[str, Any], search_terms: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Scrape records from a specific endpoint."""
        method = endpoint.get('method', 'web')
        url = endpoint.get('url')

        if method == 'api':
            return await self._scrape_api_endpoint(url, search_terms)
        elif method == 'web':
            return await self._scrape_web_endpoint(url, search_terms)
        else:
            raise ValueError(f"Unknown endpoint method: {method}")

    async def _scrape_api_endpoint(self, url: str, search_terms: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Scrape records from an API endpoint."""
        # This would implement API-specific scraping logic
        # For now, return mock data structure
        logger.debug(f"API scraping not implemented for {url}")
        return []

    async def _scrape_web_endpoint(self, url: str, search_terms: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Scrape records from a web endpoint."""
        # This would implement web scraping logic for government sites
        # For now, return mock data structure
        logger.debug(f"Web scraping not implemented for {url}")
        return []

    async def _validate_records(self, records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Validate and clean public records data."""
        validated = []

        for record in records:
            try:
                # Basic validation
                if not self._validate_record_structure(record):
                    continue

                # Content validation
                cleaned_record = self._clean_record_data(record)

                # Data quality checks
                if self._passes_quality_checks(cleaned_record):
                    validated.append(cleaned_record)

            except Exception as e:
                logger.debug(f"Record validation failed: {e}")
                continue

        return validated

    def _validate_record_structure(self, record: Dict[str, Any]) -> bool:
        """Validate basic record structure."""
        required_fields = ['record_type', 'jurisdiction']
        return all(field in record for field in required_fields)

    def _clean_record_data(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """Clean and normalize record data."""
        cleaned = record.copy()

        # Normalize dates
        for date_field in ['filed_date', 'record_date', 'effective_date']:
            if date_field in cleaned and cleaned[date_field]:
                cleaned[date_field] = self._normalize_date(cleaned[date_field])

        # Normalize addresses
        if 'address' in cleaned:
            cleaned['address'] = self._normalize_address(cleaned['address'])

        # Normalize names
        for name_field in ['owner_name', 'business_name', 'entity_name']:
            if name_field in cleaned and cleaned[name_field]:
                cleaned[name_field] = self._normalize_name(cleaned[name_field])

        return cleaned

    def _passes_quality_checks(self, record: Dict[str, Any]) -> bool:
        """Perform quality checks on record data."""
        # Check for required data
        if record.get('record_type') == 'property_records':
            if not (record.get('address') or record.get('parcel_id')):
                return False

        elif record.get('record_type') == 'business_registrations':
            if not record.get('business_name'):
                return False

        # Check data completeness
        total_fields = len(record)
        populated_fields = sum(1 for v in record.values() if v is not None and str(v).strip())

        # Require at least 60% field completion
        if populated_fields / total_fields < 0.6:
            return False

        return True

    def _anonymize_sensitive_data(self, records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Anonymize sensitive personal information."""
        anonymized = []

        for record in records:
            anon_record = record.copy()

            # Remove or mask sensitive fields
            sensitive_fields = ['ssn', 'social_security', 'drivers_license', 'passport']

            for field in sensitive_fields:
                if field in anon_record:
                    anon_record[field] = self._mask_sensitive_data(anon_record[field])

            # Mask partial sensitive data in text fields
            for field in ['description', 'notes', 'comments']:
                if field in anon_record and anon_record[field]:
                    anon_record[field] = self._mask_text_sensitive_data(anon_record[field])

            anonymized.append(anon_record)

        return anonymized

    def _mask_sensitive_data(self, value: Any) -> str:
        """Mask sensitive data values."""
        if not value:
            return ""

        value_str = str(value)

        # SSN masking
        if re.match(r'\d{3}-?\d{2}-?\d{4}', value_str):
            return "XXX-XX-XXXX"

        # General masking for unknown sensitive data
        return "REDACTED"

    def _mask_text_sensitive_data(self, text: str) -> str:
        """Mask sensitive data within text content."""
        # Mask SSNs
        text = self.validation_patterns['ssn_pattern'].sub('XXX-XX-XXXX', text)

        # Mask phone numbers
        text = self.validation_patterns['phone_pattern'].sub('(XXX) XXX-XXXX', text)

        # Mask emails
        text = self.validation_patterns['email_pattern'].sub('user@domain.com', text)

        return text

    def _normalize_date(self, date_value: Any) -> Optional[str]:
        """Normalize date values to ISO format."""
        if not date_value:
            return None

        # Handle various date formats
        # This would implement comprehensive date parsing
        return str(date_value)

    def _normalize_address(self, address: Any) -> Optional[Dict[str, Any]]:
        """Normalize address data."""
        if not address:
            return None

        try:
            # Use usaddress library if available
            if self.address_parser:
                parsed = self.address_parser.parse(address)
                return dict(parsed)
            else:
                return {"raw_address": str(address)}
        except:
            return {"raw_address": str(address)}

    def _normalize_name(self, name: str) -> str:
        """Normalize person/company names."""
        if not name:
            return ""

        # Basic normalization
        normalized = str(name).strip().title()

        # Handle common abbreviations
        normalized = re.sub(r'\bCorp\b', 'Corporation', normalized)
        normalized = re.sub(r'\bInc\b', 'Incorporated', normalized)
        normalized = re.sub(r'\bLtd\b', 'Limited', normalized)

        return normalized

    def _classify_record_sensitivity(self, record: Dict[str, Any]) -> str:
        """Classify record sensitivity level."""
        record_type = record.get('record_type', '')

        # High sensitivity records
        if record_type in ['court_records', 'vital_records']:
            return 'high'

        # Medium sensitivity
        elif record_type in ['property_records', 'business_registrations']:
            return 'medium'

        # Low sensitivity
        else:
            return 'low'

    async def _check_source_rate_limit(self, record_type: str, jurisdiction: str):
        """Check and enforce rate limits for data sources."""
        source_key = f"{record_type}_{jurisdiction}"

        if source_key not in self.source_rate_limits:
            self.source_rate_limits[source_key] = {
                'requests': [],
                'rate_limit': self.record_sources[record_type]['rate_limit']
            }

        source_limits = self.source_rate_limits[source_key]
        current_time = datetime.utcnow().timestamp()

        # Clean old requests (keep last minute)
        source_limits['requests'] = [t for t in source_limits['requests'] if current_time - t < 60]

        # Check if under rate limit
        if len(source_limits['requests']) >= source_limits['rate_limit']:
            # Calculate wait time
            oldest_request = min(source_limits['requests'])
            wait_time = 60 - (current_time - oldest_request)

            if wait_time > 0:
                logger.info(f"Rate limited for {source_key}, waiting {wait_time:.1f}s")
                await asyncio.sleep(wait_time)

        # Record this request
        source_limits['requests'].append(current_time)

    async def _log_compliance_event(self, event: Dict[str, Any]):
        """Log compliance-related events."""
        if self.public_config.compliance_logging:
            self.compliance_log.append(event)

            # Keep only recent events
            if len(self.compliance_log) > 1000:
                self.compliance_log = self.compliance_log[-1000:]

    async def _validate_result(self, result: Dict[str, Any]) -> bool:
        """Validate public records scraping result."""
        # Check basic structure
        if 'records' not in result or 'total_records' not in result:
            return False

        # Check if we have source stats
        if 'source_stats' not in result:
            return False

        # Verify we processed some jurisdictions
        jurisdictions_processed = result.get('jurisdictions_processed', [])
        return len(jurisdictions_processed) > 0

    def get_public_records_metrics(self) -> Dict[str, Any]:
        """Get public records scraping specific metrics."""
        return {
            **self.get_metrics(),
            'records_scraped': self.records_scraped,
            'jurisdictions_tracked': len(self.jurisdiction_data),
            'record_sources': len(self.record_sources),
            'compliance_events': len(self.compliance_log),
            'features_enabled': {
                'property_records': self.public_config.extract_property_records,
                'business_licenses': self.public_config.extract_business_licenses,
                'court_records': self.public_config.extract_court_records,
                'vital_records': self.public_config.extract_vital_records,
                'data_validation': self.public_config.validate_data,
                'data_anonymization': self.public_config.anonymize_sensitive_data
            },
            'record_types_available': list(self.record_sources.keys()),
            'rate_limits': {k: v.get('rate_limit', 0) for k, v in self.record_sources.items()}
        }

    async def cleanup(self) -> None:
        """Cleanup public records scraper resources."""
        await super().cleanup()

        # Clear sensitive data
        self.jurisdiction_data.clear()
        self.compliance_log.clear()
        self.source_rate_limits.clear()
        self.records_scraped = 0

        logger.info("Public records scraper cleaned up")
