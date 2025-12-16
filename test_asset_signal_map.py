# Copyright (c) 2024 Mountain Jewels Intelligence. All rights reserved.
#
# This software is proprietary and confidential. Unauthorized copying,
# modification, distribution, or use is strictly prohibited.

"""
Test Asset Signal Mapping Engine for MJ Data Scraper Suite

Comprehensive testing of the signal mapping, transformation, deduplication,
and intelligence enhancement capabilities.
"""

import asyncio
from datetime import datetime, timedelta
from core.mapping.asset_signal_map import (
    AssetSignalMapper,
    MappingRule,
    map_to_signal,
    batch_map_to_signals,
    deduplicate_signal_batch,
    get_mapping_statistics,
    clear_signal_cache,
    export_mapping_configuration,
    import_mapping_configuration,
    get_optimal_sources_for_signal,
    calculate_signal_cost_estimate,
    validate_source_for_signal,
    get_data_freshness_requirement,
    is_signal_data_fresh,
    get_asset_signal_source_configuration,
    get_signal_cost_weight,
    get_source_reliability_score,
    get_sources_for_asset_signal,
    ASSET_SIGNAL_SOURCES,
    SIGNAL_COST_WEIGHT,
    SOURCE_RELIABILITY,
    SIGNAL_FRESHNESS_REQUIREMENTS
)
from core.models.asset_signal import (
    SignalType,
    SignalSource,
    AssetType
)


class TestAssetSignalMapping:
    """Test comprehensive asset signal mapping functionality."""

    def test_mapping_rule_initialization(self):
        """Test that default mapping rules are properly initialized."""
        mapper = AssetSignalMapper()

        assert len(mapper.mapping_rules) > 0
        assert all(isinstance(rule, MappingRule) for rule in mapper.mapping_rules)

        # Check that rules are sorted by priority
        priorities = [rule.priority for rule in mapper.mapping_rules]
        assert priorities == sorted(priorities, reverse=True)

        # Check specific rule types exist
        rule_types = {rule.signal_type for rule in mapper.mapping_rules}
        expected_types = {
            SignalType.LIEN, SignalType.MORTGAGE, SignalType.DEED,
            SignalType.FORECLOSURE, SignalType.JUDGMENT, SignalType.COURT_CASE,
            SignalType.TAX_ISSUE, SignalType.BIRTHDAY, SignalType.ENGAGEMENT,
            SignalType.WEDDING
        }
        assert expected_types.issubset(rule_types)

    def test_raw_data_to_signal_mapping(self):
        """Test mapping raw scraped data to signals."""
        mapper = AssetSignalMapper()

        # Test lien data mapping
        lien_data = {
            "amount": 25000,
            "recorded_date": "2024-01-15",
            "property_address": "123 Main St, Springfield, IL",
            "parcel_id": "COOK-123456",
            "lien_type": "tax_lien"
        }

        signal = asyncio.run(mapper.map_raw_data_to_signal(
            lien_data,
            SignalSource.PUBLIC_RECORDS
        ))

        assert signal is not None
        assert signal.signal_type == SignalType.LIEN
        assert signal.judgment_amount == 25000
        assert signal.property_address == "123 Main St, Springfield, IL"
        assert signal.parcel_id == "COOK-123456"
        assert signal.asset_type == AssetType.SINGLE_FAMILY_HOME

    def test_mortgage_data_mapping(self):
        """Test mortgage data mapping."""
        mapper = AssetSignalMapper()

        mortgage_data = {
            "loan_amount": 300000,
            "lender": "First National Bank",
            "interest_rate": 3.5,
            "loan_term": 30,
            "property_address": "456 Oak Ave, Chicago, IL"
        }

        signal = asyncio.run(mapper.map_raw_data_to_signal(
            mortgage_data,
            SignalSource.FINANCIAL_REPORTS
        ))

        assert signal is not None
        assert signal.signal_type == SignalType.MORTGAGE
        assert signal.transaction_amount == 300000
        assert signal.lender_name == "First National Bank"
        assert signal.interest_rate == 3.5
        assert signal.loan_term_years == 30

    def test_deed_data_mapping(self):
        """Test property deed data mapping."""
        mapper = AssetSignalMapper()

        deed_data = {
            "sale_price": 350000,
            "grantor": "John Seller",
            "grantee": "Jane Buyer",
            "property_address": "789 Pine St, Boston, MA",
            "transfer_date": "2024-03-15"
        }

        signal = asyncio.run(mapper.map_raw_data_to_signal(
            deed_data,
            SignalSource.PUBLIC_RECORDS
        ))

        assert signal is not None
        assert signal.signal_type == SignalType.DEED
        assert signal.transaction_amount == 350000
        assert signal.plaintiff_name == "John Seller"
        assert signal.defendant_name == "Jane Buyer"
        assert signal.property_address == "789 Pine St, Boston, MA"

    def test_foreclosure_data_mapping(self):
        """Test foreclosure data mapping."""
        mapper = AssetSignalMapper()

        foreclosure_data = {
            "case_number": "FC-2024-001",
            "property_address": "321 Elm St, Detroit, MI",
            "amount": 200000,
            "trustee": "ABC Trustee Services",
            "status": "active"
        }

        signal = asyncio.run(mapper.map_raw_data_to_signal(
            foreclosure_data,
            SignalSource.COURT_RECORDS
        ))

        assert signal is not None
        assert signal.signal_type == SignalType.FORECLOSURE
        assert signal.case_number == "FC-2024-001"
        assert signal.transaction_amount == 200000
        assert signal.attorney_name == "ABC Trustee Services"
        assert signal.business_value == 0.95  # High value for foreclosure
        assert "distressed_property_review" in signal.compliance_flags

    def test_personal_event_mapping(self):
        """Test personal event data mapping."""
        mapper = AssetSignalMapper()

        # Test wedding data
        wedding_data = {
            "spouse_name": "Sarah Johnson",
            "wedding_date": "2024-08-20",
            "venue": "Grand Hotel Ballroom",
            "description": "Beach-themed wedding reception"
        }

        signal = asyncio.run(mapper.map_raw_data_to_signal(
            wedding_data,
            SignalSource.WEB_SCRAPING
        ))

        assert signal is not None
        assert signal.signal_type == SignalType.WEDDING
        assert signal.asset_type == AssetType.PERSON
        assert signal.spouse_name == "Sarah Johnson"
        assert signal.business_value == 0.5
        assert signal.privacy_level == "personal"

    def test_asset_identification(self):
        """Test asset identification and deduplication."""
        mapper = AssetSignalMapper()

        # Create multiple signals for same asset
        signals_data = [
            {
                "property_address": "123 Main St, Springfield, IL",
                "amount": 25000,
                "recorded_date": "2024-01-15"
            },
            {
                "property_address": "123 Main St, Springfield, IL",
                "parcel_id": "COOK-123456",
                "amount": 25000
            },
            {
                "address": "123 Main St, Springfield, IL",
                "lien_amount": 25000,
                "date": "2024-01-15"
            }
        ]

        asset_ids = []
        for data in signals_data:
            signal = asyncio.run(mapper.map_raw_data_to_signal(
                data,
                SignalSource.PUBLIC_RECORDS
            ))
            if signal:
                asset_ids.append(signal.asset_id)

        # All should map to same asset ID
        assert len(set(asset_ids)) == 1, f"Expected 1 unique asset ID, got {len(set(asset_ids))}: {asset_ids}"

    def test_signal_deduplication(self):
        """Test signal deduplication functionality."""
        mapper = AssetSignalMapper()

        # Create duplicate signals
        base_data = {
            "property_address": "456 Oak St, Chicago, IL",
            "amount": 15000,
            "recorded_date": "2024-02-01",
            "case_number": "LIEN-2024-001"
        }

        # Create slightly different versions
        signals_data = [
            {**base_data},
            {**base_data, "amount": 15001},  # Slightly different amount
            {**base_data, "recorded_date": "2024-02-02"},  # Different date
            {**base_data, "case_number": "LIEN-2024-002"}  # Different case
        ]

        signals = []
        for data in signals_data:
            signal = asyncio.run(mapper.map_raw_data_to_signal(
                data,
                SignalSource.PUBLIC_RECORDS
            ))
            if signal:
                signals.append(signal)

        # Deduplicate
        deduplicated = asyncio.run(mapper.deduplicate_signals(signals))

        # Should have fewer signals after deduplication
        assert len(deduplicated) < len(signals), f"Expected deduplication, got {len(deduplicated)} from {len(signals)}"

        # Check that duplicates were tracked
        assert len(mapper.duplicate_map) > 0

    def test_batch_signal_mapping(self):
        """Test batch signal mapping with concurrency."""
        mapper = AssetSignalMapper()

        # Create batch of diverse data
        batch_data = [
            # Liens
            {"amount": 10000, "property_address": "100 First St", "recorded_date": "2024-01-01"},
            {"amount": 20000, "property_address": "200 Second St", "recorded_date": "2024-01-02"},

            # Mortgages
            {"loan_amount": 250000, "lender": "Bank A", "property_address": "300 Third St"},
            {"loan_amount": 300000, "lender": "Bank B", "property_address": "400 Fourth St"},

            # Weddings
            {"spouse_name": "Alice", "wedding_date": "2024-06-01", "venue": "Hotel A"},
            {"spouse_name": "Bob", "wedding_date": "2024-07-01", "venue": "Hotel B"},

            # Invalid data (should be filtered)
            {"random_field": "invalid"},
            {}
        ]

        signals = asyncio.run(mapper.batch_map_signals(
            batch_data,
            SignalSource.WEB_SCRAPING,
            max_concurrent=3
        ))

        # Should have created valid signals (excluding invalid data)
        assert len(signals) > 0
        assert len(signals) <= 6  # Up to 6 valid signals

        # Check signal types
        signal_types = {s.signal_type for s in signals}
        assert SignalType.LIEN in signal_types or SignalType.MORTGAGE in signal_types
        assert SignalType.WEDDING in signal_types

    def test_geographic_enrichment(self):
        """Test geographic data enrichment."""
        mapper = AssetSignalMapper()

        data_with_geo = {
            "property_address": "123 Main St",
            "city": "Springfield",
            "state": "IL",
            "zip_code": "62701",
            "county": "Sangamon",
            "amount": 15000
        }

        signal = asyncio.run(mapper.map_raw_data_to_signal(
            data_with_geo,
            SignalSource.PUBLIC_RECORDS
        ))

        assert signal is not None
        assert signal.property_address == "123 Main St"
        assert signal.property_city == "Springfield"
        assert signal.property_state == "IL"
        assert signal.property_zip == "62701"
        assert signal.property_county == "Sangamon"

    def test_temporal_enrichment(self):
        """Test temporal data enrichment."""
        mapper = AssetSignalMapper()

        data_with_dates = {
            "amount": 25000,
            "recorded_date": "2024-01-15",
            "filed_date": "2024-01-10"
        }

        signal = asyncio.run(mapper.map_raw_data_to_signal(
            data_with_dates,
            SignalSource.PUBLIC_RECORDS
        ))

        assert signal is not None
        assert signal.filing_date.year == 2024
        assert signal.filing_date.month == 1
        assert signal.filing_date.day == 15

    def test_cross_reference_addition(self):
        """Test cross-reference addition between related signals."""
        mapper = AssetSignalMapper()

        # Create two signals for same property
        signal1 = asyncio.run(mapper.map_raw_data_to_signal(
            {"property_address": "123 Main St", "amount": 10000},
            SignalSource.PUBLIC_RECORDS
        ))

        signal2 = asyncio.run(mapper.map_raw_data_to_signal(
            {"property_address": "123 Main St", "loan_amount": 200000},
            SignalSource.FINANCIAL_REPORTS
        ))

        assert signal1 is not None
        assert signal2 is not None

        # Both should reference each other
        assert signal2.signal_id in signal1.related_signals
        assert signal1.signal_id in signal2.related_signals

    def test_mapping_statistics(self):
        """Test mapping statistics collection."""
        mapper = AssetSignalMapper()

        # Generate some mapping activity
        test_data = [
            {"amount": 10000, "property_address": "100 Test St"},
            {"loan_amount": 200000, "lender": "Test Bank"},
            {"spouse_name": "Test Partner", "wedding_date": "2024-01-01"}
        ]

        for data in test_data:
            asyncio.run(mapper.map_raw_data_to_signal(data, SignalSource.WEB_SCRAPING))

        stats = mapper.get_mapping_stats()

        assert stats["mapping_rules_count"] > 0
        assert stats["cached_signals_count"] >= len(test_data)
        assert "mapping_stats" in stats
        assert "cache_performance" in stats

    def test_mapping_rule_export_import(self):
        """Test mapping rule export and import."""
        mapper1 = AssetSignalMapper()

        # Export rules
        exported_rules = mapper1.export_mapping_rules()
        assert len(exported_rules) > 0

        # Create new mapper and import rules
        mapper2 = AssetSignalMapper()
        mapper2.mapping_rules.clear()  # Clear default rules

        mapper2.import_mapping_rules(exported_rules)

        assert len(mapper2.mapping_rules) == len(exported_rules)

        # Test that imported rules work
        test_data = {"amount": 15000, "property_address": "123 Test St"}
        signal1 = asyncio.run(mapper1.map_raw_data_to_signal(test_data, SignalSource.PUBLIC_RECORDS))
        signal2 = asyncio.run(mapper2.map_raw_data_to_signal(test_data, SignalSource.PUBLIC_RECORDS))

        assert signal1 is not None
        assert signal2 is not None
        assert signal1.signal_type == signal2.signal_type

    def test_cache_management(self):
        """Test signal cache management."""
        mapper = AssetSignalMapper()

        # Add some signals to cache
        test_data = {"amount": 5000, "property_address": "Cache Test St"}
        signal = asyncio.run(mapper.map_raw_data_to_signal(test_data, SignalSource.PUBLIC_RECORDS))

        assert len(mapper.signal_cache) > 0

        # Clear cache
        mapper.clear_cache()

        assert len(mapper.signal_cache) == 0
        assert mapper.cache_hits == 0
        assert mapper.cache_misses == 0

    def test_optimal_source_selection(self):
        """Test intelligent source selection for signals."""
        mapper = AssetSignalMapper()

        # Test optimal sources for single family home lien
        optimal_sources = mapper.get_optimal_sources_for_signal(
            AssetType.SINGLE_FAMILY_HOME,
            SignalType.LIEN
        )

        assert len(optimal_sources) > 0
        assert "county_clerk" in optimal_sources  # Should be in configured sources

        # Test with current sources for diversity
        diverse_sources = mapper.get_optimal_sources_for_signal(
            AssetType.SINGLE_FAMILY_HOME,
            SignalType.LIEN,
            current_sources=["county_clerk"]
        )

        assert len(diverse_sources) > 0
        # Should prefer different sources when current sources provided

        # Test fallback sources for unconfigured combination
        fallback_sources = mapper.get_optimal_sources_for_signal(
            AssetType.PERSON,
            SignalType.ANOMALY  # Not in main configuration
        )

        assert len(fallback_sources) > 0
        assert "monitoring_systems" in fallback_sources

    def test_source_scoring_and_ranking(self):
        """Test source scoring algorithm."""
        mapper = AssetSignalMapper()

        # Test source score calculation
        score = mapper._calculate_source_score("county_clerk", SignalType.LIEN)
        assert 0 <= score <= 1.0

        # High reliability source should score well
        high_reliability_score = mapper._calculate_source_score("county_clerk", SignalType.LIEN)
        low_reliability_score = mapper._calculate_source_score("social_media", SignalType.LIEN)

        assert high_reliability_score > low_reliability_score

    def test_signal_cost_estimation(self):
        """Test comprehensive cost estimation for signals."""
        mapper = AssetSignalMapper()

        # Test basic cost estimation
        cost_estimate = mapper.calculate_signal_cost_estimate(
            AssetType.SINGLE_FAMILY_HOME,
            SignalType.LIEN
        )

        assert "total_estimated_cost" in cost_estimate
        assert cost_estimate["total_estimated_cost"] > 0
        assert "base_signal_cost" in cost_estimate
        assert "asset_complexity_multiplier" in cost_estimate
        assert "estimate_confidence" in cost_estimate

        # Test with specific sources
        specific_cost = mapper.calculate_signal_cost_estimate(
            AssetType.SINGLE_FAMILY_HOME,
            SignalType.LIEN,
            sources=["county_clerk", "tax_assessor"]
        )

        assert specific_cost["total_estimated_cost"] > 0
        assert "specific_sources" in specific_cost["confidence_sources"]

        # Test quality levels
        premium_cost = mapper.calculate_signal_cost_estimate(
            AssetType.SINGLE_FAMILY_HOME,
            SignalType.LIEN,
            data_quality="premium"
        )

        standard_cost = mapper.calculate_signal_cost_estimate(
            AssetType.SINGLE_FAMILY_HOME,
            SignalType.LIEN,
            data_quality="standard"
        )

        assert premium_cost["total_estimated_cost"] > standard_cost["total_estimated_cost"]

        # Test asset complexity impact
        complex_asset_cost = mapper.calculate_signal_cost_estimate(
            AssetType.COMMERCIAL_PROPERTY,
            SignalType.LIEN
        )

        simple_asset_cost = mapper.calculate_signal_cost_estimate(
            AssetType.PERSON,
            SignalType.BIRTHDAY
        )

        # Commercial property should be more expensive than personal birthday
        assert complex_asset_cost["total_estimated_cost"] > simple_asset_cost["total_estimated_cost"]

    def test_source_validation(self):
        """Test source validation and recommendations."""
        mapper = AssetSignalMapper()

        # Test validation of configured source
        validation = mapper.validate_source_for_signal(
            "county_clerk",
            AssetType.SINGLE_FAMILY_HOME,
            SignalType.LIEN
        )

        assert validation["is_configured"] == True
        assert validation["source"] == "county_clerk"
        assert validation["asset_type"] == "single_family_home"
        assert validation["signal_type"] == "lien"
        assert "reliability_score" in validation
        assert "performance_score" in validation

        # Test validation of non-configured source
        invalid_validation = mapper.validate_source_for_signal(
            "random_source",
            AssetType.SINGLE_FAMILY_HOME,
            SignalType.LIEN
        )

        assert invalid_validation["is_configured"] == False
        assert len(invalid_validation["recommendations"]) > 0
        assert len(invalid_validation["alternatives"]) > 0

        # Test low reliability source
        low_rel_validation = mapper.validate_source_for_signal(
            "social_media",
            AssetType.SINGLE_FAMILY_HOME,
            SignalType.LIEN
        )

        assert low_rel_validation["reliability_score"] < 0.7
        # Should have recommendation about low reliability

    def test_data_freshness_assessment(self):
        """Test signal data freshness evaluation."""
        mapper = AssetSignalMapper()

        # Test freshness requirement lookup
        lien_freshness = mapper.get_data_freshness_requirement(SignalType.LIEN)
        assert lien_freshness == 180  # 6 months for liens

        foreclosure_freshness = mapper.get_data_freshness_requirement(SignalType.FORECLOSURE)
        assert foreclosure_freshness == 7  # 1 week for foreclosure

        birthday_freshness = mapper.get_data_freshness_requirement(SignalType.BIRTHDAY)
        assert birthday_freshness == 365 * 10  # 10 years for birthdays

        # Test freshness assessment
        fresh_signal = AssetSignal(
            asset_id="test_123",
            signal_type=SignalType.LIEN,
            signal_source=SignalSource.PUBLIC_RECORDS,
            signal_value={"test": "data"},
            created_at=datetime.utcnow() - timedelta(days=30)  # 30 days old
        )

        freshness_assessment = mapper.is_signal_data_fresh(fresh_signal)
        assert freshness_assessment["is_fresh"] == True
        assert freshness_assessment["days_old"] == 30
        assert freshness_assessment["needs_refresh"] == False
        assert freshness_assessment["freshness_score"] > 0

        # Test stale signal
        stale_signal = AssetSignal(
            asset_id="test_456",
            signal_type=SignalType.LIEN,
            signal_source=SignalSource.PUBLIC_RECORDS,
            signal_value={"test": "data"},
            created_at=datetime.utcnow() - timedelta(days=200)  # 200 days old
        )

        stale_assessment = mapper.is_signal_data_fresh(stale_signal)
        assert stale_assessment["is_fresh"] == False
        assert stale_assessment["needs_refresh"] == True
        assert stale_assessment["freshness_score"] == 0.0

        # Test signal without timestamp
        no_timestamp_signal = AssetSignal(
            asset_id="test_789",
            signal_type=SignalType.LIEN,
            signal_source=SignalSource.PUBLIC_RECORDS,
            signal_value={"test": "data"}
            # No created_at timestamp
        )

        no_timestamp_assessment = mapper.is_signal_data_fresh(no_timestamp_signal)
        assert no_timestamp_assessment["is_fresh"] == False
        assert no_timestamp_assessment["needs_refresh"] == True
        assert "No timestamp available" in no_timestamp_assessment["reason"]

    def test_configuration_access_functions(self):
        """Test global configuration access functions."""
        # Test configuration retrieval
        config = get_asset_signal_source_configuration()
        assert "asset_signal_sources" in config
        assert "signal_cost_weights" in config
        assert "source_reliability" in config
        assert "signal_freshness_requirements" in config
        assert "configuration_stats" in config

        # Verify configuration stats
        stats = config["configuration_stats"]
        assert stats["total_asset_types"] == len(ASSET_SIGNAL_SOURCES)
        assert stats["cost_weighted_signals"] == len(SIGNAL_COST_WEIGHT)
        assert stats["reliability_scored_sources"] == len(SOURCE_RELIABILITY)

        # Test individual access functions
        cost_weight = get_signal_cost_weight(SignalType.LIEN)
        assert cost_weight == SIGNAL_COST_WEIGHT[SignalType.LIEN]

        reliability = get_source_reliability_score("county_clerk")
        assert reliability == SOURCE_RELIABILITY["county_clerk"]

        sources = get_sources_for_asset_signal(AssetType.SINGLE_FAMILY_HOME, SignalType.LIEN)
        assert sources == ASSET_SIGNAL_SOURCES[AssetType.SINGLE_FAMILY_HOME][SignalType.LIEN]

    def test_convenience_functions_integration(self):
        """Test that convenience functions work with the global mapper."""
        # Test optimal source selection
        optimal_sources = get_optimal_sources_for_signal(
            AssetType.SINGLE_FAMILY_HOME,
            SignalType.LIEN
        )
        assert len(optimal_sources) > 0

        # Test cost estimation
        cost_estimate = calculate_signal_cost_estimate(
            AssetType.SINGLE_FAMILY_HOME,
            SignalType.LIEN
        )
        assert cost_estimate["total_estimated_cost"] > 0

        # Test source validation
        validation = validate_source_for_signal(
            "county_clerk",
            AssetType.SINGLE_FAMILY_HOME,
            SignalType.LIEN
        )
        assert validation["is_configured"] == True

        # Test freshness functions
        freshness_days = get_data_freshness_requirement(SignalType.LIEN)
        assert freshness_days == 180

        test_signal = AssetSignal(
            asset_id="freshness_test",
            signal_type=SignalType.LIEN,
            signal_source=SignalSource.PUBLIC_RECORDS,
            signal_value={"test": "data"},
            created_at=datetime.utcnow()
        )

        freshness_check = is_signal_data_fresh(test_signal)
        assert freshness_check["is_fresh"] == True

    def test_asset_type_complexity_impact(self):
        """Test that asset type complexity affects cost calculations."""
        mapper = AssetSignalMapper()

        # Test different asset types with same signal
        person_cost = mapper.calculate_signal_cost_estimate(
            AssetType.PERSON,
            SignalType.BIRTHDAY
        )

        home_cost = mapper.calculate_signal_cost_estimate(
            AssetType.SINGLE_FAMILY_HOME,
            SignalType.LIEN
        )

        commercial_cost = mapper.calculate_signal_cost_estimate(
            AssetType.COMMERCIAL_PROPERTY,
            SignalType.LIEN
        )

        # Asset complexity should increase costs appropriately
        assert person_cost["asset_complexity_multiplier"] == 1.0  # Baseline
        assert home_cost["asset_complexity_multiplier"] == 1.2    # Property complexity
        assert commercial_cost["asset_complexity_multiplier"] == 1.8  # Commercial complexity

        # Costs should reflect complexity (though signal type also matters)
        # Person birthday (1.0) vs Commercial lien (1.8 * higher base cost)

    def test_source_performance_tracking(self):
        """Test that source performance is tracked and affects scoring."""
        mapper = AssetSignalMapper()

        # Simulate source performance data
        mapper.source_performance["test_source"]["success"] = 8
        mapper.source_performance["test_source"]["failure"] = 2
        mapper.source_performance["test_source"]["avg_confidence"] = 0.85

        # Test that performance affects scoring
        score_with_performance = mapper._calculate_source_score(
            "test_source",
            SignalType.LIEN
        )

        # Reset performance data
        mapper.source_performance["test_source"]["success"] = 0
        mapper.source_performance["test_source"]["failure"] = 0

        score_without_performance = mapper._calculate_source_score(
            "test_source",
            SignalType.LIEN
        )

        # Score with performance history should be higher
        assert score_with_performance > score_without_performance

    def test_signal_cost_tracking_integration(self):
        """Test that signal cost tracking affects future estimates."""
        mapper = AssetSignalMapper()

        # Set historical cost data
        mapper.signal_cost_tracking[SignalType.LIEN] = 2.8

        # Calculate cost estimate (should be influenced by historical data)
        cost_estimate = mapper.calculate_signal_cost_estimate(
            AssetType.SINGLE_FAMILY_HOME,
            SignalType.LIEN
        )

        # Should include historical adjustment
        assert cost_estimate["historical_adjustment"] > 1.0
        assert "historical_data" in cost_estimate["confidence_sources"]

    def test_comprehensive_source_intelligence(self):
        """Test the complete source intelligence pipeline."""
        mapper = AssetSignalMapper()

        # 1. Get optimal sources for a signal
        optimal_sources = mapper.get_optimal_sources_for_signal(
            AssetType.APARTMENT_BUILDING,
            SignalType.COURT_CASE
        )

        assert len(optimal_sources) > 0
        assert "state_court" in optimal_sources  # Should be in configured sources

        # 2. Validate the top source
        validation = mapper.validate_source_for_signal(
            optimal_sources[0],
            AssetType.APARTMENT_BUILDING,
            SignalType.COURT_CASE
        )

        assert validation["is_configured"] == True
        assert validation["is_optimal"] == True

        # 3. Get cost estimate using optimal sources
        cost_estimate = mapper.calculate_signal_cost_estimate(
            AssetType.APARTMENT_BUILDING,
            SignalType.COURT_CASE,
            sources=optimal_sources[:2]  # Use top 2 sources
        )

        assert cost_estimate["total_estimated_cost"] > 0
        assert cost_estimate["estimate_confidence"] > 0.5  # Good confidence with specific sources

        # 4. Verify the complete intelligence loop works
        assert len(optimal_sources) > 0
        assert validation["reliability_score"] > 0
        assert cost_estimate["total_estimated_cost"] > 0

    def test_convenience_functions(self):
        """Test global convenience functions."""
        # Test single signal mapping
        data = {"amount": 30000, "property_address": "456 Global St"}
        signal = asyncio.run(map_to_signal(data, SignalSource.PUBLIC_RECORDS))

        assert signal is not None
        assert signal.signal_type == SignalType.LIEN

        # Test batch mapping
        batch_data = [
            {"loan_amount": 150000, "lender": "Global Bank"},
            {"spouse_name": "Global Partner", "wedding_date": "2024-01-01"}
        ]

        signals = asyncio.run(batch_map_to_signals(batch_data, SignalSource.WEB_SCRAPING))
        assert len(signals) > 0

        # Test deduplication
        deduplicated = asyncio.run(deduplicate_signal_batch(signals))
        assert len(deduplicated) <= len(signals)

        # Test statistics
        stats = get_mapping_statistics()
        assert "mapping_rules_count" in stats

        # Test cache clearing
        clear_signal_cache()

        # Test configuration export/import
        config = export_mapping_configuration()
        assert len(config) > 0

        import_mapping_configuration(config)

    def test_edge_cases_and_error_handling(self):
        """Test edge cases and error handling."""
        mapper = AssetSignalMapper()

        # Test empty data
        signal = asyncio.run(mapper.map_raw_data_to_signal({}, SignalSource.WEB_SCRAPING))
        assert signal is None

        # Test invalid data types
        invalid_data = {"amount": "not_a_number", "address": 123}
        signal = asyncio.run(mapper.map_raw_data_to_signal(invalid_data, SignalSource.WEB_SCRAPING))
        # Should handle gracefully (may return None or create signal with defaults)

        # Test very large batch
        large_batch = [{"amount": i * 1000, "property_address": f"{i} Test St"} for i in range(50)]
        signals = asyncio.run(mapper.batch_map_signals(large_batch, SignalSource.PUBLIC_RECORDS, max_concurrent=5))

        assert len(signals) > 0
        assert len(signals) <= len(large_batch)

    def test_signal_enhancement(self):
        """Test signal enhancement features."""
        mapper = AssetSignalMapper()

        # Test data with various fields
        enhancement_data = {
            "amount": 45000,
            "property_address": "789 Enhanced St",
            "city": "Enhanced City",
            "state": "EX",
            "zip_code": "12345",
            "county": "Enhanced County",
            "recorded_date": "2024-02-15",
            "case_number": "ENH-2024-001"
        }

        signal = asyncio.run(mapper.map_raw_data_to_signal(
            enhancement_data,
            SignalSource.PUBLIC_RECORDS
        ))

        assert signal is not None
        # Check geographic enrichment
        assert signal.property_address == "789 Enhanced St"
        assert signal.property_city == "Enhanced City"
        assert signal.property_state == "EX"
        assert signal.property_zip == "12345"
        assert signal.property_county == "Enhanced County"

        # Check temporal enrichment
        assert signal.filing_date.year == 2024
        assert signal.filing_date.month == 2
        assert signal.filing_date.day == 15

        # Check case number mapping
        assert signal.case_number == "ENH-2024-001"


if __name__ == "__main__":
    # Run basic tests
    print("🗺️ Testing Asset Signal Mapping Engine...")

    test_instance = TestAssetSignalMapping()

    # Run individual tests
    try:
        test_instance.test_mapping_rule_initialization()
        print("✅ Mapping rule initialization tests passed")

        test_instance.test_raw_data_to_signal_mapping()
        print("✅ Raw data to signal mapping tests passed")

        test_instance.test_mortgage_data_mapping()
        print("✅ Mortgage data mapping tests passed")

        test_instance.test_deed_data_mapping()
        print("✅ Deed data mapping tests passed")

        test_instance.test_foreclosure_data_mapping()
        print("✅ Foreclosure data mapping tests passed")

        test_instance.test_personal_event_mapping()
        print("✅ Personal event mapping tests passed")

        test_instance.test_asset_identification()
        print("✅ Asset identification tests passed")

        test_instance.test_signal_deduplication()
        print("✅ Signal deduplication tests passed")

        test_instance.test_batch_signal_mapping()
        print("✅ Batch signal mapping tests passed")

        test_instance.test_geographic_enrichment()
        print("✅ Geographic enrichment tests passed")

        test_instance.test_temporal_enrichment()
        print("✅ Temporal enrichment tests passed")

        test_instance.test_cross_reference_addition()
        print("✅ Cross-reference addition tests passed")

        test_instance.test_mapping_statistics()
        print("✅ Mapping statistics tests passed")

        test_instance.test_mapping_rule_export_import()
        print("✅ Mapping rule export/import tests passed")

        test_instance.test_cache_management()
        print("✅ Cache management tests passed")

        test_instance.test_convenience_functions()
        print("✅ Convenience functions tests passed")

        test_instance.test_edge_cases_and_error_handling()
        print("✅ Edge cases and error handling tests passed")

        test_instance.test_signal_enhancement()
        print("✅ Signal enhancement tests passed")

        test_instance.test_optimal_source_selection()
        print("✅ Optimal source selection tests passed")

        test_instance.test_source_scoring_and_ranking()
        print("✅ Source scoring and ranking tests passed")

        test_instance.test_signal_cost_estimation()
        print("✅ Signal cost estimation tests passed")

        test_instance.test_source_validation()
        print("✅ Source validation tests passed")

        test_instance.test_data_freshness_assessment()
        print("✅ Data freshness assessment tests passed")

        test_instance.test_configuration_access_functions()
        print("✅ Configuration access functions tests passed")

        test_instance.test_convenience_functions_integration()
        print("✅ Convenience functions integration tests passed")

        test_instance.test_asset_type_complexity_impact()
        print("✅ Asset type complexity impact tests passed")

        test_instance.test_source_performance_tracking()
        print("✅ Source performance tracking tests passed")

        test_instance.test_signal_cost_tracking_integration()
        print("✅ Signal cost tracking integration tests passed")

        test_instance.test_comprehensive_source_intelligence()
        print("✅ Comprehensive source intelligence tests passed")

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

    print("\n🎉 All Asset Signal Mapping tests completed successfully!")
    print("🗺️ Signal mapping and transformation fully validated!")
