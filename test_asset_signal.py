# Copyright (c) 2024 Mountain Jewels Intelligence. All rights reserved.
#
# This software is proprietary and confidential. Unauthorized copying,
# modification, distribution, or use is strictly prohibited.

"""
Test Asset Signal Model for MJ Data Scraper Suite

Comprehensive testing of the enterprise-grade asset signal model,
validation utilities, and intelligence fusion capabilities.
"""

import asyncio
from datetime import datetime, timedelta
from core.models.asset_signal import (
    AssetSignal,
    SignalType,
    SignalSource,
    AssetType,
    SignalConfidence,
    SignalValidationStatus,
    Asset,
    SignalRequest,
    create_identity_signal,
    create_location_signal,
    create_contact_signal,
    create_professional_signal,
    create_relationship_signal,
    create_anomaly_signal,
    create_lien_signal,
    create_mortgage_signal,
    create_deed_signal,
    create_foreclosure_signal,
    create_court_case_signal,
    create_judgment_signal,
    create_tax_issue_signal,
    create_birthday_signal,
    create_engagement_signal,
    create_wedding_signal,
    assess_signal_quality,
    validate_signal_completeness,
    validate_signal_consistency,
    validate_signal_timeliness,
    fuse_signals,
    deduplicate_signals
)


class TestAssetSignal:
    """Test comprehensive asset signal functionality."""

    def test_signal_creation(self):
        """Test basic signal creation and validation."""
        signal = AssetSignal(
            asset_id="person_123",
            asset_type=AssetType.PERSON,
            signal_type=SignalType.IDENTITY,
            signal_source=SignalSource.WEB_SCRAPING,
            signal_value={"name": "John Doe", "age": 35},
            confidence_score=0.85,
            source_url="https://example.com/profile/johndoe",
            source_domain="example.com"
        )

        assert signal.asset_id == "person_123"
        assert signal.asset_type == AssetType.PERSON
        assert signal.signal_type == SignalType.IDENTITY
        assert signal.confidence_score == 0.85
        assert signal.confidence_level == SignalConfidence.HIGH
        assert signal.is_valid() == True
        assert signal.is_expired() == False

    def test_signal_validation(self):
        """Test signal validation methods."""
        # Valid signal
        valid_signal = AssetSignal(
            asset_id="test_123",
            asset_type=AssetType.PERSON,
            signal_type=SignalType.IDENTITY,
            signal_source=SignalSource.WEB_SCRAPING,
            signal_value={"name": "Test User"}
        )
        assert valid_signal.is_valid() == True

        # Expired signal
        expired_signal = AssetSignal(
            asset_id="test_123",
            asset_type=AssetType.PERSON,
            signal_type=SignalType.IDENTITY,
            signal_source=SignalSource.WEB_SCRAPING,
            signal_value={"name": "Test User"},
            validity_end=datetime.utcnow() - timedelta(days=1)
        )
        assert expired_signal.is_expired() == True
        assert expired_signal.is_valid() == False

        # Future signal
        future_signal = AssetSignal(
            asset_id="test_123",
            asset_type=AssetType.PERSON,
            signal_type=SignalType.IDENTITY,
            signal_source=SignalSource.WEB_SCRAPING,
            signal_value={"name": "Test User"},
            validity_start=datetime.utcnow() + timedelta(days=1)
        )
        assert future_signal.is_valid() == False

    def test_signal_quality_assessment(self):
        """Test signal quality assessment."""
        high_quality_signal = AssetSignal(
            asset_id="test_123",
            asset_type=AssetType.PERSON,
            signal_type=SignalType.IDENTITY,
            signal_source=SignalSource.WEB_SCRAPING,
            signal_value={"name": "Test User"},
            confidence_score=0.9,
            data_quality_score=0.95,
            completeness_score=0.9,
            timeliness_score=0.85,
            consistency_score=0.9,
            reliability_score=0.95
        )

        quality = assess_signal_quality(high_quality_signal)
        assert quality["overall_quality"] > 0.8
        assert quality["quality_category"] == "high"
        assert "Signal quality is acceptable" in quality["recommendations"]

        low_quality_signal = AssetSignal(
            asset_id="test_123",
            asset_type=AssetType.PERSON,
            signal_type=SignalType.IDENTITY,
            signal_source=SignalSource.WEB_SCRAPING,
            signal_value={"name": "Test User"},
            confidence_score=0.2,
            data_quality_score=0.3,
            completeness_score=0.2,
            timeliness_score=0.1,
            consistency_score=0.3,
            reliability_score=0.2
        )

        quality = assess_signal_quality(low_quality_signal)
        assert quality["overall_quality"] < 0.5
        assert quality["quality_category"] == "low"
        assert len(quality["recommendations"]) > 1

    def test_validation_utilities(self):
        """Test signal validation utilities."""
        complete_signal = AssetSignal(
            asset_id="test_123",
            asset_type=AssetType.PERSON,
            signal_type=SignalType.IDENTITY,
            signal_source=SignalSource.WEB_SCRAPING,
            signal_value={"name": "Test User", "identifiers": {"email": "test@example.com"}},
            asset_name="Test User",
            asset_identifiers={"email": "test@example.com"}
        )

        completeness = validate_signal_completeness(complete_signal)
        assert completeness >= 0.8

        # Test consistency validation
        consistent_signal = AssetSignal(
            asset_id="test_123",
            asset_type=AssetType.PERSON,
            signal_type=SignalType.IDENTITY,
            signal_source=SignalSource.WEB_SCRAPING,
            signal_value={"name": "Test User"},
            validity_start=datetime.utcnow() - timedelta(days=1),
            validity_end=datetime.utcnow() + timedelta(days=30),
            confidence_score=0.8
        )

        consistency = validate_signal_consistency(consistent_signal)
        assert consistency >= 0.9

        # Test timeliness validation
        fresh_signal = AssetSignal(
            asset_id="test_123",
            asset_type=AssetType.PERSON,
            signal_type=SignalType.IDENTITY,
            signal_source=SignalSource.WEB_SCRAPING,
            signal_value={"name": "Test User"},
            signal_timestamp=datetime.utcnow() - timedelta(days=1)
        )

        timeliness = validate_signal_timeliness(fresh_signal)
        assert timeliness >= 0.9

    def test_signal_creation_helpers(self):
        """Test signal creation helper functions."""
        # Identity signal
        identity_signal = create_identity_signal(
            asset_id="person_123",
            signal_value={"name": "John Doe", "verified": True},
            confidence_score=0.9,
            source_url="https://linkedin.com/in/johndoe"
        )
        assert identity_signal.signal_type == SignalType.IDENTITY
        assert identity_signal.asset_type == AssetType.PERSON
        assert identity_signal.confidence_score == 0.9

        # Location signal
        location_signal = create_location_signal(
            asset_id="person_123",
            location_data={"city": "New York", "country": "USA"},
            confidence_score=0.7,
            location_country="USA",
            location_city="New York"
        )
        assert location_signal.signal_type == SignalType.LOCATION
        assert location_signal.location_country == "USA"

        # Contact signal
        contact_signal = create_contact_signal(
            asset_id="person_123",
            contact_data={"email": "john@example.com", "phone": "+1234567890"},
            confidence_score=0.6
        )
        assert contact_signal.signal_type == SignalType.CONTACT

        # Professional signal
        professional_signal = create_professional_signal(
            asset_id="person_123",
            professional_data={"company": "Tech Corp", "title": "Engineer"},
            confidence_score=0.8
        )
        assert professional_signal.signal_type == SignalType.PROFESSIONAL

        # Anomaly signal
        anomaly_signal = create_anomaly_signal(
            asset_id="person_123",
            anomaly_data={"unusual_activity": "multiple_location_changes"},
            confidence_score=0.95
        )
        assert anomaly_signal.signal_type == SignalType.ANOMALY
        assert anomaly_signal.action_required == True
        assert anomaly_signal.action_type == "investigation"

    def test_signal_audit_trail(self):
        """Test signal audit trail functionality."""
        signal = AssetSignal(
            asset_id="test_123",
            asset_type=AssetType.PERSON,
            signal_type=SignalType.IDENTITY,
            signal_source=SignalSource.WEB_SCRAPING,
            signal_value={"name": "Test User"}
        )

        # Initially empty audit trail
        assert len(signal.audit_trail) == 0

        # Add audit entry
        signal.add_audit_entry("created", {"initial_creation": True}, "system")
        assert len(signal.audit_trail) == 1
        assert signal.audit_trail[0]["action"] == "created"
        assert signal.audit_trail[0]["user"] == "system"

        # Update validation status
        signal.update_validation_status(SignalValidationStatus.VALIDATED, "manual_review", "analyst")
        assert signal.validation_status == SignalValidationStatus.VALIDATED
        assert len(signal.audit_trail) == 2
        assert signal.audit_trail[1]["action"] == "validation_status_changed"

        # Update confidence
        signal.update_confidence(0.9, "additional_verification", "analyst")
        assert signal.confidence_score == 0.9
        assert len(signal.audit_trail) == 3
        assert signal.audit_trail[2]["action"] == "confidence_updated"

        # Get audit summary
        summary = signal.get_audit_summary()
        assert summary["total_events"] == 3
        assert summary["created_by"] == "system"
        assert summary["updated_by"] == "analyst"

    def test_signal_summary_and_serialization(self):
        """Test signal summary generation and serialization."""
        signal = AssetSignal(
            asset_id="person_123",
            asset_type=AssetType.PERSON,
            signal_type=SignalType.IDENTITY,
            signal_source=SignalSource.WEB_SCRAPING,
            signal_value={"name": "John Doe", "verified": True},
            confidence_score=0.85,
            business_value=0.7,
            signal_timestamp=datetime.utcnow() - timedelta(days=5)
        )

        summary = signal.to_summary_dict()
        assert summary["signal_id"] == signal.signal_id
        assert summary["asset_id"] == "person_123"
        assert summary["confidence_score"] == 0.85
        assert summary["business_value"] == 0.7
        assert summary["is_valid"] == True
        assert abs(summary["signal_age_days"] - 5.0) < 0.1
        assert summary["quality_score"] == signal.get_quality_score()

    def test_signal_fusion(self):
        """Test signal fusion capabilities."""
        signal1 = AssetSignal(
            asset_id="person_123",
            asset_type=AssetType.PERSON,
            signal_type=SignalType.IDENTITY,
            signal_source=SignalSource.WEB_SCRAPING,
            signal_value={"name": "John Doe"},
            confidence_score=0.7
        )

        signal2 = AssetSignal(
            asset_id="person_123",
            asset_type=AssetType.PERSON,
            signal_type=SignalType.IDENTITY,
            signal_source=SignalSource.API_COLLECTION,
            signal_value={"name": "John Doe", "verified": True},
            confidence_score=0.9
        )

        # Fuse signals with highest confidence strategy
        fused = fuse_signals([signal1, signal2], fusion_method="highest_confidence")
        assert fused.confidence_score == 0.9
        assert len(fused.fusion_sources) == 2
        assert fused.fusion_method == "highest_confidence"
        assert len(fused.related_signals) == 1

        # Fuse signals with average strategy
        fused_avg = fuse_signals([signal1, signal2], fusion_method="average")
        assert fused_avg.confidence_score == 0.8  # (0.7 + 0.9) / 2

    def test_signal_deduplication(self):
        """Test signal deduplication functionality."""
        signals = [
            AssetSignal(
                asset_id="person_123",
                asset_type=AssetType.PERSON,
                signal_type=SignalType.IDENTITY,
                signal_source=SignalSource.WEB_SCRAPING,
                signal_value={"name": "John Doe"},
                confidence_score=0.8
            ),
            AssetSignal(
                asset_id="person_123",
                asset_type=AssetType.PERSON,
                signal_type=SignalType.IDENTITY,
                signal_source=SignalSource.API_COLLECTION,
                signal_value={"name": "John Doe"},
                confidence_score=0.75  # Similar confidence
            ),
            AssetSignal(
                asset_id="person_123",
                asset_type=AssetType.PERSON,
                signal_type=SignalType.CONTACT,
                signal_source=SignalSource.WEB_SCRAPING,
                signal_value={"email": "john@example.com"},
                confidence_score=0.6  # Different signal type
            )
        ]

        deduplicated = deduplicate_signals(signals, similarity_threshold=0.8)

        # Should keep 2 signals: one identity (highest confidence) and one contact (different type)
        assert len(deduplicated) == 2
        identity_signals = [s for s in deduplicated if s.signal_type == SignalType.IDENTITY]
        contact_signals = [s for s in deduplicated if s.signal_type == SignalType.CONTACT]
        assert len(identity_signals) == 1
        assert len(contact_signals) == 1
        assert identity_signals[0].confidence_score == 0.8  # Kept the higher confidence one

    def test_compliance_and_privacy(self):
        """Test compliance and privacy features."""
        signal = AssetSignal(
            asset_id="person_123",
            asset_type=AssetType.PERSON,
            signal_type=SignalType.IDENTITY,
            signal_source=SignalSource.WEB_SCRAPING,
            signal_value={"name": "John Doe", "ssn": "123-45-6789"},
            privacy_level="sensitive",
            compliance_flags=["gdpr_requires_consent", "ccpa_right_to_delete"],
            data_sensitivity="high",
            retention_policy="restricted"
        )

        compliance_summary = signal.get_compliance_summary()
        assert compliance_summary["privacy_level"] == "sensitive"
        assert len(compliance_summary["compliance_flags"]) == 2
        assert compliance_summary["compliant"] == False  # Has compliance flags

        processing_summary = signal.get_processing_summary()
        assert processing_summary["processing_success"] == True  # No processing errors
        assert processing_summary["pipeline_length"] == 0  # No processing steps

    def test_signal_lifecycle(self):
        """Test complete signal lifecycle."""
        # Create signal
        signal = create_identity_signal(
            asset_id="person_123",
            signal_value={"name": "John Doe"},
            confidence_score=0.6
        )

        # Initial state
        assert signal.validation_status == SignalValidationStatus.UNVALIDATED
        assert signal.requires_validation() == True

        # Add processing step
        signal.processing_pipeline.append("data_extraction")
        signal.processing_duration = 0.5

        # Update confidence
        signal.update_confidence(0.8, "cross_referenced_data")

        # Validate signal
        signal.update_validation_status(
            SignalValidationStatus.VALIDATED,
            "manual_review",
            "analyst"
        )

        # Add relationship
        related_signal = create_contact_signal(
            asset_id="person_123",
            contact_data={"email": "john@example.com"},
            confidence_score=0.7
        )

        signal.related_signals.append(related_signal.signal_id)

        # Final state checks
        assert signal.validation_status == SignalValidationStatus.VALIDATED
        assert signal.requires_validation() == False
        assert signal.confidence_score == 0.8
        assert len(signal.related_signals) == 1
        assert len(signal.audit_trail) == 3  # confidence update + validation + initial creation

        final_summary = signal.to_summary_dict()
        assert final_summary["is_valid"] == True
        assert final_summary["confidence_level"] == "high"

    def test_signal_metadata_and_tags(self):
        """Test signal metadata and tagging."""
        signal = AssetSignal(
            asset_id="company_456",
            asset_type=AssetType.COMPANY,
            signal_type=SignalType.FINANCIAL,
            signal_source=SignalSource.FINANCIAL_REPORTS,
            signal_value={"revenue": 1000000, "profit": 100000},
            signal_tags=["technology", "startup", "high_growth"],
            metadata={
                "industry": "software",
                "employee_count": 50,
                "funding_round": "series_a"
            },
            external_references={
                "crunchbase": "company_456",
                "linkedin": "company_456_linkedin"
            },
            correlation_ids=["scraping_job_123", "analysis_run_456"]
        )

        assert "technology" in signal.signal_tags
        assert signal.metadata["industry"] == "software"
        assert "crunchbase" in signal.external_references
        assert len(signal.correlation_ids) == 2

    def test_signal_business_intelligence(self):
        """Test business intelligence features."""
        high_value_signal = AssetSignal(
            asset_id="person_vip",
            asset_type=AssetType.PERSON,
            signal_type=SignalType.PROFESSIONAL,
            signal_source=SignalSource.WEB_SCRAPING,
            signal_value={"company": "Fortune 500 Corp", "title": "CEO"},
            business_value=0.95,
            intelligence_priority="high",
            action_required=True,
            action_type="immediate_followup"
        )

        assert high_value_signal.business_value == 0.95
        assert high_value_signal.intelligence_priority == "high"
        assert high_value_signal.action_required == True
        assert high_value_signal.action_type == "immediate_followup"

        low_value_signal = AssetSignal(
            asset_id="person_ordinary",
            asset_type=AssetType.PERSON,
            signal_type=SignalType.SOCIAL,
            signal_source=SignalSource.SOCIAL_MEDIA,
            signal_value={"platform": "twitter", "followers": 100},
            business_value=0.2,
            intelligence_priority="low"
        )

        assert low_value_signal.business_value == 0.2
        assert low_value_signal.intelligence_priority == "low"
        assert low_value_signal.action_required == False

    def test_signal_validation_errors(self):
        """Test signal validation error handling."""
        # Test invalid confidence score
        try:
            invalid_signal = AssetSignal(
                asset_id="test_123",
                asset_type=AssetType.PERSON,
                signal_type=SignalType.IDENTITY,
                signal_source=SignalSource.WEB_SCRAPING,
                signal_value={"name": "Test"},
                confidence_score=1.5  # Invalid: > 1.0
            )
            assert False, "Should have raised validation error"
        except ValueError:
            pass  # Expected

        # Test invalid validity period
        try:
            invalid_signal = AssetSignal(
                asset_id="test_123",
                asset_type=AssetType.PERSON,
                signal_type=SignalType.IDENTITY,
                signal_source=SignalSource.WEB_SCRAPING,
                signal_value={"name": "Test"},
                validity_start=datetime.utcnow() + timedelta(days=1),
                validity_end=datetime.utcnow()  # End before start
            )
            assert False, "Should have raised validation error"
        except ValueError:
            pass  # Expected

    def test_asset_model(self):
        """Test the Asset model functionality."""
        asset = Asset(
            asset_type=AssetType.SINGLE_FAMILY_HOME,
            address="123 Main St",
            city="Springfield",
            state="IL",
            zip_code="62701",
            owner_name="John Doe"
        )

        assert asset.asset_type == AssetType.SINGLE_FAMILY_HOME
        assert asset.get_full_address() == "123 Main St, Springfield, IL, 62701"
        assert asset.get_location_string() == "Springfield, IL"
        assert asset.is_property() == True

        # Test property type categorization
        assert asset.get_property_type_category() == "residential_single"

        # Test commercial property
        commercial = Asset(asset_type=AssetType.COMMERCIAL_PROPERTY)
        assert commercial.get_property_type_category() == "commercial"

    def test_signal_request_model(self):
        """Test the SignalRequest model functionality."""
        request = SignalRequest(
            signals=[SignalType.LIEN, SignalType.MORTGAGE, SignalType.DEED],
            time_window_days=90,
            min_confidence=0.7,
            max_results=100
        )

        assert len(request.signals) == 3
        assert request.time_window_days == 90
        assert request.should_include_signal(SignalType.LIEN, 0.8) == True
        assert request.should_include_signal(SignalType.LIEN, 0.5) == False  # Below min confidence
        assert request.should_include_signal(SignalType.BIRTHDAY, 0.8) == False  # Not in requested signals

        assert request.get_signal_priority(SignalType.LIEN) == "normal"

        # Test with priority signals
        priority_request = SignalRequest(
            signals=[SignalType.LIEN, SignalType.MORTGAGE],
            priority_signals=[SignalType.LIEN]
        )
        assert priority_request.get_signal_priority(SignalType.LIEN) == "high"
        assert priority_request.get_signal_priority(SignalType.MORTGAGE) == "normal"

    def test_real_estate_signal_creation(self):
        """Test real estate signal creation helpers."""
        # Test lien signal
        lien_signal = create_lien_signal(
            asset_id="property_123",
            lien_data={"amount": 50000, "type": "tax_lien"},
            confidence_score=0.9,
            property_address="123 Main St"
        )

        assert lien_signal.signal_type == SignalType.LIEN
        assert lien_signal.asset_type == AssetType.SINGLE_FAMILY_HOME
        assert lien_signal.business_value == 0.8
        assert lien_signal.intelligence_priority == "high"
        assert lien_signal.action_required == True
        assert lien_signal.action_type == "title_search_review"

        # Test mortgage signal
        mortgage_signal = create_mortgage_signal(
            asset_id="property_123",
            mortgage_data={"amount": 250000, "lender": "Bank of America"},
            lender_name="Bank of America",
            transaction_amount=250000
        )

        assert mortgage_signal.signal_type == SignalType.MORTGAGE
        assert mortgage_signal.business_value == 0.6
        assert mortgage_signal.intelligence_priority == "medium"

        # Test deed signal
        deed_signal = create_deed_signal(
            asset_id="property_123",
            deed_data={"transfer_amount": 300000, "buyer": "Jane Smith"},
            transaction_amount=300000
        )

        assert deed_signal.signal_type == SignalType.DEED
        assert deed_signal.business_value == 0.9
        assert deed_signal.intelligence_priority == "high"
        assert deed_signal.action_type == "ownership_verification"

        # Test foreclosure signal
        foreclosure_signal = create_foreclosure_signal(
            asset_id="property_123",
            foreclosure_data={"case_number": "FC-2024-001", "status": "active"},
            case_number="FC-2024-001"
        )

        assert foreclosure_signal.signal_type == SignalType.FORECLOSURE
        assert foreclosure_signal.business_value == 0.95
        assert foreclosure_signal.intelligence_priority == "critical"
        assert foreclosure_signal.action_type == "immediate_investigation"
        assert "distressed_property_review" in foreclosure_signal.compliance_flags

    def test_legal_signal_creation(self):
        """Test legal signal creation helpers."""
        # Test court case signal
        court_signal = create_court_case_signal(
            asset_id="person_123",
            court_data={"case_number": "CV-2024-001", "court": "Superior Court"},
            case_number="CV-2024-001",
            court_name="Superior Court"
        )

        assert court_signal.signal_type == SignalType.COURT_CASE
        assert court_signal.business_value == 0.75
        assert court_signal.data_sensitivity == "high"
        assert court_signal.action_type == "legal_review"

        # Test judgment signal
        judgment_signal = create_judgment_signal(
            asset_id="person_123",
            judgment_data={"amount": 15000, "plaintiff": "ABC Corp"},
            judgment_amount=15000,
            plaintiff_name="ABC Corp"
        )

        assert judgment_signal.signal_type == SignalType.JUDGMENT
        assert judgment_signal.business_value == 0.85
        assert "financial_liability_review" in judgment_signal.compliance_flags

        # Test tax issue signal
        tax_signal = create_tax_issue_signal(
            asset_id="property_123",
            tax_data={"amount": 5000, "year": 2023},
            tax_amount=5000,
            tax_year=2023
        )

        assert tax_signal.signal_type == SignalType.TAX_ISSUE
        assert tax_signal.business_value == 0.7
        assert "tax_compliance_review" in tax_signal.compliance_flags

    def test_personal_event_signal_creation(self):
        """Test personal event signal creation helpers."""
        # Test birthday signal
        birthday_signal = create_birthday_signal(
            asset_id="person_123",
            birthday_data={"date": "1990-05-15", "age": 34}
        )

        assert birthday_signal.signal_type == SignalType.BIRTHDAY
        assert birthday_signal.asset_type == AssetType.PERSON
        assert birthday_signal.business_value == 0.3
        assert birthday_signal.privacy_level == "personal"

        # Test engagement signal
        engagement_signal = create_engagement_signal(
            asset_id="person_123",
            engagement_data={"partner": "Jane Doe", "announced": "2024-01-15"}
        )

        assert engagement_signal.signal_type == SignalType.ENGAGEMENT
        assert engagement_signal.business_value == 0.4

        # Test wedding signal
        wedding_signal = create_wedding_signal(
            asset_id="person_123",
            wedding_data={"spouse": "Jane Doe", "date": "2024-06-15"},
            spouse_name="Jane Doe"
        )

        assert wedding_signal.signal_type == SignalType.WEDDING
        assert wedding_signal.business_value == 0.5
        assert wedding_signal.spouse_name == "Jane Doe"

    def test_signal_property_intelligence(self):
        """Test property-specific intelligence methods."""
        # Create a lien signal with property data
        lien_signal = AssetSignal(
            asset_id="property_123",
            asset_type=AssetType.SINGLE_FAMILY_HOME,
            signal_type=SignalType.LIEN,
            signal_source=SignalSource.PUBLIC_RECORDS,
            signal_value={"lien_amount": 25000, "recorded_date": "2024-01-15"},
            property_address="456 Oak St",
            property_city="Springfield",
            property_state="IL",
            parcel_id="123-456-789",
            judgment_amount=25000
        )

        assert lien_signal.is_property_signal() == True
        assert lien_signal.is_legal_signal() == True
        assert lien_signal.get_property_address() == "456 Oak St"
        assert lien_signal.get_financial_impact() == 25000

        # Test property intelligence
        intelligence = lien_signal.get_property_intelligence()
        assert intelligence["property_address"] == "456 Oak St"
        assert intelligence["parcel_id"] == "123-456-789"
        assert intelligence["financial_impact"] == 25000
        assert intelligence["signal_type"] == "lien"
        assert "risk_assessment" in intelligence

    def test_signal_risk_assessment(self):
        """Test comprehensive risk assessment for signals."""
        # High-risk foreclosure signal
        foreclosure_signal = AssetSignal(
            asset_id="property_123",
            signal_type=SignalType.FORECLOSURE,
            signal_source=SignalSource.COURT_RECORDS,
            signal_value={"case_status": "active", "amount": 300000},
            transaction_amount=300000,
            confidence_score=0.95
        )

        risk_assessment = foreclosure_signal.get_risk_assessment()
        assert risk_assessment["risk_level"] == "critical"
        assert risk_assessment["risk_score"] >= 0.9
        assert len(risk_assessment["recommendations"]) > 0
        assert "IMMEDIATE ACTION REQUIRED" in risk_assessment["recommendations"][0]

        # Medium-risk court case
        court_signal = AssetSignal(
            asset_id="person_123",
            signal_type=SignalType.COURT_CASE,
            signal_source=SignalSource.COURT_RECORDS,
            signal_value={"case_type": "civil", "amount": 15000},
            judgment_amount=15000,
            confidence_score=0.8
        )

        risk_assessment = court_signal.get_risk_assessment()
        assert risk_assessment["risk_level"] == "medium"
        assert risk_assessment["financial_impact"] == 15000

        # Low-risk mortgage
        mortgage_signal = AssetSignal(
            asset_id="property_123",
            signal_type=SignalType.MORTGAGE,
            signal_source=SignalSource.PUBLIC_RECORDS,
            signal_value={"loan_amount": 200000},
            transaction_amount=200000,
            confidence_score=0.85
        )

        risk_assessment = mortgage_signal.get_risk_assessment()
        assert risk_assessment["risk_level"] == "low"

    def test_signal_compliance_requirements(self):
        """Test compliance requirements for different signal types."""
        # Judgment signal
        judgment_signal = create_judgment_signal(
            asset_id="person_123",
            judgment_data={"amount": 50000}
        )

        compliance_reqs = judgment_signal.get_compliance_requirements()
        assert "financial_liability_review" in compliance_reqs
        assert "title_search_required" in compliance_reqs

        # Tax issue signal
        tax_signal = create_tax_issue_signal(
            asset_id="property_123",
            tax_data={"amount": 8000}
        )

        compliance_reqs = tax_signal.get_compliance_requirements()
        assert "tax_compliance_review" in compliance_reqs

        # Personal event signal
        birthday_signal = create_birthday_signal(
            asset_id="person_123",
            birthday_data={"date": "1990-01-01"}
        )

        compliance_reqs = birthday_signal.get_compliance_requirements()
        assert "privacy_compliance_check" in compliance_reqs

    def test_signal_legal_parties(self):
        """Test legal parties extraction from signals."""
        judgment_signal = AssetSignal(
            asset_id="case_123",
            signal_type=SignalType.JUDGMENT,
            signal_source=SignalSource.COURT_RECORDS,
            signal_value={"case_details": "Contract dispute"},
            plaintiff_name="ABC Corp",
            defendant_name="John Doe",
            attorney_name="Jane Smith"
        )

        parties = judgment_signal.get_legal_parties()
        assert parties["plaintiff"] == "ABC Corp"
        assert parties["defendant"] == "John Doe"
        assert parties["attorney"] == "Jane Smith"
        assert parties["owner"] is None  # Not set in this signal

    def test_signal_signal_type_classification(self):
        """Test signal type classification methods."""
        # Property signals
        lien_signal = AssetSignal(
            asset_id="prop_123",
            signal_type=SignalType.LIEN,
            signal_source=SignalSource.PUBLIC_RECORDS,
            signal_value={}
        )
        assert lien_signal.is_property_signal() == True
        assert lien_signal.is_legal_signal() == True

        mortgage_signal = AssetSignal(
            asset_id="prop_123",
            signal_type=SignalType.MORTGAGE,
            signal_source=SignalSource.PUBLIC_RECORDS,
            signal_value={}
        )
        assert mortgage_signal.is_property_signal() == True
        assert mortgage_signal.is_legal_signal() == False

        # Legal signals
        court_signal = AssetSignal(
            asset_id="person_123",
            signal_type=SignalType.COURT_CASE,
            signal_source=SignalSource.COURT_RECORDS,
            signal_value={}
        )
        assert court_signal.is_property_signal() == False
        assert court_signal.is_legal_signal() == True

        # Personal event signals
        birthday_signal = AssetSignal(
            asset_id="person_123",
            signal_type=SignalType.BIRTHDAY,
            signal_source=SignalSource.WEB_SCRAPING,
            signal_value={}
        )
        assert birthday_signal.is_property_signal() == False
        assert birthday_signal.is_legal_signal() == False
        assert birthday_signal.is_personal_event_signal() == True

    def test_signal_with_property_fields(self):
        """Test signal with comprehensive property information."""
        deed_signal = AssetSignal(
            asset_id="property_456",
            asset_type=AssetType.SINGLE_FAMILY_HOME,
            signal_type=SignalType.DEED,
            signal_source=SignalSource.PUBLIC_RECORDS,
            signal_value={"transfer_date": "2024-03-15", "sale_price": 350000},
            property_address="789 Elm Street",
            property_city="Riverside",
            property_county="Cook",
            property_state="IL",
            property_zip="60546",
            parcel_id="COOK-123456",
            property_value=350000,
            transaction_amount=350000,
            plaintiff_name="Seller LLC",
            defendant_name="Buyer Family"
        )

        # Test property address retrieval
        assert deed_signal.get_property_address() == "789 Elm Street"

        # Test financial impact
        assert deed_signal.get_financial_impact() == 350000

        # Test legal parties
        parties = deed_signal.get_legal_parties()
        assert parties["plaintiff"] == "Seller LLC"
        assert parties["defendant"] == "Buyer Family"

        # Test property intelligence
        intelligence = deed_signal.get_property_intelligence()
        assert intelligence["property_address"] == "789 Elm Street"
        assert intelligence["parcel_id"] == "COOK-123456"
        assert intelligence["property_value"] == 350000
        assert intelligence["ownership_change"] == True
        assert intelligence["due_diligence_priority"] == "critical"

    def test_signal_event_fields(self):
        """Test signal with event-specific fields."""
        wedding_signal = AssetSignal(
            asset_id="person_789",
            asset_type=AssetType.PERSON,
            signal_type=SignalType.WEDDING,
            signal_source=SignalSource.WEB_SCRAPING,
            signal_value={"ceremony_date": "2024-08-20", "reception_venue": "Grand Hotel"},
            spouse_name="Sarah Johnson",
            event_date=datetime(2024, 8, 20),
            venue_name="Grand Hotel",
            event_description="Beach wedding ceremony followed by reception"
        )

        assert wedding_signal.spouse_name == "Sarah Johnson"
        assert wedding_signal.event_date.year == 2024
        assert wedding_signal.event_date.month == 8
        assert wedding_signal.venue_name == "Grand Hotel"
        assert "Beach wedding" in wedding_signal.event_description

    def test_signal_tax_fields(self):
        """Test signal with tax-specific fields."""
        tax_signal = AssetSignal(
            asset_id="property_101",
            signal_type=SignalType.TAX_ISSUE,
            signal_source=SignalSource.PUBLIC_RECORDS,
            signal_value={"delinquent_amount": 3200, "tax_year": 2023, "due_date": "2024-03-15"},
            tax_year=2023,
            tax_amount=3200,
            tax_type="property",
            filing_date=datetime(2024, 3, 15)
        )

        assert tax_signal.tax_year == 2023
        assert tax_signal.tax_amount == 3200
        assert tax_signal.tax_type == "property"
        assert tax_signal.filing_date.year == 2024

        # Test financial impact
        assert tax_signal.get_financial_impact() == 3200


if __name__ == "__main__":
    # Run basic tests
    print("🛰️ Testing Asset Signal Model...")

    test_instance = TestAssetSignal()

    # Run individual tests
    try:
        test_instance.test_signal_creation()
        print("✅ Signal creation tests passed")

        test_instance.test_signal_validation()
        print("✅ Signal validation tests passed")

        test_instance.test_signal_quality_assessment()
        print("✅ Signal quality assessment tests passed")

        test_instance.test_validation_utilities()
        print("✅ Validation utilities tests passed")

        test_instance.test_signal_creation_helpers()
        print("✅ Signal creation helpers tests passed")

        test_instance.test_signal_audit_trail()
        print("✅ Signal audit trail tests passed")

        test_instance.test_signal_summary_and_serialization()
        print("✅ Signal summary and serialization tests passed")

        test_instance.test_signal_fusion()
        print("✅ Signal fusion tests passed")

        test_instance.test_signal_deduplication()
        print("✅ Signal deduplication tests passed")

        test_instance.test_compliance_and_privacy()
        print("✅ Compliance and privacy tests passed")

        test_instance.test_signal_lifecycle()
        print("✅ Signal lifecycle tests passed")

        test_instance.test_signal_metadata_and_tags()
        print("✅ Signal metadata and tags tests passed")

        test_instance.test_signal_business_intelligence()
        print("✅ Signal business intelligence tests passed")

        test_instance.test_signal_validation_errors()
        print("✅ Signal validation error tests passed")

        test_instance.test_asset_model()
        print("✅ Asset model tests passed")

        test_instance.test_signal_request_model()
        print("✅ Signal request model tests passed")

        test_instance.test_real_estate_signal_creation()
        print("✅ Real estate signal creation tests passed")

        test_instance.test_legal_signal_creation()
        print("✅ Legal signal creation tests passed")

        test_instance.test_personal_event_signal_creation()
        print("✅ Personal event signal creation tests passed")

        test_instance.test_signal_property_intelligence()
        print("✅ Signal property intelligence tests passed")

        test_instance.test_signal_risk_assessment()
        print("✅ Signal risk assessment tests passed")

        test_instance.test_signal_compliance_requirements()
        print("✅ Signal compliance requirements tests passed")

        test_instance.test_signal_legal_parties()
        print("✅ Signal legal parties tests passed")

        test_instance.test_signal_signal_type_classification()
        print("✅ Signal type classification tests passed")

        test_instance.test_signal_with_property_fields()
        print("✅ Signal with property fields tests passed")

        test_instance.test_signal_event_fields()
        print("✅ Signal event fields tests passed")

        test_instance.test_signal_tax_fields()
        print("✅ Signal tax fields tests passed")

    print("\n🎉 All Asset Signal tests completed successfully!")
    print("🏆 Enterprise-grade signal intelligence fully validated!")
    print("🏠 Real estate and legal signal processing fully tested!")

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
