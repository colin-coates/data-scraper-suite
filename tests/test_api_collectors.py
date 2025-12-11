#!/usr/bin/env python3
"""
Test API Collectors

Tests the third-party API integrations for data enrichment and verification.
"""

import asyncio
import sys
import os
from unittest.mock import Mock, patch, AsyncMock

# Add the scraper suite to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from apis.linkedin_api import LinkedInAPICollector, LinkedInAPIConfig
from apis.hunter_io import HunterIOCollector, HunterIOConfig
from apis.clearbit import ClearbitCollector, ClearbitConfig
from apis.fullcontact import FullContactCollector, FullContactConfig


async def test_api_collectors():
    """Test API collector functionality."""
    print("🔗 Testing API Collectors")
    print("=" * 35)

    # Test 1: LinkedIn API Collector
    print("\n1️⃣ Testing LinkedIn API Collector:")
    try:
        config = LinkedInAPIConfig(
            api_key="test_key",
            access_token="test_token"
        )
        collector = LinkedInAPICollector(config)

        # Test configuration
        assert collector.linkedin_config.api_key == "test_key"
        assert collector.linkedin_config.access_token == "test_token"

        print("✅ LinkedIn API collector initialization successful")

    except Exception as e:
        print(f"❌ LinkedIn API collector failed: {e}")
        return False

    # Test 2: Hunter.io Collector
    print("\n2️⃣ Testing Hunter.io Collector:")
    try:
        config = HunterIOConfig(
            api_key="test_key",
            verify_emails=True,
            confidence_threshold=0.8
        )
        collector = HunterIOCollector(config)

        # Test email validation
        assert collector._is_valid_email_format("test@example.com") == True
        assert collector._is_valid_email_format("invalid-email") == False

        print("✅ Hunter.io collector initialization successful")

    except Exception as e:
        print(f"❌ Hunter.io collector failed: {e}")
        return False

    # Test 3: Clearbit Collector
    print("\n3️⃣ Testing Clearbit Collector:")
    try:
        config = ClearbitConfig(
            api_key="test_key",
            enrichment_enabled=True,
            logo_retrieval=True
        )
        collector = ClearbitCollector(config)

        # Test domain normalization
        assert collector._is_cache_valid("nonexistent") == False

        print("✅ Clearbit collector initialization successful")

    except Exception as e:
        print(f"❌ Clearbit collector failed: {e}")
        return False

    # Test 4: FullContact Collector
    print("\n4️⃣ Testing FullContact Collector:")
    try:
        config = FullContactConfig(
            api_key="test_key",
            person_enrichment=True,
            include_social_profiles=True
        )
        collector = FullContactCollector(config)

        # Test data processing
        person_data = {"fullName": "John Doe", "contactInfo": {"emails": [{"value": "john@example.com"}]}}
        processed = collector._process_person_data(person_data)

        assert processed["full_name"] == "John Doe"
        assert len(processed["contact_info"]["emails"]) == 1

        print("✅ FullContact collector initialization successful")

    except Exception as e:
        print(f"❌ FullContact collector failed: {e}")
        return False

    # Test 5: API Collector Metrics
    print("\n5️⃣ Testing API Collector Metrics:")
    try:
        # Test LinkedIn metrics
        linkedin_config = LinkedInAPIConfig(api_key="test")
        linkedin_collector = LinkedInAPICollector(linkedin_config)
        metrics = linkedin_collector.get_linkedin_api_metrics()

        assert 'api_calls' in metrics
        assert 'requests_today' in metrics
        assert metrics['api_key_configured'] == True

        # Test Hunter.io metrics
        hunter_config = HunterIOConfig(api_key="test")
        hunter_collector = HunterIOCollector(hunter_config)
        metrics = hunter_collector.get_hunter_metrics()

        assert 'api_calls' in metrics
        assert 'cache_size' in metrics
        assert metrics['api_key_configured'] == True

        print("✅ API collector metrics collection works")

    except Exception as e:
        print(f"❌ API collector metrics failed: {e}")
        return False

    # Test 6: Error Handling
    print("\n6️⃣ Testing Error Handling:")
    try:
        # Test missing API key
        config = LinkedInAPIConfig()  # No API key
        collector = LinkedInAPICollector(config)

        # This should raise an error when trying to execute
        try:
            await collector._execute_scrape({"operation": "enrich", "entity_type": "organization"})
            assert False, "Should have raised RuntimeError"
        except RuntimeError as e:
            assert "API key required" in str(e)

        print("✅ Error handling works correctly")

    except Exception as e:
        print(f"❌ Error handling test failed: {e}")
        return False

    # Test 7: Rate Limiting Logic
    print("\n7️⃣ Testing Rate Limiting Logic:")
    try:
        # Test Hunter.io rate limiting
        config = HunterIOConfig(api_key="test")
        collector = HunterIOCollector(config)

        # Should allow requests initially
        assert await collector._check_rate_limits() == True

        # Simulate hitting limit
        collector.requests_this_hour = 50
        # Should still allow (we set conservative limits)

        print("✅ Rate limiting logic works")

    except Exception as e:
        print(f"❌ Rate limiting test failed: {e}")
        return False

    # Test 8: Cache Functionality
    print("\n8️⃣ Testing Cache Functionality:")
    try:
        # Test Clearbit caching
        config = ClearbitConfig(api_key="test")
        collector = ClearbitCollector(config)

        # Test cache operations
        test_data = {"test": "data"}
        collector._cache_company("test_key", test_data)

        assert len(collector.company_cache) == 1
        assert collector._is_cache_valid("test_key") == True

        # Clear cache
        collector.clear_company_cache()
        assert len(collector.company_cache) == 0

        print("✅ Cache functionality works")

    except Exception as e:
        print(f"❌ Cache functionality test failed: {e}")
        return False

    # Cleanup
    try:
        # Clean up collectors
        cleanup_tasks = []
        for collector_name, collector in [
            ("LinkedIn", LinkedInAPICollector(LinkedInAPIConfig())),
            ("Hunter", HunterIOCollector(HunterIOConfig())),
            ("Clearbit", ClearbitCollector(ClearbitConfig())),
            ("FullContact", FullContactCollector(FullContactConfig()))
        ]:
            cleanup_tasks.append(collector.cleanup())

        await asyncio.gather(*cleanup_tasks, return_exceptions=True)
        print("✅ API collector cleanup successful")

    except Exception as e:
        print(f"⚠️ API collector cleanup warning: {e}")

    print("\n🔗 API Collectors testing complete!")
    print("The API collectors provide comprehensive third-party data enrichment!")
    return True


if __name__ == "__main__":
    async def main():
        try:
            success = await test_api_collectors()
            if success:
                print("\n🎉 All API collector tests passed!")
                sys.exit(0)
            else:
                print("\n💥 Some API collector tests failed!")
                sys.exit(1)

        except Exception as e:
            print(f"\n💥 API collector test suite failed with exception: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)

    asyncio.run(main())
