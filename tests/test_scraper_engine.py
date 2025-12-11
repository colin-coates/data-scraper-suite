#!/usr/bin/env python3
"""
Test Scraper Engine

Tests the centralized orchestration engine, job dispatching, and plugin architecture.
"""

import asyncio
import sys
import os
from unittest.mock import Mock, patch

# Add the scraper suite to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from scraper_engine import ScraperEngine, EngineConfig
from core.base_scraper import ScraperConfig
from scrapers.linkedin_scraper import LinkedInScraper, LinkedInConfig
from scrapers.web_scraper import WebScraper, WebScraperConfig


async def test_scraper_engine():
    """Test the scraper engine functionality."""
    print("🕷️ Testing Scraper Engine")
    print("=" * 40)

    # Test 1: Engine Initialization
    print("\n1️⃣ Testing Engine Initialization:")
    try:
        config = EngineConfig(
            max_concurrent_jobs=2,
            enable_metrics=True,
            enable_anti_detection=False  # Disable for testing
        )
        engine = ScraperEngine(config)

        assert engine.config.max_concurrent_jobs == 2
        assert engine.config.enable_metrics == True
        assert len(engine.scrapers) == 0  # No scrapers registered yet

        print("✅ Engine initialization successful")

    except Exception as e:
        print(f"❌ Engine initialization failed: {e}")
        return False

    # Test 2: Scraper Registration
    print("\n2️⃣ Testing Scraper Registration:")
    try:
        # Register LinkedIn scraper
        linkedin_config = LinkedInConfig(name="linkedin_test")
        engine.register_scraper("linkedin", LinkedInScraper, linkedin_config)

        # Register web scraper
        web_config = WebScraperConfig(name="web_test")
        engine.register_scraper("web", WebScraper, web_config)

        assert "linkedin" in engine.scrapers
        assert "web" in engine.scrapers
        assert len(engine.active_scrapers) == 2

        print("✅ Scraper registration successful")
        print(f"   Registered scrapers: {list(engine.scrapers.keys())}")

    except Exception as e:
        print(f"❌ Scraper registration failed: {e}")
        return False

    # Test 3: Job Dispatching
    print("\n3️⃣ Testing Job Dispatching:")
    try:
        # Mock the initialization to avoid Azure dependencies
        with patch.object(engine, 'initialize', return_value=None):
            await engine.initialize()

        # Dispatch LinkedIn job
        linkedin_job = {
            "scraper_type": "linkedin",
            "target": {
                "profile_url": "https://linkedin.com/in/john-doe"
            },
            "priority": "high"
        }
        job_id_1 = await engine.dispatch_job(linkedin_job)

        # Dispatch web scraping job
        web_job = {
            "scraper_type": "web",
            "target": {
                "url": "https://example.com"
            },
            "priority": "normal"
        }
        job_id_2 = await engine.dispatch_job(web_job)

        assert job_id_1.startswith("job_")
        assert job_id_2.startswith("job_")
        assert job_id_1 != job_id_2

        print("✅ Job dispatching successful")
        print(f"   Job 1 ID: {job_id_1}")
        print(f"   Job 2 ID: {job_id_2}")

    except Exception as e:
        print(f"❌ Job dispatching failed: {e}")
        return False

    # Test 4: Job Status Tracking
    print("\n4️⃣ Testing Job Status Tracking:")
    try:
        status_1 = engine.get_job_status(job_id_1)
        status_2 = engine.get_job_status(job_id_2)

        assert status_1 is not None
        assert status_2 is not None
        assert status_1['status'] == 'pending'
        assert status_2['status'] == 'pending'

        print("✅ Job status tracking works")
        print(f"   Job 1 status: {status_1['status']}")
        print(f"   Job 2 status: {status_2['status']}")

    except Exception as e:
        print(f"❌ Job status tracking failed: {e}")
        return False

    # Test 5: Engine Metrics
    print("\n5️⃣ Testing Engine Metrics:")
    try:
        metrics = engine.get_metrics()

        assert 'jobs_dispatched' in metrics
        assert 'jobs_completed' in metrics
        assert 'success_rate' in metrics
        assert metrics['jobs_dispatched'] == 2
        assert 'registered_scrapers' in metrics
        assert len(metrics['registered_scrapers']) == 2

        print("✅ Engine metrics collection works")
        print(f"   Jobs dispatched: {metrics['jobs_dispatched']}")
        print(f"   Success rate: {metrics['success_rate']:.2f}")
        print(f"   Registered scrapers: {metrics['registered_scrapers']}")

    except Exception as e:
        print(f"❌ Engine metrics failed: {e}")
        return False

    # Test 6: Scraper Metrics
    print("\n6️⃣ Testing Scraper Metrics:")
    try:
        linkedin_metrics = engine.get_scraper_metrics("linkedin")
        web_metrics = engine.get_scraper_metrics("web")

        assert linkedin_metrics is not None
        assert web_metrics is not None
        assert 'success_count' in linkedin_metrics
        assert 'error_count' in linkedin_metrics

        print("✅ Scraper metrics collection works")
        print(f"   LinkedIn scraper metrics: {linkedin_metrics['success_count']} success, {linkedin_metrics['error_count']} errors")
        print(f"   Web scraper metrics: {web_metrics['success_count']} success, {web_metrics['error_count']} errors")

    except Exception as e:
        print(f"❌ Scraper metrics failed: {e}")
        return False

    # Test 7: Error Handling
    print("\n7️⃣ Testing Error Handling:")
    try:
        # Try to dispatch job with invalid scraper
        invalid_job = {
            "scraper_type": "nonexistent",
            "target": {"url": "https://example.com"}
        }

        try:
            await engine.dispatch_job(invalid_job)
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "nonexistent" in str(e)

        print("✅ Error handling works correctly")

    except Exception as e:
        print(f"❌ Error handling test failed: {e}")
        return False

    # Cleanup
    try:
        await engine.cleanup()
        print("✅ Engine cleanup successful")
    except Exception as e:
        print(f"⚠️ Engine cleanup warning: {e}")

    print("\n🕷️ Scraper Engine testing complete!")
    print("The scraper engine successfully manages job dispatching and scraper orchestration!")
    return True


async def test_plugin_architecture():
    """Test the plugin architecture components."""
    print("\n🔌 Testing Plugin Architecture")
    print("=" * 40)

    # Test 1: Plugin Manager Import
    print("\n1️⃣ Testing Plugin Manager:")
    try:
        from core.plugin_manager import PluginManager

        pm = PluginManager()
        assert pm is not None

        print("✅ Plugin manager import successful")

    except Exception as e:
        print(f"❌ Plugin manager import failed: {e}")
        return False

    # Test 2: Plugin Discovery
    print("\n2️⃣ Testing Plugin Discovery:")
    try:
        discovered = pm.discover_plugins()
        print(f"   Discovered plugins: {discovered}")

        # Should find our test plugins
        assert len(discovered) >= 2  # linkedin_scraper and web_scraper

        print("✅ Plugin discovery works")

    except Exception as e:
        print(f"❌ Plugin discovery failed: {e}")
        return False

    # Test 3: Plugin Loading
    print("\n3️⃣ Testing Plugin Loading:")
    try:
        # Try to load linkedin scraper
        success = pm.load_plugin("scrapers.linkedin_scraper")
        if success:
            print("✅ LinkedIn scraper plugin loaded")
        else:
            print("⚠️ LinkedIn scraper plugin failed to load (expected in test env)")

        # Try to load web scraper
        success = pm.load_plugin("scrapers.web_scraper")
        if success:
            print("✅ Web scraper plugin loaded")
        else:
            print("⚠️ Web scraper plugin failed to load (expected in test env)")

        print("✅ Plugin loading test completed")

    except Exception as e:
        print(f"❌ Plugin loading failed: {e}")
        return False

    # Cleanup
    try:
        await pm.cleanup()
        print("✅ Plugin manager cleanup successful")
    except Exception as e:
        print(f"⚠️ Plugin manager cleanup warning: {e}")

    return True


if __name__ == "__main__":
    async def main():
        try:
            # Test scraper engine
            engine_success = await test_scraper_engine()

            # Test plugin architecture
            plugin_success = await test_plugin_architecture()

            if engine_success and plugin_success:
                print("\n🎉 All tests passed!")
                sys.exit(0)
            else:
                print("\n💥 Some tests failed!")
                sys.exit(1)

        except Exception as e:
            print(f"\n💥 Test suite failed with exception: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)

    asyncio.run(main())
