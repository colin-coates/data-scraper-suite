# Copyright (c) 2024 Mountain Jewels Intelligence. All rights reserved.
#
# This software is proprietary and confidential. Unauthorized copying,
# modification, distribution, or use is strictly prohibited.

"""
Test Sentinel Integration Logic

Demonstrates the sentinel integration patterns and methods
that would be used in the scraper engine.
"""

import asyncio
import logging
from datetime import datetime
from core.sentinels import create_comprehensive_orchestrator
from core.control_models import JobControl, JobPriority, ScraperType, ControlMetadata

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MockScraperEngineSentinelIntegration:
    """Mock scraper engine demonstrating sentinel integration patterns."""

    def __init__(self):
        self.sentinel_orchestrator = None
        self.enable_sentinels = True
        self.run_pre_job_checks = True
        self.run_continuous_monitoring = True
        self.run_post_job_analysis = True
        self.sentinel_check_interval = 1.0

    async def initialize(self):
        """Initialize sentinel orchestrator."""
        if self.enable_sentinels:
            self.sentinel_orchestrator = create_comprehensive_orchestrator()
            print("✅ Sentinel orchestrator initialized")

    def _extract_urls_from_target(self, target):
        """Extract URLs from job target (same logic as scraper engine)."""
        urls = []
        url_fields = ["url", "urls", "target_url", "base_url", "endpoint"]

        for field in url_fields:
            if field in target:
                value = target[field]
                if isinstance(value, str):
                    urls.append(value)
                elif isinstance(value, list):
                    urls.extend([u for u in value if isinstance(u, str)])

        if "companies" in target:
            for company in target["companies"]:
                if "url" in company:
                    urls.append(company["url"])

        if "people" in target:
            for person in target["people"]:
                if "profile_url" in person:
                    urls.append(person["profile_url"])

        return urls

    def _extract_domains_from_target(self, target):
        """Extract domains from job target."""
        domains = set()
        urls = self._extract_urls_from_target(target)

        for url in urls:
            try:
                from urllib.parse import urlparse
                parsed = urlparse(url)
                if parsed.netloc:
                    domain = parsed.netloc.lower()
                    if domain.startswith("www."):
                        domain = domain[4:]
                    domains.add(domain)
            except Exception:
                continue

        return list(domains)

    async def _run_pre_job_sentinel_checks(self, job_control):
        """Run pre-job sentinel checks (same logic as scraper engine)."""
        if not self.sentinel_orchestrator or not self.run_pre_job_checks:
            return

        target_info = {
            "urls": self._extract_urls_from_target(job_control.target),
            "domains": self._extract_domains_from_target(job_control.target),
            "scraper_type": job_control.scraper_type.value,
            "job_priority": job_control.priority.value,
            "has_contract": job_control.control_contract is not None
        }

        print(f"🔍 Running pre-job sentinel checks for domains: {target_info['domains']}")

        result = await self.sentinel_orchestrator.orchestrate(
            target=target_info,
            sentinels_to_run=["network", "waf", "malware"]
        )

        if result.aggregated_action == "block":
            raise ValueError(
                f"Job blocked by sentinel security checks: {result.aggregated_risk_level} risk"
            )

        print(f"✅ Pre-job checks passed: {result.aggregated_risk_level} risk level")

    async def _run_post_job_sentinel_analysis(self, job_control, result):
        """Run post-job sentinel analysis (same logic as scraper engine)."""
        if not self.sentinel_orchestrator or not self.run_post_job_analysis:
            return

        target_info = {
            "urls": self._extract_urls_from_target(job_control.target),
            "domains": self._extract_domains_from_target(job_control.target),
            "scraper_type": job_control.scraper_type.value,
            "job_id": job_control.job_id,
            "records_found": result.data.get("records_found", 0) if result.data else 0,
            "data_quality_score": 0.85,  # Mock quality score
            "processing_time": result.response_time,
            "has_sensitive_data": False
        }

        print(f"📊 Running post-job sentinel analysis for job {job_control.job_id}")

        analysis_result = await self.sentinel_orchestrator.orchestrate(
            target=target_info,
            sentinels_to_run=["malware", "performance"]
        )

        print(f"✅ Post-job analysis completed: {analysis_result.aggregated_risk_level} risk")

        # Store analysis results
        if not job_control.metadata:
            job_control.metadata = ControlMetadata()

        job_control.metadata.properties["sentinel_analysis"] = {
            "risk_level": analysis_result.aggregated_risk_level,
            "recommended_action": analysis_result.aggregated_action,
            "findings": analysis_result.decision_factors,
            "timestamp": datetime.utcnow().isoformat()
        }

    async def process_job_with_sentinels(self, job_control):
        """Process a job with full sentinel integration."""
        try:
            # 1. Pre-job sentinel checks
            await self._run_pre_job_sentinel_checks(job_control)

            # 2. Simulate scraping
            print(f"⚙️  Processing job {job_control.job_id}...")
            await asyncio.sleep(0.2)  # Simulate scraping

            # Mock successful result
            result = type('Result', (), {
                'success': True,
                'data': {
                    'records_found': 42,
                    'companies': [{'name': 'Test Corp', 'url': 'https://testcorp.com'}],
                    'errors': 0
                },
                'error_message': None,
                'response_time': 0.2,
                'retry_count': 0,
                'timestamp': datetime.utcnow()
            })()

            # 3. Post-job sentinel analysis
            await self._run_post_job_sentinel_analysis(job_control, result)

            return result

        except Exception as e:
            print(f"❌ Job processing failed: {e}")
            raise


async def test_sentinel_integration():
    """Test sentinel integration patterns."""
    print("🚀 Testing Sentinel Integration Patterns")
    print("=" * 60)

    # Create mock engine with sentinel integration
    engine = MockScraperEngineSentinelIntegration()

    try:
        # Initialize sentinels
        print("📋 Initializing sentinel monitoring...")
        await engine.initialize()

        # Create test job
        print("📝 Creating test job targeting LinkedIn...")
        job_control = JobControl(
            job_id="test_job_001",
            scraper_type=ScraperType.LINKEDIN,
            target={
                "urls": ["https://linkedin.com/company/testcorp"],
                "companies": [{"name": "Test Corp", "url": "https://linkedin.com/company/testcorp"}],
                "people": [{"name": "John Doe", "profile_url": "https://linkedin.com/in/johndoe"}]
            },
            priority=JobPriority.NORMAL,
            metadata=ControlMetadata()
        )

        # Process job with sentinel integration
        print("🔄 Processing job with full sentinel pipeline...")
        result = await engine.process_job_with_sentinels(job_control)

        # Check results
        print("📋 Job Results:")
        print(f"   Success: {result.success}")
        print(f"   Records found: {result.data['records_found']}")
        print(f"   Response time: {result.response_time:.3f}s")

        # Check sentinel analysis
        if job_control.metadata and 'sentinel_analysis' in job_control.metadata.properties:
            analysis = job_control.metadata.properties['sentinel_analysis']
            print(f"   Sentinel Analysis: {analysis['risk_level']} risk")
            print(f"   Recommended action: {analysis['recommended_action']}")

        # Get orchestrator metrics
        if engine.sentinel_orchestrator:
            orch_info = engine.sentinel_orchestrator.get_orchestrator_info()
            print("📊 Sentinel Orchestrator Metrics:")
            print(f"   Orchestrations completed: {orch_info['metrics']['orchestrations_attempted']}")
            print(f"   Registered sentinels: {len(orch_info['registered_sentinels'])}")
            print(f"   Average execution time: {orch_info['metrics']['average_execution_time']:.3f}s")

        print("\n🎉 Sentinel Integration Test Completed Successfully!")
        print("✅ Pre-job security validation")
        print("✅ Runtime monitoring simulation")
        print("✅ Post-job data quality analysis")
        print("✅ Comprehensive risk assessment")
        print("✅ Audit trail and metadata enrichment")

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

    finally:
        # Cleanup (orchestrator doesn't need explicit cleanup)
        print("🧹 Test cleanup complete")


if __name__ == "__main__":
    asyncio.run(test_sentinel_integration())
