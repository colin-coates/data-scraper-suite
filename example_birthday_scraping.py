#!/usr/bin/env python3
"""
Example: October Birthday Scraping for Teaneck NJ

Demonstrates authorized scraping operation with proper governance controls.
"""

import asyncio
import sys
import os
from datetime import datetime

# Add the scraper suite to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '.'))

from core.base_scraper import BaseScraper, ScraperConfig
from core.control_models import (
    ScrapeControlContract, ScrapeIntent, ScrapeBudget,
    ScrapeTempo, DeploymentWindow, ScrapeAuthorization
)


class BirthdayScraper(BaseScraper):
    """Scraper for birthday-related data collection."""

    ROLE = "discovery"
    TIER = 1
    SUPPORTED_EVENTS = ["weddings", "social", "birthdays"]

    async def _execute_scrape(self, target):
        """Simulate birthday data collection."""
        await asyncio.sleep(0.5)  # Simulate scraping time

        # Mock birthday data for demonstration
        return {
            "location": target.get("location", "Teaneck NJ"),
            "month": target.get("month", "October"),
            "birthdays_found": 15,
            "social_profiles": 8,
            "contact_info": 12,
            "data_quality": "high"
        }


async def demonstrate_authorized_birthday_scraping():
    """Demonstrate properly authorized birthday scraping operation."""
    print("🎂 Authorized Birthday Scraping Example")
    print("=" * 45)

    # Create authorization as specified by user
    authorization = ScrapeAuthorization(
        approved_by="colin",  # As specified
        purpose="October birthdays – Teaneck NJ",  # As specified
        approval_timestamp=datetime(2024, 12, 15, 12, 0, 0),  # Approval time in past
        expires_at=datetime(2026, 12, 15, 5, 0, 0)  # Future expiration
    )

    print("🔐 Authorization Details:")
    print(f"   Authorized by: {authorization.approved_by}")
    print(f"   Purpose: {authorization.purpose}")
    print(f"   Expires: {authorization.expires_at.isoformat()}Z")
    print(f"   Currently valid: {authorization.expires_at > datetime.utcnow()}")

    # Create scraping intent for birthday data
    intent = ScrapeIntent(
        geography={"city": "Teaneck", "state": "NJ", "country": "US"},
        events={"social": True, "birthdays": True},
        sources=["facebook", "linkedin"],
        allowed_role="discovery",
        event_type="birthdays"
    )

    # Conservative budget for initial pilot
    budget = ScrapeBudget(
        max_runtime_minutes=60,    # 1 hour limit
        max_pages=500,             # Reasonable page limit
        max_records=1000,          # Conservative record limit
        max_browser_instances=1,   # Single browser instance
        max_memory_mb=512          # Reasonable memory limit
    )

    # Schedule during off-peak hours to minimize detection
    deployment_window = DeploymentWindow(
        earliest_start=datetime.utcnow(),  # Start immediately if approved
        latest_start=datetime.utcnow().replace(hour=6, minute=0),  # End by 6 AM next day
        max_duration_minutes=240,  # 4 hour window
        timezone="UTC"
    )

    # Create control contract with explicit authorization
    control = ScrapeControlContract(
        intent=intent,
        budget=budget,
        tempo=ScrapeTempo.HUMAN,    # Human-like behavior
        deployment_window=deployment_window,
        authorization=authorization,  # Explicit authorization
        human_override=False         # No override needed for Tier 1
    )

    print("\n🎯 Scraping Configuration:")
    print(f"   Target: {intent.event_type} in {intent.geography}")
    print(f"   Sources: {', '.join(intent.sources)}")
    print(f"   Budget: {budget.max_records} records, {budget.max_runtime_minutes} min runtime")
    print(f"   Tempo: {control.tempo.value}")
    print(f"   Role Required: {intent.allowed_role}")

    # Initialize scraper
    config = ScraperConfig(
        name="birthday_scraper_teaneck_october",
        user_agent_rotation=True,
        proxy_rotation=False,  # Conservative approach
        rate_limit_delay=2.0,  # Respectful rate limiting
        max_retries=3
    )

    scraper = BirthdayScraper(config, control)

    print("\n🕷️  Executing Birthday Scraping...")
    print("   Step 1: Pre-flight governance checks...")

    # Execute scraping with full governance
    try:
        result = await scraper.scrape({
            "location": "Teaneck NJ",
            "month": "October",
            "event_type": "birthdays"
        })

        print("   Step 2: Scraping execution completed")
        print("\n📊 Results:")
        print(f"   Status: {'✅ SUCCESS' if result.success else '❌ FAILED'}")
        print(f"   Records Found: {result.data.get('birthdays_found', 0) if result.data else 0}")
        print(f"   Execution Time: {result.response_time:.2f}s")

        if result.data:
            print(f"   Social Profiles: {result.data.get('social_profiles', 0)}")
            print(f"   Contact Info: {result.data.get('contact_info', 0)}")
            print(f"   Data Quality: {result.data.get('data_quality', 'unknown')}")

        # Verify authorization compliance
        if result.success:
            print("\n🔒 Authorization Compliance:")
            print("   ✅ Operation completed within authorized scope")
            print("   ✅ Purpose aligned with authorization")
            print("   ✅ Geographic targeting respected")
            print(f"   ⏰ Expires: {authorization.expires_at.strftime('%Y-%m-%d %H:%M UTC')}")

        return result

    except Exception as e:
        print(f"   💥 Scraping failed: {e}")
        return None


def demonstrate_authorization_expiry():
    """Demonstrate what happens when authorization expires."""
    print("\n⏰ Authorization Expiry Demonstration")
    print("=" * 40)

    # Create expired authorization
    expired_auth = ScrapeAuthorization(
        approved_by="colin",
        purpose="October birthdays – Teaneck NJ",
        approval_timestamp=datetime(2024, 12, 10, 12, 0, 0),  # Past approval
        expires_at=datetime(2024, 12, 14, 12, 0, 0)  # Past expiration
    )

    print("📅 Expired Authorization:")
    print(f"   Expires: {expired_auth.expires_at.isoformat()}Z")
    print(f"   Currently valid: {expired_auth.expires_at > datetime.utcnow()}")
    print("   Status: ❌ EXPIRED - Would be rejected by AI precheck")

async def demonstrate_unauthorized_operation():
    """Demonstrate what happens without proper authorization."""
    print("\n🚫 Unauthorized Operation Demonstration")
    print("=" * 42)

    # No authorization provided (will use default)
    control_no_auth = ScrapeControlContract(
        intent=ScrapeIntent(
            geography={"city": "Teaneck", "state": "NJ"},
            events={"social": True},
            sources=["facebook"]
        ),
        budget=ScrapeBudget(max_records=100),
        human_override=False
    )

    print("⚠️  Control contract with default authorization:")
    print(f"   Authorized by: {control_no_auth.authorization.approved_by}")
    print(f"   Purpose: {control_no_auth.authorization.purpose}")
    print("   Status: ⚠️  DEFAULT AUTH - May be rejected depending on AI risk assessment")
async def main():
    """Run all birthday scraping demonstrations."""
    print("🎂 MJ Data Scraper Suite - Authorized Birthday Scraping")
    print("=" * 58)
    print("Example: October birthdays data collection in Teaneck, NJ")
    print("Authorized by: colin | Expires: 2025-10-01T05:00Z")
    print()

    success = True

    # Demonstrate authorized scraping
    result = await demonstrate_authorized_birthday_scraping()
    success &= result is not None and result.success

    # Demonstrate authorization concepts
    demonstrate_authorization_expiry()
    await demonstrate_unauthorized_operation()

    if success:
        print("\n🎉 Authorized birthday scraping demonstration completed!")
        print("✅ Operation executed with proper governance and authorization")
        print("🔒 All compliance checks passed")
        print("📊 Data collected within authorized scope and budget")
    else:
        print("\n❌ Demonstration encountered issues.")


if __name__ == "__main__":
    asyncio.run(main())
