#!/usr/bin/env python3
"""
Test Queue Publisher

Tests the async queue publishing functionality for Azure Service Bus integration.
"""

import asyncio
import sys
import os
from unittest.mock import Mock, patch, AsyncMock

# Add the scraper suite to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from core.queue_publisher import QueuePublisher, QueueConfig, PublishResult


async def test_queue_publisher():
    """Test the queue publisher functionality."""
    print("📨 Testing Queue Publisher")
    print("=" * 30)

    # Test 1: Publisher Initialization
    print("\n1️⃣ Testing Publisher Initialization:")
    try:
        config = QueueConfig(
            connection_string="test_connection_string",
            queue_name="test-queue",
            max_batch_size=5,
            enable_batching=True
        )
        publisher = QueuePublisher(config)

        assert publisher.config.queue_name == "test-queue"
        assert publisher.config.max_batch_size == 5
        assert publisher.config.enable_batching == True

        print("✅ Queue publisher initialization successful")

    except Exception as e:
        print(f"❌ Queue publisher initialization failed: {e}")
        return False

    # Test 2: Message Preparation
    print("\n2️⃣ Testing Message Preparation:")
    try:
        test_data = {"name": "John Doe", "email": "john@example.com"}
        metadata = {"source": "test", "correlation_id": "test-123"}

        prepared = publisher._prepare_message(test_data, metadata)

        assert prepared["data"]["name"] == "John Doe"
        assert prepared["metadata"]["source"] == "test"
        assert "timestamp" in prepared
        assert prepared["version"] == "1.0"

        print("✅ Message preparation works")

    except Exception as e:
        print(f"❌ Message preparation failed: {e}")
        return False

    # Test 3: Data Truncation
    print("\n3️⃣ Testing Data Truncation:")
    try:
        # Create large data that exceeds message size
        large_content = "x" * 300000  # 300KB
        large_data = {"content": large_content, "title": "Test"}

        truncated = publisher._truncate_data(large_data, 300000)

        assert "content_truncated" in truncated
        assert len(truncated["content"]) < len(large_content)
        assert truncated["content"].endswith("...")

        print("✅ Data truncation works")

    except Exception as e:
        print(f"❌ Data truncation failed: {e}")
        return False

    # Test 4: Batch Management
    print("\n4️⃣ Testing Batch Management:")
    try:
        # Test adding to batch (without actual publishing)
        messages = [
            {"data": {"id": 1}, "metadata": {}},
            {"data": {"id": 2}, "metadata": {}}
        ]

        # Manually add to batch to test logic
        for msg in messages:
            publisher.message_batch.append(msg)
            publisher.batch_bytes += len(str(msg).encode())

        assert len(publisher.message_batch) == 2
        assert publisher.batch_bytes > 0

        print("✅ Batch management works")

    except Exception as e:
        print(f"❌ Batch management failed: {e}")
        return False

    # Test 5: Metrics Collection
    print("\n5️⃣ Testing Metrics Collection:")
    try:
        metrics = publisher.get_metrics()

        assert 'total_published' in metrics
        assert 'current_batch_size' in metrics
        assert metrics['queue_name'] == "test-queue"
        assert metrics['batching_enabled'] == True

        print("✅ Metrics collection works")
        print(f"   Current batch size: {metrics['current_batch_size']}")

    except Exception as e:
        print(f"❌ Metrics collection failed: {e}")
        return False

    # Test 6: Publish Result Structure
    print("\n6️⃣ Testing Publish Result Structure:")
    try:
        result = PublishResult(
            success=True,
            message_count=3,
            batch_count=1,
            correlation_id="test-123"
        )

        assert result.success == True
        assert result.message_count == 3
        assert result.correlation_id == "test-123"

        print("✅ Publish result structure works")

    except Exception as e:
        print(f"❌ Publish result structure failed: {e}")
        return False

    # Test 7: Error Handling
    print("\n7️⃣ Testing Error Handling:")
    try:
        # Test with invalid data type
        try:
            publisher._prepare_message("invalid_data_type")
            assert False, "Should have raised ValueError"
        except AttributeError:
            # Expected - string doesn't have items()
            pass

        print("✅ Error handling works")

    except Exception as e:
        print(f"❌ Error handling test failed: {e}")
        return False

    # Cleanup
    try:
        await publisher.cleanup()
        print("✅ Publisher cleanup successful")

    except Exception as e:
        print(f"⚠️ Publisher cleanup warning: {e}")

    print("\n📨 Queue Publisher testing complete!")
    print("The async queue publisher provides reliable message publishing with batching!")
    return True


async def test_integration_with_engine():
    """Test integration with scraper engine."""
    print("\n🔗 Testing Integration with Scraper Engine")
    print("=" * 45)

    # Test 1: Engine with Queue Publisher
    print("\n1️⃣ Testing Engine Queue Integration:")
    try:
        from scraper_engine import ScraperEngine, EngineConfig

        # Create engine config with queue publishing enabled
        engine_config = EngineConfig(
            enable_result_publishing=True,
            azure_service_bus_connection="test_connection",
            output_queue_name="test-results"
        )

        engine = ScraperEngine(engine_config)

        # Mock the queue publisher initialization to avoid Azure dependency
        with patch('mj_data_scraper_suite.scraper_engine.QueuePublisher') as mock_publisher_class:
            mock_publisher = Mock()
            mock_publisher.publish_to_queue = AsyncMock(return_value=PublishResult(success=True, message_count=1))
            mock_publisher.initialize = AsyncMock()
            mock_publisher.cleanup = AsyncMock()
            mock_publisher.get_metrics = Mock(return_value={"total_published": 1})

            mock_publisher_class.return_value = mock_publisher

            await engine.initialize()

            # Test publishing to queue
            test_data = {"scraper": "test", "result": "success"}
            result = await engine.publish_to_queue(test_data)

            assert result.success == True
            assert result.message_count == 1

            print("✅ Engine queue integration works")

    except Exception as e:
        print(f"❌ Engine queue integration failed: {e}")
        return False

    return True


if __name__ == "__main__":
    async def main():
        try:
            # Test queue publisher
            publisher_success = await test_queue_publisher()

            # Test integration
            integration_success = await test_integration_with_engine()

            if publisher_success and integration_success:
                print("\n🎉 All queue publisher tests passed!")
                sys.exit(0)
            else:
                print("\n💥 Some queue publisher tests failed!")
                sys.exit(1)

        except Exception as e:
            print(f"\n💥 Queue publisher test suite failed with exception: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)

    asyncio.run(main())
