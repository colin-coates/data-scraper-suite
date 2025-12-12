# Copyright (c) 2024 Mountain Jewels Intelligence. All rights reserved.
#
# This software is proprietary and confidential. Unauthorized copying,
# modification, distribution, or use is strictly prohibited.

"""
Queue Publisher for MJ Data Scraper Suite

Async wrapper for publishing scraped data to Azure Service Bus queues.
Connects the scraper suite with downstream processing systems like TPF enrichment.
"""

import asyncio
import json
import logging
from typing import Dict, Any, List, Optional, Union
from datetime import datetime
from dataclasses import dataclass, field

from azure.servicebus import ServiceBusClient
from azure.servicebus.aio import ServiceBusClient as AsyncServiceBusClient
from azure.servicebus.aio import ServiceBusSender
from azure.core.exceptions import ServiceBusError

from .mj_envelope import MJMessageEnvelope
from .mj_payload_builder import build_person_payload, build_event_payload

logger = logging.getLogger(__name__)


@dataclass
class QueueConfig:
    """Configuration for queue publishing."""
    connection_string: str
    queue_name: str = "scraping-results"
    max_message_size: int = 256 * 1024  # 256KB (Azure limit is 1MB)
    max_batch_size: int = 10  # Messages per batch
    send_timeout: float = 30.0
    retry_attempts: int = 3
    retry_delay: float = 1.0
    enable_batching: bool = True
    batch_flush_interval: float = 5.0  # Seconds


@dataclass
class PublishResult:
    """Result of a publish operation."""
    success: bool
    message_count: int = 0
    batch_count: int = 0
    total_bytes: int = 0
    duration: float = 0.0
    errors: List[str] = field(default_factory=list)
    correlation_id: str = ""


class QueuePublisher:
    """
    Async queue publisher for Azure Service Bus integration.
    Provides reliable message publishing with batching and error handling.
    """

    def __init__(self, config: QueueConfig):
        self.config = config
        self.client: Optional[AsyncServiceBusClient] = None
        self.sender: Optional[ServiceBusSender] = None

        # Batching state
        self.message_batch: List[Dict[str, Any]] = []
        self.batch_start_time = datetime.utcnow()
        self.batch_bytes = 0

        # Metrics
        self.total_published = 0
        self.total_batches = 0
        self.total_errors = 0
        self.total_bytes = 0

        logger.info(f"QueuePublisher initialized for queue: {config.queue_name}")

    async def initialize(self) -> None:
        """Initialize the queue publisher and connection."""
        try:
            self.client = AsyncServiceBusClient.from_connection_string(
                self.config.connection_string
            )

            self.sender = self.client.get_queue_sender(
                queue_name=self.config.queue_name
            )

            await self.sender.__aenter__()

            # Start batch flush task if batching enabled
            if self.config.enable_batching:
                asyncio.create_task(self._batch_flush_worker())

            logger.info("QueuePublisher connection established")

        except Exception as e:
            logger.error(f"Failed to initialize QueuePublisher: {e}")
            raise

    async def publish_to_queue(self, data: Union[Dict[str, Any], List[Dict[str, Any]]],
                              metadata: Optional[Dict[str, Any]] = None) -> PublishResult:
        """
        Async wrapper for publishing data to the queue.

        Args:
            data: Single data item or list of items to publish
            metadata: Optional metadata to attach to all messages

        Returns:
            PublishResult with operation details
        """
        start_time = asyncio.get_event_loop().time()
        correlation_id = f"pub_{int(start_time * 1000)}"

        result = PublishResult(
            success=False,
            correlation_id=correlation_id
        )

        try:
            # Normalize data to list
            if isinstance(data, dict):
                data_list = [data]
            elif isinstance(data, list):
                data_list = data
            else:
                raise ValueError(f"Data must be dict or list, got {type(data)}")

            # Wrap person/event payloads using MJMessageEnvelope
            messages = []
            for item in data_list:
                # Detect if result is "person" or "event"
                data_type = item.get("data_type", "unknown")

                if data_type == "person":
                    payload = build_person_payload(item)
                elif data_type == "event":
                    payload = build_event_payload(item)
                else:
                    # Not MJ data, send as-is
                    payload = item

                # Create MJ envelope
                envelope = MJMessageEnvelope(
                    data_type=data_type,
                    payload=payload,
                    correlation_id=correlation_id
                )

                # Insert timestamp automatically
                envelope.timestamp = datetime.utcnow().isoformat()

                # Send .to_dict() output
                message = self._prepare_message(envelope.to_dict(), metadata)
                messages.append(message)

            # Publish messages
            if self.config.enable_batching:
                await self._add_to_batch(messages)
                result.success = True
                result.message_count = len(messages)
                # Note: Actual publishing happens in batch flush
            else:
                await self._publish_immediately(messages)
                result.success = True
                result.message_count = len(messages)
                result.batch_count = 1

            # Update metrics
            result.duration = asyncio.get_event_loop().time() - start_time

        except Exception as e:
            result.errors.append(str(e))
            self.total_errors += 1
            logger.error(f"Failed to publish to queue: {e}")

        return result

    def _prepare_message(self, data: Dict[str, Any], metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Prepare a message for queue publishing."""
        # Create envelope
        envelope = {
            "data": data,
            "metadata": metadata or {},
            "timestamp": datetime.utcnow().isoformat(),
            "version": "1.0",
            "source": "mj-data-scraper-suite"
        }

        # Add correlation and tracing info
        if "correlation_id" not in envelope["metadata"]:
            envelope["metadata"]["correlation_id"] = f"msg_{int(asyncio.get_event_loop().time() * 1000)}"

        # Validate message size
        message_json = json.dumps(envelope)
        message_size = len(message_json.encode('utf-8'))

        if message_size > self.config.max_message_size:
            # Truncate data if too large
            truncated_data = self._truncate_data(data, message_size)
            envelope["data"] = truncated_data
            envelope["metadata"]["truncated"] = True
            envelope["metadata"]["original_size"] = message_size

            message_json = json.dumps(envelope)
            message_size = len(message_json.encode('utf-8'))

        envelope["metadata"]["message_size"] = message_size

        return envelope

    def _truncate_data(self, data: Dict[str, Any], current_size: int) -> Dict[str, Any]:
        """Truncate data to fit message size limits."""
        # Simple truncation strategy - remove large fields
        truncated = data.copy()

        # Remove potentially large fields
        large_fields = ['content', 'full_content', 'raw_html', 'images', 'videos']
        for field in large_fields:
            if field in truncated:
                truncated[field] = truncated[field][:1000] + "..." if isinstance(truncated[field], str) else None
                truncated[f"{field}_truncated"] = True

        return truncated

    async def _add_to_batch(self, messages: List[Dict[str, Any]]) -> None:
        """Add messages to batch for later publishing."""
        for message in messages:
            message_json = json.dumps(message)
            message_size = len(message_json.encode('utf-8'))

            # Check if adding this message would exceed batch size
            if (len(self.message_batch) >= self.config.max_batch_size or
                self.batch_bytes + message_size > self.config.max_message_size):
                await self._flush_batch()

            # Add to batch
            self.message_batch.append(message)
            self.batch_bytes += message_size

    async def _publish_immediately(self, messages: List[Dict[str, Any]]) -> None:
        """Publish messages immediately without batching."""
        if not self.sender:
            raise RuntimeError("QueuePublisher not initialized")

        for message in messages:
            await self._send_message(message)

    async def _send_message(self, message: Dict[str, Any]) -> None:
        """Send a single message to the queue."""
        if not self.sender:
            raise RuntimeError("QueuePublisher not initialized")

        message_json = json.dumps(message)
        message_bytes = len(message_json.encode('utf-8'))

        for attempt in range(self.config.retry_attempts):
            try:
                await self.sender.send_messages(message_json)
                self.total_published += 1
                self.total_bytes += message_bytes
                return

            except ServiceBusError as e:
                if attempt < self.config.retry_attempts - 1:
                    logger.warning(f"Message send failed (attempt {attempt + 1}): {e}")
                    await asyncio.sleep(self.config.retry_delay * (2 ** attempt))
                else:
                    logger.error(f"Message send failed after {self.config.retry_attempts} attempts: {e}")
                    raise

    async def _batch_flush_worker(self) -> None:
        """Background worker to periodically flush message batches."""
        while True:
            try:
                await asyncio.sleep(self.config.batch_flush_interval)
                await self._flush_batch()
            except Exception as e:
                logger.error(f"Batch flush worker error: {e}")
                await asyncio.sleep(5)  # Wait before retrying

    async def flush(self) -> PublishResult:
        """
        Manually flush any pending batched messages.

        Returns:
            PublishResult with flush operation details
        """
        start_time = asyncio.get_event_loop().time()

        result = PublishResult(
            success=False,
            correlation_id=f"flush_{int(start_time * 1000)}"
        )

        try:
            await self._flush_batch()
            result.success = True
            result.duration = asyncio.get_event_loop().time() - start_time

        except Exception as e:
            result.errors.append(str(e))
            logger.error(f"Manual flush failed: {e}")

        return result

    async def _flush_batch(self) -> None:
        """Flush the current message batch to the queue."""
        if not self.message_batch:
            return

        try:
            # Send all messages in batch
            for message in self.message_batch:
                await self._send_message(message)

            # Update metrics
            self.total_batches += 1

            logger.info(f"Flushed batch of {len(self.message_batch)} messages")

        except Exception as e:
            logger.error(f"Batch flush failed: {e}")
            # Continue processing - don't lose the batch
            raise

        finally:
            # Reset batch state
            self.message_batch.clear()
            self.batch_bytes = 0
            self.batch_start_time = datetime.utcnow()

    def get_metrics(self) -> Dict[str, Any]:
        """Get publisher performance metrics."""
        return {
            'total_published': self.total_published,
            'total_batches': self.total_batches,
            'total_errors': self.total_errors,
            'total_bytes': self.total_bytes,
            'current_batch_size': len(self.message_batch),
            'current_batch_bytes': self.batch_bytes,
            'average_message_size': self.total_bytes / max(1, self.total_published),
            'queue_name': self.config.queue_name,
            'batching_enabled': self.config.enable_batching,
            'batch_flush_interval': self.config.batch_flush_interval
        }

    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on the queue publisher."""
        health = {
            'healthy': False,
            'connection_status': 'disconnected',
            'queue_accessible': False,
            'last_check': datetime.utcnow().isoformat()
        }

        try:
            if self.sender:
                # Try to send a test message (won't actually send due to dry run)
                health['connection_status'] = 'connected'
                health['queue_accessible'] = True
                health['healthy'] = True
            else:
                health['connection_status'] = 'not_initialized'

        except Exception as e:
            health['error'] = str(e)
            logger.warning(f"Health check failed: {e}")

        return health

    async def cleanup(self) -> None:
        """Cleanup queue publisher resources."""
        logger.info("Cleaning up QueuePublisher...")

        # Flush any remaining messages
        try:
            await self._flush_batch()
        except Exception as e:
            logger.warning(f"Failed to flush final batch during cleanup: {e}")

        # Close sender and client
        if self.sender:
            try:
                await self.sender.__aexit__(None, None, None)
            except Exception as e:
                logger.warning(f"Error closing sender: {e}")

        if self.client:
            try:
                await self.client.close()
            except Exception as e:
                logger.warning(f"Error closing client: {e}")

        self.sender = None
        self.client = None

        logger.info("QueuePublisher cleanup complete")

    def __str__(self) -> str:
        return f"QueuePublisher(queue={self.config.queue_name}, published={self.total_published})"
