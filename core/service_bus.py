# Copyright (c) 2024 Mountain Jewels Intelligence. All rights reserved.
#
# This software is proprietary and confidential. Unauthorized copying,
# modification, distribution, or use is strictly prohibited.

"""
Azure Service Bus Integration for MJ Data Scraper Suite

Provides message publishing and consuming for the scraper pipeline:
- Scraper → scraper-work queue → Workers
- Workers → enrichment-tasks queue → Enrichment Workers
- Results → ingestion-events queue → Data Pipeline
"""

import asyncio
import json
import logging
import os
from typing import Dict, Any, Optional, Callable, Awaitable, List
from dataclasses import dataclass, asdict
from datetime import datetime
from enum import Enum

from azure.servicebus.aio import ServiceBusClient, ServiceBusSender, ServiceBusReceiver
from azure.servicebus import ServiceBusMessage, ServiceBusReceivedMessage
from azure.identity.aio import DefaultAzureCredential

logger = logging.getLogger(__name__)


class QueueName(Enum):
    """Available Service Bus queues."""
    SCRAPER_WORK = "scraper-work"
    ENRICHMENT_TASKS = "enrichment-tasks"
    ENRICHMENT_EVENTS = "enrichment-events"
    INGESTION_EVENTS = "ingestion-events"
    MATCHING_TASKS = "matching-tasks"
    KLAVIYO_CAMPAIGNS = "klaviyo-campaigns"
    DEAD_LETTER = "dead-letter-monitor"


class MessageType(Enum):
    """Types of messages in the pipeline."""
    SCRAPE_JOB = "scrape_job"
    SCRAPE_RESULT = "scrape_result"
    ENRICHMENT_REQUEST = "enrichment_request"
    ENRICHMENT_RESULT = "enrichment_result"
    INGESTION_EVENT = "ingestion_event"
    ALERT = "alert"


@dataclass
class PipelineMessage:
    """Standard message format for the pipeline."""
    message_type: str
    job_id: str
    data: Dict[str, Any]
    source: str = "scraper"
    timestamp: str = None
    correlation_id: Optional[str] = None
    priority: int = 1  # 1=low, 2=normal, 3=high, 4=critical
    retry_count: int = 0
    max_retries: int = 3

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.utcnow().isoformat()

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(asdict(self))

    @classmethod
    def from_json(cls, json_str: str) -> "PipelineMessage":
        """Create from JSON string."""
        data = json.loads(json_str)
        return cls(**data)


class ServiceBusManager:
    """
    Manages Azure Service Bus connections for the scraper pipeline.
    
    Flow:
    1. Scraper receives job → processes → publishes result to enrichment-tasks
    2. Enrichment worker receives → enriches → publishes to ingestion-events
    3. Data pipeline receives → stores in Cosmos DB
    """

    def __init__(
        self,
        connection_string: Optional[str] = None,
        namespace: Optional[str] = None,
        use_managed_identity: bool = False
    ):
        """
        Initialize Service Bus manager.

        Args:
            connection_string: Service Bus connection string
            namespace: Service Bus namespace (for managed identity)
            use_managed_identity: Use Azure Managed Identity instead of connection string
        """
        self.connection_string = connection_string or os.getenv("AZURE_SERVICEBUS_CONNECTION_STRING")
        self.namespace = namespace or os.getenv("AZURE_SERVICEBUS_NAMESPACE", "mjsb-dev")
        self.use_managed_identity = use_managed_identity

        self._client: Optional[ServiceBusClient] = None
        self._senders: Dict[str, ServiceBusSender] = {}
        self._receivers: Dict[str, ServiceBusReceiver] = {}
        self._credential: Optional[DefaultAzureCredential] = None

        # Metrics
        self.messages_sent = 0
        self.messages_received = 0
        self.messages_failed = 0

    async def connect(self) -> None:
        """Establish connection to Service Bus."""
        if self._client is not None:
            return

        try:
            if self.use_managed_identity:
                self._credential = DefaultAzureCredential()
                fully_qualified_namespace = f"{self.namespace}.servicebus.windows.net"
                self._client = ServiceBusClient(
                    fully_qualified_namespace=fully_qualified_namespace,
                    credential=self._credential
                )
            else:
                if not self.connection_string:
                    raise ValueError("Service Bus connection string not provided")
                self._client = ServiceBusClient.from_connection_string(self.connection_string)

            logger.info(f"Connected to Service Bus: {self.namespace}")

        except Exception as e:
            logger.error(f"Failed to connect to Service Bus: {e}")
            raise

    async def _get_sender(self, queue_name: str) -> ServiceBusSender:
        """Get or create a sender for a queue."""
        if queue_name not in self._senders:
            await self.connect()
            self._senders[queue_name] = self._client.get_queue_sender(queue_name)
        return self._senders[queue_name]

    async def _get_receiver(self, queue_name: str) -> ServiceBusReceiver:
        """Get or create a receiver for a queue."""
        if queue_name not in self._receivers:
            await self.connect()
            self._receivers[queue_name] = self._client.get_queue_receiver(queue_name)
        return self._receivers[queue_name]

    async def send_message(
        self,
        queue: QueueName,
        message: PipelineMessage,
        session_id: Optional[str] = None
    ) -> bool:
        """
        Send a message to a queue.

        Args:
            queue: Target queue
            message: Message to send
            session_id: Optional session ID for ordered processing

        Returns:
            True if sent successfully
        """
        try:
            sender = await self._get_sender(queue.value)

            sb_message = ServiceBusMessage(
                body=message.to_json(),
                content_type="application/json",
                correlation_id=message.correlation_id,
                subject=message.message_type,
                session_id=session_id
            )

            # Set message priority via application properties
            sb_message.application_properties = {
                "priority": message.priority,
                "source": message.source,
                "job_id": message.job_id
            }

            await sender.send_messages(sb_message)
            self.messages_sent += 1

            logger.debug(f"Sent message to {queue.value}: {message.job_id}")
            return True

        except Exception as e:
            self.messages_failed += 1
            logger.error(f"Failed to send message to {queue.value}: {e}")
            return False

    async def send_batch(
        self,
        queue: QueueName,
        messages: List[PipelineMessage]
    ) -> int:
        """
        Send multiple messages as a batch.

        Args:
            queue: Target queue
            messages: List of messages to send

        Returns:
            Number of messages sent successfully
        """
        try:
            sender = await self._get_sender(queue.value)
            batch = await sender.create_message_batch()

            sent_count = 0
            for message in messages:
                sb_message = ServiceBusMessage(
                    body=message.to_json(),
                    content_type="application/json",
                    subject=message.message_type
                )
                try:
                    batch.add_message(sb_message)
                    sent_count += 1
                except ValueError:
                    # Batch is full, send it and create a new one
                    await sender.send_messages(batch)
                    self.messages_sent += sent_count
                    batch = await sender.create_message_batch()
                    batch.add_message(sb_message)
                    sent_count = 1

            if sent_count > 0:
                await sender.send_messages(batch)
                self.messages_sent += sent_count

            logger.info(f"Sent batch of {sent_count} messages to {queue.value}")
            return sent_count

        except Exception as e:
            logger.error(f"Failed to send batch to {queue.value}: {e}")
            return 0

    async def receive_messages(
        self,
        queue: QueueName,
        max_messages: int = 10,
        max_wait_time: int = 5
    ) -> List[PipelineMessage]:
        """
        Receive messages from a queue.

        Args:
            queue: Source queue
            max_messages: Maximum messages to receive
            max_wait_time: Maximum time to wait (seconds)

        Returns:
            List of received messages
        """
        try:
            receiver = await self._get_receiver(queue.value)
            
            received = await receiver.receive_messages(
                max_message_count=max_messages,
                max_wait_time=max_wait_time
            )

            messages = []
            for msg in received:
                try:
                    pipeline_msg = PipelineMessage.from_json(str(msg))
                    messages.append(pipeline_msg)
                    await receiver.complete_message(msg)
                    self.messages_received += 1
                except Exception as e:
                    logger.error(f"Failed to process message: {e}")
                    await receiver.dead_letter_message(msg, reason=str(e))
                    self.messages_failed += 1

            return messages

        except Exception as e:
            logger.error(f"Failed to receive from {queue.value}: {e}")
            return []

    async def start_consumer(
        self,
        queue: QueueName,
        handler: Callable[[PipelineMessage], Awaitable[bool]],
        max_concurrent: int = 5
    ) -> None:
        """
        Start a continuous message consumer.

        Args:
            queue: Queue to consume from
            handler: Async function to handle each message
            max_concurrent: Maximum concurrent message handlers
        """
        receiver = await self._get_receiver(queue.value)
        semaphore = asyncio.Semaphore(max_concurrent)

        logger.info(f"Starting consumer for {queue.value}")

        async def process_message(msg: ServiceBusReceivedMessage):
            async with semaphore:
                try:
                    pipeline_msg = PipelineMessage.from_json(str(msg))
                    success = await handler(pipeline_msg)

                    if success:
                        await receiver.complete_message(msg)
                        self.messages_received += 1
                    else:
                        # Retry or dead letter
                        if pipeline_msg.retry_count < pipeline_msg.max_retries:
                            await receiver.abandon_message(msg)
                        else:
                            await receiver.dead_letter_message(msg, reason="Max retries exceeded")
                            self.messages_failed += 1

                except Exception as e:
                    logger.error(f"Handler error: {e}")
                    await receiver.dead_letter_message(msg, reason=str(e))
                    self.messages_failed += 1

        async for msg in receiver:
            asyncio.create_task(process_message(msg))

    # Convenience methods for common operations

    async def publish_scrape_result(
        self,
        job_id: str,
        scraper_type: str,
        url: str,
        data: Dict[str, Any],
        success: bool,
        error: Optional[str] = None
    ) -> bool:
        """
        Publish a scrape result to the enrichment queue.

        Args:
            job_id: Job identifier
            scraper_type: Type of scraper used
            url: URL that was scraped
            data: Scraped data
            success: Whether scrape was successful
            error: Error message if failed

        Returns:
            True if published successfully
        """
        message = PipelineMessage(
            message_type=MessageType.SCRAPE_RESULT.value,
            job_id=job_id,
            source="scraper",
            data={
                "scraper_type": scraper_type,
                "url": url,
                "success": success,
                "error": error,
                "result": data if success else None,
                "scraped_at": datetime.utcnow().isoformat()
            }
        )

        return await self.send_message(QueueName.ENRICHMENT_TASKS, message)

    async def publish_enrichment_request(
        self,
        job_id: str,
        contact_data: Dict[str, Any],
        enrichment_types: List[str] = None
    ) -> bool:
        """
        Publish an enrichment request.

        Args:
            job_id: Job identifier
            contact_data: Contact data to enrich
            enrichment_types: Types of enrichment to perform

        Returns:
            True if published successfully
        """
        message = PipelineMessage(
            message_type=MessageType.ENRICHMENT_REQUEST.value,
            job_id=job_id,
            source="scraper",
            data={
                "contact": contact_data,
                "enrichment_types": enrichment_types or ["email", "phone", "social", "company"]
            }
        )

        return await self.send_message(QueueName.ENRICHMENT_TASKS, message)

    async def publish_ingestion_event(
        self,
        job_id: str,
        event_type: str,
        data: Dict[str, Any]
    ) -> bool:
        """
        Publish an event to the ingestion pipeline.

        Args:
            job_id: Job identifier
            event_type: Type of event
            data: Event data

        Returns:
            True if published successfully
        """
        message = PipelineMessage(
            message_type=MessageType.INGESTION_EVENT.value,
            job_id=job_id,
            source="scraper",
            data={
                "event_type": event_type,
                "payload": data
            }
        )

        return await self.send_message(QueueName.INGESTION_EVENTS, message)

    async def publish_alert(
        self,
        job_id: str,
        alert_type: str,
        message: str,
        severity: str = "warning"
    ) -> bool:
        """
        Publish an alert to the dead letter monitor.

        Args:
            job_id: Related job ID
            alert_type: Type of alert
            message: Alert message
            severity: Alert severity (info, warning, error, critical)

        Returns:
            True if published successfully
        """
        pipeline_msg = PipelineMessage(
            message_type=MessageType.ALERT.value,
            job_id=job_id,
            source="scraper",
            priority=4 if severity == "critical" else 3,
            data={
                "alert_type": alert_type,
                "message": message,
                "severity": severity
            }
        )

        return await self.send_message(QueueName.DEAD_LETTER, pipeline_msg)

    def get_metrics(self) -> Dict[str, Any]:
        """Get Service Bus metrics."""
        return {
            "messages_sent": self.messages_sent,
            "messages_received": self.messages_received,
            "messages_failed": self.messages_failed,
            "success_rate": self.messages_sent / max(1, self.messages_sent + self.messages_failed),
            "active_senders": list(self._senders.keys()),
            "active_receivers": list(self._receivers.keys()),
            "namespace": self.namespace
        }

    async def close(self) -> None:
        """Close all connections."""
        for sender in self._senders.values():
            await sender.close()
        for receiver in self._receivers.values():
            await receiver.close()
        if self._client:
            await self._client.close()
        if self._credential:
            await self._credential.close()

        self._senders.clear()
        self._receivers.clear()
        self._client = None

        logger.info("Service Bus connections closed")


# Global instance
service_bus = ServiceBusManager()
