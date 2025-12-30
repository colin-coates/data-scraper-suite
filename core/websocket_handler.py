# Copyright (c) 2024 Mountain Jewels Intelligence. All rights reserved.
#
# This software is proprietary and confidential. Unauthorized copying,
# modification, distribution, or use is strictly prohibited.

"""
WebSocket Handler for Real-time Scraper Updates

Provides live streaming of:
- Job status updates
- Scraping progress
- Sentinel alerts
- Cost tracking
- Error notifications
"""

import asyncio
import json
import logging
from typing import Dict, Any, List, Optional, Set, Callable
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
import weakref

from fastapi import WebSocket, WebSocketDisconnect

logger = logging.getLogger(__name__)


class EventType(Enum):
    """Types of WebSocket events."""
    JOB_STARTED = "job_started"
    JOB_PROGRESS = "job_progress"
    JOB_COMPLETED = "job_completed"
    JOB_FAILED = "job_failed"
    SCRAPE_SUCCESS = "scrape_success"
    SCRAPE_ERROR = "scrape_error"
    SENTINEL_ALERT = "sentinel_alert"
    RATE_LIMIT = "rate_limit"
    COST_UPDATE = "cost_update"
    SYSTEM_STATUS = "system_status"
    HEARTBEAT = "heartbeat"


@dataclass
class WebSocketEvent:
    """Event to broadcast via WebSocket."""
    event_type: EventType
    data: Dict[str, Any]
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    job_id: Optional[str] = None
    severity: str = "info"  # info, warning, error, critical

    def to_json(self) -> str:
        """Convert event to JSON string."""
        return json.dumps({
            "type": self.event_type.value,
            "data": self.data,
            "timestamp": self.timestamp,
            "job_id": self.job_id,
            "severity": self.severity
        })


class ConnectionManager:
    """
    Manages WebSocket connections and broadcasts.
    
    Features:
    - Multiple client support
    - Topic-based subscriptions
    - Automatic reconnection handling
    - Message buffering for slow clients
    """

    def __init__(self, max_connections: int = 100, buffer_size: int = 100):
        """
        Initialize connection manager.

        Args:
            max_connections: Maximum concurrent connections
            buffer_size: Message buffer size per client
        """
        self.max_connections = max_connections
        self.buffer_size = buffer_size
        
        # Active connections
        self.active_connections: Set[WebSocket] = set()
        
        # Topic subscriptions: topic -> set of websockets
        self.subscriptions: Dict[str, Set[WebSocket]] = {}
        
        # Message buffers for slow clients
        self.buffers: Dict[WebSocket, List[WebSocketEvent]] = {}
        
        # Connection metadata
        self.connection_info: Dict[WebSocket, Dict[str, Any]] = {}
        
        # Event hooks
        self.on_connect: Optional[Callable[[WebSocket], None]] = None
        self.on_disconnect: Optional[Callable[[WebSocket], None]] = None
        
        # Metrics
        self.total_messages_sent = 0
        self.total_connections = 0

    async def connect(self, websocket: WebSocket, client_id: Optional[str] = None) -> bool:
        """
        Accept a new WebSocket connection.

        Args:
            websocket: The WebSocket connection
            client_id: Optional client identifier

        Returns:
            True if connection accepted
        """
        if len(self.active_connections) >= self.max_connections:
            await websocket.close(code=1013, reason="Max connections reached")
            return False

        await websocket.accept()
        self.active_connections.add(websocket)
        self.buffers[websocket] = []
        self.connection_info[websocket] = {
            "client_id": client_id,
            "connected_at": datetime.utcnow().isoformat(),
            "subscriptions": set()
        }
        
        self.total_connections += 1
        logger.info(f"WebSocket connected: {client_id or 'anonymous'} (total: {len(self.active_connections)})")

        if self.on_connect:
            self.on_connect(websocket)

        # Send welcome message
        await self.send_personal(websocket, WebSocketEvent(
            event_type=EventType.SYSTEM_STATUS,
            data={"message": "Connected to MJ Scraper Suite", "status": "connected"}
        ))

        return True

    def disconnect(self, websocket: WebSocket) -> None:
        """Handle WebSocket disconnection."""
        self.active_connections.discard(websocket)
        
        # Remove from all subscriptions
        for topic_subs in self.subscriptions.values():
            topic_subs.discard(websocket)
        
        # Cleanup
        self.buffers.pop(websocket, None)
        info = self.connection_info.pop(websocket, {})
        
        logger.info(f"WebSocket disconnected: {info.get('client_id', 'anonymous')}")

        if self.on_disconnect:
            self.on_disconnect(websocket)

    def subscribe(self, websocket: WebSocket, topic: str) -> None:
        """Subscribe a connection to a topic."""
        if topic not in self.subscriptions:
            self.subscriptions[topic] = set()
        self.subscriptions[topic].add(websocket)
        
        if websocket in self.connection_info:
            self.connection_info[websocket]["subscriptions"].add(topic)
        
        logger.debug(f"Subscribed to topic: {topic}")

    def unsubscribe(self, websocket: WebSocket, topic: str) -> None:
        """Unsubscribe a connection from a topic."""
        if topic in self.subscriptions:
            self.subscriptions[topic].discard(websocket)
        
        if websocket in self.connection_info:
            self.connection_info[websocket]["subscriptions"].discard(topic)

    async def send_personal(self, websocket: WebSocket, event: WebSocketEvent) -> bool:
        """
        Send event to a specific connection.

        Args:
            websocket: Target connection
            event: Event to send

        Returns:
            True if sent successfully
        """
        try:
            await websocket.send_text(event.to_json())
            self.total_messages_sent += 1
            return True
        except Exception as e:
            logger.warning(f"Failed to send to client: {e}")
            # Buffer for retry
            if websocket in self.buffers:
                if len(self.buffers[websocket]) < self.buffer_size:
                    self.buffers[websocket].append(event)
            return False

    async def broadcast(self, event: WebSocketEvent) -> int:
        """
        Broadcast event to all connected clients.

        Args:
            event: Event to broadcast

        Returns:
            Number of clients that received the message
        """
        sent_count = 0
        disconnected = []

        for connection in self.active_connections.copy():
            try:
                await connection.send_text(event.to_json())
                sent_count += 1
                self.total_messages_sent += 1
            except Exception:
                disconnected.append(connection)

        # Cleanup disconnected
        for conn in disconnected:
            self.disconnect(conn)

        return sent_count

    async def broadcast_to_topic(self, topic: str, event: WebSocketEvent) -> int:
        """
        Broadcast event to subscribers of a topic.

        Args:
            topic: Topic name
            event: Event to broadcast

        Returns:
            Number of subscribers that received the message
        """
        if topic not in self.subscriptions:
            return 0

        sent_count = 0
        disconnected = []

        for connection in self.subscriptions[topic].copy():
            try:
                await connection.send_text(event.to_json())
                sent_count += 1
                self.total_messages_sent += 1
            except Exception:
                disconnected.append(connection)

        for conn in disconnected:
            self.disconnect(conn)

        return sent_count

    async def flush_buffers(self) -> None:
        """Flush buffered messages to clients."""
        for websocket, buffer in list(self.buffers.items()):
            if not buffer:
                continue

            sent = []
            for event in buffer:
                try:
                    await websocket.send_text(event.to_json())
                    sent.append(event)
                    self.total_messages_sent += 1
                except Exception:
                    break

            for event in sent:
                buffer.remove(event)

    def get_connection_count(self) -> int:
        """Get number of active connections."""
        return len(self.active_connections)

    def get_metrics(self) -> Dict[str, Any]:
        """Get connection manager metrics."""
        return {
            "active_connections": len(self.active_connections),
            "total_connections": self.total_connections,
            "total_messages_sent": self.total_messages_sent,
            "topics": list(self.subscriptions.keys()),
            "topic_subscribers": {
                topic: len(subs) for topic, subs in self.subscriptions.items()
            }
        }


class ScraperEventEmitter:
    """
    Event emitter for scraper operations.
    
    Integrates with scrapers to emit real-time events.
    """

    def __init__(self, connection_manager: ConnectionManager):
        """
        Initialize event emitter.

        Args:
            connection_manager: WebSocket connection manager
        """
        self.manager = connection_manager
        self._event_queue: asyncio.Queue = asyncio.Queue()
        self._running = False
        self._task: Optional[asyncio.Task] = None

    async def start(self) -> None:
        """Start the event emitter background task."""
        if self._running:
            return

        self._running = True
        self._task = asyncio.create_task(self._process_events())
        logger.info("Scraper event emitter started")

    async def stop(self) -> None:
        """Stop the event emitter."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("Scraper event emitter stopped")

    async def _process_events(self) -> None:
        """Process events from the queue."""
        while self._running:
            try:
                event = await asyncio.wait_for(
                    self._event_queue.get(),
                    timeout=1.0
                )
                await self.manager.broadcast(event)
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Error processing event: {e}")

    def emit(self, event: WebSocketEvent) -> None:
        """
        Emit an event (non-blocking).

        Args:
            event: Event to emit
        """
        try:
            self._event_queue.put_nowait(event)
        except asyncio.QueueFull:
            logger.warning("Event queue full, dropping event")

    def emit_job_started(self, job_id: str, job_type: str, target: str) -> None:
        """Emit job started event."""
        self.emit(WebSocketEvent(
            event_type=EventType.JOB_STARTED,
            job_id=job_id,
            data={
                "job_type": job_type,
                "target": target,
                "status": "started"
            }
        ))

    def emit_job_progress(
        self,
        job_id: str,
        progress: float,
        pages_scraped: int,
        records_collected: int
    ) -> None:
        """Emit job progress event."""
        self.emit(WebSocketEvent(
            event_type=EventType.JOB_PROGRESS,
            job_id=job_id,
            data={
                "progress": progress,
                "pages_scraped": pages_scraped,
                "records_collected": records_collected
            }
        ))

    def emit_job_completed(
        self,
        job_id: str,
        total_records: int,
        duration_seconds: float,
        success: bool = True
    ) -> None:
        """Emit job completed event."""
        self.emit(WebSocketEvent(
            event_type=EventType.JOB_COMPLETED if success else EventType.JOB_FAILED,
            job_id=job_id,
            data={
                "total_records": total_records,
                "duration_seconds": duration_seconds,
                "success": success
            },
            severity="info" if success else "error"
        ))

    def emit_scrape_success(self, job_id: str, url: str, records: int) -> None:
        """Emit successful scrape event."""
        self.emit(WebSocketEvent(
            event_type=EventType.SCRAPE_SUCCESS,
            job_id=job_id,
            data={"url": url, "records": records}
        ))

    def emit_scrape_error(self, job_id: str, url: str, error: str) -> None:
        """Emit scrape error event."""
        self.emit(WebSocketEvent(
            event_type=EventType.SCRAPE_ERROR,
            job_id=job_id,
            data={"url": url, "error": error},
            severity="error"
        ))

    def emit_sentinel_alert(
        self,
        sentinel_name: str,
        alert_type: str,
        message: str,
        severity: str = "warning"
    ) -> None:
        """Emit sentinel alert event."""
        self.emit(WebSocketEvent(
            event_type=EventType.SENTINEL_ALERT,
            data={
                "sentinel": sentinel_name,
                "alert_type": alert_type,
                "message": message
            },
            severity=severity
        ))

    def emit_cost_update(
        self,
        job_id: str,
        current_cost: float,
        budget_remaining: float
    ) -> None:
        """Emit cost update event."""
        self.emit(WebSocketEvent(
            event_type=EventType.COST_UPDATE,
            job_id=job_id,
            data={
                "current_cost": current_cost,
                "budget_remaining": budget_remaining
            }
        ))

    def emit_rate_limit(self, domain: str, wait_seconds: float) -> None:
        """Emit rate limit event."""
        self.emit(WebSocketEvent(
            event_type=EventType.RATE_LIMIT,
            data={
                "domain": domain,
                "wait_seconds": wait_seconds
            },
            severity="warning"
        ))


# Global instances
connection_manager = ConnectionManager()
event_emitter = ScraperEventEmitter(connection_manager)


async def websocket_endpoint(websocket: WebSocket, client_id: Optional[str] = None):
    """
    WebSocket endpoint handler for FastAPI.
    
    Usage in FastAPI:
        @app.websocket("/ws/scraper-status")
        async def scraper_status(websocket: WebSocket):
            await websocket_endpoint(websocket)
    """
    if not await connection_manager.connect(websocket, client_id):
        return

    try:
        # Start heartbeat
        heartbeat_task = asyncio.create_task(_send_heartbeat(websocket))

        while True:
            # Receive messages from client
            data = await websocket.receive_text()
            
            try:
                message = json.loads(data)
                await _handle_client_message(websocket, message)
            except json.JSONDecodeError:
                await connection_manager.send_personal(websocket, WebSocketEvent(
                    event_type=EventType.SYSTEM_STATUS,
                    data={"error": "Invalid JSON"},
                    severity="error"
                ))

    except WebSocketDisconnect:
        pass
    finally:
        heartbeat_task.cancel()
        connection_manager.disconnect(websocket)


async def _send_heartbeat(websocket: WebSocket, interval: int = 30):
    """Send periodic heartbeat to keep connection alive."""
    while True:
        await asyncio.sleep(interval)
        try:
            await connection_manager.send_personal(websocket, WebSocketEvent(
                event_type=EventType.HEARTBEAT,
                data={"status": "alive"}
            ))
        except Exception:
            break


async def _handle_client_message(websocket: WebSocket, message: Dict[str, Any]):
    """Handle incoming client messages."""
    action = message.get("action")

    if action == "subscribe":
        topic = message.get("topic")
        if topic:
            connection_manager.subscribe(websocket, topic)
            await connection_manager.send_personal(websocket, WebSocketEvent(
                event_type=EventType.SYSTEM_STATUS,
                data={"subscribed": topic}
            ))

    elif action == "unsubscribe":
        topic = message.get("topic")
        if topic:
            connection_manager.unsubscribe(websocket, topic)
            await connection_manager.send_personal(websocket, WebSocketEvent(
                event_type=EventType.SYSTEM_STATUS,
                data={"unsubscribed": topic}
            ))

    elif action == "get_status":
        await connection_manager.send_personal(websocket, WebSocketEvent(
            event_type=EventType.SYSTEM_STATUS,
            data=connection_manager.get_metrics()
        ))
