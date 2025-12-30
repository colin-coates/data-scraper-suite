# Copyright (c) 2024 Mountain Jewels Intelligence. All rights reserved.
#
# This software is proprietary and confidential. Unauthorized copying,
# modification, distribution, or use is strictly prohibited.

"""
Security Module for MJ Data Scraper Suite

Provides security hardening features:
- Rate limiting
- API key management
- Audit logging
- Request validation
- IP allowlisting/blocklisting
"""

import asyncio
import hashlib
import hmac
import logging
import os
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Callable, Set
from functools import wraps

from fastapi import Request, HTTPException, status
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)


@dataclass
class RateLimitConfig:
    """Rate limit configuration."""
    requests_per_minute: int = 60
    requests_per_hour: int = 1000
    requests_per_day: int = 10000
    burst_limit: int = 10  # Max requests in 1 second


@dataclass
class RateLimitState:
    """Track rate limit state for a client."""
    minute_count: int = 0
    hour_count: int = 0
    day_count: int = 0
    burst_count: int = 0
    minute_reset: float = 0
    hour_reset: float = 0
    day_reset: float = 0
    burst_reset: float = 0


class RateLimiter:
    """
    Token bucket rate limiter with multiple time windows.
    
    Features:
    - Per-client rate limiting
    - Multiple time windows (minute, hour, day)
    - Burst protection
    - Configurable limits per endpoint
    """

    def __init__(self, default_config: Optional[RateLimitConfig] = None):
        """Initialize rate limiter."""
        self.default_config = default_config or RateLimitConfig()
        self.endpoint_configs: Dict[str, RateLimitConfig] = {}
        self.client_states: Dict[str, RateLimitState] = defaultdict(RateLimitState)
        self._lock = asyncio.Lock()

    def configure_endpoint(self, endpoint: str, config: RateLimitConfig) -> None:
        """Configure rate limits for a specific endpoint."""
        self.endpoint_configs[endpoint] = config

    def _get_client_key(self, request: Request) -> str:
        """Get unique client identifier."""
        # Use API key if present, otherwise IP
        api_key = request.headers.get("X-API-Key")
        if api_key:
            return f"key:{hashlib.sha256(api_key.encode()).hexdigest()[:16]}"
        
        # Get real IP (handle proxies)
        forwarded = request.headers.get("X-Forwarded-For")
        if forwarded:
            ip = forwarded.split(",")[0].strip()
        else:
            ip = request.client.host if request.client else "unknown"
        
        return f"ip:{ip}"

    async def check_rate_limit(
        self,
        request: Request,
        endpoint: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Check if request is within rate limits.

        Returns:
            Dict with 'allowed' bool and rate limit headers
        """
        client_key = self._get_client_key(request)
        config = self.endpoint_configs.get(endpoint, self.default_config)
        now = time.time()

        async with self._lock:
            state = self.client_states[client_key]

            # Reset counters if windows expired
            if now > state.burst_reset:
                state.burst_count = 0
                state.burst_reset = now + 1

            if now > state.minute_reset:
                state.minute_count = 0
                state.minute_reset = now + 60

            if now > state.hour_reset:
                state.hour_count = 0
                state.hour_reset = now + 3600

            if now > state.day_reset:
                state.day_count = 0
                state.day_reset = now + 86400

            # Check limits
            if state.burst_count >= config.burst_limit:
                return self._limit_exceeded("burst", state, config)

            if state.minute_count >= config.requests_per_minute:
                return self._limit_exceeded("minute", state, config)

            if state.hour_count >= config.requests_per_hour:
                return self._limit_exceeded("hour", state, config)

            if state.day_count >= config.requests_per_day:
                return self._limit_exceeded("day", state, config)

            # Increment counters
            state.burst_count += 1
            state.minute_count += 1
            state.hour_count += 1
            state.day_count += 1

            return {
                "allowed": True,
                "headers": {
                    "X-RateLimit-Limit": str(config.requests_per_minute),
                    "X-RateLimit-Remaining": str(config.requests_per_minute - state.minute_count),
                    "X-RateLimit-Reset": str(int(state.minute_reset)),
                }
            }

    def _limit_exceeded(
        self,
        window: str,
        state: RateLimitState,
        config: RateLimitConfig
    ) -> Dict[str, Any]:
        """Return rate limit exceeded response."""
        reset_times = {
            "burst": state.burst_reset,
            "minute": state.minute_reset,
            "hour": state.hour_reset,
            "day": state.day_reset,
        }
        limits = {
            "burst": config.burst_limit,
            "minute": config.requests_per_minute,
            "hour": config.requests_per_hour,
            "day": config.requests_per_day,
        }

        return {
            "allowed": False,
            "window": window,
            "retry_after": int(reset_times[window] - time.time()),
            "headers": {
                "X-RateLimit-Limit": str(limits[window]),
                "X-RateLimit-Remaining": "0",
                "X-RateLimit-Reset": str(int(reset_times[window])),
                "Retry-After": str(int(reset_times[window] - time.time())),
            }
        }


@dataclass
class APIKey:
    """API key record."""
    key_id: str
    key_hash: str
    name: str
    created_at: datetime
    expires_at: Optional[datetime] = None
    scopes: List[str] = field(default_factory=list)
    rate_limit: Optional[RateLimitConfig] = None
    is_active: bool = True
    last_used: Optional[datetime] = None
    usage_count: int = 0


class APIKeyManager:
    """
    Manages API keys for authentication.
    
    Features:
    - Key generation and validation
    - Scope-based permissions
    - Key rotation
    - Usage tracking
    """

    def __init__(self, secret_key: Optional[str] = None):
        """Initialize API key manager."""
        self.secret_key = secret_key or os.getenv("API_SECRET_KEY", "mj-scraper-secret")
        self.keys: Dict[str, APIKey] = {}
        self._key_lookup: Dict[str, str] = {}  # hash -> key_id

    def generate_key(
        self,
        name: str,
        scopes: Optional[List[str]] = None,
        expires_in_days: Optional[int] = None,
        rate_limit: Optional[RateLimitConfig] = None
    ) -> Tuple[str, APIKey]:
        """
        Generate a new API key.

        Returns:
            Tuple of (raw_key, APIKey record)
        """
        key_id = str(uuid.uuid4())
        raw_key = f"mj_{uuid.uuid4().hex}"
        key_hash = self._hash_key(raw_key)

        expires_at = None
        if expires_in_days:
            expires_at = datetime.utcnow() + timedelta(days=expires_in_days)

        api_key = APIKey(
            key_id=key_id,
            key_hash=key_hash,
            name=name,
            created_at=datetime.utcnow(),
            expires_at=expires_at,
            scopes=scopes or ["read"],
            rate_limit=rate_limit,
            is_active=True
        )

        self.keys[key_id] = api_key
        self._key_lookup[key_hash] = key_id

        logger.info(f"Generated API key: {name} (id: {key_id})")
        return raw_key, api_key

    def _hash_key(self, raw_key: str) -> str:
        """Hash an API key."""
        return hmac.new(
            self.secret_key.encode(),
            raw_key.encode(),
            hashlib.sha256
        ).hexdigest()

    def validate_key(self, raw_key: str) -> Optional[APIKey]:
        """
        Validate an API key.

        Returns:
            APIKey if valid, None otherwise
        """
        key_hash = self._hash_key(raw_key)
        key_id = self._key_lookup.get(key_hash)

        if not key_id:
            return None

        api_key = self.keys.get(key_id)
        if not api_key:
            return None

        if not api_key.is_active:
            return None

        if api_key.expires_at and datetime.utcnow() > api_key.expires_at:
            return None

        # Update usage
        api_key.last_used = datetime.utcnow()
        api_key.usage_count += 1

        return api_key

    def has_scope(self, api_key: APIKey, required_scope: str) -> bool:
        """Check if API key has required scope."""
        if "admin" in api_key.scopes:
            return True
        return required_scope in api_key.scopes

    def revoke_key(self, key_id: str) -> bool:
        """Revoke an API key."""
        if key_id not in self.keys:
            return False

        self.keys[key_id].is_active = False
        logger.info(f"Revoked API key: {key_id}")
        return True

    def rotate_key(self, key_id: str) -> Optional[Tuple[str, APIKey]]:
        """
        Rotate an API key (generate new, revoke old).

        Returns:
            Tuple of (new_raw_key, new_APIKey) or None
        """
        old_key = self.keys.get(key_id)
        if not old_key:
            return None

        # Generate new key with same settings
        new_raw_key, new_key = self.generate_key(
            name=old_key.name,
            scopes=old_key.scopes,
            expires_in_days=None,  # Calculate from original if needed
            rate_limit=old_key.rate_limit
        )

        # Revoke old key
        self.revoke_key(key_id)

        logger.info(f"Rotated API key: {old_key.name}")
        return new_raw_key, new_key

    def list_keys(self, include_inactive: bool = False) -> List[APIKey]:
        """List all API keys."""
        keys = list(self.keys.values())
        if not include_inactive:
            keys = [k for k in keys if k.is_active]
        return keys


@dataclass
class AuditLogEntry:
    """Audit log entry."""
    timestamp: datetime
    event_type: str
    actor: str
    resource: str
    action: str
    details: Dict[str, Any]
    ip_address: str
    user_agent: str
    success: bool
    error: Optional[str] = None


class AuditLogger:
    """
    Audit logging for security events.
    
    Features:
    - Structured logging
    - Event categorization
    - Search and filtering
    - Retention policies
    """

    def __init__(self, max_entries: int = 10000):
        """Initialize audit logger."""
        self.max_entries = max_entries
        self.entries: List[AuditLogEntry] = []
        self._lock = asyncio.Lock()

    async def log(
        self,
        event_type: str,
        actor: str,
        resource: str,
        action: str,
        request: Optional[Request] = None,
        details: Optional[Dict[str, Any]] = None,
        success: bool = True,
        error: Optional[str] = None
    ) -> AuditLogEntry:
        """Log an audit event."""
        ip_address = "unknown"
        user_agent = "unknown"

        if request:
            forwarded = request.headers.get("X-Forwarded-For")
            if forwarded:
                ip_address = forwarded.split(",")[0].strip()
            elif request.client:
                ip_address = request.client.host

            user_agent = request.headers.get("User-Agent", "unknown")

        entry = AuditLogEntry(
            timestamp=datetime.utcnow(),
            event_type=event_type,
            actor=actor,
            resource=resource,
            action=action,
            details=details or {},
            ip_address=ip_address,
            user_agent=user_agent,
            success=success,
            error=error
        )

        async with self._lock:
            self.entries.append(entry)

            # Enforce retention
            if len(self.entries) > self.max_entries:
                self.entries = self.entries[-self.max_entries:]

        # Also log to standard logger
        log_msg = f"AUDIT: {event_type} | {actor} | {action} {resource}"
        if success:
            logger.info(log_msg)
        else:
            logger.warning(f"{log_msg} | ERROR: {error}")

        return entry

    async def search(
        self,
        event_type: Optional[str] = None,
        actor: Optional[str] = None,
        resource: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        success_only: bool = False,
        limit: int = 100
    ) -> List[AuditLogEntry]:
        """Search audit logs."""
        results = self.entries.copy()

        if event_type:
            results = [e for e in results if e.event_type == event_type]

        if actor:
            results = [e for e in results if e.actor == actor]

        if resource:
            results = [e for e in results if resource in e.resource]

        if start_time:
            results = [e for e in results if e.timestamp >= start_time]

        if end_time:
            results = [e for e in results if e.timestamp <= end_time]

        if success_only:
            results = [e for e in results if e.success]

        # Sort by timestamp descending
        results.sort(key=lambda e: e.timestamp, reverse=True)

        return results[:limit]

    def get_stats(self) -> Dict[str, Any]:
        """Get audit log statistics."""
        if not self.entries:
            return {"total_entries": 0}

        event_counts = defaultdict(int)
        actor_counts = defaultdict(int)
        success_count = 0

        for entry in self.entries:
            event_counts[entry.event_type] += 1
            actor_counts[entry.actor] += 1
            if entry.success:
                success_count += 1

        return {
            "total_entries": len(self.entries),
            "success_rate": success_count / len(self.entries),
            "event_types": dict(event_counts),
            "top_actors": dict(sorted(actor_counts.items(), key=lambda x: x[1], reverse=True)[:10]),
            "oldest_entry": self.entries[0].timestamp.isoformat() if self.entries else None,
            "newest_entry": self.entries[-1].timestamp.isoformat() if self.entries else None,
        }


class IPFilter:
    """IP allowlist/blocklist filtering."""

    def __init__(self):
        """Initialize IP filter."""
        self.allowlist: Set[str] = set()
        self.blocklist: Set[str] = set()
        self.use_allowlist: bool = False  # If True, only allow IPs in allowlist

    def add_to_allowlist(self, ip: str) -> None:
        """Add IP to allowlist."""
        self.allowlist.add(ip)

    def add_to_blocklist(self, ip: str) -> None:
        """Add IP to blocklist."""
        self.blocklist.add(ip)

    def remove_from_allowlist(self, ip: str) -> None:
        """Remove IP from allowlist."""
        self.allowlist.discard(ip)

    def remove_from_blocklist(self, ip: str) -> None:
        """Remove IP from blocklist."""
        self.blocklist.discard(ip)

    def is_allowed(self, ip: str) -> bool:
        """Check if IP is allowed."""
        # Always block if in blocklist
        if ip in self.blocklist:
            return False

        # If using allowlist mode, must be in allowlist
        if self.use_allowlist:
            return ip in self.allowlist

        return True

    def get_client_ip(self, request: Request) -> str:
        """Extract client IP from request."""
        forwarded = request.headers.get("X-Forwarded-For")
        if forwarded:
            return forwarded.split(",")[0].strip()
        return request.client.host if request.client else "unknown"


# Global instances
rate_limiter = RateLimiter()
api_key_manager = APIKeyManager()
audit_logger = AuditLogger()
ip_filter = IPFilter()


# FastAPI middleware helper
async def security_middleware(request: Request, call_next):
    """Security middleware for FastAPI."""
    # Check IP filter
    client_ip = ip_filter.get_client_ip(request)
    if not ip_filter.is_allowed(client_ip):
        await audit_logger.log(
            event_type="access_denied",
            actor=client_ip,
            resource=str(request.url.path),
            action="blocked_ip",
            request=request,
            success=False,
            error="IP blocked"
        )
        return JSONResponse(
            status_code=status.HTTP_403_FORBIDDEN,
            content={"detail": "Access denied"}
        )

    # Check rate limit
    rate_check = await rate_limiter.check_rate_limit(request, request.url.path)
    if not rate_check["allowed"]:
        await audit_logger.log(
            event_type="rate_limit",
            actor=client_ip,
            resource=str(request.url.path),
            action="rate_limited",
            request=request,
            success=False,
            error=f"Rate limit exceeded ({rate_check['window']})"
        )
        return JSONResponse(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            content={"detail": "Rate limit exceeded", "retry_after": rate_check["retry_after"]},
            headers=rate_check["headers"]
        )

    # Process request
    response = await call_next(request)

    # Add rate limit headers
    for key, value in rate_check.get("headers", {}).items():
        response.headers[key] = value

    return response


# Decorator for requiring API key
def require_api_key(scopes: Optional[List[str]] = None):
    """Decorator to require API key authentication."""
    def decorator(func: Callable):
        @wraps(func)
        async def wrapper(request: Request, *args, **kwargs):
            api_key_header = request.headers.get("X-API-Key")
            
            if not api_key_header:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="API key required"
                )

            api_key = api_key_manager.validate_key(api_key_header)
            if not api_key:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid API key"
                )

            # Check scopes
            if scopes:
                for scope in scopes:
                    if not api_key_manager.has_scope(api_key, scope):
                        raise HTTPException(
                            status_code=status.HTTP_403_FORBIDDEN,
                            detail=f"Missing required scope: {scope}"
                        )

            # Add API key to request state
            request.state.api_key = api_key
            
            return await func(request, *args, **kwargs)
        return wrapper
    return decorator
