# Copyright (c) 2024 Mountain Jewels Intelligence. All rights reserved.
#
# This software is proprietary and confidential. Unauthorized copying,
# modification, distribution, or use is strictly prohibited.

"""
Authorization Gate for MJ Data Scraper Suite

Manages authorization validation, approval workflows, and access control
for scraping operations with audit trail and compliance features.
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field

from .control_models import ScrapeAuthorization

logger = logging.getLogger(__name__)


@dataclass
class AuthorizationResult:
    """Result of an authorization check."""
    authorized: bool
    reason: str = ""
    details: Dict[str, Any] = field(default_factory=dict)
    check_time: datetime = field(default_factory=datetime.utcnow)
    expires_in_seconds: Optional[int] = None


@dataclass
class AuthorizationAuditEntry:
    """Audit entry for authorization checks."""
    authorization_id: str
    check_time: datetime
    result: AuthorizationResult
    context: Dict[str, Any] = field(default_factory=dict)


class AuthorizationGate:
    """
    Authorization validation gate for scraping operations.

    Validates approvals, checks expiry, enforces policies, and maintains
    comprehensive audit trails for compliance and governance.
    """

    # Class-level audit log
    _audit_log: List[AuthorizationAuditEntry] = []

    @staticmethod
    def validate(auth) -> None:
        """
        Simple authorization validation - checks expiry only.

        Args:
            auth: Authorization object with expires_at attribute

        Raises:
            RuntimeError: If authorization has expired
        """
        now = datetime.utcnow()
        if now > auth.expires_at:
            raise RuntimeError("Authorization expired")

    @classmethod
    def validate_comprehensive(cls, authorization: ScrapeAuthorization, context: Optional[Dict[str, Any]] = None) -> AuthorizationResult:
        """
        Validate a scraping authorization.

        Args:
            authorization: Authorization to validate
            context: Optional context information for audit

        Returns:
            AuthorizationResult with validation outcome

        Raises:
            ValueError: If authorization is invalid
        """
        context = context or {}
        check_time = datetime.utcnow()

        # Create audit entry
        audit_entry = AuthorizationAuditEntry(
            authorization_id=f"{authorization.approved_by}_{int(check_time.timestamp())}",
            check_time=check_time,
            result=AuthorizationResult(authorized=False, check_time=check_time),
            context=context
        )

        try:
            # Check expiry
            if not authorization.is_valid(check_time):
                result = AuthorizationResult(
                    authorized=False,
                    reason="Authorization has expired",
                    details={
                        "expiry_time": authorization.expires_at.isoformat(),
                        "check_time": check_time.isoformat(),
                        "expired_seconds_ago": (check_time - authorization.expires_at).total_seconds()
                    },
                    check_time=check_time
                )
                audit_entry.result = result
                cls._audit_log.append(audit_entry)
                raise ValueError(result.reason)

            # Check approval timestamp (should be in the past)
            if authorization.approval_timestamp > check_time:
                result = AuthorizationResult(
                    authorized=False,
                    reason="Approval timestamp is in the future",
                    details={
                        "approval_timestamp": authorization.approval_timestamp.isoformat(),
                        "check_time": check_time.isoformat()
                    },
                    check_time=check_time
                )
                audit_entry.result = result
                cls._audit_log.append(audit_entry)
                raise ValueError(result.reason)

            # Check minimum validity period (at least 1 hour)
            validity_period = authorization.expires_at - authorization.approval_timestamp
            if validity_period < timedelta(hours=1):
                result = AuthorizationResult(
                    authorized=False,
                    reason="Authorization validity period too short",
                    details={
                        "validity_hours": validity_period.total_seconds() / 3600,
                        "minimum_required": 1.0
                    },
                    check_time=check_time
                )
                audit_entry.result = result
                cls._audit_log.append(audit_entry)
                raise ValueError(result.reason)

            # Check approver identity (basic validation)
            if not cls._validate_approver(authorization.approved_by):
                result = AuthorizationResult(
                    authorized=False,
                    reason="Invalid approver identity",
                    details={"approver": authorization.approved_by},
                    check_time=check_time
                )
                audit_entry.result = result
                cls._audit_log.append(audit_entry)
                raise ValueError(result.reason)

            # Check purpose description
            if not cls._validate_purpose(authorization.purpose):
                result = AuthorizationResult(
                    authorized=False,
                    reason="Invalid or insufficient purpose description",
                    details={"purpose": authorization.purpose},
                    check_time=check_time
                )
                audit_entry.result = result
                cls._audit_log.append(audit_entry)
                raise ValueError(result.reason)

            # All checks passed
            expires_in = int((authorization.expires_at - check_time).total_seconds())
            result = AuthorizationResult(
                authorized=True,
                reason="Authorization validated successfully",
                details={
                    "approved_by": authorization.approved_by,
                    "purpose": authorization.purpose,
                    "valid_until": authorization.expires_at.isoformat()
                },
                check_time=check_time,
                expires_in_seconds=expires_in
            )

            audit_entry.result = result
            cls._audit_log.append(audit_entry)

            logger.info(f"Authorization validated for {authorization.approved_by}: {authorization.purpose}")
            return result

        except Exception as e:
            logger.warning(f"Authorization validation failed: {e}")
            raise

    @staticmethod
    def _validate_approver(approver: str) -> bool:
        """
        Validate approver identity.

        Args:
            approver: Approver identifier to validate

        Returns:
            True if approver is valid
        """
        if not approver or not isinstance(approver, str):
            return False

        # Basic email validation for approvers
        if "@" in approver:
            local, domain = approver.split("@", 1)
            if not local or not domain or "." not in domain:
                return False

        # Check minimum length
        if len(approver.strip()) < 3:
            return False

        # Could add more sophisticated validation here:
        # - Check against approved approvers list
        # - Validate against LDAP/AD
        # - Check digital signatures

        return True

    @staticmethod
    def _validate_purpose(purpose: str) -> bool:
        """
        Validate authorization purpose.

        Args:
            purpose: Purpose description to validate

        Returns:
            True if purpose is valid
        """
        if not purpose or not isinstance(purpose, str):
            return False

        purpose = purpose.strip()

        # Check minimum length
        if len(purpose) < 10:
            return False

        # Check for required keywords (customize as needed)
        required_keywords = ["data", "scraping", "collection", "research", "analysis"]
        has_keyword = any(keyword.lower() in purpose.lower() for keyword in required_keywords)

        # Check for prohibited terms
        prohibited_terms = ["illegal", "unauthorized", "hack", "breach"]
        has_prohibited = any(term.lower() in purpose.lower() for term in prohibited_terms)

        return has_keyword and not has_prohibited

    @classmethod
    def get_audit_log(cls, limit: Optional[int] = None) -> List[AuthorizationAuditEntry]:
        """
        Get authorization audit log.

        Args:
            limit: Maximum number of entries to return

        Returns:
            List of audit entries (most recent first)
        """
        entries = sorted(cls._audit_log, key=lambda x: x.check_time, reverse=True)
        return entries[:limit] if limit else entries

    @classmethod
    def get_audit_summary(cls) -> Dict[str, Any]:
        """Get summary of authorization audit activity."""
        if not cls._audit_log:
            return {"total_checks": 0, "authorized": 0, "denied": 0}

        total_checks = len(cls._audit_log)
        authorized = sum(1 for entry in cls._audit_log if entry.result.authorized)
        denied = total_checks - authorized

        recent_entries = [entry for entry in cls._audit_log
                         if (datetime.utcnow() - entry.check_time) < timedelta(hours=24)]

        return {
            "total_checks": total_checks,
            "authorized": authorized,
            "denied": denied,
            "authorization_rate": authorized / max(1, total_checks),
            "recent_checks_24h": len(recent_entries),
            "last_check": cls._audit_log[-1].check_time if cls._audit_log else None
        }

    @classmethod
    def clear_audit_log(cls, older_than_days: int = 30) -> int:
        """
        Clear old audit log entries.

        Args:
            older_than_days: Remove entries older than this many days

        Returns:
            Number of entries removed
        """
        cutoff_time = datetime.utcnow() - timedelta(days=older_than_days)
        original_count = len(cls._audit_log)

        cls._audit_log = [
            entry for entry in cls._audit_log
            if entry.check_time > cutoff_time
        ]

        removed_count = original_count - len(cls._audit_log)
        logger.info(f"Cleared {removed_count} old audit log entries")
        return removed_count

    @staticmethod
    def create_emergency_authorization(
        purpose: str,
        duration_hours: int = 24,
        approver: str = "emergency_system"
    ) -> ScrapeAuthorization:
        """
        Create an emergency authorization for critical operations.

        Args:
            purpose: Purpose of the emergency authorization
            duration_hours: How long the authorization should be valid
            approver: Emergency approver identifier

        Returns:
            Emergency authorization (use with caution)
        """
        now = datetime.utcnow()

        authorization = ScrapeAuthorization(
            approved_by=approver,
            purpose=f"EMERGENCY: {purpose}",
            approval_timestamp=now,
            expires_at=now + timedelta(hours=duration_hours)
        )

        logger.warning(f"Emergency authorization created: {purpose} (expires: {authorization.expires_at})")
        return authorization

    @staticmethod
    def validate_authorization_chain(authorizations: List[ScrapeAuthorization]) -> AuthorizationResult:
        """
        Validate a chain of authorizations (for multi-party approval).

        Args:
            authorizations: List of authorizations to validate

        Returns:
            Combined validation result
        """
        if not authorizations:
            return AuthorizationResult(
                authorized=False,
                reason="No authorizations provided",
                check_time=datetime.utcnow()
            )

        combined_details = {}
        earliest_expiry = None

        for i, auth in enumerate(authorizations):
            try:
                result = AuthorizationGate.validate(auth, {"chain_position": i})
                if not result.authorized:
                    return AuthorizationResult(
                        authorized=False,
                        reason=f"Authorization {i} failed: {result.reason}",
                        details=result.details,
                        check_time=datetime.utcnow()
                    )

                # Track earliest expiry
                if earliest_expiry is None or auth.expires_at < earliest_expiry:
                    earliest_expiry = auth.expires_at

                combined_details[f"auth_{i}"] = {
                    "approved_by": auth.approved_by,
                    "purpose": auth.purpose,
                    "expires": auth.expires_at.isoformat()
                }

            except ValueError as e:
                return AuthorizationResult(
                    authorized=False,
                    reason=f"Authorization chain validation failed at position {i}: {e}",
                    details={"failed_position": i, "error": str(e)},
                    check_time=datetime.utcnow()
                )

        return AuthorizationResult(
            authorized=True,
            reason=f"All {len(authorizations)} authorizations in chain validated",
            details=combined_details,
            check_time=datetime.utcnow(),
            expires_in_seconds=int((earliest_expiry - datetime.utcnow()).total_seconds())
        )
