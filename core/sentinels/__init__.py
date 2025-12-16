# Copyright (c) 2024 Mountain Jewels Intelligence. All rights reserved.
#
# This software is proprietary and confidential. Unauthorized copying,
# modification, distribution, or use is strictly prohibited.

"""
Sentinel System for MJ Data Scraper Suite

Provides monitoring, alerting, and automated response capabilities for scraping operations.
Sentinels continuously monitor system health, performance, and compliance metrics.
"""

from .base import (
    BaseSentinel,
    SentinelReport,
    SentinelRunner,
    SentinelConfig,
    SentinelAlert,
    SentinelSeverity,
    SentinelStatus,
    register_sentinel,
    unregister_sentinel,
    get_registered_sentinels,
    get_sentinel,
    start_all_sentinels,
    stop_all_sentinels
)

from .performance_sentinel import PerformanceSentinel, create_performance_sentinel
from .network_sentinel import NetworkSentinel, create_network_sentinel
from .waf_sentinel import WafSentinel, create_waf_sentinel
from .malware_sentinel import MalwareSentinel, create_malware_sentinel
from .sentinel_orchestrator import (
    SentinelOrchestrator,
    OrchestrationResult,
    OrchestrationMode,
    OrchestrationStrategy,
    create_comprehensive_orchestrator,
    create_security_focused_orchestrator,
    create_performance_orchestrator
)

__all__ = [
    'BaseSentinel',
    'SentinelReport',
    'SentinelRunner',
    'SentinelConfig',
    'SentinelAlert',
    'SentinelSeverity',
    'SentinelStatus',
    'PerformanceSentinel',
    'NetworkSentinel',
    'WafSentinel',
    'MalwareSentinel',
    'SentinelOrchestrator',
    'OrchestrationResult',
    'OrchestrationMode',
    'OrchestrationStrategy',
    'create_performance_sentinel',
    'create_network_sentinel',
    'create_waf_sentinel',
    'create_malware_sentinel',
    'create_comprehensive_orchestrator',
    'create_security_focused_orchestrator',
    'create_performance_orchestrator',
    'WafSentinel',
    'register_sentinel',
    'unregister_sentinel',
    'get_registered_sentinels',
    'get_sentinel',
    'start_all_sentinels',
    'stop_all_sentinels'
]
