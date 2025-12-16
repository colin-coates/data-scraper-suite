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
    SentinelBase,
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

__all__ = [
    'SentinelBase',
    'SentinelConfig',
    'SentinelAlert',
    'SentinelSeverity',
    'SentinelStatus',
    'register_sentinel',
    'unregister_sentinel',
    'get_registered_sentinels',
    'get_sentinel',
    'start_all_sentinels',
    'stop_all_sentinels'
]
