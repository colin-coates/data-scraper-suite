# Copyright (c) 2024 Mountain Jewels Intelligence. All rights reserved.
#
# This software is proprietary and confidential. Unauthorized copying,
# modification, distribution, or use is strictly prohibited.

"""
Root-level queue publisher module.

This module provides easy access to the queue publishing functionality
by importing from the core implementation.
"""

from core.queue_publisher import QueuePublisher, QueueConfig, PublishResult

__all__ = ['QueuePublisher', 'QueueConfig', 'PublishResult']
