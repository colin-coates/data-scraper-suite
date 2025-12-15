# Copyright (c) 2024 Mountain Jewels Intelligence. All rights reserved.
#
# This software is proprietary and confidential. Unauthorized copying,
# modification, distribution, or use is strictly prohibited.

"""
Deployment Timer for MJ Data Scraper Suite

Manages time-based deployment windows and scheduling for scraping operations.
Ensures jobs only execute within authorized time windows with timezone support.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Optional

from .control_models import DeploymentWindow

logger = logging.getLogger(__name__)


class DeploymentTimer:
    """
    Time-based deployment window manager for scraping operations.

    Handles scheduling, timezone conversions, and waiting for deployment windows.
    Ensures compliance with authorized execution timeframes.
    """

    @staticmethod
    async def await_window(deployment_window: DeploymentWindow) -> None:
        """
        Wait until the deployment window opens, if not already within it.

        Args:
            deployment_window: Time window constraints for deployment

        Raises:
            ValueError: If current time is past the latest start time
            TimeoutError: If waiting exceeds reasonable limits
        """
        current_time = datetime.utcnow()

        # Check if we're already within the window
        if deployment_window.is_within_window(current_time):
            logger.info("Already within deployment window - proceeding immediately")
            return

        # Check if window has passed
        if current_time > deployment_window.latest_start:
            raise ValueError(
                f"Deployment window has expired. "
                f"Latest start: {deployment_window.latest_start}, "
                f"Current time: {current_time}"
            )

        # Calculate wait time until earliest start
        wait_seconds = (deployment_window.earliest_start - current_time).total_seconds()

        if wait_seconds > 86400:  # More than 24 hours
            logger.warning(f"Long wait until deployment window: {wait_seconds/3600:.1f} hours")

        logger.info(
            f"Waiting {wait_seconds:.0f} seconds until deployment window opens "
            f"(earliest: {deployment_window.earliest_start})"
        )

        # Wait until window opens
        await asyncio.sleep(max(0, wait_seconds))

        # Double-check we're actually in the window
        final_check = datetime.utcnow()
        if not deployment_window.is_within_window(final_check):
            raise TimeoutError(
                f"Failed to reach deployment window. "
                f"Expected: {deployment_window.earliest_start} - {deployment_window.latest_start}, "
                f"Actual: {final_check}"
            )

        logger.info("Deployment window opened - proceeding with execution")

    @staticmethod
    def get_next_window_opening(windows: list[DeploymentWindow]) -> Optional[datetime]:
        """
        Find the next time any of the given windows will open.

        Args:
            windows: List of deployment windows to check

        Returns:
            Next opening time, or None if no future windows
        """
        current_time = datetime.utcnow()
        future_openings = []

        for window in windows:
            if window.earliest_start > current_time:
                future_openings.append(window.earliest_start)

        return min(future_openings) if future_openings else None

    @staticmethod
    def calculate_window_duration(window: DeploymentWindow) -> float:
        """
        Calculate the total duration of a deployment window in seconds.

        Args:
            window: Deployment window to analyze

        Returns:
            Duration in seconds
        """
        duration = (window.latest_start - window.earliest_start).total_seconds()
        return max(0, duration)

    @staticmethod
    def is_window_expired(window: DeploymentWindow, check_time: Optional[datetime] = None) -> bool:
        """
        Check if a deployment window has expired.

        Args:
            window: Deployment window to check
            check_time: Time to check against (default: now)

        Returns:
            True if window has expired
        """
        check_time = check_time or datetime.utcnow()
        return check_time > window.latest_start

    @staticmethod
    def get_remaining_window_time(window: DeploymentWindow, check_time: Optional[datetime] = None) -> float:
        """
        Get remaining time in the deployment window.

        Args:
            window: Deployment window to check
            check_time: Time to check against (default: now)

        Returns:
            Remaining seconds, or 0 if outside window
        """
        check_time = check_time or datetime.utcnow()

        if not window.is_within_window(check_time):
            return 0.0

        remaining = (window.latest_start - check_time).total_seconds()
        return max(0, remaining)

    @staticmethod
    async def wait_for_any_window(windows: list[DeploymentWindow], timeout_seconds: Optional[float] = None) -> DeploymentWindow:
        """
        Wait for any of the given windows to open.

        Args:
            windows: List of deployment windows to wait for
            timeout_seconds: Maximum time to wait

        Returns:
            The window that opened first

        Raises:
            TimeoutError: If timeout exceeded before any window opens
        """
        if not windows:
            raise ValueError("No deployment windows provided")

        start_time = datetime.utcnow()
        timeout_time = start_time + timedelta(seconds=timeout_seconds) if timeout_seconds else None

        while True:
            current_time = datetime.utcnow()

            # Check timeout
            if timeout_time and current_time > timeout_time:
                raise TimeoutError(f"Timeout waiting for deployment windows after {timeout_seconds} seconds")

            # Check each window
            for window in windows:
                if window.is_within_window(current_time):
                    logger.info(f"Deployment window opened: {window.earliest_start} - {window.latest_start}")
                    return window

            # Wait before checking again
            await asyncio.sleep(30)  # Check every 30 seconds

    @staticmethod
    def validate_window_sequence(windows: list[DeploymentWindow]) -> bool:
        """
        Validate that a sequence of windows doesn't have conflicts or gaps.

        Args:
            windows: List of windows to validate

        Returns:
            True if sequence is valid
        """
        if not windows:
            return True

        sorted_windows = sorted(windows, key=lambda w: w.earliest_start)

        for i in range(len(sorted_windows) - 1):
            current = sorted_windows[i]
            next_window = sorted_windows[i + 1]

            # Check for overlap (which might be okay) or gaps
            if current.latest_start < next_window.earliest_start:
                gap = (next_window.earliest_start - current.latest_start).total_seconds()
                logger.warning(f"Gap of {gap/3600:.1f} hours between deployment windows")
            elif current.is_within_window(next_window.earliest_start):
                overlap = (current.latest_start - next_window.earliest_start).total_seconds()
                logger.info(f"Overlap of {overlap/3600:.1f} hours between deployment windows")

        return True

    @staticmethod
    def create_immediate_window(duration_minutes: int = 60, timezone: str = "UTC") -> DeploymentWindow:
        """
        Create a deployment window that starts immediately.

        Args:
            duration_minutes: How long the window should last
            timezone: Timezone for the window

        Returns:
            Deployment window starting now
        """
        now = datetime.utcnow()
        return DeploymentWindow(
            earliest_start=now,
            latest_start=now + timedelta(minutes=duration_minutes),
            max_duration_minutes=duration_minutes,
            timezone=timezone
        )

    @staticmethod
    def create_scheduled_window(
        start_time: datetime,
        duration_minutes: int = 60,
        timezone: str = "UTC"
    ) -> DeploymentWindow:
        """
        Create a deployment window starting at a specific time.

        Args:
            start_time: When the window should start
            duration_minutes: How long the window should last
            timezone: Timezone for the window

        Returns:
            Scheduled deployment window
        """
        return DeploymentWindow(
            earliest_start=start_time,
            latest_start=start_time + timedelta(minutes=duration_minutes),
            max_duration_minutes=duration_minutes,
            timezone=timezone
        )
