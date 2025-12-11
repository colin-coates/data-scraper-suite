# Copyright (c) 2024 Mountain Jewels Intelligence. All rights reserved.
#
# This software is proprietary and confidential. Unauthorized copying,
# modification, distribution, or use is strictly prohibited.

"""
MJ Data Scraper Suite

Enterprise web scraping platform with anti-detection capabilities,
plugin architecture, and comprehensive data collection.
"""

__version__ = "1.0.0"
__author__ = "Mountain Jewels Intelligence"
__email__ = "engineering@mountainjewels.com"

from .scraper_engine import ScraperEngine
from .core.base_scraper import BaseScraper
from .anti_detection.anti_detection import AntiDetectionLayer

__all__ = [
    "ScraperEngine",
    "BaseScraper",
    "AntiDetectionLayer",
]
