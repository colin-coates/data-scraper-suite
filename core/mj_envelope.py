# Copyright (c) 2024 Mountain Jewels Intelligence. All rights reserved.
#
# This software is proprietary and confidential. Unauthorized copying,
# modification, distribution, or use is strictly prohibited.

"""
Unified MJ envelope for person/event ingestion.
"""

class MJMessageEnvelope:
    """
    Unified MJ envelope for person/event ingestion.
    """

    def __init__(self, data_type, payload, correlation_id=None, version="1.0"):
        self.data_type = data_type
        self.payload = payload
        self.version = version
        self.source = "data-scraper-suite"
        self.correlation_id = correlation_id
        self.timestamp = None  # to be filled at send time

    def to_dict(self):
        return {
            "data_type": self.data_type,
            "payload": self.payload,
            "version": self.version,
            "source": self.source,
            "correlation_id": self.correlation_id,
            "timestamp": self.timestamp,
        }
