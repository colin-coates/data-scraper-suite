# Copyright (c) 2024 Mountain Jewels Intelligence. All rights reserved.
#
# This software is proprietary and confidential. Unauthorized copying,
# modification, distribution, or use is strictly prohibited.

"""
MJ payload builders for person and event data.
"""

def build_person_payload(raw):
    """
    Format raw scraper output into the MJ Person schema.
    """
    return {
        "first_name": raw.get("first_name"),
        "last_name": raw.get("last_name"),
        "age": raw.get("age"),
        "age_band": raw.get("age_band"),
        "gender": raw.get("gender"),
        "location": raw.get("location"),
        "wealth_tier": raw.get("wealth_tier"),
        "relationship_status": raw.get("relationship_status"),
        "event_history": raw.get("event_history", []),
        "confidence": raw.get("confidence", 0.5),
    }

def build_event_payload(raw):
    """
    Format raw scraper output into the MJ Event schema.
    """
    return {
        "event_type": raw.get("event_type"),
        "event_date": raw.get("event_date"),
        "location": raw.get("location"),
        "participants": raw.get("participants", []),
        "urgency_score": raw.get("urgency_score"),
        "wealth_signal": raw.get("wealth_signal"),
        "sentiment": raw.get("sentiment"),
        "confidence": raw.get("confidence", 0.5),
    }
