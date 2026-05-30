"""
Common types and enums for production modules.
Centralized to avoid duplication.
"""
from enum import Enum


class Severity(Enum):
    """Severity levels for health checks and incidents."""
    INFO = "INFO"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"
    EMERGENCY = "EMERGENCY"
