# Copyright 2026 Dropbox, Inc.
# Author: Andrew Yates <ayates@dropbox.com>
# Licensed under the Apache License, Version 2.0

"""
limiter.py - Rate limit types and thresholds for gh_rate_limit.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

from ai_template_scripts.shared_logging import debug_log as _shared_debug_log

__all__ = [
    "RateLimitInfo",
    "UsageStats",
    "_get_thresholds",
    "THRESHOLDS",
    "debug_log",
]

# Module-specific debug env var (AIT_DEBUG also works via shared_logging)
_MODULE_DEBUG_VAR = "AIT_GH_DEBUG"


def debug_log(msg: str) -> None:
    """Log debug message if AIT_GH_DEBUG or AIT_DEBUG is set (#1736, #2007).

    Silent exception handlers in gh_rate_limit make debugging difficult.
    This provides optional visibility controlled by env var.

    Usage:
        debug_log(f"cache read failed: {e}")

    Enable with: export AIT_GH_DEBUG=1  (or AIT_DEBUG=1 for all modules)
    """
    _shared_debug_log("gh_rate_limit", msg, module_env_var=_MODULE_DEBUG_VAR)


# Per-resource thresholds (remaining count)
# Search is 30/min, others are 5000/hr - need different thresholds
THRESHOLDS = {
    "core": {"warning": 500, "critical": 100, "block": 10},
    "graphql": {"warning": 500, "critical": 100, "block": 10},
    "search": {"warning": 10, "critical": 5, "block": 2},
    "default": {"warning": 500, "critical": 100, "block": 10},
}


def _get_thresholds(resource: str) -> dict[str, int]:
    """Get thresholds for a resource."""
    return THRESHOLDS.get(resource, THRESHOLDS["default"])


@dataclass
class RateLimitInfo:
    """Rate limit status for a single resource category."""

    resource: str
    limit: int
    remaining: int
    reset_timestamp: int

    @property
    def percent_remaining(self) -> float:
        return (self.remaining / self.limit * 100) if self.limit > 0 else 0

    @property
    def minutes_until_reset(self) -> float:
        return max(0, (self.reset_timestamp - time.time()) / 60)

    def is_critical(self) -> bool:
        t = _get_thresholds(self.resource)
        return self.remaining < t["critical"]

    def is_warning(self) -> bool:
        t = _get_thresholds(self.resource)
        return self.remaining < t["warning"]

    def should_block(self) -> bool:
        t = _get_thresholds(self.resource)
        return self.remaining < t["block"]


@dataclass
class UsageStats:
    """Usage velocity statistics for a resource."""

    resource: str
    velocity_per_min: float  # Calls per minute (negative = consumption)
    time_to_exhaustion_min: float | None  # Minutes until 0, None if not depleting
    time_to_reset_min: float  # Minutes until quota resets

    @property
    def is_depleting_before_reset(self) -> bool:
        """True if we'll run out before reset."""
        if self.time_to_exhaustion_min is None:
            return False
        return self.time_to_exhaustion_min < self.time_to_reset_min
