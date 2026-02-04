# Copyright 2026 Dropbox, Inc.
# Author: Andrew Yates <ayates@dropbox.com>
# Licensed under the Apache License, Version 2.0

"""
secondary_backoff.py - Secondary Rate Limit Handling with Exponential Backoff

GitHub's secondary rate limits (aka abuse limits) require special handling:
- Wait at least 60 seconds (or Retry-After header value if available)
- Apply exponential backoff for repeated failures
- Emit progress to stderr every 60 seconds (looper silence timeout)

See: https://docs.github.com/en/rest/using-the-rest-api/rate-limits-for-the-rest-api

Public API:
    from ai_template_scripts.gh_rate_limit.secondary_backoff import (
        is_secondary_rate_limit_error,   # Detect secondary rate limit in output
        parse_retry_after,               # Extract wait time from error message
        SecondaryRateLimitBackoff,       # Backoff state tracker
    )
"""

from __future__ import annotations

import random
import re
import sys
import time
from dataclasses import dataclass

__all__ = [
    "is_secondary_rate_limit_error",
    "parse_retry_after",
    "SecondaryRateLimitBackoff",
    "get_backoff_state",
    "reset_backoff_state",
]

# Default wait time per GitHub docs
DEFAULT_SECONDARY_WAIT_SEC = 60

# Maximum backoff time (10 minutes)
MAX_BACKOFF_SEC = 600

# Jitter factor (0.1 = ±10% randomization)
JITTER_FACTOR = 0.1


def is_secondary_rate_limit_error(output: str) -> bool:
    """Check if output indicates secondary (abuse) rate limiting.

    REQUIRES: output is a string (may be empty)
    ENSURES: Returns True if output contains secondary rate limit indicators

    Secondary rate limits are distinct from primary rate limits:
    - Primary: "API rate limit exceeded" with X-RateLimit-* headers
    - Secondary: "secondary rate limit" with Retry-After header
    """
    if not output:
        return False
    lower = output.lower()
    return "secondary rate limit" in lower or "abuse detection" in lower


def parse_retry_after(output: str) -> int | None:
    """Parse Retry-After value from error message.

    REQUIRES: output is a string (may be empty)
    ENSURES: Returns wait time in seconds, or None if not found

    GitHub CLI doesn't expose HTTP headers, but sometimes the error message
    contains timing hints like "try again in 60 seconds".
    """
    if not output:
        return None

    # Pattern: "try again in N seconds" or "retry after N seconds"
    patterns = [
        r"try again in (\d+)\s*(?:second|sec|s)",
        r"retry after (\d+)\s*(?:second|sec|s)",
        r"wait (\d+)\s*(?:second|sec|s)",
        r"(\d+)\s*(?:second|sec|s)\s*before retry",
    ]

    lower = output.lower()
    for pattern in patterns:
        match = re.search(pattern, lower)
        if match:
            return int(match.group(1))

    return None


@dataclass
class SecondaryRateLimitBackoff:
    """Track secondary rate limit backoff state.

    Usage:
        backoff = SecondaryRateLimitBackoff()
        while True:
            result = make_api_call()
            if is_secondary_rate_limit_error(result.stderr):
                wait_time = backoff.get_wait_time(result.stderr)
                backoff.wait_with_progress(wait_time)
                continue
            else:
                backoff.reset()  # Success - reset backoff
                break
    """

    consecutive_failures: int = 0
    base_wait_sec: int = DEFAULT_SECONDARY_WAIT_SEC

    def get_wait_time(self, error_output: str = "") -> float:
        """Calculate wait time with exponential backoff.

        REQUIRES: error_output is a string (may be empty)
        ENSURES: Returns wait time in seconds (between base and MAX_BACKOFF_SEC)
        """
        # Try to parse Retry-After from error message
        retry_after = parse_retry_after(error_output)
        if retry_after is not None:
            base = retry_after
        else:
            base = self.base_wait_sec

        # Apply exponential backoff for consecutive failures
        # Formula: base * 2^failures (capped at MAX_BACKOFF_SEC)
        multiplier = 2 ** self.consecutive_failures
        wait = min(base * multiplier, MAX_BACKOFF_SEC)

        # Add jitter to prevent thundering herd
        jitter = wait * JITTER_FACTOR * (2 * random.random() - 1)
        wait += jitter

        # Ensure we wait at least the base time
        return max(wait, self.base_wait_sec)

    def wait_with_progress(self, wait_sec: float) -> None:
        """Wait with progress output to prevent looper timeout.

        REQUIRES: wait_sec > 0
        ENSURES: Emits progress to stderr at least every 60 seconds

        Raises:
            ValueError: If wait_sec <= 0
        """
        if wait_sec <= 0:
            raise ValueError(f"wait_sec must be > 0, got {wait_sec}")

        remaining = wait_sec

        print(
            f"gh_rate_limit: secondary rate limit, waiting {wait_sec:.0f}s "
            f"(attempt {self.consecutive_failures + 1})",
            file=sys.stderr,
        )

        while remaining > 0:
            # Sleep in 60-second chunks to emit progress
            sleep_time = min(60.0, remaining)
            time.sleep(sleep_time)
            remaining -= sleep_time

            if remaining > 0:
                print(
                    f"gh_rate_limit: still waiting, {remaining:.0f}s remaining...",
                    file=sys.stderr,
                )

        self.consecutive_failures += 1

    def reset(self) -> None:
        """Reset backoff state after successful request."""
        self.consecutive_failures = 0


# Module-level singleton for use across requests
_backoff_state: SecondaryRateLimitBackoff | None = None


def get_backoff_state() -> SecondaryRateLimitBackoff:
    """Get global backoff state singleton.

    REQUIRES: None
    ENSURES: Returns a SecondaryRateLimitBackoff instance (creates if needed)

    Creates a new instance if none exists. Allows backoff to persist
    across multiple API calls within the same process.
    """
    global _backoff_state
    if _backoff_state is None:
        _backoff_state = SecondaryRateLimitBackoff()
    return _backoff_state


def reset_backoff_state() -> None:
    """Reset global backoff state.

    REQUIRES: None
    ENSURES: Global backoff state has consecutive_failures = 0 (if state exists)

    Call after successful API request to clear consecutive failure count.
    """
    global _backoff_state
    if _backoff_state is not None:
        _backoff_state.reset()
