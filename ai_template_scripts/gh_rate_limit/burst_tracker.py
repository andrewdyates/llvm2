# Copyright 2026 Dropbox, Inc.
# Author: Andrew Yates <ayates@dropbox.com>
# Licensed under the Apache License, Version 2.0

"""
gh_rate_limit/burst_tracker.py - Burst Rate Detection and Auto-Throttling

Detects when a single session makes too many API calls in a short window
and applies progressive delays. Prevents AIs from needing to add `sleep`
in their own loops (#3220).

Shared across both rate-limited path (gh_wrapper/RateLimiter) and identity
injection path (gh_post) via a common file-based call log.

Public API:
    from ai_template_scripts.gh_rate_limit.burst_tracker import (
        BurstTracker,        # Main tracker class
        get_burst_tracker,   # Get singleton instance
    )
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

from ai_template_scripts.shared_logging import debug_swallow

# Burst detection defaults
BURST_WINDOW_SEC = 60  # Window to count calls in
BURST_THRESHOLD = 20  # Calls in window before throttling
BURST_DELAY_BASE_SEC = 3.0  # Initial delay when throttling
BURST_DELAY_MAX_SEC = 15.0  # Maximum delay between calls
BURST_LOG_MAX_ENTRIES = 200  # Max entries in call log (rolling window)

# Call log file lives in the shared gh cache directory
CALL_LOG_FILENAME = "burst_call_log.jsonl"


class BurstTracker:
    """Tracks API call frequency and applies auto-throttle when bursting.

    Uses a JSONL file as shared state so all gh processes (wrapper and
    gh_post) see the same call history. Each entry is a timestamp line.

    Args:
        cache_dir: Directory for the call log file.
        window_sec: Time window for burst detection.
        threshold: Number of calls in window before throttling activates.
        delay_base: Initial delay in seconds when throttling.
        delay_max: Maximum delay in seconds.
    """

    def __init__(
        self,
        cache_dir: Path,
        *,
        window_sec: int = BURST_WINDOW_SEC,
        threshold: int = BURST_THRESHOLD,
        delay_base: float = BURST_DELAY_BASE_SEC,
        delay_max: float = BURST_DELAY_MAX_SEC,
    ) -> None:
        self.cache_dir = cache_dir
        self.window_sec = window_sec
        self.threshold = threshold
        self.delay_base = delay_base
        self.delay_max = delay_max
        self._log_path = cache_dir / CALL_LOG_FILENAME

    def record_and_maybe_throttle(self) -> None:
        """Record a call and sleep if burst threshold is exceeded.

        Call this before executing any gh command. If the call rate exceeds
        the threshold, this method sleeps for a progressive delay and emits
        a warning to stderr.
        """
        now = time.time()
        timestamps = self._read_timestamps(now)

        # Count calls in the current window
        window_start = now - self.window_sec
        recent_count = sum(1 for t in timestamps if t >= window_start)

        if recent_count >= self.threshold:
            # Calculate progressive delay: scales with how far over threshold
            excess = recent_count - self.threshold
            delay = min(
                self.delay_base + (excess * 0.5),
                self.delay_max,
            )
            print(
                f"[gh] Burst rate detected ({recent_count} calls in {self.window_sec}s), "
                f"throttling {delay:.1f}s",
                file=sys.stderr,
            )
            time.sleep(delay)

        # Record this call
        self._append_timestamp(now)

    def _read_timestamps(self, now: float) -> list[float]:
        """Read recent timestamps from the call log.

        Only returns timestamps within 2x the window (for pruning efficiency).
        """
        cutoff = now - (self.window_sec * 2)
        timestamps: list[float] = []

        try:
            if not self._log_path.exists():
                return timestamps
            with open(self._log_path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        t = float(line)
                        if t >= cutoff:
                            timestamps.append(t)
                    except (ValueError, TypeError):
                        continue
        except OSError:
            debug_swallow("burst_read_log")

        return timestamps

    def _append_timestamp(self, now: float) -> None:
        """Append a timestamp and prune old entries if needed."""
        try:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

            # Read existing, prune, and rewrite if too large
            timestamps = self._read_timestamps(now)
            timestamps.append(now)

            if len(timestamps) > BURST_LOG_MAX_ENTRIES:
                # Prune to keep only recent entries
                cutoff = now - (self.window_sec * 2)
                timestamps = [t for t in timestamps if t >= cutoff]

                # Rewrite entire file
                with open(self._log_path, "w") as f:
                    for t in timestamps:
                        f.write(f"{t:.3f}\n")
            else:
                # Append only
                with open(self._log_path, "a") as f:
                    f.write(f"{now:.3f}\n")
        except OSError:
            debug_swallow("burst_write_log")

    def get_recent_count(self) -> int:
        """Get the number of calls in the current window. For diagnostics."""
        now = time.time()
        timestamps = self._read_timestamps(now)
        window_start = now - self.window_sec
        return sum(1 for t in timestamps if t >= window_start)


# Singleton
_tracker: BurstTracker | None = None


def get_burst_tracker(cache_dir: Path | None = None) -> BurstTracker:
    """Get singleton BurstTracker instance.

    Args:
        cache_dir: Override cache directory. Defaults to ~/.ait_gh_cache/.
    """
    global _tracker
    if _tracker is None:
        if cache_dir is None:
            cache_dir = Path.home() / ".ait_gh_cache"
        _tracker = BurstTracker(cache_dir)
    return _tracker
