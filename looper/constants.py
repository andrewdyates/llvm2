# Copyright 2026 Your Name
# Author: Your Name
# Licensed under the Apache License, Version 2.0

# Copyright 2026 Dropbox, Inc.
# Author: Andrew Yates <ayates@dropbox.com>
# Licensed under the Apache License, Version 2.0

"""
Shared constants for looper modules.

This module provides canonical definitions of constants used across
multiple looper modules to avoid duplication and circular imports.

Part of #1972: Exit code constants consolidation.
Part of #556: Magic constants consolidation.
"""

from pathlib import Path

__all__ = [
    # Exit codes
    "EXIT_TIMEOUT",
    "EXIT_SILENCE",
    "EXIT_NOT_INITIALIZED",
    # Directories
    "LOG_DIR",
    "METRICS_DIR",
    "FLAGS_DIR",
    "ROLES_DIR",
    # Limits
    "MAX_LOG_FILES",
    "MAX_CRASH_LOG_LINES",
    "LOG_RETENTION_HOURS",
    "MAX_METRICS_LINES",
]

# --- Exit Codes ---
# Each code indicates a specific termination reason for telemetry categorization.
EXIT_TIMEOUT = 124  # AI tool timed out
EXIT_SILENCE = 125  # AI tool exceeded silence threshold
# 126 removed - early abort replaced by Maintenance Mode (#2410)
EXIT_NOT_INITIALIZED = 127  # Repo not initialized (no VISION.md + no issues)

# --- Directory Paths ---
# Central definitions for directories used by looper.
# Use these instead of inline Path() definitions to ensure consistency.
LOG_DIR = Path("worker_logs")
METRICS_DIR = Path("metrics")
FLAGS_DIR = Path(".flags")
ROLES_DIR = Path(".claude/roles")

# --- Limits ---
# Retention and size limits for logs and metrics.
MAX_LOG_FILES = 50  # Max worker log files retained
MAX_CRASH_LOG_LINES = 500  # Max lines in crash log excerpts
LOG_RETENTION_HOURS = 24  # Always keep logs from this window
MAX_METRICS_LINES = 5000  # Max lines in metrics file before rotation
