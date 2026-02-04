# Copyright 2026 Dropbox, Inc.
# Author: Andrew Yates <ayates@dropbox.com>
# Licensed under the Apache License, Version 2.0

"""
gh_rate_limit - GitHub API Rate Limit Manager Package

This package provides transparent rate limiting and caching of GitHub CLI calls.
Used by bin/gh wrapper - AIs don't interact with this directly.

Public API (library usage):
    from ai_template_scripts.gh_rate_limit import (
        RateLimiter,       # Main rate limiter with caching
        RateLimitInfo,     # Rate limit status dataclass
        UsageStats,        # Usage statistics dataclass
        ChangeLog,         # Pending changes log
        Change,            # Single queued change
        HISTORICAL_DIR,    # Path to persistent cache
        get_limiter,       # Get singleton RateLimiter instance
        get_rate_limits,   # Get current rate limit status
        get_usage_stats,   # Get recent usage statistics
        get_change_log,    # Get pending offline changes
        batch_issue_view,  # Fetch multiple issues in one query
        batch_issue_timelines,  # Fetch multiple issue timeline counts
    )

Migration Status (designs/2026-02-01-rate-limiter-decomposition.md):

Phase 1 (DONE - file-level split):
- changelog.py: Change, ChangeLog, get_change_log
- limiter.py: RateLimitInfo, UsageStats, thresholds
- batch.py: Batch issue fetch helpers
- cache.py: TtlCache, CACHE_TTLS, INVALIDATES
- historical.py: HistoricalCache, extract_issue_info_from_args
- rest_fallback.py: IssueRestFallback, helper functions
- serialize.py: SerializedFetcher, lock functions

Phase 2 (DONE - component composition):
- repo_context.py: RepoContext - repo resolution and gh binary location
- rate_state.py: RateState - rate limit tracking and quota management
- rate_limiter.py: RateLimiter facade composing all components
"""

from __future__ import annotations

from ai_template_scripts.gh_rate_limit.batch import (
    batch_issue_timelines,
    batch_issue_view,
)
from ai_template_scripts.gh_rate_limit.cache import (
    CACHE_TTLS,
    INVALIDATES,
    SEARCH_CACHE_TTL,
    TtlCache,
    format_stale_warning,
)

# Import from extracted modules
from ai_template_scripts.gh_rate_limit.changelog import (
    CHANGE_LOG_FILE,
    CHANGE_LOG_PRUNE_AGE_HOURS,
    MAX_CHANGE_LOG_ENTRIES,
    Change,
    ChangeLog,
    get_change_log,
)
from ai_template_scripts.gh_rate_limit.historical import (
    HistoricalCache,
    extract_issue_info_from_args,
)
from ai_template_scripts.gh_rate_limit.limiter import (
    RateLimitInfo,
    UsageStats,
    _get_thresholds,
    debug_log,
)

# Phase 2: Import refactored RateLimiter that composes extracted components
from ai_template_scripts.gh_rate_limit.rate_limiter import (
    CACHE_DIR,
    HISTORICAL_DIR,
    LOAD_BALANCE_THRESHOLD,
    RateLimiter,
    get_limiter,
    get_rate_limits,
    get_usage_stats,
)
from ai_template_scripts.gh_rate_limit.rate_state import RateState
from ai_template_scripts.gh_rate_limit.repo_context import RepoContext
from ai_template_scripts.gh_rate_limit.rest_fallback import (
    DEFAULT_LOAD_BALANCE_THRESHOLD,
    IssueRestFallback,
    extract_json_fields,
    extract_repo_from_args,
    has_json_flag,
    is_any_rate_limit_error,
    is_graphql_rate_limit_error,
    is_issue_list_command,
    is_issue_search_command,
    is_issue_view_command,
)
from ai_template_scripts.gh_rate_limit.secondary_backoff import (
    SecondaryRateLimitBackoff,
    get_backoff_state,
    is_secondary_rate_limit_error,
    parse_retry_after,
    reset_backoff_state,
)
from ai_template_scripts.gh_rate_limit.serialize import (
    SERIALIZE_TIMEOUT_SEC,
    SerializedFetcher,
    cleanup_stale_locks,
    lock_file,
    try_lock_file,
    unlock_file,
)

__all__ = [
    # Core classes
    "RateLimiter",
    "RateLimitInfo",
    "UsageStats",
    # Component classes (Phase 1 & 2 extractions)
    "RepoContext",
    "RateState",
    "TtlCache",
    "HistoricalCache",
    "IssueRestFallback",
    "SerializedFetcher",
    # Change tracking
    "Change",
    "ChangeLog",
    # Singletons and accessors
    "get_limiter",
    "get_rate_limits",
    "get_usage_stats",
    "get_change_log",
    # Batch operations
    "batch_issue_view",
    "batch_issue_timelines",
    # Paths
    "CACHE_DIR",
    "HISTORICAL_DIR",
    # Constants
    "CACHE_TTLS",
    "CHANGE_LOG_FILE",
    "CHANGE_LOG_PRUNE_AGE_HOURS",
    "DEFAULT_LOAD_BALANCE_THRESHOLD",
    "INVALIDATES",
    "LOAD_BALANCE_THRESHOLD",
    "MAX_CHANGE_LOG_ENTRIES",
    "SEARCH_CACHE_TTL",
    "SERIALIZE_TIMEOUT_SEC",
    # Helper functions
    "cleanup_stale_locks",
    "extract_issue_info_from_args",
    "extract_json_fields",
    "extract_repo_from_args",
    "format_stale_warning",
    "has_json_flag",
    "is_any_rate_limit_error",
    "is_graphql_rate_limit_error",
    "is_issue_list_command",
    "is_issue_search_command",
    "is_issue_view_command",
    "is_secondary_rate_limit_error",
    "lock_file",
    "parse_retry_after",
    "try_lock_file",
    "unlock_file",
    # Secondary rate limit backoff
    "SecondaryRateLimitBackoff",
    "get_backoff_state",
    "reset_backoff_state",
    # Internal (exported for testing)
    "_get_thresholds",
    # Debug utilities
    "debug_log",
]
