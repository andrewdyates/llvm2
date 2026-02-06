# Copyright 2026 Your Name
# Author: Your Name
# Licensed under the Apache License, Version 2.0

# Copyright 2026 Dropbox, Inc.
# Author: Andrew Yates <ayates@dropbox.com>
# Licensed under the Apache License, Version 2.0

"""
gh_rate_limit/rate_limiter.py - Rate Limiter Facade

Orchestrates rate limiting, caching, serialization, and REST fallback.
This is the refactored RateLimiter that composes extracted components.

Part of gh_rate_limit decomposition (designs/2026-02-01-rate-limiter-decomposition.md).

Phase 2: The RateLimiter is now a thin facade that delegates to:
- RepoContext: repo resolution and gh binary location
- RateState: rate limit tracking and quota management
- TtlCache: TTL-based response caching
- HistoricalCache: persistent issue cache
- SerializedFetcher: request serialization
- IssueRestFallback: REST API fallback
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING

from ai_template_scripts.gh_rate_limit.cache import (
    TtlCache,
    format_stale_warning,
)

# Import changelog for pending issue integration (#1854)
from ai_template_scripts.gh_rate_limit.changelog import Change, ChangeLog
from ai_template_scripts.gh_rate_limit.historical import (
    HistoricalCache,
    extract_issue_info_from_args,
)
from ai_template_scripts.gh_rate_limit.limiter import debug_log
from ai_template_scripts.gh_rate_limit.rate_state import RateState
from ai_template_scripts.gh_rate_limit.repo_context import RepoContext
from ai_template_scripts.gh_rate_limit.rest_fallback import (
    IssueRestFallback,
    extract_json_fields,
    extract_repo_from_args,
    has_json_flag,
    is_graphql_rate_limit_error,
    is_issue_list_command,
    is_issue_search_command,
    is_issue_view_command,
)
from ai_template_scripts.gh_rate_limit.secondary_backoff import (
    get_backoff_state,
    is_secondary_rate_limit_error,
    reset_backoff_state,
)
from ai_template_scripts.gh_rate_limit.serialize import (
    SerializedFetcher,
    cleanup_stale_locks,
)

if TYPE_CHECKING:
    from ai_template_scripts.gh_rate_limit.limiter import RateLimitInfo, UsageStats
    from ai_template_scripts.result import Result

# Paths
CACHE_DIR = Path.home() / ".ait_gh_cache"
HISTORICAL_DIR = CACHE_DIR / "historical"

# Proactive load balancing thresholds (#1143, #1502)
# When GraphQL remaining drops below this AND REST has more headroom, prefer REST
# 3500 = 70% of 5000 limit; triggers load balancing earlier to prevent exhaustion
LOAD_BALANCE_THRESHOLD = int(os.environ.get("AIT_GH_LOAD_BALANCE_THRESHOLD", "3500"))


class RateLimiter:
    """GitHub API rate limit manager with caching and velocity tracking.

    Used by bin/gh wrapper to transparently handle rate limits and caching.
    This is a facade that composes specialized components for each responsibility.
    """

    def __init__(self, cache_dir: Path | None = None) -> None:
        self.cache_dir = cache_dir or CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Initialize components
        self._repo_context = RepoContext()
        self._rate_state = RateState(
            cache_dir=self.cache_dir,
            get_real_gh=self._repo_context.get_real_gh,
            get_commit_hash=self._repo_context.get_commit_hash,
        )
        self._ttl_cache = TtlCache(
            cache_dir=self.cache_dir,
            get_repo=self._repo_context.get_repo,
        )
        self._historical_cache = HistoricalCache(cache_dir=self.cache_dir)
        self._serializer = SerializedFetcher(
            cache_dir=self.cache_dir,
            get_repo=self._repo_context.get_repo,
            get_ttl=self._ttl_cache.get_ttl,
        )
        self._rest_fallback = IssueRestFallback(
            get_real_gh=self._repo_context.get_real_gh,
            get_owner_repo=self._repo_context.get_owner_repo,
            rate_cache=self._rate_state.rate_cache,
            load_balance_threshold=LOAD_BALANCE_THRESHOLD,
        )
        # ChangeLog for pending issue integration (#1854)
        self._change_log = ChangeLog(cache_dir=self.cache_dir)

        # Throttle stale lock cleanup to once per hour
        self._last_lock_cleanup: float = 0

    # Backward compatibility properties for tests
    @property
    def _rate_cache(self) -> dict[str, RateLimitInfo]:
        """Backward-compatible access to rate cache."""
        return self._rate_state.rate_cache

    @_rate_cache.setter
    def _rate_cache(self, value: dict[str, RateLimitInfo]) -> None:
        """Backward-compatible setter for rate cache.

        Updates in place to maintain references held by delegates.
        """
        self._rate_state._rate_cache.clear()
        self._rate_state._rate_cache.update(value)

    @property
    def _usage_log(self) -> list[dict]:
        """Backward-compatible access to usage log."""
        return self._rate_state._usage_log

    @_usage_log.setter
    def _usage_log(self, value: list[dict]) -> None:
        """Backward-compatible setter for usage log."""
        self._rate_state._usage_log = value

    @property
    def _last_rate_check(self) -> float:
        """Backward-compatible access to last rate check timestamp."""
        return self._rate_state._last_rate_check

    @_last_rate_check.setter
    def _last_rate_check(self, value: float) -> None:
        """Backward-compatible setter for last rate check timestamp."""
        self._rate_state._last_rate_check = value

    @property
    def _repo(self) -> str | None:
        """Backward-compatible access to repo cache."""
        return self._repo_context._repo

    @_repo.setter
    def _repo(self, value: str | None) -> None:
        """Backward-compatible setter for repo cache."""
        self._repo_context._repo = value

    def _maybe_cleanup_stale_locks(self) -> None:
        """Opportunistic cleanup of stale lock files, throttled to once per hour."""
        if time.time() - self._last_lock_cleanup > 3600:
            self._last_lock_cleanup = time.time()
            try:
                self.cleanup_stale_locks()
            except Exception as e:
                debug_log(f"_maybe_cleanup_stale_locks failed: {e}")

    # --- Delegated properties and methods for backward compatibility ---

    def _get_real_gh(self) -> str:
        """Find the real gh binary."""
        return self._repo_context.get_real_gh()

    def _get_repo(self) -> str:
        """Get current repo identifier for cache namespacing."""
        return self._repo_context.get_repo()

    def _normalize_repo(self, repo: str) -> str | None:
        """Normalize repo input to owner/repo format."""
        return self._repo_context.normalize_repo(repo)

    def _get_owner_repo(self, repo_override: str | None = None) -> str | None:
        """Get owner/repo from override or git origin."""
        return self._repo_context.get_owner_repo(repo_override)

    def _get_commit_hash(self) -> str | None:
        """Get current HEAD commit hash."""
        return self._repo_context.get_commit_hash()

    # Rate state delegation
    def get_usage_stats(self, resource: str = "core") -> UsageStats | None:
        """Calculate usage velocity and time to exhaustion for a resource."""
        return self._rate_state.get_usage_stats(resource)

    def fetch_rate_limits(self) -> dict[str, RateLimitInfo]:
        """Fetch current rate limits. Cached for 60s."""
        return self._rate_state.fetch_rate_limits()

    def check_rate_limit(self, args: list[str]) -> bool:
        """Check if call is allowed. Returns False if blocked."""
        return self._rate_state.check_rate_limit(args)

    # Cache delegation
    def _cache_key(self, args: list[str]) -> str:
        """Generate cache key including repo context."""
        return self._ttl_cache.cache_key(args)

    def _get_ttl(self, args: list[str]) -> int | None:
        """Get cache TTL for command, or None if not cacheable."""
        return self._ttl_cache.get_ttl(args)

    def _is_write(self, args: list[str]) -> bool:
        """Check if command is a write operation."""
        return self._ttl_cache.is_write(args)

    def get_cached(self, args: list[str]) -> str | None:
        """Get cached response if valid. Returns stdout string or None."""
        return self._ttl_cache.get_cached(args)

    def get_stale_cached(self, args: list[str]) -> tuple[str | None, float | None]:
        """Get cached response even if expired. Returns (stdout, age_sec)."""
        return self._ttl_cache.get_stale_cached(args)

    def set_cached(self, args: list[str], stdout: str, etag: str | None = None) -> None:
        """Cache a response with optional ETag."""
        self._ttl_cache.set_cached(args, stdout, etag)
        # Also store in historical cache for individual issue views
        if is_issue_view_command(args):
            self._store_historical_issue(args, stdout)

    def get_cached_etag(self, args: list[str]) -> str | None:
        """Get cached ETag for conditional request."""
        return self._ttl_cache.get_cached_etag(args)

    def extend_cache_ttl(self, args: list[str]) -> bool:
        """Extend cache TTL by updating created_at timestamp."""
        return self._ttl_cache.extend_cache_ttl(args)

    def _invalidate_cache(self, args: list[str], repo: str | None = None) -> None:
        """Invalidate cache entries affected by a write operation.

        Args:
            args: Command arguments that triggered the write
            repo: Target repo for cross-repo operations (#2066). Can be owner/repo
                format (e.g., "dropbox-ai-prototypes/kafka2") or just repo name ("kafka2").
                If not provided, uses current directory's repo.
        """
        # Normalize repo to just the directory name for cache key matching (#2066)
        # Cache stores "kafka2", not "dropbox-ai-prototypes/kafka2"
        if repo and "/" in repo:
            target_repo = repo.split("/")[-1]
        else:
            target_repo = repo or self._get_repo()
        self._ttl_cache.invalidate(
            args,
            target_repo,
            on_historical_invalidate=self._on_historical_invalidate_callback,
        )

    def _on_historical_invalidate_callback(self, args: list[str]) -> None:
        """Callback for TtlCache.invalidate to handle historical cache (#1904)."""
        issue_num, repo_override = extract_issue_info_from_args(args)
        owner_repo = self._get_owner_repo(repo_override)
        if issue_num and owner_repo:
            self._historical_cache.invalidate_issue(owner_repo, issue_num)

    def invalidate_cache(self, args: list[str], repo: str | None = None) -> None:
        """Public cache invalidation method."""
        self._invalidate_cache(args, repo)

    # Historical cache delegation
    def _get_historical_dir(self) -> Path:
        """Get the historical cache directory."""
        return self._historical_cache.get_dir()

    def _get_historical_issue_path(self, owner_repo: str, issue_num: str) -> Path:
        """Get path for a historical issue cache file."""
        return self._historical_cache.get_issue_path(owner_repo, issue_num)

    def _store_historical_issue(self, args: list[str], stdout: str) -> None:
        """Store issue data in historical cache."""
        issue_num, repo_override = extract_issue_info_from_args(args)
        if not issue_num:
            return
        owner_repo = self._get_owner_repo(repo_override)
        if not owner_repo:
            return
        self._historical_cache.store_issue(owner_repo, issue_num, stdout)

    def get_historical_issue(self, owner_repo: str, issue_num: str) -> dict | None:
        """Get issue from historical cache."""
        return self._historical_cache.get_issue(owner_repo, issue_num)

    def _get_historical_age(self, owner_repo: str, issue_num: str) -> float | None:
        """Get age of historical cache entry in seconds."""
        return self._historical_cache.get_age(owner_repo, issue_num)

    def _invalidate_historical_issue(self, owner_repo: str, issue_num: str) -> None:
        """Remove historical cache entry for an issue."""
        self._historical_cache.invalidate_issue(owner_repo, issue_num)

    def cleanup_historical_cache(self, max_age_days: int = 30) -> tuple[int, int]:
        """Clean up old historical cache entries."""
        return self._historical_cache.cleanup(max_age_days)

    def cleanup_stale_locks(self, max_age_hours: int = 24) -> tuple[int, int]:
        """Remove stale lock files from cache directory."""
        return cleanup_stale_locks(self.cache_dir, max_age_hours)

    # Serialization delegation
    def _get_serialize_lock_path(self, args: list[str]) -> Path | None:
        """Get lock file path for serializing API calls."""
        return self._serializer.get_lock_path(args)

    # --- Pending issue integration (#1854) ---
    # Issues created while rate-limited are stored in change_log.json but were
    # invisible to gh issue list/view. These methods merge pending issues into reads.

    def _get_pending_creates_for_repo(self, owner_repo: str) -> list[Change]:
        """Get pending 'create' operations for a specific repo from change_log."""
        pending = self._change_log.get_pending(repo=owner_repo)
        # Also try without owner prefix for legacy entries
        repo_name = owner_repo.split("/")[-1] if "/" in owner_repo else owner_repo
        if repo_name != owner_repo:
            pending.extend(self._change_log.get_pending(repo=repo_name))
        return [c for c in pending if c.operation == "create"]

    def _format_pending_issue_as_json(self, change: Change, fields: list[str]) -> dict:
        """Format a pending issue Change as gh issue list JSON format.

        Pending issues use synthetic IDs: "pending-<uuid>" to distinguish from
        real GitHub issues while still being viewable.
        """
        data = change.data
        # Build a pseudo-issue matching gh issue list --json fields
        issue: dict = {
            "number": f"pending-{change.id}",  # Synthetic ID
            "title": data.get("title", ""),
            "body": data.get("body", ""),
            "state": "PENDING",  # Special state for pending issues
            "labels": [{"name": lbl} for lbl in data.get("labels", [])],
            "createdAt": time.strftime(
                "%Y-%m-%dT%H:%M:%SZ", time.gmtime(change.timestamp)
            ),
            "url": f"pending:{change.id}",  # Placeholder URL
            "author": {"login": "pending"},
        }
        # Filter to requested fields if specified
        if fields:
            return {f: issue.get(f) for f in fields if f in issue}
        return issue

    def _extract_issue_list_filters(
        self, args: list[str]
    ) -> tuple[str, list[str], int | None, bool]:
        """Extract state/label/limit filters from gh issue list args."""
        state = "open"
        labels: list[str] = []
        limit: int | None = None
        has_search = False

        i = 0
        while i < len(args):
            arg = args[i]
            if arg in ("--state", "-s") and i + 1 < len(args):
                state = args[i + 1]
                i += 2
            elif arg.startswith("--state="):
                state = arg.split("=", 1)[1]
                i += 1
            elif arg in ("--label", "-l") and i + 1 < len(args):
                labels.append(args[i + 1])
                i += 2
            elif arg.startswith("--label="):
                labels.append(arg.split("=", 1)[1])
                i += 1
            elif arg in ("--limit", "-L") and i + 1 < len(args):
                try:
                    limit = int(args[i + 1])
                except ValueError:
                    limit = None
                i += 2
            elif arg.startswith("--limit="):
                try:
                    limit = int(arg.split("=", 1)[1])
                except ValueError:
                    limit = None
                i += 1
            elif arg in ("--search", "-S"):
                has_search = True
                i += 2 if i + 1 < len(args) else 1
            elif arg.startswith("--search="):
                has_search = True
                i += 1
            else:
                i += 1

        return state, labels, limit, has_search

    def _merge_pending_into_issue_list(
        self, stdout: str, args: list[str], owner_repo: str
    ) -> str:
        """Merge pending creates into gh issue list JSON output (#1854).

        Pending issues appear at the top of the list with synthetic IDs.
        Only merges for JSON output (--json flag).
        """
        if not has_json_flag(args):
            return stdout  # Non-JSON output, can't merge

        pending = self._get_pending_creates_for_repo(owner_repo)
        if not pending:
            return stdout  # No pending issues

        state, labels, limit, has_search = self._extract_issue_list_filters(args)
        if has_search:
            return stdout  # Search filters are too complex to apply locally
        if state == "closed":
            return stdout  # Pending issues are not closed

        if labels:
            label_set = set(labels)
            pending = [
                change
                for change in pending
                if label_set.issubset(set(change.data.get("labels", [])))
            ]
            if not pending:
                return stdout

        json_fields = extract_json_fields(args)

        try:
            # Parse existing issues
            existing = json.loads(stdout) if stdout.strip() else []
            if not isinstance(existing, list):
                return stdout  # Unexpected format

            # Format pending issues
            pending_issues = [
                self._format_pending_issue_as_json(c, json_fields) for c in pending
            ]

            # Merge: pending first, then existing
            merged = pending_issues + existing
            if limit is not None and limit > 0:
                merged = merged[:limit]

            # Log merge for debugging
            print(
                f"gh_rate_limit: merged {len(pending_issues)} pending issue(s) "
                f"into list results",
                file=sys.stderr,
            )

            return json.dumps(merged) + "\n"
        except (json.JSONDecodeError, TypeError):
            return stdout  # Parse error, return original

    def _get_pending_issue_view(
        self, pending_id: str, args: list[str], owner_repo: str
    ) -> str | None:
        """Get pending issue data for gh issue view pending-<id> (#1854).

        Returns JSON string if found, None otherwise.
        """
        pending = self._get_pending_creates_for_repo(owner_repo)

        for change in pending:
            if change.id == pending_id:
                json_fields = extract_json_fields(args)
                issue_data = self._format_pending_issue_as_json(change, json_fields)
                print(
                    f"gh_rate_limit: returning pending issue {pending_id}",
                    file=sys.stderr,
                )
                return json.dumps(issue_data) + "\n"
        return None

    def _is_pending_issue_view(self, args: list[str]) -> tuple[bool, str | None]:
        """Check if args request viewing a pending issue (pending-<id>).

        Returns (is_pending, pending_id).
        """
        if not is_issue_view_command(args):
            return False, None

        issue_num, _ = extract_issue_info_from_args(args)
        if issue_num and issue_num.startswith("pending-"):
            return True, issue_num[8:]  # Strip "pending-" prefix
        return False, None

    def _maybe_merge_pending_issues(
        self, result: subprocess.CompletedProcess, args: list[str]
    ) -> subprocess.CompletedProcess:
        """Merge pending issues into gh issue list results if applicable (#1854).

        Returns modified result with pending issues prepended to the list.
        Only modifies successful issue list results with JSON output.
        """
        if result.returncode != 0:
            return result
        if not is_issue_list_command(args):
            return result
        if not has_json_flag(args):
            return result

        repo_override = extract_repo_from_args(args)
        owner_repo = self._get_owner_repo(repo_override)
        if not owner_repo:
            return result

        merged_stdout = self._merge_pending_into_issue_list(
            result.stdout, args, owner_repo
        )
        if merged_stdout == result.stdout:
            return result  # No changes

        return subprocess.CompletedProcess(
            args=result.args,
            returncode=result.returncode,
            stdout=merged_stdout,
            stderr=result.stderr,
        )

    # Command parsing helpers (delegated to rest_fallback module)
    def _is_graphql_rate_limit_error(self, output: str) -> bool:
        """Check if output indicates GraphQL rate limiting."""
        return is_graphql_rate_limit_error(output)

    def _has_json_flag(self, args: list[str]) -> bool:
        """Check if args request JSON output."""
        return has_json_flag(args)

    def _is_issue_list_command(self, args: list[str]) -> bool:
        """Check if args represent a gh issue list command."""
        return is_issue_list_command(args)

    def _is_issue_view_command(self, args: list[str]) -> bool:
        """Check if args represent a gh issue view command."""
        return is_issue_view_command(args)

    def _is_issue_search_command(self, args: list[str]) -> bool:
        """Check if args represent a gh issue list command with --search."""
        return is_issue_search_command(args)

    def _extract_json_fields(self, args: list[str]) -> list[str]:
        """Extract --json field list from command args."""
        return extract_json_fields(args)

    def _extract_issue_num_from_args(self, args: list[str]) -> str | None:
        """Extract issue number from command args."""
        issue_num, _ = extract_issue_info_from_args(args)
        return issue_num

    def _extract_repo_from_args(self, args: list[str]) -> str | None:
        """Extract repo override from command args."""
        return extract_repo_from_args(args)

    # REST fallback delegation
    def _should_prefer_rest_for_quota(self, log: bool = True) -> bool:
        """Check if REST should be preferred based on quota levels."""
        return self._rest_fallback.should_prefer_rest_for_quota(log)

    def _prefer_issue_rest(self, args: list[str]) -> bool:
        """Prefer REST for routine issue reads to reduce GraphQL usage."""
        return self._rest_fallback.prefer_issue_rest(args)

    def _is_silent_list_rate_limit(
        self, result: subprocess.CompletedProcess, args: list[str]
    ) -> bool:
        """Detect silent rate limit failure for gh issue list."""
        return self._rest_fallback.is_silent_list_rate_limit(result, args)

    # --- Composite methods that use multiple components ---

    def _format_stale_warning(
        self, source: str, age_sec: float, reason: str | None = None
    ) -> str:
        """Format a consistent stale data warning for stderr."""
        return format_stale_warning(source, age_sec, reason)

    def get_cached_or_historical(self, args: list[str]) -> str | None:
        """Get cached response, falling back to historical cache."""
        # Try fresh cache first
        cached = self.get_cached(args)
        if cached is not None:
            return cached

        # For issue view commands, try historical cache
        if not is_issue_view_command(args):
            return None

        issue_num, repo_override = extract_issue_info_from_args(args)
        if not issue_num or not issue_num.isdigit():
            return None

        owner_repo = self._get_owner_repo(repo_override)
        if not owner_repo:
            return None

        # Extract requested JSON fields from args (#1148)
        json_fields = self._extract_json_fields(args)

        historical = self.get_historical_issue(owner_repo, issue_num)
        if historical is not None:
            # Validate that historical cache contains all requested fields
            if json_fields:
                missing_fields = [f for f in json_fields if f not in historical]
                if missing_fields:
                    return None
                # Filter to only requested fields (#1751)
                output = {f: historical[f] for f in json_fields if f in historical}
            else:
                output = historical
            print(
                f"gh_rate_limit: using historical cache for issue {issue_num}",
                file=sys.stderr,
            )
            # Add trailing newline to prevent stderr interleaving (#1722)
            return json.dumps(output) + "\n"

        return None

    def get_stale_fallback(
        self, args: list[str], reason: str
    ) -> subprocess.CompletedProcess | None:
        """Get stale data with warning when API is unavailable (#1135)."""
        # Try expired TTL cache first
        stale_data, age_sec = self.get_stale_cached(args)
        if stale_data is not None and age_sec is not None:
            warning = self._format_stale_warning("cache", age_sec, reason)
            print(warning, file=sys.stderr)
            return subprocess.CompletedProcess(
                args=["gh"] + args,
                returncode=0,
                stdout=stale_data,
                stderr=warning,
            )

        # Try historical cache for issue views
        if is_issue_view_command(args):
            issue_num, repo_override = extract_issue_info_from_args(args)
            if issue_num and issue_num.isdigit():
                owner_repo = self._get_owner_repo(repo_override)
                if owner_repo:
                    json_fields = self._extract_json_fields(args)
                    historical = self.get_historical_issue(owner_repo, issue_num)
                    if historical is not None:
                        if json_fields:
                            missing = [f for f in json_fields if f not in historical]
                            if missing:
                                return None
                            # Filter to only requested fields (#1751)
                            output = {
                                f: historical[f] for f in json_fields if f in historical
                            }
                        else:
                            output = historical

                        hist_age = self._get_historical_age(owner_repo, issue_num)
                        if hist_age is not None:
                            warning = self._format_stale_warning(
                                "historical cache", hist_age, reason
                            )
                        else:
                            warning = f"gh_rate_limit: [STALE] historical ({reason})"
                        print(warning, file=sys.stderr)
                        # Add trailing newline to prevent stderr interleaving (#1722)
                        hist_stdout = json.dumps(output) + "\n"
                        return subprocess.CompletedProcess(
                            args=["gh"] + args,
                            returncode=0,
                            stdout=hist_stdout,
                            stderr=warning,
                        )

        return None

    def _serialized_fetch(
        self, args: list[str], timeout: float
    ) -> subprocess.CompletedProcess | None:
        """Fetch with serialization to prevent thundering herd.

        Includes secondary rate limit handling with exponential backoff (#2299).
        """
        lock_fd, got_lock = self._serializer.acquire_lock(args)
        if lock_fd is None:
            return None  # Not a serializable command

        try:
            if not got_lock:
                # Timeout waiting for lock - check if cache was updated
                cached = self.get_cached(args)
                if cached is not None:
                    return subprocess.CompletedProcess(
                        args=["gh"] + args,
                        returncode=0,
                        stdout=cached,
                        stderr="",
                    )
                stale_result = self.get_stale_fallback(args, "serialize timeout")
                if stale_result is not None:
                    return stale_result
                print(
                    "gh_rate_limit: serialize timeout, proceeding concurrently",
                    file=sys.stderr,
                )
                return None

            # Got lock - re-check cache (double-checked locking pattern)
            cached = self.get_cached(args)
            if cached is not None:
                return subprocess.CompletedProcess(
                    args=["gh"] + args,
                    returncode=0,
                    stdout=cached,
                    stderr="",
                )

            # Execute with secondary rate limit retry (#2299)
            # Max 3 retries for secondary rate limits
            max_secondary_retries = 3
            for attempt in range(max_secondary_retries + 1):
                # Execute the command
                try:
                    result = subprocess.run(
                        [self._get_real_gh()] + args,
                        capture_output=True,
                        text=True,
                        timeout=timeout,
                        check=False,
                    )
                except subprocess.TimeoutExpired:
                    stale_result = self.get_stale_fallback(args, "API timeout")
                    if stale_result is not None:
                        return stale_result
                    return subprocess.CompletedProcess(
                        args=["gh"] + args,
                        returncode=-1,
                        stdout="",
                        stderr=f"Timeout after {timeout}s",
                    )

                # Check for rate limit error - try REST fallback before stale
                if result.returncode != 0:
                    error_output = (result.stderr or "") + (result.stdout or "")

                    # Check for secondary rate limit (#2299)
                    if is_secondary_rate_limit_error(error_output):
                        if attempt < max_secondary_retries:
                            # Wait with backoff and retry
                            backoff = get_backoff_state()
                            wait_time = backoff.get_wait_time(error_output)
                            backoff.wait_with_progress(wait_time)
                            continue
                        # Max retries exceeded - fall through to stale fallback

                    if self._is_graphql_rate_limit_error(error_output):
                        # Try REST fallback first
                        fallback_process = None
                        # Check for search command - uses separate search API (#1777)
                        # Search API has 30/min limit - check quota before fallback
                        if is_issue_search_command(args):
                            if not self._rest_fallback.has_search_quota():
                                debug_log(
                                    "search quota exhausted, skipping REST fallback"
                                )
                                # Skip to stale fallback below
                            else:
                                fallback_result = self._issue_search_rest_fallback(
                                    args, timeout
                                )
                                if fallback_result.value is not None:
                                    if (
                                        fallback_result.ok
                                        and fallback_result.value.returncode == 0
                                    ):
                                        self.set_cached(
                                            args, fallback_result.value.stdout
                                        )
                                        reset_backoff_state()  # Success - reset backoff
                                        return fallback_result.value
                                    fallback_process = fallback_result.value
                                if fallback_result.skipped:
                                    debug_log(
                                        f"issue_search: REST skipped "
                                        f"({fallback_result.error})"
                                    )
                                elif not fallback_result.ok:
                                    print(
                                        "gh_rate_limit: REST search fallback failed "
                                        f"({fallback_result.error})",
                                        file=sys.stderr,
                                    )
                        elif is_issue_list_command(args):
                            fallback_result = self._issue_list_rest_fallback(
                                args, timeout
                            )
                            if fallback_result.value is not None:
                                if (
                                    fallback_result.ok
                                    and fallback_result.value.returncode == 0
                                ):
                                    self.set_cached(args, fallback_result.value.stdout)
                                    reset_backoff_state()  # Success - reset backoff
                                    return fallback_result.value
                                # Capture error response for fallback (#1754)
                                fallback_process = fallback_result.value
                            # Log REST failure for debugging (#1728, #1737)
                            if fallback_result.skipped:
                                if (
                                    fallback_result.error
                                    and not fallback_result.error.startswith(
                                        "unsupported_flag:"
                                    )
                                ):
                                    print(
                                        "gh_rate_limit: REST fallback skipped "
                                        f"({fallback_result.error})",
                                        file=sys.stderr,
                                    )
                            elif not fallback_result.ok:
                                print(
                                    "gh_rate_limit: REST fallback failed "
                                    f"({fallback_result.error})",
                                    file=sys.stderr,
                                )
                        elif is_issue_view_command(args):
                            fallback_result = self._issue_view_rest_fallback(
                                args, timeout
                            )
                            if fallback_result.value is not None:
                                if (
                                    fallback_result.ok
                                    and fallback_result.value.returncode == 0
                                ):
                                    reset_backoff_state()  # Success - reset backoff
                                    return fallback_result.value
                                # Capture error response for fallback (#1754)
                                fallback_process = fallback_result.value
                            # Log REST failure for debugging (#1756)
                            if fallback_result.skipped:
                                if (
                                    fallback_result.error
                                    and not fallback_result.error.startswith(
                                        "unsupported_flag:"
                                    )
                                ):
                                    print(
                                        "gh_rate_limit: REST fallback skipped "
                                        f"({fallback_result.error})",
                                        file=sys.stderr,
                                    )
                            elif not fallback_result.ok:
                                print(
                                    "gh_rate_limit: REST fallback failed "
                                    f"({fallback_result.error})",
                                    file=sys.stderr,
                                )
                        # REST fallback failed or not applicable - try stale
                        stale_result = self.get_stale_fallback(args, "rate limited")
                        if stale_result is not None:
                            return stale_result
                        # No stale - return REST error if we have one (#1754)
                        if fallback_process is not None:
                            return fallback_process
                    else:
                        # Non-rate-limit error - try stale fallback (#1135, #1689)
                        stale_result = self.get_stale_fallback(args, "API error")
                        if stale_result is not None:
                            return stale_result

                # Success or non-retryable error - exit retry loop
                break

            # Cache successful result and reset backoff state
            if result.returncode == 0:
                self.set_cached(args, result.stdout)
                reset_backoff_state()  # Reset on success

            return result

        finally:
            self._serializer.release_lock(lock_fd)

    # --- ETag conditional requests ---

    def _parse_etag_from_headers(self, response: str) -> str | None:
        """Parse ETag from response headers."""
        parts = response.split("\r\n\r\n", 1)
        if len(parts) < 2:
            parts = response.split("\n\n", 1)
        if len(parts) < 2:
            return None

        headers = parts[0]
        for line in headers.split("\n"):
            line = line.strip()
            if line.lower().startswith("etag:"):
                return line.split(":", 1)[1].strip()
        return None

    def _parse_status_from_headers(self, response: str) -> int | None:
        """Parse HTTP status code from response headers."""
        lines = response.split("\n")
        if not lines:
            return None
        first_line = lines[0].strip()
        parts = first_line.split()
        if len(parts) >= 2:
            try:
                return int(parts[1])
            except ValueError:
                return None
        return None

    def _parse_body_from_response(self, response: str) -> str:
        """Extract body from response with headers."""
        parts = response.split("\r\n\r\n", 1)
        if len(parts) == 2:
            return parts[1]
        parts = response.split("\n\n", 1)
        if len(parts) == 2:
            return parts[1]
        return response

    def api_with_etag(
        self, api_path: str, args: list[str], timeout: float
    ) -> tuple[subprocess.CompletedProcess, str | None]:
        """Make REST API call with optional ETag conditional request."""
        cached_etag = self.get_cached_etag(args)

        api_args = ["api", "-i", api_path]
        if cached_etag:
            api_args.extend(["-H", f"If-None-Match: {cached_etag}"])

        try:
            result = subprocess.run(
                [self._get_real_gh()] + api_args,
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
            )

            if result.returncode != 0:
                return result, None

            status = self._parse_status_from_headers(result.stdout)

            if status == 304:
                self.extend_cache_ttl(args)
                cached_data = self.get_cached(args)
                if cached_data is None:
                    return subprocess.CompletedProcess(
                        args=result.args,
                        returncode=0,
                        stdout="",
                        stderr="304 but no cached data found",
                    ), None
                # Buffer diagnostic in result.stderr to prevent interleaving (#1773)
                etag_msg = "gh_rate_limit: ETag 304 Not Modified (no quota consumed)\n"
                etag_stderr = (result.stderr or "") + etag_msg
                return subprocess.CompletedProcess(
                    args=result.args,
                    returncode=0,
                    stdout=cached_data,
                    stderr=etag_stderr,
                ), cached_etag

            if status is not None and status != 200:
                body = self._parse_body_from_response(result.stdout)
                return subprocess.CompletedProcess(
                    args=result.args,
                    returncode=1,
                    stdout=body,
                    stderr=f"HTTP {status}",
                ), None

            new_etag = self._parse_etag_from_headers(result.stdout)
            body = self._parse_body_from_response(result.stdout)

            return subprocess.CompletedProcess(
                args=result.args,
                returncode=0,
                stdout=body,
                stderr=result.stderr,
            ), new_etag

        except subprocess.TimeoutExpired:
            return subprocess.CompletedProcess(
                args=["gh"] + api_args,
                returncode=-1,
                stdout="",
                stderr=f"API request timeout after {timeout}s",
            ), None
        except Exception as e:
            return subprocess.CompletedProcess(
                args=["gh"] + api_args,
                returncode=-1,
                stdout="",
                stderr=str(e),
            ), None

    def _try_etag_conditional_issue_view(
        self, args: list[str], timeout: float
    ) -> subprocess.CompletedProcess | None:
        """Try ETag conditional request for gh issue view commands."""
        if not is_issue_view_command(args):
            return None

        issue_num, repo_override = extract_issue_info_from_args(args)
        if not issue_num or not issue_num.isdigit():
            return None

        owner_repo = self._get_owner_repo(repo_override)
        if not owner_repo:
            return None

        cached_etag = self.get_cached_etag(args)
        if not cached_etag:
            return None

        api_path = f"/repos/{owner_repo}/issues/{issue_num}"
        result, new_etag = self.api_with_etag(api_path, args, timeout)

        if result.returncode != 0:
            return None

        if new_etag and new_etag != cached_etag:
            self.set_cached(args, result.stdout, etag=new_etag)

        return result

    # --- REST fallback methods ---

    def _issue_list_rest_fallback(
        self, args: list[str], timeout: float
    ) -> Result[subprocess.CompletedProcess]:
        """REST fallback for gh issue list.

        Delegates to IssueRestFallback component for full implementation.
        """
        result = self._rest_fallback.issue_list_rest_fallback(args, timeout)
        if result.error:
            debug_log(f"issue_list_rest: {result.status}: {result.error}")
        return result

    def _issue_view_rest_fallback(
        self, args: list[str], timeout: float
    ) -> Result[subprocess.CompletedProcess]:
        """REST fallback for gh issue view.

        Delegates to IssueRestFallback component for full implementation.
        """
        result = self._rest_fallback.issue_view_rest_fallback(
            args,
            timeout,
            parse_etag=self._parse_etag_from_headers,
            parse_body=self._parse_body_from_response,
            set_cached=self.set_cached,
        )
        if result.error:
            debug_log(f"issue_view_rest: {result.status}: {result.error}")
        return result

    def _issue_search_rest_fallback(
        self, args: list[str], timeout: float
    ) -> Result[subprocess.CompletedProcess]:
        """REST fallback for gh issue list --search (#1777).

        Delegates to IssueRestFallback component for full implementation.
        """
        result = self._rest_fallback.issue_search_rest_fallback(args, timeout)
        if result.error:
            debug_log(f"issue_search_rest: {result.status}: {result.error}")
        return result

    # --- Unified REST fallback helper ---

    def _try_rest_and_cache(
        self, args: list[str], timeout: float, merge_pending: bool = True
    ) -> subprocess.CompletedProcess | None:
        """Try REST fallback for issue commands and cache on success.

        Unified handler for search/list/view REST fallback. Reduces code
        duplication from call() (#2333).

        Args:
            args: Command arguments
            timeout: Request timeout
            merge_pending: Whether to merge pending issues (for list commands)

        Returns:
            CompletedProcess on success, None if REST not applicable or failed
        """
        if is_issue_search_command(args):
            rest_result = self._issue_search_rest_fallback(args, timeout)
            if rest_result.ok and rest_result.value and rest_result.value.returncode == 0:
                self.set_cached(args, rest_result.value.stdout)
                return rest_result.value
        elif is_issue_list_command(args):
            rest_result = self._issue_list_rest_fallback(args, timeout)
            if rest_result.ok and rest_result.value and rest_result.value.returncode == 0:
                self.set_cached(args, rest_result.value.stdout)
                if merge_pending:
                    return self._maybe_merge_pending_issues(rest_result.value, args)
                return rest_result.value
        elif is_issue_view_command(args):
            rest_result = self._issue_view_rest_fallback(args, timeout)
            if rest_result.ok and rest_result.value and rest_result.value.returncode == 0:
                return rest_result.value
        return None

    def _try_rest_after_rate_limit(
        self, args: list[str], timeout: float
    ) -> subprocess.CompletedProcess | None:
        """Try REST fallback after GraphQL rate limit, with error capture.

        Unlike _try_rest_and_cache, this logs failures and captures error
        responses for fallback. Used in post-execution rate limit handling.

        Returns:
            CompletedProcess on success or error, None to fall through to stale
        """
        if is_issue_search_command(args):
            if not self._rest_fallback.has_search_quota():
                debug_log("search quota exhausted, skipping REST fallback")
                return None
            rest_result = self._issue_search_rest_fallback(args, timeout)
            if rest_result.value:
                if rest_result.ok and rest_result.value.returncode == 0:
                    self.set_cached(args, rest_result.value.stdout)
                    return rest_result.value
                return rest_result.value  # Return error response
            if rest_result.skipped:
                debug_log(f"issue_search: REST skipped ({rest_result.error})")
            elif not rest_result.ok:
                debug_log(f"issue_search: REST failed ({rest_result.error})")
        elif is_issue_list_command(args):
            rest_result = self._issue_list_rest_fallback(args, timeout)
            if rest_result.value:
                if rest_result.ok and rest_result.value.returncode == 0:
                    self.set_cached(args, rest_result.value.stdout)
                    return self._maybe_merge_pending_issues(rest_result.value, args)
                return rest_result.value
            if rest_result.skipped:
                debug_log(f"issue_list: REST skipped ({rest_result.error})")
            elif not rest_result.ok:
                debug_log(f"issue_list: REST failed ({rest_result.error})")
        elif is_issue_view_command(args):
            rest_result = self._issue_view_rest_fallback(args, timeout)
            if rest_result.value:
                if rest_result.ok and rest_result.value.returncode == 0:
                    return rest_result.value
                return rest_result.value
            if rest_result.skipped:
                debug_log(f"issue_view: REST skipped ({rest_result.error})")
            elif not rest_result.ok:
                debug_log(f"issue_view: REST failed ({rest_result.error})")
        return None

    def _handle_pending_issue_view(
        self, args: list[str]
    ) -> subprocess.CompletedProcess | None:
        """Handle synthetic pending issue views (#1854).

        Returns:
            None if args is not a pending issue view (PENDINGxxx format)
            CompletedProcess with returncode=0 if pending issue found
            CompletedProcess with returncode=1 if pending issue not found
        """
        is_pending, pending_id = self._is_pending_issue_view(args)
        if not (is_pending and pending_id):
            return None

        repo_override = extract_repo_from_args(args)
        owner_repo = self._get_owner_repo(repo_override)
        if owner_repo:
            pending_data = self._get_pending_issue_view(pending_id, args, owner_repo)
            if pending_data:
                return subprocess.CompletedProcess(
                    args=["gh"] + args,
                    returncode=0,
                    stdout=pending_data,
                    stderr="",
                )
        # Pending issue not found
        return subprocess.CompletedProcess(
            args=["gh"] + args,
            returncode=1,
            stdout="",
            stderr=f"Pending issue {pending_id} not found in change_log",
        )

    def _handle_rate_limit_block(
        self, args: list[str]
    ) -> subprocess.CompletedProcess | None:
        """Handle rate limit exceeded before command execution.

        Returns:
            CompletedProcess error if blocked, None to continue execution
        """
        if self.check_rate_limit(args):
            return None  # Rate limit OK, continue

        # Check if REST fallback is available for read commands
        if self._prefer_issue_rest(args) and not self._is_write(args):
            core_info = self._rate_state.rate_cache.get("core")
            if core_info and core_info.remaining > 0:
                print(
                    "gh_rate_limit: GraphQL rate-limited, using REST fallback",
                    file=sys.stderr,
                )
                return None  # Fall through to REST handling

            # REST also exhausted - try stale fallback
            stale_result = self.get_stale_fallback(args, "rate limited")
            if stale_result is not None:
                return stale_result
            return subprocess.CompletedProcess(
                args=["gh"] + args,
                returncode=1,
                stdout="",
                stderr="Rate limit exceeded (GraphQL and REST exhausted)",
            )

        # Non-REST read or write command
        if not self._is_write(args):
            stale_result = self.get_stale_fallback(args, "rate limited")
            if stale_result is not None:
                return stale_result

        return subprocess.CompletedProcess(
            args=["gh"] + args,
            returncode=1,
            stdout="",
            stderr="Rate limit exceeded",
        )

    def _handle_read_request(
        self, args: list[str], timeout: float
    ) -> subprocess.CompletedProcess | None:
        """Handle read request with cache, REST, ETag, and serialized fetch.

        Returns:
            CompletedProcess on cache hit or successful fetch, None to fall through
        """
        # Check cache (includes historical fallback for issue views)
        cached = self.get_cached_or_historical(args)
        if cached is not None:
            cached_result = subprocess.CompletedProcess(
                args=["gh"] + args,
                returncode=0,
                stdout=cached,
                stderr="",
            )
            return self._maybe_merge_pending_issues(cached_result, args)

        # Prefer REST when GraphQL quota is low (#1074)
        if self._prefer_issue_rest(args):
            rest_result = self._try_rest_and_cache(args, timeout)
            if rest_result is not None:
                return rest_result

        # Try ETag conditional request (#1674)
        etag_result = self._try_etag_conditional_issue_view(args, timeout)
        if etag_result is not None:
            return etag_result

        # Use serialized fetch to prevent thundering herd
        serialized_result = self._serialized_fetch(args, timeout)
        if serialized_result is not None:
            return self._maybe_merge_pending_issues(serialized_result, args)

        return None

    def _execute_with_retry(
        self, args: list[str], timeout: float
    ) -> subprocess.CompletedProcess:
        """Execute gh command with secondary rate limit retry (#2299).

        Returns:
            CompletedProcess from subprocess execution, stale fallback, or timeout error
        """
        max_secondary_retries = 3
        for attempt in range(max_secondary_retries + 1):
            try:
                result = subprocess.run(
                    [self._get_real_gh()] + args,
                    capture_output=True,
                    text=True,
                    timeout=timeout,
                    check=False,
                )
            except subprocess.TimeoutExpired:
                if not self._is_write(args):
                    stale_result = self.get_stale_fallback(args, "API timeout")
                    if stale_result is not None:
                        return stale_result
                return subprocess.CompletedProcess(
                    args=["gh"] + args,
                    returncode=-1,
                    stdout="",
                    stderr=f"Timeout after {timeout}s",
                )

            # Check for secondary rate limit and retry with backoff
            if result.returncode != 0:
                error_output = (result.stderr or "") + (result.stdout or "")
                if is_secondary_rate_limit_error(error_output):
                    if attempt < max_secondary_retries:
                        backoff = get_backoff_state()
                        wait_time = backoff.get_wait_time(error_output)
                        backoff.wait_with_progress(wait_time)
                        continue

            # Success or non-retryable error
            break

        return result

    def _handle_post_execution(
        self, result: subprocess.CompletedProcess, args: list[str], timeout: float
    ) -> subprocess.CompletedProcess:
        """Handle post-execution caching, rate limit fallback, and error recovery.

        Returns:
            Final CompletedProcess result
        """
        error_output = (result.stderr or "") + (result.stdout or "")
        is_rate_limited = (
            result.returncode != 0 and self._is_graphql_rate_limit_error(error_output)
        ) or self._is_silent_list_rate_limit(result, args)

        # Try REST fallback on rate limit (#1728)
        if is_rate_limited:
            fallback_result = self._try_rest_after_rate_limit(args, timeout)
            if fallback_result is not None:
                return fallback_result
            # REST failed or not applicable - try stale
            if not self._is_write(args):
                stale_result = self.get_stale_fallback(args, "rate limited")
                if stale_result is not None:
                    return stale_result

        # Cache successful reads
        if result.returncode == 0 and not self._is_write(args):
            self.set_cached(args, result.stdout)
            reset_backoff_state()

        # Invalidate cache on successful writes
        if result.returncode == 0 and self._is_write(args):
            self._invalidate_cache(args)
            reset_backoff_state()

        # Stale fallback for failed reads (#1135)
        if result.returncode != 0 and not self._is_write(args):
            stale_result = self.get_stale_fallback(args, "API error")
            if stale_result is not None:
                return stale_result

        return self._maybe_merge_pending_issues(result, args)

    # --- Main entry point ---

    def call(self, args: list[str], timeout: float = 30) -> subprocess.CompletedProcess:
        """Execute gh command with rate limiting and caching.

        This is the main entry point used by bin/gh wrapper.
        Orchestrates: pending views, rate limits, caching, REST fallback,
        subprocess execution, and post-execution handling.
        """
        # Opportunistic lock cleanup, throttled to once per hour (#1605)
        self._maybe_cleanup_stale_locks()

        # Handle pending issue views (#1854)
        pending_result = self._handle_pending_issue_view(args)
        if pending_result is not None:
            return pending_result

        # Check rate limit before execution
        rate_limit_result = self._handle_rate_limit_block(args)
        if rate_limit_result is not None:
            return rate_limit_result

        # Handle read requests with cache/REST/ETag/serialization
        if not self._is_write(args):
            read_result = self._handle_read_request(args, timeout)
            if read_result is not None:
                return read_result

        # Try REST for non-cacheable reads that fell through (#1074)
        # Note: _prefer_issue_rest returns False for write commands, so this
        # only affects reads without --json or when cache/ETag/serialized paths failed
        if self._prefer_issue_rest(args):
            rest_result = self._try_rest_and_cache(args, timeout, merge_pending=True)
            if rest_result is not None:
                return rest_result

        # Execute subprocess with retry
        result = self._execute_with_retry(args, timeout)

        # Handle post-execution caching, fallback, and error recovery
        return self._handle_post_execution(result, args, timeout)


# Singleton for bin/gh wrapper
_limiter: RateLimiter | None = None


def get_limiter() -> RateLimiter:
    """Get singleton limiter instance."""
    global _limiter
    if _limiter is None:
        _limiter = RateLimiter()
    return _limiter


def get_rate_limits() -> dict[str, RateLimitInfo]:
    """Get current rate limits. For Manager health checks."""
    return get_limiter().fetch_rate_limits()


def get_usage_stats(resource: str = "core") -> UsageStats | None:
    """Get usage velocity stats for a resource."""
    return get_limiter().get_usage_stats(resource)
