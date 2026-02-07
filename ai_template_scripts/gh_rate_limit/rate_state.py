# Copyright 2026 Dropbox, Inc.
# Author: Andrew Yates <ayates@dropbox.com>
# Licensed under the Apache License, Version 2.0

"""
gh_rate_limit/rate_state.py - Rate Limit State Management

Manages rate-cache, usage log, and check/wait logic.
Part of gh_rate_limit decomposition (designs/2026-02-01-rate-limiter-decomposition.md).

Methods extracted from RateLimiter:
- _load_rate_state, _save_rate_state, fetch_rate_limits
- check_rate_limit, get_usage_stats, _get_resource

Per-App Rate Tracking:
When AIT_USE_GITHUB_APPS=1, rate limits are tracked per-app. Each GitHub App
has independent 5000/hr quota. The current app is detected via AIT_GH_APP_ACTIVE
env var (set by gh wrapper when using app token).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from collections.abc import Callable
from pathlib import Path

from ai_template_scripts.gh_rate_limit.limiter import (
    RateLimitInfo,
    UsageStats,
    debug_log,
)
from ai_template_scripts.shared_logging import debug_swallow
from ai_template_scripts.gh_rate_limit.rest_fallback import (
    has_json_flag,
    is_issue_list_command,
    is_issue_search_command,
    is_issue_view_command,
)
from ai_template_scripts.gh_rate_limit.serialize import lock_file, unlock_file

# Velocity tracking
MAX_USAGE_LOG_ENTRIES = 60  # Keep ~1 hour of samples at 1/min

# Rate state TTL - cached quota data expires after this many seconds
# Also respects reset_timestamp - if reset has passed, data is stale
RATE_STATE_TTL_SECONDS = 300  # 5 minutes

# Overflow tracking - log file for quota overflow events
# Pulse reads this to alert Manager when overflows occur
OVERFLOW_LOG_FILE = "overflow_log.json"
MAX_OVERFLOW_LOG_ENTRIES = 100  # Keep last 100 overflow events


class RateState:
    """Rate limit state manager (#1135).

    Tracks API quota usage, persists state for cross-process visibility,
    and implements blocking/waiting logic when quota is critical.

    Supports per-app rate tracking when GitHub Apps are enabled. Each app
    has independent 5000/hr quota tracked separately.

    Args:
        cache_dir: Directory for state file.
        get_real_gh: Callable returning path to real gh binary.
        get_commit_hash: Callable returning current commit hash.
    """

    # Default identity when not using GitHub Apps
    DEFAULT_IDENTITY = "gh_auth"

    def __init__(
        self,
        cache_dir: Path,
        get_real_gh: Callable[[], str],
        get_commit_hash: Callable[[], str | None],
    ) -> None:
        self.cache_dir = cache_dir
        self._get_real_gh = get_real_gh
        self._get_commit_hash = get_commit_hash
        self._rate_cache: dict[str, RateLimitInfo] = {}
        self._usage_log: list[dict] = []  # [{t, core, graphql, search}, ...]
        self._last_rate_check: float = 0
        # Per-app rate cache: {app_name: {resource: RateLimitInfo}}
        self._app_rate_cache: dict[str, dict[str, RateLimitInfo]] = {}
        # Load existing rate state if fresh
        self._load_rate_state()

    def _get_current_repo(self) -> str | None:
        """Get the current repository name."""
        repo = os.environ.get("AIT_CURRENT_REPO")
        if not repo:
            result = subprocess.run(
                ["git", "rev-parse", "--show-toplevel"],
                capture_output=True,
                text=True,
                timeout=5,
                check=False,
            )
            if result.returncode == 0:
                repo = Path(result.stdout.strip()).name
        return repo

    def _get_current_app(self) -> str:
        """Get the current GitHub App identity.

        Returns app name from gh_apps config if using app token,
        otherwise returns 'gh_auth' for standard authentication.
        """
        if os.environ.get("AIT_GH_APP_ACTIVE") != "1":
            return self.DEFAULT_IDENTITY

        try:
            from ai_template_scripts.gh_apps.selector import get_app_for_repo

            repo = self._get_current_repo()
            if repo:
                app_name = get_app_for_repo(repo)
                if app_name:
                    return app_name
        except ImportError:
            pass

        return self.DEFAULT_IDENTITY

    def get_fallback_apps(self, repo: str | None = None) -> list[str]:
        """Get ordered list of fallback apps for overflow.

        Order: per-repo app → director app → global app (dbx-ai)

        Args:
            repo: Repository name. If None, uses current repo.

        Returns:
            List of app names in priority order.
        """
        if not repo:
            repo = self._get_current_repo()
        if not repo:
            return ["dbx-ai"]

        try:
            from ai_template_scripts.gh_apps.selector import (
                KNOWN_DIRECTOR_APPS,
                load_config,
            )

            apps: list[str] = []
            config = load_config()
            if not config:
                return ["dbx-ai"]

            per_repo_app = None
            director_app = None
            wildcard_app = None

            for app_name, app_config in config.apps.items():
                if "*" in app_config.repos:
                    wildcard_app = app_name
                    continue
                if repo not in app_config.repos:
                    continue
                if app_name in KNOWN_DIRECTOR_APPS:
                    director_app = app_name
                else:
                    per_repo_app = app_name

            if per_repo_app:
                apps.append(per_repo_app)
            if director_app:
                apps.append(director_app)
            if wildcard_app:
                apps.append(wildcard_app)

            return apps if apps else ["dbx-ai"]
        except ImportError:
            return ["dbx-ai"]

    def _log_overflow(
        self,
        resource: str,
        from_app: str,
        to_app: str,
        repo: str | None,
    ) -> None:
        """Log an overflow event for Manager visibility via pulse.

        Uses file locking to prevent corruption from concurrent writes.

        Args:
            resource: Rate limit resource (core, graphql, search).
            from_app: Primary app that was exhausted.
            to_app: Fallback app that will be used.
            repo: Repository name.
        """
        overflow_file = self.cache_dir / OVERFLOW_LOG_FILE
        lock_path = self.cache_dir / ".overflow_lock"
        lock_fd = None
        try:
            lock_fd = open(lock_path, "w")
            lock_file(lock_fd)
            try:
                # Load existing log
                if overflow_file.exists():
                    data = json.loads(overflow_file.read_text())
                else:
                    data = {"events": []}

                # Add new event
                data["events"].append({
                    "timestamp": time.time(),
                    "resource": resource,
                    "from_app": from_app,
                    "to_app": to_app,
                    "repo": repo or "unknown",
                })

                # Trim to max entries
                data["events"] = data["events"][-MAX_OVERFLOW_LOG_ENTRIES:]

                # Write back
                overflow_file.write_text(json.dumps(data))
                debug_log(f"overflow logged: {from_app} -> {to_app} for {resource}")
            finally:
                unlock_file(lock_fd)
        except Exception as e:
            debug_log(f"failed to log overflow: {e}")
        finally:
            if lock_fd is not None:
                try:
                    lock_fd.close()
                except OSError:
                    debug_swallow("lock_fd_close")

    def get_best_available_app(
        self, resource: str, repo: str | None = None
    ) -> str | None:
        """Get the best app with available quota for a resource.

        Tries apps in priority order (per-repo → director → global),
        returning the first one with remaining quota. Logs overflow
        events when falling back to lower-priority apps.

        Args:
            resource: Rate limit resource (core, graphql, search).
            repo: Repository name. If None, uses current repo.

        Returns:
            App name with available quota, or None if all exhausted.
        """
        apps = self.get_fallback_apps(repo)
        debug_log(f"get_best_available_app: checking {apps} for {resource}")

        exhausted_apps: list[str] = []

        for app_name in apps:
            app_cache = self._app_rate_cache.get(app_name, {})
            info = app_cache.get(resource)

            # No cached info - assume available
            if not info:
                debug_log(f"  {app_name}: no cached info, assuming available")
                if exhausted_apps:
                    self._log_overflow(resource, exhausted_apps[0], app_name, repo)
                return app_name

            # Stale info - assume available (quota likely reset)
            if not self._is_rate_info_fresh(info):
                debug_log(f"  {app_name}: stale, assuming available")
                if exhausted_apps:
                    self._log_overflow(resource, exhausted_apps[0], app_name, repo)
                return app_name

            if info.remaining > 0:
                debug_log(f"  {app_name}: {info.remaining} remaining")
                if exhausted_apps:
                    self._log_overflow(resource, exhausted_apps[0], app_name, repo)
                return app_name
            else:
                debug_log(f"  {app_name}: exhausted")
                exhausted_apps.append(app_name)

        debug_log("  all apps exhausted")
        return None

    @property
    def rate_cache(self) -> dict[str, RateLimitInfo]:
        """Access to rate cache for components that need it."""
        return self._rate_cache

    def _is_rate_info_fresh(self, info: RateLimitInfo) -> bool:
        """Check if rate limit info is still valid.

        Info is stale if reset_timestamp has passed (quota has reset).
        """
        now = time.time()
        if info.reset_timestamp > 0 and now > info.reset_timestamp:
            return False
        return True

    def _load_rate_state(self) -> None:
        """Load rate state from file if fresh.

        Freshness checks:
        - File timestamp within RATE_STATE_TTL_SECONDS
        - Per-resource reset_timestamp not passed (quota hasn't reset)

        Supports both legacy format (single identity) and new per-app format.
        """
        state_file = self.cache_dir / "rate_state.json"
        if not state_file.exists():
            return
        try:
            data = json.loads(state_file.read_text())
            # Always load usage log (historical data)
            self._usage_log = data.get("usage_log", [])[-MAX_USAGE_LOG_ENTRIES:]

            current_app = self._get_current_app()
            now = time.time()
            file_age = now - data.get("timestamp", 0)

            # Check file-level TTL
            if file_age >= RATE_STATE_TTL_SECONDS:
                debug_log(f"rate_state too old ({file_age:.0f}s), ignoring")
                return

            # Load per-app data if available (new format)
            if "apps" in data:
                self._app_rate_cache = {}
                for app_name, app_data in data.get("apps", {}).items():
                    self._app_rate_cache[app_name] = {}
                    for name, info_data in app_data.items():
                        info = RateLimitInfo(
                            resource=name,
                            limit=info_data.get("limit", 0),
                            remaining=info_data.get("remaining", 0),
                            reset_timestamp=info_data.get("reset", 0),
                        )
                        # Only keep fresh data (reset hasn't passed)
                        if self._is_rate_info_fresh(info):
                            self._app_rate_cache[app_name][name] = info
                        else:
                            debug_log(f"  {app_name}/{name}: stale (reset passed)")

                # Set current app's data as active rate_cache
                if current_app in self._app_rate_cache:
                    self._rate_cache = self._app_rate_cache[current_app]
                else:
                    self._rate_cache = {}
            else:
                # Legacy format: single identity under "resources"
                for name, info_data in data.get("resources", {}).items():
                    info = RateLimitInfo(
                        resource=name,
                        limit=info_data.get("limit", 0),
                        remaining=info_data.get("remaining", 0),
                        reset_timestamp=info_data.get("reset", 0),
                    )
                    if self._is_rate_info_fresh(info):
                        self._rate_cache[name] = info
                # Store in per-app cache for consistency
                self._app_rate_cache[current_app] = self._rate_cache

            self._last_rate_check = data.get("timestamp", 0)
        except Exception as e:
            debug_log(f"_load_rate_state failed: {e}")

    def _save_rate_state(self) -> None:
        """Save rate state for cross-process visibility.

        Saves in per-app format when GitHub Apps are active,
        maintaining backward compatibility with legacy format.
        """
        state_file = self.cache_dir / "rate_state.json"
        lock_path = self.cache_dir / ".lock"
        lock_fd = None
        try:
            lock_fd = open(lock_path, "w")
            lock_file(lock_fd)
            try:
                now = time.time()
                current_app = self._get_current_app()

                # Add usage log entry
                log_entry: dict[str, int | float] = {"t": now, "app": current_app}  # type: ignore[dict-item]
                for name in ["core", "graphql", "search"]:
                    if name in self._rate_cache:
                        log_entry[name] = self._rate_cache[name].remaining
                self._usage_log.append(log_entry)
                # Trim to max entries
                self._usage_log = self._usage_log[-MAX_USAGE_LOG_ENTRIES:]

                # Update per-app cache with current app's data
                self._app_rate_cache[current_app] = self._rate_cache

                # Build state with per-app format
                state: dict = {
                    "timestamp": now,
                    "commit": self._get_commit_hash(),
                    "usage_log": self._usage_log,
                }

                # If using GitHub Apps, save per-app data
                if os.environ.get("AIT_USE_GITHUB_APPS") == "1":
                    state["apps"] = {
                        app_name: {
                            name: {
                                "limit": info.limit,
                                "remaining": info.remaining,
                                "reset": info.reset_timestamp,
                            }
                            for name, info in app_data.items()
                        }
                        for app_name, app_data in self._app_rate_cache.items()
                    }
                else:
                    # Legacy format for backward compatibility
                    state["resources"] = {
                        name: {
                            "limit": info.limit,
                            "remaining": info.remaining,
                            "reset": info.reset_timestamp,
                        }
                        for name, info in self._rate_cache.items()
                    }

                state_file.write_text(json.dumps(state))
            finally:
                unlock_file(lock_fd)
        except Exception as e:
            debug_log(f"_save_rate_state failed: {e}")
        finally:
            if lock_fd is not None:
                try:
                    lock_fd.close()
                except OSError:
                    debug_swallow("lock_fd_close")

    def get_usage_stats(self, resource: str = "core") -> UsageStats | None:
        """Calculate usage velocity and time to exhaustion for a resource.

        Uses linear regression on usage_log to estimate consumption rate.
        Returns None if not enough data points.
        """
        info = self._rate_cache.get(resource)
        if not info:
            return None

        # Need at least 2 data points for velocity
        relevant = [e for e in self._usage_log if resource in e]
        if len(relevant) < 2:
            return UsageStats(
                resource=resource,
                velocity_per_min=0,
                time_to_exhaustion_min=None,
                time_to_reset_min=info.minutes_until_reset,
            )

        # Use last 10 minutes of data for velocity calculation
        now = time.time()
        recent = [e for e in relevant if now - e["t"] < 600]
        if len(recent) < 2:
            recent = relevant[-2:]  # Fall back to last 2 points

        # Calculate velocity (change in remaining per minute)
        oldest, newest = recent[0], recent[-1]
        time_delta_min = (newest["t"] - oldest["t"]) / 60
        if time_delta_min < 0.1:  # Less than 6 seconds
            return UsageStats(
                resource=resource,
                velocity_per_min=0,
                time_to_exhaustion_min=None,
                time_to_reset_min=info.minutes_until_reset,
            )

        remaining_delta = newest[resource] - oldest[resource]
        velocity = remaining_delta / time_delta_min  # Negative = consuming

        # Time to exhaustion (if depleting)
        time_to_exhaustion = None
        if velocity < 0 and info.remaining > 0:
            time_to_exhaustion = -info.remaining / velocity

        return UsageStats(
            resource=resource,
            velocity_per_min=round(velocity, 1),
            time_to_exhaustion_min=round(time_to_exhaustion, 1)
            if time_to_exhaustion
            else None,
            time_to_reset_min=round(info.minutes_until_reset, 1),
        )

    def fetch_rate_limits(self) -> dict[str, RateLimitInfo]:
        """Fetch current rate limits. Cached for 60s."""
        if time.time() - self._last_rate_check < 60 and self._rate_cache:
            return self._rate_cache
        try:
            result = subprocess.run(
                [self._get_real_gh(), "api", "/rate_limit"],
                capture_output=True,
                text=True,
                timeout=30,
                check=False,
            )
            if result.returncode != 0:
                return self._rate_cache
            data = json.loads(result.stdout)
            self._rate_cache = {}
            for name, info in data.get("resources", {}).items():
                self._rate_cache[name] = RateLimitInfo(
                    resource=name,
                    limit=info.get("limit", 0),
                    remaining=info.get("remaining", 0),
                    reset_timestamp=info.get("reset", 0),
                )
            self._last_rate_check = time.time()
            self._save_rate_state()
        except Exception as e:
            print(f"gh_rate_limit: failed to fetch: {e}", file=sys.stderr)
        return self._rate_cache

    def get_resource(self, args: list[str]) -> str:
        """Determine which rate limit resource a command uses.

        The gh CLI internally uses GraphQL for:
        - issue list/view/create/edit/close/reopen
        - pr list/view/create/edit/close/reopen
        - repo view
        - status

        These commands consume graphql quota, not core (REST) quota (#1757).

        Note: repo list/create also use GraphQL internally but are classified
        as "core" because they're low-frequency commands. The primary goal is
        tracking issue/pr GraphQL exhaustion per #1757.
        """
        if not args:
            return "core"

        # Explicit GraphQL API calls: gh api graphql ...
        # Check args[1] == "graphql" to avoid false positives like gh api /x -f graphql=y
        if args[0] == "api" and len(args) > 1 and args[1] == "graphql":
            return "graphql"

        # Search uses its own quota bucket (30 requests/minute)
        # Includes: gh search, gh api /search/..., gh issue list --search
        if args[0] == "search":
            return "search"

        # gh issue/pr list --search uses search API internally (#1869)
        # Must check before general issue/pr classification below
        if args[0] in ("issue", "pr") and len(args) > 1 and args[1] == "list":
            for arg in args:
                if arg in ("--search", "-S") or arg.startswith("--search="):
                    return "search"

        # REST search API calls: gh api /search/issues, /search/code, etc.
        if args[0] == "api" and len(args) > 1:
            api_path = args[1]
            # Check for search endpoint (handles /search/... paths)
            if api_path.startswith("/search/") or api_path.startswith("search/"):
                return "search"

        # gh CLI uses GraphQL internally for issue/pr commands
        if args[0] in ("issue", "pr"):
            return "graphql"

        # gh status fetches issues/PRs across repos using GraphQL
        if args[0] == "status":
            return "graphql"

        # repo view also uses GraphQL
        if args[0] == "repo" and len(args) > 1 and args[1] == "view":
            return "graphql"

        return "core"

    def _can_use_rest_fallback(self, args: list[str]) -> bool:
        """Check if command can use REST fallback when GraphQL is blocked (#1861).

        Issue list/view commands with --json flag can fall back to REST API.
        This allows them to proceed even when GraphQL quota is exhausted.
        """
        # Must be issue list/view with JSON flag
        if not has_json_flag(args):
            return False
        # Search commands use search quota, not REST core quota
        if is_issue_search_command(args):
            return False
        if not (is_issue_list_command(args) or is_issue_view_command(args)):
            return False
        # Check REST quota is available
        core_info = self._rate_cache.get("core")
        if not core_info or core_info.remaining <= 0:
            return False
        return True

    def check_rate_limit(self, args: list[str]) -> bool:
        """Check if call is allowed. Returns False if blocked.

        If quota is critical but reset is < 3 min away, waits for reset.
        Issue list/view commands with --json flag may proceed when GraphQL
        is blocked but REST has quota (#1861).
        """
        resource = self.get_resource(args)
        limits = self.fetch_rate_limits()
        info = limits.get(resource)
        if not info:
            return True

        if info.should_block():
            # Issue list/view with --json can use REST fallback (#1861)
            if resource == "graphql" and self._can_use_rest_fallback(args):
                debug_log(
                    "graphql blocked but REST fallback available for issue command"
                )
                return True  # Let caller handle REST fallback

            # If reset is soon, wait for it
            if info.minutes_until_reset < 3:
                wait_secs = info.minutes_until_reset * 60 + 5
                print(
                    f"gh_rate_limit: {resource} quota exhausted, "
                    f"waiting {wait_secs:.0f}s for reset",
                    file=sys.stderr,
                )
                time.sleep(wait_secs)
                self._last_rate_check = 0  # Force refresh
                return self.check_rate_limit(args)
            print(
                f"gh_rate_limit: {resource} quota critical "
                f"({info.remaining} remaining), blocking call",
                file=sys.stderr,
            )
            return False

        if info.is_warning():
            print(
                f"gh_rate_limit: {resource} quota low ({info.remaining}/{info.limit})",
                file=sys.stderr,
            )
        return True
