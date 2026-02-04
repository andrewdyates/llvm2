# Copyright 2026 Dropbox, Inc.
# Author: Andrew Yates <ayates@dropbox.com>
# Licensed under the Apache License, Version 2.0

"""
gh_rate_limit/cache.py - TTL Cache for GitHub API responses

Handles cache keying, read/write, and invalidation rules.
Part of gh_rate_limit decomposition (designs/2026-02-01-rate-limiter-decomposition.md).

Methods extracted from RateLimiter:
- _cache_key, _get_ttl, get_cached, get_stale_cached, set_cached
- invalidate_cache, _invalidate_cache
- _format_stale_warning

Related: historical.py for persistent cache, etag.py for conditional requests.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Callable
from pathlib import Path

from ai_template_scripts.gh_rate_limit.limiter import debug_log

# Cache TTLs (seconds) - 3 min default to reduce API load while staying fresh
# Workers polling simultaneously was causing rate limits; shared cache fixes this
CACHE_TTLS = {
    ("issue", "list"): 180,  # 3 min
    ("issue", "view"): 180,  # 3 min
    ("repo", "view"): 180,  # 3 min (was 1hr but stale data causes issues)
    ("label", "list"): 180,  # 3 min
    ("pr", "list"): 180,  # 3 min
}

# Search API cache TTL - longer to reduce quota pressure (#1782, #1869)
# Applies to: gh search, gh api /search/..., gh issue list --search
# Increased from 60s to 300s per #1869 - checkbox searches have unique keys
# per parent issue, so short TTL provides little reuse while exhausting quota.
SEARCH_CACHE_TTL = 300  # 5 min

# Write commands that invalidate TTL cache
# Note: edit/close/reopen also invalidate historical cache (see historical.py)
INVALIDATES = {
    ("issue", "create"): [("issue", "list")],
    ("issue", "edit"): [("issue", "list"), ("issue", "view")],  # + historical (#1904)
    ("issue", "close"): [("issue", "list"), ("issue", "view")],  # + historical
    ("issue", "reopen"): [("issue", "list"), ("issue", "view")],  # + historical
    ("issue", "comment"): [("issue", "view")],
    ("label", "create"): [("label", "list")],
    ("pr", "create"): [("pr", "list")],
    ("pr", "merge"): [("pr", "list")],
}


class TtlCache:
    """TTL-based cache for GitHub API responses.

    Manages cache keying, storage, retrieval, and invalidation.
    Used by RateLimiter to cache read operations.

    Args:
        cache_dir: Directory to store cache files.
        get_repo: Callable that returns the current repo name for cache namespacing.
    """

    def __init__(
        self,
        cache_dir: Path,
        get_repo: Callable[[], str],
    ) -> None:
        self.cache_dir = cache_dir
        self._get_repo = get_repo

    def cache_key(self, args: list[str]) -> str:
        """Generate cache key including repo context."""
        repo = self._get_repo()
        key_data = json.dumps({"repo": repo, "args": sorted(args)})
        return hashlib.sha256(key_data.encode()).hexdigest()[:16]

    def get_ttl(self, args: list[str]) -> int | None:
        """Get cache TTL for command, or None if not cacheable."""
        if not args:
            return None

        # Search API caching (#1782) - 60s to avoid exhausting 30/min quota
        # gh search ... commands
        if args[0] == "search":
            return SEARCH_CACHE_TTL

        # gh api /search/... commands (REST search API)
        if args[0] == "api" and len(args) > 1:
            api_path = args[1]
            if api_path.startswith("/search/") or api_path.startswith("search/"):
                return SEARCH_CACHE_TTL

        # gh issue list --search uses issue list TTL (already cached)
        # The --search flag doesn't change the caching behavior for issue list
        if len(args) < 2:
            return None
        return CACHE_TTLS.get((args[0], args[1]))

    def is_write(self, args: list[str]) -> bool:
        """Check if command is a write operation."""
        if len(args) < 2:
            return False
        return (args[0], args[1]) in INVALIDATES

    def get_invalidates(self, args: list[str]) -> list[tuple[str, str]]:
        """Get list of command types to invalidate for a write operation."""
        if len(args) < 2:
            return []
        cmd = (args[0], args[1])
        return INVALIDATES.get(cmd, [])

    def get_cached(self, args: list[str]) -> str | None:
        """Get cached response if valid. Returns stdout string or None."""
        ttl = self.get_ttl(args)
        if ttl is None:
            return None
        key = self.cache_key(args)
        cache_file = self.cache_dir / f"{key}.json"
        if not cache_file.exists():
            return None
        try:
            data = json.loads(cache_file.read_text())
            if time.time() - data.get("created_at", 0) < ttl:
                return data.get("stdout")
        except Exception as e:
            debug_log(f"get_cached read failed for {key}: {e}")
        return None

    def get_stale_cached(self, args: list[str]) -> tuple[str | None, float | None]:
        """Get cached response even if expired. Returns (stdout, age_sec).

        Used for stale fallback when API is unavailable (#1135).
        Returns expired TTL cache data with its age for warning messages.
        """
        ttl = self.get_ttl(args)
        if ttl is None:
            return None, None
        key = self.cache_key(args)
        cache_file = self.cache_dir / f"{key}.json"
        if not cache_file.exists():
            return None, None
        try:
            data = json.loads(cache_file.read_text())
            created_at = data.get("created_at", 0)
            age_sec = time.time() - created_at
            stdout = data.get("stdout")
            if stdout:
                return stdout, age_sec
        except Exception as e:
            debug_log(f"get_stale_cached read failed for {key}: {e}")
        return None, None

    def set_cached(self, args: list[str], stdout: str, etag: str | None = None) -> None:
        """Cache a response with optional ETag (#1674).

        Args:
            args: Command arguments for cache key
            stdout: Response body to cache
            etag: Optional ETag from response headers for conditional requests
        """
        ttl = self.get_ttl(args)
        if ttl is None:
            return
        key = self.cache_key(args)
        cache_file = self.cache_dir / f"{key}.json"
        try:
            data = {
                "repo": self._get_repo(),
                "args": args,
                "stdout": stdout,
                "created_at": time.time(),
            }
            if etag:
                data["etag"] = etag
            cache_file.write_text(json.dumps(data))
        except Exception as e:
            debug_log(f"set_cached write failed for {key}: {e}")

    def get_cached_etag(self, args: list[str]) -> str | None:
        """Get cached ETag for conditional request (#1674).

        Returns ETag if cached data exists and has an ETag, None otherwise.
        Used to make If-None-Match conditional requests.
        """
        ttl = self.get_ttl(args)
        if ttl is None:
            return None
        key = self.cache_key(args)
        cache_file = self.cache_dir / f"{key}.json"
        if not cache_file.exists():
            return None
        try:
            data = json.loads(cache_file.read_text())
            return data.get("etag")
        except Exception as e:
            debug_log(f"get_cached_etag read failed for {key}: {e}")
            return None

    def extend_cache_ttl(self, args: list[str]) -> bool:
        """Extend cache TTL by updating created_at timestamp (#1674).

        Called when a 304 Not Modified response is received.
        Returns True if cache was updated, False otherwise.
        """
        ttl = self.get_ttl(args)
        if ttl is None:
            return False
        key = self.cache_key(args)
        cache_file = self.cache_dir / f"{key}.json"
        if not cache_file.exists():
            return False
        try:
            data = json.loads(cache_file.read_text())
            data["created_at"] = time.time()
            cache_file.write_text(json.dumps(data))
            return True
        except Exception as e:
            debug_log(f"extend_cache_ttl failed for {key}: {e}")
            return False

    def invalidate(
        self,
        args: list[str],
        target_repo: str,
        on_historical_invalidate: Callable[[list[str]], None] | None = None,
    ) -> None:
        """Invalidate cache entries affected by a write operation.

        Args:
            args: Command args that triggered invalidation
            target_repo: Repo name for cache key matching
            on_historical_invalidate: Optional callback for issue edit/close/reopen
                to invalidate historical cache. Takes args as parameter.
        """
        if len(args) < 2:
            return
        cmd = (args[0], args[1])
        to_invalidate = INVALIDATES.get(cmd, [])
        for inv_cmd in to_invalidate:
            # Find and delete matching cache files for this repo + command type
            for cache_file in self.cache_dir.glob("*.json"):
                if cache_file.name in ("rate_state.json", "change_log.json"):
                    continue
                try:
                    data = json.loads(cache_file.read_text())
                    cached_args = data.get("args", [])
                    if (
                        len(cached_args) >= 2
                        and cached_args[0] == inv_cmd[0]
                        and cached_args[1] == inv_cmd[1]
                        and data.get("repo") == target_repo
                    ):
                        cache_file.unlink()
                except (json.JSONDecodeError, OSError):
                    # Skip corrupted or inaccessible files
                    pass

        # Invalidate historical cache for issue edit/close/reopen (#1163, #1904)
        if cmd in (("issue", "edit"), ("issue", "close"), ("issue", "reopen")):
            if on_historical_invalidate:
                on_historical_invalidate(args)


def format_stale_warning(source: str, age_sec: float, reason: str | None = None) -> str:
    """Format a consistent stale data warning for stderr (#1135).

    Args:
        source: Where the stale data came from (e.g., "cache", "historical")
        age_sec: How old the data is in seconds
        reason: Why we're using stale data (e.g., "rate limited", "API timeout")
    """
    if age_sec < 60:
        age_str = f"{int(age_sec)}s"
    elif age_sec < 3600:
        age_str = f"{int(age_sec / 60)}m"
    elif age_sec < 86400:
        age_str = f"{age_sec / 3600:.1f}h"
    else:
        age_str = f"{age_sec / 86400:.1f}d"

    parts = [f"gh_rate_limit: [STALE: {age_str} old]"]
    parts.append(f"using {source}")
    if reason:
        parts.append(f"({reason})")
    return " ".join(parts)
