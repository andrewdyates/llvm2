# Copyright 2026 Dropbox, Inc.
# Author: Andrew Yates <ayates@dropbox.com>
# Licensed under the Apache License, Version 2.0

"""
gh_rate_limit/historical.py - Historical Issue Cache

Persistent cache that never expires, stored with human-readable names.
Use case: old issues don't change, grep for duplicates, offline search.

Part of gh_rate_limit decomposition (designs/2026-02-01-rate-limiter-decomposition.md).

Methods extracted from RateLimiter:
- _get_historical_dir, _get_historical_issue_path, _store_historical_issue
- get_historical_issue, _get_historical_age, _invalidate_historical_issue
- cleanup_historical_cache
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from ai_template_scripts.gh_rate_limit.limiter import debug_log


class HistoricalCache:
    """Persistent cache for historical issue data (#1134).

    Never expires automatically - stores issues in human-readable structure
    for grep/ripgrep search. Used as fallback when TTL cache misses.

    Args:
        cache_dir: Base cache directory (historical/ subdirectory will be used).
    """

    def __init__(self, cache_dir: Path) -> None:
        self.cache_dir = cache_dir
        self._historical_dir = cache_dir / "historical"

    def get_dir(self) -> Path:
        """Get the historical cache directory, creating if needed."""
        self._historical_dir.mkdir(parents=True, exist_ok=True)
        return self._historical_dir

    def get_issue_path(self, owner_repo: str, issue_num: str) -> Path:
        """Get path for a historical issue cache file.

        Format: historical/<owner>/<repo>/issue-<num>.json
        Human-readable structure for grep/ripgrep search.
        """
        owner, repo = (
            owner_repo.split("/", 1) if "/" in owner_repo else ("unknown", owner_repo)
        )
        issue_dir = self.get_dir() / owner / repo
        issue_dir.mkdir(parents=True, exist_ok=True)
        return issue_dir / f"issue-{issue_num}.json"

    def store_issue(
        self,
        owner_repo: str,
        issue_num: str,
        stdout: str,
    ) -> None:
        """Store issue data in historical cache.

        Historical cache uses pretty-printed JSON for grep searchability.
        Files are named by issue number for easy browsing.
        """
        if not issue_num or not issue_num.isdigit():
            return
        if not owner_repo:
            return

        hist_path = self.get_issue_path(owner_repo, issue_num)
        try:
            # Parse and re-serialize with pretty-printing for grep searchability
            try:
                data = json.loads(stdout)
            except json.JSONDecodeError:
                data = {"raw": stdout}

            # Add metadata for diagnostics
            hist_data = {
                "issue_number": int(issue_num),
                "repo": owner_repo,
                "cached_at": time.time(),
                "data": data,
            }
            hist_path.write_text(json.dumps(hist_data, indent=2))
        except Exception as e:
            debug_log(f"store_issue failed for {owner_repo}#{issue_num}: {e}")

    def get_issue(self, owner_repo: str, issue_num: str) -> dict | None:
        """Get issue from historical cache.

        Returns the cached data dict or None if not found/invalid.

        Note: Returns None if cached data is not a dict (e.g., if cache was
        corrupted or stored a scalar value). Callers can rely on return type.
        """
        hist_path = self.get_issue_path(owner_repo, issue_num)
        if not hist_path.exists():
            return None
        try:
            data = json.loads(hist_path.read_text())
            result = data.get("data")
            # Validate type - historical cache must be a dict to be usable (#1300)
            # Sometimes REST fallback stores scalar values (e.g., comment count)
            # which are not valid for field extraction.
            if not isinstance(result, dict):
                return None
            # Normalize labels format (#1751) - ensure array of objects with 'name' key
            # Handles: string "P3, in-progress" → [{name: "P3"}, {name: "in-progress"}]
            #          array ["P3", "in-progress"] → [{name: "P3"}, {name: "in-progress"}]
            if "labels" in result:
                result = self._normalize_labels(result)
            return result
        except Exception as e:
            debug_log(f"get_issue read failed for {owner_repo}#{issue_num}: {e}")
            return None

    def _normalize_labels(self, data: dict) -> dict:
        """Normalize labels to array of objects with 'name' key (#1751).

        The gh CLI returns labels as [{name: "P3", id: ...}, ...].
        Some cached entries may have malformed formats:
        - String: "P3, in-progress, task"
        - Array of strings: ["P3", "in-progress", "task"]

        This normalizes to the expected array-of-objects format.
        """
        labels = data.get("labels")
        if labels is None:
            return data

        # Already correct format - array of dicts with 'name' key
        if isinstance(labels, list) and all(
            isinstance(lbl, dict) and "name" in lbl for lbl in labels
        ):
            return data

        # Normalize to array of objects
        normalized: list[dict] = []
        if isinstance(labels, str):
            # String format: "P3, in-progress, task"
            for name in labels.split(","):
                name = name.strip()
                if name:
                    normalized.append({"name": name})
        elif isinstance(labels, list):
            # Array of strings: ["P3", "in-progress"]
            for lbl in labels:
                if isinstance(lbl, dict) and "name" in lbl:
                    normalized.append(lbl)
                elif isinstance(lbl, str) and lbl:
                    normalized.append({"name": lbl})

        # Return new dict with normalized labels (don't mutate original)
        result = dict(data)
        result["labels"] = normalized
        return result

    def get_age(self, owner_repo: str, issue_num: str) -> float | None:
        """Get age of historical cache entry in seconds, or None if not found."""
        hist_path = self.get_issue_path(owner_repo, issue_num)
        if not hist_path.exists():
            return None
        try:
            data = json.loads(hist_path.read_text())
            cached_at = data.get("cached_at", 0)
            if cached_at:
                return time.time() - cached_at
        except Exception as e:
            debug_log(f"get_age read failed for {owner_repo}#{issue_num}: {e}")
        return None

    def invalidate_issue(self, owner_repo: str, issue_num: str) -> None:
        """Remove historical cache entry for an issue.

        Called when issue state changes (close/reopen) to prevent stale data.
        """
        hist_path = self.get_issue_path(owner_repo, issue_num)
        try:
            hist_path.unlink(missing_ok=True)
        except OSError:
            pass

    def cleanup(self, max_age_days: int = 30) -> tuple[int, int]:
        """Clean up old historical cache entries.

        Removes cached issues older than max_age_days. This is opt-in cleanup
        to prevent unbounded disk growth over time (#1363).

        Args:
            max_age_days: Remove entries older than this (default 30 days)

        Returns:
            Tuple of (files_removed, bytes_freed)
        """
        hist_dir = self.get_dir()
        if not hist_dir.exists():
            return (0, 0)

        cutoff = time.time() - (max_age_days * 86400)
        files_removed = 0
        bytes_freed = 0

        # Walk historical/<owner>/<repo>/issue-N.json
        for owner_dir in hist_dir.iterdir():
            if not owner_dir.is_dir():
                continue
            for repo_dir in owner_dir.iterdir():
                if not repo_dir.is_dir():
                    continue
                for issue_file in repo_dir.glob("issue-*.json"):
                    try:
                        st = issue_file.stat()
                        if st.st_mtime < cutoff:
                            issue_file.unlink()
                            files_removed += 1
                            bytes_freed += st.st_size
                    except OSError:
                        pass

                # Clean up empty repo directory
                try:
                    if repo_dir.exists() and not any(repo_dir.iterdir()):
                        repo_dir.rmdir()
                except OSError:
                    pass

            # Clean up empty owner directory
            try:
                if owner_dir.exists() and not any(owner_dir.iterdir()):
                    owner_dir.rmdir()
            except OSError:
                pass

        return (files_removed, bytes_freed)


def extract_issue_info_from_args(
    args: list[str],
) -> tuple[str | None, str | None]:
    """Extract issue number and repo override from command args.

    Handles args like ['issue', 'view', '123', '-R', 'owner/repo'].

    Returns:
        Tuple of (issue_num, repo_override). Either may be None.
    """
    issue_num: str | None = None
    repo_override: str | None = None

    # Flags that take arguments (their values should not be treated as issue numbers)
    flags_with_args = {
        "--repo",
        "-R",
        "--comment",
        "-c",
        "--body",
        "-b",
        "--title",
        "-t",
        "--label",
        "-l",
        "--assignee",
        "-a",
        "--milestone",
        "-m",
        "--project",
        "-p",
        "--body-file",
        "-F",
        "--json",
        "-q",
        "--jq",
        "--template",
        "-T",
        "--limit",
        "-L",
        "--state",
        "-s",
        "--search",
        "-S",
        "--author",
        "-A",
        "--mentions",
    }

    i = 2  # Skip "issue view" or "issue close"
    while i < len(args):
        arg = args[i]
        if arg in ("--repo", "-R") and i + 1 < len(args):
            repo_override = args[i + 1]
            i += 2
        elif arg.startswith("--repo="):
            repo_override = arg.split("=", 1)[1]
            i += 1
        elif arg in flags_with_args and i + 1 < len(args):
            # Skip this flag and its argument
            i += 2
        elif arg.startswith("--") and "=" in arg:
            # Flag with value in same argument (e.g., --label=bug)
            i += 1
        elif not arg.startswith("-") and issue_num is None:
            issue_num = arg.lstrip("#")
            i += 1
        else:
            i += 1

    return issue_num, repo_override
