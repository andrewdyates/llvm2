# Copyright 2026 Dropbox, Inc.
# Author: Andrew Yates <ayates@dropbox.com>
# Licensed under the Apache License, Version 2.0

"""Checkbox-to-issue conversion for looper.

Extracted from issue_manager.py per designs/2026-02-01-issue-manager-split.md.

Part of #1808.
"""

__all__ = ["CheckboxConverter", "GH_LOCK_DIR"]

import hashlib
import json
import os
import re
import time
from pathlib import Path
from typing import TextIO

from looper.issue_manager_base import IssueManagerBase
from looper.log import debug_swallow, log_info, log_warning

# fcntl is Unix-only for file locking
try:
    import fcntl

    HAS_FCNTL = True
except ImportError:
    HAS_FCNTL = False

# Lock directory for GitHub operations (per-user, not per-repo)
GH_LOCK_DIR = Path.home() / ".ait_gh_lock"


class CheckboxConverter(IssueManagerBase):
    """Convert unchecked checkboxes in issues to child issues.

    Provides deduplication via title and hash markers, file locking to prevent
    concurrent duplicate creation, and checkbox consolidation in parent issues.

    Contracts:
        REQUIRES: repo_path is a valid Path (exists at init time not required)
        REQUIRES: role is a non-empty string
        ENSURES: All methods handle subprocess failures gracefully
        ENSURES: All methods return empty/default results on error, never raise
    """

    def _acquire_checkbox_lock(
        self, timeout: float = 60.0
    ) -> tuple[TextIO | None, bool]:
        """Acquire lock for checkbox conversion operations.

        This prevents multiple looper instances from creating duplicate child
        issues when running concurrently on the same machine.

        Contracts:
            REQUIRES: timeout >= 0
            ENSURES: Returns (None, True) when HAS_FCNTL is False (best-effort)
            ENSURES: Returns (lock_file, True) only after taking an exclusive
                lock and writing the pid
            ENSURES: Returns (None, False) on open/lock failure or timeout, and
                closes any opened lock file

        Args:
            timeout: Maximum seconds to wait for lock.

        Returns:
            Tuple of (lock_file, acquired). lock_file is None if not acquired.
            Caller must call _release_checkbox_lock() with the lock_file.
        """
        if not HAS_FCNTL:
            # Windows: no flock support, proceed (best effort)
            return None, True

        # Create lock directory if needed
        GH_LOCK_DIR.mkdir(parents=True, exist_ok=True)

        # Per-repo lock file to allow different repos to run concurrently
        repo_name = self._get_repo_name()
        lock_path = GH_LOCK_DIR / f"checkbox_convert_{repo_name}.lock"

        try:
            lock_file = open(lock_path, "w")
        except OSError as e:
            debug_swallow("checkbox_lock_open", e)
            return None, False

        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                # Write PID to lock file for debugging
                lock_file.write(f"{os.getpid()}\n")
                lock_file.flush()
                return lock_file, True
            except BlockingIOError:
                time.sleep(0.5)  # Wait and retry
            except OSError as e:
                debug_swallow("checkbox_lock_flock", e)
                break

        # Failed to acquire lock
        lock_file.close()
        return None, False

    def _release_checkbox_lock(self, lock_file: TextIO | None) -> None:
        """Release checkbox conversion lock.

        Contracts:
            REQUIRES: lock_file is the value returned by _acquire_checkbox_lock
            ENSURES: No-op when lock_file is None
            ENSURES: Attempts to unlock and close the lock file, swallowing OS
                errors

        Args:
            lock_file: Lock file handle from _acquire_checkbox_lock(), or None.
        """
        if lock_file is None:
            return

        if HAS_FCNTL:
            try:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
            except OSError as e:
                debug_swallow("checkbox_lock_unlock", e)
        try:
            lock_file.close()
        except OSError as e:
            debug_swallow("checkbox_lock_close", e)

    def get_issue_checkboxes(self, issue_num: int) -> tuple[list[str], list[str], str]:
        """Get all checkboxes from issue body and comments.

        Contracts:
            REQUIRES: issue_num is a positive integer
            ENSURES: Returns (unchecked, checked, original_body) tuple
            ENSURES: unchecked and checked are deduplicated (checked wins on conflict)
            ENSURES: Items are stripped of whitespace
            ENSURES: Never raises - best-effort parsing may return partial results
        """
        unchecked = []
        checked = []
        original_body = ""

        try:
            # Get issue body
            # Avoid gh -q/--jq due to caching bugs (#1047).
            result = self._gh_run_result(
                [
                    "gh",
                    "issue",
                    "view",
                    str(issue_num),
                    "--json",
                    "body",
                ],
                timeout=10,
            )
            if result.ok and result.value.stdout:
                try:
                    data = json.loads(result.value.stdout)
                except json.JSONDecodeError as e:
                    debug_swallow("get_issue_checkboxes_body_json", e)
                    data = {}
                original_body = data.get("body") or ""
                # Use lookahead to stop at newline OR escaped newline (\n as 2 chars)
                # OR end of string. This handles bodies that were corrupted with
                # JSON-escaped content (issue #1323).
                # Note: This will truncate tasks containing literal "\n" (rare edge case).
                unchecked.extend(
                    re.findall(r"- \[ \] (.+?)(?=\n|\\n|$)", original_body)
                )
                checked.extend(
                    re.findall(
                        r"- \[x\] (.+?)(?=\n|\\n|$)", original_body, re.IGNORECASE
                    )
                )

            # Get issue comments
            comments_result = self._gh_run_result(
                [
                    "gh",
                    "issue",
                    "view",
                    str(issue_num),
                    "--json",
                    "comments",
                ],
                timeout=10,
            )
            if comments_result.ok and comments_result.value.stdout:
                try:
                    data = json.loads(comments_result.value.stdout)
                except json.JSONDecodeError as e:
                    debug_swallow("get_issue_checkboxes_comments_json", e)
                    data = {}
                comments = data.get("comments") or []
                if isinstance(comments, list):
                    for comment in comments:
                        body = ""
                        if isinstance(comment, dict):
                            body = comment.get("body") or ""
                        unchecked.extend(re.findall(r"- \[ \] (.+?)(?=\n|\\n|$)", body))
                        checked.extend(
                            re.findall(
                                r"- \[x\] (.+?)(?=\n|\\n|$)", body, re.IGNORECASE
                            )
                        )

        except Exception as e:
            debug_swallow(
                "parse_issue_checkboxes", e
            )  # Best-effort: checkbox parsing may fail on malformed bodies

        # Normalize and deduplicate
        unchecked = list(
            dict.fromkeys(item.strip() for item in unchecked if item.strip())
        )
        checked = list(dict.fromkeys(item.strip() for item in checked if item.strip()))

        # Remove items that appear in both (checked wins)
        checked_lower = {item.lower() for item in checked}
        unchecked = [item for item in unchecked if item.lower() not in checked_lower]

        return unchecked, checked, original_body

    def _item_hash(self, item_text: str, parent_num: int) -> str:
        """Generate unique hash for a checklist item.

        Contracts:
            REQUIRES: item_text is a string and parent_num is a positive integer
            ENSURES: Returns an 8-character hex hash derived from
                f"{parent_num}:{item_text.lower().strip()}"

        Args:
            item_text: The checklist item text.
            parent_num: Parent issue number.

        Returns:
            8-character hash string for uniqueness checking.
        """
        content = f"{parent_num}:{item_text.lower().strip()}"
        return hashlib.sha256(content.encode()).hexdigest()[:8]

    def _extract_titles_and_hashes(
        self,
        issues_json: str,
        parent_num: int,
        titles: set[str],
        hashes: set[str],
    ) -> None:
        """Extract titles and looper-child hashes from issue list JSON.

        Contracts:
            REQUIRES: issues_json is a JSON string containing issue dicts
            ENSURES: Lowercased titles are added to titles when present
            ENSURES: hashes collects 8-char hashes extracted from
                looper-child:{parent_num}:<hash> markers in issue bodies
            ENSURES: JSON decode errors are swallowed (best-effort parsing)

        Args:
            issues_json: JSON string from gh issue list --json title,body
            parent_num: Parent issue number for hash marker matching
            titles: Set to add lowercase titles to (mutated in place)
            hashes: Set to add hash markers to (mutated in place)
        """
        try:
            for issue in json.loads(issues_json):
                title = issue.get("title", "")
                # Handle None body (GitHub can return null for deleted bodies)
                body = issue.get("body") or ""
                if title:
                    titles.add(title.strip().lower())
                # Extract hash from marker: <!-- looper-child:N:HASH -->
                hash_match = re.search(
                    rf"looper-child:{parent_num}:([a-f0-9]{{8}})", body
                )
                if hash_match:
                    hashes.add(hash_match.group(1))
        except json.JSONDecodeError as e:
            debug_swallow(
                "extract_titles_and_hashes_json", e
            )  # Best-effort: malformed JSON from partial response

    def get_existing_child_issues(self, parent_num: int) -> tuple[set[str], set[str]]:
        """Get titles and hashes of existing child issues to avoid duplicates.

        Contracts:
            REQUIRES: parent_num is a positive integer
            ENSURES: Returns (titles, hashes) tuple
            ENSURES: titles is set of lowercase issue titles
            ENSURES: hashes is set of looper-child hash markers
            ENSURES: Searches both open and closed issues (--state all)
            ENSURES: Never raises - returns empty sets on error
            ENSURES: Uses single combined query to reduce Search API usage (#1869)
        """
        existing_titles: set[str] = set()
        existing_hashes: set[str] = set()
        try:
            # Combined search query for both patterns to reduce Search API calls (#1869)
            # Search API has 30 req/min limit - combining queries saves quota
            # Use OR pattern: "Part of #N" OR "looper-child:N:" in body
            combined_query = (
                f'"Part of #{parent_num}" OR "looper-child:{parent_num}:" in:body'
            )
            result = self._gh_run_result(
                [
                    "gh",
                    "issue",
                    "list",
                    "--state",
                    "all",
                    "--search",
                    combined_query,
                    "--json",
                    "title,body",
                    "--limit",
                    "100",  # Reasonable limit for child issues
                ],
                timeout=15,
            )
            if result.ok and result.value.stdout.strip():
                self._extract_titles_and_hashes(
                    result.value.stdout, parent_num, existing_titles, existing_hashes
                )
            elif not result.ok:
                # Log search failure for debugging - could be quota exhaustion
                stderr = result.value.stderr.strip() if result.value else ""
                if "rate limit" in stderr.lower() or "quota" in stderr.lower():
                    log_warning(
                        f"    ⚠️  Search quota exhausted - skipping deduplication for #{parent_num}"
                    )
                # Continue with empty sets - best effort deduplication

        except Exception as e:
            debug_swallow(
                "get_existing_child_issues", e
            )  # Best-effort: child issue lookup for deduplication
        return existing_titles, existing_hashes

    def _update_issue_body_with_consolidated_checkboxes(
        self,
        issue_num: int,
        unchecked: list[str],
        checked: list[str],
        original_body: str,
    ) -> bool:
        """Update issue body with consolidated and cleaned checkbox list.

        Contracts:
            REQUIRES: issue_num is a positive integer
            REQUIRES: unchecked and checked are lists of strings
            REQUIRES: original_body is a string (may be empty)
            ENSURES: Returns True on successful update, False otherwise
            ENSURES: Removes existing checkboxes and ## Checklist header
            ENSURES: Appends consolidated checkboxes with checked first
            ENSURES: Never raises - catches all exceptions
        """
        try:
            # Defensive check: if body looks like JSON-wrapped content, refuse to update
            # This catches corruption where {"body": "..."} was stored as literal body
            # (issue #1323). Better to skip consolidation than corrupt further.
            if original_body.lstrip().startswith('{"'):
                return False

            # Remove existing checkbox lines and ## Checklist header from body
            # Match checkbox until newline OR escaped newline (\n as 2 chars) OR end.
            # Handles both normal bodies and JSON-escaped content remnants.
            clean_body = re.sub(
                r"- \[[x ]\] .+?(?=\n|\\n|$)", "", original_body, flags=re.IGNORECASE
            )
            # Note: We do NOT globally replace \n with real newlines here, as that
            # would corrupt legitimate escape sequences in code examples. The regex
            # above handles the specific case of escaped newlines after checkboxes.
            clean_body = re.sub(r"\n*## Checklist\n*", "\n", clean_body)
            clean_body = clean_body.strip()

            # Build consolidated checkbox section
            checkbox_section = ""
            if checked or unchecked:
                checkbox_section = "\n\n## Checklist\n"
                for item in checked:
                    checkbox_section += f"- [x] {item}\n"
                for item in unchecked:
                    checkbox_section += f"- [ ] {item}\n"

            if clean_body:
                new_body = clean_body + checkbox_section
            else:
                new_body = checkbox_section.lstrip("\n")

            # Update issue
            result = self._gh_run_result(
                ["gh", "issue", "edit", str(issue_num), "--body", new_body],
                timeout=15,
            )
            return result.ok
        except Exception as e:
            debug_swallow("update_issue_body_checkboxes", e)
            return False

    def _get_issue_labels(self, issue_num: int) -> list[str]:
        """Fetch all label names for an issue (single API call).

        Contracts:
            REQUIRES: issue_num is a positive integer
            ENSURES: Returns list of label name strings
            ENSURES: Returns empty list on error (best-effort)
            ENSURES: Never raises - catches all exceptions
        """
        try:
            result = self._gh_run_result(
                ["gh", "issue", "view", str(issue_num), "--json", "labels"],
                timeout=10,
            )
            if result.ok and result.value.stdout.strip():
                data = json.loads(result.value.stdout)
                return [
                    label_obj.get("name", "")
                    for label_obj in data.get("labels", [])
                ]
        except Exception as e:
            debug_swallow("get_issue_labels", e)
        return []

    def _get_commit_issue_numbers(self) -> set[int]:
        """Extract issue numbers from last commit message.

        Excludes Claims #N because claiming an issue should not trigger
        checkbox conversion before any work is done (#2490).

        Contracts:
            REQUIRES: git log is available in the repo
            ENSURES: Returns an empty set if git log fails
            ENSURES: Returns a set of integers extracted from Fixes/Part of/Re:
                patterns in the last commit message
            ENSURES: Claims #N does NOT trigger checkbox conversion

        Returns:
            Set of issue numbers referenced in Fixes/Part of/Re: patterns.
        """
        result = self._run_result(
            ["git", "log", "-1", "--format=%s %b"],
            timeout=10,
        )
        if not result.ok:
            return set()

        issue_nums: set[int] = set()
        for match in re.finditer(
            r"(?:Fixes|Part of|Re:) #(\d+)",
            result.value.stdout,
            re.IGNORECASE,
        ):
            issue_nums.add(int(match.group(1)))
        return issue_nums

    def _create_child_issue(
        self, item_text: str, parent_issue: int, parent_p_label: str | None
    ) -> bool:
        """Create a child issue from a checklist item.

        Contracts:
            REQUIRES: item_text is non-empty and parent_issue is a positive integer
            ENSURES: Issue body includes Part of #{parent_issue} and the
                looper-child marker
            ENSURES: Returns True iff gh issue create succeeds (returncode 0)
            ENSURES: Inherits parent_p_label when provided

        Args:
            item_text: The checklist item text to use as issue title.
            parent_issue: Parent issue number for "Part of" reference.
            parent_p_label: Priority label to inherit (e.g., "P2"), or None.

        Returns:
            True if issue created successfully, False otherwise.
        """
        # Generate unique marker for reliable deduplication
        item_hash = self._item_hash(item_text, parent_issue)
        new_body = (
            f"Part of #{parent_issue}\n\n"
            "Auto-created from unchecked item in parent issue.\n\n"
            f"<!-- looper-child:{parent_issue}:{item_hash} -->"
        )
        # Inherit parent's P-label if present
        label_args = []
        if parent_p_label:
            label_args.extend(["--label", parent_p_label])

        create_result = self._gh_run_result(
            ["gh", "issue", "create", "--title", item_text, "--body", new_body]
            + label_args,
            timeout=30,
        )

        if create_result.ok:
            new_url = create_result.value.stdout.strip()
            log_info(
                f"    ✓ Created: {item_text[:50]}{'...' if len(item_text) > 50 else ''}"
            )
            log_info(f"      {new_url}")
            return True
        return False

    def convert_unchecked_to_issues(self) -> int:
        """Convert unchecked checkboxes in worked issues to new issues.

        Contracts:
            REQUIRES: Last commit exists in git history
            ENSURES: Returns count of issues created (>= 0)
            ENSURES: Only processes issues referenced in last commit
            ENSURES: Skips epic-labeled issues (their checklists are issue refs)
            ENSURES: Skips items that already have child issues
            ENSURES: Inherits P-label from parent issue
            ENSURES: Consolidates checkboxes in parent issue body
            ENSURES: Limits to 5 issues and 10 items per issue
            ENSURES: Uses per-repo lock to prevent concurrent duplicate creation
            ENSURES: Never raises - returns 0 on error
        """
        lock_file = None
        try:
            issue_nums = self._get_commit_issue_numbers()
            if not issue_nums:
                return 0

            # Fetch labels once per issue for epic check + P-level (#2669)
            issue_labels: dict[int, list[str]] = {
                n: self._get_issue_labels(n) for n in issue_nums
            }

            # Skip epic issues — their checklists reference existing issues (#N),
            # not plain text tasks. Converting them would create duplicates. (#2627)
            epic_issues = {n for n in issue_nums if "epic" in issue_labels.get(n, [])}
            issue_nums -= epic_issues

            if not issue_nums:
                return 0

            # Acquire per-repo lock to prevent concurrent loopers from creating
            # duplicate child issues (addresses GitHub search indexing lag)
            lock_file, acquired = self._acquire_checkbox_lock(timeout=60.0)
            if not acquired:
                log_warning(
                    "    ⚠️  Could not acquire checkbox lock (another looper running?)"
                )
                return 0

            created_count = 0

            for issue_num in list(issue_nums)[:5]:  # Limit to 5 issues
                # Get all checkboxes from body and comments
                unchecked, checked, original_body = self.get_issue_checkboxes(issue_num)

                if not unchecked and not checked:
                    continue

                # Get existing child issues to avoid duplicates
                # Now searches --state all and unique markers for robustness
                existing_titles, existing_hashes = self.get_existing_child_issues(
                    issue_num
                )

                # Get parent's P-label from cached labels (#2669)
                parent_p_label = next(
                    (l for l in issue_labels.get(issue_num, [])
                     if l in ("P0", "P1", "P2", "P3")),
                    None,
                )

                # Filter items that don't already have child issues
                # Check BOTH title AND hash to catch duplicates regardless of
                # which one was indexed first by GitHub search
                items_to_convert = []
                for item_text in unchecked[:10]:  # Limit to 10 items per issue
                    item_hash = self._item_hash(item_text, issue_num)
                    if item_text.lower() in existing_titles:
                        log_info(f"    ⊘ Skipped (title exists): {item_text[:40]}...")
                        continue
                    if item_hash in existing_hashes:
                        log_info(f"    ⊘ Skipped (hash exists): {item_text[:40]}...")
                        continue
                    items_to_convert.append(item_text)

                # Create child issues
                for item_text in items_to_convert:
                    if self._create_child_issue(item_text, issue_num, parent_p_label):
                        created_count += 1
                        checked.append(item_text)
                        unchecked.remove(item_text)

                # Consolidate checkboxes in issue body
                if original_body:
                    if self._update_issue_body_with_consolidated_checkboxes(
                        issue_num, unchecked, checked, original_body
                    ):
                        log_info(f"    ✓ Consolidated checkboxes in #{issue_num}")

            return created_count
        except Exception as e:
            # Catch-all: checkbox conversion failure logged, return 0
            log_warning(f"    ⚠️  Error converting checkboxes: {e}")
            return 0
        finally:
            # Always release lock
            self._release_checkbox_lock(lock_file)

    def process_unchecked_checkboxes(self) -> None:
        """Convert unchecked checkboxes to new issues at session end.

        Contracts:
            ENSURES: Logs status message to stdout
            ENSURES: Delegates to convert_unchecked_to_issues()
            ENSURES: Never raises - convert method handles exceptions
        """
        log_info("")
        log_info("📋 Processing checkboxes...")
        created = self.convert_unchecked_to_issues()
        if created > 0:
            log_info(f"    Created {created} new issue(s) from unchecked items")
        log_info("")
