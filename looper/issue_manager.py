# Copyright 2026 Dropbox, Inc.
# Author: Andrew Yates <ayates@dropbox.com>
# Licensed under the Apache License, Version 2.0

"""GitHub issue operations for the looper runtime.

Use via LoopRunner/IssueManager in the main loop. Direct use is intended for
tests or tooling that need issue inspection or checkbox conversion without
starting a loop. Thread safety: not thread-safe; call from a single process.

Split into submodules for maintainability (#1808):
- issue_manager_base.py: Shared subprocess helpers
- issue_manager_audit.py: Manager audit checks
- issue_manager_checkbox.py: Checkbox-to-issue conversion
See: designs/2026-02-01-issue-manager-split.md
"""

__all__ = ["GH_LOCK_DIR", "HAS_FCNTL", "IN_PROGRESS_LABELS", "IssueManager"]

import re
from pathlib import Path
from typing import Any

from ai_template_scripts.labels import IN_PROGRESS_ALL_LABELS
from looper.issue_manager_audit import IssueAuditor
from looper.issue_manager_base import IssueManagerBase
from looper.issue_manager_checkbox import GH_LOCK_DIR, CheckboxConverter
from looper.log import debug_swallow, log_info, log_warning

# Re-export from checkbox module for backward compatibility
try:
    import fcntl  # noqa: F401 - for HAS_FCNTL re-export

    HAS_FCNTL = True
except ImportError:
    HAS_FCNTL = False

# Alias for backward compatibility with code using IN_PROGRESS_LABELS
IN_PROGRESS_LABELS = IN_PROGRESS_ALL_LABELS


class IssueManager(IssueManagerBase):
    """Encapsulate GitHub issue operations for looper.

    Uses composition with IssueAuditor for manager audit functionality
    and CheckboxConverter for checkbox-to-issue conversion.

    Contracts:
        REQUIRES: repo_path is a valid Path (exists at init time not required)
        REQUIRES: role is a non-empty string (stored as provided, not normalized)
        ENSURES: All methods handle subprocess failures gracefully
        ENSURES: All methods return empty results on error, never raise
    """

    def __init__(self, repo_path: Path, role: str) -> None:
        """Initialize IssueManager.

        Contracts:
            REQUIRES: repo_path is a Path object
            REQUIRES: role is a non-empty string
            ENSURES: self.repo_path is an absolute resolved path
            ENSURES: self.role is stored as provided
            ENSURES: self._auditor is initialized for audit delegation
            ENSURES: self._checkbox is initialized for checkbox delegation
        """
        super().__init__(repo_path, role)
        self._auditor = IssueAuditor(repo_path, role)
        self._checkbox = CheckboxConverter(repo_path, role)

    # Delegate audit methods to IssueAuditor
    def check_stuck_issues(self) -> list[tuple[int, int]]:
        """Delegate to IssueAuditor."""
        return self._auditor.check_stuck_issues()

    def check_thrashing_issues(self) -> list[tuple[int, int]]:
        """Delegate to IssueAuditor."""
        return self._auditor.check_thrashing_issues()

    def check_closed_by_removal(self) -> list[tuple[int, str, int, int]]:
        """Delegate to IssueAuditor."""
        return self._auditor.check_closed_by_removal()

    def check_blocker_cycles(self) -> list[dict[str, Any]]:
        """Delegate to IssueAuditor."""
        return self._auditor.check_blocker_cycles()

    def check_escalation_sla(self, sla_days: int = 3) -> list[tuple[int, str, int]]:
        """Delegate to IssueAuditor."""
        return self._auditor.check_escalation_sla(sla_days)

    def report_stuck_issues(self, escalation_sla_days: int | None = None) -> None:
        """Delegate to IssueAuditor."""
        self._auditor.report_stuck_issues(escalation_sla_days)

    # Delegate checkbox methods to CheckboxConverter
    def get_issue_checkboxes(self, issue_num: int) -> tuple[list[str], list[str], str]:
        """Delegate to CheckboxConverter."""
        return self._checkbox.get_issue_checkboxes(issue_num)

    def get_existing_child_issues(self, parent_num: int) -> tuple[set[str], set[str]]:
        """Delegate to CheckboxConverter."""
        return self._checkbox.get_existing_child_issues(parent_num)

    def convert_unchecked_to_issues(self) -> int:
        """Delegate to CheckboxConverter."""
        return self._checkbox.convert_unchecked_to_issues()

    def process_unchecked_checkboxes(self) -> None:
        """Delegate to CheckboxConverter."""
        self._checkbox.process_unchecked_checkboxes()

    # Delegate internal checkbox methods for testing (backward compatibility)
    def _item_hash(self, item_text: str, parent_num: int) -> str:
        """Delegate to CheckboxConverter."""
        return self._checkbox._item_hash(item_text, parent_num)

    def _extract_titles_and_hashes(
        self,
        issues_json: str,
        parent_num: int,
        titles: set[str],
        hashes: set[str],
    ) -> None:
        """Delegate to CheckboxConverter."""
        return self._checkbox._extract_titles_and_hashes(
            issues_json, parent_num, titles, hashes
        )

    def _update_issue_body_with_consolidated_checkboxes(
        self,
        issue_num: int,
        unchecked: list[str],
        checked: list[str],
        original_body: str,
    ) -> bool:
        """Delegate to CheckboxConverter."""
        return self._checkbox._update_issue_body_with_consolidated_checkboxes(
            issue_num, unchecked, checked, original_body
        )

    def _has_label(self, issue_num: int, label_name: str) -> bool:
        """Check if issue has a label (delegates to _get_issue_labels)."""
        return label_name in self._checkbox._get_issue_labels(issue_num)

    def _get_issue_p_label(self, issue_num: int) -> str | None:
        """Get P-label from issue (delegates to _get_issue_labels)."""
        for label in self._checkbox._get_issue_labels(issue_num):
            if label in ("P0", "P1", "P2", "P3"):
                return label
        return None

    def _get_commit_issue_numbers(self) -> set[int]:
        """Delegate to CheckboxConverter."""
        return self._checkbox._get_commit_issue_numbers()

    def _acquire_checkbox_lock(self, timeout: float = 60.0) -> tuple:
        """Delegate to CheckboxConverter."""
        return self._checkbox._acquire_checkbox_lock(timeout)

    def _release_checkbox_lock(self, lock_file) -> None:
        """Delegate to CheckboxConverter."""
        return self._checkbox._release_checkbox_lock(lock_file)

    def create_wip_followup(self, commit_hash: str, iteration: int) -> None:
        """Create a GitHub issue to track WIP commit follow-up.

        Contracts:
            REQUIRES: commit_hash is a non-empty string (git commit hash)
            REQUIRES: iteration is a positive integer
            ENSURES: Creates issue with "wip-followup" label
            ENSURES: Issue title includes role, iteration, and short hash
            ENSURES: Prints URL on success, warning on failure
            ENSURES: Never raises - catches all exceptions
        """
        role_char = self.role[0].upper()
        title = f"WIP follow-up: [{role_char}]{iteration} ({commit_hash})"
        body = f"""Session ended with uncommitted work.

**Commit:** {commit_hash}
**Role:** {self.role.upper()}
**Iteration:** {iteration}

## Action Required
- [ ] Review changes in commit `{commit_hash}`
- [ ] Continue work OR amend with proper message OR revert if broken

---
Auto-created by looper.py to prevent work loss.
"""
        result = self._gh_run_result(
            [
                "gh",
                "issue",
                "create",
                "--title",
                title,
                "--body",
                body,
                "--label",
                "wip-followup",
            ],
            timeout=30,
        )
        if result.ok and result.value.returncode == 0:
            issue_url = result.value.stdout.strip()
            log_info(f"✓ Created follow-up issue: {issue_url}")
        elif not result.ok:
            log_warning(f"⚠️  Could not create WIP follow-up issue: {result.error}")
        else:
            log_warning(
                f"⚠️  Could not create WIP follow-up issue: {result.value.stderr}"
            )

    def cleanup_closed_issues_labels(self) -> int:
        """Remove workflow labels from closed issues.

        Issues closed via GitHub auto-close (Fixes #N on push) may retain
        stale workflow labels (needs-review, do-audit, in-progress*).
        This cleanup runs periodically to ensure queues stay accurate.

        See issue #911.

        Contracts:
            ENSURES: Returns count of cleaned issues (>= 0)
            ENSURES: Never raises - catches subprocess failures
            ENSURES: Prints status for cleaned issues

        Returns:
            Number of issues cleaned.
        """
        result = self._gh_run_result(
            ["gh", "issue", "cleanup-closed"],
            timeout=60,
        )
        if not result.ok:
            debug_swallow("cleanup_closed_issues_labels", result.error)
            return 0
        if result.value.returncode != 0:
            return 0
        # Parse output to get count: "Cleaned N issue(s)" or "No closed issues..."
        output = result.value.stderr.strip()
        if "Cleaned" in output:
            match = re.search(r"Cleaned (\d+)", output)
            if match:
                cleaned = int(match.group(1))
                if cleaned > 0:
                    log_info(
                        f"✓ Cleaned workflow labels from {cleaned} closed issue(s)"
                    )
                return cleaned
        return 0
