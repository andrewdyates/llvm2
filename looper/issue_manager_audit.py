# Copyright 2026 Dropbox, Inc.
# Author: Andrew Yates <ayates@dropbox.com>
# Licensed under the Apache License, Version 2.0

"""Manager audit checks for issue health diagnostics.

Extracted from issue_manager.py per designs/2026-02-01-issue-manager-split.md.

Part of #1808.
"""

import json
import re
from datetime import UTC, datetime
from typing import Any

from ai_template_scripts.labels import IN_PROGRESS_ALL_LABELS
from looper.issue_manager_base import IssueManagerBase
from looper.log import debug_swallow, log_info, log_warning

# Optional batch GraphQL helper for thrashing checks.
try:
    from ai_template_scripts.gh_rate_limit import batch_issue_timelines
except ImportError:
    batch_issue_timelines = None  # type: ignore[misc, assignment]

__all__ = ["IssueAuditor"]


class IssueAuditor(IssueManagerBase):
    """Manager-only health checks and issue diagnostics.

    All check_* methods return empty results if role != "manager".
    This class can be used standalone or via IssueManager delegation.

    Contracts:
        REQUIRES: repo_path is a valid Path
        REQUIRES: role is a non-empty string
        ENSURES: All check_* methods return empty results on error
    """

    def check_stuck_issues(self) -> list[tuple[int, int]]:
        """Check for issues that may be stuck (many references, not closed).

        Contracts:
            REQUIRES: self.role is set
            ENSURES: Returns empty list if self.role != "manager"
            ENSURES: Returns list of (issue_num, ref_count) tuples
            ENSURES: Only includes issues with 5+ references in last 20 commits
            ENSURES: Results sorted by ref_count descending
            ENSURES: Never raises - catches all exceptions
        """
        if self.role != "manager":
            return []

        try:
            # Get last 20 commits and count issue references
            result = self._run_result(
                ["git", "log", "--oneline", "-20", "--format=%s %b"], timeout=10
            )
            if not result.ok:
                debug_swallow("check_stuck_issues", result.error)
                return []

            # Count references to each issue (case-insensitive for auto-fixed "fixes")
            issue_counts: dict[int, int] = {}
            for match in re.finditer(
                r"(?:Fixes|Part of|Re:|Reopens) #(\d+)",
                result.value.stdout,
                re.IGNORECASE,
            ):
                issue_num = int(match.group(1))
                issue_counts[issue_num] = issue_counts.get(issue_num, 0) + 1

            # Filter to issues with 5+ references (potentially stuck)
            stuck = [(num, count) for num, count in issue_counts.items() if count >= 5]
            return sorted(stuck, key=lambda x: -x[1])  # Sort by count descending
        except Exception as e:
            debug_swallow("check_stuck_issues", e)
            return []

    def check_thrashing_issues(self) -> list[tuple[int, int]]:
        """Check for issues with multiple close/reopen cycles (thrashing).

        Contracts:
            REQUIRES: self.role is set
            ENSURES: Returns empty list if self.role != "manager"
            ENSURES: Returns list of (issue_num, reopen_count) tuples
            ENSURES: Only includes issues with 2+ reopen events
            ENSURES: Results sorted by reopen_count descending
            ENSURES: Makes 3 API calls with batch timeline fetch (2 + N on fallback)
            ENSURES: Never raises - catches all exceptions
        """
        if self.role != "manager":
            return []

        try:
            # Get repo info
            result = self._gh_run_result(
                [
                    "gh",
                    "repo",
                    "view",
                    "--json",
                    "nameWithOwner",
                    "-q",
                    ".nameWithOwner",
                ],
                timeout=10,
            )
            if not result.ok:
                debug_swallow("check_thrashing_issues_repo", result.error)
                return []
            repo = result.value.stdout.strip()

            # Get all open issues with labels in a single query (#1111 optimization)
            # This replaces 15+ sequential subprocess calls with 1 call + Python filter
            result = self._gh_run_result(
                [
                    "gh",
                    "issue",
                    "list",
                    "--state",
                    "open",
                    "--json",
                    "number,labels",
                    "--limit",
                    "200",
                ],
                timeout=15,
            )
            if not result.ok:
                debug_swallow("check_thrashing_issues_list", result.error)
                return []

            # Filter issues that have any in-progress label
            # Handle both GraphQL ({"name": "P1"}) and REST (["P1"]) formats
            in_progress_set = set(IN_PROGRESS_ALL_LABELS)
            issues = []
            for issue in json.loads(result.value.stdout):
                labels = issue.get("labels", [])
                issue_labels = {
                    lbl.get("name", lbl) if isinstance(lbl, dict) else lbl
                    for lbl in labels
                }
                if issue_labels & in_progress_set:  # Set intersection
                    issues.append(issue)
            thrashing = []
            issue_nums = [issue["number"] for issue in issues[:10]]
            reopen_counts: dict[int, int | None] = {}
            if batch_issue_timelines is not None and issue_nums:
                reopen_counts = batch_issue_timelines(
                    issue_nums,
                    repo=repo,
                    timeout=15,
                )

            batch_failed = not reopen_counts or all(
                count is None for count in reopen_counts.values()
            )

            for issue in issues[:10]:  # Limit to 10 issues to avoid API rate limits
                issue_num = issue["number"]
                reopen_count = reopen_counts.get(issue_num)
                if reopen_count is None or batch_failed:
                    # Fallback: per-issue REST timeline query
                    tl_result = self._gh_run_result(
                        [
                            "gh",
                            "api",
                            f"repos/{repo}/issues/{issue_num}/timeline",
                            "--jq",
                            '[.[] | select(.event=="reopened")] | length',
                        ],
                        timeout=10,
                    )
                    if not tl_result.ok:
                        continue
                    reopen_count = int(tl_result.value.stdout.strip() or "0")
                if reopen_count >= 2:  # 2+ reopens = thrashing
                    thrashing.append((issue_num, reopen_count))

            return sorted(thrashing, key=lambda x: -x[1])  # Sort by count descending
        except Exception as e:
            debug_swallow("check_thrashing_issues", e)
            return []

    def check_closed_by_removal(self) -> list[tuple[int, str, int, int]]:
        """Check for issues closed primarily by removing/reverting code.

        Contracts:
            REQUIRES: self.role is set
            ENSURES: Returns empty list if self.role != "manager"
            ENSURES: Returns list of (issue_num, commit_hash, additions, deletions) tuples
            ENSURES: Only includes issues where deletions > 2x additions
            ENSURES: Limits to issues closed in last 7 days
            ENSURES: Limits API calls to 10 issues
            ENSURES: Never raises - catches all exceptions
        """
        if self.role != "manager":
            return []

        try:
            # Get recently closed issues (last 7 days)
            result = self._gh_run_result(
                [
                    "gh",
                    "issue",
                    "list",
                    "--state",
                    "closed",
                    "--json",
                    "number,closedAt",
                    "-q",
                    ".[] | select(.closedAt > (now - 7*24*3600 | todate)) | .number",
                ],
                timeout=15,
            )
            if not result.ok:
                debug_swallow("check_closed_by_removal", result.error)
                return []

            closed_issues = []
            for line in result.value.stdout.strip().split("\n"):
                if line.strip():
                    try:
                        closed_issues.append(int(line.strip()))
                    except ValueError as e:
                        debug_swallow("check_closed_by_removal_issue_num", e)
                        continue

            suspicious = []

            for issue_num in closed_issues[:10]:  # Limit API calls
                # Find the closing commit (Fixes #N)
                log_result = self._run_result(
                    [
                        "git",
                        "log",
                        "--oneline",
                        "-5",
                        f"--grep=Fixes #{issue_num}",
                        "--format=%H",
                    ],
                    timeout=10,
                )
                if not log_result.ok or not log_result.value.stdout.strip():
                    continue

                commit_hash = log_result.value.stdout.strip().split("\n")[0]

                # Get diff stats for this commit
                stat_result = self._run_result(
                    ["git", "diff", "--shortstat", f"{commit_hash}^..{commit_hash}"],
                    timeout=10,
                )
                if not stat_result.ok:
                    continue

                # Parse "X files changed, Y insertions(+), Z deletions(-)"
                stat_line = stat_result.value.stdout.strip()
                additions = 0
                deletions = 0

                add_match = re.search(r"(\d+) insertion", stat_line)
                del_match = re.search(r"(\d+) deletion", stat_line)

                if add_match:
                    additions = int(add_match.group(1))
                if del_match:
                    deletions = int(del_match.group(1))

                # Flag if deletions > 2x additions (primarily removal)
                if deletions > 0 and deletions > (additions * 2):
                    suspicious.append(
                        (issue_num, commit_hash[:8], additions, deletions)
                    )

            return suspicious

        except Exception as e:
            debug_swallow("check_closed_by_removal", e)
            return []

    def _get_recently_closed_issues(self) -> list[dict[str, Any]]:
        """Get issues closed in the last 7 days.

        Contracts:
            REQUIRES: none (best-effort; gh/jq failures return empty list)
            ENSURES: Returns a list of dicts with number, title, and closedAt
                for issues closed within the last 7 days
            ENSURES: Skips malformed JSON lines; returns empty list on gh failure

        Returns:
            List of dicts with number, title, closedAt keys. Empty on error.
        """
        result = self._gh_run_result(
            [
                "gh",
                "issue",
                "list",
                "--state",
                "closed",
                "--json",
                "number,title,closedAt",
                "-q",
                ".[] | select(.closedAt > (now - 7*24*3600 | todate))",
            ],
            timeout=15,
        )
        if not result.ok:
            debug_swallow("get_recently_closed_issues", result.error)
            return []

        closed_issues = []
        for line in result.value.stdout.strip().split("\n"):
            if not line.strip():
                continue
            try:
                issue = json.loads(line)
                closed_issues.append(issue)
            except json.JSONDecodeError as e:
                debug_swallow("get_recently_closed_issues_json", e)
                continue
        return closed_issues

    def _get_closing_commit_info(
        self, issue_num: int
    ) -> tuple[str, list[str], int, int] | None:
        """Get info about the commit that closed an issue.

        Contracts:
            REQUIRES: issue_num is a positive integer
            ENSURES: Returns (commit_hash, changed_files, additions, deletions)
                when a closing commit and diff stats are found
            ENSURES: Returns None when git commands fail or no files are changed

        Args:
            issue_num: The issue number to look up.

        Returns:
            Tuple of (commit_hash, changed_files, additions, deletions) or None.
        """
        # Find the closing commit
        log_result = self._run_result(
            [
                "git",
                "log",
                "--oneline",
                "-10",
                f"--grep=Fixes #{issue_num}",
                "--format=%H %s",
            ],
            timeout=10,
        )
        if not log_result.ok or not log_result.value.stdout.strip():
            return None

        closing_line = log_result.value.stdout.strip().split("\n")[0]
        parts = closing_line.split()
        if not parts:
            return None
        closing_hash = parts[0]

        # Get files changed in closing commit
        files_result = self._run_result(
            ["git", "diff", "--name-only", f"{closing_hash}^..{closing_hash}"],
            timeout=10,
        )
        if not files_result.ok:
            return None

        changed_files = [f for f in files_result.value.stdout.strip().split("\n") if f]
        if not changed_files:
            return None

        # Get the diff stats
        stat_result = self._run_result(
            ["git", "diff", "--shortstat", f"{closing_hash}^..{closing_hash}"],
            timeout=10,
        )
        if not stat_result.ok:
            return None

        stat_line = stat_result.value.stdout.strip()
        del_match = re.search(r"(\d+) deletion", stat_line)
        add_match = re.search(r"(\d+) insertion", stat_line)
        deletions = int(del_match.group(1)) if del_match else 0
        additions = int(add_match.group(1)) if add_match else 0

        return closing_hash, changed_files, additions, deletions

    def _find_affected_open_issues(
        self, closed_num: int, changed_files: list[str], closing_hash: str
    ) -> dict[str, Any] | None:
        """Find open issues affected by a removal commit.

        Contracts:
            REQUIRES: closed_num is a positive integer
            REQUIRES: changed_files is non-empty
            ENSURES: Returns None when git log fails or no open issues are found
            ENSURES: Returns a dict containing open_issue, feature_commit,
                removal_commit, and changed_files when a candidate is found

        Args:
            closed_num: The closed issue number (to skip self-references).
            changed_files: Files changed in the closing commit.
            closing_hash: The commit hash that closed the issue.

        Returns:
            Cycle dict if found, None otherwise.
        """
        # Find commits in the last 30 days that touched these files
        log_result = self._run_result(
            [
                "git",
                "log",
                "--oneline",
                "--since=30 days ago",
                "--format=%H %s",
                "--",
                *changed_files[:5],
            ],
            timeout=15,
        )
        if not log_result.ok:
            return None

        # Look for commits that reference open issues
        for commit_line in log_result.value.stdout.strip().split("\n")[:20]:
            if not commit_line.strip():
                continue

            matches = re.findall(r"(?:Part of|Re:) #(\d+)", commit_line, re.IGNORECASE)

            for match in matches:
                ref_issue = int(match)

                # Skip self-referential
                if ref_issue == closed_num:
                    continue

                # Check if this issue is still open
                check_result = self._gh_run_result(
                    [
                        "gh",
                        "issue",
                        "view",
                        str(ref_issue),
                        "--json",
                        "state",
                        "-q",
                        ".state",
                    ],
                    timeout=5,
                )
                if not check_result.ok:
                    continue

                if check_result.value.stdout.strip() == "OPEN":
                    feature_commit = commit_line.split()[0]
                    return {
                        "open_issue": ref_issue,
                        "feature_commit": feature_commit[:8],
                        "removal_commit": closing_hash[:8],
                        "changed_files": changed_files[:3],
                    }
        return None

    def check_blocker_cycles(self) -> list[dict[str, Any]]:
        """Check for blocker cycles where fixing one issue breaks another.

        Contracts:
            REQUIRES: self.role is set
            ENSURES: Returns empty list if self.role != "manager"
            ENSURES: Returns list of cycle dicts with keys:
                     closed_issue, closed_title, open_issue, feature_commit,
                     removal_commit, changed_files
            ENSURES: Only detects cycles from issues closed in last 7 days
            ENSURES: Limits to 5 recently closed issues
            ENSURES: Never raises - catches all exceptions
        """
        if self.role != "manager":
            return []

        try:
            cycles = []
            closed_issues = self._get_recently_closed_issues()

            for closed in closed_issues[:5]:  # Limit API calls
                closed_num = closed["number"]

                commit_info = self._get_closing_commit_info(closed_num)
                if commit_info is None:
                    continue

                closing_hash, changed_files, additions, deletions = commit_info

                # Only check if this was primarily a removal
                if deletions <= additions:
                    continue

                cycle = self._find_affected_open_issues(
                    closed_num, changed_files, closing_hash
                )
                if cycle:
                    cycle["closed_issue"] = closed_num
                    cycle["closed_title"] = closed.get("title", "")
                    cycles.append(cycle)

            return cycles

        except Exception as e:
            debug_swallow("check_blocker_cycles", e)
            return []

    def check_escalation_sla(self, sla_days: int = 3) -> list[tuple[int, str, int]]:
        """Check for escalated issues exceeding SLA.

        Contracts:
            REQUIRES: self.role is set
            REQUIRES: sla_days is a positive integer
            ENSURES: Returns empty list if self.role != "manager"
            ENSURES: Returns list of (issue_num, title, age_days) tuples
            ENSURES: Only includes issues with 'escalate' label older than sla_days
            ENSURES: Results sorted by age_days descending (oldest first)
            ENSURES: Never raises - catches all exceptions
            ENSURES: Handles both Z suffix and +00:00 timezone formats
            ENSURES: Returns [] on timezone-naive datetime (TypeError caught)
        """
        if self.role != "manager":
            return []

        try:
            result = self._gh_run_result(
                [
                    "gh",
                    "issue",
                    "list",
                    "--state",
                    "open",
                    "--label",
                    "escalate",
                    "--json",
                    "number,title,createdAt",
                ],
                timeout=15,
            )

            if not result.ok:
                debug_swallow("check_escalation_sla", result.error)
                return []

            stale = []
            now = datetime.now(UTC)
            for issue in json.loads(result.value.stdout):
                created = datetime.fromisoformat(issue["createdAt"])
                age_days = (now - created).days
                if age_days >= sla_days:
                    stale.append((issue["number"], issue["title"], age_days))

            return sorted(stale, key=lambda x: -x[2])  # Oldest first

        except Exception as e:
            debug_swallow("check_escalation_sla", e)
            return []

    def report_stuck_issues(self, escalation_sla_days: int | None = None) -> None:
        """Print warning about potentially stuck issues, thrashing, and cycles.

        Args:
            escalation_sla_days: Days before escalated issues trigger SLA warning.
                Defaults to 3 if not provided (from check_escalation_sla default).

        Contracts:
            REQUIRES: self.role is set
            ENSURES: Prints diagnostic output to stdout
            ENSURES: Silent if no issues detected
            ENSURES: Never raises - delegates to check_* methods that catch exceptions
        """
        stuck = self.check_stuck_issues()
        thrashing = self.check_thrashing_issues()
        closed_by_removal = self.check_closed_by_removal()
        blocker_cycles = self.check_blocker_cycles()
        sla_days = escalation_sla_days if escalation_sla_days is not None else 3
        escalation_stale = self.check_escalation_sla(sla_days=sla_days)

        if stuck:
            log_info("")
            log_warning("⚠️  STUCK ISSUE DETECTOR:")
            for issue_num, count in stuck[:3]:  # Top 3 most referenced
                log_warning(
                    f"    #{issue_num}: {count} references in last 20 commits "
                    "without closing"
                )
            log_warning(
                "    Consider: different approach, escalate, or close as won't-fix"
            )
            log_info("")

        if thrashing:
            log_info("")
            log_warning("🔄 THRASHING DETECTOR:")
            for issue_num, reopen_count in thrashing[:3]:  # Top 3 most thrashed
                log_warning(f"    #{issue_num}: {reopen_count} close/reopen cycles")
            log_warning(
                "    Escalate: Prover (verify original fix), Researcher "
                "(systemic analysis)"
            )
            log_info("")

        if closed_by_removal:
            log_info("")
            log_warning("🗑️  CLOSED-BY-REMOVAL DETECTOR:")
            for issue_num, commit, adds, dels in closed_by_removal[:3]:
                log_warning(
                    f"    #{issue_num}: closed by {commit} (+{adds}/-{dels} lines)"
                )
            log_warning("    Review: Was root cause fixed, or just symptom hidden?")
            log_warning(
                "    Valid removal requires: documented rationale OR alternative "
                "approach"
            )
            log_info("")

        if blocker_cycles:
            log_info("")
            log_warning("🔁 BLOCKER CYCLE DETECTOR:")
            for cycle in blocker_cycles[:3]:  # Top 3 cycles
                log_warning(
                    f"    Cycle: #{cycle['open_issue']} (open) ↔ "
                    f"#{cycle['closed_issue']} (closed)"
                )
                log_warning(
                    f"      Feature added: {cycle['feature_commit']}, removed: "
                    f"{cycle['removal_commit']}"
                )
                log_warning(f"      Files: {', '.join(cycle['changed_files'][:2])}")
            log_warning(
                "    This is a blocker cycle - fixing one issue breaks the other."
            )
            log_warning(
                "    Escalate to USER: ensemble approach, accept trade-off, or redesign"
            )
            log_warning("    File issue with `blocker-cycle` label for tracking.")
            log_info("")

        if escalation_stale:
            log_info("")
            log_warning("⏰ ESCALATION SLA EXCEEDED:")
            for issue_num, title, age_days in escalation_stale[:5]:
                # Truncate title to 50 chars for readability
                title_display = title[:50] + "..." if len(title) > 50 else title
                log_warning(f"    #{issue_num}: {title_display} ({age_days} days)")
            log_warning("    Action: USER decision required")
            log_info("")
