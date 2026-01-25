# Copyright 2026 Dropbox, Inc.
# Author: Andrew Yates <ayates@dropbox.com>
# Licensed under the Apache License, Version 2.0

"""Issue management helpers for the looper runtime."""

import json
import re
import subprocess
from pathlib import Path
from typing import Any


class IssueManager:
    """Encapsulate GitHub issue operations for looper."""

    def __init__(self, repo_path: Path, role: str):
        self.repo_path = repo_path.resolve()
        self.role = role

    def _run(self, args: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        """Run a subprocess command anchored to the repo root."""
        return subprocess.run(args, cwd=self.repo_path, **kwargs)

    def check_stuck_issues(self) -> list[tuple[int, int]]:
        """Check for issues that may be stuck (many references, not closed)."""
        if self.role != "manager":
            return []

        try:
            # Get last 20 commits and count issue references
            result = self._run(
                ["git", "log", "--oneline", "-20", "--format=%s %b"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode != 0:
                return []

            # Count references to each issue (case-insensitive for auto-fixed "fixes")
            issue_counts: dict[int, int] = {}
            for match in re.finditer(
                r"(?:Fixes|Part of|Re:|Reopens) #(\d+)", result.stdout, re.IGNORECASE
            ):
                issue_num = int(match.group(1))
                issue_counts[issue_num] = issue_counts.get(issue_num, 0) + 1

            # Filter to issues with 5+ references (potentially stuck)
            stuck = [(num, count) for num, count in issue_counts.items() if count >= 5]
            return sorted(stuck, key=lambda x: -x[1])  # Sort by count descending
        except Exception:
            return []

    def check_thrashing_issues(self) -> list[tuple[int, int]]:
        """Check for issues with multiple close/reopen cycles (thrashing)."""
        if self.role != "manager":
            return []

        try:
            # Get repo info
            result = self._run(
                [
                    "gh",
                    "repo",
                    "view",
                    "--json",
                    "nameWithOwner",
                    "-q",
                    ".nameWithOwner",
                ],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode != 0:
                return []
            repo = result.stdout.strip()

            # Get open issues with in-progress label (actively worked)
            result = self._run(
                ["gh", "issue", "list", "--label", "in-progress", "--json", "number"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode != 0:
                return []

            issues = json.loads(result.stdout)
            thrashing = []

            for issue in issues[:10]:  # Limit to 10 issues to avoid API rate limits
                issue_num = issue["number"]
                # Get timeline events for this issue
                result = self._run(
                    [
                        "gh",
                        "api",
                        f"repos/{repo}/issues/{issue_num}/timeline",
                        "--jq",
                        '[.[] | select(.event=="reopened")] | length',
                    ],
                    capture_output=True,
                    text=True,
                    timeout=10,
                )
                if result.returncode == 0:
                    reopen_count = int(result.stdout.strip() or "0")
                    if reopen_count >= 2:  # 2+ reopens = thrashing
                        thrashing.append((issue_num, reopen_count))

            return sorted(thrashing, key=lambda x: -x[1])  # Sort by count descending
        except Exception:
            return []

    def check_closed_by_removal(self) -> list[tuple[int, str, int, int]]:
        """Check for issues closed primarily by removing/reverting code."""
        if self.role != "manager":
            return []

        try:
            # Get recently closed issues (last 7 days)
            result = self._run(
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
                capture_output=True,
                text=True,
                timeout=15,
            )
            if result.returncode != 0:
                return []

            closed_issues = []
            for line in result.stdout.strip().split("\n"):
                if line.strip():
                    try:
                        closed_issues.append(int(line.strip()))
                    except ValueError:
                        continue

            suspicious = []

            for issue_num in closed_issues[:10]:  # Limit API calls
                # Find the closing commit (Fixes #N)
                result = self._run(
                    [
                        "git",
                        "log",
                        "--oneline",
                        "-5",
                        f"--grep=Fixes #{issue_num}",
                        "--format=%H",
                    ],
                    capture_output=True,
                    text=True,
                    timeout=10,
                )
                if result.returncode != 0 or not result.stdout.strip():
                    continue

                commit_hash = result.stdout.strip().split("\n")[0]

                # Get diff stats for this commit
                stat_result = self._run(
                    ["git", "diff", "--shortstat", f"{commit_hash}^..{commit_hash}"],
                    capture_output=True,
                    text=True,
                    timeout=10,
                )
                if stat_result.returncode != 0:
                    continue

                # Parse "X files changed, Y insertions(+), Z deletions(-)"
                stat_line = stat_result.stdout.strip()
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

        except Exception:
            return []

    def check_blocker_cycles(self) -> list[dict[str, Any]]:
        """Check for blocker cycles where fixing one issue breaks another."""
        if self.role != "manager":
            return []

        try:
            cycles = []

            # Get recently closed issues (last 7 days)
            result = self._run(
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
                capture_output=True,
                text=True,
                timeout=15,
            )
            if result.returncode != 0:
                return []

            # Parse closed issues
            closed_issues = []
            for line in result.stdout.strip().split("\n"):
                if not line.strip():
                    continue
                try:
                    issue = json.loads(line)
                    closed_issues.append(issue)
                except json.JSONDecodeError:
                    continue

            # For each recently closed issue, check if it was closed by removing
            # code that an open issue still needs
            for closed in closed_issues[:5]:  # Limit API calls
                closed_num = closed["number"]

                # Find the closing commit
                result = self._run(
                    [
                        "git",
                        "log",
                        "--oneline",
                        "-10",
                        f"--grep=Fixes #{closed_num}",
                        "--format=%H %s",
                    ],
                    capture_output=True,
                    text=True,
                    timeout=10,
                )
                if result.returncode != 0 or not result.stdout.strip():
                    continue

                closing_line = result.stdout.strip().split("\n")[0]
                closing_hash = closing_line.split()[0]

                # Get files changed in closing commit
                result = self._run(
                    ["git", "diff", "--name-only", f"{closing_hash}^..{closing_hash}"],
                    capture_output=True,
                    text=True,
                    timeout=10,
                )
                if result.returncode != 0:
                    continue

                changed_files = [f for f in result.stdout.strip().split("\n") if f]
                if not changed_files:
                    continue

                # Get the diff to check if primarily deletions
                result = self._run(
                    ["git", "diff", "--shortstat", f"{closing_hash}^..{closing_hash}"],
                    capture_output=True,
                    text=True,
                    timeout=10,
                )
                if result.returncode != 0:
                    continue

                stat_line = result.stdout.strip()
                del_match = re.search(r"(\d+) deletion", stat_line)
                add_match = re.search(r"(\d+) insertion", stat_line)

                deletions = int(del_match.group(1)) if del_match else 0
                additions = int(add_match.group(1)) if add_match else 0

                # Only check if this was primarily a removal (deletions > additions)
                if deletions <= additions:
                    continue

                # Find commits in the last 30 days that added code to these files
                # and are referenced by still-open issues
                result = self._run(
                    [
                        "git",
                        "log",
                        "--oneline",
                        "--since=30 days ago",
                        "--format=%H %s",
                        "--",
                        *changed_files[:5],
                    ],  # Limit files
                    capture_output=True,
                    text=True,
                    timeout=15,
                )
                if result.returncode != 0:
                    continue

                # Look for commits that reference open issues
                for commit_line in result.stdout.strip().split("\n")[:20]:
                    if not commit_line.strip():
                        continue

                    # Extract issue references from commit message
                    matches = re.findall(
                        r"(?:Part of|Re:) #(\d+)", commit_line, re.IGNORECASE
                    )

                    for match in matches:
                        ref_issue = int(match)

                        # Skip if same as closed issue (self-referential)
                        if ref_issue == closed_num:
                            continue

                        # Check if this issue is still open
                        check_result = self._run(
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
                            capture_output=True,
                            text=True,
                            timeout=5,
                        )
                        if check_result.returncode != 0:
                            continue

                        if check_result.stdout.strip() == "OPEN":
                            # Found a potential cycle!
                            # The closed issue removed code that an open issue needs
                            feature_commit = commit_line.split()[0]

                            cycles.append(
                                {
                                    "closed_issue": closed_num,
                                    "closed_title": closed.get("title", ""),
                                    "open_issue": ref_issue,
                                    "feature_commit": feature_commit[:8],
                                    "removal_commit": closing_hash[:8],
                                    "changed_files": changed_files[:3],
                                }
                            )
                            break  # One cycle per closed issue is enough

            return cycles

        except Exception:
            return []

    def report_stuck_issues(self) -> None:
        """Print warning about potentially stuck issues, thrashing, and cycles."""
        stuck = self.check_stuck_issues()
        thrashing = self.check_thrashing_issues()
        closed_by_removal = self.check_closed_by_removal()
        blocker_cycles = self.check_blocker_cycles()

        if stuck:
            print()
            print("⚠️  STUCK ISSUE DETECTOR:")
            for issue_num, count in stuck[:3]:  # Top 3 most referenced
                print(
                    f"    #{issue_num}: {count} references in last 20 commits without closing"
                )
            print("    Consider: different approach, escalate, or close as won't-fix")
            print()

        if thrashing:
            print()
            print("🔄 THRASHING DETECTOR:")
            for issue_num, reopen_count in thrashing[:3]:  # Top 3 most thrashed
                print(f"    #{issue_num}: {reopen_count} close/reopen cycles")
            print(
                "    Escalate: Prover (verify original fix), Researcher (systemic analysis)"
            )
            print()

        if closed_by_removal:
            print()
            print("🗑️  CLOSED-BY-REMOVAL DETECTOR:")
            for issue_num, commit, adds, dels in closed_by_removal[:3]:
                print(f"    #{issue_num}: closed by {commit} (+{adds}/-{dels} lines)")
            print("    Review: Was root cause fixed, or just symptom hidden?")
            print(
                "    Valid removal requires: documented rationale OR alternative approach"
            )
            print()

        if blocker_cycles:
            print()
            print("🔁 BLOCKER CYCLE DETECTOR:")
            for cycle in blocker_cycles[:3]:  # Top 3 cycles
                print(
                    f"    Cycle: #{cycle['open_issue']} (open) ↔ #{cycle['closed_issue']} (closed)"
                )
                print(
                    f"      Feature added: {cycle['feature_commit']}, removed: {cycle['removal_commit']}"
                )
                print(f"      Files: {', '.join(cycle['changed_files'][:2])}")
            print("    This is a blocker cycle - fixing one issue breaks the other.")
            print(
                "    Escalate to USER: ensemble approach, accept trade-off, or redesign"
            )
            print("    File issue with `blocker-cycle` label for tracking.")
            print()

    def get_issue_checkboxes(self, issue_num: int) -> tuple[list[str], list[str], str]:
        """Get all checkboxes from issue body and comments."""
        unchecked = []
        checked = []
        original_body = ""

        try:
            # Get issue body
            result = self._run(
                [
                    "gh",
                    "issue",
                    "view",
                    str(issue_num),
                    "--json",
                    "body",
                    "-q",
                    ".body",
                ],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode == 0 and result.stdout:
                original_body = result.stdout
                unchecked.extend(re.findall(r"- \[ \] (.+?)(?:\n|$)", original_body))
                checked.extend(
                    re.findall(r"- \[x\] (.+?)(?:\n|$)", original_body, re.IGNORECASE)
                )

            # Get issue comments
            result = self._run(
                [
                    "gh",
                    "issue",
                    "view",
                    str(issue_num),
                    "--json",
                    "comments",
                    "-q",
                    ".comments[].body",
                ],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode == 0 and result.stdout:
                for line in result.stdout.split("\n"):
                    unchecked.extend(re.findall(r"- \[ \] (.+?)(?:\n|$)", line))
                    checked.extend(
                        re.findall(r"- \[x\] (.+?)(?:\n|$)", line, re.IGNORECASE)
                    )

        except Exception:
            pass

        # Normalize and deduplicate
        unchecked = list(
            dict.fromkeys(item.strip() for item in unchecked if item.strip())
        )
        checked = list(dict.fromkeys(item.strip() for item in checked if item.strip()))

        # Remove items that appear in both (checked wins)
        checked_lower = {item.lower() for item in checked}
        unchecked = [item for item in unchecked if item.lower() not in checked_lower]

        return unchecked, checked, original_body

    def get_existing_child_issues(self, parent_num: int) -> set[str]:
        """Get titles of existing child issues (Part of #N) to avoid duplicates."""
        existing = set()
        try:
            result = self._run(
                [
                    "gh",
                    "issue",
                    "list",
                    "--search",
                    f"Part of #{parent_num} in:body",
                    "--json",
                    "title",
                    "-q",
                    ".[].title",
                ],
                capture_output=True,
                text=True,
                timeout=15,
            )
            if result.returncode == 0:
                for title in result.stdout.strip().split("\n"):
                    if title.strip():
                        existing.add(title.strip().lower())
        except Exception:
            pass
        return existing

    def _update_issue_body_with_consolidated_checkboxes(
        self,
        issue_num: int,
        unchecked: list[str],
        checked: list[str],
        original_body: str,
    ) -> bool:
        """Update issue body with consolidated and cleaned checkbox list."""
        try:
            # Remove existing checkbox lines and ## Checklist header from body
            clean_body = re.sub(
                r"- \[[x ]\] .+?\n?", "", original_body, flags=re.IGNORECASE
            )
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

            new_body = clean_body + checkbox_section

            # Update issue
            result = self._run(
                ["gh", "issue", "edit", str(issue_num), "--body", new_body],
                capture_output=True,
                text=True,
                timeout=15,
            )
            return result.returncode == 0
        except Exception:
            return False

    def convert_unchecked_to_issues(self) -> int:
        """Convert unchecked checkboxes in worked issues to new issues."""
        try:
            # Get issue numbers from last commit
            result = self._run(
                ["git", "log", "-1", "--format=%s %b"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode != 0:
                return 0

            # Find issue references (case-insensitive for auto-fixed "fixes")
            issue_nums = set()
            for match in re.finditer(
                r"(?:Fixes|Part of|Re:|Claims) #(\d+)", result.stdout, re.IGNORECASE
            ):
                issue_nums.add(int(match.group(1)))

            if not issue_nums:
                return 0

            created_count = 0

            for issue_num in list(issue_nums)[:5]:  # Limit to 5 issues
                # Get all checkboxes from body and comments
                unchecked, checked, original_body = self.get_issue_checkboxes(issue_num)

                if not unchecked and not checked:
                    continue

                # Get existing child issues to avoid duplicates
                existing_children = self.get_existing_child_issues(issue_num)

                # Create new issues for unchecked items (that don't already exist)
                items_to_convert = []
                for item_text in unchecked[:10]:  # Limit to 10 items per issue
                    if item_text.lower() in existing_children:
                        print(f"    ⊘ Skipped (exists): {item_text[:40]}...")
                        continue
                    items_to_convert.append(item_text)

                for item_text in items_to_convert:
                    new_body = (
                        f"Part of #{issue_num}\n\n"
                        "Auto-created from unchecked item in parent issue."
                    )
                    create_result = self._run(
                        [
                            "gh",
                            "issue",
                            "create",
                            "--title",
                            item_text,
                            "--body",
                            new_body,
                        ],
                        capture_output=True,
                        text=True,
                        timeout=30,
                    )

                    if create_result.returncode == 0:
                        created_count += 1
                        new_url = create_result.stdout.strip()
                        print(
                            f"    ✓ Created: {item_text[:50]}{'...' if len(item_text) > 50 else ''}"
                        )
                        print(f"      {new_url}")
                        # Mark as checked now that issue exists
                        checked.append(item_text)
                        unchecked.remove(item_text)

                # Consolidate checkboxes in issue body
                if original_body:
                    if self._update_issue_body_with_consolidated_checkboxes(
                        issue_num, unchecked, checked, original_body
                    ):
                        print(f"    ✓ Consolidated checkboxes in #{issue_num}")

            return created_count
        except Exception as e:
            print(f"    ⚠️  Error converting checkboxes: {e}")
            return 0

    def process_unchecked_checkboxes(self) -> None:
        """Convert unchecked checkboxes to new issues at session end."""
        print()
        print("📋 Processing checkboxes...")
        created = self.convert_unchecked_to_issues()
        if created > 0:
            print(f"    Created {created} new issue(s) from unchecked items")
        print()

    def create_wip_followup(self, commit_hash: str, iteration: int) -> None:
        """Create a GitHub issue to track WIP commit follow-up."""
        try:
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
            result = self._run(
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
                capture_output=True,
                text=True,
                timeout=30,
            )
            if result.returncode == 0:
                # Extract issue URL from output
                issue_url = result.stdout.strip()
                print(f"✓ Created follow-up issue: {issue_url}")
            else:
                print(f"⚠️  Could not create WIP follow-up issue: {result.stderr}")
        except Exception as e:
            print(f"⚠️  Could not create WIP follow-up issue: {e}")
