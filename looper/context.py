# Copyright 2026 Dropbox, Inc.
# Author: Andrew Yates
# Licensed under the Apache License, Version 2.0

"""
looper/context.py - Session context gathering

Gathers context injected into AI prompts at session start:
- Git log (recent commits)
- GitHub issues (sampled by priority)
- Last directive (## Next from prior same-role commit)
- Other role feedback (recent commits from M > R/P > W)
- @ROLE mentions
- System status (memory/disk)
- Audit data (for manager)

Copyright 2026 Dropbox, Inc.
Created by Andrew Yates
Licensed under the Apache License, Version 2.0
"""

import json
import random
import re
import shutil
import subprocess
from pathlib import Path

from looper.telemetry import get_health_summary


def run_session_start_commands(role: str) -> dict[str, str]:
    """Execute session start commands and capture output.

    Args:
        role: Current role (worker, manager, researcher, prover)

    Returns:
        Dict with injection content:
        - git_log: Recent commits
        - gh_issues: Sampled issues (important, newest, random, oldest)
        - last_directive: ## Next section from last same-role commit
        - other_feedback: Recent commits from other roles
        - role_mentions: @ROLE mentions directed at this role
        - system_status: Memory/disk usage
        - audit_data: Pre-computed audit info (manager only)
    """
    results: dict[str, str] = {}

    # git log --oneline -10
    try:
        result = subprocess.run(
            ["git", "log", "--oneline", "-10"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            results["git_log"] = result.stdout.strip() or "(no commits)"
        else:
            results["git_log"] = f"(git log failed: {result.stderr.strip()})"
    except subprocess.TimeoutExpired:
        results["git_log"] = "(git log timed out)"
    except Exception as e:
        results["git_log"] = f"(git log error: {e})"

    # Sampled issues (role-filtered: Worker sees all, others see P0 + domain)
    results["gh_issues"] = _get_sampled_issues(role)

    # Last directive from same role's ## Next section
    results["last_directive"] = _get_last_directive(role)

    # Feedback from other roles
    results["other_feedback"] = _get_other_role_feedback(role)

    # @ROLE mentions directed at this role
    results["role_mentions"] = _get_role_mentions(role)

    # System status (memory/disk)
    results["system_status"] = _get_system_status()

    # Pre-computed audit data for manager
    if role == "manager":
        results["audit_data"] = _get_audit_data()
    else:
        results["audit_data"] = ""

    return results


def _get_sampled_issues(role: str = "worker") -> str:
    """Get role-filtered sampled issues.

    Worker sees all issues in priority order:
    - P0 (always first - system compromised)
    - do-audit (workflow gate)
    - in-progress (current work)
    - All urgent (sorted by P-level: urgent P1 > urgent P2 > urgent P3)
    - P1, P2, P3 (non-urgent by priority)
    - Newest, random, oldest (discovery)

    Non-Worker roles see P0 + domain-specific issues:
    - Manager: needs-review
    - Prover: testing
    - Researcher: research, design
    """
    # Role domain labels
    # NOTE: Labels must match what's configured in the repo. Use gh label list to verify.
    # "testing" is the standard label for test/proof work (not "test" or "proof")
    role_domains: dict[str, list[str]] = {
        "manager": ["needs-review"],
        "prover": ["testing"],
        "researcher": ["research", "design"],
    }

    try:
        result = subprocess.run(
            [
                "gh",
                "issue",
                "list",
                "--state",
                "open",
                "--limit",
                "100",
                "--json",
                "number,title,labels,createdAt",
            ],
            capture_output=True,
            text=True,
            timeout=15,
        )
        if result.returncode != 0:
            return f"(gh issue list failed: {result.stderr.strip()})"

        issues = json.loads(result.stdout)
        if not issues:
            return "(no open issues)"

        def get_labels(issue: dict) -> list[str]:
            return [label["name"] for label in issue.get("labels", [])]

        def has_label(issue: dict, label: str) -> bool:
            return label in get_labels(issue)

        def has_any_label(issue: dict, labels: list[str]) -> bool:
            issue_labels = get_labels(issue)
            return any(lbl in issue_labels for lbl in labels)

        def format_issue(issue: dict) -> str:
            labels = ", ".join(get_labels(issue)) or "-"
            return f"#{issue['number']}: {issue['title'][:60]} [{labels}]"

        def get_p_level(issue: dict) -> int:
            """Return priority level (0-3), 4 for unlabeled."""
            for p in range(4):
                if has_label(issue, f"P{p}"):
                    return p
            return 4

        # Non-Worker: show P0 + domain issues only
        domain_labels = role_domains.get(role)
        if domain_labels:
            return _get_domain_issues(
                issues,
                domain_labels,
                format_issue,
                has_label,
                has_any_label,
                get_p_level,
            )

        # Worker: full priority sampling
        # Exclude tracking issues - known limitations, not actionable work
        issues = [i for i in issues if not has_label(i, "tracking")]

        shown: set[int] = set()
        lines: list[str] = []

        # P0 first (always - system compromised)
        p0 = [i for i in issues if has_label(i, "P0")]
        for issue in p0:
            lines.append(f"[P0] {format_issue(issue)}")
            shown.add(issue["number"])

        # do-audit second (workflow gate - ready for self-audit)
        do_audit = [
            i for i in issues if has_label(i, "do-audit") and i["number"] not in shown
        ]
        for issue in do_audit[:5]:
            lines.append(f"[DO-AUDIT] {format_issue(issue)}")
            shown.add(issue["number"])

        # In-progress third (current work)
        in_progress = [
            i
            for i in issues
            if has_label(i, "in-progress") and i["number"] not in shown
        ]
        for issue in in_progress[:5]:
            lines.append(f"[IN-PROGRESS] {format_issue(issue)}")
            shown.add(issue["number"])

        # ALL urgent issues first (sorted by P-level within)
        urgent = [
            i for i in issues if has_label(i, "urgent") and i["number"] not in shown
        ]
        urgent.sort(key=get_p_level)
        for issue in urgent[:5]:
            p = get_p_level(issue)
            prefix = f"[URGENT P{p}]" if p < 4 else "[URGENT]"
            lines.append(f"{prefix} {format_issue(issue)}")
            shown.add(issue["number"])

        # P1 (top 3, non-urgent)
        p1 = [i for i in issues if has_label(i, "P1") and i["number"] not in shown]
        for issue in p1[:3]:
            lines.append(f"[P1] {format_issue(issue)}")
            shown.add(issue["number"])

        # P2 (top 2, non-urgent)
        p2 = [i for i in issues if has_label(i, "P2") and i["number"] not in shown]
        for issue in p2[:2]:
            lines.append(f"[P2] {format_issue(issue)}")
            shown.add(issue["number"])

        # P3 (top 1)
        p3 = [i for i in issues if has_label(i, "P3") and i["number"] not in shown]
        for issue in p3[:1]:
            lines.append(f"[P3] {format_issue(issue)}")
            shown.add(issue["number"])

        remaining = [i for i in issues if i["number"] not in shown]

        # Newest 2 not shown
        newest = sorted(remaining, key=lambda i: i["createdAt"], reverse=True)[:2]
        for issue in newest:
            lines.append(f"[NEW] {format_issue(issue)}")
            shown.add(issue["number"])

        remaining = [i for i in remaining if i["number"] not in shown]

        # Random 1
        if remaining:
            rand_issue = random.choice(remaining)
            lines.append(f"[RANDOM] {format_issue(rand_issue)}")
            shown.add(rand_issue["number"])
            remaining = [i for i in remaining if i["number"] not in shown]

        # Oldest 1
        if remaining:
            oldest = sorted(remaining, key=lambda i: i["createdAt"])[0]
            lines.append(f"[OLDEST] {format_issue(oldest)}")

        if not lines:
            return "(no open issues)"

        return "\n".join(lines)

    except subprocess.TimeoutExpired:
        return "(gh issue list timed out)"
    except Exception as e:
        return f"(gh issue list error: {e})"


def _get_domain_issues(
    issues: list[dict],
    domain_labels: list[str],
    format_issue,
    has_label,
    has_any_label,
    get_p_level,
) -> str:
    """Get P0 + domain-specific issues for non-Worker roles.

    Shows P0 first (all roles see system-compromised issues),
    then domain issues sorted by urgency and priority.
    """
    lines: list[str] = []
    shown: set[int] = set()

    # P0 first (all roles should see system-compromised issues)
    p0 = [i for i in issues if has_label(i, "P0")]
    for issue in p0:
        lines.append(f"[P0] {format_issue(issue)}")
        shown.add(issue["number"])

    # Domain issues: filter by label
    domain = [
        i
        for i in issues
        if has_any_label(i, domain_labels) and i["number"] not in shown
    ]

    # Sort: urgent first, then by P-level
    def domain_sort_key(issue: dict) -> tuple[int, int]:
        urgent = 0 if has_label(issue, "urgent") else 1
        return (urgent, get_p_level(issue))

    domain.sort(key=domain_sort_key)

    for issue in domain[:10]:  # Max 10 domain issues
        p = get_p_level(issue)
        is_urgent = has_label(issue, "urgent")
        if is_urgent:
            prefix = f"[URGENT P{p}]" if p < 4 else "[URGENT]"
        else:
            prefix = f"[P{p}]" if p < 4 else "[DOMAIN]"
        lines.append(f"{prefix} {format_issue(issue)}")
        shown.add(issue["number"])

    if not lines:
        return "(no issues in your domain)"

    return "\n".join(lines)


def get_issue_by_number(issue_num: int) -> str:
    """Get a single issue by number for focused audit context.

    Args:
        issue_num: The issue number to fetch.

    Returns:
        Formatted issue string, or empty string if not found.
    """
    try:
        result = subprocess.run(
            [
                "gh",
                "issue",
                "view",
                str(issue_num),
                "--json",
                "number,title,labels,state",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode != 0:
            return ""

        issue = json.loads(result.stdout)
        labels = [label["name"] for label in issue.get("labels", [])]
        label_str = ", ".join(labels) or "-"
        state = issue.get("state", "UNKNOWN")
        return f"[WORKING] #{issue['number']}: {issue['title']} [{label_str}] ({state})"

    except Exception:
        return ""


def _get_last_directive(role: str) -> str:
    """Extract ## Next section from last commit by same role.

    Args:
        role: Current role (worker, manager, etc.)

    Returns:
        The ## Next directive, or empty string if not found.
    """
    role_prefix = role[0].upper()  # W, M, R, P
    try:
        # Find last commit by this role
        result = subprocess.run(
            ["git", "log", f"--grep=^\\[{role_prefix}\\]", "--format=%H", "-1"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode != 0 or not result.stdout.strip():
            return ""

        commit_hash = result.stdout.strip()

        # Get full commit message
        result = subprocess.run(
            ["git", "log", "-1", "--format=%B", commit_hash],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode != 0:
            return ""

        msg = result.stdout

        # Extract ## Next section
        if "## Next" not in msg:
            return ""

        lines = msg.split("\n")
        in_next = False
        next_lines: list[str] = []
        for line in lines:
            if line.startswith("## Next"):
                in_next = True
                continue
            if in_next:
                if line.startswith(("## ", "---")):
                    break
                next_lines.append(line)

        directive = "\n".join(next_lines).strip()
        if directive:
            return f"From [{role_prefix}] commit {commit_hash[:7]}:\n{directive}"
        return ""

    except Exception:
        return ""


def _get_other_role_feedback(current_role: str) -> str:
    """Get recent commits from other roles, prioritized by importance.

    Priority order: Manager > Researcher/Prover > Worker
    Excludes current role from results.

    Returns:
        Formatted feedback from other roles, or empty string.
    """
    role_prefix = current_role[0].upper()
    # Priority order: M first, then R and P, then W
    priority_order = ["M", "R", "P", "W"]
    other_roles = [r for r in priority_order if r != role_prefix]

    try:
        result = subprocess.run(
            ["git", "log", "--oneline", "-30"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode != 0:
            return ""

        lines = result.stdout.strip().split("\n")

        # Collect commits by role, preserving recency within each role
        by_role: dict[str, list[str]] = {r: [] for r in other_roles}
        for line in lines:
            for role in other_roles:
                if f"[{role}]" in line:
                    if len(by_role[role]) < 2:  # Max 2 per role
                        by_role[role].append(line)
                    break

        # Build output in priority order
        feedback_lines: list[str] = []
        for role in other_roles:
            feedback_lines.extend(by_role[role])

        if not feedback_lines:
            return ""

        return "\n".join(feedback_lines[:5])  # Max 5 total

    except Exception:
        return ""


def _has_role_mention(line: str, tag: str) -> bool:
    """Check if line contains @TAG as a word (not substring).

    Uses word boundary to prevent @WORKER matching @WORKERS.
    Matches @TAG followed by colon, whitespace, or end of line.
    """
    # Pattern: @TAG followed by : or whitespace or end of string
    pattern = re.escape(tag) + r"(?:[:\s]|$)"
    return bool(re.search(pattern, line))


def _get_role_mentions(current_role: str) -> str:
    """Extract @ROLE mentions directed at the current role from recent commits.

    Searches recent commit messages for @WORKER, @PROVER, @RESEARCHER, @MANAGER, @ALL
    and returns lines directed at the current role.

    Args:
        current_role: Current role (worker, manager, researcher, prover)

    Returns:
        Formatted mentions directed at this role, or empty string.
    """
    role_upper = current_role.upper()
    role_tag = f"@{role_upper}"

    try:
        # Get full commit messages from recent commits (not just titles)
        result = subprocess.run(
            ["git", "log", "--format=%B---COMMIT_SEP---", "-20"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode != 0:
            return ""

        commits = result.stdout.split("---COMMIT_SEP---")
        mentions: list[str] = []
        seen: set[str] = set()  # Deduplicate mentions

        for commit in commits:
            if not commit.strip():
                continue

            for line in commit.split("\n"):
                line = line.strip()
                if not line:
                    continue

                # Check for direct mention or @ALL using word boundary
                if _has_role_mention(line, role_tag) or _has_role_mention(line, "@ALL"):
                    # Clean up the line - remove the @TAG prefix if it starts with it
                    clean_line = line
                    if clean_line.startswith(role_tag + ":"):
                        clean_line = clean_line[len(role_tag) + 1 :].strip()
                    elif clean_line.startswith("@ALL:"):
                        clean_line = clean_line[5:].strip()

                    # Deduplicate
                    if clean_line in seen:
                        continue
                    seen.add(clean_line)

                    mentions.append(f"- {clean_line}")
                    if len(mentions) >= 5:  # Max 5 mentions
                        break

            if len(mentions) >= 5:
                break

        if not mentions:
            return ""

        return "\n".join(mentions)

    except Exception:
        return ""


def _get_system_status() -> str:
    """Get concise system status for session start.

    Returns one line: "Mem: XX% | Disk: XX%" or empty if unavailable.
    """
    try:
        parts = []

        # Memory - macOS via vm_stat
        result = subprocess.run(["vm_stat"], capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            lines = result.stdout.split("\n")
            free = active = inactive = wired = 0
            for line in lines:
                if "Pages free:" in line:
                    free = int(line.split()[-1].rstrip("."))
                elif "Pages active:" in line:
                    active = int(line.split()[-1].rstrip("."))
                elif "Pages inactive:" in line:
                    inactive = int(line.split()[-1].rstrip("."))
                elif "Pages wired" in line:
                    wired = int(line.split()[-1].rstrip("."))

            total = free + active + inactive + wired
            if total > 0:
                used_pct = int((active + wired) * 100 / total)
                parts.append(f"Mem: {used_pct}%")

        # Disk
        disk = shutil.disk_usage(".")
        disk_pct = int(disk.used * 100 / disk.total)
        parts.append(f"Disk: {disk_pct}%")

        if not parts:
            return ""

        status = " | ".join(parts)

        # Add warning if high
        if "Mem:" in status:
            mem_val = int(status.split("Mem: ")[1].split("%")[0])
            if mem_val >= 90:
                status += " ⚠️ CRITICAL"
            elif mem_val >= 80:
                status += " ⚠️ HIGH"

        # Add looper health summary if available
        try:
            looper_health = get_health_summary(24)
            if looper_health:
                status += f"\nLooper: {looper_health}"
        except Exception:
            pass  # Telemetry unavailable or failed

        return status

    except Exception:
        return ""


def _get_audit_data() -> str:
    """Pre-run audit scripts for Manager and return output.

    Runs health_check.py, checks flags, reads metrics.
    Returns formatted output for injection into Manager context.
    """
    sections = []

    # health_check.py (fast, always try)
    health_script = Path("ai_template_scripts/health_check.py")
    if health_script.exists():
        try:
            result = subprocess.run(
                ["python3", str(health_script)],
                capture_output=True,
                text=True,
                timeout=30,
            )
            if result.stdout.strip():
                sections.append(
                    f"### health_check.py\n```\n{result.stdout.strip()}\n```"
                )
        except Exception:
            pass

    # .flags/* alerts
    flags_dir = Path(".flags")
    if flags_dir.exists():
        flags = list(flags_dir.glob("*"))
        if flags:
            flag_names = [f.name for f in flags]
            sections.append(f"### Flags\n⚠️ Active flags: {', '.join(flag_names)}")

    # metrics/latest.json summary
    metrics_file = Path("metrics/latest.json")
    if metrics_file.exists():
        try:
            metrics = json.loads(metrics_file.read_text())
            # Extract key metrics only
            summary = []
            if "open_issues" in metrics:
                summary.append(f"open_issues: {metrics['open_issues']}")
            if "error_rate" in metrics:
                summary.append(f"error_rate: {metrics['error_rate']}")
            if "commits_24h" in metrics:
                summary.append(f"commits_24h: {metrics['commits_24h']}")
            if summary:
                sections.append(f"### Metrics\n{' | '.join(summary)}")
        except Exception:
            pass

    if not sections:
        return ""

    return "## Pre-computed Audit Data\n\n" + "\n\n".join(sections)


def get_do_audit_issues() -> list[dict]:
    """Get open issues with 'do-audit' label.

    Returns:
        List of issue dicts with number, title, labels.
    """
    try:
        result = subprocess.run(
            [
                "gh",
                "issue",
                "list",
                "--state",
                "open",
                "--label",
                "do-audit",
                "--json",
                "number,title,labels",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode != 0:
            return []
        return json.loads(result.stdout)
    except Exception:
        return []


def transition_audit_to_review(issue_num: int) -> bool:
    """Transition issue from do-audit to needs-review.

    Removes 'do-audit' label and adds 'needs-review' label.

    Args:
        issue_num: The issue number to transition.

    Returns:
        True if successful, False otherwise.
    """
    try:
        # Remove do-audit
        subprocess.run(
            ["gh", "issue", "edit", str(issue_num), "--remove-label", "do-audit"],
            capture_output=True,
            timeout=10,
        )
        # Add needs-review
        result = subprocess.run(
            ["gh", "issue", "edit", str(issue_num), "--add-label", "needs-review"],
            capture_output=True,
            timeout=10,
        )
        return result.returncode == 0
    except Exception:
        return False
