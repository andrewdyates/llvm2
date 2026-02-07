# Copyright 2026 Dropbox, Inc.
# Author: Andrew Yates <ayates@dropbox.com>
# Licensed under the Apache License, Version 2.0

"""
looper/context/audit_context.py - Manager audit data gathering.

Pre-computed audit data for Manager role sessions.
"""

__all__ = ["get_audit_data"]

import json
import re
import subprocess
from datetime import datetime
from pathlib import Path

from ai_template_scripts.identity import get_identity as _get_ident
from looper.config import get_project_name, load_timeout_config
from looper.constants import FLAGS_DIR, METRICS_DIR
from looper.context.helpers import has_label
from looper.context.issue_cache import IterationIssueCache
from looper.context.system_context import run_system_health_check
from looper.log import debug_swallow
from looper.result import Result
from looper.subprocess_utils import run_gh_command

METRICS_FRESHNESS_SEC = 3600  # Suppress metrics older than 1 hour (#2969)


def _get_p1_blockers() -> str:
    """Get P1 issues with blockers for blocker_audit phase.

    Returns formatted string listing P1s with Blocked: references.
    """
    try:
        result = run_gh_command(
            [
                "issue",
                "list",
                "--label",
                "P1",
                "--json",
                "number,title,body",
                "-q",
                '.[] | select(.body | contains("Blocked:")) | "#\\(.number) \\(.title)"',
            ],
            timeout=30,
        )
        if result.ok and result.value:
            blocked_issues = result.value.strip()
            if blocked_issues:
                return f"P1s with blockers:\n{blocked_issues}"
    except Exception as e:
        debug_swallow("_get_p1_blockers", e)
    return ""


def _get_outbound_dependencies() -> str:
    """Get outbound dependency issues for cross_repo phase.

    Returns formatted string listing issues this repo filed in other repos.
    """
    try:
        project = get_project_name()
        # Search for mail issues filed by this project
        result = run_gh_command(
            [
                "search",
                "issues",
                "--owner",
                _get_ident().github_org,
                f"in:body FROM: {project}",
                "--state",
                "open",
                "--json",
                "repository,number,title,updatedAt",
                "-L",
                "10",
            ],
            timeout=30,
        )
        if result.ok and result.value:
            try:
                issues = json.loads(result.value)
                if issues:
                    lines = []
                    for issue in issues[:5]:
                        repo = issue.get("repository", {}).get("name", "?")
                        num = issue.get("number", "?")
                        title = issue.get("title", "?")[:50]
                        updated = issue.get("updatedAt", "?")[:10]
                        lines.append(f"- {repo}#{num}: {title} (updated {updated})")
                    if len(issues) > 5:
                        lines.append(f"... and {len(issues) - 5} more")
                    return "Outbound mail issues:\n" + "\n".join(lines)
            except json.JSONDecodeError:
                pass
    except Exception as e:
        debug_swallow("_get_outbound_dependencies", e)
    return ""


def _extract_parent_ref(body: str | None) -> int | None:
    """Extract parent issue number from issue body text.

    Looks for patterns like "Part of #N", "child of #N", or looper-child markers.
    Returns the parent issue number or None.
    """
    if not body:
        return None
    # looper-child comment: <!-- looper-child:NNN:hash -->
    m = re.search(r"<!--\s*looper-child:(\d+):", body)
    if m:
        return int(m.group(1))
    # "Part of #N" pattern
    m = re.search(r"Part of #(\d+)", body)
    if m:
        return int(m.group(1))
    # "child of #N" pattern
    m = re.search(r"[Cc]hild of #(\d+)", body)
    if m:
        return int(m.group(1))
    return None


def _get_needs_review_evidence() -> str:
    """Fetch last 3 comments and parent refs for needs-review issues (#2604).

    Returns formatted evidence string for needs-review issues so the Manager
    can see verification signals before deciding to close or reopen.
    Only fetches data for needs-review issues (typically 1-5).

    Uses IterationIssueCache for the issue list (#2659), avoiding a separate
    API call. Per-issue comment fetches still use gh API (not in cache).
    """
    try:
        # Filter needs-review issues from shared cache (#2659)
        cache_result = IterationIssueCache.get_all()
        if not cache_result.ok:
            return ""

        all_issues = cache_result.value or []
        issues = [i for i in all_issues if has_label(i, "needs-review")]
        if not issues:
            return ""

        evidence_sections: list[str] = []

        for issue in issues[:5]:  # Cap at 5 issues
            num = issue.get("number")
            if not num:
                continue

            parts: list[str] = []

            # Include abbreviated body (first 300 chars) for context (#2638)
            body = issue.get("body", "")
            if body:
                # Strip leading metadata line (FROM: ...) if present
                body_text = body.strip()
                if body_text.startswith("**FROM:**"):
                    body_text = body_text.split("\n", 1)[-1].strip()
                body_text = body_text.replace("\n", " ")[:300]
                if body_text:
                    parts.append(f"Body: {body_text}")

            # Extract parent reference from body
            parent = _extract_parent_ref(issue.get("body", ""))
            if parent:
                parts.append(f"Parent: #{parent}")

            # Fetch last 3 comments (1 API call per issue)
            try:
                comment_result = run_gh_command(
                    [
                        "issue",
                        "view",
                        str(num),
                        "--json",
                        "comments",
                        "--jq",
                        '.comments[-3:][] | .author.login + ": " + (.body[:200] | gsub("\n"; " "))',
                    ],
                    timeout=10,
                )
                if comment_result.ok and comment_result.value:
                    comments = comment_result.value.strip()
                    if comments:
                        parts.append(comments)
            except Exception as e:
                debug_swallow(f"_get_needs_review_evidence #{num}", e)

            if parts:
                title = issue.get("title", "")[:60]
                evidence_sections.append(
                    f"#### #{num}: {title}\n" + "\n".join(parts)
                )

        if not evidence_sections:
            return ""

        return (
            "### Needs-Review Evidence\n"
            + "\n\n".join(evidence_sections)
        )
    except Exception as e:
        debug_swallow("_get_needs_review_evidence", e)
        return ""


def get_audit_data() -> Result[str]:
    """Pre-run audit scripts for Manager and return output.

    Runs crash_analysis.py, system_health_check.py, checks flags, reads metrics.
    Returns Result with formatted output for injection into Manager context.

    Contracts:
        ENSURES: Returns Result.success with formatted audit data
        ENSURES: Returns Result.success("") if no audit data available
        ENSURES: Returns Result.failure on critical error
        ENSURES: Never raises - catches all exceptions
    """
    try:
        sections: list[str] = []

        # crash_analysis.py - system health/crash analysis
        crash_script = Path("ai_template_scripts/crash_analysis.py")
        if crash_script.exists():
            try:
                result = subprocess.run(
                    ["python3", str(crash_script)],
                    capture_output=True,
                    text=True,
                    timeout=30,
                )
                if result.stdout.strip():
                    sections.append(
                        f"### crash_analysis.py\n```\n{result.stdout.strip()}\n```"
                    )
            except Exception as e:
                debug_swallow("crash_analysis.py", e)

        # timeout_classifier.py - timeout event classification (#2310)
        timeout_script = Path("ai_template_scripts/timeout_classifier.py")
        if timeout_script.exists():
            try:
                result = subprocess.run(
                    ["python3", str(timeout_script), "--report"],
                    capture_output=True,
                    text=True,
                    timeout=30,
                )
                if result.stdout.strip():
                    # Only include if there are timeout events
                    if "Total events: 0" not in result.stdout:
                        sections.append(
                            f"### timeout_classifier.py\n```\n{result.stdout.strip()}\n```"
                        )
            except Exception as e:
                debug_swallow("timeout_classifier.py", e)

        # system_health_check.py (project integration sanity)
        health_timeout = load_timeout_config().get("health_check", 120)
        system_health = run_system_health_check(timeout_sec=health_timeout)
        if system_health.skipped:
            pass
        elif system_health.ok:
            if system_health.value:
                exit_code, output = system_health.value
                status = "OK" if exit_code == 0 else f"ERROR (exit {exit_code})"
                if output:
                    sections.append(
                        f"### system_health_check.py ({status})\n```\n{output}\n```"
                    )
                else:
                    sections.append(
                        f"### system_health_check.py ({status})\n(no output)"
                    )
        else:
            error = system_health.error or "unknown error"
            sections.append(f"### system_health_check.py (ERROR)\n```\n{error}\n```")

        # P1 blocker audit (pre-compute for blocker_audit phase)
        p1_blockers = _get_p1_blockers()
        if p1_blockers:
            sections.append(f"### P1 Blockers\n{p1_blockers}")

        # Needs-review evidence: comments + parent refs (#2604)
        nr_evidence = _get_needs_review_evidence()
        if nr_evidence:
            sections.append(nr_evidence)

        # Cross-repo dependency audit (pre-compute for cross_repo phase)
        outbound_deps = _get_outbound_dependencies()
        if outbound_deps:
            sections.append(f"### Outbound Dependencies\n{outbound_deps}")

        # .flags/* alerts
        if FLAGS_DIR.exists():
            flags = list(FLAGS_DIR.glob("*"))
            if flags:
                flag_names = [f.name for f in flags]
                sections.append(f"### Flags\n⚠️ Active flags: {', '.join(flag_names)}")
                # Surface startup_warnings content if present (#2452)
                startup_warnings_file = FLAGS_DIR / "startup_warnings"
                if startup_warnings_file.exists():
                    try:
                        content = startup_warnings_file.read_text().strip()
                        if content:
                            sections.append(
                                f"### Startup Warnings\n```\n{content}\n```"
                            )
                    except OSError as e:
                        debug_swallow("startup_warnings read", e)
                # Surface ownership conflict flag contents (#3225, #3198)
                conflict_flags = sorted(
                    f for f in flags if f.name.startswith("ownership_conflict_")
                )
                if conflict_flags:
                    conflict_lines = []
                    for cf in conflict_flags:
                        try:
                            body = cf.read_text().strip()
                            if body:
                                conflict_lines.append(
                                    f"**{cf.name}:**\n{body}"
                                )
                            else:
                                conflict_lines.append(f"**{cf.name}:** (empty)")
                        except OSError as e:
                            debug_swallow(f"ownership_conflict read {cf.name}", e)
                    if conflict_lines:
                        sections.append(
                            "### Ownership Conflicts\n"
                            + "\n".join(conflict_lines)
                        )

        # metrics/latest.json summary (with freshness gate, #2969)
        metrics_file = METRICS_DIR / "latest.json"
        if metrics_file.exists():
            try:
                metrics = json.loads(metrics_file.read_text())
                # Freshness gate: suppress stale metrics (#2969)
                ts_str = metrics.get("timestamp")
                stale = False
                if isinstance(ts_str, str):
                    try:
                        ts = datetime.fromisoformat(ts_str)
                        # Timestamps from datetime.now().isoformat() are naive
                        # local time. Compare with datetime.now() (also local).
                        now = datetime.now(ts.tzinfo)
                        age_sec = (now - ts).total_seconds()
                        stale = age_sec > METRICS_FRESHNESS_SEC
                    except (ValueError, TypeError):
                        stale = True
                else:
                    stale = True
                if not stale:
                    # Extract key metrics from pulse schema
                    summary = []
                    # issues.value.open (pulse nested structure)
                    issues = metrics.get("issues", {})
                    if isinstance(issues, dict):
                        value = issues.get("value", {})
                        if isinstance(value, dict) and "open" in value:
                            summary.append(f"open_issues: {value['open']}")
                    # crashes_24h.total for error indication
                    crashes = metrics.get("crashes_24h", {})
                    if isinstance(crashes, dict) and "total" in crashes:
                        summary.append(f"crashes_24h: {crashes['total']}")
                    # git.commits_7d (closest to velocity metric)
                    git = metrics.get("git", {})
                    if isinstance(git, dict) and "commits_7d" in git:
                        summary.append(f"commits_7d: {git['commits_7d']}")
                    if summary:
                        sections.append(f"### Metrics\n{' | '.join(summary)}")
            except Exception as e:
                debug_swallow("metrics/latest.json", e)

        if not sections:
            return Result.success("")

        return Result.success("## Pre-computed Audit Data\n\n" + "\n\n".join(sections))
    except Exception as exc:
        return Result.failure(f"audit data collection failed: {exc}")
