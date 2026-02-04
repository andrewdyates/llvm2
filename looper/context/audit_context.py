# Copyright 2026 Dropbox, Inc.
# Author: Andrew Yates <ayates@dropbox.com>
# Licensed under the Apache License, Version 2.0

"""
looper/context/audit_context.py - Manager audit data gathering.

Pre-computed audit data for Manager role sessions.
"""

__all__ = ["get_audit_data"]

import json
import subprocess
from pathlib import Path

from looper.config import load_timeout_config
from looper.constants import FLAGS_DIR, METRICS_DIR
from looper.context.system_context import run_system_health_check
from looper.log import debug_swallow
from looper.result import Result


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

        # .flags/* alerts
        if FLAGS_DIR.exists():
            flags = list(FLAGS_DIR.glob("*"))
            if flags:
                flag_names = [f.name for f in flags]
                sections.append(f"### Flags\n⚠️ Active flags: {', '.join(flag_names)}")

        # metrics/latest.json summary
        metrics_file = METRICS_DIR / "latest.json"
        if metrics_file.exists():
            try:
                metrics = json.loads(metrics_file.read_text())
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
