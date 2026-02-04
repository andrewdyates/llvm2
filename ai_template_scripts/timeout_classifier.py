# Copyright 2026 Dropbox, Inc.
# Author: Andrew Yates <ayates@dropbox.com>
# Licensed under the Apache License, Version 2.0

"""Timeout event classification tool for distinguishing crashes from expected timeouts.

This module classifies silence timeout events into categories:
- long_command: Long-running commands (benchmarks, builds) that exceed silence timeout
- stuck_query: SMT/verification queries with no progress
- network_stall: API calls or network operations that timed out
- idle_session: Session idle with no activity
- unknown: Unclassifiable timeouts

Usage:
    # Classify a single timeout event
    from ai_template_scripts.timeout_classifier import TimeoutClassifier
    classifier = TimeoutClassifier()
    result = classifier.classify_event(last_command, duration, exit_code)

    # Generate report from failures log (24h default)
    python3 -m ai_template_scripts.timeout_classifier --report
    python3 -m ai_template_scripts.timeout_classifier --report --hours 168  # 7 days
    python3 -m ai_template_scripts.timeout_classifier --report --json       # JSON output

See: #2310 for background on why crash/timeout distinction matters.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Literal

# Failure log location (renamed from crashes.log per #2310)
FAILURES_LOG = Path("worker_logs/failures.log")
# Legacy location for backwards compatibility
CRASHES_LOG = Path("worker_logs/crashes.log")
# Timeout context file (written by looper on silence timeout)
TIMEOUT_CONTEXT_FILE = Path("worker_logs/timeout_context.jsonl")

# Event categories
EventCategory = Literal[
    "long_command",
    "stuck_query",
    "network_stall",
    "idle_session",
    "error_exit",
    "signal_kill",
    "unknown",
]


@dataclass
class TimeoutEvent:
    """A single timeout/failure event with classification."""

    timestamp: datetime
    iteration: int
    ai_tool: str
    exit_code: int
    category: EventCategory
    last_command: str | None = None
    command_duration_sec: int | None = None
    message: str = ""
    confidence: float = 1.0  # Classification confidence (0.0-1.0)


@dataclass
class TimeoutReport:
    """Aggregated timeout classification report."""

    start_time: datetime
    end_time: datetime
    total_events: int
    by_category: dict[EventCategory, int] = field(default_factory=dict)
    top_commands: dict[str, int] = field(default_factory=dict)
    recommendations: list[str] = field(default_factory=list)

    def to_markdown(self) -> str:
        """Generate Markdown report.

        ENSURES: Returns valid Markdown string
        ENSURES: Includes category breakdown table
        """
        lines = [
            f"## Timeout Report ({self.start_time.date()} to {self.end_time.date()})",
            "",
            f"Total events: {self.total_events}",
            "",
            "### By Category",
            "",
            "| Category | Count | % |",
            "|----------|-------|---|",
        ]

        for cat, count in sorted(
            self.by_category.items(), key=lambda x: -x[1]
        ):
            pct = (count * 100 // self.total_events) if self.total_events > 0 else 0
            lines.append(f"| {cat} | {count} | {pct}% |")

        if self.top_commands:
            lines.extend(
                [
                    "",
                    "### Top Commands in Long Timeouts",
                    "",
                    "| Command | Count |",
                    "|---------|-------|",
                ]
            )
            for cmd, count in sorted(
                self.top_commands.items(), key=lambda x: -x[1]
            )[:10]:
                lines.append(f"| {cmd[:60]} | {count} |")

        if self.recommendations:
            lines.extend(["", "### Recommendations", ""])
            for rec in self.recommendations:
                lines.append(f"- {rec}")

        return "\n".join(lines)


class TimeoutClassifier:
    """Classify timeout events into categories.

    Classification rules:
    1. exit_code 125 (silence timeout) → check last_command
    2. exit_code 124 (iteration timeout) → long_command or stuck_query
    3. exit_code 137/-9 (SIGKILL) → signal_kill (OOM or forced)
    4. exit_code 1 → error_exit
    5. No last_command → idle_session
    """

    # Commands that are expected to run long
    LONG_COMMAND_PATTERNS = (
        "benchmark",
        "cargo test",
        "cargo build",
        "cargo check",
        "cargo clippy",
        "pytest",
        "npm test",
        "npm run",
        "make",
        "go test",
        "kani",
        "cbmc",
        "tlc",
        "z3",
    )

    # Commands that indicate SMT/verification queries
    QUERY_PATTERNS = (
        "z3",
        "cvc5",
        "kani",
        "cbmc",
        "tlc",
        "spin",
        "prover",
        "smt",
        "sat",
        "verify",
    )

    # Commands that indicate network operations
    NETWORK_PATTERNS = (
        "curl",
        "wget",
        "fetch",
        "gh ",
        "git push",
        "git pull",
        "git fetch",
        "npm install",
        "pip install",
        "cargo fetch",
    )

    def classify_event(
        self,
        last_command: str | None,
        command_duration_sec: int | None,
        exit_code: int,
    ) -> tuple[EventCategory, float]:
        """Classify a timeout event.

        Args:
            last_command: The last command that was running (may be None)
            command_duration_sec: How long the command ran before timeout
            exit_code: Process exit code (124=timeout, 125=silence, etc.)

        Returns:
            Tuple of (category, confidence) where confidence is 0.0-1.0

        REQUIRES: exit_code is an integer
        ENSURES: category is a valid EventCategory
        ENSURES: 0.0 <= confidence <= 1.0
        """
        # Signal kills (SIGKILL = 137 or -9)
        if exit_code == 137 or exit_code == -9:
            return "signal_kill", 0.95

        # Error exits
        if exit_code == 1:
            return "error_exit", 0.9

        # No command running - idle session
        if not last_command:
            return "idle_session", 0.8

        cmd_lower = last_command.lower()

        # Check for verification/SMT queries first (highest specificity)
        if any(p in cmd_lower for p in self.QUERY_PATTERNS):
            return "stuck_query", 0.85

        # Check for network operations
        if any(p in cmd_lower for p in self.NETWORK_PATTERNS):
            return "network_stall", 0.8

        # Check for known long commands
        if any(p in cmd_lower for p in self.LONG_COMMAND_PATTERNS):
            # Higher confidence if command ran for a long time
            confidence = 0.9 if (command_duration_sec or 0) > 300 else 0.75
            return "long_command", confidence

        # Unknown - couldn't classify
        return "unknown", 0.5

    def parse_failure_line(self, line: str) -> TimeoutEvent | None:
        """Parse a failure log line into a TimeoutEvent.

        Expected format:
        [2026-01-30 07:13:53] Iteration 82: claude killed due to silence (stale connection)
        [2026-01-30 20:30:55] Iteration 18: codex timed out

        REQUIRES: line is a string
        ENSURES: Returns TimeoutEvent or None if parse fails
        """
        # Pattern: [timestamp] Iteration N: tool message
        pattern = r"\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\] Iteration (\d+): (\w+) (.+)"
        match = re.match(pattern, line.strip())
        if not match:
            return None

        timestamp_str, iteration_str, ai_tool, message = match.groups()

        try:
            timestamp = datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S")
            iteration = int(iteration_str)
        except ValueError:
            return None

        # Determine exit code from message
        exit_code = self._exit_code_from_message(message)

        # Extract command info if present (not currently in log format)
        last_command = self._extract_command_from_message(message)

        # Classify the event
        category, confidence = self.classify_event(last_command, None, exit_code)

        return TimeoutEvent(
            timestamp=timestamp,
            iteration=iteration,
            ai_tool=ai_tool,
            exit_code=exit_code,
            category=category,
            last_command=last_command,
            message=message,
            confidence=confidence,
        )

    def _exit_code_from_message(self, message: str) -> int:
        """Infer exit code from failure message.

        ENSURES: Returns int exit code (125 for silence, 124 for timeout, etc.)
        """
        msg_lower = message.lower()
        if "silence" in msg_lower or "stale" in msg_lower:
            return 125
        if "timed out" in msg_lower:
            return 124
        if "signal" in msg_lower:
            return 137

        # Try to extract exit code from message
        code_match = re.search(r"code (\d+)", message)
        if code_match:
            return int(code_match.group(1))

        return 1  # Default to error

    def _extract_command_from_message(self, message: str) -> str | None:
        """Extract command from message if present.

        Current log format doesn't include command info, so this returns None.
        Future enhancement: Include last_command in failure log.
        """
        # TODO: Parse command from enhanced log format when available
        return None

    def generate_report(
        self,
        events: list[TimeoutEvent],
        hours: int = 24,
    ) -> TimeoutReport:
        """Generate a timeout classification report.

        Args:
            events: List of timeout events to analyze
            hours: Time window for report (default 24h)

        Returns:
            TimeoutReport with aggregated statistics

        REQUIRES: events is a list (may be empty)
        ENSURES: report.total_events >= 0
        ENSURES: sum(report.by_category.values()) == report.total_events
        """
        now = datetime.now()
        cutoff = now - timedelta(hours=hours)

        # Filter to time window
        recent = [e for e in events if e.timestamp >= cutoff]

        # Count by category
        by_category: dict[EventCategory, int] = {}
        top_commands: dict[str, int] = {}

        for event in recent:
            by_category[event.category] = by_category.get(event.category, 0) + 1
            if event.last_command and event.category == "long_command":
                # Extract base command (first word)
                base_cmd = event.last_command.split()[0] if event.last_command else ""
                if base_cmd:
                    top_commands[base_cmd] = top_commands.get(base_cmd, 0) + 1

        # Generate recommendations
        recommendations: list[str] = []
        total = len(recent)

        if total > 0:
            long_pct = by_category.get("long_command", 0) * 100 // total
            if long_pct > 30:
                recommendations.append(
                    f"Consider increasing silence_timeout: {long_pct}% of events "
                    "are long-running commands"
                )

            stuck_pct = by_category.get("stuck_query", 0) * 100 // total
            if stuck_pct > 20:
                recommendations.append(
                    f"Review verification timeouts: {stuck_pct}% of events "
                    "are stuck queries"
                )

            idle_pct = by_category.get("idle_session", 0) * 100 // total
            if idle_pct > 40:
                recommendations.append(
                    f"High idle rate ({idle_pct}%): Check for issue availability "
                    "or prompt issues"
                )

        return TimeoutReport(
            start_time=cutoff,
            end_time=now,
            total_events=len(recent),
            by_category=by_category,
            top_commands=top_commands,
            recommendations=recommendations,
        )


def _load_timeout_context() -> dict[str, dict]:
    """Load timeout context from jsonl file.

    Returns dict mapping ISO timestamp -> context dict for matching.

    ENSURES: Returns dict (may be empty)
    ENSURES: Never raises - catches all exceptions
    """
    context: dict[str, dict] = {}
    if not TIMEOUT_CONTEXT_FILE.exists():
        return context

    try:
        with open(TIMEOUT_CONTEXT_FILE) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                    timestamp = entry.get("timestamp", "")
                    if timestamp:
                        context[timestamp] = entry
                except json.JSONDecodeError:
                    continue
    except OSError:
        pass

    return context


def _match_context_to_event(
    event: TimeoutEvent, context: dict[str, dict]
) -> TimeoutEvent:
    """Match a timeout event to its context entry if available.

    Looks for context entries within 5 minutes of the event timestamp.

    REQUIRES: event is a valid TimeoutEvent
    ENSURES: Returns event with enriched fields if context found
    """
    event_ts = event.timestamp

    for ts_str, ctx in context.items():
        try:
            ctx_ts = datetime.fromisoformat(ts_str)
            # Match if within 5 minutes
            if abs((event_ts - ctx_ts).total_seconds()) < 300:
                # Found matching context - enrich event
                if ctx.get("last_command") and not event.last_command:
                    event.last_command = ctx["last_command"]
                if ctx.get("command_duration_sec") and not event.command_duration_sec:
                    event.command_duration_sec = ctx["command_duration_sec"]
                break
        except ValueError:
            continue

    return event


def load_events_from_log(log_path: Path | None = None) -> list[TimeoutEvent]:
    """Load timeout events from failure log.

    Tries failures.log first, falls back to crashes.log for compatibility.
    Enriches events with context from timeout_context.jsonl.

    ENSURES: Returns list of TimeoutEvent (may be empty)
    """
    classifier = TimeoutClassifier()
    events: list[TimeoutEvent] = []

    # Determine log path
    if log_path is None:
        if FAILURES_LOG.exists():
            log_path = FAILURES_LOG
        elif CRASHES_LOG.exists():
            log_path = CRASHES_LOG
        else:
            return events

    if not log_path.exists():
        return events

    # Load timeout context for enrichment
    context = _load_timeout_context()

    try:
        with open(log_path) as f:
            for line in f:
                event = classifier.parse_failure_line(line)
                if event:
                    # Try to enrich with context
                    if context:
                        event = _match_context_to_event(event, context)
                        # Re-classify with enriched info
                        if event.last_command:
                            category, confidence = classifier.classify_event(
                                event.last_command,
                                event.command_duration_sec,
                                event.exit_code,
                            )
                            event.category = category
                            event.confidence = confidence
                    events.append(event)
    except OSError:
        pass

    return events


def main() -> int:
    """CLI entry point for timeout classification.

    ENSURES: Returns 0 on success, 1 on error
    """
    parser = argparse.ArgumentParser(
        description="Classify timeout events and generate reports",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Generate 24h report
    python3 -m ai_template_scripts.timeout_classifier --report

    # Generate 7-day report
    python3 -m ai_template_scripts.timeout_classifier --report --hours 168

    # Output as JSON
    python3 -m ai_template_scripts.timeout_classifier --report --json
""",
    )
    parser.add_argument(
        "--report",
        action="store_true",
        help="Generate timeout classification report",
    )
    parser.add_argument(
        "--hours",
        type=int,
        default=24,
        help="Time window for report (default: 24)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output report as JSON",
    )
    parser.add_argument(
        "--log",
        type=Path,
        help="Path to failure log (default: worker_logs/failures.log)",
    )

    args = parser.parse_args()

    if args.report:
        events = load_events_from_log(args.log)
        classifier = TimeoutClassifier()
        report = classifier.generate_report(events, hours=args.hours)

        if args.json:
            output = {
                "start_time": report.start_time.isoformat(),
                "end_time": report.end_time.isoformat(),
                "total_events": report.total_events,
                "by_category": report.by_category,
                "top_commands": report.top_commands,
                "recommendations": report.recommendations,
            }
            print(json.dumps(output, indent=2))
        else:
            print(report.to_markdown())
        return 0

    # Default: show help
    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
