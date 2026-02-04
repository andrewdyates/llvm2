# Copyright 2026 Dropbox, Inc.
# Author: Andrew Yates <ayates@dropbox.com>
# Licensed under the Apache License, Version 2.0

"""looper/telemetry.py - Looper self-telemetry."""

__all__ = [
    "IterationMetrics",
    "LooperStats",
    "check_consecutive_abort_alert",
    "compute_stats",
    "extract_token_usage",
    "get_health_summary",
    "record_iteration",
    "update_oversight_metrics",
]

import json
import re
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from pathlib import Path

from looper.constants import (
    EXIT_NO_ISSUES,
    EXIT_SILENCE,
    EXIT_TIMEOUT,
    FLAGS_DIR,
    MAX_METRICS_LINES,
    METRICS_DIR,
)
from looper.log import debug_swallow, log_warning
from looper.subprocess_utils import run_git_command

LOOPER_METRICS_FILE = METRICS_DIR / "looper.jsonl"
OVERSIGHT_DEFAULT_WINDOW_HOURS = 24
OVERSIGHT_DEFAULT_THRESHOLD = 2.0
OVERSIGHT_DEFAULT_ALERT_HOURS = 6
ROLE_PREFIX_RE = re.compile(r"^\[(?P<role>[A-Z])\](?P<iteration>\d+)")
KNOWN_ROLES = ("W", "M", "R", "P", "U")


def _coerce_int(value: object, default: int, min_value: int | None = None) -> int:
    """Safely convert value to int, returning default on failure or below min."""
    try:
        coerced: int = int(str(value))
    except (TypeError, ValueError):
        return default
    if min_value is not None and coerced < min_value:
        return default
    return coerced


def _coerce_float(
    value: object, default: float, min_value: float | None = None
) -> float:
    """Safely convert value to float, returning default on failure or at/below min."""
    try:
        coerced: float = float(str(value))
    except (TypeError, ValueError):
        return default
    if min_value is not None and coerced <= min_value:
        return default
    return coerced


def _coerce_bool(value: object) -> bool:
    """Safely convert value to bool; handles strings like 'true', 'yes', '1'."""
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes", "y"}:
            return True
        if normalized in {"false", "0", "no", "n", ""}:
            return False
    return False


def _empty_role_counts() -> dict[str, int]:
    """Return dict with all known roles initialized to zero count."""
    return dict.fromkeys(KNOWN_ROLES, 0)


def _load_oversight_config() -> dict[str, object]:
    """Load oversight ratio config from .looper_config.json if present."""
    config_file = Path(".looper_config.json")
    if not config_file.exists():
        return {}
    try:
        raw = json.loads(config_file.read_text())
    except (json.JSONDecodeError, OSError):
        return {}
    if not isinstance(raw, dict):
        return {}
    config: dict[str, object] = {}
    nested = raw.get("oversight_ratio")
    if isinstance(nested, dict):
        config.update(nested)
    if "oversight_ratio_window_hours" in raw:
        config.setdefault("window_hours", raw["oversight_ratio_window_hours"])
    if "oversight_ratio_threshold" in raw:
        config.setdefault("threshold", raw["oversight_ratio_threshold"])
    if "oversight_ratio_alert_hours" in raw:
        config.setdefault("alert_hours", raw["oversight_ratio_alert_hours"])
    return config


def _parse_role_commit_counts(lines: list[str]) -> dict[str, int]:
    """Parse commit subject lines and count commits by role prefix [W], [M], etc."""
    counts = _empty_role_counts()
    for line in lines:
        match = ROLE_PREFIX_RE.match(line.strip())
        if not match:
            continue
        role = match.group("role")
        if role in counts:
            counts[role] += 1
    return counts


def _get_role_commit_counts(window_hours: int) -> dict[str, int]:
    """Get commit counts per role from git log within the given time window."""
    since = datetime.now() - timedelta(hours=window_hours)
    since_str = since.strftime("%Y-%m-%d %H:%M:%S")
    result = run_git_command(["log", f"--since={since_str}", "--pretty=%s"], timeout=10)
    if not result.ok or not result.value:
        return _empty_role_counts()
    lines = [line for line in result.value.splitlines() if line.strip()]
    return _parse_role_commit_counts(lines)


def _compute_oversight_ratio(
    oversight_commits: int,
    worker_commits: int,
    threshold: float,
) -> tuple[float | None, bool, str | None]:
    """Compute oversight ratio and check if above threshold; returns (ratio, exceeded, note)."""
    if worker_commits == 0:
        if oversight_commits == 0:
            return 0.0, False, "no_commits"
        return None, True, "no_worker_commits"
    ratio = oversight_commits / worker_commits
    return ratio, ratio > threshold, None


def update_oversight_metrics() -> None:
    """Update oversight ratio metrics and warn on sustained imbalance.

    REQUIRES: Called from within a git repository
    ENSURES: Updates metrics/oversight_ratio_state.json with current state
    ENSURES: Updates metrics/latest.json with oversight metrics
    ENSURES: If ratio exceeds threshold for alert_hours, emits warning and sets alert_triggered=True
    ENSURES: Never raises (catches all exceptions and logs warnings)
    """
    try:
        config = _load_oversight_config()
        window_hours = _coerce_int(
            config.get("window_hours"), OVERSIGHT_DEFAULT_WINDOW_HOURS, min_value=1
        )
        threshold = _coerce_float(
            config.get("threshold"), OVERSIGHT_DEFAULT_THRESHOLD, min_value=0.0
        )
        alert_hours = _coerce_int(
            config.get("alert_hours"), OVERSIGHT_DEFAULT_ALERT_HOURS, min_value=1
        )
        counts = _get_role_commit_counts(window_hours)
        worker_commits = counts.get("W", 0)
        oversight_commits = counts.get("M", 0) + counts.get("R", 0)
        ratio, above_threshold, ratio_note = _compute_oversight_ratio(
            oversight_commits, worker_commits, threshold
        )
        now = time.time()
        METRICS_DIR.mkdir(exist_ok=True)
        state_file = METRICS_DIR / "oversight_ratio_state.json"
        state: dict[str, object] = {}
        if state_file.exists():
            try:
                state = json.loads(state_file.read_text())
            except (json.JSONDecodeError, OSError):
                state = {}
        since_ts = state.get("above_threshold_since")
        if above_threshold:
            if not isinstance(since_ts, (int, float)):
                since_ts = now
            state["above_threshold_since"] = since_ts
        else:
            state["above_threshold_since"] = None
            state["last_alert_time"] = None
        alert_triggered = False
        if above_threshold and isinstance(since_ts, (int, float)):
            duration = now - since_ts
            if duration >= alert_hours * 3600:
                last_alert = state.get("last_alert_time")
                if not isinstance(last_alert, (int, float)) or last_alert < since_ts:
                    alert_triggered = True
                    state["last_alert_time"] = now
        state["last_checked"] = now
        state["last_ratio"] = ratio
        state_file.write_text(json.dumps(state, indent=2))
        metrics_file = METRICS_DIR / "latest.json"
        metrics: dict[str, object] = {}
        if metrics_file.exists():
            try:
                loaded = json.loads(metrics_file.read_text())
                if isinstance(loaded, dict):
                    metrics = loaded
            except (json.JSONDecodeError, OSError):
                metrics = {}
        metrics.update(
            {
                "oversight_ratio_window_hours": window_hours,
                "oversight_ratio_threshold": threshold,
                "oversight_ratio_alert_hours": alert_hours,
                "oversight_ratio_worker_commits_window": worker_commits,
                "oversight_ratio_oversight_commits_window": oversight_commits,
                "oversight_ratio": ratio,
                "oversight_ratio_above_threshold": above_threshold,
                "oversight_ratio_note": ratio_note,
                "oversight_ratio_alert_triggered": alert_triggered,
                "oversight_ratio_updated_at": datetime.now().isoformat(),
            }
        )
        if isinstance(since_ts, (int, float)):
            metrics["oversight_ratio_above_threshold_since"] = datetime.fromtimestamp(
                since_ts
            ).isoformat()
        else:
            metrics["oversight_ratio_above_threshold_since"] = None
        metrics_file.write_text(json.dumps(metrics, indent=2))
        if alert_triggered:
            r = "inf" if ratio is None else f"{ratio:.2f}:1"
            hrs = (now - since_ts) / 3600 if isinstance(since_ts, (int, float)) else 0
            log_warning(f"Warning: oversight ratio {r} > {threshold}:1 for {hrs:.1f}h")
    except Exception as e:
        log_warning(f"Warning: oversight ratio update failed: {e}")


def extract_token_usage(log_file: Path, ai_tool: str) -> dict[str, int | float]:
    """Extract token usage from AI session log.

    Optimized to avoid reading entire file into memory (#1745):
    - Claude: reads tail of file for reverse search (result appears at end)
    - Codex: streams line-by-line accumulating usage entries

    REQUIRES: log_file is a Path (may or may not exist)
    REQUIRES: ai_tool in {"claude", "codex"} (other values use codex path)
    ENSURES: Returns dict with keys {input_tokens, output_tokens, cache_read_tokens, cache_creation_tokens}
    ENSURES: All values are non-negative integers
    ENSURES: Returns {} if file doesn't exist or on parse error
    ENSURES: Never raises (catches all exceptions)

    Returns dict with:
    - input_tokens: Total input (non-cached)
    - output_tokens: Total output
    - cache_read_tokens: Tokens from cache hits
    - cache_creation_tokens: Tokens written to cache
    """
    if not log_file.exists():
        return {}
    try:
        if ai_tool == "claude":
            # Read only tail of file for reverse search (result appears at end)
            # 64KB is sufficient for typical session result metadata
            tail_size = 64 * 1024
            file_size = log_file.stat().st_size
            with log_file.open("r") as f:
                if file_size > tail_size:
                    f.seek(file_size - tail_size)
                    f.readline()  # Skip partial line after seek
                lines = f.readlines()
            for line in reversed(lines):
                # Check for legacy type=result format
                if '"type":"result"' in line or '"type": "result"' in line:
                    data = json.loads(line)
                    # Handle null/missing/non-dict usage gracefully (#1921)
                    usage = data.get("usage") or {}
                    if not isinstance(usage, dict):
                        usage = {}
                    return {
                        "input_tokens": usage.get("input_tokens", 0),
                        "output_tokens": usage.get("output_tokens", 0),
                        "cache_read_tokens": usage.get("cache_read_input_tokens", 0),
                        "cache_creation_tokens": usage.get(
                            "cache_creation_input_tokens", 0
                        ),
                    }
                # Check for newer type=turn.completed format (#1908)
                # Uses different field names: cached_input_tokens vs cache_read_input_tokens
                if (
                    '"type":"turn.completed"' in line
                    or '"type": "turn.completed"' in line
                ):
                    data = json.loads(line)
                    # Handle null/missing/non-dict usage gracefully (#1921)
                    usage = data.get("usage") or {}
                    if not isinstance(usage, dict):
                        usage = {}
                    return {
                        "input_tokens": usage.get("input_tokens", 0),
                        "output_tokens": usage.get("output_tokens", 0),
                        # turn.completed uses cached_input_tokens (not cache_read_)
                        "cache_read_tokens": usage.get("cached_input_tokens", 0),
                        "cache_creation_tokens": 0,  # Not present in turn.completed
                    }
            # Fallback: if no result or turn.completed found, accumulate from
            # assistant messages. This handles interrupted sessions (#1908).
            # Note: This may overcount if multiple assistant chunks have usage,
            # but it's better than returning empty.
            total_input = 0
            total_output = 0
            cache_read = 0
            cache_create = 0
            for line in lines:
                if '"type":"assistant"' in line or '"type": "assistant"' in line:
                    try:
                        data = json.loads(line)
                        msg = data.get("message") or {}
                        usage = msg.get("usage") or {}
                        # Handle non-dict usage gracefully (#1921)
                        if not isinstance(usage, dict):
                            continue
                        total_input += usage.get("input_tokens", 0)
                        total_output += usage.get("output_tokens", 0)
                        cache_read += usage.get("cache_read_input_tokens", 0)
                        cache_create += usage.get("cache_creation_input_tokens", 0)
                    except (json.JSONDecodeError, AttributeError):
                        continue
            if total_input > 0 or total_output > 0:
                return {
                    "input_tokens": total_input,
                    "output_tokens": total_output,
                    "cache_read_tokens": cache_read,
                    "cache_creation_tokens": cache_create,
                }
        else:
            # Stream line-by-line accumulating usage (Codex logs multiple entries)
            total_input = 0
            total_output = 0
            cached = 0
            with log_file.open("r") as f:
                for line in f:
                    if '"usage"' in line:
                        try:
                            data = json.loads(line)
                            # Handle null/missing usage gracefully (#1921)
                            usage = data.get("usage") or {}
                            if not isinstance(usage, dict):
                                continue
                            total_input += usage.get("input_tokens", 0)
                            total_output += usage.get("output_tokens", 0)
                            cached += usage.get("cached_input_tokens", 0)
                        except (json.JSONDecodeError, AttributeError):
                            continue
            if total_input > 0 or total_output > 0:
                return {
                    "input_tokens": total_input,
                    "output_tokens": total_output,
                    "cache_read_tokens": cached,
                    "cache_creation_tokens": 0,
                }
    except Exception as e:
        debug_swallow("extract_token_usage", e)
    return {}


@dataclass
class IterationMetrics:
    """Per-iteration telemetry data."""

    project: str
    role: str
    iteration: int
    session_id: str
    start_time: float
    end_time: float
    duration_seconds: float
    ai_tool: str
    ai_model: str | None
    exit_code: int
    committed: bool
    incomplete_marker: bool
    done_marker: bool
    audit_round: int
    audit_committed: bool
    audit_rounds_run: int
    rotation_phase: str | None
    working_issues: list[int]
    worker_id: int | None = None  # Multi-worker instance ID (#1373)
    log_file: str | None = None  # Canonical log path for traceability (#1463)
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_creation_tokens: int = 0
    # Tool call metrics (#1630) - see designs/2026-02-01-tool-call-metrics-format.md
    tool_call_count: int = 0  # Total tool calls this iteration
    tool_call_types: dict[str, int] | None = None  # {"Bash": 10, "Read": 15, ...}
    tool_call_duration_ms: dict[str, int] | None = None  # {"Bash": 15000, "Read": 500}
    # Crash recovery tracking (#2073) - tracks mid-iteration crash recovery events
    recovered: bool = False  # True if iteration used recovery context from crashed session


def record_iteration(metrics: IterationMetrics) -> bool:
    """Append iteration metrics to rolling log."""
    try:
        METRICS_DIR.mkdir(exist_ok=True)
        with open(LOOPER_METRICS_FILE, "a") as f:
            f.write(json.dumps(asdict(metrics)) + "\n")
        if metrics.iteration % 100 == 0:
            _prune_metrics()
        if metrics.audit_round == 0:
            update_oversight_metrics()
        return True
    except Exception as e:
        log_warning(f"Warning: telemetry write failed: {e}")
        return False


def _prune_metrics() -> None:
    """Trim metrics file to MAX_METRICS_LINES, keeping most recent entries."""
    if not LOOPER_METRICS_FILE.exists():
        return
    try:
        lines = LOOPER_METRICS_FILE.read_text().strip().split("\n")
        if len(lines) > MAX_METRICS_LINES:
            LOOPER_METRICS_FILE.write_text("\n".join(lines[-MAX_METRICS_LINES:]) + "\n")
    except Exception as e:
        debug_swallow("prune_metrics", e)


@dataclass
class LooperStats:
    """Aggregate statistics from recent iterations."""

    window_hours: int
    total_iterations: int
    success_rate: float
    avg_duration_seconds: float
    claude_count: int
    claude_success_rate: float
    codex_count: int
    codex_success_rate: float
    audit_completion_rate: float
    avg_audit_rounds: float
    crash_count: int
    timeout_count: int
    recovery_count: int  # Iterations that resumed from mid-iteration crash (#2073)
    phase_success_rates: dict[str, float]
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    cache_hit_rate: float = 0.0
    early_abort_count: int = 0  # Iterations aborted for no issues (#1641)
    consecutive_early_aborts: int = 0  # Current streak of early aborts (#1644)
    # Tool call metrics (#1630) - see designs/2026-02-01-tool-call-metrics-format.md
    total_tool_calls: int = 0  # Total tool calls in window
    tool_distribution: dict[str, int] | None = None  # Count per tool type
    tool_duration_total_ms: dict[str, int] | None = None  # Total ms per tool type
    avg_tool_calls_per_iteration: float = 0.0  # Average tool calls per iteration


def _safe_rate(numerator: float, denominator: float) -> float:
    """Compute ratio, returning 0.0 if denominator is zero to avoid division error."""
    return numerator / denominator if denominator > 0 else 0.0


def _load_recent_entries(window_hours: int) -> list[dict[str, object]]:
    """Load metrics entries from file within the specified time window."""
    cutoff = time.time() - (window_hours * 3600)
    entries: list[dict[str, object]] = []
    for line in LOOPER_METRICS_FILE.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue
        start_time = _coerce_float(entry.get("start_time"), float("-inf"))
        if start_time >= cutoff:
            entries.append(entry)
    return entries


def _partition_iterations(
    entries: list[dict[str, object]],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Split entries into main iterations (audit_round=0) and audit iterations."""
    main_iters = [e for e in entries if _coerce_int(e.get("audit_round"), 0) == 0]
    audit_iters = [e for e in entries if _coerce_int(e.get("audit_round"), 0) > 0]
    return main_iters, audit_iters


def _count_committed(entries: list[dict[str, object]]) -> int:
    """Count entries where committed flag is true."""
    return sum(1 for e in entries if _coerce_bool(e.get("committed")))


def _normalize_tool(value: object) -> str:
    """Normalize AI tool name to lowercase for consistent comparison."""
    if not isinstance(value, str):
        return ""
    return value.strip().lower()


def _filter_by_ai_tool(
    entries: list[dict[str, object]],
    tool: str,
) -> list[dict[str, object]]:
    """Filter entries to only those using the specified AI tool (claude/codex)."""
    normalized_tool = _normalize_tool(tool)
    return [e for e in entries if _normalize_tool(e.get("ai_tool")) == normalized_tool]


def _compute_phase_success_rates(
    entries: list[dict[str, object]],
) -> dict[str, float]:
    """Compute commit success rate per rotation phase."""
    phase_stats: dict[str, list[bool]] = {}
    for entry in entries:
        raw_phase = entry.get("rotation_phase")
        phase = str(raw_phase) if raw_phase else "freeform"
        phase_stats.setdefault(phase, []).append(_coerce_bool(entry.get("committed")))
    return {
        phase: _safe_rate(sum(values), len(values))
        for phase, values in phase_stats.items()
    }


def _compute_token_stats(
    entries: list[dict[str, object]],
) -> tuple[int, int, float]:
    """Sum token usage and compute cache hit rate; returns (input, output, cache_rate)."""
    total_input = sum(
        _coerce_int(e.get("input_tokens"), 0, min_value=0) for e in entries
    )
    total_output = sum(
        _coerce_int(e.get("output_tokens"), 0, min_value=0) for e in entries
    )
    total_cache_read = sum(
        _coerce_int(e.get("cache_read_tokens"), 0, min_value=0) for e in entries
    )
    total_cache_creation = sum(
        _coerce_int(e.get("cache_creation_tokens"), 0, min_value=0) for e in entries
    )
    total_possible = total_cache_read + total_input + total_cache_creation
    cache_hit_rate = total_cache_read / total_possible if total_possible > 0 else 0
    return total_input, total_output, cache_hit_rate


def _count_recoveries(entries: list[dict[str, object]]) -> int:
    """Count iterations that used crash recovery context (#2073).

    Returns count of entries where recovered=True.
    """
    return sum(1 for e in entries if _coerce_bool(e.get("recovered")))


def _compute_exit_counts(entries: list[dict[str, object]]) -> tuple[int, int, int]:
    """Compute crash, timeout, and early abort counts.

    Returns (crashes, timeouts, early_aborts).

    Exit codes:
    - 124: timeout (expected failure mode)
    - 125: silence/stale connection (expected, not a crash)
    - 126: no issues assigned (expected early abort, #1641)
    - Other non-zero without commit: crash
    """
    crashes = 0
    timeouts = 0
    early_aborts = 0
    for entry in entries:
        exit_code = _coerce_int(entry.get("exit_code"), 0)
        if exit_code == EXIT_TIMEOUT:
            timeouts += 1
            continue
        if exit_code == EXIT_SILENCE:
            continue
        if exit_code == EXIT_NO_ISSUES:
            early_aborts += 1
            continue
        if exit_code != 0 and not _coerce_bool(entry.get("committed")):
            crashes += 1
    return crashes, timeouts, early_aborts


def _compute_consecutive_early_aborts(entries: list[dict[str, object]]) -> int:
    """Compute the current streak of consecutive early aborts (#1644).

    Sorts entries by start_time and counts how many consecutive early aborts
    (exit_code 126) are at the end of the sequence. A non-126 exit code
    breaks the streak.

    REQUIRES: entries is a list of metric dicts
    ENSURES: Returns 0 if no early aborts at end
    ENSURES: Returns N if last N iterations were early aborts
    """
    if not entries:
        return 0

    # Sort by start_time ascending
    sorted_entries = sorted(
        entries, key=lambda e: _coerce_float(e.get("start_time"), 0.0)
    )

    # Count consecutive early aborts from the end
    streak = 0
    for entry in reversed(sorted_entries):
        exit_code = _coerce_int(entry.get("exit_code"), 0)
        if exit_code == EXIT_NO_ISSUES:
            streak += 1
        else:
            break

    return streak


def _compute_audit_completion_rate(entries: list[dict[str, object]]) -> float:
    """Compute fraction of audit attempts that resulted in a commit."""
    audit_committed = 0
    audit_attempts = 0
    for entry in entries:
        rounds_run = _coerce_float(entry.get("audit_rounds_run"), 0.0, min_value=0.0)
        if rounds_run > 0:
            audit_attempts += 1
        if _coerce_bool(entry.get("audit_committed")):
            audit_committed += 1
    return _safe_rate(audit_committed, audit_attempts)


def _compute_tool_call_stats(
    entries: list[dict[str, object]],
) -> tuple[int, dict[str, int], dict[str, int]]:
    """Aggregate tool call statistics across entries (#1630).

    See: designs/2026-02-01-tool-call-metrics-format.md

    Returns:
        Tuple of (total_calls, tool_distribution, tool_duration_total_ms).
        - total_calls: Sum of tool_call_count across all entries
        - tool_distribution: Aggregated count per tool type
        - tool_duration_total_ms: Aggregated duration per tool type

    ENSURES: Uses .get() with defaults for backward compatibility
    ENSURES: Old entries without tool metrics contribute zero
    """
    total_calls = 0
    tool_distribution: dict[str, int] = {}
    tool_duration_total_ms: dict[str, int] = {}

    for entry in entries:
        # Sum total tool calls (default 0 for old entries)
        total_calls += _coerce_int(entry.get("tool_call_count"), 0, min_value=0)

        # Aggregate type counts
        types = entry.get("tool_call_types")
        if isinstance(types, dict):
            for tool, count in types.items():
                if isinstance(tool, str) and isinstance(count, (int, float)):
                    tool_distribution[tool] = tool_distribution.get(tool, 0) + int(
                        count
                    )

        # Aggregate durations
        durations = entry.get("tool_call_duration_ms")
        if isinstance(durations, dict):
            for tool, ms in durations.items():
                if isinstance(tool, str) and isinstance(ms, (int, float)):
                    tool_duration_total_ms[tool] = tool_duration_total_ms.get(
                        tool, 0
                    ) + int(ms)

    return total_calls, tool_distribution, tool_duration_total_ms


def compute_stats(window_hours: int = 24) -> LooperStats | None:
    """Compute aggregate stats from recent iterations."""
    if not LOOPER_METRICS_FILE.exists():
        return None
    try:
        entries = _load_recent_entries(window_hours)
        if not entries:
            return None
        main_iters, _ = _partition_iterations(entries)
        if not main_iters:
            return None
        total = len(main_iters)
        committed = _count_committed(main_iters)
        durations = [
            _coerce_float(e.get("duration_seconds"), 0.0, min_value=0.0)
            for e in main_iters
        ]
        claude = _filter_by_ai_tool(main_iters, "claude")
        codex = _filter_by_ai_tool(main_iters, "codex")
        claude_committed = _count_committed(claude)
        codex_committed = _count_committed(codex)
        audit_rounds_total = sum(
            _coerce_float(e.get("audit_rounds_run"), 0.0, min_value=0.0)
            for e in main_iters
        )
        audit_completion_rate = _compute_audit_completion_rate(main_iters)
        crashes, timeouts, early_aborts = _compute_exit_counts(main_iters)
        recoveries = _count_recoveries(main_iters)
        consecutive_aborts = _compute_consecutive_early_aborts(main_iters)
        phase_success = _compute_phase_success_rates(main_iters)
        total_input, total_output, cache_hit_rate = _compute_token_stats(main_iters)
        # Tool call metrics (#1630)
        total_tool_calls, tool_distribution, tool_duration_total_ms = (
            _compute_tool_call_stats(main_iters)
        )
        return LooperStats(
            window_hours=window_hours,
            total_iterations=total,
            success_rate=_safe_rate(committed, total),
            avg_duration_seconds=sum(durations) / len(durations) if durations else 0,
            claude_count=len(claude),
            claude_success_rate=_safe_rate(claude_committed, len(claude)),
            codex_count=len(codex),
            codex_success_rate=_safe_rate(codex_committed, len(codex)),
            audit_completion_rate=audit_completion_rate,
            avg_audit_rounds=_safe_rate(audit_rounds_total, total),
            crash_count=crashes,
            timeout_count=timeouts,
            recovery_count=recoveries,
            phase_success_rates=phase_success,
            total_input_tokens=total_input,
            total_output_tokens=total_output,
            cache_hit_rate=cache_hit_rate,
            early_abort_count=early_aborts,
            consecutive_early_aborts=consecutive_aborts,
            # Tool call stats (#1630)
            total_tool_calls=total_tool_calls,
            tool_distribution=tool_distribution if tool_distribution else None,
            tool_duration_total_ms=(
                tool_duration_total_ms if tool_duration_total_ms else None
            ),
            avg_tool_calls_per_iteration=_safe_rate(total_tool_calls, total),
        )
    except Exception as e:
        log_warning(f"Warning: compute_stats failed: {e}")
        return None


def get_health_summary(window_hours: int = 24) -> str:
    """One-line health summary for injection into prompt."""
    stats = compute_stats(window_hours)
    if not stats:
        return ""
    parts = [
        f"Last {window_hours}h:",
        f"{stats.total_iterations} iters",
        f"{stats.success_rate:.0%} success",
        f"{stats.avg_duration_seconds / 60:.1f}m avg",
    ]
    if stats.timeout_count > 0:
        parts.append(f"{stats.timeout_count} timeouts")
    if stats.crash_count > 0:
        parts.append(f"{stats.crash_count} crashes")
    if stats.recovery_count > 0:
        parts.append(f"{stats.recovery_count} recovered")
    if stats.codex_count > 0 and stats.claude_count > 0:
        parts.append(
            f"(claude {stats.claude_success_rate:.0%}, codex {stats.codex_success_rate:.0%})"
        )
    if stats.cache_hit_rate > 0:
        parts.append(f"| Cache: {stats.cache_hit_rate:.0%}")
    # Tool call stats (#1630) - show avg per iteration if data available
    if stats.avg_tool_calls_per_iteration > 0:
        parts.append(f"| Tools: {stats.avg_tool_calls_per_iteration:.0f}/iter")
    return " ".join(parts)


# Default threshold for consecutive early aborts before alerting (#1644)
CONSECUTIVE_ABORT_THRESHOLD = 3


def check_consecutive_abort_alert(threshold: int = CONSECUTIVE_ABORT_THRESHOLD) -> bool:
    """Check and write flag if consecutive early aborts exceed threshold (#1644).

    This function computes the current streak of early aborts and writes
    a flag file if it exceeds the threshold. Called by looper after each
    iteration to provide early warning of sustained empty-issues situations.

    REQUIRES: threshold > 0
    ENSURES: Writes .flags/consecutive_early_aborts if streak >= threshold
    ENSURES: Removes flag if streak < threshold
    ENSURES: Returns True if alert was written, False otherwise

    Returns:
        True if alert flag was written, False if cleared or no action needed.
    """
    if threshold <= 0:
        threshold = CONSECUTIVE_ABORT_THRESHOLD  # Fall back to default

    flag_file = FLAGS_DIR / "consecutive_early_aborts"

    stats = compute_stats(window_hours=6)  # Look at recent window
    if not stats:
        # No stats - clear flag if present
        if flag_file.exists():
            try:
                flag_file.unlink()
            except OSError as e:
                debug_swallow("unlink_abort_flag_no_stats", e)
        return False

    streak = stats.consecutive_early_aborts

    if streak >= threshold:
        # Write alert flag
        FLAGS_DIR.mkdir(exist_ok=True)
        content = (
            f"{streak} consecutive early aborts (no issues assigned)\n"
            f"Threshold: {threshold}\n"
            f"Total early aborts in last 6h: {stats.early_abort_count}\n"
            f"\nPossible causes:\n"
            f"- No issues exist in the repo\n"
            f"- All issues are blocked, deferred, or tracking\n"
            f"- GitHub API issues (check gh_rate_limit stats)\n"
            f"\nTo resolve:\n"
            f"- File new issues: gh issue create\n"
            f"- Unblock existing: gh issue edit N --remove-label blocked\n"
            f"- Check API: cat ~/.ait_gh_cache/rate_state.json\n"
        )
        try:
            flag_file.write_text(content)
            log_warning(
                f"⚠️  Alert: {streak} consecutive early aborts - see .flags/consecutive_early_aborts"
            )
            return True
        except OSError as e:
            log_warning(f"Failed to write consecutive_early_aborts flag: {e}")
    elif flag_file.exists():
        # Clear flag - streak broken
        try:
            flag_file.unlink()
        except OSError as e:
            debug_swallow("unlink_abort_flag_clear", e)

    return False
