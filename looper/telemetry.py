# Copyright 2026 Dropbox, Inc.
# Author: Andrew Yates
# Licensed under the Apache License, Version 2.0

# looper/telemetry.py
"""
Looper self-telemetry - iteration metrics collection and analysis.

Author: Andrew Yates <ayates@dropbox.com>

Records per-iteration metrics to metrics/looper.jsonl for:
- Duration tracking
- Success/failure rates
- AI tool comparison
- Audit round analysis
"""

import json
import re
import subprocess
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from pathlib import Path

METRICS_DIR = Path("metrics")
LOOPER_METRICS_FILE = METRICS_DIR / "looper.jsonl"
MAX_METRICS_LINES = 5000  # ~2 weeks at 20 iterations/day
OVERSIGHT_DEFAULT_WINDOW_HOURS = 24
OVERSIGHT_DEFAULT_THRESHOLD = 2.0
OVERSIGHT_DEFAULT_ALERT_HOURS = 6
ROLE_PREFIX_RE = re.compile(r"^\[(?P<role>[A-Z])\](?P<iteration>\d+)")
KNOWN_ROLES = ("W", "M", "R", "P", "U")


def _coerce_int(value: object, default: int, min_value: int | None = None) -> int:
    try:
        coerced = int(value)
    except (TypeError, ValueError):
        return default
    if min_value is not None and coerced < min_value:
        return default
    return coerced


def _coerce_float(
    value: object, default: float, min_value: float | None = None
) -> float:
    try:
        coerced = float(value)
    except (TypeError, ValueError):
        return default
    if min_value is not None and coerced <= min_value:
        return default
    return coerced


def _empty_role_counts() -> dict[str, int]:
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

    # Backwards-compatible flat keys
    if "oversight_ratio_window_hours" in raw:
        config.setdefault("window_hours", raw["oversight_ratio_window_hours"])
    if "oversight_ratio_threshold" in raw:
        config.setdefault("threshold", raw["oversight_ratio_threshold"])
    if "oversight_ratio_alert_hours" in raw:
        config.setdefault("alert_hours", raw["oversight_ratio_alert_hours"])

    return config


def _parse_role_commit_counts(lines: list[str]) -> dict[str, int]:
    """Parse commit subjects and count primary role prefixes."""
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
    """Return counts of role-prefixed commits in a time window."""
    since = datetime.now() - timedelta(hours=window_hours)
    since_str = since.strftime("%Y-%m-%d %H:%M:%S")
    try:
        result = subprocess.run(
            ["git", "log", f"--since={since_str}", "--pretty=%s"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        if result.returncode != 0:
            return _empty_role_counts()
        lines = [line for line in result.stdout.splitlines() if line.strip()]
        return _parse_role_commit_counts(lines)
    except Exception:
        return _empty_role_counts()


def _compute_oversight_ratio(
    oversight_commits: int,
    worker_commits: int,
    threshold: float,
) -> tuple[float | None, bool, str | None]:
    if worker_commits == 0:
        if oversight_commits == 0:
            return 0.0, False, "no_commits"
        return None, True, "no_worker_commits"

    ratio = oversight_commits / worker_commits
    return ratio, ratio > threshold, None


def update_oversight_metrics() -> None:
    """Update oversight ratio metrics and warn on sustained imbalance."""
    try:
        config = _load_oversight_config()
        window_hours = _coerce_int(
            config.get("window_hours"),
            OVERSIGHT_DEFAULT_WINDOW_HOURS,
            min_value=1,
        )
        threshold = _coerce_float(
            config.get("threshold"),
            OVERSIGHT_DEFAULT_THRESHOLD,
            min_value=0.0,
        )
        alert_hours = _coerce_int(
            config.get("alert_hours"),
            OVERSIGHT_DEFAULT_ALERT_HOURS,
            min_value=1,
        )

        counts = _get_role_commit_counts(window_hours)
        worker_commits = counts.get("W", 0)
        oversight_commits = counts.get("M", 0) + counts.get("R", 0)
        ratio, above_threshold, ratio_note = _compute_oversight_ratio(
            oversight_commits,
            worker_commits,
            threshold,
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
            ratio_display = "inf" if ratio is None else f"{ratio:.2f}:1"
            duration_hours = (
                (now - since_ts) / 3600 if isinstance(since_ts, (int, float)) else 0
            )
            print(
                "Warning: oversight ratio "
                f"{ratio_display} exceeds {threshold}:1 for {duration_hours:.1f}h"
            )
    except Exception as e:
        print(f"Warning: oversight ratio update failed: {e}")


# Per-million-token pricing (2026-01 rates) - Part of #488
# Model: (input, output, cache_read, cache_write)
CLAUDE_PRICING: dict[str, tuple[float, float, float, float]] = {
    "claude-opus-4-5": (5.0, 25.0, 0.5, 6.25),
    "claude-sonnet-4-5": (3.0, 15.0, 0.3, 3.75),
    "claude-sonnet-4": (3.0, 15.0, 0.3, 3.75),
    "claude-haiku-4-5": (1.0, 5.0, 0.1, 1.25),
    "default": (3.0, 15.0, 0.3, 3.75),  # Assume Sonnet
}

CODEX_PRICING: dict[str, tuple[float, float, float, float]] = {
    "o4-mini": (1.1, 4.4, 0.275, 0.0),
    "gpt-4.2": (10.0, 30.0, 2.5, 0.0),
    "default": (1.1, 4.4, 0.275, 0.0),  # Assume o4-mini
}


def extract_token_usage(log_file: Path, ai_tool: str) -> dict[str, int | float]:
    """Extract token usage from AI session log.

    Claude logs have a final 'result' message with usage:
    {"type": "result", "usage": {...}, "total_cost_usd": ...}

    Codex logs have usage in response messages.

    Returns dict with:
    - input_tokens: Total input (non-cached)
    - output_tokens: Total output
    - cache_read_tokens: Tokens from cache hits
    - cache_creation_tokens: Tokens written to cache
    - total_cost_usd: Pre-calculated cost from CLI (Claude only)
    """
    if not log_file.exists():
        return {}

    try:
        content = log_file.read_text()
        lines = content.strip().split("\n")

        if ai_tool == "claude":
            # Look for result message with usage (NOT stats - stats is wrong key)
            for line in reversed(lines):
                if '"type":"result"' in line or '"type": "result"' in line:
                    data = json.loads(line)
                    usage = data.get("usage", {})
                    return {
                        "input_tokens": usage.get("input_tokens", 0),
                        "output_tokens": usage.get("output_tokens", 0),
                        "cache_read_tokens": usage.get("cache_read_input_tokens", 0),
                        "cache_creation_tokens": usage.get(
                            "cache_creation_input_tokens", 0
                        ),
                        "total_cost_usd": data.get("total_cost_usd", 0.0),
                    }
        else:
            # Codex: aggregate from response messages
            total_input = 0
            total_output = 0
            cached = 0
            for line in lines:
                if '"usage"' in line:
                    data = json.loads(line)
                    usage = data.get("usage", {})
                    total_input += usage.get("input_tokens", 0)
                    total_output += usage.get("output_tokens", 0)
                    cached += usage.get("cached_input_tokens", 0)
            if total_input > 0 or total_output > 0:
                return {
                    "input_tokens": total_input,
                    "output_tokens": total_output,
                    "cache_read_tokens": cached,
                    "cache_creation_tokens": 0,  # Codex doesn't report this
                    "total_cost_usd": 0.0,  # Codex doesn't provide, calculate manually
                }
    except Exception:
        pass

    return {}


def estimate_cost(
    input_tokens: int,
    output_tokens: int,
    cache_read_tokens: int,
    cache_creation_tokens: int,
    model: str,
    ai_tool: str,
) -> float:
    """Estimate cost in USD based on token counts and model.

    For Claude, the CLI provides total_cost_usd directly - prefer that.
    This function is for Codex or when CLI cost is unavailable.
    """
    pricing = CLAUDE_PRICING if ai_tool == "claude" else CODEX_PRICING

    # Normalize model name for lookup
    model_key = (
        model.lower().replace("-", "_").replace(".", "_") if model else "default"
    )

    # Find matching pricing
    p_input, p_output, p_cache_read, p_cache_write = pricing["default"]
    for key in pricing:
        if key == "default":
            continue
        # Normalize the pricing key the same way as model_key
        normalized_key = key.lower().replace("-", "_").replace(".", "_")
        if normalized_key in model_key or model_key.startswith(normalized_key):
            p_input, p_output, p_cache_read, p_cache_write = pricing[key]
            break

    # Calculate cost (per million tokens)
    cost = (
        (input_tokens * p_input / 1_000_000)
        + (output_tokens * p_output / 1_000_000)
        + (cache_read_tokens * p_cache_read / 1_000_000)
        + (cache_creation_tokens * p_cache_write / 1_000_000)
    )
    return round(cost, 6)  # 6 decimal places for precision


@dataclass
class IterationMetrics:
    """Per-iteration telemetry data."""

    # Identity
    project: str
    role: str
    iteration: int  # Loop iteration (not git iteration)
    session_id: str

    # Timing
    start_time: float  # Unix timestamp
    end_time: float
    duration_seconds: float

    # AI Tool
    ai_tool: str  # "claude" or "codex"
    ai_model: str | None  # Model used (claude_model or codex_model)

    # Outcome
    exit_code: int
    committed: bool  # Did this iteration produce a commit?
    incomplete_marker: bool  # Was commit marked [INCOMPLETE]?
    done_marker: bool  # Was commit marked [DONE]?

    # Audit
    audit_round: int  # 0 for main iteration, 1-N for audits
    audit_committed: bool  # Did any audit round commit?
    audit_rounds_run: int  # How many audit rounds ran?

    # Context
    rotation_phase: str | None  # Current rotation phase
    working_issues: list[int]  # Issues referenced in commit

    # Token usage (from log file stats) - Part of #488
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_creation_tokens: int = 0

    # Cost estimate - Part of #488
    estimated_cost_usd: float = 0.0


def record_iteration(metrics: IterationMetrics) -> bool:
    """Append iteration metrics to rolling log.

    Returns True if written successfully, False on error.
    Non-blocking - failures are logged but don't crash.
    """
    try:
        METRICS_DIR.mkdir(exist_ok=True)

        # Append to JSONL
        with open(LOOPER_METRICS_FILE, "a") as f:
            f.write(json.dumps(asdict(metrics)) + "\n")

        # Prune if needed (check periodically, not every write)
        if metrics.iteration % 100 == 0:
            _prune_metrics()

        if metrics.audit_round == 0:
            update_oversight_metrics()

        return True
    except Exception as e:
        print(f"Warning: telemetry write failed: {e}")
        return False


def _prune_metrics() -> None:
    """Remove old entries to keep file size bounded."""
    if not LOOPER_METRICS_FILE.exists():
        return

    try:
        lines = LOOPER_METRICS_FILE.read_text().strip().split("\n")
        if len(lines) > MAX_METRICS_LINES:
            # Keep most recent entries
            LOOPER_METRICS_FILE.write_text("\n".join(lines[-MAX_METRICS_LINES:]) + "\n")
    except Exception:
        pass  # Pruning failure is non-critical


@dataclass
class LooperStats:
    """Aggregate statistics from recent iterations."""

    window_hours: int
    total_iterations: int
    success_rate: float  # committed / total
    avg_duration_seconds: float

    # By tool
    claude_count: int
    claude_success_rate: float
    codex_count: int
    codex_success_rate: float

    # Audits
    audit_completion_rate: float  # Audits that committed / audits run
    avg_audit_rounds: float

    # Issues
    crash_count: int  # exit_code != 0 and not committed
    timeout_count: int  # exit_code == 124

    # Phases
    phase_success_rates: dict[str, float]  # phase -> success rate

    # Cost tracking - Part of #488
    total_cost_usd: float = 0.0
    avg_cost_per_iteration_usd: float = 0.0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    cache_hit_rate: float = 0.0  # cache_read / (cache_read + input + cache_creation)


def compute_stats(window_hours: int = 24) -> LooperStats | None:
    """Compute aggregate stats from recent iterations.

    Returns None if no data available.
    """
    if not LOOPER_METRICS_FILE.exists():
        return None

    try:
        cutoff = time.time() - (window_hours * 3600)

        entries = []
        for line in LOOPER_METRICS_FILE.read_text().strip().split("\n"):
            if not line.strip():
                continue
            try:
                entry = json.loads(line)
                if entry.get("start_time", 0) >= cutoff:
                    entries.append(entry)
            except json.JSONDecodeError:
                continue

        if not entries:
            return None

        # Filter to main iterations (audit_round == 0)
        main_iters = [e for e in entries if e.get("audit_round", 0) == 0]
        audit_iters = [e for e in entries if e.get("audit_round", 0) > 0]

        if not main_iters:
            return None

        # Basic stats
        total = len(main_iters)
        committed = sum(1 for e in main_iters if e.get("committed"))
        durations = [e.get("duration_seconds", 0) for e in main_iters]

        # By tool
        claude = [e for e in main_iters if e.get("ai_tool") == "claude"]
        codex = [e for e in main_iters if e.get("ai_tool") == "codex"]

        claude_committed = sum(1 for e in claude if e.get("committed"))
        codex_committed = sum(1 for e in codex if e.get("committed"))

        # Audits
        audit_committed = sum(1 for e in audit_iters if e.get("committed"))
        audit_rounds_total = sum(e.get("audit_rounds_run", 0) for e in main_iters)

        # Crashes/timeouts
        crashes = sum(
            1
            for e in main_iters
            if e.get("exit_code", 0) != 0 and not e.get("committed")
        )
        timeouts = sum(1 for e in main_iters if e.get("exit_code") == 124)

        # By phase
        phase_stats: dict[str, list[bool]] = {}
        for e in main_iters:
            phase = e.get("rotation_phase") or "freeform"
            if phase not in phase_stats:
                phase_stats[phase] = []
            phase_stats[phase].append(e.get("committed", False))

        phase_success = {
            phase: sum(results) / len(results) if results else 0
            for phase, results in phase_stats.items()
        }

        # Cost tracking - aggregate from all entries (main + audit)
        total_cost = sum(e.get("estimated_cost_usd", 0.0) for e in entries)
        total_input = sum(e.get("input_tokens", 0) for e in entries)
        total_output = sum(e.get("output_tokens", 0) for e in entries)
        total_cache_read = sum(e.get("cache_read_tokens", 0) for e in entries)
        total_cache_creation = sum(e.get("cache_creation_tokens", 0) for e in entries)

        # Cache hit rate: cache_read / (cache_read + input + cache_creation)
        total_possible_cache = total_cache_read + total_input + total_cache_creation
        cache_hit_rate = (
            total_cache_read / total_possible_cache if total_possible_cache > 0 else 0
        )

        return LooperStats(
            window_hours=window_hours,
            total_iterations=total,
            success_rate=committed / total if total > 0 else 0,
            avg_duration_seconds=sum(durations) / len(durations) if durations else 0,
            claude_count=len(claude),
            claude_success_rate=claude_committed / len(claude) if claude else 0,
            codex_count=len(codex),
            codex_success_rate=codex_committed / len(codex) if codex else 0,
            audit_completion_rate=(
                audit_committed / len(audit_iters) if audit_iters else 0
            ),
            avg_audit_rounds=audit_rounds_total / total if total > 0 else 0,
            crash_count=crashes,
            timeout_count=timeouts,
            phase_success_rates=phase_success,
            total_cost_usd=total_cost,
            avg_cost_per_iteration_usd=total_cost / total if total > 0 else 0,
            total_input_tokens=total_input,
            total_output_tokens=total_output,
            cache_hit_rate=cache_hit_rate,
        )
    except Exception as e:
        print(f"Warning: compute_stats failed: {e}")
        return None


def get_health_summary(window_hours: int = 24) -> str:
    """One-line health summary for injection into prompt.

    Returns empty string if no data available.
    """
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

    if stats.codex_count > 0 and stats.claude_count > 0:
        # Show tool comparison only if both used
        parts.append(
            f"(claude {stats.claude_success_rate:.0%}, codex {stats.codex_success_rate:.0%})"
        )

    # Cost tracking - Part of #488
    if stats.total_cost_usd > 0:
        parts.append(f"| Cost: ${stats.total_cost_usd:.2f}")
        if stats.cache_hit_rate > 0:
            parts.append(f"({stats.cache_hit_rate:.0%} cached)")

    return " ".join(parts)
