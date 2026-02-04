# Copyright 2026 Dropbox, Inc.
# Author: Andrew Yates <ayates@dropbox.com>
# Licensed under the Apache License, Version 2.0

"""
Tool call logging for fine-grained crash recovery.

See: designs/2026-02-01-tool-call-checkpointing.md

This module handles recording of individual tool calls to enable recovery
from the last completed tool call rather than restarting entire iterations.
"""

import hashlib
import json
import os
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from looper.log import log_info, log_warning

__all__ = [
    "ToolCallRecord",
    "ToolCallLogConfig",
    "ToolCallLog",
    "TOOL_CALL_LOG_VERSION",
]

TOOL_CALL_LOG_VERSION = 1


@dataclass
class ToolCallRecord:
    """Record of a single tool call for fine-grained recovery.

    See: designs/2026-02-01-tool-call-checkpointing.md

    ENSURES: Immutable record of tool call state
    ENSURES: Supports started/completed/failed status
    """

    seq: int  # Monotonically increasing sequence number per session
    tool_name: str  # Tool identifier (e.g., "Bash", "Read", "Edit")
    tool_input_hash: str  # SHA-256 hash of normalized tool input
    tool_input_preview: str  # Short preview of input (truncated)
    started_at: str  # ISO timestamp when tool call started
    status: Literal["started", "completed", "failed"]
    source: Literal["claude", "codex", "dasher"]
    completed_at: str | None = None  # ISO timestamp when completed
    result_truncated: str | None = None  # Truncated output (size-limited)
    result_path: str | None = None  # Optional file path for full output
    error_message: str | None = None  # Error message if failed

    def to_dict(self) -> dict:
        """Convert to dict for JSON serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "ToolCallRecord":
        """Create from dict (JSON deserialization).

        REQUIRES: data contains all required fields
        ENSURES: status and source are validated against allowed values
        RAISES: ValueError if status or source is not a valid literal
        RAISES: KeyError if required field is missing
        """
        status = data["status"]
        if status not in ("started", "completed", "failed"):
            raise ValueError(f"Invalid status: {status}")
        source = data["source"]
        if source not in ("claude", "codex", "dasher"):
            raise ValueError(f"Invalid source: {source}")
        return cls(
            seq=data["seq"],
            tool_name=data["tool_name"],
            tool_input_hash=data["tool_input_hash"],
            tool_input_preview=data["tool_input_preview"],
            started_at=data["started_at"],
            status=status,
            source=source,
            completed_at=data.get("completed_at"),
            result_truncated=data.get("result_truncated"),
            result_path=data.get("result_path"),
            error_message=data.get("error_message"),
        )


@dataclass
class ToolCallLogConfig:
    """Configuration for tool call log retention.

    See: designs/2026-02-01-tool-call-log-phase2.md

    Controls log size limits and pruning behavior to prevent OOM on long sessions.

    ENSURES: Limits are applied during append, not lazily
    ENSURES: Pruning preserves in-progress records (started but not completed)
    """

    max_records: int = 1000  # Max records before pruning
    max_size_bytes: int = 1_000_000  # 1MB max file size
    prune_keep_ratio: float = 0.8  # Keep 80% when pruning
    result_truncate_chars: int = 2000  # Max chars in result_truncated field


def _hash_tool_input(tool_input: str | dict) -> str:
    """Compute SHA-256 hash of normalized tool input.

    REQUIRES: tool_input is a string or dict
    ENSURES: Returns hex-encoded SHA-256 hash
    """
    if isinstance(tool_input, dict):
        # Sort keys for deterministic hashing
        normalized = json.dumps(tool_input, sort_keys=True, separators=(",", ":"))
    else:
        normalized = str(tool_input)
    return hashlib.sha256(normalized.encode()).hexdigest()[:16]  # Truncate for brevity


def _truncate_preview(text: str, max_len: int = 100) -> str:
    """Truncate text for preview display.

    ENSURES: Result length <= max_len + 3 (for "...")
    """
    if len(text) <= max_len:
        return text
    return text[:max_len] + "..."


class ToolCallLog:
    """Manages tool call log for fine-grained crash recovery.

    See: designs/2026-02-01-tool-call-checkpointing.md

    Appends tool call records atomically to a JSONL file, enabling
    recovery from the last completed tool call instead of restarting
    entire iterations.

    REQUIRES: log_file is a valid Path
    ENSURES: Atomic appends via flush+fsync after each write
    NOTE: Not thread-safe - designed for single-writer use per iteration
    """

    def __init__(
        self,
        log_file: Path,
        source: Literal["claude", "codex", "dasher"],
        result_truncate_chars: int = 2000,
        config: ToolCallLogConfig | None = None,
    ) -> None:
        """Initialize tool call log.

        Args:
            log_file: Path to JSONL file (e.g., .looper_tool_calls_worker.jsonl)
            source: AI tool being used ("claude", "codex", or "dasher")
            result_truncate_chars: Max chars to store in result_truncated field
                (deprecated - use config.result_truncate_chars instead)
            config: Optional retention configuration. If not provided, uses defaults.
        """
        self.log_file = log_file
        self.source = source
        # Use config if provided, otherwise create with result_truncate_chars override
        if config is not None:
            self.config = config
        else:
            self.config = ToolCallLogConfig(result_truncate_chars=result_truncate_chars)
        self.result_truncate_chars = self.config.result_truncate_chars
        self._seq = 0
        self._pending_records: dict[str, ToolCallRecord] = {}  # tool_id -> record

    def start(
        self,
        tool_name: str,
        tool_input: str | dict,
        tool_id: str | None = None,
    ) -> ToolCallRecord:
        """Record a tool call start.

        REQUIRES: tool_name is non-empty
        ENSURES: Returns ToolCallRecord with status="started"
        ENSURES: Record is appended to log file
        ENSURES: Record is stored in pending records for completion
        """
        self._seq += 1
        input_hash = _hash_tool_input(tool_input)
        if isinstance(tool_input, dict):
            preview = _truncate_preview(json.dumps(tool_input))
        else:
            preview = _truncate_preview(str(tool_input))

        record = ToolCallRecord(
            seq=self._seq,
            tool_name=tool_name,
            tool_input_hash=input_hash,
            tool_input_preview=preview,
            started_at=datetime.now(UTC).isoformat(),
            status="started",
            source=self.source,
        )

        # Store for later completion
        record_key = tool_id or f"{tool_name}:{self._seq}"
        self._pending_records[record_key] = record

        self._append_record(record)
        return record

    def complete(
        self,
        tool_id: str | None = None,
        tool_name: str | None = None,
        result: str | None = None,
        result_path: str | None = None,
    ) -> ToolCallRecord | None:
        """Record a tool call completion.

        REQUIRES: Either tool_id or tool_name must be provided
        ENSURES: Returns updated ToolCallRecord with status="completed"
        ENSURES: Record is appended to log file
        ENSURES: Returns None if no matching pending record found
        """
        # Find the pending record
        record_key = tool_id
        record = None

        if record_key and record_key in self._pending_records:
            record = self._pending_records.pop(record_key)
        elif tool_name:
            # Find most recent pending record with matching tool name
            for key, rec in reversed(list(self._pending_records.items())):
                if rec.tool_name == tool_name:
                    record = self._pending_records.pop(key)
                    break

        if not record:
            return None

        # Update the record
        result_truncated = None
        if result:
            if len(result) > self.result_truncate_chars:
                result_truncated = (
                    result[: self.result_truncate_chars]
                    + f"...[truncated at {self.result_truncate_chars} chars]"
                )
            else:
                result_truncated = result

        completed_record = ToolCallRecord(
            seq=record.seq,
            tool_name=record.tool_name,
            tool_input_hash=record.tool_input_hash,
            tool_input_preview=record.tool_input_preview,
            started_at=record.started_at,
            status="completed",
            source=record.source,
            completed_at=datetime.now(UTC).isoformat(),
            result_truncated=result_truncated,
            result_path=result_path,
        )

        self._append_record(completed_record)
        return completed_record

    def fail(
        self,
        tool_id: str | None = None,
        tool_name: str | None = None,
        error_message: str | None = None,
    ) -> ToolCallRecord | None:
        """Record a tool call failure.

        REQUIRES: Either tool_id or tool_name must be provided
        ENSURES: Returns updated ToolCallRecord with status="failed"
        ENSURES: Record is appended to log file
        ENSURES: Returns None if no matching pending record found
        """
        # Find the pending record (same logic as complete)
        record_key = tool_id
        record = None

        if record_key and record_key in self._pending_records:
            record = self._pending_records.pop(record_key)
        elif tool_name:
            for key, rec in reversed(list(self._pending_records.items())):
                if rec.tool_name == tool_name:
                    record = self._pending_records.pop(key)
                    break

        if not record:
            return None

        failed_record = ToolCallRecord(
            seq=record.seq,
            tool_name=record.tool_name,
            tool_input_hash=record.tool_input_hash,
            tool_input_preview=record.tool_input_preview,
            started_at=record.started_at,
            status="failed",
            source=record.source,
            completed_at=datetime.now(UTC).isoformat(),
            error_message=error_message,
        )

        self._append_record(failed_record)
        return failed_record

    def _append_record(self, record: ToolCallRecord) -> bool:
        """Append a record to the log file atomically.

        ENSURES: Record is written as single JSON line
        ENSURES: File is flushed after write
        ENSURES: Pruning is triggered if limits exceeded
        ENSURES: Returns True on success, False on error
        """
        try:
            self.log_file.parent.mkdir(exist_ok=True)
            with open(self.log_file, "a") as f:
                f.write(json.dumps(record.to_dict()) + "\n")
                f.flush()
                os.fsync(f.fileno())  # Ensure data is persisted
            # Check if pruning is needed after successful write
            self._maybe_prune()
            return True
        except OSError as e:
            log_warning(f"Warning: tool call log append failed: {e}")
            return False

    def _maybe_prune(self) -> None:
        """Prune log if limits exceeded.

        See: designs/2026-02-01-tool-call-log-phase2.md

        Pruning strategy:
        1. Check file size and record count against config limits
        2. If exceeded, keep most recent (max_records * prune_keep_ratio) records
        3. Never prune "started" records from current session (in-progress)
        4. Rewrite file atomically (temp file + rename)

        ENSURES: File stays within configured limits
        ENSURES: In-progress records (started but not completed) are preserved
        ENSURES: Pruning is logged to stderr
        """
        if not self.log_file.exists():
            return

        # Check file size
        try:
            file_size = self.log_file.stat().st_size
        except OSError:
            return

        records = self.load_records()
        if not records:
            return

        size_exceeded = file_size > self.config.max_size_bytes
        record_count_exceeded = len(records) > self.config.max_records
        if not size_exceeded and not record_count_exceeded:
            return

        # Determine how many to keep
        keep_count = int(self.config.max_records * self.config.prune_keep_ratio)
        if keep_count < 1:
            keep_count = 1

        # Separate in-progress records (started but not completed in current session)
        # These are identified by having a pending record in memory
        pending_seqs = {rec.seq for rec in self._pending_records.values()}
        in_progress = [r for r in records if r.seq in pending_seqs]
        completed = [r for r in records if r.seq not in pending_seqs]

        if not completed:
            return

        if size_exceeded and len(completed) <= keep_count:
            keep_count = max(1, int(len(completed) * self.config.prune_keep_ratio))

        # Keep most recent completed + all in-progress
        if len(completed) > keep_count:
            # Prune oldest completed records
            records_to_keep = completed[-keep_count:] + in_progress
            pruned_count = len(records) - len(records_to_keep)

            # Rewrite atomically
            try:
                temp_path = self.log_file.with_suffix(".jsonl.tmp")
                with open(temp_path, "w") as f:
                    for rec in records_to_keep:
                        f.write(json.dumps(rec.to_dict()) + "\n")
                    f.flush()
                    os.fsync(f.fileno())
                temp_path.rename(self.log_file)
                log_info(
                    f"tool_call_log: pruned {pruned_count} records "
                    f"(kept {len(records_to_keep)})"
                )
            except OSError as e:
                log_warning(f"Warning: tool call log prune failed: {e}")
                # Clean up temp file if it exists
                try:
                    temp_path.unlink(missing_ok=True)
                except OSError:
                    pass

    def get_finished_count(self) -> int:
        """Get the count of finished (completed + failed) tool calls in this session.

        ENSURES: Returns non-negative integer
        """
        return self._seq - len(self._pending_records)

    def get_last_seq(self) -> int:
        """Get the last sequence number used.

        ENSURES: Returns non-negative integer
        """
        return self._seq

    def load_records(self) -> list[ToolCallRecord]:
        """Load all records from the log file.

        ENSURES: Returns list of ToolCallRecord
        ENSURES: Returns empty list if file doesn't exist or is empty
        ENSURES: Returns partial list if some records fail to parse (best-effort)
        """
        if not self.log_file.exists():
            return []

        records = []
        try:
            with open(self.log_file) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            data = json.loads(line)
                            records.append(ToolCallRecord.from_dict(data))
                        except (json.JSONDecodeError, KeyError, ValueError) as e:
                            # Skip malformed records but continue parsing
                            log_warning(
                                f"Warning: skipping malformed tool call record: {e}"
                            )
                            continue
        except OSError as e:
            log_warning(f"Warning: failed to load tool call log: {e}")
            # Return whatever we parsed before the error
            return records

        return records

    def clear(self) -> bool:
        """Clear the log file.

        ENSURES: Log file deleted if exists
        ENSURES: Returns True on success or if file didn't exist
        """
        try:
            if self.log_file.exists():
                self.log_file.unlink()
            self._seq = 0
            self._pending_records.clear()
            return True
        except OSError as e:
            log_warning(f"Warning: tool call log clear failed: {e}")
            return False

    def get_aggregate_stats(self) -> tuple[int, dict[str, int], dict[str, int]]:
        """Return aggregate tool call statistics for telemetry.

        See: designs/2026-02-01-tool-call-metrics-format.md

        Returns:
            Tuple of (total_count, type_counts, duration_ms_per_type).
            - total_count: Total number of tool calls (started, completed, or failed)
            - type_counts: Count per tool type (e.g., {"Bash": 10, "Read": 15})
            - duration_ms_per_type: Duration in milliseconds per tool type

        ENSURES: Returns aggregates from all tool calls in the log
        ENSURES: Duration calculated from started_at to completed_at
        ENSURES: Calls without completed_at contribute to count but not duration
        """
        records = self.load_records()
        if not records:
            return 0, {}, {}

        # Track seen sequence numbers to avoid double-counting
        # (each tool call has started + completed/failed records)
        seen_seqs: set[int] = set()
        type_counts: dict[str, int] = {}
        duration_ms: dict[str, int] = {}

        for rec in records:
            tool_name = rec.tool_name
            seq = rec.seq

            # Count each unique sequence number once
            if seq not in seen_seqs:
                seen_seqs.add(seq)
                type_counts[tool_name] = type_counts.get(tool_name, 0) + 1

            # Calculate duration for completed/failed records
            if rec.status in ("completed", "failed") and rec.completed_at:
                try:
                    started_ts = datetime.fromisoformat(rec.started_at).timestamp()
                    completed_ts = datetime.fromisoformat(rec.completed_at).timestamp()
                    call_duration_ms = int((completed_ts - started_ts) * 1000)
                    if call_duration_ms >= 0:  # Sanity check
                        duration_ms[tool_name] = (
                            duration_ms.get(tool_name, 0) + call_duration_ms
                        )
                except (ValueError, TypeError):
                    # Skip malformed timestamps
                    pass

        total_count = len(seen_seqs)
        return total_count, type_counts, duration_ms

    def summarize_for_recovery(self, max_records: int = 20) -> str:
        """Generate summary of completed tool calls for recovery prompt.

        REQUIRES: max_records > 0
        ENSURES: Returns markdown-formatted summary
        ENSURES: Includes most recent completed tools and their results
        """
        if max_records <= 0:
            return ""

        records = self.load_records()
        completed = [r for r in records if r.status == "completed"]

        if not completed:
            return ""

        # Take most recent completed records
        recent = completed[-max_records:]

        lines = [
            "**Completed tool calls (from previous execution):**",
            "",
        ]

        for rec in recent:
            # Show the tool name and result preview
            result_preview = ""
            if rec.result_truncated:
                preview = rec.result_truncated[:100]
                if len(rec.result_truncated) > 100:
                    preview += "..."
                result_preview = f" → {preview}"
            lines.append(
                f"- `{rec.tool_name}` [{rec.tool_input_preview}]{result_preview}"
            )

        lines.append("")
        lines.append(
            "These tool calls completed successfully. Avoid re-running identical operations."
        )
        lines.append("")

        return "\n".join(lines)
