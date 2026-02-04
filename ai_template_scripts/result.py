# Copyright 2026 Dropbox, Inc.
# Author: Andrew Yates <ayates@dropbox.com>
# Licensed under the Apache License, Version 2.0

"""Result type for error handling without exceptions.

Provides a generic Result[T] type that wraps either a value or an error message,
with explicit status tracking for ok/error/skipped states.

Used throughout looper and scripts for operations that may fail (subprocess calls, file I/O)
or may be intentionally skipped (missing config, disabled features).
"""

__all__ = ["Result", "Status", "format_result"]

from dataclasses import dataclass
from typing import Generic, Literal, TypeVar

T = TypeVar("T")

Status = Literal["ok", "error", "skipped"]


@dataclass(frozen=True)
class Result(Generic[T]):
    """Result type with explicit status for ok/error/skipped states.

    Status semantics:
    - ok: Operation succeeded, value contains result
    - error: Operation failed, error contains message, value may contain partial data
    - skipped: Operation intentionally not performed, error contains reason
    """

    value: T | None = None
    error: str | None = None
    status: Status = "ok"

    def __post_init__(self) -> None:
        # Infer status from error field for backward compatibility
        if self.status == "ok" and self.error is not None:
            object.__setattr__(self, "status", "error")

    @property
    def ok(self) -> bool:
        return self.status == "ok"

    @property
    def skipped(self) -> bool:
        return self.status == "skipped"

    @classmethod
    def success(cls, value: T | None = None) -> "Result[T]":
        """Create a successful result with the given value."""
        return cls(value=value, status="ok")

    @classmethod
    def failure(cls, error: str, value: T | None = None) -> "Result[T]":
        """Create an error result with the given message and optional partial value."""
        return cls(value=value, error=error, status="error")

    @classmethod
    def skip(cls, reason: str) -> "Result[T]":
        """Create a skipped result with the given reason.

        Use for operations intentionally not performed (missing config, disabled features).
        """
        return cls(error=reason, status="skipped")

    def to_dict(self) -> dict[str, object]:
        """Convert to dict for JSON serialization and metrics."""
        return {
            "ok": self.ok,
            "value": self.value,
            "error": self.error,
            "status": self.status,
        }


def format_result(result: Result[str]) -> str:
    """Format a Result for prompt injection.

    - ok: returns raw value
    - error: returns "(error: <message>)" optionally followed by partial value
    - skipped: returns "(skipped: <reason>)"
    """
    value = result.value or ""

    if result.status == "skipped":
        reason = result.error or "no reason given"
        return f"(skipped: {reason})"

    if result.ok:
        return value

    error = result.error or "unknown error"
    if value:
        return f"(error: {error})\n{value}"
    return f"(error: {error})"
