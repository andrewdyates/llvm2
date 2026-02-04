# Copyright 2026 Dropbox, Inc.
# Author: Andrew Yates <ayates@dropbox.com>
# Licensed under the Apache License, Version 2.0

from __future__ import annotations

import json
import logging
import os
import sys
import traceback
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

__all__ = [
    "build_log_path",
    "debug_swallow",
    "get_logger",
    "log_debug",
    "log_error",
    "log_info",
    "log_warning",
    "setup_logging",
]

_LOGGER_NAME = "looper"
_STREAM_ATTR = "_stream"

_STANDARD_RECORD_KEYS = {
    "name",
    "msg",
    "args",
    "levelname",
    "levelno",
    "pathname",
    "filename",
    "module",
    "exc_info",
    "exc_text",
    "stack_info",
    "lineno",
    "funcName",
    "created",
    "msecs",
    "relativeCreated",
    "thread",
    "threadName",
    "processName",
    "process",
    _STREAM_ATTR,
}


class _StreamFilter(logging.Filter):
    def __init__(self, stream: str) -> None:
        super().__init__()
        self._stream = stream

    def filter(self, record: logging.LogRecord) -> bool:
        record_stream = getattr(record, _STREAM_ATTR, None)
        if record_stream is None:
            record_stream = "stderr" if record.levelno >= logging.WARNING else "stdout"
        return record_stream == self._stream


class _JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": datetime.now(UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        extra = _extract_extra_fields(record)
        if extra:
            payload["extra"] = extra
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=True)


def _extract_extra_fields(record: logging.LogRecord) -> dict[str, Any]:
    """Extract non-standard fields from a LogRecord for JSON serialization."""
    extra: dict[str, Any] = {}
    for key, value in record.__dict__.items():
        if key in _STANDARD_RECORD_KEYS:
            continue
        extra[key] = value
    return extra


def _safe_extra(fields: dict[str, Any]) -> dict[str, Any]:
    """Filter out keys that conflict with standard LogRecord attributes."""
    return {
        key: value for key, value in fields.items() if key not in _STANDARD_RECORD_KEYS
    }


def _has_handler(logger: logging.Logger, name: str) -> bool:
    """Check if logger already has a handler with the given name."""
    return any(handler.name == name for handler in logger.handlers)


def _refresh_console_streams(logger: logging.Logger) -> None:
    """Update console handlers to use current stdout/stderr after redirects."""
    for handler in logger.handlers:
        if isinstance(handler, logging.StreamHandler):
            if handler.name == "looper_stdout":
                handler.stream = sys.stdout
            elif handler.name == "looper_stderr":
                handler.stream = sys.stderr


def setup_logging(log_file: Path | None = None) -> logging.Logger:
    """Configure looper logging with console and optional JSON file output.

    Creates stdout/stderr handlers with stream filtering (DEBUG/INFO to stdout,
    WARNING+ to stderr). Optionally adds a JSON file handler for structured logs.
    Safe to call multiple times; only configures once per interpreter.
    """
    logger = logging.getLogger(_LOGGER_NAME)
    if not getattr(logger, "_looper_configured", False):
        logger.setLevel(logging.DEBUG)
        logger.propagate = False

        console_format = logging.Formatter("%(message)s")

        stdout_handler = logging.StreamHandler(sys.stdout)
        stdout_handler.name = "looper_stdout"
        stdout_handler.setLevel(logging.DEBUG)
        stdout_handler.addFilter(_StreamFilter("stdout"))
        stdout_handler.setFormatter(console_format)
        logger.addHandler(stdout_handler)

        stderr_handler = logging.StreamHandler(sys.stderr)
        stderr_handler.name = "looper_stderr"
        stderr_handler.setLevel(logging.DEBUG)
        stderr_handler.addFilter(_StreamFilter("stderr"))
        stderr_handler.setFormatter(console_format)
        logger.addHandler(stderr_handler)

        logger._looper_configured = True  # type: ignore[attr-defined]

    _refresh_console_streams(logger)

    if log_file and not _has_handler(logger, "looper_json"):
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.name = "looper_json"
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(_JsonFormatter())
        logger.addHandler(file_handler)

    return logger


def get_logger() -> logging.Logger:
    """Return the configured looper logger, initializing if needed."""
    return setup_logging()


def _log(level: int, message: str, stream: str | None, fields: dict[str, Any]) -> None:
    """Internal logging helper with optional stream routing and extra fields."""
    logger = get_logger()
    extra = _safe_extra(fields)
    if stream:
        extra[_STREAM_ATTR] = stream
    if extra:
        logger.log(level, message, extra=extra)
    else:
        logger.log(level, message)


def log_debug(message: str, *, stream: str | None = None, **fields: Any) -> None:
    """Log a DEBUG message with optional stream routing and extra fields."""
    _log(logging.DEBUG, message, stream, fields)


def log_info(message: str, *, stream: str | None = None, **fields: Any) -> None:
    """Log an INFO message with optional stream routing and extra fields."""
    _log(logging.INFO, message, stream, fields)


def log_warning(message: str, *, stream: str | None = None, **fields: Any) -> None:
    """Log a WARNING message with optional stream routing and extra fields."""
    _log(logging.WARNING, message, stream, fields)


def log_error(message: str, *, stream: str | None = None, **fields: Any) -> None:
    """Log an ERROR message with optional stream routing and extra fields."""
    _log(logging.ERROR, message, stream, fields)


def build_log_path(
    log_dir: Path, mode: str, worker_id: int | None = None, machine: str | None = None
) -> Path:
    """Build timestamped log file path with mode, worker ID, and machine suffix."""
    suffix = f"{mode}_{worker_id}" if worker_id is not None else mode
    if machine:
        suffix = f"{suffix}_{machine}"
    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    return log_dir / f"{suffix}_system_{timestamp}.jsonl"


# Debug mode: set LOOPER_DEBUG=1 to see swallowed exception tracebacks
_DEBUG = os.environ.get("LOOPER_DEBUG", "0") == "1"


def debug_swallow(operation: str, exc: BaseException | None = None) -> None:
    """Log swallowed exception in debug mode.

    Use in except blocks where exceptions are intentionally swallowed:

        try:
            do_something()
        except OSError:
            debug_swallow("do_something")  # Logs traceback if LOOPER_DEBUG=1

    When LOOPER_DEBUG=1, logs the operation name and full traceback.
    When debug is off (default), this is a no-op for zero overhead.

    Part of #387 - visibility into silently swallowed exceptions.

    Args:
        operation: Name of the operation that failed (for log message)
        exc: Optional exception to log. If None, uses current exception info.
    """
    if not _DEBUG:
        return

    timestamp = datetime.now(UTC).strftime("%H:%M:%S.%f")[:-3]
    print(f"[{timestamp}] swallowed exception in {operation}", file=sys.stderr)
    if exc is not None:
        traceback.print_exception(type(exc), exc, exc.__traceback__, file=sys.stderr)
    else:
        traceback.print_exc(file=sys.stderr)
