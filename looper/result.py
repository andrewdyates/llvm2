# Copyright 2026 Dropbox, Inc.
# Author: Andrew Yates <ayates@dropbox.com>
# Licensed under the Apache License, Version 2.0

"""Result type for error handling without exceptions.

Provides a generic Result[T] type that wraps either a value or an error message,
with explicit status tracking for ok/error/skipped states.

Used throughout looper for operations that may fail (subprocess calls, file I/O)
or may be intentionally skipped (missing config, disabled features).
"""

from ai_template_scripts.result import Result, format_result

__all__ = ["Result", "format_result"]
