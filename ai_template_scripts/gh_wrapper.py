#!/usr/bin/env python3
# Copyright 2026 Dropbox, Inc.
# Author: Andrew Yates <ayates@dropbox.com>
# Licensed under the Apache License, Version 2.0

"""
gh_wrapper.py - Transparent gh wrapper with rate limiting and caching

PURPOSE: Intercept gh commands to add rate limiting and caching.
CALLED BY: bin/gh shim (all gh commands except issue create/comment/edit/close/cleanup-closed)
REFERENCED: .claude/rules/ai_template.md (GitHub API Management section)

Uses RateLimiter for:
- Rate limit checking (blocks if quota critical)
- Response caching for read operations
- Cache invalidation on writes

AIs don't interact with this directly - they just run `gh` normally.

Public API (library usage):
    from ai_template_scripts.gh_wrapper import (
        main,  # CLI entry point wrapping gh commands
    )

CLI usage (via bin/gh shim):
    gh issue list                    # Routed through wrapper
    gh repo view                     # Routed through wrapper
    gh api ...                       # Routed through wrapper
"""

__all__ = ["main"]

import sys
from pathlib import Path

# Add parent dir to path for imports when run as script
_script_dir = Path(__file__).resolve().parent
if str(_script_dir.parent) not in sys.path:
    sys.path.insert(0, str(_script_dir.parent))

from ai_template_scripts.gh_rate_limit import get_limiter  # noqa: E402


def main() -> int:
    args = sys.argv[1:]
    if not args:
        # No args = just run gh with no args (shows help)
        limiter = get_limiter()
        result = limiter.call([])
        # Ensure stdout ends with newline to prevent stderr interleaving (#1722)
        stdout = result.stdout
        if stdout and not stdout.endswith("\n"):
            stdout = stdout + "\n"
        print(stdout, end="")
        if result.stderr:
            print(result.stderr, end="", file=sys.stderr)
        return result.returncode

    limiter = get_limiter()
    result = limiter.call(args)

    # Output stdout/stderr
    # Ensure stdout ends with newline to prevent stderr interleaving (#1722)
    if result.stdout:
        stdout = result.stdout
        if stdout and not stdout.endswith("\n"):
            stdout = stdout + "\n"
        print(stdout, end="")
    if result.stderr:
        print(result.stderr, end="", file=sys.stderr)

    return result.returncode


if __name__ == "__main__":
    sys.exit(main())
