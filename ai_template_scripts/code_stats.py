#!/usr/bin/env python3
# Copyright 2026 Dropbox, Inc.
# Author: Andrew Yates <ayates@dropbox.com>
# Licensed under the Apache License, Version 2.0

"""Shim for code_stats package entrypoint and re-exports.

Delegates __all__ to the package to prevent drift (#1948).
"""

import sys
from pathlib import Path

# Add parent dir to path for imports when run as script
_script_dir = Path(__file__).resolve().parent
if str(_script_dir.parent) not in sys.path:
    sys.path.insert(0, str(_script_dir.parent))

from ai_template_scripts import code_stats as code_stats_pkg  # noqa: E402
from ai_template_scripts.code_stats import *  # noqa: F401,F403,E402

__all__ = code_stats_pkg.__all__


def main() -> int:
    """Entry point for the code_stats shim."""
    return code_stats_pkg.main()


if __name__ == "__main__":
    raise SystemExit(main())
