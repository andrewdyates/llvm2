#!/usr/bin/env python3
# Copyright 2026 Dropbox, Inc.
# Author: Andrew Yates <ayates@dropbox.com>
# Licensed under the Apache License, Version 2.0

"""Backward compatibility shim for crash_analysis package.

This file re-exports the crash_analysis package for backward compatibility.
The implementation is now in ai_template_scripts/crash_analysis/__init__.py.

Usage:
    from ai_template_scripts.crash_analysis import get_health_report
    # OR
    python3 ai_template_scripts/crash_analysis.py --hours 24
"""

import sys
from pathlib import Path

# Support running as script (not just as module)
_repo_root = Path(__file__).resolve().parents[1]
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

# Re-export everything from the package
from ai_template_scripts.crash_analysis import *  # noqa: F401, F403, E402
from ai_template_scripts.crash_analysis import __all__  # noqa: E402

if __name__ == "__main__":
    from ai_template_scripts.crash_analysis import main

    sys.exit(main())
