#!/usr/bin/env python3
# Copyright 2026 Dropbox, Inc.
# Author: Andrew Yates <ayates@dropbox.com>
# Licensed under the Apache License, Version 2.0

"""
code_stats.py - Backward compatibility shim for code_stats package.

DEPRECATED: Use ai_template_scripts.code_stats package directly.

This file maintains backward compatibility for existing imports.
All functionality has been moved to ai_template_scripts/code_stats/ package.

For new code, import from:
    from ai_template_scripts.code_stats import analyze, FunctionMetric, etc.
"""

import sys
from pathlib import Path

# Support running as script (not just as module)
_repo_root = Path(__file__).resolve().parents[1]
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

# Re-export everything from the package for backward compatibility
from ai_template_scripts.code_stats import *  # noqa: F401, F403, E402
from ai_template_scripts.code_stats import __all__  # noqa: E402

if __name__ == "__main__":
    from ai_template_scripts.code_stats.cli import main

    sys.exit(main())
