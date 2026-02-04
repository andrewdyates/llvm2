#!/usr/bin/env python3
# Copyright 2026 Dropbox, Inc.
# Author: Andrew Yates <ayates@dropbox.com>
# Licensed under the Apache License, Version 2.0

"""Shim for gh_post package entrypoint and re-exports."""

import sys
from pathlib import Path

# Add parent dir to path for imports when run as script
_script_dir = Path(__file__).resolve().parent
if str(_script_dir.parent) not in sys.path:
    sys.path.insert(0, str(_script_dir.parent))

from ai_template_scripts import gh_post as gh_post_pkg  # noqa: E402
from ai_template_scripts.gh_post import *  # noqa: F401,F403,E402

__all__ = gh_post_pkg.__all__


def main() -> None:
    """Entry point for the gh wrapper shim."""
    gh_post_pkg.main()


if __name__ == "__main__":
    raise SystemExit(main())
