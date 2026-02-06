#!/usr/bin/env python3
# Copyright 2026 Your Name
# Author: Your Name
# Licensed under the Apache License, Version 2.0

# Copyright 2026 Dropbox, Inc.
# Author: Andrew Yates <ayates@dropbox.com>
# Licensed under the Apache License, Version 2.0

"""Shim for json_to_text package entrypoint and re-exports.

Delegates __all__ to the package to prevent drift (#1950).
"""

import sys
from pathlib import Path

# Add parent dir to path for imports when run as script
_script_dir = Path(__file__).resolve().parent
if str(_script_dir.parent) not in sys.path:
    sys.path.insert(0, str(_script_dir.parent))

from ai_template_scripts import json_to_text as json_to_text_pkg  # noqa: E402
from ai_template_scripts.json_to_text import *  # noqa: F401,F403,E402

__all__ = json_to_text_pkg.__all__


def main() -> None:
    """Entry point for the json_to_text shim."""
    return json_to_text_pkg.main()


if __name__ == "__main__":
    raise SystemExit(main())
