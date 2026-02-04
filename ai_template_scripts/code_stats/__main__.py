# Copyright 2026 Dropbox, Inc.
# Author: Andrew Yates <ayates@dropbox.com>
# Licensed under the Apache License, Version 2.0

"""CLI entry point for running as module: python -m ai_template_scripts.code_stats"""

import sys

from ai_template_scripts.code_stats.cli import main

if __name__ == "__main__":
    sys.exit(main())
