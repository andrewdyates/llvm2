# Copyright 2026 Dropbox, Inc.
# Author: Andrew Yates <ayates@dropbox.com>
# Licensed under the Apache License, Version 2.0

"""Entry point for python -m ai_template_scripts.bg_task."""

import sys

from ai_template_scripts.bg_task.cli import main

if __name__ == "__main__":
    sys.exit(main())
