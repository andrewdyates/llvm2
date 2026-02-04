# Copyright 2026 Dropbox, Inc.
# Author: Andrew Yates <ayates@dropbox.com>
# Licensed under the Apache License, Version 2.0

"""Entry point for running crash_analysis as a module."""

import sys

from ai_template_scripts.crash_analysis import main

if __name__ == "__main__":
    sys.exit(main())
