# Copyright 2026 Dropbox, Inc.
# Author: Andrew Yates <ayates@dropbox.com>
# Licensed under the Apache License, Version 2.0

"""Entry point for python -m ai_template_scripts.cargo_wrapper."""

import sys

from .cli import main

if __name__ == "__main__":
    sys.exit(main())
