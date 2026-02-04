#!/usr/bin/env python3
# Copyright 2026 Dropbox, Inc.
# Author: Andrew Yates <ayates@dropbox.com>
# Licensed under the Apache License, Version 2.0

"""
Module entrypoint for running pulse as: python -m ai_template_scripts.pulse

Part of #404: pulse.py module split
"""

if __name__ == "__main__":
    from .core import main

    main()
