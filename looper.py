#!/usr/bin/env python3
"""
looper.py - Autonomous continuous loop for AI workers and managers

Copyright 2026 Dropbox, Inc.
Created by Andrew Yates
Licensed under the Apache License, Version 2.0

Usage:
    ./looper.py worker      # Fast autonomous code loop (continuous)
    ./looper.py prover      # Proof verification loop (15-min intervals)
    ./looper.py researcher  # Research and design loop (10-min intervals)
    ./looper.py manager     # Audit and coordination loop (5-min intervals)
    ./looper.py cleanup     # Remove stale state files and exit

Module structure: See README.md "Looper Package" section for complete module listing.
"""

# Disable bytecode caching to prevent stale .pyc issues after git operations.
# Git rebase/checkout can leave file timestamps inconsistent, causing Python
# to use outdated cached bytecode instead of recompiling from source.
import sys

sys.dont_write_bytecode = True

from looper import main

if __name__ == "__main__":
    main()
