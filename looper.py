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

Module structure:
    looper/config.py   - Configuration loading, frontmatter parsing
    looper/context.py  - Session context (git log, issues, directives, feedback)
    looper/rotation.py - Phase rotation state management
    looper/hooks.py    - Git hook installation
    looper/runner.py   - LoopRunner class and main loop logic
"""

from looper import main

if __name__ == "__main__":
    main()
