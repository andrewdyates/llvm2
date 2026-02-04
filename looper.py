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

Module structure:
    looper/__init__.py       - Package exports
    looper/config.py         - Configuration loading, frontmatter parsing
    looper/context/          - Session context subpackage (git, issue, audit, system)
    looper/hooks.py          - Git hook installation
    looper/issue_manager.py  - Issue operations (gh wrapper, sampling)
    looper/iteration.py      - Iteration execution (prompt build, AI run, metrics)
    looper/result.py         - Result monad for error handling
    looper/rotation.py       - Phase rotation state management
    looper/runner.py         - LoopRunner class and main loop logic
    looper/status.py         - Status tracking (metrics, logs)
    looper/subprocess_utils.py - Result[T] wrappers for subprocess commands
    looper/sync.py           - Multi-machine zone branch syncing
    looper/telemetry.py      - Metrics collection
    looper/zones.py          - Zone-based file locking for multi-worker
"""

# Disable bytecode caching to prevent stale .pyc issues after git operations.
# Git rebase/checkout can leave file timestamps inconsistent, causing Python
# to use outdated cached bytecode instead of recompiling from source.
import sys

sys.dont_write_bytecode = True

from looper import main

if __name__ == "__main__":
    main()
