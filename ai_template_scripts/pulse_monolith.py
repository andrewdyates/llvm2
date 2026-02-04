#!/usr/bin/env python3
# Copyright 2026 Dropbox, Inc.
# Author: Andrew Yates <ayates@dropbox.com>
# Licensed under the Apache License, Version 2.0

"""Backward compatibility shim for pulse_monolith.

This module re-exports from the pulse package to maintain compatibility
with older repo-local pulse.py scripts that import from pulse_monolith.

Part of #2143: Restore pulse_monolith compatibility.

Deprecation: Repos should sync to get the new pulse.py shim that imports
directly from the pulse package. This file will be removed in a future
version after all repos have synced.
"""

# Re-export everything from pulse package for backward compatibility
# Star import re-exports all public symbols defined by pulse package
from ai_template_scripts.pulse import *  # noqa: F401,F403
