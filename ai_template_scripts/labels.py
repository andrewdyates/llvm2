# Copyright 2026 Dropbox, Inc.
# Author: Andrew Yates <ayates@dropbox.com>
# Licensed under the Apache License, Version 2.0

"""Shared label constants for GitHub issue management.

This module defines all in-progress, ownership, and workflow labels used across
ai_template_scripts and looper. Single source of truth for label constants.

Public API:
- IN_PROGRESS_PREFIX: Base prefix for all in-progress labels
- IN_PROGRESS_WORKER_IDS: Range of worker IDs (1-5)
- IN_PROGRESS_OTHER_ROLE_IDS: Range of non-worker role IDs (1-3)
- IN_PROGRESS_WORKER_LABELS: Worker in-progress labels (in-progress-W1..W5)
- IN_PROGRESS_PROVER_LABELS: Prover in-progress labels (in-progress-P1..P3)
- IN_PROGRESS_RESEARCHER_LABELS: Researcher in-progress labels (in-progress-R1..R3)
- IN_PROGRESS_MANAGER_LABELS: Manager in-progress labels (in-progress-M1..M3)
- IN_PROGRESS_ALL_LABELS: All in-progress labels including base "in-progress"
- OWNERSHIP_WORKER_LABELS: Orthogonal ownership labels (W1..W5)
- OWNERSHIP_PROVER_LABELS: Orthogonal ownership labels (prov1..prov3)
- OWNERSHIP_RESEARCHER_LABELS: Orthogonal ownership labels (R1..R3)
- OWNERSHIP_MANAGER_LABELS: Orthogonal ownership labels (M1..M3)
- OWNERSHIP_ALL_LABELS: All ownership labels combined
- WORKFLOW_LABELS: Labels cleared on issue close
"""

from __future__ import annotations

__all__ = [
    "IN_PROGRESS_PREFIX",
    "IN_PROGRESS_WORKER_IDS",
    "IN_PROGRESS_OTHER_ROLE_IDS",
    "IN_PROGRESS_WORKER_LABELS",
    "IN_PROGRESS_PROVER_LABELS",
    "IN_PROGRESS_RESEARCHER_LABELS",
    "IN_PROGRESS_MANAGER_LABELS",
    "IN_PROGRESS_ALL_LABELS",
    "OWNERSHIP_WORKER_LABELS",
    "OWNERSHIP_PROVER_LABELS",
    "OWNERSHIP_RESEARCHER_LABELS",
    "OWNERSHIP_MANAGER_LABELS",
    "OWNERSHIP_ALL_LABELS",
    "WORKFLOW_LABELS",
]

# In-progress label structure
IN_PROGRESS_PREFIX = "in-progress"
IN_PROGRESS_WORKER_IDS = range(1, 6)  # W1-W5
IN_PROGRESS_OTHER_ROLE_IDS = range(1, 4)  # P1-P3, R1-R3, M1-M3

# Role-specific in-progress labels
IN_PROGRESS_WORKER_LABELS = tuple(
    f"{IN_PROGRESS_PREFIX}-W{i}" for i in IN_PROGRESS_WORKER_IDS
)
IN_PROGRESS_PROVER_LABELS = tuple(
    f"{IN_PROGRESS_PREFIX}-P{i}" for i in IN_PROGRESS_OTHER_ROLE_IDS
)
IN_PROGRESS_RESEARCHER_LABELS = tuple(
    f"{IN_PROGRESS_PREFIX}-R{i}" for i in IN_PROGRESS_OTHER_ROLE_IDS
)
IN_PROGRESS_MANAGER_LABELS = tuple(
    f"{IN_PROGRESS_PREFIX}-M{i}" for i in IN_PROGRESS_OTHER_ROLE_IDS
)

# All in-progress labels combined
IN_PROGRESS_ALL_LABELS = (
    IN_PROGRESS_PREFIX,
    *IN_PROGRESS_WORKER_LABELS,
    *IN_PROGRESS_PROVER_LABELS,
    *IN_PROGRESS_RESEARCHER_LABELS,
    *IN_PROGRESS_MANAGER_LABELS,
)

# Orthogonal ownership labels (#1453)
# These are separate from workflow state (in-progress/do-audit/needs-review)
# and persist through the entire workflow to track who owns the issue.
# Worker ownership uses W1-W5
OWNERSHIP_WORKER_LABELS = tuple(f"W{i}" for i in IN_PROGRESS_WORKER_IDS)
# Prover ownership uses prov1-prov3 (to avoid conflict with P1-P3 priority labels)
OWNERSHIP_PROVER_LABELS = tuple(f"prov{i}" for i in IN_PROGRESS_OTHER_ROLE_IDS)
# Researcher ownership uses R1-R3
OWNERSHIP_RESEARCHER_LABELS = tuple(f"R{i}" for i in IN_PROGRESS_OTHER_ROLE_IDS)
# Manager ownership uses M1-M3
OWNERSHIP_MANAGER_LABELS = tuple(f"M{i}" for i in IN_PROGRESS_OTHER_ROLE_IDS)
# All ownership labels combined
OWNERSHIP_ALL_LABELS = (
    *OWNERSHIP_WORKER_LABELS,
    *OWNERSHIP_PROVER_LABELS,
    *OWNERSHIP_RESEARCHER_LABELS,
    *OWNERSHIP_MANAGER_LABELS,
)

# Workflow labels cleared on issue close
WORKFLOW_LABELS = (
    "needs-review",
    "do-audit",
    *IN_PROGRESS_ALL_LABELS,
    *OWNERSHIP_ALL_LABELS,  # Also clear ownership when closing
    "blocked",
    "tracking",
    "urgent",
)
