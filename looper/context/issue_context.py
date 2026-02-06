# Copyright 2026 Your Name
# Author: Your Name
# Licensed under the Apache License, Version 2.0

# Copyright 2026 Dropbox, Inc.
# Author: Andrew Yates <ayates@dropbox.com>
# Licensed under the Apache License, Version 2.0

"""
looper/context/issue_context.py - Issue sampling and retrieval.

Re-exports from focused submodules for backwards compatibility.

Modules:
- issue_cache.py: IterationIssueCache for API efficiency
- issue_sampling.py: Role-based issue sampling and retrieval
- issue_audit.py: Audit label transitions (do-audit -> needs-review)
"""

# Re-export all public APIs for backwards compatibility
from looper.context.issue_audit import (
    get_do_audit_issues,
    transition_audit_to_review,
)
from looper.context.issue_cache import (
    IterationIssueCache,
    is_feature_freeze,
)
from looper.context.issue_sampling import (
    get_active_issue,
    get_issue_by_number,
    get_issues_by_numbers,
    get_issues_structured,
    get_sampled_issues,
)

__all__ = [
    # Cache
    "IterationIssueCache",
    "is_feature_freeze",
    # Sampling
    "get_active_issue",
    "get_sampled_issues",
    "get_issue_by_number",
    "get_issues_by_numbers",
    "get_issues_structured",
    # Audit
    "get_do_audit_issues",
    "transition_audit_to_review",
]
