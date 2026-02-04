# Copyright 2026 Dropbox, Inc.
# Author: Andrew Yates
# Licensed under the Apache License, Version 2.0

"""
Health Check Base Module

Provides shared infrastructure for system_health_check.py across repos.
Each repo's scripts/system_health_check.py imports this base and adds repo-specific checks.

Exit code contract (per designs/2026-01-28-system-health-check-contract.md):
- Exit 0: All checks passed (or passed with warnings)
- Exit 1: One or more checks failed

Example usage in repos:

    from ai_template_scripts.health_check import (
        HealthCheckBase, CheckResult, Status, create_parser, standard_main
    )

    class MyHealthCheck(HealthCheckBase):
        def __init__(self):
            super().__init__()
            self.register(self.check_cargo)

        def check_cargo(self) -> CheckResult:
            ...
        check_cargo.name = "cargo_check"

    def main():
        parser = create_parser()
        args = parser.parse_args()
        hc = MyHealthCheck()
        return standard_main(hc, args)
"""

from ai_template_scripts.health_check.base import (
    Check,
    CheckResult,
    HealthCheckBase,
    Status,
)
from ai_template_scripts.health_check.cli import create_parser, standard_main

__all__ = [
    "Check",
    "CheckResult",
    "HealthCheckBase",
    "Status",
    "create_parser",
    "standard_main",
]
