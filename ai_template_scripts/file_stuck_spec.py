#!/usr/bin/env python3
# Copyright 2026 Dropbox, Inc.
# Author: Andrew Yates <ayates@dropbox.com>
# Licensed under the Apache License, Version 2.0
"""
file_stuck_spec.py - File issues for stuck verification specs to zani/tla2.

When Kani or TLA+ specs timeout or hang, this script files a structured issue
to the appropriate repo so stuck patterns can be addressed systematically.

Usage:
    python3 ai_template_scripts/file_stuck_spec.py kani --harness check_bounds --timeout 2h --property "array bounds"
    python3 ai_template_scripts/file_stuck_spec.py tla --spec Consensus.tla --timeout 3h --property "liveness"
"""

import argparse
import subprocess
import sys


def file_kani_issue(harness: str, timeout: str, prop: str, repo: str) -> int:
    """File stuck Kani proof issue to zani."""
    title = f"[stuck] Kani harness: {harness} (timeout: {timeout})"
    body = f"""## Stuck Kani Proof

**Harness:** {harness}
**Timeout:** {timeout}
**Property:** {prop}
**Source repo:** {repo}

## Context

This harness exceeded the verification timeout. Filing for systematic analysis.

## Potential causes
- [ ] Loop bound too high
- [ ] Missing invariants
- [ ] SMT solver explosion
- [ ] Memory model complexity

## Requested action
Analyze why this proof times out and either:
1. Add solver hints/invariants to make it tractable
2. Document known limitation
3. Simplify the property being verified

---
Filed by: file_stuck_spec.py from {repo}
"""

    cmd = [
        "gh", "issue", "create",
        "--repo", "dropbox-ai-prototypes/zani",
        "--title", title,
        "--body", body,
        "--label", "stuck-proof",
        "--label", "P2",
    ]

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    if result.returncode == 0:
        print(f"Filed: {result.stdout.strip()}")
        return 0
    else:
        print(f"Error: {result.stderr}", file=sys.stderr)
        return 1


def file_tla_issue(spec: str, timeout: str, prop: str, repo: str) -> int:
    """File stuck TLA+ spec issue to tla2."""
    title = f"[stuck] TLA+ spec: {spec} (timeout: {timeout})"
    body = f"""## Stuck TLA+ Spec

**Spec:** {spec}
**Timeout:** {timeout}
**Property:** {prop}
**Source repo:** {repo}

## Context

This spec exceeded the TLC verification timeout. Filing for systematic analysis.

## Potential causes
- [ ] State space explosion
- [ ] Missing symmetry reduction
- [ ] Unbounded model values
- [ ] Liveness property too complex

## Requested action
Analyze why this spec times out and either:
1. Add state constraints or symmetry sets
2. Decompose into smaller specs
3. Document known limitation

---
Filed by: file_stuck_spec.py from {repo}
"""

    cmd = [
        "gh", "issue", "create",
        "--repo", "dropbox-ai-prototypes/tla2",
        "--title", title,
        "--body", body,
        "--label", "stuck-spec",
        "--label", "P2",
    ]

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    if result.returncode == 0:
        print(f"Filed: {result.stdout.strip()}")
        return 0
    else:
        print(f"Error: {result.stderr}", file=sys.stderr)
        return 1


def get_current_repo() -> str:
    """Get current repo name from git remote."""
    try:
        result = subprocess.run(
            ["git", "remote", "get-url", "origin"],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0:
            url = result.stdout.strip()
            # Extract repo name from URL
            if "github.com" in url:
                parts = url.rstrip(".git").split("/")
                return "/".join(parts[-2:])
        return "unknown"
    except Exception:
        return "unknown"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="File stuck verification specs to zani/tla2"
    )
    subparsers = parser.add_subparsers(dest="type", required=True)

    # Kani subcommand
    kani_parser = subparsers.add_parser("kani", help="File stuck Kani proof")
    kani_parser.add_argument("--harness", required=True, help="Kani harness name")
    kani_parser.add_argument("--timeout", required=True, help="Timeout duration (e.g., 2h)")
    kani_parser.add_argument("--property", required=True, help="Property being verified")

    # TLA subcommand
    tla_parser = subparsers.add_parser("tla", help="File stuck TLA+ spec")
    tla_parser.add_argument("--spec", required=True, help="TLA+ spec filename")
    tla_parser.add_argument("--timeout", required=True, help="Timeout duration (e.g., 3h)")
    tla_parser.add_argument("--property", required=True, help="Property being checked")

    args = parser.parse_args()
    repo = get_current_repo()

    if args.type == "kani":
        return file_kani_issue(args.harness, args.timeout, args.property, repo)
    elif args.type == "tla":
        return file_tla_issue(args.spec, args.timeout, args.property, repo)
    else:
        print(f"Unknown type: {args.type}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
