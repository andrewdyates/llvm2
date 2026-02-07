#!/usr/bin/env python3
# Copyright 2026 Dropbox, Inc.
# Author: Andrew Yates <ayates@dropbox.com>
# Licensed under the Apache License, Version 2.0
"""
design_issue_audit.py - Audit design documents and issue cross-references.

Finds:
1. Design docs without issue references
2. Design docs with issue references where issue doesn't link back to design
3. Issues mentioning designs that don't exist
4. Design docs with stale Status (issue closed but Status != Implemented)

Usage:
    python3 ai_template_scripts/design_issue_audit.py [--repo PATH] [--fix]

Options:
    --repo PATH   Repository path (default: current directory)
    --fix         Output fix instructions with gh commands
    --json        Output as JSON instead of Markdown table
"""

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import NoReturn


def get_design_docs(repo_path: Path) -> dict[str, dict]:
    """Find all design documents and extract their issue references."""
    designs_dir = repo_path / "designs"
    if not designs_dir.exists():
        return {}

    # Files that are not actual designs (templates, README, etc.)
    EXCLUDED_DOCS = {"TEMPLATE.md", "README.md"}

    docs = {}
    for doc_path in designs_dir.glob("*.md"):
        if doc_path.name in EXCLUDED_DOCS:
            continue
        content = doc_path.read_text(encoding="utf-8", errors="replace")

        # Extract issue references from content
        # More specific pattern to avoid false positives:
        # - Requires whitespace, line start, or punctuation before #
        # - Avoids hex colors (#fff), markdown headers (## ), HTML entities (&#)
        # Patterns matched: #123, issue #123, Issue: #123, Related: #123, fixes #123
        issue_refs = set()
        for match in re.finditer(r'(?:^|[\s(\[,;:])#(\d+)(?=[\s)\].,;:!?]|$)', content, re.MULTILINE):
            issue_refs.add(match.group(1))

        # Check for explicit "Related Issues" or "Issue:" section
        has_issue_section = bool(
            re.search(r"##\s*(Related\s+)?Issues?|^Issue:|^Related:", content, re.MULTILINE | re.IGNORECASE)
        )

        # Extract Status field (common formats: **Status:** X, Status: X)
        status_match = re.search(r"^\*?\*?Status:?\*?\*?\s*(.+)$", content, re.MULTILINE | re.IGNORECASE)
        status = status_match.group(1).strip() if status_match else None

        docs[doc_path.name] = {
            "path": str(doc_path.relative_to(repo_path)),
            "issue_refs": sorted(issue_refs, key=int) if issue_refs else [],
            "has_issue_section": has_issue_section,
            "status": status,
        }

    return docs


def get_issues(repo_path: Path, state: str = "all") -> dict[int, dict]:
    """Get issues from GitHub.

    Args:
        repo_path: Repository path
        state: Issue state filter - "open", "closed", or "all"
    """
    try:
        result = subprocess.run(
            ["gh", "issue", "list", "--state", state, "--limit", "1000",
             "--json", "number,title,body,state"],
            capture_output=True, text=True, cwd=repo_path, timeout=120
        )
        if result.returncode != 0:
            print(f"Warning: gh issue list failed: {result.stderr}", file=sys.stderr)
            return {}

        issues_data = json.loads(result.stdout)
        issues = {}
        for issue in issues_data:
            body = issue.get("body") or ""
            title = issue.get("title") or ""

            # Check if issue references a design doc
            # Patterns: designs/foo.md, Design: designs/foo.md
            design_refs = re.findall(r"designs/[\w-]+\.md", title + " " + body)

            issues[issue["number"]] = {
                "title": title,
                "state": issue.get("state", "OPEN"),
                "design_refs": list(set(design_refs)),
                "has_design_link": bool(design_refs) or "design" in title.lower(),
            }

        return issues
    except subprocess.TimeoutExpired:
        print("Warning: gh issue list timed out", file=sys.stderr)
        return {}
    except (json.JSONDecodeError, KeyError) as e:
        print(f"Warning: Failed to parse issues: {e}", file=sys.stderr)
        return {}


def audit(repo_path: Path) -> dict:
    """Run the audit and return findings."""
    docs = get_design_docs(repo_path)
    issues = get_issues(repo_path, state="all")  # Check ALL issues, not just open

    # Status values that indicate implementation is complete
    IMPLEMENTED_STATUSES = {"implemented", "done", "complete", "merged", "closed"}

    findings = {
        "orphan_designs": [],      # Designs with no issue reference
        "unlinked_designs": [],    # Designs referenced by issue but issue doesn't link back
        "missing_designs": [],     # Issues referencing designs that don't exist
        "stale_status": [],        # Designs with stale Status (issue closed but Status != Implemented)
        "good_links": [],          # Properly linked design-issue pairs
    }

    # Check each design doc
    for doc_name, doc_info in docs.items():
        if not doc_info["issue_refs"]:
            findings["orphan_designs"].append({
                "design": doc_name,
                "path": doc_info["path"],
            })
        else:
            # Check if any referenced issue links back
            for issue_num in doc_info["issue_refs"]:
                issue_num_int = int(issue_num)
                if issue_num_int in issues:
                    issue_info = issues[issue_num_int]
                    design_path = f"designs/{doc_name}"
                    if design_path in issue_info["design_refs"]:
                        findings["good_links"].append({
                            "design": doc_name,
                            "issue": issue_num_int,
                            "state": issue_info["state"],
                        })
                    else:
                        findings["unlinked_designs"].append({
                            "design": doc_name,
                            "issue": issue_num_int,
                            "issue_title": issue_info["title"][:60],
                            "state": issue_info["state"],
                        })

    # Check issues for missing designs
    for issue_num, issue_info in issues.items():
        for design_ref in issue_info["design_refs"]:
            design_name = design_ref.replace("designs/", "")
            if design_name not in docs:
                findings["missing_designs"].append({
                    "issue": issue_num,
                    "design": design_ref,
                    "issue_title": issue_info["title"][:60],
                    "state": issue_info["state"],
                })

    # Check for stale status: design has Status != Implemented but related issue is closed
    for doc_name, doc_info in docs.items():
        status = doc_info.get("status")
        if not status:
            continue
        status_lower = status.lower()
        if status_lower in IMPLEMENTED_STATUSES:
            continue

        # Check if any related issue is closed
        for issue_num in doc_info["issue_refs"]:
            issue_num_int = int(issue_num)
            if issue_num_int in issues:
                issue_info = issues[issue_num_int]
                if issue_info["state"] == "CLOSED":
                    findings["stale_status"].append({
                        "design": doc_name,
                        "status": status,
                        "issue": issue_num_int,
                        "issue_title": issue_info["title"][:60],
                    })
                    break  # Only report once per design

    return findings


def print_fix_commands(findings: dict) -> None:
    """Print fix instructions for unlinked designs and stale status."""
    has_fixes = False

    if findings["unlinked_designs"]:
        print("# Fix commands for unlinked designs")
        print("# Run each command to add design link to issue body")
        print()

        open_count = 0
        for item in findings["unlinked_designs"]:
            if item.get("state") == "CLOSED":
                continue
            open_count += 1
            design_path = f"designs/{item['design']}"
            issue_num = item["issue"]
            print(f"# {issue_num}: {item.get('issue_title', '')[:40]}")
            print(f"gh issue comment {issue_num} --body 'Design: {design_path}'")
            print()
            has_fixes = True

        if open_count == 0:
            print("# All unlinked issues are closed - no fixes needed")
            print()

    if findings.get("stale_status"):
        print("# Fix commands for stale status")
        print("# Update Status field in each design document")
        print()
        for item in findings["stale_status"]:
            design_path = f"designs/{item['design']}"
            print(f"# {design_path}: Status is '{item['status']}' but #{item['issue']} is closed")
            print(f"# Edit {design_path} and change Status to 'Implemented'")
            print()
            has_fixes = True

    if not has_fixes:
        print("# No fixes needed - all designs properly linked and status current")


def print_markdown(findings: dict) -> None:
    """Print findings as Markdown tables."""
    total_issues = (
        len(findings["orphan_designs"]) +
        len(findings["unlinked_designs"]) +
        len(findings["missing_designs"]) +
        len(findings.get("stale_status", []))
    )

    if total_issues == 0:
        print("All design documents properly linked and status fields current.")
        return

    print(f"## Design-Issue Audit: {total_issues} issues found\n")

    if findings["orphan_designs"]:
        print("### Orphan Designs (no issue reference)")
        print("| Design | Action |")
        print("|--------|--------|")
        for item in findings["orphan_designs"]:
            action = "Add ## Related Issues section"
            print(f"| {item['design']} | {action} |")
        print()

    if findings["unlinked_designs"]:
        print("### Unlinked Designs (issue doesn't link back)")
        print("| Design | Issue | State | Fix |")
        print("|--------|-------|-------|-----|")
        for item in findings["unlinked_designs"]:
            design_path = f"designs/{item['design']}"
            state = item.get("state", "OPEN")
            fix = f"Add Design: {design_path} to #{item['issue']}"
            print(f"| {item['design']} | #{item['issue']} | {state} | {fix} |")
        print()

    if findings["missing_designs"]:
        print("### Missing Designs (referenced but don't exist)")
        print("| Issue | State | Referenced Design |")
        print("|-------|-------|-------------------|")
        for item in findings["missing_designs"]:
            state = item.get("state", "OPEN")
            print(f"| #{item['issue']} | {state} | {item['design']} |")
        print()

    if findings.get("stale_status"):
        print("### Stale Status (issue closed but Status != Implemented)")
        print("| Design | Current Status | Closed Issue | Fix |")
        print("|--------|----------------|--------------|-----|")
        for item in findings["stale_status"]:
            fix = "Update Status to Implemented"
            print(f"| {item['design']} | {item['status']} | #{item['issue']} | {fix} |")
        print()

    if findings["good_links"]:
        print(f"### Properly Linked: {len(findings['good_links'])} pairs")


def main() -> NoReturn:
    parser = argparse.ArgumentParser(description="Audit design document cross-references")
    parser.add_argument("--repo", type=Path, default=Path.cwd(), help="Repository path")
    parser.add_argument("--fix", action="store_true", help="Output fix commands")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    args = parser.parse_args()

    findings = audit(args.repo)

    if args.fix:
        print_fix_commands(findings)
    elif args.json:
        print(json.dumps(findings, indent=2))
    else:
        print_markdown(findings)

    # Exit with error code if issues found
    total_issues = (
        len(findings["orphan_designs"]) +
        len(findings["unlinked_designs"]) +
        len(findings["missing_designs"]) +
        len(findings.get("stale_status", []))
    )
    sys.exit(1 if total_issues > 0 else 0)


if __name__ == "__main__":
    main()
