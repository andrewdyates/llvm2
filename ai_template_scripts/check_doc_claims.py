#!/usr/bin/env python3
# Copyright 2026 Dropbox, Inc.
# Author: Andrew Yates
# Licensed under the Apache License, Version 2.0

"""check_doc_claims.py - Verify machine-checkable documentation claims.

Parses YAML frontmatter from CLAUDE.md, VISION.md, and README.md, verifies
that claimed code paths and markers actually exist in the codebase.

Usage:
    check_doc_claims.py                      # Check all docs, human output
    check_doc_claims.py --json               # Machine-readable output
    check_doc_claims.py --path ~/some/repo   # Check specific repo
    check_doc_claims.py --heuristic          # Keyword drift detection
    check_doc_claims.py --heuristic --keywords=custom.json  # Custom keywords

Claim format in YAML frontmatter:
---
claims:
  - type: backend
    name: z4
    code_path: src/backends/z4/
    status: active
  - type: feature
    name: BigInt
    code_marker: "// VISION: BigInt"
---

Copyright 2026 Dropbox, Inc.
Author: Andrew Yates <ayates@dropbox.com>
Licensed under the Apache License, Version 2.0
"""

from __future__ import annotations

import argparse
import functools
import json
import re
import sys
from pathlib import Path
from typing import Any, TypedDict

# Support running as script (not just as module)
_repo_root = Path(__file__).resolve().parents[1]
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from ai_template_scripts.exclude_patterns import SKIP_DIRS  # noqa: E402
from ai_template_scripts.version import get_version  # noqa: E402

__all__ = [
    "DOC_FILES",
    "DEFAULT_KEYWORDS",
    "parse_yaml_frontmatter",
    "verify_code_path",
    "verify_code_marker",
    "check_claim",
    "check_doc_claims",
    "DocClaimsResults",
    # Heuristic keyword drift detection (#1518)
    "load_keywords",
    "extract_doc_keywords",
    "check_keyword_in_code",
    "check_keywords_in_code_batch",  # Batch version for O(n*m*d) with early termination (#1699)
    "check_heuristic_drift",
    "HeuristicResult",
]

# Files to scan for claims (#2917: includes README.md for scope alignment)
DOC_FILES = ["CLAUDE.md", "VISION.md", "README.md"]

# Default keyword lists for heuristic drift detection (#1518)
# These keywords indicate capabilities that should have code backing them
DEFAULT_KEYWORDS = {
    "backends": ["cbmc", "z4", "llvm", "mir", "wasm", "kani", "lean", "coq", "tla+"],
    "solvers": ["yices", "cvc5", "minisat", "kissat", "glucose", "cadical"],
    "features": [
        "bigint",
        "hashmap",
        "chc",
        "horn",
        "smt",
        "sat",
        "bmc",
        "interpolation",
    ],
    "languages": ["rust", "python", "go", "typescript", "javascript", "swift", "c++"],
}

# YAML frontmatter pattern: --- at start, content, ---
FRONTMATTER_PATTERN = re.compile(
    r"^---\s*\n(?P<content>.*?)\n---",
    re.DOTALL,
)


def parse_yaml_frontmatter(content: str) -> dict | None:
    """Parse YAML frontmatter from document content.

    Uses a simple parser for the subset of YAML we expect (lists and dicts).
    Avoids PyYAML dependency for portability.

    REQUIRES: content is a string (may be empty)
    ENSURES: Returns parsed dict if frontmatter found, None otherwise
    """
    match = FRONTMATTER_PATTERN.match(content)
    if not match:
        return None

    yaml_content = match.group("content")
    return _parse_simple_yaml(yaml_content)


def _parse_simple_yaml(content: str) -> dict:
    """Parse simple YAML (claims list with key-value pairs).

    Supports:
    - claims: list of dicts
    - Each claim has: type, name, code_path/code_marker, status (optional)
    """
    result: dict[str, Any] = {}
    lines = content.strip().split("\n")
    current_list_key: str | None = None
    current_list: list[dict] = []
    current_item: dict[str, str] = {}
    in_list_item = False

    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue

        # Top-level key with list indicator: "claims:"
        if line[0] not in " \t" and stripped.endswith(":"):
            # Save previous list if any
            if current_list_key and current_list:
                result[current_list_key] = current_list
            key = stripped[:-1].strip()
            current_list_key = key
            current_list = []
            current_item = {}
            in_list_item = False
            continue

        # List item start: "  - type: backend"
        if stripped.startswith("- "):
            # Save previous item if any
            if current_item and in_list_item:
                current_list.append(current_item)
                current_item = {}
            in_list_item = True
            # Parse the key-value on the same line as dash
            rest = stripped[2:].strip()
            if ":" in rest:
                k, v = rest.split(":", 1)
                current_item[k.strip()] = _unquote(v.strip())
            continue

        # Continuation of list item: "    name: z4"
        if in_list_item and ":" in stripped:
            k, v = stripped.split(":", 1)
            current_item[k.strip()] = _unquote(v.strip())
            continue

    # Save final item and list
    if current_item and in_list_item:
        current_list.append(current_item)
    if current_list_key and current_list:
        result[current_list_key] = current_list

    return result


def _unquote(s: str) -> str:
    """Remove surrounding quotes from a string."""
    if len(s) >= 2:
        if (s[0] == '"' and s[-1] == '"') or (s[0] == "'" and s[-1] == "'"):
            return s[1:-1]
    return s


def verify_code_path(base_dir: Path, code_path: str) -> bool:
    """Check if a code path (file or directory) exists.

    REQUIRES: base_dir is a Path to existing directory
    REQUIRES: code_path is a relative path string
    ENSURES: Returns True iff base_dir/code_path exists
    """
    target = base_dir / code_path
    return target.exists()


def verify_code_marker(base_dir: Path, marker: str) -> bool:
    """Search for a code marker string in the codebase.

    Uses grep-like search. Returns True if marker found in any file.
    Limited to common source extensions for performance.
    Excludes DOC_FILES to avoid matching claims in the docs themselves.

    REQUIRES: base_dir is a Path to existing directory
    REQUIRES: marker is non-empty string
    ENSURES: Returns True iff marker found in any source file
    """
    # Source extensions to search (no .md to avoid matching claims in docs)
    extensions = [
        "*.py",
        "*.rs",
        "*.go",
        "*.ts",
        "*.js",
        "*.tsx",
        "*.jsx",
        "*.c",
        "*.cpp",
        "*.h",
        "*.hpp",
    ]

    # Escape special regex chars in marker
    pattern = re.escape(marker)

    for ext in extensions:
        for file_path in base_dir.rglob(ext):
            # Skip hidden dirs and common excludes
            if any(
                part.startswith(".") or part in ("node_modules", "target", "build")
                for part in file_path.parts
            ):
                continue
            try:
                content = file_path.read_text(encoding="utf-8", errors="ignore")
                if re.search(pattern, content):
                    return True
            except (OSError, UnicodeDecodeError):
                continue

    return False


def check_claim(base_dir: Path, claim: dict) -> dict:
    """Verify a single claim, return result with status.

    REQUIRES: base_dir is a Path to existing directory
    REQUIRES: claim is dict with optional "code_path" and/or "code_marker"
    ENSURES: Returns dict with "verified" key (True if all paths/markers exist)
    """
    result = {
        "type": claim.get("type", "unknown"),
        "name": claim.get("name", "unnamed"),
        "status": claim.get("status", "active"),
    }

    # Check code_path if specified
    if "code_path" in claim:
        result["code_path"] = claim["code_path"]
        result["code_path_verified"] = verify_code_path(base_dir, claim["code_path"])

    # Check code_marker if specified
    if "code_marker" in claim:
        result["code_marker"] = claim["code_marker"]
        result["code_marker_verified"] = verify_code_marker(
            base_dir, claim["code_marker"]
        )

    # Determine overall verification status.
    # A claim with neither code_path nor code_marker has no verifiable
    # assertion and must not be counted as verified (vacuous-pass guard).
    has_assertions = "code_path_verified" in result or "code_marker_verified" in result
    verified = has_assertions
    if "code_path_verified" in result and not result["code_path_verified"]:
        verified = False
    if "code_marker_verified" in result and not result["code_marker_verified"]:
        verified = False

    result["verified"] = verified
    return result


def _has_frontmatter_delimiters(content: str) -> bool:
    """Check if content starts with YAML frontmatter delimiters (``---``)."""
    return bool(re.match(r"^---\s*\n", content))


class DocClaimsResults(TypedDict):
    """Results from checking documentation claims."""

    docs: list[dict[str, Any]]
    total_claims: int
    verified_claims: int
    unverified_claims: int
    drift: list[dict[str, Any]]
    parse_errors: list[dict[str, str]]


# Heuristic keyword detection (#1518)


def load_keywords(keywords_path: Path | None) -> dict[str, list[str]]:
    """Load keywords from custom JSON file or use defaults.

    Custom JSON format:
    {
        "backends": ["cbmc", "z4", ...],
        "solvers": [...],
        "features": [...],
        ...
    }

    REQUIRES: keywords_path is None or Path to JSON file
    ENSURES: Returns dict of category -> keyword list (never empty)
    ENSURES: On error, returns DEFAULT_KEYWORDS
    """
    if keywords_path is None:
        return DEFAULT_KEYWORDS

    try:
        content = keywords_path.read_text(encoding="utf-8")
        custom = json.loads(content)
        if isinstance(custom, dict):
            # Merge with defaults - custom overrides
            merged = dict(DEFAULT_KEYWORDS)
            merged.update(custom)
            return merged
    except (OSError, json.JSONDecodeError) as e:
        print(f"Warning: Failed to load {keywords_path}: {e}", file=sys.stderr)

    return DEFAULT_KEYWORDS


def extract_doc_keywords(base_dir: Path) -> set[str]:
    """Extract all keywords mentioned in documentation.

    Scans DOC_FILES (CLAUDE.md, VISION.md, README.md) for keyword mentions.
    Returns set of lowercase keywords found.

    REQUIRES: base_dir is a Path to existing directory
    ENSURES: Returns set of lowercase strings (may be empty)
    """
    found: set[str] = set()

    for doc_name in DOC_FILES:
        doc_path = base_dir / doc_name
        if not doc_path.exists():
            continue

        try:
            content = doc_path.read_text(encoding="utf-8").lower()
            # Extract words - alphanumeric + some special chars
            words = set(re.findall(r"[a-z0-9_+-]+", content))
            found.update(words)
        except (OSError, UnicodeDecodeError):
            continue

    return found


@functools.lru_cache(maxsize=128)
def _make_keyword_pattern(keyword: str) -> re.Pattern:
    """Create regex pattern for keyword that handles special chars (#1698).

    Uses lru_cache to avoid recompiling same patterns per call.
    Standard word boundary \\b doesn't work with keywords like 'tla+' or 'c++'
    because + is not a word character. Use lookahead/lookbehind instead.
    """
    escaped = re.escape(keyword.lower())
    # For alphanumeric-only keywords, use standard word boundary
    if keyword.replace("_", "").isalnum():
        return re.compile(rf"\b{escaped}\b", re.IGNORECASE)
    # For keywords with special chars, use negative lookahead/lookbehind
    # to ensure keyword isn't part of a larger alphanumeric string
    return re.compile(rf"(?<![a-z0-9]){escaped}(?![a-z0-9])", re.IGNORECASE)


# Source extensions for keyword search
_SOURCE_EXTENSIONS = {".py", ".rs", ".go", ".ts", ".js", ".c", ".cpp", ".toml"}

# SKIP_DIRS imported from exclude_patterns.py (single source of truth, #1798)


def _should_skip_path(path: Path) -> bool:
    """Check if path should be skipped for keyword search."""
    for part in path.parts:
        # Also skip any dotfile directories (hidden files/dirs)
        if part.startswith(".") or part in SKIP_DIRS:
            return True
    return False


def check_keywords_in_code_batch(base_dir: Path, keywords: set[str]) -> set[str]:
    """Batch-check which keywords appear in codebase (O(n*m*d) instead of O(n*m*d*k)).

    Single-pass optimization (#1699): Traverses filesystem once and checks all
    keywords per file, rather than traversing once per keyword.

    REQUIRES: base_dir is a Path to existing directory
    REQUIRES: keywords is set of lowercase strings
    ENSURES: Returns subset of keywords that appear in code

    Args:
        base_dir: Repository root directory
        keywords: Set of keywords to search for (lowercase)

    Returns:
        Set of keywords that were found in code (paths or content)
    """
    if not keywords:
        return set()

    found: set[str] = set()
    remaining = set(keywords)

    # Precompute patterns for all keywords
    patterns = {kw: _make_keyword_pattern(kw) for kw in keywords}

    # Single-pass traversal
    for path in base_dir.rglob("*"):
        if not remaining:
            break  # All keywords found

        if _should_skip_path(path):
            continue

        path_name_lower = path.name.lower()

        # Check 1: Keyword in path name
        for kw in list(remaining):
            if kw in path_name_lower:
                found.add(kw)
                remaining.discard(kw)

        if not remaining:
            break

        # Check 2: Content search (only source files)
        if path.is_file() and path.suffix in _SOURCE_EXTENSIONS:
            if path.name in DOC_FILES:
                continue

            try:
                content = path.read_text(encoding="utf-8", errors="ignore")
                for kw in list(remaining):
                    if patterns[kw].search(content):
                        found.add(kw)
                        remaining.discard(kw)
            except (OSError, UnicodeDecodeError):
                continue

    return found


def check_keyword_in_code(base_dir: Path, keyword: str) -> bool:
    """Check if keyword appears in codebase (paths, imports, markers).

    Looks for:
    1. Directory/file names containing the keyword
    2. Import statements or use declarations
    3. Comments/strings mentioning the keyword

    Excludes DOC_FILES to avoid matching claims in docs themselves.

    Note: For multiple keywords, use check_keywords_in_code_batch() instead
    to avoid O(n*m*d*k) complexity (#1699).

    REQUIRES: base_dir is a Path to existing directory
    REQUIRES: keyword is non-empty string
    ENSURES: Returns True iff keyword found in code
    """
    result = check_keywords_in_code_batch(base_dir, {keyword.lower()})
    return keyword.lower() in result


class HeuristicResult(TypedDict):
    """Result from heuristic keyword drift detection."""

    doc_keywords: list[str]  # Keywords found in docs
    code_keywords: list[str]  # Keywords verified in code
    drift_keywords: list[str]  # In docs but not in code
    categories: dict[str, dict[str, Any]]  # Per-category breakdown


def check_heuristic_drift(
    base_dir: Path,
    keywords: dict[str, list[str]] | None = None,
) -> HeuristicResult:
    """Detect keyword drift between docs and code (#1518).

    REQUIRES: base_dir is a Path to existing directory
    REQUIRES: keywords is None or dict of category -> keyword list
    ENSURES: Returns HeuristicResult with doc/code/drift breakdown

    Args:
        base_dir: Repository root directory
        keywords: Keyword dict by category, or None for defaults

    Returns:
        HeuristicResult with doc/code/drift keyword lists
    """
    if keywords is None:
        keywords = DEFAULT_KEYWORDS

    # Flatten all keywords into a single set for matching
    all_keywords: set[str] = set()
    for category_keywords in keywords.values():
        all_keywords.update(kw.lower() for kw in category_keywords)

    # Find keywords mentioned in docs
    doc_words = extract_doc_keywords(base_dir)
    doc_keywords = doc_words & all_keywords

    # Check which doc keywords have code backing (batch O(n*m*d) vs O(n*m*d*k))
    code_keywords = check_keywords_in_code_batch(base_dir, doc_keywords)
    drift_keywords = doc_keywords - code_keywords

    # Build per-category breakdown
    categories: dict[str, dict[str, Any]] = {}
    for category, cat_keywords in keywords.items():
        cat_kw_lower = {kw.lower() for kw in cat_keywords}
        cat_doc = doc_keywords & cat_kw_lower
        cat_code = code_keywords & cat_kw_lower
        cat_drift = drift_keywords & cat_kw_lower
        categories[category] = {
            "in_docs": sorted(cat_doc),
            "verified": sorted(cat_code),
            "drift": sorted(cat_drift),
        }

    return {
        "doc_keywords": sorted(doc_keywords),
        "code_keywords": sorted(code_keywords),
        "drift_keywords": sorted(drift_keywords),
        "categories": categories,
    }


def check_doc_claims(base_dir: Path) -> DocClaimsResults:
    """Check all documentation claims in a repo.

    REQUIRES: base_dir is a Path to existing directory
    ENSURES: Returns DocClaimsResults with docs, totals, and drift list

    Returns dict with:
    - docs: list of doc results
    - total_claims: count
    - verified_claims: count
    - unverified_claims: count
    - drift: list of unverified claim details
    """
    results: DocClaimsResults = {
        "docs": [],
        "total_claims": 0,
        "verified_claims": 0,
        "unverified_claims": 0,
        "drift": [],
        "parse_errors": [],
    }

    for doc_name in DOC_FILES:
        doc_path = base_dir / doc_name
        doc_result: dict[str, Any] = {
            "file": doc_name,
            "exists": doc_path.exists(),
            "has_claims": False,
            "claims": [],
        }

        if doc_path.exists():
            try:
                content = doc_path.read_text(encoding="utf-8")
                frontmatter = parse_yaml_frontmatter(content)

                if frontmatter and "claims" in frontmatter:
                    doc_result["has_claims"] = True
                    claims = frontmatter["claims"]

                    for claim in claims:
                        claim_result = check_claim(base_dir, claim)
                        doc_result["claims"].append(claim_result)
                        results["total_claims"] += 1

                        if claim_result["verified"]:
                            results["verified_claims"] += 1
                        else:
                            results["unverified_claims"] += 1
                            results["drift"].append(
                                {
                                    "doc": doc_name,
                                    "claim": claim_result["name"],
                                    "type": claim_result["type"],
                                    "details": claim_result,
                                }
                            )
                elif _has_frontmatter_delimiters(content) and (
                    frontmatter is None or "claims" not in frontmatter
                ):
                    # Frontmatter delimiters present but parsing yielded no
                    # claims — likely malformed YAML (#2918).
                    doc_result["parse_error"] = (
                        "frontmatter delimiters found but no claims parsed"
                    )
                    results["parse_errors"].append({
                        "doc": doc_name,
                        "error": doc_result["parse_error"],
                    })

            except (OSError, UnicodeDecodeError) as e:
                doc_result["error"] = str(e)

        results["docs"].append(doc_result)

    return results


def main() -> int:
    """Entry point: verify documentation claims against codebase.

    REQUIRES: None (reads sys.argv)
    ENSURES: Returns 0 if no drift, 1 if drift found
    """
    parser = argparse.ArgumentParser(
        description="Verify machine-checkable documentation claims",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    check_doc_claims.py                      # Check all docs, human output
    check_doc_claims.py --json               # Machine-readable output
    check_doc_claims.py --path ~/some/repo   # Check specific repo

Claim format in YAML frontmatter:
---
claims:
  - type: backend
    name: z4
    code_path: src/backends/z4/
    status: active
  - type: feature
    name: BigInt
    code_marker: "// VISION: BigInt"
---
""",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=get_version("check_doc_claims.py"),
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output results as JSON",
    )
    parser.add_argument(
        "--path",
        type=Path,
        default=None,
        help="Base directory to scan (default: current working directory)",
    )
    parser.add_argument(
        "--help-short",
        "-H",
        action="store_true",
        help="Show short help",
    )
    parser.add_argument(
        "--heuristic",
        action="store_true",
        help="Enable heuristic keyword drift detection (#1518)",
    )
    parser.add_argument(
        "--keywords",
        type=Path,
        default=None,
        help="Custom keywords JSON file (with --heuristic)",
    )

    args = parser.parse_args()

    if args.help_short:
        print("check_doc_claims.py - Verify documentation claims against codebase")
        print("Usage: check_doc_claims.py [--json] [--path=DIR] [--heuristic]")
        return 0

    base_dir = args.path if args.path else Path.cwd()

    # Heuristic mode (#1518)
    if args.heuristic:
        keywords = load_keywords(args.keywords)
        heuristic_results = check_heuristic_drift(base_dir, keywords)

        if args.json:
            print(json.dumps(heuristic_results, indent=2))
        else:
            print("=== Heuristic Keyword Drift Detection ===\n")
            print(f"Keywords in docs: {len(heuristic_results['doc_keywords'])}")
            print(f"Verified in code: {len(heuristic_results['code_keywords'])}")
            print(f"Drift (no code):  {len(heuristic_results['drift_keywords'])}")

            if heuristic_results["drift_keywords"]:
                print("\n=== Drift Keywords ===")
                for kw in heuristic_results["drift_keywords"]:
                    print(f"  \u2717 {kw}")

            print("\n=== By Category ===")
            for category, data in heuristic_results["categories"].items():
                if data["in_docs"]:
                    print(f"\n{category.upper()}:")
                    for kw in data["in_docs"]:
                        icon = "\u2713" if kw in data["verified"] else "\u2717"
                        print(f"  [{icon}] {kw}")

        return 1 if heuristic_results["drift_keywords"] else 0

    # Standard claims mode
    results = check_doc_claims(base_dir)

    if args.json:
        print(json.dumps(results, indent=2))
    else:
        # Human-readable output
        print("=== Documentation Claims Verification ===\n")

        for doc in results["docs"]:
            if not doc["exists"]:
                print(f"{doc['file']}: not found")
                continue

            if not doc["has_claims"]:
                print(f"{doc['file']}: no claims defined")
                continue

            print(f"{doc['file']}: {len(doc['claims'])} claims")
            for claim in doc["claims"]:
                status = "PASS" if claim["verified"] else "FAIL"
                icon = "\u2713" if claim["verified"] else "\u2717"
                print(f"  [{icon}] {status} {claim['type']}/{claim['name']}")

                if not claim["verified"]:
                    if (
                        "code_path_verified" in claim
                        and not claim["code_path_verified"]
                    ):
                        print(f"      code_path not found: {claim['code_path']}")
                    if (
                        "code_marker_verified" in claim
                        and not claim["code_marker_verified"]
                    ):
                        print(f"      code_marker not found: {claim['code_marker']}")

            print()

        # Summary
        print("=== Summary ===")
        print(f"Total claims: {results['total_claims']}")
        print(f"Verified: {results['verified_claims']}")
        print(f"Unverified (drift): {results['unverified_claims']}")

        if results["drift"]:
            print("\n=== Drift Details ===")
            for d in results["drift"]:
                print(f"  {d['doc']}: {d['type']}/{d['claim']}")

    # Exit code: 0 if no drift, 1 if drift found
    return 1 if results["unverified_claims"] > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
