#!/usr/bin/env python3
"""check_security.py - PreToolUse security hook for Edit/Write.

Blocks writes containing secret patterns or injection risks.

Exit codes:
    0 - Allow tool to proceed
    2 - Block tool execution

Copyright 2026 Dropbox, Inc.
Author: Andrew Yates <ayates@dropbox.com>

Part of #1847.
"""

import json
import re
import sys
from typing import NamedTuple


class Finding(NamedTuple):
    """Security finding with pattern and description."""

    pattern: str
    description: str
    priority: str  # HIGH, MEDIUM, LOW


# Patterns to check - ordered by priority
SECRET_PATTERNS: list[Finding] = [
    # HIGH priority - definitely secrets
    Finding(
        r"-----BEGIN\s+(RSA|DSA|EC|OPENSSH|PGP)?\s*PRIVATE\s+KEY-----",
        "Private key detected",
        "HIGH",
    ),
    Finding(
        r"AKIA[0-9A-Z]{16}",
        "AWS access key ID detected",
        "HIGH",
    ),
    Finding(
        r"(?i)aws[_-]?secret[_-]?access[_-]?key\s*[=:]\s*['\"][A-Za-z0-9/+=]{40}['\"]",
        "AWS secret access key detected",
        "HIGH",
    ),
    Finding(
        r"ghp_[A-Za-z0-9]{36}",
        "GitHub personal access token detected",
        "HIGH",
    ),
    Finding(
        r"gho_[A-Za-z0-9]{36}",
        "GitHub OAuth token detected",
        "HIGH",
    ),
    # MEDIUM priority - likely secrets
    Finding(
        r"(?i)api[_-]?key\s*[=:]\s*['\"][A-Za-z0-9+/=_-]{20,}['\"]",
        "Potential API key detected",
        "MEDIUM",
    ),
    Finding(
        r"(?i)password\s*[=:]\s*['\"][^'\"]{8,}['\"]",
        "Hardcoded password detected",
        "MEDIUM",
    ),
    Finding(
        r"(?i)Bearer\s+[A-Za-z0-9._-]{20,}",
        "Bearer token detected",
        "MEDIUM",
    ),
    Finding(
        r"(?i)secret[_-]?key\s*[=:]\s*['\"][A-Za-z0-9+/=_-]{16,}['\"]",
        "Secret key detected",
        "MEDIUM",
    ),
]

INJECTION_PATTERNS: list[Finding] = [
    Finding(
        r"(?i);\s*(DROP|DELETE|TRUNCATE|UPDATE)\s+(TABLE|DATABASE|FROM)",
        "SQL injection pattern detected",
        "MEDIUM",
    ),
    Finding(
        r"\$\(\s*[^)]+\s*\)|`[^`]+`",
        "Command injection pattern (backticks or $())",
        "LOW",
    ),
]

XSS_PATTERNS: list[Finding] = [
    Finding(
        r"<script[^>]*>",
        "Script tag detected - potential XSS",
        "MEDIUM",
    ),
    Finding(
        r"(?i)on(click|dblclick|load|error|mouseover|mouseout|mousedown|mouseup|"
        r"mouseenter|mouseleave|submit|focus|blur|focusin|focusout|change|input|"
        r"keyup|keydown|keypress|contextmenu|drag|dragstart|dragend|dragover|"
        r"drop|paste|copy|cut|scroll|touchstart|touchend|touchmove)\s*=",
        "Event handler attribute detected - potential XSS",
        "MEDIUM",
    ),
    Finding(
        r"(?i)javascript\s*:",
        "JavaScript URI detected - potential XSS",
        "MEDIUM",
    ),
]

# Allowlist patterns - these are false positives
ALLOWLIST_PATTERNS = [
    r"test[_-]?api[_-]?key",  # Test keys
    r"example[_-]?password",  # Example passwords
    r"fake[_-]?(secret|key|token)",  # Fake credentials
    r"your[_-]?(api[_-]?key|password|token)",  # Placeholder text
    r"<your[_-]",  # Template placeholders
    r"PLACEHOLDER",  # Explicit placeholders
    r"xxx+",  # Redacted values
    r"TODO.*key",  # TODO comments about keys
]


def is_allowlisted(content: str, match_start: int, match_end: int) -> bool:
    """Check if the match is in an allowlisted context.

    Contracts:
        REQUIRES: 0 <= match_start <= match_end <= len(content)
        ENSURES: Returns True if any allowlist pattern matches the context
        ENSURES: Returns False if no allowlist patterns match
    """
    # Get context around match (50 chars before and after)
    context_start = max(0, match_start - 50)
    context_end = min(len(content), match_end + 50)
    context = content[context_start:context_end].lower()

    for allowlist in ALLOWLIST_PATTERNS:
        if re.search(allowlist, context, re.IGNORECASE):
            return True
    return False


def is_test_file(file_path: str) -> bool:
    """Check if the file is a test file (more lenient with patterns).

    Contracts:
        REQUIRES: file_path is a string (may be empty)
        ENSURES: Returns True for paths containing test indicators
        ENSURES: Returns False for production code paths
    """
    test_indicators = [
        "/test",
        "/tests/",
        "_test.py",
        "_test.rs",
        ".test.ts",
        ".test.js",
        "/fixtures/",
        "/mock/",
        "/mocks/",
    ]
    return any(ind in file_path.lower() for ind in test_indicators)


def check_secrets(content: str, file_path: str) -> list[tuple[str, str]]:
    """Check for hardcoded secrets.

    Contracts:
        REQUIRES: content is a string to scan
        REQUIRES: file_path is used to determine test file leniency
        ENSURES: Returns list of (description, priority) tuples
        ENSURES: Only reports first match per pattern type
        ENSURES: Skips MEDIUM/LOW findings in test files

    Returns:
        List of (finding, priority) tuples.
    """
    findings: list[tuple[str, str]] = []

    # Be more lenient in test files
    is_test = is_test_file(file_path)

    for pattern in SECRET_PATTERNS:
        for match in re.finditer(pattern.pattern, content):
            # Skip if allowlisted
            if is_allowlisted(content, match.start(), match.end()):
                continue

            # Skip MEDIUM/LOW in test files
            if is_test and pattern.priority != "HIGH":
                continue

            findings.append((pattern.description, pattern.priority))
            # Only report first match per pattern
            break

    return findings


def is_shell_file(file_path: str) -> bool:
    """Check if the file is a shell script (lenient with command patterns).

    Contracts:
        REQUIRES: file_path is a string (may be empty)
        ENSURES: Returns True for shell script extensions
        ENSURES: Returns False for other file types
    """
    shell_indicators = [".sh", ".bash", ".zsh", "/bin/"]
    return any(ind in file_path.lower() for ind in shell_indicators)


def is_markdown_file(file_path: str) -> bool:
    """Check if the file is a markdown file (lenient with backtick patterns).

    Contracts:
        REQUIRES: file_path is a string (may be empty)
        ENSURES: Returns True for markdown file extensions
        ENSURES: Returns False for other file types
    """
    return file_path.lower().endswith((".md", ".markdown", ".mdx"))


def is_doc_file(file_path: str) -> bool:
    """Check if the file uses documentation with backticks (lenient with backtick patterns).

    Languages like Rust (/// and //! doc comments with `code`), Python (docstrings with
    `code`), and TypeScript (.d.ts with JSDoc) use backticks for inline code examples.

    Config files (TOML, JSON) and documentation formats (RST) also use backticks legitimately.

    Contracts:
        REQUIRES: file_path is a string (may be empty)
        ENSURES: Returns True for files that use backticks in documentation
        ENSURES: Returns False for other file types

    Part of #2313.
    """
    # These extensions commonly have documentation with inline code backticks
    doc_extensions = (
        ".rs",  # Rust - rustdoc uses ```rust and `code`
        ".py",  # Python - docstrings use `code`
        ".ts",  # TypeScript - JSDoc uses `code`
        ".tsx",  # TypeScript JSX
        ".js",  # JavaScript - JSDoc uses `code`
        ".jsx",  # JavaScript JSX
        ".go",  # Go - godoc uses `code`
        ".d.ts",  # TypeScript declaration files
        ".toml",  # TOML - string values may contain `code` references
        ".json",  # JSON - string values may contain `code` references
        ".rst",  # reStructuredText - uses ``code`` for inline code
    )
    return file_path.lower().endswith(doc_extensions)


def is_template_file(file_path: str) -> bool:
    """Check if the file is an HTML/JSX/template file (lenient with XSS patterns).

    These files legitimately contain script tags, event handlers, and JavaScript URIs.

    Contracts:
        REQUIRES: file_path is a string (may be empty)
        ENSURES: Returns True for HTML, JSX, TSX, Vue, Svelte, and template file extensions
        ENSURES: Returns False for other file types

    Part of #2078.
    """
    template_extensions = (
        ".html",
        ".htm",
        ".jsx",
        ".tsx",
        ".vue",
        ".svelte",
        ".hbs",
        ".ejs",
        ".jinja",
        ".jinja2",
        ".j2",
        ".njk",
        ".mustache",
        ".erb",  # Ruby ERB templates
        ".phtml",  # PHP templates
        ".twig",  # Twig templates (PHP)
    )
    return file_path.lower().endswith(template_extensions)


def is_subprocess_module(file_path: str) -> bool:
    """Check if the file is a module that legitimately uses subprocess commands.

    These modules build shell command strings for run_cmd() and similar subprocess
    utilities. The $() pattern is a legitimate bash command substitution, not injection.

    Contracts:
        REQUIRES: file_path is a string (may be empty)
        ENSURES: Returns True for known subprocess-using modules
        ENSURES: Returns False for other file types

    Part of #1954.
    """
    subprocess_modules = [
        "/pulse/",  # pulse.py modules use run_cmd with bash -c
        "/subprocess_utils.py",  # Subprocess utilities
        "/cargo_wrapper/",  # Cargo wrapper uses subprocess
        "/hooks/",  # Security hooks contain regex patterns with shell syntax
        "/looper/",  # Looper modules use markdown backticks for prompt formatting
    ]
    path_lower = file_path.lower()
    return any(mod in path_lower for mod in subprocess_modules)


def check_injection(content: str, file_path: str) -> list[tuple[str, str]]:
    """Check for injection patterns.

    Contracts:
        REQUIRES: content is a string to scan
        REQUIRES: file_path is used to determine test/shell/markdown/doc/subprocess leniency
        ENSURES: Returns empty list for test files
        ENSURES: Skips command injection check for shell, markdown, doc, and subprocess files
        ENSURES: Returns list of (description, priority) tuples

    Part of #2313: Added is_doc_file() check to allow backticks in Rust/Python/JS docs.

    Returns:
        List of (finding, priority) tuples.
    """
    # Skip injection checks in test files
    if is_test_file(file_path):
        return []

    # Skip command injection pattern in shell scripts (legitimate use),
    # markdown files (backticks are normal for inline code),
    # doc files (Rust/Python/JS use backticks for inline code in docs),
    # and subprocess modules (bash command substitution is legitimate)
    is_shell = is_shell_file(file_path)
    is_markdown = is_markdown_file(file_path)
    is_doc = is_doc_file(file_path)
    is_subprocess = is_subprocess_module(file_path)

    findings: list[tuple[str, str]] = []

    for pattern in INJECTION_PATTERNS:
        # Skip command injection pattern for shell, markdown, doc, and subprocess files
        if (
            is_shell or is_markdown or is_doc or is_subprocess
        ) and "Command injection" in pattern.description:
            continue
        if re.search(pattern.pattern, content):
            findings.append((pattern.description, pattern.priority))

    return findings


def check_xss(content: str, file_path: str) -> list[tuple[str, str]]:
    """Check for XSS patterns.

    Contracts:
        REQUIRES: content is a string to scan
        REQUIRES: file_path is used to determine test/template file leniency
        ENSURES: Returns empty list for test files
        ENSURES: Returns empty list for template files (HTML, JSX, etc.)
        ENSURES: Returns list of (description, priority) tuples

    Part of #2078.

    Returns:
        List of (finding, priority) tuples.
    """
    # Skip XSS checks in test files
    if is_test_file(file_path):
        return []

    # Skip XSS checks in template files (HTML, JSX, etc.) where these are normal
    if is_template_file(file_path):
        return []

    findings: list[tuple[str, str]] = []

    for pattern in XSS_PATTERNS:
        for match in re.finditer(pattern.pattern, content):
            # Skip if allowlisted
            if is_allowlisted(content, match.start(), match.end()):
                continue
            findings.append((pattern.description, pattern.priority))
            # Only report first match per pattern
            break

    return findings


def main() -> int:
    """Main entry point.

    Contracts:
        REQUIRES: stdin contains JSON with tool_input.new_string or tool_input.content
        ENSURES: Returns 0 (allow) when no issues found or on parse errors
        ENSURES: Returns 2 (block) when security issues detected
        ENSURES: Outputs findings to stderr when blocking
    """
    try:
        data = json.load(sys.stdin)
    except json.JSONDecodeError:
        # Invalid JSON input - allow to proceed (don't block on hook errors)
        return 0

    tool_input = data.get("tool_input", {})

    # Get content being written - handle Edit (new_string) and Write (content)
    # Use "or empty string" to handle None values from JSON
    content = tool_input.get("new_string") or ""
    if not content:
        content = tool_input.get("content") or ""

    if not content:
        return 0  # No content to check

    # Get file path for context (handle None from JSON)
    file_path = tool_input.get("file_path") or ""

    # Run checks
    findings = (
        check_secrets(content, file_path)
        + check_injection(content, file_path)
        + check_xss(content, file_path)
    )

    if findings:
        # Group by priority
        high = [f for f, p in findings if p == "HIGH"]
        medium = [f for f, p in findings if p == "MEDIUM"]
        low = [f for f, p in findings if p == "LOW"]

        print("BLOCK: Security check failed:", file=sys.stderr)

        if high:
            print("  HIGH:", file=sys.stderr)
            for finding in high:
                print(f"    - {finding}", file=sys.stderr)

        if medium:
            print("  MEDIUM:", file=sys.stderr)
            for finding in medium:
                print(f"    - {finding}", file=sys.stderr)

        if low:
            print("  LOW:", file=sys.stderr)
            for finding in low:
                print(f"    - {finding}", file=sys.stderr)

        print("", file=sys.stderr)
        print("Verify this is not a real credential.", file=sys.stderr)
        print(
            "If intentional, add to test file or use placeholder value.",
            file=sys.stderr,
        )

        return 2  # Block

    return 0  # Allow


if __name__ == "__main__":
    sys.exit(main())
