# Git Hooks

Copyright 2026 Dropbox, Inc.
Author: Andrew Yates <ayates@dropbox.com>

This document describes the git hooks provided by ai_template.

## Installation

**New repos:** Hooks are installed automatically by `init_from_template.sh`.

**Existing repos:** Run from your repository root after cloning:

```bash
./ai_template_scripts/install_hooks.sh
```

This installs the pre-commit framework and configures all hooks.

## Available Hooks

### pre-commit

**File:** `ai_template_scripts/pre-commit-hook.sh`

Runs before each commit. Performs:

1. **New file stash safety** - Warns/blocks (AI mode) when staged new files + unstaged changes could trigger pre-commit stash loss
2. **Copyright headers** - Auto-adds missing copyright headers to source files
3. **Author metadata** - Warns about missing author in Cargo.toml/pyproject.toml
4. **Verification headers** - Auto-adds placeholders for new docs in designs/, reports/, postmortems/
5. **Test ignore detection** - BLOCKS commits adding test ignores (#341)
6. **Zone enforcement** - Advisory (or strict) zone checks for multi-worker mode
7. **AppleScript syntax** - Validates `osascript` usage in staged `.sh` files (macOS only)
8. **Multi-worker safety** - Warns on stale trees, regressions, and untracked staged files (multi-worker only)
9. **TLA+ property tests** - Runs `scripts/run_tla_property_tests.py --staged` when present
10. **Build gate** - AI-only build checks for staged code (cargo check, npm run build, py_compile)
11. **Linting** - Runs ruff (Python) and shellcheck (bash) via pre-commit framework

**Testing note:** `tests/test_pre_commit_integration.py` runs `pre-commit run --all-files`
to ensure the hook chain stays green on a clean repo. If it fails, treat it as a real
lint/type failure rather than an infrastructure issue.
Source: `tests/test_pre_commit_integration.py:72`.

### commit-msg

**File:** `ai_template_scripts/commit-msg-hook.sh`

Validates commit message format. Enforces:

1. Role tag format: `[U]`, `[W]`, `[P]`, `[R]`, `[M]`
2. `Fixes #N` restrictions (Manager/User only)
3. Auto-fixes common format issues

### pre-push

**File:** `ai_template_scripts/pre-push-hook.sh`

Validates benchmark claims in commit messages before push.

**Purpose:** Catches invalid benchmark claims (e.g., "39/55 CHC") before they reach the remote branch.

**Behavior:**
- Default: Warns on INVALID claims but allows push
- Strict mode: Blocks push on INVALID claims

**Environment Variables:**

| Variable | Default | Description |
|----------|---------|-------------|
| `CLAIM_VALIDATION_STRICT` | unset | Set to `1` to block push on INVALID claims |
| `CLAIM_VALIDATION_SKIP` | unset | Set to `1` to skip validation entirely |
| `CLAIM_VALIDATION_ROOT` | `evals/results` | Override results directory |

**Examples:**

```bash
# Normal push with warnings
git push origin main

# Strict mode - block on invalid claims
CLAIM_VALIDATION_STRICT=1 git push origin main

# Skip validation temporarily
CLAIM_VALIDATION_SKIP=1 git push origin main
```

**Claim validation statuses:**
- `VALID` - Claim matches results file at claimed commit
- `INVALID` - Claim does not match results (score mismatch or invalid format)
- `UNKNOWN` - No results file found for claimed eval/commit

## Project-Specific Extensions

Add custom hooks to `.pre-commit-local.d/`. Files in this directory:
- Are NOT synced from ai_template
- Must be executable shell scripts (`*.sh`)
- Run after standard pre-commit checks
- Return non-zero to block commit

Example:

```bash
# .pre-commit-local.d/check-data.sh
#!/usr/bin/env bash
# Project-specific data validation
python3 scripts/validate_data.py
```

## Manual Hook Execution

```bash
# Run all pre-commit hooks on staged files
pre-commit run

# Run all hooks on all files
pre-commit run --all-files

# Run specific hook
pre-commit run ai-template-custom

# Update hook versions
pre-commit autoupdate
```

## Bypassing Hooks

```bash
# Skip pre-commit hooks for one commit
git commit --no-verify

# Skip pre-push hook
CLAIM_VALIDATION_SKIP=1 git push
```

Use sparingly - hooks exist to catch problems before they propagate.
