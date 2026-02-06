# Git Hooks

Copyright 2026 Dropbox, Inc.
Author: Andrew Yates <ayates@dropbox.com>

This document describes the git hooks provided by ai_template.

## Installation

**Hook installation methods:**

| Method | When | Installs |
|--------|------|----------|
| `init_from_template.sh` | New repos from template | All hooks |
| `looper.py` startup | AI session start | All hooks |
| `install_hooks.sh` | Manual installation | All hooks |
| Git wrapper auto-install | First `git commit` | commit-msg only |

**New repos:** Hooks are installed automatically by `init_from_template.sh`.

**Existing repos:** Run from your repository root after cloning:

```bash
./ai_template_scripts/install_hooks.sh
```

This installs the pre-commit framework and configures all hooks.

**Partial auto-install:** The git wrapper (`ai_template_scripts/bin/git`) automatically installs the commit-msg hook on first commit. This ensures role enforcement (Fixes #N restrictions) even in fresh clones. However, pre-commit and post-commit hooks require explicit installation.

**Manual sessions:** If you're working manually (not via looper), run `install_hooks.sh` to get the full hook suite:
- **pre-commit**: Copyright headers, test ignore detection, build gate
- **post-commit**: Claims/Unclaims label automation

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

## Local Issues

The hooks support local issue tracking via the `.issues/` directory. This enables offline development when GitHub API is unavailable.

### Local Issue Syntax

Local issues use `L`-prefixed IDs:

| Format | Example | Description |
|--------|---------|-------------|
| `#LN` | `#L42` | References local issue 42 |
| `Part of #LN` | `Part of #L1` | Links to local issue without closing |
| `Fixes #LN` | `Fixes #L1` | Closes local issue (Manager/User only) |
| `Claims #LN` | `Claims #L1` | Adds in-progress label |
| `Unclaims #LN` | `Unclaims #L1` | Removes in-progress label |
| `Reopens #LN` | `Reopens #L1` | Reopens closed local issue |

### Local Issue File Format

Local issues are stored in `.issues/L<N>.md` with YAML frontmatter:

```markdown
---
title: "Issue title"
state: open
labels: ["P2", "feature"]
created: 2026-01-15T10:00:00Z
---

Issue body content here.

## Comments
- 2026-01-15: Initial filing
```

### AIT_LOCAL_MODE Environment Variable

| Value | Behavior |
|-------|----------|
| (unset) | Normal mode - uses GitHub API for all issue operations |
| `full` | Full local mode - skips ALL GitHub API calls, validates only local issues |

In full local mode:
- commit-msg hook skips GitHub issue validation
- post-commit hook skips GitHub label operations
- Local issues (#LN) work normally
- GitHub issues (#N) are assumed to exist (no validation)

**Usage:**

```bash
# Enable full local mode for offline work
export AIT_LOCAL_MODE=full

# Work on local issues without internet
git commit -m "[W]123: Part of #L1 - Add feature"

# Disable when back online
unset AIT_LOCAL_MODE
```

### Local Issue Functions

The hooks provide these internal functions for local issue handling:

| Function | Hook | Purpose |
|----------|------|---------|
| `is_local_issue()` | Both | Check if ID matches `L[0-9]+` pattern |
| `local_issue_field()` | commit-msg | Read field from `.issues/L<N>.md` (state, title, body, labels) |
| `local_issue_edit()` | post-commit | Add/remove labels in local issue |
| `local_issue_reopen()` | post-commit | Set state to open |
| `local_issue_close()` | post-commit | Set state to closed |

### Syncing Local Issues to GitHub

Local issues are intended for temporary offline work. Sync to GitHub is tracked in #1834 (write-through planned).

Current workflow:
1. Work offline using local issues (`#LN`)
2. When online, create corresponding GitHub issues
3. Update commits to reference GitHub issue numbers
4. Delete local issue files after sync
