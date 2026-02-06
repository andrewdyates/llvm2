# Template Customization Guide

Copyright 2026 Dropbox, Inc.
Author: Andrew Yates <ayates@dropbox.com>

This guide documents how projects can customize ai_template behavior without
modifying synced files. All customization points below are project-specific
and preserved across template syncs.

## Configuration Files

| File | Purpose | Synced? |
|------|---------|---------|
| `.looper_config.json` | Looper behavior, zones, model selection | No |
| `pulse.toml` | Metrics thresholds | No |
| `.pulse_ignore` | Orphan test exclusions | No |
| `cargo_wrapper.toml` | Cargo timeouts and limits | No |
| `.ai_template_features` | Optional feature opt-in | No |
| `.ai_template_skip` | Skip files during sync | No |

### .looper_config.json

Controls looper behavior. Role-specific sections merge on top of defaults.

```json
{
  "restart_delay": 60,
  "error_delay": 300,
  "silence_timeout": 600,
  "iteration_timeout": 2700,
  "local_mode": false,
  "timeouts": {
    "git_default": 5,
    "gh_list": 15,
    "gh_view": 10,
    "health_check": 60
  },
  "claude_model": "sonnet",
  "zones": {
    "worker_1": ["looper/**", "tests/test_looper*.py"],
    "worker_2": ["ai_template_scripts/**"]
  }
}
```

**Key sections:**

| Section | Purpose |
|---------|---------|
| `restart_delay` | Seconds between successful iterations (default: 60) |
| `error_delay` | Seconds after error/crash (default: 300) |
| `iteration_timeout` | Per-iteration timeout in seconds |
| `timeouts.*` | Subprocess timeouts for git/gh/health commands |
| `local_mode` | Disable GitHub API calls |
| `zones.*` | Multi-worker file locking patterns |
| `model_routing` | Per-role AI model selection |
| `sync.strategy` | Multi-machine sync: "rebase" or "merge" |

**Full reference:** See `docs/looper.md` "Configuration Reference" section.

### pulse.toml

Customize metrics thresholds. Searched in order: `pulse.toml`, `ai_template_scripts/pulse.toml`, `.pulse.toml`.

```toml
[thresholds]
max_file_lines = 1000
max_complexity = 15
memory_warning_percent = 80
memory_critical_percent = 90
disk_warning_percent = 80
disk_critical_percent = 90
stale_issue_days = 7
long_running_process_minutes = 120

[large_files]
exclude_patterns = [
    "tests/",
    "benchmarks/",
]

[runtime]
skip_orphaned_tests = true
```

**Full reference:** See `docs/pulse.md` "Configuration" section.

### .pulse_ignore

Exclude paths from orphan test detection. One pattern per line.

```text
# Project-specific exclusions
**/benchmarks/**      # VNN-COMP integration tests
**/integration_tests/**
**/fixtures/**        # Test data directories
```

**Syntax:**
- `#` starts a comment (inline comments supported)
- Empty lines are ignored
- Use `**/` for recursive matching

### cargo_wrapper.toml

Configure cargo build/test timeouts and concurrency. Also accepts `.cargo_wrapper.toml`.

```toml
[timeouts]
build_sec = 3600   # 1 hour build timeout
test_sec = 600     # 10 minute test timeout
kani_sec = 1800    # 30 minute Kani timeout

[limits]
max_concurrent_cargo = 1  # Serialize all cargo commands

[memory_watchdog]
warn_pct = 75
critical_pct = 85
kill_patterns = ["z3", "cbmc"]

[auto_bump]
repos = ["https://github.com/dropbox-ai-prototypes/z4"]
```

**Sections:**

| Section | Purpose |
|---------|---------|
| `[timeouts]` | Per-command timeout overrides |
| `[limits]` | Concurrency limits |
| `[memory_watchdog]` | Memory pressure thresholds |
| `[auto_bump]` | Repos to auto-bump on git pull |

### .ai_template_features

Opt-in to optional features from `ai_template_scripts/optional/`. One feature per line.

```text
kani-safe
```

**Available features:** Check `ai_template_scripts/optional/` for directory names.

### .ai_template_skip

Skip specific files during template sync (target-side). Prevents template from
overwriting project-specific customizations.

```text
ruff.toml              # Keep project-specific linting config
.claude/roles/custom.md  # Keep custom roles
```

**Syntax:**
- One path per line (relative to repo root)
- Comments with `#`
- Glob patterns supported: `*.md`, `.claude/roles/*.md`

## Extension Points

### Custom Pre-commit Hooks

Add project-specific checks in `.pre-commit-local.d/`. Scripts must be executable.

```bash
#!/usr/bin/env bash
# .pre-commit-local.d/check-data.sh

# Validate project-specific data files
python3 scripts/validate_data.py

# Non-zero exit blocks commit
exit $?
```

**Behavior:**
- Runs after standard pre-commit checks
- Return non-zero to block the commit
- Not synced from ai_template

### Project Health Checks

Create `scripts/system_health_check.py` for project-specific health validation.
Called by looper before each iteration.

```python
#!/usr/bin/env python3
"""Project-specific health checks."""

import sys

def check_custom_requirements():
    """Check project-specific requirements."""
    # Your validation logic here
    return True

if __name__ == "__main__":
    if not check_custom_requirements():
        print("FAIL: Custom requirements not met", file=sys.stderr)
        sys.exit(1)
    print("OK: All checks passed")
```

### Role Extensions in CLAUDE.md

Add project-specific role guidance in the "Project-Specific Configuration" section of CLAUDE.md.
This section is not synced from template.

```markdown
## Project-Specific Configuration

**Primary languages:** Python, Rust

### Worker-Specific Notes

- Always run `make lint` before committing
- Integration tests require Docker

### Prover-Specific Notes

- Kani proofs are in `proofs/` directory
- Use `--tests` flag for property tests
```

### Local Mode Touch File

Create `.local_mode` to disable GitHub API calls without environment variables.

```bash
touch .local_mode
```

**Priority:** Environment (`AIT_LOCAL_MODE`) > Touch file > Config (`local_mode` in .looper_config.json)

### Local Issue Storage

With full local mode (`AIT_LOCAL_MODE=full`), issues are stored in `.issues/`:

```yaml
# .issues/L1.md
---
id: L1
title: Implement feature X
labels: [P2, feature]
state: open
---
Feature description and details here.
```

**See:** `docs/local-development.md` for complete local issue workflow.

## What NOT to Customize

These files are synced from ai_template and will be overwritten:

| Path | Reason |
|------|--------|
| `.claude/rules/*.md` | Shared AI rules |
| `.claude/roles/*.md` | Shared role definitions |
| `ai_template_scripts/` | Shared tooling |
| `looper.py`, `looper/` | Core looper (use config instead) |
| `.pre-commit-config.yaml` | Pre-commit framework config |

**Workaround:** Use `.ai_template_skip` to preserve specific files, but note this
may cause drift from template improvements.

## Customization Decision Tree

```
Need to change behavior?
│
├─ Thresholds/limits → pulse.toml or cargo_wrapper.toml
│
├─ Looper behavior → .looper_config.json
│
├─ Add validation → .pre-commit-local.d/ or scripts/system_health_check.py
│
├─ Exclude from metrics → .pulse_ignore
│
├─ Keep file during sync → .ai_template_skip
│
├─ Enable optional feature → .ai_template_features
│
└─ Role-specific guidance → CLAUDE.md Project-Specific section
```

## Related Documentation

- `docs/looper.md` - Full looper configuration reference
- `docs/pulse.md` - Metrics system and thresholds
- `docs/hooks.md` - Git hook system
- `docs/wrappers.md` - Command wrappers (cargo, gh)
- `docs/local-development.md` - Offline/local mode workflow
