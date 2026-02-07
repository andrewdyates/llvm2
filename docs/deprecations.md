# Deprecation Tracking

This document tracks deprecated items scheduled for removal.

## Scheduled for v2.0 Removal

### Functions

| Module | Function | Replacement | Issue |
|--------|----------|-------------|-------|
| `looper/sync.py` | `check_uncommitted_work_size()` | `warn_uncommitted_work()` | #1015 |
| `looper/sync.py` | `check_stale_staged_files()` | `get_staged_files()` + `warn_stale_staged_files()` or `enforce_no_stale_staged_files()` | #995 |

### Parameters

| Module | Function | Parameter | Replacement | Issue |
|--------|----------|-----------|-------------|-------|
| `looper/checkpoint/tool_call_log.py` | `ToolCallLog.__init__` | `result_truncate_chars` | `config.result_truncate_chars` | N/A (docstring only - no `warnings.warn`) |

### Modules

| Module | Replacement | Notes |
|--------|-------------|-------|
| `ai_template_scripts/pulse_monolith.py` | `pulse.py` | Legacy pulse implementation |
| `ai_template_scripts/crash_analysis.py` | `ai_template_scripts.crash_analysis` package | Backward compatibility shim |
| `ai_template_scripts/bg_task.py` | `ai_template_scripts.bg_task` package | Backward compatibility shim |

## TODOs Requiring Implementation

| File | Line | Description | Priority |
|------|------|-------------|----------|
| ~~`ai_template_scripts/timeout_classifier.py`~~ | ~~308~~ | ~~Parse command from enhanced log format~~ | ~~Medium~~ (Fixed in fdccd2d2) |

## Naming Convention Violations

| File | Function | Should Be | Standard |
|------|----------|-----------|----------|
| ~~`looper/config_validation.py:30`~~ | ~~`check_unknown_keys()`~~ | ~~`get_unknown_keys()`~~ | Fixed in #2467 |

---
*Last updated: 2026-02-07*
*Created from #2434*
