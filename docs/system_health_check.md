# System Health Check

Purpose: verify the SYSTEM is connected, not just that unit tests pass.

Run:
```bash
python3 scripts/system_health_check.py
```

## Contract (v1.0)

All `scripts/system_health_check.py` implementations across repos MUST follow this contract.

### Exit Codes (REQUIRED)

| Code | Meaning | Audit Effect |
|------|---------|--------------|
| 0 | All checks passed (or passed with warnings) | Transition allowed |
| 1 | One or more checks failed | Block transition |

**Key rule: Warnings do NOT block audits.** Health checks catch broken systems, not imperfect ones. Warnings are informational - displayed but not blocking.

**Rationale:**
- Looper needs binary pass/fail for workflow decisions
- Warning details go in stdout for human review
- Repos wanting 3-state (pass/warn/fail) use JSON output's `summary.status` field

### CLI Flags (REQUIRED)

| Flag | Description | Output |
|------|-------------|--------|
| (no args) | Run checks, human-readable output | stdout |
| `--json-output PATH` | Write JSON manifest to PATH | file + stdout |

**Note:** Auto-fix (`--fix`) belongs in pre-commit hooks, not health checks. Health checks are read-only observability.

### Timeout Budget (REQUIRED)

Health checks MUST complete within **60 seconds** (looper's subprocess timeout).

Repos with expensive checks should:
1. Use internal budget tracking (e.g., `TOTAL_BUDGET_SEC = 55`)
2. Skip expensive checks if time exhausted
3. Return warning (not failure) for time-skipped checks

### Implementation Notes

- Looper integration: `looper/runner.py` blocks audit transitions on non-zero exit
- Looper call site: `looper/context/system_context.py:131-174`
- Shared base module: `ai_template_scripts/health_check/` (base.py, cli.py)

## JSON Output

Generate a structured JSON manifest for automated consumption:

```bash
python3 scripts/system_health_check.py --json-output reports/health_manifest.json
```

The manifest includes schema version, git commit, timestamps, and per-check structured results. See [JSON Schema](#json-schema) below.

## What It Checks

- Code reachability: modules are imported by entry points
- Data usage: data files are referenced by code
- Pipeline execution: configured commands run
- Script smoke tests: critical CLI scripts respond to safe `--help`/`--version` invocations
- Report validity: reports do not contain placeholders
- Loader usage: loader functions are called by entry points

## Configuration

Edit the constants near the top of `scripts/system_health_check.py`:

- `ENTRYPOINT_GLOBS`, `ENTRYPOINT_EXCLUDE`
- `EXPLICIT_MODULE_ROOTS`
- `DATA_DIRS`, `DATA_FILE_EXTENSIONS`
- `REPORT_DIRS`, `REPORT_FILE_EXTENSIONS`, `REQUIRED_REPORT_FILES`
- `PIPELINE_COMMANDS`, `PIPELINE_TIMEOUT_SEC`
- `SCRIPT_SMOKE_COMMANDS`, `SCRIPT_SMOKE_TIMEOUT_SEC`, `REQUIRE_SCRIPT_SMOKE_CHECKS`
- `LOADER_FILE_GLOBS`, `LOADER_FUNCTION_PREFIX`

Start by configuring:
1. At least one entry point.
2. At least one pipeline command that exercises the main flow.
3. Script smoke commands that run `--help`/`--version` without side effects.
4. Required report artifacts (if you generate reports).

## Example Output (Pass)

```
============================================================
SYSTEM HEALTH CHECK - Is Everything Actually Working?
============================================================

## Code Reachability
Checking: Are all modules imported by entry points?

## Data File Usage
Checking: Are all data files loaded by code?

## Pipeline Execution
Checking: Do configured pipeline commands run?

## Report Validity
Checking: Do reports contain real data?

## Loader Function Usage
Checking: Are loader functions actually called?

============================================================
SUMMARY
============================================================

PASSED (3)
  [OK] All modules reachable from entry points.
  [OK] Reports checked with no placeholder markers.
  [OK] All loader functions referenced by entry points.

SKIPPED (2)
  [SKIP] No data directories found; skipping data usage check.
  [SKIP] No pipeline commands configured; skipping pipeline check.

============================================================
HEALTH CHECK PASSED
```

## Example Output (Fail)

```
============================================================
SYSTEM HEALTH CHECK - Is Everything Actually Working?
============================================================

## Code Reachability
Checking: Are all modules imported by entry points?

  - my_project.unused_module

============================================================
SUMMARY
============================================================

ERRORS (1)
  [ERROR] Orphan modules detected - built but never imported.

============================================================
HEALTH CHECK FAILED
The system has integration problems that need fixing.
```

## JSON Schema

When using `--json-output`, the manifest follows this structure:

```json
{
  "schema_version": "1.0",
  "generated_at": "2026-01-25T20:01:00Z",
  "git_commit": "abc123def456",
  "project": "my_project",
  "summary": {
    "status": "pass|warn|fail",
    "passed": 3,
    "warnings": 1,
    "errors": 0,
    "skipped": 2
  },
  "checks": {
    "code_reachability": {
      "status": "pass|fail|skip",
      "modules_total": 10,
      "reachable": 10,
      "orphans": [],
      "entry_points": ["scripts/main.py"]
    },
    "data_usage": {
      "status": "pass|warn|skip",
      "files_total": 5,
      "referenced": 4,
      "unreferenced": ["data/unused.csv"]
    },
    "pipeline_execution": {
      "status": "pass|fail|skip",
      "commands_total": 1,
      "passed": 1,
      "failed": []
    },
    "script_smoke_tests": {
      "status": "pass|warn|fail|skip",
      "commands_total": 5,
      "passed": 5,
      "failed": []
    },
    "report_validity": {
      "status": "pass|warn|fail|skip",
      "reports_total": 3,
      "placeholder_hits": [],
      "missing_required": []
    },
    "loader_usage": {
      "status": "pass|warn|skip",
      "loaders_total": 2,
      "referenced": 2,
      "unreferenced": []
    }
  }
}
```

**Usage:**
- QA gates can consume JSON programmatically
- Dashboards can aggregate health across repos
- Auditors can verify claims without parsing console output
- Regression detection via manifest comparison across commits
