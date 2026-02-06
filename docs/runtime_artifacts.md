<!-- Andrew Yates <ayates@dropbox.com> -->

# Runtime Artifacts

This repo generates runtime files during looper runs, benchmarking, and tooling. They
are gitignored and safe to delete when the system is not running.

## Looper State

- `.rotation_state.json` - persisted phase state
- `.rotation_state.lock` - lock file for rotation updates
- `.pid_*` - per-role looper PID files
- `.{role}_status.json` / `.{role}_{id}_status.json` - per-role status snapshots (dot-prefixed)
- `.iteration_*` and `.commit_tag_*` - iteration metadata
- `.looper_checkpoint_*.json` - crash recovery checkpoints
- `.looper_config.json` - project-specific config overrides (see docs/looper.md)
- `.background_tasks/` - background task queues
- `worker_logs/` - iteration logs and crash reports
- `STOP` / `STOP_<ROLE>` - manual stop signal files

## Hooks and Pulse

- `.claude_read_timestamps` - hook freshness tracking
- `metrics/` - pulse metrics JSON (see schema below)
- `.flags/` - runtime flags
- `logs/` - runtime logs

### Pulse Metrics Schema (v1.0.0)

Metrics JSON files include versioning for downstream consumers:

```json
{
  "schema_version": "1.0.0",   // Increment on breaking changes
  "collector": "pulse",        // Identifies collection source
  "timestamp": "...",          // ISO 8601 collection time
  "repo": "...",               // Repository name
  // ... other fields
}
```

**Key fields:**
- `loc` - Lines of code by language
- `tests` - Test count and recent results
- `proofs` - Proof coverage (Kani, TLA+, Lean, SMT)
- `issues` - Issue counts and velocity
- `system` - Memory, disk usage
- `quality` - Code quality metrics

**Retention:** Files in `metrics/` are retained for 7 days, then archived to
`metrics/archive/`. See `METRICS_RETENTION_DAYS` in `ai_template_scripts/pulse/constants.py`.

## Issue Mirror

- `.issues/` - local GitHub issue export for offline search

## Tool Caches and Test Output

- `__pycache__/`, `.mypy_cache/`, `.ruff_cache/`, `.pytest_cache/`, `.hypothesis/`
- `.coverage`

## Benchmarks and Evals

- `benchmarks/**/results/`, `benchmarks/**/logs/`, `benchmarks/**/data/`
- `evals/**/results/`, `evals/**/logs/`

## Cleanup

Delete the paths above when you are done running the system. To remove all gitignored
artifacts at once, run `git clean -fdX` (destructive; removes all ignored files).
