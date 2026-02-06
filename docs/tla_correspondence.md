<!-- Verified: 4ca5435 | 2026-01-30T22:43:34Z | [W]5 -->

# TLA+ Implementation Correspondence

Author: Andrew Yates <ayates@dropbox.com>

This document maps TLA+ specifications to Python implementations, enabling
traceability between formal models and code.

## Cargo Lock Protocol

**TLA+ Specs:**
- `tla/cargo_lock.tla` - Base lock protocol
- `tla/cargo_lock_with_toctou.tla` - TOCTOU race condition model

**Python Implementation:** `ai_template_scripts/cargo_wrapper.py`

### Actions

| TLA+ Action | Python Function | Line | Notes |
|------------|-----------------|------|-------|
| `TryAcquire(p)` | `acquire_lock()` | 680 | Uses O_CREAT\|O_EXCL for atomic creation |
| `FailAcquire(p)` | `acquire_lock()` return False | 723 | FileExistsError caught |
| `Release(p)` | `release_lock()` | 732 | Verifies PID ownership before delete |
| `Die(p)` | (external) | - | Process crash/exit - not in Python |
| `ForceReleaseStale(p)` | `force_release_stale_lock()` | 762 | Atomic rename then delete |
| `Retry(p)` | caller loop | - | Retry logic in `try_with_lock()` |
| `Tick` | (external) | - | Wall clock advance |
| `Respawn(p)` | (external) | - | OS process restart with PID reuse |

### TOCTOU-Specific Actions (cargo_lock_with_toctou.tla)

| TLA+ Action | Python Function | Line | Notes |
|------------|-----------------|------|-------|
| `BeginStaleCheck(p)` | `is_lock_stale()` read lock file | 610-616 | Read PID from lock file |
| `FinishStaleCheckPositive(p)` | `is_lock_stale()` return True | 621-644 | Lock appears stale |
| `FinishStaleCheckNegative(p)` | `is_lock_stale()` return False | 642 | Lock holder verified alive |
| `AbortForceRelease(p)` | `force_release_stale_lock()` re-check | 777-778 | TOCTOU mitigation: re-verify before release |
| `RespawnFast(p)` | (external) | - | Fast PID reuse within 2s tolerance |

### Invariants

| TLA+ Invariant | Verified By | Notes |
|---------------|-------------|-------|
| `MutualExclusion` | O_CREAT\|O_EXCL atomicity | At most one holder |
| `LockConsistency` | `acquire_lock()` writes PID | Lock file contains valid PID |
| `HolderMatchesLock` | `release_lock()` PID check | Only owner can release |
| `NoPhantomHolding` | `_lock_held` global | Holding implies lock file exists |
| `NoIncorrectForceRelease` | `is_lock_stale()` re-check | TOCTOU mitigation |

### External Events

| Event | Python Representation | TLA+ Action |
|-------|----------------------|-------------|
| Process death | Process exits/crashes | `Die(p)` |
| PID reuse | OS reuses PID for new process | `Respawn(p)`, `RespawnFast(p)` |
| Time passage | `datetime.now()` calls | `Tick` |

## Iteration Tag Protocol

**TLA+ Specs:**
- `tla/iteration_tags.tla` - Base iteration assignment
- `tla/iteration_tags_with_commit_failure.tla` - Commit failure model

**Python Implementation:** `looper/runner_git.py`

### Actions

| TLA+ Action | Python Function | Line | Notes |
|------------|-----------------|------|-------|
| `TryAcquireLock(s)` | `_acquire_lock_with_timeout()` success | 92 | fcntl.flock() succeeds |
| `FailAcquireLock(s)` | `_acquire_lock_with_timeout()` BlockingIOError | 108 | Lock held, retry loop |
| `RetryAcquireLock(s)` | `_acquire_lock_with_timeout()` retry success | 103-107 | Loop until acquired |
| `TimeoutAcquiring(s)` | `_acquire_lock_with_timeout()` timeout | 114 | Returns False after 30s |
| `ComputeAndRelease(s)` | `get_git_iteration()` | 116 | Compute max+1, write, release |

### Commit Failure Actions (iteration_tags_with_commit_failure.tla)

| TLA+ Action | Python Function | Line | Notes |
|------------|-----------------|------|-------|
| `CommitToGitSuccess(s)` | (external) | - | Git commit succeeds |
| `CommitToGitFail(s)` | (external) | - | Pre-commit hook or network failure |
| `CommitExhausted(s)` | (external) | - | Max retries exceeded |

### Invariants

| TLA+ Invariant | Verified By | Notes |
|---------------|-------------|-------|
| `LockMutex` | fcntl.LOCK_EX | At most one lock holder |
| `UniqueIterations` | `max(file, git) + 1` | Each iteration unique |
| `HolderConsistency` | Lock acquired before compute | Only holder reads/writes |
| `ValidAssignments` | `max() + 1` computation | Iterations are positive |
| `GapDetection` | Expected to fail | Detects commit failure gaps |

### External Events

| Event | Python Representation | TLA+ Action |
|-------|----------------------|-------------|
| Git commit success | `git commit` exit 0 | `CommitToGitSuccess(s)` |
| Git commit failure | `git commit` non-zero exit | `CommitToGitFail(s)` |
| Session crash | Process exits before commit | `CommitExhausted(s)` |

## Correspondence Verification

To verify correspondence is maintained:

1. **Code changes**: When modifying lock/iteration code, check TLA+ action mapping
2. **TLA+ changes**: When extending specs, update this document and correspondence.json
3. **Property tests**: `python3 scripts/run_tla_property_tests.py --staged` (pre-commit runs this when TLA+ specs or mapped Python files change)

## Rust Correspondence Pattern

For Rust repos, use the z4-based pattern documented in
`docs/tla-rust-patterns.md`, including the `TLA_TO_RUST_MAPPING.md` template and
`tla_invariant_<invariant_name>` proptest naming. The concrete reference lives
in the z4 repo at `crates/z4-tla-bridge/src/lib.rs` and
`docs/TLA_TO_RUST_MAPPING.md`. The `z4-tla-bridge` crate standardizes TLC
execution and error classification via `TlcRunner`, `TlcOutcome`,
`TlcViolation`, and `TlcErrorCode`.

## Key Design Decisions

### TOCTOU Mitigation (cargo_lock)

The `force_release_stale_lock()` function re-verifies staleness immediately before
the atomic rename. This shrinks the TOCTOU window from milliseconds to microseconds.
See `#624` and `tla/cargo_lock_with_toctou.tla:196-235` for the formal model.

### Gap Detection (iteration_tags)

When commits fail after iteration assignment, gaps appear in git history.
The `GapDetection` invariant in `iteration_tags_with_commit_failure.tla:263-269`
intentionally fails to demonstrate this scenario. The system tolerates gaps
because `max(file, git) + 1` always produces a unique next iteration.

## References

- Design: `designs/2026-01-30-tla-implementation-correspondence-research.md`
- TLA+ specs: `tla/*.tla`
- Python: `ai_template_scripts/cargo_wrapper.py`, `looper/runner_git.py`
- Property tests: `tests/test_tla_properties.py`
