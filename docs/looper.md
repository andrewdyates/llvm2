# Looper Infrastructure

The looper (`looper.py`) runs AI sessions in an autonomous loop, managing iterations,
context injection, crash recovery, and multi-worker coordination.

## Environment Variables

Looper sets these environment variables at session start (via `setup_git_identity()`):

| Variable | Purpose | Example |
|----------|---------|---------|
| `AI_PROJECT` | Current project name | `ai_template` |
| `AI_ROLE` | Current role (uppercase) | `WORKER`, `MANAGER` |
| `AI_ITERATION` | Commit tag iteration number | `1001` |
| `AI_SESSION` | Session ID (6-char hex) | `a1b2c3` |
| `AI_MACHINE` | Machine hostname | `K3WXHC41F4` |
| `AI_WORKER_ID` | Worker ID for multi-worker mode | `1`, `2` (unset if single) |
| `AI_MACHINE_PREFIX` | Machine prefix for multi-machine | `sat` (unset if single) |
| `AI_CODER` | AI tool in use (`claude-code`, `codex`, or `dasher`) | `claude-code` |
| `AIT_VERSION` | ai_template commit hash | `abc1234` |
| `AIT_SYNCED` | ai_template sync timestamp | `2026-01-28T...` |
| `GIT_AUTHOR_NAME` | Git author for commits | `ai_template-worker-1001` |
| `GIT_AUTHOR_EMAIL` | Git author email | `a1b2c3@host.project.ai-fleet` |
| `GIT_COMMITTER_NAME` | Git committer for commits (same as author) | `ai_template-worker-1001` |
| `GIT_COMMITTER_EMAIL` | Git committer email (same as author) | `a1b2c3@host.project.ai-fleet` |
| `CLAUDE_CODE_VERSION` | Claude Code CLI version (set when `AI_CODER=claude-code`) | `1.2.3` |
| `CODEX_CLI_VERSION` | Codex CLI version (set when `AI_CODER=codex`) | `0.1.2504111205` |
| `DASHER_VERSION` | Dasher CLI version (set when `AI_CODER=dasher`) | `0.5.1` |

### Per-Iteration Variables

Looper manages these environment variables per iteration (via `_run_iteration()`):

| Variable | Purpose | Example |
|----------|---------|---------|
| `AI_PHASE` | Current rotation phase name, or `freeform` when no phase is selected, or `none` for USER mode | `high_priority`, `freeform` |
| `AI_INPUT_ISSUES` | Comma-separated issue numbers included in the current prompt | `3251,3220,3216` |
| `AI_THEME` | Active theme name from `.looper_config.json` (unset when no theme configured) | `reliability` |

`AI_PHASE` is set every iteration. `AI_INPUT_ISSUES` is set on main iterations only (retains its value during audit rounds). `AI_THEME` is only set when a theme is active and is cleared when the theme is removed from config.

### Optional Configuration Variables

Looper also reads these environment variables when set (they are not set by looper itself):

| Variable | Purpose | Example |
|----------|---------|---------|
| `LOOPER_DEBUG` | When `1`, log swallowed exception tracebacks to stderr | `1` |
| `AIT_COORD_DIR` | Directory for PID/coordination files (set via `spawn_session.sh --coord-dir`) | `/tmp/ait_coord` |
| `WORKER3_NORMAL_MODE` | When `1`, disable Worker 3 P3-only specialization (use normal priority sampling) | `1` |

### Usage in Scripts

Scripts can access these variables to determine context:

```python
import os

# Check if running in multi-worker mode
worker_id = os.environ.get("AI_WORKER_ID")
if worker_id:
    print(f"Running as Worker {worker_id}")

# Get current project
project = os.environ.get("AI_PROJECT", "unknown")
```

## Zone-Based File Locking

Zones prevent edit conflicts when multiple workers operate on the same repo.
Each worker is assigned a zone (set of glob patterns) and can only edit files
within their zone.

### Configuration

Configure zones in `.looper_config.json`:

```json
{
  "zones": {
    "worker_1": ["looper/**", "tests/test_looper*.py"],
    "worker_2": ["ai_template_scripts/**"]
  }
}
```

If no zones configured, all workers share full access (existing behavior).

### How It Works

1. **Pattern matching**: `file_in_zone()` checks if a file path matches any zone pattern
2. **Access control**: `can_edit_file()` returns False if file is outside worker's zone
3. **Exclusive locks**: `ZoneLock` provides temporary exclusive access for complex operations

### API

```python
from looper.zones import (
    get_worker_zone_patterns,  # Get patterns for a worker
    file_in_zone,              # Check if file matches patterns
    can_edit_file,             # Check if worker can edit file
    get_zone_status,           # Get all zone configuration
    ZoneLock,                  # Context manager for exclusive access
)

# Check if worker 1 can edit a file
patterns = get_worker_zone_patterns(worker_id=1)
if file_in_zone("looper/runner.py", patterns):
    print("Can edit")

# Exclusive zone access
with ZoneLock(worker_id=1) as lock:
    if lock.acquired:
        # Do multi-file operations
        pass
```

### CLI Usage

```bash
# Check if files are in zone
AI_WORKER_ID=1 python -m looper.zones check file1.py file2.py

# Show zone status
AI_WORKER_ID=1 python -m looper.zones status
```

## Checkpoint Recovery

Enables automatic recovery from crashes by persisting session state to disk.
Sessions that crash resume from their last checkpoint rather than starting fresh.

### How It Works

1. **Write checkpoint**: Before each iteration, looper writes state to
   `.looper_checkpoint_{mode}.json` (or `.looper_checkpoint_{mode}_{id}.json`)

2. **Detect crash**: At startup, checks if checkpoint exists with:
   - Different session_id (not our session)
   - Dead PID (process no longer running)

3. **Inject recovery context**: If crash detected, injects recovery info into prompt

### Checkpoint State

The checkpoint captures:

```python
@dataclass
class LooperCheckpoint:
    schema_version: int       # Version for forward compatibility
    session_id: str           # Unique session ID
    mode: str                 # Role (worker, manager, etc.)
    worker_id: int | None     # Worker ID if multi-worker
    iteration: int            # Current iteration
    timestamp: str            # ISO timestamp
    state: CheckpointState    # Working issues, phase, todo list, last tool
    context: CheckpointContext  # Modified files, uncommitted changes
    crash_signature: CrashSignature  # PID, hostname, started_at
```

### Recovery Context

When recovering, the AI sees:

```markdown
## RECOVERY: Resuming from crash

The previous session crashed at iteration 42. Recovery context:

**Last known state:**
- Working on issues: #123, #124
- Phase: implementation
- Last tool: `Edit`

**Progress before crash:**
- [x] Read requirements
- [ ] Implement feature (IN PROGRESS when crashed)
- [ ] Write tests

**Files modified (uncommitted):**
- src/main.py
- tests/test_main.py

**Instruction:** Review the uncommitted changes and continue...
```

### Configuration

No configuration needed. Checkpoint files are automatically managed.

## Session Coordination

Manages session lifecycle and graceful shutdown via STOP files and audit logs.

### Stop File Hierarchy

STOP files signal sessions to shut down gracefully. Files are checked in order
from most specific to least specific:

```
STOP_W1        # Stop worker 1 only (consumed after stop)
STOP_WORKER    # Stop all workers (expires, not consumed)
STOP           # Stop all roles (expires, not consumed)
```

**Check order for Worker 1:** `STOP_W1` → `STOP_WORKER` → `STOP`

**Instance-specific files** (e.g., `STOP_W1`, `STOP_M1`) are consumed after the
session stops. **Shared files** (e.g., `STOP_WORKER`, `STOP`) expire after
`AIT_STOP_EXPIRY_SEC` (default 3600 = 1 hour) and are cleared on next spawn.

### Stop File Content

STOP files can optionally contain a reason:

```bash
# Simple (still works)
touch STOP_W1

# With reason (recommended)
echo "switching to z4 work" > STOP_W1
```

Content is logged to the session audit log but not required. Empty = "no reason given".

### Session Audit Log

Session start/stop events are logged to `worker_logs/session.log` (or `$AIT_COORD_DIR/session.log` if set):

```
2026-02-04T14:30:00 W1 START pid=12345
2026-02-04T14:45:00 W2 START pid=12346
2026-02-04T15:00:00 W1 STOP  reason="switching to z4" by=user clean=true
2026-02-04T15:05:00 W2 STOP  reason="" by=signal clean=false
```

### Stop File Expiry

STOP files older than `AIT_STOP_EXPIRY_SEC` (default 3600 seconds = 1 hour) are
considered stale:

- **spawn_session.sh** clears expired files with warning before starting
- **looper** ignores expired STOP files (logs warning, continues running)

**Why expiry works:** If you `touch STOP`, all running sessions stop within
seconds. An hour later, the file is definitely stale - no active session is
waiting for it.

### Spawn Behavior

`spawn_session.sh` clears applicable STOP files before starting:

| Spawning | Clears |
|----------|--------|
| `spawn_session.sh worker --id=1` | `STOP_W1`, `STOP_WORKER`, `STOP` |
| `spawn_session.sh worker` | `STOP_WORKER`, `STOP` |
| `spawn_session.sh manager` | `STOP_MANAGER`, `STOP` |

Output shows reason and age:
```
Cleared stale STOP files: STOP (reason: switching tasks) [expired: 120m old]
```

### Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `AIT_STOP_EXPIRY_SEC` | `3600` | Seconds before STOP files are considered stale |
| `AIT_COORD_DIR` | `.` | Directory for coordination files (set via `spawn_session.sh --coord-dir`) |

### Related

- Design: `designs/2026-02-04-session-coordination.md`
- Implementation: #2368, #2369

## Multi-Machine Sync

Handles syncing zone branches with `origin/main` in multi-machine mode.
Secondary machines work on zone branches that need to stay up-to-date with main.

### Sync Mechanisms

There are three sync mechanisms that can apply to branch-based modes:

| Mechanism | Config Keys | When It Runs | Applies To |
|-----------|-------------|--------------|------------|
| **Startup sync** | `sync_on_startup`, `sync_strategy` | Once at looper start | All branches |
| **Legacy periodic sync** | `sync_interval_iterations`, `sync_strategy` | Every N iterations (if N > 0) | All branches |
| **Multi-machine sync** | `sync.strategy`, `sync.trigger`, etc. | Per `sync.trigger` setting | Multi-machine mode only |

**Key distinction:** The `sync` section (with nested keys like `sync.strategy`) applies **only**
in multi-machine mode (when `--machine` flag is used). The legacy root-level keys (`sync_on_startup`,
`sync_strategy`, `sync_interval_iterations`) apply to all branch-based modes.

**Precedence and interaction:**
- Both legacy periodic sync and multi-machine sync can run in the same iteration—they are independent.
- Legacy periodic sync runs based on iteration count modulo `sync_interval_iterations`.
- Multi-machine sync runs based on `sync.trigger` setting.
- To avoid redundant syncs in multi-machine mode:
  - Keep `sync_interval_iterations: 0` (default) to rely only on multi-machine sync, or
  - Set `sync.trigger: "manual"` to rely only on legacy periodic sync.

### Configuration

Configure multi-machine sync in `.looper_config.json`:

```json
{
  "sync": {
    "strategy": "rebase",           // "rebase" (default) or "merge"
    "trigger": "iteration_start",   // When to sync
    "auto_stash": true,             // Auto-stash uncommitted changes
    "conflict_action": "abort"      // "abort" or "continue_diverged"
  }
}
```

**Note:** The `sync` section only takes effect when running with `--machine` flag.
For single-machine branch work, use the legacy keys documented in "Git Sync" below.

### Sync Triggers

| Trigger | Behavior |
|---------|----------|
| `iteration_start` | Sync before each iteration (default) |
| `manual` | Only sync when explicitly requested |
| `hourly` | Sync approximately once per hour |

### Sync Status Codes

```python
class SyncStatus(Enum):
    UP_TO_DATE = "up_to_date"      # Already current with main
    SYNCED = "synced"              # Successfully synced
    CONFLICT = "conflict"          # Sync failed due to conflicts
    DIVERGED = "diverged"          # Branch diverged, sync skipped
    BLOCKED = "blocked"            # Uncommitted changes block sync
    FETCH_FAILED = "fetch_failed"  # Failed to fetch from origin
    NOT_APPLICABLE = "not_applicable"  # On main branch
    ERROR = "error"                # Other error
```

### API

```python
from looper.sync import (
    SyncConfig,
    SyncResult,
    sync_from_main,
    get_current_branch,
    has_uncommitted_changes,
    get_uncommitted_changes_result,  # Result[bool] variant
)

# Perform sync
config = SyncConfig(strategy="rebase", auto_stash=True)
result = sync_from_main(config)

if result.ok:
    print(f"Synced {result.commits_pulled} commits")
else:
    print(f"Sync failed: {result.reason}")

# For callers needing error details, use the Result-returning variant (preferred):
changes_result = get_uncommitted_changes_result()
if not changes_result.ok:
    print(f"Git error: {changes_result.error}")
elif changes_result.value:
    print("Uncommitted changes exist")
```

### Multi-Machine Setup

To run in multi-machine mode:

```bash
# Start looper on zone branch
./looper.py worker --machine=sat --branch=zone/sat

# Or just specify machine (branch auto-derived)
./looper.py worker --machine=sat
```

This:
1. Creates or switches to the zone branch
2. Sets `AI_MACHINE_PREFIX` env var for commit tags
3. Enables auto-sync with main
4. Auto-creates PRs for zone branches

## Phase Rotation

Manages phase rotation for roles that cycle through different focus areas.
Each role has weighted phases that determine work focus per iteration.

### How It Works

1. **Weighted phases**: Each phase has a weight (default 1) defined in role files
2. **Score calculation**: `score = weight × hours_since_last_run + starvation_bonus`
3. **Starvation prevention**: After `starvation_hours` (default 24), low-weight phases
   get a bonus ensuring they run at least once per day
4. **State persistence**: `.rotation_state.json` tracks last run time per phase per role
5. **Freeform iterations**: Every Nth iteration (default 3) is "freeform"

### Configuration

Phases are defined in role files (`.claude/roles/{role}.md`) using special syntax:

```markdown
<!-- PHASE:high_priority weight:3 -->
**High Priority Work** - Critical path and blockers

Pick from: P0 (always first), then P1 issues.
<!-- /PHASE:high_priority -->

<!-- PHASE:normal_work weight:2 -->
**Normal Work** - Features and bugs

Pick from: P2 issues.
<!-- /PHASE:normal_work -->
```

### Role-Specific Phases

| Role | Phases | Purpose |
|------|--------|---------|
| Worker | `high_priority`, `normal_work`, `quality` | Priority tiers |
| Manager | Audit phases | Review cycles |
| Researcher | Research phases | Research focus |
| Prover | Verification phases | Proof types |

### Scoring Formula

```
score = weight × max(hours_since_last_run, 1.0)

if hours > starvation_hours:
    overtime = min(hours - starvation_hours, 168)
    score += overtime × overtime
```

The starvation bonus is quadratic to ensure long-neglected phases get selected
regardless of their weight.

### State File

`.rotation_state.json` structure:

```json
{
  "worker": {
    "high_priority": {
      "last_run": "2026-01-28T10:30:00+00:00"
    },
    "normal_work": {
      "last_run": "2026-01-28T09:15:00+00:00"
    }
  },
  "manager": {
    "priority_review": {
      "last_run": "2026-01-28T08:00:00+00:00"
    }
  }
}
```

### Configuration Options

In `.claude/roles/shared.md` frontmatter:

```yaml
---
freeform_frequency: 3      # Every Nth iteration is freeform (0 to disable). Default is 3, but roles can override (e.g., worker uses 4)
starvation_hours: 24       # Hours before bonus kicks in
force_phase: null          # Override to force specific phase
---
```

### API

```python
from looper.rotation import (
    load_rotation_state,
    save_rotation_state,
    update_rotation_state,
    get_rotation_focus,
    select_phase_by_priority,
)

# Get current rotation focus
focus, phase = get_rotation_focus(
    iteration=42,
    rotation_type="work",
    phases=["high_priority", "normal_work", "quality"],
    phase_data={"high_priority": {"weight": 3}},
    role="worker",
)

# Update state after completing phase
update_rotation_state("worker", "high_priority")
```

## Context Subpackage (`looper/context/`)

Gathers context injected into AI prompts at session start.

| Module | Responsibility |
|--------|----------------|
| `git_context.py` | Git log parsing, directive extraction, @ROLE mentions, structured handoffs |
| `handoff_schema.py` | HandoffState enum, JSON schema, validation with warn-then-strict mode |
| `issue_context.py` | Issue context entry point, coordinates sampling and caching |
| `issue_sampling.py` | Role-based issue sampling with priority tier filtering |
| `issue_cache.py` | IterationIssueCache for API efficiency (#1676) |
| `issue_handoff.py` | Urgent handoff detection via `urgent-handoff` label |
| `issue_audit.py` | Audit label transitions: do-audit → needs-review |
| `system_context.py` | Memory, disk, zone status, health checks |
| `audit_context.py` | Manager-specific audit data (crash analysis, flags, metrics) |
| `uncommitted_warning.py` | Detect and warn about uncommitted changes |
| `helpers.py` | Shared utilities: label extraction, issue formatting |

### Structured Handoff Schema

Structured `## Handoff` blocks validate against `HANDOFF_JSON_SCHEMA` in
`looper/context/handoff_schema.py`. Validation is warn-only by default; set
`AIT_HANDOFF_STRICT=1` to reject unknown `state` values or malformed
`depends_on` entries. Use `get_handoff_schema()` to export the JSON schema for
prompt/tool integrations.

**Main entry point:** `run_session_start_commands(role)` in `__init__.py`

Returns dict with keys: `git_log`, `gh_issues`, `last_directive`, `other_feedback`,
`role_mentions`, `handoff_context`, `system_status`, `audit_data`.

See `looper/context/__init__.py` for full public API.

## Issue Manager (`looper/issue_manager*.py`)

Encapsulates GitHub issue operations for the looper runtime. Uses composition
to delegate functionality to specialized submodules.

### Architecture

```
IssueManager
├── IssueManagerBase   # Shared subprocess helpers, core operations
├── IssueAuditor       # Manager audit checks (stuck, thrashing, cycles)
└── CheckboxConverter  # Checkbox-to-issue conversion
```

### Key Functions

| Method | Purpose |
|--------|---------|
| `check_stuck_issues()` | Find issues stuck in progress too long |
| `check_thrashing_issues()` | Find issues with flip-flopping state changes |
| `check_closed_by_removal()` | Detect issues closed without proper workflow |
| `check_blocker_cycles()` | Find circular blocking dependencies |
| `check_escalation_sla(days)` | Find issues exceeding SLA threshold |
| `get_issue_checkboxes(num)` | Extract checkboxes from issue body |
| `convert_checkboxes_to_issues(num)` | Create child issues from unchecked boxes |

### Submodules

| Module | Responsibility |
|--------|----------------|
| `issue_manager.py` | Main facade, re-exports, delegation |
| `issue_manager_base.py` | Subprocess helpers, rate limiting |
| `issue_manager_audit.py` | Audit checks (stuck, thrashing, cycles) |
| `issue_manager_checkbox.py` | Checkbox-to-issue conversion with locking |

### API

```python
from looper.issue_manager import IssueManager

mgr = IssueManager(repo_path=Path("."), role="worker")

# Audit checks (Manager role)
stuck = mgr.check_stuck_issues()
cycles = mgr.check_blocker_cycles()

# Checkbox conversion
unchecked, checked, body = mgr.get_issue_checkboxes(123)
mgr.convert_checkboxes_to_issues(123)
```

## Telemetry (`looper/telemetry.py`)

Looper self-telemetry for iteration metrics, stats, and health monitoring.

### Key Functions

| Function | Purpose |
|----------|---------|
| `record_iteration(metrics)` | Log iteration metrics to JSONL |
| `compute_stats(hours)` | Calculate stats for time window |
| `get_health_summary()` | Get current health status |
| `extract_token_usage(output)` | Parse token usage from AI output |
| `check_consecutive_abort_alert()` | Detect consecutive aborts |
| `update_oversight_metrics()` | Update oversight ratio tracking |

### Metrics Storage

Metrics stored in `metrics/looper.jsonl` as JSONL (one `IterationMetrics` per line):

```json
{"project": "ai_template", "role": "W", "iteration": 1001, "session_id": "a1b2c3",
 "start_time": 1706454600.0, "end_time": 1706454845.0, "duration_seconds": 245.0,
 "ai_tool": "claude", "ai_model": "sonnet", "exit_code": 0, "committed": true,
 "working_issues": [123], "tool_call_count": 42}
```

### IterationMetrics

```python
@dataclass
class IterationMetrics:
    project: str              # Project name (e.g., "ai_template")
    role: str                 # W, M, R, P, U
    iteration: int            # Iteration number
    session_id: str           # Session ID (6-char hex)
    start_time: float         # Unix timestamp
    end_time: float           # Unix timestamp
    duration_seconds: float   # Iteration duration
    ai_tool: str              # "claude" or "codex"
    ai_model: str | None      # Model name if known
    exit_code: int            # AI exit code
    committed: bool           # Whether iteration committed
    incomplete_marker: bool   # Had [INCOMPLETE] in commit
    done_marker: bool         # Had [DONE] in commit
    audit_round: int          # Current audit round (0 = main)
    audit_committed: bool     # Whether audit committed
    audit_rounds_run: int     # Total audit rounds run
    rotation_phase: str | None  # Current phase name
    working_issues: list[int]   # Issues worked on
    worker_id: int | None = None  # Multi-worker instance ID
    log_file: str | None = None   # Log file path
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_creation_tokens: int = 0
    tool_call_count: int = 0      # Total tool calls
    tool_call_types: dict[str, int] | None = None   # {"Bash": 10, "Read": 15}
    tool_call_duration_ms: dict[str, int] | None = None  # {"Bash": 15000}
```

### LooperStats

```python
@dataclass
class LooperStats:
    window_hours: int             # Time window for stats
    total_iterations: int         # Total in window
    success_rate: float           # 0-1 success ratio
    avg_duration_seconds: float   # Average iteration time
    claude_count: int             # Claude iterations
    claude_success_rate: float    # Claude success ratio
    codex_count: int              # Codex iterations
    codex_success_rate: float     # Codex success ratio
    audit_completion_rate: float  # Audit success ratio
    avg_audit_rounds: float       # Average audit rounds
    crash_count: int              # Crashed iterations
    timeout_count: int            # Timed out iterations
    phase_success_rates: dict[str, float]  # Success by phase
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    cache_hit_rate: float = 0.0
    early_abort_count: int = 0    # No-issue aborts
    consecutive_early_aborts: int = 0  # Current abort streak
    total_tool_calls: int = 0
    tool_distribution: dict[str, int] | None = None
    avg_tool_calls_per_iteration: float = 0.0
```

### API

```python
from looper.telemetry import (
    compute_stats,
    get_health_summary,
    LooperStats,
)

# Get stats for last 24 hours
stats = compute_stats(hours=24)
print(f"Success rate: {stats.success_rate:.1%}")
print(f"Avg duration: {stats.avg_duration_seconds:.1f}s")

# Health summary
health = get_health_summary()
```

Note: `IterationMetrics` is created internally by the runner. Use `compute_stats()`
and `get_health_summary()` to read aggregated telemetry data.

## Runner Architecture

The `LoopRunner` class is the main orchestrator for autonomous AI sessions. It uses
a mixin architecture to keep code modular and under line count thresholds.

### Class Composition

```
LoopRunner
├── RunnerBase          # Core state, initialization, cleanup
├── RunnerLoopMixin     # Main run() loop, iteration orchestration
├── RunnerIterationMixin # Single iteration execution delegation
├── RunnerAuditMixin    # do-audit → needs-review workflow
├── RunnerSyncMixin     # Multi-machine branch sync (zone branches)
├── RunnerGitMixin      # Git operations, identity setup
└── RunnerControlMixin  # Signal handling, STOP file detection
```

### Key Components

| Module | Purpose |
|--------|---------|
| `runner.py` | Main entrypoint, CLI parsing, `LoopRunner` class |
| `runner_base.py` | `RunnerBase` - init, setup, dependencies, PID management |
| `runner_loop.py` | `RunnerLoopMixin` - main loop, delay logic, iteration flow |
| `runner_iteration.py` | `RunnerIterationMixin` - delegates to `IterationRunner` |
| `runner_audit.py` | `RunnerAuditMixin` - audit rounds, issue transitions |
| `runner_sync.py` | `RunnerSyncMixin` - rebase/merge from main |
| `runner_git.py` | `RunnerGitMixin` - identity, sync staleness checks |
| `runner_control.py` | `RunnerControlMixin` - signals, graceful shutdown |
| `runner_prompt.py` | Prompt debugging utility for testing/inspection |
| `iteration.py` | `IterationRunner` - single iteration: prompt, subprocess, output |
| `iteration_process.py` | `ProcessManager` - AI subprocess execution and lifecycle |
| `iteration_prompt.py` | `PromptBuilder` - prompt construction and assembly |
| `iteration_tool_tracking.py` | Tool call tracking and event processing |

### Initialization Flow

1. **CLI parsing** (`main()`) - mode, worker_id, machine, branch flags
2. **Dirty worktree check** - fails if uncommitted changes (override: `--allow-dirty`)
3. **`RunnerBase.__init__`** - config load, managers, checkpoint setup
4. **`setup()`** - dependency checks, git identity, hooks, PID file
5. **Crash recovery** - checkpoint detection, context injection
6. **Main loop** - `run()` iterates until STOP file or signal

### Execution Flow

```
run() loop:
  ├── Check STOP files
  ├── Pre-iteration sync (multi-machine)
  ├── Run iteration via IterationRunner
  │     ├── Build prompt (context injection)
  │     ├── Spawn AI subprocess (claude/codex)
  │     ├── Monitor for timeout/silence
  │     └── Parse output, extract issues
  ├── Audit rounds (if do-audit issues exist)
  │     ├── Resume same session (claude --resume)
  │     └── Transition do-audit → needs-review
  ├── Write checkpoint
  ├── Update telemetry
  └── Delay before next iteration
```

### Multi-Instance Support

Workers support multiple concurrent instances via `--id=N`:

- **File paths**: `worker_1`, `worker_2` suffixes for PID/status/iteration files
- **Environment**: `AI_WORKER_ID=N` for scripts and commit identity
- **File tracking**: Per-worker `.file_tracker_N.json` for edit scope
- **Zones**: Optional zone patterns to partition file access

### Subprocess Management

The runner manages AI tool subprocesses with:

- **Timeout detection**: Configurable per-iteration timeout
- **Silence detection**: 10-minute silence timeout triggers intervention
- **Process groups**: Clean termination of subprocess trees
- **Exit code handling**: Maps AI exit codes to iteration outcomes

### Login Shell Considerations (Codex/Dasher)

AI tools that spawn login shells (e.g., codex via `/bin/zsh -lc`) bypass PATH modifications
made by looper at runtime. This affects the cargo wrapper and gh rate limiter.

**Problem:** When codex runs cargo commands, they bypass serialization because the login
shell doesn't inherit the `ai_template_scripts/bin` PATH prefix set by looper.

**Solution:** Add to `~/.zprofile` (login shells) or `~/.zshenv` (all shells):

```bash
# AI template wrappers (cargo serialization, gh rate limiting)
export PATH="$HOME/<project>/ai_template_scripts/bin:$PATH"
```

Replace `<project>` with the main project directory name. The init script
(`init_from_template.sh`) checks for this configuration and warns if missing.
Note: `zsh -lc` does not read `~/.zshrc`, so `~/.zprofile` or `~/.zshenv` is required.

See `README.md` "Shell Configuration" section for details.

## Configuration Reference

All config keys in `.looper_config.json`. Keys are read from project config,
with role-specific sections (`worker`, `manager`, etc.) merged on top.

### Core Settings

| Key | Default | Description |
|-----|---------|-------------|
| `restart_delay` | `60` | Seconds to wait between successful iterations |
| `error_delay` | `300` | Seconds to wait after error/crash |
| `silence_timeout` | `600` | Seconds of silence before intervention (10 min) |
| `cleanup_closed_interval` | `10` | Iterations between closed issue cleanup |
| `sync_interval_iterations` | `0` | Legacy periodic git sync interval (0 = disabled). See "Git Sync" |
| `uncommitted_warn_threshold` | `100` | Lines of uncommitted changes to warn about |
| `iteration_timeout` | `2700` (worker), `7200` (others) | Per-iteration timeout in seconds |
| `local_mode` | `false` | Enable local mode (disable GitHub API calls) |

### Timeout Overrides

Top-level `timeouts` allow tuning subprocess timeouts (seconds).

Example `.looper_config.json`:

```json
{
  "timeouts": {
    "git_default": 5,
    "gh_list": 15,
    "gh_view": 10,
    "health_check": 60,
    "max_silence": 3600
  }
}
```

| Key | Default | Description |
|-----|---------|-------------|
| `git_default` | `5` | Default timeout for git commands in context gathering |
| `gh_list` | `15` | `gh issue list` timeout for issue cache/sampling |
| `gh_view` | `10` | `gh issue view/edit` timeout for issue lookups/label edits |
| `health_check` | `60` | `scripts/system_health_check.py` timeout |
| `max_silence` | `3600` | Absolute silence timeout for long-running commands |

### Git Wrapper Settings

Top-level config for the git wrapper (`ai_template_scripts/bin/git`).

| Key | Default | Description |
|-----|---------|-------------|
| `git_lock_wait_sec` | `300` | Seconds to wait for git commit lock before timeout |

**Environment variable override:** `AIT_GIT_LOCK_WAIT_S=N` overrides this config.

Example `.looper_config.json`:

```json
{
  "git_lock_wait_sec": 600
}
```

### Local Mode

Disable GitHub API calls for offline development or testing.

**Enable via any one method:**
1. **Environment:** `AIT_LOCAL_MODE=1`
2. **Touch file:** `touch .local_mode` in repo root
3. **Config:** `local_mode: true` in `.looper_config.json`

Environment variable takes priority, then touch file, then config.

**Effects when enabled:**
- `gh` commands return early with "(local mode - GitHub API disabled)"
- Issue fetching and mirror refresh skipped
- Auth/network checks skipped at startup
- Git operations and commit hooks still run normally

**When to use:**
- Offline development without network access
- Avoiding rate limits during intensive testing
- Testing looper logic without API calls
- CI environments without GitHub credentials

#### Full Local Mode (`AIT_LOCAL_MODE=full`)

Full local mode provides complete offline operation with local issue storage.

**Enable:** `AIT_LOCAL_MODE=full` (via environment or `.looper_config.json`)

**Key features:**
- Issues stored in `.issues/L*.md` with YAML frontmatter
- Full `gh issue` command support (create, list, view, edit, close, comment)
- Sync back to GitHub with `sync_local_issues.py`

**See:** [`docs/local-development.md`](local-development.md) for complete documentation including:
- Local issue format and workflow examples
- Syncing to/from GitHub
- Limitations and configuration options

### AI Model Selection

| Key | Default | Description |
|-----|---------|-------------|
| `claude_model` | `null` | Override Claude model (e.g., `"sonnet"`, `"opus"`) |
| `dasher_probability` | `0.0` | Probability of using Dasher (0-1) |
| `dasher_model` | `null` | Model for Dasher sessions |
| `codex_probability` | `0.0` | Probability of using Codex (0-1) |
| `codex_model` | `null` | Model for Codex sessions |
| `codex_models` | `[]` | List of models for round-robin Codex selection |
| `gemini_probability` | `0.0` | Probability of using Gemini (0-1) |

### Per-Role Model Routing (#1888)

For explicit per-role model selection without editing role frontmatter, use `model_routing`:

```json
{
  "model_routing": {
    "default": {
      "claude_model": "sonnet",
      "codex_model": "gpt-4o",
      "codex_models": ["gpt-4o", "gpt-4o-mini"],
      "dasher_model": "gpt-4o-mini"
    },
    "roles": {
      "worker": {
        "codex_models": ["gpt-5.2", "gpt-4o"],
        "claude_model": "sonnet"
      },
      "manager": {
        "claude_model": "opus"
      },
      "researcher": {
        "claude_model": "sonnet"
      },
      "prover": {
        "claude_model": "opus"
      }
    },
    "audit": {
      "claude_model": "opus"
    }
  }
}
```

**Routing precedence** (highest to lowest):
1. `model_routing.audit` when `audit_round > 0`
2. `model_routing.roles.<role>` when present
3. `model_routing.default` when present
4. Legacy keys (`claude_model`, `codex_model`, etc.)

### Model Switching for Resumed Sessions

Control what happens when model selection changes during a resumed session:

```json
{
  "model_switching": {
    "enabled": false,
    "allowed_tools": ["claude"],
    "strategy": "restart_session",
    "preserve_history": false
  }
}
```

| Key | Default | Description |
|-----|---------|-------------|
| `enabled` | `false` | Allow model switching in resumed sessions |
| `allowed_tools` | `[]` | Tools that can switch (default: none) |
| `strategy` | `"restart_session"` | `restart_session` drops resume ID, `resume_with_model` keeps it (future) |
| `preserve_history` | `false` | Reserved for future `/model` command support |

When `enabled=false` (default), if the selected model changes between iterations,
the session will pin to the previous model to maintain context continuity.

### Audit Configuration

| Key | Default | Description |
|-----|---------|-------------|
| `auto_audit` | `true` | Enable automatic do-audit → needs-review transitions |
| `audit_max_rounds` | `5` | Maximum audit rounds per iteration |
| `audit_min_issues` | `3` | Minimum issues to find per audit phase |
| `audit_max_rounds_p0..p3` | unset | Optional per-priority max rounds override (highest-priority do-audit issue wins) |
| `audit_min_issues_p0..p3` | unset | Optional per-priority min issues override (highest-priority do-audit issue wins) |
| `auto_pr` | `true` | Auto-create PRs for zone branches |

When `audit_*_pX` overrides are configured, looper applies the values for the
highest-priority issue in the current `do-audit` set (P0 > P1 > P2 > P3). If
an override is absent, it falls back to the global `audit_max_rounds` /
`audit_min_issues` values.
When the effective `audit_max_rounds` is `0`, looper skips follow-up audit rounds
for that iteration.

### Checkpoint Settings

Control checkpoint recovery behavior. See "Checkpoint Recovery" section above for details.

| Key | Default | Description |
|-----|---------|-------------|
| `checkpoint_enabled` | `true` | Enable checkpoint recovery |
| `checkpoint_recovery_max_tokens` | `2000` | Max tokens in recovery context |
| `checkpoint_tool_output_truncate` | `1000` | Truncate tool outputs in checkpoints |

### Issue Sampling

Control how many issues are shown in startup summaries.

```json
{
  "issue_sampling": {
    "do_audit": 5,
    "in_progress": 5,
    "urgent": 5,
    "P1": 3,
    "P2": 2,
    "P3": 1,
    "new": 2,
    "random": 1,
    "oldest": 1,
    "domain": 10
  }
}
```

| Key | Default | Description |
|-----|---------|-------------|
| `issue_sampling.do_audit` | `5` | Max do-audit issues shown |
| `issue_sampling.in_progress` | `5` | Max in-progress issues shown |
| `issue_sampling.urgent` | `5` | Max urgent issues shown |
| `issue_sampling.P1` | `3` | Max P1 issues shown |
| `issue_sampling.P2` | `2` | Max P2 issues shown |
| `issue_sampling.P3` | `1` | Max P3 issues shown |
| `issue_sampling.new` | `2` | Max newest issues shown |
| `issue_sampling.random` | `1` | Max random issues shown |
| `issue_sampling.oldest` | `1` | Max oldest issues shown |
| `issue_sampling.domain` | `10` | Max domain issues shown for non-worker roles |

**Limit = 0 Behavior (Issue #2375):**

Setting a tier limit to `0` **disables** that tier (no issues shown), except for P0 which is always unlimited. This allows fine-grained control over which issue types appear in the prompt.

| Setting | Behavior |
|---------|----------|
| `"P1": 0` | P1 tier disabled (no `[P1]` prefix issues) |
| `"P2": 0` | P2 tier disabled (no `[P2]` prefix issues) |
| `"new": 0` | Discovery tier disabled (no `[NEW]` issues) |
| `"P0": 0` | **Exception**: P0 always unlimited (treated as unlimited) |

Note: Issues may still appear through other tiers. For example, a P1 issue disabled in the P1 tier could appear in the `[NEW]` discovery tier if not shown elsewhere.

**Example:** Disable all tiers except P0 and do-audit:

```json
{
  "issue_sampling": {
    "P1": 0,
    "P2": 0,
    "P3": 0,
    "new": 0,
    "random": 0,
    "oldest": 0
  }
}
```

### Issue Cache Limit

The issue sampling algorithm operates on a cached set of issues, limited by default to 200:

| Key | Default | Description |
|-----|---------|-------------|
| `issue_cache_limit` | `200` | Maximum issues to fetch for sampling |

**Impact:** For repos with >200 open issues, issues beyond the limit are never fetched and thus invisible to sampling. Since sampling prioritizes by tier (P0 → do-audit → in-progress → urgent → P1 → P2 → P3), P3 issues may be invisible if higher-priority issues fill the cache. GitHub returns issues ordered by update time, so stale P3 issues are most at risk.

**When to increase:** If your repo has >200 open issues (check with `gh issue list --state open --json number -q 'length'`), consider increasing `issue_cache_limit` to ensure P3 issues are sampled.

**Example:**

```json
{
  "issue_cache_limit": 500
}
```

**Tradeoffs:**
- Higher limits increase GitHub API usage and startup time
- Most repos have <200 issues, so the default is typically sufficient
- For repos with large backlogs, 300-500 provides better P3 coverage

### Phase Rotation

| Key | Default | Description |
|-----|---------|-------------|
| `rotation_type` | `""` | Rotation mode: `""` (disabled), `"audit"`, `"research"`, `"verification"`, `"work"` |
| `rotation_phases` | `[]` | List of phase names to rotate through |
| `phase_data` | `{}` | Per-phase config (weights, descriptions) |
| `freeform_frequency` | `3` | Every Nth iteration is freeform (0 to disable) |
| `force_phase` | `null` | Override to force specific phase |
| `starvation_hours` | `24` | Hours before starvation bonus kicks in |

### Git Sync

| Key | Default | Description |
|-----|---------|-------------|
| `sync_on_startup` | `true` | Sync with origin on looper start (zone branches only) |
| `sync_strategy` | `"rebase"` | Sync method: `"rebase"` or `"merge"` |
| `sync_interval_iterations` | `0` | Legacy: sync every N iterations (0 = disabled). See note below. |
| `staged_check_abort` | `false` | Abort if staged changes exist |
| `git_author_name` | `null` | Override git author name (rarely needed) |

**Sync mechanism precedence:**
1. **Startup sync**: Uses `sync_on_startup` + `sync_strategy` (legacy keys, always active)
2. **Pre-iteration sync**: Uses `sync` section config when `trigger=iteration_start` (multi-machine only)
3. **Periodic sync**: Uses `sync_interval_iterations` + `sync_strategy` (legacy, runs if interval > 0)

For new setups, prefer the `sync` section config. Legacy keys remain for backwards compatibility.

### Monitoring

| Key | Default | Description |
|-----|---------|-------------|
| `pulse_interval_minutes` | `30` | Minutes between pulse.py health checks |
| `scrub_logs` | `false` | Enable log scrubbing via log_scrubber.py |
| `memory_watchdog_enabled` | `true` | Enable memory pressure monitoring |
| `memory_watchdog_threshold` | `"critical"` | Memory pressure level to trigger action |

### Escalation

| Key | Default | Description |
|-----|---------|-------------|
| `escalation_sla_days` | `null` | Days before issues escalate (null = disabled) |

### Oversight (Telemetry)

Nested under `"oversight"` key:

| Key | Default | Description |
|-----|---------|-------------|
| `window_hours` | `168` | Hours of history to analyze (1 week) |
| `threshold` | `0.7` | Warning threshold for metrics |
| `alert_hours` | `24` | Hours between alerts |

### Example Configuration

```json
{
  "restart_delay": 120,
  "audit_min_issues": 5,
  "rotation_type": "work",
  "rotation_phases": ["high_priority", "normal_work", "quality"],
  "phase_data": {
    "high_priority": {"weight": 3},
    "normal_work": {"weight": 2},
    "quality": {"weight": 1}
  },
  "worker": {
    "freeform_frequency": 4,
    "codex_probability": 0.1
  },
  "manager": {
    "audit_max_rounds": 3
  }
}
```

### Per-Role Overrides

Role-specific settings can override project defaults:

```json
{
  "restart_delay": 60,
  "worker": {
    "restart_delay": 120
  },
  "manager": {
    "restart_delay": 300
  }
}
```

The merged config for `worker` role would have `restart_delay: 120`.

### Per-Instance Overrides

For multi-worker setups, per-instance settings override role defaults:

```json
{
  "worker": {
    "restart_delay": 60
  },
  "worker_1": {
    "force_phase": "high_priority"
  },
  "worker_3": {
    "force_phase": "quality",
    "iteration_timeout": 1800
  }
}
```

Instance keys use `{role}_{id}` format (e.g., `worker_1`, `worker_2`).
These override both project defaults and role defaults.

Config precedence (highest to lowest):
1. Per-instance overrides (`worker_1`)
2. Role overrides (`worker`)
3. Project defaults (root level)
4. Role file frontmatter
5. Shared file frontmatter

## Memory Logging (`looper/memory_logger.py`)

OOM debugging infrastructure captures memory state before and after commands.

### Key Functions

| Function | Purpose |
|----------|---------|
| `get_memory_state()` | Returns `MemoryState` with used/free/total GB and pressure level |
| `get_process_tree_memory_mb(pid)` | Total memory of process tree (parent + all descendants) |
| `log_pre_command_memory(cmd)` | Log memory state BEFORE tool calls |
| `log_post_crash_memory(cmd, exit_code)` | Log memory state after crash/abnormal exit |
| `is_oom_signal(exit_code)` | Detect OOM kills (exit 137 or signal -9) |

### MemoryState Fields

```python
@dataclass
class MemoryState:
    used_gb: float        # Active + wired memory
    free_gb: float        # Free + inactive + speculative
    total_gb: float       # Total tracked pages
    pressure_level: str   # "normal", "warning", "critical"
    timestamp: float      # Unix timestamp
```

### Log Location

Memory logs are stored in `worker_logs/memory.log` as JSONL:

```json
{"timestamp": "2026-01-28T10:30:00", "event": "pre_command", "command": "cargo build",
 "memory": {"used_gb": 12.5, "free_gb": 3.5, "used_percent": 78, "pressure_level": "warning"}}
```

### Debugging OOM Crashes

```bash
# Check for OOM kills in logs
grep '"was_oom": true' worker_logs/memory.log

# View memory pressure before crashes
grep '"event": "post_crash"' worker_logs/memory.log | jq .memory.pressure_level

# Quick memory check
python3 -c "from looper.memory_logger import get_memory_state; print(get_memory_state())"
```

### Configuration

Memory monitoring is controlled by config keys:
- `memory_watchdog_enabled` (default: true) - Enable memory pressure monitoring
- `memory_watchdog_threshold` (default: "critical") - Pressure level to trigger action

## Design Rationale: Async-First Architecture

Looper uses an **async-first design** where roles coordinate via git commits rather than
synchronous function calls. This differs from popular agent frameworks (LangGraph, Swarm,
CrewAI) that use synchronous handoffs. The async approach is intentional and provides
several benefits for autonomous AI systems.

### Why Async-First?

| Principle | Sync Frameworks | Looper (Async) |
|-----------|-----------------|----------------|
| **Durability** | State in memory; crashes lose work | Git commits survive crashes |
| **Observability** | Debug logs, tracing | Full git history audit trail |
| **Human oversight** | Callbacks, breakpoints | Intervention between iterations |
| **Multi-machine** | Shared state servers | Zone branches, distributed git |
| **Role isolation** | Functions call functions | Async handoffs via commits/issues |

### Trade-offs vs Synchronous Frameworks

**What looper intentionally does NOT implement:**

1. **Synchronous handoffs** (like Swarm's `return Agent`) - Would require in-memory state
   that doesn't survive crashes. Async commit-based handoffs are more durable.

2. **Dynamic routing** (like CrewAI's manager delegation) - The static role rotation with
   priority weighting is simpler and more predictable. Dynamic routing would add complexity
   without clear benefit for autonomous AI operations.

3. **Inline retry loops** (like Anthropic's evaluator-optimizer) - Verification happens in
   the next Prover session, not inline. This separation of concerns keeps Worker focused
   on implementation and Prover on correctness.

**What looper does better:**

1. **Durable state** - Every commit is a checkpoint. In-memory frameworks lose state on
   crash; looper resumes from the last commit.

2. **Full audit trail** - Git log provides complete history of what each role did, when,
   and why. External frameworks require separate logging/tracing infrastructure.

3. **Graceful degradation** - If one role's session fails, others continue. No cascade
   failures from synchronous dependencies.

4. **Human-compatible pace** - Humans can review commits, intervene via STOP files, adjust
   priorities via issue labels - all without modifying running sessions.

### Comparison Table

| Feature | LangGraph | Swarm | CrewAI | Looper |
|---------|-----------|-------|--------|--------|
| State checkpoint | In-memory | In-memory | In-memory | **Git commits** |
| Crash recovery | From checkpoint | Restart | Restart | **Resume from commit** |
| Audit trail | Logs | Logs | Logs | **Git history** |
| Multi-machine | Shared DB | None | None | **Zone branches** |
| Role coordination | Graph edges | Returns | Manager | **Commits + Issues** |
| Human oversight | Callbacks | None | Callbacks | **STOP files, labels** |

### When Sync Might Be Better

Synchronous frameworks are better suited for:

- **Short-lived sessions** where crash recovery doesn't matter
- **Tight feedback loops** requiring sub-second handoffs
- **Stateless operations** without long-term context

Looper's async design optimizes for **long-running autonomous systems** where durability
and observability matter more than latency.

### Further Reading

- Research: `reports/research/2026-02-03-external-agent-orchestration-patterns.md`
- Related issue: #2305

## API Reference

The looper package exports a public API with three stability levels:

| Level | Meaning |
|-------|---------|
| **STABLE** | Safe for external use. Changes require deprecation period. |
| **INTERNAL** | Exported for testing/debugging. May change without notice. |
| **EXPERIMENTAL** | Subject to change as features evolve. |

### Stable API

These classes, functions, and constants are the primary public interface:

```python
from looper import (
    # Classes - primary entry points [STABLE]
    LoopRunner,            # Main orchestrator for AI sessions
    IterationRunner,       # Single iteration execution
    IterationResult,       # Iteration outcome data
    IssueManager,          # GitHub issue operations
    StatusManager,         # Status file management
    CheckpointManager,     # Crash recovery state
    CheckpointState,       # Checkpoint data structure
    RecoveryContext,       # Recovery context for prompts

    # Functions [STABLE]
    load_role_config,      # Load role configuration
    load_project_config,   # Load project configuration
    main,                  # CLI entry point

    # Constants [STABLE]
    ROLES_DIR,             # Path to role files
    LOG_DIR,               # Path to log directory
    LOG_RETENTION_HOURS,   # Log retention period
    EXIT_NO_ISSUES,        # Exit code: no issues to work
    EXIT_SILENCE,          # Exit code: silence timeout
    EXIT_TIMEOUT,          # Exit code: iteration timeout

    # Logging [STABLE]
    log_debug,             # Debug level logging
    log_info,              # Info level logging
    log_warning,           # Warning level logging
    log_error,             # Error level logging
    get_logger,            # Get a named logger
    setup_logging,         # Initialize logging
    build_log_path,        # Construct log file path
)
```

### Model Routing API

Per-role model selection (see Configuration Reference for config format):

```python
from looper import (
    # Model routing [STABLE]
    ModelRouter,           # Routes model selection by role
    ModelSelection,        # Selected model result
    AiTool,                # AI tool enum (CLAUDE, CODEX, DASHER)
    ModelSwitchingPolicy,  # Model switching behavior config
)

# Usage
router = ModelRouter(config)
selection = router.select(role="worker", audit_round=0)
print(f"Tool: {selection.tool}, Model: {selection.model}")
```

### Internal Helpers

These are exported for testing but may change without notice:

```python
from looper import (
    # Testing/debugging only [INTERNAL]
    run_session_start_commands,  # Context injection
    get_rotation_focus,          # Phase rotation
    parse_frontmatter,           # YAML frontmatter parsing
    parse_phase_blocks,          # Phase block parsing
    inject_content,              # Template injection
    build_codex_context,         # Codex-specific context
    install_hooks,               # Git hook installation
    show_prompt,                 # Prompt debugging
    check_concurrent_sessions,   # Session conflict check
    get_project_name,            # Project name detection
    validate_config,             # Config validation
    build_audit_prompt,          # Audit prompt construction
    extract_issue_numbers,       # Issue number extraction
    load_rotation_state,         # Rotation state loading
    save_rotation_state,         # Rotation state saving
    update_rotation_state,       # Rotation state update
    select_phase_by_priority,    # Phase priority selection
)
```

### Experimental API

Multi-machine sync and zone management - subject to change:

```python
from looper import (
    # Multi-machine sync [EXPERIMENTAL]
    SyncConfig,                 # Sync configuration
    SyncResult,                 # Sync operation result
    SyncStatus,                 # Sync status enum
    sync_from_main,             # Sync zone branch with main
    get_current_branch,         # Get current git branch
    has_uncommitted_changes,    # Check for uncommitted changes
    get_uncommitted_changes_result,  # Result[bool] variant
    get_uncommitted_line_count,      # Count uncommitted lines
    get_staged_files,           # List staged files
    get_commits_behind,         # Count commits behind main
    get_conflict_files,         # List conflict files

    # Deprecated sync functions (use warn_* instead)
    check_uncommitted_work_size,  # → warn_uncommitted_work()
    check_stale_staged_files,     # → get_staged_files() + warn_stale_staged_files()

    # Zone management [EXPERIMENTAL]
    WorkerInfo,            # Worker zone info
    ZoneLock,              # Zone exclusive lock
    ZoneStatus,            # Zone status enum
    can_edit_file,         # Check file edit permission
    check_files_in_zone,   # Batch zone check
    file_in_zone,          # Single file zone check
    get_worker_zone_patterns,   # Get worker's patterns
    get_zone_status,       # Get zone config
    get_zone_status_line,  # One-line zone status
    load_zone_config,      # Load zone configuration

    # Checkpoint [EXPERIMENTAL]
    get_checkpoint_filename,  # Checkpoint file path
)
```

### Stability Policy

- **STABLE**: Requires deprecation warning for 2+ releases before removal
- **INTERNAL**: May change or be removed at any time
- **EXPERIMENTAL**: May change as feature matures; stabilization planned

See `looper/__init__.py` docstring for the authoritative stability definitions.

## Related Documentation

- `designs/2026-01-25-auto-sync-protocol.md` - Multi-machine sync design
- `designs/2026-01-26-checkpoint-recovery.md` - Checkpoint recovery design
- `.claude/roles/` - Role configuration files
