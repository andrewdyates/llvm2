# Wrapper Architecture

Copyright 2026 Dropbox, Inc.
Author: Andrew Yates <ayates@dropbox.com>

ai_template provides shell wrappers that intercept common commands to add guardrails,
serialization, and rate limiting for autonomous AI sessions.

## Overview

| Wrapper | Lines | Purpose |
|---------|-------|---------|
| git | 628 | Branch blocking, commit serialization, destructive op guards |
| gh | 345 | Rate limiting, caching, REST fallback, AI identity injection |
| cargo | 153 | Build serialization, timeout management |
| grep | 114 | Memory pressure detection for recursive operations |

Location: `ai_template_scripts/bin/`

## Why Wrappers Exist

Wrappers solve problems specific to autonomous AI operations:

1. **Resource contention**: Multiple AI workers building simultaneously cause OOM and deadlocks
2. **API limits**: GitHub API rate limits (5,000+ requests/hour per token) are quickly exhausted by many parallel sessions
3. **Safety guardrails**: AIs shouldn't create branches, bypass hooks, or run destructive commands
4. **Coordination**: Multi-worker mode requires commit serialization and file tracking
5. **Memory protection**: Recursive grep on large codebases can exhaust system memory

Wrappers are transparent - AIs run normal commands (`git`, `gh`, `cargo`) without knowing
they're being intercepted.

## How Wrappers Are Installed

Looper prepends `ai_template_scripts/bin/` to PATH at session start via
`setup_wrapper_path()` in `looper/runner.py:316`. This ensures wrapper scripts
execute instead of system binaries.

```python
# looper/runner.py
def setup_wrapper_path() -> None:
    wrapper_bin = Path("ai_template_scripts/bin").resolve()
    os.environ["PATH"] = f"{wrapper_bin}{os.pathsep}{current_path}"
```

**Login shell consideration**: Tools like Codex spawn login shells (`/bin/zsh -lc`) that
reset PATH. Add to `~/.zprofile` for persistent wrapper installation:

```bash
export PATH="$HOME/<project>/ai_template_scripts/bin:$PATH"
```

## Git Wrapper

**Source**: `ai_template_scripts/bin/git`

### Behavior

| Feature | Description |
|---------|-------------|
| Branch blocking | Blocks `branch <name>`, `checkout -b`, `switch -c` (except `zone/*`) |
| Stash blocking | Blocks all `git stash` operations (use WIP commits instead) |
| Destructive guards | Blocks `restore`, `checkout --`, `reset --hard`, `clean -f` when workers active |
| Commit serialization | Per-repo lock prevents concurrent commits across sessions |
| Hook installation | Auto-installs `commit-msg` hook if missing |
| --no-verify blocking | Blocks hook bypass for AI roles (hooks enforce workflow) |
| Fixes keyword enforcement | Blocks Worker role from using `Fixes #N` (Manager only) |
| Multi-worker warnings | Advisory warnings when staging files outside worker's tracked set |
| Worker file display | Shows active worker files on `git status` |

### Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `AI_ROLE` | - | Role (WORKER/PROVER/RESEARCHER/MANAGER/USER) - set by looper |
| `AIT_ALLOW_NO_VERIFY` | `0` | Set to `1` to allow `--no-verify` for AI roles |
| `AIT_GIT_LOCK_WAIT_S` | `300` | Seconds to wait for commit lock |
| `AIT_GIT_LOCK_DISABLE` | `0` | Set to `1` to disable commit locking |

### Bypassing

```bash
# Override destructive operation guard
git restore --ait-force-destructive -- file.txt

# Use zone/* branches for multi-machine mode
git checkout -b zone/sat-machine
```

## gh Wrapper

**Source**: `ai_template_scripts/bin/gh`
**Python modules**: `gh_wrapper.py`, `gh_rate_limit.py`, `gh_post.py`

### Behavior

| Feature | Description |
|---------|-------------|
| Rate limiting | Blocks when quota critical, switches REST/GraphQL automatically |
| Response caching | Per-command TTL cache for read operations (`~/.ait_gh_cache/`); see table below |
| AI identity injection | Adds role/session/iteration footer to issue operations |
| PR attribution blocking | Blocks `Co-Authored-By: Claude` patterns in PR bodies |
| Write queue | Queues mutations when rate-limited for later replay |
| Stale fallback | Returns cached data with warning when API fails |

**TTL by command**

| Command | TTL |
|---------|-----|
| `gh issue list` | `20s` |
| `gh issue view` | `20s` |
| `gh pr list` | `20s` |
| `gh repo view` | `180s` |
| `gh label list` | `180s` |
| `gh search ...`, `gh api /search/...` | `300s` |

### Command Routing

| Command | Handler | Notes |
|---------|---------|-------|
| `gh issue create/comment/edit/close` | `gh_post.py` | Identity injection |
| `gh pr create` | Rate limiter + attribution check | Blocks AI attribution |
| All other `gh` commands | `gh_wrapper.py` → `gh_rate_limit.py` | Standard rate limiting |

### Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `AIT_LOCAL_MODE` | - | Set to `full` for complete local mode (no API calls) |
| `AIT_WRITE_THROUGH` | `1` | Write-through to local `.issues/` (enabled by default) |
| `AIT_DEBUG` | `0` | Set to `1` for debug logging |
| `AIT_GH_DEBUG` | `0` | Set to `1` for gh-specific debug logging |
| `AIT_USE_GITHUB_APPS` | `0` | Set to `1` to enable GitHub App authentication |

### Write-Through Storage (#1834)

Issue operations write to both GitHub API and local `.issues/` directory, ensuring local
state is always current even when GitHub API is unavailable or rate-limited.

**Behavior:**
- `gh issue create/comment/edit/close` mirrors to `.issues/<number>.md`
- When rate-limited, writes to local store immediately (local is authoritative)
- Queued operations use `Q<timestamp>` prefix until synced

**Storage location:**
```
.issues/
├── 42.md              # Mirrored GitHub issue #42
├── 100.md             # Mirrored GitHub issue #100
├── Q20260203123456.md # Queued issue (not yet synced)
└── _meta.json         # Metadata for local-only issues
```

**Disabling:**
```bash
# Disable write-through (not recommended)
AIT_WRITE_THROUGH=0 gh issue create --title "..."
```

### GitHub Apps (Rate Limit Scaling)

GitHub App installation access tokens have a primary REST API rate limit per installation.
GitHub documents this as a *minimum* of 5,000 requests/hour (15,000/hour for GitHub
Enterprise Cloud organizations). For non-Enterprise Cloud orgs, installations scale beyond
that minimum (+50 requests/hour per repo beyond 20, +50 requests/hour per org user beyond
20), capped at 12,500 requests/hour.

When `AIT_USE_GITHUB_APPS=1`, the gh wrapper injects app installation tokens instead of
using personal authentication, which reduces contention vs shared tokens.

**Architecture**: Three-tier lookup:

| Tier | Description | Examples |
|------|-------------|----------|
| 1. Per-repo apps | High-activity repos get dedicated apps | `z4-ai`, `zani-ai`, `dasher-ai` |
| 2. Per-director apps | Fallback by org director | `math-ai`, `lang-ai`, `tool-ai` |
| 3. User sessions | No injection, use normal `gh auth` | Interactive user sessions |

**Per-repo apps (18 dedicated apps):**
- ai_template, z4, dasher, zani, tla2, gamma-crown, sunder, certus
- dterm, kafka2, leadership, dashnews, dashboard, tMIR, LLVM2, tRust, mly, benchmarker

**Per-director apps (8 fallback apps):**

| App | Director | Covers |
|-----|----------|--------|
| `math-ai` | MATH | lean5, dashprove, zksolve, proverif-rs, galg |
| `ml-ai` | ML | model_mlx_migration, voice |
| `lang-ai` | LANG | tSwift, tC, rustc-index-verified |
| `tool-ai` | TOOL | dashterm2, dterm-alacritty, dashmap, codex_dashflow, etc. |
| `know-ai` | KNOW | sg, chunker, video_audio_extracts, dashextract, pdfium_fast |
| `rs-ai` | RS | claude_code_rs |
| `app-ai` | APP | dashpresent |
| `dbx-ai` | DBX | dbx_datacenter, dbx_unitq |

**Total primary capacity (floor)**: ~130K requests/hour (26 apps × 5,000/hr minimum).
Actual primary quota may be higher (depending on org plan and installation size), but
effective throughput can still be lower due to secondary rate limits.

**Secondary rate limits**: In addition to primary quotas, GitHub enforces secondary limits
(concurrency, per-endpoint points/minute, and CPU time). When the API returns 403/429
indicating a secondary limit, honor `Retry-After` when present; otherwise wait at least
60 seconds and use exponential backoff between retries.

**Setup**:
```bash
# One-time: Install dependencies and login
pip install playwright PyJWT pyyaml
playwright install chromium
python3 -m ai_template_scripts.gh_apps.setup login

# Create app for a project (fully automated after login)
python3 -m ai_template_scripts.gh_apps.setup create-app z4

# Verify token generation
python3 -m ai_template_scripts.gh_apps.get_token --repo ai_template
```

**Configuration**: `~/.ait_gh_apps/config.yaml`

```yaml
org: dropbox-ai-prototypes
apps:
  # Per-repo apps
  z4-ai:
    app_id: 123456
    installation_id: 789012
    private_key: ~/.ait_gh_apps/z4-ai.pem
    repo: z4
  # Per-director fallback apps
  math-ai:
    app_id: 234567
    installation_id: 890123
    private_key: ~/.ait_gh_apps/math-ai.pem
    repos: ["lean5", "dashprove", "zksolve", "proverif-rs", "galg"]
```

**Rate tracking**: When apps are enabled, `rate_state.json` tracks per-app quotas:
```bash
cat ~/.ait_gh_cache/rate_state.json | jq .apps
```

**Fallback behavior**:
- If token generation fails or returns empty, `GH_TOKEN` is not set and `gh` uses default
  `gh auth login` credentials automatically
- If a valid token is generated but the subsequent API call returns 401/403, there is no
  automatic retry with fallback credentials (you must clear the bad token manually)

**Repo-scoped tokens (least privilege)**:
Installation access tokens are scoped to the current repo at issuance time. This minimizes
blast radius if a token leaks (logs, env, file cache), since a director app's token won't
grant access to other repos in the director's domain. Token cache keys include repo to
prevent cross-repo reuse.

- Design: `designs/2026-02-03-gh-app-token-least-privilege.md`
- Implementation: `ai_template_scripts/gh_apps/token.py`

### Cache Location

```
~/.ait_gh_cache/
├── rate_state.json      # Rate limit state (quotas, reset times)
├── change_log.json      # Queued writes when rate-limited
├── historical/          # Persistent issue data cache
└── *.json               # TTL-cached API responses
```

### Bypassing

```bash
# Bypass wrapper entirely (direct to real gh)
gh --no-wrapper issue list

# Wrapper-specific help/version
gh --wrapper-help
gh --wrapper-version
```

## Cargo Wrapper

**Source**: `ai_template_scripts/bin/cargo`
**Python module**: `cargo_wrapper.py`

### Behavior

| Feature | Description |
|---------|-------------|
| Build serialization | Per-repo lock prevents concurrent builds (OOM protection) |
| Timeout management | 1-hour build timeout, 10-minute test timeout (configurable) |
| Progress output | Emits status every 60s to prevent looper silence timeout |
| Lock cleanup | Auto-cleans stale locks from dead processes |

### Serialized Commands

These commands go through `cargo_wrapper.py` for serialization:
- `build` / `b`
- `test` / `t`
- `check` / `c`
- `run` / `r`
- `doc` / `d`
- `clippy`
- `bench`
- `miri`

Other commands (`add`, `new`, `version`, etc.) pass directly to real cargo.

### Configuration

Config file: `cargo_wrapper.toml` or `.cargo_wrapper.toml` in repo root.

```toml
[timeouts]
build_sec = 3600    # 1 hour (default)
test_sec = 600      # 10 minutes (default)
kani_sec = 1800     # 30 minutes for Kani
```

| Variable | Default | Description |
|----------|---------|-------------|
| `AIT_REAL_CARGO` | auto-detected | Path to real cargo (set by wrapper) |

### Lock Location

```
~/.ait_cargo_lock/<repo>/
├── lock.pid         # Build lock PID
├── lock.json        # Build lock metadata
├── lock.test.pid    # Test lock PID
├── lock.test.json   # Test lock metadata
└── builds.log       # Build history log
```

### Bypassing

```bash
# Bypass wrapper entirely
cargo --no-wrapper build

# Wrapper-specific help/version
cargo --wrapper-help
cargo --wrapper-version
```

## Grep Wrapper

**Source**: `ai_template_scripts/bin/grep`

### Behavior

| Feature | Description |
|---------|-------------|
| Memory pressure detection | Checks system memory before recursive grep (macOS only) |
| Critical block | Blocks recursive grep when memory >85% used |
| Warning | Warns when memory 70-85% used, suggests ripgrep |

### Memory Pressure Levels

| Level | Memory Used | Action |
|-------|-------------|--------|
| Normal | <70% | Pass through |
| Warning | 70-85% | Warn, suggest `rg` alternative |
| Critical | >85% | Block with error |

### Configuration

No configuration variables. Memory thresholds are hardcoded.

### Bypassing

Use `rg` (ripgrep) directly - it's not wrapped and handles memory better.

## Troubleshooting

### Git Commit Lock Timeout

**Symptoms**: `Timed out waiting for git commit lock`
**Check**: Look for stale lock files
```bash
ls -la .git/ait_commit_lock/
```
**Solution**:
1. Check if another session is committing
2. If stale: `rm -rf .git/ait_commit_lock/`
3. Bypass: `AIT_GIT_LOCK_DISABLE=1 git commit -m "..."`

### GitHub Rate Limiting

**Symptoms**: `gh_rate_limit: REST fallback activated` or commands hang
**Check**:
```bash
cat ~/.ait_gh_cache/rate_state.json | jq .resources
```
**Solution**:
1. Wait for reset time (shown in rate_state.json)
2. Wrapper auto-switches between REST and GraphQL
3. Clear stale cache: `find ~/.ait_gh_cache -maxdepth 1 -name '*.json' ! -name 'rate_state.json' ! -name 'change_log.json' -delete`

### Cargo Build Lock Held

**Symptoms**: `cargo_wrapper: waiting for build lock`
**Check**:
```bash
./ai_template_scripts/cargo_lock_info.py
```
**Solution**:
1. Verify no legitimate build is running
2. Clear stale lock: `rm ~/.ait_cargo_lock/<repo>/lock.*`
3. Bypass: `cargo --no-wrapper build`

### Grep Blocked by Memory Pressure

**Symptoms**: `BLOCKED: System memory pressure is CRITICAL`
**Solution**: Use `rg` (ripgrep) instead - not wrapped and memory-efficient

### Branch Creation Blocked

**Symptoms**: `ERROR: Branch creation is blocked`
**Reason**: AI workers must work on main to avoid merge conflicts
**Solution**:
1. Commit small incremental changes to main
2. Use `zone/*` branches only for multi-machine mode

### Wrapper Not Intercepting Commands

**Symptoms**: System binary runs instead of wrapper
**Check**:
```bash
which git   # Should show ai_template_scripts/bin/git
echo $PATH  # Should have ai_template_scripts/bin first
```
**Solution**:
1. Verify looper started the session (sets PATH)
2. For login shells, add to `~/.zprofile`:
   ```bash
   export PATH="$HOME/<project>/ai_template_scripts/bin:$PATH"
   ```

## Design Principles

When adding new wrapper features:

1. **Transparent**: AIs shouldn't need to know about wrappers
2. **Fail-safe**: On error, prefer blocking over silent pass-through
3. **Bypassable**: Provide `--no-wrapper` or env var override
4. **Logged**: Emit progress for long operations (prevents silence timeout)
5. **Documented**: Update this file and add config to looper.md

### Adding a New Wrapper

1. Create shell script in `ai_template_scripts/bin/<command>`
2. Find real binary (skip this wrapper in PATH)
3. Implement feature with clear bypass mechanism
4. Add to this documentation
5. Test both direct and looper-spawned execution

## Related Documentation

- `docs/looper.md` - Looper infrastructure (installs wrappers)
- `docs/troubleshooting.md` - General troubleshooting
- `.claude/rules/ai_template.md` - GitHub API Management section
