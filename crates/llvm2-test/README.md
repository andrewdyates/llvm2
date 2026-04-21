# `llvm2-test` — unified LLVM2 test & verification CLI

One Rust binary. No shell scripts, no Python, no Makefile.

Design reference:
designs/2026-04-19-proving-llvm2-replaces-llvm.md

## Install

From a repo checkout:

```
cargo build -p llvm2-test --release
./target/release/llvm2-test --help
```

## Quick start

```sh
# Environment check — tells you what tools are missing for each WS.
llvm2-test doctor
llvm2-test doctor --for fuzz --format json

# Render this week's dashboard from whatever results are on disk.
llvm2-test report

# Ratchets — called by CI. Fails non-zero on regression.
llvm2-test ratchet shell-isolation
llvm2-test ratchet schema

# Each workstream has its own subcommand (some are WS0 stubs):
llvm2-test matrix    --format human    # WS1
llvm2-test suite                       # WS2
llvm2-test fuzz      --driver csmith   # WS3
llvm2-test rustc     smoke             # WS4
llvm2-test bootstrap --stage 1         # WS5
llvm2-test ecosystem --top 100         # WS6
llvm2-test prove     --width w8        # WS7
llvm2-test pipeline  regalloc          # WS8
```

## Command reference

<!-- BEGIN: help -->

The authoritative version of this section is printed by
`llvm2-test --help`. A future CI job will overwrite the block between
these markers from `clap`'s output to prevent drift. For now the full
reference lives under `docs/testing/`.

### Global flags

Present on every subcommand:

| Flag | Description |
|---|---|
| `-f`, `--format <fmt>` | Output format: `human` \| `json` \| `junit` |
| `-o`, `--out <path>` | Write machine-readable result artifact here |
| `--config <path>` | Config file (TOML) |
| `--timeout <secs>` | Per-unit timeout |
| `--parallel <n>` | Worker count (honors cargo-lock) |
| `-q`, `--quiet` | Suppress progress bars |
| `-v`, `--verbose` | Verbose logging (repeat for debug/trace) |
| `--no-cache` | Ignore proof / corpus caches |
| `--dry-run` | Print what would run; don't invoke external tools |

### Subcommands

| Subcommand | Purpose | Page |
|---|---|---|
| `matrix` | Workspace unit/integration test matrix (WS1) | docs/testing/matrix.md |
| `suite` | `llvm-test-suite` corpus runner (WS2) | docs/testing/suite.md |
| `fuzz` | csmith / yarpgen / tmir-gen differential fuzzers (WS3) | docs/testing/fuzz.md |
| `rustc` | `rustc_codegen_llvm2` + rustc UI harness (WS4) | docs/testing/rustc.md |
| `bootstrap` | rustc self-host via LLVM2 (WS5) | docs/testing/bootstrap.md |
| `ecosystem` | top-100 crates.io `cargo test` (WS6) | docs/testing/ecosystem.md |
| `prove` | z4 lowering obligations (WS7) | docs/testing/prove.md |
| `pipeline` | RA / scheduler / emit proofs (WS8) | docs/testing/pipeline.md |
| `report` | Weekly dashboard generator (WS9) | docs/testing/report.md |
| `ratchet` | CI invariants (shell isolation, schema drift) | |
| `doctor` | Environment check | |

<!-- END: help -->

## Exit codes

| Code | Meaning |
|---:|---|
| `0` | all units passed |
| `1` | at least one unit failed (user-level) |
| `2` | environment broken / tool missing / not-yet-implemented |
| `64` | invalid argv |
| `70` | internal bug (panic caught at `main`) |

## Layout

See docs/testing/architecture.md.

## Adding a subcommand

See docs/testing/architecture.md.
