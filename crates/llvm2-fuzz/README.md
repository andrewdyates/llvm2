# llvm2-fuzz

Differential fuzzing harness for LLVM2. Part of **WS3** (workstream 3 —
differential fuzzing) in the "proving LLVM2 replaces LLVM" plan.

Three drivers, each a workspace binary:

| Binary            | Oracle                          | Status (2026-04-19)             |
|-------------------|---------------------------------|---------------------------------|
| `tmir-gen`        | tMIR interpreter (in-process)   | **Live** — runs today           |
| `csmith-driver`   | clang -O0                        | Unavailable: needs `brew install csmith` + WS2 (C->tMIR) |
| `yarpgen-driver`  | clang -O0                        | Unavailable: build from source + WS2 (C->tMIR)           |

## Running a campaign

```
scripts/fuzz_campaign.sh --driver tmir-gen --duration 300 --out evals/results/fuzz/$(date +%F)
```

Arguments:
- `--driver`: one of `tmir-gen | csmith-driver | yarpgen-driver`
- `--duration`: seconds to run before writing the JSON (default 300)
- `--out`: directory where `<driver>.json` and repros will be written

## Output schema

Every driver writes `<driver>.json` in the out directory:

```jsonc
{
  "driver": "tmir-gen",
  "status": "ok",               // or "unavailable"
  "reason": null,               // human string when status != "ok"
  "duration_secs": 300,
  "runs": 4213057,
  "timeouts": 0,
  "crashes": 0,
  "miscompiles": 0,
  "repros": [
    { "seed": 1234, "minimized_input_path": "...", "summary": "..." }
  ],
  "started_at": "2026-04-19T21:55:00Z",
  "finished_at": "2026-04-19T22:00:00Z"
}
```

## How tmir-gen works

For each iteration:
1. Pick a PRNG seed (SplitMix64).
2. Generate a valid tMIR module: one function with N i64 params and M
   random binops over the existing value set.
3. Call the tMIR interpreter (`llvm2_codegen::interpreter::interpret`) on
   a panel of sample inputs (well-known extremes + random).
4. Translate tMIR → `llvm2-lower::Function` and compile the function at
   both `O0` and `O2` via `Pipeline::compile_function`.
5. Miscompile signals:
   - **Interpreter panic** on generated tMIR → logged as a crash.
   - **Compiler panic** on valid tMIR → logged as a crash.
   - **O0/O2 compilation-success divergence** → logged as a miscompile
     (definite optimizer or legalization bug).

This is a conservative oracle. It does **not** JIT-execute compiled code
and compare against the interpreter — that requires aarch64-host-only
setup and careful ABI handling, and is a follow-up (see the WS3 issue).

## Adding a miscompile auto-filer

Set `LLVM2_AUTOFILE=1` in the environment before running:

```
LLVM2_AUTOFILE=1 scripts/fuzz_campaign.sh --driver tmir-gen --duration 300 --out ...
```

When a miscompile is detected, the driver calls
scripts/file_miscompile_issue.sh with `(driver, repro_path)`. That
script opens a GitHub issue labeled `miscompile` `bug` `P1` with the
repro path and reproduction commands.

Auto-filing is **off by default** so first campaigns don't spam issues.

## Unblocking the external drivers

`csmith-driver` and `yarpgen-driver` need:
1. The external generator on `PATH` (see install hints in the JSON
   `reason` field).
2. A C → tMIR importer (workstream WS2 in the "proving" plan). Once
   that lands, update the drivers to shell out to the importer and then
   `llvm2-cli` at each opt level, running both binaries and diffing
   stdout + exit code.

## Going from 5-minute to 1-hour campaigns

The MVP caps each driver at 5 minutes so agent sessions complete. The
known blockers for scaling up:

- **Repro storage.** We currently save up to 32 repros per campaign as
  individual JSON files. A 1h campaign may find hundreds — switch to
  dedup + minimization before storage.
- **Coverage feedback.** No coverage-guided generation yet; all seeds
  are unweighted. A 1h campaign spends most time revisiting the same
  shapes. Wire in line/branch coverage or use AFL++-style evolutionary
  scheduling.
- **JIT execution.** Without JIT-run-and-compare, the oracle is weak.
  The follow-up is to JIT the compiled function and compare its result
  against the interpreter on the same inputs.

All three are WS3 follow-ups, tracked on the main WS3 issue.
