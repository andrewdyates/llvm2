// WS1 — workspace unit/integration test matrix.
//
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Copyright 2026 Andrew Yates | License: Apache-2.0

//! `llvm2-test matrix` — run the workspace unit-test matrix.
//!
//! See `docs/testing/matrix.md` and WS1 in
//! `designs/2026-04-19-proving-llvm2-replaces-llvm.md`.

use clap::Args;

use super::{GlobalArgs, not_yet_implemented};
use crate::results::ResultStatus;

/// Arguments for `llvm2-test matrix`.
#[derive(Args, Debug, Clone)]
#[command(
    long_about = "Run the workspace unit/integration test matrix (WS1).\n\n\
                  Shards `llvm2-codegen` by target (aarch64, x86_64, macho, \
                  elf, riscv, jit, misc) so no shard exceeds 20 min. Writes \
                  `evals/results/matrix/<iso-date>.json` with per-crate \
                  `{passed, failed, ignored, time_s}`. `ignored` must be 0 \
                  (rule #341) or the run exits 1.\n\n\
                  # Examples\n\n  \
                  llvm2-test matrix --format human\n  \
                  llvm2-test matrix --crate llvm2-codegen --shard aarch64 --format json\n  \
                  llvm2-test matrix --compare ratchet/tests_baseline.json"
)]
pub struct MatrixArgs {
    /// Only run the named crate (e.g. `llvm2-codegen`).
    #[arg(long, value_name = "CRATE")]
    pub crate_filter: Option<String>,

    /// Shard of a crate to run (e.g. `aarch64`, `macho`). Requires `--crate`.
    #[arg(long, value_name = "NAME")]
    pub shard: Option<String>,

    /// Only run tests changed since the given git ref.
    #[arg(long, value_name = "REF")]
    pub since: Option<String>,

    /// Baseline JSON file to diff the run against.
    #[arg(long, value_name = "PATH")]
    pub compare: Option<std::path::PathBuf>,
}

/// Entry point. Stub until WS1 lands its implementation.
pub fn run(_global: &GlobalArgs, _args: &MatrixArgs) -> anyhow::Result<ResultStatus> {
    not_yet_implemented("matrix")
}
