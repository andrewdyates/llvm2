// WS3 — differential fuzzers (csmith, yarpgen, tmir-gen).
//
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Copyright 2026 Andrew Yates | License: Apache-2.0

//! `llvm2-test fuzz` — differential fuzzing.
//!
//! See `docs/testing/fuzz.md` and WS3 in
//! `designs/2026-04-19-proving-llvm2-replaces-llvm.md`.

use clap::{Args, ValueEnum};

use super::{GlobalArgs, not_yet_implemented};
use crate::results::ResultStatus;

/// Which fuzz driver to run.
#[derive(Clone, Copy, Debug, ValueEnum)]
#[value(rename_all = "lower")]
pub enum Driver {
    /// csmith random C program generator.
    Csmith,
    /// YARPGen random C/C++ program generator.
    Yarpgen,
    /// In-tree tMIR random IR generator.
    TmirGen,
    /// Run all drivers for the requested duration.
    All,
}

/// Arguments for `llvm2-test fuzz`.
#[derive(Args, Debug, Clone)]
#[command(
    long_about = "Run differential fuzzers (WS3).\n\n\
                  Each driver generates a program, compiles it with the \
                  oracle (clang / rustc-LLVM) and LLVM2, executes both, \
                  and diffs results. Miscompiles are minimized with \
                  `creduce` and auto-filed as GitHub issues labeled \
                  `miscompile` `bug` `P1`. Writes \
                  `evals/results/fuzz/<iso-date>/<driver>.json`.\n\n\
                  # Examples\n\n  \
                  llvm2-test fuzz --driver csmith --duration 10m\n  \
                  llvm2-test fuzz --driver all --duration 1h --format json\n  \
                  llvm2-test fuzz --driver yarpgen --seeds 1000 --reduce"
)]
pub struct FuzzArgs {
    /// Which driver to run.
    #[arg(long, value_enum, default_value_t = Driver::All)]
    pub driver: Driver,

    /// Wall-clock duration (humantime, e.g. `10s`, `1h`, `24h`).
    #[arg(long, value_name = "DUR", default_value = "10m")]
    pub duration: String,

    /// Number of seeds to run (cap; earlier of duration or seeds wins).
    #[arg(long, value_name = "N")]
    pub seeds: Option<u64>,

    /// Optimization level used when compiling generated programs.
    #[arg(long, value_name = "N", default_value_t = 2)]
    pub optlevel: u8,

    /// Run creduce on every miscompile before filing.
    #[arg(long)]
    pub reduce: bool,
}

/// Entry point. Stub until WS3 lands.
pub fn run(_global: &GlobalArgs, _args: &FuzzArgs) -> anyhow::Result<ResultStatus> {
    not_yet_implemented("fuzz")
}
