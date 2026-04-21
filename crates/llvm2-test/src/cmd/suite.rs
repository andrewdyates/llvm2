// WS2 — llvm-test-suite SingleSource external corpus runner.
//
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Copyright 2026 Dropbox, Inc. | License: Apache-2.0

//! `llvm2-test suite` — external LLVM test-suite runner.
//!
//! See `docs/testing/suite.md` and WS2 in
//! `designs/2026-04-19-proving-llvm2-replaces-llvm.md`.

use clap::Args;

use super::{GlobalArgs, not_yet_implemented};
use crate::results::ResultStatus;

/// Arguments for `llvm2-test suite`.
#[derive(Args, Debug, Clone)]
#[command(
    long_about = "Run the external llvm-test-suite SingleSource corpus (WS2).\n\n\
                  clang `-S -emit-llvm` -> `llvm2-llvm-import` (library) -> \
                  tMIR -> LLVM2 -> link -> run, diff stdout/stderr against \
                  `clang -O0` golden. Writes \
                  `evals/results/suite/<iso-date>.json`.\n\n\
                  # Examples\n\n  \
                  llvm2-test suite --filter 'SingleSource/UnitTests/*.c' --format human\n  \
                  llvm2-test suite --optlevel 2 --format json\n  \
                  llvm2-test suite --clone-corpus"
)]
pub struct SuiteArgs {
    /// Glob filter over test-suite source paths.
    #[arg(long, value_name = "GLOB")]
    pub filter: Option<String>,

    /// Optimization level for the LLVM2 compile (0-3).
    #[arg(long, value_name = "N", default_value_t = 0)]
    pub optlevel: u8,

    /// Clone `llvm-test-suite` into `~/llvm-test-suite-ref/` if missing.
    #[arg(long)]
    pub clone_corpus: bool,
}

/// Entry point. Stub until WS2 lands.
pub fn run(_global: &GlobalArgs, _args: &SuiteArgs) -> anyhow::Result<ResultStatus> {
    not_yet_implemented("suite")
}
