// WS4 — rustc_codegen_llvm2 + rustc UI harness.
//
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Copyright 2026 Dropbox, Inc. | License: Apache-2.0

//! `llvm2-test rustc` — drive `rustc_codegen_llvm2`.
//!
//! See `docs/testing/rustc.md` and WS4 in
//! `designs/2026-04-19-proving-llvm2-replaces-llvm.md`.

use clap::{Args, Subcommand};

use super::{GlobalArgs, not_yet_implemented};
use crate::results::ResultStatus;

/// Subcommand selector.
#[derive(Subcommand, Debug, Clone)]
pub enum RustcCommand {
    /// Compile and run a trivial hello-world via LLVM2.
    Smoke,
    /// Run the full rustc UI harness against LLVM2.
    Ui,
    /// Print rustc-MIR opcode coverage of the `tmir-from-rustc-mir` adapter.
    FeatureCoverage,
}

/// Arguments for `llvm2-test rustc`.
#[derive(Args, Debug, Clone)]
#[command(
    long_about = "Drive `rustc_codegen_llvm2` + rustc UI tests (WS4).\n\n\
                  `rustc smoke` sanity-compiles `hello.rs`. `rustc ui` runs \
                  the full UI harness and writes a per-test JSON record. \
                  `rustc feature-coverage` reports which rustc-MIR opcodes \
                  the adapter currently translates.\n\n\
                  # Examples\n\n  \
                  llvm2-test rustc smoke\n  \
                  llvm2-test rustc ui --format json --out evals/results/rustc/2026-04-19.json\n  \
                  llvm2-test rustc feature-coverage --format human"
)]
pub struct RustcArgs {
    /// Subcommand.
    #[command(subcommand)]
    pub cmd: RustcCommand,
}

/// Entry point. Stub until WS4 lands.
pub fn run(_global: &GlobalArgs, _args: &RustcArgs) -> anyhow::Result<ResultStatus> {
    not_yet_implemented("rustc")
}
