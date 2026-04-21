// WS8 — prove the rest of the pipeline: RA, scheduler, Mach-O writer.
//
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Copyright 2026 Dropbox, Inc. | License: Apache-2.0

//! `llvm2-test pipeline` — non-ISel proof drivers.

use clap::{Args, Subcommand};

use super::{GlobalArgs, not_yet_implemented};
use crate::results::ResultStatus;

/// Which stage's proof to discharge.
#[derive(Subcommand, Debug, Clone)]
pub enum PipelineCommand {
    /// Register-allocator correctness theorem.
    Regalloc,
    /// Scheduler correctness theorem.
    Schedule,
    /// Mach-O writer round-trip theorem.
    Emit,
}

/// Arguments for `llvm2-test pipeline`.
#[derive(Args, Debug, Clone)]
#[command(
    long_about = "Prove non-ISel stages (WS8): register allocator, \
                  scheduler, Mach-O emitter. Writes \
                  `evals/results/pipeline/<stage>/<iso-date>.json`.\n\n\
                  # Examples\n\n  \
                  llvm2-test pipeline regalloc\n  \
                  llvm2-test pipeline schedule --format json\n  \
                  llvm2-test pipeline emit"
)]
pub struct PipelineArgs {
    /// Which stage to prove.
    #[command(subcommand)]
    pub cmd: PipelineCommand,
}

/// Entry point. Stub until WS8 lands.
pub fn run(_global: &GlobalArgs, _args: &PipelineArgs) -> anyhow::Result<ResultStatus> {
    not_yet_implemented("pipeline")
}
