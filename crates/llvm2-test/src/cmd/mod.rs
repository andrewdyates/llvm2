// Subcommand dispatcher for `llvm2-test`.
//
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Copyright 2026 Dropbox, Inc. | License: Apache-2.0

//! Per-workstream subcommand modules.
//!
//! Each WS-N workstream's runnable surface lives under `cmd/<ws>.rs`.
//! Stubs here return `ResultStatus::NotImplemented` (exit 2) until the
//! respective workstream fills them in.

use std::path::PathBuf;

use crate::OutputFormat;
use crate::results::ResultStatus;

pub mod bootstrap;
pub mod doctor;
pub mod ecosystem;
pub mod fuzz;
pub mod lint_linux;
pub mod matrix;
pub mod pipeline;
pub mod prove;
pub mod ratchet;
pub mod report;
pub mod rustc;
pub mod suite;

/// Global arguments present on every subcommand. Built from the
/// top-level `Cli` struct in `main.rs` and passed to each `run()`.
#[derive(Clone, Debug)]
pub struct GlobalArgs {
    /// Output format for the human-facing view and the exit-code story.
    pub format: OutputFormat,
    /// If set, write the machine-readable artifact here.
    pub out: Option<PathBuf>,
    /// Config file path override.
    pub config: Option<PathBuf>,
    /// Per-unit timeout override, in seconds.
    pub timeout: Option<u64>,
    /// Worker count override.
    pub parallel: Option<usize>,
    /// Suppress progress bars.
    pub quiet: bool,
    /// Verbosity counter (0 = info, 1 = debug, 2+ = trace).
    pub verbose: u8,
    /// Ignore caches for this run.
    pub no_cache: bool,
    /// Print-only; do not invoke external tools.
    pub dry_run: bool,
}

impl GlobalArgs {
    /// Is JSON output requested?
    #[must_use]
    pub fn is_json(&self) -> bool {
        matches!(self.format, OutputFormat::Json | OutputFormat::Junit)
    }
}

/// Default "not yet implemented" path used by WS1-WS9 stubs.
///
/// Prints a human-readable pointer to the design doc and returns
/// `ResultStatus::NotImplemented` (exit 2).
pub(crate) fn not_yet_implemented(cmd: &str) -> anyhow::Result<ResultStatus> {
    eprintln!(
        "subcommand `{cmd}`: not yet implemented - see \
         `designs/2026-04-19-proving-llvm2-replaces-llvm.md` and \
         `docs/testing/{cmd}.md`"
    );
    Ok(ResultStatus::NotImplemented)
}
