// WS7 — discharge z4 lowering obligations.
//
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Copyright 2026 Dropbox, Inc. | License: Apache-2.0

//! `llvm2-test prove` — z4 lowering-obligation driver.

use clap::{Args, ValueEnum};

use super::{GlobalArgs, not_yet_implemented};
use crate::results::ResultStatus;

/// Bitwidth target for the prove run.
#[derive(Clone, Copy, Debug, ValueEnum)]
#[value(rename_all = "lower")]
pub enum Width {
    /// 8-bit.
    W8,
    /// 16-bit.
    W16,
    /// 32-bit.
    W32,
    /// 64-bit.
    W64,
    /// Parametric width (BV theory).
    Parametric,
}

/// Arguments for `llvm2-test prove`.
#[derive(Args, Debug, Clone)]
#[command(
    long_about = "Discharge z4 lowering obligations (WS7).\n\n\
                  Drives the z4 SMT solver against the obligation set \
                  produced by `llvm2-verify`. Timeout counts as a \
                  failure. Writes \
                  `evals/results/prove/<iso-date>.json`.\n\n\
                  # Examples\n\n  \
                  llvm2-test prove --width w8\n  \
                  llvm2-test prove --width w32 --obligation 'aarch64::*'\n  \
                  llvm2-test prove --width parametric --timeout-per-query 600"
)]
pub struct ProveArgs {
    /// Bitwidth to prove at.
    #[arg(long, value_enum, default_value_t = Width::W8)]
    pub width: Width,

    /// Glob over obligation names.
    #[arg(long, value_name = "GLOB")]
    pub obligation: Option<String>,

    /// Per-query timeout in seconds.
    #[arg(long, value_name = "SECS", default_value_t = 120)]
    pub timeout_per_query: u64,

    /// Proof cache directory.
    #[arg(long, value_name = "PATH")]
    pub cache_dir: Option<std::path::PathBuf>,
}

/// Entry point. Stub until WS7 lands.
pub fn run(_global: &GlobalArgs, _args: &ProveArgs) -> anyhow::Result<ResultStatus> {
    not_yet_implemented("prove")
}
