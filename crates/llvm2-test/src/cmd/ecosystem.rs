// WS6 — top-100 crates.io cargo-test smoke.
//
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Copyright 2026 Dropbox, Inc. | License: Apache-2.0

//! `llvm2-test ecosystem` — crates.io smoke via `-Zcodegen-backend=llvm2`.

use clap::Args;

use super::{GlobalArgs, not_yet_implemented};
use crate::results::ResultStatus;

/// Arguments for `llvm2-test ecosystem`.
#[derive(Args, Debug, Clone)]
#[command(
    long_about = "Run `cargo test` on top-N crates.io crates (WS6).\n\n\
                  For each crate: pin a version, fetch into the cache, \
                  `cargo test -Zcodegen-backend=llvm2`, record outcome. \
                  Writes `evals/results/ecosystem/<iso-date>.json`.\n\n\
                  # Examples\n\n  \
                  llvm2-test ecosystem --top 100\n  \
                  llvm2-test ecosystem --crate serde --format json"
)]
pub struct EcosystemArgs {
    /// Number of top crates to test.
    #[arg(long, value_name = "N", default_value_t = 100)]
    pub top: u32,

    /// Test a single crate by name.
    #[arg(long, value_name = "NAME")]
    pub crate_name: Option<String>,

    /// Directory for cloned sources; defaults to `~/.llvm2-test/ecosystem-cache/`.
    #[arg(long, value_name = "PATH")]
    pub cache_dir: Option<std::path::PathBuf>,
}

/// Entry point. Stub until WS6 lands.
pub fn run(_global: &GlobalArgs, _args: &EcosystemArgs) -> anyhow::Result<ResultStatus> {
    not_yet_implemented("ecosystem")
}
