// WS5 — bootstrap rustc with LLVM2.
//
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Copyright 2026 Dropbox, Inc. | License: Apache-2.0

//! `llvm2-test bootstrap` — stage-1 / stage-2 rustc self-host via LLVM2.
//!
//! See `docs/testing/bootstrap.md` and WS5 in
//! `designs/2026-04-19-proving-llvm2-replaces-llvm.md`.

use clap::Args;

use super::{GlobalArgs, not_yet_implemented};
use crate::results::ResultStatus;

/// Arguments for `llvm2-test bootstrap`.
#[derive(Args, Debug, Clone)]
#[command(
    long_about = "Bootstrap rustc with LLVM2 (WS5). Contingent on WS4 UI \
                  >=95%.\n\n\
                  Drives `x.py build --codegen-backends=llvm2` against a \
                  checked-out rustc tree. Writes a JSON record including \
                  stage reached, wall-clock time, compile-time ratio vs \
                  LLVM.\n\n\
                  # Examples\n\n  \
                  llvm2-test bootstrap --stage 1\n  \
                  llvm2-test bootstrap --stage 2 --rustc-src ~/rustc-ref"
)]
pub struct BootstrapArgs {
    /// Which rustc bootstrap stage to target.
    #[arg(long, value_name = "N", default_value_t = 1)]
    pub stage: u8,

    /// Path to a rustc source checkout.
    #[arg(long, value_name = "PATH")]
    pub rustc_src: Option<std::path::PathBuf>,
}

/// Entry point. Stub until WS5 lands.
pub fn run(_global: &GlobalArgs, _args: &BootstrapArgs) -> anyhow::Result<ResultStatus> {
    not_yet_implemented("bootstrap")
}
