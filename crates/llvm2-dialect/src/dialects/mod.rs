// llvm2-dialect - Sample dialects (verif, tmir, machir) and conversions
//
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Copyright 2026 Dropbox, Inc. | License: Apache-2.0

//! Proof-of-concept dialects used by the end-to-end lowering test:
//!
//! * [`verif`] — a verification-layer dialect with one op
//!   (`verif.fingerprint_batch_stub`).
//! * [`tmir`] — a small subset of tMIR ops (`tmir.add`, `tmir.xor`,
//!   `tmir.const`, `tmir.ret`).
//! * [`machir`] — a small subset of MachIR ops mapping to `AArch64Opcode`.
//! * [`conversions`] — `VerifToTmir` and `TmirToMachir` [`ConversionPattern`]s.

pub mod conversions;
pub mod machir;
pub mod tmir;
pub mod verif;

pub use conversions::{TmirToMachir, VerifToTmir, register_all};
