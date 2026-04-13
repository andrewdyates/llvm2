// llvm2-opt - Optimization passes
//
// Author: Andrew Yates <ayates@dropbox.com>
// Copyright 2026 Dropbox, Inc. | License: Apache-2.0

//! Individual optimization passes, each verified for correctness.
//!
//! This module re-exports the pass implementations from their dedicated
//! submodules for convenient access.

pub use crate::cfg_simplify::CfgSimplify;
pub use crate::const_fold::ConstantFolding;
pub use crate::copy_prop::CopyPropagation;
pub use crate::cse::CommonSubexprElim;
pub use crate::dce::DeadCodeElimination;
pub use crate::licm::LoopInvariantCodeMotion;
pub use crate::peephole::Peephole;
pub use crate::proof_opts::ProofOptimization;
