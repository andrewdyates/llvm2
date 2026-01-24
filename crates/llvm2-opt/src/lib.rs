// llvm2-opt - Verified optimizations
//
// Author: Andrew Yates <ayates@dropbox.com>
// Copyright 2026 Dropbox, Inc. | License: Apache-2.0

//! Verified optimization passes for LLVM2.
//!
//! Each optimization is proven to preserve semantics using z4.
//!
//! Optimizations include:
//! - Constant folding
//! - Dead code elimination
//! - Peephole optimizations
//! - Common subexpression elimination

pub mod passes;
pub mod pipeline;

// pub use pipeline::OptimizationPipeline;
