// llvm2-opt - Optimization pipeline
//
// Author: Andrew Yates <ayates@dropbox.com>
// Copyright 2026 Dropbox, Inc. | License: Apache-2.0

//! Optimization pipeline configuration and execution.

/// Optimization level.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OptLevel {
    /// No optimizations (fastest compile).
    O0,
    /// Basic optimizations.
    O1,
    /// Standard optimizations.
    O2,
    /// Aggressive optimizations.
    O3,
    /// Size optimization.
    Os,
}

/// Optimization pipeline configuration.
pub struct OptimizationPipeline {
    pub level: OptLevel,
}
