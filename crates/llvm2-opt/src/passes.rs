// llvm2-opt - Optimization passes
//
// Author: Andrew Yates <ayates@dropbox.com>
// Copyright 2026 Dropbox, Inc. | License: Apache-2.0

//! Individual optimization passes, each verified for correctness.

/// Constant folding pass - evaluates constant expressions at compile time.
pub struct ConstantFolding;

/// Dead code elimination - removes unreachable code.
pub struct DeadCodeElimination;

/// Peephole optimizations - local instruction pattern matching.
pub struct Peephole;

/// Common subexpression elimination.
pub struct CSE;
