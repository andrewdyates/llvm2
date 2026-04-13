// llvm2-opt - Verified optimizations
//
// Author: Andrew Yates <ayates@dropbox.com>
// Copyright 2026 Dropbox, Inc. | License: Apache-2.0

//! Verified optimization passes for LLVM2.
//!
//! Each optimization is proven to preserve semantics using z4.
//!
//! # Architecture
//!
//! ```text
//! PassManager { passes: [ConstFold, CopyProp, Peephole, DCE] }
//!     │
//!     ├── run_once(func)              → single pass
//!     └── run_to_fixpoint(func, max)  → iterate until stable
//! ```
//!
//! # Passes
//!
//! | Pass | Description |
//! |------|-------------|
//! | [`DeadCodeElimination`] | Remove instructions whose defs are unused |
//! | [`ConstantFolding`] | Evaluate constant expressions at compile time |
//! | [`CopyPropagation`] | Replace uses of `mov dst, src` with `src` |
//! | [`Peephole`] | AArch64-specific instruction simplification |
//!
//! # Memory Effects Model
//!
//! The [`effects`] module classifies each opcode as Pure, Load, Store,
//! or Call. This is used by DCE and will be used by future CSE/LICM passes.
//!
//! # Usage
//!
//! ```rust,no_run
//! use llvm2_opt::pipeline::{OptimizationPipeline, OptLevel};
//! use llvm2_ir::MachFunction;
//!
//! // Build and run at O2:
//! // let pipeline = OptimizationPipeline::new(OptLevel::O2);
//! // let stats = pipeline.run(&mut func);
//! ```

pub mod const_fold;
pub mod copy_prop;
pub mod dce;
pub mod effects;
pub mod pass_manager;
pub mod passes;
pub mod peephole;
pub mod pipeline;

// Re-export the most important types at crate root.
pub use pass_manager::{MachinePass, PassManager, PassStats};
pub use pipeline::{OptLevel, OptimizationPipeline};
