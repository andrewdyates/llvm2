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
//! PassManager { passes: [ConstFold, CopyProp, CSE, LICM, Peephole, DCE, CfgSimplify] }
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
//! | [`CommonSubexprElim`] | Eliminate redundant computations (dominator-based) |
//! | [`LoopInvariantCodeMotion`] | Hoist loop-invariant computations to preheader |
//! | [`ProofOptimization`] | Consume tMIR proof annotations to eliminate runtime checks |
//! | [`AddrModeFormation`] | Fold ADD+LDR/STR into rich AArch64 addressing modes |
//! | [`CmpSelectCombine`] | Diamond CFG to CSEL/CSET conditional select formation |
//! | [`CfgSimplify`] | Simplify CFG: branch folding, empty block elim, unreachable removal |
//! | [`const_materialize`] | Optimal constant materialization (MOVZ/MOVK, logical imm, MOVN) |
//!
//! # Memory Effects Model
//!
//! The [`effects`] module classifies each opcode as Pure, Load, Store,
//! or Call. This is used by DCE, CSE, and LICM to ensure safety.
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

pub mod addr_mode;
pub mod cfg_simplify;
pub mod cmp_select;
pub mod const_fold;
pub mod const_materialize;
pub mod copy_prop;
pub mod cse;
pub mod dce;
pub mod dom;
pub mod effects;
pub mod licm;
pub mod loops;
pub mod pass_manager;
pub mod passes;
pub mod peephole;
pub mod pipeline;
pub mod proof_opts;
pub mod scheduler;
pub mod vectorize;

// Re-export the most important types at crate root.
pub use pass_manager::{MachinePass, PassManager, PassStats};
pub use pipeline::{OptLevel, OptimizationPipeline};
