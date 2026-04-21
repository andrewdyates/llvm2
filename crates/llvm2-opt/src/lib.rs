// llvm2-opt - Verified optimizations
//
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Copyright 2026 Andrew Yates | License: Apache-2.0

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
//! | [`GlobalValueNumbering`] | Value-number-based redundancy elimination with load numbering |
//! | [`LoopInvariantCodeMotion`] | Hoist loop-invariant computations to preheader |
//! | [`ProofOptimization`] | Consume tMIR proof annotations to eliminate runtime checks |
//! | [`AddrModeFormation`] | Fold ADD+LDR/STR into rich AArch64 addressing modes |
//! | [`CmpSelectCombine`] | Diamond CFG to CSEL/CSET conditional select formation |
//! | [`IfConversion`] | General diamond/triangle CFG to CSEL/CSINC/CSNEG |
//! | [`CmpBranchFusion`] | Fuse CMP/TST + BCond into CBZ/CBNZ/TBZ/TBNZ |
//! | [`TailCallOptimization`] | Replace tail calls with branches to eliminate stack growth |
//! | [`VectorizationPass`](vectorize::VectorizationPass) | NEON auto-vectorization: scalar loops to SIMD |
//! | [`CfgSimplify`] | Simplify CFG: branch folding, empty block elim, unreachable removal |
//! | [`const_materialize`] | Optimal constant materialization (MOVZ/MOVK, logical imm, MOVN) |
//!
//! # Memory Effects Model
//!
//! The [`effects`] module classifies each opcode as Pure, Load, Store,
//! or Call. This is used by DCE, CSE, GVN, and LICM to ensure safety.
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
pub mod cache;
pub mod cfg_simplify;
pub mod cmp_branch_fusion;
pub mod cmp_select;
pub mod const_fold;
pub mod const_materialize;
pub mod copy_prop;
pub mod if_convert;
pub mod cse;
pub mod dce;
pub mod dom;
pub mod effects;
pub mod gvn;
pub mod inline;
pub mod interfaces;
pub mod licm;
pub mod loop_unroll;
pub mod loops;
pub mod pass_manager;
pub mod passes;
pub mod peephole;
pub mod pgo;
pub mod pipeline;
pub mod proof_opts;
pub mod rewrite;
pub mod scheduler;
pub mod sroa;
pub mod strength_reduce;
pub mod tail_call;
pub mod vectorize;

// Re-export the most important types at crate root.
pub use cache::{
    CACHE_KEY_VERSION, CacheBackend, CacheKey, CacheStats, FileCache, InMemoryCache, STABLE_HASH_SEED,
    STABLE_HASH_SEED_HI, StableHasher, StatsCache, stable_hash,
};
pub use interfaces::{DivergenceClass, OpInterfaces};
pub use pass_manager::{AnalysisCache, MachinePass, PassManager, PassStats};
pub use pgo::{
    CounterInjectionPass, CounterMap, CounterSite, PipelineConfig, ProfData, ProfDataError,
    build_profdata_from_counters, inject_block_counters,
};
pub use pipeline::{OptLevel, OptimizationPipeline};
pub use rewrite::{
    DeclarativeRewritePass, RewriteAction, RewriteEngine, RewriteStats, Rule, RuleBuilder,
};
pub use sroa::ScalarReplacementOfAggregates;
