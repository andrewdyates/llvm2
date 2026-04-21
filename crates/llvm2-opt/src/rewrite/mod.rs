// llvm2-opt - Declarative pattern rewrite framework
//
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Copyright 2026 Dropbox, Inc. | License: Apache-2.0

//! Declarative pattern rewrite framework (PDL-style).
//!
//! Rewrite rules are declared as composable pieces: a [`Matcher`] selects
//! candidate instructions, a stack of [`Constraint`]s filters them by
//! semantic predicate, and a [`Rewriter`] produces the replacement
//! [`RewriteAction`]. A [`RewriteEngine`] drives the rule set to a fixed
//! point over a [`MachFunction`].
//!
//! See `designs/2026-04-18-rewrite-and-interfaces.md` for the design.
//!
//! ```text
//! Engine → Rule → Matcher + Constraints + Rewriter
//!            │
//!            └── benefit: i32 (higher wins on conflict)
//! ```
//!
//! # Example
//!
//! ```
//! use llvm2_opt::rewrite::{patterns, RewriteEngine};
//! use llvm2_ir::{MachFunction, Signature};
//!
//! let mut func = MachFunction::new("demo".into(), Signature::new(vec![], vec![]));
//! let mut engine = RewriteEngine::new();
//! patterns::register_migrated(&mut engine);
//! let _stats = engine.run_to_fixpoint(&mut func, 16);
//! ```

pub mod constraint;
pub mod engine;
pub mod matcher;
pub mod pass;
pub mod patterns;
pub mod rewriter;
pub mod rule;

pub use constraint::{
    Constraint, DefinedByCategory, DefinedByOneOf, DefinedByOpcode, DefinerImmEquals,
    DefinerOperandEqualsOuter, ImmEquals, ImmIs, ImmIsPowerOfTwo, ImmNegativeNonMin,
    InterfacePure, OperandsEqual,
};
pub use engine::{RewriteEngine, RewriteStats};
pub use matcher::{CategoryMatcher, MatchCtx, Matcher, OpcodeMatcher};
pub use pass::DeclarativeRewritePass;
pub use rewriter::{RewriteAction, Rewriter, RewriterFn};
pub use rule::{Rule, RuleBuilder};
