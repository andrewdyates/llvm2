// llvm2-verify - Verification backend
//
// Author: Andrew Yates <ayates@dropbox.com>
// Copyright 2026 Dropbox, Inc. | License: Apache-2.0

//! Verification backend for LLVM2.
//!
//! This crate provides formal verification of tMIR -> AArch64 lowering rules
//! using SMT bitvector solving. Each lowering rule is proven correct by
//! showing semantic equivalence: for all inputs satisfying preconditions,
//! the tMIR semantics and AArch64 semantics produce the same result.
//!
//! # Architecture
//!
//! ```text
//! tmir_semantics    -- tMIR instruction semantics as SmtExpr
//! aarch64_semantics -- AArch64 instruction semantics as SmtExpr
//! lowering_proof    -- Proof obligations pairing both sides
//! smt               -- Self-contained bitvector expression AST + evaluator
//! verify            -- High-level verification interface
//! ```
//!
//! # Verification modes
//!
//! 1. **Mock verification** (default): evaluates proof obligations using
//!    concrete Rust arithmetic (exhaustive for small widths, random sampling
//!    for 32/64-bit). No external solver needed.
//!
//! 2. **z4 verification** (future, feature-gated): serializes proof obligations
//!    to SMT-LIB2 and solves via z4 for full formal guarantees.
//!
//! # Example
//!
//! ```rust
//! use llvm2_verify::lowering_proof::{proof_iadd_i32, verify_by_evaluation};
//! use llvm2_verify::verify::VerificationResult;
//!
//! let obligation = proof_iadd_i32();
//! let result = verify_by_evaluation(&obligation);
//! assert!(matches!(result, VerificationResult::Valid));
//! ```

pub mod smt;
pub mod tmir_semantics;
pub mod aarch64_semantics;
pub mod nzcv;
pub mod lowering_proof;
pub mod peephole_proofs;
pub mod opt_proofs;
pub mod const_fold_proofs;
pub mod cse_licm_proofs;
pub mod memory_model;
pub mod verify;

pub use verify::{VerificationResult, Verifier};
pub use lowering_proof::{ProofObligation, verify_by_evaluation};
pub use smt::{SmtExpr, SmtSort};
