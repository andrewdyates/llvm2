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
//! # Verification strength levels
//!
//! Each proof obligation is verified at one of three strength levels
//! (see [`verify::VerificationStrength`] for full details):
//!
//! | Level | Bit-width | Strategy | Guarantee |
//! |-------|-----------|----------|-----------|
//! | **Exhaustive** | <= 8 (with <= 2 inputs) | All 2^(w*n) input combinations | Complete for that width |
//! | **Statistical** | > 8 (32-bit, 64-bit) | Edge cases + 100K random samples | Probabilistic, not formal |
//! | **Formal** | Any | SMT solver (z4/z3) | Complete mathematical proof |
//!
//! ## Current status
//!
//! The default verification mode uses **mock evaluation** via
//! [`lowering_proof::verify_by_evaluation`]:
//! - 8-bit proofs run **exhaustive** verification (all 65,536 input pairs tested)
//! - 32/64-bit proofs run **statistical** verification (36 edge-case combos +
//!   100,000 random samples per proof)
//!
//! The 32/64-bit statistical verification provides high confidence but is
//! **not a formal proof**. Structured or adversarial bugs could theoretically
//! hide in the untested ~2^64 input space.
//!
//! ## Path to formal verification (z4)
//!
//! 1. **Current** -- Mock evaluation (this module): fast, catches regressions,
//!    exhaustive for 8-bit, statistical for 32/64-bit.
//! 2. **Available** -- z4/z3 CLI via [`z4_bridge`]: serialize proof obligations
//!    to SMT-LIB2 format with [`lowering_proof::ProofObligation::to_smt2`],
//!    pipe to an external z3/z4 solver for complete formal proofs.
//! 3. **Future** -- z4 native API (feature-gated `z4`): in-process SMT solving
//!    with no subprocess overhead. When this becomes the default, mock evaluation
//!    will serve as a fast pre-check before the formal proof.
//!
//! ## Configuring sample count
//!
//! The number of random samples for statistical verification is configurable
//! via [`lowering_proof::VerificationConfig`]:
//!
//! ```rust
//! use llvm2_verify::lowering_proof::{
//!     proof_iadd_i32, verify_by_evaluation_with_config, VerificationConfig,
//! };
//! use llvm2_verify::verify::VerificationResult;
//!
//! let config = VerificationConfig::with_sample_count(500_000);
//! let obligation = proof_iadd_i32();
//! let result = verify_by_evaluation_with_config(&obligation, &config);
//! assert!(matches!(result, VerificationResult::Valid));
//! ```
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
pub mod dce_proofs;
pub mod copy_prop_proofs;
pub mod cfg_proofs;
pub mod memory_model;
pub mod memory_proofs;
pub mod verify;
pub mod z4_bridge;
pub mod synthesis;
pub mod cegis;
pub mod rule_discovery;
pub mod neon_semantics;
pub mod ane_semantics;
pub mod gpu_semantics;
pub mod unified_synthesis;
pub mod neon_lowering_proofs;
pub mod vectorization_proofs;
pub mod ane_precision_proofs;

pub use verify::{VerificationResult, VerificationReport, ProofResult, Verifier, VerificationStrength};
pub use lowering_proof::{ProofObligation, verify_by_evaluation, verify_by_evaluation_with_config,
    VerificationConfig, DEFAULT_SAMPLE_COUNT, EXHAUSTIVE_WIDTH_THRESHOLD};
pub use smt::{SmtError, SmtExpr, SmtSort};
pub use z4_bridge::{Z4Config, Z4Result, verify_with_z4};
pub use cegis::{CegisLoop, CegisResult, ConcreteInput};
pub use rule_discovery::{RuleDiscovery, RuleProposal, RuleResult, RuleDatabase, DiscoveryStats};
