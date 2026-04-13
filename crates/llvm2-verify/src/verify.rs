// llvm2-verify/verify.rs - Verification interface
//
// Author: Andrew Yates <ayates@dropbox.com>
// Copyright 2026 Dropbox, Inc. | License: Apache-2.0

//! High-level verification interface.

use llvm2_lower::Function;
use thiserror::Error;

/// Verification result.
#[derive(Debug, Clone)]
pub enum VerificationResult {
    /// Verification succeeded - property holds for all inputs.
    Valid,
    /// Verification failed - counterexample found.
    Invalid { counterexample: String },
    /// Verification inconclusive (timeout, unknown, z4 not available).
    Unknown { reason: String },
}

/// Verification error.
#[derive(Debug, Error)]
pub enum VerifyError {
    #[error("encoding error: {0}")]
    Encoding(String),
    #[error("solver error: {0}")]
    Solver(String),
}

/// Function verifier (whole-function verification, future use).
pub struct Verifier {
    timeout_ms: u64,
}

impl Verifier {
    /// Create a new verifier with default settings.
    pub fn new() -> Self {
        Self { timeout_ms: 30000 }
    }

    /// Set solver timeout in milliseconds.
    pub fn with_timeout(mut self, timeout_ms: u64) -> Self {
        self.timeout_ms = timeout_ms;
        self
    }

    /// Get the configured timeout.
    pub fn timeout_ms(&self) -> u64 {
        self.timeout_ms
    }

    /// Verify that a transformation is semantics-preserving.
    ///
    /// This is a placeholder for whole-function verification.
    /// For per-rule verification, use [`crate::lowering_proof`] directly.
    pub fn verify_transformation(
        &self,
        _original: &Function,
        _transformed: &Function,
    ) -> Result<VerificationResult, VerifyError> {
        // TODO: Implement whole-function verification using z4.
        // For now, per-rule verification is in lowering_proof.rs.
        Ok(VerificationResult::Unknown {
            reason: "whole-function verification not yet implemented; use lowering_proof::verify_by_evaluation for per-rule proofs".to_string(),
        })
    }
}

impl Default for Verifier {
    fn default() -> Self {
        Self::new()
    }
}
