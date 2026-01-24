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
    /// Verification succeeded - property holds.
    Valid,
    /// Verification failed - counterexample found.
    Invalid { counterexample: String },
    /// Verification inconclusive (timeout, unknown).
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

/// Function verifier.
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

    /// Verify that a transformation is semantics-preserving.
    pub fn verify_transformation(
        &self,
        _original: &Function,
        _transformed: &Function,
    ) -> Result<VerificationResult, VerifyError> {
        // TODO: Implement verification using z4
        Ok(VerificationResult::Unknown {
            reason: "not yet implemented".to_string(),
        })
    }
}

impl Default for Verifier {
    fn default() -> Self {
        Self::new()
    }
}
