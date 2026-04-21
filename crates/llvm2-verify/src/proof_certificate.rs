// llvm2-verify/proof_certificate.rs - Proof certificate chain
//
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Copyright 2026 Dropbox, Inc. | License: Apache-2.0
//
// Proof certificates record the outcome of verification proofs so they can be
// persisted, inspected, and chained together. They connect tRust's
// TrustDisposition to LLVM2's verification proofs by providing a serializable
// evidence trail.
//
// A CertificateChain collects all proof certificates for a compilation unit
// (e.g., a function) and can be serialized to/from JSON for persistence.
//
// Reference: designs/2026-04-13-verification-architecture.md

//! Proof certificate chain for verification persistence.
//!
//! [`ProofCertificate`] records the outcome of a single proof obligation,
//! including the solver used, verification strength, duration, and a formula
//! hash for cache invalidation.
//!
//! [`CertificateChain`] collects certificates for a compilation unit and
//! provides JSON serialization for persistence and inspection.
//!
//! # Example
//!
//! ```rust
//! use llvm2_verify::proof_certificate::{
//!     generate_certificate, generate_certificate_chain, CertificateResult,
//! };
//! use llvm2_verify::lowering_proof::proof_iadd_i8;
//!
//! let obligation = proof_iadd_i8();
//! let cert = generate_certificate(&obligation);
//! assert_eq!(cert.result, CertificateResult::Verified);
//!
//! let chain = generate_certificate_chain("test_fn", &[obligation]);
//! assert!(chain.all_verified());
//! ```

use std::time::{Instant, SystemTime, UNIX_EPOCH};

use thiserror::Error;

use llvm2_opt::cache::StableHasher;

use crate::lowering_proof::{ProofObligation, TransvalCheckKind, verify_by_evaluation};
use crate::verify::{VerificationResult, VerificationStrength};

// ---------------------------------------------------------------------------
// CertificateResult
// ---------------------------------------------------------------------------

/// Outcome of a single proof certificate.
#[derive(Debug, Clone, PartialEq)]
pub enum CertificateResult {
    /// Proof succeeded -- property holds for all inputs.
    Verified,
    /// Proof failed -- counterexample found.
    Failed { counterexample: String },
    /// Solver timed out before reaching a conclusion.
    Timeout { seconds: f64 },
    /// Proof was not attempted.
    Skipped { reason: String },
}

impl CertificateResult {
    /// Returns true if the result is Verified.
    pub fn is_verified(&self) -> bool {
        matches!(self, CertificateResult::Verified)
    }

    /// Returns a short string tag for serialization.
    fn tag(&self) -> &'static str {
        match self {
            CertificateResult::Verified => "verified",
            CertificateResult::Failed { .. } => "failed",
            CertificateResult::Timeout { .. } => "timeout",
            CertificateResult::Skipped { .. } => "skipped",
        }
    }
}

impl std::fmt::Display for CertificateResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CertificateResult::Verified => write!(f, "Verified"),
            CertificateResult::Failed { counterexample } => {
                write!(f, "Failed({})", counterexample)
            }
            CertificateResult::Timeout { seconds } => write!(f, "Timeout({:.2}s)", seconds),
            CertificateResult::Skipped { reason } => write!(f, "Skipped({})", reason),
        }
    }
}

// ---------------------------------------------------------------------------
// SolverUsed
// ---------------------------------------------------------------------------

/// Which solver backend was used for verification.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SolverUsed {
    /// Exhaustive concrete evaluation (all input combinations).
    MockExhaustive,
    /// Random sampling with the given number of samples.
    MockStatistical { samples: u64 },
    /// z4 CLI subprocess.
    Z4Cli,
    /// z4 in-process native API.
    Z4Native,
    /// z3 CLI subprocess.
    Z3Cli,
}

impl SolverUsed {
    /// Returns a short string tag for serialization.
    fn tag(&self) -> &'static str {
        match self {
            SolverUsed::MockExhaustive => "mock_exhaustive",
            SolverUsed::MockStatistical { .. } => "mock_statistical",
            SolverUsed::Z4Cli => "z4_cli",
            SolverUsed::Z4Native => "z4_native",
            SolverUsed::Z3Cli => "z3_cli",
        }
    }
}

impl std::fmt::Display for SolverUsed {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SolverUsed::MockExhaustive => write!(f, "MockExhaustive"),
            SolverUsed::MockStatistical { samples } => {
                write!(f, "MockStatistical({})", samples)
            }
            SolverUsed::Z4Cli => write!(f, "Z4Cli"),
            SolverUsed::Z4Native => write!(f, "Z4Native"),
            SolverUsed::Z3Cli => write!(f, "Z3Cli"),
        }
    }
}

// ---------------------------------------------------------------------------
// ProofCertificate
// ---------------------------------------------------------------------------

/// A certificate recording the outcome of verifying a single proof obligation.
#[derive(Debug, Clone)]
pub struct ProofCertificate {
    /// Name of the proof obligation (e.g., "Iadd_I32 -> ADDWrr").
    pub obligation_name: String,
    /// Verification outcome.
    pub result: CertificateResult,
    /// Which solver backend was used.
    pub solver: SolverUsed,
    /// Strength of the verification applied.
    pub strength: VerificationStrength,
    /// Proof category from the obligation (if set).
    pub check_kind: Option<TransvalCheckKind>,
    /// Hash of the negated equivalence formula, for cache invalidation.
    pub formula_hash: u64,
    /// Unix epoch seconds when this certificate was generated.
    pub timestamp_epoch_secs: u64,
    /// Duration of the verification in milliseconds.
    pub duration_ms: u64,
}

// ---------------------------------------------------------------------------
// CertificateChain
// ---------------------------------------------------------------------------

/// An ordered collection of proof certificates for a compilation unit.
#[derive(Debug, Clone)]
pub struct CertificateChain {
    /// Name of the compilation unit (e.g., function name).
    pub compilation_unit: String,
    /// Ordered list of proof certificates.
    pub certificates: Vec<ProofCertificate>,
    /// Unix epoch seconds when this chain was created.
    pub created_epoch_secs: u64,
}

impl CertificateChain {
    /// Create a new empty certificate chain for the given compilation unit.
    pub fn new(compilation_unit: String) -> Self {
        let created = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        Self {
            compilation_unit,
            certificates: Vec::new(),
            created_epoch_secs: created,
        }
    }

    /// Add a certificate to the chain.
    pub fn add(&mut self, cert: ProofCertificate) {
        self.certificates.push(cert);
    }

    /// Verify the chain: check that all certificates are Verified.
    pub fn verify_chain(&self) -> ChainVerificationResult {
        if self.certificates.is_empty() {
            return ChainVerificationResult::Empty;
        }

        let summary = self.summary();

        if summary.failed == 0 && summary.timeout == 0 && summary.skipped == 0 {
            ChainVerificationResult::AllVerified {
                count: summary.verified,
            }
        } else {
            ChainVerificationResult::HasFailures {
                verified: summary.verified,
                failed: summary.failed,
                skipped: summary.skipped,
                timeout: summary.timeout,
            }
        }
    }

    /// Returns true if all certificates in the chain are Verified.
    pub fn all_verified(&self) -> bool {
        !self.certificates.is_empty()
            && self.certificates.iter().all(|c| c.result.is_verified())
    }

    /// Returns references to all failed certificates.
    pub fn failed_certificates(&self) -> Vec<&ProofCertificate> {
        self.certificates
            .iter()
            .filter(|c| matches!(c.result, CertificateResult::Failed { .. }))
            .collect()
    }

    /// Compute a summary of the chain.
    pub fn summary(&self) -> ChainSummary {
        let mut verified = 0;
        let mut failed = 0;
        let mut timeout = 0;
        let mut skipped = 0;
        let mut total_duration_ms = 0u64;

        for cert in &self.certificates {
            match &cert.result {
                CertificateResult::Verified => verified += 1,
                CertificateResult::Failed { .. } => failed += 1,
                CertificateResult::Timeout { .. } => timeout += 1,
                CertificateResult::Skipped { .. } => skipped += 1,
            }
            total_duration_ms = total_duration_ms.saturating_add(cert.duration_ms);
        }

        ChainSummary {
            total: self.certificates.len(),
            verified,
            failed,
            timeout,
            skipped,
            total_duration_ms,
        }
    }

    /// Serialize the chain to JSON (manual, no serde dependency).
    pub fn to_json(&self) -> String {
        let mut out = String::from("{\n");
        out.push_str(&format!(
            "  \"compilation_unit\": \"{}\",\n",
            escape_json(&self.compilation_unit)
        ));
        out.push_str(&format!(
            "  \"created_epoch_secs\": {},\n",
            self.created_epoch_secs
        ));
        out.push_str("  \"certificates\": [\n");

        for (i, cert) in self.certificates.iter().enumerate() {
            out.push_str("    {\n");
            out.push_str(&format!(
                "      \"obligation_name\": \"{}\",\n",
                escape_json(&cert.obligation_name)
            ));
            out.push_str(&format!(
                "      \"result\": \"{}\",\n",
                cert.result.tag()
            ));

            // Result detail (counterexample, timeout seconds, skip reason)
            match &cert.result {
                CertificateResult::Failed { counterexample } => {
                    out.push_str(&format!(
                        "      \"counterexample\": \"{}\",\n",
                        escape_json(counterexample)
                    ));
                }
                CertificateResult::Timeout { seconds } => {
                    out.push_str(&format!("      \"timeout_seconds\": {},\n", seconds));
                }
                CertificateResult::Skipped { reason } => {
                    out.push_str(&format!(
                        "      \"skip_reason\": \"{}\",\n",
                        escape_json(reason)
                    ));
                }
                CertificateResult::Verified => {}
            }

            out.push_str(&format!("      \"solver\": \"{}\",\n", cert.solver.tag()));

            // Solver detail (samples for statistical)
            if let SolverUsed::MockStatistical { samples } = &cert.solver {
                out.push_str(&format!("      \"solver_samples\": {},\n", samples));
            }

            out.push_str(&format!(
                "      \"strength\": \"{}\",\n",
                strength_to_tag(&cert.strength)
            ));
            if let VerificationStrength::Statistical { sample_count } = &cert.strength {
                out.push_str(&format!(
                    "      \"strength_samples\": {},\n",
                    sample_count
                ));
            }

            if let Some(kind) = &cert.check_kind {
                out.push_str(&format!("      \"check_kind\": \"{}\",\n", kind));
            }

            out.push_str(&format!(
                "      \"formula_hash\": {},\n",
                cert.formula_hash
            ));
            out.push_str(&format!(
                "      \"timestamp_epoch_secs\": {},\n",
                cert.timestamp_epoch_secs
            ));
            out.push_str(&format!("      \"duration_ms\": {}\n", cert.duration_ms));

            if i + 1 < self.certificates.len() {
                out.push_str("    },\n");
            } else {
                out.push_str("    }\n");
            }
        }

        out.push_str("  ]\n");
        out.push('}');
        out
    }

    /// Deserialize a chain from JSON (manual parsing, no serde dependency).
    pub fn from_json(json: &str) -> Result<Self, CertificateError> {
        let compilation_unit = extract_string_field(json, "compilation_unit")?;
        let created_epoch_secs = extract_u64_field(json, "created_epoch_secs")?;

        let certs_start = json
            .find("\"certificates\"")
            .ok_or_else(|| CertificateError::MissingField("certificates".to_string()))?;
        let array_start = json[certs_start..]
            .find('[')
            .ok_or_else(|| CertificateError::JsonParseError("missing [ for certificates".to_string()))?
            + certs_start;
        let array_end = find_matching_bracket(json, array_start)
            .ok_or_else(|| CertificateError::JsonParseError("unmatched [ in certificates".to_string()))?;

        let array_content = &json[array_start + 1..array_end];
        let mut certificates = Vec::new();

        // Split on objects by finding matched { }
        let mut pos = 0;
        let bytes = array_content.as_bytes();
        while pos < bytes.len() {
            if bytes[pos] == b'{' {
                let obj_end = find_matching_brace(array_content, pos)
                    .ok_or_else(|| {
                        CertificateError::JsonParseError("unmatched { in certificate".to_string())
                    })?;
                let obj_str = &array_content[pos..=obj_end];
                let cert = parse_certificate(obj_str)?;
                certificates.push(cert);
                pos = obj_end + 1;
            } else {
                pos += 1;
            }
        }

        Ok(CertificateChain {
            compilation_unit,
            certificates,
            created_epoch_secs,
        })
    }
}

impl std::fmt::Display for CertificateChain {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let summary = self.summary();
        write!(
            f,
            "CertificateChain({}: {}/{} verified, {} failed, {} timeout, {} skipped, {}ms)",
            self.compilation_unit,
            summary.verified,
            summary.total,
            summary.failed,
            summary.timeout,
            summary.skipped,
            summary.total_duration_ms
        )
    }
}

// ---------------------------------------------------------------------------
// ChainVerificationResult
// ---------------------------------------------------------------------------

/// Result of verifying an entire certificate chain.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ChainVerificationResult {
    /// All certificates are Verified.
    AllVerified { count: usize },
    /// Some certificates are not Verified.
    HasFailures {
        verified: usize,
        failed: usize,
        skipped: usize,
        timeout: usize,
    },
    /// The chain is empty (no certificates).
    Empty,
}

// ---------------------------------------------------------------------------
// ChainSummary
// ---------------------------------------------------------------------------

/// Summary statistics for a certificate chain.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChainSummary {
    /// Total number of certificates.
    pub total: usize,
    /// Number of Verified certificates.
    pub verified: usize,
    /// Number of Failed certificates.
    pub failed: usize,
    /// Number of Timeout certificates.
    pub timeout: usize,
    /// Number of Skipped certificates.
    pub skipped: usize,
    /// Total verification duration in milliseconds.
    pub total_duration_ms: u64,
}

// ---------------------------------------------------------------------------
// CertificateError
// ---------------------------------------------------------------------------

/// Errors during certificate chain serialization/deserialization.
#[derive(Debug, Error)]
pub enum CertificateError {
    /// JSON parsing error.
    #[error("JSON parse error: {0}")]
    JsonParseError(String),
    /// Required field missing.
    #[error("missing field: {0}")]
    MissingField(String),
    /// Invalid result value.
    #[error("invalid result: {0}")]
    InvalidResult(String),
}

// ---------------------------------------------------------------------------
// Certificate generation
// ---------------------------------------------------------------------------

/// Generate a proof certificate by running verification on the given obligation.
///
/// This function:
/// 1. Computes a formula hash from the negated equivalence expression
/// 2. Runs `verify_by_evaluation` on the obligation
/// 3. Records the outcome, duration, solver, and strength
pub fn generate_certificate(obligation: &ProofObligation) -> ProofCertificate {
    let formula_hash = compute_formula_hash(obligation);
    let strength = VerificationStrength::for_obligation(obligation);
    let solver = strength_to_solver(&strength);

    let timestamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();

    let start = Instant::now();
    let result = verify_by_evaluation(obligation);
    let duration_ms = start.elapsed().as_millis() as u64;

    let cert_result = match result {
        VerificationResult::Valid => CertificateResult::Verified,
        VerificationResult::Invalid { counterexample } => {
            CertificateResult::Failed { counterexample }
        }
        VerificationResult::Unknown { reason } => {
            if reason.to_lowercase().contains("timeout") {
                CertificateResult::Timeout {
                    seconds: duration_ms as f64 / 1000.0,
                }
            } else {
                CertificateResult::Skipped { reason }
            }
        }
    };

    ProofCertificate {
        obligation_name: obligation.name.clone(),
        result: cert_result,
        solver,
        strength,
        check_kind: obligation.category,
        formula_hash,
        timestamp_epoch_secs: timestamp,
        duration_ms,
    }
}

/// Generate a certificate chain by verifying all obligations for a compilation unit.
pub fn generate_certificate_chain(
    compilation_unit: &str,
    obligations: &[ProofObligation],
) -> CertificateChain {
    let mut chain = CertificateChain::new(compilation_unit.to_string());
    for obligation in obligations {
        let cert = generate_certificate(obligation);
        chain.add(cert);
    }
    chain
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Compute a hash of the negated equivalence formula for cache invalidation.
fn compute_formula_hash(obligation: &ProofObligation) -> u64 {
    let formula = obligation.negated_equivalence();
    let debug_str = format!("{:?}", formula);
    let mut hasher = StableHasher::new();
    hasher.write(debug_str.as_bytes());
    hasher.finish64()
}

/// Map verification strength to the solver that was used.
fn strength_to_solver(strength: &VerificationStrength) -> SolverUsed {
    match strength {
        VerificationStrength::Exhaustive => SolverUsed::MockExhaustive,
        VerificationStrength::Statistical { sample_count } => SolverUsed::MockStatistical {
            samples: *sample_count,
        },
        VerificationStrength::Formal => SolverUsed::Z4Native,
    }
}

/// Convert VerificationStrength to a short tag for JSON.
fn strength_to_tag(strength: &VerificationStrength) -> &'static str {
    match strength {
        VerificationStrength::Exhaustive => "exhaustive",
        VerificationStrength::Statistical { .. } => "statistical",
        VerificationStrength::Formal => "formal",
    }
}

/// Escape a string for JSON output.
fn escape_json(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => {
                out.push_str(&format!("\\u{:04x}", c as u32));
            }
            c => out.push(c),
        }
    }
    out
}

/// Extract a string field from a flat JSON object.
fn extract_string_field(json: &str, field: &str) -> Result<String, CertificateError> {
    let pattern = format!("\"{}\"", field);
    let field_pos = json
        .find(&pattern)
        .ok_or_else(|| CertificateError::MissingField(field.to_string()))?;
    let after_key = &json[field_pos + pattern.len()..];
    // Skip ': "'
    let val_start = after_key
        .find('"')
        .ok_or_else(|| CertificateError::JsonParseError(format!("no value for {}", field)))?;
    let val_content = &after_key[val_start + 1..];
    let val_end = find_unescaped_quote(val_content)
        .ok_or_else(|| CertificateError::JsonParseError(format!("unterminated string for {}", field)))?;
    Ok(unescape_json(&val_content[..val_end]))
}

/// Extract a u64 field from a flat JSON object.
fn extract_u64_field(json: &str, field: &str) -> Result<u64, CertificateError> {
    let pattern = format!("\"{}\"", field);
    let field_pos = json
        .find(&pattern)
        .ok_or_else(|| CertificateError::MissingField(field.to_string()))?;
    let after_key = &json[field_pos + pattern.len()..];
    // Skip ': '
    let colon_pos = after_key
        .find(':')
        .ok_or_else(|| CertificateError::JsonParseError(format!("no colon after {}", field)))?;
    let after_colon = after_key[colon_pos + 1..].trim_start();
    // Read digits
    let num_end = after_colon
        .find(|c: char| !c.is_ascii_digit())
        .unwrap_or(after_colon.len());
    let num_str = &after_colon[..num_end];
    num_str
        .parse::<u64>()
        .map_err(|e| CertificateError::JsonParseError(format!("bad u64 for {}: {}", field, e)))
}

/// Extract an optional f64 field from a flat JSON object.
fn extract_f64_field(json: &str, field: &str) -> Option<f64> {
    let pattern = format!("\"{}\"", field);
    let field_pos = json.find(&pattern)?;
    let after_key = &json[field_pos + pattern.len()..];
    let colon_pos = after_key.find(':')?;
    let after_colon = after_key[colon_pos + 1..].trim_start();
    let num_end = after_colon
        .find(|c: char| !c.is_ascii_digit() && c != '.' && c != '-')
        .unwrap_or(after_colon.len());
    after_colon[..num_end].parse::<f64>().ok()
}

/// Find the index of the first unescaped double-quote.
fn find_unescaped_quote(s: &str) -> Option<usize> {
    let mut escaped = false;
    for (i, c) in s.char_indices() {
        if escaped {
            escaped = false;
            continue;
        }
        if c == '\\' {
            escaped = true;
            continue;
        }
        if c == '"' {
            return Some(i);
        }
    }
    None
}

/// Unescape a JSON string value.
fn unescape_json(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut chars = s.chars();
    while let Some(c) = chars.next() {
        if c == '\\' {
            match chars.next() {
                Some('"') => out.push('"'),
                Some('\\') => out.push('\\'),
                Some('n') => out.push('\n'),
                Some('r') => out.push('\r'),
                Some('t') => out.push('\t'),
                Some(other) => {
                    out.push('\\');
                    out.push(other);
                }
                None => out.push('\\'),
            }
        } else {
            out.push(c);
        }
    }
    out
}

/// Find the matching ] for a [ at position `start`.
fn find_matching_bracket(s: &str, start: usize) -> Option<usize> {
    let mut depth = 0;
    let mut in_string = false;
    let mut escape_next = false;
    for (i, c) in s[start..].char_indices() {
        if escape_next {
            escape_next = false;
            continue;
        }
        if c == '\\' && in_string {
            escape_next = true;
            continue;
        }
        if c == '"' {
            in_string = !in_string;
            continue;
        }
        if in_string {
            continue;
        }
        if c == '[' {
            depth += 1;
        } else if c == ']' {
            depth -= 1;
            if depth == 0 {
                return Some(start + i);
            }
        }
    }
    None
}

/// Find the matching } for a { at position `start`.
fn find_matching_brace(s: &str, start: usize) -> Option<usize> {
    let mut depth = 0;
    let mut in_string = false;
    let mut escape_next = false;
    for (i, c) in s[start..].char_indices() {
        if escape_next {
            escape_next = false;
            continue;
        }
        if c == '\\' && in_string {
            escape_next = true;
            continue;
        }
        if c == '"' {
            in_string = !in_string;
            continue;
        }
        if in_string {
            continue;
        }
        if c == '{' {
            depth += 1;
        } else if c == '}' {
            depth -= 1;
            if depth == 0 {
                return Some(start + i);
            }
        }
    }
    None
}

/// Parse a single certificate from a JSON object string.
fn parse_certificate(json: &str) -> Result<ProofCertificate, CertificateError> {
    let obligation_name = extract_string_field(json, "obligation_name")?;
    let result_tag = extract_string_field(json, "result")?;

    let result = match result_tag.as_str() {
        "verified" => CertificateResult::Verified,
        "failed" => {
            let cex = extract_string_field(json, "counterexample")
                .unwrap_or_else(|_| String::new());
            CertificateResult::Failed {
                counterexample: cex,
            }
        }
        "timeout" => {
            let secs = extract_f64_field(json, "timeout_seconds").unwrap_or(0.0);
            CertificateResult::Timeout { seconds: secs }
        }
        "skipped" => {
            let reason = extract_string_field(json, "skip_reason")
                .unwrap_or_else(|_| String::new());
            CertificateResult::Skipped { reason }
        }
        other => {
            return Err(CertificateError::InvalidResult(other.to_string()));
        }
    };

    let solver_tag = extract_string_field(json, "solver")?;
    let solver = match solver_tag.as_str() {
        "mock_exhaustive" => SolverUsed::MockExhaustive,
        "mock_statistical" => {
            let samples = extract_u64_field(json, "solver_samples").unwrap_or(100_000);
            SolverUsed::MockStatistical { samples }
        }
        "z4_cli" => SolverUsed::Z4Cli,
        "z4_native" => SolverUsed::Z4Native,
        "z3_cli" => SolverUsed::Z3Cli,
        other => {
            return Err(CertificateError::InvalidResult(format!(
                "unknown solver: {}",
                other
            )));
        }
    };

    let strength_tag = extract_string_field(json, "strength")?;
    let strength = match strength_tag.as_str() {
        "exhaustive" => VerificationStrength::Exhaustive,
        "statistical" => {
            let samples = extract_u64_field(json, "strength_samples").unwrap_or(100_000);
            VerificationStrength::Statistical {
                sample_count: samples,
            }
        }
        "formal" => VerificationStrength::Formal,
        other => {
            return Err(CertificateError::InvalidResult(format!(
                "unknown strength: {}",
                other
            )));
        }
    };

    let check_kind = extract_string_field(json, "check_kind")
        .ok()
        .and_then(|s| parse_check_kind(&s));

    let formula_hash = extract_u64_field(json, "formula_hash")?;
    let timestamp_epoch_secs = extract_u64_field(json, "timestamp_epoch_secs")?;
    let duration_ms = extract_u64_field(json, "duration_ms")?;

    Ok(ProofCertificate {
        obligation_name,
        result,
        solver,
        strength,
        check_kind,
        formula_hash,
        timestamp_epoch_secs,
        duration_ms,
    })
}

/// Parse a TransvalCheckKind from its Display string.
fn parse_check_kind(s: &str) -> Option<TransvalCheckKind> {
    match s {
        "data_flow" => Some(TransvalCheckKind::DataFlow),
        "control_flow" => Some(TransvalCheckKind::ControlFlow),
        "return_value" => Some(TransvalCheckKind::ReturnValue),
        "termination" => Some(TransvalCheckKind::Termination),
        "instruction_lowering" => Some(TransvalCheckKind::InstructionLowering),
        "peephole" => Some(TransvalCheckKind::PeepholeOptimization),
        "memory" => Some(TransvalCheckKind::MemoryModel),
        "regalloc" => Some(TransvalCheckKind::RegisterAllocation),
        "vectorization" => Some(TransvalCheckKind::Vectorization),
        _ => None,
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::smt::SmtExpr;

    /// Helper: create a trivially valid proof obligation (bvadd(a,b) == bvadd(a,b)).
    fn make_simple_obligation(name: &str) -> ProofObligation {
        let a = SmtExpr::var("a", 8);
        let b = SmtExpr::var("b", 8);
        ProofObligation {
            name: name.to_string(),
            tmir_expr: SmtExpr::bvadd(a.clone(), b.clone()),
            aarch64_expr: SmtExpr::bvadd(a, b),
            inputs: vec![("a".to_string(), 8), ("b".to_string(), 8)],
            preconditions: vec![],
            fp_inputs: vec![],
            category: Some(TransvalCheckKind::InstructionLowering),
        }
    }

    #[test]
    fn test_certificate_result_display() {
        let v = CertificateResult::Verified;
        assert_eq!(format!("{:?}", v), "Verified");
        assert_eq!(format!("{}", v), "Verified");

        let f = CertificateResult::Failed {
            counterexample: "a=1, b=2".to_string(),
        };
        assert!(format!("{}", f).contains("Failed"));

        let t = CertificateResult::Timeout { seconds: 5.5 };
        assert!(format!("{}", t).contains("Timeout"));

        let s = CertificateResult::Skipped {
            reason: "no solver".to_string(),
        };
        assert!(format!("{}", s).contains("Skipped"));
    }

    #[test]
    fn test_generate_certificate_for_simple_proof() {
        let obligation = make_simple_obligation("test_iadd_i8");
        let cert = generate_certificate(&obligation);

        assert_eq!(cert.obligation_name, "test_iadd_i8");
        assert_eq!(cert.result, CertificateResult::Verified);
        assert_eq!(cert.solver, SolverUsed::MockExhaustive);
        assert_eq!(cert.strength, VerificationStrength::Exhaustive);
        assert_eq!(
            cert.check_kind,
            Some(TransvalCheckKind::InstructionLowering)
        );
        assert!(cert.formula_hash != 0);
        assert!(cert.timestamp_epoch_secs > 0);
    }

    #[test]
    fn test_certificate_chain_empty() {
        let chain = CertificateChain::new("empty_fn".to_string());
        assert_eq!(chain.verify_chain(), ChainVerificationResult::Empty);
        assert!(!chain.all_verified());
        assert!(chain.failed_certificates().is_empty());
    }

    #[test]
    fn test_certificate_chain_all_verified() {
        let mut chain = CertificateChain::new("verified_fn".to_string());
        for i in 0..3 {
            let obligation = make_simple_obligation(&format!("proof_{}", i));
            let cert = generate_certificate(&obligation);
            chain.add(cert);
        }

        assert!(chain.all_verified());
        assert_eq!(
            chain.verify_chain(),
            ChainVerificationResult::AllVerified { count: 3 }
        );
        assert!(chain.failed_certificates().is_empty());
    }

    #[test]
    fn test_certificate_chain_with_failure() {
        let mut chain = CertificateChain::new("mixed_fn".to_string());

        // Add a verified certificate
        let obligation = make_simple_obligation("good_proof");
        let cert = generate_certificate(&obligation);
        chain.add(cert);

        // Add a manually-created failed certificate
        chain.add(ProofCertificate {
            obligation_name: "bad_proof".to_string(),
            result: CertificateResult::Failed {
                counterexample: "a=0xff, b=0x01".to_string(),
            },
            solver: SolverUsed::MockExhaustive,
            strength: VerificationStrength::Exhaustive,
            check_kind: None,
            formula_hash: 12345,
            timestamp_epoch_secs: 1000,
            duration_ms: 50,
        });

        assert!(!chain.all_verified());
        assert_eq!(chain.failed_certificates().len(), 1);

        match chain.verify_chain() {
            ChainVerificationResult::HasFailures {
                verified,
                failed,
                skipped,
                timeout,
            } => {
                assert_eq!(verified, 1);
                assert_eq!(failed, 1);
                assert_eq!(skipped, 0);
                assert_eq!(timeout, 0);
            }
            other => panic!("expected HasFailures, got {:?}", other),
        }
    }

    #[test]
    fn test_chain_summary() {
        let mut chain = CertificateChain::new("summary_fn".to_string());

        // Two verified
        for i in 0..2 {
            let obligation = make_simple_obligation(&format!("proof_{}", i));
            chain.add(generate_certificate(&obligation));
        }

        // One timeout
        chain.add(ProofCertificate {
            obligation_name: "timeout_proof".to_string(),
            result: CertificateResult::Timeout { seconds: 30.0 },
            solver: SolverUsed::Z3Cli,
            strength: VerificationStrength::Formal,
            check_kind: None,
            formula_hash: 99999,
            timestamp_epoch_secs: 1000,
            duration_ms: 30000,
        });

        // One skipped
        chain.add(ProofCertificate {
            obligation_name: "skipped_proof".to_string(),
            result: CertificateResult::Skipped {
                reason: "solver not available".to_string(),
            },
            solver: SolverUsed::Z4Native,
            strength: VerificationStrength::Formal,
            check_kind: None,
            formula_hash: 88888,
            timestamp_epoch_secs: 1000,
            duration_ms: 0,
        });

        let summary = chain.summary();
        assert_eq!(summary.total, 4);
        assert_eq!(summary.verified, 2);
        assert_eq!(summary.failed, 0);
        assert_eq!(summary.timeout, 1);
        assert_eq!(summary.skipped, 1);
        assert!(summary.total_duration_ms >= 30000);
    }

    #[test]
    fn test_json_roundtrip() {
        let mut chain = CertificateChain::new("roundtrip_fn".to_string());

        // Add a verified certificate
        let obligation = make_simple_obligation("iadd_i8");
        let cert = generate_certificate(&obligation);
        let original_hash = cert.formula_hash;
        let original_name = cert.obligation_name.clone();
        chain.add(cert);

        // Add a failed certificate
        chain.add(ProofCertificate {
            obligation_name: "bad_rule".to_string(),
            result: CertificateResult::Failed {
                counterexample: "a=255, b=1".to_string(),
            },
            solver: SolverUsed::MockStatistical { samples: 50000 },
            strength: VerificationStrength::Statistical {
                sample_count: 50000,
            },
            check_kind: Some(TransvalCheckKind::DataFlow),
            formula_hash: 42,
            timestamp_epoch_secs: 1713200000,
            duration_ms: 123,
        });

        let json = chain.to_json();

        // Parse it back
        let parsed = CertificateChain::from_json(&json).expect("JSON roundtrip failed");

        assert_eq!(parsed.compilation_unit, "roundtrip_fn");
        assert_eq!(parsed.certificates.len(), 2);

        // Check first certificate
        let c0 = &parsed.certificates[0];
        assert_eq!(c0.obligation_name, original_name);
        assert_eq!(c0.result, CertificateResult::Verified);
        assert_eq!(c0.solver, SolverUsed::MockExhaustive);
        assert_eq!(c0.strength, VerificationStrength::Exhaustive);
        assert_eq!(c0.formula_hash, original_hash);

        // Check second certificate
        let c1 = &parsed.certificates[1];
        assert_eq!(c1.obligation_name, "bad_rule");
        assert!(matches!(c1.result, CertificateResult::Failed { .. }));
        assert_eq!(
            c1.solver,
            SolverUsed::MockStatistical { samples: 50000 }
        );
        assert_eq!(
            c1.strength,
            VerificationStrength::Statistical {
                sample_count: 50000
            }
        );
        assert_eq!(c1.formula_hash, 42);
        assert_eq!(c1.timestamp_epoch_secs, 1713200000);
        assert_eq!(c1.duration_ms, 123);
    }

    #[test]
    fn test_generate_certificate_chain_fn() {
        let obligations: Vec<ProofObligation> = (0..3)
            .map(|i| make_simple_obligation(&format!("chain_proof_{}", i)))
            .collect();

        let chain = generate_certificate_chain("test_function", &obligations);

        assert_eq!(chain.compilation_unit, "test_function");
        assert_eq!(chain.certificates.len(), 3);
        assert!(chain.all_verified());

        for (i, cert) in chain.certificates.iter().enumerate() {
            assert_eq!(cert.obligation_name, format!("chain_proof_{}", i));
            assert_eq!(cert.result, CertificateResult::Verified);
        }
    }

    #[test]
    fn test_formula_hash_stability() {
        // Same obligation should produce the same hash
        let o1 = make_simple_obligation("same_proof");
        let o2 = make_simple_obligation("same_proof");
        let h1 = compute_formula_hash(&o1);
        let h2 = compute_formula_hash(&o2);
        assert_eq!(h1, h2);

        // Different obligation should produce a different hash
        let o3 = make_simple_obligation("different_proof");
        // Same formula structure -- hash should still be equal since the
        // negated equivalence is the same (just different name, which
        // is not part of the formula).
        let h3 = compute_formula_hash(&o3);
        assert_eq!(h1, h3); // name not hashed, formula is identical

        // Truly different formula
        let a = SmtExpr::var("a", 8);
        let b = SmtExpr::var("b", 8);
        let o4 = ProofObligation {
            name: "sub_proof".to_string(),
            tmir_expr: SmtExpr::bvsub(a.clone(), b.clone()),
            aarch64_expr: SmtExpr::bvsub(a, b),
            inputs: vec![("a".to_string(), 8), ("b".to_string(), 8)],
            preconditions: vec![],
            fp_inputs: vec![],
            category: None,
        };
        let h4 = compute_formula_hash(&o4);
        assert_ne!(h1, h4);
    }

    #[test]
    fn test_chain_display() {
        let mut chain = CertificateChain::new("display_fn".to_string());
        let obligation = make_simple_obligation("proof_0");
        chain.add(generate_certificate(&obligation));

        let display = format!("{}", chain);
        assert!(display.contains("display_fn"));
        assert!(display.contains("1/1 verified"));
    }
}
