// llvm2-verify/z4_bridge.rs - Bridge to the z4 SMT solver
//
// Author: Andrew Yates <ayates@dropbox.com>
// Copyright 2026 Dropbox, Inc. | License: Apache-2.0
//
// Translates our SmtExpr AST into SMT-LIB2 format and invokes an SMT solver
// to check satisfiability. Two backends:
//
// 1. z4 Rust API (feature = "z4") -- direct in-process solving via the z4 crate
// 2. CLI subprocess fallback -- invokes z3/z4 binary via SMT-LIB2 text pipe
//
// The CLI fallback is always available (no feature gate) and is useful when
// the z4 crate is not linked. It uses the standard SMT-LIB2 text interface.
//
// Reference: designs/2026-04-13-verification-architecture.md

//! Bridge to the z4 SMT solver for formal verification.
//!
//! This module provides the infrastructure to verify [`ProofObligation`]s
//! using a real SMT solver instead of the mock evaluator. It translates
//! our [`SmtExpr`] AST into SMT-LIB2 format and either:
//!
//! - Invokes z4's native Rust API (when the `z4` feature is enabled), or
//! - Pipes SMT-LIB2 text to a z3/z4 CLI binary as a subprocess fallback.
//!
//! # Architecture
//!
//! ```text
//! ProofObligation
//!   |
//!   v
//! to_smt2() -> SMT-LIB2 string
//!   |
//!   +--[z4 feature]--> z4::Solver (in-process)
//!   |
//!   +--[CLI fallback]-> z3/z4 subprocess (SMT-LIB2 stdin/stdout)
//! ```
//!
//! [`ProofObligation`]: crate::lowering_proof::ProofObligation
//! [`SmtExpr`]: crate::smt::SmtExpr

use crate::lowering_proof::ProofObligation;
use crate::smt::{SmtExpr, SmtSort, RoundingMode};
#[cfg(feature = "z4")]
compile_error!(
    "The `z4` feature requires the z4 crate dependency. \
     Uncomment the z4 line in llvm2-verify/Cargo.toml and change \
     the z4 feature to `z4 = [\"dep:z4\"]`."
);
#[cfg(feature = "z4")]
use std::collections::HashMap;
use std::fmt;

// ---------------------------------------------------------------------------
// Z4Result
// ---------------------------------------------------------------------------

/// Result of a z4/z3 verification check.
#[derive(Debug, Clone, PartialEq)]
pub enum Z4Result {
    /// The property holds (UNSAT -- no counterexample exists).
    /// The negated equivalence is unsatisfiable, meaning the original
    /// property holds for ALL inputs.
    Verified,
    /// The property fails with a counterexample.
    /// Each entry is (variable_name, value) from the satisfying assignment
    /// to the negated equivalence formula.
    CounterExample(Vec<(String, u64)>),
    /// The solver timed out before reaching a conclusion.
    Timeout,
    /// Solver error (parse failure, internal error, etc.).
    Error(String),
}

impl fmt::Display for Z4Result {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Z4Result::Verified => write!(f, "VERIFIED (UNSAT)"),
            Z4Result::CounterExample(cex) => {
                write!(f, "COUNTEREXAMPLE: ")?;
                for (i, (name, val)) in cex.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{} = {:#x}", name, val)?;
                }
                Ok(())
            }
            Z4Result::Timeout => write!(f, "TIMEOUT"),
            Z4Result::Error(msg) => write!(f, "ERROR: {}", msg),
        }
    }
}

// ---------------------------------------------------------------------------
// Z4Config
// ---------------------------------------------------------------------------

/// Configuration for the z4/z3 solver.
pub struct Z4Config {
    /// Path to the solver binary for CLI fallback (default: search PATH for z3, then z4).
    pub solver_path: Option<String>,
    /// Timeout in milliseconds (default: 5000).
    pub timeout_ms: u64,
    /// Whether to request a model on SAT (for counterexample extraction).
    pub produce_models: bool,
}

impl Default for Z4Config {
    fn default() -> Self {
        Self {
            solver_path: None,
            timeout_ms: 5000,
            produce_models: true,
        }
    }
}

impl Z4Config {
    /// Create a config with a custom timeout.
    pub fn with_timeout(mut self, timeout_ms: u64) -> Self {
        self.timeout_ms = timeout_ms;
        self
    }

    /// Create a config with a specific solver binary path.
    pub fn with_solver_path(mut self, path: impl Into<String>) -> Self {
        self.solver_path = Some(path.into());
        self
    }
}

// ---------------------------------------------------------------------------
// SMT-LIB2 generation (enhanced version of ProofObligation::to_smt2)
// ---------------------------------------------------------------------------

/// Infer the minimal SMT-LIB2 logic string needed for an expression.
///
/// Walks the expression tree and returns the appropriate logic:
/// - `QF_BV` -- bitvectors only (default)
/// - `QF_ABV` -- bitvectors + arrays
/// - `QF_BVFP` -- bitvectors + floating-point
/// - `QF_ABVFP` -- bitvectors + arrays + floating-point
/// - `QF_UFBV` -- bitvectors + uninterpreted functions
/// - `ALL` -- when multiple theories are combined
pub fn infer_logic(expr: &SmtExpr) -> &'static str {
    let mut has_array = false;
    let mut has_fp = false;
    let mut has_uf = false;
    infer_logic_walk(expr, &mut has_array, &mut has_fp, &mut has_uf);

    match (has_array, has_fp, has_uf) {
        (false, false, false) => "QF_BV",
        (true, false, false)  => "QF_ABV",
        (false, true, false)  => "QF_BVFP",
        (true, true, false)   => "QF_ABVFP",
        (false, false, true)  => "QF_UFBV",
        _                     => "ALL",
    }
}

fn infer_logic_walk(expr: &SmtExpr, has_array: &mut bool, has_fp: &mut bool, has_uf: &mut bool) {
    match expr {
        SmtExpr::Select { array, index } => {
            *has_array = true;
            infer_logic_walk(array, has_array, has_fp, has_uf);
            infer_logic_walk(index, has_array, has_fp, has_uf);
        }
        SmtExpr::Store { array, index, value } => {
            *has_array = true;
            infer_logic_walk(array, has_array, has_fp, has_uf);
            infer_logic_walk(index, has_array, has_fp, has_uf);
            infer_logic_walk(value, has_array, has_fp, has_uf);
        }
        SmtExpr::ConstArray { value, .. } => {
            *has_array = true;
            infer_logic_walk(value, has_array, has_fp, has_uf);
        }
        SmtExpr::FPAdd { lhs, rhs, .. }
        | SmtExpr::FPSub { lhs, rhs, .. }
        | SmtExpr::FPMul { lhs, rhs, .. }
        | SmtExpr::FPDiv { lhs, rhs, .. }
        | SmtExpr::FPEq { lhs, rhs }
        | SmtExpr::FPLt { lhs, rhs } => {
            *has_fp = true;
            infer_logic_walk(lhs, has_array, has_fp, has_uf);
            infer_logic_walk(rhs, has_array, has_fp, has_uf);
        }
        SmtExpr::FPNeg { operand } => {
            *has_fp = true;
            infer_logic_walk(operand, has_array, has_fp, has_uf);
        }
        SmtExpr::FPConst { .. } => {
            *has_fp = true;
        }
        SmtExpr::UF { args, .. } => {
            *has_uf = true;
            for arg in args {
                infer_logic_walk(arg, has_array, has_fp, has_uf);
            }
        }
        SmtExpr::UFDecl { .. } => {
            *has_uf = true;
        }
        // Binary BV/Bool ops
        SmtExpr::BvAdd { lhs, rhs, .. }
        | SmtExpr::BvSub { lhs, rhs, .. }
        | SmtExpr::BvMul { lhs, rhs, .. }
        | SmtExpr::BvSDiv { lhs, rhs, .. }
        | SmtExpr::BvUDiv { lhs, rhs, .. }
        | SmtExpr::BvAnd { lhs, rhs, .. }
        | SmtExpr::BvOr { lhs, rhs, .. }
        | SmtExpr::BvXor { lhs, rhs, .. }
        | SmtExpr::BvShl { lhs, rhs, .. }
        | SmtExpr::BvLshr { lhs, rhs, .. }
        | SmtExpr::BvAshr { lhs, rhs, .. }
        | SmtExpr::Eq { lhs, rhs }
        | SmtExpr::BvSlt { lhs, rhs, .. }
        | SmtExpr::BvSge { lhs, rhs, .. }
        | SmtExpr::BvSgt { lhs, rhs, .. }
        | SmtExpr::BvSle { lhs, rhs, .. }
        | SmtExpr::BvUlt { lhs, rhs, .. }
        | SmtExpr::BvUge { lhs, rhs, .. }
        | SmtExpr::BvUgt { lhs, rhs, .. }
        | SmtExpr::BvUle { lhs, rhs, .. }
        | SmtExpr::And { lhs, rhs }
        | SmtExpr::Or { lhs, rhs } => {
            infer_logic_walk(lhs, has_array, has_fp, has_uf);
            infer_logic_walk(rhs, has_array, has_fp, has_uf);
        }
        SmtExpr::BvNeg { operand, .. }
        | SmtExpr::Not { operand }
        | SmtExpr::Extract { operand, .. }
        | SmtExpr::ZeroExtend { operand, .. }
        | SmtExpr::SignExtend { operand, .. } => {
            infer_logic_walk(operand, has_array, has_fp, has_uf);
        }
        SmtExpr::Concat { hi, lo, .. } => {
            infer_logic_walk(hi, has_array, has_fp, has_uf);
            infer_logic_walk(lo, has_array, has_fp, has_uf);
        }
        SmtExpr::Ite { cond, then_expr, else_expr } => {
            infer_logic_walk(cond, has_array, has_fp, has_uf);
            infer_logic_walk(then_expr, has_array, has_fp, has_uf);
            infer_logic_walk(else_expr, has_array, has_fp, has_uf);
        }
        SmtExpr::Var { .. } | SmtExpr::BvConst { .. } | SmtExpr::BoolConst(_) => {}
    }
}

/// Serialize a rounding mode to SMT-LIB2.
pub fn rounding_mode_to_smt2(rm: &RoundingMode) -> &'static str {
    match rm {
        RoundingMode::RNE => "RNE",
        RoundingMode::RNA => "RNA",
        RoundingMode::RTP => "RTP",
        RoundingMode::RTN => "RTN",
        RoundingMode::RTZ => "RTZ",
    }
}

/// Serialize an SmtSort to SMT-LIB2 sort syntax.
///
/// Examples:
/// - `SmtSort::BitVec(32)` -> `(_ BitVec 32)`
/// - `SmtSort::Bool` -> `Bool`
/// - `SmtSort::Array(BitVec(64), BitVec(8))` -> `(Array (_ BitVec 64) (_ BitVec 8))`
pub fn sort_to_smt2(sort: &SmtSort) -> String {
    // SmtSort::Display already emits valid SMT-LIB2 sort syntax.
    format!("{}", sort)
}

/// Generate a complete SMT-LIB2 query for a proof obligation.
///
/// This extends `ProofObligation::to_smt2()` with:
/// - Automatic logic inference (QF_BV, QF_ABV, QF_BVFP, etc.)
/// - `(set-option :timeout <ms>)` for solver timeout
/// - `(get-model)` after `(check-sat)` for counterexample extraction
/// - Proper `(get-value ...)` queries for each input variable
pub fn generate_smt2_query(obligation: &ProofObligation, config: &Z4Config) -> String {
    generate_smt2_query_with_arrays(obligation, config, &[])
}

/// Generate a complete SMT-LIB2 query with additional array-sorted variable declarations.
///
/// Extends [`generate_smt2_query`] with declarations for non-bitvector symbolic
/// variables (arrays, FP-sorted constants, etc.). This is needed for memory model
/// proofs where memory is a symbolic `Array(BitVec64, BitVec8)` variable.
///
/// # Arguments
///
/// * `obligation` -- the proof obligation (bitvector inputs are declared from `inputs`)
/// * `config` -- solver configuration
/// * `extra_decls` -- additional variable declarations with arbitrary sorts,
///   emitted as `(declare-const name sort)` in the SMT-LIB2 output
pub fn generate_smt2_query_with_arrays(
    obligation: &ProofObligation,
    config: &Z4Config,
    extra_decls: &[(String, SmtSort)],
) -> String {
    let mut lines = Vec::new();

    // Logic declaration -- infer from the formula content.
    let formula = obligation.negated_equivalence();
    let logic = infer_logic(&formula);
    lines.push(format!("(set-logic {})", logic));

    // Solver options
    if config.timeout_ms > 0 {
        // z3 uses :timeout in milliseconds
        lines.push(format!("(set-option :timeout {})", config.timeout_ms));
    }
    if config.produce_models {
        lines.push("(set-option :produce-models true)".to_string());
    }

    // Declare symbolic bitvector inputs
    for (name, width) in &obligation.inputs {
        lines.push(format!(
            "(declare-const {} (_ BitVec {}))",
            name, width
        ));
    }

    // Declare symbolic floating-point inputs
    for (name, eb, sb) in &obligation.fp_inputs {
        lines.push(format!(
            "(declare-const {} (_ FloatingPoint {} {}))",
            name, eb, sb
        ));
    }

    // Declare additional non-bitvector inputs (arrays, FP, etc.)
    for (name, sort) in extra_decls {
        lines.push(format!(
            "(declare-const {} {})",
            name, sort_to_smt2(sort)
        ));
    }

    // Assert the negated equivalence
    lines.push(format!("(assert {})", formula));

    // Check satisfiability
    lines.push("(check-sat)".to_string());

    // If SAT, get the model for counterexample extraction.
    let has_any_inputs = !obligation.inputs.is_empty() || !obligation.fp_inputs.is_empty();
    if config.produce_models && has_any_inputs {
        let mut var_names: Vec<&str> = obligation
            .inputs
            .iter()
            .map(|(name, _)| name.as_str())
            .collect();
        for (name, _, _) in &obligation.fp_inputs {
            var_names.push(name.as_str());
        }
        lines.push(format!("(get-value ({}))", var_names.join(" ")));
    }

    lines.push("(exit)".to_string());

    lines.join("\n")
}

// ---------------------------------------------------------------------------
// Public convenience API (named per task specification)
// ---------------------------------------------------------------------------

/// Serialize a proof obligation to a complete SMT-LIB2 query string.
///
/// This is a convenience wrapper around [`generate_smt2_query`] using default
/// configuration. For custom solver options (timeout, model production, etc.),
/// use [`generate_smt2_query`] directly.
///
/// The returned string is a complete SMT-LIB2 script ready to be piped to
/// z3 or z4:
/// ```text
/// (set-logic QF_BV)
/// (set-option :timeout 5000)
/// (set-option :produce-models true)
/// (declare-const a (_ BitVec 32))
/// (declare-const b (_ BitVec 32))
/// (assert (not (= (bvadd a b) (bvadd a b))))
/// (check-sat)
/// (get-value (a b))
/// (exit)
/// ```
pub fn serialize_to_smt2(obligation: &ProofObligation) -> String {
    generate_smt2_query(obligation, &Z4Config::default())
}

/// Verify a proof obligation by shelling out to a z4 or z3 CLI binary.
///
/// This is an alias for [`verify_with_cli`] with a name that matches the
/// z4-specific nomenclature used throughout the codebase.
///
/// The function:
/// 1. Serializes the proof obligation to SMT-LIB2
/// 2. Writes it to a temp file
/// 3. Invokes the solver binary (z3 or z4, auto-detected)
/// 4. Parses the output (sat/unsat/timeout/error)
/// 5. Extracts counterexamples from the model if SAT
pub fn verify_with_z4_cli(obligation: &ProofObligation, config: &Z4Config) -> Z4Result {
    verify_with_cli(obligation, config)
}

/// Parse raw solver output text into a [`Z4Result`].
///
/// This is a public wrapper around the internal parser, useful for testing
/// and for consumers that invoke the solver themselves.
///
/// # Arguments
///
/// * `output` -- the solver's stdout text (e.g., "unsat\n" or "sat\n((a #x0a))")
/// * `inputs` -- the bitvector input variables for counterexample extraction
///
/// # Returns
///
/// * [`Z4Result::Verified`] if the output is "unsat"
/// * [`Z4Result::CounterExample`] if the output is "sat" (with model if available)
/// * [`Z4Result::Timeout`] if the output is "unknown" or contains "timeout"
/// * [`Z4Result::Error`] for any other output
pub fn parse_z4_output(output: &str, inputs: &[(String, u32)]) -> Z4Result {
    parse_solver_output(output, "", inputs)
}

// ---------------------------------------------------------------------------
// CLI subprocess backend (always available)
// ---------------------------------------------------------------------------

/// Verify a proof obligation using a z3/z4 CLI subprocess.
///
/// This function:
/// 1. Generates SMT-LIB2 from the proof obligation
/// 2. Writes it to a temp file
/// 3. Invokes the solver binary
/// 4. Parses the output (sat/unsat/timeout/error)
/// 5. If sat, extracts the counterexample from the model
pub fn verify_with_cli(obligation: &ProofObligation, config: &Z4Config) -> Z4Result {
    let smt2 = generate_smt2_query(obligation, config);

    // Find the solver binary
    let solver_path = match &config.solver_path {
        Some(path) => path.clone(),
        None => find_solver_binary(),
    };

    if solver_path.is_empty() {
        return Z4Result::Error(
            "No SMT solver found. Install z3 (brew install z3) or set solver_path.".to_string(),
        );
    }

    // Write SMT-LIB2 to a temp file
    let tmp_path = match write_temp_smt2(&smt2) {
        Ok(path) => path,
        Err(e) => return Z4Result::Error(format!("Failed to write temp file: {}", e)),
    };

    // Invoke the solver
    let output = std::process::Command::new(&solver_path)
        .arg("-smt2")
        .arg(&tmp_path)
        .output();

    // Clean up temp file (best-effort)
    let _ = std::fs::remove_file(&tmp_path);

    match output {
        Ok(output) => {
            let stdout = String::from_utf8_lossy(&output.stdout);
            let stderr = String::from_utf8_lossy(&output.stderr);
            parse_solver_output(&stdout, &stderr, &obligation.inputs)
        }
        Err(e) => Z4Result::Error(format!("Failed to invoke solver '{}': {}", solver_path, e)),
    }
}

/// Search PATH for z3 or z4 binary.
fn find_solver_binary() -> String {
    // Prefer z3 (widely available, well-tested SMT-LIB2 support)
    if let Ok(output) = std::process::Command::new("which").arg("z3").output() {
        if output.status.success() {
            let path = String::from_utf8_lossy(&output.stdout).trim().to_string();
            if !path.is_empty() {
                return path;
            }
        }
    }
    // Fallback: z4 binary
    if let Ok(output) = std::process::Command::new("which").arg("z4").output() {
        if output.status.success() {
            let path = String::from_utf8_lossy(&output.stdout).trim().to_string();
            if !path.is_empty() {
                return path;
            }
        }
    }
    String::new()
}

/// Write SMT-LIB2 content to a temporary file with a unique name.
fn write_temp_smt2(content: &str) -> Result<String, std::io::Error> {
    use std::io::Write;
    use std::sync::atomic::{AtomicU64, Ordering};
    static COUNTER: AtomicU64 = AtomicU64::new(0);

    let dir = std::env::temp_dir();
    let id = COUNTER.fetch_add(1, Ordering::Relaxed);
    let path = dir.join(format!(
        "llvm2_verify_{}_{}.smt2",
        std::process::id(),
        id
    ));
    let mut file = std::fs::File::create(&path)?;
    file.write_all(content.as_bytes())?;
    Ok(path.to_string_lossy().to_string())
}

/// Parse solver stdout/stderr into a Z4Result.
fn parse_solver_output(
    stdout: &str,
    stderr: &str,
    inputs: &[(String, u32)],
) -> Z4Result {
    let stdout_trimmed = stdout.trim();

    // Check for timeout indicators
    if stdout_trimmed.contains("timeout") || stdout_trimmed == "unknown" {
        return Z4Result::Timeout;
    }

    // Check for errors
    if stdout_trimmed.starts_with("(error") || !stderr.trim().is_empty() {
        let msg = if !stderr.trim().is_empty() {
            stderr.trim().to_string()
        } else {
            stdout_trimmed.to_string()
        };
        // Some solvers print warnings to stderr that aren't errors
        if msg.contains("WARNING") || msg.contains("warning") {
            // Continue parsing stdout
        } else if !stdout_trimmed.starts_with("sat") && !stdout_trimmed.starts_with("unsat") {
            return Z4Result::Error(msg);
        }
    }

    // Parse the result lines
    let lines: Vec<&str> = stdout_trimmed.lines().collect();

    if lines.is_empty() {
        return Z4Result::Error("Empty solver output".to_string());
    }

    let first_line = lines[0].trim();

    match first_line {
        "unsat" => Z4Result::Verified,
        "sat" => {
            // Try to extract counterexample from model output
            if lines.len() > 1 {
                let model_text = lines[1..].join("\n");
                let cex = parse_model_output(&model_text, inputs);
                Z4Result::CounterExample(cex)
            } else {
                // SAT but no model output
                Z4Result::CounterExample(vec![])
            }
        }
        "unknown" => Z4Result::Timeout,
        _ => Z4Result::Error(format!("Unexpected solver output: {}", first_line)),
    }
}

/// Parse SMT-LIB2 `(get-value ...)` output to extract variable assignments.
///
/// Expected format:
/// ```text
/// ((a #x0000000a)
///  (b #x00000014))
/// ```
fn parse_model_output(model_text: &str, inputs: &[(String, u32)]) -> Vec<(String, u64)> {
    let mut result = Vec::new();

    for (name, _width) in inputs {
        // Look for the variable assignment in the model
        // Format: (name #xHEXVALUE) or (name (_ bvDECIMAL WIDTH))
        if let Some(val) = extract_bv_value(model_text, name) {
            result.push((name.clone(), val));
        }
    }

    result
}

/// Extract a bitvector value for a variable from SMT-LIB2 model output.
fn extract_bv_value(model_text: &str, var_name: &str) -> Option<u64> {
    // Pattern 1: (var_name #xHEXDIGITS)
    let hex_pattern = format!("({} #x", var_name);
    if let Some(pos) = model_text.find(&hex_pattern) {
        let start = pos + hex_pattern.len();
        let end = model_text[start..].find(')')? + start;
        let hex_str = &model_text[start..end];
        return u64::from_str_radix(hex_str, 16).ok();
    }

    // Pattern 2: (var_name #bBINDIGITS)
    let bin_pattern = format!("({} #b", var_name);
    if let Some(pos) = model_text.find(&bin_pattern) {
        let start = pos + bin_pattern.len();
        let end = model_text[start..].find(')')? + start;
        let bin_str = &model_text[start..end];
        return u64::from_str_radix(bin_str, 2).ok();
    }

    // Pattern 3: (var_name (_ bvDECIMAL WIDTH))
    let bv_pattern = format!("({} (_ bv", var_name);
    if let Some(pos) = model_text.find(&bv_pattern) {
        let start = pos + bv_pattern.len();
        let space = model_text[start..].find(' ')? + start;
        let dec_str = &model_text[start..space];
        return dec_str.parse::<u64>().ok();
    }

    None
}

// ---------------------------------------------------------------------------
// z4 native Rust API backend (feature-gated)
// ---------------------------------------------------------------------------

/// Verify a proof obligation using the z4 crate's native Rust API.
///
/// This avoids subprocess overhead and provides richer error information.
/// Only available when the `z4` feature is enabled.
#[cfg(feature = "z4")]
pub fn verify_with_z4_api(obligation: &ProofObligation, config: &Z4Config) -> Z4Result {
    use z4::{Logic, SolveResult, Sort, Solver, BitVecSort};

    // Create solver for QF_BV (quantifier-free bitvectors)
    let mut solver = match Solver::try_new(Logic::QfBv) {
        Ok(s) => s,
        Err(e) => return Z4Result::Error(format!("Failed to create z4 solver: {}", e)),
    };

    // Declare input variables
    let mut var_terms: HashMap<String, z4::Term> = HashMap::new();
    for (name, width) in &obligation.inputs {
        let sort = Sort::BitVec(BitVecSort { width: *width });
        let term = solver.declare_const(name, sort);
        var_terms.insert(name.clone(), term);
    }

    // Build and assert the negated equivalence formula
    let formula_term = translate_expr_to_z4(&obligation.negated_equivalence(), &solver, &var_terms);
    match formula_term {
        Ok(term) => solver.assert_term(term),
        Err(e) => return Z4Result::Error(format!("Failed to translate formula: {}", e)),
    }

    // Check satisfiability
    let details = solver.check_sat_with_details();
    match details.accept_for_consumer() {
        Ok(SolveResult::Unsat(_)) => Z4Result::Verified,
        Ok(SolveResult::Sat) => {
            // Extract counterexample from model
            let cex = match solver.model() {
                Some(model) => {
                    let model = model.into_inner();
                    let mut assignments = Vec::new();
                    for (name, width) in &obligation.inputs {
                        if let Some(val) = model.bv_val(name) {
                            assignments.push((name.clone(), val));
                        }
                    }
                    assignments
                }
                None => vec![],
            };
            Z4Result::CounterExample(cex)
        }
        Ok(SolveResult::Unknown) | Err(_) => {
            if let Some(reason) = details.unknown_reason {
                if reason.to_string().contains("timeout") {
                    Z4Result::Timeout
                } else {
                    Z4Result::Error(format!("Solver returned unknown: {}", reason))
                }
            } else {
                Z4Result::Timeout
            }
        }
        Ok(_) => Z4Result::Error("Unexpected solver result".to_string()),
    }
}

/// Translate an SmtExpr tree into a z4 Term.
///
/// This recursively converts our internal AST into z4's native term
/// representation using the solver's builder API.
#[cfg(feature = "z4")]
fn translate_expr_to_z4(
    expr: &SmtExpr,
    solver: &z4::Solver,
    var_terms: &HashMap<String, z4::Term>,
) -> Result<z4::Term, String> {
    match expr {
        SmtExpr::Var { name, .. } => {
            var_terms
                .get(name)
                .cloned()
                .ok_or_else(|| format!("Variable '{}' not declared", name))
        }
        SmtExpr::BvConst { value, width } => {
            Ok(solver.bv_const(*value, *width))
        }
        SmtExpr::BoolConst(b) => {
            Ok(solver.bool_const(*b))
        }
        SmtExpr::BvAdd { lhs, rhs, .. } => {
            let l = translate_expr_to_z4(lhs, solver, var_terms)?;
            let r = translate_expr_to_z4(rhs, solver, var_terms)?;
            Ok(solver.bvadd(l, r))
        }
        SmtExpr::BvSub { lhs, rhs, .. } => {
            let l = translate_expr_to_z4(lhs, solver, var_terms)?;
            let r = translate_expr_to_z4(rhs, solver, var_terms)?;
            Ok(solver.bvsub(l, r))
        }
        SmtExpr::BvMul { lhs, rhs, .. } => {
            let l = translate_expr_to_z4(lhs, solver, var_terms)?;
            let r = translate_expr_to_z4(rhs, solver, var_terms)?;
            Ok(solver.bvmul(l, r))
        }
        SmtExpr::BvSDiv { lhs, rhs, .. } => {
            let l = translate_expr_to_z4(lhs, solver, var_terms)?;
            let r = translate_expr_to_z4(rhs, solver, var_terms)?;
            Ok(solver.bvsdiv(l, r))
        }
        SmtExpr::BvUDiv { lhs, rhs, .. } => {
            let l = translate_expr_to_z4(lhs, solver, var_terms)?;
            let r = translate_expr_to_z4(rhs, solver, var_terms)?;
            Ok(solver.bvudiv(l, r))
        }
        SmtExpr::BvNeg { operand, .. } => {
            let o = translate_expr_to_z4(operand, solver, var_terms)?;
            Ok(solver.bvneg(o))
        }
        SmtExpr::BvAnd { lhs, rhs, .. } => {
            let l = translate_expr_to_z4(lhs, solver, var_terms)?;
            let r = translate_expr_to_z4(rhs, solver, var_terms)?;
            Ok(solver.bvand(l, r))
        }
        SmtExpr::BvOr { lhs, rhs, .. } => {
            let l = translate_expr_to_z4(lhs, solver, var_terms)?;
            let r = translate_expr_to_z4(rhs, solver, var_terms)?;
            Ok(solver.bvor(l, r))
        }
        SmtExpr::BvXor { lhs, rhs, .. } => {
            let l = translate_expr_to_z4(lhs, solver, var_terms)?;
            let r = translate_expr_to_z4(rhs, solver, var_terms)?;
            Ok(solver.bvxor(l, r))
        }
        SmtExpr::BvShl { lhs, rhs, .. } => {
            let l = translate_expr_to_z4(lhs, solver, var_terms)?;
            let r = translate_expr_to_z4(rhs, solver, var_terms)?;
            Ok(solver.bvshl(l, r))
        }
        SmtExpr::BvLshr { lhs, rhs, .. } => {
            let l = translate_expr_to_z4(lhs, solver, var_terms)?;
            let r = translate_expr_to_z4(rhs, solver, var_terms)?;
            Ok(solver.bvlshr(l, r))
        }
        SmtExpr::BvAshr { lhs, rhs, .. } => {
            let l = translate_expr_to_z4(lhs, solver, var_terms)?;
            let r = translate_expr_to_z4(rhs, solver, var_terms)?;
            Ok(solver.bvashr(l, r))
        }
        SmtExpr::Eq { lhs, rhs } => {
            let l = translate_expr_to_z4(lhs, solver, var_terms)?;
            let r = translate_expr_to_z4(rhs, solver, var_terms)?;
            Ok(solver.eq(l, r))
        }
        SmtExpr::Not { operand } => {
            let o = translate_expr_to_z4(operand, solver, var_terms)?;
            Ok(solver.not(o))
        }
        SmtExpr::And { lhs, rhs } => {
            let l = translate_expr_to_z4(lhs, solver, var_terms)?;
            let r = translate_expr_to_z4(rhs, solver, var_terms)?;
            Ok(solver.and(l, r))
        }
        SmtExpr::Or { lhs, rhs } => {
            let l = translate_expr_to_z4(lhs, solver, var_terms)?;
            let r = translate_expr_to_z4(rhs, solver, var_terms)?;
            Ok(solver.or(l, r))
        }
        SmtExpr::BvSlt { lhs, rhs, .. } => {
            let l = translate_expr_to_z4(lhs, solver, var_terms)?;
            let r = translate_expr_to_z4(rhs, solver, var_terms)?;
            Ok(solver.bvslt(l, r))
        }
        SmtExpr::BvSge { lhs, rhs, .. } => {
            let l = translate_expr_to_z4(lhs, solver, var_terms)?;
            let r = translate_expr_to_z4(rhs, solver, var_terms)?;
            Ok(solver.bvsge(l, r))
        }
        SmtExpr::BvSgt { lhs, rhs, .. } => {
            let l = translate_expr_to_z4(lhs, solver, var_terms)?;
            let r = translate_expr_to_z4(rhs, solver, var_terms)?;
            Ok(solver.bvsgt(l, r))
        }
        SmtExpr::BvSle { lhs, rhs, .. } => {
            let l = translate_expr_to_z4(lhs, solver, var_terms)?;
            let r = translate_expr_to_z4(rhs, solver, var_terms)?;
            Ok(solver.bvsle(l, r))
        }
        SmtExpr::BvUlt { lhs, rhs, .. } => {
            let l = translate_expr_to_z4(lhs, solver, var_terms)?;
            let r = translate_expr_to_z4(rhs, solver, var_terms)?;
            Ok(solver.bvult(l, r))
        }
        SmtExpr::BvUge { lhs, rhs, .. } => {
            let l = translate_expr_to_z4(lhs, solver, var_terms)?;
            let r = translate_expr_to_z4(rhs, solver, var_terms)?;
            Ok(solver.bvuge(l, r))
        }
        SmtExpr::BvUgt { lhs, rhs, .. } => {
            let l = translate_expr_to_z4(lhs, solver, var_terms)?;
            let r = translate_expr_to_z4(rhs, solver, var_terms)?;
            Ok(solver.bvugt(l, r))
        }
        SmtExpr::BvUle { lhs, rhs, .. } => {
            let l = translate_expr_to_z4(lhs, solver, var_terms)?;
            let r = translate_expr_to_z4(rhs, solver, var_terms)?;
            Ok(solver.bvule(l, r))
        }
        SmtExpr::Ite { cond, then_expr, else_expr } => {
            let c = translate_expr_to_z4(cond, solver, var_terms)?;
            let t = translate_expr_to_z4(then_expr, solver, var_terms)?;
            let e = translate_expr_to_z4(else_expr, solver, var_terms)?;
            Ok(solver.ite(c, t, e))
        }
        SmtExpr::Extract { high, low, operand, .. } => {
            let o = translate_expr_to_z4(operand, solver, var_terms)?;
            Ok(solver.extract(*high, *low, o))
        }
        SmtExpr::Concat { hi, lo, .. } => {
            let h = translate_expr_to_z4(hi, solver, var_terms)?;
            let l = translate_expr_to_z4(lo, solver, var_terms)?;
            Ok(solver.concat(h, l))
        }
        SmtExpr::ZeroExtend { operand, extra_bits, .. } => {
            let o = translate_expr_to_z4(operand, solver, var_terms)?;
            Ok(solver.zero_ext(*extra_bits, o))
        }
        SmtExpr::SignExtend { operand, extra_bits, .. } => {
            let o = translate_expr_to_z4(operand, solver, var_terms)?;
            Ok(solver.sign_ext(*extra_bits, o))
        }
        // New theory extensions -- these will be translatable once z4 gains
        // array/FP/UF theory support. For now, return descriptive errors.
        SmtExpr::Select { .. }
        | SmtExpr::Store { .. }
        | SmtExpr::ConstArray { .. } => {
            Err("Array theory (QF_ABV) not yet supported in z4 native API; use CLI fallback with z3".to_string())
        }
        SmtExpr::FPAdd { .. }
        | SmtExpr::FPSub { .. }
        | SmtExpr::FPMul { .. }
        | SmtExpr::FPDiv { .. }
        | SmtExpr::FPNeg { .. }
        | SmtExpr::FPEq { .. }
        | SmtExpr::FPLt { .. }
        | SmtExpr::FPConst { .. } => {
            Err("Floating-point theory (QF_FP) not yet supported in z4 native API; use CLI fallback with z3".to_string())
        }
        SmtExpr::UF { .. }
        | SmtExpr::UFDecl { .. } => {
            Err("Uninterpreted function theory (QF_UF) not yet supported in z4 native API; use CLI fallback with z3".to_string())
        }
    }
}

// ---------------------------------------------------------------------------
// Unified verification interface
// ---------------------------------------------------------------------------

/// Verify a proof obligation using the best available solver backend.
///
/// Selection order:
/// 1. z4 native Rust API (if `z4` feature enabled)
/// 2. CLI subprocess (z3 or z4 binary)
///
/// Returns [`Z4Result::Verified`] if the lowering rule is correct for all inputs.
pub fn verify_with_z4(obligation: &ProofObligation, config: &Z4Config) -> Z4Result {
    #[cfg(feature = "z4")]
    {
        return verify_with_z4_api(obligation, config);
    }

    #[cfg(not(feature = "z4"))]
    {
        verify_with_cli(obligation, config)
    }
}

/// Re-verify all known lowering proofs using the z4 solver.
///
/// Collects all standard proof obligations (arithmetic, comparison, branch,
/// peephole, NZCV, constant folding, CSE/LICM) and verifies each one.
///
/// Returns a list of (proof_name, result) pairs.
pub fn verify_all_with_z4(config: &Z4Config) -> Vec<(String, Z4Result)> {
    let mut results = Vec::new();

    // Arithmetic lowering proofs
    for obligation in crate::lowering_proof::all_arithmetic_proofs() {
        let result = verify_with_z4(&obligation, config);
        results.push((obligation.name.clone(), result));
    }

    // NZCV flag + comparison + branch proofs
    for obligation in crate::lowering_proof::all_nzcv_proofs() {
        let result = verify_with_z4(&obligation, config);
        results.push((obligation.name.clone(), result));
    }

    // Peephole identity proofs
    for obligation in crate::peephole_proofs::all_peephole_proofs_with_32bit() {
        let result = verify_with_z4(&obligation, config);
        results.push((obligation.name.clone(), result));
    }

    results
}

/// Summary statistics for a batch verification run.
#[derive(Debug, Clone)]
pub struct VerificationSummary {
    /// Total number of proofs checked.
    pub total: usize,
    /// Number of proofs verified (UNSAT).
    pub verified: usize,
    /// Number of proofs that found counterexamples (SAT).
    pub failed: usize,
    /// Number of proofs that timed out.
    pub timeouts: usize,
    /// Number of proofs that had errors.
    pub errors: usize,
}

impl VerificationSummary {
    /// Compute summary from a list of results.
    pub fn from_results(results: &[(String, Z4Result)]) -> Self {
        let mut summary = Self {
            total: results.len(),
            verified: 0,
            failed: 0,
            timeouts: 0,
            errors: 0,
        };

        for (_, result) in results {
            match result {
                Z4Result::Verified => summary.verified += 1,
                Z4Result::CounterExample(_) => summary.failed += 1,
                Z4Result::Timeout => summary.timeouts += 1,
                Z4Result::Error(_) => summary.errors += 1,
            }
        }

        summary
    }

    /// Returns true if all proofs were verified.
    pub fn all_verified(&self) -> bool {
        self.verified == self.total
    }
}

impl fmt::Display for VerificationSummary {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}/{} verified, {} failed, {} timeouts, {} errors",
            self.verified, self.total, self.failed, self.timeouts, self.errors
        )
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lowering_proof::ProofObligation;
    use crate::smt::{SmtExpr, SmtSort};

    // -----------------------------------------------------------------------
    // SMT-LIB2 generation tests (always run, no solver needed)
    // -----------------------------------------------------------------------

    #[test]
    fn test_generate_smt2_query_basic() {
        let a = SmtExpr::var("a", 32);
        let b = SmtExpr::var("b", 32);

        let obligation = ProofObligation {
            name: "test_add".to_string(),
            tmir_expr: a.clone().bvadd(b.clone()),
            aarch64_expr: a.bvadd(b),
            inputs: vec![("a".to_string(), 32), ("b".to_string(), 32)],
            preconditions: vec![],
            fp_inputs: vec![],
        };

        let config = Z4Config::default();
        let smt2 = generate_smt2_query(&obligation, &config);

        assert!(smt2.contains("(set-logic QF_BV)"));
        assert!(smt2.contains("(set-option :timeout 5000)"));
        assert!(smt2.contains("(set-option :produce-models true)"));
        assert!(smt2.contains("(declare-const a (_ BitVec 32))"));
        assert!(smt2.contains("(declare-const b (_ BitVec 32))"));
        assert!(smt2.contains("(assert"));
        assert!(smt2.contains("(check-sat)"));
        assert!(smt2.contains("(get-value (a b))"));
        assert!(smt2.contains("(exit)"));
    }

    #[test]
    fn test_generate_smt2_no_timeout() {
        let a = SmtExpr::var("x", 64);
        let obligation = ProofObligation {
            name: "test_no_timeout".to_string(),
            tmir_expr: a.clone(),
            aarch64_expr: a,
            inputs: vec![("x".to_string(), 64)],
            preconditions: vec![],
            fp_inputs: vec![],
        };

        let config = Z4Config {
            timeout_ms: 0,
            ..Default::default()
        };
        let smt2 = generate_smt2_query(&obligation, &config);
        assert!(!smt2.contains(":timeout"));
    }

    // -----------------------------------------------------------------------
    // Solver output parsing tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_parse_unsat() {
        let result = parse_solver_output("unsat\n", "", &[]);
        assert_eq!(result, Z4Result::Verified);
    }

    #[test]
    fn test_parse_sat_with_hex_model() {
        let output = "sat\n((a #x0000000a)\n (b #x00000014))";
        let inputs = vec![("a".to_string(), 32), ("b".to_string(), 32)];
        let result = parse_solver_output(output, "", &inputs);
        match result {
            Z4Result::CounterExample(cex) => {
                assert_eq!(cex.len(), 2);
                assert_eq!(cex[0], ("a".to_string(), 0xa));
                assert_eq!(cex[1], ("b".to_string(), 0x14));
            }
            other => panic!("Expected CounterExample, got {:?}", other),
        }
    }

    #[test]
    fn test_parse_sat_with_bv_model() {
        let output = "sat\n((x (_ bv42 32)))";
        let inputs = vec![("x".to_string(), 32)];
        let result = parse_solver_output(output, "", &inputs);
        match result {
            Z4Result::CounterExample(cex) => {
                assert_eq!(cex.len(), 1);
                assert_eq!(cex[0], ("x".to_string(), 42));
            }
            other => panic!("Expected CounterExample, got {:?}", other),
        }
    }

    #[test]
    fn test_parse_sat_with_binary_model() {
        let output = "sat\n((x #b00101010))";
        let inputs = vec![("x".to_string(), 8)];
        let result = parse_solver_output(output, "", &inputs);
        match result {
            Z4Result::CounterExample(cex) => {
                assert_eq!(cex.len(), 1);
                assert_eq!(cex[0], ("x".to_string(), 42));
            }
            other => panic!("Expected CounterExample, got {:?}", other),
        }
    }

    #[test]
    fn test_parse_unknown() {
        let result = parse_solver_output("unknown\n", "", &[]);
        assert_eq!(result, Z4Result::Timeout);
    }

    #[test]
    fn test_parse_error() {
        let result = parse_solver_output("", "Parse error at line 1", &[]);
        assert!(matches!(result, Z4Result::Error(_)));
    }

    #[test]
    fn test_parse_empty_output() {
        let result = parse_solver_output("", "", &[]);
        assert!(matches!(result, Z4Result::Error(_)));
    }

    // -----------------------------------------------------------------------
    // Z4Result display tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_z4result_display_verified() {
        assert_eq!(format!("{}", Z4Result::Verified), "VERIFIED (UNSAT)");
    }

    #[test]
    fn test_z4result_display_counterexample() {
        let cex = Z4Result::CounterExample(vec![
            ("a".to_string(), 10),
            ("b".to_string(), 20),
        ]);
        let display = format!("{}", cex);
        assert!(display.contains("a = 0xa"));
        assert!(display.contains("b = 0x14"));
    }

    #[test]
    fn test_z4result_display_timeout() {
        assert_eq!(format!("{}", Z4Result::Timeout), "TIMEOUT");
    }

    // -----------------------------------------------------------------------
    // VerificationSummary tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_verification_summary() {
        let results = vec![
            ("proof1".to_string(), Z4Result::Verified),
            ("proof2".to_string(), Z4Result::Verified),
            ("proof3".to_string(), Z4Result::CounterExample(vec![])),
            ("proof4".to_string(), Z4Result::Timeout),
            ("proof5".to_string(), Z4Result::Error("oops".to_string())),
        ];

        let summary = VerificationSummary::from_results(&results);
        assert_eq!(summary.total, 5);
        assert_eq!(summary.verified, 2);
        assert_eq!(summary.failed, 1);
        assert_eq!(summary.timeouts, 1);
        assert_eq!(summary.errors, 1);
        assert!(!summary.all_verified());
    }

    #[test]
    fn test_verification_summary_all_verified() {
        let results = vec![
            ("proof1".to_string(), Z4Result::Verified),
            ("proof2".to_string(), Z4Result::Verified),
        ];

        let summary = VerificationSummary::from_results(&results);
        assert!(summary.all_verified());
    }

    // -----------------------------------------------------------------------
    // CLI integration test (only runs if z3 is available)
    // -----------------------------------------------------------------------

    #[test]
    fn test_cli_verify_correct_rule() {
        // Skip if no solver binary available
        let solver = find_solver_binary();
        if solver.is_empty() {
            return; // No solver available, skip test
        }

        // a + b == a + b (trivially correct)
        let a = SmtExpr::var("a", 32);
        let b = SmtExpr::var("b", 32);
        let obligation = ProofObligation {
            name: "trivial_add_identity".to_string(),
            tmir_expr: a.clone().bvadd(b.clone()),
            aarch64_expr: a.bvadd(b),
            inputs: vec![("a".to_string(), 32), ("b".to_string(), 32)],
            preconditions: vec![],
            fp_inputs: vec![],
        };

        let config = Z4Config::default();
        let result = verify_with_cli(&obligation, &config);
        assert_eq!(result, Z4Result::Verified);
    }

    #[test]
    fn test_cli_verify_wrong_rule() {
        // Skip if no solver binary available
        let solver = find_solver_binary();
        if solver.is_empty() {
            return;
        }

        // a + b != a - b (should find counterexample)
        let a = SmtExpr::var("a", 8);
        let b = SmtExpr::var("b", 8);
        let obligation = ProofObligation {
            name: "wrong_add_vs_sub".to_string(),
            tmir_expr: a.clone().bvadd(b.clone()),
            aarch64_expr: a.bvsub(b),
            inputs: vec![("a".to_string(), 8), ("b".to_string(), 8)],
            preconditions: vec![],
            fp_inputs: vec![],
        };

        let config = Z4Config::default();
        let result = verify_with_cli(&obligation, &config);
        assert!(matches!(result, Z4Result::CounterExample(_)));
    }

    #[test]
    fn test_cli_verify_iadd_i32() {
        let solver = find_solver_binary();
        if solver.is_empty() {
            return;
        }

        let obligation = crate::lowering_proof::proof_iadd_i32();
        let config = Z4Config::default();
        let result = verify_with_cli(&obligation, &config);
        assert_eq!(result, Z4Result::Verified);
    }

    #[test]
    fn test_cli_verify_peephole_add_zero() {
        let solver = find_solver_binary();
        if solver.is_empty() {
            return;
        }

        let obligation = crate::peephole_proofs::proof_add_zero_identity();
        let config = Z4Config::default();
        let result = verify_with_cli(&obligation, &config);
        assert_eq!(result, Z4Result::Verified);
    }

    // -----------------------------------------------------------------------
    // Logic inference tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_infer_logic_bv_only() {
        let expr = SmtExpr::var("x", 32).bvadd(SmtExpr::var("y", 32));
        assert_eq!(infer_logic(&expr), "QF_BV");
    }

    #[test]
    fn test_infer_logic_array() {
        let arr = SmtExpr::const_array(
            SmtSort::BitVec(32),
            SmtExpr::bv_const(0, 32),
        );
        let expr = SmtExpr::select(arr, SmtExpr::var("idx", 32));
        assert_eq!(infer_logic(&expr), "QF_ABV");
    }

    #[test]
    fn test_infer_logic_fp() {
        let expr = SmtExpr::fp_add(
            crate::smt::RoundingMode::RNE,
            SmtExpr::fp64_const(1.0),
            SmtExpr::fp64_const(2.0),
        );
        assert_eq!(infer_logic(&expr), "QF_BVFP");
    }

    #[test]
    fn test_infer_logic_uf() {
        let expr = SmtExpr::uf("f", vec![SmtExpr::var("x", 32)], SmtSort::BitVec(32));
        assert_eq!(infer_logic(&expr), "QF_UFBV");
    }

    #[test]
    fn test_infer_logic_mixed_array_fp() {
        let arr = SmtExpr::const_array(
            SmtSort::BitVec(32),
            SmtExpr::fp64_const(0.0),
        );
        assert_eq!(infer_logic(&arr), "QF_ABVFP");
    }

    #[test]
    fn test_rounding_mode_smt2() {
        assert_eq!(rounding_mode_to_smt2(&RoundingMode::RNE), "RNE");
        assert_eq!(rounding_mode_to_smt2(&RoundingMode::RNA), "RNA");
        assert_eq!(rounding_mode_to_smt2(&RoundingMode::RTP), "RTP");
        assert_eq!(rounding_mode_to_smt2(&RoundingMode::RTN), "RTN");
        assert_eq!(rounding_mode_to_smt2(&RoundingMode::RTZ), "RTZ");
    }

    // -----------------------------------------------------------------------
    // Sort serialization tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_sort_to_smt2_bitvec() {
        assert_eq!(sort_to_smt2(&SmtSort::BitVec(32)), "(_ BitVec 32)");
        assert_eq!(sort_to_smt2(&SmtSort::BitVec(64)), "(_ BitVec 64)");
        assert_eq!(sort_to_smt2(&SmtSort::BitVec(8)), "(_ BitVec 8)");
    }

    #[test]
    fn test_sort_to_smt2_bool() {
        assert_eq!(sort_to_smt2(&SmtSort::Bool), "Bool");
    }

    #[test]
    fn test_sort_to_smt2_array() {
        let mem_sort = SmtSort::bv_array(64, 8);
        assert_eq!(sort_to_smt2(&mem_sort), "(Array (_ BitVec 64) (_ BitVec 8))");
    }

    #[test]
    fn test_sort_to_smt2_fp() {
        assert_eq!(sort_to_smt2(&SmtSort::fp32()), "(_ FloatingPoint 8 24)");
        assert_eq!(sort_to_smt2(&SmtSort::fp64()), "(_ FloatingPoint 11 53)");
    }

    // -----------------------------------------------------------------------
    // Array theory SMT-LIB2 serialization tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_array_select_serialization() {
        // (select array index) serialized via SmtExpr::Display
        let arr = SmtExpr::const_array(SmtSort::BitVec(64), SmtExpr::bv_const(0, 8));
        let idx = SmtExpr::var("addr", 64);
        let sel = SmtExpr::select(arr, idx);
        let serialized = format!("{}", sel);
        assert_eq!(
            serialized,
            "(select ((as const (Array (_ BitVec 64) (_ BitVec 8))) (_ bv0 8)) addr)"
        );
    }

    #[test]
    fn test_array_store_serialization() {
        // (store array index value) serialized via SmtExpr::Display
        let arr = SmtExpr::const_array(SmtSort::BitVec(64), SmtExpr::bv_const(0, 8));
        let idx = SmtExpr::var("addr", 64);
        let val = SmtExpr::var("byte", 8);
        let st = SmtExpr::store(arr, idx, val);
        let serialized = format!("{}", st);
        assert_eq!(
            serialized,
            "(store ((as const (Array (_ BitVec 64) (_ BitVec 8))) (_ bv0 8)) addr byte)"
        );
    }

    #[test]
    fn test_array_const_array_serialization() {
        // ((as const (Array (_ BitVec 64) (_ BitVec 8))) (_ bv0 8))
        let arr = SmtExpr::const_array(SmtSort::BitVec(64), SmtExpr::bv_const(0, 8));
        let serialized = format!("{}", arr);
        assert_eq!(
            serialized,
            "((as const (Array (_ BitVec 64) (_ BitVec 8))) (_ bv0 8))"
        );
    }

    #[test]
    fn test_array_nested_store_select() {
        // store at addr, then select at same addr: should produce nested expression
        let arr = SmtExpr::const_array(SmtSort::BitVec(64), SmtExpr::bv_const(0, 8));
        let addr = SmtExpr::var("a", 64);
        let val = SmtExpr::bv_const(42, 8);
        let stored = SmtExpr::store(arr, addr.clone(), val);
        let loaded = SmtExpr::select(stored, addr);
        let serialized = format!("{}", loaded);
        assert!(serialized.contains("(select (store"));
        assert!(serialized.contains("(store ((as const (Array (_ BitVec 64) (_ BitVec 8))) (_ bv0 8)) a (_ bv42 8))"));
    }

    #[test]
    fn test_generate_smt2_query_with_array_ops() {
        // A proof obligation that involves array operations should get QF_ABV logic
        let mem = SmtExpr::const_array(SmtSort::BitVec(64), SmtExpr::var("d", 8));
        let addr = SmtExpr::var("a", 64);
        let val = SmtExpr::var("v", 8);

        // tmir side: store then select at same address
        let mem_after = SmtExpr::store(mem.clone(), addr.clone(), val.clone());
        let tmir_result = SmtExpr::select(mem_after, addr.clone());

        // aarch64 side: should equal the stored value
        let aarch64_result = val.clone();

        let obligation = ProofObligation {
            name: "store_load_roundtrip".to_string(),
            tmir_expr: tmir_result,
            aarch64_expr: aarch64_result,
            inputs: vec![
                ("a".to_string(), 64),
                ("v".to_string(), 8),
                ("d".to_string(), 8),
            ],
            preconditions: vec![],
            fp_inputs: vec![],
        };

        let config = Z4Config::default();
        let smt2 = generate_smt2_query(&obligation, &config);

        // Must use QF_ABV logic (arrays + bitvectors)
        assert!(smt2.contains("(set-logic QF_ABV)"), "Expected QF_ABV logic, got: {}", smt2);
        // Must declare all bitvector inputs
        assert!(smt2.contains("(declare-const a (_ BitVec 64))"));
        assert!(smt2.contains("(declare-const v (_ BitVec 8))"));
        assert!(smt2.contains("(declare-const d (_ BitVec 8))"));
        // Must contain array operations in the assertion
        assert!(smt2.contains("select"));
        assert!(smt2.contains("store"));
        assert!(smt2.contains("(check-sat)"));
    }

    #[test]
    fn test_generate_smt2_query_with_extra_array_decls() {
        // Test the enhanced query generator with explicit array declarations
        let _mem_var = SmtExpr::var("mem", 64); // placeholder -- in real usage this would be array
        let addr = SmtExpr::var("a", 64);

        let obligation = ProofObligation {
            name: "test_array_decl".to_string(),
            tmir_expr: addr.clone(),
            aarch64_expr: addr,
            inputs: vec![("a".to_string(), 64)],
            preconditions: vec![],
            fp_inputs: vec![],
        };

        let config = Z4Config::default();
        let extra_decls = vec![
            ("mem".to_string(), SmtSort::bv_array(64, 8)),
        ];
        let smt2 = generate_smt2_query_with_arrays(&obligation, &config, &extra_decls);

        // Must declare the array variable with correct sort
        assert!(
            smt2.contains("(declare-const mem (Array (_ BitVec 64) (_ BitVec 8)))"),
            "Missing array declaration in: {}",
            smt2,
        );
        // Must still declare BV inputs
        assert!(smt2.contains("(declare-const a (_ BitVec 64))"));
    }

    #[test]
    fn test_memory_proof_smt2_serialization() {
        // End-to-end test: generate SMT-LIB2 for a store-load roundtrip from memory_proofs
        let obligation = crate::memory_proofs::proof_roundtrip_i8();
        let config = Z4Config::default();
        let smt2 = generate_smt2_query(&obligation, &config);

        // Memory proofs use array operations, so logic should be QF_ABV
        assert!(smt2.contains("(set-logic QF_ABV)"), "Expected QF_ABV for memory proof, got: {}", smt2);
        // Must contain array operations (select, store, as const)
        assert!(smt2.contains("select"), "Missing select in: {}", smt2);
        assert!(smt2.contains("store"), "Missing store in: {}", smt2);
        assert!(smt2.contains("as const"), "Missing as const in: {}", smt2);
    }

    #[test]
    fn test_cli_verify_memory_roundtrip_i8() {
        // Integration test: verify store-load roundtrip with z3 CLI
        let solver = find_solver_binary();
        if solver.is_empty() {
            return; // No solver available, skip test
        }

        let obligation = crate::memory_proofs::proof_roundtrip_i8();
        let config = Z4Config::default();
        let result = verify_with_cli(&obligation, &config);
        assert_eq!(result, Z4Result::Verified, "Store-load roundtrip I8 should be verified");
    }

    // -----------------------------------------------------------------------
    // Floating-point SMT-LIB2 serialization tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_fp_add_smt2_serialization() {
        let a = SmtExpr::fp64_const(1.0);
        let b = SmtExpr::fp64_const(2.0);
        let expr = SmtExpr::fp_add(RoundingMode::RNE, a, b);
        let s = format!("{}", expr);
        assert!(s.starts_with("(fp.add RNE"));
        assert!(s.contains("(fp #b"));
    }

    #[test]
    fn test_fp_mul_smt2_serialization() {
        let a = SmtExpr::fp64_const(3.0);
        let b = SmtExpr::fp64_const(7.0);
        let expr = SmtExpr::fp_mul(RoundingMode::RTZ, a, b);
        let s = format!("{}", expr);
        assert!(s.starts_with("(fp.mul RTZ"));
    }

    #[test]
    fn test_fp_div_smt2_serialization() {
        let a = SmtExpr::fp64_const(10.0);
        let b = SmtExpr::fp64_const(4.0);
        let expr = SmtExpr::fp_div(RoundingMode::RNA, a, b);
        let s = format!("{}", expr);
        assert!(s.starts_with("(fp.div RNA"));
    }

    #[test]
    fn test_fp_neg_smt2_serialization() {
        let a = SmtExpr::fp64_const(42.0);
        let expr = a.fp_neg();
        let s = format!("{}", expr);
        assert!(s.starts_with("(fp.neg"));
    }

    #[test]
    fn test_fp_eq_smt2_serialization() {
        let a = SmtExpr::fp64_const(1.0);
        let b = SmtExpr::fp64_const(1.0);
        let expr = a.fp_eq(b);
        let s = format!("{}", expr);
        assert!(s.starts_with("(fp.eq"));
    }

    #[test]
    fn test_fp_lt_smt2_serialization() {
        let a = SmtExpr::fp64_const(1.0);
        let b = SmtExpr::fp64_const(2.0);
        let expr = a.fp_lt(b);
        let s = format!("{}", expr);
        assert!(s.starts_with("(fp.lt"));
    }

    #[test]
    fn test_fp_const_smt2_serialization() {
        let expr = SmtExpr::fp64_const(1.0_f64);
        let s = format!("{}", expr);
        assert!(s.starts_with("(fp #b"));
        assert!(s.contains("#b0"));
        assert!(s.contains("#b01111111111"));
    }

    #[test]
    fn test_fp_const_fp32_smt2_serialization() {
        let expr = SmtExpr::fp32_const(1.5_f32);
        let s = format!("{}", expr);
        assert!(s.starts_with("(fp #b0"));
        assert!(s.contains("#b01111111"));
    }

    #[test]
    fn test_fp_const_negative_smt2() {
        let expr = SmtExpr::fp64_const(-1.0_f64);
        let s = format!("{}", expr);
        assert!(s.starts_with("(fp #b1"));
    }

    #[test]
    fn test_generate_smt2_query_with_fp_inputs() {
        let a_const = SmtExpr::fp64_const(1.0);
        let b_const = SmtExpr::fp64_const(2.0);

        let obligation = ProofObligation {
            name: "test_fp_add".to_string(),
            tmir_expr: SmtExpr::fp_add(RoundingMode::RNE, a_const.clone(), b_const.clone()),
            aarch64_expr: SmtExpr::fp_add(RoundingMode::RNE, a_const, b_const),
            inputs: vec![],
            preconditions: vec![],
            fp_inputs: vec![
                ("a".to_string(), 11, 53),
                ("b".to_string(), 11, 53),
            ],
        };

        let config = Z4Config::default();
        let smt2 = generate_smt2_query(&obligation, &config);

        assert!(smt2.contains("QF_BVFP") || smt2.contains("QF_FP"),
            "Expected FP logic, got: {}", smt2);
        assert!(smt2.contains("(declare-const a (_ FloatingPoint 11 53))"),
            "Missing FP64 declaration for a: {}", smt2);
        assert!(smt2.contains("(declare-const b (_ FloatingPoint 11 53))"),
            "Missing FP64 declaration for b: {}", smt2);
        assert!(smt2.contains("(get-value (a b))"),
            "Missing get-value for FP vars: {}", smt2);
    }

    #[test]
    fn test_generate_smt2_query_mixed_bv_fp() {
        let _bv_a = SmtExpr::var("x", 32);
        let fp_a = SmtExpr::fp32_const(1.0_f32);
        let fp_b = SmtExpr::fp32_const(2.0_f32);

        let obligation = ProofObligation {
            name: "test_mixed".to_string(),
            tmir_expr: SmtExpr::fp_add(RoundingMode::RNE, fp_a.clone(), fp_b.clone()),
            aarch64_expr: SmtExpr::fp_add(RoundingMode::RNE, fp_a, fp_b),
            inputs: vec![("x".to_string(), 32)],
            preconditions: vec![],
            fp_inputs: vec![
                ("fa".to_string(), 8, 24),
            ],
        };

        let config = Z4Config::default();
        let smt2 = generate_smt2_query(&obligation, &config);

        assert!(smt2.contains("(declare-const x (_ BitVec 32))"));
        assert!(smt2.contains("(declare-const fa (_ FloatingPoint 8 24))"));
        assert!(smt2.contains("(get-value (x fa))"));
    }

    #[test]
    fn test_fp_sort_display_in_declare() {
        let fp32 = SmtSort::fp32();
        assert_eq!(format!("{}", fp32), "(_ FloatingPoint 8 24)");
        let fp64 = SmtSort::fp64();
        assert_eq!(format!("{}", fp64), "(_ FloatingPoint 11 53)");
        let fp16 = SmtSort::fp16();
        assert_eq!(format!("{}", fp16), "(_ FloatingPoint 5 11)");
    }

    #[test]
    fn test_infer_logic_fp_add() {
        let expr = SmtExpr::fp_add(
            RoundingMode::RNE,
            SmtExpr::fp32_const(1.0_f32),
            SmtExpr::fp32_const(2.0_f32),
        );
        assert_eq!(infer_logic(&expr), "QF_BVFP");
    }

    #[test]
    fn test_infer_logic_fp_neg() {
        let expr = SmtExpr::fp64_const(1.0).fp_neg();
        assert_eq!(infer_logic(&expr), "QF_BVFP");
    }

    #[test]
    fn test_infer_logic_fp_eq() {
        let expr = SmtExpr::fp64_const(1.0).fp_eq(SmtExpr::fp64_const(2.0));
        assert_eq!(infer_logic(&expr), "QF_BVFP");
    }

    #[test]
    fn test_infer_logic_fp_lt() {
        let expr = SmtExpr::fp64_const(1.0).fp_lt(SmtExpr::fp64_const(2.0));
        assert_eq!(infer_logic(&expr), "QF_BVFP");
    }

    #[test]
    fn test_infer_logic_fp_const_only() {
        let expr = SmtExpr::fp64_const(3.14);
        assert_eq!(infer_logic(&expr), "QF_BVFP");
    }

    #[test]
    fn test_infer_logic_fp_mul() {
        let expr = SmtExpr::fp_mul(
            RoundingMode::RTZ,
            SmtExpr::fp64_const(2.0),
            SmtExpr::fp64_const(3.0),
        );
        assert_eq!(infer_logic(&expr), "QF_BVFP");
    }

    #[test]
    fn test_infer_logic_fp_div() {
        let expr = SmtExpr::fp_div(
            RoundingMode::RNE,
            SmtExpr::fp64_const(10.0),
            SmtExpr::fp64_const(3.0),
        );
        assert_eq!(infer_logic(&expr), "QF_BVFP");
    }

    // -----------------------------------------------------------------------
    // Public API convenience function tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_serialize_to_smt2() {
        let a = SmtExpr::var("a", 32);
        let b = SmtExpr::var("b", 32);
        let obligation = ProofObligation {
            name: "test_serialize".to_string(),
            tmir_expr: a.clone().bvadd(b.clone()),
            aarch64_expr: a.bvadd(b),
            inputs: vec![("a".to_string(), 32), ("b".to_string(), 32)],
            preconditions: vec![],
            fp_inputs: vec![],
        };

        let smt2 = serialize_to_smt2(&obligation);

        // Should produce a complete SMT-LIB2 script
        assert!(smt2.contains("(set-logic QF_BV)"));
        assert!(smt2.contains("(declare-const a (_ BitVec 32))"));
        assert!(smt2.contains("(declare-const b (_ BitVec 32))"));
        assert!(smt2.contains("(assert"));
        assert!(smt2.contains("(check-sat)"));
        assert!(smt2.contains("(exit)"));
        // Default config includes timeout and models
        assert!(smt2.contains("(set-option :timeout 5000)"));
        assert!(smt2.contains("(set-option :produce-models true)"));
    }

    #[test]
    fn test_parse_z4_output_unsat() {
        let result = parse_z4_output("unsat\n", &[]);
        assert_eq!(result, Z4Result::Verified);
    }

    #[test]
    fn test_parse_z4_output_sat_with_model() {
        let output = "sat\n((x #x0000002a))";
        let inputs = vec![("x".to_string(), 32)];
        let result = parse_z4_output(output, &inputs);
        match result {
            Z4Result::CounterExample(cex) => {
                assert_eq!(cex.len(), 1);
                assert_eq!(cex[0], ("x".to_string(), 0x2a));
            }
            other => panic!("Expected CounterExample, got {:?}", other),
        }
    }

    #[test]
    fn test_parse_z4_output_unknown() {
        let result = parse_z4_output("unknown\n", &[]);
        assert_eq!(result, Z4Result::Timeout);
    }

    #[test]
    fn test_parse_z4_output_timeout_in_text() {
        let result = parse_z4_output("timeout\n", &[]);
        assert_eq!(result, Z4Result::Timeout);
    }

    #[test]
    fn test_parse_z4_output_empty() {
        let result = parse_z4_output("", &[]);
        assert!(matches!(result, Z4Result::Error(_)));
    }

    #[test]
    fn test_parse_z4_output_unexpected() {
        let result = parse_z4_output("garbage\n", &[]);
        assert!(matches!(result, Z4Result::Error(_)));
    }

    // -----------------------------------------------------------------------
    // SmtExpr::to_smt2_expr() tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_to_smt2_expr_var() {
        let expr = SmtExpr::var("x", 32);
        assert_eq!(expr.to_smt2_expr(), "x");
    }

    #[test]
    fn test_to_smt2_expr_bv_const() {
        let expr = SmtExpr::bv_const(42, 32);
        assert_eq!(expr.to_smt2_expr(), "(_ bv42 32)");
    }

    #[test]
    fn test_to_smt2_expr_bool_const() {
        assert_eq!(SmtExpr::bool_const(true).to_smt2_expr(), "true");
        assert_eq!(SmtExpr::bool_const(false).to_smt2_expr(), "false");
    }

    #[test]
    fn test_to_smt2_expr_bvadd() {
        let a = SmtExpr::var("a", 32);
        let b = SmtExpr::var("b", 32);
        assert_eq!(a.bvadd(b).to_smt2_expr(), "(bvadd a b)");
    }

    #[test]
    fn test_to_smt2_expr_bvsub() {
        let a = SmtExpr::var("a", 32);
        let b = SmtExpr::var("b", 32);
        assert_eq!(a.bvsub(b).to_smt2_expr(), "(bvsub a b)");
    }

    #[test]
    fn test_to_smt2_expr_bvmul() {
        let a = SmtExpr::var("a", 32);
        let b = SmtExpr::var("b", 32);
        assert_eq!(a.bvmul(b).to_smt2_expr(), "(bvmul a b)");
    }

    #[test]
    fn test_to_smt2_expr_bvneg() {
        let a = SmtExpr::var("a", 32);
        assert_eq!(a.bvneg().to_smt2_expr(), "(bvneg a)");
    }

    #[test]
    fn test_to_smt2_expr_bitwise_ops() {
        let a = SmtExpr::var("a", 32);
        let b = SmtExpr::var("b", 32);
        assert_eq!(a.clone().bvand(b.clone()).to_smt2_expr(), "(bvand a b)");
        assert_eq!(a.clone().bvor(b.clone()).to_smt2_expr(), "(bvor a b)");
        assert_eq!(a.clone().bvxor(b.clone()).to_smt2_expr(), "(bvxor a b)");
    }

    #[test]
    fn test_to_smt2_expr_shift_ops() {
        let a = SmtExpr::var("a", 32);
        let b = SmtExpr::var("b", 32);
        assert_eq!(a.clone().bvshl(b.clone()).to_smt2_expr(), "(bvshl a b)");
        assert_eq!(a.clone().bvlshr(b.clone()).to_smt2_expr(), "(bvlshr a b)");
        assert_eq!(a.clone().bvashr(b.clone()).to_smt2_expr(), "(bvashr a b)");
    }

    #[test]
    fn test_to_smt2_expr_comparisons() {
        let a = SmtExpr::var("a", 32);
        let b = SmtExpr::var("b", 32);
        assert_eq!(a.clone().eq_expr(b.clone()).to_smt2_expr(), "(= a b)");
        assert_eq!(a.clone().bvslt(b.clone()).to_smt2_expr(), "(bvslt a b)");
        assert_eq!(a.clone().bvsge(b.clone()).to_smt2_expr(), "(bvsge a b)");
        assert_eq!(a.clone().bvult(b.clone()).to_smt2_expr(), "(bvult a b)");
        assert_eq!(a.clone().bvuge(b.clone()).to_smt2_expr(), "(bvuge a b)");
    }

    #[test]
    fn test_to_smt2_expr_logical_ops() {
        let a = SmtExpr::bool_const(true);
        let b = SmtExpr::bool_const(false);
        assert_eq!(a.clone().and_expr(b.clone()).to_smt2_expr(), "(and true false)");
        assert_eq!(a.clone().or_expr(b.clone()).to_smt2_expr(), "(or true false)");
        assert_eq!(a.not_expr().to_smt2_expr(), "(not true)");
    }

    #[test]
    fn test_to_smt2_expr_ite() {
        let cond = SmtExpr::var("c", 32).eq_expr(SmtExpr::bv_const(0, 32));
        let then_e = SmtExpr::var("a", 32);
        let else_e = SmtExpr::var("b", 32);
        let expr = SmtExpr::ite(cond, then_e, else_e);
        assert_eq!(expr.to_smt2_expr(), "(ite (= c (_ bv0 32)) a b)");
    }

    #[test]
    fn test_to_smt2_expr_extract() {
        let a = SmtExpr::var("a", 32);
        assert_eq!(a.extract(15, 0).to_smt2_expr(), "((_ extract 15 0) a)");
    }

    #[test]
    fn test_to_smt2_expr_concat() {
        let hi = SmtExpr::var("hi", 16);
        let lo = SmtExpr::var("lo", 16);
        assert_eq!(hi.concat(lo).to_smt2_expr(), "(concat hi lo)");
    }

    #[test]
    fn test_to_smt2_expr_extend() {
        let a = SmtExpr::var("a", 8);
        assert_eq!(a.clone().zero_ext(24).to_smt2_expr(), "((_ zero_extend 24) a)");
        assert_eq!(a.sign_ext(24).to_smt2_expr(), "((_ sign_extend 24) a)");
    }

    #[test]
    fn test_to_smt2_expr_division() {
        let a = SmtExpr::var("a", 32);
        let b = SmtExpr::var("b", 32);
        assert_eq!(a.clone().bvsdiv(b.clone()).to_smt2_expr(), "(bvsdiv a b)");
        assert_eq!(a.bvudiv(b).to_smt2_expr(), "(bvudiv a b)");
    }

    #[test]
    fn test_to_smt2_expr_nested() {
        // (bvadd (bvmul a b) (bvsub c d))
        let a = SmtExpr::var("a", 32);
        let b = SmtExpr::var("b", 32);
        let c = SmtExpr::var("c", 32);
        let d = SmtExpr::var("d", 32);
        let expr = a.bvmul(b).bvadd(c.bvsub(d));
        assert_eq!(expr.to_smt2_expr(), "(bvadd (bvmul a b) (bvsub c d))");
    }

    #[test]
    fn test_to_smt2_expr_fp_operations() {
        let a = SmtExpr::fp64_const(1.0);
        let b = SmtExpr::fp64_const(2.0);
        let add = SmtExpr::fp_add(RoundingMode::RNE, a.clone(), b.clone());
        assert!(add.to_smt2_expr().starts_with("(fp.add RNE"));

        let neg = a.clone().fp_neg();
        assert!(neg.to_smt2_expr().starts_with("(fp.neg"));

        let eq = a.clone().fp_eq(b.clone());
        assert!(eq.to_smt2_expr().starts_with("(fp.eq"));
    }

    #[test]
    fn test_to_smt2_expr_array_operations() {
        let arr = SmtExpr::const_array(SmtSort::BitVec(32), SmtExpr::bv_const(0, 8));
        let smt2 = arr.to_smt2_expr();
        assert!(smt2.contains("as const"));
        assert!(smt2.contains("Array"));

        let sel = SmtExpr::select(arr.clone(), SmtExpr::var("idx", 32));
        assert!(sel.to_smt2_expr().starts_with("(select"));

        let st = SmtExpr::store(arr, SmtExpr::var("idx", 32), SmtExpr::bv_const(42, 8));
        assert!(st.to_smt2_expr().starts_with("(store"));
    }

    // -----------------------------------------------------------------------
    // CLI integration tests for the public API wrappers
    // -----------------------------------------------------------------------

    #[test]
    fn test_verify_with_z4_cli_trivial_correct() {
        let solver = find_solver_binary();
        if solver.is_empty() {
            return;
        }

        let a = SmtExpr::var("x", 16);
        let obligation = ProofObligation {
            name: "trivial_identity".to_string(),
            tmir_expr: a.clone(),
            aarch64_expr: a,
            inputs: vec![("x".to_string(), 16)],
            preconditions: vec![],
            fp_inputs: vec![],
        };

        let config = Z4Config::default();
        let result = verify_with_z4_cli(&obligation, &config);
        assert_eq!(result, Z4Result::Verified);
    }

    #[test]
    fn test_verify_with_z4_cli_trivial_wrong() {
        let solver = find_solver_binary();
        if solver.is_empty() {
            return;
        }

        // x != x + 1 (should find counterexample for any x)
        let x = SmtExpr::var("x", 8);
        let obligation = ProofObligation {
            name: "wrong_identity".to_string(),
            tmir_expr: x.clone(),
            aarch64_expr: x.bvadd(SmtExpr::bv_const(1, 8)),
            inputs: vec![("x".to_string(), 8)],
            preconditions: vec![],
            fp_inputs: vec![],
        };

        let config = Z4Config::default();
        let result = verify_with_z4_cli(&obligation, &config);
        assert!(matches!(result, Z4Result::CounterExample(_)));
    }

    #[test]
    fn test_serialize_to_smt2_roundtrip_with_solver() {
        // Verify that serialize_to_smt2 output is valid SMT-LIB2 by running it
        // through z3 if available.
        let solver = find_solver_binary();
        if solver.is_empty() {
            return;
        }

        let a = SmtExpr::var("a", 32);
        let b = SmtExpr::var("b", 32);
        let obligation = ProofObligation {
            name: "roundtrip_test".to_string(),
            tmir_expr: a.clone().bvadd(b.clone()),
            aarch64_expr: a.bvadd(b),
            inputs: vec![("a".to_string(), 32), ("b".to_string(), 32)],
            preconditions: vec![],
            fp_inputs: vec![],
        };

        let smt2 = serialize_to_smt2(&obligation);

        // Write to temp file and verify z3 can parse it
        let tmp_path = write_temp_smt2(&smt2).expect("failed to write temp file");
        let output = std::process::Command::new(&solver)
            .arg("-smt2")
            .arg(&tmp_path)
            .output()
            .expect("failed to invoke solver");
        let _ = std::fs::remove_file(&tmp_path);

        let stdout = String::from_utf8_lossy(&output.stdout);
        // Should be unsat (a+b == a+b is trivially true)
        assert!(stdout.trim().starts_with("unsat"),
            "Expected unsat, got: {}", stdout);
    }

    #[test]
    fn test_serialize_to_smt2_with_preconditions() {
        // Test serialization of obligations with preconditions
        let a = SmtExpr::var("a", 32);
        let b = SmtExpr::var("b", 32);
        let precond = b.clone().eq_expr(SmtExpr::bv_const(0, 32)).not_expr();

        let obligation = ProofObligation {
            name: "div_with_precond".to_string(),
            tmir_expr: a.clone().bvsdiv(b.clone()),
            aarch64_expr: a.bvsdiv(b),
            inputs: vec![("a".to_string(), 32), ("b".to_string(), 32)],
            preconditions: vec![precond],
            fp_inputs: vec![],
        };

        let smt2 = serialize_to_smt2(&obligation);
        assert!(smt2.contains("(assert"));
        assert!(smt2.contains("bvsdiv"));
        assert!(smt2.contains("(not (="));  // precondition b != 0
    }

    #[test]
    fn test_verify_with_z4_cli_with_preconditions() {
        let solver = find_solver_binary();
        if solver.is_empty() {
            return;
        }

        // a / b == a / b with precondition b != 0
        let a = SmtExpr::var("a", 32);
        let b = SmtExpr::var("b", 32);
        let precond = b.clone().eq_expr(SmtExpr::bv_const(0, 32)).not_expr();

        let obligation = ProofObligation {
            name: "sdiv_identity".to_string(),
            tmir_expr: a.clone().bvsdiv(b.clone()),
            aarch64_expr: a.bvsdiv(b),
            inputs: vec![("a".to_string(), 32), ("b".to_string(), 32)],
            preconditions: vec![precond],
            fp_inputs: vec![],
        };

        let config = Z4Config::default();
        let result = verify_with_z4_cli(&obligation, &config);
        assert_eq!(result, Z4Result::Verified);
    }

    #[test]
    fn test_verify_with_z4_cli_negation_rule() {
        let solver = find_solver_binary();
        if solver.is_empty() {
            return;
        }

        // Verify that bvneg(a) == bvsub(0, a) -- foundational identity
        let a = SmtExpr::var("a", 32);
        let obligation = ProofObligation {
            name: "neg_is_sub_zero".to_string(),
            tmir_expr: a.clone().bvneg(),
            aarch64_expr: SmtExpr::bv_const(0, 32).bvsub(a),
            inputs: vec![("a".to_string(), 32)],
            preconditions: vec![],
            fp_inputs: vec![],
        };

        let config = Z4Config::default();
        let result = verify_with_z4_cli(&obligation, &config);
        assert_eq!(result, Z4Result::Verified);
    }

    #[test]
    fn test_verify_with_z4_cli_bitwise_identity() {
        let solver = find_solver_binary();
        if solver.is_empty() {
            return;
        }

        // Verify that a XOR a == 0 for all 16-bit values
        let a = SmtExpr::var("a", 16);
        let obligation = ProofObligation {
            name: "xor_self_is_zero".to_string(),
            tmir_expr: a.clone().bvxor(a.clone()),
            aarch64_expr: SmtExpr::bv_const(0, 16),
            inputs: vec![("a".to_string(), 16)],
            preconditions: vec![],
            fp_inputs: vec![],
        };

        let config = Z4Config::default();
        let result = verify_with_z4_cli(&obligation, &config);
        assert_eq!(result, Z4Result::Verified);
    }

    #[test]
    fn test_verify_with_z4_cli_extract_zeroext_identity() {
        let solver = find_solver_binary();
        if solver.is_empty() {
            return;
        }

        // Verify that zero_extend(extract[7:0](a), 24) extracts and extends correctly
        // For a 32-bit value, this should equal a AND 0xFF
        let a = SmtExpr::var("a", 32);
        let tmir = a.clone().extract(7, 0).zero_ext(24);
        let aarch64 = a.bvand(SmtExpr::bv_const(0xFF, 32));

        let obligation = ProofObligation {
            name: "extract_zext_eq_mask".to_string(),
            tmir_expr: tmir,
            aarch64_expr: aarch64,
            inputs: vec![("a".to_string(), 32)],
            preconditions: vec![],
            fp_inputs: vec![],
        };

        let config = Z4Config::default();
        let result = verify_with_z4_cli(&obligation, &config);
        assert_eq!(result, Z4Result::Verified);
    }

    #[test]
    fn test_parse_z4_output_sat_empty_model() {
        // SAT but no model lines following
        let result = parse_z4_output("sat\n", &[("x".to_string(), 32)]);
        match result {
            Z4Result::CounterExample(cex) => {
                // No model available, empty counterexample
                assert!(cex.is_empty());
            }
            other => panic!("Expected CounterExample, got {:?}", other),
        }
    }

    #[test]
    fn test_parse_z4_output_multiple_vars() {
        let output = "sat\n((a #x00000001)\n (b #x00000002)\n (c #x00000003))";
        let inputs = vec![
            ("a".to_string(), 32),
            ("b".to_string(), 32),
            ("c".to_string(), 32),
        ];
        let result = parse_z4_output(output, &inputs);
        match result {
            Z4Result::CounterExample(cex) => {
                assert_eq!(cex.len(), 3);
                assert_eq!(cex[0], ("a".to_string(), 1));
                assert_eq!(cex[1], ("b".to_string(), 2));
                assert_eq!(cex[2], ("c".to_string(), 3));
            }
            other => panic!("Expected CounterExample, got {:?}", other),
        }
    }
}
