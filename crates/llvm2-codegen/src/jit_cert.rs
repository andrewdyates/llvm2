// llvm2-codegen/jit_cert.rs - Proof certificates attached to JIT buffers
//
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Copyright 2026 Andrew Yates | License: Apache-2.0

//! Proof certificates for JIT-compiled functions (issue #348).
//!
//! Attaches a [`JitCertificate`] to each function compiled by
//! [`crate::jit::JitCompiler`] when [`crate::jit::JitConfig::verify`] is
//! true. The certificate bundles:
//!
//! - the function name and its byte range within the executable buffer,
//! - a coarse (tMIR op, MachInst index range) provenance map,
//! - a full [`llvm2_verify::proof_certificate::CertificateChain`] summarizing
//!   what was verified by [`llvm2_verify::verify_function`], and
//! - a cheap `replay_check()` that re-hashes the chain for tamper detection.
//!
//! This is the Phase-1 cut of the plan in
//! `reports/2026-04-18-jit-proof-certs-plan.md`. Real z4 SMT replay is
//! deferred to a future `proof-certs-full` feature.
//!
//! Callers (e.g. tla2) use the certificate to assert "the JIT-compiled
//! machine code for `fn add` has been formally checked against the tMIR
//! that produced it", without needing to re-run the full verification
//! pipeline themselves.
//!
//! # Example
//!
//! ```no_run
//! # use llvm2_codegen::jit::{JitCompiler, JitConfig};
//! # use std::collections::HashMap;
//! let jit = JitCompiler::new(JitConfig { verify: true, ..Default::default() });
//! # let functions = vec![];
//! let buf = jit.compile_raw(&functions, &HashMap::new()).unwrap();
//! if let Some(cert) = buf.certificate("add") {
//!     assert!(cert.is_verified());
//!     assert!(cert.replay_check());
//! }
//! ```

use std::ops::Range;

#[cfg(feature = "verify")]
use std::collections::hash_map::DefaultHasher;
#[cfg(feature = "verify")]
use std::hash::Hasher;
#[cfg(feature = "verify")]
use llvm2_ir::{AArch64Opcode, MachFunction};
#[cfg(feature = "verify")]
use llvm2_verify::function_verifier::{FunctionVerificationReport, InstructionVerificationResult};
#[cfg(feature = "verify")]
use llvm2_verify::lowering_proof::TransvalCheckKind;
#[cfg(feature = "verify")]
use llvm2_verify::proof_certificate::{
    CertificateChain, CertificateResult, ProofCertificate, SolverUsed,
};
#[cfg(feature = "verify")]
use llvm2_verify::verify::VerificationStrength;

// ---------------------------------------------------------------------------
// TmirPair
// ---------------------------------------------------------------------------

/// A coarse provenance entry: one tMIR operation and the contiguous range of
/// MachInst indices that implements it after lowering.
///
/// Phase 1 populates this via a small opcode → tMIR-name lookup table. Later
/// phases will replace the table with the real tMIR-to-MachInst provenance
/// map from `llvm2_ir::provenance`.
#[derive(Debug, Clone)]
pub struct TmirPair {
    /// Name of the tMIR operation (e.g. "Iadd_I32"). `"<opaque>"` when the
    /// opcode is not covered by the Phase-1 lookup table.
    pub tmir_op: String,
    /// Contiguous half-open range of indices into `MachFunction::insts` that
    /// implements this tMIR op.
    pub mach_insts: Range<u32>,
}

// ---------------------------------------------------------------------------
// JitCertificate
// ---------------------------------------------------------------------------

/// A proof certificate for a single JIT-compiled function.
///
/// Constructed from a [`FunctionVerificationReport`] at JIT compile time
/// when verification is enabled. The certificate is immutable after
/// construction and can be queried, displayed, exported as JSON, or
/// replay-checked.
///
/// When the `verify` feature is disabled at build time, this type still
/// exists but its constructor is unreachable — `ExecutableBuffer` will
/// return `None` for every `certificate(name)` call.
#[derive(Debug, Clone)]
pub struct JitCertificate {
    /// Canonical function name (matches `MachFunction::name`).
    function: String,
    /// Byte range `[start, end)` of this function's machine code within the
    /// owning `ExecutableBuffer`.
    code_range: Range<u64>,
    /// tMIR → MachInst provenance. One entry per encoded MachInst in
    /// insertion order. Empty in the no-verify build.
    tmir_pairs: Vec<TmirPair>,
    /// Certificate chain produced by `llvm2-verify`. Empty when no
    /// instruction in the function matched a proof obligation.
    #[cfg(feature = "verify")]
    chain: CertificateChain,
    /// Verified flag: true iff the verification report reports no failed or
    /// unverified instructions (skipped pseudo-ops do not count).
    verified: bool,
    /// Coverage percentage from the verification report
    /// (verified / (total - skipped) * 100).
    coverage_pct: f64,
}

impl JitCertificate {
    /// Build a certificate from a verification report, function, and the
    /// byte range the encoded function occupies in the executable buffer.
    ///
    /// This is the primary constructor used by `JitCompiler::compile_raw`.
    /// The returned certificate is self-contained and does not borrow
    /// from `func`.
    #[cfg(feature = "verify")]
    pub(crate) fn from_report(
        func: &MachFunction,
        report: &FunctionVerificationReport,
        code_range: Range<u64>,
    ) -> Self {
        let function = func.name.clone();
        let tmir_pairs = build_tmir_pairs(func);

        let mut chain = CertificateChain::new(function.clone());
        for ir in &report.instructions {
            if let Some(cert) = instruction_report_to_cert(ir) {
                chain.add(cert);
            }
        }

        let verified = report.all_verified();
        let coverage_pct = report.coverage_percent();

        Self {
            function,
            code_range,
            tmir_pairs,
            chain,
            verified,
            coverage_pct,
        }
    }

    /// Stub constructor for `verify`-disabled builds. Never populated in
    /// practice because the no-verify `ExecutableBuffer` path skips
    /// certificate construction entirely; exposed so the struct compiles.
    #[cfg(not(feature = "verify"))]
    #[allow(dead_code)]
    pub(crate) fn empty(function: String, code_range: Range<u64>) -> Self {
        Self {
            function,
            code_range,
            tmir_pairs: Vec::new(),
            verified: false,
            coverage_pct: 0.0,
        }
    }

    /// Function name this certificate covers.
    pub fn function(&self) -> &str {
        &self.function
    }

    /// Half-open byte range `[start, end)` of this function's code within
    /// the owning `ExecutableBuffer`.
    pub fn code_range(&self) -> Range<u64> {
        self.code_range.clone()
    }

    /// tMIR → MachInst provenance entries.
    pub fn tmir_pairs(&self) -> &[TmirPair] {
        &self.tmir_pairs
    }

    /// Returns true iff every non-pseudo machine instruction in this
    /// function was matched against a proof obligation and verified.
    pub fn is_verified(&self) -> bool {
        self.verified
    }

    /// Coverage percentage: verified / (total - skipped) * 100.
    pub fn coverage_percent(&self) -> f64 {
        self.coverage_pct
    }

    /// Underlying verification chain. Available only with the `verify`
    /// feature enabled.
    #[cfg(feature = "verify")]
    pub fn chain(&self) -> &CertificateChain {
        &self.chain
    }

    /// Cheap replay check.
    ///
    /// Re-derives a stable hash from each certificate's
    /// `(obligation_name, solver, strength, result)` tuple and confirms it
    /// matches the stored `formula_hash`. This catches in-memory tampering
    /// with the certificate without re-running the SMT solver.
    ///
    /// A full solver-backed replay is gated behind the future
    /// `proof-certs-full` feature.
    #[cfg(feature = "verify")]
    pub fn replay_check(&self) -> bool {
        for cert in &self.chain.certificates {
            if cert.formula_hash != expected_formula_hash(cert) {
                return false;
            }
        }
        // A certificate with an empty chain is considered consistent —
        // `replay_check` only fails when at least one stored entry has
        // been tampered with. `is_verified` captures the "had any proof"
        // question for callers that care.
        true
    }

    /// Stub replay check for no-verify builds; always returns `true`.
    #[cfg(not(feature = "verify"))]
    pub fn replay_check(&self) -> bool {
        true
    }

    /// Serialize this certificate to a compact JSON object. When the
    /// `verify` feature is disabled the `chain` field is omitted.
    pub fn to_json(&self) -> String {
        let mut out = String::from("{\n");
        out.push_str(&format!(
            "  \"function\": \"{}\",\n",
            escape_json(&self.function)
        ));
        out.push_str(&format!(
            "  \"code_range\": [{}, {}],\n",
            self.code_range.start, self.code_range.end
        ));
        out.push_str(&format!("  \"verified\": {},\n", self.verified));
        out.push_str(&format!(
            "  \"coverage_percent\": {:.4},\n",
            self.coverage_pct
        ));
        out.push_str("  \"tmir_pairs\": [");
        for (i, p) in self.tmir_pairs.iter().enumerate() {
            if i > 0 {
                out.push_str(", ");
            }
            out.push_str(&format!(
                "{{\"tmir_op\": \"{}\", \"mach_insts\": [{}, {}]}}",
                escape_json(&p.tmir_op),
                p.mach_insts.start,
                p.mach_insts.end
            ));
        }
        out.push_str("]");

        #[cfg(feature = "verify")]
        {
            out.push_str(",\n  \"chain\": ");
            out.push_str(&self.chain.to_json());
        }

        out.push_str("\n}");
        out
    }
}

// ---------------------------------------------------------------------------
// Internal helpers (feature-gated — all live inside the verify world)
// ---------------------------------------------------------------------------

/// Map an AArch64 opcode to a tMIR operation name for the Phase-1 lookup
/// table. Returns "<opaque>" for any opcode not in the lookup.
#[cfg(feature = "verify")]
fn opcode_to_tmir_op(opcode: AArch64Opcode) -> &'static str {
    use AArch64Opcode::*;
    match opcode {
        AddRR | AddRI | AddRIShift12 => "Iadd_I32",
        SubRR | SubRI => "Isub_I32",
        MulRR => "Imul_I32",
        Neg => "Ineg_I32",
        SDiv => "Isdiv_I32",
        UDiv => "Iudiv_I32",
        CmpRR | CmpRI => "Icmp",
        Tst => "Itst",
        BCond | Bcc => "Ibrcond",
        AndRR | AndRI => "Iand",
        OrrRR | OrrRI => "Ior",
        EorRR | EorRI => "Ixor",
        LslRR | LslRI => "Ishl",
        LsrRR | LsrRI => "Ilshr",
        AsrRR | AsrRI => "Iashr",
        CSet => "Icset",
        FaddRR => "Fadd",
        FsubRR => "Fsub",
        FmulRR => "Fmul",
        FnegRR => "Fneg",
        _ => "<opaque>",
    }
}

/// Build a one-to-one TmirPair list: each MachInst in `func.insts`
/// produces a single pair with a unit range `[i, i+1)`. Coalescing runs
/// of the same tMIR op is intentionally left for Phase 2 when real
/// provenance is available.
#[cfg(feature = "verify")]
fn build_tmir_pairs(func: &MachFunction) -> Vec<TmirPair> {
    func.insts
        .iter()
        .enumerate()
        .map(|(i, inst)| TmirPair {
            tmir_op: opcode_to_tmir_op(inst.opcode).to_string(),
            mach_insts: (i as u32)..((i as u32) + 1),
        })
        .collect()
}

/// Convert a single `InstructionVerificationResult` into a synthesized
/// [`ProofCertificate`]. Returns `None` for `Skipped` results (pseudo-ops
/// have no proof to record) and `None` for `Unverified` results (no proof
/// obligation matched — not a failure, just absence). `Failed` and
/// `Verified` both yield certificates.
#[cfg(feature = "verify")]
fn instruction_report_to_cert(
    ir: &llvm2_verify::function_verifier::InstructionReport,
) -> Option<ProofCertificate> {
    let (obligation_name, result, strength, check_kind) = match &ir.result {
        InstructionVerificationResult::Verified {
            proof_name,
            category: _,
            strength,
        } => (
            proof_name.clone(),
            CertificateResult::Verified,
            strength.clone(),
            Some(TransvalCheckKind::InstructionLowering),
        ),
        InstructionVerificationResult::Failed { proof_name, detail } => (
            proof_name.clone(),
            CertificateResult::Failed {
                counterexample: detail.clone(),
            },
            VerificationStrength::Exhaustive,
            Some(TransvalCheckKind::InstructionLowering),
        ),
        InstructionVerificationResult::Unverified { .. }
        | InstructionVerificationResult::Skipped { .. } => return None,
    };

    let solver = strength_to_solver(&strength);
    let mut cert = ProofCertificate {
        obligation_name,
        result,
        solver,
        strength,
        check_kind,
        formula_hash: 0,
        timestamp_epoch_secs: now_epoch_secs(),
        duration_ms: 0,
    };
    cert.formula_hash = expected_formula_hash(&cert);
    Some(cert)
}

/// Map a verification strength back to the solver tag the verifier would
/// have used. Mirrors `proof_certificate::strength_to_solver` without
/// duplicating the helper.
#[cfg(feature = "verify")]
fn strength_to_solver(strength: &VerificationStrength) -> SolverUsed {
    match strength {
        VerificationStrength::Exhaustive => SolverUsed::MockExhaustive,
        VerificationStrength::Statistical { sample_count } => SolverUsed::MockStatistical {
            samples: *sample_count,
        },
        VerificationStrength::Formal => SolverUsed::Z4Native,
    }
}

/// Derive a stable hash for a certificate from its
/// (obligation_name, solver_tag, strength, result) tuple. Used both at
/// construction time and by `replay_check` to detect tampering. The hash
/// is deliberately name-based rather than formula-based: the full SMT
/// formula is not serialized into the certificate in Phase 1.
#[cfg(feature = "verify")]
fn expected_formula_hash(cert: &ProofCertificate) -> u64 {
    let mut h = DefaultHasher::new();
    h.write(cert.obligation_name.as_bytes());
    h.write(format!("{:?}", cert.solver).as_bytes());
    h.write(format!("{:?}", cert.strength).as_bytes());
    // Only the tag of the result, not the counterexample/reason — we
    // want the hash stable across identical Verified entries.
    h.write(result_tag(&cert.result).as_bytes());
    h.finish()
}

#[cfg(feature = "verify")]
fn result_tag(r: &CertificateResult) -> &'static str {
    match r {
        CertificateResult::Verified => "verified",
        CertificateResult::Failed { .. } => "failed",
        CertificateResult::Timeout { .. } => "timeout",
        CertificateResult::Skipped { .. } => "skipped",
    }
}

#[cfg(feature = "verify")]
fn now_epoch_secs() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

/// Minimal JSON escape usable from the jit module's `export_proofs`.
pub(crate) fn escape_for_export(s: &str) -> String {
    escape_json(s)
}

/// Minimal JSON escape for function names and tMIR op strings.
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

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[cfg(all(test, feature = "verify"))]
mod tests {
    use super::*;
    use llvm2_ir::types::InstId;
    use llvm2_ir::{MachInst, Signature};

    fn make_add_i32_func() -> MachFunction {
        let mut func = MachFunction::new("add".to_string(), Signature::new(vec![], vec![]));
        func.insts.push(MachInst::new(AArch64Opcode::AddRR, vec![]));
        func.blocks[0].insts.push(InstId(0));
        func
    }

    #[test]
    fn opcode_to_tmir_op_covers_basic_arith() {
        assert_eq!(opcode_to_tmir_op(AArch64Opcode::AddRR), "Iadd_I32");
        assert_eq!(opcode_to_tmir_op(AArch64Opcode::SubRR), "Isub_I32");
        assert_eq!(opcode_to_tmir_op(AArch64Opcode::MulRR), "Imul_I32");
        assert_eq!(opcode_to_tmir_op(AArch64Opcode::Neg), "Ineg_I32");
    }

    #[test]
    fn build_tmir_pairs_single_add() {
        let func = make_add_i32_func();
        let pairs = build_tmir_pairs(&func);
        assert_eq!(pairs.len(), 1);
        assert_eq!(pairs[0].tmir_op, "Iadd_I32");
        assert_eq!(pairs[0].mach_insts, 0u32..1);
    }

    #[test]
    fn certificate_roundtrip_from_report() {
        let func = make_add_i32_func();
        let report = llvm2_verify::verify_function(&func);
        let cert = JitCertificate::from_report(&func, &report, 0u64..4u64);

        assert_eq!(cert.function(), "add");
        assert_eq!(cert.code_range(), 0u64..4u64);
        assert!(cert.is_verified(), "AddRR must verify");
        assert!(
            cert.coverage_percent() >= 99.9,
            "coverage = {}",
            cert.coverage_percent()
        );
        let pairs = cert.tmir_pairs();
        assert_eq!(pairs.len(), 1);
        assert_eq!(pairs[0].tmir_op, "Iadd_I32");

        let chain = cert.chain();
        assert!(!chain.certificates.is_empty());
        assert!(chain.all_verified());
    }

    #[test]
    fn replay_check_passes_on_untampered_certificate() {
        let func = make_add_i32_func();
        let report = llvm2_verify::verify_function(&func);
        let cert = JitCertificate::from_report(&func, &report, 0u64..4u64);
        assert!(cert.replay_check());
    }

    #[test]
    fn replay_check_detects_hash_tampering() {
        let func = make_add_i32_func();
        let report = llvm2_verify::verify_function(&func);
        let mut cert = JitCertificate::from_report(&func, &report, 0u64..4u64);
        // Tamper with the first certificate's stored hash.
        assert!(!cert.chain.certificates.is_empty());
        cert.chain.certificates[0].formula_hash =
            cert.chain.certificates[0].formula_hash.wrapping_add(1);
        assert!(
            !cert.replay_check(),
            "replay_check must fail after hash tampering"
        );
    }

    #[test]
    fn to_json_emits_core_fields() {
        let func = make_add_i32_func();
        let report = llvm2_verify::verify_function(&func);
        let cert = JitCertificate::from_report(&func, &report, 0u64..4u64);
        let json = cert.to_json();
        assert!(json.contains("\"function\": \"add\""), "json: {json}");
        assert!(json.contains("\"verified\": true"), "json: {json}");
        assert!(json.contains("\"tmir_pairs\""), "json: {json}");
        assert!(json.contains("Iadd_I32"), "json: {json}");
        assert!(json.contains("\"chain\""), "json: {json}");
    }
}
