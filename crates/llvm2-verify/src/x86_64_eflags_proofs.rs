// llvm2-verify/x86_64_eflags_proofs.rs - x86-64 EFLAGS direct proof obligations
//
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Copyright 2026 Andrew Yates | License: Apache-2.0
//
// Direct z4 proof obligations for x86-64 CMP/SUB flag writes (SF/ZF/CF/OF).
//
// These proofs pin the flag-layer semantics independent of the SETcc chain,
// so a regression in CF/OF computation surfaces here rather than being
// misattributed to the 10-obligation icmp lowering suite.
//
// Pattern mirrors `lowering_proof::proof_nzcv_*_flag_i32` (AArch64 NZCV flag
// correctness lemmas): each proof compares the EFLAGS expression produced by
// `x86_64_eflags::encode_cmp_eflags` (converted to BV1 via ITE) against a
// direct SMT encoding of the reference flag semantics.
//
// Reference: Intel 64 and IA-32 Architectures Software Developer's Manual,
//            Vol. 1 §3.4.3 "EFLAGS Register".
// Reference: #458 (filing issue), #407 (epic), #434 (all_x86_64_proofs registry).

//! Direct proof obligations for x86-64 CMP/SUB flag writes.
//!
//! The existing `x86_64_lowering_proofs::proof_x86_icmp_*` obligations exercise
//! EFLAGS indirectly through the SETcc chain. This module pins each flag
//! directly: SF, ZF, CF, OF for both CMP (alias of SUB-discard-result) and
//! SUB. Both operations share the same flag-write semantics, so regressions
//! in one code path are caught here at the semantic layer rather than
//! bubbling up through an icmp lowering proof failure.
//!
//! Scope: i8 (the z4 smoke lane's strict-verification width).

use crate::lowering_proof::ProofObligation;
use crate::smt::SmtExpr;
use crate::x86_64_eflags::encode_cmp_eflags;

/// Shared helper: convert a Bool-sorted SMT expression to a 1-bit bitvector.
///
/// `true -> bv1(1)`, `false -> bv1(0)`. Matches the B1 convention used by
/// `proof_nzcv_*_flag_i32`.
fn bool_to_bv1(b: SmtExpr) -> SmtExpr {
    SmtExpr::ite(b, SmtExpr::bv_const(1, 1), SmtExpr::bv_const(0, 1))
}

// ===========================================================================
// CMP flag proofs (i8)
// ===========================================================================
//
// CMP src1, src2 is semantically SUB with the result discarded. The flag
// writes are identical to SUB's. These four proofs pin the CMP path so a
// regression in the CMP encoding surfaces without touching SETcc.

/// Proof: `CMP SF(a, b) == (a - b)[7]` at i8.
///
/// SF is defined as the sign bit (MSB) of the subtraction result.
pub fn proof_cmp_sf_i8() -> ProofObligation {
    let a = SmtExpr::var("a", 8);
    let b = SmtExpr::var("b", 8);

    let flags = encode_cmp_eflags(a.clone(), b.clone(), 8);

    // Reference: MSB of (a - b).
    let diff = a.clone().bvsub(b.clone());
    let msb_bool = diff.extract(7, 7).eq_expr(SmtExpr::bv_const(1, 1));

    let sf_bv = bool_to_bv1(flags.sf);
    let ref_bv = bool_to_bv1(msb_bool);

    ProofObligation {
        name: "x86_64: CMP_SF_flag_I8".to_string(),
        tmir_expr: ref_bv,
        aarch64_expr: sf_bv,
        inputs: vec![("a".to_string(), 8), ("b".to_string(), 8)],
        preconditions: vec![],
        fp_inputs: vec![],
        category: None,
    }
}

/// Proof: `CMP ZF(a, b) == ((a - b) == 0)` at i8.
pub fn proof_cmp_zf_i8() -> ProofObligation {
    let a = SmtExpr::var("a", 8);
    let b = SmtExpr::var("b", 8);

    let flags = encode_cmp_eflags(a.clone(), b.clone(), 8);

    let diff = a.clone().bvsub(b.clone());
    let is_zero = diff.eq_expr(SmtExpr::bv_const(0, 8));

    let zf_bv = bool_to_bv1(flags.zf);
    let ref_bv = bool_to_bv1(is_zero);

    ProofObligation {
        name: "x86_64: CMP_ZF_flag_I8".to_string(),
        tmir_expr: ref_bv,
        aarch64_expr: zf_bv,
        inputs: vec![("a".to_string(), 8), ("b".to_string(), 8)],
        preconditions: vec![],
        fp_inputs: vec![],
        category: None,
    }
}

/// Proof: `CMP CF(a, b) == (a <_u b)` at i8.
///
/// Intel CF semantics for SUB: set when an unsigned borrow occurred. This is
/// the inverse of AArch64 NZCV C. The reference spec is `bvult(a, b)`.
pub fn proof_cmp_cf_i8() -> ProofObligation {
    let a = SmtExpr::var("a", 8);
    let b = SmtExpr::var("b", 8);

    let flags = encode_cmp_eflags(a.clone(), b.clone(), 8);

    let ult = a.clone().bvult(b.clone());

    let cf_bv = bool_to_bv1(flags.cf);
    let ref_bv = bool_to_bv1(ult);

    ProofObligation {
        name: "x86_64: CMP_CF_flag_I8".to_string(),
        tmir_expr: ref_bv,
        aarch64_expr: cf_bv,
        inputs: vec![("a".to_string(), 8), ("b".to_string(), 8)],
        preconditions: vec![],
        fp_inputs: vec![],
        category: None,
    }
}

/// Proof: `CMP OF(a, b) == signed_overflow(a - b)` at i8.
///
/// OF is the signed-subtraction overflow predicate:
///   `(sign(a) != sign(b)) AND (sign(a) != sign(a - b))`.
pub fn proof_cmp_of_i8() -> ProofObligation {
    let a = SmtExpr::var("a", 8);
    let b = SmtExpr::var("b", 8);

    let flags = encode_cmp_eflags(a.clone(), b.clone(), 8);

    let diff = a.clone().bvsub(b.clone());
    let a_msb = a.extract(7, 7);
    let b_msb = b.extract(7, 7);
    let r_msb = diff.extract(7, 7);
    let overflow = a_msb.clone().eq_expr(b_msb).not_expr()
        .and_expr(a_msb.eq_expr(r_msb).not_expr());

    let of_bv = bool_to_bv1(flags.of_);
    let ref_bv = bool_to_bv1(overflow);

    ProofObligation {
        name: "x86_64: CMP_OF_flag_I8".to_string(),
        tmir_expr: ref_bv,
        aarch64_expr: of_bv,
        inputs: vec![("a".to_string(), 8), ("b".to_string(), 8)],
        preconditions: vec![],
        fp_inputs: vec![],
        category: None,
    }
}

// ===========================================================================
// SUB flag proofs (i8)
// ===========================================================================
//
// SUB writes the same four flags as CMP. These proofs are structurally
// identical obligations but named against SUB so a regression in the
// "SUB keeps result, writes flags" path (for setcc combining, peephole
// fusion, etc.) surfaces separately from the CMP path.
//
// The tMIR side would be `Isub(a, b)` for the numeric result; here we pin
// the flag writes, which share the same reference spec as CMP.

/// Proof: `SUB SF(a, b) == (a - b)[7]` at i8.
pub fn proof_sub_sf_i8() -> ProofObligation {
    let a = SmtExpr::var("a", 8);
    let b = SmtExpr::var("b", 8);

    let flags = encode_cmp_eflags(a.clone(), b.clone(), 8);

    let diff = a.clone().bvsub(b.clone());
    let msb_bool = diff.extract(7, 7).eq_expr(SmtExpr::bv_const(1, 1));

    let sf_bv = bool_to_bv1(flags.sf);
    let ref_bv = bool_to_bv1(msb_bool);

    ProofObligation {
        name: "x86_64: SUB_SF_flag_I8".to_string(),
        tmir_expr: ref_bv,
        aarch64_expr: sf_bv,
        inputs: vec![("a".to_string(), 8), ("b".to_string(), 8)],
        preconditions: vec![],
        fp_inputs: vec![],
        category: None,
    }
}

/// Proof: `SUB ZF(a, b) == ((a - b) == 0)` at i8.
pub fn proof_sub_zf_i8() -> ProofObligation {
    let a = SmtExpr::var("a", 8);
    let b = SmtExpr::var("b", 8);

    let flags = encode_cmp_eflags(a.clone(), b.clone(), 8);

    let diff = a.clone().bvsub(b.clone());
    let is_zero = diff.eq_expr(SmtExpr::bv_const(0, 8));

    let zf_bv = bool_to_bv1(flags.zf);
    let ref_bv = bool_to_bv1(is_zero);

    ProofObligation {
        name: "x86_64: SUB_ZF_flag_I8".to_string(),
        tmir_expr: ref_bv,
        aarch64_expr: zf_bv,
        inputs: vec![("a".to_string(), 8), ("b".to_string(), 8)],
        preconditions: vec![],
        fp_inputs: vec![],
        category: None,
    }
}

/// Proof: `SUB CF(a, b) == (a <_u b)` at i8.
pub fn proof_sub_cf_i8() -> ProofObligation {
    let a = SmtExpr::var("a", 8);
    let b = SmtExpr::var("b", 8);

    let flags = encode_cmp_eflags(a.clone(), b.clone(), 8);

    let ult = a.clone().bvult(b.clone());

    let cf_bv = bool_to_bv1(flags.cf);
    let ref_bv = bool_to_bv1(ult);

    ProofObligation {
        name: "x86_64: SUB_CF_flag_I8".to_string(),
        tmir_expr: ref_bv,
        aarch64_expr: cf_bv,
        inputs: vec![("a".to_string(), 8), ("b".to_string(), 8)],
        preconditions: vec![],
        fp_inputs: vec![],
        category: None,
    }
}

/// Proof: `SUB OF(a, b) == signed_overflow(a - b)` at i8.
pub fn proof_sub_of_i8() -> ProofObligation {
    let a = SmtExpr::var("a", 8);
    let b = SmtExpr::var("b", 8);

    let flags = encode_cmp_eflags(a.clone(), b.clone(), 8);

    let diff = a.clone().bvsub(b.clone());
    let a_msb = a.extract(7, 7);
    let b_msb = b.extract(7, 7);
    let r_msb = diff.extract(7, 7);
    let overflow = a_msb.clone().eq_expr(b_msb).not_expr()
        .and_expr(a_msb.eq_expr(r_msb).not_expr());

    let of_bv = bool_to_bv1(flags.of_);
    let ref_bv = bool_to_bv1(overflow);

    ProofObligation {
        name: "x86_64: SUB_OF_flag_I8".to_string(),
        tmir_expr: ref_bv,
        aarch64_expr: of_bv,
        inputs: vec![("a".to_string(), 8), ("b".to_string(), 8)],
        preconditions: vec![],
        fp_inputs: vec![],
        category: None,
    }
}

// ===========================================================================
// CMP writes i32 — direct flag-correctness proofs (issue #458)
// ===========================================================================

/// Proof: `CMP ZF(a, b) == ((a - b) == 0)` at i32.
pub fn proof_x86_cmp_writes_zf_i32() -> ProofObligation {
    let a = SmtExpr::var("a", 32);
    let b = SmtExpr::var("b", 32);

    let flags = encode_cmp_eflags(a.clone(), b.clone(), 32);

    let diff = a.clone().bvsub(b.clone());
    let is_zero = diff.eq_expr(SmtExpr::bv_const(0, 32));

    let zf_bv = bool_to_bv1(flags.zf);
    let ref_bv = bool_to_bv1(is_zero);

    ProofObligation {
        name: "x86_64: CMP_ZF_writes_I32".to_string(),
        tmir_expr: ref_bv,
        aarch64_expr: zf_bv,
        inputs: vec![("a".to_string(), 32), ("b".to_string(), 32)],
        preconditions: vec![],
        fp_inputs: vec![],
        category: None,
    }
}

/// Proof: `CMP SF(a, b) == (a - b)[31]` at i32.
pub fn proof_x86_cmp_writes_sf_i32() -> ProofObligation {
    let a = SmtExpr::var("a", 32);
    let b = SmtExpr::var("b", 32);

    let flags = encode_cmp_eflags(a.clone(), b.clone(), 32);

    let diff = a.clone().bvsub(b.clone());
    let msb_bool = diff.extract(31, 31).eq_expr(SmtExpr::bv_const(1, 1));

    let sf_bv = bool_to_bv1(flags.sf);
    let ref_bv = bool_to_bv1(msb_bool);

    ProofObligation {
        name: "x86_64: CMP_SF_writes_I32".to_string(),
        tmir_expr: ref_bv,
        aarch64_expr: sf_bv,
        inputs: vec![("a".to_string(), 32), ("b".to_string(), 32)],
        preconditions: vec![],
        fp_inputs: vec![],
        category: None,
    }
}

/// Proof: `CMP CF(a, b) == (a <_u b)` at i32.
pub fn proof_x86_cmp_writes_cf_i32() -> ProofObligation {
    let a = SmtExpr::var("a", 32);
    let b = SmtExpr::var("b", 32);

    let flags = encode_cmp_eflags(a.clone(), b.clone(), 32);

    let ult = a.clone().bvult(b.clone());

    let cf_bv = bool_to_bv1(flags.cf);
    let ref_bv = bool_to_bv1(ult);

    ProofObligation {
        name: "x86_64: CMP_CF_writes_I32".to_string(),
        tmir_expr: ref_bv,
        aarch64_expr: cf_bv,
        inputs: vec![("a".to_string(), 32), ("b".to_string(), 32)],
        preconditions: vec![],
        fp_inputs: vec![],
        category: None,
    }
}

/// Proof: `CMP OF(a, b) == signed_overflow(a - b)` at i32.
pub fn proof_x86_cmp_writes_of_i32() -> ProofObligation {
    let a = SmtExpr::var("a", 32);
    let b = SmtExpr::var("b", 32);

    let flags = encode_cmp_eflags(a.clone(), b.clone(), 32);

    let diff = a.clone().bvsub(b.clone());
    let a_msb = a.extract(31, 31);
    let b_msb = b.extract(31, 31);
    let r_msb = diff.extract(31, 31);
    let overflow = a_msb.clone().eq_expr(b_msb).not_expr()
        .and_expr(a_msb.eq_expr(r_msb).not_expr());

    let of_bv = bool_to_bv1(flags.of_);
    let ref_bv = bool_to_bv1(overflow);

    ProofObligation {
        name: "x86_64: CMP_OF_writes_I32".to_string(),
        tmir_expr: ref_bv,
        aarch64_expr: of_bv,
        inputs: vec![("a".to_string(), 32), ("b".to_string(), 32)],
        preconditions: vec![],
        fp_inputs: vec![],
        category: None,
    }
}

pub fn all_x86_64_cmp_writes_i32_proofs() -> Vec<ProofObligation> {
    vec![
        proof_x86_cmp_writes_zf_i32(),
        proof_x86_cmp_writes_sf_i32(),
        proof_x86_cmp_writes_cf_i32(),
        proof_x86_cmp_writes_of_i32(),
    ]
}

/// Return all 8 direct x86-64 EFLAGS flag proofs at i8.
///
/// Four CMP + four SUB obligations. These supplement the ten indirect
/// `proof_x86_icmp_*_i32` proofs in `x86_64_lowering_proofs.rs` by pinning
/// the flag-write semantics directly.
pub fn all_x86_64_eflags_proofs() -> Vec<ProofObligation> {
    vec![
        proof_cmp_sf_i8(),
        proof_cmp_zf_i8(),
        proof_cmp_cf_i8(),
        proof_cmp_of_i8(),
        proof_sub_sf_i8(),
        proof_sub_zf_i8(),
        proof_sub_cf_i8(),
        proof_sub_of_i8(),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lowering_proof::verify_by_evaluation;
    use crate::verify::VerificationResult;

    fn assert_valid(obligation: &ProofObligation) {
        let result = verify_by_evaluation(obligation);
        assert!(
            matches!(result, VerificationResult::Valid),
            "Proof '{}' failed: {:?}",
            obligation.name,
            result
        );
    }

    #[test]
    fn test_proof_cmp_sf_i8() {
        assert_valid(&proof_cmp_sf_i8());
    }

    #[test]
    fn test_proof_cmp_zf_i8() {
        assert_valid(&proof_cmp_zf_i8());
    }

    #[test]
    fn test_proof_cmp_cf_i8() {
        assert_valid(&proof_cmp_cf_i8());
    }

    #[test]
    fn test_proof_cmp_of_i8() {
        assert_valid(&proof_cmp_of_i8());
    }

    #[test]
    fn test_proof_sub_sf_i8() {
        assert_valid(&proof_sub_sf_i8());
    }

    #[test]
    fn test_proof_sub_zf_i8() {
        assert_valid(&proof_sub_zf_i8());
    }

    #[test]
    fn test_proof_sub_cf_i8() {
        assert_valid(&proof_sub_cf_i8());
    }

    #[test]
    fn test_proof_sub_of_i8() {
        assert_valid(&proof_sub_of_i8());
    }

    #[test]
    fn test_proof_x86_cmp_writes_zf_i32() {
        assert_valid(&proof_x86_cmp_writes_zf_i32());
    }

    #[test]
    fn test_proof_x86_cmp_writes_sf_i32() {
        assert_valid(&proof_x86_cmp_writes_sf_i32());
    }

    #[test]
    fn test_proof_x86_cmp_writes_cf_i32() {
        assert_valid(&proof_x86_cmp_writes_cf_i32());
    }

    #[test]
    fn test_proof_x86_cmp_writes_of_i32() {
        assert_valid(&proof_x86_cmp_writes_of_i32());
    }

    #[test]
    fn test_all_x86_64_cmp_writes_i32_proofs_registered() {
        let proofs = all_x86_64_cmp_writes_i32_proofs();
        assert_eq!(proofs.len(), 4, "Expected 4 x86-64 CMP writes i32 proofs");
        for p in &proofs {
            assert_valid(p);
        }
    }

    #[test]
    fn test_all_x86_64_eflags_proofs_registered() {
        let proofs = all_x86_64_eflags_proofs();
        assert_eq!(proofs.len(), 8, "Expected 8 x86-64 EFLAGS flag proofs");
        for p in &proofs {
            assert_valid(p);
        }
    }
}
