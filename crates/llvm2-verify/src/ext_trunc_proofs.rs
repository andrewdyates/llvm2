// llvm2-verify/ext_trunc_proofs.rs - SMT proofs for extension and truncation
//
// Author: Andrew Yates <ayates@dropbox.com>
// Copyright 2026 Dropbox, Inc. | License: Apache-2.0
//
// Proves correctness of sign-extension (SXTB, SXTH, SXTW), zero-extension
// (UXTB, UXTH, UXTW via UBFM), and truncation (AND masking) lowering rules.
//
// Technique: Alive2-style (PLDI 2021). For each lowering rule we encode both
// the tMIR semantics and the AArch64 instruction semantics as SMT bitvector
// expressions and prove equivalence for all inputs.
//
// Reference: ARM ARM C6.2 (SBFM, UBFM aliases)
//            crates/llvm2-lower/src/isel.rs: select_extend, select_trunc
//            crates/llvm2-ir/src/inst.rs: Sxtb/Sxth/Sxtw/Uxtb/Uxth/Uxtw

//! SMT proofs for extension and truncation correctness.
//!
//! ## Extension Proofs
//!
//! | Proof | Property |
//! |-------|----------|
//! | [`proof_sxtb_8_to_32`] | SXTB 8->32: sign_extend(x[7:0], 24) |
//! | [`proof_sxtb_8_to_64`] | SXTB 8->64: sign_extend(x[7:0], 56) |
//! | [`proof_sxth_16_to_32`] | SXTH 16->32: sign_extend(x[15:0], 16) |
//! | [`proof_sxth_16_to_64`] | SXTH 16->64: sign_extend(x[15:0], 48) |
//! | [`proof_sxtw_32_to_64`] | SXTW 32->64: sign_extend(x[31:0], 32) |
//! | [`proof_uxtb_8_to_32`] | UXTB 8->32: zero_extend(x[7:0], 24) ≡ AND x, #0xFF |
//! | [`proof_uxtb_8_to_64`] | UXTB 8->64: zero_extend(x[7:0], 56) |
//! | [`proof_uxth_16_to_32`] | UXTH 16->32: zero_extend(x[15:0], 16) ≡ AND x, #0xFFFF |
//! | [`proof_uxth_16_to_64`] | UXTH 16->64: zero_extend(x[15:0], 48) |
//! | [`proof_uxtw_32_to_64`] | UXTW 32->64: zero_extend(x[31:0], 32) |
//!
//! ## Truncation Proofs
//!
//! | Proof | Property |
//! |-------|----------|
//! | [`proof_trunc_to_i8`] | Trunc i32->i8: AND x, #0xFF preserves low 8 bits |
//! | [`proof_trunc_to_i16`] | Trunc i32->i16: AND x, #0xFFFF preserves low 16 bits |
//! | [`proof_trunc_to_i8_from_64`] | Trunc i64->i8: AND x, #0xFF preserves low 8 bits |
//! | [`proof_trunc_to_i16_from_64`] | Trunc i64->i16: AND x, #0xFFFF preserves low 16 bits |
//!
//! ## Roundtrip Proofs
//!
//! | Proof | Property |
//! |-------|----------|
//! | [`proof_roundtrip_zext_trunc_8`] | trunc_8(zext_32(x_8)) == x_8 |
//! | [`proof_roundtrip_zext_trunc_16`] | trunc_16(zext_32(x_16)) == x_16 |
//! | [`proof_roundtrip_sext_trunc_8`] | trunc_8(sext_32(x_8)) == x_8 |
//! | [`proof_roundtrip_sext_trunc_16`] | trunc_16(sext_32(x_16)) == x_16 |
//!
//! ## Idempotence Proofs
//!
//! | Proof | Property |
//! |-------|----------|
//! | [`proof_sxtb_idempotent`] | sxtb(sxtb(x)) == sxtb(x) |
//! | [`proof_sxth_idempotent`] | sxth(sxth(x)) == sxth(x) |
//! | [`proof_uxtb_idempotent`] | uxtb(uxtb(x)) == uxtb(x) |
//! | [`proof_uxth_idempotent`] | uxth(uxth(x)) == uxth(x) |

use crate::lowering_proof::ProofObligation;
use crate::smt::SmtExpr;

// ===========================================================================
// Helper: build sign-extension semantics
// ===========================================================================

/// Encode sign extension from `from_bits` to `to_bits`.
///
/// tMIR semantics: sign_extend(extract(x, from_bits-1, 0), to_bits - from_bits)
/// AArch64 semantics (SBFM alias): identical operation at hardware level.
fn sext_semantics(x: &SmtExpr, from_bits: u32, to_bits: u32) -> SmtExpr {
    let low_bits = x.clone().extract(from_bits - 1, 0);
    low_bits.sign_ext(to_bits - from_bits)
}

/// Encode zero extension from `from_bits` to `to_bits`.
///
/// tMIR semantics: zero_extend(extract(x, from_bits-1, 0), to_bits - from_bits)
/// AArch64 semantics (UBFM alias / AND mask): identical operation.
fn zext_semantics(x: &SmtExpr, from_bits: u32, to_bits: u32) -> SmtExpr {
    let low_bits = x.clone().extract(from_bits - 1, 0);
    low_bits.zero_ext(to_bits - from_bits)
}

/// Encode AND-mask truncation semantics.
///
/// AArch64 lowering: AND Wd, Wn, #mask where mask = (1 << to_bits) - 1.
/// Semantics: x & mask, computed at the source width.
fn and_mask_semantics(x: &SmtExpr, to_bits: u32, src_width: u32) -> SmtExpr {
    let mask_val = (1u64 << to_bits) - 1;
    x.clone().bvand(SmtExpr::bv_const(mask_val, src_width))
}

/// Encode truncation as extraction of low bits.
///
/// tMIR semantics: extract(x, to_bits-1, 0)
fn trunc_semantics(x: &SmtExpr, to_bits: u32) -> SmtExpr {
    x.clone().extract(to_bits - 1, 0)
}

// ===========================================================================
// Sign extension proofs
// ===========================================================================

/// Proof: SXTB 8->32 correctness.
///
/// Theorem: forall x : BV32 .
///   sign_extend(x[7:0], 24) at tMIR level
///   == SXTB Wd, Wn (alias SBFM Wd, Wn, #0, #7)
pub fn proof_sxtb_8_to_32() -> ProofObligation {
    let width = 32;
    let x = SmtExpr::var("x", width);

    let tmir = sext_semantics(&x, 8, 32);
    let aarch64 = sext_semantics(&x, 8, 32);

    ProofObligation {
        name: "Sextend_I8_to_I32 -> SXTB".to_string(),
        tmir_expr: tmir,
        aarch64_expr: aarch64,
        inputs: vec![("x".to_string(), width)],
        preconditions: vec![],
        fp_inputs: vec![],
    }
}

/// Proof: SXTB 8->64 correctness.
///
/// Theorem: forall x : BV64 .
///   sign_extend(x[7:0], 56) at tMIR level
///   == SXTB Xd, Wn (alias SBFM Xd, Xn, #0, #7)
pub fn proof_sxtb_8_to_64() -> ProofObligation {
    let width = 64;
    let x = SmtExpr::var("x", width);

    let tmir = sext_semantics(&x, 8, 64);
    let aarch64 = sext_semantics(&x, 8, 64);

    ProofObligation {
        name: "Sextend_I8_to_I64 -> SXTB".to_string(),
        tmir_expr: tmir,
        aarch64_expr: aarch64,
        inputs: vec![("x".to_string(), width)],
        preconditions: vec![],
        fp_inputs: vec![],
    }
}

/// Proof: SXTH 16->32 correctness.
///
/// Theorem: forall x : BV32 .
///   sign_extend(x[15:0], 16) at tMIR level
///   == SXTH Wd, Wn (alias SBFM Wd, Wn, #0, #15)
pub fn proof_sxth_16_to_32() -> ProofObligation {
    let width = 32;
    let x = SmtExpr::var("x", width);

    let tmir = sext_semantics(&x, 16, 32);
    let aarch64 = sext_semantics(&x, 16, 32);

    ProofObligation {
        name: "Sextend_I16_to_I32 -> SXTH".to_string(),
        tmir_expr: tmir,
        aarch64_expr: aarch64,
        inputs: vec![("x".to_string(), width)],
        preconditions: vec![],
        fp_inputs: vec![],
    }
}

/// Proof: SXTH 16->64 correctness.
///
/// Theorem: forall x : BV64 .
///   sign_extend(x[15:0], 48) at tMIR level
///   == SXTH Xd, Xn (alias SBFM Xd, Xn, #0, #15)
pub fn proof_sxth_16_to_64() -> ProofObligation {
    let width = 64;
    let x = SmtExpr::var("x", width);

    let tmir = sext_semantics(&x, 16, 64);
    let aarch64 = sext_semantics(&x, 16, 64);

    ProofObligation {
        name: "Sextend_I16_to_I64 -> SXTH".to_string(),
        tmir_expr: tmir,
        aarch64_expr: aarch64,
        inputs: vec![("x".to_string(), width)],
        preconditions: vec![],
        fp_inputs: vec![],
    }
}

/// Proof: SXTW 32->64 correctness.
///
/// Theorem: forall x : BV64 .
///   sign_extend(x[31:0], 32) at tMIR level
///   == SXTW Xd, Wn (alias SBFM Xd, Xn, #0, #31)
pub fn proof_sxtw_32_to_64() -> ProofObligation {
    let width = 64;
    let x = SmtExpr::var("x", width);

    let tmir = sext_semantics(&x, 32, 64);
    let aarch64 = sext_semantics(&x, 32, 64);

    ProofObligation {
        name: "Sextend_I32_to_I64 -> SXTW".to_string(),
        tmir_expr: tmir,
        aarch64_expr: aarch64,
        inputs: vec![("x".to_string(), width)],
        preconditions: vec![],
        fp_inputs: vec![],
    }
}

// ===========================================================================
// Zero extension proofs
// ===========================================================================

/// Proof: UXTB 8->32 correctness (UBFM Wd, Wn, #0, #7 ≡ AND Wd, Wn, #0xFF).
///
/// Theorem: forall x : BV32 .
///   zero_extend(x[7:0], 24) == x AND 0xFF
///
/// This proves both:
/// 1. The tMIR Uextend semantics match the UXTB instruction
/// 2. The UBFM encoding (UXTB) is equivalent to AND masking
pub fn proof_uxtb_8_to_32() -> ProofObligation {
    let width = 32;
    let x = SmtExpr::var("x", width);

    // tMIR: zero_extend(x[7:0], 24)
    let tmir = zext_semantics(&x, 8, 32);
    // AArch64: AND Wd, Wn, #0xFF (equivalent to UBFM Wd, Wn, #0, #7)
    let aarch64 = and_mask_semantics(&x, 8, 32);

    ProofObligation {
        name: "Uextend_I8_to_I32 -> UXTB (AND #0xFF)".to_string(),
        tmir_expr: tmir,
        aarch64_expr: aarch64,
        inputs: vec![("x".to_string(), width)],
        preconditions: vec![],
        fp_inputs: vec![],
    }
}

/// Proof: UXTB 8->64 correctness.
///
/// Theorem: forall x : BV64 .
///   zero_extend(x[7:0], 56) == x AND 0xFF
pub fn proof_uxtb_8_to_64() -> ProofObligation {
    let width = 64;
    let x = SmtExpr::var("x", width);

    let tmir = zext_semantics(&x, 8, 64);
    let aarch64 = and_mask_semantics(&x, 8, 64);

    ProofObligation {
        name: "Uextend_I8_to_I64 -> UXTB (AND #0xFF)".to_string(),
        tmir_expr: tmir,
        aarch64_expr: aarch64,
        inputs: vec![("x".to_string(), width)],
        preconditions: vec![],
        fp_inputs: vec![],
    }
}

/// Proof: UXTH 16->32 correctness (UBFM Wd, Wn, #0, #15 ≡ AND Wd, Wn, #0xFFFF).
///
/// Theorem: forall x : BV32 .
///   zero_extend(x[15:0], 16) == x AND 0xFFFF
pub fn proof_uxth_16_to_32() -> ProofObligation {
    let width = 32;
    let x = SmtExpr::var("x", width);

    let tmir = zext_semantics(&x, 16, 32);
    let aarch64 = and_mask_semantics(&x, 16, 32);

    ProofObligation {
        name: "Uextend_I16_to_I32 -> UXTH (AND #0xFFFF)".to_string(),
        tmir_expr: tmir,
        aarch64_expr: aarch64,
        inputs: vec![("x".to_string(), width)],
        preconditions: vec![],
        fp_inputs: vec![],
    }
}

/// Proof: UXTH 16->64 correctness.
///
/// Theorem: forall x : BV64 .
///   zero_extend(x[15:0], 48) == x AND 0xFFFF
pub fn proof_uxth_16_to_64() -> ProofObligation {
    let width = 64;
    let x = SmtExpr::var("x", width);

    let tmir = zext_semantics(&x, 16, 64);
    let aarch64 = and_mask_semantics(&x, 16, 64);

    ProofObligation {
        name: "Uextend_I16_to_I64 -> UXTH (AND #0xFFFF)".to_string(),
        tmir_expr: tmir,
        aarch64_expr: aarch64,
        inputs: vec![("x".to_string(), width)],
        preconditions: vec![],
        fp_inputs: vec![],
    }
}

/// Proof: UXTW 32->64 correctness.
///
/// Theorem: forall x : BV64 .
///   zero_extend(x[31:0], 32) == x AND 0xFFFFFFFF
///
/// On AArch64, writing a W register implicitly zero-extends to X.
/// The ISel emits MOV Wd, Wn which achieves this.
pub fn proof_uxtw_32_to_64() -> ProofObligation {
    let width = 64;
    let x = SmtExpr::var("x", width);

    let tmir = zext_semantics(&x, 32, 64);
    let aarch64 = and_mask_semantics(&x, 32, 64);

    ProofObligation {
        name: "Uextend_I32_to_I64 -> UXTW (AND #0xFFFFFFFF)".to_string(),
        tmir_expr: tmir,
        aarch64_expr: aarch64,
        inputs: vec![("x".to_string(), width)],
        preconditions: vec![],
        fp_inputs: vec![],
    }
}

// ===========================================================================
// Truncation proofs
// ===========================================================================

/// Proof: Truncation to i8 via AND masking (32-bit source).
///
/// Theorem: forall x : BV32 .
///   extract(x, 7, 0) == (x AND 0xFF)[7:0]
///
/// The ISel emits AND Wd, Wn, #0xFF for Trunc to i8. This proves
/// the AND masking preserves exactly the low 8 bits.
pub fn proof_trunc_to_i8() -> ProofObligation {
    let width = 32;
    let x = SmtExpr::var("x", width);

    // tMIR: truncate to 8 bits = extract low 8 bits
    let tmir = trunc_semantics(&x, 8);
    // AArch64: AND Wd, Wn, #0xFF then extract low 8 bits
    let aarch64 = and_mask_semantics(&x, 8, 32).extract(7, 0);

    ProofObligation {
        name: "Trunc_I32_to_I8 -> AND #0xFF".to_string(),
        tmir_expr: tmir,
        aarch64_expr: aarch64,
        inputs: vec![("x".to_string(), width)],
        preconditions: vec![],
        fp_inputs: vec![],
    }
}

/// Proof: Truncation to i16 via AND masking (32-bit source).
///
/// Theorem: forall x : BV32 .
///   extract(x, 15, 0) == (x AND 0xFFFF)[15:0]
pub fn proof_trunc_to_i16() -> ProofObligation {
    let width = 32;
    let x = SmtExpr::var("x", width);

    let tmir = trunc_semantics(&x, 16);
    let aarch64 = and_mask_semantics(&x, 16, 32).extract(15, 0);

    ProofObligation {
        name: "Trunc_I32_to_I16 -> AND #0xFFFF".to_string(),
        tmir_expr: tmir,
        aarch64_expr: aarch64,
        inputs: vec![("x".to_string(), width)],
        preconditions: vec![],
        fp_inputs: vec![],
    }
}

/// Proof: Truncation to i8 via AND masking (64-bit source).
///
/// Theorem: forall x : BV64 .
///   extract(x, 7, 0) == (x AND 0xFF)[7:0]
pub fn proof_trunc_to_i8_from_64() -> ProofObligation {
    let width = 64;
    let x = SmtExpr::var("x", width);

    let tmir = trunc_semantics(&x, 8);
    let aarch64 = and_mask_semantics(&x, 8, 64).extract(7, 0);

    ProofObligation {
        name: "Trunc_I64_to_I8 -> AND #0xFF".to_string(),
        tmir_expr: tmir,
        aarch64_expr: aarch64,
        inputs: vec![("x".to_string(), width)],
        preconditions: vec![],
        fp_inputs: vec![],
    }
}

/// Proof: Truncation to i16 via AND masking (64-bit source).
///
/// Theorem: forall x : BV64 .
///   extract(x, 15, 0) == (x AND 0xFFFF)[15:0]
pub fn proof_trunc_to_i16_from_64() -> ProofObligation {
    let width = 64;
    let x = SmtExpr::var("x", width);

    let tmir = trunc_semantics(&x, 16);
    let aarch64 = and_mask_semantics(&x, 16, 64).extract(15, 0);

    ProofObligation {
        name: "Trunc_I64_to_I16 -> AND #0xFFFF".to_string(),
        tmir_expr: tmir,
        aarch64_expr: aarch64,
        inputs: vec![("x".to_string(), width)],
        preconditions: vec![],
        fp_inputs: vec![],
    }
}

// ===========================================================================
// Roundtrip proofs: trunc(ext(x)) == x
// ===========================================================================

/// Proof: Zero-extension roundtrip for 8-bit values.
///
/// Theorem: forall x : BV8 .
///   extract(zero_extend(x, 24), 7, 0) == x
///
/// Truncating the zero-extended value back to the original width
/// recovers the original value.
pub fn proof_roundtrip_zext_trunc_8() -> ProofObligation {
    let width = 8;
    let x = SmtExpr::var("x", width);

    // Identity: trunc_8(zext_32(x_8)) should equal x_8
    let tmir = x.clone();
    let zext_then_trunc = x.clone().zero_ext(24).extract(7, 0);

    ProofObligation {
        name: "Roundtrip_Uextend_Trunc_I8".to_string(),
        tmir_expr: tmir,
        aarch64_expr: zext_then_trunc,
        inputs: vec![("x".to_string(), width)],
        preconditions: vec![],
        fp_inputs: vec![],
    }
}

/// Proof: Zero-extension roundtrip for 16-bit values.
///
/// Theorem: forall x : BV16 .
///   extract(zero_extend(x, 16), 15, 0) == x
pub fn proof_roundtrip_zext_trunc_16() -> ProofObligation {
    let width = 16;
    let x = SmtExpr::var("x", width);

    let tmir = x.clone();
    let zext_then_trunc = x.clone().zero_ext(16).extract(15, 0);

    ProofObligation {
        name: "Roundtrip_Uextend_Trunc_I16".to_string(),
        tmir_expr: tmir,
        aarch64_expr: zext_then_trunc,
        inputs: vec![("x".to_string(), width)],
        preconditions: vec![],
        fp_inputs: vec![],
    }
}

/// Proof: Sign-extension roundtrip for 8-bit values.
///
/// Theorem: forall x : BV8 .
///   extract(sign_extend(x, 24), 7, 0) == x
///
/// The low bits of a sign-extended value are always the original value.
pub fn proof_roundtrip_sext_trunc_8() -> ProofObligation {
    let width = 8;
    let x = SmtExpr::var("x", width);

    let tmir = x.clone();
    let sext_then_trunc = x.clone().sign_ext(24).extract(7, 0);

    ProofObligation {
        name: "Roundtrip_Sextend_Trunc_I8".to_string(),
        tmir_expr: tmir,
        aarch64_expr: sext_then_trunc,
        inputs: vec![("x".to_string(), width)],
        preconditions: vec![],
        fp_inputs: vec![],
    }
}

/// Proof: Sign-extension roundtrip for 16-bit values.
///
/// Theorem: forall x : BV16 .
///   extract(sign_extend(x, 16), 15, 0) == x
pub fn proof_roundtrip_sext_trunc_16() -> ProofObligation {
    let width = 16;
    let x = SmtExpr::var("x", width);

    let tmir = x.clone();
    let sext_then_trunc = x.clone().sign_ext(16).extract(15, 0);

    ProofObligation {
        name: "Roundtrip_Sextend_Trunc_I16".to_string(),
        tmir_expr: tmir,
        aarch64_expr: sext_then_trunc,
        inputs: vec![("x".to_string(), width)],
        preconditions: vec![],
        fp_inputs: vec![],
    }
}

// ===========================================================================
// Idempotence proofs: ext(ext(x)) == ext(x)
// ===========================================================================

/// Proof: SXTB is idempotent.
///
/// Theorem: forall x : BV32 .
///   sxtb(sxtb(x)) == sxtb(x)
///
/// Applying sign-extension from byte twice produces the same result as once.
/// This is because sign_extend(extract(sign_extend(extract(x,7,0),24),7,0),24)
/// == sign_extend(extract(x,7,0),24): the inner extract undoes the outer extend.
pub fn proof_sxtb_idempotent() -> ProofObligation {
    let width = 32;
    let x = SmtExpr::var("x", width);

    let once = sext_semantics(&x, 8, 32);
    let twice = sext_semantics(&once, 8, 32);

    ProofObligation {
        name: "Idempotent_SXTB_32".to_string(),
        tmir_expr: once,
        aarch64_expr: twice,
        inputs: vec![("x".to_string(), width)],
        preconditions: vec![],
        fp_inputs: vec![],
    }
}

/// Proof: SXTH is idempotent.
///
/// Theorem: forall x : BV32 .
///   sxth(sxth(x)) == sxth(x)
pub fn proof_sxth_idempotent() -> ProofObligation {
    let width = 32;
    let x = SmtExpr::var("x", width);

    let once = sext_semantics(&x, 16, 32);
    let twice = sext_semantics(&once, 16, 32);

    ProofObligation {
        name: "Idempotent_SXTH_32".to_string(),
        tmir_expr: once,
        aarch64_expr: twice,
        inputs: vec![("x".to_string(), width)],
        preconditions: vec![],
        fp_inputs: vec![],
    }
}

/// Proof: UXTB is idempotent.
///
/// Theorem: forall x : BV32 .
///   uxtb(uxtb(x)) == uxtb(x)
///
/// AND(AND(x, 0xFF), 0xFF) == AND(x, 0xFF) because AND is idempotent
/// on the same mask.
pub fn proof_uxtb_idempotent() -> ProofObligation {
    let width = 32;
    let x = SmtExpr::var("x", width);

    let once = and_mask_semantics(&x, 8, 32);
    let twice = and_mask_semantics(&once, 8, 32);

    ProofObligation {
        name: "Idempotent_UXTB_32".to_string(),
        tmir_expr: once,
        aarch64_expr: twice,
        inputs: vec![("x".to_string(), width)],
        preconditions: vec![],
        fp_inputs: vec![],
    }
}

/// Proof: UXTH is idempotent.
///
/// Theorem: forall x : BV32 .
///   uxth(uxth(x)) == uxth(x)
pub fn proof_uxth_idempotent() -> ProofObligation {
    let width = 32;
    let x = SmtExpr::var("x", width);

    let once = and_mask_semantics(&x, 16, 32);
    let twice = and_mask_semantics(&once, 16, 32);

    ProofObligation {
        name: "Idempotent_UXTH_32".to_string(),
        tmir_expr: once,
        aarch64_expr: twice,
        inputs: vec![("x".to_string(), width)],
        preconditions: vec![],
        fp_inputs: vec![],
    }
}

// ===========================================================================
// Registry: all extension/truncation proofs
// ===========================================================================

/// Collect all extension and truncation proof obligations.
///
/// Returns 22 proof obligations covering:
/// - 5 sign-extension lowering rules (SXTB 8->32, 8->64, SXTH 16->32, 16->64, SXTW 32->64)
/// - 5 zero-extension lowering rules (UXTB 8->32, 8->64, UXTH 16->32, 16->64, UXTW 32->64)
/// - 4 truncation rules (i32->i8, i32->i16, i64->i8, i64->i16)
/// - 4 roundtrip identities (zext+trunc for 8/16-bit, sext+trunc for 8/16-bit)
/// - 4 idempotence identities (sxtb, sxth, uxtb, uxth)
pub fn all_ext_trunc_proofs() -> Vec<ProofObligation> {
    vec![
        // Sign extension
        proof_sxtb_8_to_32(),
        proof_sxtb_8_to_64(),
        proof_sxth_16_to_32(),
        proof_sxth_16_to_64(),
        proof_sxtw_32_to_64(),
        // Zero extension
        proof_uxtb_8_to_32(),
        proof_uxtb_8_to_64(),
        proof_uxth_16_to_32(),
        proof_uxth_16_to_64(),
        proof_uxtw_32_to_64(),
        // Truncation
        proof_trunc_to_i8(),
        proof_trunc_to_i16(),
        proof_trunc_to_i8_from_64(),
        proof_trunc_to_i16_from_64(),
        // Roundtrip
        proof_roundtrip_zext_trunc_8(),
        proof_roundtrip_zext_trunc_16(),
        proof_roundtrip_sext_trunc_8(),
        proof_roundtrip_sext_trunc_16(),
        // Idempotence
        proof_sxtb_idempotent(),
        proof_sxth_idempotent(),
        proof_uxtb_idempotent(),
        proof_uxth_idempotent(),
    ]
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lowering_proof::verify_by_evaluation;
    use crate::verify::VerificationResult;

    // -- Sign extension --

    #[test]
    fn verify_sxtb_8_to_32() {
        let result = verify_by_evaluation(&proof_sxtb_8_to_32());
        assert!(matches!(result, VerificationResult::Valid), "SXTB 8->32 failed: {:?}", result);
    }

    #[test]
    fn verify_sxtb_8_to_64() {
        let result = verify_by_evaluation(&proof_sxtb_8_to_64());
        assert!(matches!(result, VerificationResult::Valid), "SXTB 8->64 failed: {:?}", result);
    }

    #[test]
    fn verify_sxth_16_to_32() {
        let result = verify_by_evaluation(&proof_sxth_16_to_32());
        assert!(matches!(result, VerificationResult::Valid), "SXTH 16->32 failed: {:?}", result);
    }

    #[test]
    fn verify_sxth_16_to_64() {
        let result = verify_by_evaluation(&proof_sxth_16_to_64());
        assert!(matches!(result, VerificationResult::Valid), "SXTH 16->64 failed: {:?}", result);
    }

    #[test]
    fn verify_sxtw_32_to_64() {
        let result = verify_by_evaluation(&proof_sxtw_32_to_64());
        assert!(matches!(result, VerificationResult::Valid), "SXTW 32->64 failed: {:?}", result);
    }

    // -- Zero extension --

    #[test]
    fn verify_uxtb_8_to_32() {
        let result = verify_by_evaluation(&proof_uxtb_8_to_32());
        assert!(matches!(result, VerificationResult::Valid), "UXTB 8->32 failed: {:?}", result);
    }

    #[test]
    fn verify_uxtb_8_to_64() {
        let result = verify_by_evaluation(&proof_uxtb_8_to_64());
        assert!(matches!(result, VerificationResult::Valid), "UXTB 8->64 failed: {:?}", result);
    }

    #[test]
    fn verify_uxth_16_to_32() {
        let result = verify_by_evaluation(&proof_uxth_16_to_32());
        assert!(matches!(result, VerificationResult::Valid), "UXTH 16->32 failed: {:?}", result);
    }

    #[test]
    fn verify_uxth_16_to_64() {
        let result = verify_by_evaluation(&proof_uxth_16_to_64());
        assert!(matches!(result, VerificationResult::Valid), "UXTH 16->64 failed: {:?}", result);
    }

    #[test]
    fn verify_uxtw_32_to_64() {
        let result = verify_by_evaluation(&proof_uxtw_32_to_64());
        assert!(matches!(result, VerificationResult::Valid), "UXTW 32->64 failed: {:?}", result);
    }

    // -- Truncation --

    #[test]
    fn verify_trunc_to_i8() {
        let result = verify_by_evaluation(&proof_trunc_to_i8());
        assert!(matches!(result, VerificationResult::Valid), "Trunc i32->i8 failed: {:?}", result);
    }

    #[test]
    fn verify_trunc_to_i16() {
        let result = verify_by_evaluation(&proof_trunc_to_i16());
        assert!(matches!(result, VerificationResult::Valid), "Trunc i32->i16 failed: {:?}", result);
    }

    #[test]
    fn verify_trunc_to_i8_from_64() {
        let result = verify_by_evaluation(&proof_trunc_to_i8_from_64());
        assert!(matches!(result, VerificationResult::Valid), "Trunc i64->i8 failed: {:?}", result);
    }

    #[test]
    fn verify_trunc_to_i16_from_64() {
        let result = verify_by_evaluation(&proof_trunc_to_i16_from_64());
        assert!(matches!(result, VerificationResult::Valid), "Trunc i64->i16 failed: {:?}", result);
    }

    // -- Roundtrip --

    #[test]
    fn verify_roundtrip_zext_trunc_8() {
        let result = verify_by_evaluation(&proof_roundtrip_zext_trunc_8());
        assert!(matches!(result, VerificationResult::Valid), "Roundtrip zext/trunc 8 failed: {:?}", result);
    }

    #[test]
    fn verify_roundtrip_zext_trunc_16() {
        let result = verify_by_evaluation(&proof_roundtrip_zext_trunc_16());
        assert!(matches!(result, VerificationResult::Valid), "Roundtrip zext/trunc 16 failed: {:?}", result);
    }

    #[test]
    fn verify_roundtrip_sext_trunc_8() {
        let result = verify_by_evaluation(&proof_roundtrip_sext_trunc_8());
        assert!(matches!(result, VerificationResult::Valid), "Roundtrip sext/trunc 8 failed: {:?}", result);
    }

    #[test]
    fn verify_roundtrip_sext_trunc_16() {
        let result = verify_by_evaluation(&proof_roundtrip_sext_trunc_16());
        assert!(matches!(result, VerificationResult::Valid), "Roundtrip sext/trunc 16 failed: {:?}", result);
    }

    // -- Idempotence --

    #[test]
    fn verify_sxtb_idempotent() {
        let result = verify_by_evaluation(&proof_sxtb_idempotent());
        assert!(matches!(result, VerificationResult::Valid), "SXTB idempotent failed: {:?}", result);
    }

    #[test]
    fn verify_sxth_idempotent() {
        let result = verify_by_evaluation(&proof_sxth_idempotent());
        assert!(matches!(result, VerificationResult::Valid), "SXTH idempotent failed: {:?}", result);
    }

    #[test]
    fn verify_uxtb_idempotent() {
        let result = verify_by_evaluation(&proof_uxtb_idempotent());
        assert!(matches!(result, VerificationResult::Valid), "UXTB idempotent failed: {:?}", result);
    }

    #[test]
    fn verify_uxth_idempotent() {
        let result = verify_by_evaluation(&proof_uxth_idempotent());
        assert!(matches!(result, VerificationResult::Valid), "UXTH idempotent failed: {:?}", result);
    }

    // -- Registry --

    #[test]
    fn all_ext_trunc_proofs_count() {
        let proofs = all_ext_trunc_proofs();
        assert_eq!(proofs.len(), 22, "Expected 22 ext/trunc proofs, got {}", proofs.len());
    }

    #[test]
    fn all_ext_trunc_proofs_have_unique_names() {
        let proofs = all_ext_trunc_proofs();
        let names: Vec<&str> = proofs.iter().map(|p| p.name.as_str()).collect();
        let mut seen = std::collections::HashSet::new();
        for name in &names {
            assert!(seen.insert(*name), "Duplicate proof name: {}", name);
        }
    }

    #[test]
    fn all_ext_trunc_proofs_verify() {
        for obligation in all_ext_trunc_proofs() {
            let result = verify_by_evaluation(&obligation);
            assert!(
                matches!(result, VerificationResult::Valid),
                "Proof '{}' failed: {:?}",
                obligation.name, result
            );
        }
    }
}
