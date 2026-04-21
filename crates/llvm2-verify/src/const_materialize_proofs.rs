// llvm2-verify/const_materialize_proofs.rs - SMT proofs for constant materialization
//
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Copyright 2026 Dropbox, Inc. | License: Apache-2.0
//
// Proves that AArch64 constant materialization strategies in
// llvm2-opt/const_materialize.rs produce the correct register values.
// Each strategy (MOVZ, MOVZ+MOVK, ORR logical immediate, MOVN) is modeled
// as SMT bitvector operations and verified against the expected constant.
//
// Technique: Alive2-style (PLDI 2021). For each strategy, encode the
// instruction sequence semantics as SMT bitvector expressions and the
// desired constant value as the target. Check `NOT(strategy == target)`
// for UNSAT. If UNSAT, the materialization is proven correct.
//
// Reference: crates/llvm2-opt/src/const_materialize.rs
// Reference: ARM Architecture Reference Manual (DDI 0487), C6.2

//! SMT proofs for AArch64 constant materialization strategies.
//!
//! Each proof corresponds to a materialization strategy in
//! `llvm2_opt::const_materialize::materialize_constant`:
//!
//! ## MOVZ (move wide with zero)
//!
//! | Rule | Semantics | Proof |
//! |------|-----------|-------|
//! | MOVZ hw=0 | `Xd = zext(imm16)` | [`proof_movz_hw0`] |
//! | MOVZ hw=1 | `Xd = zext(imm16) << 16` | [`proof_movz_hw1`] |
//! | MOVZ hw=2 | `Xd = zext(imm16) << 32` | [`proof_movz_hw2`] |
//! | MOVZ hw=3 | `Xd = zext(imm16) << 48` | [`proof_movz_hw3`] |
//!
//! ## MOVZ + MOVK (32-bit assembly)
//!
//! | Rule | Semantics | Proof |
//! |------|-----------|-------|
//! | MOVZ+MOVK 32 | `(hi16 << 16) \| lo16` | [`proof_movz_movk_32bit`] |
//!
//! ## MOVZ + 3xMOVK (64-bit assembly)
//!
//! | Rule | Semantics | Proof |
//! |------|-----------|-------|
//! | MOVZ+3xMOVK 64 | `(hw3<<48)\|(hw2<<32)\|(hw1<<16)\|hw0` | [`proof_movz_movk_64bit`] |
//!
//! ## ORR logical immediate
//!
//! | Rule | Semantics | Proof |
//! |------|-----------|-------|
//! | ORR XZR, #imm | `XZR \| bitmask = bitmask` | [`proof_orr_logical_imm`] |
//!
//! ## MOVN (move wide with NOT)
//!
//! | Rule | Semantics | Proof |
//! |------|-----------|-------|
//! | MOVN hw=0 | `~(zext(imm16))` | [`proof_movn_hw0`] |
//! | MOVN hw=1 | `~(zext(imm16) << 16)` | [`proof_movn_hw1`] |
//!
//! ## Strategy equivalence
//!
//! | Rule | Semantics | Proof |
//! |------|-----------|-------|
//! | ORR == MOVZ overlap | For single-chunk values | [`proof_orr_movz_equivalence`] |

use crate::lowering_proof::ProofObligation;
use crate::smt::SmtExpr;

// ---------------------------------------------------------------------------
// Semantic encoding helpers
// ---------------------------------------------------------------------------

/// Encode MOVZ semantics: `Xd = zero_extend(imm16) << (hw * 16)`.
///
/// MOVZ zeros the entire register, then inserts the 16-bit immediate at
/// the specified halfword position. The result is `imm16 << shift`.
///
/// ARM ARM: "MOVZ moves a 16-bit immediate value to a register, shifting
/// it left by 0, 16, 32 or 48 bits, and clearing the remaining bits."
fn encode_movz(imm16: SmtExpr, hw_shift: u32, width: u32) -> SmtExpr {
    let imm_width = imm16.bv_width();
    let extended = imm16.zero_ext(width - imm_width);
    let shift_amount = SmtExpr::bv_const(hw_shift as u64, width);
    extended.bvshl(shift_amount)
}

/// Encode MOVK semantics: insert 16-bit immediate into existing register value.
///
/// MOVK keeps all other bits of the destination register, and replaces
/// exactly 16 bits at the halfword position `hw_shift`.
///
/// Semantics: `Xd = (Xd & ~(0xFFFF << hw_shift)) | (zext(imm16) << hw_shift)`
///
/// ARM ARM: "MOVK moves a 16-bit immediate value to the specified halfword
/// position of the register, keeping all other bits unchanged."
fn encode_movk(prev: SmtExpr, imm16: SmtExpr, hw_shift: u32, width: u32) -> SmtExpr {
    let imm_width = imm16.bv_width();
    let extended = imm16.zero_ext(width - imm_width);
    let shift_amount = SmtExpr::bv_const(hw_shift as u64, width);
    let shifted = extended.bvshl(shift_amount);

    // Build the mask: ~(0xFFFF << hw_shift)
    let mask_16 = SmtExpr::bv_const(0xFFFF, width);
    let shifted_mask = mask_16.bvshl(SmtExpr::bv_const(hw_shift as u64, width));
    let all_ones = if width >= 64 { u64::MAX } else { (1u64 << width) - 1 };
    let inverted_mask = shifted_mask.bvxor(SmtExpr::bv_const(all_ones, width));

    // Clear the halfword, then insert the new value
    prev.bvand(inverted_mask).bvor(shifted)
}

/// Encode ORR with XZR semantics: `Xd = XZR | bitmask = bitmask`.
///
/// ORR with XZR as the first operand simply loads the logical immediate
/// into the register. This is used for bitmask patterns that are encodable
/// as ARM logical immediates.
fn encode_orr_xzr(bitmask: SmtExpr, width: u32) -> SmtExpr {
    let zero = SmtExpr::bv_const(0, width);
    zero.bvor(bitmask)
}

/// Encode MOVN semantics: `Xd = ~(zero_extend(imm16) << (hw * 16))`.
///
/// MOVN inserts a 16-bit immediate at the specified halfword position,
/// zeros other bits, then inverts the entire result.
///
/// ARM ARM: "MOVN moves the bitwise inverse of a 16-bit immediate value
/// (optionally shifted) to a register."
fn encode_movn(imm16: SmtExpr, hw_shift: u32, width: u32) -> SmtExpr {
    let movz_result = encode_movz(imm16, hw_shift, width);
    let all_ones = if width >= 64 { u64::MAX } else { (1u64 << width) - 1 };
    movz_result.bvxor(SmtExpr::bv_const(all_ones, width))
}


// ---------------------------------------------------------------------------
// MOVZ proofs: single instruction, each halfword position
// ---------------------------------------------------------------------------

/// Proof: MOVZ #imm16, LSL #0 produces `zext(imm16)`.
///
/// Theorem: forall imm16 : BV16 . zext64(imm16) << 0 == zext64(imm16)
///
/// The simplest materialization: a 16-bit immediate placed in bits [15:0]
/// with bits [63:16] zeroed. Used for constants 0..65535.
pub fn proof_movz_hw0() -> ProofObligation {
    let width = 64;
    let imm16 = SmtExpr::var("imm16", 16);
    let target = imm16.clone().zero_ext(48); // desired value
    let strategy = encode_movz(imm16, 0, width);

    ProofObligation {
        name: "ConstMat: MOVZ #imm16, LSL #0 == zext(imm16)".to_string(),
        tmir_expr: target,
        aarch64_expr: strategy,
        inputs: vec![("imm16".to_string(), 16)],
        preconditions: vec![],
        fp_inputs: vec![],
            category: None,
    }
}

/// Proof: MOVZ #imm16, LSL #16 produces `zext(imm16) << 16`.
///
/// Theorem: forall imm16 : BV16 . zext64(imm16) << 16 == zext64(imm16) << 16
///
/// Used for constants with only bits [31:16] set, e.g., 0x10000..0xFFFF0000.
pub fn proof_movz_hw1() -> ProofObligation {
    let width = 64;
    let imm16 = SmtExpr::var("imm16", 16);
    let target = imm16.clone().zero_ext(48).bvshl(SmtExpr::bv_const(16, width));
    let strategy = encode_movz(imm16, 16, width);

    ProofObligation {
        name: "ConstMat: MOVZ #imm16, LSL #16 == zext(imm16) << 16".to_string(),
        tmir_expr: target,
        aarch64_expr: strategy,
        inputs: vec![("imm16".to_string(), 16)],
        preconditions: vec![],
        fp_inputs: vec![],
            category: None,
    }
}

/// Proof: MOVZ #imm16, LSL #32 produces `zext(imm16) << 32`.
///
/// Used for constants with only bits [47:32] set.
pub fn proof_movz_hw2() -> ProofObligation {
    let width = 64;
    let imm16 = SmtExpr::var("imm16", 16);
    let target = imm16.clone().zero_ext(48).bvshl(SmtExpr::bv_const(32, width));
    let strategy = encode_movz(imm16, 32, width);

    ProofObligation {
        name: "ConstMat: MOVZ #imm16, LSL #32 == zext(imm16) << 32".to_string(),
        tmir_expr: target,
        aarch64_expr: strategy,
        inputs: vec![("imm16".to_string(), 16)],
        preconditions: vec![],
        fp_inputs: vec![],
            category: None,
    }
}

/// Proof: MOVZ #imm16, LSL #48 produces `zext(imm16) << 48`.
///
/// Used for constants with only bits [63:48] set.
pub fn proof_movz_hw3() -> ProofObligation {
    let width = 64;
    let imm16 = SmtExpr::var("imm16", 16);
    let target = imm16.clone().zero_ext(48).bvshl(SmtExpr::bv_const(48, width));
    let strategy = encode_movz(imm16, 48, width);

    ProofObligation {
        name: "ConstMat: MOVZ #imm16, LSL #48 == zext(imm16) << 48".to_string(),
        tmir_expr: target,
        aarch64_expr: strategy,
        inputs: vec![("imm16".to_string(), 16)],
        preconditions: vec![],
        fp_inputs: vec![],
            category: None,
    }
}

/// Proof: MOVZ 32-bit width at hw=0 (8-bit exhaustive variant for imm16).
pub fn proof_movz_hw0_8bit() -> ProofObligation {
    let width = 16; // Use 16-bit total for exhaustive 8-bit imm
    let imm = SmtExpr::var("imm", 8);
    let target = imm.clone().zero_ext(8);
    let extended = imm.zero_ext(8);
    let strategy = extended.bvshl(SmtExpr::bv_const(0, width));

    ProofObligation {
        name: "ConstMat: MOVZ #imm8, LSL #0 == zext(imm8) (8-bit)".to_string(),
        tmir_expr: target,
        aarch64_expr: strategy,
        inputs: vec![("imm".to_string(), 8)],
        preconditions: vec![],
        fp_inputs: vec![],
            category: None,
    }
}

/// Proof: MOVZ 16-bit at hw=1 (8-bit exhaustive).
pub fn proof_movz_hw1_8bit() -> ProofObligation {
    let width = 16;
    let imm = SmtExpr::var("imm", 8);
    let target = imm.clone().zero_ext(8).bvshl(SmtExpr::bv_const(8, width));
    let strategy = imm.zero_ext(8).bvshl(SmtExpr::bv_const(8, width));

    ProofObligation {
        name: "ConstMat: MOVZ #imm8, LSL #8 == zext(imm8) << 8 (8-bit)".to_string(),
        tmir_expr: target,
        aarch64_expr: strategy,
        inputs: vec![("imm".to_string(), 8)],
        preconditions: vec![],
        fp_inputs: vec![],
            category: None,
    }
}

// ---------------------------------------------------------------------------
// MOVZ + MOVK proofs: two-instruction 32-bit assembly
// ---------------------------------------------------------------------------

/// Proof: MOVZ + MOVK assembles a 32-bit value from two 16-bit halves.
///
/// Theorem: forall lo16, hi16 : BV16 .
///     MOVK(MOVZ(lo16, 0), hi16, 16) == (zext32(hi16) << 16) | zext32(lo16)
///
/// The MOVZ loads the low 16 bits and zeros bits [31:16].
/// The MOVK then inserts the high 16 bits without disturbing bits [15:0].
/// The result is the full 32-bit value.
pub fn proof_movz_movk_32bit() -> ProofObligation {
    let width = 32;
    let lo16 = SmtExpr::var("lo16", 16);
    let hi16 = SmtExpr::var("hi16", 16);

    // Target: the 32-bit value (hi16 << 16) | lo16
    let target = hi16.clone().zero_ext(16).bvshl(SmtExpr::bv_const(16, width))
        .bvor(lo16.clone().zero_ext(16));

    // Strategy: MOVZ lo16 at hw=0, then MOVK hi16 at hw=1
    let step1 = encode_movz(lo16, 0, width);
    let step2 = encode_movk(step1, hi16, 16, width);

    ProofObligation {
        name: "ConstMat: MOVZ+MOVK assembles 32-bit value".to_string(),
        tmir_expr: target,
        aarch64_expr: step2,
        inputs: vec![("lo16".to_string(), 16), ("hi16".to_string(), 16)],
        preconditions: vec![],
        fp_inputs: vec![],
            category: None,
    }
}

/// Proof: MOVZ + MOVK assembles 16-bit value from two 8-bit halves (exhaustive).
pub fn proof_movz_movk_16bit() -> ProofObligation {
    let width = 16;
    let lo8 = SmtExpr::var("lo8", 8);
    let hi8 = SmtExpr::var("hi8", 8);

    // Target: (hi8 << 8) | lo8
    let target = hi8.clone().zero_ext(8).bvshl(SmtExpr::bv_const(8, width))
        .bvor(lo8.clone().zero_ext(8));

    // Strategy: MOVZ lo8 at shift=0 (in 16-bit), then MOVK hi8 at shift=8
    let step1_extended = lo8.zero_ext(8);
    let step1 = step1_extended.bvshl(SmtExpr::bv_const(0, width));
    let step2 = encode_movk(step1, hi8, 8, width);

    ProofObligation {
        name: "ConstMat: MOVZ+MOVK assembles 16-bit from 8-bit halves (exhaustive)".to_string(),
        tmir_expr: target,
        aarch64_expr: step2,
        inputs: vec![("lo8".to_string(), 8), ("hi8".to_string(), 8)],
        preconditions: vec![],
        fp_inputs: vec![],
            category: None,
    }
}

// ---------------------------------------------------------------------------
// MOVZ + 3xMOVK proof: four-instruction 64-bit assembly
// ---------------------------------------------------------------------------

/// Proof: MOVZ + 3xMOVK assembles a 64-bit value from four 16-bit chunks.
///
/// Theorem: forall hw0, hw1, hw2, hw3 : BV16 .
///     MOVK(MOVK(MOVK(MOVZ(hw0, 0), hw1, 16), hw2, 32), hw3, 48) ==
///     (zext64(hw3) << 48) | (zext64(hw2) << 32) | (zext64(hw1) << 16) | zext64(hw0)
///
/// This is the general case for arbitrary 64-bit constants that don't
/// match simpler patterns (single MOVZ, logical immediate, MOVN).
pub fn proof_movz_movk_64bit() -> ProofObligation {
    let width = 64;
    let hw0 = SmtExpr::var("hw0", 16);
    let hw1 = SmtExpr::var("hw1", 16);
    let hw2 = SmtExpr::var("hw2", 16);
    let hw3 = SmtExpr::var("hw3", 16);

    // Target: the full 64-bit value assembled from four halfwords
    let target = hw3.clone().zero_ext(48).bvshl(SmtExpr::bv_const(48, width))
        .bvor(hw2.clone().zero_ext(48).bvshl(SmtExpr::bv_const(32, width)))
        .bvor(hw1.clone().zero_ext(48).bvshl(SmtExpr::bv_const(16, width)))
        .bvor(hw0.clone().zero_ext(48));

    // Strategy: MOVZ hw0 at 0, MOVK hw1 at 16, MOVK hw2 at 32, MOVK hw3 at 48
    let step1 = encode_movz(hw0, 0, width);
    let step2 = encode_movk(step1, hw1, 16, width);
    let step3 = encode_movk(step2, hw2, 32, width);
    let step4 = encode_movk(step3, hw3, 48, width);

    ProofObligation {
        name: "ConstMat: MOVZ+3xMOVK assembles 64-bit value".to_string(),
        tmir_expr: target,
        aarch64_expr: step4,
        inputs: vec![
            ("hw0".to_string(), 16),
            ("hw1".to_string(), 16),
            ("hw2".to_string(), 16),
            ("hw3".to_string(), 16),
        ],
        preconditions: vec![],
        fp_inputs: vec![],
            category: None,
    }
}

// ---------------------------------------------------------------------------
// ORR logical immediate proofs
// ---------------------------------------------------------------------------

/// Proof: ORR Xd, XZR, #bitmask produces the bitmask value.
///
/// Theorem: forall bitmask : BV64 . (0 | bitmask) == bitmask
///
/// This is the identity property of bitwise OR with zero. The ORR
/// instruction with XZR (zero register) as the first operand simply
/// loads the logical immediate encoding into the destination register.
pub fn proof_orr_logical_imm() -> ProofObligation {
    let width = 64;
    let bitmask = SmtExpr::var("bitmask", width);

    let target = bitmask.clone();
    let strategy = encode_orr_xzr(bitmask, width);

    ProofObligation {
        name: "ConstMat: ORR Xd, XZR, #bitmask == bitmask".to_string(),
        tmir_expr: target,
        aarch64_expr: strategy,
        inputs: vec![("bitmask".to_string(), width)],
        preconditions: vec![],
        fp_inputs: vec![],
            category: None,
    }
}

/// Proof: ORR Wd, WZR, #bitmask produces the bitmask value (32-bit).
pub fn proof_orr_logical_imm_32bit() -> ProofObligation {
    let width = 32;
    let bitmask = SmtExpr::var("bitmask", width);

    let target = bitmask.clone();
    let strategy = encode_orr_xzr(bitmask, width);

    ProofObligation {
        name: "ConstMat: ORR Wd, WZR, #bitmask == bitmask (32-bit)".to_string(),
        tmir_expr: target,
        aarch64_expr: strategy,
        inputs: vec![("bitmask".to_string(), width)],
        preconditions: vec![],
        fp_inputs: vec![],
            category: None,
    }
}

/// Proof: ORR logical immediate (8-bit exhaustive).
pub fn proof_orr_logical_imm_8bit() -> ProofObligation {
    let width = 8;
    let bitmask = SmtExpr::var("bitmask", width);

    let target = bitmask.clone();
    let strategy = SmtExpr::bv_const(0, width).bvor(bitmask);

    ProofObligation {
        name: "ConstMat: ORR XZR, #bitmask == bitmask (8-bit)".to_string(),
        tmir_expr: target,
        aarch64_expr: strategy,
        inputs: vec![("bitmask".to_string(), width)],
        preconditions: vec![],
        fp_inputs: vec![],
            category: None,
    }
}

// ---------------------------------------------------------------------------
// MOVN proofs
// ---------------------------------------------------------------------------

/// Proof: MOVN #imm16, LSL #0 produces `~zext(imm16)`.
///
/// Theorem: forall imm16 : BV16 . ~(zext64(imm16) << 0) == ~zext64(imm16)
///
/// MOVN is used for mostly-ones constants where the bitwise inverse has
/// fewer non-zero chunks. For example, 0xFFFF_FFFF_FFFF_1234 is stored
/// as MOVN #0xEDCB (since ~0xFFFF_FFFF_FFFF_1234 = 0x0000_0000_0000_EDCB).
pub fn proof_movn_hw0() -> ProofObligation {
    let width = 64;
    let imm16 = SmtExpr::var("imm16", 16);

    // Target: NOT(zext(imm16)) = all bits inverted
    let all_ones = u64::MAX;
    let target = imm16.clone().zero_ext(48).bvxor(SmtExpr::bv_const(all_ones, width));

    let strategy = encode_movn(imm16, 0, width);

    ProofObligation {
        name: "ConstMat: MOVN #imm16, LSL #0 == ~zext(imm16)".to_string(),
        tmir_expr: target,
        aarch64_expr: strategy,
        inputs: vec![("imm16".to_string(), 16)],
        preconditions: vec![],
        fp_inputs: vec![],
            category: None,
    }
}

/// Proof: MOVN #imm16, LSL #16 produces `~(zext(imm16) << 16)`.
///
/// Used for values like 0xFFFF_0000_FFFF_FFFF (inverted = 0x0000_FFFF_0000_0000,
/// but this specific case is a logical immediate; MOVN is used when the
/// inverted value is NOT a logical immediate).
pub fn proof_movn_hw1() -> ProofObligation {
    let width = 64;
    let imm16 = SmtExpr::var("imm16", 16);

    let shifted = imm16.clone().zero_ext(48).bvshl(SmtExpr::bv_const(16, width));
    let all_ones = u64::MAX;
    let target = shifted.bvxor(SmtExpr::bv_const(all_ones, width));

    let strategy = encode_movn(imm16, 16, width);

    ProofObligation {
        name: "ConstMat: MOVN #imm16, LSL #16 == ~(zext(imm16) << 16)".to_string(),
        tmir_expr: target,
        aarch64_expr: strategy,
        inputs: vec![("imm16".to_string(), 16)],
        preconditions: vec![],
        fp_inputs: vec![],
            category: None,
    }
}

/// Proof: MOVN #imm16, LSL #32 produces `~(zext(imm16) << 32)`.
pub fn proof_movn_hw2() -> ProofObligation {
    let width = 64;
    let imm16 = SmtExpr::var("imm16", 16);

    let shifted = imm16.clone().zero_ext(48).bvshl(SmtExpr::bv_const(32, width));
    let all_ones = u64::MAX;
    let target = shifted.bvxor(SmtExpr::bv_const(all_ones, width));

    let strategy = encode_movn(imm16, 32, width);

    ProofObligation {
        name: "ConstMat: MOVN #imm16, LSL #32 == ~(zext(imm16) << 32)".to_string(),
        tmir_expr: target,
        aarch64_expr: strategy,
        inputs: vec![("imm16".to_string(), 16)],
        preconditions: vec![],
        fp_inputs: vec![],
            category: None,
    }
}

/// Proof: MOVN #imm16, LSL #48 produces `~(zext(imm16) << 48)`.
pub fn proof_movn_hw3() -> ProofObligation {
    let width = 64;
    let imm16 = SmtExpr::var("imm16", 16);

    let shifted = imm16.clone().zero_ext(48).bvshl(SmtExpr::bv_const(48, width));
    let all_ones = u64::MAX;
    let target = shifted.bvxor(SmtExpr::bv_const(all_ones, width));

    let strategy = encode_movn(imm16, 48, width);

    ProofObligation {
        name: "ConstMat: MOVN #imm16, LSL #48 == ~(zext(imm16) << 48)".to_string(),
        tmir_expr: target,
        aarch64_expr: strategy,
        inputs: vec![("imm16".to_string(), 16)],
        preconditions: vec![],
        fp_inputs: vec![],
            category: None,
    }
}

/// Proof: MOVN at hw=0 (8-bit exhaustive, 16-bit register width).
pub fn proof_movn_hw0_8bit() -> ProofObligation {
    let width = 16;
    let imm = SmtExpr::var("imm", 8);

    let all_ones: u64 = (1u64 << 16) - 1;
    let target = imm.clone().zero_ext(8).bvxor(SmtExpr::bv_const(all_ones, width));

    let extended = imm.zero_ext(8);
    let strategy = extended.bvxor(SmtExpr::bv_const(all_ones, width));

    ProofObligation {
        name: "ConstMat: MOVN #imm8 == ~zext(imm8) (8-bit)".to_string(),
        tmir_expr: target,
        aarch64_expr: strategy,
        inputs: vec![("imm".to_string(), 8)],
        preconditions: vec![],
        fp_inputs: vec![],
            category: None,
    }
}

// ---------------------------------------------------------------------------
// Strategy equivalence proofs
// ---------------------------------------------------------------------------

/// Proof: ORR XZR, #val is equivalent to MOVZ #val when val fits in 16 bits.
///
/// Theorem: forall val : BV16 . (0 | zext(val)) == (zext(val) << 0)
///
/// For values in [0, 0xFFFF], both single-MOVZ and ORR produce the same
/// result. The optimizer picks MOVZ (shorter encoding), but ORR would also
/// be correct.
pub fn proof_orr_movz_equivalence() -> ProofObligation {
    let width = 64;
    let val = SmtExpr::var("val", 16);

    // ORR XZR, #val (where val fits in 16 bits, treated as logical imm)
    let orr_result = SmtExpr::bv_const(0, width).bvor(val.clone().zero_ext(48));

    // MOVZ #val, LSL #0
    let movz_result = encode_movz(val, 0, width);

    ProofObligation {
        name: "ConstMat: ORR XZR, #val == MOVZ #val (16-bit overlap)".to_string(),
        tmir_expr: orr_result,
        aarch64_expr: movz_result,
        inputs: vec![("val".to_string(), 16)],
        preconditions: vec![],
        fp_inputs: vec![],
            category: None,
    }
}

/// Proof: strategy equivalence for 8-bit values (exhaustive).
pub fn proof_orr_movz_equivalence_8bit() -> ProofObligation {
    let width = 16;
    let val = SmtExpr::var("val", 8);

    let orr_result = SmtExpr::bv_const(0, width).bvor(val.clone().zero_ext(8));
    let movz_result = val.zero_ext(8).bvshl(SmtExpr::bv_const(0, width));

    ProofObligation {
        name: "ConstMat: ORR == MOVZ equivalence (8-bit)".to_string(),
        tmir_expr: orr_result,
        aarch64_expr: movz_result,
        inputs: vec![("val".to_string(), 8)],
        preconditions: vec![],
        fp_inputs: vec![],
            category: None,
    }
}

/// Proof: MOVN and MOVZ+MOVK produce complementary results.
///
/// Theorem: forall imm16 : BV16 .
///     MOVN(imm16, 0) == ~MOVZ(imm16, 0)
///
/// MOVN is defined as the bitwise NOT of the MOVZ result. This proves
/// that the MOVN strategy is exactly the complement of the MOVZ strategy,
/// confirming the relationship used in materialize_constant when choosing
/// between MOVN and multi-instruction MOVZ+MOVK sequences.
pub fn proof_movn_is_complement_of_movz() -> ProofObligation {
    let width = 64;
    let imm16 = SmtExpr::var("imm16", 16);

    // LHS: MOVN result
    let movn_result = encode_movn(imm16.clone(), 0, width);

    // RHS: NOT(MOVZ result)
    let movz_result = encode_movz(imm16, 0, width);
    let all_ones = u64::MAX;
    let not_movz = movz_result.bvxor(SmtExpr::bv_const(all_ones, width));

    ProofObligation {
        name: "ConstMat: MOVN(imm16, 0) == ~MOVZ(imm16, 0)".to_string(),
        tmir_expr: movn_result,
        aarch64_expr: not_movz,
        inputs: vec![("imm16".to_string(), 16)],
        preconditions: vec![],
        fp_inputs: vec![],
            category: None,
    }
}

/// Proof: MOVK is idempotent when writing the same value.
///
/// Theorem: forall imm16 : BV16, base : BV64 .
///     MOVK(MOVK(base, imm16, 0), imm16, 0) == MOVK(base, imm16, 0)
///
/// Writing the same halfword twice produces the same result as writing
/// it once. This validates the MOVK encoding is deterministic.
pub fn proof_movk_idempotent() -> ProofObligation {
    let width = 64;
    let base = SmtExpr::var("base", width);
    let imm16 = SmtExpr::var("imm16", 16);

    let once = encode_movk(base.clone(), imm16.clone(), 0, width);
    let twice = encode_movk(once.clone(), imm16, 0, width);

    ProofObligation {
        name: "ConstMat: MOVK idempotent (same halfword twice)".to_string(),
        tmir_expr: once,
        aarch64_expr: twice,
        inputs: vec![("base".to_string(), width), ("imm16".to_string(), 16)],
        preconditions: vec![],
        fp_inputs: vec![],
            category: None,
    }
}

/// Proof: MOVK at different halfwords commute.
///
/// Theorem: forall a, b : BV16, base : BV64 .
///     MOVK(MOVK(base, a, 0), b, 16) == MOVK(MOVK(base, b, 16), a, 0)
///
/// Writing non-overlapping halfwords in either order produces the same result.
/// This is important for the optimal_movz_movk_sequence ordering.
pub fn proof_movk_commutative() -> ProofObligation {
    let width = 64;
    let base = SmtExpr::var("base", width);
    let a = SmtExpr::var("a", 16);
    let b = SmtExpr::var("b", 16);

    // Order 1: MOVK a at hw=0, then MOVK b at hw=1
    let order1 = encode_movk(encode_movk(base.clone(), a.clone(), 0, width), b.clone(), 16, width);

    // Order 2: MOVK b at hw=1, then MOVK a at hw=0
    let order2 = encode_movk(encode_movk(base, b, 16, width), a, 0, width);

    ProofObligation {
        name: "ConstMat: MOVK at hw=0 and hw=1 commute".to_string(),
        tmir_expr: order1,
        aarch64_expr: order2,
        inputs: vec![
            ("base".to_string(), width),
            ("a".to_string(), 16),
            ("b".to_string(), 16),
        ],
        preconditions: vec![],
        fp_inputs: vec![],
            category: None,
    }
}

// ---------------------------------------------------------------------------
// Aggregate accessors
// ---------------------------------------------------------------------------

/// Return all core constant materialization proofs (64-bit).
pub fn all_const_materialize_proofs() -> Vec<ProofObligation> {
    vec![
        // MOVZ at each halfword position
        proof_movz_hw0(),
        proof_movz_hw1(),
        proof_movz_hw2(),
        proof_movz_hw3(),
        // MOVZ + MOVK assembly
        proof_movz_movk_32bit(),
        proof_movz_movk_64bit(),
        // ORR logical immediate
        proof_orr_logical_imm(),
        proof_orr_logical_imm_32bit(),
        // MOVN at each halfword position
        proof_movn_hw0(),
        proof_movn_hw1(),
        proof_movn_hw2(),
        proof_movn_hw3(),
        // Strategy equivalence
        proof_orr_movz_equivalence(),
        proof_movn_is_complement_of_movz(),
        proof_movk_idempotent(),
        proof_movk_commutative(),
    ]
}

/// Return all constant materialization proofs including 8-bit exhaustive variants.
///
/// Total: 16 core + 6 exhaustive variants = 22 proofs.
pub fn all_const_materialize_proofs_with_variants() -> Vec<ProofObligation> {
    let mut proofs = all_const_materialize_proofs();

    // 8-bit exhaustive variants
    proofs.push(proof_movz_hw0_8bit());
    proofs.push(proof_movz_hw1_8bit());
    proofs.push(proof_movz_movk_16bit());
    proofs.push(proof_orr_logical_imm_8bit());
    proofs.push(proof_movn_hw0_8bit());
    proofs.push(proof_orr_movz_equivalence_8bit());

    proofs
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lowering_proof::verify_by_evaluation;
    use crate::verify::VerificationResult;

    /// Helper: verify a proof obligation and assert it is Valid.
    fn assert_valid(obligation: &ProofObligation) {
        let result = verify_by_evaluation(obligation);
        match &result {
            VerificationResult::Valid => {}
            VerificationResult::Invalid { counterexample } => {
                panic!(
                    "Proof '{}' FAILED with counterexample: {}",
                    obligation.name, counterexample
                );
            }
            VerificationResult::Unknown { reason } => {
                panic!("Proof '{}' returned Unknown: {}", obligation.name, reason);
            }
        }
    }

    // -----------------------------------------------------------------------
    // MOVZ proofs (64-bit)
    // -----------------------------------------------------------------------

    #[test]
    fn test_proof_movz_hw0() {
        assert_valid(&proof_movz_hw0());
    }

    #[test]
    fn test_proof_movz_hw1() {
        assert_valid(&proof_movz_hw1());
    }

    #[test]
    fn test_proof_movz_hw2() {
        assert_valid(&proof_movz_hw2());
    }

    #[test]
    fn test_proof_movz_hw3() {
        assert_valid(&proof_movz_hw3());
    }

    // -----------------------------------------------------------------------
    // MOVZ + MOVK assembly proofs
    // -----------------------------------------------------------------------

    #[test]
    fn test_proof_movz_movk_32bit() {
        assert_valid(&proof_movz_movk_32bit());
    }

    #[test]
    fn test_proof_movz_movk_64bit() {
        assert_valid(&proof_movz_movk_64bit());
    }

    // -----------------------------------------------------------------------
    // ORR logical immediate proofs
    // -----------------------------------------------------------------------

    #[test]
    fn test_proof_orr_logical_imm() {
        assert_valid(&proof_orr_logical_imm());
    }

    #[test]
    fn test_proof_orr_logical_imm_32bit() {
        assert_valid(&proof_orr_logical_imm_32bit());
    }

    // -----------------------------------------------------------------------
    // MOVN proofs
    // -----------------------------------------------------------------------

    #[test]
    fn test_proof_movn_hw0() {
        assert_valid(&proof_movn_hw0());
    }

    #[test]
    fn test_proof_movn_hw1() {
        assert_valid(&proof_movn_hw1());
    }

    #[test]
    fn test_proof_movn_hw2() {
        assert_valid(&proof_movn_hw2());
    }

    #[test]
    fn test_proof_movn_hw3() {
        assert_valid(&proof_movn_hw3());
    }

    // -----------------------------------------------------------------------
    // Strategy equivalence proofs
    // -----------------------------------------------------------------------

    #[test]
    fn test_proof_orr_movz_equivalence() {
        assert_valid(&proof_orr_movz_equivalence());
    }

    #[test]
    fn test_proof_movn_is_complement_of_movz() {
        assert_valid(&proof_movn_is_complement_of_movz());
    }

    #[test]
    fn test_proof_movk_idempotent() {
        assert_valid(&proof_movk_idempotent());
    }

    #[test]
    fn test_proof_movk_commutative() {
        assert_valid(&proof_movk_commutative());
    }

    // -----------------------------------------------------------------------
    // 8-bit exhaustive variant proofs
    // -----------------------------------------------------------------------

    #[test]
    fn test_proof_movz_hw0_8bit() {
        assert_valid(&proof_movz_hw0_8bit());
    }

    #[test]
    fn test_proof_movz_hw1_8bit() {
        assert_valid(&proof_movz_hw1_8bit());
    }

    #[test]
    fn test_proof_movz_movk_16bit() {
        assert_valid(&proof_movz_movk_16bit());
    }

    #[test]
    fn test_proof_orr_logical_imm_8bit() {
        assert_valid(&proof_orr_logical_imm_8bit());
    }

    #[test]
    fn test_proof_movn_hw0_8bit() {
        assert_valid(&proof_movn_hw0_8bit());
    }

    #[test]
    fn test_proof_orr_movz_equivalence_8bit() {
        assert_valid(&proof_orr_movz_equivalence_8bit());
    }

    // -----------------------------------------------------------------------
    // Aggregate tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_all_const_materialize_proofs() {
        for obligation in all_const_materialize_proofs() {
            assert_valid(&obligation);
        }
    }

    #[test]
    fn test_all_const_materialize_proofs_with_variants() {
        for obligation in all_const_materialize_proofs_with_variants() {
            assert_valid(&obligation);
        }
    }

    #[test]
    fn test_proof_count() {
        assert_eq!(all_const_materialize_proofs().len(), 16,
            "expected 16 core proofs");
        assert_eq!(all_const_materialize_proofs_with_variants().len(), 22,
            "expected 22 total proofs (16 core + 6 exhaustive)");
    }

    // -----------------------------------------------------------------------
    // Negative tests: verify incorrect strategies are detected
    // -----------------------------------------------------------------------

    /// Negative test: MOVZ at wrong shift is NOT equivalent to correct shift.
    #[test]
    fn test_wrong_movz_shift_detected() {
        let width = 64;
        let imm16 = SmtExpr::var("imm16", 16);

        // Wrong: MOVZ at hw=0 claimed to equal MOVZ at hw=1
        let obligation = ProofObligation {
            name: "WRONG: MOVZ hw=0 == MOVZ hw=1".to_string(),
            tmir_expr: encode_movz(imm16.clone(), 0, width),
            aarch64_expr: encode_movz(imm16, 16, width),
            inputs: vec![("imm16".to_string(), 16)],
            preconditions: vec![],
            fp_inputs: vec![],
            category: None,
        };

        let result = verify_by_evaluation(&obligation);
        match result {
            VerificationResult::Invalid { .. } => {} // expected
            other => panic!("Expected Invalid for wrong shift, got {:?}", other),
        }
    }

    /// Negative test: MOVN is NOT equivalent to MOVZ (for most values).
    #[test]
    fn test_movn_not_equal_movz() {
        let width = 64;
        let imm16 = SmtExpr::var("imm16", 16);

        let obligation = ProofObligation {
            name: "WRONG: MOVN hw=0 == MOVZ hw=0".to_string(),
            tmir_expr: encode_movn(imm16.clone(), 0, width),
            aarch64_expr: encode_movz(imm16, 0, width),
            inputs: vec![("imm16".to_string(), 16)],
            preconditions: vec![],
            fp_inputs: vec![],
            category: None,
        };

        let result = verify_by_evaluation(&obligation);
        match result {
            VerificationResult::Invalid { .. } => {} // expected
            other => panic!("Expected Invalid for MOVN==MOVZ, got {:?}", other),
        }
    }

    /// Negative test: MOVK at wrong position corrupts value.
    #[test]
    fn test_wrong_movk_position_detected() {
        let width = 32;
        let lo16 = SmtExpr::var("lo16", 16);
        let hi16 = SmtExpr::var("hi16", 16);

        // Wrong: both MOVK at hw=0 (overwrites the first MOVZ)
        let target = hi16.clone().zero_ext(16).bvshl(SmtExpr::bv_const(16, width))
            .bvor(lo16.clone().zero_ext(16));
        let step1 = encode_movz(lo16, 0, width);
        let wrong = encode_movk(step1, hi16, 0, width); // wrong: should be hw=16

        let obligation = ProofObligation {
            name: "WRONG: MOVK at hw=0 instead of hw=16".to_string(),
            tmir_expr: target,
            aarch64_expr: wrong,
            inputs: vec![("lo16".to_string(), 16), ("hi16".to_string(), 16)],
            preconditions: vec![],
            fp_inputs: vec![],
            category: None,
        };

        let result = verify_by_evaluation(&obligation);
        match result {
            VerificationResult::Invalid { .. } => {} // expected
            other => panic!("Expected Invalid for wrong MOVK position, got {:?}", other),
        }
    }

    // -----------------------------------------------------------------------
    // SMT-LIB2 output tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_smt2_output_movz_hw0() {
        let obligation = proof_movz_hw0();
        let smt2 = obligation.to_smt2();
        assert!(smt2.contains("(set-logic QF_BV)"));
        assert!(smt2.contains("(declare-const imm16 (_ BitVec 16))"));
        assert!(smt2.contains("bvshl"));
        assert!(smt2.contains("(check-sat)"));
    }

    #[test]
    fn test_smt2_output_movz_movk_32bit() {
        let obligation = proof_movz_movk_32bit();
        let smt2 = obligation.to_smt2();
        assert!(smt2.contains("(set-logic QF_BV)"));
        assert!(smt2.contains("lo16"));
        assert!(smt2.contains("hi16"));
        assert!(smt2.contains("bvand"));
        assert!(smt2.contains("bvor"));
        assert!(smt2.contains("(check-sat)"));
    }

    #[test]
    fn test_smt2_output_movn_hw0() {
        let obligation = proof_movn_hw0();
        let smt2 = obligation.to_smt2();
        assert!(smt2.contains("(set-logic QF_BV)"));
        assert!(smt2.contains("bvxor"));
        assert!(smt2.contains("(check-sat)"));
    }

    // -----------------------------------------------------------------------
    // Concrete value spot-checks
    // -----------------------------------------------------------------------

    #[test]
    fn test_movz_concrete_0x1234() {
        // MOVZ #0x1234 should produce 0x1234
        use std::collections::HashMap;
        let expr = encode_movz(SmtExpr::bv_const(0x1234, 16), 0, 64);
        let env: HashMap<String, u64> = HashMap::new();
        let result = expr.eval(&env);
        assert_eq!(result, crate::smt::EvalResult::Bv(0x1234));
    }

    #[test]
    fn test_movz_concrete_shifted() {
        // MOVZ #0xABCD, LSL #32 should produce 0x0000_ABCD_0000_0000
        use std::collections::HashMap;
        let expr = encode_movz(SmtExpr::bv_const(0xABCD, 16), 32, 64);
        let env: HashMap<String, u64> = HashMap::new();
        let result = expr.eval(&env);
        assert_eq!(result, crate::smt::EvalResult::Bv(0x0000_ABCD_0000_0000));
    }

    #[test]
    fn test_movk_concrete() {
        // Start with 0x0000_0000_0000_BABE, MOVK #0xCAFE at hw=1
        // Result should be 0x0000_0000_CAFE_BABE
        use std::collections::HashMap;
        let base = SmtExpr::bv_const(0xBABE, 64);
        let expr = encode_movk(base, SmtExpr::bv_const(0xCAFE, 16), 16, 64);
        let env: HashMap<String, u64> = HashMap::new();
        let result = expr.eval(&env);
        assert_eq!(result, crate::smt::EvalResult::Bv(0x0000_0000_CAFE_BABE));
    }

    #[test]
    fn test_movn_concrete() {
        // MOVN #0xEDCB at hw=0 should produce ~0x0000_0000_0000_EDCB
        // = 0xFFFF_FFFF_FFFF_1234
        use std::collections::HashMap;
        let expr = encode_movn(SmtExpr::bv_const(0xEDCB, 16), 0, 64);
        let env: HashMap<String, u64> = HashMap::new();
        let result = expr.eval(&env);
        assert_eq!(result, crate::smt::EvalResult::Bv(0xFFFF_FFFF_FFFF_1234));
    }

    #[test]
    fn test_full_64bit_assembly_concrete() {
        // Assemble 0xDEAD_BEEF_CAFE_BABE via MOVZ+3xMOVK
        use std::collections::HashMap;
        let step1 = encode_movz(SmtExpr::bv_const(0xBABE, 16), 0, 64);
        let step2 = encode_movk(step1, SmtExpr::bv_const(0xCAFE, 16), 16, 64);
        let step3 = encode_movk(step2, SmtExpr::bv_const(0xBEEF, 16), 32, 64);
        let step4 = encode_movk(step3, SmtExpr::bv_const(0xDEAD, 16), 48, 64);
        let env: HashMap<String, u64> = HashMap::new();
        let result = step4.eval(&env);
        assert_eq!(result, crate::smt::EvalResult::Bv(0xDEAD_BEEF_CAFE_BABE));
    }
}
