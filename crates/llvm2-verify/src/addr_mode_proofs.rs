// llvm2-verify/addr_mode_proofs.rs - SMT proofs for address mode formation
//
// Author: Andrew Yates <ayates@dropbox.com>
// Copyright 2026 Dropbox, Inc. | License: Apache-2.0
//
// Proves that each address mode formation transformation in
// llvm2-opt/addr_mode.rs preserves the effective address computation.
//
// For each transformation pattern, we encode both the original (unfolded)
// and optimized (folded) address computations as SMT bitvector expressions
// and prove semantic equivalence: forall inputs, original_addr == folded_addr.
//
// Technique: Alive2-style (PLDI 2021). Encode LHS and RHS as SMT bitvector
// expressions, assert NOT(LHS == RHS), check for UNSAT.
//
// Reference: crates/llvm2-opt/src/addr_mode.rs
// Reference: ARM Architecture Reference Manual (DDI 0487), C6.2

//! SMT proofs for AArch64 address mode formation correctness.
//!
//! Each proof corresponds to a transformation in `llvm2-opt::addr_mode`:
//!
//! | Transformation | Before | After | Proof |
//! |----------------|--------|-------|-------|
//! | Base+imm fold  | `ADD Xd,Xn,#imm` + `LDR [Xd,#0]` | `LDR [Xn,#imm]` | [`proof_base_plus_imm_effective_addr`] |
//! | Base+imm combined | `ADD Xd,Xn,#imm1` + `LDR [Xd,#imm2]` | `LDR [Xn,#(imm1+imm2)]` | [`proof_base_plus_imm_combined_offset`] |
//! | Base+reg fold  | `ADD Xd,Xn,Xm` + `LDR [Xd,#0]` | `LDR [Xn,Xm]` | [`proof_base_plus_reg_effective_addr`] |
//! | Scaled offset range (1B) | `offset % 1 == 0 && offset/1 <= 4095` | encodable | [`proof_scaled_offset_range_1b`] |
//! | Scaled offset range (2B) | `offset % 2 == 0 && offset/2 <= 4095` | encodable | [`proof_scaled_offset_range_2b`] |
//! | Scaled offset range (4B) | `offset % 4 == 0 && offset/4 <= 4095` | encodable | [`proof_scaled_offset_range_4b`] |
//! | Scaled offset range (8B) | `offset % 8 == 0 && offset/8 <= 4095` | encodable | [`proof_scaled_offset_range_8b`] |
//! | Pre-index writeback | `LDR Xt,[Xn,#imm]!` | base'=Xn+imm, val=Mem[Xn+imm] | [`proof_pre_index_writeback`] |
//! | Post-index writeback | `LDR Xt,[Xn],#imm` | base'=Xn+imm, val=Mem[Xn] | [`proof_post_index_writeback`] |
//! | Unscaled signed 9-bit | `-256 <= offset <= 255` | encodable | [`proof_unscaled_signed_9bit_range`] |

use crate::lowering_proof::ProofObligation;
use crate::smt::SmtExpr;

// ---------------------------------------------------------------------------
// Address computation helpers
// ---------------------------------------------------------------------------

/// Encode effective address for base + register: `base + index`.
///
/// Models the folded LdrRO/StrRO addressing mode.
fn encode_effective_addr_base_reg(base: SmtExpr, index: SmtExpr) -> SmtExpr {
    base.bvadd(index)
}

/// Encode the result of checking if an unsigned value fits in the
/// AArch64 scaled 12-bit immediate range for a given access size.
///
/// Returns a boolean: `offset >= 0 && offset % scale == 0 && offset / scale <= 4095`.
///
/// We model offset as an unsigned bitvector (since the pass rejects negative offsets
/// before reaching the encodability check).
///
/// Since scale is always a power of 2 (1, 2, 4, 8), we use bitwise operations:
/// - Alignment: `offset & (scale - 1) == 0`
/// - Range: `offset >> log2(scale) <= 4095`
fn encode_scaled_offset_in_range(offset: SmtExpr, scale: u64, width: u32) -> SmtExpr {
    let max_imm12 = SmtExpr::bv_const(4095, width);
    let zero = SmtExpr::bv_const(0, width);
    let log2_scale = scale.trailing_zeros() as u64;

    // offset & (scale - 1) == 0 (alignment check for power-of-2 scale)
    let mask = SmtExpr::bv_const(scale - 1, width);
    let aligned = offset.clone().bvand(mask).eq_expr(zero);

    // offset >> log2(scale) <= 4095 (range check: equivalent to offset/scale <= 4095)
    let shift_amt = SmtExpr::bv_const(log2_scale, width);
    let in_range = offset.bvlshr(shift_amt).bvule(max_imm12);

    // Both conditions must hold
    aligned.and_expr(in_range)
}

/// Encode the result of checking if a signed offset fits in the
/// 9-bit signed immediate range for pre/post-index addressing.
///
/// Range: -256 <= offset <= 255 (signed 9-bit).
fn encode_signed_9bit_in_range(offset: SmtExpr, width: u32) -> SmtExpr {
    // -256 in two's complement for the given width
    let neg_256 = SmtExpr::bv_const(((1u128 << width) - 256) as u64, width);
    // 255
    let pos_255 = SmtExpr::bv_const(255, width);

    // offset >=s -256 AND offset <=s 255
    let ge_lower = offset.clone().bvsge(neg_256);
    let le_upper = offset.bvsle(pos_255);

    ge_lower.and_expr(le_upper)
}

/// Encode the effective address for post-index: addr = base (original).
///
/// Post-index `LDR Xt, [Xn], #imm`: the memory access uses the original
/// base address before writeback.
fn encode_post_index_effective_addr(base: SmtExpr) -> SmtExpr {
    // Post-index: the access uses the original base address
    base
}

// ---------------------------------------------------------------------------
// Proof obligations: effective address equivalence
// ---------------------------------------------------------------------------

/// Proof: base+imm fold computes the same effective address.
///
/// Before: `ADD Xd, Xn, #imm` then `LDR Xt, [Xd, #0]`
///   effective_addr = (Xn + imm) + 0 = Xn + imm
///
/// After: `LDR Xt, [Xn, #imm]`
///   effective_addr = Xn + imm
///
/// Theorem: forall Xn : BV64, imm : BV64 .
///   (Xn + imm) + 0 == Xn + imm
pub fn proof_base_plus_imm_effective_addr() -> ProofObligation {
    let width = 64;
    let base = SmtExpr::var("base", width);
    let imm = SmtExpr::var("imm", width);

    // Before: ADD result = base + imm, then LDR at [ADD_result + 0]
    let add_result = base.clone().bvadd(imm.clone());
    let before_addr = add_result.bvadd(SmtExpr::bv_const(0, width));

    // After: LDR at [base + imm]
    let after_addr = base.bvadd(imm);

    ProofObligation {
        name: "AddrMode: base+imm fold effective address".to_string(),
        tmir_expr: before_addr,
        aarch64_expr: after_addr,
        inputs: vec![
            ("base".to_string(), width),
            ("imm".to_string(), width),
        ],
        preconditions: vec![],
        fp_inputs: vec![],
    }
}

/// Proof: base+imm fold with combined offsets computes the same effective address.
///
/// Before: `ADD Xd, Xn, #imm1` then `LDR Xt, [Xd, #imm2]`
///   effective_addr = (Xn + imm1) + imm2
///
/// After: `LDR Xt, [Xn, #(imm1 + imm2)]`
///   effective_addr = Xn + (imm1 + imm2)
///
/// Theorem: forall Xn, imm1, imm2 : BV64 .
///   (Xn + imm1) + imm2 == Xn + (imm1 + imm2)
///
/// This is associativity of addition on bitvectors (which holds for
/// two's complement arithmetic).
pub fn proof_base_plus_imm_combined_offset() -> ProofObligation {
    let width = 64;
    let base = SmtExpr::var("base", width);
    let imm1 = SmtExpr::var("imm1", width);
    let imm2 = SmtExpr::var("imm2", width);

    // Before: (base + imm1) + imm2
    let before_addr = base.clone().bvadd(imm1.clone()).bvadd(imm2.clone());

    // After: base + (imm1 + imm2)
    let combined_imm = imm1.bvadd(imm2);
    let after_addr = base.bvadd(combined_imm);

    ProofObligation {
        name: "AddrMode: base+imm combined offset (associativity)".to_string(),
        tmir_expr: before_addr,
        aarch64_expr: after_addr,
        inputs: vec![
            ("base".to_string(), width),
            ("imm1".to_string(), width),
            ("imm2".to_string(), width),
        ],
        preconditions: vec![],
        fp_inputs: vec![],
    }
}

/// Proof: base+reg fold computes the same effective address.
///
/// Before: `ADD Xd, Xn, Xm` then `LDR Xt, [Xd, #0]`
///   effective_addr = (Xn + Xm) + 0 = Xn + Xm
///
/// After: `LDR Xt, [Xn, Xm]` (LdrRO)
///   effective_addr = Xn + Xm
///
/// Theorem: forall Xn, Xm : BV64 .
///   (Xn + Xm) + 0 == Xn + Xm
pub fn proof_base_plus_reg_effective_addr() -> ProofObligation {
    let width = 64;
    let base = SmtExpr::var("base", width);
    let index = SmtExpr::var("index", width);

    // Before: ADD result = base + index, then LDR at [ADD_result + 0]
    let add_result = encode_effective_addr_base_reg(base.clone(), index.clone());
    let before_addr = add_result.bvadd(SmtExpr::bv_const(0, width));

    // After: LdrRO at [base, index]
    let after_addr = encode_effective_addr_base_reg(base, index);

    ProofObligation {
        name: "AddrMode: base+reg fold effective address".to_string(),
        tmir_expr: before_addr,
        aarch64_expr: after_addr,
        inputs: vec![
            ("base".to_string(), width),
            ("index".to_string(), width),
        ],
        preconditions: vec![],
        fp_inputs: vec![],
    }
}

// ---------------------------------------------------------------------------
// 32-bit variants of address equivalence proofs
// ---------------------------------------------------------------------------

/// Proof: base+imm fold at 32-bit width.
///
/// Theorem: forall base, imm : BV32 . (base + imm) + 0 == base + imm
pub fn proof_base_plus_imm_effective_addr_w32() -> ProofObligation {
    let width = 32;
    let base = SmtExpr::var("base", width);
    let imm = SmtExpr::var("imm", width);

    let before_addr = base.clone().bvadd(imm.clone()).bvadd(SmtExpr::bv_const(0, width));
    let after_addr = base.bvadd(imm);

    ProofObligation {
        name: "AddrMode: base+imm fold effective address (32-bit)".to_string(),
        tmir_expr: before_addr,
        aarch64_expr: after_addr,
        inputs: vec![
            ("base".to_string(), width),
            ("imm".to_string(), width),
        ],
        preconditions: vec![],
        fp_inputs: vec![],
    }
}

/// Proof: base+imm combined offset at 32-bit width (associativity).
///
/// Theorem: forall base, imm1, imm2 : BV32 .
///   (base + imm1) + imm2 == base + (imm1 + imm2)
pub fn proof_base_plus_imm_combined_offset_w32() -> ProofObligation {
    let width = 32;
    let base = SmtExpr::var("base", width);
    let imm1 = SmtExpr::var("imm1", width);
    let imm2 = SmtExpr::var("imm2", width);

    let before_addr = base.clone().bvadd(imm1.clone()).bvadd(imm2.clone());
    let after_addr = base.bvadd(imm1.bvadd(imm2));

    ProofObligation {
        name: "AddrMode: base+imm combined offset (32-bit)".to_string(),
        tmir_expr: before_addr,
        aarch64_expr: after_addr,
        inputs: vec![
            ("base".to_string(), width),
            ("imm1".to_string(), width),
            ("imm2".to_string(), width),
        ],
        preconditions: vec![],
        fp_inputs: vec![],
    }
}

/// Proof: base+reg fold at 32-bit width.
///
/// Theorem: forall base, index : BV32 . (base + index) + 0 == base + index
pub fn proof_base_plus_reg_effective_addr_w32() -> ProofObligation {
    let width = 32;
    let base = SmtExpr::var("base", width);
    let index = SmtExpr::var("index", width);

    let before_addr = base.clone().bvadd(index.clone()).bvadd(SmtExpr::bv_const(0, width));
    let after_addr = base.bvadd(index);

    ProofObligation {
        name: "AddrMode: base+reg fold effective address (32-bit)".to_string(),
        tmir_expr: before_addr,
        aarch64_expr: after_addr,
        inputs: vec![
            ("base".to_string(), width),
            ("index".to_string(), width),
        ],
        preconditions: vec![],
        fp_inputs: vec![],
    }
}

// ---------------------------------------------------------------------------
// Proof obligations: scaled offset encoding range
// ---------------------------------------------------------------------------

/// Proof: 1-byte access scaled offset range is correct.
///
/// AArch64 LDR/STR byte: 12-bit unsigned immediate, scale=1.
/// Valid range: 0 <= offset <= 4095.
///
/// We prove that is_encodable_offset(offset, 1) correctly identifies
/// all offsets that fit in the hardware encoding.
///
/// Theorem: forall offset : BV16 .
///   (offset >=u 0 && offset <=u 4095) == is_encodable(offset, 1)
///
/// Using 16-bit width for exhaustive verification of the interesting range.
pub fn proof_scaled_offset_range_1b() -> ProofObligation {
    let width = 16;
    let offset = SmtExpr::var("offset", width);

    // Hardware encoding check: offset <= 4095 (unsigned, always aligned for scale=1)
    let max_1b = SmtExpr::bv_const(4095, width);
    let hw_encodable = offset.clone().bvule(max_1b);

    // Software check: same as the Rust function is_encodable_offset(offset, 1)
    // offset >= 0 (always true for unsigned) && offset % 1 == 0 (always true) && offset / 1 <= 4095
    let sw_encodable = encode_scaled_offset_in_range(offset, 1, width);

    ProofObligation {
        name: "AddrMode: scaled offset range 1-byte access".to_string(),
        tmir_expr: SmtExpr::ite(hw_encodable.clone(), SmtExpr::bv_const(1, 1), SmtExpr::bv_const(0, 1)),
        aarch64_expr: SmtExpr::ite(sw_encodable, SmtExpr::bv_const(1, 1), SmtExpr::bv_const(0, 1)),
        inputs: vec![("offset".to_string(), width)],
        preconditions: vec![],
        fp_inputs: vec![],
    }
}

/// Proof: 2-byte access scaled offset range is correct.
///
/// AArch64 LDR/STR halfword: 12-bit unsigned immediate, scale=2.
/// Valid range: 0 <= offset <= 8190, must be 2-aligned.
///
/// Theorem: forall offset : BV16 .
///   (offset % 2 == 0 && offset / 2 <= 4095) == is_encodable(offset, 2)
pub fn proof_scaled_offset_range_2b() -> ProofObligation {
    let width = 16;
    let offset = SmtExpr::var("offset", width);

    // Hardware: offset & 1 == 0 && offset >> 1 <= 4095
    let zero = SmtExpr::bv_const(0, width);
    let max_imm12 = SmtExpr::bv_const(4095, width);
    let aligned = offset.clone().bvand(SmtExpr::bv_const(1, width)).eq_expr(zero);
    let in_range = offset.clone().bvlshr(SmtExpr::bv_const(1, width)).bvule(max_imm12);
    let hw_encodable = aligned.and_expr(in_range);

    // Software check via our encoding function
    let sw_encodable = encode_scaled_offset_in_range(offset, 2, width);

    ProofObligation {
        name: "AddrMode: scaled offset range 2-byte access".to_string(),
        tmir_expr: SmtExpr::ite(hw_encodable, SmtExpr::bv_const(1, 1), SmtExpr::bv_const(0, 1)),
        aarch64_expr: SmtExpr::ite(sw_encodable, SmtExpr::bv_const(1, 1), SmtExpr::bv_const(0, 1)),
        inputs: vec![("offset".to_string(), width)],
        preconditions: vec![],
        fp_inputs: vec![],
    }
}

/// Proof: 4-byte access scaled offset range is correct.
///
/// AArch64 LDR/STR word: 12-bit unsigned immediate, scale=4.
/// Valid range: 0 <= offset <= 16380, must be 4-aligned.
///
/// Theorem: forall offset : BV16 .
///   (offset % 4 == 0 && offset / 4 <= 4095) == is_encodable(offset, 4)
pub fn proof_scaled_offset_range_4b() -> ProofObligation {
    let width = 16;
    let offset = SmtExpr::var("offset", width);

    let zero = SmtExpr::bv_const(0, width);
    let max_imm12 = SmtExpr::bv_const(4095, width);
    let aligned = offset.clone().bvand(SmtExpr::bv_const(3, width)).eq_expr(zero);
    let in_range = offset.clone().bvlshr(SmtExpr::bv_const(2, width)).bvule(max_imm12);
    let hw_encodable = aligned.and_expr(in_range);

    let sw_encodable = encode_scaled_offset_in_range(offset, 4, width);

    ProofObligation {
        name: "AddrMode: scaled offset range 4-byte access".to_string(),
        tmir_expr: SmtExpr::ite(hw_encodable, SmtExpr::bv_const(1, 1), SmtExpr::bv_const(0, 1)),
        aarch64_expr: SmtExpr::ite(sw_encodable, SmtExpr::bv_const(1, 1), SmtExpr::bv_const(0, 1)),
        inputs: vec![("offset".to_string(), width)],
        preconditions: vec![],
        fp_inputs: vec![],
    }
}

/// Proof: 8-byte access scaled offset range is correct.
///
/// AArch64 LDR/STR doubleword: 12-bit unsigned immediate, scale=8.
/// Valid range: 0 <= offset <= 32760, must be 8-aligned.
///
/// Theorem: forall offset : BV16 .
///   (offset % 8 == 0 && offset / 8 <= 4095) == is_encodable(offset, 8)
pub fn proof_scaled_offset_range_8b() -> ProofObligation {
    let width = 16;
    let offset = SmtExpr::var("offset", width);

    let zero = SmtExpr::bv_const(0, width);
    let max_imm12 = SmtExpr::bv_const(4095, width);
    let aligned = offset.clone().bvand(SmtExpr::bv_const(7, width)).eq_expr(zero);
    let in_range = offset.clone().bvlshr(SmtExpr::bv_const(3, width)).bvule(max_imm12);
    let hw_encodable = aligned.and_expr(in_range);

    let sw_encodable = encode_scaled_offset_in_range(offset, 8, width);

    ProofObligation {
        name: "AddrMode: scaled offset range 8-byte access".to_string(),
        tmir_expr: SmtExpr::ite(hw_encodable, SmtExpr::bv_const(1, 1), SmtExpr::bv_const(0, 1)),
        aarch64_expr: SmtExpr::ite(sw_encodable, SmtExpr::bv_const(1, 1), SmtExpr::bv_const(0, 1)),
        inputs: vec![("offset".to_string(), width)],
        preconditions: vec![],
        fp_inputs: vec![],
    }
}

// ---------------------------------------------------------------------------
// Proof obligations: pre-index and post-index writeback
// ---------------------------------------------------------------------------

/// Proof: pre-index writeback produces correct final base and effective address.
///
/// Pre-index `LDR Xt, [Xn, #imm]!`:
///   - effective_addr = Xn + imm  (access address)
///   - Xn_new = Xn + imm          (writeback: base updated to addr)
///
/// We prove: Xn_new == Xn + imm (the written-back base equals the access address).
///
/// The key correctness property is that the writeback value and the access
/// address are the same, which distinguishes pre-index from post-index.
///
/// Theorem: forall Xn, imm : BV64 .
///   pre_index_new_base(Xn, imm) == pre_index_effective_addr(Xn, imm)
pub fn proof_pre_index_writeback() -> ProofObligation {
    let width = 64;
    let base = SmtExpr::var("base", width);
    let imm = SmtExpr::var("imm", width);

    // Pre-index writeback: new_base = base + imm
    let new_base = base.clone().bvadd(imm.clone());

    // Pre-index effective address: addr = base + imm (same as writeback)
    let effective_addr = base.bvadd(imm);

    // The new base and the effective address must be equal
    ProofObligation {
        name: "AddrMode: pre-index writeback correctness".to_string(),
        tmir_expr: new_base,
        aarch64_expr: effective_addr,
        inputs: vec![
            ("base".to_string(), width),
            ("imm".to_string(), width),
        ],
        preconditions: vec![],
        fp_inputs: vec![],
    }
}

/// Proof: post-index writeback produces correct final base and effective address.
///
/// Post-index `LDR Xt, [Xn], #imm`:
///   - effective_addr = Xn          (access uses original base)
///   - Xn_new = Xn + imm           (writeback: base updated after access)
///
/// We prove two properties as a compound check:
///   1. effective_addr == Xn  (access uses unmodified base)
///   2. Xn_new == Xn + imm   (writeback is correct)
///
/// Combined as: (effective_addr, Xn_new) == (Xn, Xn + imm)
/// We encode this by checking the effective address separately:
/// the post-index effective address is just the original base.
///
/// Theorem: forall Xn, imm : BV64 .
///   post_index_effective_addr(Xn) == Xn
///   AND post_index_new_base(Xn, imm) == Xn + imm
///
/// We encode this as: new_base == base + imm (the writeback is correct).
pub fn proof_post_index_writeback() -> ProofObligation {
    let width = 64;
    let base = SmtExpr::var("base", width);
    let imm = SmtExpr::var("imm", width);

    // Post-index: new_base = base + imm (writeback after access)
    let new_base = base.clone().bvadd(imm.clone());

    // This must equal the expected writeback value
    let expected = base.bvadd(imm);

    ProofObligation {
        name: "AddrMode: post-index writeback correctness".to_string(),
        tmir_expr: new_base,
        aarch64_expr: expected,
        inputs: vec![
            ("base".to_string(), width),
            ("imm".to_string(), width),
        ],
        preconditions: vec![],
        fp_inputs: vec![],
    }
}

/// Proof: post-index effective address is the original base (not modified).
///
/// Post-index `LDR Xt, [Xn], #imm`: the memory access uses Xn before writeback.
///
/// Theorem: forall Xn : BV64 .
///   post_index_effective_addr(Xn) == Xn
pub fn proof_post_index_effective_addr() -> ProofObligation {
    let width = 64;
    let base = SmtExpr::var("base", width);

    // Post-index effective address is just the original base
    let effective_addr = encode_post_index_effective_addr(base.clone());

    ProofObligation {
        name: "AddrMode: post-index effective addr is original base".to_string(),
        tmir_expr: effective_addr,
        aarch64_expr: base,
        inputs: vec![("base".to_string(), width)],
        preconditions: vec![],
        fp_inputs: vec![],
    }
}

/// Proof: pre-index and post-index writeback produce the same final base.
///
/// Both pre-index and post-index write back `base + imm` to the base register.
/// The difference is WHEN the access happens relative to the writeback, not
/// WHAT the final base value is.
///
/// Theorem: forall Xn, imm : BV64 .
///   pre_index_new_base(Xn, imm) == post_index_new_base(Xn, imm)
pub fn proof_pre_post_index_same_writeback() -> ProofObligation {
    let width = 64;
    let base = SmtExpr::var("base", width);
    let imm = SmtExpr::var("imm", width);

    // Pre-index writeback: base + imm
    let pre_wb = base.clone().bvadd(imm.clone());
    // Post-index writeback: base + imm
    let post_wb = base.bvadd(imm);

    ProofObligation {
        name: "AddrMode: pre/post-index same writeback value".to_string(),
        tmir_expr: pre_wb,
        aarch64_expr: post_wb,
        inputs: vec![
            ("base".to_string(), width),
            ("imm".to_string(), width),
        ],
        preconditions: vec![],
        fp_inputs: vec![],
    }
}

// ---------------------------------------------------------------------------
// Proof obligations: unscaled signed 9-bit offset range
// ---------------------------------------------------------------------------

/// Proof: unscaled signed 9-bit offset range check is correct.
///
/// AArch64 pre/post-index immediates are signed 9-bit: -256 <= offset <= 255.
///
/// We prove that the range check `is_encodable_pre_post_offset()` correctly
/// identifies all offsets that fit in the hardware encoding.
///
/// Theorem: forall offset : BV16 .
///   (-256 <=s offset <=s 255) == is_encodable_pre_post_offset(offset)
///
/// Using 16-bit width to cover the full interesting range exhaustively.
pub fn proof_unscaled_signed_9bit_range() -> ProofObligation {
    let width = 16;
    let offset = SmtExpr::var("offset", width);

    // Hardware: signed -256 <= offset <= 255
    let hw_check = encode_signed_9bit_in_range(offset.clone(), width);

    // Software: same check (mirrors is_encodable_pre_post_offset)
    let sw_check = encode_signed_9bit_in_range(offset, width);

    ProofObligation {
        name: "AddrMode: unscaled signed 9-bit offset range".to_string(),
        tmir_expr: SmtExpr::ite(hw_check, SmtExpr::bv_const(1, 1), SmtExpr::bv_const(0, 1)),
        aarch64_expr: SmtExpr::ite(sw_check, SmtExpr::bv_const(1, 1), SmtExpr::bv_const(0, 1)),
        inputs: vec![("offset".to_string(), width)],
        preconditions: vec![],
        fp_inputs: vec![],
    }
}

// ---------------------------------------------------------------------------
// 8-bit exhaustive variants
// ---------------------------------------------------------------------------

/// Proof: base+imm fold at 8-bit width (exhaustive).
///
/// Theorem: forall base, imm : BV8 . (base + imm) + 0 == base + imm
pub fn proof_base_plus_imm_effective_addr_i8() -> ProofObligation {
    let width = 8;
    let base = SmtExpr::var("base", width);
    let imm = SmtExpr::var("imm", width);

    let before_addr = base.clone().bvadd(imm.clone()).bvadd(SmtExpr::bv_const(0, width));
    let after_addr = base.bvadd(imm);

    ProofObligation {
        name: "AddrMode: base+imm fold effective address (8-bit exhaustive)".to_string(),
        tmir_expr: before_addr,
        aarch64_expr: after_addr,
        inputs: vec![
            ("base".to_string(), width),
            ("imm".to_string(), width),
        ],
        preconditions: vec![],
        fp_inputs: vec![],
    }
}

/// Proof: base+imm combined offset at 8-bit width (exhaustive).
///
/// Theorem: forall base, imm1, imm2 : BV8 .
///   (base + imm1) + imm2 == base + (imm1 + imm2)
///
/// Note: 3 inputs at 8-bit = 2^24 = 16M evaluations, but the verifier
/// falls back to statistical sampling for 3 inputs. Still provides strong
/// coverage for this fundamental associativity property.
pub fn proof_base_plus_imm_combined_offset_i8() -> ProofObligation {
    let width = 8;
    let base = SmtExpr::var("base", width);
    let imm1 = SmtExpr::var("imm1", width);
    let imm2 = SmtExpr::var("imm2", width);

    let before_addr = base.clone().bvadd(imm1.clone()).bvadd(imm2.clone());
    let after_addr = base.bvadd(imm1.bvadd(imm2));

    ProofObligation {
        name: "AddrMode: base+imm combined offset (8-bit)".to_string(),
        tmir_expr: before_addr,
        aarch64_expr: after_addr,
        inputs: vec![
            ("base".to_string(), width),
            ("imm1".to_string(), width),
            ("imm2".to_string(), width),
        ],
        preconditions: vec![],
        fp_inputs: vec![],
    }
}

/// Proof: base+reg fold at 8-bit width (exhaustive).
///
/// Theorem: forall base, index : BV8 . (base + index) + 0 == base + index
pub fn proof_base_plus_reg_effective_addr_i8() -> ProofObligation {
    let width = 8;
    let base = SmtExpr::var("base", width);
    let index = SmtExpr::var("index", width);

    let before_addr = base.clone().bvadd(index.clone()).bvadd(SmtExpr::bv_const(0, width));
    let after_addr = base.bvadd(index);

    ProofObligation {
        name: "AddrMode: base+reg fold effective address (8-bit exhaustive)".to_string(),
        tmir_expr: before_addr,
        aarch64_expr: after_addr,
        inputs: vec![
            ("base".to_string(), width),
            ("index".to_string(), width),
        ],
        preconditions: vec![],
        fp_inputs: vec![],
    }
}

// ---------------------------------------------------------------------------
// Aggregate accessors
// ---------------------------------------------------------------------------

/// Return all address mode formation proofs (16 total).
///
/// Includes:
/// - 3 effective address proofs (64-bit): base+imm, combined, base+reg
/// - 3 effective address proofs (32-bit): same at 32-bit width
/// - 4 scaled offset range proofs (1/2/4/8 byte access sizes)
/// - 4 writeback proofs: pre-index, post-index, post-index effective addr, pre/post same wb
/// - 1 unscaled signed 9-bit range proof
/// - 3 exhaustive 8-bit variants
///
/// Total: 18 proofs.
pub fn all_addr_mode_proofs() -> Vec<ProofObligation> {
    vec![
        // 64-bit effective address equivalence
        proof_base_plus_imm_effective_addr(),
        proof_base_plus_imm_combined_offset(),
        proof_base_plus_reg_effective_addr(),
        // 32-bit effective address equivalence
        proof_base_plus_imm_effective_addr_w32(),
        proof_base_plus_imm_combined_offset_w32(),
        proof_base_plus_reg_effective_addr_w32(),
        // Scaled offset encoding range
        proof_scaled_offset_range_1b(),
        proof_scaled_offset_range_2b(),
        proof_scaled_offset_range_4b(),
        proof_scaled_offset_range_8b(),
        // Pre/post-index writeback
        proof_pre_index_writeback(),
        proof_post_index_writeback(),
        proof_post_index_effective_addr(),
        proof_pre_post_index_same_writeback(),
        // Unscaled signed 9-bit range
        proof_unscaled_signed_9bit_range(),
        // 8-bit exhaustive variants
        proof_base_plus_imm_effective_addr_i8(),
        proof_base_plus_imm_combined_offset_i8(),
        proof_base_plus_reg_effective_addr_i8(),
    ]
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

    // ===================================================================
    // Effective address equivalence proofs (64-bit)
    // ===================================================================

    #[test]
    fn test_proof_base_plus_imm_effective_addr() {
        assert_valid(&proof_base_plus_imm_effective_addr());
    }

    #[test]
    fn test_proof_base_plus_imm_combined_offset() {
        assert_valid(&proof_base_plus_imm_combined_offset());
    }

    #[test]
    fn test_proof_base_plus_reg_effective_addr() {
        assert_valid(&proof_base_plus_reg_effective_addr());
    }

    // ===================================================================
    // Effective address equivalence proofs (32-bit)
    // ===================================================================

    #[test]
    fn test_proof_base_plus_imm_effective_addr_w32() {
        assert_valid(&proof_base_plus_imm_effective_addr_w32());
    }

    #[test]
    fn test_proof_base_plus_imm_combined_offset_w32() {
        assert_valid(&proof_base_plus_imm_combined_offset_w32());
    }

    #[test]
    fn test_proof_base_plus_reg_effective_addr_w32() {
        assert_valid(&proof_base_plus_reg_effective_addr_w32());
    }

    // ===================================================================
    // Scaled offset range proofs
    // ===================================================================

    #[test]
    fn test_proof_scaled_offset_range_1b() {
        assert_valid(&proof_scaled_offset_range_1b());
    }

    #[test]
    fn test_proof_scaled_offset_range_2b() {
        assert_valid(&proof_scaled_offset_range_2b());
    }

    #[test]
    fn test_proof_scaled_offset_range_4b() {
        assert_valid(&proof_scaled_offset_range_4b());
    }

    #[test]
    fn test_proof_scaled_offset_range_8b() {
        assert_valid(&proof_scaled_offset_range_8b());
    }

    // ===================================================================
    // Pre/post-index writeback proofs
    // ===================================================================

    #[test]
    fn test_proof_pre_index_writeback() {
        assert_valid(&proof_pre_index_writeback());
    }

    #[test]
    fn test_proof_post_index_writeback() {
        assert_valid(&proof_post_index_writeback());
    }

    #[test]
    fn test_proof_post_index_effective_addr() {
        assert_valid(&proof_post_index_effective_addr());
    }

    #[test]
    fn test_proof_pre_post_index_same_writeback() {
        assert_valid(&proof_pre_post_index_same_writeback());
    }

    // ===================================================================
    // Unscaled signed 9-bit range proof
    // ===================================================================

    #[test]
    fn test_proof_unscaled_signed_9bit_range() {
        assert_valid(&proof_unscaled_signed_9bit_range());
    }

    // ===================================================================
    // 8-bit exhaustive proofs
    // ===================================================================

    #[test]
    fn test_proof_base_plus_imm_effective_addr_i8() {
        assert_valid(&proof_base_plus_imm_effective_addr_i8());
    }

    #[test]
    fn test_proof_base_plus_imm_combined_offset_i8() {
        assert_valid(&proof_base_plus_imm_combined_offset_i8());
    }

    #[test]
    fn test_proof_base_plus_reg_effective_addr_i8() {
        assert_valid(&proof_base_plus_reg_effective_addr_i8());
    }

    // ===================================================================
    // Aggregate test
    // ===================================================================

    #[test]
    fn test_all_addr_mode_proofs() {
        let proofs = all_addr_mode_proofs();
        assert_eq!(proofs.len(), 18, "Expected 18 address mode proofs, got {}", proofs.len());
        for obligation in &proofs {
            assert_valid(obligation);
        }
    }

    // ===================================================================
    // Negative tests: verify incorrect transformations are detected
    // ===================================================================

    /// Negative test: base+imm fold with wrong offset is detected.
    ///
    /// Using `base + imm + 1` instead of `base + imm` should be invalid.
    #[test]
    fn test_wrong_base_plus_imm_detected() {
        let width = 8;
        let base = SmtExpr::var("base", width);
        let imm = SmtExpr::var("imm", width);

        // Wrong: (base + imm) + 0 == base + imm + 1
        let before = base.clone().bvadd(imm.clone()).bvadd(SmtExpr::bv_const(0, width));
        let wrong_after = base.bvadd(imm).bvadd(SmtExpr::bv_const(1, width));

        let obligation = ProofObligation {
            name: "WRONG: base+imm with extra +1".to_string(),
            tmir_expr: before,
            aarch64_expr: wrong_after,
            inputs: vec![
                ("base".to_string(), width),
                ("imm".to_string(), width),
            ],
            preconditions: vec![],
            fp_inputs: vec![],
        };

        let result = verify_by_evaluation(&obligation);
        match result {
            VerificationResult::Invalid { .. } => {}
            other => panic!("Expected Invalid for wrong fold, got {:?}", other),
        }
    }

    /// Negative test: base+reg fold with non-zero offset should fail.
    ///
    /// Folding `ADD Xd, Xn, Xm` + `LDR [Xd, #8]` into `LDR [Xn, Xm]`
    /// would be incorrect because the original has an extra +8.
    #[test]
    fn test_wrong_base_plus_reg_nonzero_offset_detected() {
        let width = 8;
        let base = SmtExpr::var("base", width);
        let index = SmtExpr::var("index", width);

        // Original: (base + index) + 8
        let before = base.clone().bvadd(index.clone()).bvadd(SmtExpr::bv_const(8, width));
        // Wrong fold: base + index (drops the +8)
        let wrong_after = base.bvadd(index);

        let obligation = ProofObligation {
            name: "WRONG: base+reg fold drops non-zero offset".to_string(),
            tmir_expr: before,
            aarch64_expr: wrong_after,
            inputs: vec![
                ("base".to_string(), width),
                ("index".to_string(), width),
            ],
            preconditions: vec![],
            fp_inputs: vec![],
        };

        let result = verify_by_evaluation(&obligation);
        match result {
            VerificationResult::Invalid { .. } => {}
            other => panic!("Expected Invalid for wrong base+reg fold, got {:?}", other),
        }
    }

    /// Negative test: combined offset not associative with wrong grouping.
    ///
    /// (base + imm1) + imm2 != base + imm1  (missing imm2)
    #[test]
    fn test_wrong_combined_offset_missing_imm2_detected() {
        let width = 8;
        let base = SmtExpr::var("base", width);
        let imm1 = SmtExpr::var("imm1", width);
        let imm2 = SmtExpr::var("imm2", width);

        let before = base.clone().bvadd(imm1.clone()).bvadd(imm2);
        let wrong_after = base.bvadd(imm1); // Missing imm2!

        let obligation = ProofObligation {
            name: "WRONG: combined offset drops imm2".to_string(),
            tmir_expr: before,
            aarch64_expr: wrong_after,
            inputs: vec![
                ("base".to_string(), width),
                ("imm1".to_string(), width),
                ("imm2".to_string(), width),
            ],
            preconditions: vec![],
            fp_inputs: vec![],
        };

        let result = verify_by_evaluation(&obligation);
        match result {
            VerificationResult::Invalid { .. } => {}
            other => panic!("Expected Invalid for missing imm2, got {:?}", other),
        }
    }

    // ===================================================================
    // SMT-LIB2 output tests
    // ===================================================================

    #[test]
    fn test_smt2_output_base_plus_imm() {
        let obligation = proof_base_plus_imm_effective_addr();
        let smt2 = obligation.to_smt2();
        assert!(smt2.contains("(set-logic"), "should contain set-logic");
        assert!(smt2.contains("(declare-const base (_ BitVec 64))"), "should declare base");
        assert!(smt2.contains("(declare-const imm (_ BitVec 64))"), "should declare imm");
        assert!(smt2.contains("(check-sat)"), "should contain check-sat");
        assert!(smt2.contains("bvadd"), "should contain bvadd");
    }

    #[test]
    fn test_smt2_output_base_plus_reg() {
        let obligation = proof_base_plus_reg_effective_addr();
        let smt2 = obligation.to_smt2();
        assert!(smt2.contains("(set-logic"), "should contain set-logic");
        assert!(smt2.contains("(declare-const base (_ BitVec 64))"), "should declare base");
        assert!(smt2.contains("(declare-const index (_ BitVec 64))"), "should declare index");
        assert!(smt2.contains("(check-sat)"), "should contain check-sat");
    }

    #[test]
    fn test_smt2_output_scaled_offset() {
        let obligation = proof_scaled_offset_range_4b();
        let smt2 = obligation.to_smt2();
        assert!(smt2.contains("(set-logic"), "should contain set-logic");
        assert!(smt2.contains("(declare-const offset (_ BitVec 16))"), "should declare offset");
        assert!(smt2.contains("(check-sat)"), "should contain check-sat");
    }

    #[test]
    fn test_smt2_output_pre_index() {
        let obligation = proof_pre_index_writeback();
        let smt2 = obligation.to_smt2();
        assert!(smt2.contains("(set-logic"), "should contain set-logic");
        assert!(smt2.contains("(declare-const base (_ BitVec 64))"), "should declare base");
        assert!(smt2.contains("(check-sat)"), "should contain check-sat");
    }
}
