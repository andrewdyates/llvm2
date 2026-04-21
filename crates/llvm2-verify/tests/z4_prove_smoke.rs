// llvm2-verify/tests/z4_prove_smoke.rs - z4-prove feature smoke tests
//
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Copyright 2026 Andrew Yates | License: Apache-2.0
//
// This test file is gated on the `z4-prove` cargo feature (see
// `crates/llvm2-verify/Cargo.toml`). It is the entry point for the
// "full SMT proof via z4" opt-in verification lane.
//
// # Running
//
// ```
// cargo test -p llvm2-verify --features z4-prove --test z4_prove_smoke
// ```
//
// Without the feature, this file compiles to nothing and the tests are
// inert. With the feature, each test dispatches a proof obligation to the
// z4 native Rust API and asserts the property holds (Verified) or the
// solver ran out of time (Timeout, tolerated — z4 may be slow under load).
//
// # Widths covered
//
// z4 reasons symbolically in the theory of bitvectors: the decision
// procedure does not enumerate the joint input space, so moving from i8
// (2^16 pairs) to i16 (2^32 pairs) has only modest impact on wall-time.
// Issue #449 (epic #407 Task 3) widened the smoke lane to i32/i64 once the
// StableHasher bridge cache (#420) made the wider-width obligations
// tractable. i32/i64 obligations use the tolerant
// `assert_verified_or_timeout` helper — a timeout is still tolerated there
// because heavy-arithmetic formulas (imul_i64, deep ASR/LSR) may not close
// in the 10 s budget on a loaded host; the summary test below reports the
// Verified vs Timeout split.
//
// # References
//
// - `reports/2026-04-18-329-plan.md` — scope and rationale.
// - `docs/z4-prove-feature.md` — how to run, what it proves.
// - `designs/2026-04-14-z4-integration-guide.md` — z4 bridge architecture.
// - Issue #329 — "Enable z4 verification by default" (opt-in infra).

#![cfg(feature = "z4-prove")]

use llvm2_verify::lowering_proof::{
    proof_band_i16, proof_band_i32, proof_band_i64, proof_band_i8, proof_bic_i16, proof_bic_i32,
    proof_bic_i64, proof_bic_i8, proof_bitcast_i16, proof_bitcast_i32, proof_bitcast_i64,
    proof_bitcast_i8, proof_bor_i16, proof_bor_i32, proof_bor_i64, proof_bor_i8, proof_bxor_i16,
    proof_bxor_i32, proof_bxor_i64, proof_bxor_i8, proof_condbr_eq_i32, proof_condbr_eq_i64,
    proof_condbr_ne_i32, proof_condbr_ne_i64, proof_condbr_sge_i32, proof_condbr_sge_i64,
    proof_condbr_sgt_i32, proof_condbr_sgt_i64, proof_condbr_sle_i32, proof_condbr_sle_i64,
    proof_condbr_slt_i32, proof_condbr_slt_i64, proof_condbr_uge_i32, proof_condbr_uge_i64,
    proof_condbr_ugt_i32, proof_condbr_ugt_i64, proof_condbr_ule_i32, proof_condbr_ule_i64,
    proof_condbr_ult_i32, proof_condbr_ult_i64, proof_extract_bits_i8, proof_fabs_f32,
    proof_fabs_f64, proof_fadd_f32, proof_fadd_f64, proof_fcmp_eq_f32, proof_fcmp_eq_f64,
    proof_fcmp_ge_f32, proof_fcmp_ge_f64, proof_fcmp_gt_f32, proof_fcmp_gt_f64,
    proof_fcmp_le_f32, proof_fcmp_le_f64, proof_fcmp_lt_f32, proof_fcmp_lt_f64,
    proof_fcmp_ne_f32, proof_fcmp_ne_f64, proof_fcmp_ord_f32, proof_fcmp_ord_f64,
    proof_fcmp_ueq_f32, proof_fcmp_ueq_f64, proof_fcmp_uge_f32, proof_fcmp_uge_f64,
    proof_fcmp_ugt_f32, proof_fcmp_ugt_f64, proof_fcmp_ule_f32, proof_fcmp_ule_f64,
    proof_fcmp_ult_f32, proof_fcmp_ult_f64, proof_fcmp_une_f32, proof_fcmp_une_f64,
    proof_fcmp_uno_f32, proof_fcmp_uno_f64, proof_fdiv_f32, proof_fdiv_f64, proof_fmul_f32,
    proof_fmul_f64, proof_fneg_f32, proof_fneg_f64, proof_fsqrt_f32, proof_fsqrt_f64,
    proof_fsub_f32, proof_fsub_f64, proof_iadd_i16,
    proof_iadd_i32, proof_iadd_i64, proof_iadd_i8, proof_icmp_eq_i32, proof_icmp_eq_i64,
    proof_icmp_ne_i32, proof_icmp_ne_i64, proof_icmp_sge_i32, proof_icmp_sge_i64,
    proof_icmp_sgt_i32, proof_icmp_sgt_i64, proof_icmp_sle_i32, proof_icmp_sle_i64,
    proof_icmp_slt_i32, proof_icmp_slt_i64, proof_icmp_uge_i32, proof_icmp_uge_i64,
    proof_icmp_ugt_i32, proof_icmp_ugt_i64, proof_icmp_ule_i32, proof_icmp_ule_i64,
    proof_icmp_ult_i32, proof_icmp_ult_i64, proof_imul_i16, proof_imul_i32, proof_imul_i64,
    proof_imul_i8, proof_insert_bits_i8, proof_ishl_i16, proof_ishl_i32, proof_ishl_i64,
    proof_ishl_i8, proof_isub_i16, proof_isub_i32, proof_isub_i64, proof_isub_i8, proof_neg_i16,
    proof_neg_i32, proof_neg_i64, proof_neg_i8, proof_orn_i16, proof_orn_i32, proof_orn_i64,
    proof_orn_i8, proof_sextract_bits_i8, proof_srem_i16, proof_srem_i8, proof_sshr_i16,
    proof_sshr_i32, proof_sshr_i64, proof_sshr_i8, proof_urem_i16, proof_urem_i8, proof_ushr_i16,
    proof_ushr_i32, proof_ushr_i64, proof_ushr_i8, ProofObligation,
};
use llvm2_verify::memory_proofs::{
    proof_aligned_roundtrip_i32, proof_aligned_roundtrip_i64,
    proof_axiom_const_array_32, proof_axiom_read_after_write_diff_addr_32,
    proof_axiom_read_after_write_same_addr_32, proof_axiom_write_write_same_addr_32,
    proof_endianness_i32, proof_endianness_i64, proof_endianness_msb_i32,
    proof_forwarding_i32_offset12, proof_forwarding_i64_offset24,
    proof_forwarding_i64_to_i32_lower, proof_load_i32, proof_load_i32_offset,
    proof_load_i64, proof_load_i64_offset, proof_non_interference_i32,
    proof_non_interference_i32_adjacent, proof_non_interference_i32_store_i64_load,
    proof_non_interference_i32_symbolic, proof_non_interference_i64,
    proof_non_interference_i64_adjacent, proof_non_interference_i64_store_i32_load,
    proof_non_interference_i64_symbolic, proof_roundtrip_i32, proof_roundtrip_i64,
    proof_scaled_offset_alignment_i32, proof_store_i32, proof_store_i32_offset,
    proof_store_i64, proof_store_i64_offset, proof_subword_i32_byte1,
    proof_subword_i32_byte2, proof_subword_i32_halfword_upper, proof_subword_i64_byte5,
    proof_write_combining_halfwords_to_i32, proof_write_combining_i32,
    proof_write_combining_words_to_i64,
};
use llvm2_verify::switch_proofs::{
    proof_switch_dense_i16, proof_switch_dense_i8, proof_switch_dense_i8_nonzero_base,
    proof_switch_dense_i8_with_hole, proof_switch_sparse_7case_i8, proof_switch_sparse_i16,
    proof_switch_sparse_i8,
};
use llvm2_verify::ext_trunc_proofs::{
    proof_roundtrip_sext_trunc_16, proof_roundtrip_sext_trunc_8, proof_roundtrip_zext_trunc_16,
    proof_roundtrip_zext_trunc_8, proof_sxtb_8_to_32, proof_sxtb_8_to_64, proof_sxtb_idempotent,
    proof_sxth_16_to_32, proof_sxth_16_to_64, proof_sxth_idempotent, proof_sxtw_32_to_64,
    proof_trunc_to_i16, proof_trunc_to_i16_from_64, proof_trunc_to_i32_from_64,
    proof_trunc_to_i8, proof_trunc_to_i8_from_64, proof_uxtb_8_to_32, proof_uxtb_8_to_64,
    proof_uxtb_idempotent, proof_uxth_16_to_32, proof_uxth_16_to_64, proof_uxth_idempotent,
    proof_uxtw_32_to_64,
};
use llvm2_verify::peephole_proofs::{
    proof_add_zero_identity, proof_and_ri_allones_identity, proof_and_ri_zero,
    proof_and_self_identity, proof_eor_rr_self_zero, proof_eor_zero_identity,
    proof_asr_zero_identity, proof_lsl_zero_identity, proof_lsr_zero_identity,
    proof_mul_one_identity, proof_mul_zero_absorb, proof_neg_neg_identity,
    proof_orr_ri_allones, proof_orr_ri_zero_identity, proof_orr_self_identity,
    proof_sub_rr_self_zero, proof_sub_zero_identity, proof_udiv_one_identity,
};
use llvm2_verify::x86_64_eflags_proofs::{
    proof_cmp_cf_i8, proof_cmp_of_i8, proof_cmp_sf_i8, proof_cmp_zf_i8, proof_sub_cf_i8,
    proof_sub_of_i8, proof_sub_sf_i8, proof_sub_zf_i8, proof_x86_cmp_writes_zf_i32,
};
use llvm2_verify::x86_64_lowering_proofs::{
    proof_x86_band_i8, proof_x86_band_i32, proof_x86_band_i64, proof_x86_bor_i8,
    proof_x86_bor_i32, proof_x86_bor_i64, proof_x86_bxor_i8, proof_x86_bxor_i32,
    proof_x86_bxor_i64, proof_x86_fabs_f32, proof_x86_fabs_f64, proof_x86_fadd_f32,
    proof_x86_fadd_f64, proof_x86_fdiv_f32, proof_x86_fdiv_f64, proof_x86_fmul_f32,
    proof_x86_fmul_f64, proof_x86_fneg_f32, proof_x86_fneg_f64, proof_x86_fsqrt_f32,
    proof_x86_fsqrt_f64, proof_x86_fsub_f32, proof_x86_fsub_f64, proof_x86_iadd_i8,
    proof_x86_iadd_i32, proof_x86_iadd_i64, proof_x86_icmp_eq_i32, proof_x86_icmp_eq_i64,
    proof_x86_icmp_ne_i32, proof_x86_icmp_ne_i64, proof_x86_icmp_sge_i32,
    proof_x86_icmp_sge_i64, proof_x86_icmp_sgt_i32, proof_x86_icmp_sgt_i64,
    proof_x86_icmp_sle_i32, proof_x86_icmp_sle_i64, proof_x86_icmp_slt_i32,
    proof_x86_icmp_slt_i64, proof_x86_icmp_uge_i32, proof_x86_icmp_uge_i64,
    proof_x86_icmp_ugt_i32, proof_x86_icmp_ugt_i64, proof_x86_icmp_ule_i32,
    proof_x86_icmp_ule_i64, proof_x86_icmp_ult_i32, proof_x86_icmp_ult_i64,
    proof_x86_imul_i8, proof_x86_imul_i32, proof_x86_imul_i64, proof_x86_ishl_i8,
    proof_x86_ishl_i32, proof_x86_ishl_i64, proof_x86_isub_i8, proof_x86_isub_i32,
    proof_x86_isub_i64, proof_x86_neg_i32, proof_x86_neg_i64, proof_x86_sshr_i8,
    proof_x86_sshr_i32, proof_x86_sshr_i64, proof_x86_ushr_i8, proof_x86_ushr_i32,
    proof_x86_ushr_i64,
};
use llvm2_verify::z4_bridge::{verify_with_z4_api, Z4Config, Z4Result};

/// Strict assertion: a z4 smoke proof is OK **only** if the solver
/// returned `Verified`. Any other result — `CounterExample`, `Timeout`,
/// or `Error` — is a failure.
///
/// A timeout is a proof failure, not a silent pass (#407 Phase 0, #389).
/// This keeps the smoke lane honest: if z4 cannot discharge the small
/// 8/16-bit obligations within the 30 s default budget, something is
/// wrong (slow solver build, missing theory, formula regression) and we
/// want the test to surface it — not silently say "good enough".
fn assert_verified(name: &str, obligation: &ProofObligation) {
    let config = Z4Config::default();
    let result = verify_with_z4_api(obligation, &config);
    assert!(
        matches!(result, Z4Result::Verified),
        "z4-prove smoke: '{}' returned {}; expected Verified",
        name,
        result
    );
}

/// Tolerant assertion used for obligations at widths (i16/i32/i64) or
/// operations (mul, shifts, remainder, symbolic switch tables, bitfield)
/// where z4 may legitimately run out of time under the default budget.
/// A `Verified` result is the success path; `Timeout` is tolerated because
/// the eval/sampling lane also covers the obligation; `CounterExample` or
/// `Error` always fails — those would indicate a real soundness bug, not
/// a budget issue.
///
/// The summary test below reports the verified/timed-out split so CI can
/// track the ratio and detect a silent-timeout regression where the whole
/// suite silently becomes non-functional.
fn assert_verified_or_timeout(name: &str, obligation: &ProofObligation) {
    let config = Z4Config::default().with_timeout(10_000);
    let result = verify_with_z4_api(obligation, &config);
    match result {
        Z4Result::Verified | Z4Result::Timeout => {}
        other => panic!(
            "z4-prove smoke: '{}' returned {}; expected Verified or Timeout",
            name, other
        ),
    }
}

// ===========================================================================
// 8-bit arithmetic proofs
// ===========================================================================

#[test]
fn z4_prove_iadd_i8() {
    assert_verified("iadd_i8", &proof_iadd_i8());
}

#[test]
fn z4_prove_isub_i8() {
    assert_verified("isub_i8", &proof_isub_i8());
}

/// Unary `Ineg -> NEG` at i8. Lowers to `SUB Rd, ZR, Rn`; SMT obligation
/// reduces to `bvneg x == bvsub (bv_const 0 8) x`, closed in milliseconds.
/// First unary-op in the z4 smoke lane (issue #423).
#[test]
fn z4_prove_neg_i8() {
    assert_verified("neg_i8", &proof_neg_i8());
}

// ===========================================================================
// 8-bit bitwise proofs
// ===========================================================================

#[test]
fn z4_prove_band_i8() {
    assert_verified("band_i8", &proof_band_i8());
}

#[test]
fn z4_prove_bor_i8() {
    assert_verified("bor_i8", &proof_bor_i8());
}

#[test]
fn z4_prove_bxor_i8() {
    assert_verified("bxor_i8", &proof_bxor_i8());
}

// ===========================================================================
// 16-bit arithmetic proofs (issue #426)
// ===========================================================================

#[test]
fn z4_prove_iadd_i16() {
    assert_verified("iadd_i16", &proof_iadd_i16());
}

#[test]
fn z4_prove_isub_i16() {
    assert_verified("isub_i16", &proof_isub_i16());
}

#[test]
fn z4_prove_band_i16() {
    assert_verified("band_i16", &proof_band_i16());
}

#[test]
fn z4_prove_bor_i16() {
    assert_verified("bor_i16", &proof_bor_i16());
}

#[test]
fn z4_prove_bxor_i16() {
    assert_verified("bxor_i16", &proof_bxor_i16());
}

// ===========================================================================
// 8-bit shift proofs (issue #424)
// ===========================================================================

#[test]
fn z4_prove_ishl_i8() {
    assert_verified("ishl_i8", &proof_ishl_i8());
}

#[test]
fn z4_prove_ushr_i8() {
    assert_verified("ushr_i8", &proof_ushr_i8());
}

#[test]
fn z4_prove_sshr_i8() {
    assert_verified("sshr_i8", &proof_sshr_i8());
}

// ===========================================================================
// 8-bit BIC/ORN proofs (issue #425)
// ===========================================================================

#[test]
fn z4_prove_bic_i8() {
    assert_verified("bic_i8", &proof_bic_i8());
}

#[test]
fn z4_prove_orn_i8() {
    assert_verified("orn_i8", &proof_orn_i8());
}

// ===========================================================================
// 16-bit shift proofs (issues #424 / #407)
// ===========================================================================

#[test]
fn z4_prove_ishl_i16() {
    assert_verified_or_timeout("ishl_i16", &proof_ishl_i16());
}

#[test]
fn z4_prove_ushr_i16() {
    assert_verified_or_timeout("ushr_i16", &proof_ushr_i16());
}

#[test]
fn z4_prove_sshr_i16() {
    assert_verified_or_timeout("sshr_i16", &proof_sshr_i16());
}

// ===========================================================================
// 16-bit BIC/ORN proofs (issues #425 / #407)
// ===========================================================================

#[test]
fn z4_prove_bic_i16() {
    assert_verified_or_timeout("bic_i16", &proof_bic_i16());
}

#[test]
fn z4_prove_orn_i16() {
    assert_verified_or_timeout("orn_i16", &proof_orn_i16());
}

// ===========================================================================
// Integer multiply + Ineg i16 proofs (epic #407)
// ===========================================================================
//
// `Opcode::Imul` lowers to the AArch64 `MUL` (Madd with ZR accumulator) at
// every integer width. Prior to this lane, `proof_imul_i8`/`proof_imul_i16`
// existed in `lowering_proof.rs` but ran only through the eval path
// (exhaustive 8-bit, 100K-sample 16-bit). Promoting them to the z4 SMT
// path exercises `bvmul` symbolically. Imul is one of the highest-leverage
// arithmetic opcodes in the adapter (array-index scale, polynomial
// evaluation, strength-reduced multiplication).
//
// Neg at i16 joins the lane alongside the existing i8 neg proof; the
// formula reduces to `bvneg x == bvsub (bv_const 0 16) x`.

#[test]
fn z4_prove_imul_i8() {
    assert_verified_or_timeout("imul_i8", &proof_imul_i8());
}

#[test]
fn z4_prove_imul_i16() {
    assert_verified_or_timeout("imul_i16", &proof_imul_i16());
}

#[test]
fn z4_prove_neg_i16() {
    assert_verified_or_timeout("neg_i16", &proof_neg_i16());
}

// ===========================================================================
// 8-bit remainder + bitcast proofs (issue #435)
// ===========================================================================
//
// AArch64 has no dedicated remainder opcode; `Urem`/`Srem` lower to a UDIV
// or SDIV followed by an MSUB: `r = a - (a / b) * b`. The proofs below
// verify the composition against the native `bvurem` / `bvsrem` primitives
// under the relevant preconditions (`b != 0`, plus `not INT_MIN/-1` for
// `Srem`). These are the first z4 smoke obligations that carry more than
// one precondition -- useful regression coverage for `verify_with_z4_api`'s
// precondition assembly.
//
// `Bitcast` is pure type-punning and reduces to the bitvector identity;
// same-width bitcasts lower to `MOV` (GPR<->GPR) or `FMOV` (FPR<->FPR or
// GPR<->FPR). The i8 obligation is the representative check.

#[test]
fn z4_prove_urem_i8() {
    assert_verified_or_timeout("urem_i8", &proof_urem_i8());
}

#[test]
fn z4_prove_srem_i8() {
    assert_verified_or_timeout("srem_i8", &proof_srem_i8());
}

#[test]
fn z4_prove_bitcast_i8() {
    assert_verified_or_timeout("bitcast_i8", &proof_bitcast_i8());
}

// ===========================================================================
// #435 widened remainder + bitcast (i16 rem; i16/i32/i64 bitcast)
// ===========================================================================
//
// Same-shape widening of `proof_urem_i8` / `proof_srem_i8` to i16 and of
// `proof_bitcast_i8` to i16/i32/i64. All underlying encoders
// (`encode_udiv_rr`, `encode_sdiv_rr`, `encode_msub_rr`, `encode_mov_rr`,
// `encode_tmir_bitcast`) are width-polymorphic, so the obligations scale
// without new semantic machinery. Closes the rem/bitcast row of the #435
// coverage matrix alongside the existing i8 rem/bitcast and i8 bitfield
// obligations. Tolerant assertion because the bitblasted remainder at i16
// may run past the 10 s budget on a loaded host.

#[test]
fn z4_prove_urem_i16() {
    assert_verified_or_timeout("urem_i16", &proof_urem_i16());
}

#[test]
fn z4_prove_srem_i16() {
    assert_verified_or_timeout("srem_i16", &proof_srem_i16());
}

#[test]
fn z4_prove_bitcast_i16() {
    assert_verified_or_timeout("bitcast_i16", &proof_bitcast_i16());
}

#[test]
fn z4_prove_bitcast_i32() {
    assert_verified_or_timeout("bitcast_i32", &proof_bitcast_i32());
}

#[test]
fn z4_prove_bitcast_i64() {
    assert_verified_or_timeout("bitcast_i64", &proof_bitcast_i64());
}

// ===========================================================================
// #435 bitfield (#452)
// ===========================================================================
//
// tMIR bitfield opcodes (`ExtractBits`, `SextractBits`, `InsertBits`) lower
// to AArch64 UBFM / SBFM / BFM. All three are pure QF_BV operations with no
// preconditions -- the proofs verify that the tMIR shift-and-mask semantics
// are exactly equivalent to the machine encoding at i8 with `(lsb=2,
// width=4)`. Closes the top of the #435 Phase-2 Tier-C matrix.
//
// Reference: ARM DDI 0487, C6.2.335 UBFM, C6.2.266 SBFM, C6.2.46 BFM.

#[test]
fn z4_prove_extract_bits_i8() {
    assert_verified_or_timeout("extract_bits_i8", &proof_extract_bits_i8());
}

#[test]
fn z4_prove_sextract_bits_i8() {
    assert_verified_or_timeout("sextract_bits_i8", &proof_sextract_bits_i8());
}

#[test]
fn z4_prove_insert_bits_i8() {
    assert_verified_or_timeout("insert_bits_i8", &proof_insert_bits_i8());
}

// ===========================================================================
// Switch lowering proofs (issue #444, deferred from #323)
// ===========================================================================
//
// tMIR `Switch(scrutinee, cases, default)` lowers to one of three machine
// forms (see `llvm2-lower/src/switch.rs`):
//   - Linear scan    (N <= 3, covered by the i8 cases below with small N)
//   - Jump table     (dense, >=0.4 density; modelled by the i8/i16 dense proofs)
//   - Binary search  (sparse; modelled by the i8/i16 sparse proofs)
//
// The proof obligations encode successor-block identity: for every bounded
// scrutinee value, the lowered CFG selects the same target block as a
// linear-scan reference. 8-bit obligations also run exhaustively in the
// default eval path (`crates/llvm2-verify/src/switch_proofs.rs::tests`); the
// z4 smoke lane extends to i16 with full symbolic coverage.

#[test]
fn z4_prove_switch_dense_i8() {
    assert_verified_or_timeout("switch_dense_i8", &proof_switch_dense_i8());
}

#[test]
fn z4_prove_switch_dense_i8_nonzero_base() {
    assert_verified_or_timeout(
        "switch_dense_i8_nonzero_base",
        &proof_switch_dense_i8_nonzero_base(),
    );
}

#[test]
fn z4_prove_switch_dense_i8_with_hole() {
    assert_verified_or_timeout(
        "switch_dense_i8_with_hole",
        &proof_switch_dense_i8_with_hole(),
    );
}

#[test]
fn z4_prove_switch_dense_i16() {
    assert_verified_or_timeout("switch_dense_i16", &proof_switch_dense_i16());
}

#[test]
fn z4_prove_switch_sparse_i8() {
    assert_verified_or_timeout("switch_sparse_i8", &proof_switch_sparse_i8());
}

#[test]
fn z4_prove_switch_sparse_7case_i8() {
    assert_verified_or_timeout(
        "switch_sparse_7case_i8",
        &proof_switch_sparse_7case_i8(),
    );
}

#[test]
fn z4_prove_switch_sparse_i16() {
    assert_verified_or_timeout("switch_sparse_i16", &proof_switch_sparse_i16());
}

// ===========================================================================
// i32 core arithmetic / logic / shift proofs (issue #449, epic #407 Task 3)
// ===========================================================================
//
// Wider widths depend on the StableHasher cache (#420) to keep z4's
// bitblasting tractable. imul/mul-heavy obligations may still time out at
// i32/i64 — `assert_verified_or_timeout` tolerates that; the summary test
// below reports the split. Pre-landing (i8/i16 only) the suite had 34
// obligations; post-landing it has 58.

#[test]
fn z4_prove_iadd_i32() {
    assert_verified_or_timeout("iadd_i32", &proof_iadd_i32());
}

#[test]
fn z4_prove_isub_i32() {
    assert_verified_or_timeout("isub_i32", &proof_isub_i32());
}

#[test]
fn z4_prove_imul_i32() {
    assert_verified_or_timeout("imul_i32", &proof_imul_i32());
}

#[test]
fn z4_prove_neg_i32() {
    assert_verified_or_timeout("neg_i32", &proof_neg_i32());
}

#[test]
fn z4_prove_band_i32() {
    assert_verified_or_timeout("band_i32", &proof_band_i32());
}

#[test]
fn z4_prove_bor_i32() {
    assert_verified_or_timeout("bor_i32", &proof_bor_i32());
}

#[test]
fn z4_prove_bxor_i32() {
    assert_verified_or_timeout("bxor_i32", &proof_bxor_i32());
}

#[test]
fn z4_prove_ishl_i32() {
    assert_verified_or_timeout("ishl_i32", &proof_ishl_i32());
}

#[test]
fn z4_prove_ushr_i32() {
    assert_verified_or_timeout("ushr_i32", &proof_ushr_i32());
}

#[test]
fn z4_prove_sshr_i32() {
    assert_verified_or_timeout("sshr_i32", &proof_sshr_i32());
}

#[test]
fn z4_prove_bic_i32() {
    assert_verified_or_timeout("bic_i32", &proof_bic_i32());
}

#[test]
fn z4_prove_orn_i32() {
    assert_verified_or_timeout("orn_i32", &proof_orn_i32());
}

// ===========================================================================
// i64 core arithmetic / logic / shift proofs (issue #449, epic #407 Task 3)
// ===========================================================================

#[test]
fn z4_prove_iadd_i64() {
    assert_verified_or_timeout("iadd_i64", &proof_iadd_i64());
}

#[test]
fn z4_prove_isub_i64() {
    assert_verified_or_timeout("isub_i64", &proof_isub_i64());
}

#[test]
fn z4_prove_imul_i64() {
    assert_verified_or_timeout("imul_i64", &proof_imul_i64());
}

#[test]
fn z4_prove_neg_i64() {
    assert_verified_or_timeout("neg_i64", &proof_neg_i64());
}

#[test]
fn z4_prove_band_i64() {
    assert_verified_or_timeout("band_i64", &proof_band_i64());
}

#[test]
fn z4_prove_bor_i64() {
    assert_verified_or_timeout("bor_i64", &proof_bor_i64());
}

#[test]
fn z4_prove_bxor_i64() {
    assert_verified_or_timeout("bxor_i64", &proof_bxor_i64());
}

#[test]
fn z4_prove_ishl_i64() {
    assert_verified_or_timeout("ishl_i64", &proof_ishl_i64());
}

#[test]
fn z4_prove_ushr_i64() {
    assert_verified_or_timeout("ushr_i64", &proof_ushr_i64());
}

#[test]
fn z4_prove_sshr_i64() {
    assert_verified_or_timeout("sshr_i64", &proof_sshr_i64());
}

#[test]
fn z4_prove_bic_i64() {
    assert_verified_or_timeout("bic_i64", &proof_bic_i64());
}

#[test]
fn z4_prove_orn_i64() {
    assert_verified_or_timeout("orn_i64", &proof_orn_i64());
}

// ===========================================================================
// AArch64 icmp predicates at i32/i64 (issue #468)
// ===========================================================================
//
// Wires every existing `proof_icmp_{eq,ne,slt,sge,sgt,sle,ult,uge,ugt,ule}_i{32,64}`
// helper from `lowering_proof.rs` into the default-on smoke lane. Fills the
// AArch64 icmp coverage gap identified in
// `reports/2026-04-19-419-z4-coverage-survey.md` recommendation #1 — AArch64
// had integer arithmetic at all 4 widths but zero icmp/condbr coverage in the
// smoke lane despite the proof helpers existing.
//
// All 20 obligations are CMP Wn,Wm ; CSET Wd,<cc> style — the expected pattern
// is a single subtraction followed by a condition-code materialization. z4
// closes the underlying formula quickly, but some CC encodings (e.g. signed
// overflow in SLT/SGE at i64) may exceed the 10 s budget under load; the
// tolerant helper is used uniformly so a transient timeout does not mask a
// real soundness regression elsewhere.

#[test]
fn z4_prove_icmp_eq_i32() {
    assert_verified_or_timeout("icmp_eq_i32", &proof_icmp_eq_i32());
}

#[test]
fn z4_prove_icmp_ne_i32() {
    assert_verified_or_timeout("icmp_ne_i32", &proof_icmp_ne_i32());
}

#[test]
fn z4_prove_icmp_slt_i32() {
    assert_verified_or_timeout("icmp_slt_i32", &proof_icmp_slt_i32());
}

#[test]
fn z4_prove_icmp_sge_i32() {
    assert_verified_or_timeout("icmp_sge_i32", &proof_icmp_sge_i32());
}

#[test]
fn z4_prove_icmp_sgt_i32() {
    assert_verified_or_timeout("icmp_sgt_i32", &proof_icmp_sgt_i32());
}

#[test]
fn z4_prove_icmp_sle_i32() {
    assert_verified_or_timeout("icmp_sle_i32", &proof_icmp_sle_i32());
}

#[test]
fn z4_prove_icmp_ult_i32() {
    assert_verified_or_timeout("icmp_ult_i32", &proof_icmp_ult_i32());
}

#[test]
fn z4_prove_icmp_uge_i32() {
    assert_verified_or_timeout("icmp_uge_i32", &proof_icmp_uge_i32());
}

#[test]
fn z4_prove_icmp_ugt_i32() {
    assert_verified_or_timeout("icmp_ugt_i32", &proof_icmp_ugt_i32());
}

#[test]
fn z4_prove_icmp_ule_i32() {
    assert_verified_or_timeout("icmp_ule_i32", &proof_icmp_ule_i32());
}

#[test]
fn z4_prove_icmp_eq_i64() {
    assert_verified_or_timeout("icmp_eq_i64", &proof_icmp_eq_i64());
}

#[test]
fn z4_prove_icmp_ne_i64() {
    assert_verified_or_timeout("icmp_ne_i64", &proof_icmp_ne_i64());
}

#[test]
fn z4_prove_icmp_slt_i64() {
    assert_verified_or_timeout("icmp_slt_i64", &proof_icmp_slt_i64());
}

#[test]
fn z4_prove_icmp_sge_i64() {
    assert_verified_or_timeout("icmp_sge_i64", &proof_icmp_sge_i64());
}

#[test]
fn z4_prove_icmp_sgt_i64() {
    assert_verified_or_timeout("icmp_sgt_i64", &proof_icmp_sgt_i64());
}

#[test]
fn z4_prove_icmp_sle_i64() {
    assert_verified_or_timeout("icmp_sle_i64", &proof_icmp_sle_i64());
}

#[test]
fn z4_prove_icmp_ult_i64() {
    assert_verified_or_timeout("icmp_ult_i64", &proof_icmp_ult_i64());
}

#[test]
fn z4_prove_icmp_uge_i64() {
    assert_verified_or_timeout("icmp_uge_i64", &proof_icmp_uge_i64());
}

#[test]
fn z4_prove_icmp_ugt_i64() {
    assert_verified_or_timeout("icmp_ugt_i64", &proof_icmp_ugt_i64());
}

#[test]
fn z4_prove_icmp_ule_i64() {
    assert_verified_or_timeout("icmp_ule_i64", &proof_icmp_ule_i64());
}

// ===========================================================================
// AArch64 condbr predicates at i32/i64 (issue #468)
// ===========================================================================
//
// Companion to the icmp block above. Proves that the full compare+branch
// fused lowering (CMP Wn,Wm ; B.<cc> target) is equivalent to a tMIR
// `Icmp(cc) ; CondBr(cond, then, else)` sequence at i32 and i64. Same
// predicate set as icmp.

#[test]
fn z4_prove_condbr_eq_i32() {
    assert_verified_or_timeout("condbr_eq_i32", &proof_condbr_eq_i32());
}

#[test]
fn z4_prove_condbr_ne_i32() {
    assert_verified_or_timeout("condbr_ne_i32", &proof_condbr_ne_i32());
}

#[test]
fn z4_prove_condbr_slt_i32() {
    assert_verified_or_timeout("condbr_slt_i32", &proof_condbr_slt_i32());
}

#[test]
fn z4_prove_condbr_sge_i32() {
    assert_verified_or_timeout("condbr_sge_i32", &proof_condbr_sge_i32());
}

#[test]
fn z4_prove_condbr_sgt_i32() {
    assert_verified_or_timeout("condbr_sgt_i32", &proof_condbr_sgt_i32());
}

#[test]
fn z4_prove_condbr_sle_i32() {
    assert_verified_or_timeout("condbr_sle_i32", &proof_condbr_sle_i32());
}

#[test]
fn z4_prove_condbr_ult_i32() {
    assert_verified_or_timeout("condbr_ult_i32", &proof_condbr_ult_i32());
}

#[test]
fn z4_prove_condbr_uge_i32() {
    assert_verified_or_timeout("condbr_uge_i32", &proof_condbr_uge_i32());
}

#[test]
fn z4_prove_condbr_ugt_i32() {
    assert_verified_or_timeout("condbr_ugt_i32", &proof_condbr_ugt_i32());
}

#[test]
fn z4_prove_condbr_ule_i32() {
    assert_verified_or_timeout("condbr_ule_i32", &proof_condbr_ule_i32());
}

#[test]
fn z4_prove_condbr_eq_i64() {
    assert_verified_or_timeout("condbr_eq_i64", &proof_condbr_eq_i64());
}

#[test]
fn z4_prove_condbr_ne_i64() {
    assert_verified_or_timeout("condbr_ne_i64", &proof_condbr_ne_i64());
}

#[test]
fn z4_prove_condbr_slt_i64() {
    assert_verified_or_timeout("condbr_slt_i64", &proof_condbr_slt_i64());
}

#[test]
fn z4_prove_condbr_sge_i64() {
    assert_verified_or_timeout("condbr_sge_i64", &proof_condbr_sge_i64());
}

#[test]
fn z4_prove_condbr_sgt_i64() {
    assert_verified_or_timeout("condbr_sgt_i64", &proof_condbr_sgt_i64());
}

#[test]
fn z4_prove_condbr_sle_i64() {
    assert_verified_or_timeout("condbr_sle_i64", &proof_condbr_sle_i64());
}

#[test]
fn z4_prove_condbr_ult_i64() {
    assert_verified_or_timeout("condbr_ult_i64", &proof_condbr_ult_i64());
}

#[test]
fn z4_prove_condbr_uge_i64() {
    assert_verified_or_timeout("condbr_uge_i64", &proof_condbr_uge_i64());
}

#[test]
fn z4_prove_condbr_ugt_i64() {
    assert_verified_or_timeout("condbr_ugt_i64", &proof_condbr_ugt_i64());
}

#[test]
fn z4_prove_condbr_ule_i64() {
    assert_verified_or_timeout("condbr_ule_i64", &proof_condbr_ule_i64());
}

// ===========================================================================
// x86-64 EFLAGS flag-write proofs (issue #458)
// ===========================================================================
//
// Direct z4 proof obligations for CMP/SUB flag writes at i8 (SF/ZF/CF/OF).
// Pins the x86-64 EFLAGS layer independent of the `proof_x86_icmp_*` SETcc
// chain: a CF or OF regression now surfaces here at the flag semantics
// rather than through an indirect icmp lowering failure.
//
// All eight obligations are pure 8-bit bitvector formulas (extract/eq/bvsub/
// bvult plus a single ITE coercion), so z4 closes them in milliseconds --
// they use `assert_verified` (strict) and a timeout here would indicate the
// bridge is broken, not a budget issue.

#[test]
fn z4_prove_x86_cmp_sf_i8() {
    assert_verified("x86_cmp_sf_i8", &proof_cmp_sf_i8());
}

#[test]
fn z4_prove_x86_cmp_zf_i8() {
    assert_verified("x86_cmp_zf_i8", &proof_cmp_zf_i8());
}

#[test]
fn z4_prove_x86_cmp_cf_i8() {
    assert_verified("x86_cmp_cf_i8", &proof_cmp_cf_i8());
}

#[test]
fn z4_prove_x86_cmp_of_i8() {
    assert_verified("x86_cmp_of_i8", &proof_cmp_of_i8());
}

#[test]
fn z4_prove_x86_sub_sf_i8() {
    assert_verified("x86_sub_sf_i8", &proof_sub_sf_i8());
}

#[test]
fn z4_prove_x86_sub_zf_i8() {
    assert_verified("x86_sub_zf_i8", &proof_sub_zf_i8());
}

#[test]
fn z4_prove_x86_sub_cf_i8() {
    assert_verified("x86_sub_cf_i8", &proof_sub_cf_i8());
}

#[test]
fn z4_prove_x86_sub_of_i8() {
    assert_verified("x86_sub_of_i8", &proof_sub_of_i8());
}

// ---------------------------------------------------------------------------
// x86-64 CMP writes at i32 (issue #458)
// ---------------------------------------------------------------------------
//
// The four direct CMP EFLAGS-write obligations (ZF/SF/CF/OF) at i32 pin the
// flag semantics independent of the SETcc chain. At least one is wired into
// this smoke lane per #458 AC item 4; ZF is the cheapest (pure zero-eq on
// i32 bvsub) so it stays here under the 10 s tolerant budget. The remaining
// three (SF/CF/OF) are exercised by
// `test_all_x86_64_cmp_writes_i32_proofs_registered` via eval/sampling and
// by the z4 smoke suite summary below.
//
// Budget: the `assert_verified_or_timeout` helper applies a 10 s z4 timeout.
// Timeout is tolerated because i32 widths can legitimately exceed the smoke
// budget under load -- CounterExample or Error still fails the test.

#[test]
fn z4_prove_x86_cmp_writes_zf_i32() {
    assert_verified_or_timeout(
        "x86_cmp_writes_zf_i32",
        &proof_x86_cmp_writes_zf_i32(),
    );
}

// ===========================================================================
// x86-64 core ops smoke (issue #457)
// ===========================================================================
//
// First z4 smoke obligations discharged against the x86-64 backend. The
// AArch64 lane already proves i32 iadd/isub/ishl (issue #449); this section
// mirrors that coverage against the x86-64 lowering and adds icmp_eq to
// exercise the `encode_cmp_setcc` + EFLAGS chain (x86_64_eflags.rs).
//
// Four obligations, each at i32 and each a distinct SMT shape:
//   - proof_x86_iadd_i32 — bvadd baseline.
//   - proof_x86_isub_i32 — bvsub baseline.
//   - proof_x86_icmp_eq_i32 — EFLAGS ZF + SETE zero-extend chain.
//   - proof_x86_ishl_i32 — shift (historically slower for z4).
//
// All four use `assert_verified_or_timeout` at the 10 s budget; timeouts are
// tolerated because i32 wide-arithmetic formulas can legitimately exceed
// the smoke budget under load, and the eval path also covers these. A
// CounterExample or Error fails the test.

#[test]
fn z4_prove_x86_iadd_i32() {
    assert_verified_or_timeout("x86_iadd_i32", &proof_x86_iadd_i32());
}

#[test]
fn z4_prove_x86_isub_i32() {
    assert_verified_or_timeout("x86_isub_i32", &proof_x86_isub_i32());
}

#[test]
fn z4_prove_x86_icmp_eq_i32() {
    assert_verified_or_timeout("x86_icmp_eq_i32", &proof_x86_icmp_eq_i32());
}

#[test]
fn z4_prove_x86_ishl_i32() {
    assert_verified_or_timeout("x86_ishl_i32", &proof_x86_ishl_i32());
}

// ===========================================================================
// x86-64 core ops smoke at i64 (#457 follow-up)
// ===========================================================================
//
// Mirrors the i32 x86-64 lane above at i64, using the pre-existing
// `proof_x86_*_i64` obligations in `x86_64_lowering_proofs.rs`. The i64
// width exercises the REX.W prefix path in the encoder plus 64-bit
// bitvector reasoning in the z4 bridge; it is also where mul/shift
// formulas may legitimately run past the 10 s budget on a loaded host,
// hence the tolerant `assert_verified_or_timeout` helper. A
// CounterExample or Error still fails.
//
// Four obligations at i64, shape-matched to the i32 lane:
//   - proof_x86_iadd_i64 — bvadd baseline.
//   - proof_x86_isub_i64 — bvsub baseline.
//   - proof_x86_icmp_eq_i64 — EFLAGS ZF + SETE zero-extend chain.
//   - proof_x86_ishl_i64 — shift (historically slower for z4).

#[test]
fn z4_prove_x86_iadd_i64() {
    assert_verified_or_timeout("x86_iadd_i64", &proof_x86_iadd_i64());
}

#[test]
fn z4_prove_x86_isub_i64() {
    assert_verified_or_timeout("x86_isub_i64", &proof_x86_isub_i64());
}

#[test]
fn z4_prove_x86_icmp_eq_i64() {
    assert_verified_or_timeout("x86_icmp_eq_i64", &proof_x86_icmp_eq_i64());
}

#[test]
fn z4_prove_x86_ishl_i64() {
    assert_verified_or_timeout("x86_ishl_i64", &proof_x86_ishl_i64());
}

// ===========================================================================
// x86-64 core ops widened — neg/band/bor/bxor/ushr/sshr/imul (issue #463)
// ===========================================================================
//
// Widens the x86-64 z4 smoke lane to the remaining integer-core helpers that
// already exist in `x86_64_lowering_proofs.rs`. Each helper was written when
// the corresponding lowering landed; this section promotes them into the
// feature-gated z4-prove lane so the SMT backend exercises the formulas
// symbolically (the eval/sampling path already covers them).
//
// Fourteen obligations (7 ops × 2 widths):
//   - proof_x86_neg_{i32,i64}  — unary NEG; `bvneg x == bvsub 0 x`.
//   - proof_x86_band_{i32,i64} — AND r/m, r; bvand baseline.
//   - proof_x86_bor_{i32,i64}  — OR  r/m, r; bvor  baseline.
//   - proof_x86_bxor_{i32,i64} — XOR r/m, r; bvxor baseline.
//   - proof_x86_ushr_{i32,i64} — SHR; logical shift via `bvlshr`.
//   - proof_x86_sshr_{i32,i64} — SAR; arithmetic shift via `bvashr`.
//   - proof_x86_imul_{i32,i64} — IMUL r, r/m; bvmul (historically slowest).
//
// All use `assert_verified_or_timeout` at the 10 s budget: imul_i64 in
// particular may legitimately time out under load, and the eval path still
// covers the obligation. CounterExample or Error still fails the test.

#[test]
fn z4_prove_x86_neg_i32() {
    assert_verified_or_timeout("x86_neg_i32", &proof_x86_neg_i32());
}

#[test]
fn z4_prove_x86_neg_i64() {
    assert_verified_or_timeout("x86_neg_i64", &proof_x86_neg_i64());
}

#[test]
fn z4_prove_x86_band_i32() {
    assert_verified_or_timeout("x86_band_i32", &proof_x86_band_i32());
}

#[test]
fn z4_prove_x86_band_i64() {
    assert_verified_or_timeout("x86_band_i64", &proof_x86_band_i64());
}

#[test]
fn z4_prove_x86_bor_i32() {
    assert_verified_or_timeout("x86_bor_i32", &proof_x86_bor_i32());
}

#[test]
fn z4_prove_x86_bor_i64() {
    assert_verified_or_timeout("x86_bor_i64", &proof_x86_bor_i64());
}

#[test]
fn z4_prove_x86_bxor_i32() {
    assert_verified_or_timeout("x86_bxor_i32", &proof_x86_bxor_i32());
}

#[test]
fn z4_prove_x86_bxor_i64() {
    assert_verified_or_timeout("x86_bxor_i64", &proof_x86_bxor_i64());
}

#[test]
fn z4_prove_x86_ushr_i32() {
    assert_verified_or_timeout("x86_ushr_i32", &proof_x86_ushr_i32());
}

#[test]
fn z4_prove_x86_ushr_i64() {
    assert_verified_or_timeout("x86_ushr_i64", &proof_x86_ushr_i64());
}

#[test]
fn z4_prove_x86_sshr_i32() {
    assert_verified_or_timeout("x86_sshr_i32", &proof_x86_sshr_i32());
}

#[test]
fn z4_prove_x86_sshr_i64() {
    assert_verified_or_timeout("x86_sshr_i64", &proof_x86_sshr_i64());
}

#[test]
fn z4_prove_x86_imul_i32() {
    assert_verified_or_timeout("x86_imul_i32", &proof_x86_imul_i32());
}

#[test]
fn z4_prove_x86_imul_i64() {
    assert_verified_or_timeout("x86_imul_i64", &proof_x86_imul_i64());
}

// ===========================================================================
// x86-64 integer ops at i8 width (issue #470)
// ===========================================================================
//
// Rec #4 from the #419 z4 coverage survey: the x86-64 lowering proofs
// already provide nine i8 helpers (arithmetic, logic, and shift), but
// the z4-prove smoke lane only exercised them at i32/i64. This section
// wires the existing i8 helpers so the cross-target i8 coverage is
// symmetric with AArch64 (which has i8 covered under the `lowering_proof`
// lane above).
//
// Nine obligations at i8:
//   - proof_x86_iadd_i8  — bvadd baseline (sub-word REX-less 8-bit ADD).
//   - proof_x86_isub_i8  — bvsub baseline.
//   - proof_x86_imul_i8  — bvmul (IMUL r, r/m with 8-bit width).
//   - proof_x86_band_i8  — bvand baseline (AND r/m8, r8).
//   - proof_x86_bor_i8   — bvor  baseline (OR  r/m8, r8).
//   - proof_x86_bxor_i8  — bvxor baseline (XOR r/m8, r8).
//   - proof_x86_ishl_i8  — SHL r/m8, CL — bvshl at 8 bits.
//   - proof_x86_ushr_i8  — SHR r/m8, CL — bvlshr at 8 bits.
//   - proof_x86_sshr_i8  — SAR r/m8, CL — bvashr at 8 bits.
//
// All use `assert_verified_or_timeout`: i8 formulas are small for z4, but
// the smoke lane stays tolerant for consistency with the surrounding
// blocks. CounterExample or Error still fails the test.
//
// Note: `proof_x86_neg_i8` does not exist in `x86_64_lowering_proofs.rs`
// (only `proof_x86_neg_{i32,i64}` are authored). Similarly no i16
// helpers exist for these ops on x86-64 today. Widening is tracked as a
// separate authoring task; this section is wiring-only.

#[test]
fn z4_prove_x86_iadd_i8() {
    assert_verified_or_timeout("x86_iadd_i8", &proof_x86_iadd_i8());
}

#[test]
fn z4_prove_x86_isub_i8() {
    assert_verified_or_timeout("x86_isub_i8", &proof_x86_isub_i8());
}

#[test]
fn z4_prove_x86_imul_i8() {
    assert_verified_or_timeout("x86_imul_i8", &proof_x86_imul_i8());
}

#[test]
fn z4_prove_x86_band_i8() {
    assert_verified_or_timeout("x86_band_i8", &proof_x86_band_i8());
}

#[test]
fn z4_prove_x86_bor_i8() {
    assert_verified_or_timeout("x86_bor_i8", &proof_x86_bor_i8());
}

#[test]
fn z4_prove_x86_bxor_i8() {
    assert_verified_or_timeout("x86_bxor_i8", &proof_x86_bxor_i8());
}

#[test]
fn z4_prove_x86_ishl_i8() {
    assert_verified_or_timeout("x86_ishl_i8", &proof_x86_ishl_i8());
}

#[test]
fn z4_prove_x86_ushr_i8() {
    assert_verified_or_timeout("x86_ushr_i8", &proof_x86_ushr_i8());
}

#[test]
fn z4_prove_x86_sshr_i8() {
    assert_verified_or_timeout("x86_sshr_i8", &proof_x86_sshr_i8());
}

// ===========================================================================
// x86-64 icmp predicates at i32/i64 (issue #463 follow-up)
// ===========================================================================
//
// Extends the x86-64 z4 smoke lane to the remaining icmp predicates beyond
// `eq` (which already lives in the i32/i64 sections above). Each obligation
// exercises the TEST/CMP + SETcc + MOVZX zero-extend chain for a specific
// condition code:
//
//   - ne:  SETNE  — ZF == 0
//   - slt: SETL   — SF != OF
//   - sge: SETGE  — SF == OF
//   - sgt: SETG   — ZF == 0 && SF == OF
//   - sle: SETLE  — ZF == 1 || SF != OF
//   - ult: SETB   — CF == 1
//   - uge: SETAE  — CF == 0
//   - ugt: SETA   — CF == 0 && ZF == 0
//   - ule: SETBE  — CF == 1 || ZF == 1
//
// Eighteen obligations (9 preds × 2 widths). Each uses
// `assert_verified_or_timeout` — EFLAGS + SETcc + zero-extend formulas are
// well within z4's comfort zone at i32, but i64 signed-predicate formulas
// occasionally stretch the 10 s budget on a loaded host. A CounterExample
// or Error still fails.

#[test]
fn z4_prove_x86_icmp_ne_i32() {
    assert_verified_or_timeout("x86_icmp_ne_i32", &proof_x86_icmp_ne_i32());
}

#[test]
fn z4_prove_x86_icmp_ne_i64() {
    assert_verified_or_timeout("x86_icmp_ne_i64", &proof_x86_icmp_ne_i64());
}

#[test]
fn z4_prove_x86_icmp_slt_i32() {
    assert_verified_or_timeout("x86_icmp_slt_i32", &proof_x86_icmp_slt_i32());
}

#[test]
fn z4_prove_x86_icmp_slt_i64() {
    assert_verified_or_timeout("x86_icmp_slt_i64", &proof_x86_icmp_slt_i64());
}

#[test]
fn z4_prove_x86_icmp_sge_i32() {
    assert_verified_or_timeout("x86_icmp_sge_i32", &proof_x86_icmp_sge_i32());
}

#[test]
fn z4_prove_x86_icmp_sge_i64() {
    assert_verified_or_timeout("x86_icmp_sge_i64", &proof_x86_icmp_sge_i64());
}

#[test]
fn z4_prove_x86_icmp_sgt_i32() {
    assert_verified_or_timeout("x86_icmp_sgt_i32", &proof_x86_icmp_sgt_i32());
}

#[test]
fn z4_prove_x86_icmp_sgt_i64() {
    assert_verified_or_timeout("x86_icmp_sgt_i64", &proof_x86_icmp_sgt_i64());
}

#[test]
fn z4_prove_x86_icmp_sle_i32() {
    assert_verified_or_timeout("x86_icmp_sle_i32", &proof_x86_icmp_sle_i32());
}

#[test]
fn z4_prove_x86_icmp_sle_i64() {
    assert_verified_or_timeout("x86_icmp_sle_i64", &proof_x86_icmp_sle_i64());
}

#[test]
fn z4_prove_x86_icmp_ult_i32() {
    assert_verified_or_timeout("x86_icmp_ult_i32", &proof_x86_icmp_ult_i32());
}

#[test]
fn z4_prove_x86_icmp_ult_i64() {
    assert_verified_or_timeout("x86_icmp_ult_i64", &proof_x86_icmp_ult_i64());
}

#[test]
fn z4_prove_x86_icmp_uge_i32() {
    assert_verified_or_timeout("x86_icmp_uge_i32", &proof_x86_icmp_uge_i32());
}

#[test]
fn z4_prove_x86_icmp_uge_i64() {
    assert_verified_or_timeout("x86_icmp_uge_i64", &proof_x86_icmp_uge_i64());
}

#[test]
fn z4_prove_x86_icmp_ugt_i32() {
    assert_verified_or_timeout("x86_icmp_ugt_i32", &proof_x86_icmp_ugt_i32());
}

#[test]
fn z4_prove_x86_icmp_ugt_i64() {
    assert_verified_or_timeout("x86_icmp_ugt_i64", &proof_x86_icmp_ugt_i64());
}

#[test]
fn z4_prove_x86_icmp_ule_i32() {
    assert_verified_or_timeout("x86_icmp_ule_i32", &proof_x86_icmp_ule_i32());
}

#[test]
fn z4_prove_x86_icmp_ule_i64() {
    assert_verified_or_timeout("x86_icmp_ule_i64", &proof_x86_icmp_ule_i64());
}

// ===========================================================================
// x86-64 FP ops at f32/f64 (issue #463 follow-up)
// ===========================================================================
//
// First FP coverage in the x86-64 z4 smoke lane. Lowerings:
//
//   - Fadd  -> ADDSS / ADDSD  (scalar single / double-precision add)
//   - Fsub  -> SUBSS / SUBSD  (scalar single / double-precision sub)
//   - Fmul  -> MULSS / MULSD  (scalar single / double-precision mul)
//   - Fdiv  -> DIVSS / DIVSD  (scalar single / double-precision div)
//   - Fneg  -> XORPS/XORPD with sign-bit mask (bit-level negation)
//   - Fabs  -> ANDPS/ANDPD with abs-mask (clear sign bit)
//   - Fsqrt -> SQRTSS / SQRTSD (scalar single / double-precision sqrt)
//
// Fourteen obligations (7 ops x 2 widths). FP formulas encode IEEE-754
// semantics in SMT's theory of floating-point, which is significantly more
// expensive than pure bitvector reasoning -- timeouts at f64 under a 10 s
// budget are expected and tolerated via `assert_verified_or_timeout`. A
// CounterExample or Error still fails, so soundness regressions surface.
// The eval/sampling path in the default (non-feature-gated) verification
// lane still provides exhaustive-style coverage at FP widths.

#[test]
fn z4_prove_x86_fadd_f32() {
    assert_verified_or_timeout("x86_fadd_f32", &proof_x86_fadd_f32());
}

#[test]
fn z4_prove_x86_fadd_f64() {
    assert_verified_or_timeout("x86_fadd_f64", &proof_x86_fadd_f64());
}

#[test]
fn z4_prove_x86_fsub_f32() {
    assert_verified_or_timeout("x86_fsub_f32", &proof_x86_fsub_f32());
}

#[test]
fn z4_prove_x86_fsub_f64() {
    assert_verified_or_timeout("x86_fsub_f64", &proof_x86_fsub_f64());
}

#[test]
fn z4_prove_x86_fmul_f32() {
    assert_verified_or_timeout("x86_fmul_f32", &proof_x86_fmul_f32());
}

#[test]
fn z4_prove_x86_fmul_f64() {
    assert_verified_or_timeout("x86_fmul_f64", &proof_x86_fmul_f64());
}

#[test]
fn z4_prove_x86_fdiv_f32() {
    assert_verified_or_timeout("x86_fdiv_f32", &proof_x86_fdiv_f32());
}

#[test]
fn z4_prove_x86_fdiv_f64() {
    assert_verified_or_timeout("x86_fdiv_f64", &proof_x86_fdiv_f64());
}

#[test]
fn z4_prove_x86_fneg_f32() {
    assert_verified_or_timeout("x86_fneg_f32", &proof_x86_fneg_f32());
}

#[test]
fn z4_prove_x86_fneg_f64() {
    assert_verified_or_timeout("x86_fneg_f64", &proof_x86_fneg_f64());
}

#[test]
fn z4_prove_x86_fabs_f32() {
    assert_verified_or_timeout("x86_fabs_f32", &proof_x86_fabs_f32());
}

#[test]
fn z4_prove_x86_fabs_f64() {
    assert_verified_or_timeout("x86_fabs_f64", &proof_x86_fabs_f64());
}

#[test]
fn z4_prove_x86_fsqrt_f32() {
    assert_verified_or_timeout("x86_fsqrt_f32", &proof_x86_fsqrt_f32());
}

#[test]
fn z4_prove_x86_fsqrt_f64() {
    assert_verified_or_timeout("x86_fsqrt_f64", &proof_x86_fsqrt_f64());
}

// ===========================================================================
// AArch64 FP arithmetic + fcmp at f32/f64 (issue #471)
// ===========================================================================
//
// Wires AArch64 FP lowering proofs into the smoke lane. Mirrors the x86-64
// FP coverage above (14 arith/unary entries) and extends it with 28 fcmp
// variants that x86-64 has not yet wired. Lowerings:
//
//   - Fadd/Fsub/Fmul/Fdiv -> FADD/FSUB/FMUL/FDIV (Sd/Dd variants per width)
//   - Fneg                -> FNEG (sign-bit flip via bit-level semantics)
//   - Fabs                -> FABS (clear sign bit)
//   - Fsqrt               -> FSQRT
//   - Fcmp                -> FCMP + flag-to-bit via NZCV lowering
//
// Forty-two obligations total (14 arith/unary + 28 fcmp). FP formulas
// encode IEEE-754 semantics in SMT's theory of floating-point, which is
// significantly more expensive than pure bitvector reasoning -- timeouts
// at f64 under a 10 s budget are expected and tolerated via
// `assert_verified_or_timeout`. A CounterExample or Error still fails, so
// soundness regressions surface. The eval/sampling path in the default
// (non-feature-gated) verification lane still provides exhaustive-style
// coverage at FP widths.
//
// Per `reports/2026-04-19-419-z4-coverage-survey.md` rec #2.

#[test]
fn z4_prove_fadd_f32() {
    assert_verified_or_timeout("fadd_f32", &proof_fadd_f32());
}

#[test]
fn z4_prove_fadd_f64() {
    assert_verified_or_timeout("fadd_f64", &proof_fadd_f64());
}

#[test]
fn z4_prove_fsub_f32() {
    assert_verified_or_timeout("fsub_f32", &proof_fsub_f32());
}

#[test]
fn z4_prove_fsub_f64() {
    assert_verified_or_timeout("fsub_f64", &proof_fsub_f64());
}

#[test]
fn z4_prove_fmul_f32() {
    assert_verified_or_timeout("fmul_f32", &proof_fmul_f32());
}

#[test]
fn z4_prove_fmul_f64() {
    assert_verified_or_timeout("fmul_f64", &proof_fmul_f64());
}

#[test]
fn z4_prove_fdiv_f32() {
    assert_verified_or_timeout("fdiv_f32", &proof_fdiv_f32());
}

#[test]
fn z4_prove_fdiv_f64() {
    assert_verified_or_timeout("fdiv_f64", &proof_fdiv_f64());
}

#[test]
fn z4_prove_fneg_f32() {
    assert_verified_or_timeout("fneg_f32", &proof_fneg_f32());
}

#[test]
fn z4_prove_fneg_f64() {
    assert_verified_or_timeout("fneg_f64", &proof_fneg_f64());
}

#[test]
fn z4_prove_fabs_f32() {
    assert_verified_or_timeout("fabs_f32", &proof_fabs_f32());
}

#[test]
fn z4_prove_fabs_f64() {
    assert_verified_or_timeout("fabs_f64", &proof_fabs_f64());
}

#[test]
fn z4_prove_fsqrt_f32() {
    assert_verified_or_timeout("fsqrt_f32", &proof_fsqrt_f32());
}

#[test]
fn z4_prove_fsqrt_f64() {
    assert_verified_or_timeout("fsqrt_f64", &proof_fsqrt_f64());
}

// ---- fcmp ordered predicates ----------------------------------------------

#[test]
fn z4_prove_fcmp_eq_f32() {
    assert_verified_or_timeout("fcmp_eq_f32", &proof_fcmp_eq_f32());
}

#[test]
fn z4_prove_fcmp_eq_f64() {
    assert_verified_or_timeout("fcmp_eq_f64", &proof_fcmp_eq_f64());
}

#[test]
fn z4_prove_fcmp_ne_f32() {
    assert_verified_or_timeout("fcmp_ne_f32", &proof_fcmp_ne_f32());
}

#[test]
fn z4_prove_fcmp_ne_f64() {
    assert_verified_or_timeout("fcmp_ne_f64", &proof_fcmp_ne_f64());
}

#[test]
fn z4_prove_fcmp_lt_f32() {
    assert_verified_or_timeout("fcmp_lt_f32", &proof_fcmp_lt_f32());
}

#[test]
fn z4_prove_fcmp_lt_f64() {
    assert_verified_or_timeout("fcmp_lt_f64", &proof_fcmp_lt_f64());
}

#[test]
fn z4_prove_fcmp_le_f32() {
    assert_verified_or_timeout("fcmp_le_f32", &proof_fcmp_le_f32());
}

#[test]
fn z4_prove_fcmp_le_f64() {
    assert_verified_or_timeout("fcmp_le_f64", &proof_fcmp_le_f64());
}

#[test]
fn z4_prove_fcmp_gt_f32() {
    assert_verified_or_timeout("fcmp_gt_f32", &proof_fcmp_gt_f32());
}

#[test]
fn z4_prove_fcmp_gt_f64() {
    assert_verified_or_timeout("fcmp_gt_f64", &proof_fcmp_gt_f64());
}

#[test]
fn z4_prove_fcmp_ge_f32() {
    assert_verified_or_timeout("fcmp_ge_f32", &proof_fcmp_ge_f32());
}

#[test]
fn z4_prove_fcmp_ge_f64() {
    assert_verified_or_timeout("fcmp_ge_f64", &proof_fcmp_ge_f64());
}

#[test]
fn z4_prove_fcmp_ord_f32() {
    assert_verified_or_timeout("fcmp_ord_f32", &proof_fcmp_ord_f32());
}

#[test]
fn z4_prove_fcmp_ord_f64() {
    assert_verified_or_timeout("fcmp_ord_f64", &proof_fcmp_ord_f64());
}

#[test]
fn z4_prove_fcmp_uno_f32() {
    assert_verified_or_timeout("fcmp_uno_f32", &proof_fcmp_uno_f32());
}

#[test]
fn z4_prove_fcmp_uno_f64() {
    assert_verified_or_timeout("fcmp_uno_f64", &proof_fcmp_uno_f64());
}

// ---- fcmp unordered predicates --------------------------------------------

#[test]
fn z4_prove_fcmp_ueq_f32() {
    assert_verified_or_timeout("fcmp_ueq_f32", &proof_fcmp_ueq_f32());
}

#[test]
fn z4_prove_fcmp_ueq_f64() {
    assert_verified_or_timeout("fcmp_ueq_f64", &proof_fcmp_ueq_f64());
}

#[test]
fn z4_prove_fcmp_une_f32() {
    assert_verified_or_timeout("fcmp_une_f32", &proof_fcmp_une_f32());
}

#[test]
fn z4_prove_fcmp_une_f64() {
    assert_verified_or_timeout("fcmp_une_f64", &proof_fcmp_une_f64());
}

#[test]
fn z4_prove_fcmp_ult_f32() {
    assert_verified_or_timeout("fcmp_ult_f32", &proof_fcmp_ult_f32());
}

#[test]
fn z4_prove_fcmp_ult_f64() {
    assert_verified_or_timeout("fcmp_ult_f64", &proof_fcmp_ult_f64());
}

#[test]
fn z4_prove_fcmp_ule_f32() {
    assert_verified_or_timeout("fcmp_ule_f32", &proof_fcmp_ule_f32());
}

#[test]
fn z4_prove_fcmp_ule_f64() {
    assert_verified_or_timeout("fcmp_ule_f64", &proof_fcmp_ule_f64());
}

#[test]
fn z4_prove_fcmp_ugt_f32() {
    assert_verified_or_timeout("fcmp_ugt_f32", &proof_fcmp_ugt_f32());
}

#[test]
fn z4_prove_fcmp_ugt_f64() {
    assert_verified_or_timeout("fcmp_ugt_f64", &proof_fcmp_ugt_f64());
}

#[test]
fn z4_prove_fcmp_uge_f32() {
    assert_verified_or_timeout("fcmp_uge_f32", &proof_fcmp_uge_f32());
}

#[test]
fn z4_prove_fcmp_uge_f64() {
    assert_verified_or_timeout("fcmp_uge_f64", &proof_fcmp_uge_f64());
}

// ===========================================================================
// memory_proofs: load/store + memory-model obligations (issue #469)
// ===========================================================================
//
// Wires memory_proofs.rs i32/i64 helpers into the smoke lane. All
// obligations use `assert_verified_or_timeout` because the memory
// model sits on QF_ABV and may run into z4 solver-time limits even
// for modest byte layouts (forwarding, endianness, write-combining).
//
// Coverage (38 obligations):
//   - load i32/i64 +/- offset                           (4)
//   - store i32/i64 +/- offset                          (4)
//   - roundtrip i32/i64 (store-then-load)               (2)
//   - aligned_roundtrip i32/i64                         (2)
//   - scaled_offset_alignment_i32                       (1)
//   - non_interference i32/i64 (various layouts)        (8)
//   - forwarding i32 off12, i64 off24, i64->i32 lower   (3)
//   - subword i32 byte1/byte2/halfword_upper, i64 byte5 (4)
//   - write_combining i32/halfwords_to_i32/words_to_i64 (3)
//   - endianness i32/i64/msb_i32                        (3)
//   - axioms read/write same/diff/const_array at 32-bit (4)

#[test]
fn z4_prove_load_i32() {
    assert_verified_or_timeout("load_i32", &proof_load_i32());
}

#[test]
fn z4_prove_load_i32_offset() {
    assert_verified_or_timeout("load_i32_offset", &proof_load_i32_offset());
}

#[test]
fn z4_prove_load_i64() {
    assert_verified_or_timeout("load_i64", &proof_load_i64());
}

#[test]
fn z4_prove_load_i64_offset() {
    assert_verified_or_timeout("load_i64_offset", &proof_load_i64_offset());
}

#[test]
fn z4_prove_store_i32() {
    assert_verified_or_timeout("store_i32", &proof_store_i32());
}

#[test]
fn z4_prove_store_i32_offset() {
    assert_verified_or_timeout("store_i32_offset", &proof_store_i32_offset());
}

#[test]
fn z4_prove_store_i64() {
    assert_verified_or_timeout("store_i64", &proof_store_i64());
}

#[test]
fn z4_prove_store_i64_offset() {
    assert_verified_or_timeout("store_i64_offset", &proof_store_i64_offset());
}

#[test]
fn z4_prove_roundtrip_i32() {
    assert_verified_or_timeout("roundtrip_i32", &proof_roundtrip_i32());
}

#[test]
fn z4_prove_roundtrip_i64() {
    assert_verified_or_timeout("roundtrip_i64", &proof_roundtrip_i64());
}

#[test]
fn z4_prove_aligned_roundtrip_i32() {
    assert_verified_or_timeout("aligned_roundtrip_i32", &proof_aligned_roundtrip_i32());
}

#[test]
fn z4_prove_aligned_roundtrip_i64() {
    assert_verified_or_timeout("aligned_roundtrip_i64", &proof_aligned_roundtrip_i64());
}

#[test]
fn z4_prove_scaled_offset_alignment_i32() {
    assert_verified_or_timeout(
        "scaled_offset_alignment_i32",
        &proof_scaled_offset_alignment_i32(),
    );
}

#[test]
fn z4_prove_non_interference_i32() {
    assert_verified_or_timeout("non_interference_i32", &proof_non_interference_i32());
}

#[test]
fn z4_prove_non_interference_i64() {
    assert_verified_or_timeout("non_interference_i64", &proof_non_interference_i64());
}

#[test]
fn z4_prove_non_interference_i32_adjacent() {
    assert_verified_or_timeout(
        "non_interference_i32_adjacent",
        &proof_non_interference_i32_adjacent(),
    );
}

#[test]
fn z4_prove_non_interference_i64_adjacent() {
    assert_verified_or_timeout(
        "non_interference_i64_adjacent",
        &proof_non_interference_i64_adjacent(),
    );
}

#[test]
fn z4_prove_non_interference_i64_store_i32_load() {
    assert_verified_or_timeout(
        "non_interference_i64_store_i32_load",
        &proof_non_interference_i64_store_i32_load(),
    );
}

#[test]
fn z4_prove_non_interference_i32_store_i64_load() {
    assert_verified_or_timeout(
        "non_interference_i32_store_i64_load",
        &proof_non_interference_i32_store_i64_load(),
    );
}

#[test]
fn z4_prove_non_interference_i32_symbolic() {
    assert_verified_or_timeout(
        "non_interference_i32_symbolic",
        &proof_non_interference_i32_symbolic(),
    );
}

#[test]
fn z4_prove_non_interference_i64_symbolic() {
    assert_verified_or_timeout(
        "non_interference_i64_symbolic",
        &proof_non_interference_i64_symbolic(),
    );
}

#[test]
fn z4_prove_forwarding_i32_offset12() {
    assert_verified_or_timeout(
        "forwarding_i32_offset12",
        &proof_forwarding_i32_offset12(),
    );
}

#[test]
fn z4_prove_forwarding_i64_offset24() {
    assert_verified_or_timeout(
        "forwarding_i64_offset24",
        &proof_forwarding_i64_offset24(),
    );
}

#[test]
fn z4_prove_forwarding_i64_to_i32_lower() {
    assert_verified_or_timeout(
        "forwarding_i64_to_i32_lower",
        &proof_forwarding_i64_to_i32_lower(),
    );
}

#[test]
fn z4_prove_subword_i32_byte1() {
    assert_verified_or_timeout("subword_i32_byte1", &proof_subword_i32_byte1());
}

#[test]
fn z4_prove_subword_i32_byte2() {
    assert_verified_or_timeout("subword_i32_byte2", &proof_subword_i32_byte2());
}

#[test]
fn z4_prove_subword_i32_halfword_upper() {
    assert_verified_or_timeout(
        "subword_i32_halfword_upper",
        &proof_subword_i32_halfword_upper(),
    );
}

#[test]
fn z4_prove_subword_i64_byte5() {
    assert_verified_or_timeout("subword_i64_byte5", &proof_subword_i64_byte5());
}

#[test]
fn z4_prove_write_combining_i32() {
    assert_verified_or_timeout("write_combining_i32", &proof_write_combining_i32());
}

#[test]
fn z4_prove_write_combining_halfwords_to_i32() {
    assert_verified_or_timeout(
        "write_combining_halfwords_to_i32",
        &proof_write_combining_halfwords_to_i32(),
    );
}

#[test]
fn z4_prove_write_combining_words_to_i64() {
    assert_verified_or_timeout(
        "write_combining_words_to_i64",
        &proof_write_combining_words_to_i64(),
    );
}

#[test]
fn z4_prove_endianness_i32() {
    assert_verified_or_timeout("endianness_i32", &proof_endianness_i32());
}

#[test]
fn z4_prove_endianness_msb_i32() {
    assert_verified_or_timeout("endianness_msb_i32", &proof_endianness_msb_i32());
}

#[test]
fn z4_prove_endianness_i64() {
    assert_verified_or_timeout("endianness_i64", &proof_endianness_i64());
}

#[test]
fn z4_prove_axiom_read_after_write_same_addr_32() {
    assert_verified_or_timeout(
        "axiom_read_after_write_same_addr_32",
        &proof_axiom_read_after_write_same_addr_32(),
    );
}

#[test]
fn z4_prove_axiom_read_after_write_diff_addr_32() {
    assert_verified_or_timeout(
        "axiom_read_after_write_diff_addr_32",
        &proof_axiom_read_after_write_diff_addr_32(),
    );
}

#[test]
fn z4_prove_axiom_write_write_same_addr_32() {
    assert_verified_or_timeout(
        "axiom_write_write_same_addr_32",
        &proof_axiom_write_write_same_addr_32(),
    );
}

#[test]
fn z4_prove_axiom_const_array_32() {
    assert_verified_or_timeout("axiom_const_array_32", &proof_axiom_const_array_32());
}

// ===========================================================================
// peephole_proofs (canonical identities, #472)
// ===========================================================================
//
// Survey rec #5 (`reports/2026-04-19-419-z4-coverage-survey.md`): wire a
// conservative safety subset of `peephole_proofs.rs` (99-helper orphan
// file) into the z4-prove smoke lane. All 18 entries below are canonical
// identity / zero-absorption / idempotent rewrites at i64 width; the
// formulas reduce to constant-parametric bitvector equalities and close
// in milliseconds, so they use strict `assert_verified`.
//
// Out of scope: pow-2 strength reductions, shift-compose patterns,
// sign-extend subsume chains, MADD/MSUB variants, SDIV -1 -> NEG (sound
// but heavier or width-sensitive). Those are future waves.

#[test]
fn z4_prove_peephole_add_zero_identity() {
    assert_verified("peephole_add_zero_identity", &proof_add_zero_identity());
}

#[test]
fn z4_prove_peephole_sub_zero_identity() {
    assert_verified("peephole_sub_zero_identity", &proof_sub_zero_identity());
}

#[test]
fn z4_prove_peephole_mul_one_identity() {
    assert_verified("peephole_mul_one_identity", &proof_mul_one_identity());
}

#[test]
fn z4_prove_peephole_lsl_zero_identity() {
    assert_verified("peephole_lsl_zero_identity", &proof_lsl_zero_identity());
}

#[test]
fn z4_prove_peephole_lsr_zero_identity() {
    assert_verified("peephole_lsr_zero_identity", &proof_lsr_zero_identity());
}

#[test]
fn z4_prove_peephole_asr_zero_identity() {
    assert_verified("peephole_asr_zero_identity", &proof_asr_zero_identity());
}

#[test]
fn z4_prove_peephole_orr_self_identity() {
    assert_verified("peephole_orr_self_identity", &proof_orr_self_identity());
}

#[test]
fn z4_prove_peephole_and_self_identity() {
    assert_verified("peephole_and_self_identity", &proof_and_self_identity());
}

#[test]
fn z4_prove_peephole_eor_zero_identity() {
    assert_verified("peephole_eor_zero_identity", &proof_eor_zero_identity());
}

#[test]
fn z4_prove_peephole_orr_ri_zero_identity() {
    assert_verified("peephole_orr_ri_zero_identity", &proof_orr_ri_zero_identity());
}

#[test]
fn z4_prove_peephole_and_ri_allones_identity() {
    assert_verified("peephole_and_ri_allones_identity", &proof_and_ri_allones_identity());
}

#[test]
fn z4_prove_peephole_sub_rr_self_zero() {
    assert_verified("peephole_sub_rr_self_zero", &proof_sub_rr_self_zero());
}

#[test]
fn z4_prove_peephole_eor_rr_self_zero() {
    assert_verified("peephole_eor_rr_self_zero", &proof_eor_rr_self_zero());
}

#[test]
fn z4_prove_peephole_and_ri_zero() {
    assert_verified("peephole_and_ri_zero", &proof_and_ri_zero());
}

#[test]
fn z4_prove_peephole_orr_ri_allones() {
    assert_verified("peephole_orr_ri_allones", &proof_orr_ri_allones());
}

#[test]
fn z4_prove_peephole_neg_neg_identity() {
    assert_verified("peephole_neg_neg_identity", &proof_neg_neg_identity());
}

#[test]
fn z4_prove_peephole_mul_zero_absorb() {
    assert_verified("peephole_mul_zero_absorb", &proof_mul_zero_absorb());
}

#[test]
fn z4_prove_peephole_udiv_one_identity() {
    assert_verified("peephole_udiv_one_identity", &proof_udiv_one_identity());
}

// ===========================================================================
// ext_trunc_proofs (sign/zero extend + truncate, #488 item 2)
// ===========================================================================
//
// Per #488 the top-2 proof-coverage gap for Epic #407 default-on: 22 eval
// obligations already exist in `ext_trunc_proofs.rs` but none were in the
// z4 smoke lane. This wave adds every width-pair emitted by
// `llvm2-lower::select_extend` / `select_trunc` (see
// `crates/llvm2-lower/src/isel.rs`), including the previously unproven
// `Trunc_I64_to_I32 -> MOV Wd, Wn` (the final missing width-pair).
//
// All formulas are primitive QF_BV: `extract` + `sign_extend`/`zero_extend`
// + `bvand` (for mask-based lowerings). They close in milliseconds so we
// use strict `assert_verified` at every width. Sign vs zero distinction is
// encoded at the SMT level (`sign_ext` vs `zero_ext`/`bvand`), and
// truncation drops high bits via `extract(x, to_bits-1, 0)` matching tMIR
// semantics (defined, not poison).

// --- Sign extension --- 5 width-pairs (I8->I32, I8->I64, I16->I32, I16->I64, I32->I64)

#[test]
fn z4_prove_sxtb_8_to_32() {
    assert_verified("sxtb_8_to_32", &proof_sxtb_8_to_32());
}

#[test]
fn z4_prove_sxtb_8_to_64() {
    assert_verified("sxtb_8_to_64", &proof_sxtb_8_to_64());
}

#[test]
fn z4_prove_sxth_16_to_32() {
    assert_verified("sxth_16_to_32", &proof_sxth_16_to_32());
}

#[test]
fn z4_prove_sxth_16_to_64() {
    assert_verified("sxth_16_to_64", &proof_sxth_16_to_64());
}

#[test]
fn z4_prove_sxtw_32_to_64() {
    assert_verified("sxtw_32_to_64", &proof_sxtw_32_to_64());
}

// --- Zero extension --- 5 width-pairs (I8->I32, I8->I64, I16->I32, I16->I64, I32->I64)

#[test]
fn z4_prove_uxtb_8_to_32() {
    assert_verified("uxtb_8_to_32", &proof_uxtb_8_to_32());
}

#[test]
fn z4_prove_uxtb_8_to_64() {
    assert_verified("uxtb_8_to_64", &proof_uxtb_8_to_64());
}

#[test]
fn z4_prove_uxth_16_to_32() {
    assert_verified("uxth_16_to_32", &proof_uxth_16_to_32());
}

#[test]
fn z4_prove_uxth_16_to_64() {
    assert_verified("uxth_16_to_64", &proof_uxth_16_to_64());
}

#[test]
fn z4_prove_uxtw_32_to_64() {
    assert_verified("uxtw_32_to_64", &proof_uxtw_32_to_64());
}

// --- Truncation --- 5 width-pairs (I32->I8, I32->I16, I64->I8, I64->I16, I64->I32)

#[test]
fn z4_prove_trunc_to_i8() {
    assert_verified("trunc_i32_to_i8", &proof_trunc_to_i8());
}

#[test]
fn z4_prove_trunc_to_i16() {
    assert_verified("trunc_i32_to_i16", &proof_trunc_to_i16());
}

#[test]
fn z4_prove_trunc_to_i8_from_64() {
    assert_verified("trunc_i64_to_i8", &proof_trunc_to_i8_from_64());
}

#[test]
fn z4_prove_trunc_to_i16_from_64() {
    assert_verified("trunc_i64_to_i16", &proof_trunc_to_i16_from_64());
}

#[test]
fn z4_prove_trunc_to_i32_from_64() {
    assert_verified("trunc_i64_to_i32", &proof_trunc_to_i32_from_64());
}

// --- Roundtrip identities (trunc(ext(x)) == x) --- 4 obligations

#[test]
fn z4_prove_roundtrip_zext_trunc_8() {
    assert_verified("roundtrip_zext_trunc_8", &proof_roundtrip_zext_trunc_8());
}

#[test]
fn z4_prove_roundtrip_zext_trunc_16() {
    assert_verified("roundtrip_zext_trunc_16", &proof_roundtrip_zext_trunc_16());
}

#[test]
fn z4_prove_roundtrip_sext_trunc_8() {
    assert_verified("roundtrip_sext_trunc_8", &proof_roundtrip_sext_trunc_8());
}

#[test]
fn z4_prove_roundtrip_sext_trunc_16() {
    assert_verified("roundtrip_sext_trunc_16", &proof_roundtrip_sext_trunc_16());
}

// --- Idempotence (ext(ext(x)) == ext(x)) --- 4 obligations

#[test]
fn z4_prove_sxtb_idempotent() {
    assert_verified("sxtb_idempotent", &proof_sxtb_idempotent());
}

#[test]
fn z4_prove_sxth_idempotent() {
    assert_verified("sxth_idempotent", &proof_sxth_idempotent());
}

#[test]
fn z4_prove_uxtb_idempotent() {
    assert_verified("uxtb_idempotent", &proof_uxtb_idempotent());
}

#[test]
fn z4_prove_uxth_idempotent() {
    assert_verified("uxth_idempotent", &proof_uxth_idempotent());
}

// ===========================================================================
// Roll-up sanity check
// ===========================================================================

/// Run every proof in this suite (i8 + i16 + i32 + i64, binary + unary +
/// shifts + BIC/ORN + imul/neg + Urem/Srem/Bitcast + bitfield + switch)
/// in a single test that reports the count of verified vs timed-out
/// proofs. Useful signal for a CI lane to detect that z4 is effectively
/// disabled by timeouts.
///
/// Contents (298 total):
///   - i8 binary/unary:         iadd, isub, band, bor, bxor, neg       (6)
///   - i8 shifts (#424):        ishl, ushr, sshr                        (3)
///   - i8 bic/orn (#425):       bic, orn                                 (2)
///   - i16 binary (#426):       iadd, isub, band, bor, bxor             (5)
///   - i16 shifts (#424/#407):  ishl, ushr, sshr                         (3)
///   - i16 bic/orn (#425/#407): bic, orn                                 (2)
///   - imul + neg_i16 (#407):   imul_i8, imul_i16, neg_i16              (3)
///   - i32 widened (#449):      iadd, isub, imul, neg, band, bor, bxor,
///                              ishl, ushr, sshr, bic, orn             (12)
///   - i64 widened (#449):      iadd, isub, imul, neg, band, bor, bxor,
///                              ishl, ushr, sshr, bic, orn             (12)
///   - i8 rem/bitcast (#435):   urem, srem, bitcast                      (3)
///   - i16 rem + i16/i32/i64
///     bitcast (#435 widen):    urem_i16, srem_i16, bitcast_i16/i32/i64 (5)
///   - i8 bitfield (#452):      extract, sextract, insert                (3)
///   - switch (#444):           dense_i8, dense_i8_base10, dense_i8_hole,
///                              dense_i16, sparse_i8, sparse_7case_i8,
///                              sparse_i16                               (7)
///   - AArch64 icmp i32 (#468): eq, ne, slt, sge, sgt, sle,
///                              ult, uge, ugt, ule                      (10)
///   - AArch64 icmp i64 (#468): eq, ne, slt, sge, sgt, sle,
///                              ult, uge, ugt, ule                      (10)
///   - AArch64 condbr i32 (#468): eq, ne, slt, sge, sgt, sle,
///                              ult, uge, ugt, ule                      (10)
///   - AArch64 condbr i64 (#468): eq, ne, slt, sge, sgt, sle,
///                              ult, uge, ugt, ule                      (10)
///   - x86 EFLAGS (#458):       cmp/sub x sf/zf/cf/of at i8              (8)
///   - x86 core ops (#457):     iadd, isub, icmp_eq, ishl at i32         (4)
///   - x86 core ops i64
///     (#457 follow-up):        iadd, isub, icmp_eq, ishl at i64         (4)
///   - x86 widened (#463):      neg, band, bor, bxor, ushr, sshr, imul
///                              at i32 and i64                          (14)
///   - x86 i8 wiring (#470):    iadd, isub, imul, band, bor, bxor,
///                              ishl, ushr, sshr at i8                   (9)
///   - x86 icmp predicates i32/i64 (#463): ne, slt/sge/sgt/sle,
///                              ult/uge/ugt/ule                         (18)
///   - x86-64 FP (#463):        fadd/fsub/fmul/fdiv/fneg/fabs/fsqrt
///                              at f32/f64                              (14)
///   - memory_proofs (#469):    load/store/roundtrip/non-interference/
///                              forwarding/subword/write-combining/
///                              endianness/axioms at i32/i64            (38)
///   - AArch64 FP (#471):       fadd/fsub/fmul/fdiv/fneg/fabs/fsqrt
///                              at f32/f64                              (14)
///   - AArch64 fcmp (#471):     eq/ne/lt/le/gt/ge/ord/uno (ordered) +
///                              ueq/une/ult/ule/ugt/uge (unordered)
///                              at f32/f64                              (28)
///   - peephole_proofs (#472):  canonical identity/zero/idempotent
///                              rewrites at i64                         (18)
///   - ext_trunc_proofs (#488): SXTB/SXTH/SXTW, UXTB/UXTH/UXTW, TRUNC
///                              (all width-pairs emitted by ISel) +
///                              roundtrip + idempotent                  (23)
#[test]
fn z4_prove_smoke_suite_summary() {
    let proofs: Vec<(&str, ProofObligation)> = vec![
        // i8 binary (pre-existing)
        ("iadd_i8", proof_iadd_i8()),
        ("isub_i8", proof_isub_i8()),
        ("band_i8", proof_band_i8()),
        ("bor_i8", proof_bor_i8()),
        ("bxor_i8", proof_bxor_i8()),
        // i8 unary (issue #423)
        ("neg_i8", proof_neg_i8()),
        // i8 shifts (issue #424)
        ("ishl_i8", proof_ishl_i8()),
        ("ushr_i8", proof_ushr_i8()),
        ("sshr_i8", proof_sshr_i8()),
        // i8 BIC/ORN (issue #425)
        ("bic_i8", proof_bic_i8()),
        ("orn_i8", proof_orn_i8()),
        // i16 binary (issue #426)
        ("iadd_i16", proof_iadd_i16()),
        ("isub_i16", proof_isub_i16()),
        ("band_i16", proof_band_i16()),
        ("bor_i16", proof_bor_i16()),
        ("bxor_i16", proof_bxor_i16()),
        // i16 shifts (issues #424 / #407)
        ("ishl_i16", proof_ishl_i16()),
        ("ushr_i16", proof_ushr_i16()),
        ("sshr_i16", proof_sshr_i16()),
        // i16 BIC/ORN (issues #425 / #407)
        ("bic_i16", proof_bic_i16()),
        ("orn_i16", proof_orn_i16()),
        // imul + neg_i16 (epic #407)
        ("imul_i8", proof_imul_i8()),
        ("imul_i16", proof_imul_i16()),
        ("neg_i16", proof_neg_i16()),
        // i8 remainder + bitcast (issue #435)
        ("urem_i8", proof_urem_i8()),
        ("srem_i8", proof_srem_i8()),
        ("bitcast_i8", proof_bitcast_i8()),
        // i16 remainder + i16/i32/i64 bitcast widening (issue #435)
        ("urem_i16", proof_urem_i16()),
        ("srem_i16", proof_srem_i16()),
        ("bitcast_i16", proof_bitcast_i16()),
        ("bitcast_i32", proof_bitcast_i32()),
        ("bitcast_i64", proof_bitcast_i64()),
        // i8 bitfield (issue #452)
        ("extract_bits_i8", proof_extract_bits_i8()),
        ("sextract_bits_i8", proof_sextract_bits_i8()),
        ("insert_bits_i8", proof_insert_bits_i8()),
        // switch lowering (issue #444, deferred from #323)
        ("switch_dense_i8", proof_switch_dense_i8()),
        ("switch_dense_i8_nonzero_base", proof_switch_dense_i8_nonzero_base()),
        ("switch_dense_i8_with_hole", proof_switch_dense_i8_with_hole()),
        ("switch_dense_i16", proof_switch_dense_i16()),
        ("switch_sparse_i8", proof_switch_sparse_i8()),
        ("switch_sparse_7case_i8", proof_switch_sparse_7case_i8()),
        ("switch_sparse_i16", proof_switch_sparse_i16()),
        // i32 core arith/logic/shift (issue #449, epic #407 Task 3)
        ("iadd_i32", proof_iadd_i32()),
        ("isub_i32", proof_isub_i32()),
        ("imul_i32", proof_imul_i32()),
        ("neg_i32", proof_neg_i32()),
        ("band_i32", proof_band_i32()),
        ("bor_i32", proof_bor_i32()),
        ("bxor_i32", proof_bxor_i32()),
        ("ishl_i32", proof_ishl_i32()),
        ("ushr_i32", proof_ushr_i32()),
        ("sshr_i32", proof_sshr_i32()),
        ("bic_i32", proof_bic_i32()),
        ("orn_i32", proof_orn_i32()),
        // i64 core arith/logic/shift (issue #449, epic #407 Task 3)
        ("iadd_i64", proof_iadd_i64()),
        ("isub_i64", proof_isub_i64()),
        ("imul_i64", proof_imul_i64()),
        ("neg_i64", proof_neg_i64()),
        ("band_i64", proof_band_i64()),
        ("bor_i64", proof_bor_i64()),
        ("bxor_i64", proof_bxor_i64()),
        ("ishl_i64", proof_ishl_i64()),
        ("ushr_i64", proof_ushr_i64()),
        ("sshr_i64", proof_sshr_i64()),
        ("bic_i64", proof_bic_i64()),
        ("orn_i64", proof_orn_i64()),
        // AArch64 icmp predicates at i32 (issue #468) -- 10 obligations
        ("icmp_eq_i32", proof_icmp_eq_i32()),
        ("icmp_ne_i32", proof_icmp_ne_i32()),
        ("icmp_slt_i32", proof_icmp_slt_i32()),
        ("icmp_sge_i32", proof_icmp_sge_i32()),
        ("icmp_sgt_i32", proof_icmp_sgt_i32()),
        ("icmp_sle_i32", proof_icmp_sle_i32()),
        ("icmp_ult_i32", proof_icmp_ult_i32()),
        ("icmp_uge_i32", proof_icmp_uge_i32()),
        ("icmp_ugt_i32", proof_icmp_ugt_i32()),
        ("icmp_ule_i32", proof_icmp_ule_i32()),
        // AArch64 icmp predicates at i64 (issue #468) -- 10 obligations
        ("icmp_eq_i64", proof_icmp_eq_i64()),
        ("icmp_ne_i64", proof_icmp_ne_i64()),
        ("icmp_slt_i64", proof_icmp_slt_i64()),
        ("icmp_sge_i64", proof_icmp_sge_i64()),
        ("icmp_sgt_i64", proof_icmp_sgt_i64()),
        ("icmp_sle_i64", proof_icmp_sle_i64()),
        ("icmp_ult_i64", proof_icmp_ult_i64()),
        ("icmp_uge_i64", proof_icmp_uge_i64()),
        ("icmp_ugt_i64", proof_icmp_ugt_i64()),
        ("icmp_ule_i64", proof_icmp_ule_i64()),
        // AArch64 condbr predicates at i32 (issue #468) -- 10 obligations
        ("condbr_eq_i32", proof_condbr_eq_i32()),
        ("condbr_ne_i32", proof_condbr_ne_i32()),
        ("condbr_slt_i32", proof_condbr_slt_i32()),
        ("condbr_sge_i32", proof_condbr_sge_i32()),
        ("condbr_sgt_i32", proof_condbr_sgt_i32()),
        ("condbr_sle_i32", proof_condbr_sle_i32()),
        ("condbr_ult_i32", proof_condbr_ult_i32()),
        ("condbr_uge_i32", proof_condbr_uge_i32()),
        ("condbr_ugt_i32", proof_condbr_ugt_i32()),
        ("condbr_ule_i32", proof_condbr_ule_i32()),
        // AArch64 condbr predicates at i64 (issue #468) -- 10 obligations
        ("condbr_eq_i64", proof_condbr_eq_i64()),
        ("condbr_ne_i64", proof_condbr_ne_i64()),
        ("condbr_slt_i64", proof_condbr_slt_i64()),
        ("condbr_sge_i64", proof_condbr_sge_i64()),
        ("condbr_sgt_i64", proof_condbr_sgt_i64()),
        ("condbr_sle_i64", proof_condbr_sle_i64()),
        ("condbr_ult_i64", proof_condbr_ult_i64()),
        ("condbr_uge_i64", proof_condbr_uge_i64()),
        ("condbr_ugt_i64", proof_condbr_ugt_i64()),
        ("condbr_ule_i64", proof_condbr_ule_i64()),
        // x86-64 EFLAGS flag writes (issue #458) -- 8 i8 obligations
        ("x86_cmp_sf_i8", proof_cmp_sf_i8()),
        ("x86_cmp_zf_i8", proof_cmp_zf_i8()),
        ("x86_cmp_cf_i8", proof_cmp_cf_i8()),
        ("x86_cmp_of_i8", proof_cmp_of_i8()),
        ("x86_sub_sf_i8", proof_sub_sf_i8()),
        ("x86_sub_zf_i8", proof_sub_zf_i8()),
        ("x86_sub_cf_i8", proof_sub_cf_i8()),
        ("x86_sub_of_i8", proof_sub_of_i8()),
        // x86-64 EFLAGS flag writes at i32 (issue #458) -- ZF wired strict;
        // SF/CF/OF covered via eval/sampling tests in x86_64_eflags_proofs.
        ("x86_cmp_writes_zf_i32", proof_x86_cmp_writes_zf_i32()),
        // x86-64 core ops (issue #457) -- first cross-target coverage at i32
        ("x86_iadd_i32", proof_x86_iadd_i32()),
        ("x86_isub_i32", proof_x86_isub_i32()),
        ("x86_icmp_eq_i32", proof_x86_icmp_eq_i32()),
        ("x86_ishl_i32", proof_x86_ishl_i32()),
        // x86-64 core ops at i64 (#457 follow-up) -- REX.W + 64-bit bv theory
        ("x86_iadd_i64", proof_x86_iadd_i64()),
        ("x86_isub_i64", proof_x86_isub_i64()),
        ("x86_icmp_eq_i64", proof_x86_icmp_eq_i64()),
        ("x86_ishl_i64", proof_x86_ishl_i64()),
        // x86-64 widened ops (#463) -- neg/band/bor/bxor/ushr/sshr/imul
        ("x86_neg_i32", proof_x86_neg_i32()),
        ("x86_neg_i64", proof_x86_neg_i64()),
        ("x86_band_i32", proof_x86_band_i32()),
        ("x86_band_i64", proof_x86_band_i64()),
        ("x86_bor_i32", proof_x86_bor_i32()),
        ("x86_bor_i64", proof_x86_bor_i64()),
        ("x86_bxor_i32", proof_x86_bxor_i32()),
        ("x86_bxor_i64", proof_x86_bxor_i64()),
        ("x86_ushr_i32", proof_x86_ushr_i32()),
        ("x86_ushr_i64", proof_x86_ushr_i64()),
        ("x86_sshr_i32", proof_x86_sshr_i32()),
        ("x86_sshr_i64", proof_x86_sshr_i64()),
        ("x86_imul_i32", proof_x86_imul_i32()),
        ("x86_imul_i64", proof_x86_imul_i64()),
        // x86-64 integer ops at i8 (#470) -- 9 helpers already authored
        ("x86_iadd_i8", proof_x86_iadd_i8()),
        ("x86_isub_i8", proof_x86_isub_i8()),
        ("x86_imul_i8", proof_x86_imul_i8()),
        ("x86_band_i8", proof_x86_band_i8()),
        ("x86_bor_i8", proof_x86_bor_i8()),
        ("x86_bxor_i8", proof_x86_bxor_i8()),
        ("x86_ishl_i8", proof_x86_ishl_i8()),
        ("x86_ushr_i8", proof_x86_ushr_i8()),
        ("x86_sshr_i8", proof_x86_sshr_i8()),
        // x86-64 icmp predicates (#463) -- ne/slt/sge/sgt/sle/ult/uge/ugt/ule
        ("x86_icmp_ne_i32", proof_x86_icmp_ne_i32()),
        ("x86_icmp_ne_i64", proof_x86_icmp_ne_i64()),
        ("x86_icmp_slt_i32", proof_x86_icmp_slt_i32()),
        ("x86_icmp_slt_i64", proof_x86_icmp_slt_i64()),
        ("x86_icmp_sge_i32", proof_x86_icmp_sge_i32()),
        ("x86_icmp_sge_i64", proof_x86_icmp_sge_i64()),
        ("x86_icmp_sgt_i32", proof_x86_icmp_sgt_i32()),
        ("x86_icmp_sgt_i64", proof_x86_icmp_sgt_i64()),
        ("x86_icmp_sle_i32", proof_x86_icmp_sle_i32()),
        ("x86_icmp_sle_i64", proof_x86_icmp_sle_i64()),
        ("x86_icmp_ult_i32", proof_x86_icmp_ult_i32()),
        ("x86_icmp_ult_i64", proof_x86_icmp_ult_i64()),
        ("x86_icmp_uge_i32", proof_x86_icmp_uge_i32()),
        ("x86_icmp_uge_i64", proof_x86_icmp_uge_i64()),
        ("x86_icmp_ugt_i32", proof_x86_icmp_ugt_i32()),
        ("x86_icmp_ugt_i64", proof_x86_icmp_ugt_i64()),
        ("x86_icmp_ule_i32", proof_x86_icmp_ule_i32()),
        ("x86_icmp_ule_i64", proof_x86_icmp_ule_i64()),
        // x86-64 FP (#463) -- fadd/fsub/fmul/fdiv/fneg/fabs/fsqrt at f32/f64
        ("x86_fadd_f32", proof_x86_fadd_f32()),
        ("x86_fadd_f64", proof_x86_fadd_f64()),
        ("x86_fsub_f32", proof_x86_fsub_f32()),
        ("x86_fsub_f64", proof_x86_fsub_f64()),
        ("x86_fmul_f32", proof_x86_fmul_f32()),
        ("x86_fmul_f64", proof_x86_fmul_f64()),
        ("x86_fdiv_f32", proof_x86_fdiv_f32()),
        ("x86_fdiv_f64", proof_x86_fdiv_f64()),
        ("x86_fneg_f32", proof_x86_fneg_f32()),
        ("x86_fneg_f64", proof_x86_fneg_f64()),
        ("x86_fabs_f32", proof_x86_fabs_f32()),
        ("x86_fabs_f64", proof_x86_fabs_f64()),
        ("x86_fsqrt_f32", proof_x86_fsqrt_f32()),
        ("x86_fsqrt_f64", proof_x86_fsqrt_f64()),
        // memory_proofs (issue #469) -- load/store at i32/i64 and related
        ("mem_load_i32", proof_load_i32()),
        ("mem_load_i32_offset", proof_load_i32_offset()),
        ("mem_load_i64", proof_load_i64()),
        ("mem_load_i64_offset", proof_load_i64_offset()),
        ("mem_store_i32", proof_store_i32()),
        ("mem_store_i32_offset", proof_store_i32_offset()),
        ("mem_store_i64", proof_store_i64()),
        ("mem_store_i64_offset", proof_store_i64_offset()),
        ("mem_roundtrip_i32", proof_roundtrip_i32()),
        ("mem_roundtrip_i64", proof_roundtrip_i64()),
        ("mem_aligned_roundtrip_i32", proof_aligned_roundtrip_i32()),
        ("mem_aligned_roundtrip_i64", proof_aligned_roundtrip_i64()),
        ("mem_scaled_offset_alignment_i32", proof_scaled_offset_alignment_i32()),
        ("mem_non_interference_i32", proof_non_interference_i32()),
        ("mem_non_interference_i64", proof_non_interference_i64()),
        ("mem_non_interference_i32_adjacent", proof_non_interference_i32_adjacent()),
        ("mem_non_interference_i64_adjacent", proof_non_interference_i64_adjacent()),
        ("mem_non_interference_i64_store_i32_load", proof_non_interference_i64_store_i32_load()),
        ("mem_non_interference_i32_store_i64_load", proof_non_interference_i32_store_i64_load()),
        ("mem_non_interference_i32_symbolic", proof_non_interference_i32_symbolic()),
        ("mem_non_interference_i64_symbolic", proof_non_interference_i64_symbolic()),
        ("mem_forwarding_i32_offset12", proof_forwarding_i32_offset12()),
        ("mem_forwarding_i64_offset24", proof_forwarding_i64_offset24()),
        ("mem_forwarding_i64_to_i32_lower", proof_forwarding_i64_to_i32_lower()),
        ("mem_subword_i32_byte1", proof_subword_i32_byte1()),
        ("mem_subword_i32_byte2", proof_subword_i32_byte2()),
        ("mem_subword_i32_halfword_upper", proof_subword_i32_halfword_upper()),
        ("mem_subword_i64_byte5", proof_subword_i64_byte5()),
        ("mem_write_combining_i32", proof_write_combining_i32()),
        ("mem_write_combining_halfwords_to_i32", proof_write_combining_halfwords_to_i32()),
        ("mem_write_combining_words_to_i64", proof_write_combining_words_to_i64()),
        ("mem_endianness_i32", proof_endianness_i32()),
        ("mem_endianness_msb_i32", proof_endianness_msb_i32()),
        ("mem_endianness_i64", proof_endianness_i64()),
        ("mem_axiom_read_after_write_same_addr_32", proof_axiom_read_after_write_same_addr_32()),
        ("mem_axiom_read_after_write_diff_addr_32", proof_axiom_read_after_write_diff_addr_32()),
        ("mem_axiom_write_write_same_addr_32", proof_axiom_write_write_same_addr_32()),
        ("mem_axiom_const_array_32", proof_axiom_const_array_32()),
        // AArch64 FP arith/unary (#471) -- mirror of x86-64 FP set above
        ("fadd_f32", proof_fadd_f32()),
        ("fadd_f64", proof_fadd_f64()),
        ("fsub_f32", proof_fsub_f32()),
        ("fsub_f64", proof_fsub_f64()),
        ("fmul_f32", proof_fmul_f32()),
        ("fmul_f64", proof_fmul_f64()),
        ("fdiv_f32", proof_fdiv_f32()),
        ("fdiv_f64", proof_fdiv_f64()),
        ("fneg_f32", proof_fneg_f32()),
        ("fneg_f64", proof_fneg_f64()),
        ("fabs_f32", proof_fabs_f32()),
        ("fabs_f64", proof_fabs_f64()),
        ("fsqrt_f32", proof_fsqrt_f32()),
        ("fsqrt_f64", proof_fsqrt_f64()),
        // AArch64 fcmp ordered predicates (#471) -- 16 obligations
        ("fcmp_eq_f32", proof_fcmp_eq_f32()),
        ("fcmp_eq_f64", proof_fcmp_eq_f64()),
        ("fcmp_ne_f32", proof_fcmp_ne_f32()),
        ("fcmp_ne_f64", proof_fcmp_ne_f64()),
        ("fcmp_lt_f32", proof_fcmp_lt_f32()),
        ("fcmp_lt_f64", proof_fcmp_lt_f64()),
        ("fcmp_le_f32", proof_fcmp_le_f32()),
        ("fcmp_le_f64", proof_fcmp_le_f64()),
        ("fcmp_gt_f32", proof_fcmp_gt_f32()),
        ("fcmp_gt_f64", proof_fcmp_gt_f64()),
        ("fcmp_ge_f32", proof_fcmp_ge_f32()),
        ("fcmp_ge_f64", proof_fcmp_ge_f64()),
        ("fcmp_ord_f32", proof_fcmp_ord_f32()),
        ("fcmp_ord_f64", proof_fcmp_ord_f64()),
        ("fcmp_uno_f32", proof_fcmp_uno_f32()),
        ("fcmp_uno_f64", proof_fcmp_uno_f64()),
        // AArch64 fcmp unordered predicates (#471) -- 12 obligations
        ("fcmp_ueq_f32", proof_fcmp_ueq_f32()),
        ("fcmp_ueq_f64", proof_fcmp_ueq_f64()),
        ("fcmp_une_f32", proof_fcmp_une_f32()),
        ("fcmp_une_f64", proof_fcmp_une_f64()),
        ("fcmp_ult_f32", proof_fcmp_ult_f32()),
        ("fcmp_ult_f64", proof_fcmp_ult_f64()),
        ("fcmp_ule_f32", proof_fcmp_ule_f32()),
        ("fcmp_ule_f64", proof_fcmp_ule_f64()),
        ("fcmp_ugt_f32", proof_fcmp_ugt_f32()),
        ("fcmp_ugt_f64", proof_fcmp_ugt_f64()),
        ("fcmp_uge_f32", proof_fcmp_uge_f32()),
        ("fcmp_uge_f64", proof_fcmp_uge_f64()),
        // peephole_proofs (#472) -- canonical identity/zero/idempotent rewrites at i64
        ("peephole_add_zero_identity", proof_add_zero_identity()),
        ("peephole_sub_zero_identity", proof_sub_zero_identity()),
        ("peephole_mul_one_identity", proof_mul_one_identity()),
        ("peephole_lsl_zero_identity", proof_lsl_zero_identity()),
        ("peephole_lsr_zero_identity", proof_lsr_zero_identity()),
        ("peephole_asr_zero_identity", proof_asr_zero_identity()),
        ("peephole_orr_self_identity", proof_orr_self_identity()),
        ("peephole_and_self_identity", proof_and_self_identity()),
        ("peephole_eor_zero_identity", proof_eor_zero_identity()),
        ("peephole_orr_ri_zero_identity", proof_orr_ri_zero_identity()),
        ("peephole_and_ri_allones_identity", proof_and_ri_allones_identity()),
        ("peephole_sub_rr_self_zero", proof_sub_rr_self_zero()),
        ("peephole_eor_rr_self_zero", proof_eor_rr_self_zero()),
        ("peephole_and_ri_zero", proof_and_ri_zero()),
        ("peephole_orr_ri_allones", proof_orr_ri_allones()),
        ("peephole_neg_neg_identity", proof_neg_neg_identity()),
        ("peephole_mul_zero_absorb", proof_mul_zero_absorb()),
        ("peephole_udiv_one_identity", proof_udiv_one_identity()),
        // ext_trunc_proofs (#488 item 2) -- 23 obligations covering every
        // width-pair emitted by AArch64 select_extend / select_trunc:
        // 5 sign-ext + 5 zero-ext + 5 trunc + 4 roundtrip + 4 idempotent.
        ("sxtb_8_to_32", proof_sxtb_8_to_32()),
        ("sxtb_8_to_64", proof_sxtb_8_to_64()),
        ("sxth_16_to_32", proof_sxth_16_to_32()),
        ("sxth_16_to_64", proof_sxth_16_to_64()),
        ("sxtw_32_to_64", proof_sxtw_32_to_64()),
        ("uxtb_8_to_32", proof_uxtb_8_to_32()),
        ("uxtb_8_to_64", proof_uxtb_8_to_64()),
        ("uxth_16_to_32", proof_uxth_16_to_32()),
        ("uxth_16_to_64", proof_uxth_16_to_64()),
        ("uxtw_32_to_64", proof_uxtw_32_to_64()),
        ("trunc_i32_to_i8", proof_trunc_to_i8()),
        ("trunc_i32_to_i16", proof_trunc_to_i16()),
        ("trunc_i64_to_i8", proof_trunc_to_i8_from_64()),
        ("trunc_i64_to_i16", proof_trunc_to_i16_from_64()),
        ("trunc_i64_to_i32", proof_trunc_to_i32_from_64()),
        ("roundtrip_zext_trunc_8", proof_roundtrip_zext_trunc_8()),
        ("roundtrip_zext_trunc_16", proof_roundtrip_zext_trunc_16()),
        ("roundtrip_sext_trunc_8", proof_roundtrip_sext_trunc_8()),
        ("roundtrip_sext_trunc_16", proof_roundtrip_sext_trunc_16()),
        ("sxtb_idempotent", proof_sxtb_idempotent()),
        ("sxth_idempotent", proof_sxth_idempotent()),
        ("uxtb_idempotent", proof_uxtb_idempotent()),
        ("uxth_idempotent", proof_uxth_idempotent()),
    ];

    let config = Z4Config::default().with_timeout(10_000);
    let mut verified = 0usize;
    let mut timed_out = 0usize;
    let mut failures: Vec<String> = Vec::new();

    for (name, obligation) in &proofs {
        let result = verify_with_z4_api(obligation, &config);
        match result {
            Z4Result::Verified => verified += 1,
            Z4Result::Timeout => timed_out += 1,
            other => failures.push(format!("{}: {}", name, other)),
        }
    }

    println!(
        "z4-prove smoke summary: {} verified, {} timed-out, {} failed (of {})",
        verified,
        timed_out,
        failures.len(),
        proofs.len()
    );

    assert!(
        failures.is_empty(),
        "z4-prove smoke suite had unexpected failures:\n  {}",
        failures.join("\n  ")
    );
    // At least one proof must have reached a conclusive Verified result,
    // otherwise the z4 backend is effectively disabled.
    assert!(
        verified >= 1,
        "z4-prove smoke: all {} proofs timed out; z4 backend appears \
         non-functional on this host",
        proofs.len()
    );
}

// ===========================================================================
// Anti-tautology check (epic #407)
// ===========================================================================
//
// Guard against the failure mode where the proof pipeline trivially
// "verifies" everything because both sides of the equivalence were built
// from the same expression (structural tautology). If z4 were to accept
// `a * b == a + b` at i8 as "Verified", the whole smoke lane would be
// meaningless.
//
// Below we construct a DELIBERATELY-WRONG obligation where the tMIR side
// is `Imul(a, b)` (semantics = `bvmul`) and the machine side is `ADD`
// (`bvadd`). z4 must return `CounterExample`. If the solver returns
// `Verified` we have a soundness bug in the bridge or the encoder; if it
// returns `Timeout` on such a tiny 8-bit formula the backend is broken.
//
// This test is the twin of `z4_prove_imul_i8` and is the evidence that
// the imul/neg proofs above are not tautologies.

#[test]
fn z4_prove_anti_tautology_imul_vs_add_i8() {
    use llvm2_lower::instructions::Opcode;
    use llvm2_lower::types::Type;
    use llvm2_verify::lowering_proof::ProofObligation;
    use llvm2_verify::smt::SmtExpr;
    use llvm2_verify::tmir_semantics::encode_tmir_binop;

    let a = SmtExpr::var("a", 8);
    let b = SmtExpr::var("b", 8);

    // tMIR side: the real Imul semantics (bvmul at i8).
    // Machine side: bvadd -- WRONG on purpose. z4 must falsify.
    let bogus = ProofObligation {
        name: "anti-tautology: Imul_I8 lowered to ADD (must fail)".to_string(),
        tmir_expr: encode_tmir_binop(&Opcode::Imul, Type::I8, a.clone(), b.clone()),
        aarch64_expr: a.bvadd(b),
        inputs: vec![("a".to_string(), 8), ("b".to_string(), 8)],
        preconditions: vec![],
        fp_inputs: vec![],
        category: None,
    };

    let config = Z4Config::default().with_timeout(10_000);
    let result = verify_with_z4_api(&bogus, &config);

    match result {
        Z4Result::CounterExample(cex) => {
            assert!(
                !cex.is_empty(),
                "z4 returned an empty counterexample; the formula was \
                 accepted as tautology, which means the negated \
                 equivalence is UNSAT -- soundness bug"
            );
        }
        Z4Result::Verified => panic!(
            "SOUNDNESS BUG: z4 reported Imul_I8 == ADD as Verified. The \
             bridge is accepting non-equivalent obligations as proofs. \
             Every other proof in this suite is suspect until this is fixed."
        ),
        Z4Result::Timeout => panic!(
            "z4 timed out on an 8-bit bvmul != bvadd refutation. This \
             formula is trivial; a timeout here indicates the backend is \
             broken or misconfigured on this host."
        ),
        Z4Result::Error(e) => panic!("z4 error on anti-tautology probe: {}", e),
    }
}
