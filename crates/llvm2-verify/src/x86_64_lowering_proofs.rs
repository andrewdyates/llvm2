// llvm2-verify/x86_64_lowering_proofs.rs - x86-64 lowering rule proof obligations
//
// Author: Andrew Yates <ayates@dropbox.com>
// Copyright 2026 Dropbox, Inc. | License: Apache-2.0
//
// Defines proof obligations for tMIR -> x86-64 lowering rules and verifies
// semantic equivalence using the same ProofObligation framework as AArch64.
//
// Each proof obligation pairs:
//   - tMIR instruction semantics (from tmir_semantics module)
//   - x86-64 instruction semantics (from x86_64_semantics module)
//
// and asserts: forall inputs: tmir_result == x86_64_result
//
// Reference: Intel 64 and IA-32 Architectures Software Developer's Manual
// Reference: crates/llvm2-lower/src/x86_64_isel.rs (ISel rules being verified)

//! Proof obligations for x86-64 lowering rule verification.
//!
//! Mirrors the AArch64 proof obligations in [`crate::lowering_proof`] but
//! targets x86-64 instruction semantics. Each proof function constructs a
//! [`ProofObligation`] that can be verified by evaluation or SMT solving.

use crate::lowering_proof::ProofObligation;
use crate::smt::SmtExpr;

// ===========================================================================
// Integer arithmetic lowering proofs (32-bit)
// ===========================================================================

/// Proof: `tMIR::Iadd(I32, a, b) -> x86-64 ADD r32, r32`
///
/// tMIR Iadd is wrapping addition. x86-64 ADD is also wrapping addition.
/// Both are `bvadd` in SMT.
pub fn proof_x86_iadd_i32() -> ProofObligation {
    use crate::tmir_semantics::encode_tmir_binop;
    use crate::x86_64_semantics::{encode_add_rr, X86OperandSize};
    use llvm2_lower::instructions::Opcode;
    use llvm2_lower::types::Type;

    let a = SmtExpr::var("a", 32);
    let b = SmtExpr::var("b", 32);

    ProofObligation {
        name: "x86_64: Iadd_I32 -> ADD r32,r32".to_string(),
        tmir_expr: encode_tmir_binop(&Opcode::Iadd, Type::I32, a.clone(), b.clone()),
        aarch64_expr: encode_add_rr(X86OperandSize::S32, a, b),
        inputs: vec![("a".to_string(), 32), ("b".to_string(), 32)],
        preconditions: vec![],
        fp_inputs: vec![],
    }
}

/// Proof: `tMIR::Isub(I32, a, b) -> x86-64 SUB r32, r32`
pub fn proof_x86_isub_i32() -> ProofObligation {
    use crate::tmir_semantics::encode_tmir_binop;
    use crate::x86_64_semantics::{encode_sub_rr, X86OperandSize};
    use llvm2_lower::instructions::Opcode;
    use llvm2_lower::types::Type;

    let a = SmtExpr::var("a", 32);
    let b = SmtExpr::var("b", 32);

    ProofObligation {
        name: "x86_64: Isub_I32 -> SUB r32,r32".to_string(),
        tmir_expr: encode_tmir_binop(&Opcode::Isub, Type::I32, a.clone(), b.clone()),
        aarch64_expr: encode_sub_rr(X86OperandSize::S32, a, b),
        inputs: vec![("a".to_string(), 32), ("b".to_string(), 32)],
        preconditions: vec![],
        fp_inputs: vec![],
    }
}

/// Proof: `tMIR::Imul(I32, a, b) -> x86-64 IMUL r32, r32`
///
/// tMIR Imul produces the lower-width result of multiplication (wrapping).
/// x86-64 two-operand IMUL also produces the lower-width result in dst.
pub fn proof_x86_imul_i32() -> ProofObligation {
    use crate::tmir_semantics::encode_tmir_binop;
    use crate::x86_64_semantics::{encode_imul_rr, X86OperandSize};
    use llvm2_lower::instructions::Opcode;
    use llvm2_lower::types::Type;

    let a = SmtExpr::var("a", 32);
    let b = SmtExpr::var("b", 32);

    ProofObligation {
        name: "x86_64: Imul_I32 -> IMUL r32,r32".to_string(),
        tmir_expr: encode_tmir_binop(&Opcode::Imul, Type::I32, a.clone(), b.clone()),
        aarch64_expr: encode_imul_rr(X86OperandSize::S32, a, b),
        inputs: vec![("a".to_string(), 32), ("b".to_string(), 32)],
        preconditions: vec![],
        fp_inputs: vec![],
    }
}

/// Proof: `tMIR::Ineg(I32, a) -> x86-64 NEG r32`
pub fn proof_x86_neg_i32() -> ProofObligation {
    use crate::tmir_semantics::encode_tmir_neg;
    use crate::x86_64_semantics::{encode_neg, X86OperandSize};
    use llvm2_lower::types::Type;

    let a = SmtExpr::var("a", 32);

    ProofObligation {
        name: "x86_64: Neg_I32 -> NEG r32".to_string(),
        tmir_expr: encode_tmir_neg(Type::I32, a.clone()),
        aarch64_expr: encode_neg(X86OperandSize::S32, a),
        inputs: vec![("a".to_string(), 32)],
        preconditions: vec![],
        fp_inputs: vec![],
    }
}

// ===========================================================================
// Integer arithmetic lowering proofs (64-bit)
// ===========================================================================

/// Proof: `tMIR::Iadd(I64, a, b) -> x86-64 ADD r64, r64`
pub fn proof_x86_iadd_i64() -> ProofObligation {
    use crate::tmir_semantics::encode_tmir_binop;
    use crate::x86_64_semantics::{encode_add_rr, X86OperandSize};
    use llvm2_lower::instructions::Opcode;
    use llvm2_lower::types::Type;

    let a = SmtExpr::var("a", 64);
    let b = SmtExpr::var("b", 64);

    ProofObligation {
        name: "x86_64: Iadd_I64 -> ADD r64,r64".to_string(),
        tmir_expr: encode_tmir_binop(&Opcode::Iadd, Type::I64, a.clone(), b.clone()),
        aarch64_expr: encode_add_rr(X86OperandSize::S64, a, b),
        inputs: vec![("a".to_string(), 64), ("b".to_string(), 64)],
        preconditions: vec![],
        fp_inputs: vec![],
    }
}

/// Proof: `tMIR::Isub(I64, a, b) -> x86-64 SUB r64, r64`
pub fn proof_x86_isub_i64() -> ProofObligation {
    use crate::tmir_semantics::encode_tmir_binop;
    use crate::x86_64_semantics::{encode_sub_rr, X86OperandSize};
    use llvm2_lower::instructions::Opcode;
    use llvm2_lower::types::Type;

    let a = SmtExpr::var("a", 64);
    let b = SmtExpr::var("b", 64);

    ProofObligation {
        name: "x86_64: Isub_I64 -> SUB r64,r64".to_string(),
        tmir_expr: encode_tmir_binop(&Opcode::Isub, Type::I64, a.clone(), b.clone()),
        aarch64_expr: encode_sub_rr(X86OperandSize::S64, a, b),
        inputs: vec![("a".to_string(), 64), ("b".to_string(), 64)],
        preconditions: vec![],
        fp_inputs: vec![],
    }
}

/// Proof: `tMIR::Imul(I64, a, b) -> x86-64 IMUL r64, r64`
pub fn proof_x86_imul_i64() -> ProofObligation {
    use crate::tmir_semantics::encode_tmir_binop;
    use crate::x86_64_semantics::{encode_imul_rr, X86OperandSize};
    use llvm2_lower::instructions::Opcode;
    use llvm2_lower::types::Type;

    let a = SmtExpr::var("a", 64);
    let b = SmtExpr::var("b", 64);

    ProofObligation {
        name: "x86_64: Imul_I64 -> IMUL r64,r64".to_string(),
        tmir_expr: encode_tmir_binop(&Opcode::Imul, Type::I64, a.clone(), b.clone()),
        aarch64_expr: encode_imul_rr(X86OperandSize::S64, a, b),
        inputs: vec![("a".to_string(), 64), ("b".to_string(), 64)],
        preconditions: vec![],
        fp_inputs: vec![],
    }
}

/// Proof: `tMIR::Ineg(I64, a) -> x86-64 NEG r64`
pub fn proof_x86_neg_i64() -> ProofObligation {
    use crate::tmir_semantics::encode_tmir_neg;
    use crate::x86_64_semantics::{encode_neg, X86OperandSize};
    use llvm2_lower::types::Type;

    let a = SmtExpr::var("a", 64);

    ProofObligation {
        name: "x86_64: Neg_I64 -> NEG r64".to_string(),
        tmir_expr: encode_tmir_neg(Type::I64, a.clone()),
        aarch64_expr: encode_neg(X86OperandSize::S64, a),
        inputs: vec![("a".to_string(), 64)],
        preconditions: vec![],
        fp_inputs: vec![],
    }
}

// ===========================================================================
// Division lowering proofs
// ===========================================================================

/// Proof: `tMIR::Sdiv(I32, a, b) -> x86-64 IDIV r32` (quotient in EAX)
///
/// The x86-64 ISel emits CDQ (sign-extend EAX to EDX:EAX) + IDIV.
/// The quotient in EAX matches tMIR's signed division semantic.
/// Precondition: divisor != 0.
pub fn proof_x86_sdiv_i32() -> ProofObligation {
    use crate::tmir_semantics::{encode_tmir_binop, precondition};
    use crate::x86_64_semantics::{encode_idiv_quotient, X86OperandSize};
    use llvm2_lower::instructions::Opcode;
    use llvm2_lower::types::Type;

    let a = SmtExpr::var("a", 32);
    let b = SmtExpr::var("b", 32);

    let mut preconditions = vec![];
    if let Some(pre) = precondition(&Opcode::Sdiv, Type::I32, &a, &b) {
        preconditions.push(pre);
    }

    ProofObligation {
        name: "x86_64: Sdiv_I32 -> IDIV r32 (quotient)".to_string(),
        tmir_expr: encode_tmir_binop(&Opcode::Sdiv, Type::I32, a.clone(), b.clone()),
        aarch64_expr: encode_idiv_quotient(X86OperandSize::S32, a, b),
        inputs: vec![("a".to_string(), 32), ("b".to_string(), 32)],
        preconditions,
        fp_inputs: vec![],
    }
}

/// Proof: `tMIR::Sdiv(I64, a, b) -> x86-64 IDIV r64` (quotient in RAX)
pub fn proof_x86_sdiv_i64() -> ProofObligation {
    use crate::tmir_semantics::{encode_tmir_binop, precondition};
    use crate::x86_64_semantics::{encode_idiv_quotient, X86OperandSize};
    use llvm2_lower::instructions::Opcode;
    use llvm2_lower::types::Type;

    let a = SmtExpr::var("a", 64);
    let b = SmtExpr::var("b", 64);

    let mut preconditions = vec![];
    if let Some(pre) = precondition(&Opcode::Sdiv, Type::I64, &a, &b) {
        preconditions.push(pre);
    }

    ProofObligation {
        name: "x86_64: Sdiv_I64 -> IDIV r64 (quotient)".to_string(),
        tmir_expr: encode_tmir_binop(&Opcode::Sdiv, Type::I64, a.clone(), b.clone()),
        aarch64_expr: encode_idiv_quotient(X86OperandSize::S64, a, b),
        inputs: vec![("a".to_string(), 64), ("b".to_string(), 64)],
        preconditions,
        fp_inputs: vec![],
    }
}

/// Proof: `tMIR::Udiv(I32, a, b) -> x86-64 DIV r32` (quotient in EAX)
pub fn proof_x86_udiv_i32() -> ProofObligation {
    use crate::tmir_semantics::{encode_tmir_binop, precondition};
    use crate::x86_64_semantics::{encode_div_quotient, X86OperandSize};
    use llvm2_lower::instructions::Opcode;
    use llvm2_lower::types::Type;

    let a = SmtExpr::var("a", 32);
    let b = SmtExpr::var("b", 32);

    let mut preconditions = vec![];
    if let Some(pre) = precondition(&Opcode::Udiv, Type::I32, &a, &b) {
        preconditions.push(pre);
    }

    ProofObligation {
        name: "x86_64: Udiv_I32 -> DIV r32 (quotient)".to_string(),
        tmir_expr: encode_tmir_binop(&Opcode::Udiv, Type::I32, a.clone(), b.clone()),
        aarch64_expr: encode_div_quotient(X86OperandSize::S32, a, b),
        inputs: vec![("a".to_string(), 32), ("b".to_string(), 32)],
        preconditions,
        fp_inputs: vec![],
    }
}

/// Proof: `tMIR::Udiv(I64, a, b) -> x86-64 DIV r64` (quotient in RAX)
pub fn proof_x86_udiv_i64() -> ProofObligation {
    use crate::tmir_semantics::{encode_tmir_binop, precondition};
    use crate::x86_64_semantics::{encode_div_quotient, X86OperandSize};
    use llvm2_lower::instructions::Opcode;
    use llvm2_lower::types::Type;

    let a = SmtExpr::var("a", 64);
    let b = SmtExpr::var("b", 64);

    let mut preconditions = vec![];
    if let Some(pre) = precondition(&Opcode::Udiv, Type::I64, &a, &b) {
        preconditions.push(pre);
    }

    ProofObligation {
        name: "x86_64: Udiv_I64 -> DIV r64 (quotient)".to_string(),
        tmir_expr: encode_tmir_binop(&Opcode::Udiv, Type::I64, a.clone(), b.clone()),
        aarch64_expr: encode_div_quotient(X86OperandSize::S64, a, b),
        inputs: vec![("a".to_string(), 64), ("b".to_string(), 64)],
        preconditions,
        fp_inputs: vec![],
    }
}

// ===========================================================================
// Bitwise lowering proofs (32-bit)
// ===========================================================================

/// Proof: `tMIR::Band(I32, a, b) -> x86-64 AND r32, r32`
pub fn proof_x86_band_i32() -> ProofObligation {
    use crate::tmir_semantics::encode_tmir_bitwise_binop;
    use crate::x86_64_semantics::{encode_and_rr, X86OperandSize};
    use llvm2_lower::instructions::Opcode;
    use llvm2_lower::types::Type;

    let a = SmtExpr::var("a", 32);
    let b = SmtExpr::var("b", 32);

    ProofObligation {
        name: "x86_64: Band_I32 -> AND r32,r32".to_string(),
        tmir_expr: encode_tmir_bitwise_binop(&Opcode::Band, Type::I32, a.clone(), b.clone()),
        aarch64_expr: encode_and_rr(X86OperandSize::S32, a, b),
        inputs: vec![("a".to_string(), 32), ("b".to_string(), 32)],
        preconditions: vec![],
        fp_inputs: vec![],
    }
}

/// Proof: `tMIR::Bor(I32, a, b) -> x86-64 OR r32, r32`
pub fn proof_x86_bor_i32() -> ProofObligation {
    use crate::tmir_semantics::encode_tmir_bitwise_binop;
    use crate::x86_64_semantics::{encode_or_rr, X86OperandSize};
    use llvm2_lower::instructions::Opcode;
    use llvm2_lower::types::Type;

    let a = SmtExpr::var("a", 32);
    let b = SmtExpr::var("b", 32);

    ProofObligation {
        name: "x86_64: Bor_I32 -> OR r32,r32".to_string(),
        tmir_expr: encode_tmir_bitwise_binop(&Opcode::Bor, Type::I32, a.clone(), b.clone()),
        aarch64_expr: encode_or_rr(X86OperandSize::S32, a, b),
        inputs: vec![("a".to_string(), 32), ("b".to_string(), 32)],
        preconditions: vec![],
        fp_inputs: vec![],
    }
}

/// Proof: `tMIR::Bxor(I32, a, b) -> x86-64 XOR r32, r32`
pub fn proof_x86_bxor_i32() -> ProofObligation {
    use crate::tmir_semantics::encode_tmir_bitwise_binop;
    use crate::x86_64_semantics::{encode_xor_rr, X86OperandSize};
    use llvm2_lower::instructions::Opcode;
    use llvm2_lower::types::Type;

    let a = SmtExpr::var("a", 32);
    let b = SmtExpr::var("b", 32);

    ProofObligation {
        name: "x86_64: Bxor_I32 -> XOR r32,r32".to_string(),
        tmir_expr: encode_tmir_bitwise_binop(&Opcode::Bxor, Type::I32, a.clone(), b.clone()),
        aarch64_expr: encode_xor_rr(X86OperandSize::S32, a, b),
        inputs: vec![("a".to_string(), 32), ("b".to_string(), 32)],
        preconditions: vec![],
        fp_inputs: vec![],
    }
}

/// Proof: `tMIR::Bnot(I32, a) -> x86-64 NOT r32`
pub fn proof_x86_bnot_i32() -> ProofObligation {
    use crate::tmir_semantics::encode_tmir_bnot;
    use crate::x86_64_semantics::{encode_not, X86OperandSize};
    use llvm2_lower::types::Type;

    let a = SmtExpr::var("a", 32);

    ProofObligation {
        name: "x86_64: Bnot_I32 -> NOT r32".to_string(),
        tmir_expr: encode_tmir_bnot(Type::I32, a.clone()),
        aarch64_expr: encode_not(X86OperandSize::S32, a),
        inputs: vec![("a".to_string(), 32)],
        preconditions: vec![],
        fp_inputs: vec![],
    }
}

// ===========================================================================
// Bitwise lowering proofs (64-bit)
// ===========================================================================

/// Proof: `tMIR::Band(I64, a, b) -> x86-64 AND r64, r64`
pub fn proof_x86_band_i64() -> ProofObligation {
    use crate::tmir_semantics::encode_tmir_bitwise_binop;
    use crate::x86_64_semantics::{encode_and_rr, X86OperandSize};
    use llvm2_lower::instructions::Opcode;
    use llvm2_lower::types::Type;

    let a = SmtExpr::var("a", 64);
    let b = SmtExpr::var("b", 64);

    ProofObligation {
        name: "x86_64: Band_I64 -> AND r64,r64".to_string(),
        tmir_expr: encode_tmir_bitwise_binop(&Opcode::Band, Type::I64, a.clone(), b.clone()),
        aarch64_expr: encode_and_rr(X86OperandSize::S64, a, b),
        inputs: vec![("a".to_string(), 64), ("b".to_string(), 64)],
        preconditions: vec![],
        fp_inputs: vec![],
    }
}

/// Proof: `tMIR::Bor(I64, a, b) -> x86-64 OR r64, r64`
pub fn proof_x86_bor_i64() -> ProofObligation {
    use crate::tmir_semantics::encode_tmir_bitwise_binop;
    use crate::x86_64_semantics::{encode_or_rr, X86OperandSize};
    use llvm2_lower::instructions::Opcode;
    use llvm2_lower::types::Type;

    let a = SmtExpr::var("a", 64);
    let b = SmtExpr::var("b", 64);

    ProofObligation {
        name: "x86_64: Bor_I64 -> OR r64,r64".to_string(),
        tmir_expr: encode_tmir_bitwise_binop(&Opcode::Bor, Type::I64, a.clone(), b.clone()),
        aarch64_expr: encode_or_rr(X86OperandSize::S64, a, b),
        inputs: vec![("a".to_string(), 64), ("b".to_string(), 64)],
        preconditions: vec![],
        fp_inputs: vec![],
    }
}

/// Proof: `tMIR::Bxor(I64, a, b) -> x86-64 XOR r64, r64`
pub fn proof_x86_bxor_i64() -> ProofObligation {
    use crate::tmir_semantics::encode_tmir_bitwise_binop;
    use crate::x86_64_semantics::{encode_xor_rr, X86OperandSize};
    use llvm2_lower::instructions::Opcode;
    use llvm2_lower::types::Type;

    let a = SmtExpr::var("a", 64);
    let b = SmtExpr::var("b", 64);

    ProofObligation {
        name: "x86_64: Bxor_I64 -> XOR r64,r64".to_string(),
        tmir_expr: encode_tmir_bitwise_binop(&Opcode::Bxor, Type::I64, a.clone(), b.clone()),
        aarch64_expr: encode_xor_rr(X86OperandSize::S64, a, b),
        inputs: vec![("a".to_string(), 64), ("b".to_string(), 64)],
        preconditions: vec![],
        fp_inputs: vec![],
    }
}

/// Proof: `tMIR::Bnot(I64, a) -> x86-64 NOT r64`
pub fn proof_x86_bnot_i64() -> ProofObligation {
    use crate::tmir_semantics::encode_tmir_bnot;
    use crate::x86_64_semantics::{encode_not, X86OperandSize};
    use llvm2_lower::types::Type;

    let a = SmtExpr::var("a", 64);

    ProofObligation {
        name: "x86_64: Bnot_I64 -> NOT r64".to_string(),
        tmir_expr: encode_tmir_bnot(Type::I64, a.clone()),
        aarch64_expr: encode_not(X86OperandSize::S64, a),
        inputs: vec![("a".to_string(), 64)],
        preconditions: vec![],
        fp_inputs: vec![],
    }
}

// ===========================================================================
// Shift lowering proofs (32-bit)
// ===========================================================================

/// Proof: `tMIR::Ishl(I32, a, b) -> x86-64 SHL r32, CL`
///
/// Both tMIR and x86-64 shift left operations produce the same result.
/// x86-64 masks the shift amount to 5 bits for 32-bit operands; the tMIR
/// semantics also use bvshl which matches this behavior for in-range amounts.
pub fn proof_x86_ishl_i32() -> ProofObligation {
    use crate::tmir_semantics::encode_tmir_shift;
    use crate::x86_64_semantics::{encode_shl_rr, X86OperandSize};
    use llvm2_lower::instructions::Opcode;
    use llvm2_lower::types::Type;

    let a = SmtExpr::var("a", 32);
    let b = SmtExpr::var("b", 32);

    ProofObligation {
        name: "x86_64: Ishl_I32 -> SHL r32,CL".to_string(),
        tmir_expr: encode_tmir_shift(&Opcode::Ishl, Type::I32, a.clone(), b.clone()),
        aarch64_expr: encode_shl_rr(X86OperandSize::S32, a, b),
        inputs: vec![("a".to_string(), 32), ("b".to_string(), 32)],
        preconditions: vec![],
        fp_inputs: vec![],
    }
}

/// Proof: `tMIR::Ushr(I32, a, b) -> x86-64 SHR r32, CL`
pub fn proof_x86_ushr_i32() -> ProofObligation {
    use crate::tmir_semantics::encode_tmir_shift;
    use crate::x86_64_semantics::{encode_shr_rr, X86OperandSize};
    use llvm2_lower::instructions::Opcode;
    use llvm2_lower::types::Type;

    let a = SmtExpr::var("a", 32);
    let b = SmtExpr::var("b", 32);

    ProofObligation {
        name: "x86_64: Ushr_I32 -> SHR r32,CL".to_string(),
        tmir_expr: encode_tmir_shift(&Opcode::Ushr, Type::I32, a.clone(), b.clone()),
        aarch64_expr: encode_shr_rr(X86OperandSize::S32, a, b),
        inputs: vec![("a".to_string(), 32), ("b".to_string(), 32)],
        preconditions: vec![],
        fp_inputs: vec![],
    }
}

/// Proof: `tMIR::Sshr(I32, a, b) -> x86-64 SAR r32, CL`
pub fn proof_x86_sshr_i32() -> ProofObligation {
    use crate::tmir_semantics::encode_tmir_shift;
    use crate::x86_64_semantics::{encode_sar_rr, X86OperandSize};
    use llvm2_lower::instructions::Opcode;
    use llvm2_lower::types::Type;

    let a = SmtExpr::var("a", 32);
    let b = SmtExpr::var("b", 32);

    ProofObligation {
        name: "x86_64: Sshr_I32 -> SAR r32,CL".to_string(),
        tmir_expr: encode_tmir_shift(&Opcode::Sshr, Type::I32, a.clone(), b.clone()),
        aarch64_expr: encode_sar_rr(X86OperandSize::S32, a, b),
        inputs: vec![("a".to_string(), 32), ("b".to_string(), 32)],
        preconditions: vec![],
        fp_inputs: vec![],
    }
}

// ===========================================================================
// Shift lowering proofs (64-bit)
// ===========================================================================

/// Proof: `tMIR::Ishl(I64, a, b) -> x86-64 SHL r64, CL`
pub fn proof_x86_ishl_i64() -> ProofObligation {
    use crate::tmir_semantics::encode_tmir_shift;
    use crate::x86_64_semantics::{encode_shl_rr, X86OperandSize};
    use llvm2_lower::instructions::Opcode;
    use llvm2_lower::types::Type;

    let a = SmtExpr::var("a", 64);
    let b = SmtExpr::var("b", 64);

    ProofObligation {
        name: "x86_64: Ishl_I64 -> SHL r64,CL".to_string(),
        tmir_expr: encode_tmir_shift(&Opcode::Ishl, Type::I64, a.clone(), b.clone()),
        aarch64_expr: encode_shl_rr(X86OperandSize::S64, a, b),
        inputs: vec![("a".to_string(), 64), ("b".to_string(), 64)],
        preconditions: vec![],
        fp_inputs: vec![],
    }
}

/// Proof: `tMIR::Ushr(I64, a, b) -> x86-64 SHR r64, CL`
pub fn proof_x86_ushr_i64() -> ProofObligation {
    use crate::tmir_semantics::encode_tmir_shift;
    use crate::x86_64_semantics::{encode_shr_rr, X86OperandSize};
    use llvm2_lower::instructions::Opcode;
    use llvm2_lower::types::Type;

    let a = SmtExpr::var("a", 64);
    let b = SmtExpr::var("b", 64);

    ProofObligation {
        name: "x86_64: Ushr_I64 -> SHR r64,CL".to_string(),
        tmir_expr: encode_tmir_shift(&Opcode::Ushr, Type::I64, a.clone(), b.clone()),
        aarch64_expr: encode_shr_rr(X86OperandSize::S64, a, b),
        inputs: vec![("a".to_string(), 64), ("b".to_string(), 64)],
        preconditions: vec![],
        fp_inputs: vec![],
    }
}

/// Proof: `tMIR::Sshr(I64, a, b) -> x86-64 SAR r64, CL`
pub fn proof_x86_sshr_i64() -> ProofObligation {
    use crate::tmir_semantics::encode_tmir_shift;
    use crate::x86_64_semantics::{encode_sar_rr, X86OperandSize};
    use llvm2_lower::instructions::Opcode;
    use llvm2_lower::types::Type;

    let a = SmtExpr::var("a", 64);
    let b = SmtExpr::var("b", 64);

    ProofObligation {
        name: "x86_64: Sshr_I64 -> SAR r64,CL".to_string(),
        tmir_expr: encode_tmir_shift(&Opcode::Sshr, Type::I64, a.clone(), b.clone()),
        aarch64_expr: encode_sar_rr(X86OperandSize::S64, a, b),
        inputs: vec![("a".to_string(), 64), ("b".to_string(), 64)],
        preconditions: vec![],
        fp_inputs: vec![],
    }
}

// ===========================================================================
// 8-bit exhaustive proofs (complete verification)
// ===========================================================================

/// Proof: `tMIR::Iadd(I8, a, b) -> x86-64 ADD (8-bit)`
///
/// 8-bit proofs are verified exhaustively (all 65,536 input pairs).
pub fn proof_x86_iadd_i8() -> ProofObligation {
    use crate::tmir_semantics::encode_tmir_binop;
    use crate::x86_64_semantics::{encode_add_rr, X86OperandSize};
    use llvm2_lower::instructions::Opcode;
    use llvm2_lower::types::Type;

    let a = SmtExpr::var("a", 8);
    let b = SmtExpr::var("b", 8);

    ProofObligation {
        name: "x86_64: Iadd_I8 -> ADD (8-bit)".to_string(),
        tmir_expr: encode_tmir_binop(&Opcode::Iadd, Type::I8, a.clone(), b.clone()),
        aarch64_expr: encode_add_rr(X86OperandSize::S32, a, b),
        inputs: vec![("a".to_string(), 8), ("b".to_string(), 8)],
        preconditions: vec![],
        fp_inputs: vec![],
    }
}

/// Proof: `tMIR::Isub(I8, a, b) -> x86-64 SUB (8-bit)`
pub fn proof_x86_isub_i8() -> ProofObligation {
    use crate::tmir_semantics::encode_tmir_binop;
    use crate::x86_64_semantics::{encode_sub_rr, X86OperandSize};
    use llvm2_lower::instructions::Opcode;
    use llvm2_lower::types::Type;

    let a = SmtExpr::var("a", 8);
    let b = SmtExpr::var("b", 8);

    ProofObligation {
        name: "x86_64: Isub_I8 -> SUB (8-bit)".to_string(),
        tmir_expr: encode_tmir_binop(&Opcode::Isub, Type::I8, a.clone(), b.clone()),
        aarch64_expr: encode_sub_rr(X86OperandSize::S32, a, b),
        inputs: vec![("a".to_string(), 8), ("b".to_string(), 8)],
        preconditions: vec![],
        fp_inputs: vec![],
    }
}

/// Proof: `tMIR::Imul(I8, a, b) -> x86-64 IMUL (8-bit)`
pub fn proof_x86_imul_i8() -> ProofObligation {
    use crate::tmir_semantics::encode_tmir_binop;
    use crate::x86_64_semantics::{encode_imul_rr, X86OperandSize};
    use llvm2_lower::instructions::Opcode;
    use llvm2_lower::types::Type;

    let a = SmtExpr::var("a", 8);
    let b = SmtExpr::var("b", 8);

    ProofObligation {
        name: "x86_64: Imul_I8 -> IMUL (8-bit)".to_string(),
        tmir_expr: encode_tmir_binop(&Opcode::Imul, Type::I8, a.clone(), b.clone()),
        aarch64_expr: encode_imul_rr(X86OperandSize::S32, a, b),
        inputs: vec![("a".to_string(), 8), ("b".to_string(), 8)],
        preconditions: vec![],
        fp_inputs: vec![],
    }
}

// ===========================================================================
// Collect all proofs
// ===========================================================================

/// Return all x86-64 lowering proof obligations.
///
/// This provides a single entry point for running all x86-64 verification
/// proofs, analogous to how the AArch64 proofs are collected.
pub fn all_x86_64_proofs() -> Vec<ProofObligation> {
    vec![
        // Arithmetic (32-bit)
        proof_x86_iadd_i32(),
        proof_x86_isub_i32(),
        proof_x86_imul_i32(),
        proof_x86_neg_i32(),
        // Arithmetic (64-bit)
        proof_x86_iadd_i64(),
        proof_x86_isub_i64(),
        proof_x86_imul_i64(),
        proof_x86_neg_i64(),
        // Division
        proof_x86_sdiv_i32(),
        proof_x86_sdiv_i64(),
        proof_x86_udiv_i32(),
        proof_x86_udiv_i64(),
        // Bitwise (32-bit)
        proof_x86_band_i32(),
        proof_x86_bor_i32(),
        proof_x86_bxor_i32(),
        proof_x86_bnot_i32(),
        // Bitwise (64-bit)
        proof_x86_band_i64(),
        proof_x86_bor_i64(),
        proof_x86_bxor_i64(),
        proof_x86_bnot_i64(),
        // Shifts (32-bit)
        proof_x86_ishl_i32(),
        proof_x86_ushr_i32(),
        proof_x86_sshr_i32(),
        // Shifts (64-bit)
        proof_x86_ishl_i64(),
        proof_x86_ushr_i64(),
        proof_x86_sshr_i64(),
        // 8-bit exhaustive
        proof_x86_iadd_i8(),
        proof_x86_isub_i8(),
        proof_x86_imul_i8(),
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

    // -----------------------------------------------------------------------
    // Arithmetic lowering proof tests (32-bit)
    // -----------------------------------------------------------------------

    #[test]
    fn test_x86_64_iadd_i32_proof() {
        let obligation = proof_x86_iadd_i32();
        let result = verify_by_evaluation(&obligation);
        assert!(matches!(result, VerificationResult::Valid), "x86-64 Iadd_I32 proof failed: {:?}", result);
    }

    #[test]
    fn test_x86_64_isub_i32_proof() {
        let obligation = proof_x86_isub_i32();
        let result = verify_by_evaluation(&obligation);
        assert!(matches!(result, VerificationResult::Valid), "x86-64 Isub_I32 proof failed: {:?}", result);
    }

    #[test]
    fn test_x86_64_imul_i32_proof() {
        let obligation = proof_x86_imul_i32();
        let result = verify_by_evaluation(&obligation);
        assert!(matches!(result, VerificationResult::Valid), "x86-64 Imul_I32 proof failed: {:?}", result);
    }

    #[test]
    fn test_x86_64_neg_i32_proof() {
        let obligation = proof_x86_neg_i32();
        let result = verify_by_evaluation(&obligation);
        assert!(matches!(result, VerificationResult::Valid), "x86-64 Neg_I32 proof failed: {:?}", result);
    }

    // -----------------------------------------------------------------------
    // Arithmetic lowering proof tests (64-bit)
    // -----------------------------------------------------------------------

    #[test]
    fn test_x86_64_iadd_i64_proof() {
        let obligation = proof_x86_iadd_i64();
        let result = verify_by_evaluation(&obligation);
        assert!(matches!(result, VerificationResult::Valid), "x86-64 Iadd_I64 proof failed: {:?}", result);
    }

    #[test]
    fn test_x86_64_isub_i64_proof() {
        let obligation = proof_x86_isub_i64();
        let result = verify_by_evaluation(&obligation);
        assert!(matches!(result, VerificationResult::Valid), "x86-64 Isub_I64 proof failed: {:?}", result);
    }

    #[test]
    fn test_x86_64_imul_i64_proof() {
        let obligation = proof_x86_imul_i64();
        let result = verify_by_evaluation(&obligation);
        assert!(matches!(result, VerificationResult::Valid), "x86-64 Imul_I64 proof failed: {:?}", result);
    }

    #[test]
    fn test_x86_64_neg_i64_proof() {
        let obligation = proof_x86_neg_i64();
        let result = verify_by_evaluation(&obligation);
        assert!(matches!(result, VerificationResult::Valid), "x86-64 Neg_I64 proof failed: {:?}", result);
    }

    // -----------------------------------------------------------------------
    // Division lowering proof tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_x86_64_sdiv_i32_proof() {
        let obligation = proof_x86_sdiv_i32();
        let result = verify_by_evaluation(&obligation);
        assert!(matches!(result, VerificationResult::Valid), "x86-64 Sdiv_I32 proof failed: {:?}", result);
    }

    #[test]
    fn test_x86_64_sdiv_i64_proof() {
        let obligation = proof_x86_sdiv_i64();
        let result = verify_by_evaluation(&obligation);
        assert!(matches!(result, VerificationResult::Valid), "x86-64 Sdiv_I64 proof failed: {:?}", result);
    }

    #[test]
    fn test_x86_64_udiv_i32_proof() {
        let obligation = proof_x86_udiv_i32();
        let result = verify_by_evaluation(&obligation);
        assert!(matches!(result, VerificationResult::Valid), "x86-64 Udiv_I32 proof failed: {:?}", result);
    }

    #[test]
    fn test_x86_64_udiv_i64_proof() {
        let obligation = proof_x86_udiv_i64();
        let result = verify_by_evaluation(&obligation);
        assert!(matches!(result, VerificationResult::Valid), "x86-64 Udiv_I64 proof failed: {:?}", result);
    }

    // -----------------------------------------------------------------------
    // Bitwise lowering proof tests (32-bit)
    // -----------------------------------------------------------------------

    #[test]
    fn test_x86_64_band_i32_proof() {
        let obligation = proof_x86_band_i32();
        let result = verify_by_evaluation(&obligation);
        assert!(matches!(result, VerificationResult::Valid), "x86-64 Band_I32 proof failed: {:?}", result);
    }

    #[test]
    fn test_x86_64_bor_i32_proof() {
        let obligation = proof_x86_bor_i32();
        let result = verify_by_evaluation(&obligation);
        assert!(matches!(result, VerificationResult::Valid), "x86-64 Bor_I32 proof failed: {:?}", result);
    }

    #[test]
    fn test_x86_64_bxor_i32_proof() {
        let obligation = proof_x86_bxor_i32();
        let result = verify_by_evaluation(&obligation);
        assert!(matches!(result, VerificationResult::Valid), "x86-64 Bxor_I32 proof failed: {:?}", result);
    }

    #[test]
    fn test_x86_64_bnot_i32_proof() {
        let obligation = proof_x86_bnot_i32();
        let result = verify_by_evaluation(&obligation);
        assert!(matches!(result, VerificationResult::Valid), "x86-64 Bnot_I32 proof failed: {:?}", result);
    }

    // -----------------------------------------------------------------------
    // Bitwise lowering proof tests (64-bit)
    // -----------------------------------------------------------------------

    #[test]
    fn test_x86_64_band_i64_proof() {
        let obligation = proof_x86_band_i64();
        let result = verify_by_evaluation(&obligation);
        assert!(matches!(result, VerificationResult::Valid), "x86-64 Band_I64 proof failed: {:?}", result);
    }

    #[test]
    fn test_x86_64_bor_i64_proof() {
        let obligation = proof_x86_bor_i64();
        let result = verify_by_evaluation(&obligation);
        assert!(matches!(result, VerificationResult::Valid), "x86-64 Bor_I64 proof failed: {:?}", result);
    }

    #[test]
    fn test_x86_64_bxor_i64_proof() {
        let obligation = proof_x86_bxor_i64();
        let result = verify_by_evaluation(&obligation);
        assert!(matches!(result, VerificationResult::Valid), "x86-64 Bxor_I64 proof failed: {:?}", result);
    }

    #[test]
    fn test_x86_64_bnot_i64_proof() {
        let obligation = proof_x86_bnot_i64();
        let result = verify_by_evaluation(&obligation);
        assert!(matches!(result, VerificationResult::Valid), "x86-64 Bnot_I64 proof failed: {:?}", result);
    }

    // -----------------------------------------------------------------------
    // Shift lowering proof tests (32-bit)
    // -----------------------------------------------------------------------

    #[test]
    fn test_x86_64_ishl_i32_proof() {
        let obligation = proof_x86_ishl_i32();
        let result = verify_by_evaluation(&obligation);
        assert!(matches!(result, VerificationResult::Valid), "x86-64 Ishl_I32 proof failed: {:?}", result);
    }

    #[test]
    fn test_x86_64_ushr_i32_proof() {
        let obligation = proof_x86_ushr_i32();
        let result = verify_by_evaluation(&obligation);
        assert!(matches!(result, VerificationResult::Valid), "x86-64 Ushr_I32 proof failed: {:?}", result);
    }

    #[test]
    fn test_x86_64_sshr_i32_proof() {
        let obligation = proof_x86_sshr_i32();
        let result = verify_by_evaluation(&obligation);
        assert!(matches!(result, VerificationResult::Valid), "x86-64 Sshr_I32 proof failed: {:?}", result);
    }

    // -----------------------------------------------------------------------
    // Shift lowering proof tests (64-bit)
    // -----------------------------------------------------------------------

    #[test]
    fn test_x86_64_ishl_i64_proof() {
        let obligation = proof_x86_ishl_i64();
        let result = verify_by_evaluation(&obligation);
        assert!(matches!(result, VerificationResult::Valid), "x86-64 Ishl_I64 proof failed: {:?}", result);
    }

    #[test]
    fn test_x86_64_ushr_i64_proof() {
        let obligation = proof_x86_ushr_i64();
        let result = verify_by_evaluation(&obligation);
        assert!(matches!(result, VerificationResult::Valid), "x86-64 Ushr_I64 proof failed: {:?}", result);
    }

    #[test]
    fn test_x86_64_sshr_i64_proof() {
        let obligation = proof_x86_sshr_i64();
        let result = verify_by_evaluation(&obligation);
        assert!(matches!(result, VerificationResult::Valid), "x86-64 Sshr_I64 proof failed: {:?}", result);
    }

    // -----------------------------------------------------------------------
    // 8-bit exhaustive proof tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_x86_64_iadd_i8_proof() {
        let obligation = proof_x86_iadd_i8();
        let result = verify_by_evaluation(&obligation);
        assert!(matches!(result, VerificationResult::Valid), "x86-64 Iadd_I8 proof failed: {:?}", result);
    }

    #[test]
    fn test_x86_64_isub_i8_proof() {
        let obligation = proof_x86_isub_i8();
        let result = verify_by_evaluation(&obligation);
        assert!(matches!(result, VerificationResult::Valid), "x86-64 Isub_I8 proof failed: {:?}", result);
    }

    #[test]
    fn test_x86_64_imul_i8_proof() {
        let obligation = proof_x86_imul_i8();
        let result = verify_by_evaluation(&obligation);
        assert!(matches!(result, VerificationResult::Valid), "x86-64 Imul_I8 proof failed: {:?}", result);
    }

    // -----------------------------------------------------------------------
    // Meta: verify all proofs at once
    // -----------------------------------------------------------------------

    #[test]
    fn test_all_x86_64_proofs() {
        let proofs = all_x86_64_proofs();
        assert_eq!(proofs.len(), 29, "Expected 29 x86-64 proof obligations");

        for proof in &proofs {
            let result = verify_by_evaluation(proof);
            assert!(
                matches!(result, VerificationResult::Valid),
                "Proof '{}' failed: {:?}",
                proof.name,
                result,
            );
        }
    }
}
