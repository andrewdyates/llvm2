// llvm2-lower/abi.rs - Apple AArch64 calling convention
//
// Author: Andrew Yates <ayates@dropbox.com>
// Copyright 2026 Dropbox, Inc. | License: Apache-2.0
//
// Reference: LLVM AArch64CallingConvention.td, AArch64ISelLowering.cpp
// Reference: ARM AAPCS64 + Apple arm64 ABI delta (DarwinPCS)

//! Apple AArch64 (arm64, DarwinPCS) calling convention.
//!
//! This implements the Apple variant of the AAPCS64 calling convention used on
//! macOS Apple Silicon. Key differences from standard AAPCS64:
//! - X18 is reserved (platform register, do not touch)
//! - Frame pointer (X29) is mandatory on Darwin
//! - Variadic arguments are always passed on the stack (not in registers)
//! - Stack alignment is 16 bytes

use crate::types::Type;

// Import physical register from llvm2-ir (canonical source of truth).
pub use llvm2_ir::regs::PReg;

// ---------------------------------------------------------------------------
// ABI register constants — re-exported from llvm2-ir (single source of truth)
// ---------------------------------------------------------------------------

/// GPR constants for ABI use.
///
/// These re-export the canonical constants from `llvm2_ir::regs` so that
/// downstream code can write `gpr::X0` without a long import path.
pub mod gpr {
    pub use llvm2_ir::regs::{
        X0, X1, X2, X3, X4, X5, X6, X7, X8, X9, X10, X11, X12, X13, X14, X15,
        X16, X17, X18, X19, X20, X21, X22, X23, X24, X25, X26, X27, X28,
        FP, LR,
        // FPR128 (V/Q registers) — 128-bit NEON/SIMD
        V0, V1, V2, V3, V4, V5, V6, V7,
        V8, V9, V10, V11, V12, V13, V14, V15,
        V16, V17, V18, V19, V20, V21, V22, V23,
        V24, V25, V26, V27, V28, V29, V30, V31,
        // FPR64 (D registers) — 64-bit FP, aliases lower 64 bits of V
        D0, D1, D2, D3, D4, D5, D6, D7,
        // FPR32 (S registers) — 32-bit FP, aliases lower 32 bits of V
        S0, S1, S2, S3, S4, S5, S6, S7,
    };
}

// ---------------------------------------------------------------------------
// Argument location classification
// ---------------------------------------------------------------------------

/// Where a single argument or return value is placed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ArgLocation {
    /// In a general-purpose register (X0-X7 for args, X0-X7 for returns).
    Reg(PReg),
    /// On the stack at `[SP + offset]`, occupying `size` bytes.
    Stack { offset: i64, size: u32 },
    /// Large aggregate passed/returned indirectly via pointer in `ptr_reg`.
    /// Caller allocates memory, callee accesses via pointer.
    /// For returns, X8 holds the pointer (sret convention).
    Indirect { ptr_reg: PReg },
    /// Multiple consecutive typed registers for HFA (Homogeneous Floating-point
    /// Aggregate) passing. Each register is listed explicitly with the correct
    /// register class (S0-S3 for F32 HFA, D0-D3 for F64 HFA).
    ///
    /// Per AAPCS64, HFA members are passed in consecutive FPR registers of the
    /// element type's width. A 2-member F32 HFA uses S0, S1 (not V0, V1).
    ///
    /// This variant replaces the previous encoding where HFAs were represented
    /// as a single `Reg(V0)` with an implicit count. Explicit register lists
    /// enable unambiguous code generation.
    RegSequence(Vec<PReg>),
}

// ---------------------------------------------------------------------------
// Aggregate classification result
// ---------------------------------------------------------------------------

/// Result of classifying an aggregate or variadic argument for the ABI.
///
/// Unlike `ArgLocation` which is used during full parameter list classification
/// (with register allocation state), `ClassifyResult` describes the *category*
/// of passing convention for a single type, independent of register allocation.
/// This is useful for callers that need to know the passing strategy before
/// doing full argument lowering.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ClassifyResult {
    /// Passed in one or more general-purpose registers.
    /// `regs` contains placeholder registers (e.g., X0/X1) -- actual register
    /// assignment depends on the full parameter list context.
    InRegs { regs: Vec<PReg> },
    /// Passed on the stack at the given offset with the given size and alignment.
    OnStack { offset: i64, size: u32, align: u32 },
    /// Passed indirectly via a pointer in `ptr_reg`.
    /// Caller allocates memory; callee accesses via the pointer.
    Indirect { ptr_reg: PReg },
    /// Homogeneous Floating-point Aggregate: 1-4 same-type FP members
    /// passed in consecutive V registers.
    Hfa { base_ty: HfaBaseType, count: usize, first_reg: PReg },
}

/// The base floating-point element type of an HFA (Homogeneous Floating-point
/// Aggregate). Per AAPCS64, only F32 and F64 are valid HFA element types.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HfaBaseType {
    /// 32-bit float (float / f32)
    F32,
    /// 64-bit float (double / f64)
    F64,
}

// ---------------------------------------------------------------------------
// Register arrays (static, following LLVM AArch64ISelLowering.cpp)
// ---------------------------------------------------------------------------

/// GPR argument registers: X0-X7
const GPR_ARG_REGS: [PReg; 8] = [
    gpr::X0, gpr::X1, gpr::X2, gpr::X3,
    gpr::X4, gpr::X5, gpr::X6, gpr::X7,
];

/// FPR argument registers: V0-V7 (128-bit view, used for all FP/SIMD args)
const FPR_ARG_REGS: [PReg; 8] = [
    gpr::V0, gpr::V1, gpr::V2, gpr::V3,
    gpr::V4, gpr::V5, gpr::V6, gpr::V7,
];

/// S-register argument sequence: S0-S7 (typed 32-bit view of V0-V7 for F32 args).
///
/// These alias the same physical registers as V0-V7 / D0-D7 but use the
/// Fpr32 register class, enabling size-correct instruction selection.
const S_ARG_REGS: [PReg; 8] = [
    gpr::S0, gpr::S1, gpr::S2, gpr::S3,
    gpr::S4, gpr::S5, gpr::S6, gpr::S7,
];

/// D-register argument sequence: D0-D7 (typed 64-bit view of V0-V7 for F64 args).
///
/// These alias the same physical registers as V0-V7 but use the Fpr64
/// register class.
const D_ARG_REGS: [PReg; 8] = [
    gpr::D0, gpr::D1, gpr::D2, gpr::D3,
    gpr::D4, gpr::D5, gpr::D6, gpr::D7,
];

/// Callee-saved GPRs: X19-X28 + FP(X29) + LR(X30)
/// LR is saved by the call itself but must be restored on return.
const CALLEE_SAVED_GPRS: [PReg; 12] = [
    gpr::X19, gpr::X20, gpr::X21, gpr::X22,
    gpr::X23, gpr::X24, gpr::X25, gpr::X26,
    gpr::X27, gpr::X28, gpr::FP, gpr::LR,
];

/// Callee-saved FPRs: V8-V15 (lower 64 bits only per AAPCS64).
const CALLEE_SAVED_FPRS: [PReg; 8] = [
    gpr::V8, gpr::V9, gpr::V10, gpr::V11,
    gpr::V12, gpr::V13, gpr::V14, gpr::V15,
];

/// Call-clobbered GPRs: X0-X18 (everything not callee-saved or SP).
/// X18 is reserved on Apple but still clobbered across calls.
const CALL_CLOBBER_GPRS: [PReg; 19] = [
    gpr::X0, gpr::X1, gpr::X2, gpr::X3,
    gpr::X4, gpr::X5, gpr::X6, gpr::X7,
    gpr::X8, gpr::X9, gpr::X10, gpr::X11,
    gpr::X12, gpr::X13, gpr::X14, gpr::X15,
    gpr::X16, gpr::X17, gpr::X18,
];

/// Call-clobbered FPRs: V0-V7, V16-V31 (everything not callee-saved V8-V15).
///
/// Per AAPCS64, V0-V7 are argument/result registers (clobbered by calls)
/// and V16-V31 are caller-saved temporaries.
const CALL_CLOBBER_FPRS: [PReg; 24] = [
    gpr::V0,  gpr::V1,  gpr::V2,  gpr::V3,
    gpr::V4,  gpr::V5,  gpr::V6,  gpr::V7,
    gpr::V16, gpr::V17, gpr::V18, gpr::V19,
    gpr::V20, gpr::V21, gpr::V22, gpr::V23,
    gpr::V24, gpr::V25, gpr::V26, gpr::V27,
    gpr::V28, gpr::V29, gpr::V30, gpr::V31,
];

// ---------------------------------------------------------------------------
// Apple AArch64 ABI
// ---------------------------------------------------------------------------

/// Apple AArch64 (DarwinPCS) calling convention implementation.
///
/// Follows the ARM AAPCS64 base with Apple-specific modifications:
/// - X18 reserved (platform register)
/// - X29 (FP) mandatory
/// - Variadic args always on stack
/// - 16-byte stack alignment
pub struct AppleAArch64ABI;

impl AppleAArch64ABI {
    /// Detect if a type is a Homogeneous Floating-point Aggregate (HFA).
    ///
    /// An HFA is a struct or array of 1-4 members of the same floating-point type
    /// (F32 or F64). Per AAPCS64 and Apple ABI, HFAs are passed/returned in
    /// consecutive FPR registers (V0-V7 for args, V0-V3 for returns).
    ///
    /// Returns `Some((element_type, count))` if HFA, `None` otherwise.
    ///
    /// Reference: AAPCS64 sec. 6.4.2 "Homogeneous Aggregates"
    /// Reference: Apple ARM64 ABI documentation
    fn detect_hfa(ty: &Type) -> Option<(Type, usize)> {
        match ty {
            Type::Struct(fields) => {
                if fields.is_empty() || fields.len() > 4 {
                    return None;
                }
                // All fields must be the same FP type (F32 or F64).
                // Nested structs/arrays: recursively flatten to check.
                let mut flat_fields = Vec::new();
                for field in fields {
                    Self::flatten_hfa_fields(field, &mut flat_fields);
                }
                if flat_fields.is_empty() || flat_fields.len() > 4 {
                    return None;
                }
                let base = &flat_fields[0];
                if !matches!(base, Type::F32 | Type::F64) {
                    return None;
                }
                if flat_fields.iter().all(|f| f == base) {
                    Some((base.clone(), flat_fields.len()))
                } else {
                    None
                }
            }
            Type::Array(elem, count) => {
                let count = *count as usize;
                if count == 0 || count > 4 {
                    return None;
                }
                // Element must be F32 or F64 (or itself an HFA that flattens to same type).
                match elem.as_ref() {
                    Type::F32 => Some((Type::F32, count)),
                    Type::F64 => Some((Type::F64, count)),
                    _ => {
                        // Check if element is itself an HFA (e.g., array of float-structs)
                        if let Some((base, inner_count)) = Self::detect_hfa(elem) {
                            let total = count * inner_count;
                            if total <= 4 {
                                Some((base, total))
                            } else {
                                None
                            }
                        } else {
                            None
                        }
                    }
                }
            }
            _ => None,
        }
    }

    /// Recursively flatten struct/array fields to their leaf FP types for HFA detection.
    fn flatten_hfa_fields(ty: &Type, out: &mut Vec<Type>) {
        match ty {
            Type::F32 | Type::F64 => out.push(ty.clone()),
            Type::Struct(fields) => {
                for field in fields {
                    Self::flatten_hfa_fields(field, out);
                }
            }
            Type::Array(elem, count) => {
                for _ in 0..*count {
                    Self::flatten_hfa_fields(elem, out);
                }
            }
            // Any non-FP leaf type makes this not an HFA
            _ => out.push(ty.clone()),
        }
    }

    /// Classify function parameters into register/stack locations.
    ///
    /// Rules (Apple arm64, non-variadic):
    /// - Integer/pointer types (I8, I16, I32, I64, B1): next available X0-X7
    /// - Float types (F32, F64): next available V0-V7
    /// - HFA (Homogeneous Floating-point Aggregate): consecutive V0-V7 registers
    /// - I128: uses two consecutive GPR slots (must start on even register for
    ///   alignment on standard AAPCS; Apple relaxes this but we follow the
    ///   conservative rule for correctness)
    /// - Overflow to stack, 16-byte aligned slots
    /// - Aggregates > 16 bytes: indirect via pointer in next GPR
    pub fn classify_params(params: &[Type]) -> Vec<ArgLocation> {
        let mut result = Vec::with_capacity(params.len());
        let mut gpr_idx: usize = 0;
        let mut fpr_idx: usize = 0;
        let mut stack_offset: i64 = 0;

        for ty in params {
            match ty {
                // Integer and boolean types -> GPR
                Type::B1 | Type::I8 | Type::I16 | Type::I32 | Type::I64 => {
                    if gpr_idx < GPR_ARG_REGS.len() {
                        result.push(ArgLocation::Reg(GPR_ARG_REGS[gpr_idx]));
                        gpr_idx += 1;
                    } else {
                        let size = if ty.bytes() < 8 { 8 } else { ty.bytes() };
                        result.push(ArgLocation::Stack {
                            offset: stack_offset,
                            size,
                        });
                        stack_offset += align_up(size as i64, 8);
                    }
                }

                // 128-bit integer -> two GPRs or stack
                Type::I128 => {
                    // Need 2 consecutive GPRs. Apple doesn't require even
                    // alignment but we do for simplicity.
                    if gpr_idx + 1 < GPR_ARG_REGS.len() {
                        // For I128 we return the first register; the second is
                        // implicitly the next one. A real ABI lowering would
                        // produce two locations; for the scaffold we use
                        // Indirect to signal the pair.
                        result.push(ArgLocation::Indirect {
                            ptr_reg: GPR_ARG_REGS[gpr_idx],
                        });
                        gpr_idx += 2;
                    } else {
                        gpr_idx = GPR_ARG_REGS.len(); // exhaust remaining
                        result.push(ArgLocation::Stack {
                            offset: stack_offset,
                            size: 16,
                        });
                        stack_offset += 16;
                    }
                }

                // Float types -> FPR
                Type::F32 | Type::F64 => {
                    if fpr_idx < FPR_ARG_REGS.len() {
                        result.push(ArgLocation::Reg(FPR_ARG_REGS[fpr_idx]));
                        fpr_idx += 1;
                    } else {
                        let size = ty.bytes();
                        result.push(ArgLocation::Stack {
                            offset: stack_offset,
                            size,
                        });
                        stack_offset += align_up(size as i64, 8);
                    }
                }

                // 128-bit NEON/SIMD vector -> FPR (V0-V7)
                //
                // Per AAPCS64/Apple ABI, SIMD&FP arguments use the same V0-V7
                // register sequence as scalar float arguments, with independent
                // numbering from the GPR sequence.
                Type::V128 => {
                    if fpr_idx < FPR_ARG_REGS.len() {
                        result.push(ArgLocation::Reg(FPR_ARG_REGS[fpr_idx]));
                        fpr_idx += 1;
                    } else {
                        // 128-bit vectors are 16-byte aligned on stack.
                        stack_offset = align_up(stack_offset, 16);
                        result.push(ArgLocation::Stack {
                            offset: stack_offset,
                            size: 16,
                        });
                        stack_offset += 16;
                    }
                }

                // Aggregate types: check for HFA first, then size-based classification.
                Type::Struct(_) | Type::Array(_, _) => {
                    // HFA: 1-4 same-type FP fields -> consecutive typed FPR registers.
                    // Per AAPCS64, HFA members use the element type's register class:
                    //   F32 HFA -> S0-S7 (Fpr32 class)
                    //   F64 HFA -> D0-D7 (Fpr64 class)
                    // If there aren't enough FPR slots for the entire HFA, the
                    // whole thing goes on the stack (all-or-nothing rule).
                    //
                    // Reference: AAPCS64 sec. 6.4.2, Apple ARM64 ABI
                    // Reference: LLVM AArch64CallingConvention.td line 390+
                    if let Some((base_ty, count)) = Self::detect_hfa(ty) {
                        if fpr_idx + count <= FPR_ARG_REGS.len() {
                            // Select the typed register array based on HFA element type.
                            let typed_regs: &[PReg] = match base_ty {
                                Type::F32 => &S_ARG_REGS,
                                Type::F64 => &D_ARG_REGS,
                                _ => &FPR_ARG_REGS, // fallback (shouldn't happen)
                            };
                            // Build explicit register sequence for code generation.
                            let regs: Vec<PReg> = (fpr_idx..fpr_idx + count)
                                .map(|i| typed_regs[i])
                                .collect();
                            result.push(ArgLocation::RegSequence(regs));
                            fpr_idx += count;
                        } else {
                            // Not enough FPR slots: entire HFA goes on stack.
                            let size = ty.bytes();
                            result.push(ArgLocation::Stack {
                                offset: stack_offset,
                                size,
                            });
                            stack_offset += align_up(size as i64, 8);
                        }
                    } else {
                        // Non-HFA aggregate: pass in GPRs or indirect.
                        let size = ty.bytes();
                        if size <= 16 && gpr_idx < GPR_ARG_REGS.len() {
                            // Small aggregates: pass in register(s)
                            if size <= 8 {
                                result.push(ArgLocation::Reg(GPR_ARG_REGS[gpr_idx]));
                                gpr_idx += 1;
                            } else if gpr_idx + 1 < GPR_ARG_REGS.len() {
                                // 9-16 bytes: two registers
                                result.push(ArgLocation::Indirect {
                                    ptr_reg: GPR_ARG_REGS[gpr_idx],
                                });
                                gpr_idx += 2;
                            } else {
                                result.push(ArgLocation::Stack {
                                    offset: stack_offset,
                                    size,
                                });
                                stack_offset += align_up(size as i64, 8);
                            }
                        } else if size > 16 {
                            // Large aggregates: pass indirect via pointer
                            if gpr_idx < GPR_ARG_REGS.len() {
                                result.push(ArgLocation::Indirect {
                                    ptr_reg: GPR_ARG_REGS[gpr_idx],
                                });
                                gpr_idx += 1;
                            } else {
                                result.push(ArgLocation::Stack {
                                    offset: stack_offset,
                                    size: 8, // pointer size
                                });
                                stack_offset += 8;
                            }
                        } else {
                            result.push(ArgLocation::Stack {
                                offset: stack_offset,
                                size,
                            });
                            stack_offset += align_up(size as i64, 8);
                        }
                    }
                }
            }
        }

        result
    }

    /// Classify return values.
    ///
    /// Rules:
    /// - Integer: X0 (single), X0+X1 (pair)
    /// - Float: V0 (single), V0+V1 (pair)
    /// - HFA aggregate: V0-V3 (1-4 FP registers)
    /// - Small aggregate (<= 16 bytes, non-HFA): X0 (<=8 bytes), X0+X1 (<=16 bytes)
    /// - Large aggregate (> 16 bytes): indirect via X8 pointer (sret)
    /// - Up to 4 return values in registers per AAPCS64
    pub fn classify_returns(returns: &[Type]) -> Vec<ArgLocation> {
        let mut result = Vec::with_capacity(returns.len());
        let mut gpr_idx: usize = 0;
        let mut fpr_idx: usize = 0;

        for ty in returns {
            match ty {
                Type::B1 | Type::I8 | Type::I16 | Type::I32 | Type::I64 => {
                    if gpr_idx < GPR_ARG_REGS.len() {
                        result.push(ArgLocation::Reg(GPR_ARG_REGS[gpr_idx]));
                        gpr_idx += 1;
                    } else {
                        // Too many return values for registers -> indirect
                        result.push(ArgLocation::Indirect { ptr_reg: gpr::X8 });
                    }
                }

                Type::I128 => {
                    if gpr_idx + 1 < GPR_ARG_REGS.len() {
                        result.push(ArgLocation::Indirect {
                            ptr_reg: GPR_ARG_REGS[gpr_idx],
                        });
                        gpr_idx += 2;
                    } else {
                        result.push(ArgLocation::Indirect { ptr_reg: gpr::X8 });
                    }
                }

                Type::F32 | Type::F64 => {
                    if fpr_idx < FPR_ARG_REGS.len() {
                        result.push(ArgLocation::Reg(FPR_ARG_REGS[fpr_idx]));
                        fpr_idx += 1;
                    } else {
                        result.push(ArgLocation::Indirect { ptr_reg: gpr::X8 });
                    }
                }

                // 128-bit NEON/SIMD vector return -> V0
                Type::V128 => {
                    if fpr_idx < FPR_ARG_REGS.len() {
                        result.push(ArgLocation::Reg(FPR_ARG_REGS[fpr_idx]));
                        fpr_idx += 1;
                    } else {
                        result.push(ArgLocation::Indirect { ptr_reg: gpr::X8 });
                    }
                }

                // Aggregate returns: HFA in typed FPR regs, small structs in X regs, large via sret.
                Type::Struct(_) | Type::Array(_, _) => {
                    // Check for HFA: return in consecutive typed FPR registers.
                    // F32 HFA -> S0-S3, F64 HFA -> D0-D3.
                    if let Some((base_ty, count)) = Self::detect_hfa(ty) {
                        if fpr_idx + count <= 4 {
                            // HFA return in typed FPR registers (max 4).
                            let typed_regs: &[PReg] = match base_ty {
                                Type::F32 => &S_ARG_REGS,
                                Type::F64 => &D_ARG_REGS,
                                _ => &FPR_ARG_REGS,
                            };
                            let regs: Vec<PReg> = (fpr_idx..fpr_idx + count)
                                .map(|i| typed_regs[i])
                                .collect();
                            result.push(ArgLocation::RegSequence(regs));
                            fpr_idx += count;
                        } else {
                            result.push(ArgLocation::Indirect { ptr_reg: gpr::X8 });
                        }
                    } else {
                        // Non-HFA aggregate: small structs in X0/X1, large via sret.
                        let size = ty.bytes();
                        if size <= 8 && gpr_idx < GPR_ARG_REGS.len() {
                            result.push(ArgLocation::Reg(GPR_ARG_REGS[gpr_idx]));
                            gpr_idx += 1;
                        } else if size <= 16 && gpr_idx + 1 < GPR_ARG_REGS.len() {
                            // 9-16 bytes: two GPRs. Record the first; second is implicit.
                            result.push(ArgLocation::Indirect {
                                ptr_reg: GPR_ARG_REGS[gpr_idx],
                            });
                            gpr_idx += 2;
                        } else {
                            // > 16 bytes or no registers left: indirect via X8.
                            result.push(ArgLocation::Indirect { ptr_reg: gpr::X8 });
                        }
                    }
                }
            }
        }

        result
    }

    /// Callee-saved GPRs that a function must preserve.
    /// Includes X19-X28, FP (X29), LR (X30).
    pub fn callee_saved_gprs() -> &'static [PReg] {
        &CALLEE_SAVED_GPRS
    }

    /// Callee-saved FPRs (V8-V15, lower 64 bits only).
    pub fn callee_saved_fprs() -> &'static [PReg] {
        &CALLEE_SAVED_FPRS
    }

    /// Call-clobbered GPRs (X0-X18).
    pub fn call_clobber_gprs() -> &'static [PReg] {
        &CALL_CLOBBER_GPRS
    }

    /// Call-clobbered FPRs (V0-V7, V16-V31).
    ///
    /// These registers are NOT preserved across function calls. V8-V15 are
    /// callee-saved (lower 64 bits only on Apple AArch64).
    pub fn call_clobber_fprs() -> &'static [PReg] {
        &CALL_CLOBBER_FPRS
    }

    /// Typed FPR argument registers for F32: S0-S7.
    ///
    /// Returns the 32-bit view of the argument FPR registers. S0-S7 alias
    /// the lower 32 bits of V0-V7 and share the same FPR allocation sequence.
    pub fn s_arg_regs() -> &'static [PReg] {
        &S_ARG_REGS
    }

    /// Typed FPR argument registers for F64: D0-D7.
    ///
    /// Returns the 64-bit view of the argument FPR registers. D0-D7 alias
    /// the lower 64 bits of V0-V7 and share the same FPR allocation sequence.
    pub fn d_arg_regs() -> &'static [PReg] {
        &D_ARG_REGS
    }

    /// Classify a single floating-point or SIMD argument, returning the
    /// appropriately-typed register.
    ///
    /// Unlike `classify_params()` which always returns V registers (Fpr128),
    /// this method returns the correctly-sized register alias:
    /// - F32  -> S0-S7 (Fpr32 class)
    /// - F64  -> D0-D7 (Fpr64 class)
    /// - V128 -> V0-V7 (Fpr128 class)
    ///
    /// Returns `Some((location, next_fpr_idx))` if an FPR slot is available,
    /// `None` if all FPR slots are exhausted (caller should use stack).
    ///
    /// This is useful for instruction selection where the register class
    /// must match the operand size (e.g., `FMOV S0, S1` vs `FMOV D0, D1`).
    pub fn classify_fp_arg(ty: &Type, fpr_idx: usize) -> Option<(ArgLocation, usize)> {
        if fpr_idx >= FPR_ARG_REGS.len() {
            return None;
        }
        match ty {
            Type::F32 => Some((ArgLocation::Reg(S_ARG_REGS[fpr_idx]), fpr_idx + 1)),
            Type::F64 => Some((ArgLocation::Reg(D_ARG_REGS[fpr_idx]), fpr_idx + 1)),
            Type::V128 => Some((ArgLocation::Reg(FPR_ARG_REGS[fpr_idx]), fpr_idx + 1)),
            _ => None,
        }
    }

    /// Classify a 128-bit NEON vector argument.
    ///
    /// NEON vectors use the same V0-V7 FPR sequence as scalar FP arguments.
    /// Returns `Some((location, next_fpr_idx))` if an FPR slot is available,
    /// `None` if all FPR slots are exhausted.
    pub fn classify_vector_arg(fpr_idx: usize) -> Option<(ArgLocation, usize)> {
        if fpr_idx >= FPR_ARG_REGS.len() {
            return None;
        }
        Some((ArgLocation::Reg(FPR_ARG_REGS[fpr_idx]), fpr_idx + 1))
    }

    /// Total stack space consumed by overflow arguments.
    pub fn stack_args_size(params: &[Type]) -> i64 {
        let locs = Self::classify_params(params);
        locs.iter()
            .filter_map(|loc| {
                if let ArgLocation::Stack { offset, size } = loc {
                    Some(offset + align_up(*size as i64, 8))
                } else {
                    None
                }
            })
            .max()
            .unwrap_or(0)
    }

    /// Classify parameters for a variadic function call (Apple AArch64 ABI).
    ///
    /// Apple arm64 variadic convention:
    /// - Fixed params (`params[0..fixed_count]`) follow normal ABI classification
    ///   (GPR X0-X7, FPR V0-V7, then stack overflow).
    /// - Variadic params (`params[fixed_count..]`) are ALL placed on the stack,
    ///   each 8-byte aligned, regardless of type. This differs from standard
    ///   AAPCS64 which allows variadic args in registers.
    ///
    /// Reference: Apple ARM64 Function Calling Conventions, "Variadic Functions"
    /// Reference: LLVM AArch64ISelLowering.cpp CC_AArch64_DarwinPCS_VarArg
    pub fn classify_params_variadic(fixed_count: usize, params: &[Type]) -> Vec<ArgLocation> {
        let fixed_params = if fixed_count <= params.len() {
            &params[..fixed_count]
        } else {
            params
        };

        // Classify fixed params normally.
        let mut result = Self::classify_params(fixed_params);

        // Determine the stack offset after fixed parameter overflow.
        let mut stack_offset: i64 = result
            .iter()
            .filter_map(|loc| {
                if let ArgLocation::Stack { offset, size } = loc {
                    Some(offset + align_up(*size as i64, 8))
                } else {
                    None
                }
            })
            .max()
            .unwrap_or(0);

        // All variadic arguments go on the stack, 8-byte aligned.
        for ty in params.iter().skip(fixed_count) {
            let size = if ty.bytes() < 8 { 8 } else { ty.bytes() };
            result.push(ArgLocation::Stack {
                offset: stack_offset,
                size,
            });
            stack_offset += align_up(size as i64, 8);
        }

        result
    }

    /// Total stack space consumed by overflow arguments in a variadic call.
    pub fn stack_args_size_variadic(fixed_count: usize, params: &[Type]) -> i64 {
        let locs = Self::classify_params_variadic(fixed_count, params);
        locs.iter()
            .filter_map(|loc| {
                if let ArgLocation::Stack { offset, size } = loc {
                    Some(offset + align_up(*size as i64, 8))
                } else {
                    None
                }
            })
            .max()
            .unwrap_or(0)
    }

    // -----------------------------------------------------------------------
    // Standalone classification helpers (ClassifyResult API)
    // -----------------------------------------------------------------------

    /// Classify how an aggregate type is passed as a parameter.
    ///
    /// This performs type-level classification without consuming register
    /// allocation state. The returned `ClassifyResult` uses placeholder
    /// registers (X0/X1 for GPR, V0 for FPR) to indicate the *kind* of
    /// passing convention. Actual register assignment happens during full
    /// parameter list classification via `classify_params()`.
    ///
    /// Rules (Apple AArch64 / AAPCS64):
    /// - HFA (1-4 same FP fields) -> `Hfa` (consecutive V registers)
    /// - Aggregate <= 8 bytes -> `InRegs` (one GPR)
    /// - Aggregate 9-16 bytes -> `InRegs` (two GPRs)
    /// - Aggregate > 16 bytes -> `Indirect` (pointer in GPR)
    ///
    /// Reference: AAPCS64 sec. 6.4.2, Apple ARM64 ABI documentation
    pub fn classify_aggregate(ty: &Type) -> ClassifyResult {
        // Check for HFA first (takes priority over size-based classification).
        if let Some((base_ty, count)) = Self::classify_hfa(ty) {
            return ClassifyResult::Hfa {
                base_ty,
                count,
                first_reg: gpr::V0,
            };
        }

        // Non-HFA aggregate: size-based classification.
        match ty.bytes() {
            0..=8 => ClassifyResult::InRegs {
                regs: vec![gpr::X0],
            },
            9..=16 => ClassifyResult::InRegs {
                regs: vec![gpr::X0, gpr::X1],
            },
            _ => ClassifyResult::Indirect { ptr_reg: gpr::X0 },
        }
    }

    /// Classify a single variadic argument (Apple AArch64 ABI).
    ///
    /// On Apple arm64, ALL variadic arguments are passed on the stack,
    /// regardless of type. Each argument occupies at least 8 bytes and is
    /// 8-byte aligned. This differs from standard AAPCS64 which allows
    /// variadic args in registers.
    ///
    /// Returns `(classification, next_stack_offset)` so callers can chain
    /// multiple variadic argument classifications.
    ///
    /// Reference: Apple ARM64 Function Calling Conventions, "Variadic Functions"
    pub fn classify_variadic(ty: &Type, stack_offset: i64) -> (ClassifyResult, i64) {
        let offset = align_up(stack_offset, 8);
        let size = ty.bytes().max(8);
        let next_offset = offset + align_up(size as i64, 8);

        (
            ClassifyResult::OnStack {
                offset,
                size,
                align: 8,
            },
            next_offset,
        )
    }

    /// Public wrapper around `detect_hfa` that returns `HfaBaseType` instead
    /// of `Type`.
    ///
    /// Returns `Some((base_type, count))` if the type is a Homogeneous
    /// Floating-point Aggregate with 1-4 members of the same FP type,
    /// `None` otherwise.
    pub fn classify_hfa(ty: &Type) -> Option<(HfaBaseType, usize)> {
        Self::detect_hfa(ty).and_then(|(base_ty, count)| match base_ty {
            Type::F32 => Some((HfaBaseType::F32, count)),
            Type::F64 => Some((HfaBaseType::F64, count)),
            _ => None,
        })
    }
}

/// Align `value` up to the next multiple of `align`.
fn align_up(value: i64, align: i64) -> i64 {
    (value + align - 1) & !(align - 1)
}

// ===========================================================================
// Exception Handling / Stack Unwinding ABI
// ===========================================================================
//
// Reference: Apple Mach-O compact_unwind_encoding.h
// Reference: DWARF 5 spec, Section 6.4 (Call Frame Information)
// Reference: ~/llvm-project-ref/llvm/lib/Target/AArch64/MCTargetDesc/AArch64AsmBackend.cpp
//            (generateCompactUnwindEncoding, line 576)

// ---------------------------------------------------------------------------
// UnwindInfo — callee-saved register locations relative to frame pointer
// ---------------------------------------------------------------------------

/// Describes where callee-saved registers are stored relative to the
/// frame pointer, for stack unwinding purposes.
///
/// This is the ABI-level representation of unwind information. It captures
/// the register save locations and frame shape that the unwinder needs to
/// reconstruct the caller's register state.
///
/// On AArch64 Darwin, FP (X29) and LR (X30) are always saved as the first
/// pair at the top of the callee-saved area. Additional callee-saved registers
/// are stored at decreasing offsets below FP.
///
/// # Example layout
///
/// ```text
///     [FP + 0]   → saved X29 (FP)
///     [FP + 8]   → saved X30 (LR)
///     [FP - 8]   → saved X20
///     [FP - 16]  → saved X19
///     [FP - 24]  → saved V9  (D9, lower 64 bits)
///     [FP - 32]  → saved V8  (D8, lower 64 bits)
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UnwindInfo {
    /// Callee-saved register save locations, in save order.
    /// Each entry records a register and its offset from the frame pointer.
    pub saved_registers: Vec<SavedRegister>,

    /// Total size of the stack frame in bytes (16-byte aligned).
    pub frame_size: u32,

    /// Whether the function uses a frame pointer (X29).
    /// Always true on Apple AArch64.
    pub has_frame_pointer: bool,

    /// Whether the function is a leaf (makes no calls).
    /// Leaf functions may use a simplified unwind encoding.
    pub is_leaf: bool,

    /// Whether the function has dynamic stack allocation (alloca).
    /// Dynamic frames cannot use compact unwind encoding and require
    /// full DWARF CFI fallback.
    pub has_dynamic_alloc: bool,

    /// Personality function symbol name, if any.
    /// Used for C++ exception handling (__gxx_personality_v0),
    /// Rust panics (__rust_eh_personality), etc.
    pub personality: Option<String>,

    /// Whether a Language-Specific Data Area (LSDA) is present.
    /// The LSDA contains landing pad and type info tables for
    /// exception dispatch.
    pub has_lsda: bool,
}

/// A single callee-saved register and its save location.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SavedRegister {
    /// The physical register that was saved.
    pub reg: PReg,
    /// Offset from the frame pointer (FP / X29) where this register is stored.
    /// Negative values indicate positions below FP (the normal case for
    /// callee-saved registers other than FP/LR).
    pub fp_offset: i32,
    /// Whether this is a floating-point / SIMD register (V-reg / D-reg).
    pub is_fpr: bool,
}

impl UnwindInfo {
    /// Create unwind info for a leaf function with no callee-saved registers
    /// beyond FP/LR (the minimal Apple AArch64 frame).
    ///
    /// This is the common case for simple leaf functions.
    pub fn leaf_frame() -> Self {
        Self {
            saved_registers: vec![
                SavedRegister { reg: gpr::FP, fp_offset: 0, is_fpr: false },
                SavedRegister { reg: gpr::LR, fp_offset: 8, is_fpr: false },
            ],
            frame_size: 16,
            has_frame_pointer: true,
            is_leaf: true,
            has_dynamic_alloc: false,
            personality: None,
            has_lsda: false,
        }
    }

    /// Create unwind info for a standard frame with the given callee-saved
    /// registers (in addition to the mandatory FP/LR pair).
    ///
    /// Registers are laid out in pairs below FP at decreasing offsets.
    /// The `additional_pairs` parameter lists register pairs in save order
    /// (each pair is `(reg1, reg2, is_fpr)`).
    pub fn standard_frame(
        additional_pairs: &[(PReg, PReg, bool)],
        frame_size: u32,
        is_leaf: bool,
    ) -> Self {
        let mut saved = Vec::with_capacity(2 + additional_pairs.len() * 2);

        // FP/LR always at the top.
        saved.push(SavedRegister { reg: gpr::FP, fp_offset: 0, is_fpr: false });
        saved.push(SavedRegister { reg: gpr::LR, fp_offset: 8, is_fpr: false });

        // Additional callee-saved pairs at decreasing offsets.
        for (i, &(reg1, reg2, is_fpr)) in additional_pairs.iter().enumerate() {
            let base_offset = -((i as i32 + 1) * 16);
            saved.push(SavedRegister { reg: reg1, fp_offset: base_offset + 8, is_fpr });
            saved.push(SavedRegister { reg: reg2, fp_offset: base_offset, is_fpr });
        }

        Self {
            saved_registers: saved,
            frame_size,
            has_frame_pointer: true,
            is_leaf,
            has_dynamic_alloc: false,
            personality: None,
            has_lsda: false,
        }
    }

    /// Returns the number of callee-saved register pairs (including FP/LR).
    pub fn num_saved_pairs(&self) -> usize {
        // FP/LR is 1 pair, additional registers come in pairs of 2.
        self.saved_registers.len().div_ceil(2)
    }

    /// Returns only the GPR save entries (non-FPR).
    pub fn saved_gprs(&self) -> Vec<&SavedRegister> {
        self.saved_registers.iter().filter(|r| !r.is_fpr).collect()
    }

    /// Returns only the FPR/SIMD save entries.
    pub fn saved_fprs(&self) -> Vec<&SavedRegister> {
        self.saved_registers.iter().filter(|r| r.is_fpr).collect()
    }
}

// ---------------------------------------------------------------------------
// CompactUnwindEntry — ABI-level compact unwind encoding
// ---------------------------------------------------------------------------

// Darwin compact unwind mode constants (ABI-level, matching Apple spec).
// These mirror the constants in llvm2-codegen/frame.rs but are defined
// here so that the ABI module is self-contained.

/// Standard frame-pointer-based unwind mode.
const COMPACT_MODE_FRAME: u32 = 0x0400_0000;
/// Frameless leaf function unwind mode.
#[allow(dead_code)] // Defined for API completeness; will be used for frameless function support.
const COMPACT_MODE_FRAMELESS: u32 = 0x0200_0000;
/// Fallback to full DWARF FDE.
const COMPACT_MODE_DWARF: u32 = 0x0300_0000;

/// Compact unwind register-pair flags for GPRs.
const COMPACT_X19_X20: u32 = 0x0000_0001;
const COMPACT_X21_X22: u32 = 0x0000_0002;
const COMPACT_X23_X24: u32 = 0x0000_0004;
const COMPACT_X25_X26: u32 = 0x0000_0008;
const COMPACT_X27_X28: u32 = 0x0000_0010;

/// Compact unwind register-pair flags for FPRs (D-regs).
const COMPACT_D8_D9: u32 = 0x0000_0100;
const COMPACT_D10_D11: u32 = 0x0000_0200;
const COMPACT_D12_D13: u32 = 0x0000_0400;
const COMPACT_D14_D15: u32 = 0x0000_0800;

/// A compact unwind table entry for one function.
///
/// This is the ABI-level representation of what goes into the
/// `__LD,__compact_unwind` Mach-O section. Each function gets one
/// 32-byte entry that tells the unwinder how to restore the caller's
/// register state.
///
/// # Compact encoding format (ARM64)
///
/// ```text
/// Bits 27-24: mode (0x04=FRAME, 0x02=FRAMELESS, 0x03=DWARF)
/// Bits  4- 0: GPR pair flags (X19/X20 through X27/X28)
/// Bits 11- 8: FPR pair flags (D8/D9 through D14/D15)
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CompactUnwindEntry {
    /// The 32-bit compact unwind encoding.
    pub encoding: u32,
    /// Start offset of the function (will be relocated in the object file).
    pub function_start: u64,
    /// Length of the function in bytes.
    pub function_length: u32,
    /// Personality function pointer (0 if no exception handling).
    pub personality: u64,
    /// Language-Specific Data Area pointer (0 if no LSDA).
    pub lsda: u64,
}

impl CompactUnwindEntry {
    /// Returns true if this entry requires DWARF CFI fallback.
    pub fn needs_dwarf_fallback(&self) -> bool {
        (self.encoding & 0x0F00_0000) == COMPACT_MODE_DWARF
    }

    /// Returns the mode portion of the encoding.
    pub fn mode(&self) -> u32 {
        self.encoding & 0x0F00_0000
    }
}

/// Generate a compact unwind entry from `UnwindInfo` and function metadata.
///
/// Encodes the frame shape as a 32-bit compact unwind encoding per the
/// Apple arm64 format. Falls back to `COMPACT_MODE_DWARF` when the frame
/// cannot be described by compact unwind (frameless, dynamic alloc, etc.).
///
/// # Arguments
///
/// * `info` — Unwind info describing callee-saved register locations.
/// * `function_start` — Start address of the function (virtual, relocated later).
/// * `function_length` — Length of the function in bytes.
pub fn generate_compact_unwind(
    info: &UnwindInfo,
    function_start: u64,
    function_length: u32,
) -> CompactUnwindEntry {
    // Dynamic alloc or no frame pointer => DWARF fallback.
    if info.has_dynamic_alloc || !info.has_frame_pointer {
        let personality_flag = if info.personality.is_some() { 1 } else { 0 };
        let lsda_flag = if info.has_lsda { 1 } else { 0 };
        return CompactUnwindEntry {
            encoding: COMPACT_MODE_DWARF,
            function_start,
            function_length,
            personality: personality_flag,
            lsda: lsda_flag,
        };
    }

    let mut encoding = COMPACT_MODE_FRAME;

    // Scan saved registers for known callee-saved pairs.
    // We look at pairs of saved GPR/FPR registers and set the corresponding
    // compact unwind flags.
    let gprs: Vec<&SavedRegister> = info.saved_registers.iter()
        .filter(|r| !r.is_fpr)
        .collect();

    // Check for each known GPR pair.
    let has_reg_pair = |r1_enc: u16, r2_enc: u16| -> bool {
        gprs.iter().any(|r| r.reg.encoding() == r1_enc)
            && gprs.iter().any(|r| r.reg.encoding() == r2_enc)
    };

    if has_reg_pair(19, 20) { encoding |= COMPACT_X19_X20; }
    if has_reg_pair(21, 22) { encoding |= COMPACT_X21_X22; }
    if has_reg_pair(23, 24) { encoding |= COMPACT_X23_X24; }
    if has_reg_pair(25, 26) { encoding |= COMPACT_X25_X26; }
    if has_reg_pair(27, 28) { encoding |= COMPACT_X27_X28; }

    // FPR pairs: V8-V15 (encoding 72-79 in our unified PReg scheme).
    let fprs: Vec<&SavedRegister> = info.saved_registers.iter()
        .filter(|r| r.is_fpr)
        .collect();

    let has_fpr_pair = |r1_enc: u16, r2_enc: u16| -> bool {
        fprs.iter().any(|r| r.reg.encoding() == r1_enc)
            && fprs.iter().any(|r| r.reg.encoding() == r2_enc)
    };

    if has_fpr_pair(72, 73) { encoding |= COMPACT_D8_D9; }    // V8/V9
    if has_fpr_pair(74, 75) { encoding |= COMPACT_D10_D11; }  // V10/V11
    if has_fpr_pair(76, 77) { encoding |= COMPACT_D12_D13; }  // V12/V13
    if has_fpr_pair(78, 79) { encoding |= COMPACT_D14_D15; }  // V14/V15

    // Propagate personality and LSDA flags from UnwindInfo.
    // The compact unwind entry's personality and lsda fields hold symbol
    // references that the linker resolves. At this ABI-level representation
    // we use sentinel values: personality=1 means "has personality" (the
    // actual symbol is in UnwindInfo.personality), lsda=1 means "has LSDA".
    // The codegen layer maps these to real relocations.
    let personality_flag = if info.personality.is_some() { 1 } else { 0 };
    let lsda_flag = if info.has_lsda { 1 } else { 0 };

    CompactUnwindEntry {
        encoding,
        function_start,
        function_length,
        personality: personality_flag,
        lsda: lsda_flag,
    }
}

// ---------------------------------------------------------------------------
// DwarfCfiOp — DWARF Call Frame Information operations
// ---------------------------------------------------------------------------

/// A DWARF CFI (Call Frame Information) operation.
///
/// These operations describe how the unwinder should interpret the stack
/// frame at each point in the function's code. They are emitted as part
/// of an FDE (Frame Description Entry) in the `__TEXT,__eh_frame` section.
///
/// Used as a fallback when compact unwind encoding is insufficient
/// (e.g., dynamic stack allocation, unusual frame layouts).
///
/// Reference: DWARF 5 spec, Section 6.4.2 (CFA Definition Instructions)
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DwarfCfiOp {
    /// DW_CFA_def_cfa: Define the CFA (Canonical Frame Address) as
    /// `register + offset`. This is the base address from which all
    /// register save locations are computed.
    ///
    /// Initially, CFA = SP + 0 (before the prologue).
    /// After `STP X29, X30, [SP, #-N]!`, CFA = SP + N.
    /// After `MOV X29, SP`, CFA = FP + N.
    DefCfa {
        /// DWARF register number for the CFA base.
        /// 31 = SP, 29 = FP (X29).
        register: u16,
        /// Unsigned offset added to the register value.
        offset: u32,
    },

    /// DW_CFA_offset: Register is saved at `CFA - (factored_offset * data_align_factor)`.
    ///
    /// For AArch64, `data_align_factor = -8`, so a factored offset of 2
    /// means the register is at `CFA - 16`.
    Offset {
        /// DWARF register number of the saved register.
        /// GPR: X0-X30 = 0-30, SP = 31.
        /// FPR: D0-D31 = 64-95.
        register: u16,
        /// Factored offset (multiplied by data_align_factor to get byte offset).
        factored_offset: u32,
    },

    /// DW_CFA_register: Register's value is found in another register.
    ///
    /// Used when a register is moved rather than spilled to the stack
    /// (e.g., `MOV X19, X0` in a custom calling convention).
    Register {
        /// DWARF register number of the register being described.
        register: u16,
        /// DWARF register number where the value is stored.
        value_register: u16,
    },

    /// DW_CFA_restore: Register is restored to its initial (CIE) state.
    ///
    /// Used in the epilogue to indicate that a previously-saved register
    /// is back to its caller-provided value.
    Restore {
        /// DWARF register number of the restored register.
        register: u16,
    },
}

/// AArch64 data alignment factor for DWARF CFI.
/// Stack grows downward in 8-byte increments.
const DWARF_DATA_ALIGN_FACTOR: i32 = -8;

/// DWARF register number for SP (stack pointer) on AArch64.
const DWARF_REG_SP: u16 = 31;
/// DWARF register number for FP (frame pointer / X29) on AArch64.
const DWARF_REG_FP: u16 = 29;
/// DWARF register number for LR (link register / X30) on AArch64.
const DWARF_REG_LR: u16 = 30;

/// Generate DWARF CFI operations from `UnwindInfo`.
///
/// Produces the sequence of CFI operations that describe the function's
/// frame setup. These are used when compact unwind encoding is insufficient
/// (the compact unwind entry uses `UNWIND_ARM64_MODE_DWARF`).
///
/// # Prologue CFI sequence
///
/// 1. After `STP X29, X30, [SP, #-N]!`:
///    - `DefCfa { SP, N }` — CFA moves up by N
///    - `Offset { FP, 2 }` — FP saved at CFA-16
///    - `Offset { LR, 1 }` — LR saved at CFA-8
///
/// 2. After `MOV X29, SP`:
///    - `DefCfa { FP, N }` — CFA now based on FP
///
/// 3. For each callee-saved register:
///    - `Offset { reg, factored_offset }` — register saved at CFA - offset
pub fn generate_dwarf_cfi(info: &UnwindInfo) -> Vec<DwarfCfiOp> {
    let mut ops = Vec::new();

    if !info.has_frame_pointer {
        // Frameless: just define CFA = SP + frame_size.
        ops.push(DwarfCfiOp::DefCfa {
            register: DWARF_REG_SP,
            offset: info.frame_size,
        });
        return ops;
    }

    // Step 1: After STP X29, X30, [SP, #-callee_saved_area]!
    // CFA = SP + callee_saved_area
    let callee_saved_area = info.num_saved_pairs() as u32 * 16;
    ops.push(DwarfCfiOp::DefCfa {
        register: DWARF_REG_SP,
        offset: callee_saved_area,
    });

    // FP saved at CFA - 16: factored offset = 16 / 8 = 2
    ops.push(DwarfCfiOp::Offset {
        register: DWARF_REG_FP,
        factored_offset: 2,
    });

    // LR saved at CFA - 8: factored offset = 8 / 8 = 1
    ops.push(DwarfCfiOp::Offset {
        register: DWARF_REG_LR,
        factored_offset: 1,
    });

    // Step 2: After MOV X29, SP — CFA now based on FP
    ops.push(DwarfCfiOp::DefCfa {
        register: DWARF_REG_FP,
        offset: callee_saved_area,
    });

    // Step 3: Each callee-saved register (skip FP and LR, already handled)
    for saved in &info.saved_registers {
        // Skip FP (encoding 29) and LR (encoding 30) — already described above.
        let enc = saved.reg.encoding();
        if !saved.is_fpr && (enc == 29 || enc == 30) {
            continue;
        }

        // DWARF register number.
        let dwarf_reg = if saved.is_fpr {
            // V-registers: our encoding V0=64..V31=95, DWARF D0=64..D31=95
            enc
        } else {
            // X-registers: encoding 0-31, DWARF 0-31
            enc
        };

        // Compute factored offset from CFA.
        // saved.fp_offset is relative to FP.
        // CFA = FP + callee_saved_area.
        // Register is at FP + saved.fp_offset = CFA - callee_saved_area + saved.fp_offset.
        // Byte offset from CFA = callee_saved_area - saved.fp_offset (note: fp_offset is negative for callee-saves below FP).
        // Factored offset = byte_offset / abs(data_align_factor).
        let byte_offset_from_cfa = callee_saved_area as i32 - saved.fp_offset;
        let factored = byte_offset_from_cfa / (-DWARF_DATA_ALIGN_FACTOR);

        ops.push(DwarfCfiOp::Offset {
            register: dwarf_reg,
            factored_offset: factored as u32,
        });
    }

    ops
}

// ===========================================================================
// Exception Handling ABI — Landing Pads, LSDA, Personality Functions
// ===========================================================================
//
// Reference: Itanium C++ ABI: Exception Handling
//            https://itanium-cxx-abi.github.io/cxx-abi/abi-eh.html
// Reference: DWARF 5 spec, Section 6.4.1 (Structure of Call Frame Information)
// Reference: LLVM libunwind: src/DwarfParser.hpp (LSDA parsing)
// Reference: ~/llvm-project-ref/llvm/include/llvm/MC/MCStreamer.h
//            (personality + LSDA emission)
//
// On Apple AArch64, exception handling uses the "zero-cost" Itanium-derived
// ABI. The runtime uses DWARF unwind tables to walk the stack, and the
// Language-Specific Data Area (LSDA) to dispatch exceptions to the correct
// landing pad. The personality function (e.g., __gxx_personality_v0 for C++)
// interprets the LSDA and decides whether to catch, filter, or propagate.
//
// Layout of the LSDA (per Itanium EH ABI):
//
// ┌─────────────────────────────┐
// │  LSDA Header                │  lpstart encoding, type table encoding,
// │                             │  call site table encoding
// ├─────────────────────────────┤
// │  Call Site Table             │  Maps PC ranges to landing pads + action
// │    entry 0                  │
// │    entry 1                  │
// │    ...                      │
// ├─────────────────────────────┤
// │  Action Table                │  Chains of type filter references
// │    action 0                 │
// │    action 1                 │
// │    ...                      │
// ├─────────────────────────────┤
// │  Type Table                  │  Pointers to typeinfo objects (C++ RTTI)
// │    type 0                   │
// │    type 1                   │
// │    ...                      │
// └─────────────────────────────┘

// ---------------------------------------------------------------------------
// Personality function selection
// ---------------------------------------------------------------------------

/// Known personality function identifiers for exception handling.
///
/// The personality function is called by the unwinder for each frame during
/// exception propagation. It interprets the LSDA to determine whether the
/// current frame should handle the exception (catch), filter it, or
/// continue unwinding.
///
/// Reference: Itanium C++ ABI sec. 2.5.2 "Personality Routine"
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum PersonalityFunction {
    /// C++ personality: `__gxx_personality_v0`
    /// Handles C++ try/catch, RTTI-based type matching, and cleanup.
    GxxPersonalityV0,

    /// Rust personality: `__rust_eh_personality`
    /// Handles Rust panic unwinding with cleanup-only semantics
    /// (Rust does not catch by type, only by cleanup).
    RustEhPersonality,

    /// Objective-C personality: `__objc_personality_v0`
    /// Handles Objective-C @try/@catch/@finally on Apple platforms.
    ObjcPersonalityV0,

    /// C personality (GCC __attribute__((cleanup))): `__gcc_personality_v0`
    /// Minimal personality for C code with cleanup attributes.
    GccPersonalityV0,

    /// Custom personality function referenced by symbol name.
    /// Used for language runtimes not covered by the standard set.
    Custom(String),
}

impl PersonalityFunction {
    /// Returns the linker symbol name for this personality function.
    pub fn symbol_name(&self) -> &str {
        match self {
            Self::GxxPersonalityV0 => "__gxx_personality_v0",
            Self::RustEhPersonality => "__rust_eh_personality",
            Self::ObjcPersonalityV0 => "__objc_personality_v0",
            Self::GccPersonalityV0 => "__gcc_personality_v0",
            Self::Custom(name) => name,
        }
    }

    /// Select the appropriate personality function for a source language.
    ///
    /// This is a convenience constructor for the common case where the
    /// personality function is determined by the source language alone.
    pub fn for_language(lang: SourceLanguage) -> Self {
        match lang {
            SourceLanguage::Cpp => Self::GxxPersonalityV0,
            SourceLanguage::Rust => Self::RustEhPersonality,
            SourceLanguage::ObjectiveC => Self::ObjcPersonalityV0,
            SourceLanguage::C => Self::GccPersonalityV0,
        }
    }
}

/// Source language for personality function selection.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SourceLanguage {
    /// C++ (try/catch/throw)
    Cpp,
    /// Rust (panic/catch_unwind)
    Rust,
    /// Objective-C (@try/@catch/@throw)
    ObjectiveC,
    /// C (cleanup attributes only)
    C,
}

// ---------------------------------------------------------------------------
// Landing pad
// ---------------------------------------------------------------------------

/// A landing pad descriptor for one exception-handling entry point.
///
/// A landing pad is a code location where control transfers when an
/// exception is caught or cleanup is needed during stack unwinding.
/// Each landing pad corresponds to a `catch`, `catch(...)`, or cleanup
/// block in the source language.
///
/// Reference: Itanium C++ ABI sec. 2.5.1 "Landing Pad"
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LandingPad {
    /// Offset of the landing pad code from the function start (in bytes).
    /// The unwinder jumps to `function_start + landing_pad_offset` when
    /// dispatching to this landing pad.
    pub landing_pad_offset: u32,

    /// Types that this landing pad catches.
    ///
    /// Each entry is an index into the LSDA type table. Index 0 is
    /// reserved for "catch-all" (C++ `catch(...)`).
    ///
    /// For C++: each index refers to a `std::type_info*` in the type table.
    /// For Rust: typically empty (Rust uses cleanup-only semantics).
    pub catch_type_indices: Vec<u32>,

    /// Type filter indices for exception specification filters.
    ///
    /// Negative indices in the action table encode exception specification
    /// filters. If the thrown type does NOT match any type in the filter
    /// list, `std::unexpected()` (C++03) or `std::terminate()` is called.
    ///
    /// Each entry is an index into the type table. The filter matches if
    /// the thrown type is NOT in this set.
    pub filter_type_indices: Vec<u32>,

    /// Whether this landing pad is a cleanup-only handler.
    ///
    /// Cleanup landing pads execute destructor/drop code but do not
    /// catch the exception. After cleanup, unwinding continues.
    /// All Rust landing pads are cleanup-only (panics are not caught
    /// by type). C++ cleanup corresponds to local object destructors.
    pub is_cleanup: bool,
}

impl LandingPad {
    /// Create a catch-all landing pad (C++ `catch(...)`).
    pub fn catch_all(landing_pad_offset: u32) -> Self {
        Self {
            landing_pad_offset,
            catch_type_indices: vec![0], // 0 = catch-all
            filter_type_indices: Vec::new(),
            is_cleanup: false,
        }
    }

    /// Create a typed catch landing pad (C++ `catch(SomeType&)`).
    ///
    /// `type_index` is the 1-based index into the LSDA type table
    /// for the caught type's `std::type_info`.
    pub fn catch_typed(landing_pad_offset: u32, type_index: u32) -> Self {
        Self {
            landing_pad_offset,
            catch_type_indices: vec![type_index],
            filter_type_indices: Vec::new(),
            is_cleanup: false,
        }
    }

    /// Create a cleanup-only landing pad (destructors/drops, no catch).
    pub fn cleanup(landing_pad_offset: u32) -> Self {
        Self {
            landing_pad_offset,
            catch_type_indices: Vec::new(),
            filter_type_indices: Vec::new(),
            is_cleanup: true,
        }
    }

    /// Returns true if this landing pad catches any exception type
    /// (either specific types or catch-all).
    pub fn has_catch(&self) -> bool {
        !self.catch_type_indices.is_empty()
    }

    /// Returns true if this landing pad has exception specification filters.
    pub fn has_filter(&self) -> bool {
        !self.filter_type_indices.is_empty()
    }

    /// Returns true if this is a catch-all landing pad (catches everything).
    pub fn is_catch_all(&self) -> bool {
        self.catch_type_indices.contains(&0)
    }
}

// ---------------------------------------------------------------------------
// LSDA — Language-Specific Data Area
// ---------------------------------------------------------------------------

/// A call site table entry in the LSDA.
///
/// Each entry maps a range of PC values to a landing pad and an action
/// chain. During unwinding, the personality function searches the call
/// site table for the current PC to determine the correct handler.
///
/// Reference: Itanium C++ ABI sec. 2.5.4 "Call Site Table"
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CallSiteEntry {
    /// Start of the call site range, as an offset from the function start.
    pub start_offset: u32,
    /// Length of the call site range in bytes.
    pub length: u32,
    /// Landing pad offset from the function start, or 0 if no landing pad.
    /// A value of 0 means this call site has no handler (exception
    /// propagates to the caller).
    pub landing_pad_offset: u32,
    /// Index into the action table (1-based), or 0 if no action.
    /// The action table entry determines which types are caught or
    /// filtered by the associated landing pad.
    pub action_index: u32,
}

impl CallSiteEntry {
    /// Returns true if this call site has a landing pad.
    pub fn has_landing_pad(&self) -> bool {
        self.landing_pad_offset != 0
    }

    /// Returns true if this call site has an associated action chain.
    pub fn has_action(&self) -> bool {
        self.action_index != 0
    }
}

/// An action table entry in the LSDA.
///
/// Action entries form chains (linked lists via `next_offset`). Each entry
/// specifies a type filter: positive values index into the type table for
/// catch handlers; negative values index into the type table for exception
/// specification filters; zero means cleanup.
///
/// Reference: Itanium C++ ABI sec. 2.5.5 "Action Table"
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ActionEntry {
    /// Type filter value:
    /// - Positive: 1-based index into the type table (catch handler).
    ///   Matches if the thrown type matches type_table[filter - 1].
    /// - Zero: cleanup action (no type matching, just run cleanup code).
    /// - Negative: exception specification filter. The absolute value is
    ///   a 1-based index into a filter list in the type table.
    pub type_filter: i32,
    /// Offset to the next action entry in the chain (in bytes from
    /// the start of this entry), or 0 if this is the last action.
    /// This enables multiple catch clauses per landing pad.
    pub next_offset: i32,
}

impl ActionEntry {
    /// Returns true if this is a cleanup action (type_filter == 0).
    pub fn is_cleanup(&self) -> bool {
        self.type_filter == 0
    }

    /// Returns true if this is a catch handler (type_filter > 0).
    pub fn is_catch(&self) -> bool {
        self.type_filter > 0
    }

    /// Returns true if this is an exception specification filter (type_filter < 0).
    pub fn is_filter(&self) -> bool {
        self.type_filter < 0
    }

    /// Returns true if there is a next action in the chain.
    pub fn has_next(&self) -> bool {
        self.next_offset != 0
    }
}

/// A type table reference in the LSDA.
///
/// The type table contains pointers to type descriptor objects (e.g.,
/// `std::type_info*` for C++). These are used by the personality function
/// to determine if a thrown exception matches a catch clause's type.
///
/// Entries are indexed from the END of the type table (index 1 is the
/// last entry, index 2 is second-to-last, etc.). This reverse indexing
/// is mandated by the Itanium ABI.
///
/// Reference: Itanium C++ ABI sec. 2.5.6 "Type Table"
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TypeTableEntry {
    /// Symbol name of the type descriptor (e.g., "_ZTIi" for `int`,
    /// "_ZTISt9exception" for `std::exception`).
    /// Empty string represents a catch-all (null type pointer).
    pub type_info_symbol: String,
    /// Encoding format for the type pointer in the LSDA.
    pub encoding: TypeTableEncoding,
}

/// Encoding format for type table pointers in the LSDA.
///
/// Determines how the personality function reads type descriptor pointers
/// from the type table section of the LSDA.
///
/// Reference: DWARF Exception Header Encoding (DW_EH_PE_*)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TypeTableEncoding {
    /// Absolute pointer (DW_EH_PE_absptr). Full pointer-width value.
    AbsPtr,
    /// PC-relative signed 4-byte offset (DW_EH_PE_pcrel | DW_EH_PE_sdata4).
    /// Standard on Apple AArch64.
    PcRelSData4,
    /// Indirect pointer: the value at the encoded address is itself a pointer
    /// to the type descriptor (DW_EH_PE_indirect | DW_EH_PE_pcrel | DW_EH_PE_sdata4).
    /// Used when type descriptors are in a different DSO.
    IndirectPcRel,
    /// Omitted (DW_EH_PE_omit). No type table present.
    Omit,
}

impl TypeTableEncoding {
    /// Returns the DWARF exception header encoding byte (DW_EH_PE_*).
    pub fn dwarf_encoding(self) -> u8 {
        match self {
            Self::AbsPtr => 0x00,         // DW_EH_PE_absptr
            Self::PcRelSData4 => 0x1B,    // DW_EH_PE_pcrel | DW_EH_PE_sdata4
            Self::IndirectPcRel => 0x9B,  // DW_EH_PE_indirect | DW_EH_PE_pcrel | DW_EH_PE_sdata4
            Self::Omit => 0xFF,           // DW_EH_PE_omit
        }
    }
}

// ---------------------------------------------------------------------------
// ExceptionHandlingInfo — per-function exception handling metadata
// ---------------------------------------------------------------------------

/// Complete exception handling information for a single function.
///
/// Collects all the EH metadata needed to generate the LSDA, personality
/// function reference, and compact unwind / DWARF CFI augmentation data.
///
/// This is the ABI-level representation. The codegen layer uses this to
/// emit the `__gcc_except_tab` section (LSDA) and to set the personality
/// and LSDA pointers in compact unwind / DWARF FDE entries.
///
/// # Usage
///
/// ```ignore
/// let mut eh = ExceptionHandlingInfo::new(PersonalityFunction::GxxPersonalityV0);
///
/// // Add a landing pad for a try/catch block
/// eh.add_landing_pad(LandingPad::catch_typed(0x40, 1));
///
/// // Add call site entries covering the try region
/// eh.add_call_site(CallSiteEntry {
///     start_offset: 0x10,
///     length: 0x20,
///     landing_pad_offset: 0x40,
///     action_index: 1,
/// });
///
/// // Add action and type table entries
/// eh.add_action(ActionEntry { type_filter: 1, next_offset: 0 });
/// eh.add_type(TypeTableEntry {
///     type_info_symbol: "_ZTIi".to_string(),
///     encoding: TypeTableEncoding::PcRelSData4,
/// });
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExceptionHandlingInfo {
    /// The personality function for this function.
    pub personality: PersonalityFunction,

    /// Landing pads (exception entry points) in this function.
    pub landing_pads: Vec<LandingPad>,

    /// Call site table entries (PC ranges mapped to landing pads).
    pub call_sites: Vec<CallSiteEntry>,

    /// Action table entries (type filter chains for each call site).
    pub actions: Vec<ActionEntry>,

    /// Type table entries (type descriptors for catch/filter matching).
    pub type_table: Vec<TypeTableEntry>,

    /// Encoding used for type table pointers. Default: PcRelSData4 on Apple.
    pub type_table_encoding: TypeTableEncoding,
}

impl ExceptionHandlingInfo {
    /// Create exception handling info with the given personality function.
    pub fn new(personality: PersonalityFunction) -> Self {
        Self {
            personality,
            landing_pads: Vec::new(),
            call_sites: Vec::new(),
            actions: Vec::new(),
            type_table: Vec::new(),
            type_table_encoding: TypeTableEncoding::PcRelSData4,
        }
    }

    /// Create exception handling info for C++ code.
    pub fn for_cpp() -> Self {
        Self::new(PersonalityFunction::GxxPersonalityV0)
    }

    /// Create exception handling info for Rust code.
    pub fn for_rust() -> Self {
        Self::new(PersonalityFunction::RustEhPersonality)
    }

    /// Add a landing pad.
    pub fn add_landing_pad(&mut self, lp: LandingPad) {
        self.landing_pads.push(lp);
    }

    /// Add a call site table entry.
    pub fn add_call_site(&mut self, cs: CallSiteEntry) {
        self.call_sites.push(cs);
    }

    /// Add an action table entry.
    pub fn add_action(&mut self, action: ActionEntry) {
        self.actions.push(action);
    }

    /// Add a type table entry.
    pub fn add_type(&mut self, ty: TypeTableEntry) {
        self.type_table.push(ty);
    }

    /// Returns true if there are any landing pads.
    pub fn has_landing_pads(&self) -> bool {
        !self.landing_pads.is_empty()
    }

    /// Returns true if any landing pad catches exceptions (not cleanup-only).
    pub fn has_catch_handlers(&self) -> bool {
        self.landing_pads.iter().any(|lp| lp.has_catch())
    }

    /// Returns true if any landing pad is cleanup-only.
    pub fn has_cleanup_handlers(&self) -> bool {
        self.landing_pads.iter().any(|lp| lp.is_cleanup)
    }

    /// Returns the total number of entries in the call site table.
    pub fn call_site_count(&self) -> usize {
        self.call_sites.len()
    }

    /// Returns the total number of entries in the action table.
    pub fn action_count(&self) -> usize {
        self.actions.len()
    }

    /// Returns the total number of entries in the type table.
    pub fn type_count(&self) -> usize {
        self.type_table.len()
    }

    /// Returns the personality function symbol name.
    pub fn personality_symbol(&self) -> &str {
        self.personality.symbol_name()
    }

    /// Validate the internal consistency of the exception handling info.
    ///
    /// Checks:
    /// - All call site landing pad offsets correspond to a registered landing pad
    /// - All action indices are within range
    /// - All type filter indices in actions are within the type table range
    ///
    /// Returns a list of validation errors (empty if valid).
    pub fn validate(&self) -> Vec<String> {
        let mut errors = Vec::new();

        // Check that call site landing pads correspond to registered landing pads.
        let lp_offsets: Vec<u32> = self.landing_pads.iter()
            .map(|lp| lp.landing_pad_offset)
            .collect();

        for (i, cs) in self.call_sites.iter().enumerate() {
            if cs.has_landing_pad() && !lp_offsets.contains(&cs.landing_pad_offset) {
                errors.push(format!(
                    "call site {} references landing pad at offset 0x{:x} which is not registered",
                    i, cs.landing_pad_offset
                ));
            }
            if cs.has_action() && cs.action_index as usize > self.actions.len() {
                errors.push(format!(
                    "call site {} has action_index {} but only {} actions exist",
                    i, cs.action_index, self.actions.len()
                ));
            }
        }

        // Check type filter indices in actions.
        for (i, action) in self.actions.iter().enumerate() {
            if action.is_catch() && action.type_filter as usize > self.type_table.len() {
                errors.push(format!(
                    "action {} has type_filter {} but only {} types in type table",
                    i, action.type_filter, self.type_table.len()
                ));
            }
        }

        errors
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn classify_simple_int_params() {
        // fn foo(i32, i32) -> first two in X0, X1
        let locs = AppleAArch64ABI::classify_params(&[Type::I32, Type::I32]);
        assert_eq!(locs.len(), 2);
        assert_eq!(locs[0], ArgLocation::Reg(gpr::X0));
        assert_eq!(locs[1], ArgLocation::Reg(gpr::X1));
    }

    #[test]
    fn classify_mixed_int_float_params() {
        // fn bar(i64, f64, i32, f32) -> X0, V0, X1, V1
        let locs =
            AppleAArch64ABI::classify_params(&[Type::I64, Type::F64, Type::I32, Type::F32]);
        assert_eq!(locs.len(), 4);
        assert_eq!(locs[0], ArgLocation::Reg(gpr::X0));
        assert_eq!(locs[1], ArgLocation::Reg(gpr::V0));
        assert_eq!(locs[2], ArgLocation::Reg(gpr::X1));
        assert_eq!(locs[3], ArgLocation::Reg(gpr::V1));
    }

    #[test]
    fn classify_overflow_to_stack() {
        // 9 integer params -> X0-X7 in regs, 9th on stack
        let params = vec![Type::I64; 9];
        let locs = AppleAArch64ABI::classify_params(&params);
        assert_eq!(locs.len(), 9);
        for i in 0..8 {
            assert_eq!(locs[i], ArgLocation::Reg(GPR_ARG_REGS[i]));
        }
        assert!(matches!(locs[8], ArgLocation::Stack { offset: 0, size: 8 }));
    }

    #[test]
    fn classify_simple_return() {
        let locs = AppleAArch64ABI::classify_returns(&[Type::I64]);
        assert_eq!(locs.len(), 1);
        assert_eq!(locs[0], ArgLocation::Reg(gpr::X0));
    }

    #[test]
    fn classify_float_return() {
        let locs = AppleAArch64ABI::classify_returns(&[Type::F64]);
        assert_eq!(locs.len(), 1);
        assert_eq!(locs[0], ArgLocation::Reg(gpr::V0));
    }

    #[test]
    fn callee_saved_regs_correct() {
        let gprs = AppleAArch64ABI::callee_saved_gprs();
        assert_eq!(gprs.len(), 12); // X19-X28 + FP + LR
        assert_eq!(gprs[0], gpr::X19);
        assert_eq!(gprs[11], gpr::LR);

        let fprs = AppleAArch64ABI::callee_saved_fprs();
        assert_eq!(fprs.len(), 8); // V8-V15
        assert_eq!(fprs[0], gpr::V8);
    }

    #[test]
    fn call_clobber_includes_x18() {
        let clobber = AppleAArch64ABI::call_clobber_gprs();
        assert!(clobber.contains(&gpr::X18));
        assert!(!clobber.contains(&gpr::X19)); // callee-saved, not clobbered
    }

    #[test]
    fn stack_args_size_no_overflow() {
        // 4 integer args -> all in registers, 0 stack
        let params: Vec<Type> = (0..4).map(|_| Type::I32).collect();
        assert_eq!(AppleAArch64ABI::stack_args_size(&params), 0);
    }

    #[test]
    fn stack_args_size_with_overflow() {
        // 10 integer args -> 2 overflow on stack (8 bytes each)
        let params: Vec<Type> = (0..10).map(|_| Type::I64).collect();
        let size = AppleAArch64ABI::stack_args_size(&params);
        assert_eq!(size, 16); // two 8-byte slots
    }

    // ===================================================================
    // Variadic call ABI tests (Apple AArch64)
    // ===================================================================

    #[test]
    fn variadic_fixed_args_in_registers() {
        // printf(const char* fmt, ...) called with printf(fmt, 42, 3.14)
        // fixed_count=1: fmt -> X0 (register)
        // variadic: 42(I32) -> stack[0], 3.14(F64) -> stack[8]
        let params = vec![Type::I64, Type::I32, Type::F64];
        let locs = AppleAArch64ABI::classify_params_variadic(1, &params);

        assert_eq!(locs.len(), 3);
        // Fixed arg: fmt in X0
        assert_eq!(locs[0], ArgLocation::Reg(gpr::X0));
        // Variadic arg 1: i32 on stack at offset 0, stored as 8 bytes
        assert!(matches!(locs[1], ArgLocation::Stack { offset: 0, size: 8 }));
        // Variadic arg 2: f64 on stack at offset 8
        assert!(matches!(locs[2], ArgLocation::Stack { offset: 8, size: 8 }));
    }

    #[test]
    fn variadic_two_fixed_two_varargs() {
        // fn(i64, i64, ...) called with (a, b, c, d) where c,d are variadic
        let params = vec![Type::I64, Type::I64, Type::I64, Type::I64];
        let locs = AppleAArch64ABI::classify_params_variadic(2, &params);

        assert_eq!(locs.len(), 4);
        // Fixed args in registers
        assert_eq!(locs[0], ArgLocation::Reg(gpr::X0));
        assert_eq!(locs[1], ArgLocation::Reg(gpr::X1));
        // Variadic args on stack
        assert!(matches!(locs[2], ArgLocation::Stack { offset: 0, size: 8 }));
        assert!(matches!(locs[3], ArgLocation::Stack { offset: 8, size: 8 }));
    }

    #[test]
    fn variadic_no_varargs_is_same_as_normal() {
        // Variadic function called with only fixed args (no variadic args)
        let params = vec![Type::I32, Type::I64];
        let normal_locs = AppleAArch64ABI::classify_params(&params);
        let variadic_locs = AppleAArch64ABI::classify_params_variadic(2, &params);

        assert_eq!(normal_locs, variadic_locs);
    }

    #[test]
    fn variadic_fixed_overflow_then_varargs() {
        // 9 fixed integer args (8 in regs, 1 on stack), then 1 variadic
        let mut params: Vec<Type> = vec![Type::I64; 9];
        params.push(Type::I64); // variadic
        let locs = AppleAArch64ABI::classify_params_variadic(9, &params);

        assert_eq!(locs.len(), 10);
        // X0-X7 in registers
        for i in 0..8 {
            assert_eq!(locs[i], ArgLocation::Reg(GPR_ARG_REGS[i]));
        }
        // 9th fixed arg on stack
        assert!(matches!(locs[8], ArgLocation::Stack { offset: 0, size: 8 }));
        // variadic arg on stack after fixed overflow
        assert!(matches!(locs[9], ArgLocation::Stack { offset: 8, size: 8 }));
    }

    #[test]
    fn variadic_float_args_on_stack() {
        // Apple ABI: variadic floats go on stack, NOT in FPR registers.
        // fn(i64, ...) called with (ptr, 1.0f32, 2.0f64)
        let params = vec![Type::I64, Type::F32, Type::F64];
        let locs = AppleAArch64ABI::classify_params_variadic(1, &params);

        assert_eq!(locs.len(), 3);
        assert_eq!(locs[0], ArgLocation::Reg(gpr::X0));
        // F32 variadic -> stack (8-byte slot)
        assert!(matches!(locs[1], ArgLocation::Stack { offset: 0, size: 8 }));
        // F64 variadic -> stack
        assert!(matches!(locs[2], ArgLocation::Stack { offset: 8, size: 8 }));
    }

    #[test]
    fn variadic_stack_args_size() {
        // printf(fmt, 42, 3.14) -> fmt in X0, two varargs on stack
        let params = vec![Type::I64, Type::I32, Type::F64];
        let size = AppleAArch64ABI::stack_args_size_variadic(1, &params);
        assert_eq!(size, 16); // two 8-byte slots
    }

    #[test]
    fn variadic_zero_fixed_all_on_stack() {
        // Degenerate case: 0 fixed params, all args are variadic
        let params = vec![Type::I64, Type::I32];
        let locs = AppleAArch64ABI::classify_params_variadic(0, &params);

        assert_eq!(locs.len(), 2);
        // Both on stack
        assert!(matches!(locs[0], ArgLocation::Stack { offset: 0, size: 8 }));
        assert!(matches!(locs[1], ArgLocation::Stack { offset: 8, size: 8 }));
    }

    // ===================================================================
    // HFA (Homogeneous Floating-point Aggregate) tests
    // ===================================================================

    #[test]
    fn detect_hfa_two_f32_fields() {
        // struct Complex32 { float re, im; } -> HFA: 2x F32
        let ty = Type::Struct(vec![Type::F32, Type::F32]);
        let hfa = AppleAArch64ABI::detect_hfa(&ty);
        assert_eq!(hfa, Some((Type::F32, 2)));
    }

    #[test]
    fn detect_hfa_four_f32_fields() {
        // struct Vec4 { float x, y, z, w; } -> HFA: 4x F32
        let ty = Type::Struct(vec![Type::F32, Type::F32, Type::F32, Type::F32]);
        let hfa = AppleAArch64ABI::detect_hfa(&ty);
        assert_eq!(hfa, Some((Type::F32, 4)));
    }

    #[test]
    fn detect_hfa_two_f64_fields() {
        // struct Complex64 { double re, im; } -> HFA: 2x F64
        let ty = Type::Struct(vec![Type::F64, Type::F64]);
        let hfa = AppleAArch64ABI::detect_hfa(&ty);
        assert_eq!(hfa, Some((Type::F64, 2)));
    }

    #[test]
    fn detect_hfa_mixed_types_not_hfa() {
        // struct { float x; int y; } -> NOT HFA (mixed types)
        let ty = Type::Struct(vec![Type::F32, Type::I32]);
        let hfa = AppleAArch64ABI::detect_hfa(&ty);
        assert_eq!(hfa, None);
    }

    #[test]
    fn detect_hfa_five_fields_not_hfa() {
        // struct { float a, b, c, d, e; } -> NOT HFA (> 4 fields)
        let ty = Type::Struct(vec![Type::F32, Type::F32, Type::F32, Type::F32, Type::F32]);
        let hfa = AppleAArch64ABI::detect_hfa(&ty);
        assert_eq!(hfa, None);
    }

    #[test]
    fn detect_hfa_array_of_f32() {
        // float[3] -> HFA: 3x F32
        let ty = Type::Array(Box::new(Type::F32), 3);
        let hfa = AppleAArch64ABI::detect_hfa(&ty);
        assert_eq!(hfa, Some((Type::F32, 3)));
    }

    #[test]
    fn detect_hfa_array_of_f64_too_large() {
        // double[5] -> NOT HFA (> 4 elements)
        let ty = Type::Array(Box::new(Type::F64), 5);
        let hfa = AppleAArch64ABI::detect_hfa(&ty);
        assert_eq!(hfa, None);
    }

    #[test]
    fn detect_hfa_empty_struct_not_hfa() {
        let ty = Type::Struct(vec![]);
        let hfa = AppleAArch64ABI::detect_hfa(&ty);
        assert_eq!(hfa, None);
    }

    #[test]
    fn detect_hfa_integer_struct_not_hfa() {
        let ty = Type::Struct(vec![Type::I32, Type::I32]);
        let hfa = AppleAArch64ABI::detect_hfa(&ty);
        assert_eq!(hfa, None);
    }

    #[test]
    fn classify_hfa_param_two_f32_in_v_regs() {
        // struct { float, float } passed as HFA -> S0, S1 (typed FPR registers)
        let hfa = Type::Struct(vec![Type::F32, Type::F32]);
        let locs = AppleAArch64ABI::classify_params(&[hfa]);
        assert_eq!(locs.len(), 1);
        // F32 HFA uses S registers (Fpr32 class), not V registers.
        assert_eq!(locs[0], ArgLocation::RegSequence(vec![gpr::S0, gpr::S1]));
    }

    #[test]
    fn classify_hfa_param_four_f32_in_v_regs() {
        // struct { float, float, float, float } -> S0-S3 (typed FPR registers)
        let hfa = Type::Struct(vec![Type::F32, Type::F32, Type::F32, Type::F32]);
        let locs = AppleAArch64ABI::classify_params(&[hfa]);
        assert_eq!(locs.len(), 1);
        assert_eq!(locs[0], ArgLocation::RegSequence(vec![gpr::S0, gpr::S1, gpr::S2, gpr::S3]));
    }

    #[test]
    fn classify_hfa_param_with_preceding_floats() {
        // fn(f64, HFA{f32, f32}) -> V0 for f64, then HFA in S1+S2
        // The f64 scalar consumes FPR slot 0, HFA starts at slot 1.
        let hfa = Type::Struct(vec![Type::F32, Type::F32]);
        let locs = AppleAArch64ABI::classify_params(&[Type::F64, hfa]);
        assert_eq!(locs.len(), 2);
        assert_eq!(locs[0], ArgLocation::Reg(gpr::V0));
        assert_eq!(locs[1], ArgLocation::RegSequence(vec![gpr::S1, gpr::S2]));
    }

    #[test]
    fn classify_hfa_param_overflow_to_stack() {
        // 7 floats used up V0-V6, then HFA(2x F32) can't fit -> stack
        let mut params: Vec<Type> = vec![Type::F32; 7];
        params.push(Type::Struct(vec![Type::F32, Type::F32])); // needs 2 FPR slots
        let locs = AppleAArch64ABI::classify_params(&params);
        assert_eq!(locs.len(), 8);
        // V0-V6 for the 7 scalars
        for i in 0..7 {
            assert_eq!(locs[i], ArgLocation::Reg(FPR_ARG_REGS[i]));
        }
        // HFA needs 2 slots but only 1 left -> stack
        assert!(matches!(locs[7], ArgLocation::Stack { .. }));
    }

    #[test]
    fn classify_non_hfa_struct_in_gpr() {
        // struct { i32, f32 } -> NOT HFA (mixed types), small (8 bytes) -> X0
        let non_hfa = Type::Struct(vec![Type::I32, Type::F32]);
        let locs = AppleAArch64ABI::classify_params(&[non_hfa]);
        assert_eq!(locs.len(), 1);
        assert_eq!(locs[0], ArgLocation::Reg(gpr::X0));
    }

    // ===================================================================
    // Small struct register return tests
    // ===================================================================

    #[test]
    fn return_small_struct_8_bytes_in_x0() {
        // struct { i32, i32 } = 8 bytes -> return in X0
        let ty = Type::Struct(vec![Type::I32, Type::I32]);
        let locs = AppleAArch64ABI::classify_returns(&[ty]);
        assert_eq!(locs.len(), 1);
        assert_eq!(locs[0], ArgLocation::Reg(gpr::X0));
    }

    #[test]
    fn return_small_struct_16_bytes_in_x0_x1() {
        // struct { i64, i64 } = 16 bytes -> return in X0+X1 (Indirect{X0} encoding)
        let ty = Type::Struct(vec![Type::I64, Type::I64]);
        let locs = AppleAArch64ABI::classify_returns(&[ty]);
        assert_eq!(locs.len(), 1);
        // 16-byte struct uses the Indirect{X0} encoding (two GPRs, first recorded).
        assert_eq!(locs[0], ArgLocation::Indirect { ptr_reg: gpr::X0 });
    }

    #[test]
    fn return_large_struct_via_sret() {
        // struct { i64, i64, i64 } = 24 bytes -> indirect via X8
        let ty = Type::Struct(vec![Type::I64, Type::I64, Type::I64]);
        let locs = AppleAArch64ABI::classify_returns(&[ty]);
        assert_eq!(locs.len(), 1);
        assert_eq!(locs[0], ArgLocation::Indirect { ptr_reg: gpr::X8 });
    }

    #[test]
    fn return_hfa_two_f32_in_v0() {
        // struct { float, float } -> HFA return in S0, S1 (typed FPR registers)
        let hfa = Type::Struct(vec![Type::F32, Type::F32]);
        let locs = AppleAArch64ABI::classify_returns(&[hfa]);
        assert_eq!(locs.len(), 1);
        assert_eq!(locs[0], ArgLocation::RegSequence(vec![gpr::S0, gpr::S1]));
    }

    #[test]
    fn return_hfa_four_f64_in_v0() {
        // struct { double, double, double, double } -> HFA return in D0-D3
        let hfa = Type::Struct(vec![Type::F64, Type::F64, Type::F64, Type::F64]);
        let locs = AppleAArch64ABI::classify_returns(&[hfa]);
        assert_eq!(locs.len(), 1);
        assert_eq!(locs[0], ArgLocation::RegSequence(vec![gpr::D0, gpr::D1, gpr::D2, gpr::D3]));
    }

    #[test]
    fn return_hfa_array_three_f32() {
        // float[3] -> HFA return in S0-S2
        let hfa = Type::Array(Box::new(Type::F32), 3);
        let locs = AppleAArch64ABI::classify_returns(&[hfa]);
        assert_eq!(locs.len(), 1);
        assert_eq!(locs[0], ArgLocation::RegSequence(vec![gpr::S0, gpr::S1, gpr::S2]));
    }

    #[test]
    fn return_single_field_struct_in_x0() {
        // struct { i64 } = 8 bytes -> return in X0 (not HFA, <= 8 bytes)
        let ty = Type::Struct(vec![Type::I64]);
        let locs = AppleAArch64ABI::classify_returns(&[ty]);
        assert_eq!(locs.len(), 1);
        assert_eq!(locs[0], ArgLocation::Reg(gpr::X0));
    }

    // ===================================================================
    // Coverage expansion: 7-arg boundary (all GPR slots filled)
    // ===================================================================

    #[test]
    fn classify_exactly_7_int_args_all_in_regs() {
        // 7 integer args -> X0-X6, all in registers
        let params: Vec<Type> = (0..7).map(|_| Type::I64).collect();
        let locs = AppleAArch64ABI::classify_params(&params);
        assert_eq!(locs.len(), 7);
        for i in 0..7 {
            assert_eq!(locs[i], ArgLocation::Reg(GPR_ARG_REGS[i]));
        }
    }

    #[test]
    fn classify_exactly_8_int_args_all_in_regs() {
        // 8 integer args -> X0-X7, all in registers (uses all GPR slots)
        let params: Vec<Type> = (0..8).map(|_| Type::I64).collect();
        let locs = AppleAArch64ABI::classify_params(&params);
        assert_eq!(locs.len(), 8);
        for i in 0..8 {
            assert_eq!(locs[i], ArgLocation::Reg(GPR_ARG_REGS[i]));
        }
    }

    #[test]
    fn classify_8th_arg_on_stack_boundary() {
        // 9 args: first 8 in X0-X7, 9th on stack at offset 0
        let params: Vec<Type> = (0..9).map(|_| Type::I32).collect();
        let locs = AppleAArch64ABI::classify_params(&params);
        assert_eq!(locs.len(), 9);
        for i in 0..8 {
            assert_eq!(locs[i], ArgLocation::Reg(GPR_ARG_REGS[i]));
        }
        // 9th arg: on stack, I32 promoted to 8 bytes
        assert!(matches!(locs[8], ArgLocation::Stack { offset: 0, size: 8 }));
    }

    // ===================================================================
    // Coverage expansion: return classification for all scalar types
    // ===================================================================

    #[test]
    fn classify_return_b1() {
        let locs = AppleAArch64ABI::classify_returns(&[Type::B1]);
        assert_eq!(locs.len(), 1);
        assert_eq!(locs[0], ArgLocation::Reg(gpr::X0));
    }

    #[test]
    fn classify_return_i8() {
        let locs = AppleAArch64ABI::classify_returns(&[Type::I8]);
        assert_eq!(locs.len(), 1);
        assert_eq!(locs[0], ArgLocation::Reg(gpr::X0));
    }

    #[test]
    fn classify_return_i16() {
        let locs = AppleAArch64ABI::classify_returns(&[Type::I16]);
        assert_eq!(locs.len(), 1);
        assert_eq!(locs[0], ArgLocation::Reg(gpr::X0));
    }

    #[test]
    fn classify_return_i32() {
        let locs = AppleAArch64ABI::classify_returns(&[Type::I32]);
        assert_eq!(locs.len(), 1);
        assert_eq!(locs[0], ArgLocation::Reg(gpr::X0));
    }

    #[test]
    fn classify_return_f32() {
        let locs = AppleAArch64ABI::classify_returns(&[Type::F32]);
        assert_eq!(locs.len(), 1);
        assert_eq!(locs[0], ArgLocation::Reg(gpr::V0));
    }

    #[test]
    fn classify_return_i128() {
        // I128 returns use two GPRs: Indirect{X0} encoding
        let locs = AppleAArch64ABI::classify_returns(&[Type::I128]);
        assert_eq!(locs.len(), 1);
        assert_eq!(locs[0], ArgLocation::Indirect { ptr_reg: gpr::X0 });
    }

    // ===================================================================
    // Coverage expansion: I128 parameter classification
    // ===================================================================

    #[test]
    fn classify_i128_param_in_register_pair() {
        // I128 uses two consecutive GPR slots
        let locs = AppleAArch64ABI::classify_params(&[Type::I128]);
        assert_eq!(locs.len(), 1);
        assert_eq!(locs[0], ArgLocation::Indirect { ptr_reg: gpr::X0 });
    }

    #[test]
    fn classify_i128_param_after_6_gpr_args_goes_to_stack() {
        // 6 i64 args use X0-X5, then I128 needs 2 consecutive but only X6-X7 left.
        // X6+X7 is 2 consecutive GPRs, so it should fit.
        let mut params: Vec<Type> = vec![Type::I64; 6];
        params.push(Type::I128);
        let locs = AppleAArch64ABI::classify_params(&params);
        assert_eq!(locs.len(), 7);
        // I128 at index 6: needs 2 GPRs starting at X6 (X6+X7)
        assert_eq!(locs[6], ArgLocation::Indirect { ptr_reg: gpr::X6 });
    }

    #[test]
    fn classify_i128_param_after_7_gpr_args_goes_to_stack() {
        // 7 i64 args use X0-X6, then I128 needs 2 consecutive but only X7 left.
        // Should overflow to stack.
        let mut params: Vec<Type> = vec![Type::I64; 7];
        params.push(Type::I128);
        let locs = AppleAArch64ABI::classify_params(&params);
        assert_eq!(locs.len(), 8);
        // I128 at index 7: not enough consecutive GPRs -> stack
        assert!(matches!(locs[7], ArgLocation::Stack { size: 16, .. }));
    }

    // ===================================================================
    // Coverage expansion: non-HFA aggregate parameter classification
    // ===================================================================

    #[test]
    fn classify_small_non_hfa_struct_in_gpr() {
        // struct { i32, i16 } = 6 bytes (with padding -> 8) -> fits in 1 GPR
        let small = Type::Struct(vec![Type::I32, Type::I16]);
        assert!(small.bytes() <= 8);
        let locs = AppleAArch64ABI::classify_params(&[small]);
        assert_eq!(locs.len(), 1);
        assert_eq!(locs[0], ArgLocation::Reg(gpr::X0));
    }

    #[test]
    fn classify_medium_non_hfa_struct_in_two_gprs() {
        // struct { i64, i32 } = 16 bytes -> 2 GPRs (Indirect encoding)
        let medium = Type::Struct(vec![Type::I64, Type::I32]);
        assert!(medium.bytes() > 8 && medium.bytes() <= 16);
        let locs = AppleAArch64ABI::classify_params(&[medium]);
        assert_eq!(locs.len(), 1);
        assert_eq!(locs[0], ArgLocation::Indirect { ptr_reg: gpr::X0 });
    }

    #[test]
    fn classify_large_non_hfa_struct_indirect() {
        // struct { i64, i64, i64 } = 24 bytes > 16 -> indirect via pointer
        let large = Type::Struct(vec![Type::I64, Type::I64, Type::I64]);
        assert!(large.bytes() > 16);
        let locs = AppleAArch64ABI::classify_params(&[large]);
        assert_eq!(locs.len(), 1);
        assert_eq!(locs[0], ArgLocation::Indirect { ptr_reg: gpr::X0 });
    }

    // ===================================================================
    // Coverage expansion: nested HFA detection
    // ===================================================================

    #[test]
    fn detect_hfa_nested_struct_two_f32() {
        // struct { struct { f32 }, struct { f32 } } -> 2x F32 HFA
        let inner = Type::Struct(vec![Type::F32]);
        let outer = Type::Struct(vec![inner.clone(), inner]);
        let hfa = AppleAArch64ABI::detect_hfa(&outer);
        assert_eq!(hfa, Some((Type::F32, 2)));
    }

    #[test]
    fn detect_hfa_array_of_struct_f64() {
        // [struct { f64 }; 2] -> 2x F64 HFA
        let inner = Type::Struct(vec![Type::F64]);
        let arr = Type::Array(Box::new(inner), 2);
        let hfa = AppleAArch64ABI::detect_hfa(&arr);
        assert_eq!(hfa, Some((Type::F64, 2)));
    }

    #[test]
    fn detect_hfa_nested_too_many_flattened_fields() {
        // struct { struct { f32, f32 }, struct { f32, f32, f32 } } -> 5 flattened fields > 4
        let two_f32 = Type::Struct(vec![Type::F32, Type::F32]);
        let three_f32 = Type::Struct(vec![Type::F32, Type::F32, Type::F32]);
        let outer = Type::Struct(vec![two_f32, three_f32]);
        let hfa = AppleAArch64ABI::detect_hfa(&outer);
        assert_eq!(hfa, None, "5 flattened FP fields exceeds HFA limit of 4");
    }

    #[test]
    fn detect_hfa_single_f64_field() {
        // struct { f64 } -> HFA: 1x F64
        let ty = Type::Struct(vec![Type::F64]);
        let hfa = AppleAArch64ABI::detect_hfa(&ty);
        assert_eq!(hfa, Some((Type::F64, 1)));
    }

    #[test]
    fn detect_hfa_zero_element_array_not_hfa() {
        let ty = Type::Array(Box::new(Type::F32), 0);
        let hfa = AppleAArch64ABI::detect_hfa(&ty);
        assert_eq!(hfa, None);
    }

    // ===================================================================
    // Coverage expansion: mixed GPR/FPR argument interleaving
    // ===================================================================

    #[test]
    fn classify_mixed_8_gpr_8_fpr_all_in_regs() {
        // 8 ints + 8 floats: all should fit since GPR and FPR have independent pools
        let mut params = Vec::new();
        for _i in 0..8 {
            params.push(Type::I64);
            params.push(Type::F64);
        }
        let locs = AppleAArch64ABI::classify_params(&params);
        assert_eq!(locs.len(), 16);
        // All should be in registers (8 GPR + 8 FPR)
        for loc in &locs {
            assert!(matches!(loc, ArgLocation::Reg(_)),
                "Expected all 16 args in registers, got {:?}", loc);
        }
    }

    #[test]
    fn classify_float_overflow_9_fpr_args() {
        // 9 float args: first 8 in V0-V7, 9th on stack
        let params: Vec<Type> = (0..9).map(|_| Type::F64).collect();
        let locs = AppleAArch64ABI::classify_params(&params);
        assert_eq!(locs.len(), 9);
        for i in 0..8 {
            assert_eq!(locs[i], ArgLocation::Reg(FPR_ARG_REGS[i]));
        }
        assert!(matches!(locs[8], ArgLocation::Stack { offset: 0, size: 8 }));
    }

    // ===================================================================
    // Coverage expansion: align_up helper
    // ===================================================================

    #[test]
    fn align_up_values() {
        assert_eq!(align_up(0, 8), 0);
        assert_eq!(align_up(1, 8), 8);
        assert_eq!(align_up(7, 8), 8);
        assert_eq!(align_up(8, 8), 8);
        assert_eq!(align_up(9, 8), 16);
        assert_eq!(align_up(16, 16), 16);
        assert_eq!(align_up(1, 16), 16);
    }

    // ===================================================================
    // Coverage expansion: empty params and returns
    // ===================================================================

    #[test]
    fn classify_no_params() {
        let locs = AppleAArch64ABI::classify_params(&[]);
        assert!(locs.is_empty());
    }

    #[test]
    fn classify_no_returns() {
        let locs = AppleAArch64ABI::classify_returns(&[]);
        assert!(locs.is_empty());
    }

    #[test]
    fn stack_args_size_empty() {
        assert_eq!(AppleAArch64ABI::stack_args_size(&[]), 0);
    }

    // ===================================================================
    // Coverage expansion: multiple stack overflow args sizing
    // ===================================================================

    #[test]
    fn stack_args_size_three_overflow() {
        // 11 integer args -> 3 overflow on stack (8 bytes each)
        let params: Vec<Type> = (0..11).map(|_| Type::I64).collect();
        let size = AppleAArch64ABI::stack_args_size(&params);
        assert_eq!(size, 24); // three 8-byte slots
    }

    #[test]
    fn variadic_stack_args_size_multiple_varargs() {
        // printf(fmt, a, b, c) -> fmt in X0, 3 varargs on stack
        let params = vec![Type::I64, Type::I32, Type::I64, Type::F64];
        let size = AppleAArch64ABI::stack_args_size_variadic(1, &params);
        assert_eq!(size, 24); // three 8-byte slots
    }

    // ===================================================================
    // Coverage expansion: B1 parameter classification
    // ===================================================================

    #[test]
    fn classify_b1_param_in_gpr() {
        let locs = AppleAArch64ABI::classify_params(&[Type::B1]);
        assert_eq!(locs.len(), 1);
        assert_eq!(locs[0], ArgLocation::Reg(gpr::X0));
    }

    #[test]
    fn classify_i8_param_in_gpr() {
        let locs = AppleAArch64ABI::classify_params(&[Type::I8]);
        assert_eq!(locs.len(), 1);
        assert_eq!(locs[0], ArgLocation::Reg(gpr::X0));
    }

    #[test]
    fn classify_i16_param_in_gpr() {
        let locs = AppleAArch64ABI::classify_params(&[Type::I16]);
        assert_eq!(locs.len(), 1);
        assert_eq!(locs[0], ArgLocation::Reg(gpr::X0));
    }

    #[test]
    fn classify_f32_param_in_fpr() {
        let locs = AppleAArch64ABI::classify_params(&[Type::F32]);
        assert_eq!(locs.len(), 1);
        assert_eq!(locs[0], ArgLocation::Reg(gpr::V0));
    }

    // ===================================================================
    // Exception Handling / Stack Unwinding ABI tests
    // ===================================================================

    // --- UnwindInfo construction tests ---

    #[test]
    fn unwind_leaf_frame_has_fp_lr() {
        let info = UnwindInfo::leaf_frame();
        assert_eq!(info.saved_registers.len(), 2);
        assert_eq!(info.saved_registers[0].reg, gpr::FP);
        assert_eq!(info.saved_registers[0].fp_offset, 0);
        assert_eq!(info.saved_registers[1].reg, gpr::LR);
        assert_eq!(info.saved_registers[1].fp_offset, 8);
        assert_eq!(info.frame_size, 16);
        assert!(info.has_frame_pointer);
        assert!(info.is_leaf);
        assert!(!info.has_dynamic_alloc);
        assert!(info.personality.is_none());
        assert!(!info.has_lsda);
    }

    #[test]
    fn unwind_leaf_frame_num_pairs() {
        let info = UnwindInfo::leaf_frame();
        assert_eq!(info.num_saved_pairs(), 1); // FP/LR = 1 pair
    }

    #[test]
    fn unwind_standard_frame_with_gpr_pair() {
        // Frame with FP/LR + X19/X20 pair.
        let info = UnwindInfo::standard_frame(
            &[(gpr::X19, gpr::X20, false)],
            32,
            false,
        );
        assert_eq!(info.saved_registers.len(), 4);
        assert_eq!(info.frame_size, 32);
        assert!(!info.is_leaf);
        assert_eq!(info.num_saved_pairs(), 2); // FP/LR + X19/X20
    }

    #[test]
    fn unwind_standard_frame_register_offsets() {
        // FP/LR + X19/X20: X19 at FP-8, X20 at FP-16.
        let info = UnwindInfo::standard_frame(
            &[(gpr::X19, gpr::X20, false)],
            32,
            true,
        );
        // FP at offset 0
        assert_eq!(info.saved_registers[0].fp_offset, 0);
        // LR at offset 8
        assert_eq!(info.saved_registers[1].fp_offset, 8);
        // X19 at offset -16 + 8 = -8
        assert_eq!(info.saved_registers[2].fp_offset, -8);
        // X20 at offset -16
        assert_eq!(info.saved_registers[3].fp_offset, -16);
    }

    #[test]
    fn unwind_standard_frame_multiple_pairs() {
        // FP/LR + X19/X20 + X21/X22 + V8/V9.
        let info = UnwindInfo::standard_frame(
            &[
                (gpr::X19, gpr::X20, false),
                (gpr::X21, gpr::X22, false),
                (gpr::V8, gpr::V9, true),
            ],
            64,
            false,
        );
        assert_eq!(info.saved_registers.len(), 8); // 2 + 2 + 2 + 2
        assert_eq!(info.num_saved_pairs(), 4);
        assert_eq!(info.frame_size, 64);
    }

    #[test]
    fn unwind_info_saved_gprs_filter() {
        let info = UnwindInfo::standard_frame(
            &[
                (gpr::X19, gpr::X20, false),
                (gpr::V8, gpr::V9, true),
            ],
            48,
            false,
        );
        let gprs = info.saved_gprs();
        let fprs = info.saved_fprs();
        // GPRs: FP, LR, X19, X20 = 4
        assert_eq!(gprs.len(), 4);
        // FPRs: V8, V9 = 2
        assert_eq!(fprs.len(), 2);
    }

    // --- CompactUnwindEntry generation tests ---

    #[test]
    fn compact_unwind_leaf_frame() {
        let info = UnwindInfo::leaf_frame();
        let entry = generate_compact_unwind(&info, 0x1000, 64);

        assert_eq!(entry.encoding, COMPACT_MODE_FRAME);
        assert_eq!(entry.function_start, 0x1000);
        assert_eq!(entry.function_length, 64);
        assert_eq!(entry.personality, 0);
        assert_eq!(entry.lsda, 0);
        assert!(!entry.needs_dwarf_fallback());
    }

    #[test]
    fn compact_unwind_with_x19_x20() {
        let info = UnwindInfo::standard_frame(
            &[(gpr::X19, gpr::X20, false)],
            32,
            false,
        );
        let entry = generate_compact_unwind(&info, 0, 128);

        assert_eq!(entry.encoding, COMPACT_MODE_FRAME | COMPACT_X19_X20);
        assert!(!entry.needs_dwarf_fallback());
    }

    #[test]
    fn compact_unwind_with_all_gpr_pairs() {
        let info = UnwindInfo::standard_frame(
            &[
                (gpr::X19, gpr::X20, false),
                (gpr::X21, gpr::X22, false),
                (gpr::X23, gpr::X24, false),
                (gpr::X25, gpr::X26, false),
                (gpr::X27, gpr::X28, false),
            ],
            96,
            false,
        );
        let entry = generate_compact_unwind(&info, 0, 256);

        let expected = COMPACT_MODE_FRAME
            | COMPACT_X19_X20
            | COMPACT_X21_X22
            | COMPACT_X23_X24
            | COMPACT_X25_X26
            | COMPACT_X27_X28;
        assert_eq!(entry.encoding, expected);
    }

    #[test]
    fn compact_unwind_with_fpr_pairs() {
        let info = UnwindInfo::standard_frame(
            &[
                (gpr::V8, gpr::V9, true),
                (gpr::V10, gpr::V11, true),
            ],
            48,
            false,
        );
        let entry = generate_compact_unwind(&info, 0, 96);

        let expected = COMPACT_MODE_FRAME | COMPACT_D8_D9 | COMPACT_D10_D11;
        assert_eq!(entry.encoding, expected);
    }

    #[test]
    fn compact_unwind_mixed_gpr_fpr() {
        let info = UnwindInfo::standard_frame(
            &[
                (gpr::X19, gpr::X20, false),
                (gpr::V8, gpr::V9, true),
                (gpr::V14, gpr::V15, true),
            ],
            64,
            false,
        );
        let entry = generate_compact_unwind(&info, 0, 128);

        let expected = COMPACT_MODE_FRAME
            | COMPACT_X19_X20
            | COMPACT_D8_D9
            | COMPACT_D14_D15;
        assert_eq!(entry.encoding, expected);
    }

    #[test]
    fn compact_unwind_dynamic_alloc_falls_back_to_dwarf() {
        let mut info = UnwindInfo::leaf_frame();
        info.has_dynamic_alloc = true;
        let entry = generate_compact_unwind(&info, 0, 32);

        assert!(entry.needs_dwarf_fallback());
        assert_eq!(entry.mode(), COMPACT_MODE_DWARF);
    }

    #[test]
    fn compact_unwind_no_frame_pointer_falls_back_to_dwarf() {
        let mut info = UnwindInfo::leaf_frame();
        info.has_frame_pointer = false;
        let entry = generate_compact_unwind(&info, 0, 32);

        assert!(entry.needs_dwarf_fallback());
    }

    // --- DwarfCfiOp generation tests ---

    #[test]
    fn dwarf_cfi_leaf_frame_basic_sequence() {
        let info = UnwindInfo::leaf_frame();
        let ops = generate_dwarf_cfi(&info);

        // Should have: DefCfa(SP, 16), Offset(FP, 2), Offset(LR, 1), DefCfa(FP, 16)
        assert_eq!(ops.len(), 4);

        // First: DefCfa SP, 16
        assert_eq!(ops[0], DwarfCfiOp::DefCfa {
            register: DWARF_REG_SP,
            offset: 16,
        });

        // FP saved at CFA-16 (factored: 2)
        assert_eq!(ops[1], DwarfCfiOp::Offset {
            register: DWARF_REG_FP,
            factored_offset: 2,
        });

        // LR saved at CFA-8 (factored: 1)
        assert_eq!(ops[2], DwarfCfiOp::Offset {
            register: DWARF_REG_LR,
            factored_offset: 1,
        });

        // DefCfa FP, 16
        assert_eq!(ops[3], DwarfCfiOp::DefCfa {
            register: DWARF_REG_FP,
            offset: 16,
        });
    }

    #[test]
    fn dwarf_cfi_with_callee_saves() {
        let info = UnwindInfo::standard_frame(
            &[(gpr::X19, gpr::X20, false)],
            32,
            false,
        );
        let ops = generate_dwarf_cfi(&info);

        // 4 base ops + 2 callee-save ops (X19, X20)
        assert_eq!(ops.len(), 6);

        // First DefCfa should use callee_saved_area = 2 pairs * 16 = 32
        assert_eq!(ops[0], DwarfCfiOp::DefCfa {
            register: DWARF_REG_SP,
            offset: 32,
        });

        // After FP/LR offsets and DefCfa(FP), we get X19 and X20 offsets.
        // X19 at FP-8: CFA offset = 32 - (-8) = 40, factored = 40/8 = 5
        assert_eq!(ops[4], DwarfCfiOp::Offset {
            register: 19,
            factored_offset: 5,
        });

        // X20 at FP-16: CFA offset = 32 - (-16) = 48, factored = 48/8 = 6
        assert_eq!(ops[5], DwarfCfiOp::Offset {
            register: 20,
            factored_offset: 6,
        });
    }

    #[test]
    fn dwarf_cfi_with_fpr_saves() {
        let info = UnwindInfo::standard_frame(
            &[(gpr::V8, gpr::V9, true)],
            32,
            false,
        );
        let ops = generate_dwarf_cfi(&info);

        // 4 base ops + 2 FPR ops
        assert_eq!(ops.len(), 6);

        // V8 (encoding 72, DWARF 72) at FP-8
        assert_eq!(ops[4], DwarfCfiOp::Offset {
            register: 72,
            factored_offset: 5, // (32 - (-8)) / 8 = 5
        });

        // V9 (encoding 73, DWARF 73) at FP-16
        assert_eq!(ops[5], DwarfCfiOp::Offset {
            register: 73,
            factored_offset: 6, // (32 - (-16)) / 8 = 6
        });
    }

    #[test]
    fn dwarf_cfi_frameless_function() {
        let info = UnwindInfo {
            saved_registers: vec![],
            frame_size: 32,
            has_frame_pointer: false,
            is_leaf: true,
            has_dynamic_alloc: false,
            personality: None,
            has_lsda: false,
        };
        let ops = generate_dwarf_cfi(&info);

        // Frameless: just DefCfa SP, frame_size
        assert_eq!(ops.len(), 1);
        assert_eq!(ops[0], DwarfCfiOp::DefCfa {
            register: DWARF_REG_SP,
            offset: 32,
        });
    }

    #[test]
    fn dwarf_cfi_op_variants_constructible() {
        // Verify all DwarfCfiOp variants can be constructed and compared.
        let def_cfa = DwarfCfiOp::DefCfa { register: 31, offset: 16 };
        let offset = DwarfCfiOp::Offset { register: 29, factored_offset: 2 };
        let reg = DwarfCfiOp::Register { register: 19, value_register: 0 };
        let restore = DwarfCfiOp::Restore { register: 19 };

        assert_ne!(def_cfa, offset);
        assert_ne!(reg, restore);
        assert_eq!(def_cfa.clone(), def_cfa);
    }

    // --- UnwindInfo with personality/LSDA ---

    #[test]
    fn unwind_info_with_personality() {
        let mut info = UnwindInfo::leaf_frame();
        info.personality = Some("__gxx_personality_v0".to_string());
        info.has_lsda = true;
        assert!(info.personality.is_some());
        assert!(info.has_lsda);
    }

    #[test]
    fn compact_unwind_entry_mode_extraction() {
        let entry = CompactUnwindEntry {
            encoding: COMPACT_MODE_FRAME | COMPACT_X19_X20 | COMPACT_D8_D9,
            function_start: 0,
            function_length: 64,
            personality: 0,
            lsda: 0,
        };
        assert_eq!(entry.mode(), COMPACT_MODE_FRAME);
        assert!(!entry.needs_dwarf_fallback());
    }

    // ===================================================================
    // SIMD/FP vector argument passing tests
    // ===================================================================

    // --- Pure FP function: (f32, f32) -> f32 ---

    #[test]
    fn classify_pure_f32_function() {
        // fn add_f32(f32, f32) -> f32
        // Both args in V0, V1; return in V0
        let locs = AppleAArch64ABI::classify_params(&[Type::F32, Type::F32]);
        assert_eq!(locs.len(), 2);
        assert_eq!(locs[0], ArgLocation::Reg(gpr::V0));
        assert_eq!(locs[1], ArgLocation::Reg(gpr::V1));

        let ret = AppleAArch64ABI::classify_returns(&[Type::F32]);
        assert_eq!(ret.len(), 1);
        assert_eq!(ret[0], ArgLocation::Reg(gpr::V0));
    }

    // --- Mixed int + FP: (i64, f64, i32, f32) -> f64 ---

    #[test]
    fn classify_mixed_int_fp_with_return() {
        // fn mixed(i64, f64, i32, f32) -> f64
        // i64 -> X0, f64 -> V0, i32 -> X1, f32 -> V1
        let locs = AppleAArch64ABI::classify_params(
            &[Type::I64, Type::F64, Type::I32, Type::F32],
        );
        assert_eq!(locs.len(), 4);
        assert_eq!(locs[0], ArgLocation::Reg(gpr::X0));   // i64
        assert_eq!(locs[1], ArgLocation::Reg(gpr::V0));   // f64
        assert_eq!(locs[2], ArgLocation::Reg(gpr::X1));   // i32
        assert_eq!(locs[3], ArgLocation::Reg(gpr::V1));   // f32

        // Return f64 -> V0
        let ret = AppleAArch64ABI::classify_returns(&[Type::F64]);
        assert_eq!(ret.len(), 1);
        assert_eq!(ret[0], ArgLocation::Reg(gpr::V0));
    }

    // --- >8 FP args: overflow to stack ---

    #[test]
    fn classify_fp_overflow_10_f32_args() {
        // 10 f32 args: first 8 in V0-V7, last 2 on stack
        let params: Vec<Type> = (0..10).map(|_| Type::F32).collect();
        let locs = AppleAArch64ABI::classify_params(&params);
        assert_eq!(locs.len(), 10);
        for i in 0..8 {
            assert_eq!(locs[i], ArgLocation::Reg(FPR_ARG_REGS[i]),
                "F32 arg {} should be in V{}", i, i);
        }
        // 9th and 10th on stack (4-byte F32, promoted to 8-byte aligned slot)
        assert!(matches!(locs[8], ArgLocation::Stack { offset: 0, size: 4 }));
        assert!(matches!(locs[9], ArgLocation::Stack { offset: 8, size: 4 }));
    }

    #[test]
    fn classify_fp_overflow_9_f64_args() {
        // 9 f64 args: first 8 in V0-V7, 9th on stack
        let params: Vec<Type> = (0..9).map(|_| Type::F64).collect();
        let locs = AppleAArch64ABI::classify_params(&params);
        assert_eq!(locs.len(), 9);
        for i in 0..8 {
            assert_eq!(locs[i], ArgLocation::Reg(FPR_ARG_REGS[i]));
        }
        assert!(matches!(locs[8], ArgLocation::Stack { offset: 0, size: 8 }));
    }

    // --- 128-bit NEON vector arg passing ---

    #[test]
    fn classify_v128_param_in_fpr() {
        // Single 128-bit vector -> V0
        let locs = AppleAArch64ABI::classify_params(&[Type::V128]);
        assert_eq!(locs.len(), 1);
        assert_eq!(locs[0], ArgLocation::Reg(gpr::V0));
    }

    #[test]
    fn classify_v128_return() {
        // V128 return -> V0
        let ret = AppleAArch64ABI::classify_returns(&[Type::V128]);
        assert_eq!(ret.len(), 1);
        assert_eq!(ret[0], ArgLocation::Reg(gpr::V0));
    }

    #[test]
    fn classify_multiple_v128_params() {
        // 4 vector args -> V0, V1, V2, V3
        let params = vec![Type::V128; 4];
        let locs = AppleAArch64ABI::classify_params(&params);
        assert_eq!(locs.len(), 4);
        for i in 0..4 {
            assert_eq!(locs[i], ArgLocation::Reg(FPR_ARG_REGS[i]));
        }
    }

    #[test]
    fn classify_v128_overflow_to_stack() {
        // 9 vector args: V0-V7 in regs, 9th on stack (16-byte aligned)
        let params: Vec<Type> = (0..9).map(|_| Type::V128).collect();
        let locs = AppleAArch64ABI::classify_params(&params);
        assert_eq!(locs.len(), 9);
        for i in 0..8 {
            assert_eq!(locs[i], ArgLocation::Reg(FPR_ARG_REGS[i]));
        }
        assert!(matches!(locs[8], ArgLocation::Stack { offset: 0, size: 16 }));
    }

    #[test]
    fn classify_mixed_v128_and_f64() {
        // V128 and F64 share the same FPR sequence
        // fn(v128, f64, v128) -> V0, V1, V2
        let locs = AppleAArch64ABI::classify_params(
            &[Type::V128, Type::F64, Type::V128],
        );
        assert_eq!(locs.len(), 3);
        assert_eq!(locs[0], ArgLocation::Reg(gpr::V0));  // v128
        assert_eq!(locs[1], ArgLocation::Reg(gpr::V1));  // f64
        assert_eq!(locs[2], ArgLocation::Reg(gpr::V2));  // v128
    }

    #[test]
    fn classify_mixed_int_v128() {
        // int and vector args use independent register pools
        // fn(i64, v128, i64, v128) -> X0, V0, X1, V1
        let locs = AppleAArch64ABI::classify_params(
            &[Type::I64, Type::V128, Type::I64, Type::V128],
        );
        assert_eq!(locs.len(), 4);
        assert_eq!(locs[0], ArgLocation::Reg(gpr::X0));
        assert_eq!(locs[1], ArgLocation::Reg(gpr::V0));
        assert_eq!(locs[2], ArgLocation::Reg(gpr::X1));
        assert_eq!(locs[3], ArgLocation::Reg(gpr::V1));
    }

    // --- FP return values ---

    #[test]
    fn classify_f32_return() {
        let ret = AppleAArch64ABI::classify_returns(&[Type::F32]);
        assert_eq!(ret.len(), 1);
        assert_eq!(ret[0], ArgLocation::Reg(gpr::V0));
    }

    #[test]
    fn classify_multiple_fp_returns() {
        // Returning (f64, f32) -> V0, V1
        let ret = AppleAArch64ABI::classify_returns(&[Type::F64, Type::F32]);
        assert_eq!(ret.len(), 2);
        assert_eq!(ret[0], ArgLocation::Reg(gpr::V0));
        assert_eq!(ret[1], ArgLocation::Reg(gpr::V1));
    }

    // --- HFA return (struct with 2-4 floats) ---

    #[test]
    fn classify_hfa_return_two_f32() {
        // struct { f32, f32 } -> HFA return in S0, S1 (typed FPR registers)
        let hfa = Type::Struct(vec![Type::F32, Type::F32]);
        let ret = AppleAArch64ABI::classify_returns(&[hfa]);
        assert_eq!(ret.len(), 1);
        assert_eq!(ret[0], ArgLocation::RegSequence(vec![gpr::S0, gpr::S1]));
    }

    #[test]
    fn classify_hfa_return_three_f64() {
        // struct { f64, f64, f64 } -> HFA return in D0, D1, D2
        let hfa = Type::Struct(vec![Type::F64, Type::F64, Type::F64]);
        let ret = AppleAArch64ABI::classify_returns(&[hfa]);
        assert_eq!(ret.len(), 1);
        assert_eq!(ret[0], ArgLocation::RegSequence(vec![gpr::D0, gpr::D1, gpr::D2]));
    }

    #[test]
    fn classify_hfa_return_four_f32() {
        // struct { f32, f32, f32, f32 } -> HFA in S0-S3
        let hfa = Type::Struct(vec![Type::F32, Type::F32, Type::F32, Type::F32]);
        let ret = AppleAArch64ABI::classify_returns(&[hfa]);
        assert_eq!(ret.len(), 1);
        assert_eq!(ret[0], ArgLocation::RegSequence(vec![gpr::S0, gpr::S1, gpr::S2, gpr::S3]));
    }

    // --- classify_fp_arg() typed register helper ---

    #[test]
    fn classify_fp_arg_f32_returns_s_register() {
        let (loc, next) = AppleAArch64ABI::classify_fp_arg(&Type::F32, 0).unwrap();
        assert_eq!(loc, ArgLocation::Reg(gpr::S0));
        assert_eq!(next, 1);
    }

    #[test]
    fn classify_fp_arg_f64_returns_d_register() {
        let (loc, next) = AppleAArch64ABI::classify_fp_arg(&Type::F64, 0).unwrap();
        assert_eq!(loc, ArgLocation::Reg(gpr::D0));
        assert_eq!(next, 1);
    }

    #[test]
    fn classify_fp_arg_v128_returns_v_register() {
        let (loc, next) = AppleAArch64ABI::classify_fp_arg(&Type::V128, 0).unwrap();
        assert_eq!(loc, ArgLocation::Reg(gpr::V0));
        assert_eq!(next, 1);
    }

    #[test]
    fn classify_fp_arg_at_index_3() {
        // F32 at fpr_idx=3 -> S3
        let (loc, next) = AppleAArch64ABI::classify_fp_arg(&Type::F32, 3).unwrap();
        assert_eq!(loc, ArgLocation::Reg(gpr::S3));
        assert_eq!(next, 4);
    }

    #[test]
    fn classify_fp_arg_exhausted_returns_none() {
        // fpr_idx=8 (all slots used) -> None
        assert!(AppleAArch64ABI::classify_fp_arg(&Type::F64, 8).is_none());
    }

    #[test]
    fn classify_fp_arg_non_fp_type_returns_none() {
        // Integer type is not a FP arg
        assert!(AppleAArch64ABI::classify_fp_arg(&Type::I64, 0).is_none());
    }

    // --- classify_vector_arg() helper ---

    #[test]
    fn classify_vector_arg_at_zero() {
        let (loc, next) = AppleAArch64ABI::classify_vector_arg(0).unwrap();
        assert_eq!(loc, ArgLocation::Reg(gpr::V0));
        assert_eq!(next, 1);
    }

    #[test]
    fn classify_vector_arg_at_7() {
        let (loc, next) = AppleAArch64ABI::classify_vector_arg(7).unwrap();
        assert_eq!(loc, ArgLocation::Reg(gpr::V7));
        assert_eq!(next, 8);
    }

    #[test]
    fn classify_vector_arg_exhausted() {
        assert!(AppleAArch64ABI::classify_vector_arg(8).is_none());
    }

    // --- call_clobber_fprs ---

    #[test]
    fn call_clobber_fprs_correct() {
        let clobber = AppleAArch64ABI::call_clobber_fprs();
        assert_eq!(clobber.len(), 24); // V0-V7 (8) + V16-V31 (16)
        // V0-V7 are arg registers, clobbered
        assert!(clobber.contains(&gpr::V0));
        assert!(clobber.contains(&gpr::V7));
        // V8-V15 are callee-saved, NOT in clobber list
        assert!(!clobber.contains(&gpr::V8));
        assert!(!clobber.contains(&gpr::V15));
        // V16-V31 are clobbered
        assert!(clobber.contains(&gpr::V16));
        assert!(clobber.contains(&gpr::V31));
    }

    // --- S/D register accessor tests ---

    #[test]
    fn s_arg_regs_correct() {
        let s_regs = AppleAArch64ABI::s_arg_regs();
        assert_eq!(s_regs.len(), 8);
        assert_eq!(s_regs[0], gpr::S0);
        assert_eq!(s_regs[7], gpr::S7);
    }

    #[test]
    fn d_arg_regs_correct() {
        let d_regs = AppleAArch64ABI::d_arg_regs();
        assert_eq!(d_regs.len(), 8);
        assert_eq!(d_regs[0], gpr::D0);
        assert_eq!(d_regs[7], gpr::D7);
    }

    // --- V128 in variadic context ---

    #[test]
    fn variadic_v128_on_stack() {
        // fn(i64, ...) called with (ptr, v128_arg)
        // Fixed: ptr -> X0, Variadic: v128 -> stack (16 bytes)
        let params = vec![Type::I64, Type::V128];
        let locs = AppleAArch64ABI::classify_params_variadic(1, &params);
        assert_eq!(locs.len(), 2);
        assert_eq!(locs[0], ArgLocation::Reg(gpr::X0));
        assert!(matches!(locs[1], ArgLocation::Stack { offset: 0, size: 16 }));
    }

    // --- V128 stack args size ---

    #[test]
    fn stack_args_size_with_v128_overflow() {
        // 9 v128 args: 8 in regs, 1 on stack (16 bytes)
        let params: Vec<Type> = (0..9).map(|_| Type::V128).collect();
        let size = AppleAArch64ABI::stack_args_size(&params);
        assert_eq!(size, 16);
    }

    // --- V128 type properties ---

    #[test]
    fn v128_type_properties() {
        assert_eq!(Type::V128.bytes(), 16);
        assert_eq!(Type::V128.bits(), 128);
        assert_eq!(Type::V128.storage_bits(), 128);
        assert_eq!(Type::V128.align(), 16);
        assert!(!Type::V128.is_aggregate());
        assert!(Type::V128.is_scalar());
    }

    // ===================================================================
    // ClassifyResult API tests
    // ===================================================================

    /// Helper: assert a classify_variadic result is OnStack with expected values.
    fn assert_variadic_stack(
        result: (ClassifyResult, i64),
        expected_offset: i64,
        expected_size: u32,
        expected_next: i64,
    ) {
        let (classify, next) = result;
        assert_eq!(
            classify,
            ClassifyResult::OnStack {
                offset: expected_offset,
                size: expected_size,
                align: 8,
            }
        );
        assert_eq!(next, expected_next);
    }

    // --- ClassifyResult construction and equality ---

    #[test]
    fn classify_result_in_regs_construction_and_equality() {
        let lhs = ClassifyResult::InRegs {
            regs: vec![gpr::X0, gpr::X1],
        };
        let rhs = ClassifyResult::InRegs {
            regs: vec![gpr::X0, gpr::X1],
        };
        assert_eq!(lhs, rhs);
    }

    #[test]
    fn classify_result_on_stack_construction_and_equality() {
        let lhs = ClassifyResult::OnStack {
            offset: 16,
            size: 8,
            align: 8,
        };
        let rhs = ClassifyResult::OnStack {
            offset: 16,
            size: 8,
            align: 8,
        };
        assert_eq!(lhs, rhs);
    }

    #[test]
    fn classify_result_indirect_construction_and_equality() {
        let lhs = ClassifyResult::Indirect { ptr_reg: gpr::X0 };
        let rhs = ClassifyResult::Indirect { ptr_reg: gpr::X0 };
        assert_eq!(lhs, rhs);
    }

    #[test]
    fn classify_result_hfa_construction_and_equality() {
        let lhs = ClassifyResult::Hfa {
            base_ty: HfaBaseType::F32,
            count: 4,
            first_reg: gpr::V0,
        };
        let rhs = ClassifyResult::Hfa {
            base_ty: HfaBaseType::F32,
            count: 4,
            first_reg: gpr::V0,
        };
        assert_eq!(lhs, rhs);
    }

    // --- classify_aggregate tests ---

    #[test]
    fn classify_aggregate_small_struct_in_one_gpr() {
        // struct { i32, i16 } = 8 bytes -> InRegs(X0)
        let ty = Type::Struct(vec![Type::I32, Type::I16]);
        assert_eq!(
            AppleAArch64ABI::classify_aggregate(&ty),
            ClassifyResult::InRegs {
                regs: vec![gpr::X0],
            }
        );
    }

    #[test]
    fn classify_aggregate_medium_struct_in_two_gprs() {
        // struct { i64, i32 } = 16 bytes -> InRegs(X0, X1)
        let ty = Type::Struct(vec![Type::I64, Type::I32]);
        assert_eq!(
            AppleAArch64ABI::classify_aggregate(&ty),
            ClassifyResult::InRegs {
                regs: vec![gpr::X0, gpr::X1],
            }
        );
    }

    #[test]
    fn classify_aggregate_large_struct_indirect() {
        // struct { i64, i64, i64 } = 24 bytes -> Indirect(X0)
        let ty = Type::Struct(vec![Type::I64, Type::I64, Type::I64]);
        assert_eq!(
            AppleAArch64ABI::classify_aggregate(&ty),
            ClassifyResult::Indirect { ptr_reg: gpr::X0 }
        );
    }

    #[test]
    fn classify_aggregate_hfa_struct() {
        // struct { f32, f32 } -> HFA(F32, 2, V0)
        let ty = Type::Struct(vec![Type::F32, Type::F32]);
        assert_eq!(
            AppleAArch64ABI::classify_aggregate(&ty),
            ClassifyResult::Hfa {
                base_ty: HfaBaseType::F32,
                count: 2,
                first_reg: gpr::V0,
            }
        );
    }

    #[test]
    fn classify_aggregate_empty_struct() {
        // struct {} = 0 bytes -> InRegs(X0) (0 is <= 8)
        let ty = Type::Struct(vec![]);
        assert_eq!(
            AppleAArch64ABI::classify_aggregate(&ty),
            ClassifyResult::InRegs {
                regs: vec![gpr::X0],
            }
        );
    }

    #[test]
    fn classify_aggregate_single_field_struct() {
        // struct { i64 } = 8 bytes -> InRegs(X0)
        let ty = Type::Struct(vec![Type::I64]);
        assert_eq!(
            AppleAArch64ABI::classify_aggregate(&ty),
            ClassifyResult::InRegs {
                regs: vec![gpr::X0],
            }
        );
    }

    #[test]
    fn classify_aggregate_nested_struct() {
        // struct { struct { i32, i16 }, i8 } -> 12 bytes -> InRegs(X0, X1)
        let inner = Type::Struct(vec![Type::I32, Type::I16]);
        let ty = Type::Struct(vec![inner, Type::I8]);
        assert_eq!(
            AppleAArch64ABI::classify_aggregate(&ty),
            ClassifyResult::InRegs {
                regs: vec![gpr::X0, gpr::X1],
            }
        );
    }

    #[test]
    fn classify_aggregate_hfa_array() {
        // f64[3] -> HFA(F64, 3, V0)
        let ty = Type::Array(Box::new(Type::F64), 3);
        assert_eq!(
            AppleAArch64ABI::classify_aggregate(&ty),
            ClassifyResult::Hfa {
                base_ty: HfaBaseType::F64,
                count: 3,
                first_reg: gpr::V0,
            }
        );
    }

    // --- classify_variadic tests ---

    #[test]
    fn classify_variadic_i32_goes_on_stack() {
        assert_variadic_stack(AppleAArch64ABI::classify_variadic(&Type::I32, 0), 0, 8, 8);
    }

    #[test]
    fn classify_variadic_i64_goes_on_stack() {
        assert_variadic_stack(AppleAArch64ABI::classify_variadic(&Type::I64, 0), 0, 8, 8);
    }

    #[test]
    fn classify_variadic_f32_goes_on_stack() {
        assert_variadic_stack(AppleAArch64ABI::classify_variadic(&Type::F32, 0), 0, 8, 8);
    }

    #[test]
    fn classify_variadic_f64_goes_on_stack() {
        assert_variadic_stack(AppleAArch64ABI::classify_variadic(&Type::F64, 0), 0, 8, 8);
    }

    #[test]
    fn classify_variadic_v128_goes_on_stack() {
        assert_variadic_stack(AppleAArch64ABI::classify_variadic(&Type::V128, 0), 0, 16, 16);
    }

    #[test]
    fn classify_variadic_small_struct_goes_on_stack() {
        let ty = Type::Struct(vec![Type::I32, Type::I16]);
        assert_variadic_stack(AppleAArch64ABI::classify_variadic(&ty, 0), 0, 8, 8);
    }

    #[test]
    fn classify_variadic_large_struct_goes_on_stack() {
        let ty = Type::Struct(vec![Type::I64, Type::I64, Type::I64]);
        assert_variadic_stack(AppleAArch64ABI::classify_variadic(&ty, 0), 0, 24, 24);
    }

    #[test]
    fn classify_variadic_aligns_incoming_offset() {
        // stack_offset=5 -> aligned to 8, then 8-byte I32 slot
        assert_variadic_stack(AppleAArch64ABI::classify_variadic(&Type::I32, 5), 8, 8, 16);
    }

    #[test]
    fn classify_variadic_empty_struct_promotes_to_eight_bytes() {
        let ty = Type::Struct(vec![]);
        assert_variadic_stack(AppleAArch64ABI::classify_variadic(&ty, 0), 0, 8, 8);
    }

    #[test]
    fn classify_variadic_chains_offsets() {
        // Chain: I32 at offset 0, then V128 at next offset
        let (first, next) = AppleAArch64ABI::classify_variadic(&Type::I32, 0);
        assert_eq!(
            first,
            ClassifyResult::OnStack {
                offset: 0,
                size: 8,
                align: 8,
            }
        );
        let (second, final_next) = AppleAArch64ABI::classify_variadic(&Type::V128, next);
        assert_eq!(
            second,
            ClassifyResult::OnStack {
                offset: 8,
                size: 16,
                align: 8,
            }
        );
        assert_eq!(final_next, 24);
    }

    // --- classify_hfa tests ---

    #[test]
    fn classify_hfa_one_f32_field() {
        let ty = Type::Struct(vec![Type::F32]);
        assert_eq!(
            AppleAArch64ABI::classify_hfa(&ty),
            Some((HfaBaseType::F32, 1))
        );
    }

    #[test]
    fn classify_hfa_two_f32_fields() {
        let ty = Type::Struct(vec![Type::F32, Type::F32]);
        assert_eq!(
            AppleAArch64ABI::classify_hfa(&ty),
            Some((HfaBaseType::F32, 2))
        );
    }

    #[test]
    fn classify_hfa_three_f32_fields_array() {
        let ty = Type::Array(Box::new(Type::F32), 3);
        assert_eq!(
            AppleAArch64ABI::classify_hfa(&ty),
            Some((HfaBaseType::F32, 3))
        );
    }

    #[test]
    fn classify_hfa_four_f32_fields_nested() {
        let ty = Type::Struct(vec![
            Type::Struct(vec![Type::F32, Type::F32]),
            Type::Struct(vec![Type::F32, Type::F32]),
        ]);
        assert_eq!(
            AppleAArch64ABI::classify_hfa(&ty),
            Some((HfaBaseType::F32, 4))
        );
    }

    #[test]
    fn classify_hfa_one_f64_field() {
        let ty = Type::Struct(vec![Type::F64]);
        assert_eq!(
            AppleAArch64ABI::classify_hfa(&ty),
            Some((HfaBaseType::F64, 1))
        );
    }

    #[test]
    fn classify_hfa_two_f64_fields() {
        let ty = Type::Struct(vec![Type::F64, Type::F64]);
        assert_eq!(
            AppleAArch64ABI::classify_hfa(&ty),
            Some((HfaBaseType::F64, 2))
        );
    }

    #[test]
    fn classify_hfa_three_f64_fields_nested() {
        // struct { f64, f64[2] } -> 3x F64 HFA
        let ty = Type::Struct(vec![Type::F64, Type::Array(Box::new(Type::F64), 2)]);
        assert_eq!(
            AppleAArch64ABI::classify_hfa(&ty),
            Some((HfaBaseType::F64, 3))
        );
    }

    #[test]
    fn classify_hfa_four_f64_fields() {
        let ty = Type::Array(Box::new(Type::F64), 4);
        assert_eq!(
            AppleAArch64ABI::classify_hfa(&ty),
            Some((HfaBaseType::F64, 4))
        );
    }

    #[test]
    fn classify_hfa_mixed_fields_not_hfa() {
        // struct { f32, f64 } -> NOT HFA (mixed FP types)
        let ty = Type::Struct(vec![Type::F32, Type::F64]);
        assert_eq!(AppleAArch64ABI::classify_hfa(&ty), None);
    }

    #[test]
    fn classify_hfa_array_of_structs() {
        // [struct { f32 }; 4] -> 4x F32 HFA
        let elem = Type::Struct(vec![Type::F32]);
        let ty = Type::Array(Box::new(elem), 4);
        assert_eq!(
            AppleAArch64ABI::classify_hfa(&ty),
            Some((HfaBaseType::F32, 4))
        );
    }

    #[test]
    fn classify_hfa_nested_structs() {
        // struct { struct { f64 }, struct { f64 } } -> 2x F64
        let ty = Type::Struct(vec![
            Type::Struct(vec![Type::F64]),
            Type::Struct(vec![Type::F64]),
        ]);
        assert_eq!(
            AppleAArch64ABI::classify_hfa(&ty),
            Some((HfaBaseType::F64, 2))
        );
    }

    #[test]
    fn classify_hfa_zero_size_array_is_not_hfa() {
        let ty = Type::Array(Box::new(Type::F32), 0);
        assert_eq!(AppleAArch64ABI::classify_hfa(&ty), None);
    }

    #[test]
    fn classify_hfa_five_field_struct_is_not_hfa() {
        let ty = Type::Struct(vec![Type::F32; 5]);
        assert_eq!(AppleAArch64ABI::classify_hfa(&ty), None);
    }

    #[test]
    fn classify_hfa_nested_mixed_struct_is_not_hfa() {
        // struct { struct { f32 }, struct { f64 } } -> NOT HFA (mixed)
        let ty = Type::Struct(vec![
            Type::Struct(vec![Type::F32]),
            Type::Struct(vec![Type::F64]),
        ]);
        assert_eq!(AppleAArch64ABI::classify_hfa(&ty), None);
    }

    // ===================================================================
    // HFA typed register and RegSequence tests (issue #140)
    // ===================================================================

    #[test]
    fn classify_hfa_param_f64_uses_d_registers() {
        // struct { f64, f64 } -> HFA in D0, D1
        let hfa = Type::Struct(vec![Type::F64, Type::F64]);
        let locs = AppleAArch64ABI::classify_params(&[hfa]);
        assert_eq!(locs.len(), 1);
        assert_eq!(locs[0], ArgLocation::RegSequence(vec![gpr::D0, gpr::D1]));
    }

    #[test]
    fn classify_hfa_param_f64_four_members() {
        // struct { f64, f64, f64, f64 } -> HFA in D0-D3
        let hfa = Type::Struct(vec![Type::F64, Type::F64, Type::F64, Type::F64]);
        let locs = AppleAArch64ABI::classify_params(&[hfa]);
        assert_eq!(locs.len(), 1);
        assert_eq!(locs[0], ArgLocation::RegSequence(vec![gpr::D0, gpr::D1, gpr::D2, gpr::D3]));
    }

    #[test]
    fn classify_hfa_param_single_f32() {
        // struct { f32 } -> single-member HFA in S0
        let hfa = Type::Struct(vec![Type::F32]);
        let locs = AppleAArch64ABI::classify_params(&[hfa]);
        assert_eq!(locs.len(), 1);
        assert_eq!(locs[0], ArgLocation::RegSequence(vec![gpr::S0]));
    }

    #[test]
    fn classify_hfa_param_single_f64() {
        // struct { f64 } -> single-member HFA in D0
        let hfa = Type::Struct(vec![Type::F64]);
        let locs = AppleAArch64ABI::classify_params(&[hfa]);
        assert_eq!(locs.len(), 1);
        assert_eq!(locs[0], ArgLocation::RegSequence(vec![gpr::D0]));
    }

    #[test]
    fn classify_hfa_param_mixed_with_gpr_args() {
        // fn(i64, HFA{f32, f32, f32}, i32) -> X0, S0-S2, X1
        // GPR and FPR allocation are independent.
        let hfa = Type::Struct(vec![Type::F32, Type::F32, Type::F32]);
        let locs = AppleAArch64ABI::classify_params(&[Type::I64, hfa, Type::I32]);
        assert_eq!(locs.len(), 3);
        assert_eq!(locs[0], ArgLocation::Reg(gpr::X0));
        assert_eq!(locs[1], ArgLocation::RegSequence(vec![gpr::S0, gpr::S1, gpr::S2]));
        assert_eq!(locs[2], ArgLocation::Reg(gpr::X1));
    }

    #[test]
    fn classify_hfa_param_after_f64_scalar_uses_d_regs() {
        // fn(f64, HFA{f64, f64}) -> V0 for scalar f64, D1+D2 for HFA
        let hfa = Type::Struct(vec![Type::F64, Type::F64]);
        let locs = AppleAArch64ABI::classify_params(&[Type::F64, hfa]);
        assert_eq!(locs.len(), 2);
        assert_eq!(locs[0], ArgLocation::Reg(gpr::V0));
        assert_eq!(locs[1], ArgLocation::RegSequence(vec![gpr::D1, gpr::D2]));
    }

    #[test]
    fn classify_hfa_param_three_f64_after_five_scalars() {
        // fn(f64, f64, f64, f64, f64, HFA{f64, f64, f64})
        // Scalars consume FPR slots 0-4. HFA needs 3 slots (5,6,7) -> fits.
        let mut params: Vec<Type> = vec![Type::F64; 5];
        params.push(Type::Struct(vec![Type::F64, Type::F64, Type::F64]));
        let locs = AppleAArch64ABI::classify_params(&params);
        assert_eq!(locs.len(), 6);
        assert_eq!(locs[5], ArgLocation::RegSequence(vec![gpr::D5, gpr::D6, gpr::D7]));
    }

    #[test]
    fn classify_hfa_param_overflow_when_partially_fits() {
        // fn(f64, f64, f64, f64, f64, f64, HFA{f64, f64, f64})
        // 6 scalars consume FPR 0-5. HFA needs 3 but only 2 left -> stack.
        let mut params: Vec<Type> = vec![Type::F64; 6];
        params.push(Type::Struct(vec![Type::F64, Type::F64, Type::F64]));
        let locs = AppleAArch64ABI::classify_params(&params);
        assert_eq!(locs.len(), 7);
        assert!(matches!(locs[6], ArgLocation::Stack { .. }));
    }

    #[test]
    fn classify_hfa_return_single_f32() {
        // struct { f32 } -> return in S0
        let hfa = Type::Struct(vec![Type::F32]);
        let ret = AppleAArch64ABI::classify_returns(&[hfa]);
        assert_eq!(ret.len(), 1);
        assert_eq!(ret[0], ArgLocation::RegSequence(vec![gpr::S0]));
    }

    #[test]
    fn classify_hfa_return_single_f64() {
        // struct { f64 } -> return in D0
        let hfa = Type::Struct(vec![Type::F64]);
        let ret = AppleAArch64ABI::classify_returns(&[hfa]);
        assert_eq!(ret.len(), 1);
        assert_eq!(ret[0], ArgLocation::RegSequence(vec![gpr::D0]));
    }

    #[test]
    fn classify_hfa_return_four_f64() {
        // struct { f64, f64, f64, f64 } -> return in D0-D3
        let hfa = Type::Struct(vec![Type::F64, Type::F64, Type::F64, Type::F64]);
        let ret = AppleAArch64ABI::classify_returns(&[hfa]);
        assert_eq!(ret.len(), 1);
        assert_eq!(ret[0], ArgLocation::RegSequence(vec![gpr::D0, gpr::D1, gpr::D2, gpr::D3]));
    }

    #[test]
    fn classify_hfa_return_array_of_f64() {
        // f64[2] -> HFA return in D0, D1
        let hfa = Type::Array(Box::new(Type::F64), 2);
        let ret = AppleAArch64ABI::classify_returns(&[hfa]);
        assert_eq!(ret.len(), 1);
        assert_eq!(ret[0], ArgLocation::RegSequence(vec![gpr::D0, gpr::D1]));
    }

    // ===================================================================
    // Large struct passing/return tests (issue #140)
    // ===================================================================

    #[test]
    fn classify_large_struct_32_bytes_indirect_via_gpr() {
        // struct { i64, i64, i64, i64 } = 32 bytes > 16 -> indirect
        let large = Type::Struct(vec![Type::I64, Type::I64, Type::I64, Type::I64]);
        assert_eq!(large.bytes(), 32);
        let locs = AppleAArch64ABI::classify_params(&[large]);
        assert_eq!(locs.len(), 1);
        assert_eq!(locs[0], ArgLocation::Indirect { ptr_reg: gpr::X0 });
    }

    #[test]
    fn classify_large_struct_param_after_gprs_used() {
        // fn(i64, i64, i64, i64, i64, i64, i64, large_struct)
        // All 7 GPR slots X0-X6 used, large struct uses X7.
        let large = Type::Struct(vec![Type::I64, Type::I64, Type::I64]);
        let mut params: Vec<Type> = vec![Type::I64; 7];
        params.push(large);
        let locs = AppleAArch64ABI::classify_params(&params);
        assert_eq!(locs.len(), 8);
        assert_eq!(locs[7], ArgLocation::Indirect { ptr_reg: gpr::X7 });
    }

    #[test]
    fn classify_large_struct_param_no_gprs_left_goes_to_stack() {
        // fn(i64 x8, large_struct) -> all 8 GPRs used, pointer goes on stack
        let large = Type::Struct(vec![Type::I64, Type::I64, Type::I64]);
        let mut params: Vec<Type> = vec![Type::I64; 8];
        params.push(large);
        let locs = AppleAArch64ABI::classify_params(&params);
        assert_eq!(locs.len(), 9);
        // Large struct: pointer on stack (8 bytes for the pointer)
        assert!(matches!(locs[8], ArgLocation::Stack { size: 8, .. }));
    }

    #[test]
    fn classify_large_struct_return_via_x8() {
        // Return struct > 16 bytes via sret (X8)
        let large = Type::Struct(vec![Type::I64, Type::I64, Type::I64, Type::I64]);
        assert!(large.bytes() > 16);
        let locs = AppleAArch64ABI::classify_returns(&[large]);
        assert_eq!(locs.len(), 1);
        assert_eq!(locs[0], ArgLocation::Indirect { ptr_reg: gpr::X8 });
    }

    #[test]
    fn classify_17_byte_struct_indirect() {
        // Struct of exactly 17 bytes (just over the 16-byte threshold)
        // struct { i64, i64, i8 } -> size depends on alignment/padding
        // Actually: 8 + 8 + 1 = 17 bytes, padded to 24 with alignment
        let ty = Type::Struct(vec![Type::I64, Type::I64, Type::I8]);
        let size = ty.bytes();
        assert!(size > 16, "expected > 16 bytes, got {}", size);
        let locs = AppleAArch64ABI::classify_params(&[ty]);
        assert_eq!(locs.len(), 1);
        assert_eq!(locs[0], ArgLocation::Indirect { ptr_reg: gpr::X0 });
    }

    #[test]
    fn classify_large_struct_mixed_with_hfa() {
        // fn(large_struct, HFA{f32, f32}) -> X0 indirect, S0+S1 for HFA
        // GPR and FPR allocation are independent.
        let large = Type::Struct(vec![Type::I64, Type::I64, Type::I64]);
        let hfa = Type::Struct(vec![Type::F32, Type::F32]);
        let locs = AppleAArch64ABI::classify_params(&[large, hfa]);
        assert_eq!(locs.len(), 2);
        assert_eq!(locs[0], ArgLocation::Indirect { ptr_reg: gpr::X0 });
        assert_eq!(locs[1], ArgLocation::RegSequence(vec![gpr::S0, gpr::S1]));
    }

    #[test]
    fn classify_hfa_and_large_struct_interleaved() {
        // fn(HFA{f64, f64}, i32, large_struct, f32, HFA{f32, f32, f32})
        // HFA{f64,f64} -> D0, D1 (FPR 0-1)
        // i32 -> X0 (GPR 0)
        // large_struct -> X1 indirect (GPR 1)
        // f32 -> V2 (FPR 2)
        // HFA{f32,f32,f32} -> S3, S4, S5 (FPR 3-5)
        let hfa_f64 = Type::Struct(vec![Type::F64, Type::F64]);
        let large = Type::Struct(vec![Type::I64, Type::I64, Type::I64]);
        let hfa_f32 = Type::Struct(vec![Type::F32, Type::F32, Type::F32]);
        let locs = AppleAArch64ABI::classify_params(&[hfa_f64, Type::I32, large, Type::F32, hfa_f32]);
        assert_eq!(locs.len(), 5);
        assert_eq!(locs[0], ArgLocation::RegSequence(vec![gpr::D0, gpr::D1]));
        assert_eq!(locs[1], ArgLocation::Reg(gpr::X0));
        assert_eq!(locs[2], ArgLocation::Indirect { ptr_reg: gpr::X1 });
        assert_eq!(locs[3], ArgLocation::Reg(gpr::V2));
        assert_eq!(locs[4], ArgLocation::RegSequence(vec![gpr::S3, gpr::S4, gpr::S5]));
    }

    // ===================================================================
    // RegSequence variant tests
    // ===================================================================

    #[test]
    fn arg_location_reg_sequence_equality() {
        let a = ArgLocation::RegSequence(vec![gpr::S0, gpr::S1]);
        let b = ArgLocation::RegSequence(vec![gpr::S0, gpr::S1]);
        assert_eq!(a, b);
    }

    #[test]
    fn arg_location_reg_sequence_inequality() {
        let a = ArgLocation::RegSequence(vec![gpr::S0, gpr::S1]);
        let b = ArgLocation::RegSequence(vec![gpr::D0, gpr::D1]);
        assert_ne!(a, b);
    }

    #[test]
    fn arg_location_reg_sequence_clone() {
        let a = ArgLocation::RegSequence(vec![gpr::D0, gpr::D1, gpr::D2]);
        let b = a.clone();
        assert_eq!(a, b);
    }

    #[test]
    fn arg_location_reg_sequence_debug() {
        let loc = ArgLocation::RegSequence(vec![gpr::S0]);
        let debug_str = format!("{:?}", loc);
        assert!(debug_str.contains("RegSequence"));
    }

    // ===================================================================
    // Exception Handling ABI tests
    // ===================================================================

    // --- PersonalityFunction tests ---

    #[test]
    fn personality_gxx_symbol_name() {
        let p = PersonalityFunction::GxxPersonalityV0;
        assert_eq!(p.symbol_name(), "__gxx_personality_v0");
    }

    #[test]
    fn personality_rust_symbol_name() {
        let p = PersonalityFunction::RustEhPersonality;
        assert_eq!(p.symbol_name(), "__rust_eh_personality");
    }

    #[test]
    fn personality_objc_symbol_name() {
        let p = PersonalityFunction::ObjcPersonalityV0;
        assert_eq!(p.symbol_name(), "__objc_personality_v0");
    }

    #[test]
    fn personality_gcc_symbol_name() {
        let p = PersonalityFunction::GccPersonalityV0;
        assert_eq!(p.symbol_name(), "__gcc_personality_v0");
    }

    #[test]
    fn personality_custom_symbol_name() {
        let p = PersonalityFunction::Custom("__my_personality".to_string());
        assert_eq!(p.symbol_name(), "__my_personality");
    }

    #[test]
    fn personality_for_language_cpp() {
        assert_eq!(
            PersonalityFunction::for_language(SourceLanguage::Cpp),
            PersonalityFunction::GxxPersonalityV0
        );
    }

    #[test]
    fn personality_for_language_rust() {
        assert_eq!(
            PersonalityFunction::for_language(SourceLanguage::Rust),
            PersonalityFunction::RustEhPersonality
        );
    }

    #[test]
    fn personality_for_language_objc() {
        assert_eq!(
            PersonalityFunction::for_language(SourceLanguage::ObjectiveC),
            PersonalityFunction::ObjcPersonalityV0
        );
    }

    #[test]
    fn personality_for_language_c() {
        assert_eq!(
            PersonalityFunction::for_language(SourceLanguage::C),
            PersonalityFunction::GccPersonalityV0
        );
    }

    #[test]
    fn personality_equality() {
        assert_eq!(PersonalityFunction::GxxPersonalityV0, PersonalityFunction::GxxPersonalityV0);
        assert_ne!(PersonalityFunction::GxxPersonalityV0, PersonalityFunction::RustEhPersonality);
    }

    #[test]
    fn personality_clone() {
        let p = PersonalityFunction::Custom("__test".to_string());
        let p2 = p.clone();
        assert_eq!(p, p2);
    }

    // --- LandingPad tests ---

    #[test]
    fn landing_pad_catch_all_construction() {
        let lp = LandingPad::catch_all(0x40);
        assert_eq!(lp.landing_pad_offset, 0x40);
        assert_eq!(lp.catch_type_indices, vec![0]);
        assert!(lp.filter_type_indices.is_empty());
        assert!(!lp.is_cleanup);
    }

    #[test]
    fn landing_pad_catch_all_predicates() {
        let lp = LandingPad::catch_all(0x40);
        assert!(lp.has_catch());
        assert!(!lp.has_filter());
        assert!(lp.is_catch_all());
        assert!(!lp.is_cleanup);
    }

    #[test]
    fn landing_pad_catch_typed_construction() {
        let lp = LandingPad::catch_typed(0x80, 3);
        assert_eq!(lp.landing_pad_offset, 0x80);
        assert_eq!(lp.catch_type_indices, vec![3]);
        assert!(lp.filter_type_indices.is_empty());
        assert!(!lp.is_cleanup);
    }

    #[test]
    fn landing_pad_catch_typed_predicates() {
        let lp = LandingPad::catch_typed(0x80, 3);
        assert!(lp.has_catch());
        assert!(!lp.has_filter());
        assert!(!lp.is_catch_all()); // type index 3 is not catch-all
        assert!(!lp.is_cleanup);
    }

    #[test]
    fn landing_pad_cleanup_construction() {
        let lp = LandingPad::cleanup(0xC0);
        assert_eq!(lp.landing_pad_offset, 0xC0);
        assert!(lp.catch_type_indices.is_empty());
        assert!(lp.filter_type_indices.is_empty());
        assert!(lp.is_cleanup);
    }

    #[test]
    fn landing_pad_cleanup_predicates() {
        let lp = LandingPad::cleanup(0xC0);
        assert!(!lp.has_catch());
        assert!(!lp.has_filter());
        assert!(!lp.is_catch_all());
        assert!(lp.is_cleanup);
    }

    #[test]
    fn landing_pad_with_filter() {
        let lp = LandingPad {
            landing_pad_offset: 0x50,
            catch_type_indices: Vec::new(),
            filter_type_indices: vec![1, 2],
            is_cleanup: false,
        };
        assert!(!lp.has_catch());
        assert!(lp.has_filter());
        assert!(!lp.is_catch_all());
    }

    #[test]
    fn landing_pad_with_catch_and_filter() {
        let lp = LandingPad {
            landing_pad_offset: 0x60,
            catch_type_indices: vec![1],
            filter_type_indices: vec![2, 3],
            is_cleanup: false,
        };
        assert!(lp.has_catch());
        assert!(lp.has_filter());
        assert!(!lp.is_catch_all());
    }

    #[test]
    fn landing_pad_equality() {
        let lp1 = LandingPad::catch_all(0x40);
        let lp2 = LandingPad::catch_all(0x40);
        assert_eq!(lp1, lp2);

        let lp3 = LandingPad::cleanup(0x40);
        assert_ne!(lp1, lp3);
    }

    // --- CallSiteEntry tests ---

    #[test]
    fn call_site_entry_with_landing_pad() {
        let cs = CallSiteEntry {
            start_offset: 0x10,
            length: 0x20,
            landing_pad_offset: 0x40,
            action_index: 1,
        };
        assert!(cs.has_landing_pad());
        assert!(cs.has_action());
    }

    #[test]
    fn call_site_entry_no_landing_pad() {
        let cs = CallSiteEntry {
            start_offset: 0x10,
            length: 0x20,
            landing_pad_offset: 0,
            action_index: 0,
        };
        assert!(!cs.has_landing_pad());
        assert!(!cs.has_action());
    }

    #[test]
    fn call_site_entry_landing_pad_no_action() {
        let cs = CallSiteEntry {
            start_offset: 0x10,
            length: 0x08,
            landing_pad_offset: 0x30,
            action_index: 0,
        };
        assert!(cs.has_landing_pad());
        assert!(!cs.has_action());
    }

    #[test]
    fn call_site_entry_equality() {
        let cs1 = CallSiteEntry {
            start_offset: 0x10,
            length: 0x20,
            landing_pad_offset: 0x40,
            action_index: 1,
        };
        let cs2 = cs1.clone();
        assert_eq!(cs1, cs2);
    }

    // --- ActionEntry tests ---

    #[test]
    fn action_entry_catch() {
        let action = ActionEntry { type_filter: 1, next_offset: 0 };
        assert!(action.is_catch());
        assert!(!action.is_cleanup());
        assert!(!action.is_filter());
        assert!(!action.has_next());
    }

    #[test]
    fn action_entry_cleanup() {
        let action = ActionEntry { type_filter: 0, next_offset: 0 };
        assert!(!action.is_catch());
        assert!(action.is_cleanup());
        assert!(!action.is_filter());
        assert!(!action.has_next());
    }

    #[test]
    fn action_entry_filter() {
        let action = ActionEntry { type_filter: -1, next_offset: 0 };
        assert!(!action.is_catch());
        assert!(!action.is_cleanup());
        assert!(action.is_filter());
        assert!(!action.has_next());
    }

    #[test]
    fn action_entry_with_chain() {
        let action = ActionEntry { type_filter: 2, next_offset: 4 };
        assert!(action.is_catch());
        assert!(action.has_next());
    }

    #[test]
    fn action_entry_equality() {
        let a1 = ActionEntry { type_filter: 1, next_offset: 0 };
        let a2 = ActionEntry { type_filter: 1, next_offset: 0 };
        assert_eq!(a1, a2);

        let a3 = ActionEntry { type_filter: 2, next_offset: 0 };
        assert_ne!(a1, a3);
    }

    // --- TypeTableEntry and TypeTableEncoding tests ---

    #[test]
    fn type_table_encoding_dwarf_values() {
        assert_eq!(TypeTableEncoding::AbsPtr.dwarf_encoding(), 0x00);
        assert_eq!(TypeTableEncoding::PcRelSData4.dwarf_encoding(), 0x1B);
        assert_eq!(TypeTableEncoding::IndirectPcRel.dwarf_encoding(), 0x9B);
        assert_eq!(TypeTableEncoding::Omit.dwarf_encoding(), 0xFF);
    }

    #[test]
    fn type_table_entry_construction() {
        let entry = TypeTableEntry {
            type_info_symbol: "_ZTIi".to_string(),
            encoding: TypeTableEncoding::PcRelSData4,
        };
        assert_eq!(entry.type_info_symbol, "_ZTIi");
        assert_eq!(entry.encoding, TypeTableEncoding::PcRelSData4);
    }

    #[test]
    fn type_table_entry_catch_all_empty_symbol() {
        let entry = TypeTableEntry {
            type_info_symbol: String::new(),
            encoding: TypeTableEncoding::PcRelSData4,
        };
        assert!(entry.type_info_symbol.is_empty());
    }

    #[test]
    fn type_table_entry_equality() {
        let e1 = TypeTableEntry {
            type_info_symbol: "_ZTIi".to_string(),
            encoding: TypeTableEncoding::PcRelSData4,
        };
        let e2 = e1.clone();
        assert_eq!(e1, e2);
    }

    #[test]
    fn type_table_encoding_equality() {
        assert_eq!(TypeTableEncoding::AbsPtr, TypeTableEncoding::AbsPtr);
        assert_ne!(TypeTableEncoding::AbsPtr, TypeTableEncoding::Omit);
    }

    // --- ExceptionHandlingInfo tests ---

    #[test]
    fn eh_info_new_cpp() {
        let eh = ExceptionHandlingInfo::for_cpp();
        assert_eq!(eh.personality, PersonalityFunction::GxxPersonalityV0);
        assert!(eh.landing_pads.is_empty());
        assert!(eh.call_sites.is_empty());
        assert!(eh.actions.is_empty());
        assert!(eh.type_table.is_empty());
        assert_eq!(eh.type_table_encoding, TypeTableEncoding::PcRelSData4);
    }

    #[test]
    fn eh_info_new_rust() {
        let eh = ExceptionHandlingInfo::for_rust();
        assert_eq!(eh.personality, PersonalityFunction::RustEhPersonality);
    }

    #[test]
    fn eh_info_add_landing_pad() {
        let mut eh = ExceptionHandlingInfo::for_cpp();
        eh.add_landing_pad(LandingPad::catch_all(0x40));
        eh.add_landing_pad(LandingPad::cleanup(0x80));

        assert!(eh.has_landing_pads());
        assert_eq!(eh.landing_pads.len(), 2);
        assert!(eh.has_catch_handlers());
        assert!(eh.has_cleanup_handlers());
    }

    #[test]
    fn eh_info_only_catch_no_cleanup() {
        let mut eh = ExceptionHandlingInfo::for_cpp();
        eh.add_landing_pad(LandingPad::catch_typed(0x40, 1));

        assert!(eh.has_catch_handlers());
        assert!(!eh.has_cleanup_handlers());
    }

    #[test]
    fn eh_info_only_cleanup_no_catch() {
        let mut eh = ExceptionHandlingInfo::for_rust();
        eh.add_landing_pad(LandingPad::cleanup(0x40));

        assert!(!eh.has_catch_handlers());
        assert!(eh.has_cleanup_handlers());
    }

    #[test]
    fn eh_info_add_call_site() {
        let mut eh = ExceptionHandlingInfo::for_cpp();
        eh.add_call_site(CallSiteEntry {
            start_offset: 0x10,
            length: 0x20,
            landing_pad_offset: 0x40,
            action_index: 1,
        });
        assert_eq!(eh.call_site_count(), 1);
    }

    #[test]
    fn eh_info_add_action() {
        let mut eh = ExceptionHandlingInfo::for_cpp();
        eh.add_action(ActionEntry { type_filter: 1, next_offset: 0 });
        eh.add_action(ActionEntry { type_filter: 0, next_offset: 0 });
        assert_eq!(eh.action_count(), 2);
    }

    #[test]
    fn eh_info_add_type() {
        let mut eh = ExceptionHandlingInfo::for_cpp();
        eh.add_type(TypeTableEntry {
            type_info_symbol: "_ZTIi".to_string(),
            encoding: TypeTableEncoding::PcRelSData4,
        });
        assert_eq!(eh.type_count(), 1);
    }

    #[test]
    fn eh_info_personality_symbol() {
        let eh = ExceptionHandlingInfo::for_cpp();
        assert_eq!(eh.personality_symbol(), "__gxx_personality_v0");

        let eh_rust = ExceptionHandlingInfo::for_rust();
        assert_eq!(eh_rust.personality_symbol(), "__rust_eh_personality");
    }

    #[test]
    fn eh_info_validate_empty_is_valid() {
        let eh = ExceptionHandlingInfo::for_cpp();
        assert!(eh.validate().is_empty());
    }

    #[test]
    fn eh_info_validate_consistent_info() {
        let mut eh = ExceptionHandlingInfo::for_cpp();

        // Add matching landing pad, call site, action, and type
        eh.add_landing_pad(LandingPad::catch_typed(0x40, 1));
        eh.add_call_site(CallSiteEntry {
            start_offset: 0x10,
            length: 0x20,
            landing_pad_offset: 0x40,
            action_index: 1,
        });
        eh.add_action(ActionEntry { type_filter: 1, next_offset: 0 });
        eh.add_type(TypeTableEntry {
            type_info_symbol: "_ZTIi".to_string(),
            encoding: TypeTableEncoding::PcRelSData4,
        });

        let errors = eh.validate();
        assert!(errors.is_empty(), "Expected no errors, got: {:?}", errors);
    }

    #[test]
    fn eh_info_validate_detects_missing_landing_pad() {
        let mut eh = ExceptionHandlingInfo::for_cpp();
        // Call site references landing pad 0x40 but no landing pad registered
        eh.add_call_site(CallSiteEntry {
            start_offset: 0x10,
            length: 0x20,
            landing_pad_offset: 0x40,
            action_index: 0,
        });

        let errors = eh.validate();
        assert_eq!(errors.len(), 1);
        assert!(errors[0].contains("landing pad at offset 0x40"));
    }

    #[test]
    fn eh_info_validate_detects_out_of_range_action_index() {
        let mut eh = ExceptionHandlingInfo::for_cpp();
        eh.add_landing_pad(LandingPad::catch_all(0x40));
        // Call site references action 5 but no actions exist
        eh.add_call_site(CallSiteEntry {
            start_offset: 0x10,
            length: 0x20,
            landing_pad_offset: 0x40,
            action_index: 5,
        });

        let errors = eh.validate();
        assert_eq!(errors.len(), 1);
        assert!(errors[0].contains("action_index 5"));
    }

    #[test]
    fn eh_info_validate_detects_out_of_range_type_filter() {
        let mut eh = ExceptionHandlingInfo::for_cpp();
        // Action references type 3 but type table is empty
        eh.add_action(ActionEntry { type_filter: 3, next_offset: 0 });

        let errors = eh.validate();
        assert_eq!(errors.len(), 1);
        assert!(errors[0].contains("type_filter 3"));
    }

    #[test]
    fn eh_info_validate_no_landing_pad_offset_zero_is_ok() {
        let mut eh = ExceptionHandlingInfo::for_cpp();
        // Call site with landing_pad_offset=0 (no handler) is valid
        eh.add_call_site(CallSiteEntry {
            start_offset: 0x10,
            length: 0x20,
            landing_pad_offset: 0,
            action_index: 0,
        });

        let errors = eh.validate();
        assert!(errors.is_empty());
    }

    #[test]
    fn eh_info_equality() {
        let eh1 = ExceptionHandlingInfo::for_cpp();
        let eh2 = ExceptionHandlingInfo::for_cpp();
        assert_eq!(eh1, eh2);
    }

    #[test]
    fn eh_info_clone() {
        let mut eh = ExceptionHandlingInfo::for_cpp();
        eh.add_landing_pad(LandingPad::catch_all(0x40));
        let eh2 = eh.clone();
        assert_eq!(eh, eh2);
    }

    // --- Full C++ exception handling scenario ---

    #[test]
    fn full_cpp_try_catch_scenario() {
        // Simulate: void foo() { try { bar(); } catch(int) { ... } catch(...) { ... } }
        let mut eh = ExceptionHandlingInfo::for_cpp();

        // Type table: index 1 = int
        eh.add_type(TypeTableEntry {
            type_info_symbol: "_ZTIi".to_string(),
            encoding: TypeTableEncoding::PcRelSData4,
        });

        // Landing pad for catch(int) at offset 0x40
        eh.add_landing_pad(LandingPad::catch_typed(0x40, 1));

        // Landing pad for catch(...) at offset 0x60
        eh.add_landing_pad(LandingPad::catch_all(0x60));

        // Action chain: action 1 catches int, chains to action 2 (catch-all)
        eh.add_action(ActionEntry { type_filter: 1, next_offset: 4 });
        eh.add_action(ActionEntry { type_filter: 0, next_offset: 0 }); // catch-all = cleanup

        // Call site covering the try region
        eh.add_call_site(CallSiteEntry {
            start_offset: 0x10,
            length: 0x20,
            landing_pad_offset: 0x40,
            action_index: 1,
        });

        // Assertions
        assert!(eh.has_landing_pads());
        assert!(eh.has_catch_handlers());
        assert_eq!(eh.call_site_count(), 1);
        assert_eq!(eh.action_count(), 2);
        assert_eq!(eh.type_count(), 1);
        assert_eq!(eh.personality_symbol(), "__gxx_personality_v0");

        // Validate internal consistency
        let errors = eh.validate();
        assert!(errors.is_empty(), "Expected valid, got: {:?}", errors);
    }

    // --- Full Rust panic unwinding scenario ---

    #[test]
    fn full_rust_panic_unwind_scenario() {
        // Simulate: fn foo() { let _guard = Guard::new(); bar(); }
        // Rust uses cleanup-only landing pads (drop glue), no type-based catch.
        let mut eh = ExceptionHandlingInfo::for_rust();

        // Cleanup landing pad for Guard's destructor
        eh.add_landing_pad(LandingPad::cleanup(0x30));

        // Call site for bar() call
        eh.add_call_site(CallSiteEntry {
            start_offset: 0x10,
            length: 0x08,
            landing_pad_offset: 0x30,
            action_index: 1,
        });

        // Action: cleanup only (type_filter = 0)
        eh.add_action(ActionEntry { type_filter: 0, next_offset: 0 });

        assert!(eh.has_landing_pads());
        assert!(!eh.has_catch_handlers());
        assert!(eh.has_cleanup_handlers());
        assert_eq!(eh.personality_symbol(), "__rust_eh_personality");

        let errors = eh.validate();
        assert!(errors.is_empty(), "Expected valid, got: {:?}", errors);
    }

    // --- SourceLanguage tests ---

    #[test]
    fn source_language_equality() {
        assert_eq!(SourceLanguage::Cpp, SourceLanguage::Cpp);
        assert_ne!(SourceLanguage::Cpp, SourceLanguage::Rust);
        assert_ne!(SourceLanguage::C, SourceLanguage::ObjectiveC);
    }

    #[test]
    fn source_language_clone() {
        let lang = SourceLanguage::Rust;
        let lang2 = lang;
        assert_eq!(lang, lang2);
    }

    // ===================================================================
    // Compact unwind personality/LSDA propagation tests
    // ===================================================================

    #[test]
    fn compact_unwind_propagates_personality_flag() {
        let mut info = UnwindInfo::standard_frame(
            &[(gpr::X19, gpr::X20, false)],
            32,
            false,
        );
        info.personality = Some("__gxx_personality_v0".to_string());
        info.has_lsda = false;
        let entry = generate_compact_unwind(&info, 0, 128);

        // personality flag should be 1 (has personality)
        assert_eq!(entry.personality, 1);
        // lsda flag should be 0 (no LSDA)
        assert_eq!(entry.lsda, 0);
    }

    #[test]
    fn compact_unwind_propagates_lsda_flag() {
        let mut info = UnwindInfo::standard_frame(
            &[(gpr::X19, gpr::X20, false)],
            32,
            false,
        );
        info.personality = Some("__gxx_personality_v0".to_string());
        info.has_lsda = true;
        let entry = generate_compact_unwind(&info, 0, 128);

        assert_eq!(entry.personality, 1);
        assert_eq!(entry.lsda, 1);
    }

    #[test]
    fn compact_unwind_no_personality_no_lsda() {
        let info = UnwindInfo::leaf_frame();
        let entry = generate_compact_unwind(&info, 0, 64);

        assert_eq!(entry.personality, 0);
        assert_eq!(entry.lsda, 0);
    }

    #[test]
    fn compact_unwind_dwarf_fallback_propagates_personality() {
        // Dynamic alloc forces DWARF fallback, but personality/LSDA
        // flags should still be propagated.
        let mut info = UnwindInfo::leaf_frame();
        info.has_dynamic_alloc = true;
        info.personality = Some("__rust_eh_personality".to_string());
        info.has_lsda = true;
        let entry = generate_compact_unwind(&info, 0, 32);

        assert!(entry.needs_dwarf_fallback());
        assert_eq!(entry.personality, 1);
        assert_eq!(entry.lsda, 1);
    }

    #[test]
    fn compact_unwind_frameless_dwarf_fallback_propagates_personality() {
        let info = UnwindInfo {
            saved_registers: vec![],
            frame_size: 32,
            has_frame_pointer: false,
            is_leaf: true,
            has_dynamic_alloc: false,
            personality: Some("__objc_personality_v0".to_string()),
            has_lsda: true,
        };
        let entry = generate_compact_unwind(&info, 0, 64);

        assert!(entry.needs_dwarf_fallback());
        assert_eq!(entry.personality, 1);
        assert_eq!(entry.lsda, 1);
    }
}
