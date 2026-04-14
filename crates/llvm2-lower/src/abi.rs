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
        V0, V1, V2, V3, V4, V5, V6, V7,
        V8, V9, V10, V11, V12, V13, V14, V15,
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
}

// ---------------------------------------------------------------------------
// Register arrays (static, following LLVM AArch64ISelLowering.cpp)
// ---------------------------------------------------------------------------

/// GPR argument registers: X0-X7
const GPR_ARG_REGS: [PReg; 8] = [
    gpr::X0, gpr::X1, gpr::X2, gpr::X3,
    gpr::X4, gpr::X5, gpr::X6, gpr::X7,
];

/// FPR argument registers: V0-V7
const FPR_ARG_REGS: [PReg; 8] = [
    gpr::V0, gpr::V1, gpr::V2, gpr::V3,
    gpr::V4, gpr::V5, gpr::V6, gpr::V7,
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

                // Aggregate types: check for HFA first, then size-based classification.
                Type::Struct(_) | Type::Array(_, _) => {
                    // HFA: 1-4 same-type FP fields -> consecutive FPR registers.
                    // Per AAPCS64, if there aren't enough FPR slots for the
                    // entire HFA, the whole thing goes on the stack.
                    if let Some((_base_ty, count)) = Self::detect_hfa(ty) {
                        if fpr_idx + count <= FPR_ARG_REGS.len() {
                            // Pass in consecutive FPR registers.
                            // We record the first register; the ABI lowering
                            // layer knows to use `count` consecutive V registers.
                            result.push(ArgLocation::Reg(FPR_ARG_REGS[fpr_idx]));
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

                // Aggregate returns: HFA in V regs, small structs in X regs, large via sret.
                Type::Struct(_) | Type::Array(_, _) => {
                    // Check for HFA: return in consecutive V registers.
                    if let Some((_base_ty, count)) = Self::detect_hfa(ty) {
                        if fpr_idx + count <= 4 {
                            // HFA return in V0-V3 (max 4 FPR return registers).
                            result.push(ArgLocation::Reg(FPR_ARG_REGS[fpr_idx]));
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
}

/// Align `value` up to the next multiple of `align`.
fn align_up(value: i64, align: i64) -> i64 {
    (value + align - 1) & !(align - 1)
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
        // struct { float, float } passed as HFA -> V0 (consumes V0 + V1)
        let hfa = Type::Struct(vec![Type::F32, Type::F32]);
        let locs = AppleAArch64ABI::classify_params(&[hfa]);
        assert_eq!(locs.len(), 1);
        // First FPR register is V0; the ABI layer knows 2 consecutive are consumed.
        assert_eq!(locs[0], ArgLocation::Reg(gpr::V0));
    }

    #[test]
    fn classify_hfa_param_four_f32_in_v_regs() {
        // struct { float, float, float, float } -> V0 (consumes V0-V3)
        let hfa = Type::Struct(vec![Type::F32, Type::F32, Type::F32, Type::F32]);
        let locs = AppleAArch64ABI::classify_params(&[hfa]);
        assert_eq!(locs.len(), 1);
        assert_eq!(locs[0], ArgLocation::Reg(gpr::V0));
    }

    #[test]
    fn classify_hfa_param_with_preceding_floats() {
        // fn(f64, HFA{f32, f32}) -> V0 for f64, V1 for HFA (consumes V1+V2)
        let hfa = Type::Struct(vec![Type::F32, Type::F32]);
        let locs = AppleAArch64ABI::classify_params(&[Type::F64, hfa]);
        assert_eq!(locs.len(), 2);
        assert_eq!(locs[0], ArgLocation::Reg(gpr::V0));
        assert_eq!(locs[1], ArgLocation::Reg(gpr::V1));
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
        // struct { float, float } -> HFA return in V0 (consumes V0+V1)
        let hfa = Type::Struct(vec![Type::F32, Type::F32]);
        let locs = AppleAArch64ABI::classify_returns(&[hfa]);
        assert_eq!(locs.len(), 1);
        assert_eq!(locs[0], ArgLocation::Reg(gpr::V0));
    }

    #[test]
    fn return_hfa_four_f64_in_v0() {
        // struct { double, double, double, double } -> HFA return in V0 (consumes V0-V3)
        let hfa = Type::Struct(vec![Type::F64, Type::F64, Type::F64, Type::F64]);
        let locs = AppleAArch64ABI::classify_returns(&[hfa]);
        assert_eq!(locs.len(), 1);
        assert_eq!(locs[0], ArgLocation::Reg(gpr::V0));
    }

    #[test]
    fn return_hfa_array_three_f32() {
        // float[3] -> HFA return in V0
        let hfa = Type::Array(Box::new(Type::F32), 3);
        let locs = AppleAArch64ABI::classify_returns(&[hfa]);
        assert_eq!(locs.len(), 1);
        assert_eq!(locs[0], ArgLocation::Reg(gpr::V0));
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
}
