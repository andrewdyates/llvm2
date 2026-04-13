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
    /// Classify function parameters into register/stack locations.
    ///
    /// Rules (Apple arm64, non-variadic):
    /// - Integer/pointer types (I8, I16, I32, I64, B1): next available X0-X7
    /// - Float types (F32, F64): next available V0-V7
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

                // Aggregate types: pass by pointer (indirect) if > 16 bytes,
                // otherwise pack into register(s).
                Type::Struct(_) | Type::Array(_, _) => {
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

        result
    }

    /// Classify return values.
    ///
    /// Rules:
    /// - Integer: X0 (single), X0+X1 (pair)
    /// - Float: V0 (single), V0+V1 (pair)
    /// - Large aggregate returns: indirect via X8 pointer (sret)
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

                // Aggregate returns: indirect via X8 (sret convention)
                Type::Struct(_) | Type::Array(_, _) => {
                    result.push(ArgLocation::Indirect { ptr_reg: gpr::X8 });
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
}
