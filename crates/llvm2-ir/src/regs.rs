// llvm2-ir - Shared machine IR model
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Copyright 2026 Andrew Yates | License: Apache-2.0

//! Register types — unified re-exports from the comprehensive `aarch64_regs` module.
//!
//! **This module is the public API for register types.** All register definitions
//! (PReg, RegClass, CondCode, ShiftType, ExtendType) live in `aarch64_regs`
//! and are re-exported here for backward compatibility.
//!
//! PReg encoding: see [`crate::aarch64_regs`] for the full encoding scheme.
//! GPR64: 0-30 = X0-X30, 31 = SP. GPR32: 32-62 = W0-W30, 63 = WSP.
//! FPR128: 64-95 = V0-V31. FPR64: 96-127 = D0-D31. FPR32: 128-159 = S0-S31.

// Re-export canonical types from aarch64_regs.
pub use crate::aarch64_regs::{
    // Core types
    PReg, RegClass, CondCode, ShiftType, ExtendType,
    // GPR64 constants (X registers)
    X0, X1, X2, X3, X4, X5, X6, X7, X8, X9, X10, X11, X12, X13, X14, X15,
    X16, X17, X18, X19, X20, X21, X22, X23, X24, X25, X26, X27, X28, X29, X30,
    SP, FP, LR,
    // GPR32 constants (W registers)
    W0, W1, W2, W3, W4, W5, W6, W7, W8, W9, W10, W11, W12, W13, W14, W15,
    W16, W17, W18, W19, W20, W21, W22, W23, W24, W25, W26, W27, W28, W29, W30,
    WSP,
    // FPR128 constants (V registers)
    V0, V1, V2, V3, V4, V5, V6, V7, V8, V9, V10, V11, V12, V13, V14, V15,
    V16, V17, V18, V19, V20, V21, V22, V23, V24, V25, V26, V27, V28, V29, V30, V31,
    // FPR64 constants (D registers)
    D0, D1, D2, D3, D4, D5, D6, D7, D8, D9, D10, D11, D12, D13, D14, D15,
    D16, D17, D18, D19, D20, D21, D22, D23, D24, D25, D26, D27, D28, D29, D30, D31,
    // FPR32 constants (S registers)
    S0, S1, S2, S3, S4, S5, S6, S7, S8, S9, S10, S11, S12, S13, S14, S15,
    S16, S17, S18, S19, S20, S21, S22, S23, S24, S25, S26, S27, S28, S29, S30, S31,
    // FPR16 constants (H registers)
    H0, H1, H2, H3, H4, H5, H6, H7, H8, H9, H10, H11, H12, H13, H14, H15,
    H16, H17, H18, H19, H20, H21, H22, H23, H24, H25, H26, H27, H28, H29, H30, H31,
    // FPR8 constants (B registers)
    B0, B1, B2, B3, B4, B5, B6, B7, B8, B9, B10, B11, B12, B13, B14, B15,
    B16, B17, B18, B19, B20, B21, B22, B23, B24, B25, B26, B27, B28, B29, B30, B31,
    // Special registers
    XZR, WZR, NZCV, FPCR, FPSR,
    // Register class arrays
    ALL_GPRS, ALLOCATABLE_GPRS, CALLEE_SAVED_GPRS, CALLER_SAVED_GPRS,
    CALL_CLOBBER_GPRS, ARG_GPRS, RET_GPRS, TEMP_GPRS,
    ALL_FPRS, ALLOCATABLE_FPRS, CALLEE_SAVED_FPRS, CALLER_SAVED_FPRS,
    ARG_FPRS, RET_FPRS,
    // Helper functions
    preg_name, preg_class, hw_encoding, is_callee_saved, is_caller_saved,
    gpr64_to_gpr32, gpr32_to_gpr64,
    fpr128_to_fpr64, fpr128_to_fpr32, fpr128_to_fpr16, fpr128_to_fpr8,
    fpr64_to_fpr128, fpr32_to_fpr128,
    reg_number, regs_overlap,
    // CondCode aliases
    CC, CS,
};

/// Virtual register — SSA value before register allocation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct VReg {
    pub id: u32,
    pub class: RegClass,
}

impl VReg {
    pub fn new(id: u32, class: RegClass) -> Self {
        Self { id, class }
    }
}

impl core::fmt::Display for VReg {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "v{}", self.id)
    }
}

impl RegClass {
    /// Select the register class for a given IR type.
    ///
    /// For aggregate types (struct/array), returns Gpr64 since aggregates are
    /// passed by pointer or decomposed into scalar loads/stores.
    pub fn for_type(ty: &crate::function::Type) -> Self {
        use crate::function::Type;
        match ty {
            Type::I8 | Type::I16 | Type::I32 | Type::B1 => RegClass::Gpr32,
            Type::I64 | Type::Ptr | Type::I128 => RegClass::Gpr64,
            Type::F32 => RegClass::Fpr32,
            Type::F64 => RegClass::Fpr64,
            // Aggregates are handled via pointers at the machine level.
            Type::Struct(_) | Type::Array(_, _) => RegClass::Gpr64,
        }
    }
}

/// Special AArch64 registers that are not allocatable.
///
/// This enum provides a way to reference SP/XZR/WZR as operands.
/// For the physical register constants themselves, use the PReg constants
/// directly (e.g., `regs::SP`, `regs::XZR`, `regs::WZR`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SpecialReg {
    /// Stack pointer (encoded as register 31 in some instructions).
    SP,
    /// 64-bit zero register (encoded as register 31 in other instructions).
    XZR,
    /// 32-bit zero register.
    WZR,
}
