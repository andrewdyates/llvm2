// llvm2-ir - Shared machine IR model
// Author: Andrew Yates <ayates@dropbox.com>
// Copyright 2026 Dropbox, Inc. | License: Apache-2.0

//! AArch64 register model.
//!
//! PReg encoding: 0-30 = GPR (X0-X30), 32-63 = FPR (V0-V31).
//! Apple AArch64 calling convention register allocation rules are
//! encoded via the ALLOCATABLE_GPRS, CALLEE_SAVED_GPRS, and
//! ALLOCATABLE_FPRS arrays.

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

/// Physical register — AArch64 hardware register.
///
/// Encoding: 0-30 = GPR (X0-X30), 32-63 = FPR (V0-V31).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PReg(pub u8);

impl PReg {
    /// Returns true if this is a general-purpose register (X0-X30).
    pub fn is_gpr(&self) -> bool {
        self.0 < 32
    }

    /// Returns true if this is a floating-point/SIMD register (V0-V31).
    pub fn is_fpr(&self) -> bool {
        self.0 >= 32 && self.0 < 64
    }

    /// Returns the hardware register number (0-30 for GPR, 0-31 for FPR).
    pub fn hw_enc(&self) -> u8 {
        if self.is_gpr() {
            self.0
        } else {
            self.0 - 32
        }
    }

    /// Alias for `hw_enc()` — returns the register number within its class.
    pub fn hw_index(&self) -> u8 {
        self.hw_enc()
    }
}

impl core::fmt::Display for PReg {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        if self.is_gpr() {
            match self.0 {
                29 => write!(f, "fp"),
                30 => write!(f, "lr"),
                n => write!(f, "x{n}"),
            }
        } else if self.is_fpr() {
            write!(f, "v{}", self.0 - 32)
        } else {
            write!(f, "?{}", self.0)
        }
    }
}

// GPR constants (X0-X30)
pub const X0: PReg = PReg(0);
pub const X1: PReg = PReg(1);
pub const X2: PReg = PReg(2);
pub const X3: PReg = PReg(3);
pub const X4: PReg = PReg(4);
pub const X5: PReg = PReg(5);
pub const X6: PReg = PReg(6);
pub const X7: PReg = PReg(7);
pub const X8: PReg = PReg(8);
pub const X9: PReg = PReg(9);
pub const X10: PReg = PReg(10);
pub const X11: PReg = PReg(11);
pub const X12: PReg = PReg(12);
pub const X13: PReg = PReg(13);
pub const X14: PReg = PReg(14);
pub const X15: PReg = PReg(15);
pub const X16: PReg = PReg(16);
pub const X17: PReg = PReg(17);
pub const X18: PReg = PReg(18);
pub const X19: PReg = PReg(19);
pub const X20: PReg = PReg(20);
pub const X21: PReg = PReg(21);
pub const X22: PReg = PReg(22);
pub const X23: PReg = PReg(23);
pub const X24: PReg = PReg(24);
pub const X25: PReg = PReg(25);
pub const X26: PReg = PReg(26);
pub const X27: PReg = PReg(27);
pub const X28: PReg = PReg(28);
pub const X29: PReg = PReg(29);
pub const X30: PReg = PReg(30);

// FPR/SIMD constants (V0-V31)
pub const V0: PReg = PReg(32);
pub const V1: PReg = PReg(33);
pub const V2: PReg = PReg(34);
pub const V3: PReg = PReg(35);
pub const V4: PReg = PReg(36);
pub const V5: PReg = PReg(37);
pub const V6: PReg = PReg(38);
pub const V7: PReg = PReg(39);
pub const V8: PReg = PReg(40);
pub const V9: PReg = PReg(41);
pub const V10: PReg = PReg(42);
pub const V11: PReg = PReg(43);
pub const V12: PReg = PReg(44);
pub const V13: PReg = PReg(45);
pub const V14: PReg = PReg(46);
pub const V15: PReg = PReg(47);
pub const V16: PReg = PReg(48);
pub const V17: PReg = PReg(49);
pub const V18: PReg = PReg(50);
pub const V19: PReg = PReg(51);
pub const V20: PReg = PReg(52);
pub const V21: PReg = PReg(53);
pub const V22: PReg = PReg(54);
pub const V23: PReg = PReg(55);
pub const V24: PReg = PReg(56);
pub const V25: PReg = PReg(57);
pub const V26: PReg = PReg(58);
pub const V27: PReg = PReg(59);
pub const V28: PReg = PReg(60);
pub const V29: PReg = PReg(61);
pub const V30: PReg = PReg(62);
pub const V31: PReg = PReg(63);

/// Register class — determines which physical register file a value lives in.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RegClass {
    /// 32-bit general-purpose (W0-W30)
    Gpr32,
    /// 64-bit general-purpose (X0-X30)
    Gpr64,
    /// 32-bit floating-point (S0-S31)
    Fpr32,
    /// 64-bit floating-point (D0-D31)
    Fpr64,
    /// 128-bit SIMD vector (V0-V31)
    Vec128,
}

/// Special AArch64 registers that are not allocatable.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SpecialReg {
    /// Stack pointer (encoded as register 31 in some instructions).
    SP,
    /// 64-bit zero register (encoded as register 31 in other instructions).
    XZR,
    /// 32-bit zero register.
    WZR,
}

/// Allocatable GPRs: X0-X15, X19-X28.
///
/// Excludes: X16-X17 (IP scratch, reserved by linker), X18 (reserved on Apple),
/// X29 (frame pointer, mandatory on Darwin), X30 (link register).
pub const ALLOCATABLE_GPRS: &[PReg] = &[
    X0, X1, X2, X3, X4, X5, X6, X7, X8, X9, X10, X11, X12, X13, X14, X15,
    X19, X20, X21, X22, X23, X24, X25, X26, X27, X28,
];

/// Callee-saved GPRs: X19-X28.
pub const CALLEE_SAVED_GPRS: &[PReg] = &[
    X19, X20, X21, X22, X23, X24, X25, X26, X27, X28,
];

/// All FPR/SIMD registers are allocatable: V0-V31.
pub const ALLOCATABLE_FPRS: &[PReg] = &[
    V0, V1, V2, V3, V4, V5, V6, V7, V8, V9, V10, V11, V12, V13, V14, V15,
    V16, V17, V18, V19, V20, V21, V22, V23, V24, V25, V26, V27, V28, V29, V30, V31,
];

/// Callee-saved FPRs: V8-V15 (lower 64 bits only on Apple AArch64).
pub const CALLEE_SAVED_FPRS: &[PReg] = &[
    V8, V9, V10, V11, V12, V13, V14, V15,
];
