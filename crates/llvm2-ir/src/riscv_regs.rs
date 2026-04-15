// llvm2-ir - RISC-V register model
//
// Author: Andrew Yates <ayates@dropbox.com>
// Copyright 2026 Dropbox, Inc. | License: Apache-2.0
//
// Reference: RISC-V Unprivileged ISA Specification (Volume 1, Version 20191213)
// Reference: RISC-V ELF psABI Specification (calling convention)

//! RISC-V RV64GC register definitions.
//!
//! Encoding scheme for [`RiscVPReg`]:
//! - `0..=31`   GPR (x0-x31, 64-bit integer registers)
//! - `32..=63`  FPR (f0-f31, 64-bit double-precision floating-point)
//!
//! Register numbering within each class matches the RISC-V hardware encoding
//! (x0=0, x1=1, ..., x31=31 for GPR; f0=0, f1=1, ..., f31=31 for FPR).
//!
//! ABI names:
//! - x0=zero, x1=ra, x2=sp, x3=gp, x4=tp
//! - x5-x7=t0-t2, x8=s0/fp, x9=s1
//! - x10-x17=a0-a7, x18-x27=s2-s11, x28-x31=t3-t6
//! - f0-f7=ft0-ft7, f8-f9=fs0-fs1, f10-f17=fa0-fa7
//! - f18-f27=fs2-fs11, f28-f31=ft8-ft11

// ---------------------------------------------------------------------------
// RiscVPReg -- Physical Register
// ---------------------------------------------------------------------------

/// A physical RISC-V register identified by a unique encoding index.
///
/// The encoding is internal to LLVM2 and organizes registers by class.
/// Use [`riscv_hw_encoding`] to get the 5-bit field value that appears
/// in machine instructions.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct RiscVPReg(u16);

impl RiscVPReg {
    /// Create a RiscVPReg from a raw encoding value.
    #[inline]
    pub const fn new(encoding: u16) -> Self {
        Self(encoding)
    }

    /// Return the raw encoding value.
    #[inline]
    pub const fn encoding(self) -> u16 {
        self.0
    }

    /// Returns true if this is a general-purpose register (x0-x31).
    #[inline]
    pub const fn is_gpr(self) -> bool {
        self.0 <= 31
    }

    /// Returns true if this is a floating-point register (f0-f31).
    #[inline]
    pub const fn is_fpr(self) -> bool {
        self.0 >= 32 && self.0 <= 63
    }

    /// Returns true if this register is allocatable for register allocation.
    ///
    /// Excludes:
    /// - x0/zero (hardwired zero, reads as 0, writes are discarded)
    /// - x2/sp (stack pointer)
    /// - x3/gp (global pointer, reserved by linker)
    /// - x4/tp (thread pointer, reserved by OS)
    ///
    /// All FPR registers are allocatable.
    #[inline]
    pub fn is_allocatable(self) -> bool {
        if self.is_gpr() {
            let hw = self.hw_enc();
            hw != 0 && hw != 2 && hw != 3 && hw != 4
        } else {
            self.is_fpr()
        }
    }

    /// Returns the 5-bit hardware encoding for this register.
    ///
    /// This is the value that appears in instruction register fields.
    #[inline]
    pub fn hw_enc(self) -> u8 {
        riscv_hw_encoding(self)
    }

    /// Returns the register class for this register.
    #[inline]
    pub fn reg_class(self) -> RiscVRegClass {
        riscv_preg_class(self)
    }
}

impl core::fmt::Debug for RiscVPReg {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "RiscVPReg({}:{})", self.0, riscv_preg_name(*self))
    }
}

impl core::fmt::Display for RiscVPReg {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str(riscv_preg_name(*self))
    }
}

// ---------------------------------------------------------------------------
// RiscV Register class
// ---------------------------------------------------------------------------

/// Register class for a RISC-V physical register.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum RiscVRegClass {
    /// 64-bit general-purpose (x0-x31 on RV64)
    Gpr,
    /// 64-bit double-precision floating-point (f0-f31)
    Fpr64,
}

impl RiscVRegClass {
    /// Size in bits of registers in this class.
    #[inline]
    pub const fn size_bits(self) -> u32 {
        match self {
            Self::Gpr => 64,
            Self::Fpr64 => 64,
        }
    }

    /// Size in bytes of registers in this class.
    #[inline]
    pub const fn size_bytes(self) -> u32 {
        self.size_bits() / 8
    }
}

// ===========================================================================
// GPR: x0-x31 (encodings 0-31)
// ===========================================================================
// Numeric names (x0-x31) match the hardware register encoding directly.

pub const X0:  RiscVPReg = RiscVPReg(0);
pub const X1:  RiscVPReg = RiscVPReg(1);
pub const X2:  RiscVPReg = RiscVPReg(2);
pub const X3:  RiscVPReg = RiscVPReg(3);
pub const X4:  RiscVPReg = RiscVPReg(4);
pub const X5:  RiscVPReg = RiscVPReg(5);
pub const X6:  RiscVPReg = RiscVPReg(6);
pub const X7:  RiscVPReg = RiscVPReg(7);
pub const X8:  RiscVPReg = RiscVPReg(8);
pub const X9:  RiscVPReg = RiscVPReg(9);
pub const X10: RiscVPReg = RiscVPReg(10);
pub const X11: RiscVPReg = RiscVPReg(11);
pub const X12: RiscVPReg = RiscVPReg(12);
pub const X13: RiscVPReg = RiscVPReg(13);
pub const X14: RiscVPReg = RiscVPReg(14);
pub const X15: RiscVPReg = RiscVPReg(15);
pub const X16: RiscVPReg = RiscVPReg(16);
pub const X17: RiscVPReg = RiscVPReg(17);
pub const X18: RiscVPReg = RiscVPReg(18);
pub const X19: RiscVPReg = RiscVPReg(19);
pub const X20: RiscVPReg = RiscVPReg(20);
pub const X21: RiscVPReg = RiscVPReg(21);
pub const X22: RiscVPReg = RiscVPReg(22);
pub const X23: RiscVPReg = RiscVPReg(23);
pub const X24: RiscVPReg = RiscVPReg(24);
pub const X25: RiscVPReg = RiscVPReg(25);
pub const X26: RiscVPReg = RiscVPReg(26);
pub const X27: RiscVPReg = RiscVPReg(27);
pub const X28: RiscVPReg = RiscVPReg(28);
pub const X29: RiscVPReg = RiscVPReg(29);
pub const X30: RiscVPReg = RiscVPReg(30);
pub const X31: RiscVPReg = RiscVPReg(31);

// ABI name aliases for integer registers
/// x0: hardwired zero
pub const ZERO: RiscVPReg = X0;
/// x1: return address
pub const RA:   RiscVPReg = X1;
/// x2: stack pointer
pub const SP:   RiscVPReg = X2;
/// x3: global pointer
pub const GP:   RiscVPReg = X3;
/// x4: thread pointer
pub const TP:   RiscVPReg = X4;
/// x5: temporary 0
pub const T0:   RiscVPReg = X5;
/// x6: temporary 1
pub const T1:   RiscVPReg = X6;
/// x7: temporary 2
pub const T2:   RiscVPReg = X7;
/// x8: saved register 0 / frame pointer
pub const S0:   RiscVPReg = X8;
/// x8: frame pointer (alias for s0)
pub const FP:   RiscVPReg = X8;
/// x9: saved register 1
pub const S1:   RiscVPReg = X9;
/// x10: argument 0 / return value 0
pub const A0:   RiscVPReg = X10;
/// x11: argument 1 / return value 1
pub const A1:   RiscVPReg = X11;
/// x12: argument 2
pub const A2:   RiscVPReg = X12;
/// x13: argument 3
pub const A3:   RiscVPReg = X13;
/// x14: argument 4
pub const A4:   RiscVPReg = X14;
/// x15: argument 5
pub const A5:   RiscVPReg = X15;
/// x16: argument 6
pub const A6:   RiscVPReg = X16;
/// x17: argument 7
pub const A7:   RiscVPReg = X17;
/// x18: saved register 2
pub const S2:   RiscVPReg = X18;
/// x19: saved register 3
pub const S3:   RiscVPReg = X19;
/// x20: saved register 4
pub const S4:   RiscVPReg = X20;
/// x21: saved register 5
pub const S5:   RiscVPReg = X21;
/// x22: saved register 6
pub const S6:   RiscVPReg = X22;
/// x23: saved register 7
pub const S7:   RiscVPReg = X23;
/// x24: saved register 8
pub const S8:   RiscVPReg = X24;
/// x25: saved register 9
pub const S9:   RiscVPReg = X25;
/// x26: saved register 10
pub const S10:  RiscVPReg = X26;
/// x27: saved register 11
pub const S11:  RiscVPReg = X27;
/// x28: temporary 3
pub const T3:   RiscVPReg = X28;
/// x29: temporary 4
pub const T4:   RiscVPReg = X29;
/// x30: temporary 5
pub const T5:   RiscVPReg = X30;
/// x31: temporary 6
pub const T6:   RiscVPReg = X31;

// ===========================================================================
// FPR: f0-f31 (encodings 32-63)
// ===========================================================================

pub const F0:  RiscVPReg = RiscVPReg(32);
pub const F1:  RiscVPReg = RiscVPReg(33);
pub const F2:  RiscVPReg = RiscVPReg(34);
pub const F3:  RiscVPReg = RiscVPReg(35);
pub const F4:  RiscVPReg = RiscVPReg(36);
pub const F5:  RiscVPReg = RiscVPReg(37);
pub const F6:  RiscVPReg = RiscVPReg(38);
pub const F7:  RiscVPReg = RiscVPReg(39);
pub const F8:  RiscVPReg = RiscVPReg(40);
pub const F9:  RiscVPReg = RiscVPReg(41);
pub const F10: RiscVPReg = RiscVPReg(42);
pub const F11: RiscVPReg = RiscVPReg(43);
pub const F12: RiscVPReg = RiscVPReg(44);
pub const F13: RiscVPReg = RiscVPReg(45);
pub const F14: RiscVPReg = RiscVPReg(46);
pub const F15: RiscVPReg = RiscVPReg(47);
pub const F16: RiscVPReg = RiscVPReg(48);
pub const F17: RiscVPReg = RiscVPReg(49);
pub const F18: RiscVPReg = RiscVPReg(50);
pub const F19: RiscVPReg = RiscVPReg(51);
pub const F20: RiscVPReg = RiscVPReg(52);
pub const F21: RiscVPReg = RiscVPReg(53);
pub const F22: RiscVPReg = RiscVPReg(54);
pub const F23: RiscVPReg = RiscVPReg(55);
pub const F24: RiscVPReg = RiscVPReg(56);
pub const F25: RiscVPReg = RiscVPReg(57);
pub const F26: RiscVPReg = RiscVPReg(58);
pub const F27: RiscVPReg = RiscVPReg(59);
pub const F28: RiscVPReg = RiscVPReg(60);
pub const F29: RiscVPReg = RiscVPReg(61);
pub const F30: RiscVPReg = RiscVPReg(62);
pub const F31: RiscVPReg = RiscVPReg(63);

// ABI name aliases for floating-point registers
/// f0: FP temporary 0
pub const FT0:  RiscVPReg = F0;
/// f1: FP temporary 1
pub const FT1:  RiscVPReg = F1;
/// f2: FP temporary 2
pub const FT2:  RiscVPReg = F2;
/// f3: FP temporary 3
pub const FT3:  RiscVPReg = F3;
/// f4: FP temporary 4
pub const FT4:  RiscVPReg = F4;
/// f5: FP temporary 5
pub const FT5:  RiscVPReg = F5;
/// f6: FP temporary 6
pub const FT6:  RiscVPReg = F6;
/// f7: FP temporary 7
pub const FT7:  RiscVPReg = F7;
/// f8: FP saved register 0
pub const FS0:  RiscVPReg = F8;
/// f9: FP saved register 1
pub const FS1:  RiscVPReg = F9;
/// f10: FP argument 0 / return value 0
pub const FA0:  RiscVPReg = F10;
/// f11: FP argument 1 / return value 1
pub const FA1:  RiscVPReg = F11;
/// f12: FP argument 2
pub const FA2:  RiscVPReg = F12;
/// f13: FP argument 3
pub const FA3:  RiscVPReg = F13;
/// f14: FP argument 4
pub const FA4:  RiscVPReg = F14;
/// f15: FP argument 5
pub const FA5:  RiscVPReg = F15;
/// f16: FP argument 6
pub const FA6:  RiscVPReg = F16;
/// f17: FP argument 7
pub const FA7:  RiscVPReg = F17;
/// f18: FP saved register 2
pub const FS2:  RiscVPReg = F18;
/// f19: FP saved register 3
pub const FS3:  RiscVPReg = F19;
/// f20: FP saved register 4
pub const FS4:  RiscVPReg = F20;
/// f21: FP saved register 5
pub const FS5:  RiscVPReg = F21;
/// f22: FP saved register 6
pub const FS6:  RiscVPReg = F22;
/// f23: FP saved register 7
pub const FS7:  RiscVPReg = F23;
/// f24: FP saved register 8
pub const FS8:  RiscVPReg = F24;
/// f25: FP saved register 9
pub const FS9:  RiscVPReg = F25;
/// f26: FP saved register 10
pub const FS10: RiscVPReg = F26;
/// f27: FP saved register 11
pub const FS11: RiscVPReg = F27;
/// f28: FP temporary 8
pub const FT8:  RiscVPReg = F28;
/// f29: FP temporary 9
pub const FT9:  RiscVPReg = F29;
/// f30: FP temporary 10
pub const FT10: RiscVPReg = F30;
/// f31: FP temporary 11
pub const FT11: RiscVPReg = F31;

// ===========================================================================
// Register class arrays -- RISC-V calling convention
// ===========================================================================
// Reference: RISC-V ELF psABI Specification, Chapter 1 (Register Convention)

/// All 32 integer GPRs (x0-x31).
pub const RISCV_ALL_GPRS: [RiscVPReg; 32] = [
    X0,  X1,  X2,  X3,  X4,  X5,  X6,  X7,
    X8,  X9,  X10, X11, X12, X13, X14, X15,
    X16, X17, X18, X19, X20, X21, X22, X23,
    X24, X25, X26, X27, X28, X29, X30, X31,
];

/// Allocatable GPRs for register allocation.
///
/// Excludes:
/// - x0/zero (hardwired zero)
/// - x2/sp (stack pointer)
/// - x3/gp (global pointer, reserved by linker)
/// - x4/tp (thread pointer, reserved by OS)
pub const RISCV_ALLOCATABLE_GPRS: [RiscVPReg; 28] = [
    X1,                                         // ra
    X5,  X6,  X7,                               // t0-t2
    X8,  X9,                                    // s0-s1
    X10, X11, X12, X13, X14, X15, X16, X17,    // a0-a7
    X18, X19, X20, X21, X22, X23, X24, X25,    // s2-s9
    X26, X27,                                   // s10-s11
    X28, X29, X30, X31,                         // t3-t6
];

/// Callee-saved (non-volatile) GPRs per RISC-V calling convention.
///
/// A called function must save and restore these if it uses them.
/// s0-s11 (x8-x9, x18-x27)
pub const RISCV_CALLEE_SAVED_GPRS: [RiscVPReg; 12] = [
    S0, S1,
    S2, S3, S4, S5, S6, S7, S8, S9, S10, S11,
];

/// Caller-saved (volatile) GPRs per RISC-V calling convention.
///
/// These are clobbered by function calls. The caller must save them if needed.
/// ra, t0-t6, a0-a7
pub const RISCV_CALLER_SAVED_GPRS: [RiscVPReg; 16] = [
    RA,
    T0, T1, T2,
    A0, A1, A2, A3, A4, A5, A6, A7,
    T3, T4, T5, T6,
];

/// Argument-passing GPRs (RISC-V calling convention order).
///
/// Integer arguments are passed in: a0-a7 (x10-x17)
pub const RISCV_ARG_GPRS: [RiscVPReg; 8] = [A0, A1, A2, A3, A4, A5, A6, A7];

/// Return-value GPRs. a0 for the first return value, a1 for the second.
pub const RISCV_RET_GPRS: [RiscVPReg; 2] = [A0, A1];

/// All 32 floating-point registers (f0-f31).
pub const RISCV_ALL_FPRS: [RiscVPReg; 32] = [
    F0,  F1,  F2,  F3,  F4,  F5,  F6,  F7,
    F8,  F9,  F10, F11, F12, F13, F14, F15,
    F16, F17, F18, F19, F20, F21, F22, F23,
    F24, F25, F26, F27, F28, F29, F30, F31,
];

/// Allocatable FPRs (all 32 are allocatable).
pub const RISCV_ALLOCATABLE_FPRS: [RiscVPReg; 32] = RISCV_ALL_FPRS;

/// Callee-saved FPRs per RISC-V calling convention.
///
/// fs0-fs11 (f8-f9, f18-f27)
pub const RISCV_CALLEE_SAVED_FPRS: [RiscVPReg; 12] = [
    FS0, FS1,
    FS2, FS3, FS4, FS5, FS6, FS7, FS8, FS9, FS10, FS11,
];

/// Caller-saved FPRs per RISC-V calling convention.
///
/// ft0-ft11, fa0-fa7
pub const RISCV_CALLER_SAVED_FPRS: [RiscVPReg; 20] = [
    FT0, FT1, FT2, FT3, FT4, FT5, FT6, FT7,
    FA0, FA1, FA2, FA3, FA4, FA5, FA6, FA7,
    FT8, FT9, FT10, FT11,
];

/// Argument-passing FPRs (RISC-V calling convention order).
///
/// Float/double arguments are passed in: fa0-fa7 (f10-f17)
pub const RISCV_ARG_FPRS: [RiscVPReg; 8] = [FA0, FA1, FA2, FA3, FA4, FA5, FA6, FA7];

/// Return-value FPRs. fa0 for the first FP return, fa1 for the second.
pub const RISCV_RET_FPRS: [RiscVPReg; 2] = [FA0, FA1];

/// Registers clobbered by a CALL instruction (caller-saved GPRs + FPRs).
///
/// Includes all caller-saved GPRs. SP is modified by the call frame but
/// restored by the callee, so it is not listed here.
pub const RISCV_CALL_CLOBBER_GPRS: [RiscVPReg; 16] = RISCV_CALLER_SAVED_GPRS;

// ===========================================================================
// Lookup tables for name -> register mapping
// ===========================================================================

/// GPR ABI names indexed by register number (0-31).
const GPR_ABI_NAMES: [&str; 32] = [
    "zero", "ra",  "sp",  "gp",  "tp",  "t0",  "t1",  "t2",
    "s0",   "s1",  "a0",  "a1",  "a2",  "a3",  "a4",  "a5",
    "a6",   "a7",  "s2",  "s3",  "s4",  "s5",  "s6",  "s7",
    "s8",   "s9",  "s10", "s11", "t3",  "t4",  "t5",  "t6",
];

/// FPR ABI names indexed by register number (0-31).
const FPR_ABI_NAMES: [&str; 32] = [
    "ft0",  "ft1",  "ft2",  "ft3",  "ft4",  "ft5",  "ft6",  "ft7",
    "fs0",  "fs1",  "fa0",  "fa1",  "fa2",  "fa3",  "fa4",  "fa5",
    "fa6",  "fa7",  "fs2",  "fs3",  "fs4",  "fs5",  "fs6",  "fs7",
    "fs8",  "fs9",  "fs10", "fs11", "ft8",  "ft9",  "ft10", "ft11",
];

// ===========================================================================
// Helper functions
// ===========================================================================

/// Return the assembly name (ABI name) for a RISC-V physical register.
///
/// Returns ABI names: "zero", "ra", "sp", "a0", "fa0", etc.
pub fn riscv_preg_name(reg: RiscVPReg) -> &'static str {
    let e = reg.encoding();
    match e {
        0..=31 => GPR_ABI_NAMES[e as usize],
        32..=63 => FPR_ABI_NAMES[(e - 32) as usize],
        _ => "<unknown>",
    }
}

/// Return the register class for a RISC-V physical register.
pub fn riscv_preg_class(reg: RiscVPReg) -> RiscVRegClass {
    let e = reg.encoding();
    match e {
        0..=31 => RiscVRegClass::Gpr,
        32..=63 => RiscVRegClass::Fpr64,
        _ => RiscVRegClass::Gpr, // fallback
    }
}

/// Return the 5-bit hardware encoding for a RISC-V register.
///
/// This is the value used in instruction register fields (rs1, rs2, rd).
/// Both GPR and FPR use 0-31 encoding in the instruction word.
pub fn riscv_hw_encoding(reg: RiscVPReg) -> u8 {
    let e = reg.encoding();
    match e {
        0..=31 => e as u8,           // GPR: x0=0 .. x31=31
        32..=63 => (e - 32) as u8,   // FPR: f0=0 .. f31=31
        _ => 0,
    }
}

/// Return `true` if the register is callee-saved (non-volatile) per RISC-V calling convention.
///
/// Callee-saved GPRs: s0-s11 (x8-x9, x18-x27)
/// Callee-saved FPRs: fs0-fs11 (f8-f9, f18-f27)
pub fn riscv_is_callee_saved(reg: RiscVPReg) -> bool {
    let e = reg.encoding();
    match e {
        // GPR: s0=x8, s1=x9
        8..=9 => true,
        // GPR: s2-s11 = x18-x27
        18..=27 => true,
        // FPR: fs0=f8, fs1=f9 (encodings 40-41)
        40..=41 => true,
        // FPR: fs2-fs11 = f18-f27 (encodings 50-59)
        50..=59 => true,
        _ => false,
    }
}

/// Return `true` if the register is caller-saved (volatile) per RISC-V calling convention.
///
/// Caller-saved GPRs: ra (x1), t0-t6 (x5-x7, x28-x31), a0-a7 (x10-x17)
/// Caller-saved FPRs: ft0-ft11 (f0-f7, f28-f31), fa0-fa7 (f10-f17)
pub fn riscv_is_caller_saved(reg: RiscVPReg) -> bool {
    let e = reg.encoding();
    match e {
        // GPR: ra=x1
        1 => true,
        // GPR: t0-t2 = x5-x7
        5..=7 => true,
        // GPR: a0-a7 = x10-x17
        10..=17 => true,
        // GPR: t3-t6 = x28-x31
        28..=31 => true,
        // FPR: ft0-ft7 = f0-f7 (encodings 32-39)
        32..=39 => true,
        // FPR: fa0-fa7 = f10-f17 (encodings 42-49)
        42..=49 => true,
        // FPR: ft8-ft11 = f28-f31 (encodings 60-63)
        60..=63 => true,
        _ => false,
    }
}

/// Try to parse a RISC-V register from its ABI name or numeric name.
///
/// Accepts ABI names: `"zero"`, `"ra"`, `"sp"`, `"a0"`, `"fa0"`, `"ft0"`, etc.
/// Also accepts numeric names: `"x0"`, `"x31"`, `"f0"`, `"f31"`.
///
/// Returns `None` if the name is not recognized.
pub fn riscv_preg_from_name(name: &str) -> Option<RiscVPReg> {
    // Try ABI names for GPR
    for (i, &n) in GPR_ABI_NAMES.iter().enumerate() {
        if n == name {
            return Some(RiscVPReg(i as u16));
        }
    }
    // Also accept "fp" as alias for s0
    if name == "fp" {
        return Some(FP);
    }
    // Try ABI names for FPR
    for (i, &n) in FPR_ABI_NAMES.iter().enumerate() {
        if n == name {
            return Some(RiscVPReg(32 + i as u16));
        }
    }
    // Try numeric GPR names: x0-x31
    if let Some(rest) = name.strip_prefix('x')
        && let Ok(num) = rest.parse::<u16>()
            && num <= 31 {
                return Some(RiscVPReg(num));
            }
    // Try numeric FPR names: f0-f31
    if let Some(rest) = name.strip_prefix('f') {
        // Avoid matching ABI names like "fa0", "ft0", "fs0"
        if let Ok(num) = rest.parse::<u16>()
            && num <= 31 {
                return Some(RiscVPReg(32 + num));
            }
    }
    None
}

/// Return the register number (0-31) for any register.
///
/// For GPR: the x-register number (0-31).
/// For FPR: the f-register number (0-31).
pub fn riscv_reg_number(reg: RiscVPReg) -> u8 {
    riscv_hw_encoding(reg)
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_preg_names() {
        assert_eq!(riscv_preg_name(X0), "zero");
        assert_eq!(riscv_preg_name(X1), "ra");
        assert_eq!(riscv_preg_name(X2), "sp");
        assert_eq!(riscv_preg_name(X3), "gp");
        assert_eq!(riscv_preg_name(X4), "tp");
        assert_eq!(riscv_preg_name(X5), "t0");
        assert_eq!(riscv_preg_name(X8), "s0");
        assert_eq!(riscv_preg_name(X10), "a0");
        assert_eq!(riscv_preg_name(X17), "a7");
        assert_eq!(riscv_preg_name(X18), "s2");
        assert_eq!(riscv_preg_name(X27), "s11");
        assert_eq!(riscv_preg_name(X28), "t3");
        assert_eq!(riscv_preg_name(X31), "t6");
        assert_eq!(riscv_preg_name(F0), "ft0");
        assert_eq!(riscv_preg_name(F8), "fs0");
        assert_eq!(riscv_preg_name(F10), "fa0");
        assert_eq!(riscv_preg_name(F17), "fa7");
        assert_eq!(riscv_preg_name(F18), "fs2");
        assert_eq!(riscv_preg_name(F27), "fs11");
        assert_eq!(riscv_preg_name(F28), "ft8");
        assert_eq!(riscv_preg_name(F31), "ft11");
    }

    #[test]
    fn test_abi_aliases() {
        // Verify ABI alias constants match the numeric constants
        assert_eq!(ZERO, X0);
        assert_eq!(RA, X1);
        assert_eq!(SP, X2);
        assert_eq!(GP, X3);
        assert_eq!(TP, X4);
        assert_eq!(T0, X5);
        assert_eq!(T1, X6);
        assert_eq!(T2, X7);
        assert_eq!(S0, X8);
        assert_eq!(FP, X8);
        assert_eq!(S1, X9);
        assert_eq!(A0, X10);
        assert_eq!(A7, X17);
        assert_eq!(S2, X18);
        assert_eq!(S11, X27);
        assert_eq!(T3, X28);
        assert_eq!(T6, X31);

        assert_eq!(FT0, F0);
        assert_eq!(FT7, F7);
        assert_eq!(FS0, F8);
        assert_eq!(FS1, F9);
        assert_eq!(FA0, F10);
        assert_eq!(FA7, F17);
        assert_eq!(FS2, F18);
        assert_eq!(FS11, F27);
        assert_eq!(FT8, F28);
        assert_eq!(FT11, F31);
    }

    #[test]
    fn test_preg_classes() {
        assert_eq!(riscv_preg_class(X0), RiscVRegClass::Gpr);
        assert_eq!(riscv_preg_class(X31), RiscVRegClass::Gpr);
        assert_eq!(riscv_preg_class(F0), RiscVRegClass::Fpr64);
        assert_eq!(riscv_preg_class(F31), RiscVRegClass::Fpr64);
    }

    #[test]
    fn test_hw_encoding() {
        assert_eq!(riscv_hw_encoding(X0), 0);
        assert_eq!(riscv_hw_encoding(X1), 1);
        assert_eq!(riscv_hw_encoding(X31), 31);
        assert_eq!(riscv_hw_encoding(F0), 0);
        assert_eq!(riscv_hw_encoding(F1), 1);
        assert_eq!(riscv_hw_encoding(F31), 31);
    }

    #[test]
    fn test_callee_saved() {
        // GPR callee-saved: s0-s11 (x8-x9, x18-x27)
        assert!(riscv_is_callee_saved(S0));
        assert!(riscv_is_callee_saved(S1));
        assert!(riscv_is_callee_saved(S2));
        assert!(riscv_is_callee_saved(S11));

        // NOT callee-saved
        assert!(!riscv_is_callee_saved(ZERO));
        assert!(!riscv_is_callee_saved(RA));
        assert!(!riscv_is_callee_saved(SP));
        assert!(!riscv_is_callee_saved(T0));
        assert!(!riscv_is_callee_saved(A0));
        assert!(!riscv_is_callee_saved(T6));

        // FPR callee-saved: fs0-fs11 (f8-f9, f18-f27)
        assert!(riscv_is_callee_saved(FS0));
        assert!(riscv_is_callee_saved(FS1));
        assert!(riscv_is_callee_saved(FS2));
        assert!(riscv_is_callee_saved(FS11));

        // FPR NOT callee-saved
        assert!(!riscv_is_callee_saved(FT0));
        assert!(!riscv_is_callee_saved(FA0));
        assert!(!riscv_is_callee_saved(FT11));
    }

    #[test]
    fn test_caller_saved() {
        // GPR caller-saved: ra, t0-t6, a0-a7
        assert!(riscv_is_caller_saved(RA));
        assert!(riscv_is_caller_saved(T0));
        assert!(riscv_is_caller_saved(T2));
        assert!(riscv_is_caller_saved(A0));
        assert!(riscv_is_caller_saved(A7));
        assert!(riscv_is_caller_saved(T3));
        assert!(riscv_is_caller_saved(T6));

        // NOT caller-saved
        assert!(!riscv_is_caller_saved(ZERO));
        assert!(!riscv_is_caller_saved(SP));
        assert!(!riscv_is_caller_saved(GP));
        assert!(!riscv_is_caller_saved(TP));
        assert!(!riscv_is_caller_saved(S0));
        assert!(!riscv_is_caller_saved(S11));

        // FPR caller-saved: ft0-ft11, fa0-fa7
        assert!(riscv_is_caller_saved(FT0));
        assert!(riscv_is_caller_saved(FT7));
        assert!(riscv_is_caller_saved(FA0));
        assert!(riscv_is_caller_saved(FA7));
        assert!(riscv_is_caller_saved(FT8));
        assert!(riscv_is_caller_saved(FT11));

        // FPR NOT caller-saved
        assert!(!riscv_is_caller_saved(FS0));
        assert!(!riscv_is_caller_saved(FS11));
    }

    #[test]
    fn test_is_predicates() {
        assert!(X0.is_gpr());
        assert!(X31.is_gpr());
        assert!(!X0.is_fpr());

        assert!(F0.is_fpr());
        assert!(F31.is_fpr());
        assert!(!F0.is_gpr());
    }

    #[test]
    fn test_register_class_sizes() {
        assert_eq!(RiscVRegClass::Gpr.size_bits(), 64);
        assert_eq!(RiscVRegClass::Fpr64.size_bits(), 64);
        assert_eq!(RiscVRegClass::Gpr.size_bytes(), 8);
        assert_eq!(RiscVRegClass::Fpr64.size_bytes(), 8);
    }

    #[test]
    fn test_allocatable_gprs_exclude_reserved() {
        assert!(!RISCV_ALLOCATABLE_GPRS.contains(&ZERO)); // x0/zero
        assert!(!RISCV_ALLOCATABLE_GPRS.contains(&SP));   // x2/sp
        assert!(!RISCV_ALLOCATABLE_GPRS.contains(&GP));   // x3/gp
        assert!(!RISCV_ALLOCATABLE_GPRS.contains(&TP));   // x4/tp
        assert!(RISCV_ALLOCATABLE_GPRS.contains(&RA));    // x1/ra is allocatable
        assert!(RISCV_ALLOCATABLE_GPRS.contains(&A0));
        assert!(RISCV_ALLOCATABLE_GPRS.contains(&S0));
        assert!(RISCV_ALLOCATABLE_GPRS.contains(&T6));
    }

    #[test]
    fn test_is_allocatable() {
        // Non-allocatable: zero, sp, gp, tp
        assert!(!ZERO.is_allocatable());
        assert!(!SP.is_allocatable());
        assert!(!GP.is_allocatable());
        assert!(!TP.is_allocatable());

        // Allocatable GPRs
        assert!(RA.is_allocatable());
        assert!(T0.is_allocatable());
        assert!(S0.is_allocatable());
        assert!(A0.is_allocatable());
        assert!(T6.is_allocatable());

        // All FPRs are allocatable
        assert!(F0.is_allocatable());
        assert!(F31.is_allocatable());
    }

    #[test]
    fn test_arg_gprs_order() {
        assert_eq!(RISCV_ARG_GPRS[0], A0);
        assert_eq!(RISCV_ARG_GPRS[1], A1);
        assert_eq!(RISCV_ARG_GPRS[2], A2);
        assert_eq!(RISCV_ARG_GPRS[3], A3);
        assert_eq!(RISCV_ARG_GPRS[4], A4);
        assert_eq!(RISCV_ARG_GPRS[5], A5);
        assert_eq!(RISCV_ARG_GPRS[6], A6);
        assert_eq!(RISCV_ARG_GPRS[7], A7);
    }

    #[test]
    fn test_arg_fprs_order() {
        assert_eq!(RISCV_ARG_FPRS[0], FA0);
        assert_eq!(RISCV_ARG_FPRS[1], FA1);
        assert_eq!(RISCV_ARG_FPRS[7], FA7);
    }

    #[test]
    fn test_display() {
        assert_eq!(format!("{}", ZERO), "zero");
        assert_eq!(format!("{}", RA), "ra");
        assert_eq!(format!("{}", SP), "sp");
        assert_eq!(format!("{}", A0), "a0");
        assert_eq!(format!("{}", FA0), "fa0");
        assert_eq!(format!("{}", FT11), "ft11");
    }

    #[test]
    fn test_preg_from_name() {
        // ABI names
        assert_eq!(riscv_preg_from_name("zero"), Some(ZERO));
        assert_eq!(riscv_preg_from_name("ra"), Some(RA));
        assert_eq!(riscv_preg_from_name("sp"), Some(SP));
        assert_eq!(riscv_preg_from_name("gp"), Some(GP));
        assert_eq!(riscv_preg_from_name("tp"), Some(TP));
        assert_eq!(riscv_preg_from_name("t0"), Some(T0));
        assert_eq!(riscv_preg_from_name("s0"), Some(S0));
        assert_eq!(riscv_preg_from_name("fp"), Some(FP));
        assert_eq!(riscv_preg_from_name("a0"), Some(A0));
        assert_eq!(riscv_preg_from_name("a7"), Some(A7));
        assert_eq!(riscv_preg_from_name("s11"), Some(S11));
        assert_eq!(riscv_preg_from_name("t6"), Some(T6));

        // FPR ABI names
        assert_eq!(riscv_preg_from_name("ft0"), Some(FT0));
        assert_eq!(riscv_preg_from_name("fs0"), Some(FS0));
        assert_eq!(riscv_preg_from_name("fa0"), Some(FA0));
        assert_eq!(riscv_preg_from_name("fa7"), Some(FA7));
        assert_eq!(riscv_preg_from_name("fs11"), Some(FS11));
        assert_eq!(riscv_preg_from_name("ft11"), Some(FT11));

        // Numeric names
        assert_eq!(riscv_preg_from_name("x0"), Some(X0));
        assert_eq!(riscv_preg_from_name("x31"), Some(X31));
        assert_eq!(riscv_preg_from_name("f0"), Some(F0));
        assert_eq!(riscv_preg_from_name("f31"), Some(F31));

        // Unknown
        assert_eq!(riscv_preg_from_name("v0"), None);
        assert_eq!(riscv_preg_from_name(""), None);
        assert_eq!(riscv_preg_from_name("x32"), None);
        assert_eq!(riscv_preg_from_name("f32"), None);
    }

    #[test]
    fn test_caller_callee_partition_gprs() {
        // For each allocatable GPR, it should be either caller-saved or callee-saved
        // (but not both and not neither).
        for &gpr in &RISCV_ALLOCATABLE_GPRS {
            let caller = riscv_is_caller_saved(gpr);
            let callee = riscv_is_callee_saved(gpr);
            assert!(
                caller ^ callee,
                "{:?} is caller_saved={}, callee_saved={} (expected exactly one)",
                gpr, caller, callee
            );
        }
    }

    #[test]
    fn test_caller_callee_partition_fprs() {
        // For each FPR, it should be either caller-saved or callee-saved.
        for &fpr in &RISCV_ALL_FPRS {
            let caller = riscv_is_caller_saved(fpr);
            let callee = riscv_is_callee_saved(fpr);
            assert!(
                caller ^ callee,
                "{:?} is caller_saved={}, callee_saved={} (expected exactly one)",
                fpr, caller, callee
            );
        }
    }

    #[test]
    fn test_hw_encoding_consistency() {
        // All GPR encodings should be 0-31
        for i in 0..32u16 {
            let reg = RiscVPReg(i);
            assert_eq!(riscv_hw_encoding(reg), i as u8);
        }
        // All FPR encodings should be 0-31
        for i in 0..32u16 {
            let reg = RiscVPReg(32 + i);
            assert_eq!(riscv_hw_encoding(reg), i as u8);
        }
    }

    #[test]
    fn test_reg_class_method_matches_function() {
        let test_regs = [X0, X31, A0, S0, RA, SP, F0, F31, FA0, FS0];
        for &reg in &test_regs {
            assert_eq!(reg.reg_class(), riscv_preg_class(reg), "mismatch for {:?}", reg);
        }
    }

    #[test]
    fn test_callee_saved_array_consistency() {
        // Every register in callee-saved GPR list should pass is_callee_saved
        for &reg in &RISCV_CALLEE_SAVED_GPRS {
            assert!(riscv_is_callee_saved(reg), "{:?} in array but not callee-saved", reg);
        }
        for &reg in &RISCV_CALLEE_SAVED_FPRS {
            assert!(riscv_is_callee_saved(reg), "{:?} in array but not callee-saved", reg);
        }
    }

    #[test]
    fn test_caller_saved_array_consistency() {
        // Every register in caller-saved GPR list should pass is_caller_saved
        for &reg in &RISCV_CALLER_SAVED_GPRS {
            assert!(riscv_is_caller_saved(reg), "{:?} in array but not caller-saved", reg);
        }
        for &reg in &RISCV_CALLER_SAVED_FPRS {
            assert!(riscv_is_caller_saved(reg), "{:?} in array but not caller-saved", reg);
        }
    }

    #[test]
    fn test_register_counts() {
        assert_eq!(RISCV_ALL_GPRS.len(), 32);
        assert_eq!(RISCV_ALLOCATABLE_GPRS.len(), 28);
        assert_eq!(RISCV_CALLEE_SAVED_GPRS.len(), 12);
        assert_eq!(RISCV_CALLER_SAVED_GPRS.len(), 16);
        assert_eq!(RISCV_ARG_GPRS.len(), 8);
        assert_eq!(RISCV_RET_GPRS.len(), 2);

        assert_eq!(RISCV_ALL_FPRS.len(), 32);
        assert_eq!(RISCV_ALLOCATABLE_FPRS.len(), 32);
        assert_eq!(RISCV_CALLEE_SAVED_FPRS.len(), 12);
        assert_eq!(RISCV_CALLER_SAVED_FPRS.len(), 20);
        assert_eq!(RISCV_ARG_FPRS.len(), 8);
        assert_eq!(RISCV_RET_FPRS.len(), 2);
    }
}
