// llvm2-ir - AArch64 register model
//
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Copyright 2026 Andrew Yates | License: Apache-2.0
//
// Reference: llvm-project-ref/llvm/lib/Target/AArch64/AArch64RegisterInfo.td
// Reference: ARM Architecture Reference Manual (ARMv8-A)

//! Comprehensive AArch64 register definitions.
//!
//! Encoding scheme for [`PReg`]:
//! - `0..=30`   GPR64  (X0-X30)
//! - `31`       SP
//! - `32..=62`  GPR32  (W0-W30, aliases lower 32 bits of X0-X30)
//! - `63`       WSP
//! - `64..=95`  FPR128 (V0-V31 / Q0-Q31)
//! - `96..=127` FPR64  (D0-D31, aliases lower 64 bits of V0-V31)
//! - `128..=159` FPR32 (S0-S31, aliases lower 32 bits of V0-V31)
//! - `160`      XZR
//! - `161`      WZR
//! - `162`      NZCV (condition flags)
//! - `163`      FPCR (floating-point control register)
//! - `164`      FPSR (floating-point status register)
//! - `165..=196` FPR16 (H0-H31, aliases lower 16 bits of V0-V31)
//! - `197..=228` FPR8  (B0-B31, aliases lower 8 bits of V0-V31)

// ---------------------------------------------------------------------------
// PReg — Physical Register
// ---------------------------------------------------------------------------

/// A physical register identified by a unique encoding index.
///
/// The encoding is internal to LLVM2 and does NOT match the ARM hardware
/// encoding directly. Use [`hw_encoding`] to get the 5-bit field value
/// that appears in machine instructions.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct PReg(u16);

impl PReg {
    /// Create a PReg from a raw encoding value.
    #[inline]
    pub const fn new(encoding: u16) -> Self {
        Self(encoding)
    }

    /// Return the raw encoding value.
    #[inline]
    pub const fn encoding(self) -> u16 {
        self.0
    }

    /// Returns true if this is a 64-bit general-purpose register (X0-X30 or SP).
    #[inline]
    pub const fn is_gpr(self) -> bool {
        self.0 <= 31
    }

    /// Returns true if this is a floating-point/SIMD register (any FPR class).
    #[inline]
    pub const fn is_fpr(self) -> bool {
        self.0 >= 64 && self.0 <= 228
    }

    /// Returns the 5-bit hardware encoding for this register.
    ///
    /// Alias for the free function [`hw_encoding`]. Provided as a method
    /// for backward compatibility.
    #[inline]
    pub fn hw_enc(self) -> u8 {
        hw_encoding(self)
    }

    /// Alias for `hw_enc()`.
    #[inline]
    pub fn hw_index(self) -> u8 {
        self.hw_enc()
    }
}

impl core::fmt::Debug for PReg {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "PReg({}:{})", self.0, preg_name(*self))
    }
}

impl core::fmt::Display for PReg {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str(preg_name(*self))
    }
}

// ---------------------------------------------------------------------------
// Register class
// ---------------------------------------------------------------------------

/// Broad register class for an AArch64 physical register.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum RegClass {
    /// 64-bit general-purpose (X0-X30, SP, XZR)
    Gpr64,
    /// 32-bit general-purpose (W0-W30, WSP, WZR) — aliases low 32 bits of GPR64
    Gpr32,
    /// 128-bit SIMD/FP (V0-V31 / Q0-Q31)
    Fpr128,
    /// 64-bit SIMD/FP (D0-D31) — aliases low 64 bits of FPR128
    Fpr64,
    /// 32-bit SIMD/FP (S0-S31) — aliases low 32 bits of FPR128
    Fpr32,
    /// 16-bit SIMD/FP (H0-H31) — aliases low 16 bits of FPR128
    Fpr16,
    /// 8-bit SIMD/FP (B0-B31) — aliases low 8 bits of FPR128
    Fpr8,
    /// System/special (NZCV, FPCR, FPSR)
    System,
}

impl RegClass {
    /// Size in bits of registers in this class.
    #[inline]
    pub const fn size_bits(self) -> u32 {
        match self {
            Self::Gpr64 => 64,
            Self::Gpr32 => 32,
            Self::Fpr128 => 128,
            Self::Fpr64 => 64,
            Self::Fpr32 => 32,
            Self::Fpr16 => 16,
            Self::Fpr8 => 8,
            Self::System => 32,
        }
    }

    /// Size in bytes of registers in this class.
    #[inline]
    pub const fn size_bytes(self) -> u32 {
        self.size_bits() / 8
    }
}

// ===========================================================================
// GPR64: X0-X30, SP  (encodings 0-31)
// ===========================================================================

pub const X0:  PReg = PReg(0);
pub const X1:  PReg = PReg(1);
pub const X2:  PReg = PReg(2);
pub const X3:  PReg = PReg(3);
pub const X4:  PReg = PReg(4);
pub const X5:  PReg = PReg(5);
pub const X6:  PReg = PReg(6);
pub const X7:  PReg = PReg(7);
pub const X8:  PReg = PReg(8);
pub const X9:  PReg = PReg(9);
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
/// Frame pointer (X29). Aliased as FP in LLVM.
pub const X29: PReg = PReg(29);
/// Link register (X30). Aliased as LR in LLVM.
pub const X30: PReg = PReg(30);
/// Stack pointer. HW encoding 31, but context-dependent (SP vs XZR).
pub const SP:  PReg = PReg(31);

/// Convenience alias: frame pointer = X29.
pub const FP: PReg = X29;
/// Convenience alias: link register = X30.
pub const LR: PReg = X30;

// ===========================================================================
// GPR32: W0-W30, WSP  (encodings 32-63)
// ===========================================================================

pub const W0:  PReg = PReg(32);
pub const W1:  PReg = PReg(33);
pub const W2:  PReg = PReg(34);
pub const W3:  PReg = PReg(35);
pub const W4:  PReg = PReg(36);
pub const W5:  PReg = PReg(37);
pub const W6:  PReg = PReg(38);
pub const W7:  PReg = PReg(39);
pub const W8:  PReg = PReg(40);
pub const W9:  PReg = PReg(41);
pub const W10: PReg = PReg(42);
pub const W11: PReg = PReg(43);
pub const W12: PReg = PReg(44);
pub const W13: PReg = PReg(45);
pub const W14: PReg = PReg(46);
pub const W15: PReg = PReg(47);
pub const W16: PReg = PReg(48);
pub const W17: PReg = PReg(49);
pub const W18: PReg = PReg(50);
pub const W19: PReg = PReg(51);
pub const W20: PReg = PReg(52);
pub const W21: PReg = PReg(53);
pub const W22: PReg = PReg(54);
pub const W23: PReg = PReg(55);
pub const W24: PReg = PReg(56);
pub const W25: PReg = PReg(57);
pub const W26: PReg = PReg(58);
pub const W27: PReg = PReg(59);
pub const W28: PReg = PReg(60);
pub const W29: PReg = PReg(61);
pub const W30: PReg = PReg(62);
/// 32-bit stack pointer alias.
pub const WSP: PReg = PReg(63);

// ===========================================================================
// FPR128 / NEON: V0-V31 (Q0-Q31)  (encodings 64-95)
// ===========================================================================

pub const V0:  PReg = PReg(64);
pub const V1:  PReg = PReg(65);
pub const V2:  PReg = PReg(66);
pub const V3:  PReg = PReg(67);
pub const V4:  PReg = PReg(68);
pub const V5:  PReg = PReg(69);
pub const V6:  PReg = PReg(70);
pub const V7:  PReg = PReg(71);
pub const V8:  PReg = PReg(72);
pub const V9:  PReg = PReg(73);
pub const V10: PReg = PReg(74);
pub const V11: PReg = PReg(75);
pub const V12: PReg = PReg(76);
pub const V13: PReg = PReg(77);
pub const V14: PReg = PReg(78);
pub const V15: PReg = PReg(79);
pub const V16: PReg = PReg(80);
pub const V17: PReg = PReg(81);
pub const V18: PReg = PReg(82);
pub const V19: PReg = PReg(83);
pub const V20: PReg = PReg(84);
pub const V21: PReg = PReg(85);
pub const V22: PReg = PReg(86);
pub const V23: PReg = PReg(87);
pub const V24: PReg = PReg(88);
pub const V25: PReg = PReg(89);
pub const V26: PReg = PReg(90);
pub const V27: PReg = PReg(91);
pub const V28: PReg = PReg(92);
pub const V29: PReg = PReg(93);
pub const V30: PReg = PReg(94);
pub const V31: PReg = PReg(95);

// ===========================================================================
// FPR64: D0-D31  (encodings 96-127)
// ===========================================================================

pub const D0:  PReg = PReg(96);
pub const D1:  PReg = PReg(97);
pub const D2:  PReg = PReg(98);
pub const D3:  PReg = PReg(99);
pub const D4:  PReg = PReg(100);
pub const D5:  PReg = PReg(101);
pub const D6:  PReg = PReg(102);
pub const D7:  PReg = PReg(103);
pub const D8:  PReg = PReg(104);
pub const D9:  PReg = PReg(105);
pub const D10: PReg = PReg(106);
pub const D11: PReg = PReg(107);
pub const D12: PReg = PReg(108);
pub const D13: PReg = PReg(109);
pub const D14: PReg = PReg(110);
pub const D15: PReg = PReg(111);
pub const D16: PReg = PReg(112);
pub const D17: PReg = PReg(113);
pub const D18: PReg = PReg(114);
pub const D19: PReg = PReg(115);
pub const D20: PReg = PReg(116);
pub const D21: PReg = PReg(117);
pub const D22: PReg = PReg(118);
pub const D23: PReg = PReg(119);
pub const D24: PReg = PReg(120);
pub const D25: PReg = PReg(121);
pub const D26: PReg = PReg(122);
pub const D27: PReg = PReg(123);
pub const D28: PReg = PReg(124);
pub const D29: PReg = PReg(125);
pub const D30: PReg = PReg(126);
pub const D31: PReg = PReg(127);

// ===========================================================================
// FPR32: S0-S31  (encodings 128-159)
// ===========================================================================

pub const S0:  PReg = PReg(128);
pub const S1:  PReg = PReg(129);
pub const S2:  PReg = PReg(130);
pub const S3:  PReg = PReg(131);
pub const S4:  PReg = PReg(132);
pub const S5:  PReg = PReg(133);
pub const S6:  PReg = PReg(134);
pub const S7:  PReg = PReg(135);
pub const S8:  PReg = PReg(136);
pub const S9:  PReg = PReg(137);
pub const S10: PReg = PReg(138);
pub const S11: PReg = PReg(139);
pub const S12: PReg = PReg(140);
pub const S13: PReg = PReg(141);
pub const S14: PReg = PReg(142);
pub const S15: PReg = PReg(143);
pub const S16: PReg = PReg(144);
pub const S17: PReg = PReg(145);
pub const S18: PReg = PReg(146);
pub const S19: PReg = PReg(147);
pub const S20: PReg = PReg(148);
pub const S21: PReg = PReg(149);
pub const S22: PReg = PReg(150);
pub const S23: PReg = PReg(151);
pub const S24: PReg = PReg(152);
pub const S25: PReg = PReg(153);
pub const S26: PReg = PReg(154);
pub const S27: PReg = PReg(155);
pub const S28: PReg = PReg(156);
pub const S29: PReg = PReg(157);
pub const S30: PReg = PReg(158);
pub const S31: PReg = PReg(159);

// ===========================================================================
// Special registers  (encodings 160-164)
// ===========================================================================

/// Zero register (64-bit). HW encoding 31, same as SP — context decides.
pub const XZR:  PReg = PReg(160);
/// Zero register (32-bit). HW encoding 31, same as WSP.
pub const WZR:  PReg = PReg(161);
/// NZCV condition flags register.
pub const NZCV: PReg = PReg(162);
/// Floating-point control register.
pub const FPCR: PReg = PReg(163);
/// Floating-point status register.
pub const FPSR: PReg = PReg(164);

// ===========================================================================
// FPR16: H0-H31  (encodings 165-196)
// ===========================================================================

pub const H0:  PReg = PReg(165);
pub const H1:  PReg = PReg(166);
pub const H2:  PReg = PReg(167);
pub const H3:  PReg = PReg(168);
pub const H4:  PReg = PReg(169);
pub const H5:  PReg = PReg(170);
pub const H6:  PReg = PReg(171);
pub const H7:  PReg = PReg(172);
pub const H8:  PReg = PReg(173);
pub const H9:  PReg = PReg(174);
pub const H10: PReg = PReg(175);
pub const H11: PReg = PReg(176);
pub const H12: PReg = PReg(177);
pub const H13: PReg = PReg(178);
pub const H14: PReg = PReg(179);
pub const H15: PReg = PReg(180);
pub const H16: PReg = PReg(181);
pub const H17: PReg = PReg(182);
pub const H18: PReg = PReg(183);
pub const H19: PReg = PReg(184);
pub const H20: PReg = PReg(185);
pub const H21: PReg = PReg(186);
pub const H22: PReg = PReg(187);
pub const H23: PReg = PReg(188);
pub const H24: PReg = PReg(189);
pub const H25: PReg = PReg(190);
pub const H26: PReg = PReg(191);
pub const H27: PReg = PReg(192);
pub const H28: PReg = PReg(193);
pub const H29: PReg = PReg(194);
pub const H30: PReg = PReg(195);
pub const H31: PReg = PReg(196);

// ===========================================================================
// FPR8: B0-B31  (encodings 197-228)
// ===========================================================================

pub const B0:  PReg = PReg(197);
pub const B1:  PReg = PReg(198);
pub const B2:  PReg = PReg(199);
pub const B3:  PReg = PReg(200);
pub const B4:  PReg = PReg(201);
pub const B5:  PReg = PReg(202);
pub const B6:  PReg = PReg(203);
pub const B7:  PReg = PReg(204);
pub const B8:  PReg = PReg(205);
pub const B9:  PReg = PReg(206);
pub const B10: PReg = PReg(207);
pub const B11: PReg = PReg(208);
pub const B12: PReg = PReg(209);
pub const B13: PReg = PReg(210);
pub const B14: PReg = PReg(211);
pub const B15: PReg = PReg(212);
pub const B16: PReg = PReg(213);
pub const B17: PReg = PReg(214);
pub const B18: PReg = PReg(215);
pub const B19: PReg = PReg(216);
pub const B20: PReg = PReg(217);
pub const B21: PReg = PReg(218);
pub const B22: PReg = PReg(219);
pub const B23: PReg = PReg(220);
pub const B24: PReg = PReg(221);
pub const B25: PReg = PReg(222);
pub const B26: PReg = PReg(223);
pub const B27: PReg = PReg(224);
pub const B28: PReg = PReg(225);
pub const B29: PReg = PReg(226);
pub const B30: PReg = PReg(227);
pub const B31: PReg = PReg(228);

// ===========================================================================
// Register class arrays — AAPCS64 calling convention
// ===========================================================================
// Reference: ARM Architecture Procedure Call Standard for AArch64 (AAPCS64)
// Reference: AArch64RegisterInfo.td lines 191-263 (register class definitions)

/// All 64-bit GPRs: X0-X30 (excludes SP/XZR).
pub const ALL_GPRS: [PReg; 31] = [
    X0, X1, X2, X3, X4, X5, X6, X7,
    X8, X9, X10, X11, X12, X13, X14, X15,
    X16, X17, X18, X19, X20, X21, X22, X23,
    X24, X25, X26, X27, X28, X29, X30,
];

/// Allocatable GPRs for register allocation.
///
/// Excludes:
/// - X8  (indirect result location / platform register)
/// - X16 (IP0, intra-procedure-call scratch / veneer)
/// - X17 (IP1, intra-procedure-call scratch / veneer)
/// - X18 (platform register, reserved on many OSes including Darwin/iOS)
/// - X29 (frame pointer)
/// - X30 (link register, preserved across calls but not allocatable)
pub const ALLOCATABLE_GPRS: [PReg; 25] = [
    X0, X1, X2, X3, X4, X5, X6, X7,
    X9, X10, X11, X12, X13, X14, X15,
    X19, X20, X21, X22, X23, X24, X25, X26, X27, X28,
];

/// Caller-saved (volatile) GPRs — clobbered by function calls.
///
/// These do NOT survive across a call. The caller must save them if needed.
pub const CALLER_SAVED_GPRS: [PReg; 15] = [
    X0, X1, X2, X3, X4, X5, X6, X7,
    X9, X10, X11, X12, X13, X14, X15,
];

/// Callee-saved (non-volatile) GPRs — preserved across calls.
///
/// A called function must save and restore these if it uses them.
pub const CALLEE_SAVED_GPRS: [PReg; 10] = [
    X19, X20, X21, X22, X23, X24, X25, X26, X27, X28,
];

/// Registers clobbered by a function call (superset of caller-saved).
///
/// Includes: argument regs, IP scratch regs, platform reg, LR.
/// Reference: AAPCS64 section 6.1.1
pub const CALL_CLOBBER_GPRS: [PReg; 20] = [
    X0, X1, X2, X3, X4, X5, X6, X7,
    X8, X9, X10, X11, X12, X13, X14, X15,
    X16, X17, X18, X30,
];

/// Argument-passing GPRs (X0-X7). AAPCS64 section 6.4.2.
pub const ARG_GPRS: [PReg; 8] = [X0, X1, X2, X3, X4, X5, X6, X7];

/// Return-value GPRs (X0-X7). AAPCS64 section 6.5.
pub const RET_GPRS: [PReg; 8] = [X0, X1, X2, X3, X4, X5, X6, X7];

/// Temporary / scratch GPRs: X9-X15 (caller-saved, not argument regs).
pub const TEMP_GPRS: [PReg; 7] = [X9, X10, X11, X12, X13, X14, X15];

// ---------------------------------------------------------------------------
// FPR class arrays
// ---------------------------------------------------------------------------

/// All 128-bit SIMD/FP registers.
pub const ALL_FPRS: [PReg; 32] = [
    V0,  V1,  V2,  V3,  V4,  V5,  V6,  V7,
    V8,  V9,  V10, V11, V12, V13, V14, V15,
    V16, V17, V18, V19, V20, V21, V22, V23,
    V24, V25, V26, V27, V28, V29, V30, V31,
];

/// Allocatable FPRs — all 32 vector registers are allocatable.
pub const ALLOCATABLE_FPRS: [PReg; 32] = ALL_FPRS;

/// Callee-saved FPR registers: V8-V15 (lower 64 bits only per AAPCS64).
///
/// AAPCS64 requires the callee to preserve the lower 64 bits (D8-D15)
/// of V8-V15. The upper 64 bits are caller-saved.
pub const CALLEE_SAVED_FPRS: [PReg; 8] = [V8, V9, V10, V11, V12, V13, V14, V15];

/// Caller-saved FPR registers: V0-V7 and V16-V31.
pub const CALLER_SAVED_FPRS: [PReg; 24] = [
    V0,  V1,  V2,  V3,  V4,  V5,  V6,  V7,
    V16, V17, V18, V19, V20, V21, V22, V23,
    V24, V25, V26, V27, V28, V29, V30, V31,
];

/// Argument-passing FPRs (V0-V7). AAPCS64 section 6.4.2.
pub const ARG_FPRS: [PReg; 8] = [V0, V1, V2, V3, V4, V5, V6, V7];

/// Return-value FPRs (V0-V7). AAPCS64 section 6.5.
pub const RET_FPRS: [PReg; 8] = [V0, V1, V2, V3, V4, V5, V6, V7];

// ===========================================================================
// Lookup tables for name → PReg mapping
// ===========================================================================

/// GPR64 names indexed by register number (0-30 = "x0"-"x30", 31 = "sp").
const GPR64_NAMES: [&str; 32] = [
    "x0",  "x1",  "x2",  "x3",  "x4",  "x5",  "x6",  "x7",
    "x8",  "x9",  "x10", "x11", "x12", "x13", "x14", "x15",
    "x16", "x17", "x18", "x19", "x20", "x21", "x22", "x23",
    "x24", "x25", "x26", "x27", "x28", "x29", "x30", "sp",
];

/// GPR32 names indexed by register number (0-30 = "w0"-"w30", 31 = "wsp").
const GPR32_NAMES: [&str; 32] = [
    "w0",  "w1",  "w2",  "w3",  "w4",  "w5",  "w6",  "w7",
    "w8",  "w9",  "w10", "w11", "w12", "w13", "w14", "w15",
    "w16", "w17", "w18", "w19", "w20", "w21", "w22", "w23",
    "w24", "w25", "w26", "w27", "w28", "w29", "w30", "wsp",
];

/// FPR128 (V/Q) names indexed by register number.
const FPR128_NAMES: [&str; 32] = [
    "v0",  "v1",  "v2",  "v3",  "v4",  "v5",  "v6",  "v7",
    "v8",  "v9",  "v10", "v11", "v12", "v13", "v14", "v15",
    "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23",
    "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31",
];

/// FPR64 (D) names indexed by register number.
const FPR64_NAMES: [&str; 32] = [
    "d0",  "d1",  "d2",  "d3",  "d4",  "d5",  "d6",  "d7",
    "d8",  "d9",  "d10", "d11", "d12", "d13", "d14", "d15",
    "d16", "d17", "d18", "d19", "d20", "d21", "d22", "d23",
    "d24", "d25", "d26", "d27", "d28", "d29", "d30", "d31",
];

/// FPR32 (S) names indexed by register number.
const FPR32_NAMES: [&str; 32] = [
    "s0",  "s1",  "s2",  "s3",  "s4",  "s5",  "s6",  "s7",
    "s8",  "s9",  "s10", "s11", "s12", "s13", "s14", "s15",
    "s16", "s17", "s18", "s19", "s20", "s21", "s22", "s23",
    "s24", "s25", "s26", "s27", "s28", "s29", "s30", "s31",
];

/// FPR16 (H) names indexed by register number.
const FPR16_NAMES: [&str; 32] = [
    "h0",  "h1",  "h2",  "h3",  "h4",  "h5",  "h6",  "h7",
    "h8",  "h9",  "h10", "h11", "h12", "h13", "h14", "h15",
    "h16", "h17", "h18", "h19", "h20", "h21", "h22", "h23",
    "h24", "h25", "h26", "h27", "h28", "h29", "h30", "h31",
];

/// FPR8 (B) names indexed by register number.
const FPR8_NAMES: [&str; 32] = [
    "b0",  "b1",  "b2",  "b3",  "b4",  "b5",  "b6",  "b7",
    "b8",  "b9",  "b10", "b11", "b12", "b13", "b14", "b15",
    "b16", "b17", "b18", "b19", "b20", "b21", "b22", "b23",
    "b24", "b25", "b26", "b27", "b28", "b29", "b30", "b31",
];

// ===========================================================================
// Helper functions
// ===========================================================================

/// Return the assembly name for a physical register (e.g., `"x0"`, `"v31"`, `"sp"`).
pub fn preg_name(reg: PReg) -> &'static str {
    let e = reg.encoding();
    match e {
        0..=31 => GPR64_NAMES[e as usize],
        32..=63 => GPR32_NAMES[(e - 32) as usize],
        64..=95 => FPR128_NAMES[(e - 64) as usize],
        96..=127 => FPR64_NAMES[(e - 96) as usize],
        128..=159 => FPR32_NAMES[(e - 128) as usize],
        160 => "xzr",
        161 => "wzr",
        162 => "nzcv",
        163 => "fpcr",
        164 => "fpsr",
        165..=196 => FPR16_NAMES[(e - 165) as usize],
        197..=228 => FPR8_NAMES[(e - 197) as usize],
        _ => "<unknown>",
    }
}

/// Return the register class for a physical register.
pub fn preg_class(reg: PReg) -> RegClass {
    let e = reg.encoding();
    match e {
        0..=31 => RegClass::Gpr64,
        32..=63 => RegClass::Gpr32,
        64..=95 => RegClass::Fpr128,
        96..=127 => RegClass::Fpr64,
        128..=159 => RegClass::Fpr32,
        160 => RegClass::Gpr64,   // XZR is a GPR64
        161 => RegClass::Gpr32,   // WZR is a GPR32
        162..=164 => RegClass::System,
        165..=196 => RegClass::Fpr16,
        197..=228 => RegClass::Fpr8,
        _ => RegClass::System,
    }
}

/// Return the 5-bit hardware encoding for a register.
///
/// This is the value that appears in the `Rd`, `Rn`, `Rm` fields of
/// AArch64 machine instructions. SP and XZR both encode as 31.
pub fn hw_encoding(reg: PReg) -> u8 {
    let e = reg.encoding();
    match e {
        // GPR64: X0-X30 encode as 0-30, SP encodes as 31
        0..=31 => e as u8,
        // GPR32: W0-W30 encode as 0-30, WSP encodes as 31
        32..=63 => (e - 32) as u8,
        // FPR128: V0-V31 encode as 0-31
        64..=95 => (e - 64) as u8,
        // FPR64: D0-D31 encode as 0-31
        96..=127 => (e - 96) as u8,
        // FPR32: S0-S31 encode as 0-31
        128..=159 => (e - 128) as u8,
        // XZR encodes as 31
        160 => 31,
        // WZR encodes as 31
        161 => 31,
        // System regs — not directly encodable in GPR/FPR fields
        162..=164 => 0,
        // FPR16: H0-H31 encode as 0-31
        165..=196 => (e - 165) as u8,
        // FPR8: B0-B31 encode as 0-31
        197..=228 => (e - 197) as u8,
        _ => 0,
    }
}

/// Return `true` if the register is callee-saved (non-volatile) per AAPCS64.
pub fn is_callee_saved(reg: PReg) -> bool {
    let e = reg.encoding();
    match e {
        // GPR64: X19-X28 are callee-saved
        19..=28 => true,
        // GPR32: W19-W28 are callee-saved (aliases of X19-X28)
        51..=60 => true,
        // FPR128: V8-V15 are callee-saved (lower 64 bits)
        72..=79 => true,
        // FPR64: D8-D15 are callee-saved
        104..=111 => true,
        // FPR32: S8-S15 are callee-saved (subset of V8-V15)
        136..=143 => true,
        _ => false,
    }
}

/// Return `true` if the register is caller-saved (volatile) per AAPCS64.
pub fn is_caller_saved(reg: PReg) -> bool {
    let e = reg.encoding();
    match e {
        // GPR64: X0-X15 (excluding X8 IP, X16-X17 are scratch but special)
        0..=7 | 9..=15 => true,
        // GPR32: corresponding W registers
        32..=39 | 41..=47 => true,
        // FPR128: V0-V7, V16-V31
        64..=71 | 80..=95 => true,
        // FPR64: D0-D7, D16-D31
        96..=103 | 112..=127 => true,
        // FPR32: S0-S7, S16-S31
        128..=135 | 144..=159 => true,
        _ => false,
    }
}

/// Convert a 64-bit GPR (X register) to its 32-bit alias (W register).
///
/// Returns `None` if the input is not a GPR64 (X0-X30).
/// SP→WSP is handled separately. XZR→WZR is handled separately.
pub fn gpr64_to_gpr32(reg: PReg) -> Option<PReg> {
    let e = reg.encoding();
    match e {
        // X0-X30 → W0-W30
        0..=30 => Some(PReg(e + 32)),
        // SP → WSP
        31 => Some(WSP),
        // XZR → WZR
        160 => Some(WZR),
        _ => None,
    }
}

/// Convert a 32-bit GPR (W register) to its 64-bit alias (X register).
///
/// Returns `None` if the input is not a GPR32 (W0-W30).
pub fn gpr32_to_gpr64(reg: PReg) -> Option<PReg> {
    let e = reg.encoding();
    match e {
        // W0-W30 → X0-X30
        32..=62 => Some(PReg(e - 32)),
        // WSP → SP
        63 => Some(SP),
        // WZR → XZR
        161 => Some(XZR),
        _ => None,
    }
}

/// Convert a 128-bit FPR (V register) to its 64-bit alias (D register).
pub fn fpr128_to_fpr64(reg: PReg) -> Option<PReg> {
    let e = reg.encoding();
    match e {
        64..=95 => Some(PReg(e + 32)),
        _ => None,
    }
}

/// Convert a 128-bit FPR (V register) to its 32-bit alias (S register).
pub fn fpr128_to_fpr32(reg: PReg) -> Option<PReg> {
    let e = reg.encoding();
    match e {
        64..=95 => Some(PReg(e + 64)),
        _ => None,
    }
}

/// Convert a 128-bit FPR (V register) to its 16-bit alias (H register).
pub fn fpr128_to_fpr16(reg: PReg) -> Option<PReg> {
    let e = reg.encoding();
    match e {
        64..=95 => Some(PReg(e + 101)),
        _ => None,
    }
}

/// Convert a 128-bit FPR (V register) to its 8-bit alias (B register).
pub fn fpr128_to_fpr8(reg: PReg) -> Option<PReg> {
    let e = reg.encoding();
    match e {
        64..=95 => Some(PReg(e + 133)),
        _ => None,
    }
}

/// Convert a 64-bit FPR (D register) to its 128-bit alias (V register).
pub fn fpr64_to_fpr128(reg: PReg) -> Option<PReg> {
    let e = reg.encoding();
    match e {
        96..=127 => Some(PReg(e - 32)),
        _ => None,
    }
}

/// Convert a 32-bit FPR (S register) to its 128-bit alias (V register).
pub fn fpr32_to_fpr128(reg: PReg) -> Option<PReg> {
    let e = reg.encoding();
    match e {
        128..=159 => Some(PReg(e - 64)),
        _ => None,
    }
}

/// Return the register number (0-31) for any register in a given class.
///
/// This is the logical index within the register file, NOT the hardware
/// encoding (though for most registers they are the same).
pub fn reg_number(reg: PReg) -> Option<u8> {
    let e = reg.encoding();
    match e {
        0..=30 => Some(e as u8),          // X0-X30
        31 => Some(31),                    // SP
        32..=62 => Some((e - 32) as u8),   // W0-W30
        63 => Some(31),                    // WSP
        64..=95 => Some((e - 64) as u8),   // V0-V31
        96..=127 => Some((e - 96) as u8),  // D0-D31
        128..=159 => Some((e - 128) as u8), // S0-S31
        160 => Some(31),                   // XZR
        161 => Some(31),                   // WZR
        165..=196 => Some((e - 165) as u8), // H0-H31
        197..=228 => Some((e - 197) as u8), // B0-B31
        _ => None,
    }
}

/// Return `true` if two registers overlap (share physical storage).
///
/// For example, X0 overlaps W0, V0 overlaps D0/S0/H0/B0.
pub fn regs_overlap(a: PReg, b: PReg) -> bool {
    if a == b {
        return true;
    }

    // Get the "root" register number and class group for each
    let a_root = reg_root(a);
    let b_root = reg_root(b);

    match (a_root, b_root) {
        (Some((num_a, group_a)), Some((num_b, group_b))) => {
            num_a == num_b && group_a == group_b
        }
        _ => false,
    }
}

/// Return the root register number and class group (0=GPR, 1=FPR).
fn reg_root(reg: PReg) -> Option<(u8, u8)> {
    let e = reg.encoding();
    match e {
        0..=31 => Some((e as u8, 0)),          // GPR64
        32..=63 => Some(((e - 32) as u8, 0)),  // GPR32 aliases GPR64
        64..=95 => Some(((e - 64) as u8, 1)),  // FPR128
        96..=127 => Some(((e - 96) as u8, 1)), // FPR64 aliases FPR128
        128..=159 => Some(((e - 128) as u8, 1)), // FPR32 aliases FPR128
        160 => Some((31, 0)),                   // XZR aliases GPR group
        161 => Some((31, 0)),                   // WZR aliases GPR group
        165..=196 => Some(((e - 165) as u8, 1)), // FPR16 aliases FPR128
        197..=228 => Some(((e - 197) as u8, 1)), // FPR8 aliases FPR128
        _ => None,
    }
}

// ===========================================================================
// AArch64 Condition Codes
// ===========================================================================
// Reference: ARM Architecture Reference Manual, C1.2.4

/// AArch64 condition code for conditional branches, selects, and compares.
///
/// The 4-bit encoding matches the hardware encoding used in instruction
/// fields (e.g., `B.cond`, `CSEL`, `CCMP`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum CondCode {
    /// Equal (Z == 1)
    EQ = 0b0000,
    /// Not equal (Z == 0)
    NE = 0b0001,
    /// Carry set / unsigned higher or same (C == 1)
    HS = 0b0010,
    /// Carry clear / unsigned lower (C == 0)
    LO = 0b0011,
    /// Minus / negative (N == 1)
    MI = 0b0100,
    /// Plus / positive or zero (N == 0)
    PL = 0b0101,
    /// Overflow (V == 1)
    VS = 0b0110,
    /// No overflow (V == 0)
    VC = 0b0111,
    /// Unsigned higher (C == 1 && Z == 0)
    HI = 0b1000,
    /// Unsigned lower or same (C == 0 || Z == 1)
    LS = 0b1001,
    /// Signed greater or equal (N == V)
    GE = 0b1010,
    /// Signed less than (N != V)
    LT = 0b1011,
    /// Signed greater than (Z == 0 && N == V)
    GT = 0b1100,
    /// Signed less or equal (Z == 1 || N != V)
    LE = 0b1101,
    /// Always (unconditional)
    AL = 0b1110,
    /// Never (reserved, behaves as AL in most contexts)
    NV = 0b1111,
}

/// Alternate names that are standard ARM assembly aliases.
pub const CC: CondCode = CondCode::LO;  // Carry clear = unsigned lower
pub const CS: CondCode = CondCode::HS;  // Carry set = unsigned higher or same

impl CondCode {
    /// Return the 4-bit hardware encoding.
    #[inline]
    pub const fn encoding(self) -> u8 {
        self as u8
    }

    /// Invert the condition (logical negation).
    ///
    /// The ARM architecture defines inverted conditions by flipping bit 0
    /// of the encoding, except for AL/NV which are unconditional.
    ///
    /// | Original | Inverted |
    /// |----------|----------|
    /// | EQ       | NE       |
    /// | HS (CS)  | LO (CC)  |
    /// | MI       | PL       |
    /// | VS       | VC       |
    /// | HI       | LS       |
    /// | GE       | LT       |
    /// | GT       | LE       |
    /// | AL       | NV       |
    #[inline]
    pub const fn invert(self) -> Self {
        // Safety: flipping bit 0 of a valid 4-bit code produces a valid 4-bit code.
        // All 16 values of the enum are defined, so this is always valid.
        let inv = (self as u8) ^ 1;
        // SAFETY: all 16 values 0b0000..=0b1111 are valid CondCode variants.
        unsafe { core::mem::transmute::<u8, CondCode>(inv) }
    }

    /// Return the assembly mnemonic suffix (e.g., `"eq"`, `"ne"`, `"hs"`).
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::EQ => "eq",
            Self::NE => "ne",
            Self::HS => "hs",
            Self::LO => "lo",
            Self::MI => "mi",
            Self::PL => "pl",
            Self::VS => "vs",
            Self::VC => "vc",
            Self::HI => "hi",
            Self::LS => "ls",
            Self::GE => "ge",
            Self::LT => "lt",
            Self::GT => "gt",
            Self::LE => "le",
            Self::AL => "al",
            Self::NV => "nv",
        }
    }

    /// Try to parse a condition code from its assembly mnemonic.
    pub fn from_str(s: &str) -> Option<Self> {
        match s {
            "eq" => Some(Self::EQ),
            "ne" => Some(Self::NE),
            "hs" | "cs" => Some(Self::HS),
            "lo" | "cc" => Some(Self::LO),
            "mi" => Some(Self::MI),
            "pl" => Some(Self::PL),
            "vs" => Some(Self::VS),
            "vc" => Some(Self::VC),
            "hi" => Some(Self::HI),
            "ls" => Some(Self::LS),
            "ge" => Some(Self::GE),
            "lt" => Some(Self::LT),
            "gt" => Some(Self::GT),
            "le" => Some(Self::LE),
            "al" => Some(Self::AL),
            "nv" => Some(Self::NV),
            _ => None,
        }
    }

    /// Return `true` if this is an unconditional code (AL or NV).
    #[inline]
    pub const fn is_unconditional(self) -> bool {
        matches!(self, Self::AL | Self::NV)
    }

    /// Return `true` if this is a signed comparison condition.
    #[inline]
    pub const fn is_signed(self) -> bool {
        matches!(self, Self::GE | Self::LT | Self::GT | Self::LE)
    }

    /// Return `true` if this is an unsigned comparison condition.
    #[inline]
    pub const fn is_unsigned(self) -> bool {
        matches!(self, Self::HI | Self::LS | Self::HS | Self::LO)
    }
}

impl core::fmt::Display for CondCode {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str(self.as_str())
    }
}

// ===========================================================================
// Shift types for shifted register operands
// ===========================================================================
// Reference: ARM ARM C6.2.2

/// Shift type applied to a register operand.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum ShiftType {
    /// Logical shift left
    LSL = 0b00,
    /// Logical shift right
    LSR = 0b01,
    /// Arithmetic shift right
    ASR = 0b10,
    /// Rotate right (used in some data-processing instructions)
    ROR = 0b11,
}

impl ShiftType {
    /// Return the 2-bit hardware encoding.
    #[inline]
    pub const fn encoding(self) -> u8 {
        self as u8
    }

    /// Return the assembly mnemonic.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::LSL => "lsl",
            Self::LSR => "lsr",
            Self::ASR => "asr",
            Self::ROR => "ror",
        }
    }
}

impl core::fmt::Display for ShiftType {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str(self.as_str())
    }
}

// ===========================================================================
// Extend types for extended register operands
// ===========================================================================
// Reference: ARM ARM C6.2.2

/// Extend type applied to a register operand.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum ExtendType {
    /// Unsigned byte extend
    UXTB = 0b000,
    /// Unsigned halfword extend
    UXTH = 0b001,
    /// Unsigned word extend (32→64, same as LSL for 32-bit source)
    UXTW = 0b010,
    /// Unsigned doubleword extend (64→64, identity for shift)
    UXTX = 0b011,
    /// Signed byte extend
    SXTB = 0b100,
    /// Signed halfword extend
    SXTH = 0b101,
    /// Signed word extend (32→64)
    SXTW = 0b110,
    /// Signed doubleword extend (64→64, identity for shift)
    SXTX = 0b111,
}

impl ExtendType {
    /// Return the 3-bit hardware encoding.
    #[inline]
    pub const fn encoding(self) -> u8 {
        self as u8
    }

    /// Return the assembly mnemonic.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::UXTB => "uxtb",
            Self::UXTH => "uxth",
            Self::UXTW => "uxtw",
            Self::UXTX => "uxtx",
            Self::SXTB => "sxtb",
            Self::SXTH => "sxth",
            Self::SXTW => "sxtw",
            Self::SXTX => "sxtx",
        }
    }

    /// Return `true` if this is a signed extend.
    #[inline]
    pub const fn is_signed(self) -> bool {
        (self as u8) & 0b100 != 0
    }

    /// Return the source width in bits before extension.
    #[inline]
    pub const fn source_bits(self) -> u32 {
        match self {
            Self::UXTB | Self::SXTB => 8,
            Self::UXTH | Self::SXTH => 16,
            Self::UXTW | Self::SXTW => 32,
            Self::UXTX | Self::SXTX => 64,
        }
    }
}

impl core::fmt::Display for ExtendType {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str(self.as_str())
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_preg_names() {
        assert_eq!(preg_name(X0), "x0");
        assert_eq!(preg_name(X30), "x30");
        assert_eq!(preg_name(SP), "sp");
        assert_eq!(preg_name(W0), "w0");
        assert_eq!(preg_name(W30), "w30");
        assert_eq!(preg_name(WSP), "wsp");
        assert_eq!(preg_name(V0), "v0");
        assert_eq!(preg_name(V31), "v31");
        assert_eq!(preg_name(D0), "d0");
        assert_eq!(preg_name(D31), "d31");
        assert_eq!(preg_name(S0), "s0");
        assert_eq!(preg_name(S31), "s31");
        assert_eq!(preg_name(H0), "h0");
        assert_eq!(preg_name(H31), "h31");
        assert_eq!(preg_name(B0), "b0");
        assert_eq!(preg_name(B31), "b31");
        assert_eq!(preg_name(XZR), "xzr");
        assert_eq!(preg_name(WZR), "wzr");
        assert_eq!(preg_name(NZCV), "nzcv");
        assert_eq!(preg_name(FPCR), "fpcr");
        assert_eq!(preg_name(FPSR), "fpsr");
    }

    #[test]
    fn test_preg_classes() {
        assert_eq!(preg_class(X0), RegClass::Gpr64);
        assert_eq!(preg_class(X30), RegClass::Gpr64);
        assert_eq!(preg_class(SP), RegClass::Gpr64);
        assert_eq!(preg_class(W0), RegClass::Gpr32);
        assert_eq!(preg_class(WSP), RegClass::Gpr32);
        assert_eq!(preg_class(V0), RegClass::Fpr128);
        assert_eq!(preg_class(D0), RegClass::Fpr64);
        assert_eq!(preg_class(S0), RegClass::Fpr32);
        assert_eq!(preg_class(H0), RegClass::Fpr16);
        assert_eq!(preg_class(B0), RegClass::Fpr8);
        assert_eq!(preg_class(XZR), RegClass::Gpr64);
        assert_eq!(preg_class(WZR), RegClass::Gpr32);
        assert_eq!(preg_class(NZCV), RegClass::System);
    }

    #[test]
    fn test_hw_encoding() {
        assert_eq!(hw_encoding(X0), 0);
        assert_eq!(hw_encoding(X30), 30);
        assert_eq!(hw_encoding(SP), 31);
        assert_eq!(hw_encoding(XZR), 31);
        assert_eq!(hw_encoding(W0), 0);
        assert_eq!(hw_encoding(W30), 30);
        assert_eq!(hw_encoding(WSP), 31);
        assert_eq!(hw_encoding(WZR), 31);
        assert_eq!(hw_encoding(V0), 0);
        assert_eq!(hw_encoding(V31), 31);
        assert_eq!(hw_encoding(D15), 15);
        assert_eq!(hw_encoding(S7), 7);
        assert_eq!(hw_encoding(H0), 0);
        assert_eq!(hw_encoding(B31), 31);
    }

    #[test]
    fn test_callee_saved() {
        // GPR64 callee-saved: X19-X28
        assert!(!is_callee_saved(X0));
        assert!(!is_callee_saved(X18));
        assert!(is_callee_saved(X19));
        assert!(is_callee_saved(X28));
        assert!(!is_callee_saved(X29));
        assert!(!is_callee_saved(X30));

        // GPR32 callee-saved: W19-W28
        assert!(is_callee_saved(W19));
        assert!(is_callee_saved(W28));
        assert!(!is_callee_saved(W0));

        // FPR callee-saved: V8-V15
        assert!(!is_callee_saved(V0));
        assert!(is_callee_saved(V8));
        assert!(is_callee_saved(V15));
        assert!(!is_callee_saved(V16));

        // D8-D15 also callee-saved
        assert!(is_callee_saved(D8));
        assert!(is_callee_saved(D15));
        assert!(!is_callee_saved(D16));
    }

    #[test]
    fn test_gpr_conversion() {
        assert_eq!(gpr64_to_gpr32(X0), Some(W0));
        assert_eq!(gpr64_to_gpr32(X30), Some(W30));
        assert_eq!(gpr64_to_gpr32(SP), Some(WSP));
        assert_eq!(gpr64_to_gpr32(XZR), Some(WZR));
        assert_eq!(gpr64_to_gpr32(V0), None);

        assert_eq!(gpr32_to_gpr64(W0), Some(X0));
        assert_eq!(gpr32_to_gpr64(W30), Some(X30));
        assert_eq!(gpr32_to_gpr64(WSP), Some(SP));
        assert_eq!(gpr32_to_gpr64(WZR), Some(XZR));
        assert_eq!(gpr32_to_gpr64(V0), None);
    }

    #[test]
    fn test_fpr_conversion() {
        assert_eq!(fpr128_to_fpr64(V0), Some(D0));
        assert_eq!(fpr128_to_fpr64(V31), Some(D31));
        assert_eq!(fpr128_to_fpr32(V0), Some(S0));
        assert_eq!(fpr128_to_fpr32(V31), Some(S31));
        assert_eq!(fpr128_to_fpr16(V0), Some(H0));
        assert_eq!(fpr128_to_fpr16(V31), Some(H31));
        assert_eq!(fpr128_to_fpr8(V0), Some(B0));
        assert_eq!(fpr128_to_fpr8(V31), Some(B31));

        assert_eq!(fpr64_to_fpr128(D0), Some(V0));
        assert_eq!(fpr64_to_fpr128(D31), Some(V31));
        assert_eq!(fpr32_to_fpr128(S0), Some(V0));
        assert_eq!(fpr32_to_fpr128(S31), Some(V31));
    }

    #[test]
    fn test_regs_overlap() {
        assert!(regs_overlap(X0, W0));
        assert!(regs_overlap(W0, X0));
        assert!(regs_overlap(V0, D0));
        assert!(regs_overlap(V0, S0));
        assert!(regs_overlap(V0, H0));
        assert!(regs_overlap(V0, B0));
        assert!(regs_overlap(D0, S0));

        assert!(!regs_overlap(X0, X1));
        assert!(!regs_overlap(X0, W1));
        assert!(!regs_overlap(V0, V1));
        assert!(!regs_overlap(X0, V0)); // different groups
    }

    #[test]
    fn test_cond_code_invert() {
        assert_eq!(CondCode::EQ.invert(), CondCode::NE);
        assert_eq!(CondCode::NE.invert(), CondCode::EQ);
        assert_eq!(CondCode::HS.invert(), CondCode::LO);
        assert_eq!(CondCode::LO.invert(), CondCode::HS);
        assert_eq!(CondCode::MI.invert(), CondCode::PL);
        assert_eq!(CondCode::PL.invert(), CondCode::MI);
        assert_eq!(CondCode::VS.invert(), CondCode::VC);
        assert_eq!(CondCode::VC.invert(), CondCode::VS);
        assert_eq!(CondCode::HI.invert(), CondCode::LS);
        assert_eq!(CondCode::LS.invert(), CondCode::HI);
        assert_eq!(CondCode::GE.invert(), CondCode::LT);
        assert_eq!(CondCode::LT.invert(), CondCode::GE);
        assert_eq!(CondCode::GT.invert(), CondCode::LE);
        assert_eq!(CondCode::LE.invert(), CondCode::GT);
        assert_eq!(CondCode::AL.invert(), CondCode::NV);
        assert_eq!(CondCode::NV.invert(), CondCode::AL);
    }

    #[test]
    fn test_cond_code_encoding() {
        assert_eq!(CondCode::EQ.encoding(), 0b0000);
        assert_eq!(CondCode::NE.encoding(), 0b0001);
        assert_eq!(CondCode::HS.encoding(), 0b0010);
        assert_eq!(CondCode::LO.encoding(), 0b0011);
        assert_eq!(CondCode::MI.encoding(), 0b0100);
        assert_eq!(CondCode::PL.encoding(), 0b0101);
        assert_eq!(CondCode::VS.encoding(), 0b0110);
        assert_eq!(CondCode::VC.encoding(), 0b0111);
        assert_eq!(CondCode::HI.encoding(), 0b1000);
        assert_eq!(CondCode::LS.encoding(), 0b1001);
        assert_eq!(CondCode::GE.encoding(), 0b1010);
        assert_eq!(CondCode::LT.encoding(), 0b1011);
        assert_eq!(CondCode::GT.encoding(), 0b1100);
        assert_eq!(CondCode::LE.encoding(), 0b1101);
        assert_eq!(CondCode::AL.encoding(), 0b1110);
        assert_eq!(CondCode::NV.encoding(), 0b1111);
    }

    #[test]
    fn test_cond_code_parse() {
        assert_eq!(CondCode::from_str("eq"), Some(CondCode::EQ));
        assert_eq!(CondCode::from_str("ne"), Some(CondCode::NE));
        assert_eq!(CondCode::from_str("hs"), Some(CondCode::HS));
        assert_eq!(CondCode::from_str("cs"), Some(CondCode::HS));
        assert_eq!(CondCode::from_str("lo"), Some(CondCode::LO));
        assert_eq!(CondCode::from_str("cc"), Some(CondCode::LO));
        assert_eq!(CondCode::from_str("xyz"), None);
    }

    #[test]
    fn test_cond_code_properties() {
        assert!(CondCode::GE.is_signed());
        assert!(CondCode::LT.is_signed());
        assert!(CondCode::GT.is_signed());
        assert!(CondCode::LE.is_signed());
        assert!(!CondCode::EQ.is_signed());

        assert!(CondCode::HI.is_unsigned());
        assert!(CondCode::LS.is_unsigned());
        assert!(CondCode::HS.is_unsigned());
        assert!(CondCode::LO.is_unsigned());
        assert!(!CondCode::EQ.is_unsigned());

        assert!(CondCode::AL.is_unconditional());
        assert!(CondCode::NV.is_unconditional());
        assert!(!CondCode::EQ.is_unconditional());
    }

    #[test]
    fn test_register_class_arrays() {
        // ALLOCATABLE_GPRS should not contain X8, X16, X17, X18, X29, X30
        assert!(!ALLOCATABLE_GPRS.contains(&X8));
        assert!(!ALLOCATABLE_GPRS.contains(&X16));
        assert!(!ALLOCATABLE_GPRS.contains(&X17));
        assert!(!ALLOCATABLE_GPRS.contains(&X18));
        assert!(!ALLOCATABLE_GPRS.contains(&X29));
        assert!(!ALLOCATABLE_GPRS.contains(&X30));

        // But should contain X0-X7, X9-X15, X19-X28
        assert!(ALLOCATABLE_GPRS.contains(&X0));
        assert!(ALLOCATABLE_GPRS.contains(&X7));
        assert!(ALLOCATABLE_GPRS.contains(&X9));
        assert!(ALLOCATABLE_GPRS.contains(&X15));
        assert!(ALLOCATABLE_GPRS.contains(&X19));
        assert!(ALLOCATABLE_GPRS.contains(&X28));

        // ARG_GPRS = X0-X7
        assert_eq!(ARG_GPRS.len(), 8);
        assert_eq!(ARG_GPRS[0], X0);
        assert_eq!(ARG_GPRS[7], X7);

        // ARG_FPRS = V0-V7
        assert_eq!(ARG_FPRS.len(), 8);
        assert_eq!(ARG_FPRS[0], V0);
        assert_eq!(ARG_FPRS[7], V7);

        // CALLEE_SAVED_GPRS = X19-X28
        assert_eq!(CALLEE_SAVED_GPRS.len(), 10);
        assert_eq!(CALLEE_SAVED_GPRS[0], X19);
        assert_eq!(CALLEE_SAVED_GPRS[9], X28);

        // CALLEE_SAVED_FPRS = V8-V15
        assert_eq!(CALLEE_SAVED_FPRS.len(), 8);
        assert_eq!(CALLEE_SAVED_FPRS[0], V8);
        assert_eq!(CALLEE_SAVED_FPRS[7], V15);
    }

    #[test]
    fn test_shift_type() {
        assert_eq!(ShiftType::LSL.encoding(), 0b00);
        assert_eq!(ShiftType::LSR.encoding(), 0b01);
        assert_eq!(ShiftType::ASR.encoding(), 0b10);
        assert_eq!(ShiftType::ROR.encoding(), 0b11);
        assert_eq!(ShiftType::LSL.as_str(), "lsl");
    }

    #[test]
    fn test_extend_type() {
        assert_eq!(ExtendType::UXTB.encoding(), 0b000);
        assert_eq!(ExtendType::SXTX.encoding(), 0b111);
        assert!(!ExtendType::UXTB.is_signed());
        assert!(ExtendType::SXTB.is_signed());
        assert_eq!(ExtendType::UXTB.source_bits(), 8);
        assert_eq!(ExtendType::UXTH.source_bits(), 16);
        assert_eq!(ExtendType::UXTW.source_bits(), 32);
        assert_eq!(ExtendType::UXTX.source_bits(), 64);
    }

    #[test]
    fn test_reg_class_sizes() {
        assert_eq!(RegClass::Gpr64.size_bits(), 64);
        assert_eq!(RegClass::Gpr32.size_bits(), 32);
        assert_eq!(RegClass::Fpr128.size_bits(), 128);
        assert_eq!(RegClass::Fpr64.size_bits(), 64);
        assert_eq!(RegClass::Fpr32.size_bits(), 32);
        assert_eq!(RegClass::Fpr16.size_bits(), 16);
        assert_eq!(RegClass::Fpr8.size_bits(), 8);

        assert_eq!(RegClass::Gpr64.size_bytes(), 8);
        assert_eq!(RegClass::Fpr128.size_bytes(), 16);
    }

    #[test]
    fn test_display() {
        assert_eq!(format!("{}", X0), "x0");
        assert_eq!(format!("{}", SP), "sp");
        assert_eq!(format!("{}", V31), "v31");
        assert_eq!(format!("{}", CondCode::EQ), "eq");
        assert_eq!(format!("{}", ShiftType::LSL), "lsl");
        assert_eq!(format!("{}", ExtendType::SXTW), "sxtw");
    }
}
