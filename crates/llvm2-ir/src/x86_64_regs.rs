// llvm2-ir - x86-64 register model
//
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Copyright 2026 Dropbox, Inc. | License: Apache-2.0
//
// Reference: ~/llvm-project-ref/llvm/lib/Target/X86/X86RegisterInfo.td
// Reference: System V AMD64 ABI (https://refspecs.linuxbase.org/elf/x86_64-abi-0.99.pdf)

//! x86-64 register definitions.
//!
//! Encoding scheme for [`X86PReg`]:
//! - `0..=15`    GPR64  (RAX, RCX, RDX, RBX, RSP, RBP, RSI, RDI, R8-R15)
//! - `16..=31`   GPR32  (EAX, ECX, EDX, EBX, ESP, EBP, ESI, EDI, R8D-R15D)
//! - `32..=47`   GPR16  (AX, CX, DX, BX, SP, BP, SI, DI, R8W-R15W)
//! - `48..=63`   GPR8   (AL, CL, DL, BL, SPL, BPL, SIL, DIL, R8B-R15B)
//! - `64..=79`   XMM    (XMM0-XMM15)
//! - `80`        RFLAGS
//! - `81`        RIP
//!
//! Register numbering within each class follows the standard x86-64 encoding
//! (RAX=0, RCX=1, RDX=2, RBX=3, RSP=4, RBP=5, RSI=6, RDI=7, R8-R15=8-15).

// ---------------------------------------------------------------------------
// X86PReg — Physical Register
// ---------------------------------------------------------------------------

/// A physical x86-64 register identified by a unique encoding index.
///
/// The encoding is internal to LLVM2 and organizes registers by class.
/// Use [`x86_hw_encoding`] to get the ModR/M register field value.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct X86PReg(u16);

impl X86PReg {
    /// Create an X86PReg from a raw encoding value.
    #[inline]
    pub const fn new(encoding: u16) -> Self {
        Self(encoding)
    }

    /// Return the raw encoding value.
    #[inline]
    pub const fn encoding(self) -> u16 {
        self.0
    }

    /// Returns true if this is a 64-bit general-purpose register.
    #[inline]
    pub const fn is_gpr64(self) -> bool {
        self.0 <= 15
    }

    /// Returns true if this is a 32-bit general-purpose register.
    #[inline]
    pub const fn is_gpr32(self) -> bool {
        self.0 >= 16 && self.0 <= 31
    }

    /// Returns true if this is a 16-bit general-purpose register.
    #[inline]
    pub const fn is_gpr16(self) -> bool {
        self.0 >= 32 && self.0 <= 47
    }

    /// Returns true if this is an 8-bit general-purpose register.
    #[inline]
    pub const fn is_gpr8(self) -> bool {
        self.0 >= 48 && self.0 <= 63
    }

    /// Returns true if this is any GPR (64, 32, 16, or 8 bit).
    #[inline]
    pub const fn is_gpr(self) -> bool {
        self.0 <= 63
    }

    /// Returns true if this is an XMM SIMD/FP register.
    #[inline]
    pub const fn is_xmm(self) -> bool {
        self.0 >= 64 && self.0 <= 79
    }

    /// Returns true if this is a system/special register (RFLAGS, RIP).
    #[inline]
    pub const fn is_system(self) -> bool {
        self.0 == 80 || self.0 == 81
    }

    /// Returns true if this register is allocatable for register allocation.
    ///
    /// RSP (stack pointer) and RBP (frame pointer) and their sub-register
    /// aliases are excluded. System registers (RFLAGS, RIP) are also excluded.
    #[inline]
    pub fn is_allocatable(self) -> bool {
        let hw = self.hw_enc();
        // Exclude RSP (hw=4), RBP (hw=5), and system regs
        self.is_gpr() && hw != 4 && hw != 5
            || self.is_xmm()
    }

    /// Returns the 4-bit hardware encoding for this register.
    ///
    /// This is the value that appears in ModR/M, SIB, and REX fields.
    #[inline]
    pub fn hw_enc(self) -> u8 {
        x86_hw_encoding(self)
    }

    /// Returns true if accessing this register requires a REX prefix.
    ///
    /// Registers R8-R15 and their sub-registers need REX.B or REX.R.
    /// SPL, BPL, SIL, DIL also need a REX prefix (to distinguish from AH-DH).
    #[inline]
    pub fn needs_rex(self) -> bool {
        let e = self.0;
        match e {
            // GPR64: R8-R15 (encodings 8-15)
            8..=15 => true,
            // GPR32: R8D-R15D (encodings 24-31)
            24..=31 => true,
            // GPR16: R8W-R15W (encodings 40-47)
            40..=47 => true,
            // GPR8: R8B-R15B (encodings 56-63), plus SPL/BPL/SIL/DIL (52-55)
            52..=63 => true,
            // XMM8-XMM15 (encodings 72-79)
            72..=79 => true,
            _ => false,
        }
    }

    /// Returns the register class for this register.
    #[inline]
    pub fn reg_class(self) -> X86RegClass {
        x86_preg_class(self)
    }
}

impl core::fmt::Debug for X86PReg {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "X86PReg({}:{})", self.0, x86_preg_name(*self))
    }
}

impl core::fmt::Display for X86PReg {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str(x86_preg_name(*self))
    }
}

// ---------------------------------------------------------------------------
// X86 Register class
// ---------------------------------------------------------------------------

/// Register class for an x86-64 physical register.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum X86RegClass {
    /// 64-bit general-purpose (RAX-R15)
    Gpr64,
    /// 32-bit general-purpose (EAX-R15D) -- aliases low 32 bits of GPR64
    Gpr32,
    /// 16-bit general-purpose (AX-R15W) -- aliases low 16 bits of GPR64
    Gpr16,
    /// 8-bit general-purpose (AL-R15B) -- aliases low 8 bits of GPR64
    Gpr8,
    /// 128-bit SSE (XMM0-XMM15)
    Xmm128,
    /// System/special (RFLAGS, RIP)
    System,
}

impl X86RegClass {
    /// Size in bits of registers in this class.
    #[inline]
    pub const fn size_bits(self) -> u32 {
        match self {
            Self::Gpr64 => 64,
            Self::Gpr32 => 32,
            Self::Gpr16 => 16,
            Self::Gpr8 => 8,
            Self::Xmm128 => 128,
            Self::System => 64,
        }
    }

    /// Size in bytes of registers in this class.
    #[inline]
    pub const fn size_bytes(self) -> u32 {
        self.size_bits() / 8
    }
}

// ===========================================================================
// GPR64: RAX-R15 (encodings 0-15)
// ===========================================================================
// Order matches x86-64 register encoding: RAX=0, RCX=1, RDX=2, RBX=3,
// RSP=4, RBP=5, RSI=6, RDI=7, R8-R15=8-15.

pub const RAX: X86PReg = X86PReg(0);
pub const RCX: X86PReg = X86PReg(1);
pub const RDX: X86PReg = X86PReg(2);
pub const RBX: X86PReg = X86PReg(3);
pub const RSP: X86PReg = X86PReg(4);
pub const RBP: X86PReg = X86PReg(5);
pub const RSI: X86PReg = X86PReg(6);
pub const RDI: X86PReg = X86PReg(7);
pub const R8:  X86PReg = X86PReg(8);
pub const R9:  X86PReg = X86PReg(9);
pub const R10: X86PReg = X86PReg(10);
pub const R11: X86PReg = X86PReg(11);
pub const R12: X86PReg = X86PReg(12);
pub const R13: X86PReg = X86PReg(13);
pub const R14: X86PReg = X86PReg(14);
pub const R15: X86PReg = X86PReg(15);

// ===========================================================================
// GPR32: EAX-R15D (encodings 16-31)
// ===========================================================================

pub const EAX:  X86PReg = X86PReg(16);
pub const ECX:  X86PReg = X86PReg(17);
pub const EDX:  X86PReg = X86PReg(18);
pub const EBX:  X86PReg = X86PReg(19);
pub const ESP:  X86PReg = X86PReg(20);
pub const EBP:  X86PReg = X86PReg(21);
pub const ESI:  X86PReg = X86PReg(22);
pub const EDI:  X86PReg = X86PReg(23);
pub const R8D:  X86PReg = X86PReg(24);
pub const R9D:  X86PReg = X86PReg(25);
pub const R10D: X86PReg = X86PReg(26);
pub const R11D: X86PReg = X86PReg(27);
pub const R12D: X86PReg = X86PReg(28);
pub const R13D: X86PReg = X86PReg(29);
pub const R14D: X86PReg = X86PReg(30);
pub const R15D: X86PReg = X86PReg(31);

// ===========================================================================
// GPR16: AX-R15W (encodings 32-47)
// ===========================================================================

pub const AX:   X86PReg = X86PReg(32);
pub const CX:   X86PReg = X86PReg(33);
pub const DX:   X86PReg = X86PReg(34);
pub const BX:   X86PReg = X86PReg(35);
pub const SP16: X86PReg = X86PReg(36);
pub const BP16: X86PReg = X86PReg(37);
pub const SI:   X86PReg = X86PReg(38);
pub const DI:   X86PReg = X86PReg(39);
pub const R8W:  X86PReg = X86PReg(40);
pub const R9W:  X86PReg = X86PReg(41);
pub const R10W: X86PReg = X86PReg(42);
pub const R11W: X86PReg = X86PReg(43);
pub const R12W: X86PReg = X86PReg(44);
pub const R13W: X86PReg = X86PReg(45);
pub const R14W: X86PReg = X86PReg(46);
pub const R15W: X86PReg = X86PReg(47);

// ===========================================================================
// GPR8: AL-R15B (encodings 48-63)
// ===========================================================================

pub const AL:   X86PReg = X86PReg(48);
pub const CL:   X86PReg = X86PReg(49);
pub const DL:   X86PReg = X86PReg(50);
pub const BL:   X86PReg = X86PReg(51);
pub const SPL:  X86PReg = X86PReg(52);
pub const BPL:  X86PReg = X86PReg(53);
pub const SIL:  X86PReg = X86PReg(54);
pub const DIL:  X86PReg = X86PReg(55);
pub const R8B:  X86PReg = X86PReg(56);
pub const R9B:  X86PReg = X86PReg(57);
pub const R10B: X86PReg = X86PReg(58);
pub const R11B: X86PReg = X86PReg(59);
pub const R12B: X86PReg = X86PReg(60);
pub const R13B: X86PReg = X86PReg(61);
pub const R14B: X86PReg = X86PReg(62);
pub const R15B: X86PReg = X86PReg(63);

// ===========================================================================
// XMM: XMM0-XMM15 (encodings 64-79)
// ===========================================================================

pub const XMM0:  X86PReg = X86PReg(64);
pub const XMM1:  X86PReg = X86PReg(65);
pub const XMM2:  X86PReg = X86PReg(66);
pub const XMM3:  X86PReg = X86PReg(67);
pub const XMM4:  X86PReg = X86PReg(68);
pub const XMM5:  X86PReg = X86PReg(69);
pub const XMM6:  X86PReg = X86PReg(70);
pub const XMM7:  X86PReg = X86PReg(71);
pub const XMM8:  X86PReg = X86PReg(72);
pub const XMM9:  X86PReg = X86PReg(73);
pub const XMM10: X86PReg = X86PReg(74);
pub const XMM11: X86PReg = X86PReg(75);
pub const XMM12: X86PReg = X86PReg(76);
pub const XMM13: X86PReg = X86PReg(77);
pub const XMM14: X86PReg = X86PReg(78);
pub const XMM15: X86PReg = X86PReg(79);

// ===========================================================================
// Special registers (encodings 80-81)
// ===========================================================================

/// RFLAGS status register.
pub const RFLAGS: X86PReg = X86PReg(80);
/// Instruction pointer (not directly addressable).
pub const RIP: X86PReg = X86PReg(81);

// ===========================================================================
// Register class arrays — System V AMD64 ABI
// ===========================================================================
// Reference: System V AMD64 ABI, section 3.2.1 (registers and calling convention)

/// All 64-bit GPRs (RAX-R15).
pub const X86_ALL_GPRS: [X86PReg; 16] = [
    RAX, RCX, RDX, RBX, RSP, RBP, RSI, RDI,
    R8, R9, R10, R11, R12, R13, R14, R15,
];

/// Allocatable GPRs for register allocation.
///
/// Excludes:
/// - RSP (stack pointer, not allocatable)
/// - RBP (frame pointer, not allocatable when frame pointer is used)
pub const X86_ALLOCATABLE_GPRS: [X86PReg; 14] = [
    RAX, RCX, RDX, RBX, RSI, RDI,
    R8, R9, R10, R11, R12, R13, R14, R15,
];

/// Callee-saved (non-volatile) GPRs per System V AMD64 ABI.
///
/// A called function must save and restore these if it uses them.
/// System V ABI: RBX, RBP, R12-R15
pub const X86_CALLEE_SAVED_GPRS: [X86PReg; 6] = [
    RBX, RBP, R12, R13, R14, R15,
];

/// Caller-saved (volatile) GPRs per System V AMD64 ABI.
///
/// These are clobbered by function calls. The caller must save them if needed.
/// System V ABI: RAX, RCX, RDX, RSI, RDI, R8-R11
pub const X86_CALLER_SAVED_GPRS: [X86PReg; 9] = [
    RAX, RCX, RDX, RSI, RDI, R8, R9, R10, R11,
];

/// Argument-passing GPRs (System V AMD64 ABI order).
///
/// Arguments are passed in: RDI, RSI, RDX, RCX, R8, R9
pub const X86_ARG_GPRS: [X86PReg; 6] = [RDI, RSI, RDX, RCX, R8, R9];

/// Return-value GPRs. RAX for the first integer return value, RDX for the second.
pub const X86_RET_GPRS: [X86PReg; 2] = [RAX, RDX];

/// All XMM registers.
pub const X86_ALL_XMMS: [X86PReg; 16] = [
    XMM0,  XMM1,  XMM2,  XMM3,  XMM4,  XMM5,  XMM6,  XMM7,
    XMM8,  XMM9,  XMM10, XMM11, XMM12, XMM13, XMM14, XMM15,
];

/// Allocatable XMM registers (all 16 are allocatable in System V).
pub const X86_ALLOCATABLE_XMMS: [X86PReg; 16] = X86_ALL_XMMS;

/// Caller-saved XMM registers per System V AMD64 ABI.
///
/// All XMM registers are caller-saved in System V (unlike Windows x64 where XMM6-XMM15 are callee-saved).
pub const X86_CALLER_SAVED_XMMS: [X86PReg; 16] = X86_ALL_XMMS;

/// Argument-passing XMM registers (System V AMD64 ABI).
///
/// Float/double arguments passed in: XMM0-XMM7
pub const X86_ARG_XMMS: [X86PReg; 8] = [
    XMM0, XMM1, XMM2, XMM3, XMM4, XMM5, XMM6, XMM7,
];

/// Return-value XMM registers. XMM0 for the first FP return, XMM1 for the second.
pub const X86_RET_XMMS: [X86PReg; 2] = [XMM0, XMM1];

/// Registers clobbered by a CALL instruction (superset of caller-saved).
///
/// Includes all caller-saved GPRs. RSP is also modified (by the implicit
/// push of the return address), but it is restored by RET so it is not
/// listed here. RFLAGS is clobbered but handled separately.
pub const X86_CALL_CLOBBER_GPRS: [X86PReg; 9] = X86_CALLER_SAVED_GPRS;

/// Allocatable 32-bit GPRs (sub-register aliases of allocatable 64-bit GPRs).
///
/// Excludes ESP (alias of RSP) and EBP (alias of RBP).
pub const X86_ALLOCATABLE_GPR32S: [X86PReg; 14] = [
    EAX, ECX, EDX, EBX, ESI, EDI,
    R8D, R9D, R10D, R11D, R12D, R13D, R14D, R15D,
];

/// Allocatable 16-bit GPRs (sub-register aliases of allocatable 64-bit GPRs).
///
/// Excludes SP16 (alias of RSP) and BP16 (alias of RBP).
pub const X86_ALLOCATABLE_GPR16S: [X86PReg; 14] = [
    AX, CX, DX, BX, SI, DI,
    R8W, R9W, R10W, R11W, R12W, R13W, R14W, R15W,
];

/// Allocatable 8-bit GPRs (sub-register aliases of allocatable 64-bit GPRs).
///
/// Excludes SPL (alias of RSP) and BPL (alias of RBP).
pub const X86_ALLOCATABLE_GPR8S: [X86PReg; 14] = [
    AL, CL, DL, BL, SIL, DIL,
    R8B, R9B, R10B, R11B, R12B, R13B, R14B, R15B,
];

/// Callee-saved 32-bit GPR aliases (sub-registers of callee-saved 64-bit GPRs).
pub const X86_CALLEE_SAVED_GPR32S: [X86PReg; 6] = [
    EBX, EBP, R12D, R13D, R14D, R15D,
];

// ===========================================================================
// Lookup tables for name -> register mapping
// ===========================================================================

/// GPR64 names indexed by register number.
const GPR64_NAMES: [&str; 16] = [
    "rax", "rcx", "rdx", "rbx", "rsp", "rbp", "rsi", "rdi",
    "r8",  "r9",  "r10", "r11", "r12", "r13", "r14", "r15",
];

/// GPR32 names indexed by register number.
const GPR32_NAMES: [&str; 16] = [
    "eax",  "ecx",  "edx",  "ebx",  "esp",  "ebp",  "esi",  "edi",
    "r8d",  "r9d",  "r10d", "r11d", "r12d", "r13d", "r14d", "r15d",
];

/// GPR16 names indexed by register number.
const GPR16_NAMES: [&str; 16] = [
    "ax",   "cx",   "dx",   "bx",   "sp",   "bp",   "si",   "di",
    "r8w",  "r9w",  "r10w", "r11w", "r12w", "r13w", "r14w", "r15w",
];

/// GPR8 names indexed by register number.
const GPR8_NAMES: [&str; 16] = [
    "al",   "cl",   "dl",   "bl",   "spl",  "bpl",  "sil",  "dil",
    "r8b",  "r9b",  "r10b", "r11b", "r12b", "r13b", "r14b", "r15b",
];

/// XMM names indexed by register number.
const XMM_NAMES: [&str; 16] = [
    "xmm0",  "xmm1",  "xmm2",  "xmm3",  "xmm4",  "xmm5",  "xmm6",  "xmm7",
    "xmm8",  "xmm9",  "xmm10", "xmm11", "xmm12", "xmm13", "xmm14", "xmm15",
];

// ===========================================================================
// Helper functions
// ===========================================================================

/// Return the assembly name for an x86-64 physical register.
pub fn x86_preg_name(reg: X86PReg) -> &'static str {
    let e = reg.encoding();
    match e {
        0..=15 => GPR64_NAMES[e as usize],
        16..=31 => GPR32_NAMES[(e - 16) as usize],
        32..=47 => GPR16_NAMES[(e - 32) as usize],
        48..=63 => GPR8_NAMES[(e - 48) as usize],
        64..=79 => XMM_NAMES[(e - 64) as usize],
        80 => "rflags",
        81 => "rip",
        _ => "<unknown>",
    }
}

/// Return the register class for an x86-64 physical register.
pub fn x86_preg_class(reg: X86PReg) -> X86RegClass {
    let e = reg.encoding();
    match e {
        0..=15 => X86RegClass::Gpr64,
        16..=31 => X86RegClass::Gpr32,
        32..=47 => X86RegClass::Gpr16,
        48..=63 => X86RegClass::Gpr8,
        64..=79 => X86RegClass::Xmm128,
        80..=81 => X86RegClass::System,
        _ => X86RegClass::System,
    }
}

/// Return the 4-bit hardware encoding for an x86-64 register.
///
/// This is the value used in ModR/M reg fields, SIB base/index, and
/// determines REX.B/REX.R extension bits.
pub fn x86_hw_encoding(reg: X86PReg) -> u8 {
    let e = reg.encoding();
    match e {
        0..=15 => e as u8,           // GPR64: RAX=0 .. R15=15
        16..=31 => (e - 16) as u8,   // GPR32: same numbering
        32..=47 => (e - 32) as u8,   // GPR16: same numbering
        48..=63 => (e - 48) as u8,   // GPR8: same numbering
        64..=79 => (e - 64) as u8,   // XMM: 0-15
        _ => 0,
    }
}

/// Return `true` if the register is callee-saved (non-volatile) per System V AMD64 ABI.
pub fn x86_is_callee_saved(reg: X86PReg) -> bool {
    let e = reg.encoding();
    match e {
        // GPR64: RBX=3, RBP=5, R12=12, R13=13, R14=14, R15=15
        3 | 5 | 12..=15 => true,
        // GPR32 aliases: EBX=19, EBP=21, R12D=28..R15D=31
        19 | 21 | 28..=31 => true,
        // GPR16 aliases: BX=35, BP=37, R12W=44..R15W=47
        35 | 37 | 44..=47 => true,
        // GPR8 aliases: BL=51, BPL=53, R12B=60..R15B=63
        51 | 53 | 60..=63 => true,
        // XMM registers are ALL caller-saved in System V (none callee-saved)
        _ => false,
    }
}

/// Return `true` if the register is caller-saved (volatile) per System V AMD64 ABI.
pub fn x86_is_caller_saved(reg: X86PReg) -> bool {
    let e = reg.encoding();
    match e {
        // GPR64: RAX=0, RCX=1, RDX=2, RSI=6, RDI=7, R8-R11=8-11
        0..=2 | 6..=11 => true,
        // GPR32 aliases
        16..=18 | 22..=27 => true,
        // GPR16 aliases
        32..=34 | 38..=43 => true,
        // GPR8 aliases
        48..=50 | 54..=59 => true,
        // All XMM registers are caller-saved in System V
        64..=79 => true,
        _ => false,
    }
}

/// Convert a 64-bit GPR to its 32-bit alias.
///
/// Returns `None` if the input is not a GPR64.
pub fn x86_gpr64_to_gpr32(reg: X86PReg) -> Option<X86PReg> {
    if reg.0 <= 15 {
        Some(X86PReg(reg.0 + 16))
    } else {
        None
    }
}

/// Convert a 32-bit GPR to its 64-bit alias.
///
/// Returns `None` if the input is not a GPR32.
pub fn x86_gpr32_to_gpr64(reg: X86PReg) -> Option<X86PReg> {
    if reg.0 >= 16 && reg.0 <= 31 {
        Some(X86PReg(reg.0 - 16))
    } else {
        None
    }
}

/// Return `true` if two x86-64 registers overlap (share physical storage).
///
/// For example, RAX overlaps EAX, AX, AL.
pub fn x86_regs_overlap(a: X86PReg, b: X86PReg) -> bool {
    if a == b {
        return true;
    }
    let a_root = x86_reg_root(a);
    let b_root = x86_reg_root(b);
    match (a_root, b_root) {
        (Some((num_a, group_a)), Some((num_b, group_b))) => {
            num_a == num_b && group_a == group_b
        }
        _ => false,
    }
}

/// Return the root register number and class group (0=GPR, 1=XMM).
fn x86_reg_root(reg: X86PReg) -> Option<(u8, u8)> {
    let e = reg.encoding();
    match e {
        0..=15 => Some((e as u8, 0)),         // GPR64
        16..=31 => Some(((e - 16) as u8, 0)), // GPR32 aliases GPR64
        32..=47 => Some(((e - 32) as u8, 0)), // GPR16 aliases GPR64
        48..=63 => Some(((e - 48) as u8, 0)), // GPR8 aliases GPR64
        64..=79 => Some(((e - 64) as u8, 1)), // XMM
        _ => None,
    }
}

/// Convert a 64-bit GPR to its 16-bit alias.
///
/// Returns `None` if the input is not a GPR64.
pub fn x86_gpr64_to_gpr16(reg: X86PReg) -> Option<X86PReg> {
    if reg.0 <= 15 {
        Some(X86PReg(reg.0 + 32))
    } else {
        None
    }
}

/// Convert a 64-bit GPR to its 8-bit alias.
///
/// Returns `None` if the input is not a GPR64.
pub fn x86_gpr64_to_gpr8(reg: X86PReg) -> Option<X86PReg> {
    if reg.0 <= 15 {
        Some(X86PReg(reg.0 + 48))
    } else {
        None
    }
}

/// Convert a 16-bit GPR to its 64-bit alias.
///
/// Returns `None` if the input is not a GPR16.
pub fn x86_gpr16_to_gpr64(reg: X86PReg) -> Option<X86PReg> {
    if reg.0 >= 32 && reg.0 <= 47 {
        Some(X86PReg(reg.0 - 32))
    } else {
        None
    }
}

/// Convert an 8-bit GPR to its 64-bit alias.
///
/// Returns `None` if the input is not a GPR8.
pub fn x86_gpr8_to_gpr64(reg: X86PReg) -> Option<X86PReg> {
    if reg.0 >= 48 && reg.0 <= 63 {
        Some(X86PReg(reg.0 - 48))
    } else {
        None
    }
}

/// Convert a 32-bit GPR to its 16-bit alias.
///
/// Returns `None` if the input is not a GPR32.
pub fn x86_gpr32_to_gpr16(reg: X86PReg) -> Option<X86PReg> {
    if reg.0 >= 16 && reg.0 <= 31 {
        Some(X86PReg(reg.0 + 16))
    } else {
        None
    }
}

/// Convert a 32-bit GPR to its 8-bit alias.
///
/// Returns `None` if the input is not a GPR32.
pub fn x86_gpr32_to_gpr8(reg: X86PReg) -> Option<X86PReg> {
    if reg.0 >= 16 && reg.0 <= 31 {
        Some(X86PReg(reg.0 + 32))
    } else {
        None
    }
}

/// Return the register number (0-15) for any register in its class.
///
/// This is the logical index within the register file and matches the
/// hardware encoding for GPR and XMM registers. Returns `None` for
/// system registers (RFLAGS, RIP).
pub fn x86_reg_number(reg: X86PReg) -> Option<u8> {
    let e = reg.encoding();
    match e {
        0..=15 => Some(e as u8),           // GPR64
        16..=31 => Some((e - 16) as u8),   // GPR32
        32..=47 => Some((e - 32) as u8),   // GPR16
        48..=63 => Some((e - 48) as u8),   // GPR8
        64..=79 => Some((e - 64) as u8),   // XMM
        _ => None,
    }
}

/// Try to parse an x86-64 register from its assembly name.
///
/// Accepts lowercase names: `"rax"`, `"eax"`, `"ax"`, `"al"`, `"xmm0"`,
/// `"rflags"`, `"rip"`, etc.
///
/// Returns `None` if the name is not recognized.
pub fn x86_preg_from_name(name: &str) -> Option<X86PReg> {
    // GPR64
    for (i, &n) in GPR64_NAMES.iter().enumerate() {
        if n == name {
            return Some(X86PReg(i as u16));
        }
    }
    // GPR32
    for (i, &n) in GPR32_NAMES.iter().enumerate() {
        if n == name {
            return Some(X86PReg(16 + i as u16));
        }
    }
    // GPR16
    for (i, &n) in GPR16_NAMES.iter().enumerate() {
        if n == name {
            return Some(X86PReg(32 + i as u16));
        }
    }
    // GPR8
    for (i, &n) in GPR8_NAMES.iter().enumerate() {
        if n == name {
            return Some(X86PReg(48 + i as u16));
        }
    }
    // XMM
    for (i, &n) in XMM_NAMES.iter().enumerate() {
        if n == name {
            return Some(X86PReg(64 + i as u16));
        }
    }
    // Special
    match name {
        "rflags" => Some(RFLAGS),
        "rip" => Some(RIP),
        _ => None,
    }
}

/// Return the 64-bit GPR that contains the given register as a sub-register.
///
/// For any GPR (64/32/16/8-bit), returns the corresponding GPR64.
/// For XMM and system registers, returns `None`.
pub fn x86_containing_gpr64(reg: X86PReg) -> Option<X86PReg> {
    let e = reg.encoding();
    match e {
        0..=15 => Some(reg),                     // Already GPR64
        16..=31 => Some(X86PReg(e - 16)),         // GPR32 -> GPR64
        32..=47 => Some(X86PReg(e - 32)),         // GPR16 -> GPR64
        48..=63 => Some(X86PReg(e - 48)),         // GPR8 -> GPR64
        _ => None,
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
        assert_eq!(x86_preg_name(RAX), "rax");
        assert_eq!(x86_preg_name(RCX), "rcx");
        assert_eq!(x86_preg_name(RDX), "rdx");
        assert_eq!(x86_preg_name(RBX), "rbx");
        assert_eq!(x86_preg_name(RSP), "rsp");
        assert_eq!(x86_preg_name(RBP), "rbp");
        assert_eq!(x86_preg_name(RSI), "rsi");
        assert_eq!(x86_preg_name(RDI), "rdi");
        assert_eq!(x86_preg_name(R8), "r8");
        assert_eq!(x86_preg_name(R15), "r15");
        assert_eq!(x86_preg_name(EAX), "eax");
        assert_eq!(x86_preg_name(R15D), "r15d");
        assert_eq!(x86_preg_name(AL), "al");
        assert_eq!(x86_preg_name(R15B), "r15b");
        assert_eq!(x86_preg_name(XMM0), "xmm0");
        assert_eq!(x86_preg_name(XMM15), "xmm15");
        assert_eq!(x86_preg_name(RFLAGS), "rflags");
        assert_eq!(x86_preg_name(RIP), "rip");
    }

    #[test]
    fn test_preg_classes() {
        assert_eq!(x86_preg_class(RAX), X86RegClass::Gpr64);
        assert_eq!(x86_preg_class(R15), X86RegClass::Gpr64);
        assert_eq!(x86_preg_class(EAX), X86RegClass::Gpr32);
        assert_eq!(x86_preg_class(R15D), X86RegClass::Gpr32);
        assert_eq!(x86_preg_class(AX), X86RegClass::Gpr16);
        assert_eq!(x86_preg_class(AL), X86RegClass::Gpr8);
        assert_eq!(x86_preg_class(XMM0), X86RegClass::Xmm128);
        assert_eq!(x86_preg_class(XMM15), X86RegClass::Xmm128);
        assert_eq!(x86_preg_class(RFLAGS), X86RegClass::System);
        assert_eq!(x86_preg_class(RIP), X86RegClass::System);
    }

    #[test]
    fn test_hw_encoding() {
        assert_eq!(x86_hw_encoding(RAX), 0);
        assert_eq!(x86_hw_encoding(RCX), 1);
        assert_eq!(x86_hw_encoding(RBX), 3);
        assert_eq!(x86_hw_encoding(RSP), 4);
        assert_eq!(x86_hw_encoding(R8), 8);
        assert_eq!(x86_hw_encoding(R15), 15);
        assert_eq!(x86_hw_encoding(EAX), 0);
        assert_eq!(x86_hw_encoding(R15D), 15);
        assert_eq!(x86_hw_encoding(AL), 0);
        assert_eq!(x86_hw_encoding(R15B), 15);
        assert_eq!(x86_hw_encoding(XMM0), 0);
        assert_eq!(x86_hw_encoding(XMM15), 15);
    }

    #[test]
    fn test_callee_saved() {
        // System V callee-saved: RBX, RBP, R12-R15
        assert!(x86_is_callee_saved(RBX));
        assert!(x86_is_callee_saved(RBP));
        assert!(x86_is_callee_saved(R12));
        assert!(x86_is_callee_saved(R13));
        assert!(x86_is_callee_saved(R14));
        assert!(x86_is_callee_saved(R15));

        // NOT callee-saved
        assert!(!x86_is_callee_saved(RAX));
        assert!(!x86_is_callee_saved(RCX));
        assert!(!x86_is_callee_saved(RDX));
        assert!(!x86_is_callee_saved(RSI));
        assert!(!x86_is_callee_saved(RDI));
        assert!(!x86_is_callee_saved(R8));
        assert!(!x86_is_callee_saved(R11));

        // 32-bit aliases of callee-saved
        assert!(x86_is_callee_saved(EBX));
        assert!(x86_is_callee_saved(EBP));
        assert!(x86_is_callee_saved(R12D));

        // XMM are NOT callee-saved in System V
        assert!(!x86_is_callee_saved(XMM0));
        assert!(!x86_is_callee_saved(XMM15));
    }

    #[test]
    fn test_caller_saved() {
        // System V caller-saved: RAX, RCX, RDX, RSI, RDI, R8-R11
        assert!(x86_is_caller_saved(RAX));
        assert!(x86_is_caller_saved(RCX));
        assert!(x86_is_caller_saved(RDX));
        assert!(x86_is_caller_saved(RSI));
        assert!(x86_is_caller_saved(RDI));
        assert!(x86_is_caller_saved(R8));
        assert!(x86_is_caller_saved(R9));
        assert!(x86_is_caller_saved(R10));
        assert!(x86_is_caller_saved(R11));

        // NOT caller-saved (callee-saved)
        assert!(!x86_is_caller_saved(RBX));
        assert!(!x86_is_caller_saved(RBP));
        assert!(!x86_is_caller_saved(R12));
        assert!(!x86_is_caller_saved(R15));

        // All XMM are caller-saved in System V
        assert!(x86_is_caller_saved(XMM0));
        assert!(x86_is_caller_saved(XMM15));
    }

    #[test]
    fn test_gpr_conversion() {
        assert_eq!(x86_gpr64_to_gpr32(RAX), Some(EAX));
        assert_eq!(x86_gpr64_to_gpr32(R15), Some(R15D));
        assert_eq!(x86_gpr64_to_gpr32(RSP), Some(ESP));
        assert_eq!(x86_gpr64_to_gpr32(XMM0), None);

        assert_eq!(x86_gpr32_to_gpr64(EAX), Some(RAX));
        assert_eq!(x86_gpr32_to_gpr64(R15D), Some(R15));
        assert_eq!(x86_gpr32_to_gpr64(ESP), Some(RSP));
        assert_eq!(x86_gpr32_to_gpr64(RAX), None);
    }

    #[test]
    fn test_regs_overlap() {
        assert!(x86_regs_overlap(RAX, EAX));
        assert!(x86_regs_overlap(EAX, RAX));
        assert!(x86_regs_overlap(RAX, AX));
        assert!(x86_regs_overlap(RAX, AL));
        assert!(x86_regs_overlap(EAX, AL));

        assert!(!x86_regs_overlap(RAX, RBX));
        assert!(!x86_regs_overlap(EAX, EBX));
        assert!(!x86_regs_overlap(RAX, XMM0)); // different groups
    }

    #[test]
    fn test_needs_rex() {
        assert!(!RAX.needs_rex());
        assert!(!RBX.needs_rex());
        assert!(R8.needs_rex());
        assert!(R15.needs_rex());
        assert!(R8D.needs_rex());
        assert!(R15D.needs_rex());
        assert!(SPL.needs_rex());
        assert!(BPL.needs_rex());
        assert!(SIL.needs_rex());
        assert!(DIL.needs_rex());
        assert!(!AL.needs_rex());
        assert!(R8B.needs_rex());
        assert!(!XMM0.needs_rex());
        assert!(XMM8.needs_rex());
    }

    #[test]
    fn test_register_class_sizes() {
        assert_eq!(X86RegClass::Gpr64.size_bits(), 64);
        assert_eq!(X86RegClass::Gpr32.size_bits(), 32);
        assert_eq!(X86RegClass::Gpr16.size_bits(), 16);
        assert_eq!(X86RegClass::Gpr8.size_bits(), 8);
        assert_eq!(X86RegClass::Xmm128.size_bits(), 128);

        assert_eq!(X86RegClass::Gpr64.size_bytes(), 8);
        assert_eq!(X86RegClass::Gpr32.size_bytes(), 4);
        assert_eq!(X86RegClass::Xmm128.size_bytes(), 16);
    }

    #[test]
    fn test_allocatable_gprs_exclude_rsp_rbp() {
        assert!(!X86_ALLOCATABLE_GPRS.contains(&RSP));
        assert!(!X86_ALLOCATABLE_GPRS.contains(&RBP));
        assert!(X86_ALLOCATABLE_GPRS.contains(&RAX));
        assert!(X86_ALLOCATABLE_GPRS.contains(&R15));
    }

    #[test]
    fn test_arg_gprs_system_v_order() {
        assert_eq!(X86_ARG_GPRS[0], RDI);
        assert_eq!(X86_ARG_GPRS[1], RSI);
        assert_eq!(X86_ARG_GPRS[2], RDX);
        assert_eq!(X86_ARG_GPRS[3], RCX);
        assert_eq!(X86_ARG_GPRS[4], R8);
        assert_eq!(X86_ARG_GPRS[5], R9);
    }

    #[test]
    fn test_display() {
        assert_eq!(format!("{}", RAX), "rax");
        assert_eq!(format!("{}", RSP), "rsp");
        assert_eq!(format!("{}", XMM15), "xmm15");
        assert_eq!(format!("{}", R8D), "r8d");
    }

    #[test]
    fn test_is_predicates() {
        assert!(RAX.is_gpr64());
        assert!(RAX.is_gpr());
        assert!(!RAX.is_gpr32());
        assert!(!RAX.is_xmm());

        assert!(EAX.is_gpr32());
        assert!(EAX.is_gpr());
        assert!(!EAX.is_gpr64());

        assert!(XMM0.is_xmm());
        assert!(!XMM0.is_gpr());

        // New GPR16/GPR8 predicates
        assert!(AX.is_gpr16());
        assert!(R15W.is_gpr16());
        assert!(!AX.is_gpr64());
        assert!(!AX.is_gpr32());
        assert!(AX.is_gpr());

        assert!(AL.is_gpr8());
        assert!(R15B.is_gpr8());
        assert!(!AL.is_gpr64());
        assert!(!AL.is_gpr16());
        assert!(AL.is_gpr());

        // System register predicates
        assert!(RFLAGS.is_system());
        assert!(RIP.is_system());
        assert!(!RAX.is_system());
        assert!(!XMM0.is_system());
    }

    #[test]
    fn test_gpr16_gpr8_conversion() {
        // GPR64 -> GPR16
        assert_eq!(x86_gpr64_to_gpr16(RAX), Some(AX));
        assert_eq!(x86_gpr64_to_gpr16(R15), Some(R15W));
        assert_eq!(x86_gpr64_to_gpr16(RSP), Some(SP16));
        assert_eq!(x86_gpr64_to_gpr16(XMM0), None);
        assert_eq!(x86_gpr64_to_gpr16(EAX), None);

        // GPR64 -> GPR8
        assert_eq!(x86_gpr64_to_gpr8(RAX), Some(AL));
        assert_eq!(x86_gpr64_to_gpr8(R15), Some(R15B));
        assert_eq!(x86_gpr64_to_gpr8(RSP), Some(SPL));
        assert_eq!(x86_gpr64_to_gpr8(XMM0), None);

        // GPR16 -> GPR64
        assert_eq!(x86_gpr16_to_gpr64(AX), Some(RAX));
        assert_eq!(x86_gpr16_to_gpr64(R15W), Some(R15));
        assert_eq!(x86_gpr16_to_gpr64(SP16), Some(RSP));
        assert_eq!(x86_gpr16_to_gpr64(RAX), None);

        // GPR8 -> GPR64
        assert_eq!(x86_gpr8_to_gpr64(AL), Some(RAX));
        assert_eq!(x86_gpr8_to_gpr64(R15B), Some(R15));
        assert_eq!(x86_gpr8_to_gpr64(SPL), Some(RSP));
        assert_eq!(x86_gpr8_to_gpr64(RAX), None);

        // GPR32 -> GPR16
        assert_eq!(x86_gpr32_to_gpr16(EAX), Some(AX));
        assert_eq!(x86_gpr32_to_gpr16(R15D), Some(R15W));
        assert_eq!(x86_gpr32_to_gpr16(RAX), None);

        // GPR32 -> GPR8
        assert_eq!(x86_gpr32_to_gpr8(EAX), Some(AL));
        assert_eq!(x86_gpr32_to_gpr8(R15D), Some(R15B));
        assert_eq!(x86_gpr32_to_gpr8(RAX), None);
    }

    #[test]
    fn test_reg_number() {
        assert_eq!(x86_reg_number(RAX), Some(0));
        assert_eq!(x86_reg_number(RCX), Some(1));
        assert_eq!(x86_reg_number(R15), Some(15));
        assert_eq!(x86_reg_number(EAX), Some(0));
        assert_eq!(x86_reg_number(R15D), Some(15));
        assert_eq!(x86_reg_number(AX), Some(0));
        assert_eq!(x86_reg_number(AL), Some(0));
        assert_eq!(x86_reg_number(R15B), Some(15));
        assert_eq!(x86_reg_number(XMM0), Some(0));
        assert_eq!(x86_reg_number(XMM15), Some(15));
        assert_eq!(x86_reg_number(RFLAGS), None);
        assert_eq!(x86_reg_number(RIP), None);
    }

    #[test]
    fn test_preg_from_name() {
        // GPR64
        assert_eq!(x86_preg_from_name("rax"), Some(RAX));
        assert_eq!(x86_preg_from_name("rsp"), Some(RSP));
        assert_eq!(x86_preg_from_name("r15"), Some(R15));

        // GPR32
        assert_eq!(x86_preg_from_name("eax"), Some(EAX));
        assert_eq!(x86_preg_from_name("r15d"), Some(R15D));

        // GPR16
        assert_eq!(x86_preg_from_name("ax"), Some(AX));
        assert_eq!(x86_preg_from_name("r8w"), Some(R8W));

        // GPR8
        assert_eq!(x86_preg_from_name("al"), Some(AL));
        assert_eq!(x86_preg_from_name("spl"), Some(SPL));
        assert_eq!(x86_preg_from_name("r15b"), Some(R15B));

        // XMM
        assert_eq!(x86_preg_from_name("xmm0"), Some(XMM0));
        assert_eq!(x86_preg_from_name("xmm15"), Some(XMM15));

        // Special
        assert_eq!(x86_preg_from_name("rflags"), Some(RFLAGS));
        assert_eq!(x86_preg_from_name("rip"), Some(RIP));

        // Unknown
        assert_eq!(x86_preg_from_name("ymm0"), None);
        assert_eq!(x86_preg_from_name(""), None);
    }

    #[test]
    fn test_containing_gpr64() {
        assert_eq!(x86_containing_gpr64(RAX), Some(RAX));
        assert_eq!(x86_containing_gpr64(EAX), Some(RAX));
        assert_eq!(x86_containing_gpr64(AX), Some(RAX));
        assert_eq!(x86_containing_gpr64(AL), Some(RAX));
        assert_eq!(x86_containing_gpr64(R15), Some(R15));
        assert_eq!(x86_containing_gpr64(R15D), Some(R15));
        assert_eq!(x86_containing_gpr64(R15W), Some(R15));
        assert_eq!(x86_containing_gpr64(R15B), Some(R15));
        assert_eq!(x86_containing_gpr64(RSP), Some(RSP));
        assert_eq!(x86_containing_gpr64(ESP), Some(RSP));
        assert_eq!(x86_containing_gpr64(SP16), Some(RSP));
        assert_eq!(x86_containing_gpr64(SPL), Some(RSP));
        assert_eq!(x86_containing_gpr64(XMM0), None);
        assert_eq!(x86_containing_gpr64(RFLAGS), None);
    }

    #[test]
    fn test_is_allocatable() {
        // Allocatable GPRs
        assert!(RAX.is_allocatable());
        assert!(RCX.is_allocatable());
        assert!(R15.is_allocatable());

        // Non-allocatable: RSP, RBP
        assert!(!RSP.is_allocatable());
        assert!(!RBP.is_allocatable());

        // Sub-register aliases of RSP/RBP are also non-allocatable
        assert!(!ESP.is_allocatable());
        assert!(!EBP.is_allocatable());
        assert!(!SP16.is_allocatable());
        assert!(!BP16.is_allocatable());
        assert!(!SPL.is_allocatable());
        assert!(!BPL.is_allocatable());

        // XMM registers are allocatable
        assert!(XMM0.is_allocatable());
        assert!(XMM15.is_allocatable());

        // System registers are not allocatable
        assert!(!RFLAGS.is_allocatable());
        assert!(!RIP.is_allocatable());
    }

    #[test]
    fn test_allocatable_subreg_arrays() {
        // GPR32 allocatable excludes ESP, EBP
        assert!(!X86_ALLOCATABLE_GPR32S.contains(&ESP));
        assert!(!X86_ALLOCATABLE_GPR32S.contains(&EBP));
        assert!(X86_ALLOCATABLE_GPR32S.contains(&EAX));
        assert!(X86_ALLOCATABLE_GPR32S.contains(&R15D));
        assert_eq!(X86_ALLOCATABLE_GPR32S.len(), 14);

        // GPR16 allocatable excludes SP16, BP16
        assert!(!X86_ALLOCATABLE_GPR16S.contains(&SP16));
        assert!(!X86_ALLOCATABLE_GPR16S.contains(&BP16));
        assert!(X86_ALLOCATABLE_GPR16S.contains(&AX));
        assert!(X86_ALLOCATABLE_GPR16S.contains(&R15W));
        assert_eq!(X86_ALLOCATABLE_GPR16S.len(), 14);

        // GPR8 allocatable excludes SPL, BPL
        assert!(!X86_ALLOCATABLE_GPR8S.contains(&SPL));
        assert!(!X86_ALLOCATABLE_GPR8S.contains(&BPL));
        assert!(X86_ALLOCATABLE_GPR8S.contains(&AL));
        assert!(X86_ALLOCATABLE_GPR8S.contains(&R15B));
        assert_eq!(X86_ALLOCATABLE_GPR8S.len(), 14);
    }

    #[test]
    fn test_callee_saved_subreg_consistency() {
        // Every register in callee-saved 32-bit list should be a sub-register
        // of a register in the callee-saved 64-bit list.
        for &gpr32 in &X86_CALLEE_SAVED_GPR32S {
            let gpr64 = x86_gpr32_to_gpr64(gpr32).unwrap();
            assert!(
                X86_CALLEE_SAVED_GPRS.contains(&gpr64),
                "{:?} (from {:?}) not in callee-saved GPRs",
                gpr64, gpr32
            );
        }
    }

    #[test]
    fn test_caller_callee_partition() {
        // For each allocatable GPR, it should be either caller-saved or callee-saved
        // (but not both and not neither). RSP/RBP are excluded from both sets.
        for &gpr in &X86_ALLOCATABLE_GPRS {
            let caller = x86_is_caller_saved(gpr);
            let callee = x86_is_callee_saved(gpr);
            assert!(
                caller ^ callee,
                "{:?} is caller_saved={}, callee_saved={} (expected exactly one)",
                gpr, caller, callee
            );
        }
    }

    #[test]
    fn test_hw_encoding_consistency_across_aliases() {
        // All aliases of the same register should have the same hw encoding.
        for i in 0..16u16 {
            let gpr64 = X86PReg(i);
            let gpr32 = X86PReg(i + 16);
            let gpr16 = X86PReg(i + 32);
            let gpr8 = X86PReg(i + 48);
            let expected = i as u8;
            assert_eq!(x86_hw_encoding(gpr64), expected);
            assert_eq!(x86_hw_encoding(gpr32), expected);
            assert_eq!(x86_hw_encoding(gpr16), expected);
            assert_eq!(x86_hw_encoding(gpr8), expected);
        }
    }

    #[test]
    fn test_reg_class_method_matches_function() {
        // The reg_class() method should return the same thing as x86_preg_class()
        let test_regs = [RAX, EAX, AX, AL, XMM0, RFLAGS, RIP, R15, R15D, R15W, R15B, XMM15];
        for &reg in &test_regs {
            assert_eq!(reg.reg_class(), x86_preg_class(reg), "mismatch for {:?}", reg);
        }
    }
}
