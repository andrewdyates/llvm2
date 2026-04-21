// llvm2-regalloc/x86_adapter.rs - x86-64 register allocator adapter
//
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Copyright 2026 Dropbox, Inc. | License: Apache-2.0
//
// Reference: System V AMD64 ABI (https://refspecs.linuxbase.org/elf/x86_64-abi-0.99.pdf)
// Reference: ~/llvm-project-ref/llvm/lib/Target/X86/X86RegisterInfo.td

//! x86-64 register allocator adapter.
//!
//! This module bridges the x86-64 register model ([`llvm2_ir::x86_64_regs`])
//! to the target-independent register allocator infrastructure which operates
//! on [`PReg`] and [`RegClass`].
//!
//! ## Encoding scheme
//!
//! The regalloc uses `PReg(u16)` as an abstract register identifier. AArch64
//! occupies encodings 0..=228. To avoid collision, x86-64 registers are
//! encoded in a separate range:
//!
//! | Range       | x86-64 class | Registers       |
//! |-------------|-------------- |-----------------|
//! | 512..=527   | GPR64         | RAX-R15         |
//! | 528..=543   | GPR32         | EAX-R15D        |
//! | 544..=559   | XMM128        | XMM0-XMM15      |
//!
//! ## Register aliasing
//!
//! x86-64 has register aliasing: EAX aliases the low 32 bits of RAX. The
//! adapter exposes this through [`x86_preg_aliases`] so the allocator can
//! model interference between different-width views of the same physical
//! register.
//!
//! ## Two-address constraints
//!
//! x86-64 uses two-address form (e.g., `ADD RAX, RBX` means `RAX = RAX + RBX`).
//! This is represented via [`TiedOperand`] constraints that tell the allocator
//! the destination must be the same physical register as a source operand.
//!
//! ## Call clobber sets
//!
//! System V AMD64 ABI defines which registers are caller-saved (clobbered by
//! calls) and callee-saved (preserved across calls). The adapter provides
//! these sets for call-crossing interval analysis.

use crate::machine_types::{PReg, RegClass};
use llvm2_ir::x86_64_regs::{self, X86PReg};
use std::collections::{HashMap, HashSet};

// ===========================================================================
// PReg encoding constants for x86-64
// ===========================================================================
// Offset from X86PReg encoding to PReg encoding, by class.

/// Base PReg encoding for x86-64 GPR64 registers (RAX=512, RCX=513, ...).
pub const X86_PREG_GPR64_BASE: u16 = 512;
/// Base PReg encoding for x86-64 GPR32 registers (EAX=528, ECX=529, ...).
pub const X86_PREG_GPR32_BASE: u16 = 528;
/// Base PReg encoding for x86-64 XMM registers (XMM0=544, XMM1=545, ...).
pub const X86_PREG_XMM_BASE: u16 = 544;

// ===========================================================================
// X86PReg <-> PReg conversion
// ===========================================================================

/// Convert an [`X86PReg`] to a regalloc [`PReg`].
///
/// Maps x86-64 GPR64 (encoding 0-15) to PReg(512-527),
/// GPR32 (encoding 16-31) to PReg(528-543),
/// and XMM (encoding 64-79) to PReg(544-559).
///
/// Returns `None` for GPR16, GPR8, system registers, and other
/// non-allocatable register classes that the regalloc does not handle.
pub fn x86_to_preg(x86: X86PReg) -> Option<PReg> {
    let enc = x86.encoding();
    match enc {
        0..=15 => Some(PReg::new(X86_PREG_GPR64_BASE + enc)),
        16..=31 => Some(PReg::new(X86_PREG_GPR32_BASE + (enc - 16))),
        64..=79 => Some(PReg::new(X86_PREG_XMM_BASE + (enc - 64))),
        _ => None,
    }
}

/// Convert a regalloc [`PReg`] back to an [`X86PReg`].
///
/// Inverse of [`x86_to_preg`]. Returns `None` if the PReg encoding
/// does not fall within the x86-64 range.
pub fn preg_to_x86(preg: PReg) -> Option<X86PReg> {
    let enc = preg.encoding();
    match enc {
        512..=527 => Some(X86PReg::new(enc - X86_PREG_GPR64_BASE)),
        528..=543 => Some(X86PReg::new(16 + (enc - X86_PREG_GPR32_BASE))),
        544..=559 => Some(X86PReg::new(64 + (enc - X86_PREG_XMM_BASE))),
        _ => None,
    }
}

/// Returns `true` if the given [`PReg`] is in the x86-64 encoding range.
pub fn is_x86_preg(preg: PReg) -> bool {
    let enc = preg.encoding();
    (512..=527).contains(&enc)
        || (528..=543).contains(&enc)
        || (544..=559).contains(&enc)
}

// ===========================================================================
// Allocatable register sets
// ===========================================================================

/// Return allocatable x86-64 physical registers for the register allocator.
///
/// Excludes RSP (stack pointer) and RBP (frame pointer) from GPR64/GPR32.
/// All 16 XMM registers are allocatable.
///
/// The returned `PReg` values use the x86-64 encoding range (512+).
pub fn x86_64_allocatable_regs() -> HashMap<RegClass, Vec<PReg>> {
    let mut regs = HashMap::new();

    // GPR64: 14 allocatable registers (RAX-R15, excluding RSP=4 and RBP=5).
    let gpr64: Vec<PReg> = x86_64_regs::X86_ALLOCATABLE_GPRS
        .iter()
        .filter_map(|r| x86_to_preg(*r))
        .collect();
    regs.insert(RegClass::Gpr64, gpr64);

    // GPR32: 14 allocatable registers (EAX-R15D, excluding ESP and EBP).
    let gpr32: Vec<PReg> = x86_64_regs::X86_ALLOCATABLE_GPR32S
        .iter()
        .filter_map(|r| x86_to_preg(*r))
        .collect();
    regs.insert(RegClass::Gpr32, gpr32);

    // XMM: all 16 XMM registers are allocatable (used for FPR128/FPR64/FPR32).
    let xmm: Vec<PReg> = x86_64_regs::X86_ALLOCATABLE_XMMS
        .iter()
        .filter_map(|r| x86_to_preg(*r))
        .collect();
    // Map XMM to all FP register classes (they share the same physical regs).
    regs.insert(RegClass::Fpr128, xmm.clone());
    regs.insert(RegClass::Fpr64, xmm.clone());
    regs.insert(RegClass::Fpr32, xmm);

    regs
}

// ===========================================================================
// Call clobber sets
// ===========================================================================

/// Return x86-64 caller-saved (volatile) registers as a set of [`PReg`].
///
/// System V AMD64 ABI: RAX, RCX, RDX, RSI, RDI, R8-R11 (GPR),
/// and all XMM0-XMM15 (all XMM are caller-saved in System V).
pub fn x86_64_caller_saved_regs() -> HashSet<PReg> {
    let mut regs = HashSet::new();

    // GPR caller-saved
    for &r in &x86_64_regs::X86_CALLER_SAVED_GPRS {
        if let Some(preg) = x86_to_preg(r) {
            regs.insert(preg);
        }
    }

    // All XMM are caller-saved in System V
    for &r in &x86_64_regs::X86_CALLER_SAVED_XMMS {
        if let Some(preg) = x86_to_preg(r) {
            regs.insert(preg);
        }
    }

    regs
}

/// Return x86-64 callee-saved (non-volatile) registers as a set of [`PReg`].
///
/// System V AMD64 ABI: RBX, RBP, R12-R15.
/// Note: RBP is callee-saved but NOT allocatable (frame pointer).
pub fn x86_64_callee_saved_regs() -> HashSet<PReg> {
    let mut regs = HashSet::new();

    for &r in &x86_64_regs::X86_CALLEE_SAVED_GPRS {
        if let Some(preg) = x86_to_preg(r) {
            regs.insert(preg);
        }
    }

    regs
}

// ===========================================================================
// Register aliasing
// ===========================================================================

/// Return the set of [`PReg`] values that alias (overlap) the given x86-64 register.
///
/// For a GPR64 register like RAX, the aliases include EAX (the 32-bit view).
/// For a GPR32 register like EAX, the aliases include RAX (the 64-bit parent).
/// XMM registers have no sub-register aliases in our model.
///
/// The returned set always includes the register itself.
pub fn x86_preg_aliases(preg: PReg) -> Vec<PReg> {
    let enc = preg.encoding();
    match enc {
        // GPR64 (512..=527): aliases the corresponding GPR32 (528..=543).
        512..=527 => {
            let reg_num = enc - X86_PREG_GPR64_BASE;
            vec![preg, PReg::new(X86_PREG_GPR32_BASE + reg_num)]
        }
        // GPR32 (528..=543): aliases the corresponding GPR64 (512..=527).
        528..=543 => {
            let reg_num = enc - X86_PREG_GPR32_BASE;
            vec![preg, PReg::new(X86_PREG_GPR64_BASE + reg_num)]
        }
        // XMM (544..=559): no sub-register aliases in our model.
        544..=559 => vec![preg],
        // Not an x86-64 register.
        _ => vec![preg],
    }
}

/// Return `true` if two x86-64 [`PReg`] values alias (share physical storage).
///
/// This is the [`PReg`]-level equivalent of [`x86_64_regs::x86_regs_overlap`].
pub fn x86_pregs_overlap(a: PReg, b: PReg) -> bool {
    if a == b {
        return true;
    }
    let aliases = x86_preg_aliases(a);
    aliases.contains(&b)
}

// ===========================================================================
// Two-address constraint
// ===========================================================================

/// Represents a tied operand constraint for x86-64 two-address instructions.
///
/// In x86-64, most ALU instructions are two-address: `ADD dst, src` means
/// `dst = dst + src`. The destination register is both read and written.
/// The register allocator must ensure the destination VReg gets the same
/// physical register as the first source operand.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TiedOperand {
    /// Index of the destination operand (typically 0).
    pub def_idx: u32,
    /// Index of the source operand that must be allocated to the same register.
    pub use_idx: u32,
}

impl TiedOperand {
    /// The standard x86-64 two-address constraint: def[0] is tied to use[0].
    ///
    /// For an instruction like `ADD v2, v0, v1` (three-address ISel form),
    /// the constraint says v2 (def[0]) must be allocated to the same
    /// physical register as v0 (use[0]). The pipeline inserts a MOV copy
    /// if they differ.
    pub const STANDARD: Self = Self {
        def_idx: 0,
        use_idx: 0,
    };
}

/// Return `true` if the given opcode has a two-address constraint.
///
/// Most x86-64 ALU instructions require the destination to be the same
/// register as the first source. Notable exceptions: MOV, LEA, IMUL r,r,imm.
pub fn is_two_address_opcode(opcode: u16) -> bool {
    // Opcode constants from x86_64_ops — using raw u16 values since the
    // regalloc crate doesn't depend on x86_64_ops directly.
    // This table covers the common two-address ALU opcodes.
    //
    // The caller (pipeline) maps X86Opcode enum values to u16 before
    // calling into the regalloc.
    matches!(
        opcode,
        // ADD, SUB, AND, OR, XOR (register-register forms)
        0x01 | 0x29 | 0x21 | 0x09 | 0x31
        // ADD, SUB, AND, OR, XOR (register-immediate forms)
        | 0x81
        // IMUL r,r (two-operand form)
        | 0x0FAF
        // NEG, NOT, INC, DEC (unary in-place)
        | 0xF7
        // Shifts: SHL, SHR, SAR
        | 0xD3 | 0xC1
    )
}

// ===========================================================================
// AllocConfig factory
// ===========================================================================

/// Create an [`AllocConfig`] for x86-64 using the linear scan allocator.
pub fn x86_64_alloc_config() -> crate::AllocConfig {
    crate::AllocConfig {
        allocatable_regs: x86_64_allocatable_regs(),
        strategy: crate::AllocStrategy::LinearScan,
        enable_coalescing: true,
        enable_remat: true,
        enable_spill_slot_reuse: true,
        hints: HashMap::new(),
    }
}

/// Create an [`AllocConfig`] for x86-64 using the greedy allocator.
pub fn x86_64_greedy_alloc_config() -> crate::AllocConfig {
    crate::AllocConfig {
        allocatable_regs: x86_64_allocatable_regs(),
        strategy: crate::AllocStrategy::Greedy,
        enable_coalescing: true,
        enable_remat: true,
        enable_spill_slot_reuse: true,
        hints: HashMap::new(),
    }
}

// ===========================================================================
// Allocation result translation
// ===========================================================================

/// Translate a regalloc allocation result from [`PReg`] to [`X86PReg`].
///
/// Takes the `VReg -> PReg` mapping produced by the allocator and converts
/// each PReg to its corresponding X86PReg. Returns `None` entries for any
/// PReg that doesn't map to an x86-64 register (which would indicate a bug).
pub fn translate_allocation(
    allocation: &HashMap<crate::machine_types::VReg, PReg>,
) -> HashMap<crate::machine_types::VReg, X86PReg> {
    allocation
        .iter()
        .filter_map(|(&vreg, &preg)| {
            preg_to_x86(preg).map(|x86| (vreg, x86))
        })
        .collect()
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use llvm2_ir::x86_64_regs::*;

    // -----------------------------------------------------------------------
    // PReg <-> X86PReg conversion
    // -----------------------------------------------------------------------

    #[test]
    fn test_gpr64_roundtrip() {
        // All 16 GPR64 registers should survive roundtrip conversion.
        for &gpr in &X86_ALL_GPRS {
            let preg = x86_to_preg(gpr).unwrap_or_else(|| panic!("failed to convert {:?}", gpr));
            let back = preg_to_x86(preg).unwrap_or_else(|| panic!("failed to convert back {:?}", preg));
            assert_eq!(gpr, back, "roundtrip failed for {:?}", gpr);
        }
    }

    #[test]
    fn test_gpr32_roundtrip() {
        let gpr32s = [EAX, ECX, EDX, EBX, ESP, EBP, ESI, EDI,
                      R8D, R9D, R10D, R11D, R12D, R13D, R14D, R15D];
        for &gpr in &gpr32s {
            let preg = x86_to_preg(gpr).unwrap_or_else(|| panic!("failed to convert {:?}", gpr));
            let back = preg_to_x86(preg).unwrap_or_else(|| panic!("failed to convert back {:?}", preg));
            assert_eq!(gpr, back, "roundtrip failed for {:?}", gpr);
        }
    }

    #[test]
    fn test_xmm_roundtrip() {
        for &xmm in &X86_ALL_XMMS {
            let preg = x86_to_preg(xmm).unwrap_or_else(|| panic!("failed to convert {:?}", xmm));
            let back = preg_to_x86(preg).unwrap_or_else(|| panic!("failed to convert back {:?}", preg));
            assert_eq!(xmm, back, "roundtrip failed for {:?}", xmm);
        }
    }

    #[test]
    fn test_gpr16_not_convertible() {
        // GPR16 registers (AX, CX, etc.) should not map to PReg.
        assert!(x86_to_preg(AX).is_none());
        assert!(x86_to_preg(R15W).is_none());
    }

    #[test]
    fn test_gpr8_not_convertible() {
        // GPR8 registers (AL, CL, etc.) should not map to PReg.
        assert!(x86_to_preg(AL).is_none());
        assert!(x86_to_preg(R15B).is_none());
    }

    #[test]
    fn test_system_not_convertible() {
        assert!(x86_to_preg(RFLAGS).is_none());
        assert!(x86_to_preg(RIP).is_none());
    }

    #[test]
    fn test_is_x86_preg() {
        let rax_preg = x86_to_preg(RAX).unwrap();
        assert!(is_x86_preg(rax_preg));

        let xmm0_preg = x86_to_preg(XMM0).unwrap();
        assert!(is_x86_preg(xmm0_preg));

        // AArch64 PReg should not be x86.
        assert!(!is_x86_preg(PReg::new(0)));   // X0
        assert!(!is_x86_preg(PReg::new(64)));  // V0
    }

    // -----------------------------------------------------------------------
    // Encoding range correctness
    // -----------------------------------------------------------------------

    #[test]
    fn test_preg_encoding_ranges_disjoint() {
        // x86-64 GPR64 range should not overlap GPR32 or XMM ranges.
        for i in 512..=527u16 {
            assert!(!(528..=543).contains(&i), "GPR64 overlaps GPR32");
            assert!(!(544..=559).contains(&i), "GPR64 overlaps XMM");
        }
        for i in 528..=543u16 {
            assert!(!(544..=559).contains(&i), "GPR32 overlaps XMM");
        }
    }

    #[test]
    fn test_preg_encoding_no_aarch64_collision() {
        // AArch64 uses encodings 0..=228. x86-64 starts at 512.
        let rax_preg = x86_to_preg(RAX).unwrap();
        assert!(rax_preg.encoding() >= 512);
    }

    // -----------------------------------------------------------------------
    // Allocatable register sets
    // -----------------------------------------------------------------------

    #[test]
    fn test_allocatable_gpr64_count() {
        let regs = x86_64_allocatable_regs();
        let gpr64 = regs.get(&RegClass::Gpr64).expect("missing GPR64");
        // 14 allocatable GPRs (16 total minus RSP and RBP).
        assert_eq!(gpr64.len(), 14, "expected 14 allocatable GPR64");
    }

    #[test]
    fn test_allocatable_gpr32_count() {
        let regs = x86_64_allocatable_regs();
        let gpr32 = regs.get(&RegClass::Gpr32).expect("missing GPR32");
        assert_eq!(gpr32.len(), 14, "expected 14 allocatable GPR32");
    }

    #[test]
    fn test_allocatable_xmm_count() {
        let regs = x86_64_allocatable_regs();
        let xmm = regs.get(&RegClass::Fpr128).expect("missing Fpr128/XMM");
        assert_eq!(xmm.len(), 16, "expected 16 allocatable XMM");
    }

    #[test]
    fn test_allocatable_excludes_rsp_rbp() {
        let regs = x86_64_allocatable_regs();
        let gpr64 = regs.get(&RegClass::Gpr64).unwrap();

        let rsp_preg = x86_to_preg(RSP).unwrap();
        let rbp_preg = x86_to_preg(RBP).unwrap();

        assert!(!gpr64.contains(&rsp_preg), "RSP should not be allocatable");
        assert!(!gpr64.contains(&rbp_preg), "RBP should not be allocatable");
    }

    #[test]
    fn test_allocatable_gpr32_excludes_esp_ebp() {
        let regs = x86_64_allocatable_regs();
        let gpr32 = regs.get(&RegClass::Gpr32).unwrap();

        let esp_preg = x86_to_preg(ESP).unwrap();
        let ebp_preg = x86_to_preg(EBP).unwrap();

        assert!(!gpr32.contains(&esp_preg), "ESP should not be allocatable");
        assert!(!gpr32.contains(&ebp_preg), "EBP should not be allocatable");
    }

    // -----------------------------------------------------------------------
    // Call clobber sets
    // -----------------------------------------------------------------------

    #[test]
    fn test_caller_saved_contains_expected() {
        let cs = x86_64_caller_saved_regs();
        // System V: RAX, RCX, RDX, RSI, RDI, R8-R11 are caller-saved.
        assert!(cs.contains(&x86_to_preg(RAX).unwrap()));
        assert!(cs.contains(&x86_to_preg(RCX).unwrap()));
        assert!(cs.contains(&x86_to_preg(RDX).unwrap()));
        assert!(cs.contains(&x86_to_preg(RSI).unwrap()));
        assert!(cs.contains(&x86_to_preg(RDI).unwrap()));
        assert!(cs.contains(&x86_to_preg(R8).unwrap()));
        assert!(cs.contains(&x86_to_preg(R11).unwrap()));
    }

    #[test]
    fn test_caller_saved_excludes_callee_saved() {
        let caller = x86_64_caller_saved_regs();
        // RBX, R12-R15 are callee-saved, not caller-saved.
        assert!(!caller.contains(&x86_to_preg(RBX).unwrap()));
        assert!(!caller.contains(&x86_to_preg(R12).unwrap()));
        assert!(!caller.contains(&x86_to_preg(R15).unwrap()));
    }

    #[test]
    fn test_callee_saved_contains_expected() {
        let cs = x86_64_callee_saved_regs();
        // System V: RBX, RBP, R12-R15 are callee-saved.
        assert!(cs.contains(&x86_to_preg(RBX).unwrap()));
        assert!(cs.contains(&x86_to_preg(RBP).unwrap()));
        assert!(cs.contains(&x86_to_preg(R12).unwrap()));
        assert!(cs.contains(&x86_to_preg(R13).unwrap()));
        assert!(cs.contains(&x86_to_preg(R14).unwrap()));
        assert!(cs.contains(&x86_to_preg(R15).unwrap()));
    }

    #[test]
    fn test_caller_callee_disjoint() {
        let caller = x86_64_caller_saved_regs();
        let callee = x86_64_callee_saved_regs();
        for preg in &caller {
            assert!(
                !callee.contains(preg),
                "PReg {:?} is in both caller-saved and callee-saved",
                preg
            );
        }
    }

    #[test]
    fn test_caller_saved_count() {
        let cs = x86_64_caller_saved_regs();
        // 9 GPRs (RAX, RCX, RDX, RSI, RDI, R8-R11) + 16 XMM = 25.
        assert_eq!(cs.len(), 25, "expected 25 caller-saved regs (9 GPR + 16 XMM)");
    }

    #[test]
    fn test_callee_saved_count() {
        let cs = x86_64_callee_saved_regs();
        // 6 GPRs: RBX, RBP, R12-R15.
        assert_eq!(cs.len(), 6, "expected 6 callee-saved regs");
    }

    // -----------------------------------------------------------------------
    // Register aliasing
    // -----------------------------------------------------------------------

    #[test]
    fn test_gpr64_aliases_gpr32() {
        let rax_preg = x86_to_preg(RAX).unwrap();
        let eax_preg = x86_to_preg(EAX).unwrap();

        let aliases = x86_preg_aliases(rax_preg);
        assert!(aliases.contains(&rax_preg));
        assert!(aliases.contains(&eax_preg));
    }

    #[test]
    fn test_gpr32_aliases_gpr64() {
        let eax_preg = x86_to_preg(EAX).unwrap();
        let rax_preg = x86_to_preg(RAX).unwrap();

        let aliases = x86_preg_aliases(eax_preg);
        assert!(aliases.contains(&eax_preg));
        assert!(aliases.contains(&rax_preg));
    }

    #[test]
    fn test_xmm_no_aliases() {
        let xmm0_preg = x86_to_preg(XMM0).unwrap();
        let aliases = x86_preg_aliases(xmm0_preg);
        assert_eq!(aliases.len(), 1, "XMM should have no sub-register aliases");
        assert_eq!(aliases[0], xmm0_preg);
    }

    #[test]
    fn test_pregs_overlap_same_register() {
        let rax_preg = x86_to_preg(RAX).unwrap();
        assert!(x86_pregs_overlap(rax_preg, rax_preg));
    }

    #[test]
    fn test_pregs_overlap_aliased() {
        let rax_preg = x86_to_preg(RAX).unwrap();
        let eax_preg = x86_to_preg(EAX).unwrap();
        assert!(x86_pregs_overlap(rax_preg, eax_preg));
        assert!(x86_pregs_overlap(eax_preg, rax_preg));
    }

    #[test]
    fn test_pregs_no_overlap_different_regs() {
        let rax_preg = x86_to_preg(RAX).unwrap();
        let rbx_preg = x86_to_preg(RBX).unwrap();
        assert!(!x86_pregs_overlap(rax_preg, rbx_preg));
    }

    #[test]
    fn test_pregs_no_overlap_gpr_xmm() {
        let rax_preg = x86_to_preg(RAX).unwrap();
        let xmm0_preg = x86_to_preg(XMM0).unwrap();
        assert!(!x86_pregs_overlap(rax_preg, xmm0_preg));
    }

    // -----------------------------------------------------------------------
    // Two-address constraints
    // -----------------------------------------------------------------------

    #[test]
    fn test_tied_operand_standard() {
        let tied = TiedOperand::STANDARD;
        assert_eq!(tied.def_idx, 0);
        assert_eq!(tied.use_idx, 0);
    }

    // -----------------------------------------------------------------------
    // AllocConfig factory
    // -----------------------------------------------------------------------

    #[test]
    fn test_x86_64_alloc_config_linear_scan() {
        let config = x86_64_alloc_config();
        assert_eq!(config.strategy, crate::AllocStrategy::LinearScan);
        assert!(config.enable_coalescing);
        assert!(config.enable_remat);
        assert!(config.enable_spill_slot_reuse);

        // Should have GPR64, GPR32, Fpr128, Fpr64, Fpr32.
        assert!(config.allocatable_regs.contains_key(&RegClass::Gpr64));
        assert!(config.allocatable_regs.contains_key(&RegClass::Gpr32));
        assert!(config.allocatable_regs.contains_key(&RegClass::Fpr128));
    }

    #[test]
    fn test_x86_64_greedy_alloc_config() {
        let config = x86_64_greedy_alloc_config();
        assert_eq!(config.strategy, crate::AllocStrategy::Greedy);
    }

    // -----------------------------------------------------------------------
    // Allocation through the real allocator
    // -----------------------------------------------------------------------

    #[test]
    fn test_allocate_with_x86_config() {
        use crate::machine_types::*;

        // Build a simple function with 3 GPR64 VRegs.
        let mut func = RegAllocFunction {
            name: "x86_test".into(),
            insts: vec![
                RegAllocInst {
                    opcode: 1,
                    defs: vec![RegAllocOperand::VReg(VReg::new(0, RegClass::Gpr64))],
                    uses: vec![RegAllocOperand::Imm(42)],
                    implicit_defs: Vec::new(),
                    implicit_uses: Vec::new(),
                    flags: InstFlags::default(),
                },
                RegAllocInst {
                    opcode: 1,
                    defs: vec![RegAllocOperand::VReg(VReg::new(1, RegClass::Gpr64))],
                    uses: vec![RegAllocOperand::Imm(99)],
                    implicit_defs: Vec::new(),
                    implicit_uses: Vec::new(),
                    flags: InstFlags::default(),
                },
                RegAllocInst {
                    opcode: 2,
                    defs: vec![RegAllocOperand::VReg(VReg::new(2, RegClass::Gpr64))],
                    uses: vec![
                        RegAllocOperand::VReg(VReg::new(0, RegClass::Gpr64)),
                        RegAllocOperand::VReg(VReg::new(1, RegClass::Gpr64)),
                    ],
                    implicit_defs: Vec::new(),
                    implicit_uses: Vec::new(),
                    flags: InstFlags::default(),
                },
                RegAllocInst {
                    opcode: 3,
                    defs: vec![],
                    uses: vec![RegAllocOperand::VReg(VReg::new(2, RegClass::Gpr64))],
                    implicit_defs: Vec::new(),
                    implicit_uses: Vec::new(),
                    flags: InstFlags::default(),
                },
            ],
            blocks: vec![RegAllocBlock {
                insts: vec![InstId(0), InstId(1), InstId(2), InstId(3)],
                preds: Vec::new(),
                succs: Vec::new(),
                loop_depth: 0,
            }],
            block_order: vec![BlockId(0)],
            entry_block: BlockId(0),
            next_vreg: 3,
            next_stack_slot: 0,
            stack_slots: std::collections::HashMap::new(),
        };

        let config = x86_64_alloc_config();
        let result = crate::allocate(&mut func, &config).expect("x86-64 allocation failed");

        // All 3 VRegs should be allocated.
        assert_eq!(result.allocation.len(), 3);
        assert!(result.spills.is_empty());

        // All allocated PRegs should be in the x86-64 GPR64 range.
        for &preg in result.allocation.values() {
            assert!(
                is_x86_preg(preg),
                "allocated PReg {:?} is not in x86-64 range",
                preg
            );
        }
    }

    #[test]
    fn test_allocate_with_x86_greedy() {
        use crate::machine_types::*;

        let mut func = RegAllocFunction {
            name: "x86_greedy_test".into(),
            insts: vec![
                RegAllocInst {
                    opcode: 1,
                    defs: vec![RegAllocOperand::VReg(VReg::new(0, RegClass::Gpr64))],
                    uses: vec![RegAllocOperand::Imm(1)],
                    implicit_defs: Vec::new(),
                    implicit_uses: Vec::new(),
                    flags: InstFlags::default(),
                },
                RegAllocInst {
                    opcode: 2,
                    defs: vec![],
                    uses: vec![RegAllocOperand::VReg(VReg::new(0, RegClass::Gpr64))],
                    implicit_defs: Vec::new(),
                    implicit_uses: Vec::new(),
                    flags: InstFlags::default(),
                },
            ],
            blocks: vec![RegAllocBlock {
                insts: vec![InstId(0), InstId(1)],
                preds: Vec::new(),
                succs: Vec::new(),
                loop_depth: 0,
            }],
            block_order: vec![BlockId(0)],
            entry_block: BlockId(0),
            next_vreg: 1,
            next_stack_slot: 0,
            stack_slots: std::collections::HashMap::new(),
        };

        let config = x86_64_greedy_alloc_config();
        let result = crate::allocate(&mut func, &config).expect("x86-64 greedy allocation failed");

        assert_eq!(result.allocation.len(), 1);
        assert!(result.spills.is_empty());
    }

    #[test]
    fn test_translate_allocation() {
        use crate::machine_types::VReg;

        let v0 = VReg::new(0, RegClass::Gpr64);
        let v1 = VReg::new(1, RegClass::Gpr64);
        let rax_preg = x86_to_preg(RAX).unwrap();
        let rbx_preg = x86_to_preg(RBX).unwrap();

        let mut allocation = HashMap::new();
        allocation.insert(v0, rax_preg);
        allocation.insert(v1, rbx_preg);

        let x86_alloc = translate_allocation(&allocation);
        assert_eq!(x86_alloc.len(), 2);
        assert_eq!(x86_alloc[&v0], RAX);
        assert_eq!(x86_alloc[&v1], RBX);
    }

    #[test]
    fn test_high_pressure_causes_spills() {
        use crate::machine_types::*;

        // Create 16 simultaneously live GPR64 VRegs — only 14 allocatable,
        // so at least 2 must spill.
        let n = 16u32;
        let mut insts = Vec::new();
        let mut inst_ids = Vec::new();

        for i in 0..n {
            let inst = RegAllocInst {
                opcode: 1,
                defs: vec![RegAllocOperand::VReg(VReg::new(i, RegClass::Gpr64))],
                uses: vec![RegAllocOperand::Imm(i as i64)],
                implicit_defs: Vec::new(),
                implicit_uses: Vec::new(),
                flags: InstFlags::default(),
            };
            inst_ids.push(InstId(insts.len() as u32));
            insts.push(inst);
        }

        for i in 0..n {
            let inst = RegAllocInst {
                opcode: 2,
                defs: vec![],
                uses: vec![RegAllocOperand::VReg(VReg::new(i, RegClass::Gpr64))],
                implicit_defs: Vec::new(),
                implicit_uses: Vec::new(),
                flags: InstFlags::default(),
            };
            inst_ids.push(InstId(insts.len() as u32));
            insts.push(inst);
        }

        let mut func = RegAllocFunction {
            name: "high_pressure".into(),
            insts,
            blocks: vec![RegAllocBlock {
                insts: inst_ids,
                preds: Vec::new(),
                succs: Vec::new(),
                loop_depth: 0,
            }],
            block_order: vec![BlockId(0)],
            entry_block: BlockId(0),
            next_vreg: n,
            next_stack_slot: 0,
            stack_slots: std::collections::HashMap::new(),
        };

        let config = x86_64_alloc_config();
        let result = crate::allocate(&mut func, &config).expect("allocation failed");

        let total = result.allocation.len() + result.spills.len();
        assert!(total > 0);
        assert!(
            result.allocation.len() <= 14,
            "cannot allocate more than 14 GPR64: got {}",
            result.allocation.len()
        );
    }

    #[test]
    fn test_xmm_allocation() {
        use crate::machine_types::*;

        // Allocate 2 FPR64 (XMM) VRegs.
        let mut func = RegAllocFunction {
            name: "xmm_test".into(),
            insts: vec![
                RegAllocInst {
                    opcode: 1,
                    defs: vec![RegAllocOperand::VReg(VReg::new(0, RegClass::Fpr64))],
                    uses: vec![RegAllocOperand::FImm(1.0)],
                    implicit_defs: Vec::new(),
                    implicit_uses: Vec::new(),
                    flags: InstFlags::default(),
                },
                RegAllocInst {
                    opcode: 1,
                    defs: vec![RegAllocOperand::VReg(VReg::new(1, RegClass::Fpr64))],
                    uses: vec![RegAllocOperand::FImm(2.0)],
                    implicit_defs: Vec::new(),
                    implicit_uses: Vec::new(),
                    flags: InstFlags::default(),
                },
                RegAllocInst {
                    opcode: 2,
                    defs: vec![],
                    uses: vec![
                        RegAllocOperand::VReg(VReg::new(0, RegClass::Fpr64)),
                        RegAllocOperand::VReg(VReg::new(1, RegClass::Fpr64)),
                    ],
                    implicit_defs: Vec::new(),
                    implicit_uses: Vec::new(),
                    flags: InstFlags::default(),
                },
            ],
            blocks: vec![RegAllocBlock {
                insts: vec![InstId(0), InstId(1), InstId(2)],
                preds: Vec::new(),
                succs: Vec::new(),
                loop_depth: 0,
            }],
            block_order: vec![BlockId(0)],
            entry_block: BlockId(0),
            next_vreg: 2,
            next_stack_slot: 0,
            stack_slots: std::collections::HashMap::new(),
        };

        let config = x86_64_alloc_config();
        let result = crate::allocate(&mut func, &config).expect("XMM allocation failed");

        assert_eq!(result.allocation.len(), 2);
        assert!(result.spills.is_empty());

        // Both should be in the XMM range.
        for &preg in result.allocation.values() {
            let enc = preg.encoding();
            assert!(
                (X86_PREG_XMM_BASE..=X86_PREG_XMM_BASE + 15).contains(&enc),
                "expected XMM preg, got encoding {}",
                enc
            );
        }
    }

    #[test]
    fn test_mixed_gpr_xmm_allocation() {
        use crate::machine_types::*;

        // Mix GPR64 and FPR64 VRegs.
        let mut func = RegAllocFunction {
            name: "mixed_test".into(),
            insts: vec![
                RegAllocInst {
                    opcode: 1,
                    defs: vec![RegAllocOperand::VReg(VReg::new(0, RegClass::Gpr64))],
                    uses: vec![RegAllocOperand::Imm(1)],
                    implicit_defs: Vec::new(),
                    implicit_uses: Vec::new(),
                    flags: InstFlags::default(),
                },
                RegAllocInst {
                    opcode: 1,
                    defs: vec![RegAllocOperand::VReg(VReg::new(1, RegClass::Fpr64))],
                    uses: vec![RegAllocOperand::FImm(2.0)],
                    implicit_defs: Vec::new(),
                    implicit_uses: Vec::new(),
                    flags: InstFlags::default(),
                },
                RegAllocInst {
                    opcode: 2,
                    defs: vec![],
                    uses: vec![
                        RegAllocOperand::VReg(VReg::new(0, RegClass::Gpr64)),
                        RegAllocOperand::VReg(VReg::new(1, RegClass::Fpr64)),
                    ],
                    implicit_defs: Vec::new(),
                    implicit_uses: Vec::new(),
                    flags: InstFlags::default(),
                },
            ],
            blocks: vec![RegAllocBlock {
                insts: vec![InstId(0), InstId(1), InstId(2)],
                preds: Vec::new(),
                succs: Vec::new(),
                loop_depth: 0,
            }],
            block_order: vec![BlockId(0)],
            entry_block: BlockId(0),
            next_vreg: 2,
            next_stack_slot: 0,
            stack_slots: std::collections::HashMap::new(),
        };

        let config = x86_64_alloc_config();
        let result = crate::allocate(&mut func, &config).expect("mixed allocation failed");

        assert_eq!(result.allocation.len(), 2);

        // v0 (GPR64) should be in GPR range, v1 (Fpr64) should be in XMM range.
        let v0 = VReg::new(0, RegClass::Gpr64);
        let v1 = VReg::new(1, RegClass::Fpr64);
        let v0_enc = result.allocation[&v0].encoding();
        let v1_enc = result.allocation[&v1].encoding();

        assert!(
            (X86_PREG_GPR64_BASE..=X86_PREG_GPR64_BASE + 15).contains(&v0_enc),
            "v0 should be GPR64, got encoding {}",
            v0_enc
        );
        assert!(
            (X86_PREG_XMM_BASE..=X86_PREG_XMM_BASE + 15).contains(&v1_enc),
            "v1 should be XMM, got encoding {}",
            v1_enc
        );
    }

    #[test]
    fn test_call_crossing_with_x86_regs() {
        use crate::call_clobber;
        use crate::liveness::LiveInterval;
        use crate::machine_types::*;

        // Build function: def v0, call, use v0.
        let insts = vec![
            RegAllocInst {
                opcode: 1,
                defs: vec![RegAllocOperand::VReg(VReg::new(0, RegClass::Gpr64))],
                uses: vec![RegAllocOperand::Imm(42)],
                implicit_defs: Vec::new(),
                implicit_uses: Vec::new(),
                flags: InstFlags::default(),
            },
            RegAllocInst {
                opcode: 0xCA,
                defs: vec![],
                uses: vec![],
                implicit_defs: Vec::new(),
                implicit_uses: Vec::new(),
                flags: InstFlags::IS_CALL,
            },
            RegAllocInst {
                opcode: 2,
                defs: vec![],
                uses: vec![RegAllocOperand::VReg(VReg::new(0, RegClass::Gpr64))],
                implicit_defs: Vec::new(),
                implicit_uses: Vec::new(),
                flags: InstFlags::default(),
            },
        ];

        let func = RegAllocFunction {
            name: "call_test".into(),
            insts,
            blocks: vec![RegAllocBlock {
                insts: vec![InstId(0), InstId(1), InstId(2)],
                preds: Vec::new(),
                succs: Vec::new(),
                loop_depth: 0,
            }],
            block_order: vec![BlockId(0)],
            entry_block: BlockId(0),
            next_vreg: 1,
            next_stack_slot: 0,
            stack_slots: std::collections::HashMap::new(),
        };

        let numbering: HashMap<InstId, u32> =
            (0..3).map(|i| (InstId(i), i)).collect();

        let mut interval = LiveInterval::new(VReg::new(0, RegClass::Gpr64));
        interval.add_range(0, 3);
        let intervals = HashMap::from([(0u32, interval)]);

        let crossings = call_clobber::find_call_crossings(&func, &intervals, &numbering);
        assert_eq!(crossings.len(), 1);
        assert_eq!(crossings[0].live_across.len(), 1);

        // Compute hints with x86-64 callee-saved set.
        let callee_saved = x86_64_callee_saved_regs();
        let allocatable = x86_64_allocatable_regs();
        let hints = call_clobber::compute_call_crossing_hints(
            &crossings,
            &callee_saved,
            &allocatable,
        );

        // v0 should get callee-saved hints.
        let v0 = VReg::new(0, RegClass::Gpr64);
        assert!(hints.contains_key(&v0), "v0 should have callee-saved hints");
        for &hint_preg in &hints[&v0] {
            assert!(
                callee_saved.contains(&hint_preg),
                "hint {:?} should be callee-saved",
                hint_preg
            );
        }
    }

    #[test]
    fn test_empty_function_x86() {
        use crate::machine_types::*;

        let mut func = RegAllocFunction {
            name: "empty_x86".into(),
            insts: Vec::new(),
            blocks: vec![RegAllocBlock {
                insts: Vec::new(),
                preds: Vec::new(),
                succs: Vec::new(),
                loop_depth: 0,
            }],
            block_order: vec![BlockId(0)],
            entry_block: BlockId(0),
            next_vreg: 0,
            next_stack_slot: 0,
            stack_slots: std::collections::HashMap::new(),
        };

        let config = x86_64_alloc_config();
        let result = crate::allocate(&mut func, &config).expect("empty function failed");
        assert!(result.allocation.is_empty());
        assert!(result.spills.is_empty());
    }
}
