// llvm2-codegen/target.rs - Target architectures and target-generic info
//
// Author: Andrew Yates <ayates@dropbox.com>
// Copyright 2026 Dropbox, Inc. | License: Apache-2.0

//! Target architecture definitions with per-target register info and calling conventions.
//!
//! This module provides a unified [`Target`] enum and per-target accessors for
//! register allocation constraints, calling convention details, and stack layout
//! parameters. The design is extensible: adding a new target requires adding an
//! enum variant and implementing the per-target methods.

use llvm2_ir::aarch64_regs;
use llvm2_ir::riscv_regs;
use llvm2_ir::x86_64_regs;

/// Supported target architectures.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Target {
    /// x86-64 (AMD64)
    X86_64,
    /// AArch64 (ARM64)
    Aarch64,
    /// RISC-V 64-bit
    Riscv64,
}

impl Target {
    /// Returns the pointer size in bytes for this target.
    pub fn pointer_bytes(self) -> u32 {
        match self {
            Target::X86_64 | Target::Aarch64 | Target::Riscv64 => 8,
        }
    }

    /// Returns the name of this target.
    pub fn name(self) -> &'static str {
        match self {
            Target::X86_64 => "x86_64",
            Target::Aarch64 => "aarch64",
            Target::Riscv64 => "riscv64",
        }
    }

    /// Returns the required stack alignment in bytes.
    pub fn stack_alignment(self) -> u32 {
        match self {
            // Both x86-64 System V and AArch64 require 16-byte stack alignment.
            Target::X86_64 | Target::Aarch64 => 16,
            // RISC-V: 16-byte alignment for RV64.
            Target::Riscv64 => 16,
        }
    }

    /// Returns the number of integer argument-passing registers.
    pub fn num_arg_gprs(self) -> usize {
        match self {
            Target::X86_64 => x86_64_regs::X86_ARG_GPRS.len(),     // 6 (RDI,RSI,RDX,RCX,R8,R9)
            Target::Aarch64 => aarch64_regs::ARG_GPRS.len(),        // 8 (X0-X7)
            Target::Riscv64 => riscv_regs::RISCV_ARG_GPRS.len(), // a0-a7
        }
    }

    /// Returns the number of floating-point argument-passing registers.
    pub fn num_arg_fprs(self) -> usize {
        match self {
            Target::X86_64 => x86_64_regs::X86_ARG_XMMS.len(),     // 8 (XMM0-XMM7)
            Target::Aarch64 => aarch64_regs::ARG_FPRS.len(),        // 8 (V0-V7)
            Target::Riscv64 => riscv_regs::RISCV_ARG_FPRS.len(), // fa0-fa7
        }
    }

    /// Returns the number of callee-saved GPRs.
    pub fn num_callee_saved_gprs(self) -> usize {
        match self {
            Target::X86_64 => x86_64_regs::X86_CALLEE_SAVED_GPRS.len(), // 6 (RBX,RBP,R12-R15)
            Target::Aarch64 => aarch64_regs::CALLEE_SAVED_GPRS.len(),    // 10 (X19-X28)
            Target::Riscv64 => riscv_regs::RISCV_CALLEE_SAVED_GPRS.len(), // s0-s11
        }
    }

    /// Returns the number of allocatable GPRs.
    pub fn num_allocatable_gprs(self) -> usize {
        match self {
            Target::X86_64 => x86_64_regs::X86_ALLOCATABLE_GPRS.len(), // 14
            Target::Aarch64 => aarch64_regs::ALLOCATABLE_GPRS.len(),    // 25
            Target::Riscv64 => riscv_regs::RISCV_ALLOCATABLE_GPRS.len(),
        }
    }

    /// Returns true if this target uses a frame pointer by default.
    ///
    /// Apple AArch64 always requires a frame pointer. x86-64 can omit it
    /// with -fomit-frame-pointer but we default to using it.
    pub fn requires_frame_pointer(self) -> bool {
        match self {
            Target::Aarch64 => true,  // Apple AArch64 mandate
            Target::X86_64 => false,  // Optional, but recommended
            Target::Riscv64 => false,
        }
    }

    /// Returns the calling convention description for this target.
    pub fn calling_convention(self) -> CallingConvention {
        match self {
            Target::Aarch64 => CallingConvention {
                name: "aapcs64",
                num_arg_gprs: 8,
                num_arg_fprs: 8,
                num_ret_gprs: 8,
                num_ret_fprs: 8,
                red_zone_size: 128,
                shadow_space: 0,
            },
            Target::X86_64 => CallingConvention {
                name: "sysv_amd64",
                num_arg_gprs: 6,
                num_arg_fprs: 8,
                num_ret_gprs: 2,
                num_ret_fprs: 2,
                red_zone_size: 128, // System V AMD64 has a 128-byte red zone
                shadow_space: 0,    // No shadow space in System V (Windows x64 has 32 bytes)
            },
            Target::Riscv64 => CallingConvention {
                name: "riscv_lp64d",
                num_arg_gprs: 8,
                num_arg_fprs: 8,
                num_ret_gprs: 2,
                num_ret_fprs: 2,
                red_zone_size: 0, // RISC-V has no red zone
                shadow_space: 0,
            },
        }
    }
}

/// Describes a calling convention's key parameters.
///
/// This is a target-generic description that captures the essential
/// constraints for ABI lowering, without being tied to specific register types.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CallingConvention {
    /// Name of the calling convention (e.g., "sysv_amd64", "aapcs64").
    pub name: &'static str,
    /// Number of integer/pointer argument-passing registers.
    pub num_arg_gprs: usize,
    /// Number of floating-point argument-passing registers.
    pub num_arg_fprs: usize,
    /// Number of integer return-value registers.
    pub num_ret_gprs: usize,
    /// Number of floating-point return-value registers.
    pub num_ret_fprs: usize,
    /// Red zone size in bytes (area below SP that leaf functions may use).
    pub red_zone_size: u32,
    /// Shadow space / home space required above return address (Windows x64 = 32, others = 0).
    pub shadow_space: u32,
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_target_names() {
        assert_eq!(Target::X86_64.name(), "x86_64");
        assert_eq!(Target::Aarch64.name(), "aarch64");
        assert_eq!(Target::Riscv64.name(), "riscv64");
    }

    #[test]
    fn test_pointer_bytes() {
        assert_eq!(Target::X86_64.pointer_bytes(), 8);
        assert_eq!(Target::Aarch64.pointer_bytes(), 8);
        assert_eq!(Target::Riscv64.pointer_bytes(), 8);
    }

    #[test]
    fn test_stack_alignment() {
        assert_eq!(Target::X86_64.stack_alignment(), 16);
        assert_eq!(Target::Aarch64.stack_alignment(), 16);
        assert_eq!(Target::Riscv64.stack_alignment(), 16);
    }

    #[test]
    fn test_arg_register_counts() {
        // System V: 6 GPR args, 8 XMM args
        assert_eq!(Target::X86_64.num_arg_gprs(), 6);
        assert_eq!(Target::X86_64.num_arg_fprs(), 8);

        // AAPCS64: 8 GPR args, 8 FPR args
        assert_eq!(Target::Aarch64.num_arg_gprs(), 8);
        assert_eq!(Target::Aarch64.num_arg_fprs(), 8);
    }

    #[test]
    fn test_callee_saved_counts() {
        // System V: RBX, RBP, R12-R15 = 6
        assert_eq!(Target::X86_64.num_callee_saved_gprs(), 6);
        // AAPCS64: X19-X28 = 10
        assert_eq!(Target::Aarch64.num_callee_saved_gprs(), 10);
    }

    #[test]
    fn test_allocatable_gprs() {
        // x86-64: 16 GPRs - RSP - RBP = 14
        assert_eq!(Target::X86_64.num_allocatable_gprs(), 14);
        // AArch64: 25 (excludes X8, X16-X18, X29, X30)
        assert_eq!(Target::Aarch64.num_allocatable_gprs(), 25);
    }

    #[test]
    fn test_frame_pointer_requirement() {
        // Apple AArch64 requires frame pointer
        assert!(Target::Aarch64.requires_frame_pointer());
        // x86-64 does not require it
        assert!(!Target::X86_64.requires_frame_pointer());
    }

    #[test]
    fn test_calling_convention_aarch64() {
        let cc = Target::Aarch64.calling_convention();
        assert_eq!(cc.name, "aapcs64");
        assert_eq!(cc.num_arg_gprs, 8);
        assert_eq!(cc.num_arg_fprs, 8);
        assert_eq!(cc.num_ret_gprs, 8);
        assert_eq!(cc.num_ret_fprs, 8);
        assert_eq!(cc.red_zone_size, 128);
        assert_eq!(cc.shadow_space, 0);
    }

    #[test]
    fn test_calling_convention_x86_64() {
        let cc = Target::X86_64.calling_convention();
        assert_eq!(cc.name, "sysv_amd64");
        assert_eq!(cc.num_arg_gprs, 6);
        assert_eq!(cc.num_arg_fprs, 8);
        assert_eq!(cc.num_ret_gprs, 2);
        assert_eq!(cc.num_ret_fprs, 2);
        assert_eq!(cc.red_zone_size, 128);
        assert_eq!(cc.shadow_space, 0);
    }

    #[test]
    fn test_calling_convention_riscv64() {
        let cc = Target::Riscv64.calling_convention();
        assert_eq!(cc.name, "riscv_lp64d");
        assert_eq!(cc.num_arg_gprs, 8);
        assert_eq!(cc.red_zone_size, 0);
    }

    #[test]
    fn test_riscv64_arg_register_counts() {
        // RISC-V LP64D: 8 GPR args (a0-a7), 8 FPR args (fa0-fa7)
        assert_eq!(Target::Riscv64.num_arg_gprs(), 8);
        assert_eq!(Target::Riscv64.num_arg_fprs(), 8);
    }

    #[test]
    fn test_riscv64_callee_saved_count() {
        // RISC-V: s0-s11 = 12
        assert_eq!(Target::Riscv64.num_callee_saved_gprs(), 12);
    }

    #[test]
    fn test_riscv64_allocatable_gprs() {
        // RISC-V: 32 GPRs - x0/zero - x2/sp - x3/gp - x4/tp = 28
        assert_eq!(Target::Riscv64.num_allocatable_gprs(), 28);
    }

    #[test]
    fn test_riscv64_frame_pointer() {
        assert!(!Target::Riscv64.requires_frame_pointer());
    }

    #[test]
    fn test_target_equality() {
        assert_eq!(Target::X86_64, Target::X86_64);
        assert_ne!(Target::X86_64, Target::Aarch64);
        assert_ne!(Target::Riscv64, Target::Aarch64);
    }
}
