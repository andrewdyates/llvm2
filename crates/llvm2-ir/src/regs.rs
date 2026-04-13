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

impl RegClass {
    /// Select the register class for a given IR type.
    pub fn for_type(ty: crate::function::Type) -> Self {
        use crate::function::Type;
        match ty {
            Type::I8 | Type::I16 | Type::I32 | Type::B1 => RegClass::Gpr32,
            Type::I64 | Type::Ptr | Type::I128 => RegClass::Gpr64,
            Type::F32 => RegClass::Fpr32,
            Type::F64 => RegClass::Fpr64,
        }
    }
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::function::Type;

    // ---- VReg tests ----

    #[test]
    fn vreg_creation() {
        let v = VReg::new(0, RegClass::Gpr64);
        assert_eq!(v.id, 0);
        assert_eq!(v.class, RegClass::Gpr64);
    }

    #[test]
    fn vreg_creation_different_classes() {
        for (id, class) in [
            (0, RegClass::Gpr32),
            (1, RegClass::Gpr64),
            (100, RegClass::Fpr32),
            (u32::MAX, RegClass::Fpr64),
            (42, RegClass::Vec128),
        ] {
            let v = VReg::new(id, class);
            assert_eq!(v.id, id);
            assert_eq!(v.class, class);
        }
    }

    #[test]
    fn vreg_equality() {
        let a = VReg::new(5, RegClass::Gpr64);
        let b = VReg::new(5, RegClass::Gpr64);
        let c = VReg::new(5, RegClass::Gpr32);
        let d = VReg::new(6, RegClass::Gpr64);
        assert_eq!(a, b);
        assert_ne!(a, c); // same id, different class
        assert_ne!(a, d); // different id, same class
    }

    #[test]
    fn vreg_copy_clone() {
        let a = VReg::new(7, RegClass::Fpr32);
        let b = a; // Copy
        let c = a.clone(); // Clone
        assert_eq!(a, b);
        assert_eq!(a, c);
    }

    #[test]
    fn vreg_hash() {
        use std::collections::HashSet;
        let mut set = HashSet::new();
        set.insert(VReg::new(0, RegClass::Gpr64));
        set.insert(VReg::new(0, RegClass::Gpr64)); // duplicate
        set.insert(VReg::new(1, RegClass::Gpr64));
        assert_eq!(set.len(), 2);
    }

    #[test]
    fn vreg_display() {
        assert_eq!(format!("{}", VReg::new(0, RegClass::Gpr64)), "v0");
        assert_eq!(format!("{}", VReg::new(42, RegClass::Fpr32)), "v42");
        assert_eq!(format!("{}", VReg::new(999, RegClass::Vec128)), "v999");
    }

    // ---- PReg tests ----

    #[test]
    fn preg_encoding_constants() {
        // GPRs should have encoding matching their index
        assert_eq!(X0.0, 0);
        assert_eq!(X15.0, 15);
        assert_eq!(X30.0, 30);

        // FPRs should start at 32
        assert_eq!(V0.0, 32);
        assert_eq!(V15.0, 47);
        assert_eq!(V31.0, 63);
    }

    #[test]
    fn preg_is_gpr() {
        assert!(X0.is_gpr());
        assert!(X15.is_gpr());
        assert!(X30.is_gpr());
        assert!(!V0.is_gpr());
        assert!(!V31.is_gpr());
    }

    #[test]
    fn preg_is_fpr() {
        assert!(V0.is_fpr());
        assert!(V15.is_fpr());
        assert!(V31.is_fpr());
        assert!(!X0.is_fpr());
        assert!(!X30.is_fpr());
    }

    #[test]
    fn preg_hw_enc() {
        // GPR hw_enc is identity (0-30)
        assert_eq!(X0.hw_enc(), 0);
        assert_eq!(X15.hw_enc(), 15);
        assert_eq!(X30.hw_enc(), 30);

        // FPR hw_enc subtracts 32
        assert_eq!(V0.hw_enc(), 0);
        assert_eq!(V15.hw_enc(), 15);
        assert_eq!(V31.hw_enc(), 31);
    }

    #[test]
    fn preg_hw_index_is_hw_enc() {
        for r in [X0, X15, X30, V0, V15, V31] {
            assert_eq!(r.hw_index(), r.hw_enc());
        }
    }

    #[test]
    fn preg_display() {
        assert_eq!(format!("{}", X0), "x0");
        assert_eq!(format!("{}", X15), "x15");
        assert_eq!(format!("{}", X29), "fp");
        assert_eq!(format!("{}", X30), "lr");
        assert_eq!(format!("{}", V0), "v0");
        assert_eq!(format!("{}", V31), "v31");
    }

    #[test]
    fn preg_display_unknown() {
        let unknown = PReg(64); // beyond FPR range
        assert_eq!(format!("{}", unknown), "?64");
    }

    #[test]
    fn preg_equality() {
        assert_eq!(X0, PReg(0));
        assert_eq!(V0, PReg(32));
        assert_ne!(X0, X1);
        assert_ne!(X0, V0);
    }

    #[test]
    fn preg_hash() {
        use std::collections::HashSet;
        let mut set = HashSet::new();
        set.insert(X0);
        set.insert(X0); // duplicate
        set.insert(X1);
        assert_eq!(set.len(), 2);
    }

    // ---- RegClass tests ----

    #[test]
    fn regclass_for_type_integer() {
        assert_eq!(RegClass::for_type(Type::I8), RegClass::Gpr32);
        assert_eq!(RegClass::for_type(Type::I16), RegClass::Gpr32);
        assert_eq!(RegClass::for_type(Type::I32), RegClass::Gpr32);
        assert_eq!(RegClass::for_type(Type::B1), RegClass::Gpr32);
        assert_eq!(RegClass::for_type(Type::I64), RegClass::Gpr64);
        assert_eq!(RegClass::for_type(Type::I128), RegClass::Gpr64);
        assert_eq!(RegClass::for_type(Type::Ptr), RegClass::Gpr64);
    }

    #[test]
    fn regclass_for_type_float() {
        assert_eq!(RegClass::for_type(Type::F32), RegClass::Fpr32);
        assert_eq!(RegClass::for_type(Type::F64), RegClass::Fpr64);
    }

    #[test]
    fn regclass_variants_are_distinct() {
        let classes = [
            RegClass::Gpr32,
            RegClass::Gpr64,
            RegClass::Fpr32,
            RegClass::Fpr64,
            RegClass::Vec128,
        ];
        for i in 0..classes.len() {
            for j in (i + 1)..classes.len() {
                assert_ne!(classes[i], classes[j]);
            }
        }
    }

    // ---- SpecialReg tests ----

    #[test]
    fn special_reg_variants() {
        let sp = SpecialReg::SP;
        let xzr = SpecialReg::XZR;
        let wzr = SpecialReg::WZR;

        assert_ne!(sp, xzr);
        assert_ne!(sp, wzr);
        assert_ne!(xzr, wzr);
    }

    #[test]
    fn special_reg_copy_clone_hash() {
        use std::collections::HashSet;
        let a = SpecialReg::SP;
        let b = a; // Copy
        let c = a.clone(); // Clone
        assert_eq!(a, b);
        assert_eq!(a, c);

        let mut set = HashSet::new();
        set.insert(SpecialReg::SP);
        set.insert(SpecialReg::XZR);
        set.insert(SpecialReg::WZR);
        set.insert(SpecialReg::SP); // duplicate
        assert_eq!(set.len(), 3);
    }

    // ---- Allocatable register array tests ----

    #[test]
    fn allocatable_gprs_count() {
        // Should be 26: X0-X15 (16) + X19-X28 (10), minus nothing
        assert_eq!(ALLOCATABLE_GPRS.len(), 26);
    }

    #[test]
    fn allocatable_gprs_excludes_reserved() {
        // X16, X17 (IP scratch), X18 (Apple reserved), X29 (FP), X30 (LR) excluded
        assert!(!ALLOCATABLE_GPRS.contains(&X16));
        assert!(!ALLOCATABLE_GPRS.contains(&X17));
        assert!(!ALLOCATABLE_GPRS.contains(&X18));
        assert!(!ALLOCATABLE_GPRS.contains(&X29));
        assert!(!ALLOCATABLE_GPRS.contains(&X30));
    }

    #[test]
    fn allocatable_gprs_includes_expected() {
        for i in 0..=15 {
            assert!(ALLOCATABLE_GPRS.contains(&PReg(i)));
        }
        for i in 19..=28 {
            assert!(ALLOCATABLE_GPRS.contains(&PReg(i)));
        }
    }

    #[test]
    fn callee_saved_gprs_count() {
        assert_eq!(CALLEE_SAVED_GPRS.len(), 10);
    }

    #[test]
    fn callee_saved_gprs_are_x19_to_x28() {
        for (idx, i) in (19..=28).enumerate() {
            assert_eq!(CALLEE_SAVED_GPRS[idx], PReg(i));
        }
    }

    #[test]
    fn allocatable_fprs_count() {
        assert_eq!(ALLOCATABLE_FPRS.len(), 32);
    }

    #[test]
    fn callee_saved_fprs_count() {
        assert_eq!(CALLEE_SAVED_FPRS.len(), 8);
    }

    #[test]
    fn callee_saved_fprs_are_v8_to_v15() {
        for (idx, i) in (40..=47).enumerate() {
            assert_eq!(CALLEE_SAVED_FPRS[idx], PReg(i));
        }
    }
}
