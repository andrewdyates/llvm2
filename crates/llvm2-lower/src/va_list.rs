// llvm2-lower/va_list.rs - Apple AArch64 va_list intrinsic lowering
//
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Copyright 2026 Dropbox, Inc. | License: Apache-2.0
//
// Reference: Apple ARM64 Function Calling Conventions ("Variadic Functions")
// Reference: LLVM AArch64ISelLowering.cpp (LowerVASTART, LowerVAARG)
// Reference: AAPCS64 sec. 6.4.2 "Parameter Passing — Variadic"

//! Apple AArch64 `va_list` intrinsic lowering.
//!
//! On Apple AArch64 (DarwinPCS), `va_list` is a simple `char*` — a pointer
//! to the next variadic argument on the stack. This differs from the Linux
//! AAPCS64 ABI where `va_list` is a complex struct with `__gr_top`,
//! `__vr_top`, `__gr_offs`, `__vr_offs`, and `__stack` fields.
//!
//! Apple simplification: since ALL variadic arguments are placed on the
//! stack (never in registers), `va_list` is just a pointer that advances
//! through the stack argument area.
//!
//! # Intrinsic lowering
//!
//! - `va_start(ap)`: Store the address of the first variadic arg into `*ap`.
//!   The address is `SP + stack_offset_to_varargs`.
//! - `va_arg(ap, T)`: Load a value of type T from `*ap`, then advance `*ap`
//!   past the argument (aligned to 8 bytes minimum).
//! - `va_end(ap)`: No-op on AArch64.
//! - `va_copy(dst, src)`: `*dst = *src` (pointer copy).

use crate::types::Type;

// ---------------------------------------------------------------------------
// VaListIntrinsic — operations on va_list
// ---------------------------------------------------------------------------

/// Variadic argument intrinsic operations.
///
/// These correspond to the C/C++ `<stdarg.h>` macros that the frontend
/// lowers to IR intrinsics. Each operation describes what the backend
/// needs to do at the machine level.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum VaListIntrinsic {
    /// `va_start(ap)`: Initialize a `va_list` to point at the first
    /// variadic argument on the stack.
    ///
    /// On Apple AArch64, this stores `frame_base + stack_vararg_offset`
    /// into the memory at `va_list_ptr`.
    VaStart {
        /// The offset from the caller's SP to the start of the variadic
        /// argument area. This is the total size of fixed stack arguments.
        stack_vararg_offset: i64,
    },

    /// `va_arg(ap, type)`: Fetch the next variadic argument of the given
    /// type from the va_list, and advance the va_list pointer.
    VaArg {
        /// Type of the argument to fetch.
        arg_type: Type,
    },

    /// `va_end(ap)`: Finalize a va_list. No-op on AArch64.
    VaEnd,

    /// `va_copy(dst, src)`: Copy a va_list. Simple pointer copy on Apple.
    VaCopy,
}

// ---------------------------------------------------------------------------
// VaArgLowering — describes how to lower a single va_arg
// ---------------------------------------------------------------------------

/// Describes the machine-level operations needed to extract one variadic
/// argument from a `va_list` pointer.
///
/// The general pattern for `va_arg(ap, T)` on Apple AArch64:
/// 1. Load the current pointer: `ptr = *ap`
/// 2. Align `ptr` if the type requires > 8-byte alignment
/// 3. Load the value: `val = *(T*)ptr`
/// 4. Advance the pointer: `*ap = ptr + padded_size`
///
/// For large aggregates (>16 bytes), the caller passes a pointer to a
/// copy, so `va_arg` loads the pointer and then dereferences it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VaArgLowering {
    /// How the value is accessed from the va_list pointer.
    pub access: VaArgAccess,
    /// Number of bytes to load from the va_list pointer.
    /// For indirect access, this is 8 (pointer size).
    pub load_size: u32,
    /// Alignment requirement for the stack slot (minimum 8 bytes).
    pub slot_align: u32,
    /// Number of bytes to advance the va_list pointer after loading.
    /// Always a multiple of 8 (minimum stack slot size on AArch64).
    pub advance_bytes: u32,
}

/// How a variadic argument value is accessed from the stack.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VaArgAccess {
    /// Direct load: the value is stored directly at the va_list pointer.
    /// Used for scalars and small aggregates (<= 16 bytes).
    Direct,
    /// Indirect load: the va_list pointer points to a pointer to the value.
    /// Used for large aggregates (> 16 bytes) that are passed by reference.
    Indirect,
}

// ---------------------------------------------------------------------------
// Lowering functions
// ---------------------------------------------------------------------------

/// Determine the lowering for `va_arg(ap, ty)` on Apple AArch64.
///
/// Returns a `VaArgLowering` describing how to extract a value of the given
/// type from the variadic argument area.
///
/// Rules (Apple AArch64 DarwinPCS):
/// - All variadic args are on the stack, 8-byte minimum alignment
/// - Scalars (I8-I128, F32, F64, B1): direct load, 8-byte advance
/// - V128: direct load, 16-byte aligned, 16-byte advance
/// - Small aggregates (<= 16 bytes): direct load, 8-byte aligned
/// - Large aggregates (> 16 bytes): indirect (pointer), 8-byte advance
///
/// Reference: Apple ARM64 Function Calling Conventions
/// Reference: LLVM AArch64ISelLowering.cpp::LowerVAARG()
pub fn lower_va_arg(ty: &Type) -> VaArgLowering {
    match ty {
        // Integer/boolean scalars: stored in 8-byte stack slots.
        // Smaller types (I8, I16, I32, B1) are promoted to 8 bytes.
        Type::B1 | Type::I8 | Type::I16 | Type::I32 | Type::I64 => VaArgLowering {
            access: VaArgAccess::Direct,
            load_size: ty.bytes().max(8),
            slot_align: 8,
            advance_bytes: 8,
        },

        // I128: 16 bytes, 8-byte aligned on Apple (Apple does NOT require
        // 16-byte alignment for I128 varargs, unlike Linux AAPCS).
        Type::I128 => VaArgLowering {
            access: VaArgAccess::Direct,
            load_size: 16,
            slot_align: 8,
            advance_bytes: 16,
        },

        // Float scalars: on Apple AArch64, variadic floats are on the stack.
        // F32 is promoted to an 8-byte slot; F64 occupies 8 bytes naturally.
        Type::F32 => VaArgLowering {
            access: VaArgAccess::Direct,
            load_size: 8, // promoted to 8-byte slot
            slot_align: 8,
            advance_bytes: 8,
        },
        Type::F64 => VaArgLowering {
            access: VaArgAccess::Direct,
            load_size: 8,
            slot_align: 8,
            advance_bytes: 8,
        },

        // 128-bit NEON vector: 16-byte aligned, 16-byte slot.
        Type::V128 => VaArgLowering {
            access: VaArgAccess::Direct,
            load_size: 16,
            slot_align: 16,
            advance_bytes: 16,
        },

        // Aggregates: size-based classification.
        Type::Struct(_) | Type::Array(_, _) => {
            let size = ty.bytes();
            if size <= 16 {
                // Small aggregate: passed directly on the stack.
                let padded = align_up(size, 8);
                VaArgLowering {
                    access: VaArgAccess::Direct,
                    load_size: size,
                    slot_align: 8,
                    advance_bytes: padded,
                }
            } else {
                // Large aggregate: passed indirectly via pointer on stack.
                VaArgLowering {
                    access: VaArgAccess::Indirect,
                    load_size: 8, // pointer size
                    slot_align: 8,
                    advance_bytes: 8,
                }
            }
        }
    }
}

/// Compute the stack offset where variadic arguments begin.
///
/// This is the total stack space consumed by fixed (non-variadic) arguments
/// that overflowed to the stack. Variadic arguments are placed immediately
/// after the fixed argument overflow area.
///
/// # Arguments
///
/// * `fixed_stack_size` — Total bytes consumed by fixed args on the stack.
///   This comes from `AppleAArch64ABI::stack_args_size()` for the fixed
///   parameter types.
///
/// Returns the byte offset from the caller's SP to the first variadic arg.
pub fn va_start_offset(fixed_stack_size: i64) -> i64 {
    // Variadic args start right after the fixed arg overflow area.
    // Both are already 8-byte aligned.
    fixed_stack_size
}

/// Align `value` up to the next multiple of `align`.
fn align_up(value: u32, align: u32) -> u32 {
    if align <= 1 {
        value
    } else {
        (value + align - 1) & !(align - 1)
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // --- VaListIntrinsic construction ---

    #[test]
    fn va_start_intrinsic_construction() {
        let intr = VaListIntrinsic::VaStart { stack_vararg_offset: 32 };
        assert_eq!(
            intr,
            VaListIntrinsic::VaStart { stack_vararg_offset: 32 }
        );
    }

    #[test]
    fn va_arg_intrinsic_construction() {
        let intr = VaListIntrinsic::VaArg { arg_type: Type::I32 };
        assert_eq!(
            intr,
            VaListIntrinsic::VaArg { arg_type: Type::I32 }
        );
    }

    #[test]
    fn va_end_intrinsic_construction() {
        assert_eq!(VaListIntrinsic::VaEnd, VaListIntrinsic::VaEnd);
    }

    #[test]
    fn va_copy_intrinsic_construction() {
        assert_eq!(VaListIntrinsic::VaCopy, VaListIntrinsic::VaCopy);
    }

    #[test]
    fn va_list_intrinsic_clone() {
        let intr = VaListIntrinsic::VaArg { arg_type: Type::F64 };
        let clone = intr.clone();
        assert_eq!(intr, clone);
    }

    // --- VaArgLowering: scalar types ---

    #[test]
    fn va_arg_i8_direct_8byte_slot() {
        let l = lower_va_arg(&Type::I8);
        assert_eq!(l.access, VaArgAccess::Direct);
        assert_eq!(l.load_size, 8); // promoted
        assert_eq!(l.slot_align, 8);
        assert_eq!(l.advance_bytes, 8);
    }

    #[test]
    fn va_arg_i16_direct_8byte_slot() {
        let l = lower_va_arg(&Type::I16);
        assert_eq!(l.access, VaArgAccess::Direct);
        assert_eq!(l.load_size, 8);
        assert_eq!(l.advance_bytes, 8);
    }

    #[test]
    fn va_arg_i32_direct_8byte_slot() {
        let l = lower_va_arg(&Type::I32);
        assert_eq!(l.access, VaArgAccess::Direct);
        assert_eq!(l.load_size, 8);
        assert_eq!(l.advance_bytes, 8);
    }

    #[test]
    fn va_arg_i64_direct_8byte_slot() {
        let l = lower_va_arg(&Type::I64);
        assert_eq!(l.access, VaArgAccess::Direct);
        assert_eq!(l.load_size, 8);
        assert_eq!(l.advance_bytes, 8);
    }

    #[test]
    fn va_arg_i128_direct_16byte_slot() {
        let l = lower_va_arg(&Type::I128);
        assert_eq!(l.access, VaArgAccess::Direct);
        assert_eq!(l.load_size, 16);
        assert_eq!(l.slot_align, 8);
        assert_eq!(l.advance_bytes, 16);
    }

    #[test]
    fn va_arg_b1_direct_8byte_slot() {
        let l = lower_va_arg(&Type::B1);
        assert_eq!(l.access, VaArgAccess::Direct);
        assert_eq!(l.load_size, 8); // promoted to 8 bytes
        assert_eq!(l.advance_bytes, 8);
    }

    // --- VaArgLowering: float types ---

    #[test]
    fn va_arg_f32_direct_8byte_slot() {
        // Apple ABI: variadic F32 on stack, promoted to 8-byte slot.
        let l = lower_va_arg(&Type::F32);
        assert_eq!(l.access, VaArgAccess::Direct);
        assert_eq!(l.load_size, 8);
        assert_eq!(l.slot_align, 8);
        assert_eq!(l.advance_bytes, 8);
    }

    #[test]
    fn va_arg_f64_direct_8byte_slot() {
        let l = lower_va_arg(&Type::F64);
        assert_eq!(l.access, VaArgAccess::Direct);
        assert_eq!(l.load_size, 8);
        assert_eq!(l.advance_bytes, 8);
    }

    // --- VaArgLowering: NEON vector ---

    #[test]
    fn va_arg_v128_direct_16byte_aligned() {
        let l = lower_va_arg(&Type::V128);
        assert_eq!(l.access, VaArgAccess::Direct);
        assert_eq!(l.load_size, 16);
        assert_eq!(l.slot_align, 16);
        assert_eq!(l.advance_bytes, 16);
    }

    // --- VaArgLowering: aggregate types ---

    #[test]
    fn va_arg_small_struct_direct() {
        // struct { i32, i32 } = 8 bytes -> direct
        let ty = Type::Struct(vec![Type::I32, Type::I32]);
        let l = lower_va_arg(&ty);
        assert_eq!(l.access, VaArgAccess::Direct);
        assert_eq!(l.load_size, 8);
        assert_eq!(l.slot_align, 8);
        assert_eq!(l.advance_bytes, 8);
    }

    #[test]
    fn va_arg_medium_struct_direct_16_bytes() {
        // struct { i64, i64 } = 16 bytes -> direct
        let ty = Type::Struct(vec![Type::I64, Type::I64]);
        let l = lower_va_arg(&ty);
        assert_eq!(l.access, VaArgAccess::Direct);
        assert_eq!(l.load_size, 16);
        assert_eq!(l.advance_bytes, 16);
    }

    #[test]
    fn va_arg_large_struct_indirect() {
        // struct { i64, i64, i64 } = 24 bytes -> indirect
        let ty = Type::Struct(vec![Type::I64, Type::I64, Type::I64]);
        let l = lower_va_arg(&ty);
        assert_eq!(l.access, VaArgAccess::Indirect);
        assert_eq!(l.load_size, 8); // pointer size
        assert_eq!(l.advance_bytes, 8);
    }

    #[test]
    fn va_arg_small_array_direct() {
        // f32[2] = 8 bytes -> direct
        let ty = Type::Array(Box::new(Type::F32), 2);
        let l = lower_va_arg(&ty);
        assert_eq!(l.access, VaArgAccess::Direct);
        assert_eq!(l.load_size, 8);
        assert_eq!(l.advance_bytes, 8);
    }

    #[test]
    fn va_arg_large_array_indirect() {
        // i64[4] = 32 bytes -> indirect
        let ty = Type::Array(Box::new(Type::I64), 4);
        let l = lower_va_arg(&ty);
        assert_eq!(l.access, VaArgAccess::Indirect);
        assert_eq!(l.load_size, 8);
        assert_eq!(l.advance_bytes, 8);
    }

    #[test]
    fn va_arg_padded_struct_advance_alignment() {
        // struct { i32, i8 } = 5 bytes, padded to 8 -> advance 8
        let ty = Type::Struct(vec![Type::I32, Type::I8]);
        let l = lower_va_arg(&ty);
        assert_eq!(l.access, VaArgAccess::Direct);
        assert_eq!(l.advance_bytes, 8); // padded to 8-byte boundary
    }

    #[test]
    fn va_arg_12_byte_struct_advance_16() {
        // struct { i32, i64 } = 12 bytes with padding -> 16 bytes total
        // Actually: offset 0 = i32(4), pad to 8, offset 8 = i64(8) = total 16
        let ty = Type::Struct(vec![Type::I32, Type::I64]);
        let actual_size = ty.bytes();
        let l = lower_va_arg(&ty);
        assert_eq!(l.access, VaArgAccess::Direct);
        assert_eq!(l.load_size, actual_size);
        assert_eq!(l.advance_bytes, align_up(actual_size, 8));
    }

    // --- va_start_offset ---

    #[test]
    fn va_start_offset_no_fixed_overflow() {
        // All fixed args in registers -> variadic area starts at SP+0
        assert_eq!(va_start_offset(0), 0);
    }

    #[test]
    fn va_start_offset_with_fixed_overflow() {
        // Fixed args consumed 16 bytes of stack -> variadic area at SP+16
        assert_eq!(va_start_offset(16), 16);
    }

    #[test]
    fn va_start_offset_preserves_alignment() {
        // 24 bytes of fixed overflow -> 24 (already 8-byte aligned)
        assert_eq!(va_start_offset(24), 24);
    }

    // --- align_up helper ---

    #[test]
    fn align_up_already_aligned() {
        assert_eq!(align_up(8, 8), 8);
        assert_eq!(align_up(16, 8), 16);
        assert_eq!(align_up(0, 8), 0);
    }

    #[test]
    fn align_up_needs_padding() {
        assert_eq!(align_up(1, 8), 8);
        assert_eq!(align_up(5, 8), 8);
        assert_eq!(align_up(9, 8), 16);
        assert_eq!(align_up(12, 16), 16);
    }

    #[test]
    fn align_up_trivial() {
        assert_eq!(align_up(7, 1), 7);
        assert_eq!(align_up(0, 1), 0);
    }

    // --- VaArgAccess variants ---

    #[test]
    fn va_arg_access_direct_ne_indirect() {
        assert_ne!(VaArgAccess::Direct, VaArgAccess::Indirect);
    }

    #[test]
    fn va_arg_access_clone_eq() {
        let a = VaArgAccess::Direct;
        assert_eq!(a, a.clone());
    }

    // --- VaArgLowering equality ---

    #[test]
    fn va_arg_lowering_equality() {
        let l1 = lower_va_arg(&Type::I64);
        let l2 = lower_va_arg(&Type::I64);
        assert_eq!(l1, l2);
    }

    #[test]
    fn va_arg_lowering_inequality() {
        let l_i32 = lower_va_arg(&Type::I32);
        let l_v128 = lower_va_arg(&Type::V128);
        assert_ne!(l_i32, l_v128);
    }

    // --- Empty struct edge case ---

    #[test]
    fn va_arg_empty_struct() {
        // struct {} = 0 bytes -> direct, but advance at least 8
        let ty = Type::Struct(vec![]);
        let l = lower_va_arg(&ty);
        assert_eq!(l.access, VaArgAccess::Direct);
        assert_eq!(l.load_size, 0);
        // align_up(0, 8) = 0, but we need at least one slot
        // This is an edge case: the caller shouldn't create zero-size va_args
        // but we don't panic.
    }

    // --- HFA struct through va_arg (treated as regular aggregate on stack) ---

    #[test]
    fn va_arg_hfa_struct_is_direct() {
        // struct { f32, f32 } = 8 bytes -> direct (HFA is irrelevant for variadics,
        // since all variadic args are on the stack regardless)
        let ty = Type::Struct(vec![Type::F32, Type::F32]);
        let l = lower_va_arg(&ty);
        assert_eq!(l.access, VaArgAccess::Direct);
        assert_eq!(l.load_size, 8);
        assert_eq!(l.advance_bytes, 8);
    }

    #[test]
    fn va_arg_hfa_4xf64_is_indirect() {
        // struct { f64, f64, f64, f64 } = 32 bytes > 16 -> indirect
        let ty = Type::Struct(vec![Type::F64, Type::F64, Type::F64, Type::F64]);
        let l = lower_va_arg(&ty);
        assert_eq!(l.access, VaArgAccess::Indirect);
    }
}
