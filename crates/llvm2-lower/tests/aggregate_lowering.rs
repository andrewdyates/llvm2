// aggregate_lowering.rs — Phase 1 aggregate regression tests (#391)
//
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Copyright 2026 Andrew Yates | License: Apache-2.0

//! Aggregate lowering regression tests.
//!
//! These tests verify that aggregate operations (StructGep, ArrayGep,
//! memcpy/memset intrinsics) survive the ISel path without panics and
//! produce machine instructions with the expected shape.
//!
//! End-to-end survival through the optimization pipeline (O0/O1/O2/O3)
//! is covered by `crates/llvm2-opt/tests/aggregate_pipeline.rs`.
//!
//! Design: `designs/2026-04-18-aggregate-lowering.md`.

use llvm2_lower::instructions::{Block, Instruction, Opcode, Value};
use llvm2_lower::isel::{AArch64Opcode, ISelOperand, InstructionSelector};
use llvm2_lower::function::Signature;
use llvm2_lower::types::Type;

/// Build an ISel with two pointer-sized formal arguments.
/// Value(0) = base pointer, Value(1) = index (both I64).
fn make_isel_ptr_index() -> (InstructionSelector, Block) {
    let sig = Signature {
        params: vec![Type::I64, Type::I64],
        returns: vec![],
    };
    let mut isel = InstructionSelector::new("aggregate_test".to_string(), sig.clone());
    let entry = Block(0);
    isel.lower_formal_arguments(&sig, entry).unwrap();
    (isel, entry)
}

/// Build an ISel with a single pointer-sized formal argument (Value(0)).
fn make_isel_ptr() -> (InstructionSelector, Block) {
    let sig = Signature {
        params: vec![Type::I64],
        returns: vec![],
    };
    let mut isel = InstructionSelector::new("aggregate_test".to_string(), sig.clone());
    let entry = Block(0);
    isel.lower_formal_arguments(&sig, entry).unwrap();
    (isel, entry)
}

#[test]
fn struct_gep_then_load_lowers_cleanly() {
    // Simulate:
    //   p: *Struct{I32, I64} = arg0
    //   field1_addr = StructGep(p, 1)   // offset 8
    //   v = Load<I64>(field1_addr)
    let (mut isel, entry) = make_isel_ptr();
    let struct_ty = Type::Struct(vec![Type::I32, Type::I64]);
    let insts = vec![
        Instruction {
            opcode: Opcode::StructGep { struct_ty, field_index: 1 },
            args: vec![Value(0)],
            results: vec![Value(1)],
        },
        Instruction {
            opcode: Opcode::Load { ty: Type::I64 },
            args: vec![Value(1)],
            results: vec![Value(2)],
        },
    ];
    isel.select_block(entry, &insts).unwrap();
    let mfunc = isel.finalize();
    let mblock = &mfunc.blocks[&entry];

    let gep_pos = mblock
        .insts
        .iter()
        .position(|i| matches!(i.opcode, AArch64Opcode::AddRI))
        .expect("expected ADD for StructGep");
    assert!(gep_pos < mblock.insts.len() - 1, "ADD should precede LDR");
    let next_op = mblock.insts[gep_pos + 1].opcode;
    assert!(
        matches!(
            next_op,
            AArch64Opcode::LdrRI
                | AArch64Opcode::LdrbRI
                | AArch64Opcode::LdrhRI
                | AArch64Opcode::LdrRO
        ),
        "StructGep should be followed by a load, got {:?}",
        next_op
    );
}

#[test]
fn array_gep_i32_lowers_to_shift_and_add() {
    // Simulate an i32 array element address computation.
    // Value(0) = base pointer, Value(1) = index (both I64 formal args).
    let (mut isel, entry) = make_isel_ptr_index();
    let insts = vec![Instruction {
        opcode: Opcode::ArrayGep { elem_ty: Type::I32 },
        args: vec![Value(0), Value(1)],
        results: vec![Value(2)],
    }];
    isel.select_block(entry, &insts).unwrap();
    let mfunc = isel.finalize();
    let mblock = &mfunc.blocks[&entry];

    let lsl_pos = mblock
        .insts
        .iter()
        .position(|i| i.opcode == AArch64Opcode::LslRI)
        .expect("expected LSL for i32 array index scaling");
    // Verify the shift amount is log2(4) = 2.
    let lsl_imm = mblock.insts[lsl_pos]
        .operands
        .iter()
        .find_map(|op| match op {
            ISelOperand::Imm(v) => Some(*v),
            _ => None,
        })
        .expect("LSL should carry an immediate");
    assert_eq!(lsl_imm, 2, "i32 array: LSL #2");
    // AddRR must follow.
    let add_pos = mblock
        .insts
        .iter()
        .skip(lsl_pos + 1)
        .position(|i| i.opcode == AArch64Opcode::AddRR);
    assert!(add_pos.is_some(), "expected AddRR after LSL");
}

#[test]
fn nested_struct_of_array_field_offset() {
    // struct Outer { I32 head; [I64; 4] tail; }
    //   head at offset 0, tail at offset 8 (aligned to 8 for I64)
    // tail[i] = ArrayGep(StructGep(p, 1), i)
    let (mut isel, entry) = make_isel_ptr_index();

    let tail_ty = Type::Array(Box::new(Type::I64), 4);
    let outer_ty = Type::Struct(vec![Type::I32, tail_ty.clone()]);

    let insts = vec![
        // tail_addr = gep(p, field 1)  -> offset 8
        Instruction {
            opcode: Opcode::StructGep {
                struct_ty: outer_ty,
                field_index: 1,
            },
            args: vec![Value(0)],
            results: vec![Value(2)],
        },
        // elem_addr = array_gep(tail_addr, i)  -> +i*8
        Instruction {
            opcode: Opcode::ArrayGep { elem_ty: Type::I64 },
            args: vec![Value(2), Value(1)],
            results: vec![Value(3)],
        },
    ];
    isel.select_block(entry, &insts).unwrap();
    let mfunc = isel.finalize();
    let mblock = &mfunc.blocks[&entry];

    let add_imm8 = mblock.insts.iter().filter(|i| {
        i.opcode == AArch64Opcode::AddRI
            && i.operands
                .iter()
                .any(|op| matches!(op, ISelOperand::Imm(8)))
    }).count();
    let lsl_imm3 = mblock.insts.iter().filter(|i| {
        i.opcode == AArch64Opcode::LslRI
            && i.operands
                .iter()
                .any(|op| matches!(op, ISelOperand::Imm(3)))
    }).count();
    let add_rr = mblock
        .insts
        .iter()
        .filter(|i| i.opcode == AArch64Opcode::AddRR)
        .count();

    assert_eq!(add_imm8, 1, "expected StructGep ADD with imm 8");
    assert_eq!(lsl_imm3, 1, "expected ArrayGep LSL with imm 3 (i64)");
    assert!(add_rr >= 1, "expected at least one AddRR from ArrayGep");
}

#[test]
fn array_gep_chained_elements_produce_two_computations() {
    // Two ArrayGep ops into the same array must emit two independent
    // LSL + AddRR pairs. ISel does not fold or CSE.
    let (mut isel, entry) = make_isel_ptr_index();

    let insts = vec![
        // elem_addr0 = array_gep(base, index)
        Instruction {
            opcode: Opcode::ArrayGep { elem_ty: Type::I32 },
            args: vec![Value(0), Value(1)],
            results: vec![Value(2)],
        },
        // elem_addr1 = array_gep(base, index)  (same args, to test non-folding)
        Instruction {
            opcode: Opcode::ArrayGep { elem_ty: Type::I32 },
            args: vec![Value(0), Value(1)],
            results: vec![Value(3)],
        },
    ];
    isel.select_block(entry, &insts).unwrap();
    let mfunc = isel.finalize();
    let mblock = &mfunc.blocks[&entry];

    let lsl_count = mblock
        .insts
        .iter()
        .filter(|i| i.opcode == AArch64Opcode::LslRI)
        .count();
    assert_eq!(lsl_count, 2, "two array-geps → two LSLs (no ISel-time CSE)");
}
