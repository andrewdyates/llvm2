// fuzz/fuzz_targets/fuzz_encode_instruction.rs
//
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Copyright 2026 Andrew Yates | License: Apache-2.0
//
// libFuzzer target shadowing `panic_fuzz_encode.rs`. Derives a random
// (opcode, operand list) tuple from the fuzzer's byte stream and feeds it
// to `encode_instruction`. The contract is identical to the proptest
// harness: encode must return `Ok(u32)` or `Err(EncodeError)` — never
// panic, never abort.

#![no_main]

use libfuzzer_sys::fuzz_target;

use llvm2_codegen::aarch64::encode::encode_instruction;
use llvm2_ir::inst::{AArch64Opcode, MachInst};
use llvm2_ir::operand::MachOperand;

// A compact opcode table. We index into it with one fuzz byte, so the
// fuzzer's byte stream maps deterministically to opcodes. Keeping this in
// sync with `panic_fuzz_encode::opcode_strategy` is a deliberate soft
// invariant: the cargo-fuzz target and the proptest target should share
// an alphabet so corpus entries are mutually meaningful.
const OPCODES: &[AArch64Opcode] = &[
    AArch64Opcode::AddRR, AArch64Opcode::AddRI, AArch64Opcode::SubRR, AArch64Opcode::SubRI,
    AArch64Opcode::MulRR, AArch64Opcode::Msub, AArch64Opcode::Smull, AArch64Opcode::Umull,
    AArch64Opcode::SDiv, AArch64Opcode::UDiv, AArch64Opcode::Neg,
    AArch64Opcode::AndRR, AArch64Opcode::AndRI, AArch64Opcode::OrrRR, AArch64Opcode::EorRR,
    AArch64Opcode::LslRR, AArch64Opcode::LsrRR, AArch64Opcode::AsrRR,
    AArch64Opcode::CmpRR, AArch64Opcode::CmpRI, AArch64Opcode::Tst,
    AArch64Opcode::MovR, AArch64Opcode::MovI, AArch64Opcode::Movz, AArch64Opcode::Movn,
    AArch64Opcode::LdrRI, AArch64Opcode::StrRI, AArch64Opcode::LdrbRI, AArch64Opcode::StrbRI,
    AArch64Opcode::BCond, AArch64Opcode::Cbz, AArch64Opcode::Cbnz,
    AArch64Opcode::Br, AArch64Opcode::Bl, AArch64Opcode::Blr, AArch64Opcode::Ret,
    AArch64Opcode::FaddRR, AArch64Opcode::FsubRR, AArch64Opcode::FmulRR,
    AArch64Opcode::Phi, AArch64Opcode::Copy, AArch64Opcode::Nop,
];

fn take_u32(data: &[u8], pos: &mut usize) -> u32 {
    let mut buf = [0u8; 4];
    for slot in buf.iter_mut() {
        if *pos < data.len() {
            *slot = data[*pos];
            *pos += 1;
        }
    }
    u32::from_le_bytes(buf)
}

fn take_i64(data: &[u8], pos: &mut usize) -> i64 {
    let mut buf = [0u8; 8];
    for slot in buf.iter_mut() {
        if *pos < data.len() {
            *slot = data[*pos];
            *pos += 1;
        }
    }
    i64::from_le_bytes(buf)
}

fn derive_operand(data: &[u8], pos: &mut usize) -> MachOperand {
    let tag = if *pos < data.len() {
        let v = data[*pos];
        *pos += 1;
        v
    } else {
        0
    };
    match tag % 4 {
        0 => MachOperand::Imm(take_i64(data, pos)),
        1 => MachOperand::FImm(f64::from_bits(take_u32(data, pos) as u64)),
        2 => MachOperand::Block(llvm2_ir::types::BlockId(take_u32(data, pos) % 8)),
        _ => MachOperand::Imm(0),
    }
}

fuzz_target!(|data: &[u8]| {
    if data.is_empty() {
        return;
    }
    let mut pos = 0;
    let opcode_byte = data[pos];
    pos += 1;
    let opcode = OPCODES[opcode_byte as usize % OPCODES.len()];

    let noperands = if pos < data.len() {
        let v = data[pos] as usize;
        pos += 1;
        v % 5
    } else {
        0
    };
    let mut operands = Vec::with_capacity(noperands);
    for _ in 0..noperands {
        operands.push(derive_operand(data, &mut pos));
    }

    let inst = MachInst::new(opcode, operands);
    let _ = encode_instruction(&inst);
});
