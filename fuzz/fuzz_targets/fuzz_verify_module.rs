// fuzz/fuzz_targets/fuzz_verify_module.rs
//
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Copyright 2026 Andrew Yates | License: Apache-2.0
//
// libFuzzer target shadowing `panic_fuzz_verify.rs`. Builds a compact
// `MachFunction` from the fuzzer byte stream and drives it through
// `verify_function`. The contract is: the verifier must produce a
// `FunctionVerificationReport` (possibly with skipped / unverified
// entries) — never panic.

#![no_main]

use libfuzzer_sys::fuzz_target;

use llvm2_ir::function::{MachBlock, MachFunction, Signature, Type};
use llvm2_ir::inst::{AArch64Opcode, MachInst};
use llvm2_ir::operand::MachOperand;
use llvm2_ir::types::{BlockId, InstId};
use llvm2_verify::function_verifier::verify_function;

const OPCODES: &[AArch64Opcode] = &[
    AArch64Opcode::AddRR, AArch64Opcode::AddRI, AArch64Opcode::SubRR, AArch64Opcode::SubRI,
    AArch64Opcode::MulRR, AArch64Opcode::SDiv, AArch64Opcode::UDiv, AArch64Opcode::Neg,
    AArch64Opcode::AndRR, AArch64Opcode::OrrRR,
    AArch64Opcode::LslRI, AArch64Opcode::LsrRR, AArch64Opcode::AsrRR,
    AArch64Opcode::CmpRR, AArch64Opcode::CmpRI, AArch64Opcode::Tst,
    AArch64Opcode::LdrRI, AArch64Opcode::StrRI,
    AArch64Opcode::FaddRR, AArch64Opcode::FsubRR, AArch64Opcode::FmulRR,
    AArch64Opcode::CSet, AArch64Opcode::BCond,
    // Pseudos
    AArch64Opcode::Phi, AArch64Opcode::Copy, AArch64Opcode::Nop,
];

fuzz_target!(|data: &[u8]| {
    let mut func = MachFunction::new("_fuzz".into(), Signature::new(vec![], vec![Type::I64]));
    func.blocks.clear();
    func.block_order.clear();

    // Seed one block.
    let mut block = MachBlock::new();

    // Derive up to 16 instructions from the byte stream.
    let nsteps = data.len().min(16);
    for i in 0..nsteps {
        let b = data[i];
        let opcode = OPCODES[b as usize % OPCODES.len()];
        let nops = ((b >> 4) & 0x3) as usize;
        let operands = (0..nops)
            .map(|j| MachOperand::Imm((b as i64).wrapping_mul(j as i64 + 1)))
            .collect::<Vec<_>>();
        let inst_id = func.insts.len() as u32;
        func.insts.push(MachInst::new(opcode, operands));
        block.insts.push(InstId(inst_id));

        // Every 5th byte, inject an out-of-range InstId to exercise the
        // verifier's bounds guard.
        if i % 5 == 4 {
            block.insts.push(InstId(u32::MAX - i as u32));
        }
    }
    func.blocks.push(block);
    func.block_order.push(BlockId(0));
    func.entry = BlockId(0);

    let _ = verify_function(&func);
});
