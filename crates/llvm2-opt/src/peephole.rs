// llvm2-opt - AArch64 Peephole Optimizations
//
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Copyright 2026 Andrew Yates | License: Apache-2.0

//! AArch64-specific peephole optimizations.
//!
//! Peephole optimizations are local pattern-matching transformations that
//! simplify or eliminate individual instructions or short instruction
//! sequences.
//!
//! # Patterns Implemented
//!
//! ## Identity Simplifications (instruction -> MOV)
//!
//! | # | Pattern | Transformation |
//! |---|---------|---------------|
//! | 1 | `mov x0, x0` | Delete (self-move is no-op) |
//! | 2 | `add x0, x1, #0` | `mov x0, x1` (additive identity) |
//! | 3 | `sub x0, x1, #0` | `mov x0, x1` (subtractive identity) |
//! | 4 | `lsl x0, x1, #0` | `mov x0, x1` (shift by zero) |
//! | 5 | `lsr x0, x1, #0` | `mov x0, x1` (shift by zero) |
//! | 6 | `asr x0, x1, #0` | `mov x0, x1` (shift by zero) |
//! | 7 | `orr x0, x1, x1` | `mov x0, x1` (OR self = identity) |
//! | 8 | `and x0, x1, x1` | `mov x0, x1` (AND self = identity) |
//! | 9 | `eor x0, x1, #0` | `mov x0, x1` (XOR with zero) |
//! | 10 | `orr x0, x1, #0` | `mov x0, x1` (OR with zero) |
//! | 11 | `and x0, x1, #-1` | `mov x0, x1` (AND with all-ones) |
//!
//! ## Zero-Producing Operations (instruction -> MOV #0)
//!
//! | # | Pattern | Transformation |
//! |---|---------|---------------|
//! | 12 | `sub x0, x1, x1` | `mov x0, #0` (SUB self = zero) |
//! | 13 | `eor x0, x1, x1` | `mov x0, #0` (XOR self = zero) |
//! | 14 | `and x0, x1, #0` | `mov x0, #0` (AND with zero) |
//!
//! ## Constant-Producing Operations (instruction -> MOV #const)
//!
//! | # | Pattern | Transformation |
//! |---|---------|---------------|
//! | 15 | `orr x0, x1, #-1` | `mov x0, #-1` (OR with all-ones) |
//!
//! ## Strength Reduction (instruction -> cheaper instruction)
//!
//! | # | Pattern | Transformation |
//! |---|---------|---------------|
//! | 16 | `add x0, x1, x1` | `lsl x0, x1, #1` (double = shift left) |
//!
//! ## Nop Elimination
//!
//! | # | Pattern | Transformation |
//! |---|---------|---------------|
//! | 17 | `nop` | Delete |
//!
//! ## Immediate Canonicalization
//!
//! | # | Pattern | Transformation |
//! |---|---------|---------------|
//! | 18 | `add x0, x1, #-N` (N>0) | `sub x0, x1, #N` (negative add → positive sub) |
//! | 19 | `sub x0, x1, #-N` (N>0) | `add x0, x1, #N` (negative sub → positive add) |
//!
//! ## Multi-Instruction Patterns (def-use chain within same block)
//!
//! | # | Pattern | Transformation |
//! |---|---------|---------------|
//! | 20 | `neg(neg(x))` | `mov x` (double negation elimination) |
//! | 21 | `add x, neg(y)` | `sub x, y` (add-of-negation → subtraction) |
//! | 22 | `sub x, neg(y)` | `add x, y` (sub-of-negation → addition) |
//! | 23 | `mul x, #2^k` | `lsl x, #k` (multiply by power of 2 → shift) |
//! | 24 | `udiv x, #2^k` | `lsr x, #k` (unsigned divide by power of 2 → shift) |
//! | 25 | `mul x, #1` | `mov x` (multiply by one identity) |
//! | 26 | `bic x, y, y` | `mov x, #0` (BIC self: x AND NOT(x) = 0) |
//! | 27 | `orn x, y, y` | `mov x, #-1` (ORN self: x OR NOT(x) = all-ones) |
//! | 28 | `sxtb(sxtb(x))` | `sxtb(x)` (double sign extension elimination) |
//! | 29 | `lsl(lsr(x, n), n)` | `and x, mask` (shift-pair to mask, clear low bits) |
//! | 30 | `neg(sub(x, y))` | `sub y, x` (negate subtraction swaps operands) |
//! | 31 | `mul x, #-1` | `neg x` (multiply by negative one → negation) |
//! | 32 | `lsr(lsl(x, n), n)` | `and x, mask` (shift-pair to mask, clear high bits) |
//! | 33 | `sdiv x, #1` | `mov x` (signed division by one identity) |
//! | 34 | `mul x, #0` | `mov x, #0` (multiply by zero absorbing) |
//! | 35 | `madd x, y, #1, z` | `add x, y, z` (multiply-add with multiplier 1) |
//! | 36 | `madd x, y, #0, z` | `mov x, z` (multiply-add with multiplier 0) |
//! | 37 | `msub x, y, #1, z` | `sub x, z, y` (multiply-sub with multiplier 1) |
//! | 38 | `msub x, y, #0, z` | `mov x, z` (multiply-sub with multiplier 0) |
//! | 39 | `sxth(sxtb(x))` | `sxtb(x)` (narrower extension subsumes wider) |
//! | 40 | `sxtw(sxtb(x))` or `sxtw(sxth(x))` | narrower ext (narrower subsumes wider) |
//! | 41 | `add(sub(x, y), y)` | `mov x` (cross-instruction cancellation) |
//! | 42 | `sub(add(x, y), y)` | `mov x` (cross-instruction cancellation) |
//! | 43 | `csel x, y, y, cond` | `mov x, y` (identical arms eliminates condition) |
//! | 44 | `udiv x, #1` | `mov x` (unsigned divide by one identity) |
//! | 45 | `sdiv x, #-1` | `neg x` (signed divide by negative one) |
//! | 46 | `uxtb(uxtb(x))` | `uxtb(x)` (double zero extension idempotent) |
//! | 47 | `uxth(uxth(x))` | `uxth(x)` (double zero extension idempotent) |
//! | 48 | `uxth(uxtb(x))` | `uxtb(x)` (narrower zero ext subsumes wider) |
//! | 49 | `uxtw(uxtb(x))` | `uxtb(x)` (narrower zero ext subsumes wider) |
//! | 50 | `uxtw(uxth(x))` | `uxth(x)` (narrower zero ext subsumes wider) |
//! | 51 | `add(x, sub(y, x))` | `mov y` (cross-instruction cancellation) |
//! | 52 | `eor(eor(x, y), y)` | `mov x` (XOR cancellation) |
//!
//! # Note on cmp+b.cond Fusing
//!
//! CMP+BCond fusing into CBZ/CBNZ/TBZ/TBNZ is handled by the
//! [`cmp_branch_fusion`](crate::cmp_branch_fusion) pass, which runs after
//! peephole. It requires cross-instruction analysis (the CMP must be
//! immediately followed by the BCond with no intervening flag-setting
//! instructions).

use std::collections::HashMap;

use llvm2_ir::{AArch64Opcode, AArch64Target, BlockId, InstId, MachFunction, MachInst, MachOperand, OpcodeCategory, TargetInfo};

use crate::pass_manager::MachinePass;

/// AArch64 peephole optimization pass.
pub struct Peephole;

impl MachinePass for Peephole {
    fn name(&self) -> &str {
        "peephole"
    }

    fn run(&mut self, func: &mut MachFunction) -> bool {
        let mut changed = false;
        let mut to_delete: Vec<InstId> = Vec::new();

        for block_id in func.block_order.clone() {
            // Build def map for multi-instruction patterns within this block.
            let def_map = build_def_map(func, block_id);

            let block = func.block(block_id);
            for &inst_id in block.insts.clone().iter() {
                // Try single-instruction patterns first.
                let result = match try_peephole(func.inst(inst_id)) {
                    PeepholeResult::NoChange => {
                        // Try multi-instruction patterns (def-use chain).
                        try_peephole_with_defs(func.inst(inst_id), func, &def_map)
                    }
                    other => other,
                };

                match result {
                    PeepholeResult::NoChange => {}
                    PeepholeResult::Replace(mut new_inst) => {
                        // Preserve proof annotation from the original instruction.
                        // The peephole simplification preserves semantics
                        // (e.g., x + 0 -> x), so the proof remains valid.
                        new_inst.proof = func.inst(inst_id).proof;
                        // Preserve source_loc (issue #376): peephole replaces
                        // one instruction with a semantically equivalent one,
                        // which still corresponds to the same source line/column.
                        // Dropping source_loc here causes `<optimized out>`
                        // line-info regression at -O3.
                        if new_inst.source_loc.is_none() {
                            new_inst.source_loc = func.inst(inst_id).source_loc;
                        }
                        *func.inst_mut(inst_id) = new_inst;
                        changed = true;
                    }
                    PeepholeResult::Delete => {
                        to_delete.push(inst_id);
                        changed = true;
                    }
                }
            }
        }

        // Remove deleted instructions from blocks.
        if !to_delete.is_empty() {
            let delete_set: std::collections::HashSet<InstId> =
                to_delete.into_iter().collect();
            for block_id in func.block_order.clone() {
                let block = func.block_mut(block_id);
                block.insts.retain(|id| !delete_set.contains(id));
            }
        }

        changed
    }
}

/// Result of attempting a peephole optimization on a single instruction.
enum PeepholeResult {
    /// No optimization applies.
    NoChange,
    /// Replace the instruction with a new one.
    Replace(MachInst),
    /// Delete the instruction entirely.
    Delete,
}

/// Try to apply a peephole optimization to a single instruction.
///
/// Dispatch uses [`OpcodeCategory`] for patterns that are algebraic identities
/// (true on any target): add(x,0)=x, sub(x,0)=x, shift(x,0)=x, etc. The
/// category determines *which* rule applies; replacement opcodes are obtained
/// from [`TargetInfo`] methods (`mov_rr()`, `mov_ri()`, `shl_ri()`, etc.)
/// so patterns are target-independent.
///
/// Generic instruction properties are also used:
/// - `is_nop()` for nop elimination (any target)
/// - `is_move()` for self-move detection (any target)
///
/// Multi-instruction patterns and target-specific idioms remain in the
/// AArch64-specific match below.
fn try_peephole(inst: &MachInst) -> PeepholeResult {
    // Generic: delete nops on any target.
    if inst.is_nop() {
        return PeepholeResult::Delete;
    }
    // Generic: delete self-moves on any target.
    if inst.is_move() {
        return peephole_mov(inst);
    }

    // Category-based dispatch for algebraic identity patterns.
    // These rules are target-independent (the algebra is the same on
    // AArch64, x86-64, RISC-V, etc.). Replacement opcodes come from
    // TargetInfo methods for multi-target support.
    let category = inst.opcode.categorize();
    match category {
        // -- Register-immediate identity/constant --
        OpcodeCategory::AddRI => peephole_add_ri(inst),
        OpcodeCategory::SubRI => peephole_sub_ri(inst),
        OpcodeCategory::ShlRI | OpcodeCategory::ShrRI | OpcodeCategory::SarRI => {
            peephole_shift_ri_zero(inst)
        }
        OpcodeCategory::XorRI => peephole_eor_ri(inst),
        OpcodeCategory::OrRI => peephole_orr_ri(inst),
        OpcodeCategory::AndRI => peephole_and_ri(inst),
        // -- Register-register self patterns --
        OpcodeCategory::OrRR | OpcodeCategory::AndRR => peephole_logical_self(inst),
        OpcodeCategory::SubRR => peephole_sub_rr(inst),
        OpcodeCategory::XorRR => peephole_eor_rr(inst),
        OpcodeCategory::AddRR => peephole_add_rr(inst),
        // BicRR/OrnRR don't have OpcodeCategory equivalents yet — match by opcode.
        OpcodeCategory::Other => match inst.opcode {
            AArch64Opcode::BicRR => peephole_bic_self(inst),
            AArch64Opcode::OrnRR => peephole_orn_self(inst),
            AArch64Opcode::Csel => peephole_csel_same_arms(inst),
            _ => PeepholeResult::NoChange,
        },
        _ => PeepholeResult::NoChange,
    }
}

/// `mov x0, x0` → delete (self-move is no-op).
fn peephole_mov(inst: &MachInst) -> PeepholeResult {
    if inst.operands.len() < 2 {
        return PeepholeResult::NoChange;
    }

    // Check if src == dst (self-move).
    match (&inst.operands[0], &inst.operands[1]) {
        (MachOperand::VReg(dst), MachOperand::VReg(src)) if dst.id == src.id => {
            PeepholeResult::Delete
        }
        (MachOperand::PReg(dst), MachOperand::PReg(src)) if dst == src => {
            PeepholeResult::Delete
        }
        _ => PeepholeResult::NoChange,
    }
}

/// `add x0, x1, #0` → `mov x0, x1` (pattern 2: additive identity)
/// `add x0, x1, #-N` → `sub x0, x1, #N` (pattern 18: negative immediate canonicalization)
fn peephole_add_ri(inst: &MachInst) -> PeepholeResult {
    if inst.operands.len() < 3 {
        return PeepholeResult::NoChange;
    }

    match &inst.operands[2] {
        // Pattern 2: add x, y, #0 → mov x, y
        MachOperand::Imm(0) => {
            let new_inst = MachInst::new(
                AArch64Target::mov_rr(),
                vec![inst.operands[0].clone(), inst.operands[1].clone()],
            );
            PeepholeResult::Replace(new_inst)
        }
        // Pattern 18: add x, y, #-N (N>0) → sub x, y, #N
        MachOperand::Imm(val) if *val < 0 && *val != i64::MIN => {
            let new_inst = MachInst::new(
                AArch64Target::sub_ri(),
                vec![
                    inst.operands[0].clone(),
                    inst.operands[1].clone(),
                    MachOperand::Imm(-*val),
                ],
            );
            PeepholeResult::Replace(new_inst)
        }
        _ => PeepholeResult::NoChange,
    }
}

/// `sub x0, x1, #0` → `mov x0, x1` (pattern 3: subtractive identity)
/// `sub x0, x1, #-N` → `add x0, x1, #N` (pattern 19: negative immediate canonicalization)
fn peephole_sub_ri(inst: &MachInst) -> PeepholeResult {
    if inst.operands.len() < 3 {
        return PeepholeResult::NoChange;
    }

    match &inst.operands[2] {
        // Pattern 3: sub x, y, #0 → mov x, y
        MachOperand::Imm(0) => {
            let new_inst = MachInst::new(
                AArch64Target::mov_rr(),
                vec![inst.operands[0].clone(), inst.operands[1].clone()],
            );
            PeepholeResult::Replace(new_inst)
        }
        // Pattern 19: sub x, y, #-N (N>0) → add x, y, #N
        MachOperand::Imm(val) if *val < 0 && *val != i64::MIN => {
            let new_inst = MachInst::new(
                AArch64Target::add_ri(),
                vec![
                    inst.operands[0].clone(),
                    inst.operands[1].clone(),
                    MachOperand::Imm(-*val),
                ],
            );
            PeepholeResult::Replace(new_inst)
        }
        _ => PeepholeResult::NoChange,
    }
}

/// `lsl/lsr/asr x0, x1, #0` → `mov x0, x1`
fn peephole_shift_ri_zero(inst: &MachInst) -> PeepholeResult {
    if inst.operands.len() < 3 {
        return PeepholeResult::NoChange;
    }

    if let MachOperand::Imm(0) = &inst.operands[2] {
        let new_inst = MachInst::new(
            AArch64Target::mov_rr(),
            vec![inst.operands[0].clone(), inst.operands[1].clone()],
        );
        PeepholeResult::Replace(new_inst)
    } else {
        PeepholeResult::NoChange
    }
}

/// `orr x0, x1, x1` → `mov x0, x1` (OR with self = identity)
/// `and x0, x1, x1` → `mov x0, x1` (AND with self = identity)
fn peephole_logical_self(inst: &MachInst) -> PeepholeResult {
    if inst.operands.len() < 3 {
        return PeepholeResult::NoChange;
    }

    // Check if both source operands are the same VReg.
    match (&inst.operands[1], &inst.operands[2]) {
        (MachOperand::VReg(a), MachOperand::VReg(b)) if a.id == b.id => {
            let new_inst = MachInst::new(
                AArch64Target::mov_rr(),
                vec![inst.operands[0].clone(), inst.operands[1].clone()],
            );
            PeepholeResult::Replace(new_inst)
        }
        _ => PeepholeResult::NoChange,
    }
}

// =========================================================================
// New patterns: bitwise register-immediate
// =========================================================================

/// `eor x0, x1, #0` → `mov x0, x1` (XOR with zero is identity).
fn peephole_eor_ri(inst: &MachInst) -> PeepholeResult {
    if inst.operands.len() < 3 {
        return PeepholeResult::NoChange;
    }

    if let MachOperand::Imm(0) = &inst.operands[2] {
        let new_inst = MachInst::new(
            AArch64Target::mov_rr(),
            vec![inst.operands[0].clone(), inst.operands[1].clone()],
        );
        PeepholeResult::Replace(new_inst)
    } else {
        PeepholeResult::NoChange
    }
}

/// `orr x0, x1, #0` → `mov x0, x1` (OR with zero is identity).
/// `orr x0, x1, #-1` → `mov x0, #-1` (OR with all-ones is all-ones).
fn peephole_orr_ri(inst: &MachInst) -> PeepholeResult {
    if inst.operands.len() < 3 {
        return PeepholeResult::NoChange;
    }

    match &inst.operands[2] {
        MachOperand::Imm(0) => {
            // OR with zero is identity.
            let new_inst = MachInst::new(
                AArch64Target::mov_rr(),
                vec![inst.operands[0].clone(), inst.operands[1].clone()],
            );
            PeepholeResult::Replace(new_inst)
        }
        MachOperand::Imm(-1) => {
            // OR with all-ones produces all-ones.
            let new_inst = MachInst::new(
                AArch64Target::mov_ri(),
                vec![inst.operands[0].clone(), MachOperand::Imm(-1)],
            );
            PeepholeResult::Replace(new_inst)
        }
        _ => PeepholeResult::NoChange,
    }
}

/// `and x0, x1, #0` → `mov x0, #0` (AND with zero is zero).
/// `and x0, x1, #-1` → `mov x0, x1` (AND with all-ones is identity).
fn peephole_and_ri(inst: &MachInst) -> PeepholeResult {
    if inst.operands.len() < 3 {
        return PeepholeResult::NoChange;
    }

    match &inst.operands[2] {
        MachOperand::Imm(0) => {
            // AND with zero produces zero.
            let new_inst = MachInst::new(
                AArch64Target::mov_ri(),
                vec![inst.operands[0].clone(), MachOperand::Imm(0)],
            );
            PeepholeResult::Replace(new_inst)
        }
        MachOperand::Imm(-1) => {
            // AND with all-ones is identity.
            let new_inst = MachInst::new(
                AArch64Target::mov_rr(),
                vec![inst.operands[0].clone(), inst.operands[1].clone()],
            );
            PeepholeResult::Replace(new_inst)
        }
        _ => PeepholeResult::NoChange,
    }
}

// =========================================================================
// New patterns: register-register self operations
// =========================================================================

/// `sub x0, x1, x1` → `mov x0, #0` (SUB self is zero).
fn peephole_sub_rr(inst: &MachInst) -> PeepholeResult {
    if inst.operands.len() < 3 {
        return PeepholeResult::NoChange;
    }

    match (&inst.operands[1], &inst.operands[2]) {
        (MachOperand::VReg(a), MachOperand::VReg(b)) if a.id == b.id => {
            let new_inst = MachInst::new(
                AArch64Target::mov_ri(),
                vec![inst.operands[0].clone(), MachOperand::Imm(0)],
            );
            PeepholeResult::Replace(new_inst)
        }
        _ => PeepholeResult::NoChange,
    }
}

/// `eor x0, x1, x1` → `mov x0, #0` (XOR self is zero).
fn peephole_eor_rr(inst: &MachInst) -> PeepholeResult {
    if inst.operands.len() < 3 {
        return PeepholeResult::NoChange;
    }

    match (&inst.operands[1], &inst.operands[2]) {
        (MachOperand::VReg(a), MachOperand::VReg(b)) if a.id == b.id => {
            let new_inst = MachInst::new(
                AArch64Target::mov_ri(),
                vec![inst.operands[0].clone(), MachOperand::Imm(0)],
            );
            PeepholeResult::Replace(new_inst)
        }
        _ => PeepholeResult::NoChange,
    }
}

/// `add x0, x1, x1` → `lsl x0, x1, #1` (double via shift).
///
/// x + x = 2*x = x << 1. Shift is cheaper than add on most pipelines
/// because it avoids the carry chain.
fn peephole_add_rr(inst: &MachInst) -> PeepholeResult {
    if inst.operands.len() < 3 {
        return PeepholeResult::NoChange;
    }

    match (&inst.operands[1], &inst.operands[2]) {
        (MachOperand::VReg(a), MachOperand::VReg(b)) if a.id == b.id => {
            let new_inst = MachInst::new(
                AArch64Target::shl_ri(),
                vec![
                    inst.operands[0].clone(),
                    inst.operands[1].clone(),
                    MachOperand::Imm(1),
                ],
            );
            PeepholeResult::Replace(new_inst)
        }
        _ => PeepholeResult::NoChange,
    }
}

// =========================================================================
// Helper: power-of-two detection
// =========================================================================

/// Check if `val` is a positive power of two and return log2(val).
///
/// Returns `None` for val <= 0 or non-power-of-two values.
fn is_power_of_two(val: i64) -> Option<u32> {
    if val > 0 && (val & (val - 1)) == 0 {
        Some(val.trailing_zeros())
    } else {
        None
    }
}

// =========================================================================
// Multi-instruction pattern infrastructure
// =========================================================================

/// Build a map from VReg id → InstId for all value-producing instructions
/// in a block. Used for def-use chain lookups in multi-instruction patterns.
fn build_def_map(func: &MachFunction, block_id: BlockId) -> HashMap<u32, InstId> {
    let mut map = HashMap::new();
    let block = func.block(block_id);
    for &inst_id in &block.insts {
        let inst = func.inst(inst_id);
        if inst.opcode.produces_value() {
            if let Some(MachOperand::VReg(dst)) = inst.operands.first() {
                map.insert(dst.id, inst_id);
            }
        }
    }
    map
}

/// Look up the defining instruction for a VReg within the same block.
fn lookup_def<'a>(
    vreg_id: u32,
    func: &'a MachFunction,
    def_map: &HashMap<u32, InstId>,
) -> Option<&'a MachInst> {
    def_map.get(&vreg_id).map(|&id| func.inst(id))
}

/// Try multi-instruction peephole patterns that require looking up the
/// defining instruction of source VRegs.
///
/// ## Patterns:
/// - 20: `neg(neg(x))` → `mov x` (double negation)
/// - 21: `add x, neg(y)` → `sub x, y` (add of negation)
/// - 22: `sub x, neg(y)` → `add x, y` (sub of negation)
/// - 23: `mul x, #2^k` → `lsl x, #k` (multiply by power of 2)
/// - 24: `udiv x, #2^k` → `lsr x, #k` (unsigned div by power of 2)
/// - 25: `mul x, #1` → `mov x` (multiply by one identity)
/// - 28: `sxtb(sxtb(x))` → `sxtb(x)` (double extension elimination)
/// - 29: `lsl(lsr(x, n), n)` → `and x, mask` (shift-pair clear low bits)
/// - 30: `neg(sub(x, y))` → `sub(y, x)` (negate subtraction)
/// - 31: `mul x, #-1` → `neg x` (multiply by -1)
/// - 32: `lsr(lsl(x, n), n)` → `and x, mask` (shift-pair clear high bits)
/// - 33: `sdiv x, #1` → `mov x` (signed division by one identity)
/// - 34: `mul x, #0` → `mov x, #0` (multiply by zero absorbing)
/// - 35: `madd x, y, #1, z` → `add x, y, z` (multiply-add with multiplier 1)
/// - 36: `madd x, y, #0, z` → `mov x, z` (multiply-add with multiplier 0)
/// - 37: `msub x, y, #1, z` → `sub x, z, y` (multiply-sub with multiplier 1)
/// - 38: `msub x, y, #0, z` → `mov x, z` (multiply-sub with multiplier 0)
/// - 39: `sxth(sxtb(x))` → `sxtb(x)` (narrower extension subsumes wider)
/// - 40: `sxtw(sxtb(x))` or `sxtw(sxth(x))` → narrower (subsumes wider)
/// - 41: `add(sub(x, y), y)` → `mov x` (cross-instruction cancellation)
/// - 42: `sub(add(x, y), y)` → `mov x` (cross-instruction cancellation)
/// - 44: `udiv x, #1` → `mov x` (unsigned divide by one)
/// - 45: `sdiv x, #-1` → `neg x` (signed divide by negative one)
/// - 46: `uxtb(uxtb(x))` → `uxtb(x)` (double zero extension idempotent)
/// - 47: `uxth(uxth(x))` → `uxth(x)` (double zero extension idempotent)
/// - 48: `uxth(uxtb(x))` → `uxtb(x)` (narrower zero ext subsumes wider)
/// - 49: `uxtw(uxtb(x))` → `uxtb(x)` (narrower zero ext subsumes wider)
/// - 50: `uxtw(uxth(x))` → `uxth(x)` (narrower zero ext subsumes wider)
/// - 51: `add(x, sub(y, x))` → `mov y` (cross-instruction cancellation)
/// - 52: `eor(eor(x, y), y)` → `mov x` (XOR cancellation)
fn try_peephole_with_defs(
    inst: &MachInst,
    func: &MachFunction,
    def_map: &HashMap<u32, InstId>,
) -> PeepholeResult {
    match inst.opcode {
        // Patterns 20 / 30 (neg-neg, neg-sub-swap) migrated to the
        // declarative rewrite framework (#393 Wave 3/4). The peephole
        // pass no longer handles `Neg` — the declarative pass fires
        // first in the pipeline.
        AArch64Opcode::Neg => PeepholeResult::NoChange,
        // Patterns 21 (add-of-neg), 41 (add-sub-cancel), 51 (add-sub commuted)
        // migrated to Wave 4/5. AddRR is now handled entirely by the
        // declarative rewrite framework.
        AArch64Opcode::AddRR => PeepholeResult::NoChange,
        // Patterns 22 (sub-of-neg), 42 (sub-add-cancel) migrated to
        // Wave 4/5. SubRR is now handled entirely by the declarative
        // framework.
        AArch64Opcode::SubRR => PeepholeResult::NoChange,
        // Pattern 52 (eor cancellation) migrated to Wave 5.
        AArch64Opcode::EorRR => PeepholeResult::NoChange,
        // Patterns 25/31/34 (mul-by-constant) migrated to Wave 5. Pattern
        // 23 (mul by power-of-two → LSL) remains here.
        //
        // BLOCKED on constraint API extension (#393):
        //   Needs `DefinerImmIs { idx, def_operand_idx, pred: fn(i64) -> bool }`
        //   — predicate-based immediate match on a definer, analogous to the
        //   existing `ImmIs` but reaching through `def_map`. The rewriter
        //   also needs access to the concrete matched immediate (to compute
        //   `log2(value)` for the shift amount). The current API only
        //   supports equality (`DefinerImmEquals { value: i64 }`), so we'd
        //   need either a side-channel in `MatchCtx` or a new constraint
        //   that records the matched imm into the ctx for the rewriter.
        //   Same extension would also unblock pattern 24 (UDIV pow2).
        AArch64Opcode::MulRR => peephole_mul_pow2(inst, func, def_map),
        // Patterns 33/44/45 (div-by-one / sdiv-by-neg-one) migrated to
        // Wave 5. Pattern 24 (udiv by power of 2 → LSR) remains here.
        //
        // BLOCKED on same `DefinerImmIs` predicate + matched-imm-readback
        // extension described above (#393). Until then, power-of-two
        // detection must live in the legacy helper.
        AArch64Opcode::UDiv => peephole_udiv_pow2(inst, func, def_map),
        AArch64Opcode::SDiv => PeepholeResult::NoChange,
        // Patterns 35/36 (madd #1 / #0) and 37/38 (msub #1 / #0) migrated to
        // Wave 6. MADD/MSUB are now handled entirely by the declarative
        // framework. See `rewrite::patterns::rule_madd_by_*` and
        // `rewrite::patterns::rule_msub_by_*`.
        AArch64Opcode::Madd => PeepholeResult::NoChange,
        AArch64Opcode::Msub => PeepholeResult::NoChange,
        AArch64Opcode::Sxtb | AArch64Opcode::Sxth | AArch64Opcode::Sxtw => {
            // Patterns 39/40 (narrower-subsumes-wider) migrated to Wave 4.
            // Only the same-opcode double-ext collapse (28) remains.
            peephole_double_ext(inst, func, def_map)
        }
        AArch64Opcode::Uxtb | AArch64Opcode::Uxth | AArch64Opcode::Uxtw => {
            // Patterns 48/49/50 (narrower-subsumes-wider zext) migrated
            // to Wave 4. Only same-opcode double-zext (46/47) remains.
            peephole_double_zext(inst, func, def_map)
        }
        // Patterns 29 (`lsl(lsr(x,#k),#k)` → `and x, #mask_hi`) and
        // 32 (`lsr(lsl(x,#k),#k)` → `and x, #mask_lo`) remain here.
        //
        // BLOCKED on a two-imm equality constraint (#393):
        //   Needs `OuterImmEqualsDefinerImm { outer_imm_idx, outer_def_idx,
        //   def_imm_idx }` — proves the outer shift amount matches the
        //   definer shift amount. Additionally the rewriter needs to
        //   compute the AND mask at runtime (`(1<<w - 1) << k` or
        //   `(1<<w - 1) >> k`) which requires either a compute-on-match
        //   hook or surfacing the matched imm value through `MatchCtx`.
        //   Until that extension lands, these collapses live in the
        //   legacy helpers.
        AArch64Opcode::LslRI => peephole_lsl_lsr_to_and(inst, func, def_map),
        AArch64Opcode::LsrRI => peephole_lsr_lsl_to_and(inst, func, def_map),
        _ => PeepholeResult::NoChange,
    }
}

// Pattern 20 (neg-neg double-negation) migrated to the declarative
// rewrite framework in Wave 3 (#393). See
// `rewrite::patterns::rule_neg_neg`.

// Patterns 21 (add-of-neg) and 22 (sub-of-neg) migrated to the declarative
// rewrite framework in Wave 4 (#393). See
// `rewrite::patterns::rule_add_neg_rhs`, `rule_add_neg_lhs`, and
// `rule_sub_neg_to_add`.

/// Pattern 23: `mul x, #2^k` → `lsl x, #k` (multiply by power of 2).
///
/// When one operand of MUL is defined by a MovI loading a power of two,
/// replace with a left shift. MUL is commutative, so both operands are
/// checked.
fn peephole_mul_pow2(
    inst: &MachInst,
    func: &MachFunction,
    def_map: &HashMap<u32, InstId>,
) -> PeepholeResult {
    if inst.operands.len() < 3 {
        return PeepholeResult::NoChange;
    }

    // Check operand[2] first.
    if let Some(result) = try_mul_pow2_operand(inst, 2, 1, func, def_map) {
        return result;
    }
    // Check operand[1] (commutative).
    if let Some(result) = try_mul_pow2_operand(inst, 1, 2, func, def_map) {
        return result;
    }

    PeepholeResult::NoChange
}

/// Helper for pattern 23: check if operand at `const_idx` is a power-of-2
/// MovI, and if so, replace MUL with LSL using operand at `other_idx`.
fn try_mul_pow2_operand(
    inst: &MachInst,
    const_idx: usize,
    other_idx: usize,
    func: &MachFunction,
    def_map: &HashMap<u32, InstId>,
) -> Option<PeepholeResult> {
    if let MachOperand::VReg(vr) = &inst.operands[const_idx] {
        if let Some(def_inst) = lookup_def(vr.id, func, def_map) {
            if def_inst.opcode == AArch64Opcode::MovI && def_inst.operands.len() >= 2 {
                if let MachOperand::Imm(val) = def_inst.operands[1] {
                    if let Some(shift) = is_power_of_two(val) {
                        let new_inst = MachInst::new(
                            AArch64Target::shl_ri(),
                            vec![
                                inst.operands[0].clone(),
                                inst.operands[other_idx].clone(),
                                MachOperand::Imm(shift as i64),
                            ],
                        );
                        return Some(PeepholeResult::Replace(new_inst));
                    }
                }
            }
        }
    }
    None
}

// Patterns 25/31/34 (MUL by constant) migrated to the declarative rewrite
// framework in Wave 5 (#393). See `rewrite::patterns::rule_mul_by_zero_*`,
// `rule_mul_by_one_*`, and `rule_mul_by_neg_one_*`. Pattern 23 (MUL by
// power of 2) remains here — handled by `peephole_mul_pow2` above.

// =========================================================================
// Pattern 26: BIC self → MOV #0 (BicRR x, y, y → mov x, #0)
// =========================================================================

/// `bic x0, x1, x1` → `mov x0, #0` (BIC self: x AND NOT(x) = 0).
fn peephole_bic_self(inst: &MachInst) -> PeepholeResult {
    if inst.operands.len() < 3 {
        return PeepholeResult::NoChange;
    }

    match (&inst.operands[1], &inst.operands[2]) {
        (MachOperand::VReg(a), MachOperand::VReg(b)) if a.id == b.id => {
            let new_inst = MachInst::new(
                AArch64Target::mov_ri(),
                vec![inst.operands[0].clone(), MachOperand::Imm(0)],
            );
            PeepholeResult::Replace(new_inst)
        }
        _ => PeepholeResult::NoChange,
    }
}

// =========================================================================
// Pattern 27: ORN self → MOV #-1 (OrnRR x, y, y → mov x, #-1)
// =========================================================================

/// `orn x0, x1, x1` → `mov x0, #-1` (ORN self: x OR NOT(x) = all-ones).
fn peephole_orn_self(inst: &MachInst) -> PeepholeResult {
    if inst.operands.len() < 3 {
        return PeepholeResult::NoChange;
    }

    match (&inst.operands[1], &inst.operands[2]) {
        (MachOperand::VReg(a), MachOperand::VReg(b)) if a.id == b.id => {
            let new_inst = MachInst::new(
                AArch64Target::mov_ri(),
                vec![inst.operands[0].clone(), MachOperand::Imm(-1)],
            );
            PeepholeResult::Replace(new_inst)
        }
        _ => PeepholeResult::NoChange,
    }
}

// =========================================================================
// Pattern 28: Double sign extension elimination
// =========================================================================

/// `sxtb(sxtb(x))` → `sxtb(x)`, `sxth(sxth(x))` → `sxth(x)`,
/// `sxtw(sxtw(x))` → `sxtw(x)` (sign extension is idempotent).
fn peephole_double_ext(
    inst: &MachInst,
    func: &MachFunction,
    def_map: &HashMap<u32, InstId>,
) -> PeepholeResult {
    if inst.operands.len() < 2 {
        return PeepholeResult::NoChange;
    }

    if let MachOperand::VReg(src) = &inst.operands[1] {
        if let Some(def_inst) = lookup_def(src.id, func, def_map) {
            // If the source was defined by the same extension opcode,
            // replace with the extension applied to the inner source.
            if def_inst.opcode == inst.opcode && def_inst.operands.len() >= 2 {
                let new_inst = MachInst::new(
                    inst.opcode,
                    vec![inst.operands[0].clone(), def_inst.operands[1].clone()],
                );
                return PeepholeResult::Replace(new_inst);
            }
        }
    }
    PeepholeResult::NoChange
}

// =========================================================================
// Pattern 29: LSL(LSR(x, n), n) → AND x, mask (clear low bits)
// =========================================================================

/// `lsl(lsr(x, n), n)` → `and x, mask` where mask = !((1 << n) - 1).
///
/// Shifting right by n then left by the same n clears the low n bits.
/// This is equivalent to AND with a mask that has the low n bits zeroed.
fn peephole_lsl_lsr_to_and(
    inst: &MachInst,
    func: &MachFunction,
    def_map: &HashMap<u32, InstId>,
) -> PeepholeResult {
    if inst.operands.len() < 3 {
        return PeepholeResult::NoChange;
    }

    // inst is LslRI: check that shift amount is in range.
    if let MachOperand::Imm(lsl_shift) = &inst.operands[2] {
        let k = *lsl_shift;
        if k < 1 || k > 62 {
            return PeepholeResult::NoChange;
        }

        // Check if source of LSL was defined by LSR with same shift amount.
        if let MachOperand::VReg(src) = &inst.operands[1] {
            if let Some(def_inst) = lookup_def(src.id, func, def_map) {
                if def_inst.opcode == AArch64Opcode::LsrRI
                    && def_inst.operands.len() >= 3
                {
                    if let MachOperand::Imm(lsr_shift) = &def_inst.operands[2] {
                        if *lsr_shift == k {
                            // mask = !((1u64 << k) - 1) = clear low k bits
                            let mask = !((1u64 << k) - 1) as i64;
                            let new_inst = MachInst::new(
                                AArch64Opcode::AndRI,
                                vec![
                                    inst.operands[0].clone(),
                                    def_inst.operands[1].clone(),
                                    MachOperand::Imm(mask),
                                ],
                            );
                            return PeepholeResult::Replace(new_inst);
                        }
                    }
                }
            }
        }
    }
    PeepholeResult::NoChange
}

// Pattern 30 (neg-of-sub swap) migrated to the declarative rewrite
// framework in Wave 4 (#393). See `rewrite::patterns::rule_neg_sub_swap`.

// =========================================================================
// Pattern 32: LSR(LSL(x, n), n) → AND x, mask (clear high bits)
// =========================================================================

/// `lsr(lsl(x, n), n)` → `and x, mask` where mask = (1 << (64 - n)) - 1.
///
/// Shifting left by n then right by the same n clears the high n bits.
/// This is equivalent to zero-extension / masking off the top bits.
fn peephole_lsr_lsl_to_and(
    inst: &MachInst,
    func: &MachFunction,
    def_map: &HashMap<u32, InstId>,
) -> PeepholeResult {
    if inst.operands.len() < 3 {
        return PeepholeResult::NoChange;
    }

    // inst is LsrRI: check that shift amount is in range.
    if let MachOperand::Imm(lsr_shift) = &inst.operands[2] {
        let k = *lsr_shift;
        if k < 1 || k > 62 {
            return PeepholeResult::NoChange;
        }

        // Check if source of LSR was defined by LSL with same shift amount.
        if let MachOperand::VReg(src) = &inst.operands[1] {
            if let Some(def_inst) = lookup_def(src.id, func, def_map) {
                if def_inst.opcode == AArch64Opcode::LslRI
                    && def_inst.operands.len() >= 3
                {
                    if let MachOperand::Imm(lsl_shift) = &def_inst.operands[2] {
                        if *lsl_shift == k {
                            // mask = (1u64 << (64 - k)) - 1 = clear high k bits
                            // Equivalent to u64::MAX >> k
                            let mask = (u64::MAX >> k) as i64;
                            let new_inst = MachInst::new(
                                AArch64Opcode::AndRI,
                                vec![
                                    inst.operands[0].clone(),
                                    def_inst.operands[1].clone(),
                                    MachOperand::Imm(mask),
                                ],
                            );
                            return PeepholeResult::Replace(new_inst);
                        }
                    }
                }
            }
        }
    }
    PeepholeResult::NoChange
}

// Pattern 33 (sdiv by 1) is now handled by peephole_sdiv_chain below.

// Patterns 35-36 (MADD imm 0/1) migrated to the declarative rewrite
// framework in Wave 6 (#393). See `rewrite::patterns::rule_madd_by_*`.
// The legacy `peephole_madd` helper was removed as part of the
// zero-warnings cleanup — its switch arm in `apply_pattern` already
// returns `NoChange` (see the `AArch64Opcode::Madd` branch above).

// Patterns 37-38 (MSUB imm 0/1) migrated to the declarative rewrite
// framework in Wave 6 (#393). See `rewrite::patterns::rule_msub_by_*`.
// The legacy `peephole_msub` helper was removed for the same reason.

// Patterns 39/40 (narrower sign-extension subsumes wider) migrated to
// the declarative rewrite framework in Wave 4 (#393). See
// `rewrite::patterns::rule_sxth_of_sxtb` and
// `rule_sxtw_of_narrower_sext`.

// Patterns 41 (ADD-SUB cancel), 42 (SUB-ADD cancel), 51 (ADD-SUB commuted)
// migrated to the declarative rewrite framework in Wave 5 (#393). See
// `rewrite::patterns::rule_add_of_sub_cancel_lhs_def`,
// `rule_add_of_sub_cancel_rhs_def`, `rule_sub_of_add_cancel_rhs`, and
// `rule_sub_of_add_cancel_lhs`.

// =========================================================================
// Pattern 43: CSEL Xd, Xn, Xm, cond where Xn==Xm → MOV Xd, Xn
// =========================================================================

/// `csel Xd, Xn, Xm, cond` where Xn == Xm → `mov Xd, Xn`.
///
/// If both arms of a conditional select are the same register, the condition
/// is irrelevant and the instruction is an unconditional move.
/// CSEL operands: [dst, true_src, false_src, Imm(cond_code)].
fn peephole_csel_same_arms(inst: &MachInst) -> PeepholeResult {
    if inst.operands.len() < 4 {
        return PeepholeResult::NoChange;
    }

    match (&inst.operands[1], &inst.operands[2]) {
        (MachOperand::VReg(a), MachOperand::VReg(b)) if a.id == b.id => {
            let new_inst = MachInst::new(
                AArch64Opcode::MovR,
                vec![inst.operands[0].clone(), inst.operands[1].clone()],
            );
            PeepholeResult::Replace(new_inst)
        }
        _ => PeepholeResult::NoChange,
    }
}

// =========================================================================
// Pattern 24: UDIV x, #2^k → LSR x, #k (unsigned divide by power of 2)
// =========================================================================

/// `udiv x, y, MovI(#2^k)` → `lsr x, y, #k`.
///
/// Patterns 33/44/45 (UDIV/SDIV by ±1) migrated to the declarative
/// rewrite framework in Wave 5 (#393). Pattern 24 remains here because
/// it requires a power-of-two predicate on a definer's immediate plus a
/// `log2` computation in the rewriter — neither is yet exposed by a
/// declarative constraint.
fn peephole_udiv_pow2(
    inst: &MachInst,
    func: &MachFunction,
    def_map: &HashMap<u32, InstId>,
) -> PeepholeResult {
    if inst.operands.len() < 3 {
        return PeepholeResult::NoChange;
    }

    if let MachOperand::VReg(divisor) = &inst.operands[2] {
        if let Some(def_inst) = lookup_def(divisor.id, func, def_map) {
            if def_inst.opcode == AArch64Opcode::MovI && def_inst.operands.len() >= 2 {
                if let MachOperand::Imm(val) = def_inst.operands[1] {
                    // Pattern 24: UDIV x, #2^k → LSR x, #k.
                    // We must not fire on #1 (shift=0) — that case is
                    // migrated to `rule_udiv_by_one` in the declarative
                    // framework and would collide here. `is_power_of_two`
                    // treats 1 as a power of two, so guard explicitly.
                    if val > 1 {
                        if let Some(shift) = is_power_of_two(val) {
                            let new_inst = MachInst::new(
                                AArch64Opcode::LsrRI,
                                vec![
                                    inst.operands[0].clone(),
                                    inst.operands[1].clone(),
                                    MachOperand::Imm(shift as i64),
                                ],
                            );
                            return PeepholeResult::Replace(new_inst);
                        }
                    }
                }
            }
        }
    }

    PeepholeResult::NoChange
}

// =========================================================================
// Patterns 46-47: Double zero extension elimination (idempotent)
// =========================================================================

/// `uxtb(uxtb(x))` → `uxtb(x)`, `uxth(uxth(x))` → `uxth(x)`,
/// `uxtw(uxtw(x))` → `uxtw(x)` (zero extension is idempotent).
fn peephole_double_zext(
    inst: &MachInst,
    func: &MachFunction,
    def_map: &HashMap<u32, InstId>,
) -> PeepholeResult {
    if inst.operands.len() < 2 {
        return PeepholeResult::NoChange;
    }

    if let MachOperand::VReg(src) = &inst.operands[1] {
        if let Some(def_inst) = lookup_def(src.id, func, def_map) {
            // If the source was defined by the same zero-extension opcode,
            // replace with the extension applied to the inner source.
            if def_inst.opcode == inst.opcode && def_inst.operands.len() >= 2 {
                let new_inst = MachInst::new(
                    inst.opcode,
                    vec![inst.operands[0].clone(), def_inst.operands[1].clone()],
                );
                return PeepholeResult::Replace(new_inst);
            }
        }
    }
    PeepholeResult::NoChange
}

// Patterns 48/49/50 (narrower zero-extension subsumes wider) migrated to
// the declarative rewrite framework in Wave 4 (#393). See
// `rewrite::patterns::rule_uxth_of_uxtb` and
// `rule_uxtw_of_narrower_zext`.

// Pattern 52 (EOR-EOR cancellation, four commutations) migrated to the
// declarative rewrite framework in Wave 5 (#393). See
// `rewrite::patterns::rule_eor_cancel_*`.

#[cfg(test)]
mod tests {
    use super::*;
    use llvm2_ir::{
        AArch64Opcode, InstId, MachFunction, MachInst, MachOperand,
        ProofAnnotation, RegClass, Signature, VReg,
    };
    use crate::pass_manager::MachinePass;

    fn vreg(id: u32) -> MachOperand {
        MachOperand::VReg(VReg::new(id, RegClass::Gpr64))
    }

    fn imm(val: i64) -> MachOperand {
        MachOperand::Imm(val)
    }

    fn make_func_with_insts(insts: Vec<MachInst>) -> MachFunction {
        let mut func = MachFunction::new(
            "test_peephole".to_string(),
            Signature::new(vec![], vec![]),
        );
        let block = func.entry;
        for inst in insts {
            let id = func.push_inst(inst);
            func.append_inst(block, id);
        }
        func
    }

    #[test]
    fn test_delete_self_move() {
        let mov = MachInst::new(AArch64Opcode::MovR, vec![vreg(0), vreg(0)]);
        let ret = MachInst::new(AArch64Opcode::Ret, vec![]);
        let mut func = make_func_with_insts(vec![mov, ret]);

        let mut peep = Peephole;
        assert!(peep.run(&mut func));

        let block = func.block(func.entry);
        assert_eq!(block.insts.len(), 1); // only ret
    }

    #[test]
    fn test_add_zero_to_mov() {
        // add v0, v1, #0 → mov v0, v1
        let add = MachInst::new(AArch64Opcode::AddRI, vec![vreg(0), vreg(1), imm(0)]);
        let ret = MachInst::new(AArch64Opcode::Ret, vec![]);
        let mut func = make_func_with_insts(vec![add, ret]);

        let mut peep = Peephole;
        assert!(peep.run(&mut func));

        let inst = func.inst(InstId(0));
        assert_eq!(inst.opcode, AArch64Opcode::MovR);
        assert_eq!(inst.operands[0], vreg(0));
        assert_eq!(inst.operands[1], vreg(1));
    }

    #[test]
    fn test_sub_zero_to_mov() {
        let sub = MachInst::new(AArch64Opcode::SubRI, vec![vreg(0), vreg(1), imm(0)]);
        let ret = MachInst::new(AArch64Opcode::Ret, vec![]);
        let mut func = make_func_with_insts(vec![sub, ret]);

        let mut peep = Peephole;
        assert!(peep.run(&mut func));

        let inst = func.inst(InstId(0));
        assert_eq!(inst.opcode, AArch64Opcode::MovR);
    }

    #[test]
    fn test_lsl_zero_to_mov() {
        let lsl = MachInst::new(AArch64Opcode::LslRI, vec![vreg(0), vreg(1), imm(0)]);
        let ret = MachInst::new(AArch64Opcode::Ret, vec![]);
        let mut func = make_func_with_insts(vec![lsl, ret]);

        let mut peep = Peephole;
        assert!(peep.run(&mut func));

        let inst = func.inst(InstId(0));
        assert_eq!(inst.opcode, AArch64Opcode::MovR);
    }

    #[test]
    fn test_lsr_zero_to_mov() {
        let lsr = MachInst::new(AArch64Opcode::LsrRI, vec![vreg(0), vreg(1), imm(0)]);
        let ret = MachInst::new(AArch64Opcode::Ret, vec![]);
        let mut func = make_func_with_insts(vec![lsr, ret]);

        let mut peep = Peephole;
        assert!(peep.run(&mut func));

        let inst = func.inst(InstId(0));
        assert_eq!(inst.opcode, AArch64Opcode::MovR);
    }

    #[test]
    fn test_orr_self_to_mov() {
        // orr v0, v1, v1 → mov v0, v1
        let orr = MachInst::new(AArch64Opcode::OrrRR, vec![vreg(0), vreg(1), vreg(1)]);
        let ret = MachInst::new(AArch64Opcode::Ret, vec![]);
        let mut func = make_func_with_insts(vec![orr, ret]);

        let mut peep = Peephole;
        assert!(peep.run(&mut func));

        let inst = func.inst(InstId(0));
        assert_eq!(inst.opcode, AArch64Opcode::MovR);
    }

    #[test]
    fn test_and_self_to_mov() {
        let and = MachInst::new(AArch64Opcode::AndRR, vec![vreg(0), vreg(1), vreg(1)]);
        let ret = MachInst::new(AArch64Opcode::Ret, vec![]);
        let mut func = make_func_with_insts(vec![and, ret]);

        let mut peep = Peephole;
        assert!(peep.run(&mut func));

        let inst = func.inst(InstId(0));
        assert_eq!(inst.opcode, AArch64Opcode::MovR);
    }

    #[test]
    fn test_no_change_nonzero_add() {
        let add = MachInst::new(AArch64Opcode::AddRI, vec![vreg(0), vreg(1), imm(5)]);
        let ret = MachInst::new(AArch64Opcode::Ret, vec![]);
        let mut func = make_func_with_insts(vec![add, ret]);

        let mut peep = Peephole;
        assert!(!peep.run(&mut func));
    }

    #[test]
    fn test_no_change_different_regs_orr() {
        let orr = MachInst::new(AArch64Opcode::OrrRR, vec![vreg(0), vreg(1), vreg(2)]);
        let ret = MachInst::new(AArch64Opcode::Ret, vec![]);
        let mut func = make_func_with_insts(vec![orr, ret]);

        let mut peep = Peephole;
        assert!(!peep.run(&mut func));
    }

    #[test]
    fn test_delete_nop() {
        let nop = MachInst::new(AArch64Opcode::Nop, vec![]);
        let ret = MachInst::new(AArch64Opcode::Ret, vec![]);
        let mut func = make_func_with_insts(vec![nop, ret]);

        let mut peep = Peephole;
        assert!(peep.run(&mut func));

        let block = func.block(func.entry);
        assert_eq!(block.insts.len(), 1);
    }

    #[test]
    fn test_multiple_peepholes_in_one_pass() {
        // self-move + add-zero: both should be optimized
        let mov = MachInst::new(AArch64Opcode::MovR, vec![vreg(0), vreg(0)]);
        let add = MachInst::new(AArch64Opcode::AddRI, vec![vreg(1), vreg(2), imm(0)]);
        let ret = MachInst::new(AArch64Opcode::Ret, vec![]);
        let mut func = make_func_with_insts(vec![mov, add, ret]);

        let mut peep = Peephole;
        assert!(peep.run(&mut func));

        let block = func.block(func.entry);
        // Self-move deleted, add replaced with mov
        assert_eq!(block.insts.len(), 2); // mov (from add) + ret

        let add_inst = func.inst(InstId(1));
        assert_eq!(add_inst.opcode, AArch64Opcode::MovR);
    }

    // ---- Proof annotation preservation tests ----

    #[test]
    fn test_add_zero_preserves_proof() {
        // add v0, v1, #0 [NoOverflow] -> mov v0, v1 [NoOverflow]
        let add = MachInst::new(AArch64Opcode::AddRI, vec![vreg(0), vreg(1), imm(0)])
            .with_proof(ProofAnnotation::NoOverflow);
        let ret = MachInst::new(AArch64Opcode::Ret, vec![]);
        let mut func = make_func_with_insts(vec![add, ret]);

        let mut peep = Peephole;
        assert!(peep.run(&mut func));

        let inst = func.inst(InstId(0));
        assert_eq!(inst.opcode, AArch64Opcode::MovR);
        assert_eq!(inst.proof, Some(ProofAnnotation::NoOverflow));
    }

    #[test]
    fn test_sub_zero_preserves_proof() {
        // sub v0, v1, #0 [InBounds] -> mov v0, v1 [InBounds]
        let sub = MachInst::new(AArch64Opcode::SubRI, vec![vreg(0), vreg(1), imm(0)])
            .with_proof(ProofAnnotation::InBounds);
        let ret = MachInst::new(AArch64Opcode::Ret, vec![]);
        let mut func = make_func_with_insts(vec![sub, ret]);

        let mut peep = Peephole;
        assert!(peep.run(&mut func));

        let inst = func.inst(InstId(0));
        assert_eq!(inst.opcode, AArch64Opcode::MovR);
        assert_eq!(inst.proof, Some(ProofAnnotation::InBounds));
    }

    #[test]
    fn test_shift_zero_preserves_proof() {
        // lsl v0, v1, #0 [NotNull] -> mov v0, v1 [NotNull]
        let lsl = MachInst::new(AArch64Opcode::LslRI, vec![vreg(0), vreg(1), imm(0)])
            .with_proof(ProofAnnotation::NotNull);
        let ret = MachInst::new(AArch64Opcode::Ret, vec![]);
        let mut func = make_func_with_insts(vec![lsl, ret]);

        let mut peep = Peephole;
        assert!(peep.run(&mut func));

        let inst = func.inst(InstId(0));
        assert_eq!(inst.opcode, AArch64Opcode::MovR);
        assert_eq!(inst.proof, Some(ProofAnnotation::NotNull));
    }

    #[test]
    fn test_logical_self_preserves_proof() {
        // orr v0, v1, v1 [ValidBorrow] -> mov v0, v1 [ValidBorrow]
        let orr = MachInst::new(AArch64Opcode::OrrRR, vec![vreg(0), vreg(1), vreg(1)])
            .with_proof(ProofAnnotation::ValidBorrow);
        let ret = MachInst::new(AArch64Opcode::Ret, vec![]);
        let mut func = make_func_with_insts(vec![orr, ret]);

        let mut peep = Peephole;
        assert!(peep.run(&mut func));

        let inst = func.inst(InstId(0));
        assert_eq!(inst.opcode, AArch64Opcode::MovR);
        assert_eq!(inst.proof, Some(ProofAnnotation::ValidBorrow));
    }

    #[test]
    fn test_no_proof_no_proof_after_peephole() {
        // add v0, v1, #0 (no proof) -> mov v0, v1 (no proof)
        let add = MachInst::new(AArch64Opcode::AddRI, vec![vreg(0), vreg(1), imm(0)]);
        let ret = MachInst::new(AArch64Opcode::Ret, vec![]);
        let mut func = make_func_with_insts(vec![add, ret]);

        let mut peep = Peephole;
        assert!(peep.run(&mut func));

        let inst = func.inst(InstId(0));
        assert_eq!(inst.opcode, AArch64Opcode::MovR);
        assert!(inst.proof.is_none());
    }

    // ================================================================
    // New pattern tests: EorRI
    // ================================================================

    #[test]
    fn test_eor_ri_zero_to_mov() {
        // eor v0, v1, #0 -> mov v0, v1
        let eor = MachInst::new(AArch64Opcode::EorRI, vec![vreg(0), vreg(1), imm(0)]);
        let ret = MachInst::new(AArch64Opcode::Ret, vec![]);
        let mut func = make_func_with_insts(vec![eor, ret]);

        let mut peep = Peephole;
        assert!(peep.run(&mut func));

        let inst = func.inst(InstId(0));
        assert_eq!(inst.opcode, AArch64Opcode::MovR);
        assert_eq!(inst.operands[0], vreg(0));
        assert_eq!(inst.operands[1], vreg(1));
    }

    #[test]
    fn test_eor_ri_nonzero_no_change() {
        let eor = MachInst::new(AArch64Opcode::EorRI, vec![vreg(0), vreg(1), imm(0xFF)]);
        let ret = MachInst::new(AArch64Opcode::Ret, vec![]);
        let mut func = make_func_with_insts(vec![eor, ret]);

        let mut peep = Peephole;
        assert!(!peep.run(&mut func));
    }

    #[test]
    fn test_eor_ri_zero_preserves_proof() {
        let eor = MachInst::new(AArch64Opcode::EorRI, vec![vreg(0), vreg(1), imm(0)])
            .with_proof(ProofAnnotation::NoOverflow);
        let ret = MachInst::new(AArch64Opcode::Ret, vec![]);
        let mut func = make_func_with_insts(vec![eor, ret]);

        let mut peep = Peephole;
        assert!(peep.run(&mut func));

        let inst = func.inst(InstId(0));
        assert_eq!(inst.opcode, AArch64Opcode::MovR);
        assert_eq!(inst.proof, Some(ProofAnnotation::NoOverflow));
    }

    // ================================================================
    // New pattern tests: OrrRI
    // ================================================================

    #[test]
    fn test_orr_ri_zero_to_mov() {
        // orr v0, v1, #0 -> mov v0, v1
        let orr = MachInst::new(AArch64Opcode::OrrRI, vec![vreg(0), vreg(1), imm(0)]);
        let ret = MachInst::new(AArch64Opcode::Ret, vec![]);
        let mut func = make_func_with_insts(vec![orr, ret]);

        let mut peep = Peephole;
        assert!(peep.run(&mut func));

        let inst = func.inst(InstId(0));
        assert_eq!(inst.opcode, AArch64Opcode::MovR);
        assert_eq!(inst.operands[0], vreg(0));
        assert_eq!(inst.operands[1], vreg(1));
    }

    #[test]
    fn test_orr_ri_allones_to_movi() {
        // orr v0, v1, #-1 -> mov v0, #-1
        let orr = MachInst::new(AArch64Opcode::OrrRI, vec![vreg(0), vreg(1), imm(-1)]);
        let ret = MachInst::new(AArch64Opcode::Ret, vec![]);
        let mut func = make_func_with_insts(vec![orr, ret]);

        let mut peep = Peephole;
        assert!(peep.run(&mut func));

        let inst = func.inst(InstId(0));
        assert_eq!(inst.opcode, AArch64Opcode::MovI);
        assert_eq!(inst.operands[0], vreg(0));
        assert_eq!(inst.operands[1], imm(-1));
    }

    #[test]
    fn test_orr_ri_nonzero_no_change() {
        let orr = MachInst::new(AArch64Opcode::OrrRI, vec![vreg(0), vreg(1), imm(0x0F)]);
        let ret = MachInst::new(AArch64Opcode::Ret, vec![]);
        let mut func = make_func_with_insts(vec![orr, ret]);

        let mut peep = Peephole;
        assert!(!peep.run(&mut func));
    }

    #[test]
    fn test_orr_ri_zero_preserves_proof() {
        let orr = MachInst::new(AArch64Opcode::OrrRI, vec![vreg(0), vreg(1), imm(0)])
            .with_proof(ProofAnnotation::InBounds);
        let ret = MachInst::new(AArch64Opcode::Ret, vec![]);
        let mut func = make_func_with_insts(vec![orr, ret]);

        let mut peep = Peephole;
        assert!(peep.run(&mut func));

        let inst = func.inst(InstId(0));
        assert_eq!(inst.opcode, AArch64Opcode::MovR);
        assert_eq!(inst.proof, Some(ProofAnnotation::InBounds));
    }

    // ================================================================
    // New pattern tests: AndRI
    // ================================================================

    #[test]
    fn test_and_ri_zero_to_movi_zero() {
        // and v0, v1, #0 -> mov v0, #0
        let and = MachInst::new(AArch64Opcode::AndRI, vec![vreg(0), vreg(1), imm(0)]);
        let ret = MachInst::new(AArch64Opcode::Ret, vec![]);
        let mut func = make_func_with_insts(vec![and, ret]);

        let mut peep = Peephole;
        assert!(peep.run(&mut func));

        let inst = func.inst(InstId(0));
        assert_eq!(inst.opcode, AArch64Opcode::MovI);
        assert_eq!(inst.operands[0], vreg(0));
        assert_eq!(inst.operands[1], imm(0));
    }

    #[test]
    fn test_and_ri_allones_to_mov() {
        // and v0, v1, #-1 -> mov v0, v1
        let and = MachInst::new(AArch64Opcode::AndRI, vec![vreg(0), vreg(1), imm(-1)]);
        let ret = MachInst::new(AArch64Opcode::Ret, vec![]);
        let mut func = make_func_with_insts(vec![and, ret]);

        let mut peep = Peephole;
        assert!(peep.run(&mut func));

        let inst = func.inst(InstId(0));
        assert_eq!(inst.opcode, AArch64Opcode::MovR);
        assert_eq!(inst.operands[0], vreg(0));
        assert_eq!(inst.operands[1], vreg(1));
    }

    #[test]
    fn test_and_ri_nonzero_no_change() {
        let and = MachInst::new(AArch64Opcode::AndRI, vec![vreg(0), vreg(1), imm(0xFF)]);
        let ret = MachInst::new(AArch64Opcode::Ret, vec![]);
        let mut func = make_func_with_insts(vec![and, ret]);

        let mut peep = Peephole;
        assert!(!peep.run(&mut func));
    }

    #[test]
    fn test_and_ri_allones_preserves_proof() {
        let and = MachInst::new(AArch64Opcode::AndRI, vec![vreg(0), vreg(1), imm(-1)])
            .with_proof(ProofAnnotation::NotNull);
        let ret = MachInst::new(AArch64Opcode::Ret, vec![]);
        let mut func = make_func_with_insts(vec![and, ret]);

        let mut peep = Peephole;
        assert!(peep.run(&mut func));

        let inst = func.inst(InstId(0));
        assert_eq!(inst.opcode, AArch64Opcode::MovR);
        assert_eq!(inst.proof, Some(ProofAnnotation::NotNull));
    }

    // ================================================================
    // New pattern tests: SubRR (self -> zero)
    // ================================================================

    #[test]
    fn test_sub_rr_self_to_zero() {
        // sub v0, v1, v1 -> mov v0, #0
        let sub = MachInst::new(AArch64Opcode::SubRR, vec![vreg(0), vreg(1), vreg(1)]);
        let ret = MachInst::new(AArch64Opcode::Ret, vec![]);
        let mut func = make_func_with_insts(vec![sub, ret]);

        let mut peep = Peephole;
        assert!(peep.run(&mut func));

        let inst = func.inst(InstId(0));
        assert_eq!(inst.opcode, AArch64Opcode::MovI);
        assert_eq!(inst.operands[0], vreg(0));
        assert_eq!(inst.operands[1], imm(0));
    }

    #[test]
    fn test_sub_rr_different_regs_no_change() {
        let sub = MachInst::new(AArch64Opcode::SubRR, vec![vreg(0), vreg(1), vreg(2)]);
        let ret = MachInst::new(AArch64Opcode::Ret, vec![]);
        let mut func = make_func_with_insts(vec![sub, ret]);

        let mut peep = Peephole;
        assert!(!peep.run(&mut func));
    }

    #[test]
    fn test_sub_rr_self_preserves_proof() {
        let sub = MachInst::new(AArch64Opcode::SubRR, vec![vreg(0), vreg(1), vreg(1)])
            .with_proof(ProofAnnotation::NoOverflow);
        let ret = MachInst::new(AArch64Opcode::Ret, vec![]);
        let mut func = make_func_with_insts(vec![sub, ret]);

        let mut peep = Peephole;
        assert!(peep.run(&mut func));

        let inst = func.inst(InstId(0));
        assert_eq!(inst.opcode, AArch64Opcode::MovI);
        assert_eq!(inst.proof, Some(ProofAnnotation::NoOverflow));
    }

    // ================================================================
    // New pattern tests: EorRR (self -> zero)
    // ================================================================

    #[test]
    fn test_eor_rr_self_to_zero() {
        // eor v0, v1, v1 -> mov v0, #0
        let eor = MachInst::new(AArch64Opcode::EorRR, vec![vreg(0), vreg(1), vreg(1)]);
        let ret = MachInst::new(AArch64Opcode::Ret, vec![]);
        let mut func = make_func_with_insts(vec![eor, ret]);

        let mut peep = Peephole;
        assert!(peep.run(&mut func));

        let inst = func.inst(InstId(0));
        assert_eq!(inst.opcode, AArch64Opcode::MovI);
        assert_eq!(inst.operands[0], vreg(0));
        assert_eq!(inst.operands[1], imm(0));
    }

    #[test]
    fn test_eor_rr_different_regs_no_change() {
        let eor = MachInst::new(AArch64Opcode::EorRR, vec![vreg(0), vreg(1), vreg(2)]);
        let ret = MachInst::new(AArch64Opcode::Ret, vec![]);
        let mut func = make_func_with_insts(vec![eor, ret]);

        let mut peep = Peephole;
        assert!(!peep.run(&mut func));
    }

    #[test]
    fn test_eor_rr_self_preserves_proof() {
        let eor = MachInst::new(AArch64Opcode::EorRR, vec![vreg(0), vreg(1), vreg(1)])
            .with_proof(ProofAnnotation::ValidBorrow);
        let ret = MachInst::new(AArch64Opcode::Ret, vec![]);
        let mut func = make_func_with_insts(vec![eor, ret]);

        let mut peep = Peephole;
        assert!(peep.run(&mut func));

        let inst = func.inst(InstId(0));
        assert_eq!(inst.opcode, AArch64Opcode::MovI);
        assert_eq!(inst.proof, Some(ProofAnnotation::ValidBorrow));
    }

    // ================================================================
    // New pattern tests: AddRR (self -> lsl #1)
    // ================================================================

    #[test]
    fn test_add_rr_self_to_lsl() {
        // add v0, v1, v1 -> lsl v0, v1, #1
        let add = MachInst::new(AArch64Opcode::AddRR, vec![vreg(0), vreg(1), vreg(1)]);
        let ret = MachInst::new(AArch64Opcode::Ret, vec![]);
        let mut func = make_func_with_insts(vec![add, ret]);

        let mut peep = Peephole;
        assert!(peep.run(&mut func));

        let inst = func.inst(InstId(0));
        assert_eq!(inst.opcode, AArch64Opcode::LslRI);
        assert_eq!(inst.operands[0], vreg(0));
        assert_eq!(inst.operands[1], vreg(1));
        assert_eq!(inst.operands[2], imm(1));
    }

    #[test]
    fn test_add_rr_different_regs_no_change() {
        let add = MachInst::new(AArch64Opcode::AddRR, vec![vreg(0), vreg(1), vreg(2)]);
        let ret = MachInst::new(AArch64Opcode::Ret, vec![]);
        let mut func = make_func_with_insts(vec![add, ret]);

        let mut peep = Peephole;
        assert!(!peep.run(&mut func));
    }

    #[test]
    fn test_add_rr_self_preserves_proof() {
        let add = MachInst::new(AArch64Opcode::AddRR, vec![vreg(0), vreg(1), vreg(1)])
            .with_proof(ProofAnnotation::InBounds);
        let ret = MachInst::new(AArch64Opcode::Ret, vec![]);
        let mut func = make_func_with_insts(vec![add, ret]);

        let mut peep = Peephole;
        assert!(peep.run(&mut func));

        let inst = func.inst(InstId(0));
        assert_eq!(inst.opcode, AArch64Opcode::LslRI);
        assert_eq!(inst.proof, Some(ProofAnnotation::InBounds));
    }

    // ================================================================
    // Combined new pattern tests
    // ================================================================

    #[test]
    fn test_multiple_new_peepholes_in_one_pass() {
        // sub v0, v1, v1 (zero) + eor v2, v3, #0 (identity) + orr v4, v5, #-1 (all-ones)
        let sub = MachInst::new(AArch64Opcode::SubRR, vec![vreg(0), vreg(1), vreg(1)]);
        let eor = MachInst::new(AArch64Opcode::EorRI, vec![vreg(2), vreg(3), imm(0)]);
        let orr = MachInst::new(AArch64Opcode::OrrRI, vec![vreg(4), vreg(5), imm(-1)]);
        let ret = MachInst::new(AArch64Opcode::Ret, vec![]);
        let mut func = make_func_with_insts(vec![sub, eor, orr, ret]);

        let mut peep = Peephole;
        assert!(peep.run(&mut func));

        // sub self -> MovI #0
        let inst0 = func.inst(InstId(0));
        assert_eq!(inst0.opcode, AArch64Opcode::MovI);
        assert_eq!(inst0.operands[1], imm(0));

        // eor #0 -> MovR
        let inst1 = func.inst(InstId(1));
        assert_eq!(inst1.opcode, AArch64Opcode::MovR);

        // orr #-1 -> MovI #-1
        let inst2 = func.inst(InstId(2));
        assert_eq!(inst2.opcode, AArch64Opcode::MovI);
        assert_eq!(inst2.operands[1], imm(-1));
    }

    #[test]
    fn test_and_zero_plus_add_self_combined() {
        // and v0, v1, #0 (zero) + add v2, v3, v3 (double via lsl)
        let and = MachInst::new(AArch64Opcode::AndRI, vec![vreg(0), vreg(1), imm(0)]);
        let add = MachInst::new(AArch64Opcode::AddRR, vec![vreg(2), vreg(3), vreg(3)]);
        let ret = MachInst::new(AArch64Opcode::Ret, vec![]);
        let mut func = make_func_with_insts(vec![and, add, ret]);

        let mut peep = Peephole;
        assert!(peep.run(&mut func));

        // and #0 -> MovI #0
        let inst0 = func.inst(InstId(0));
        assert_eq!(inst0.opcode, AArch64Opcode::MovI);
        assert_eq!(inst0.operands[1], imm(0));

        // add self -> LslRI #1
        let inst1 = func.inst(InstId(1));
        assert_eq!(inst1.opcode, AArch64Opcode::LslRI);
        assert_eq!(inst1.operands[2], imm(1));
    }

    // ================================================================
    // Pattern 18: AddRI negative immediate canonicalization
    // ================================================================

    #[test]
    fn test_add_ri_negative_imm_to_sub() {
        // add v0, v1, #-5 -> sub v0, v1, #5
        let add = MachInst::new(AArch64Opcode::AddRI, vec![vreg(0), vreg(1), imm(-5)]);
        let ret = MachInst::new(AArch64Opcode::Ret, vec![]);
        let mut func = make_func_with_insts(vec![add, ret]);

        let mut peep = Peephole;
        assert!(peep.run(&mut func));

        let inst = func.inst(InstId(0));
        assert_eq!(inst.opcode, AArch64Opcode::SubRI);
        assert_eq!(inst.operands[0], vreg(0));
        assert_eq!(inst.operands[1], vreg(1));
        assert_eq!(inst.operands[2], imm(5));
    }

    #[test]
    fn test_add_ri_negative_imm_preserves_proof() {
        let add = MachInst::new(AArch64Opcode::AddRI, vec![vreg(0), vreg(1), imm(-10)])
            .with_proof(ProofAnnotation::NoOverflow);
        let ret = MachInst::new(AArch64Opcode::Ret, vec![]);
        let mut func = make_func_with_insts(vec![add, ret]);

        let mut peep = Peephole;
        assert!(peep.run(&mut func));

        let inst = func.inst(InstId(0));
        assert_eq!(inst.opcode, AArch64Opcode::SubRI);
        assert_eq!(inst.proof, Some(ProofAnnotation::NoOverflow));
    }

    #[test]
    fn test_add_ri_positive_imm_no_change() {
        let add = MachInst::new(AArch64Opcode::AddRI, vec![vreg(0), vreg(1), imm(5)]);
        let ret = MachInst::new(AArch64Opcode::Ret, vec![]);
        let mut func = make_func_with_insts(vec![add, ret]);

        let mut peep = Peephole;
        assert!(!peep.run(&mut func));
    }

    // ================================================================
    // Pattern 19: SubRI negative immediate canonicalization
    // ================================================================

    #[test]
    fn test_sub_ri_negative_imm_to_add() {
        // sub v0, v1, #-7 -> add v0, v1, #7
        let sub = MachInst::new(AArch64Opcode::SubRI, vec![vreg(0), vreg(1), imm(-7)]);
        let ret = MachInst::new(AArch64Opcode::Ret, vec![]);
        let mut func = make_func_with_insts(vec![sub, ret]);

        let mut peep = Peephole;
        assert!(peep.run(&mut func));

        let inst = func.inst(InstId(0));
        assert_eq!(inst.opcode, AArch64Opcode::AddRI);
        assert_eq!(inst.operands[0], vreg(0));
        assert_eq!(inst.operands[1], vreg(1));
        assert_eq!(inst.operands[2], imm(7));
    }

    #[test]
    fn test_sub_ri_negative_imm_preserves_proof() {
        let sub = MachInst::new(AArch64Opcode::SubRI, vec![vreg(0), vreg(1), imm(-3)])
            .with_proof(ProofAnnotation::InBounds);
        let ret = MachInst::new(AArch64Opcode::Ret, vec![]);
        let mut func = make_func_with_insts(vec![sub, ret]);

        let mut peep = Peephole;
        assert!(peep.run(&mut func));

        let inst = func.inst(InstId(0));
        assert_eq!(inst.opcode, AArch64Opcode::AddRI);
        assert_eq!(inst.proof, Some(ProofAnnotation::InBounds));
    }

    // Patterns 20 (neg-neg), 21 (add-of-neg), and 22 (sub-of-neg) migrated
    // to the declarative rewrite framework in Wave 3/4 (#393). Their
    // behavioural coverage lives in `rewrite::patterns::tests`
    // (`neg_neg_collapses_to_mov`, `add_neg_rhs_rewrites_to_sub`,
    // `add_neg_lhs_rewrites_to_sub_with_swap`,
    // `sub_neg_rewrites_to_add`).

    // ================================================================
    // Pattern 23: MUL x, #2^k → LSL x, #k
    // ================================================================

    #[test]
    fn test_mul_pow2_to_lsl() {
        // v1 = mov #8
        // v2 = mul v0, v1 → v2 = lsl v0, #3
        let movi = MachInst::new(AArch64Opcode::MovI, vec![vreg(1), imm(8)]);
        let mul = MachInst::new(AArch64Opcode::MulRR, vec![vreg(2), vreg(0), vreg(1)]);
        let ret = MachInst::new(AArch64Opcode::Ret, vec![]);
        let mut func = make_func_with_insts(vec![movi, mul, ret]);

        let mut peep = Peephole;
        assert!(peep.run(&mut func));

        let inst = func.inst(InstId(1));
        assert_eq!(inst.opcode, AArch64Opcode::LslRI);
        assert_eq!(inst.operands[0], vreg(2));
        assert_eq!(inst.operands[1], vreg(0));
        assert_eq!(inst.operands[2], imm(3)); // log2(8) = 3
    }

    #[test]
    fn test_mul_pow2_commutative() {
        // v1 = mov #4
        // v2 = mul v1, v0 → v2 = lsl v0, #2
        let movi = MachInst::new(AArch64Opcode::MovI, vec![vreg(1), imm(4)]);
        let mul = MachInst::new(AArch64Opcode::MulRR, vec![vreg(2), vreg(1), vreg(0)]);
        let ret = MachInst::new(AArch64Opcode::Ret, vec![]);
        let mut func = make_func_with_insts(vec![movi, mul, ret]);

        let mut peep = Peephole;
        assert!(peep.run(&mut func));

        let inst = func.inst(InstId(1));
        assert_eq!(inst.opcode, AArch64Opcode::LslRI);
        assert_eq!(inst.operands[0], vreg(2));
        assert_eq!(inst.operands[1], vreg(0));
        assert_eq!(inst.operands[2], imm(2)); // log2(4) = 2
    }

    #[test]
    fn test_mul_non_pow2_no_change() {
        // v1 = mov #5 (not power of 2)
        // v2 = mul v0, v1 → no change
        let movi = MachInst::new(AArch64Opcode::MovI, vec![vreg(1), imm(5)]);
        let mul = MachInst::new(AArch64Opcode::MulRR, vec![vreg(2), vreg(0), vreg(1)]);
        let ret = MachInst::new(AArch64Opcode::Ret, vec![]);
        let mut func = make_func_with_insts(vec![movi, mul, ret]);

        let mut peep = Peephole;
        assert!(!peep.run(&mut func));
    }

    #[test]
    fn test_mul_pow2_preserves_proof() {
        let movi = MachInst::new(AArch64Opcode::MovI, vec![vreg(1), imm(2)]);
        let mul = MachInst::new(AArch64Opcode::MulRR, vec![vreg(2), vreg(0), vreg(1)])
            .with_proof(ProofAnnotation::NoOverflow);
        let ret = MachInst::new(AArch64Opcode::Ret, vec![]);
        let mut func = make_func_with_insts(vec![movi, mul, ret]);

        let mut peep = Peephole;
        assert!(peep.run(&mut func));

        let inst = func.inst(InstId(1));
        assert_eq!(inst.opcode, AArch64Opcode::LslRI);
        assert_eq!(inst.proof, Some(ProofAnnotation::NoOverflow));
    }

    // ================================================================
    // Pattern 24: UDIV x, #2^k → LSR x, #k
    // ================================================================

    #[test]
    fn test_udiv_pow2_to_lsr() {
        // v1 = mov #16
        // v2 = udiv v0, v1 → v2 = lsr v0, #4
        let movi = MachInst::new(AArch64Opcode::MovI, vec![vreg(1), imm(16)]);
        let div = MachInst::new(AArch64Opcode::UDiv, vec![vreg(2), vreg(0), vreg(1)]);
        let ret = MachInst::new(AArch64Opcode::Ret, vec![]);
        let mut func = make_func_with_insts(vec![movi, div, ret]);

        let mut peep = Peephole;
        assert!(peep.run(&mut func));

        let inst = func.inst(InstId(1));
        assert_eq!(inst.opcode, AArch64Opcode::LsrRI);
        assert_eq!(inst.operands[0], vreg(2));
        assert_eq!(inst.operands[1], vreg(0));
        assert_eq!(inst.operands[2], imm(4)); // log2(16) = 4
    }

    #[test]
    fn test_udiv_non_pow2_no_change() {
        let movi = MachInst::new(AArch64Opcode::MovI, vec![vreg(1), imm(7)]);
        let div = MachInst::new(AArch64Opcode::UDiv, vec![vreg(2), vreg(0), vreg(1)]);
        let ret = MachInst::new(AArch64Opcode::Ret, vec![]);
        let mut func = make_func_with_insts(vec![movi, div, ret]);

        let mut peep = Peephole;
        assert!(!peep.run(&mut func));
    }

    #[test]
    fn test_udiv_pow2_preserves_proof() {
        let movi = MachInst::new(AArch64Opcode::MovI, vec![vreg(1), imm(32)]);
        let div = MachInst::new(AArch64Opcode::UDiv, vec![vreg(2), vreg(0), vreg(1)])
            .with_proof(ProofAnnotation::NonZeroDivisor);
        let ret = MachInst::new(AArch64Opcode::Ret, vec![]);
        let mut func = make_func_with_insts(vec![movi, div, ret]);

        let mut peep = Peephole;
        assert!(peep.run(&mut func));

        let inst = func.inst(InstId(1));
        assert_eq!(inst.opcode, AArch64Opcode::LsrRI);
        assert_eq!(inst.proof, Some(ProofAnnotation::NonZeroDivisor));
    }

    // ================================================================
    // Helper: is_power_of_two tests
    // ================================================================

    #[test]
    fn test_is_power_of_two() {
        assert_eq!(is_power_of_two(1), Some(0));
        assert_eq!(is_power_of_two(2), Some(1));
        assert_eq!(is_power_of_two(4), Some(2));
        assert_eq!(is_power_of_two(8), Some(3));
        assert_eq!(is_power_of_two(16), Some(4));
        assert_eq!(is_power_of_two(1024), Some(10));
        assert_eq!(is_power_of_two(0), None);
        assert_eq!(is_power_of_two(-1), None);
        assert_eq!(is_power_of_two(3), None);
        assert_eq!(is_power_of_two(5), None);
        assert_eq!(is_power_of_two(6), None);
    }

    // ================================================================
    // Pattern 25: MUL x, #1 → MOV x
    // Migrated to rewrite/patterns.rs (rule_mul_by_one_rhs / rule_mul_by_one_lhs).
    // Tests moved to rewrite::patterns::tests.
    // ================================================================

    // ================================================================
    // Pattern 26: BIC self → MOV #0
    // ================================================================

    #[test]
    fn test_bic_self_to_zero() {
        // bic v0, v1, v1 → mov v0, #0
        let bic = MachInst::new(AArch64Opcode::BicRR, vec![vreg(0), vreg(1), vreg(1)]);
        let ret = MachInst::new(AArch64Opcode::Ret, vec![]);
        let mut func = make_func_with_insts(vec![bic, ret]);

        let mut peep = Peephole;
        assert!(peep.run(&mut func));

        let inst = func.inst(InstId(0));
        assert_eq!(inst.opcode, AArch64Opcode::MovI);
        assert_eq!(inst.operands[0], vreg(0));
        assert_eq!(inst.operands[1], imm(0));
    }

    #[test]
    fn test_bic_different_regs_no_change() {
        let bic = MachInst::new(AArch64Opcode::BicRR, vec![vreg(0), vreg(1), vreg(2)]);
        let ret = MachInst::new(AArch64Opcode::Ret, vec![]);
        let mut func = make_func_with_insts(vec![bic, ret]);

        let mut peep = Peephole;
        assert!(!peep.run(&mut func));
    }

    #[test]
    fn test_bic_self_preserves_proof() {
        let bic = MachInst::new(AArch64Opcode::BicRR, vec![vreg(0), vreg(1), vreg(1)])
            .with_proof(ProofAnnotation::NoOverflow);
        let ret = MachInst::new(AArch64Opcode::Ret, vec![]);
        let mut func = make_func_with_insts(vec![bic, ret]);

        let mut peep = Peephole;
        assert!(peep.run(&mut func));

        let inst = func.inst(InstId(0));
        assert_eq!(inst.opcode, AArch64Opcode::MovI);
        assert_eq!(inst.proof, Some(ProofAnnotation::NoOverflow));
    }

    // ================================================================
    // Pattern 27: ORN self → MOV #-1
    // ================================================================

    #[test]
    fn test_orn_self_to_allones() {
        // orn v0, v1, v1 → mov v0, #-1
        let orn = MachInst::new(AArch64Opcode::OrnRR, vec![vreg(0), vreg(1), vreg(1)]);
        let ret = MachInst::new(AArch64Opcode::Ret, vec![]);
        let mut func = make_func_with_insts(vec![orn, ret]);

        let mut peep = Peephole;
        assert!(peep.run(&mut func));

        let inst = func.inst(InstId(0));
        assert_eq!(inst.opcode, AArch64Opcode::MovI);
        assert_eq!(inst.operands[0], vreg(0));
        assert_eq!(inst.operands[1], imm(-1));
    }

    #[test]
    fn test_orn_different_regs_no_change() {
        let orn = MachInst::new(AArch64Opcode::OrnRR, vec![vreg(0), vreg(1), vreg(2)]);
        let ret = MachInst::new(AArch64Opcode::Ret, vec![]);
        let mut func = make_func_with_insts(vec![orn, ret]);

        let mut peep = Peephole;
        assert!(!peep.run(&mut func));
    }

    #[test]
    fn test_orn_self_preserves_proof() {
        let orn = MachInst::new(AArch64Opcode::OrnRR, vec![vreg(0), vreg(1), vreg(1)])
            .with_proof(ProofAnnotation::ValidBorrow);
        let ret = MachInst::new(AArch64Opcode::Ret, vec![]);
        let mut func = make_func_with_insts(vec![orn, ret]);

        let mut peep = Peephole;
        assert!(peep.run(&mut func));

        let inst = func.inst(InstId(0));
        assert_eq!(inst.opcode, AArch64Opcode::MovI);
        assert_eq!(inst.proof, Some(ProofAnnotation::ValidBorrow));
    }

    // ================================================================
    // Pattern 28: Double sign extension elimination
    // ================================================================

    #[test]
    fn test_double_sxtb() {
        // v1 = sxtb v0
        // v2 = sxtb v1 → v2 = sxtb v0
        let sxtb1 = MachInst::new(AArch64Opcode::Sxtb, vec![vreg(1), vreg(0)]);
        let sxtb2 = MachInst::new(AArch64Opcode::Sxtb, vec![vreg(2), vreg(1)]);
        let ret = MachInst::new(AArch64Opcode::Ret, vec![]);
        let mut func = make_func_with_insts(vec![sxtb1, sxtb2, ret]);

        let mut peep = Peephole;
        assert!(peep.run(&mut func));

        let inst = func.inst(InstId(1));
        assert_eq!(inst.opcode, AArch64Opcode::Sxtb);
        assert_eq!(inst.operands[0], vreg(2));
        assert_eq!(inst.operands[1], vreg(0));
    }

    #[test]
    fn test_double_sxth() {
        let sxth1 = MachInst::new(AArch64Opcode::Sxth, vec![vreg(1), vreg(0)]);
        let sxth2 = MachInst::new(AArch64Opcode::Sxth, vec![vreg(2), vreg(1)]);
        let ret = MachInst::new(AArch64Opcode::Ret, vec![]);
        let mut func = make_func_with_insts(vec![sxth1, sxth2, ret]);

        let mut peep = Peephole;
        assert!(peep.run(&mut func));

        let inst = func.inst(InstId(1));
        assert_eq!(inst.opcode, AArch64Opcode::Sxth);
        assert_eq!(inst.operands[0], vreg(2));
        assert_eq!(inst.operands[1], vreg(0));
    }

    #[test]
    fn test_double_sxtw() {
        let sxtw1 = MachInst::new(AArch64Opcode::Sxtw, vec![vreg(1), vreg(0)]);
        let sxtw2 = MachInst::new(AArch64Opcode::Sxtw, vec![vreg(2), vreg(1)]);
        let ret = MachInst::new(AArch64Opcode::Ret, vec![]);
        let mut func = make_func_with_insts(vec![sxtw1, sxtw2, ret]);

        let mut peep = Peephole;
        assert!(peep.run(&mut func));

        let inst = func.inst(InstId(1));
        assert_eq!(inst.opcode, AArch64Opcode::Sxtw);
        assert_eq!(inst.operands[0], vreg(2));
        assert_eq!(inst.operands[1], vreg(0));
    }

    #[test]
    fn test_sxtb_after_add_no_change() {
        // v1 = add v0, #5
        // v2 = sxtb v1 → no simplification (v1 not from sxtb)
        let add = MachInst::new(AArch64Opcode::AddRI, vec![vreg(1), vreg(0), imm(5)]);
        let sxtb = MachInst::new(AArch64Opcode::Sxtb, vec![vreg(2), vreg(1)]);
        let ret = MachInst::new(AArch64Opcode::Ret, vec![]);
        let mut func = make_func_with_insts(vec![add, sxtb, ret]);

        let mut peep = Peephole;
        assert!(!peep.run(&mut func));
    }

    #[test]
    fn test_double_sxtb_preserves_proof() {
        let sxtb1 = MachInst::new(AArch64Opcode::Sxtb, vec![vreg(1), vreg(0)]);
        let sxtb2 = MachInst::new(AArch64Opcode::Sxtb, vec![vreg(2), vreg(1)])
            .with_proof(ProofAnnotation::InBounds);
        let ret = MachInst::new(AArch64Opcode::Ret, vec![]);
        let mut func = make_func_with_insts(vec![sxtb1, sxtb2, ret]);

        let mut peep = Peephole;
        assert!(peep.run(&mut func));

        let inst = func.inst(InstId(1));
        assert_eq!(inst.opcode, AArch64Opcode::Sxtb);
        assert_eq!(inst.proof, Some(ProofAnnotation::InBounds));
    }

    // ================================================================
    // Pattern 29: LSL(LSR(x, n), n) → AND x, mask
    // ================================================================

    #[test]
    fn test_lsl_lsr_to_and() {
        // v1 = lsr v0, #4
        // v2 = lsl v1, #4 → v2 = and v0, #0xFFFFFFFFFFFFFFF0
        let lsr = MachInst::new(AArch64Opcode::LsrRI, vec![vreg(1), vreg(0), imm(4)]);
        let lsl = MachInst::new(AArch64Opcode::LslRI, vec![vreg(2), vreg(1), imm(4)]);
        let ret = MachInst::new(AArch64Opcode::Ret, vec![]);
        let mut func = make_func_with_insts(vec![lsr, lsl, ret]);

        let mut peep = Peephole;
        assert!(peep.run(&mut func));

        let inst = func.inst(InstId(1));
        assert_eq!(inst.opcode, AArch64Opcode::AndRI);
        assert_eq!(inst.operands[0], vreg(2));
        assert_eq!(inst.operands[1], vreg(0));
        // mask = !((1 << 4) - 1) = !0xF = 0xFFFFFFFFFFFFFFF0
        let expected_mask = !((1u64 << 4) - 1) as i64;
        assert_eq!(inst.operands[2], imm(expected_mask));
    }

    #[test]
    fn test_lsl_lsr_different_shifts_no_change() {
        // v1 = lsr v0, #3
        // v2 = lsl v1, #5 → no change (different shift amounts)
        let lsr = MachInst::new(AArch64Opcode::LsrRI, vec![vreg(1), vreg(0), imm(3)]);
        let lsl = MachInst::new(AArch64Opcode::LslRI, vec![vreg(2), vreg(1), imm(5)]);
        let ret = MachInst::new(AArch64Opcode::Ret, vec![]);
        let mut func = make_func_with_insts(vec![lsr, lsl, ret]);

        let mut peep = Peephole;
        assert!(!peep.run(&mut func));
    }

    #[test]
    fn test_lsl_lsr_preserves_proof() {
        let lsr = MachInst::new(AArch64Opcode::LsrRI, vec![vreg(1), vreg(0), imm(8)]);
        let lsl = MachInst::new(AArch64Opcode::LslRI, vec![vreg(2), vreg(1), imm(8)])
            .with_proof(ProofAnnotation::NoOverflow);
        let ret = MachInst::new(AArch64Opcode::Ret, vec![]);
        let mut func = make_func_with_insts(vec![lsr, lsl, ret]);

        let mut peep = Peephole;
        assert!(peep.run(&mut func));

        let inst = func.inst(InstId(1));
        assert_eq!(inst.opcode, AArch64Opcode::AndRI);
        assert_eq!(inst.proof, Some(ProofAnnotation::NoOverflow));
    }

    // Pattern 30 (neg-of-sub swap) migrated to the declarative rewrite
    // framework in Wave 4 (#393). Behavioural coverage lives in
    // `rewrite::patterns::tests::neg_sub_swaps_operands` and
    // `neg_sub_does_not_fire_over_non_sub_definer`.

    // ================================================================
    // Pattern 31: MUL x, #-1 → NEG x
    // Migrated to rewrite/patterns.rs (rule_mul_by_neg_one_rhs / rule_mul_by_neg_one_lhs).
    // Tests moved to rewrite::patterns::tests.
    // ================================================================

    // ================================================================
    // Pattern 32: LSR(LSL(x, n), n) → AND x, mask
    // ================================================================

    #[test]
    fn test_lsr_lsl_to_and() {
        // v1 = lsl v0, #8
        // v2 = lsr v1, #8 → v2 = and v0, #(u64::MAX >> 8)
        let lsl = MachInst::new(AArch64Opcode::LslRI, vec![vreg(1), vreg(0), imm(8)]);
        let lsr = MachInst::new(AArch64Opcode::LsrRI, vec![vreg(2), vreg(1), imm(8)]);
        let ret = MachInst::new(AArch64Opcode::Ret, vec![]);
        let mut func = make_func_with_insts(vec![lsl, lsr, ret]);

        let mut peep = Peephole;
        assert!(peep.run(&mut func));

        let inst = func.inst(InstId(1));
        assert_eq!(inst.opcode, AArch64Opcode::AndRI);
        assert_eq!(inst.operands[0], vreg(2));
        assert_eq!(inst.operands[1], vreg(0));
        let expected_mask = (u64::MAX >> 8) as i64;
        assert_eq!(inst.operands[2], imm(expected_mask));
    }

    #[test]
    fn test_lsr_lsl_different_shifts_no_change() {
        let lsl = MachInst::new(AArch64Opcode::LslRI, vec![vreg(1), vreg(0), imm(4)]);
        let lsr = MachInst::new(AArch64Opcode::LsrRI, vec![vreg(2), vreg(1), imm(6)]);
        let ret = MachInst::new(AArch64Opcode::Ret, vec![]);
        let mut func = make_func_with_insts(vec![lsl, lsr, ret]);

        let mut peep = Peephole;
        assert!(!peep.run(&mut func));
    }

    #[test]
    fn test_lsr_lsl_preserves_proof() {
        let lsl = MachInst::new(AArch64Opcode::LslRI, vec![vreg(1), vreg(0), imm(16)]);
        let lsr = MachInst::new(AArch64Opcode::LsrRI, vec![vreg(2), vreg(1), imm(16)])
            .with_proof(ProofAnnotation::InBounds);
        let ret = MachInst::new(AArch64Opcode::Ret, vec![]);
        let mut func = make_func_with_insts(vec![lsl, lsr, ret]);

        let mut peep = Peephole;
        assert!(peep.run(&mut func));

        let inst = func.inst(InstId(1));
        assert_eq!(inst.opcode, AArch64Opcode::AndRI);
        assert_eq!(inst.proof, Some(ProofAnnotation::InBounds));
    }

    // ================================================================
    // Pattern 33: SDIV x, #1 → MOV x
    // Migrated to rewrite/patterns.rs (rule_sdiv_by_one).
    // Tests moved to rewrite::patterns::tests.
    // ================================================================

    // ================================================================
    // Pattern 34: MUL x, #0 → MOV x, #0
    // Migrated to rewrite/patterns.rs (rule_mul_by_zero_rhs / rule_mul_by_zero_lhs).
    // Tests moved to rewrite::patterns::tests.
    // ================================================================

    // ================================================================
    // Patterns 35 (MADD #1), 36 (MADD #0), 37 (MSUB #1), 38 (MSUB #0)
    // Migrated to rewrite/patterns.rs (rule_madd_by_zero_at_{rn,rm},
    // rule_madd_by_one_at_{rn,rm}, rule_msub_by_zero_at_{rn,rm},
    // rule_msub_by_one_at_{rn,rm}).
    // Tests moved to rewrite::patterns::tests — see
    // `madd_zero_at_r{m,n}_*`, `madd_one_at_r{m,n}_*`, and the
    // analogous `msub_*` tests, plus `full_ruleset_wave6_madd_msub_mixed`.
    // ================================================================

    // Patterns 39 (sxth-of-sxtb) and 40 (sxtw-of-sxtb/sxth) migrated to
    // the declarative rewrite framework in Wave 4 (#393). Behavioural
    // coverage lives in `rewrite::patterns::tests::sxth_of_sxtb_*`,
    // `sxtw_of_sxtb_*`, `sxtw_of_sxth_*`.

    #[test]
    fn test_sxth_after_add_no_narrow() {
        // v1 = add v0, #5
        // v2 = sxth v1 → no simplification (v1 not from narrower ext)
        let add = MachInst::new(AArch64Opcode::AddRI, vec![vreg(1), vreg(0), imm(5)]);
        let sxth = MachInst::new(AArch64Opcode::Sxth, vec![vreg(2), vreg(1)]);
        let ret = MachInst::new(AArch64Opcode::Ret, vec![]);
        let mut func = make_func_with_insts(vec![add, sxth, ret]);

        let mut peep = Peephole;
        assert!(!peep.run(&mut func));
    }

    // ===================================================================
    // Tests for patterns 41-52 (Wave 48)
    // ===================================================================

    // Patterns 41 (add-sub cancel) and 42 (sub-add cancel)
    // migrated to the declarative rewrite framework in Wave 5 (#393).
    // Behavioural coverage lives in rewrite::patterns::tests.

    #[test]
    fn test_csel_same_arms_pattern43() {
        // csel v2, v1, v1, EQ → mov v2, v1
        let csel = MachInst::new(
            AArch64Opcode::Csel,
            vec![vreg(2), vreg(1), vreg(1), imm(0)],
        );
        let ret = MachInst::new(AArch64Opcode::Ret, vec![]);
        let mut func = make_func_with_insts(vec![csel, ret]);

        let mut peep = Peephole;
        assert!(peep.run(&mut func));

        let inst = func.inst(InstId(0));
        assert_eq!(inst.opcode, AArch64Opcode::MovR);
        assert_eq!(inst.operands[1], vreg(1));
    }

    #[test]
    fn test_csel_different_arms_no_change() {
        // csel v2, v0, v1, EQ → no change (different arms)
        let csel = MachInst::new(
            AArch64Opcode::Csel,
            vec![vreg(2), vreg(0), vreg(1), imm(0)],
        );
        let ret = MachInst::new(AArch64Opcode::Ret, vec![]);
        let mut func = make_func_with_insts(vec![csel, ret]);

        let mut peep = Peephole;
        assert!(!peep.run(&mut func));
    }

    // Patterns 44 (udiv x, 1 → mov x) and 45 (sdiv x, -1 → neg x)
    // migrated to the declarative rewrite framework in Wave 5 (#393).
    // Behavioural coverage lives in rewrite::patterns::tests.

    #[test]
    fn test_double_uxtb_pattern46() {
        // v1 = uxtb v0
        // v2 = uxtb v1 → v2 = uxtb v0
        let ext1 = MachInst::new(AArch64Opcode::Uxtb, vec![vreg(1), vreg(0)]);
        let ext2 = MachInst::new(AArch64Opcode::Uxtb, vec![vreg(2), vreg(1)]);
        let ret = MachInst::new(AArch64Opcode::Ret, vec![]);
        let mut func = make_func_with_insts(vec![ext1, ext2, ret]);

        let mut peep = Peephole;
        assert!(peep.run(&mut func));

        let inst = func.inst(InstId(1));
        assert_eq!(inst.opcode, AArch64Opcode::Uxtb);
        assert_eq!(inst.operands[1], vreg(0));
    }

    #[test]
    fn test_double_uxth_pattern47() {
        // v1 = uxth v0
        // v2 = uxth v1 → v2 = uxth v0
        let ext1 = MachInst::new(AArch64Opcode::Uxth, vec![vreg(1), vreg(0)]);
        let ext2 = MachInst::new(AArch64Opcode::Uxth, vec![vreg(2), vreg(1)]);
        let ret = MachInst::new(AArch64Opcode::Ret, vec![]);
        let mut func = make_func_with_insts(vec![ext1, ext2, ret]);

        let mut peep = Peephole;
        assert!(peep.run(&mut func));

        let inst = func.inst(InstId(1));
        assert_eq!(inst.opcode, AArch64Opcode::Uxth);
        assert_eq!(inst.operands[1], vreg(0));
    }

    // Patterns 48 (uxth-of-uxtb), 49 (uxtw-of-uxtb), 50 (uxtw-of-uxth)
    // migrated to the declarative rewrite framework in Wave 4 (#393).
    // Behavioural coverage lives in
    // `rewrite::patterns::tests::uxth_of_uxtb_*`,
    // `uxtw_of_uxtb_*`, `uxtw_of_uxth_*`.

    // Pattern 51 (add-sub y-x cancel) and pattern 52 (eor-eor self-cancel,
    // both operand orders) migrated to the declarative rewrite framework
    // in Wave 5 (#393). Behavioural coverage lives in rewrite::patterns::tests.

    // ------------------------------------------------------------------
    // source_loc preservation across peephole rewrites (issue #376).
    //
    // Peephole replaces `add x, y, #0` with `mov x, y` by constructing a
    // fresh MachInst. Without explicit propagation, the new MachInst has
    // source_loc: None, which breaks DWARF __debug_line emission and
    // causes lldb to show incorrect source lines at -O3.
    // ------------------------------------------------------------------

    #[test]
    fn test_source_loc_preserved_across_peephole_replace() {
        use llvm2_ir::SourceLoc;
        let loc = SourceLoc { file: 1, line: 42, col: 5 };
        // add v0, v1, #0 (with source_loc) → mov v0, v1 (must keep source_loc)
        let mut add = MachInst::new(AArch64Opcode::AddRI, vec![vreg(0), vreg(1), imm(0)]);
        add.source_loc = Some(loc);
        let ret = MachInst::new(AArch64Opcode::Ret, vec![]);
        let mut func = make_func_with_insts(vec![add, ret]);

        let mut peep = Peephole;
        assert!(peep.run(&mut func));

        let inst = func.inst(InstId(0));
        assert_eq!(inst.opcode, AArch64Opcode::MovR);
        assert_eq!(
            inst.source_loc,
            Some(loc),
            "peephole must preserve source_loc when replacing MachInst (issue #376)"
        );
    }

    #[test]
    fn test_source_loc_preserved_across_shift_zero_to_mov() {
        use llvm2_ir::SourceLoc;
        let loc = SourceLoc { file: 2, line: 100, col: 0 };
        // lsl v0, v1, #0 (with source_loc) → mov v0, v1
        let mut lsl = MachInst::new(AArch64Opcode::LslRI, vec![vreg(0), vreg(1), imm(0)]);
        lsl.source_loc = Some(loc);
        let ret = MachInst::new(AArch64Opcode::Ret, vec![]);
        let mut func = make_func_with_insts(vec![lsl, ret]);

        let mut peep = Peephole;
        assert!(peep.run(&mut func));

        let inst = func.inst(InstId(0));
        assert_eq!(inst.opcode, AArch64Opcode::MovR);
        assert_eq!(
            inst.source_loc,
            Some(loc),
            "peephole must preserve source_loc for shift-by-zero rewrite (issue #376)"
        );
    }
}
