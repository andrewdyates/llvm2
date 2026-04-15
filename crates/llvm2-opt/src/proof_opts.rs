// llvm2-opt - Proof-consuming optimizations
//
// Author: Andrew Yates <ayates@dropbox.com>
// Copyright 2026 Dropbox, Inc. | License: Apache-2.0

//! Proof-consuming optimization pass for LLVM2.
//!
//! This is the unique value proposition of LLVM2: tMIR carries formally
//! verified proof annotations that enable optimizations no other compiler
//! can safely perform. When tMIR proves a property about a value, we can
//! eliminate the runtime check that protects against violation of that
//! property.
//!
//! # Proof Types and Their Optimizations
//!
//! | Proof Annotation | Codegen Pattern Eliminated |
//! |------------------|---------------------------|
//! | `NoOverflow`     | `adds/subs + b.vs trap` → plain `add/sub` |
//! | `InBounds`       | `cmp idx, len + b.hs panic` → remove guard |
//! | `NotNull`        | `cbz ptr, panic_label` → remove guard |
//! | `ValidBorrow`    | Refines memory aliasing (enables reordering) |
//! | `PositiveRefCount` | `retain + release` → remove pair |
//! | `NonZeroDivisor` | `cbz divisor, trap + udiv/sdiv` → plain `udiv/sdiv` |
//! | `ValidShift`     | `cmp amt, #64 + b.ge trap + lsl/lsr/asr` → plain shift |
//!
//! # Safety
//!
//! These optimizations are sound ONLY because tMIR has formally proven the
//! preconditions using z4. The proof annotations represent verified facts,
//! not hints or heuristics. Removing a bounds check without a proof would
//! be a miscompilation; with the proof, it is a verified optimization.
//!
//! # Algorithm
//!
//! Single forward pass over each block's instruction list:
//! 1. For each instruction, check if it has a `ProofAnnotation`.
//! 2. If so, apply the corresponding pattern transformation.
//! 3. Track statistics for diagnostics.
//!
//! The pass also scans for multi-instruction patterns (e.g., ADDS followed
//! by TrapOverflow) where the proof annotation is on the first instruction
//! of the sequence.
//!
//! Reference: designs/2026-04-12-aarch64-backend.md, "Proof-Enabled Optimizations"

use std::collections::HashSet;

use llvm2_ir::{
    AArch64Opcode, InstFlags, InstId, MachFunction, MachOperand, ProofAnnotation,
};

use crate::pass_manager::MachinePass;

/// Statistics collected during proof-consuming optimization.
#[derive(Debug, Clone, Default)]
pub struct ProofOptStats {
    /// Number of overflow checks eliminated (NoOverflow).
    pub overflow_checks_eliminated: u32,
    /// Number of bounds checks eliminated (InBounds).
    pub bounds_checks_eliminated: u32,
    /// Number of null checks eliminated (NotNull).
    pub null_checks_eliminated: u32,
    /// Number of retain/release pairs eliminated (PositiveRefCount).
    pub refcount_pairs_eliminated: u32,
    /// Number of memory reordering opportunities enabled (ValidBorrow).
    pub alias_refinements: u32,
    /// Number of division-by-zero checks eliminated (NonZeroDivisor).
    pub divzero_checks_eliminated: u32,
    /// Number of shift-range checks eliminated (ValidShift).
    pub shift_checks_eliminated: u32,
    /// Number of loads promoted to pure for aggressive CSE (Pure).
    pub pure_cse_enabled: u32,
}

/// Proof-consuming optimization pass.
///
/// Consumes tMIR proof annotations to eliminate runtime safety checks
/// that have been formally verified as unnecessary.
pub struct ProofOptimization {
    stats: ProofOptStats,
}

impl ProofOptimization {
    /// Create a new proof optimization pass.
    pub fn new() -> Self {
        Self {
            stats: ProofOptStats::default(),
        }
    }

    /// Returns optimization statistics from the last run.
    pub fn stats(&self) -> &ProofOptStats {
        &self.stats
    }
}

impl Default for ProofOptimization {
    fn default() -> Self {
        Self::new()
    }
}

impl MachinePass for ProofOptimization {
    fn name(&self) -> &str {
        "proof-opts"
    }

    fn run(&mut self, func: &mut MachFunction) -> bool {
        self.stats = ProofOptStats::default();
        let mut changed = false;

        // Collect instructions to delete across all blocks.
        let mut to_delete: HashSet<InstId> = HashSet::new();

        for block_id in func.block_order.clone() {
            let block_insts: Vec<InstId> = func.block(block_id).insts.clone();

            for (pos, &inst_id) in block_insts.iter().enumerate() {
                let proof = func.inst(inst_id).proof;

                match proof {
                    Some(ProofAnnotation::NoOverflow) => {
                        if self.apply_no_overflow(func, &block_insts, pos, &mut to_delete)
                        {
                            changed = true;
                        }
                    }
                    Some(ProofAnnotation::InBounds) => {
                        if self.apply_in_bounds(func, &block_insts, pos, &mut to_delete) {
                            changed = true;
                        }
                    }
                    Some(ProofAnnotation::NotNull) => {
                        if self.apply_not_null(func, &block_insts, pos, &mut to_delete) {
                            changed = true;
                        }
                    }
                    Some(ProofAnnotation::ValidBorrow) => {
                        if self.apply_valid_borrow(func, inst_id) {
                            changed = true;
                        }
                    }
                    Some(ProofAnnotation::PositiveRefCount) => {
                        if self.apply_positive_refcount(
                            func,
                            &block_insts,
                            pos,
                            &mut to_delete,
                        ) {
                            changed = true;
                        }
                    }
                    Some(ProofAnnotation::NonZeroDivisor) => {
                        if self.apply_non_zero_divisor(
                            func,
                            &block_insts,
                            pos,
                            &mut to_delete,
                        ) {
                            changed = true;
                        }
                    }
                    Some(ProofAnnotation::ValidShift) => {
                        if self.apply_valid_shift(
                            func,
                            &block_insts,
                            pos,
                            &mut to_delete,
                        ) {
                            changed = true;
                        }
                    }
                    Some(ProofAnnotation::Pure) => {
                        if self.apply_pure(func, inst_id) {
                            changed = true;
                        }
                    }
                    // Algebraic property proofs: preserved as metadata for
                    // downstream passes (vectorizer, parallel reduction).
                    // The proof_opts pass does not transform these directly
                    // but they are consumed by vectorize.rs and scheduler.rs.
                    Some(ProofAnnotation::Associative)
                    | Some(ProofAnnotation::Commutative)
                    | Some(ProofAnnotation::Idempotent) => {}
                    None => {}
                }
            }
        }

        // Remove deleted instructions from blocks.
        if !to_delete.is_empty() {
            for block_id in func.block_order.clone() {
                let block = func.block_mut(block_id);
                block.insts.retain(|id| !to_delete.contains(id));
            }
        }

        changed
    }
}

impl ProofOptimization {
    /// NoOverflow optimization: when tMIR proves no overflow, convert
    /// checked arithmetic to unchecked.
    ///
    /// Pattern: `ADDS/SUBS dst, a, b` [NoOverflow] + `TrapOverflow cc, panic_block`
    /// Result:  `ADD/SUB dst, a, b` (trap instruction removed)
    ///
    /// The ADDS instruction sets condition flags; the subsequent TrapOverflow
    /// branches to a panic block if overflow occurred. With the NoOverflow
    /// proof, we know overflow cannot happen, so we:
    /// 1. Replace ADDS/SUBS with plain ADD/SUB (no flag setting)
    /// 2. Remove the TrapOverflow instruction
    fn apply_no_overflow(
        &mut self,
        func: &mut MachFunction,
        block_insts: &[InstId],
        pos: usize,
        to_delete: &mut HashSet<InstId>,
    ) -> bool {
        let inst_id = block_insts[pos];
        let opcode = func.inst(inst_id).opcode;

        // Map checked opcodes to unchecked equivalents.
        let unchecked = match opcode {
            AArch64Opcode::AddsRR => AArch64Opcode::AddRR,
            AArch64Opcode::AddsRI => AArch64Opcode::AddRI,
            AArch64Opcode::SubsRR => AArch64Opcode::SubRR,
            AArch64Opcode::SubsRI => AArch64Opcode::SubRI,
            _ => return false,
        };

        // Replace the checked instruction with unchecked.
        let inst = func.inst_mut(inst_id);
        inst.opcode = unchecked;
        inst.flags = unchecked.default_flags();
        inst.proof = None;

        // Look for the trailing TrapOverflow and remove it.
        if pos + 1 < block_insts.len() {
            let next_id = block_insts[pos + 1];
            if func.inst(next_id).opcode == AArch64Opcode::TrapOverflow {
                to_delete.insert(next_id);
            }
        }

        self.stats.overflow_checks_eliminated += 1;
        true
    }

    /// InBounds optimization: when tMIR proves array access is in-bounds,
    /// eliminate the bounds check guard.
    ///
    /// Pattern: `CMP idx, len` [InBounds] + `TrapBoundsCheck panic_block`
    /// Result:  both instructions removed
    ///
    /// The CMP sets flags comparing index against array length, and
    /// TrapBoundsCheck branches to a panic block if idx >= len (unsigned).
    /// With the InBounds proof, we know idx < len always holds.
    fn apply_in_bounds(
        &mut self,
        func: &mut MachFunction,
        block_insts: &[InstId],
        pos: usize,
        to_delete: &mut HashSet<InstId>,
    ) -> bool {
        let inst_id = block_insts[pos];
        let opcode = func.inst(inst_id).opcode;

        // The InBounds proof is attached to the CMP instruction.
        if !matches!(opcode, AArch64Opcode::CmpRR | AArch64Opcode::CmpRI) {
            return false;
        }

        // Look for the trailing TrapBoundsCheck.
        if pos + 1 < block_insts.len() {
            let next_id = block_insts[pos + 1];
            if func.inst(next_id).opcode == AArch64Opcode::TrapBoundsCheck {
                // Remove both the CMP and the TrapBoundsCheck.
                to_delete.insert(inst_id);
                to_delete.insert(next_id);
                self.stats.bounds_checks_eliminated += 1;
                return true;
            }
        }

        false
    }

    /// NotNull optimization: when tMIR proves a pointer is not null,
    /// eliminate the null check guard.
    ///
    /// Pattern: `CBZ ptr, panic_label` [NotNull]
    /// Result:  instruction removed
    ///
    /// Or: `CBNZ ptr, continue_label` [NotNull] (inverted sense)
    /// Result:  replaced with unconditional branch to continue_label
    ///
    /// Also handles the pseudo-instruction pattern:
    /// Pattern: `TrapNull panic_block` [NotNull]
    /// Result:  instruction removed
    fn apply_not_null(
        &mut self,
        func: &mut MachFunction,
        block_insts: &[InstId],
        pos: usize,
        to_delete: &mut HashSet<InstId>,
    ) -> bool {
        let inst_id = block_insts[pos];
        let opcode = func.inst(inst_id).opcode;

        match opcode {
            AArch64Opcode::Cbz => {
                // CBZ branches to panic if ptr == null.
                // With NotNull proof, ptr is never null, so the branch
                // never fires. Remove it entirely.
                to_delete.insert(inst_id);
                self.stats.null_checks_eliminated += 1;
                true
            }
            AArch64Opcode::Cbnz => {
                // CBNZ branches to continue if ptr != null.
                // With NotNull proof, this always branches. Replace with
                // unconditional B to the target.
                let target = func.inst(inst_id).operands.last().cloned();
                if let Some(MachOperand::Block(target_block)) = target {
                    let inst = func.inst_mut(inst_id);
                    inst.opcode = AArch64Opcode::B;
                    inst.operands = vec![MachOperand::Block(target_block)];
                    inst.flags = AArch64Opcode::B.default_flags();
                    inst.proof = None;
                    self.stats.null_checks_eliminated += 1;
                    true
                } else {
                    false
                }
            }
            AArch64Opcode::TrapNull => {
                // Pseudo-instruction: trap if null. Remove with proof.
                to_delete.insert(inst_id);
                self.stats.null_checks_eliminated += 1;
                true
            }
            // Also handle the 2-instruction pattern: CMP + BCond where the
            // CMP compares against zero for null check.
            AArch64Opcode::CmpRI => {
                // Check if comparing against zero (null check pattern).
                let is_null_check = func
                    .inst(inst_id)
                    .operands
                    .last()
                    .map(|op| matches!(op, MachOperand::Imm(0)))
                    .unwrap_or(false);

                if is_null_check {
                    // Look for trailing BCond or TrapNull.
                    if pos + 1 < block_insts.len() {
                        let next_id = block_insts[pos + 1];
                        let next_opcode = func.inst(next_id).opcode;
                        if matches!(
                            next_opcode,
                            AArch64Opcode::BCond | AArch64Opcode::TrapNull
                        ) {
                            to_delete.insert(inst_id);
                            to_delete.insert(next_id);
                            self.stats.null_checks_eliminated += 1;
                            return true;
                        }
                    }
                }
                false
            }
            _ => false,
        }
    }

    /// ValidBorrow optimization: when tMIR proves a borrow is valid,
    /// refine the memory aliasing model to allow reordering.
    ///
    /// For loads/stores with ValidBorrow proof, we remove the memory
    /// side-effect flags that would normally prevent reordering. This
    /// allows CSE and LICM to treat these memory operations more
    /// aggressively.
    ///
    /// Specifically: a load with ValidBorrow can be treated as non-aliasing
    /// with other ValidBorrow stores, enabling the load to be hoisted or
    /// CSE'd even past intervening stores.
    fn apply_valid_borrow(
        &mut self,
        func: &mut MachFunction,
        inst_id: InstId,
    ) -> bool {
        let inst = func.inst(inst_id);

        // ValidBorrow applies to loads and stores.
        if !inst.reads_memory() && !inst.writes_memory() {
            return false;
        }

        // Mark the instruction as proof-reorderable. This flag tells
        // subsequent passes (CSE, LICM) that this memory operation can
        // be safely reordered past other memory operations because the
        // borrow validity has been formally proven.
        let inst = func.inst_mut(inst_id);
        inst.flags.insert(InstFlags::PROOF_REORDERABLE);

        self.stats.alias_refinements += 1;
        true
    }

    /// PositiveRefCount optimization: when tMIR proves the reference count
    /// is positive, eliminate redundant retain/release pairs.
    ///
    /// Pattern: `Retain ptr` [PositiveRefCount] ... `Release ptr`
    /// Result:  both instructions removed
    ///
    /// When we encounter a Retain with PositiveRefCount proof, we scan
    /// forward for a matching Release on the same pointer. If found with
    /// no intervening calls or other retains/releases on the same pointer,
    /// both are eliminated.
    fn apply_positive_refcount(
        &mut self,
        func: &mut MachFunction,
        block_insts: &[InstId],
        pos: usize,
        to_delete: &mut HashSet<InstId>,
    ) -> bool {
        let inst_id = block_insts[pos];
        if func.inst(inst_id).opcode != AArch64Opcode::Retain {
            return false;
        }

        // Get the pointer operand of the retain.
        let retain_ptr = func.inst(inst_id).operands.first().cloned();
        let retain_ptr = match retain_ptr {
            Some(op) => op,
            None => return false,
        };

        // Scan forward for a matching Release on the same pointer.
        for &later_id in &block_insts[pos + 1..] {
            let later = func.inst(later_id);

            // If we hit a call, stop — it might observe the refcount.
            if later.is_call() {
                break;
            }

            // If we hit another Retain or Release on the same pointer
            // (not the match we're looking for), stop to avoid complexity.
            if later.opcode == AArch64Opcode::Retain
                && later.operands.first() == Some(&retain_ptr) {
                    break;
                }

            if later.opcode == AArch64Opcode::Release
                && later.operands.first() == Some(&retain_ptr) {
                    // Found matching release. Remove both.
                    to_delete.insert(inst_id);
                    to_delete.insert(later_id);
                    self.stats.refcount_pairs_eliminated += 1;
                    return true;
                }
        }

        false
    }

    /// NonZeroDivisor optimization: when tMIR proves the divisor is non-zero,
    /// eliminate the division-by-zero guard.
    ///
    /// Pattern 1: `CBZ divisor, trap_block` [NonZeroDivisor]
    /// Result:    instruction removed
    ///
    /// Pattern 2: `TrapDivZero trap_block` [NonZeroDivisor]
    /// Result:    instruction removed
    ///
    /// Pattern 3: `CMP divisor, #0` [NonZeroDivisor] + `BCond EQ, trap_block`/`TrapDivZero`
    /// Result:    both instructions removed
    ///
    /// On AArch64, division by zero yields zero (not a fault), but the
    /// tMIR runtime model may still insert guards when the source language
    /// semantics require a trap. With the NonZeroDivisor proof, the guard
    /// is provably dead code.
    fn apply_non_zero_divisor(
        &mut self,
        func: &mut MachFunction,
        block_insts: &[InstId],
        pos: usize,
        to_delete: &mut HashSet<InstId>,
    ) -> bool {
        let inst_id = block_insts[pos];
        let opcode = func.inst(inst_id).opcode;

        match opcode {
            AArch64Opcode::Cbz => {
                // CBZ branches to trap if divisor == 0.
                // With NonZeroDivisor proof, divisor is never zero. Remove.
                to_delete.insert(inst_id);
                self.stats.divzero_checks_eliminated += 1;
                true
            }
            AArch64Opcode::TrapDivZero => {
                // Pseudo-instruction: trap if divisor is zero. Remove with proof.
                to_delete.insert(inst_id);
                self.stats.divzero_checks_eliminated += 1;
                true
            }
            AArch64Opcode::CmpRI => {
                // Check if comparing against zero (div-zero check pattern).
                let is_zero_check = func
                    .inst(inst_id)
                    .operands
                    .last()
                    .map(|op| matches!(op, MachOperand::Imm(0)))
                    .unwrap_or(false);

                if is_zero_check {
                    // Look for trailing BCond or TrapDivZero.
                    if pos + 1 < block_insts.len() {
                        let next_id = block_insts[pos + 1];
                        let next_opcode = func.inst(next_id).opcode;
                        if matches!(
                            next_opcode,
                            AArch64Opcode::BCond | AArch64Opcode::TrapDivZero
                        ) {
                            to_delete.insert(inst_id);
                            to_delete.insert(next_id);
                            self.stats.divzero_checks_eliminated += 1;
                            return true;
                        }
                    }
                }
                false
            }
            _ => false,
        }
    }

    /// ValidShift optimization: when tMIR proves the shift amount is in
    /// [0, bitwidth), eliminate the shift-amount range check.
    ///
    /// Pattern 1: `CMP shift_amt, #64` [ValidShift] + `TrapShiftRange trap_block`
    /// Result:    both instructions removed
    ///
    /// Pattern 2: `CMP shift_amt, #32/#64` [ValidShift] + `BCond GE, trap_block`
    /// Result:    both instructions removed
    ///
    /// Pattern 3: `TrapShiftRange trap_block` [ValidShift]
    /// Result:    instruction removed
    ///
    /// On AArch64, shift amounts outside [0, bitwidth) produce
    /// implementation-defined results. The tMIR runtime model inserts
    /// range checks when required by source-language semantics. With
    /// the ValidShift proof, the check is provably dead.
    fn apply_valid_shift(
        &mut self,
        func: &mut MachFunction,
        block_insts: &[InstId],
        pos: usize,
        to_delete: &mut HashSet<InstId>,
    ) -> bool {
        let inst_id = block_insts[pos];
        let opcode = func.inst(inst_id).opcode;

        match opcode {
            AArch64Opcode::CmpRI => {
                // Check if comparing against a bitwidth (32 or 64).
                let is_shift_range_check = func
                    .inst(inst_id)
                    .operands
                    .last()
                    .map(|op| matches!(op, MachOperand::Imm(32) | MachOperand::Imm(64)))
                    .unwrap_or(false);

                if is_shift_range_check {
                    // Look for trailing BCond or TrapShiftRange.
                    if pos + 1 < block_insts.len() {
                        let next_id = block_insts[pos + 1];
                        let next_opcode = func.inst(next_id).opcode;
                        if matches!(
                            next_opcode,
                            AArch64Opcode::BCond | AArch64Opcode::TrapShiftRange
                        ) {
                            to_delete.insert(inst_id);
                            to_delete.insert(next_id);
                            self.stats.shift_checks_eliminated += 1;
                            return true;
                        }
                    }
                }
                false
            }
            AArch64Opcode::TrapShiftRange => {
                // Pseudo-instruction: trap if shift amount out of range. Remove.
                to_delete.insert(inst_id);
                self.stats.shift_checks_eliminated += 1;
                true
            }
            _ => false,
        }
    }

    /// Pure optimization: when tMIR proves an operation is pure (no observable
    /// side effects, deterministic), promote it for aggressive CSE.
    ///
    /// A load with a Pure proof annotation means the loaded memory location
    /// is immutable (e.g., a read from a frozen/constant data structure).
    /// This enables:
    ///
    /// 1. **Aggressive CSE**: Two loads from the same address can be CSE'd
    ///    even with intervening stores (the Pure proof guarantees the loaded
    ///    value does not change).
    /// 2. **LICM**: Pure loads can be hoisted out of loops.
    ///
    /// Implementation: remove the READS_MEMORY/WRITES_MEMORY and
    /// HAS_SIDE_EFFECTS flags, making the instruction appear pure to
    /// downstream CSE/LICM passes. The proof annotation is consumed.
    fn apply_pure(
        &mut self,
        func: &mut MachFunction,
        inst_id: InstId,
    ) -> bool {
        let inst = func.inst(inst_id);

        // Pure proof is meaningful for loads (promotes them to CSE-able).
        // For already-pure instructions, no change needed.
        if !inst.reads_memory() && !inst.writes_memory() && !inst.has_side_effects() {
            return false;
        }

        // Promote the instruction: remove memory flags so CSE/LICM treat
        // it as pure. Keep the instruction opcode and operands unchanged.
        let inst = func.inst_mut(inst_id);
        inst.flags.remove(InstFlags::READS_MEMORY);
        inst.flags.remove(InstFlags::WRITES_MEMORY);
        inst.flags.remove(InstFlags::HAS_SIDE_EFFECTS);
        inst.proof = None; // Proof consumed.

        self.stats.pure_cse_enabled += 1;
        true
    }
}

// ---------------------------------------------------------------------------
// Public API: named functions for each proof-consuming optimization
// ---------------------------------------------------------------------------

/// Eliminate overflow checks when NoOverflow proof is present.
///
/// Converts checked arithmetic (ADDS/SUBS) to unchecked (ADD/SUB) and
/// removes trailing TrapOverflow instructions.
///
/// Returns the number of overflow checks eliminated.
pub fn eliminate_overflow_checks(func: &mut MachFunction) -> u32 {
    let mut pass = ProofOptimization::new();
    pass.run(func);
    pass.stats().overflow_checks_eliminated
}

/// Eliminate bounds checks when InBounds proof is present.
///
/// Removes CMP+TrapBoundsCheck patterns when the index is proven in-bounds.
///
/// Returns the number of bounds checks eliminated.
pub fn eliminate_bounds_checks(func: &mut MachFunction) -> u32 {
    let mut pass = ProofOptimization::new();
    pass.run(func);
    pass.stats().bounds_checks_eliminated
}

/// Eliminate null checks when NotNull proof is present.
///
/// Removes CBZ/TrapNull patterns and simplifies CBNZ to unconditional branches
/// when the pointer is proven non-null.
///
/// Returns the number of null checks eliminated.
pub fn eliminate_null_checks(func: &mut MachFunction) -> u32 {
    let mut pass = ProofOptimization::new();
    pass.run(func);
    pass.stats().null_checks_eliminated
}

/// Enable load/store reordering when ValidBorrow proof is present.
///
/// Marks proven-valid memory operations with `PROOF_REORDERABLE` flag,
/// allowing CSE and LICM to reorder them past other memory operations.
///
/// Returns the number of alias refinements applied.
pub fn enable_load_store_reorder(func: &mut MachFunction) -> u32 {
    let mut pass = ProofOptimization::new();
    pass.run(func);
    pass.stats().alias_refinements
}

/// Enable aggressive CSE for operations with Pure proof annotation.
///
/// Removes memory flags from proven-pure operations (e.g., loads from
/// immutable data) so they can be CSE'd and LICM'd like pure computations.
///
/// Returns the number of operations promoted to pure.
pub fn aggressive_cse(func: &mut MachFunction) -> u32 {
    let mut pass = ProofOptimization::new();
    pass.run(func);
    pass.stats().pure_cse_enabled
}

#[cfg(test)]
mod tests {
    use super::*;
    use llvm2_ir::{
        AArch64Opcode, BlockId, InstFlags, InstId, MachFunction, MachInst, MachOperand,
        ProofAnnotation, RegClass, Signature, VReg,
    };

    fn vreg(id: u32) -> MachOperand {
        MachOperand::VReg(VReg::new(id, RegClass::Gpr64))
    }

    fn imm(val: i64) -> MachOperand {
        MachOperand::Imm(val)
    }

    fn make_func_with_insts(insts: Vec<MachInst>) -> MachFunction {
        let mut func = MachFunction::new(
            "test_proof_opts".to_string(),
            Signature::new(vec![], vec![]),
        );
        let block = func.entry;
        for inst in insts {
            let id = func.push_inst(inst);
            func.append_inst(block, id);
        }
        func
    }

    // --- NoOverflow tests ---

    #[test]
    fn test_no_overflow_adds_to_add() {
        // Pattern: adds v0, v1, v2 [NoOverflow] + trap_overflow
        // Expected: add v0, v1, v2 (trap removed)
        let adds = MachInst::new(
            AArch64Opcode::AddsRR,
            vec![vreg(0), vreg(1), vreg(2)],
        )
        .with_proof(ProofAnnotation::NoOverflow);

        let trap = MachInst::new(
            AArch64Opcode::TrapOverflow,
            vec![imm(0x06), MachOperand::Block(BlockId(1))],
        );

        let ret = MachInst::new(AArch64Opcode::Ret, vec![]);
        let mut func = make_func_with_insts(vec![adds, trap, ret]);

        // Create the panic block so the function is well-formed.
        func.create_block();

        let mut pass = ProofOptimization::new();
        assert!(pass.run(&mut func));

        // The ADDS should be converted to ADD.
        let inst = func.inst(InstId(0));
        assert_eq!(inst.opcode, AArch64Opcode::AddRR);
        assert!(inst.proof.is_none());

        // The TrapOverflow should be removed from the block.
        let block = func.block(func.entry);
        assert_eq!(block.insts.len(), 2); // add + ret

        // Verify stats.
        assert_eq!(pass.stats().overflow_checks_eliminated, 1);
    }

    #[test]
    fn test_no_overflow_subs_to_sub() {
        let subs = MachInst::new(
            AArch64Opcode::SubsRI,
            vec![vreg(0), vreg(1), imm(42)],
        )
        .with_proof(ProofAnnotation::NoOverflow);

        let trap = MachInst::new(
            AArch64Opcode::TrapOverflow,
            vec![imm(0x06), MachOperand::Block(BlockId(1))],
        );

        let ret = MachInst::new(AArch64Opcode::Ret, vec![]);
        let mut func = make_func_with_insts(vec![subs, trap, ret]);
        func.create_block();

        let mut pass = ProofOptimization::new();
        assert!(pass.run(&mut func));

        let inst = func.inst(InstId(0));
        assert_eq!(inst.opcode, AArch64Opcode::SubRI);
        assert_eq!(pass.stats().overflow_checks_eliminated, 1);
    }

    #[test]
    fn test_no_overflow_without_trap_still_converts() {
        // ADDS with NoOverflow but no following TrapOverflow.
        // Should still convert to ADD.
        let adds = MachInst::new(
            AArch64Opcode::AddsRR,
            vec![vreg(0), vreg(1), vreg(2)],
        )
        .with_proof(ProofAnnotation::NoOverflow);

        let ret = MachInst::new(AArch64Opcode::Ret, vec![]);
        let mut func = make_func_with_insts(vec![adds, ret]);

        let mut pass = ProofOptimization::new();
        assert!(pass.run(&mut func));

        let inst = func.inst(InstId(0));
        assert_eq!(inst.opcode, AArch64Opcode::AddRR);
    }

    // --- InBounds tests ---

    #[test]
    fn test_in_bounds_eliminates_check() {
        // Pattern: cmp idx, len [InBounds] + trap_bounds_check
        // Expected: both removed
        let cmp = MachInst::new(
            AArch64Opcode::CmpRR,
            vec![vreg(0), vreg(1)],
        )
        .with_proof(ProofAnnotation::InBounds);

        let trap = MachInst::new(
            AArch64Opcode::TrapBoundsCheck,
            vec![MachOperand::Block(BlockId(1))],
        );

        let ldr = MachInst::new(
            AArch64Opcode::LdrRI,
            vec![vreg(2), vreg(3), imm(0)],
        );
        let ret = MachInst::new(AArch64Opcode::Ret, vec![]);
        let mut func = make_func_with_insts(vec![cmp, trap, ldr, ret]);
        func.create_block();

        let mut pass = ProofOptimization::new();
        assert!(pass.run(&mut func));

        // CMP and TrapBoundsCheck should be removed, LDR and RET remain.
        let block = func.block(func.entry);
        assert_eq!(block.insts.len(), 2); // ldr + ret

        assert_eq!(pass.stats().bounds_checks_eliminated, 1);
    }

    #[test]
    fn test_in_bounds_cmp_without_trap_no_change() {
        // CMP with InBounds but no TrapBoundsCheck following — no change.
        let cmp = MachInst::new(
            AArch64Opcode::CmpRR,
            vec![vreg(0), vreg(1)],
        )
        .with_proof(ProofAnnotation::InBounds);

        let ret = MachInst::new(AArch64Opcode::Ret, vec![]);
        let mut func = make_func_with_insts(vec![cmp, ret]);

        let mut pass = ProofOptimization::new();
        assert!(!pass.run(&mut func));
    }

    // --- NotNull tests ---

    #[test]
    fn test_not_null_eliminates_cbz() {
        // Pattern: cbz ptr, panic_block [NotNull]
        // Expected: cbz removed
        let cbz = MachInst::new(
            AArch64Opcode::Cbz,
            vec![vreg(0), MachOperand::Block(BlockId(1))],
        )
        .with_proof(ProofAnnotation::NotNull);

        let ldr = MachInst::new(
            AArch64Opcode::LdrRI,
            vec![vreg(1), vreg(0), imm(0)],
        );
        let ret = MachInst::new(AArch64Opcode::Ret, vec![]);
        let mut func = make_func_with_insts(vec![cbz, ldr, ret]);
        func.create_block();

        let mut pass = ProofOptimization::new();
        assert!(pass.run(&mut func));

        let block = func.block(func.entry);
        assert_eq!(block.insts.len(), 2); // ldr + ret

        assert_eq!(pass.stats().null_checks_eliminated, 1);
    }

    #[test]
    fn test_not_null_cbnz_to_unconditional_branch() {
        // Pattern: cbnz ptr, continue_block [NotNull]
        // Expected: b continue_block (always branches since ptr != null)
        let cbnz = MachInst::new(
            AArch64Opcode::Cbnz,
            vec![vreg(0), MachOperand::Block(BlockId(1))],
        )
        .with_proof(ProofAnnotation::NotNull);

        let mut func = make_func_with_insts(vec![cbnz]);
        func.create_block();

        let mut pass = ProofOptimization::new();
        assert!(pass.run(&mut func));

        let inst = func.inst(InstId(0));
        assert_eq!(inst.opcode, AArch64Opcode::B);
        assert_eq!(inst.operands.len(), 1);
        assert_eq!(inst.operands[0], MachOperand::Block(BlockId(1)));
        assert!(inst.proof.is_none());

        assert_eq!(pass.stats().null_checks_eliminated, 1);
    }

    #[test]
    fn test_not_null_trap_null_eliminated() {
        // Pattern: trap_null panic_block [NotNull]
        // Expected: removed
        let trap = MachInst::new(
            AArch64Opcode::TrapNull,
            vec![MachOperand::Block(BlockId(1))],
        )
        .with_proof(ProofAnnotation::NotNull);

        let ret = MachInst::new(AArch64Opcode::Ret, vec![]);
        let mut func = make_func_with_insts(vec![trap, ret]);
        func.create_block();

        let mut pass = ProofOptimization::new();
        assert!(pass.run(&mut func));

        let block = func.block(func.entry);
        assert_eq!(block.insts.len(), 1); // only ret

        assert_eq!(pass.stats().null_checks_eliminated, 1);
    }

    #[test]
    fn test_not_null_cmp_zero_bcond_eliminated() {
        // Pattern: cmp ptr, #0 [NotNull] + bcond EQ, panic_block
        // Expected: both removed
        let cmp = MachInst::new(
            AArch64Opcode::CmpRI,
            vec![vreg(0), imm(0)],
        )
        .with_proof(ProofAnnotation::NotNull);

        let bcond = MachInst::new(
            AArch64Opcode::BCond,
            vec![imm(0x00), MachOperand::Block(BlockId(1))],
        );

        let ret = MachInst::new(AArch64Opcode::Ret, vec![]);
        let mut func = make_func_with_insts(vec![cmp, bcond, ret]);
        func.create_block();

        let mut pass = ProofOptimization::new();
        assert!(pass.run(&mut func));

        let block = func.block(func.entry);
        assert_eq!(block.insts.len(), 1); // only ret

        assert_eq!(pass.stats().null_checks_eliminated, 1);
    }

    // --- ValidBorrow tests ---

    #[test]
    fn test_valid_borrow_refines_load() {
        // A load with ValidBorrow proof.
        let ldr = MachInst::new(
            AArch64Opcode::LdrRI,
            vec![vreg(0), vreg(1), imm(0)],
        )
        .with_proof(ProofAnnotation::ValidBorrow);

        let ret = MachInst::new(AArch64Opcode::Ret, vec![]);
        let mut func = make_func_with_insts(vec![ldr, ret]);

        let mut pass = ProofOptimization::new();
        assert!(pass.run(&mut func));

        assert_eq!(pass.stats().alias_refinements, 1);
    }

    #[test]
    fn test_valid_borrow_refines_store() {
        let str_inst = MachInst::new(
            AArch64Opcode::StrRI,
            vec![vreg(0), vreg(1), imm(8)],
        )
        .with_proof(ProofAnnotation::ValidBorrow);

        let ret = MachInst::new(AArch64Opcode::Ret, vec![]);
        let mut func = make_func_with_insts(vec![str_inst, ret]);

        let mut pass = ProofOptimization::new();
        assert!(pass.run(&mut func));

        assert_eq!(pass.stats().alias_refinements, 1);
    }

    #[test]
    fn test_valid_borrow_non_memory_no_effect() {
        // ValidBorrow on a non-memory instruction should not count.
        let add = MachInst::new(
            AArch64Opcode::AddRR,
            vec![vreg(0), vreg(1), vreg(2)],
        )
        .with_proof(ProofAnnotation::ValidBorrow);

        let ret = MachInst::new(AArch64Opcode::Ret, vec![]);
        let mut func = make_func_with_insts(vec![add, ret]);

        let mut pass = ProofOptimization::new();
        assert!(!pass.run(&mut func));

        assert_eq!(pass.stats().alias_refinements, 0);
    }

    // --- PositiveRefCount tests ---

    #[test]
    fn test_positive_refcount_eliminates_retain_release_pair() {
        // Pattern: retain ptr [PositiveRefCount] + release ptr
        // Expected: both removed
        let retain = MachInst::new(
            AArch64Opcode::Retain,
            vec![vreg(0)],
        )
        .with_proof(ProofAnnotation::PositiveRefCount);

        let release = MachInst::new(
            AArch64Opcode::Release,
            vec![vreg(0)],
        );

        let ret = MachInst::new(AArch64Opcode::Ret, vec![]);
        let mut func = make_func_with_insts(vec![retain, release, ret]);

        let mut pass = ProofOptimization::new();
        assert!(pass.run(&mut func));

        let block = func.block(func.entry);
        assert_eq!(block.insts.len(), 1); // only ret

        assert_eq!(pass.stats().refcount_pairs_eliminated, 1);
    }

    #[test]
    fn test_positive_refcount_no_match_different_ptr() {
        // retain v0 [PositiveRefCount] + release v1 — different ptrs, no elim.
        let retain = MachInst::new(
            AArch64Opcode::Retain,
            vec![vreg(0)],
        )
        .with_proof(ProofAnnotation::PositiveRefCount);

        let release = MachInst::new(
            AArch64Opcode::Release,
            vec![vreg(1)],
        );

        let ret = MachInst::new(AArch64Opcode::Ret, vec![]);
        let mut func = make_func_with_insts(vec![retain, release, ret]);

        let mut pass = ProofOptimization::new();
        assert!(!pass.run(&mut func));

        let block = func.block(func.entry);
        assert_eq!(block.insts.len(), 3); // all preserved
    }

    #[test]
    fn test_positive_refcount_call_blocks_elimination() {
        // retain v0 [PositiveRefCount] + bl foo + release v0
        // The call prevents elimination.
        let retain = MachInst::new(
            AArch64Opcode::Retain,
            vec![vreg(0)],
        )
        .with_proof(ProofAnnotation::PositiveRefCount);

        let call = MachInst::new(
            AArch64Opcode::Bl,
            vec![MachOperand::Imm(0)],
        );

        let release = MachInst::new(
            AArch64Opcode::Release,
            vec![vreg(0)],
        );

        let ret = MachInst::new(AArch64Opcode::Ret, vec![]);
        let mut func = make_func_with_insts(vec![retain, call, release, ret]);

        let mut pass = ProofOptimization::new();
        assert!(!pass.run(&mut func));

        let block = func.block(func.entry);
        assert_eq!(block.insts.len(), 4); // all preserved
    }

    #[test]
    fn test_positive_refcount_with_intervening_instructions() {
        // retain v0 [PositiveRefCount] + add v1, v2, v3 + release v0
        // Non-memory instruction between retain/release is fine.
        let retain = MachInst::new(
            AArch64Opcode::Retain,
            vec![vreg(0)],
        )
        .with_proof(ProofAnnotation::PositiveRefCount);

        let add = MachInst::new(
            AArch64Opcode::AddRR,
            vec![vreg(1), vreg(2), vreg(3)],
        );

        let release = MachInst::new(
            AArch64Opcode::Release,
            vec![vreg(0)],
        );

        let ret = MachInst::new(AArch64Opcode::Ret, vec![]);
        let mut func = make_func_with_insts(vec![retain, add, release, ret]);

        let mut pass = ProofOptimization::new();
        assert!(pass.run(&mut func));

        let block = func.block(func.entry);
        assert_eq!(block.insts.len(), 2); // add + ret

        assert_eq!(pass.stats().refcount_pairs_eliminated, 1);
    }

    // --- Idempotency tests ---

    #[test]
    fn test_pass_is_idempotent() {
        let adds = MachInst::new(
            AArch64Opcode::AddsRR,
            vec![vreg(0), vreg(1), vreg(2)],
        )
        .with_proof(ProofAnnotation::NoOverflow);

        let trap = MachInst::new(
            AArch64Opcode::TrapOverflow,
            vec![imm(0x06), MachOperand::Block(BlockId(1))],
        );

        let ret = MachInst::new(AArch64Opcode::Ret, vec![]);
        let mut func = make_func_with_insts(vec![adds, trap, ret]);
        func.create_block();

        let mut pass = ProofOptimization::new();
        assert!(pass.run(&mut func));
        // Second run should find nothing to do.
        assert!(!pass.run(&mut func));
    }

    // --- No annotation, no optimization ---

    #[test]
    fn test_no_annotation_no_change() {
        // ADDS without proof annotation should not be optimized.
        let adds = MachInst::new(
            AArch64Opcode::AddsRR,
            vec![vreg(0), vreg(1), vreg(2)],
        );

        let trap = MachInst::new(
            AArch64Opcode::TrapOverflow,
            vec![imm(0x06), MachOperand::Block(BlockId(1))],
        );

        let ret = MachInst::new(AArch64Opcode::Ret, vec![]);
        let mut func = make_func_with_insts(vec![adds, trap, ret]);
        func.create_block();

        let mut pass = ProofOptimization::new();
        assert!(!pass.run(&mut func));

        // Everything should be preserved.
        let block = func.block(func.entry);
        assert_eq!(block.insts.len(), 3);
        assert_eq!(func.inst(InstId(0)).opcode, AArch64Opcode::AddsRR);
    }

    // --- Multi-block tests ---

    #[test]
    fn test_proof_opts_across_multiple_blocks() {
        let mut func = MachFunction::new(
            "test_multi_block".to_string(),
            Signature::new(vec![], vec![]),
        );

        // Block 0: overflow check
        let adds = MachInst::new(
            AArch64Opcode::AddsRR,
            vec![vreg(0), vreg(1), vreg(2)],
        )
        .with_proof(ProofAnnotation::NoOverflow);
        let trap_ov = MachInst::new(
            AArch64Opcode::TrapOverflow,
            vec![imm(0x06), MachOperand::Block(BlockId(2))],
        );
        let branch = MachInst::new(
            AArch64Opcode::B,
            vec![MachOperand::Block(BlockId(1))],
        );

        let adds_id = func.push_inst(adds);
        let trap_ov_id = func.push_inst(trap_ov);
        let branch_id = func.push_inst(branch);
        func.append_inst(BlockId(0), adds_id);
        func.append_inst(BlockId(0), trap_ov_id);
        func.append_inst(BlockId(0), branch_id);

        // Block 1: null check
        let bb1 = func.create_block();
        let cbz = MachInst::new(
            AArch64Opcode::Cbz,
            vec![vreg(3), MachOperand::Block(BlockId(2))],
        )
        .with_proof(ProofAnnotation::NotNull);
        let ret = MachInst::new(AArch64Opcode::Ret, vec![]);

        let cbz_id = func.push_inst(cbz);
        let ret_id = func.push_inst(ret);
        func.append_inst(bb1, cbz_id);
        func.append_inst(bb1, ret_id);

        // Block 2: panic (unused in this test)
        func.create_block();

        let mut pass = ProofOptimization::new();
        assert!(pass.run(&mut func));

        // Block 0: adds→add, trap removed, branch kept
        let block0 = func.block(BlockId(0));
        assert_eq!(block0.insts.len(), 2); // add + b
        assert_eq!(func.inst(adds_id).opcode, AArch64Opcode::AddRR);

        // Block 1: cbz removed, ret kept
        let block1 = func.block(bb1);
        assert_eq!(block1.insts.len(), 1); // ret

        assert_eq!(pass.stats().overflow_checks_eliminated, 1);
        assert_eq!(pass.stats().null_checks_eliminated, 1);
    }

    // --- NonZeroDivisor tests ---

    #[test]
    fn test_non_zero_divisor_eliminates_cbz() {
        // Pattern: cbz divisor, trap_block [NonZeroDivisor] + udiv result, dividend, divisor
        // Expected: cbz removed, udiv preserved
        let cbz = MachInst::new(
            AArch64Opcode::Cbz,
            vec![vreg(1), MachOperand::Block(BlockId(1))],
        )
        .with_proof(ProofAnnotation::NonZeroDivisor);

        let udiv = MachInst::new(
            AArch64Opcode::UDiv,
            vec![vreg(2), vreg(0), vreg(1)],
        );

        let ret = MachInst::new(AArch64Opcode::Ret, vec![]);
        let mut func = make_func_with_insts(vec![cbz, udiv, ret]);
        func.create_block();

        let mut pass = ProofOptimization::new();
        assert!(pass.run(&mut func));

        // CBZ should be removed, UDIV and RET remain.
        let block = func.block(func.entry);
        assert_eq!(block.insts.len(), 2); // udiv + ret
        assert_eq!(func.inst(InstId(1)).opcode, AArch64Opcode::UDiv);

        assert_eq!(pass.stats().divzero_checks_eliminated, 1);
    }

    #[test]
    fn test_non_zero_divisor_eliminates_trap_divzero() {
        // Pattern: trap_divzero panic_block [NonZeroDivisor]
        // Expected: removed
        let trap = MachInst::new(
            AArch64Opcode::TrapDivZero,
            vec![MachOperand::Block(BlockId(1))],
        )
        .with_proof(ProofAnnotation::NonZeroDivisor);

        let sdiv = MachInst::new(
            AArch64Opcode::SDiv,
            vec![vreg(2), vreg(0), vreg(1)],
        );

        let ret = MachInst::new(AArch64Opcode::Ret, vec![]);
        let mut func = make_func_with_insts(vec![trap, sdiv, ret]);
        func.create_block();

        let mut pass = ProofOptimization::new();
        assert!(pass.run(&mut func));

        let block = func.block(func.entry);
        assert_eq!(block.insts.len(), 2); // sdiv + ret

        assert_eq!(pass.stats().divzero_checks_eliminated, 1);
    }

    #[test]
    fn test_non_zero_divisor_cmp_zero_bcond_eliminated() {
        // Pattern: cmp divisor, #0 [NonZeroDivisor] + bcond EQ, trap_block
        // Expected: both removed
        let cmp = MachInst::new(
            AArch64Opcode::CmpRI,
            vec![vreg(1), imm(0)],
        )
        .with_proof(ProofAnnotation::NonZeroDivisor);

        let bcond = MachInst::new(
            AArch64Opcode::BCond,
            vec![imm(0x00), MachOperand::Block(BlockId(1))],
        );

        let udiv = MachInst::new(
            AArch64Opcode::UDiv,
            vec![vreg(2), vreg(0), vreg(1)],
        );

        let ret = MachInst::new(AArch64Opcode::Ret, vec![]);
        let mut func = make_func_with_insts(vec![cmp, bcond, udiv, ret]);
        func.create_block();

        let mut pass = ProofOptimization::new();
        assert!(pass.run(&mut func));

        // CMP and BCond removed, UDIV and RET remain.
        let block = func.block(func.entry);
        assert_eq!(block.insts.len(), 2); // udiv + ret

        assert_eq!(pass.stats().divzero_checks_eliminated, 1);
    }

    #[test]
    fn test_non_zero_divisor_no_proof_no_change() {
        // CBZ without proof annotation should not be optimized.
        let cbz = MachInst::new(
            AArch64Opcode::Cbz,
            vec![vreg(1), MachOperand::Block(BlockId(1))],
        );

        let udiv = MachInst::new(
            AArch64Opcode::UDiv,
            vec![vreg(2), vreg(0), vreg(1)],
        );

        let ret = MachInst::new(AArch64Opcode::Ret, vec![]);
        let mut func = make_func_with_insts(vec![cbz, udiv, ret]);
        func.create_block();

        let mut pass = ProofOptimization::new();
        assert!(!pass.run(&mut func));

        let block = func.block(func.entry);
        assert_eq!(block.insts.len(), 3); // all preserved
    }

    #[test]
    fn test_non_zero_divisor_sdiv_pattern() {
        // Same as cbz test but with SDIV instead of UDIV.
        let cbz = MachInst::new(
            AArch64Opcode::Cbz,
            vec![vreg(1), MachOperand::Block(BlockId(1))],
        )
        .with_proof(ProofAnnotation::NonZeroDivisor);

        let sdiv = MachInst::new(
            AArch64Opcode::SDiv,
            vec![vreg(2), vreg(0), vreg(1)],
        );

        let ret = MachInst::new(AArch64Opcode::Ret, vec![]);
        let mut func = make_func_with_insts(vec![cbz, sdiv, ret]);
        func.create_block();

        let mut pass = ProofOptimization::new();
        assert!(pass.run(&mut func));

        let block = func.block(func.entry);
        assert_eq!(block.insts.len(), 2); // sdiv + ret

        assert_eq!(pass.stats().divzero_checks_eliminated, 1);
    }

    // --- ValidShift tests ---

    #[test]
    fn test_valid_shift_eliminates_cmp_trap() {
        // Pattern: cmp shift_amt, #64 [ValidShift] + trap_shift_range panic_block
        // Expected: both removed, LSL preserved
        let cmp = MachInst::new(
            AArch64Opcode::CmpRI,
            vec![vreg(1), imm(64)],
        )
        .with_proof(ProofAnnotation::ValidShift);

        let trap = MachInst::new(
            AArch64Opcode::TrapShiftRange,
            vec![MachOperand::Block(BlockId(1))],
        );

        let lsl = MachInst::new(
            AArch64Opcode::LslRR,
            vec![vreg(2), vreg(0), vreg(1)],
        );

        let ret = MachInst::new(AArch64Opcode::Ret, vec![]);
        let mut func = make_func_with_insts(vec![cmp, trap, lsl, ret]);
        func.create_block();

        let mut pass = ProofOptimization::new();
        assert!(pass.run(&mut func));

        // CMP and TrapShiftRange removed, LSL and RET remain.
        let block = func.block(func.entry);
        assert_eq!(block.insts.len(), 2); // lsl + ret

        assert_eq!(pass.stats().shift_checks_eliminated, 1);
    }

    #[test]
    fn test_valid_shift_eliminates_cmp_bcond() {
        // Pattern: cmp shift_amt, #64 [ValidShift] + bcond GE, trap_block
        // Expected: both removed
        let cmp = MachInst::new(
            AArch64Opcode::CmpRI,
            vec![vreg(1), imm(64)],
        )
        .with_proof(ProofAnnotation::ValidShift);

        let bcond = MachInst::new(
            AArch64Opcode::BCond,
            vec![imm(0x0A), MachOperand::Block(BlockId(1))], // 0x0A = GE condition
        );

        let lsr = MachInst::new(
            AArch64Opcode::LsrRR,
            vec![vreg(2), vreg(0), vreg(1)],
        );

        let ret = MachInst::new(AArch64Opcode::Ret, vec![]);
        let mut func = make_func_with_insts(vec![cmp, bcond, lsr, ret]);
        func.create_block();

        let mut pass = ProofOptimization::new();
        assert!(pass.run(&mut func));

        let block = func.block(func.entry);
        assert_eq!(block.insts.len(), 2); // lsr + ret

        assert_eq!(pass.stats().shift_checks_eliminated, 1);
    }

    #[test]
    fn test_valid_shift_32bit() {
        // Pattern: cmp shift_amt, #32 [ValidShift] + trap_shift_range
        // Expected: both removed (32-bit shift width)
        let cmp = MachInst::new(
            AArch64Opcode::CmpRI,
            vec![vreg(1), imm(32)],
        )
        .with_proof(ProofAnnotation::ValidShift);

        let trap = MachInst::new(
            AArch64Opcode::TrapShiftRange,
            vec![MachOperand::Block(BlockId(1))],
        );

        let asr = MachInst::new(
            AArch64Opcode::AsrRR,
            vec![vreg(2), vreg(0), vreg(1)],
        );

        let ret = MachInst::new(AArch64Opcode::Ret, vec![]);
        let mut func = make_func_with_insts(vec![cmp, trap, asr, ret]);
        func.create_block();

        let mut pass = ProofOptimization::new();
        assert!(pass.run(&mut func));

        let block = func.block(func.entry);
        assert_eq!(block.insts.len(), 2); // asr + ret

        assert_eq!(pass.stats().shift_checks_eliminated, 1);
    }

    #[test]
    fn test_valid_shift_trap_only() {
        // Pattern: trap_shift_range panic_block [ValidShift]
        // Expected: removed
        let trap = MachInst::new(
            AArch64Opcode::TrapShiftRange,
            vec![MachOperand::Block(BlockId(1))],
        )
        .with_proof(ProofAnnotation::ValidShift);

        let lsl = MachInst::new(
            AArch64Opcode::LslRR,
            vec![vreg(2), vreg(0), vreg(1)],
        );

        let ret = MachInst::new(AArch64Opcode::Ret, vec![]);
        let mut func = make_func_with_insts(vec![trap, lsl, ret]);
        func.create_block();

        let mut pass = ProofOptimization::new();
        assert!(pass.run(&mut func));

        let block = func.block(func.entry);
        assert_eq!(block.insts.len(), 2); // lsl + ret

        assert_eq!(pass.stats().shift_checks_eliminated, 1);
    }

    #[test]
    fn test_valid_shift_no_proof_no_change() {
        // CMP with #64 but no ValidShift proof should not be optimized.
        let cmp = MachInst::new(
            AArch64Opcode::CmpRI,
            vec![vreg(1), imm(64)],
        );

        let trap = MachInst::new(
            AArch64Opcode::TrapShiftRange,
            vec![MachOperand::Block(BlockId(1))],
        );

        let lsl = MachInst::new(
            AArch64Opcode::LslRR,
            vec![vreg(2), vreg(0), vreg(1)],
        );

        let ret = MachInst::new(AArch64Opcode::Ret, vec![]);
        let mut func = make_func_with_insts(vec![cmp, trap, lsl, ret]);
        func.create_block();

        let mut pass = ProofOptimization::new();
        assert!(!pass.run(&mut func));

        let block = func.block(func.entry);
        assert_eq!(block.insts.len(), 4); // all preserved
    }

    #[test]
    fn test_valid_shift_cmp_without_trap_no_change() {
        // CMP with ValidShift but no TrapShiftRange or BCond following.
        let cmp = MachInst::new(
            AArch64Opcode::CmpRI,
            vec![vreg(1), imm(64)],
        )
        .with_proof(ProofAnnotation::ValidShift);

        let lsl = MachInst::new(
            AArch64Opcode::LslRR,
            vec![vreg(2), vreg(0), vreg(1)],
        );

        let ret = MachInst::new(AArch64Opcode::Ret, vec![]);
        let mut func = make_func_with_insts(vec![cmp, lsl, ret]);

        let mut pass = ProofOptimization::new();
        assert!(!pass.run(&mut func));
    }

    // --- Combined new + existing proof opts ---

    #[test]
    fn test_combined_divzero_and_shift_opts() {
        // Two blocks: one with NonZeroDivisor, one with ValidShift.
        let mut func = MachFunction::new(
            "test_combined_new_opts".to_string(),
            Signature::new(vec![], vec![]),
        );

        // Block 0: div-zero check
        let cbz = MachInst::new(
            AArch64Opcode::Cbz,
            vec![vreg(1), MachOperand::Block(BlockId(2))],
        )
        .with_proof(ProofAnnotation::NonZeroDivisor);
        let udiv = MachInst::new(
            AArch64Opcode::UDiv,
            vec![vreg(2), vreg(0), vreg(1)],
        );
        let branch = MachInst::new(
            AArch64Opcode::B,
            vec![MachOperand::Block(BlockId(1))],
        );

        let cbz_id = func.push_inst(cbz);
        let udiv_id = func.push_inst(udiv);
        let branch_id = func.push_inst(branch);
        func.append_inst(BlockId(0), cbz_id);
        func.append_inst(BlockId(0), udiv_id);
        func.append_inst(BlockId(0), branch_id);

        // Block 1: shift range check
        let bb1 = func.create_block();
        let cmp = MachInst::new(
            AArch64Opcode::CmpRI,
            vec![vreg(3), imm(64)],
        )
        .with_proof(ProofAnnotation::ValidShift);
        let trap_shift = MachInst::new(
            AArch64Opcode::TrapShiftRange,
            vec![MachOperand::Block(BlockId(2))],
        );
        let lsl = MachInst::new(
            AArch64Opcode::LslRR,
            vec![vreg(4), vreg(0), vreg(3)],
        );
        let ret = MachInst::new(AArch64Opcode::Ret, vec![]);

        let cmp_id = func.push_inst(cmp);
        let trap_shift_id = func.push_inst(trap_shift);
        let lsl_id = func.push_inst(lsl);
        let ret_id = func.push_inst(ret);
        func.append_inst(bb1, cmp_id);
        func.append_inst(bb1, trap_shift_id);
        func.append_inst(bb1, lsl_id);
        func.append_inst(bb1, ret_id);

        // Block 2: panic (unused)
        func.create_block();

        let mut pass = ProofOptimization::new();
        assert!(pass.run(&mut func));

        // Block 0: cbz removed → udiv + b
        let block0 = func.block(BlockId(0));
        assert_eq!(block0.insts.len(), 2);

        // Block 1: cmp + trap removed → lsl + ret
        let block1 = func.block(bb1);
        assert_eq!(block1.insts.len(), 2);

        assert_eq!(pass.stats().divzero_checks_eliminated, 1);
        assert_eq!(pass.stats().shift_checks_eliminated, 1);
    }

    // --- Pure (aggressive CSE) tests ---

    #[test]
    fn test_pure_promotes_load_to_cse_able() {
        // A load with Pure proof should have its READS_MEMORY flag removed,
        // making it eligible for CSE by downstream passes.
        let ldr = MachInst::new(
            AArch64Opcode::LdrRI,
            vec![vreg(0), vreg(1), imm(0)],
        )
        .with_proof(ProofAnnotation::Pure);

        let ret = MachInst::new(AArch64Opcode::Ret, vec![]);
        let mut func = make_func_with_insts(vec![ldr, ret]);

        let mut pass = ProofOptimization::new();
        assert!(pass.run(&mut func));

        // The load should no longer have READS_MEMORY flag.
        let inst = func.inst(InstId(0));
        assert!(!inst.flags.contains(InstFlags::READS_MEMORY));
        // The proof should be consumed.
        assert!(inst.proof.is_none());
        assert_eq!(pass.stats().pure_cse_enabled, 1);
    }

    #[test]
    fn test_pure_promotes_store_to_cse_able() {
        // A store with Pure proof should have its WRITES_MEMORY + HAS_SIDE_EFFECTS removed.
        let str_inst = MachInst::new(
            AArch64Opcode::StrRI,
            vec![vreg(0), vreg(1), imm(8)],
        )
        .with_proof(ProofAnnotation::Pure);

        let ret = MachInst::new(AArch64Opcode::Ret, vec![]);
        let mut func = make_func_with_insts(vec![str_inst, ret]);

        let mut pass = ProofOptimization::new();
        assert!(pass.run(&mut func));

        let inst = func.inst(InstId(0));
        assert!(!inst.flags.contains(InstFlags::WRITES_MEMORY));
        assert!(!inst.flags.contains(InstFlags::HAS_SIDE_EFFECTS));
        assert!(inst.proof.is_none());
        assert_eq!(pass.stats().pure_cse_enabled, 1);
    }

    #[test]
    fn test_pure_on_already_pure_instruction_no_effect() {
        // An ADD instruction is already pure — Pure proof has no effect.
        let add = MachInst::new(
            AArch64Opcode::AddRR,
            vec![vreg(0), vreg(1), vreg(2)],
        )
        .with_proof(ProofAnnotation::Pure);

        let ret = MachInst::new(AArch64Opcode::Ret, vec![]);
        let mut func = make_func_with_insts(vec![add, ret]);

        let mut pass = ProofOptimization::new();
        // Should return false because the ADD is already pure.
        assert!(!pass.run(&mut func));

        assert_eq!(pass.stats().pure_cse_enabled, 0);
    }

    #[test]
    fn test_pure_multiple_loads() {
        // Two loads with Pure proof: both should be promoted.
        let ldr1 = MachInst::new(
            AArch64Opcode::LdrRI,
            vec![vreg(0), vreg(1), imm(0)],
        )
        .with_proof(ProofAnnotation::Pure);

        let ldr2 = MachInst::new(
            AArch64Opcode::LdrRI,
            vec![vreg(2), vreg(1), imm(8)],
        )
        .with_proof(ProofAnnotation::Pure);

        let ret = MachInst::new(AArch64Opcode::Ret, vec![]);
        let mut func = make_func_with_insts(vec![ldr1, ldr2, ret]);

        let mut pass = ProofOptimization::new();
        assert!(pass.run(&mut func));

        assert_eq!(pass.stats().pure_cse_enabled, 2);

        // Both loads should have READS_MEMORY removed.
        assert!(!func.inst(InstId(0)).flags.contains(InstFlags::READS_MEMORY));
        assert!(!func.inst(InstId(1)).flags.contains(InstFlags::READS_MEMORY));
    }

    #[test]
    fn test_pure_without_proof_load_unchanged() {
        // A load without Pure proof should keep its READS_MEMORY flag.
        let ldr = MachInst::new(
            AArch64Opcode::LdrRI,
            vec![vreg(0), vreg(1), imm(0)],
        );

        let ret = MachInst::new(AArch64Opcode::Ret, vec![]);
        let mut func = make_func_with_insts(vec![ldr, ret]);

        let mut pass = ProofOptimization::new();
        assert!(!pass.run(&mut func));

        assert!(func.inst(InstId(0)).flags.contains(InstFlags::READS_MEMORY));
        assert_eq!(pass.stats().pure_cse_enabled, 0);
    }

    // --- ValidBorrow reordering flag tests ---

    #[test]
    fn test_valid_borrow_sets_reorderable_flag_on_load() {
        // A load with ValidBorrow should get the PROOF_REORDERABLE flag.
        let ldr = MachInst::new(
            AArch64Opcode::LdrRI,
            vec![vreg(0), vreg(1), imm(0)],
        )
        .with_proof(ProofAnnotation::ValidBorrow);

        let ret = MachInst::new(AArch64Opcode::Ret, vec![]);
        let mut func = make_func_with_insts(vec![ldr, ret]);

        let mut pass = ProofOptimization::new();
        assert!(pass.run(&mut func));

        let inst = func.inst(InstId(0));
        assert!(inst.flags.contains(InstFlags::PROOF_REORDERABLE));
        assert_eq!(pass.stats().alias_refinements, 1);
    }

    #[test]
    fn test_valid_borrow_sets_reorderable_flag_on_store() {
        // A store with ValidBorrow should get the PROOF_REORDERABLE flag.
        let str_inst = MachInst::new(
            AArch64Opcode::StrRI,
            vec![vreg(0), vreg(1), imm(8)],
        )
        .with_proof(ProofAnnotation::ValidBorrow);

        let ret = MachInst::new(AArch64Opcode::Ret, vec![]);
        let mut func = make_func_with_insts(vec![str_inst, ret]);

        let mut pass = ProofOptimization::new();
        assert!(pass.run(&mut func));

        let inst = func.inst(InstId(0));
        assert!(inst.flags.contains(InstFlags::PROOF_REORDERABLE));
        // Store should keep its existing memory flags since it still writes.
        assert!(inst.flags.contains(InstFlags::WRITES_MEMORY));
    }

    #[test]
    fn test_valid_borrow_load_store_pair_both_reorderable() {
        // Both a load and store with ValidBorrow get PROOF_REORDERABLE.
        let ldr = MachInst::new(
            AArch64Opcode::LdrRI,
            vec![vreg(0), vreg(1), imm(0)],
        )
        .with_proof(ProofAnnotation::ValidBorrow);

        let str_inst = MachInst::new(
            AArch64Opcode::StrRI,
            vec![vreg(2), vreg(1), imm(8)],
        )
        .with_proof(ProofAnnotation::ValidBorrow);

        let ret = MachInst::new(AArch64Opcode::Ret, vec![]);
        let mut func = make_func_with_insts(vec![ldr, str_inst, ret]);

        let mut pass = ProofOptimization::new();
        assert!(pass.run(&mut func));

        assert!(func.inst(InstId(0)).flags.contains(InstFlags::PROOF_REORDERABLE));
        assert!(func.inst(InstId(1)).flags.contains(InstFlags::PROOF_REORDERABLE));
        assert_eq!(pass.stats().alias_refinements, 2);
    }

    // --- Public API function tests ---

    #[test]
    fn test_eliminate_overflow_checks_public_api() {
        let adds = MachInst::new(
            AArch64Opcode::AddsRR,
            vec![vreg(0), vreg(1), vreg(2)],
        )
        .with_proof(ProofAnnotation::NoOverflow);

        let trap = MachInst::new(
            AArch64Opcode::TrapOverflow,
            vec![imm(0x06), MachOperand::Block(BlockId(1))],
        );

        let ret = MachInst::new(AArch64Opcode::Ret, vec![]);
        let mut func = make_func_with_insts(vec![adds, trap, ret]);
        func.create_block();

        let count = eliminate_overflow_checks(&mut func);
        assert_eq!(count, 1);

        // ADDS should now be ADD.
        assert_eq!(func.inst(InstId(0)).opcode, AArch64Opcode::AddRR);
    }

    #[test]
    fn test_eliminate_bounds_checks_public_api() {
        let cmp = MachInst::new(
            AArch64Opcode::CmpRR,
            vec![vreg(0), vreg(1)],
        )
        .with_proof(ProofAnnotation::InBounds);

        let trap = MachInst::new(
            AArch64Opcode::TrapBoundsCheck,
            vec![MachOperand::Block(BlockId(1))],
        );

        let ret = MachInst::new(AArch64Opcode::Ret, vec![]);
        let mut func = make_func_with_insts(vec![cmp, trap, ret]);
        func.create_block();

        let count = eliminate_bounds_checks(&mut func);
        assert_eq!(count, 1);

        let block = func.block(func.entry);
        assert_eq!(block.insts.len(), 1); // only ret
    }

    #[test]
    fn test_eliminate_null_checks_public_api() {
        let cbz = MachInst::new(
            AArch64Opcode::Cbz,
            vec![vreg(0), MachOperand::Block(BlockId(1))],
        )
        .with_proof(ProofAnnotation::NotNull);

        let ret = MachInst::new(AArch64Opcode::Ret, vec![]);
        let mut func = make_func_with_insts(vec![cbz, ret]);
        func.create_block();

        let count = eliminate_null_checks(&mut func);
        assert_eq!(count, 1);

        let block = func.block(func.entry);
        assert_eq!(block.insts.len(), 1); // only ret
    }

    #[test]
    fn test_enable_load_store_reorder_public_api() {
        let ldr = MachInst::new(
            AArch64Opcode::LdrRI,
            vec![vreg(0), vreg(1), imm(0)],
        )
        .with_proof(ProofAnnotation::ValidBorrow);

        let ret = MachInst::new(AArch64Opcode::Ret, vec![]);
        let mut func = make_func_with_insts(vec![ldr, ret]);

        let count = enable_load_store_reorder(&mut func);
        assert_eq!(count, 1);

        assert!(func.inst(InstId(0)).flags.contains(InstFlags::PROOF_REORDERABLE));
    }

    #[test]
    fn test_aggressive_cse_public_api() {
        let ldr = MachInst::new(
            AArch64Opcode::LdrRI,
            vec![vreg(0), vreg(1), imm(0)],
        )
        .with_proof(ProofAnnotation::Pure);

        let ret = MachInst::new(AArch64Opcode::Ret, vec![]);
        let mut func = make_func_with_insts(vec![ldr, ret]);

        let count = aggressive_cse(&mut func);
        assert_eq!(count, 1);

        assert!(!func.inst(InstId(0)).flags.contains(InstFlags::READS_MEMORY));
    }

    // --- Combined new + existing: all proof types in one function ---

    #[test]
    fn test_all_proof_types_in_one_function() {
        let mut func = MachFunction::new(
            "test_all_proofs".to_string(),
            Signature::new(vec![], vec![]),
        );

        // Block 0: NoOverflow (adds -> add) + ValidBorrow (load reorderable)
        let adds = MachInst::new(
            AArch64Opcode::AddsRR,
            vec![vreg(0), vreg(1), vreg(2)],
        )
        .with_proof(ProofAnnotation::NoOverflow);
        let trap_ov = MachInst::new(
            AArch64Opcode::TrapOverflow,
            vec![imm(0x06), MachOperand::Block(BlockId(2))],
        );
        let ldr_reorder = MachInst::new(
            AArch64Opcode::LdrRI,
            vec![vreg(3), vreg(0), imm(0)],
        )
        .with_proof(ProofAnnotation::ValidBorrow);
        let ldr_pure = MachInst::new(
            AArch64Opcode::LdrRI,
            vec![vreg(4), vreg(0), imm(8)],
        )
        .with_proof(ProofAnnotation::Pure);
        let branch = MachInst::new(
            AArch64Opcode::B,
            vec![MachOperand::Block(BlockId(1))],
        );

        let adds_id = func.push_inst(adds);
        let trap_ov_id = func.push_inst(trap_ov);
        let ldr_reorder_id = func.push_inst(ldr_reorder);
        let ldr_pure_id = func.push_inst(ldr_pure);
        let branch_id = func.push_inst(branch);
        func.append_inst(BlockId(0), adds_id);
        func.append_inst(BlockId(0), trap_ov_id);
        func.append_inst(BlockId(0), ldr_reorder_id);
        func.append_inst(BlockId(0), ldr_pure_id);
        func.append_inst(BlockId(0), branch_id);

        // Block 1: ret
        let bb1 = func.create_block();
        let ret = MachInst::new(AArch64Opcode::Ret, vec![]);
        let ret_id = func.push_inst(ret);
        func.append_inst(bb1, ret_id);

        // Block 2: panic (unused)
        func.create_block();

        let mut pass = ProofOptimization::new();
        assert!(pass.run(&mut func));

        // Block 0: adds→add (trap removed), ldr_reorder (reorderable flag), ldr_pure (pure), branch
        let block0 = func.block(BlockId(0));
        assert_eq!(block0.insts.len(), 4); // add + ldr_reorder + ldr_pure + b

        // Verify: adds → add
        assert_eq!(func.inst(adds_id).opcode, AArch64Opcode::AddRR);

        // Verify: ValidBorrow load has PROOF_REORDERABLE
        assert!(func.inst(ldr_reorder_id).flags.contains(InstFlags::PROOF_REORDERABLE));

        // Verify: Pure load has READS_MEMORY removed
        assert!(!func.inst(ldr_pure_id).flags.contains(InstFlags::READS_MEMORY));

        // Verify stats
        assert_eq!(pass.stats().overflow_checks_eliminated, 1);
        assert_eq!(pass.stats().alias_refinements, 1);
        assert_eq!(pass.stats().pure_cse_enabled, 1);
    }
}
