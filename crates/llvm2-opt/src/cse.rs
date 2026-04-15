// llvm2-opt - Common Subexpression Elimination
//
// Author: Andrew Yates <ayates@dropbox.com>
// Copyright 2026 Dropbox, Inc. | License: Apache-2.0

//! Common Subexpression Elimination (CSE) for machine-level IR.
//!
//! Identifies and eliminates redundant computations: when two instructions
//! compute the same value (same opcode and operands), the second can be
//! replaced with a reference to the first's result.
//!
//! # Safety Requirements
//!
//! **Only Pure instructions are CSE'd.** The memory-effects model in
//! [`effects`] classifies each opcode; only `MemoryEffect::Pure` instructions
//! are candidates. Loads, stores, and calls are never CSE'd because:
//! - Loads may return different values if memory was modified between them.
//! - Stores have side effects.
//! - Calls may have arbitrary side effects.
//!
//! **Dominator-based:** An instruction can only be CSE'd if a dominating
//! instruction with the same opcode and operands exists. This prevents
//! using a value from a non-dominating block (which might not have executed).
//!
//! # Commutative Instructions
//!
//! For commutative operations (add, mul, and, or, xor), operands are
//! canonicalized by sorting before hashing. This allows `add v1, v2` to
//! match `add v2, v1`.
//!
//! # Algorithm
//!
//! 1. Compute dominator tree.
//! 2. Walk blocks in dominator-tree preorder (ensures we see dominators first).
//! 3. For each Pure instruction, compute a canonical key (opcode + sorted operands).
//! 4. Look up in the available-expressions table:
//!    - If found AND the available instruction dominates this one, mark for replacement.
//!    - If not found, insert into the table.
//! 5. Apply replacements: rewrite uses of the eliminated instruction's def
//!    to use the original instruction's def.
//!
//! Reference: LLVM `MachineCSE.cpp`, GVN

use std::collections::HashMap;

use llvm2_ir::{AArch64Opcode, BlockId, InstId, MachFunction, MachOperand, ProofAnnotation, VReg};

use crate::dom::DomTree;
use crate::effects::{opcode_effect, produces_value, MemoryEffect};
use crate::pass_manager::MachinePass;

/// Common Subexpression Elimination pass.
pub struct CommonSubexprElim;

/// A canonical key for an instruction: (opcode, canonicalized operands).
/// Two instructions with the same key compute the same value.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct ExprKey {
    opcode: AArch64Opcode,
    /// Source operands only (excludes the def in operand[0]).
    /// For commutative ops, sorted for canonical form.
    operands: Vec<CanonOperand>,
}

/// A canonicalized operand for hashing purposes.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
enum CanonOperand {
    VReg(u32),
    Imm(i64),
    FImm(u64), // f64 bits for hashing
    Other,     // non-hashable operands (blocks, etc.)
}

impl From<&MachOperand> for CanonOperand {
    fn from(op: &MachOperand) -> Self {
        match op {
            MachOperand::VReg(v) => CanonOperand::VReg(v.id),
            MachOperand::Imm(i) => CanonOperand::Imm(*i),
            MachOperand::FImm(f) => CanonOperand::FImm(f.to_bits()),
            _ => CanonOperand::Other,
        }
    }
}

/// Entry in the available-expressions table.
#[derive(Debug, Clone)]
struct AvailExpr {
    /// The instruction that first computed this expression.
    inst_id: InstId,
    /// The block containing the instruction.
    block: BlockId,
    /// The VReg defined by the instruction (operand[0]).
    def_vreg: VReg,
}

impl MachinePass for CommonSubexprElim {
    fn name(&self) -> &str {
        "cse"
    }

    fn run(&mut self, func: &mut MachFunction) -> bool {
        let dom = DomTree::compute(func);
        run_cse(func, &dom)
    }
}

/// Run CSE on the function, returning true if any changes were made.
fn run_cse(func: &mut MachFunction, dom: &DomTree) -> bool {
    // Table of available expressions: key -> first occurrence.
    let mut available: HashMap<ExprKey, AvailExpr> = HashMap::new();

    // Replacement map: vreg_id of eliminated def -> vreg of original def.
    let mut replacements: HashMap<u32, VReg> = HashMap::new();

    // Instructions to remove (marked dead).
    let mut dead_insts: Vec<InstId> = Vec::new();

    // Proof annotations to merge onto surviving instructions.
    // Maps surviving inst_id -> proof from eliminated duplicate.
    let mut proof_merges: Vec<(InstId, Option<ProofAnnotation>)> = Vec::new();

    // Walk in dominator-tree preorder to see definitions before uses.
    let preorder = dom_preorder(dom, func.entry);

    for &block_id in &preorder {
        let block = func.block(block_id);
        for &inst_id in &block.insts {
            let inst = func.inst(inst_id);

            // Only CSE pure instructions that produce a value.
            if opcode_effect(inst.opcode) != MemoryEffect::Pure {
                continue;
            }
            if !produces_value(inst.opcode) {
                continue;
            }

            // Get the def vreg (operand[0]).
            let def_vreg = match &inst.operands.first() {
                Some(MachOperand::VReg(v)) => *v,
                _ => continue,
            };

            // Build canonical key from source operands.
            let key = make_expr_key(inst.opcode, &inst.operands);

            // Check if we have a key with "Other" operands — skip those
            // as they're not reliably hashable.
            if key.operands.iter().any(|o| matches!(o, CanonOperand::Other)) {
                continue;
            }

            // Look up in available expressions.
            if let Some(avail) = available.get(&key) {
                // Verify dominance: the available instruction's block must
                // dominate the current block.
                if dom.dominates(avail.block, block_id) {
                    // CSE: replace all uses of def_vreg with avail.def_vreg.
                    replacements.insert(def_vreg.id, avail.def_vreg);
                    dead_insts.push(inst_id);

                    // Merge proof annotation from eliminated instruction
                    // onto the surviving one.
                    let eliminated_proof = inst.proof;
                    proof_merges.push((avail.inst_id, eliminated_proof));
                    continue;
                }
                // If the available doesn't dominate, we could update the table
                // if WE dominate it, but that's complex. Keep it simple:
                // first-in-preorder wins.
            }

            // Insert into available expressions table.
            available.insert(
                key,
                AvailExpr {
                    inst_id,
                    block: block_id,
                    def_vreg,
                },
            );
        }
    }

    if replacements.is_empty() {
        return false;
    }

    // Apply proof merges: for each surviving instruction, merge in the
    // proof annotation from the eliminated duplicate.
    for (surviving_id, eliminated_proof) in proof_merges {
        let surviving = func.inst_mut(surviving_id);
        surviving.proof = ProofAnnotation::merge(surviving.proof, eliminated_proof);
    }

    // Apply replacements: rewrite all uses of eliminated vregs.
    for block_id in func.block_order.clone() {
        let block = func.block(block_id);
        for &inst_id in block.insts.clone().iter() {
            let inst = func.inst_mut(inst_id);
            let use_start = if produces_value(inst.opcode) { 1 } else { 0 };

            for i in use_start..inst.operands.len() {
                if let MachOperand::VReg(vreg) = &inst.operands[i]
                    && let Some(replacement) = replacements.get(&vreg.id) {
                        inst.operands[i] = MachOperand::VReg(*replacement);
                    }
            }
        }
    }

    // Remove dead instructions.
    let dead_set: std::collections::HashSet<InstId> = dead_insts.into_iter().collect();
    for block_id in func.block_order.clone() {
        let block = func.block_mut(block_id);
        block.insts.retain(|id| !dead_set.contains(id));
    }

    true
}

/// Build a canonical expression key for an instruction.
///
/// The key consists of the opcode and the source operands (excluding
/// the def in operand[0]). For commutative operations, source operands
/// are sorted to produce a canonical form.
fn make_expr_key(opcode: AArch64Opcode, operands: &[MachOperand]) -> ExprKey {
    // Source operands start at index 1 (operand[0] is the def).
    let mut canon_ops: Vec<CanonOperand> = operands[1..].iter().map(CanonOperand::from).collect();

    // For commutative operations, sort operands for canonical form.
    if is_commutative(opcode) && canon_ops.len() == 2 {
        canon_ops.sort_by(canon_operand_cmp);
    }

    ExprKey {
        opcode,
        operands: canon_ops,
    }
}

/// Returns true if the opcode is commutative (operand order doesn't matter).
fn is_commutative(opcode: AArch64Opcode) -> bool {
    use AArch64Opcode::*;
    matches!(
        opcode,
        AddRR | MulRR | AndRR | OrrRR | EorRR | FaddRR | FmulRR
    )
}

/// Comparison function for canonicalizing operand order.
fn canon_operand_cmp(a: &CanonOperand, b: &CanonOperand) -> std::cmp::Ordering {
    use std::cmp::Ordering;
    match (a, b) {
        (CanonOperand::VReg(a), CanonOperand::VReg(b)) => a.cmp(b),
        (CanonOperand::Imm(a), CanonOperand::Imm(b)) => a.cmp(b),
        (CanonOperand::FImm(a), CanonOperand::FImm(b)) => a.cmp(b),
        (CanonOperand::VReg(_), _) => Ordering::Less,
        (_, CanonOperand::VReg(_)) => Ordering::Greater,
        (CanonOperand::Imm(_), _) => Ordering::Less,
        (_, CanonOperand::Imm(_)) => Ordering::Greater,
        (CanonOperand::FImm(_), _) => Ordering::Less,
        (_, CanonOperand::FImm(_)) => Ordering::Greater,
        (CanonOperand::Other, CanonOperand::Other) => Ordering::Equal,
    }
}

/// Walk dominator tree in preorder (parent before children).
fn dom_preorder(dom: &DomTree, entry: BlockId) -> Vec<BlockId> {
    let mut order = Vec::new();
    let mut stack = vec![entry];

    while let Some(block) = stack.pop() {
        order.push(block);
        // Push children in reverse order so we visit them left-to-right.
        let children = dom.children(block);
        for &child in children.iter().rev() {
            stack.push(child);
        }
    }

    order
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pass_manager::MachinePass;
    use llvm2_ir::{
        AArch64Opcode, MachFunction, MachInst, MachOperand, ProofAnnotation, RegClass,
        Signature, VReg,
    };

    fn vreg(id: u32) -> MachOperand {
        MachOperand::VReg(VReg::new(id, RegClass::Gpr64))
    }

    fn imm(val: i64) -> MachOperand {
        MachOperand::Imm(val)
    }

    fn make_func_with_insts(insts: Vec<MachInst>) -> MachFunction {
        let mut func = MachFunction::new(
            "test_cse".to_string(),
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
    fn test_cse_identical_adds() {
        // v2 = add v0, v1
        // v3 = add v0, v1   → eliminated, v3 replaced with v2
        // v4 = sub v3, #1   → v4 = sub v2, #1
        // ret
        let a1 = MachInst::new(AArch64Opcode::AddRR, vec![vreg(2), vreg(0), vreg(1)]);
        let a2 = MachInst::new(AArch64Opcode::AddRR, vec![vreg(3), vreg(0), vreg(1)]);
        let sub = MachInst::new(AArch64Opcode::SubRI, vec![vreg(4), vreg(3), imm(1)]);
        let ret = MachInst::new(AArch64Opcode::Ret, vec![]);
        let mut func = make_func_with_insts(vec![a1, a2, sub, ret]);

        let mut cse = CommonSubexprElim;
        assert!(cse.run(&mut func));

        // a2 should be removed
        let block = func.block(func.entry);
        assert_eq!(block.insts.len(), 3); // a1, sub, ret

        // sub should now use v2 instead of v3
        let sub_inst = func.inst(block.insts[1]);
        assert_eq!(sub_inst.operands[1], vreg(2));
    }

    #[test]
    fn test_cse_commutative() {
        // v2 = add v0, v1
        // v3 = add v1, v0   → eliminated (commutative: v0+v1 == v1+v0)
        // ret
        let a1 = MachInst::new(AArch64Opcode::AddRR, vec![vreg(2), vreg(0), vreg(1)]);
        let a2 = MachInst::new(AArch64Opcode::AddRR, vec![vreg(3), vreg(1), vreg(0)]);
        let ret = MachInst::new(AArch64Opcode::Ret, vec![]);
        let mut func = make_func_with_insts(vec![a1, a2, ret]);

        let mut cse = CommonSubexprElim;
        assert!(cse.run(&mut func));

        let block = func.block(func.entry);
        assert_eq!(block.insts.len(), 2); // a1, ret
    }

    #[test]
    fn test_cse_no_cse_loads() {
        // v2 = ldr v0, #8
        // v3 = ldr v0, #8   → NOT eliminated (loads are not pure)
        // ret
        let l1 = MachInst::new(AArch64Opcode::LdrRI, vec![vreg(2), vreg(0), imm(8)]);
        let l2 = MachInst::new(AArch64Opcode::LdrRI, vec![vreg(3), vreg(0), imm(8)]);
        let ret = MachInst::new(AArch64Opcode::Ret, vec![]);
        let mut func = make_func_with_insts(vec![l1, l2, ret]);

        let mut cse = CommonSubexprElim;
        assert!(!cse.run(&mut func));

        let block = func.block(func.entry);
        assert_eq!(block.insts.len(), 3);
    }

    #[test]
    fn test_cse_different_operands() {
        // v2 = add v0, v1
        // v3 = add v0, v4   → different operands, not CSE'd
        // ret
        let a1 = MachInst::new(AArch64Opcode::AddRR, vec![vreg(2), vreg(0), vreg(1)]);
        let a2 = MachInst::new(AArch64Opcode::AddRR, vec![vreg(3), vreg(0), vreg(4)]);
        let ret = MachInst::new(AArch64Opcode::Ret, vec![]);
        let mut func = make_func_with_insts(vec![a1, a2, ret]);

        let mut cse = CommonSubexprElim;
        assert!(!cse.run(&mut func));
    }

    #[test]
    fn test_cse_different_opcodes() {
        // v2 = add v0, v1
        // v3 = sub v0, v1   → different opcode
        // ret
        let a1 = MachInst::new(AArch64Opcode::AddRR, vec![vreg(2), vreg(0), vreg(1)]);
        let s1 = MachInst::new(AArch64Opcode::SubRR, vec![vreg(3), vreg(0), vreg(1)]);
        let ret = MachInst::new(AArch64Opcode::Ret, vec![]);
        let mut func = make_func_with_insts(vec![a1, s1, ret]);

        let mut cse = CommonSubexprElim;
        assert!(!cse.run(&mut func));
    }

    #[test]
    fn test_cse_non_commutative() {
        // v2 = sub v0, v1
        // v3 = sub v1, v0   → NOT eliminated (sub is not commutative)
        // ret
        let s1 = MachInst::new(AArch64Opcode::SubRR, vec![vreg(2), vreg(0), vreg(1)]);
        let s2 = MachInst::new(AArch64Opcode::SubRR, vec![vreg(3), vreg(1), vreg(0)]);
        let ret = MachInst::new(AArch64Opcode::Ret, vec![]);
        let mut func = make_func_with_insts(vec![s1, s2, ret]);

        let mut cse = CommonSubexprElim;
        assert!(!cse.run(&mut func));
    }

    #[test]
    fn test_cse_dominator_based() {
        // Diamond CFG:
        //   bb0: v2 = add v0, v1
        //   bb1: v3 = add v0, v1  → CSE'd (bb0 dominates bb1)
        //   bb2: v4 = add v0, v1  → CSE'd (bb0 dominates bb2)
        //   bb3: ret
        let mut func = MachFunction::new(
            "test_cse_dom".to_string(),
            Signature::new(vec![], vec![]),
        );
        let bb0 = func.entry;
        let bb1 = func.create_block();
        let bb2 = func.create_block();
        let bb3 = func.create_block();

        // bb0: add + branch
        let a0 = func.push_inst(MachInst::new(
            AArch64Opcode::AddRR,
            vec![vreg(2), vreg(0), vreg(1)],
        ));
        func.append_inst(bb0, a0);
        let br0 = func.push_inst(MachInst::new(
            AArch64Opcode::BCond,
            vec![MachOperand::Block(bb1), MachOperand::Block(bb2)],
        ));
        func.append_inst(bb0, br0);

        // bb1: same add + branch
        let a1 = func.push_inst(MachInst::new(
            AArch64Opcode::AddRR,
            vec![vreg(3), vreg(0), vreg(1)],
        ));
        func.append_inst(bb1, a1);
        let br1 = func.push_inst(MachInst::new(
            AArch64Opcode::B,
            vec![MachOperand::Block(bb3)],
        ));
        func.append_inst(bb1, br1);

        // bb2: same add + branch
        let a2 = func.push_inst(MachInst::new(
            AArch64Opcode::AddRR,
            vec![vreg(4), vreg(0), vreg(1)],
        ));
        func.append_inst(bb2, a2);
        let br2 = func.push_inst(MachInst::new(
            AArch64Opcode::B,
            vec![MachOperand::Block(bb3)],
        ));
        func.append_inst(bb2, br2);

        // bb3: ret
        let ret = func.push_inst(MachInst::new(AArch64Opcode::Ret, vec![]));
        func.append_inst(bb3, ret);

        func.add_edge(bb0, bb1);
        func.add_edge(bb0, bb2);
        func.add_edge(bb1, bb3);
        func.add_edge(bb2, bb3);

        let mut cse = CommonSubexprElim;
        assert!(cse.run(&mut func));

        // bb1 and bb2 should have their adds removed.
        assert_eq!(func.block(bb1).insts.len(), 1); // just branch
        assert_eq!(func.block(bb2).insts.len(), 1); // just branch
    }

    #[test]
    fn test_cse_no_domination() {
        // Diamond: bb1 has add, bb2 has same add.
        // Neither bb1 nor bb2 dominates the other → no CSE between them.
        let mut func = MachFunction::new(
            "test_no_dom".to_string(),
            Signature::new(vec![], vec![]),
        );
        let bb0 = func.entry;
        let bb1 = func.create_block();
        let bb2 = func.create_block();
        let bb3 = func.create_block();

        let br0 = func.push_inst(MachInst::new(
            AArch64Opcode::BCond,
            vec![MachOperand::Block(bb1), MachOperand::Block(bb2)],
        ));
        func.append_inst(bb0, br0);

        let a1 = func.push_inst(MachInst::new(
            AArch64Opcode::AddRR,
            vec![vreg(2), vreg(0), vreg(1)],
        ));
        func.append_inst(bb1, a1);
        let br1 = func.push_inst(MachInst::new(
            AArch64Opcode::B,
            vec![MachOperand::Block(bb3)],
        ));
        func.append_inst(bb1, br1);

        let a2 = func.push_inst(MachInst::new(
            AArch64Opcode::AddRR,
            vec![vreg(3), vreg(0), vreg(1)],
        ));
        func.append_inst(bb2, a2);
        let br2 = func.push_inst(MachInst::new(
            AArch64Opcode::B,
            vec![MachOperand::Block(bb3)],
        ));
        func.append_inst(bb2, br2);

        let ret = func.push_inst(MachInst::new(AArch64Opcode::Ret, vec![]));
        func.append_inst(bb3, ret);

        func.add_edge(bb0, bb1);
        func.add_edge(bb0, bb2);
        func.add_edge(bb1, bb3);
        func.add_edge(bb2, bb3);

        let mut cse = CommonSubexprElim;
        assert!(!cse.run(&mut func));

        // Both adds should remain.
        assert_eq!(func.block(bb1).insts.len(), 2);
        assert_eq!(func.block(bb2).insts.len(), 2);
    }

    #[test]
    fn test_cse_idempotent() {
        let a1 = MachInst::new(AArch64Opcode::AddRR, vec![vreg(2), vreg(0), vreg(1)]);
        let a2 = MachInst::new(AArch64Opcode::AddRR, vec![vreg(3), vreg(0), vreg(1)]);
        let ret = MachInst::new(AArch64Opcode::Ret, vec![]);
        let mut func = make_func_with_insts(vec![a1, a2, ret]);

        let mut cse = CommonSubexprElim;
        assert!(cse.run(&mut func));
        // Second run should be a no-op.
        assert!(!cse.run(&mut func));
    }

    #[test]
    fn test_cse_mul_commutative() {
        // v2 = mul v0, v1
        // v3 = mul v1, v0   → CSE'd (mul is commutative)
        let m1 = MachInst::new(AArch64Opcode::MulRR, vec![vreg(2), vreg(0), vreg(1)]);
        let m2 = MachInst::new(AArch64Opcode::MulRR, vec![vreg(3), vreg(1), vreg(0)]);
        let ret = MachInst::new(AArch64Opcode::Ret, vec![]);
        let mut func = make_func_with_insts(vec![m1, m2, ret]);

        let mut cse = CommonSubexprElim;
        assert!(cse.run(&mut func));

        let block = func.block(func.entry);
        assert_eq!(block.insts.len(), 2); // m1, ret
    }

    #[test]
    fn test_cse_immediate_operands() {
        // v1 = add v0, #5
        // v2 = add v0, #5   → CSE'd
        // ret
        let a1 = MachInst::new(AArch64Opcode::AddRI, vec![vreg(1), vreg(0), imm(5)]);
        let a2 = MachInst::new(AArch64Opcode::AddRI, vec![vreg(2), vreg(0), imm(5)]);
        let ret = MachInst::new(AArch64Opcode::Ret, vec![]);
        let mut func = make_func_with_insts(vec![a1, a2, ret]);

        let mut cse = CommonSubexprElim;
        assert!(cse.run(&mut func));

        let block = func.block(func.entry);
        assert_eq!(block.insts.len(), 2); // a1, ret
    }

    // ---- Proof annotation preservation tests ----

    #[test]
    fn test_cse_preserves_surviving_proof() {
        // v2 = add v0, v1 [NoOverflow]
        // v3 = add v0, v1 (no proof) → eliminated, v3 → v2
        // Surviving instruction keeps its proof.
        let a1 = MachInst::new(AArch64Opcode::AddRR, vec![vreg(2), vreg(0), vreg(1)])
            .with_proof(ProofAnnotation::NoOverflow);
        let a2 = MachInst::new(AArch64Opcode::AddRR, vec![vreg(3), vreg(0), vreg(1)]);
        let ret = MachInst::new(AArch64Opcode::Ret, vec![]);
        let mut func = make_func_with_insts(vec![a1, a2, ret]);

        let mut cse = CommonSubexprElim;
        assert!(cse.run(&mut func));

        let block = func.block(func.entry);
        assert_eq!(block.insts.len(), 2); // a1, ret
        let surviving = func.inst(block.insts[0]);
        assert_eq!(surviving.proof, Some(ProofAnnotation::NoOverflow));
    }

    #[test]
    fn test_cse_merges_proof_from_eliminated() {
        // v2 = add v0, v1 (no proof)
        // v3 = add v0, v1 [InBounds] → eliminated, proof merged onto v2
        // Surviving instruction gets the eliminated instruction's proof.
        let a1 = MachInst::new(AArch64Opcode::AddRR, vec![vreg(2), vreg(0), vreg(1)]);
        let a2 = MachInst::new(AArch64Opcode::AddRR, vec![vreg(3), vreg(0), vreg(1)])
            .with_proof(ProofAnnotation::InBounds);
        let ret = MachInst::new(AArch64Opcode::Ret, vec![]);
        let mut func = make_func_with_insts(vec![a1, a2, ret]);

        let mut cse = CommonSubexprElim;
        assert!(cse.run(&mut func));

        let block = func.block(func.entry);
        assert_eq!(block.insts.len(), 2); // a1, ret
        let surviving = func.inst(block.insts[0]);
        assert_eq!(surviving.proof, Some(ProofAnnotation::InBounds));
    }

    #[test]
    fn test_cse_merges_same_proof() {
        // Both instructions have the same proof → surviving keeps it.
        let a1 = MachInst::new(AArch64Opcode::AddRR, vec![vreg(2), vreg(0), vreg(1)])
            .with_proof(ProofAnnotation::NotNull);
        let a2 = MachInst::new(AArch64Opcode::AddRR, vec![vreg(3), vreg(0), vreg(1)])
            .with_proof(ProofAnnotation::NotNull);
        let ret = MachInst::new(AArch64Opcode::Ret, vec![]);
        let mut func = make_func_with_insts(vec![a1, a2, ret]);

        let mut cse = CommonSubexprElim;
        assert!(cse.run(&mut func));

        let block = func.block(func.entry);
        assert_eq!(block.insts.len(), 2);
        let surviving = func.inst(block.insts[0]);
        assert_eq!(surviving.proof, Some(ProofAnnotation::NotNull));
    }

    #[test]
    fn test_cse_drops_conflicting_proofs() {
        // Different proofs → conservative merge returns None.
        let a1 = MachInst::new(AArch64Opcode::AddRR, vec![vreg(2), vreg(0), vreg(1)])
            .with_proof(ProofAnnotation::NoOverflow);
        let a2 = MachInst::new(AArch64Opcode::AddRR, vec![vreg(3), vreg(0), vreg(1)])
            .with_proof(ProofAnnotation::InBounds);
        let ret = MachInst::new(AArch64Opcode::Ret, vec![]);
        let mut func = make_func_with_insts(vec![a1, a2, ret]);

        let mut cse = CommonSubexprElim;
        assert!(cse.run(&mut func));

        let block = func.block(func.entry);
        assert_eq!(block.insts.len(), 2);
        let surviving = func.inst(block.insts[0]);
        // Different proofs → conservatively dropped
        assert!(surviving.proof.is_none());
    }
}
