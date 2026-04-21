// llvm2-opt - Global Value Numbering
//
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Copyright 2026 Andrew Yates | License: Apache-2.0

//! Global Value Numbering (GVN) for machine-level IR.
//!
//! GVN assigns a unique *value number* to each computed expression. Two
//! instructions that compute the same value (same opcode and same
//! value-numbered operands) receive the same value number. When a
//! dominating instruction already defines a value with the same number,
//! the later instruction is redundant and can be eliminated.
//!
//! # Differences from CSE
//!
//! CSE uses syntactic expression keys (opcode + literal operand lists).
//! GVN is strictly more powerful because it uses *value numbers*, which
//! enables transitive reasoning: if `v2 = add v0, v1` and later
//! `v3 = mov v2`, then any instruction using `v3` sees the same value
//! number as `v2`, enabling further elimination.
//!
//! # Memory Load Value Numbering
//!
//! Pure loads from the same address (same opcode, same base value number,
//! same offset) receive the same value number. Stores and calls act as
//! memory barriers that invalidate all load value numbers, because they
//! may modify arbitrary memory locations.
//!
//! # Scoped Value Table
//!
//! The value table uses scope push/pop aligned with the dominator tree
//! walk. When entering a dominator-tree subtree, a scope is pushed;
//! when leaving, entries added in that subtree are removed. This ensures
//! that value numbers from non-dominating blocks are never visible.
//!
//! # Algorithm
//!
//! 1. Compute dominator tree.
//! 2. Walk blocks in dominator-tree preorder (ensures dominators are
//!    processed before dominated blocks).
//! 3. For each instruction:
//!    a. If store or call: invalidate all load value numbers.
//!    b. If load that produces a value: look up in load table; if found,
//!       mark for elimination; otherwise assign a fresh value number.
//!    c. If pure and produces a value: compute value-numbered key
//!       (opcode + value numbers of source operands); look up in value
//!       table; if found, mark for elimination; otherwise assign fresh
//!       value number.
//! 4. Apply replacements: rewrite vreg uses.
//! 5. Remove dead instructions.
//!
//! # Commutative Instructions
//!
//! For commutative operations (add, mul, and, or, xor, fadd, fmul),
//! the value-numbered operands are sorted before lookup. This allows
//! `add v1, v0` to match `add v0, v1`.
//!
//! Reference: LLVM `GVN.cpp`, Briggs & Cooper "Value Numbering"

use std::collections::{HashMap, HashSet};

use llvm2_ir::{AArch64Opcode, BlockId, InstId, MachFunction, MachOperand, ProofAnnotation, VReg};

use crate::dom::DomTree;
use crate::effects::{has_tied_def_use, opcode_effect, produces_value, reads_flags, MemoryEffect};
use crate::pass_manager::{AnalysisCache, MachinePass};

/// Global Value Numbering pass.
pub struct GlobalValueNumbering;

impl MachinePass for GlobalValueNumbering {
    fn name(&self) -> &str {
        "gvn"
    }

    fn run(&mut self, func: &mut MachFunction) -> bool {
        let dom = DomTree::compute(func);
        run_gvn(func, &dom)
    }

    fn run_with_analyses(
        &mut self,
        func: &mut MachFunction,
        analyses: &mut AnalysisCache,
    ) -> bool {
        let dom = analyses.domtree(func).clone();
        run_gvn(func, &dom)
    }
}

// ---------------------------------------------------------------------------
// Value number types
// ---------------------------------------------------------------------------

/// A value number. Two expressions with the same value number compute
/// the same result.
type ValNum = u32;

/// Counter for allocating fresh value numbers.
struct ValNumAllocator {
    next: ValNum,
}

impl ValNumAllocator {
    fn new() -> Self {
        Self { next: 0 }
    }

    fn fresh(&mut self) -> ValNum {
        let vn = self.next;
        self.next += 1;
        vn
    }
}

// ---------------------------------------------------------------------------
// Expression key (opcode + value-numbered operands)
// ---------------------------------------------------------------------------

/// A value-numbered expression key.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct VNExprKey {
    opcode: AArch64Opcode,
    /// Value numbers of source operands (value-numbered, not raw vreg ids).
    operand_vns: Vec<ValNum>,
}

/// A load expression key (opcode + base value number + offset).
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct LoadKey {
    opcode: AArch64Opcode,
    base_vn: ValNum,
    offset: i64,
}

// ---------------------------------------------------------------------------
// Leader: the canonical instruction for a value number
// ---------------------------------------------------------------------------

/// Information about the first (leader) instruction for a value number.
#[derive(Debug, Clone)]
struct Leader {
    inst_id: InstId,
    def_vreg: VReg,
}

// ---------------------------------------------------------------------------
// Scoped value table
// ---------------------------------------------------------------------------

/// A scoped value table that supports push/pop aligned with dominator
/// tree traversal. Entries added in a child scope are removed when that
/// scope is popped.
struct ScopedValueTable {
    /// VReg id -> value number.
    vreg_to_vn: HashMap<u32, ValNum>,
    /// Expression key -> (value number, leader).
    expr_table: HashMap<VNExprKey, (ValNum, Leader)>,
    /// Load key -> (value number, leader).
    load_table: HashMap<LoadKey, (ValNum, Leader)>,
    /// Scope stack: each entry records keys added in that scope so
    /// they can be removed on pop.
    scope_stack: Vec<ScopeFrame>,
}

/// Records what was added during a single scope (one dominator-tree node).
#[derive(Default)]
struct ScopeFrame {
    added_vregs: Vec<u32>,
    added_exprs: Vec<VNExprKey>,
    added_loads: Vec<LoadKey>,
    /// Whether load table was cleared in this scope (stores/calls).
    /// If so, we need to restore the previous load table on pop.
    cleared_loads: Option<HashMap<LoadKey, (ValNum, Leader)>>,
}

impl ScopedValueTable {
    fn new() -> Self {
        Self {
            vreg_to_vn: HashMap::new(),
            expr_table: HashMap::new(),
            load_table: HashMap::new(),
            scope_stack: Vec::new(),
        }
    }

    fn push_scope(&mut self) {
        self.scope_stack.push(ScopeFrame::default());
    }

    fn pop_scope(&mut self) {
        let frame = self.scope_stack.pop().expect("scope underflow");
        // Remove entries added in this scope.
        for vreg_id in &frame.added_vregs {
            self.vreg_to_vn.remove(vreg_id);
        }
        for key in &frame.added_exprs {
            self.expr_table.remove(key);
        }
        for key in &frame.added_loads {
            self.load_table.remove(key);
        }
        // Restore load table if it was cleared.
        if let Some(saved) = frame.cleared_loads {
            self.load_table = saved;
        }
    }

    /// Assign a value number to a vreg.
    fn set_vreg_vn(&mut self, vreg_id: u32, vn: ValNum) {
        self.vreg_to_vn.insert(vreg_id, vn);
        if let Some(frame) = self.scope_stack.last_mut() {
            frame.added_vregs.push(vreg_id);
        }
    }

    /// Look up the value number for a vreg.
    fn get_vreg_vn(&self, vreg_id: u32) -> Option<ValNum> {
        self.vreg_to_vn.get(&vreg_id).copied()
    }

    /// Look up an expression in the value table.
    fn lookup_expr(&self, key: &VNExprKey) -> Option<&(ValNum, Leader)> {
        self.expr_table.get(key)
    }

    /// Insert an expression into the value table.
    fn insert_expr(&mut self, key: VNExprKey, vn: ValNum, leader: Leader) {
        if let Some(frame) = self.scope_stack.last_mut() {
            frame.added_exprs.push(key.clone());
        }
        self.expr_table.insert(key, (vn, leader));
    }

    /// Look up a load in the load table.
    fn lookup_load(&self, key: &LoadKey) -> Option<&(ValNum, Leader)> {
        self.load_table.get(key)
    }

    /// Insert a load into the load table.
    fn insert_load(&mut self, key: LoadKey, vn: ValNum, leader: Leader) {
        if let Some(frame) = self.scope_stack.last_mut() {
            frame.added_loads.push(key.clone());
        }
        self.load_table.insert(key, (vn, leader));
    }

    /// Invalidate all load value numbers (called on store/call).
    /// Saves the current load table so it can be restored on scope pop.
    fn kill_loads(&mut self) {
        if let Some(frame) = self.scope_stack.last_mut()
            && frame.cleared_loads.is_none() {
                frame.cleared_loads = Some(self.load_table.clone());
            }
        self.load_table.clear();
    }
}

// ---------------------------------------------------------------------------
// Commutative opcode detection
// ---------------------------------------------------------------------------

/// Returns true if the opcode is commutative (operand order doesn't matter).
///
/// Delegates to the generic [`AArch64Opcode::is_commutative`] method for
/// multi-target compatibility.
fn is_commutative(opcode: AArch64Opcode) -> bool {
    opcode.is_commutative()
}

// ---------------------------------------------------------------------------
// Value number computation for an operand
// ---------------------------------------------------------------------------

/// Get the value number for a source operand. VRegs use the vreg map;
/// immediates get a deterministic value number derived from the value
/// (we use a separate per-immediate allocation to avoid collisions).
///
/// VRegs that have not yet been assigned a value number (e.g., function
/// parameters, vregs defined before the analyzed region) get a fresh
/// value number assigned on demand. This ensures every reachable vreg
/// has a stable value number.
fn operand_vn(
    op: &MachOperand,
    table: &mut ScopedValueTable,
    imm_vns: &mut HashMap<i64, ValNum>,
    fimm_vns: &mut HashMap<u64, ValNum>,
    alloc: &mut ValNumAllocator,
) -> Option<ValNum> {
    match op {
        MachOperand::VReg(v) => {
            if let Some(vn) = table.get_vreg_vn(v.id) {
                Some(vn)
            } else {
                // First time seeing this vreg as a source — assign fresh VN.
                let vn = alloc.fresh();
                table.set_vreg_vn(v.id, vn);
                Some(vn)
            }
        }
        MachOperand::Imm(i) => {
            let vn = *imm_vns.entry(*i).or_insert_with(|| alloc.fresh());
            Some(vn)
        }
        MachOperand::FImm(f) => {
            let bits = f.to_bits();
            let vn = *fimm_vns.entry(bits).or_insert_with(|| alloc.fresh());
            Some(vn)
        }
        // Non-hashable operands (blocks, pregs, etc.) — skip GVN.
        _ => None,
    }
}

// ---------------------------------------------------------------------------
// Core GVN algorithm
// ---------------------------------------------------------------------------

/// Run GVN on the function, returning true if any changes were made.
fn run_gvn(func: &mut MachFunction, dom: &DomTree) -> bool {
    let mut alloc = ValNumAllocator::new();
    let mut table = ScopedValueTable::new();
    let mut imm_vns: HashMap<i64, ValNum> = HashMap::new();
    let mut fimm_vns: HashMap<u64, ValNum> = HashMap::new();

    // Replacement map: vreg_id of eliminated def -> vreg of leader def.
    let mut replacements: HashMap<u32, VReg> = HashMap::new();

    // Instructions to remove.
    let mut dead_insts: Vec<InstId> = Vec::new();

    // Proof annotations to merge onto surviving instructions.
    let mut proof_merges: Vec<(InstId, Option<ProofAnnotation>)> = Vec::new();

    // Pre-assign value numbers to function parameters (vregs with low ids
    // that appear as uses but never as defs within the function). We handle
    // this lazily: any vreg without a value number gets a fresh one when
    // first encountered as a source operand (see operand_vn + fallback below).

    // Walk dominator tree in preorder using a recursive-style iterative
    // traversal that pushes/pops scopes.
    dom_walk_gvn(
        func,
        dom,
        func.entry,
        &mut table,
        &mut alloc,
        &mut imm_vns,
        &mut fimm_vns,
        &mut replacements,
        &mut dead_insts,
        &mut proof_merges,
    );

    if replacements.is_empty() {
        return false;
    }

    // Apply proof merges.
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
    let dead_set: HashSet<InstId> = dead_insts.into_iter().collect();
    for block_id in func.block_order.clone() {
        let block = func.block_mut(block_id);
        block.insts.retain(|id| !dead_set.contains(id));
    }

    true
}

/// Recursive dominator-tree walk with scope push/pop.
#[allow(clippy::too_many_arguments)]
fn dom_walk_gvn(
    func: &MachFunction,
    dom: &DomTree,
    block_id: BlockId,
    table: &mut ScopedValueTable,
    alloc: &mut ValNumAllocator,
    imm_vns: &mut HashMap<i64, ValNum>,
    fimm_vns: &mut HashMap<u64, ValNum>,
    replacements: &mut HashMap<u32, VReg>,
    dead_insts: &mut Vec<InstId>,
    proof_merges: &mut Vec<(InstId, Option<ProofAnnotation>)>,
) {
    table.push_scope();

    let block = func.block(block_id);
    for &inst_id in &block.insts {
        let inst = func.inst(inst_id);
        let effect = opcode_effect(inst.opcode);

        // Stores and calls kill all load value numbers.
        if effect.writes_memory() || effect == MemoryEffect::Call {
            table.kill_loads();
        }

        // Only process instructions that produce a value.
        if !produces_value(inst.opcode) {
            continue;
        }

        // Skip instructions that read implicit NZCV flags. These
        // instructions depend on flag state set by a prior CMP/TST,
        // which is not captured in their explicit operands. Two CSet
        // instructions with the same condition code produce different
        // values if their preceding comparisons set different flags.
        if reads_flags(inst.opcode) {
            // Assign a fresh value number so downstream uses get
            // unique numbering, but do NOT insert into the expression
            // table (no elimination possible).
            if let Some(MachOperand::VReg(v)) = inst.operands.first() {
                let vn = alloc.fresh();
                table.set_vreg_vn(v.id, vn);
            }
            continue;
        }

        // Get the def vreg (operand[0]).
        let def_vreg = match &inst.operands.first() {
            Some(MachOperand::VReg(v)) => *v,
            _ => continue,
        };

        // Handle loads: value number via load table.
        if effect == MemoryEffect::Load {
            if let Some(load_key) = make_load_key(inst, table, imm_vns, fimm_vns, alloc) {
                if let Some((existing_vn, leader)) = table.lookup_load(&load_key) {
                    // Found a matching load — eliminate this one.
                    let vn = *existing_vn;
                    let leader_clone = leader.clone();
                    table.set_vreg_vn(def_vreg.id, vn);
                    replacements.insert(def_vreg.id, leader_clone.def_vreg);
                    dead_insts.push(inst_id);
                    proof_merges.push((leader_clone.inst_id, inst.proof));
                    continue;
                }
                // New load — assign fresh value number.
                let vn = alloc.fresh();
                table.set_vreg_vn(def_vreg.id, vn);
                table.insert_load(
                    load_key,
                    vn,
                    Leader {
                        inst_id,
                        def_vreg,
                    },
                );
                continue;
            }
            // Could not form a load key (e.g., non-standard operands).
            // Assign a fresh value number and move on.
            let vn = alloc.fresh();
            table.set_vreg_vn(def_vreg.id, vn);
            continue;
        }

        // Handle pure instructions.
        if effect == MemoryEffect::Pure
            && let Some(expr_key) = make_expr_key(inst, table, imm_vns, fimm_vns, alloc) {
                if let Some((existing_vn, leader)) = table.lookup_expr(&expr_key) {
                    // Found a matching expression — eliminate this one.
                    let vn = *existing_vn;
                    let leader_clone = leader.clone();
                    table.set_vreg_vn(def_vreg.id, vn);
                    replacements.insert(def_vreg.id, leader_clone.def_vreg);
                    dead_insts.push(inst_id);
                    proof_merges.push((leader_clone.inst_id, inst.proof));
                    continue;
                }
                // New expression — assign fresh value number.
                let vn = alloc.fresh();
                table.set_vreg_vn(def_vreg.id, vn);
                table.insert_expr(
                    expr_key,
                    vn,
                    Leader {
                        inst_id,
                        def_vreg,
                    },
                );
                continue;
            }

        // Fallback: non-matchable instruction, assign fresh value number.
        let vn = alloc.fresh();
        table.set_vreg_vn(def_vreg.id, vn);
    }

    // Recurse into dominator-tree children.
    for &child in dom.children(block_id) {
        dom_walk_gvn(
            func, dom, child, table, alloc, imm_vns, fimm_vns, replacements, dead_insts,
            proof_merges,
        );
    }

    table.pop_scope();
}

/// Build a value-numbered expression key for a pure instruction.
///
/// Returns `None` if any source operand cannot be value-numbered
/// (e.g., block operands, physical registers) OR if the instruction
/// has a tied def-use operand whose prior value is an implicit input
/// (e.g., MOVK).
fn make_expr_key(
    inst: &llvm2_ir::MachInst,
    table: &mut ScopedValueTable,
    imm_vns: &mut HashMap<i64, ValNum>,
    fimm_vns: &mut HashMap<u64, ValNum>,
    alloc: &mut ValNumAllocator,
) -> Option<VNExprKey> {
    // Instructions with tied def-use (e.g., MOVK) cannot be value-numbered
    // using just (opcode, source operands): the destination register's
    // prior value is also an input. Two MOVKs with identical (imm, shift)
    // but different dest registers compute DIFFERENT values.
    //
    // We could value-number them by including the def's prior VN in the
    // key, but that requires tracking pre-def VNs which the current
    // scheme does not. Conservatively skip them.
    if has_tied_def_use(inst.opcode) {
        return None;
    }

    // Source operands start at index 1 (operand[0] is the def).
    let mut op_vns = Vec::with_capacity(inst.operands.len() - 1);
    for op in &inst.operands[1..] {
        match operand_vn(op, table, imm_vns, fimm_vns, alloc) {
            Some(vn) => op_vns.push(vn),
            None => return None,
        }
    }

    // Canonicalize commutative operations.
    if is_commutative(inst.opcode) && op_vns.len() == 2
        && op_vns[0] > op_vns[1] {
            op_vns.swap(0, 1);
        }

    Some(VNExprKey {
        opcode: inst.opcode,
        operand_vns: op_vns,
    })
}

/// Build a load key for a load instruction.
///
/// Load instructions typically have the form: `def, base, offset`.
/// Returns `None` if the operands don't match this pattern.
fn make_load_key(
    inst: &llvm2_ir::MachInst,
    table: &mut ScopedValueTable,
    imm_vns: &mut HashMap<i64, ValNum>,
    fimm_vns: &mut HashMap<u64, ValNum>,
    alloc: &mut ValNumAllocator,
) -> Option<LoadKey> {
    // Expect at least: def, base, offset
    if inst.operands.len() < 3 {
        return None;
    }

    // Base register (operand[1]) must be a VReg.
    let base_vn = operand_vn(&inst.operands[1], table, imm_vns, fimm_vns, alloc)?;

    // Offset (operand[2]) must be an immediate.
    let offset = match &inst.operands[2] {
        MachOperand::Imm(i) => *i,
        _ => return None,
    };

    Some(LoadKey {
        opcode: inst.opcode,
        base_vn,
        offset,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pass_manager::MachinePass;
    use llvm2_ir::{
        AArch64Opcode, MachFunction, MachInst, MachOperand, ProofAnnotation, RegClass, Signature,
        VReg,
    };

    fn vreg(id: u32) -> MachOperand {
        MachOperand::VReg(VReg::new(id, RegClass::Gpr64))
    }

    fn imm(val: i64) -> MachOperand {
        MachOperand::Imm(val)
    }

    fn make_func_with_insts(insts: Vec<MachInst>) -> MachFunction {
        let mut func = MachFunction::new(
            "test_gvn".to_string(),
            Signature::new(vec![], vec![]),
        );
        let block = func.entry;
        for inst in insts {
            let id = func.push_inst(inst);
            func.append_inst(block, id);
        }
        func
    }

    // ---- Basic value numbering tests ----

    #[test]
    fn test_gvn_identical_adds() {
        // v2 = add v0, v1
        // v3 = add v0, v1   -> eliminated, v3 replaced with v2
        // v4 = sub v3, #1   -> v4 = sub v2, #1
        // ret
        let a1 = MachInst::new(AArch64Opcode::AddRR, vec![vreg(2), vreg(0), vreg(1)]);
        let a2 = MachInst::new(AArch64Opcode::AddRR, vec![vreg(3), vreg(0), vreg(1)]);
        let sub = MachInst::new(AArch64Opcode::SubRI, vec![vreg(4), vreg(3), imm(1)]);
        let ret = MachInst::new(AArch64Opcode::Ret, vec![]);
        let mut func = make_func_with_insts(vec![a1, a2, sub, ret]);

        let mut gvn = GlobalValueNumbering;
        assert!(gvn.run(&mut func));

        // a2 should be removed
        let block = func.block(func.entry);
        assert_eq!(block.insts.len(), 3); // a1, sub, ret

        // sub should now use v2 instead of v3
        let sub_inst = func.inst(block.insts[1]);
        assert_eq!(sub_inst.operands[1], vreg(2));
    }

    #[test]
    fn test_gvn_commutative() {
        // v2 = add v0, v1
        // v3 = add v1, v0   -> eliminated (commutative)
        // ret
        let a1 = MachInst::new(AArch64Opcode::AddRR, vec![vreg(2), vreg(0), vreg(1)]);
        let a2 = MachInst::new(AArch64Opcode::AddRR, vec![vreg(3), vreg(1), vreg(0)]);
        let ret = MachInst::new(AArch64Opcode::Ret, vec![]);
        let mut func = make_func_with_insts(vec![a1, a2, ret]);

        let mut gvn = GlobalValueNumbering;
        assert!(gvn.run(&mut func));

        let block = func.block(func.entry);
        assert_eq!(block.insts.len(), 2); // a1, ret
    }

    #[test]
    fn test_gvn_non_commutative() {
        // v2 = sub v0, v1
        // v3 = sub v1, v0   -> NOT eliminated (sub is not commutative)
        // ret
        let s1 = MachInst::new(AArch64Opcode::SubRR, vec![vreg(2), vreg(0), vreg(1)]);
        let s2 = MachInst::new(AArch64Opcode::SubRR, vec![vreg(3), vreg(1), vreg(0)]);
        let ret = MachInst::new(AArch64Opcode::Ret, vec![]);
        let mut func = make_func_with_insts(vec![s1, s2, ret]);

        let mut gvn = GlobalValueNumbering;
        assert!(!gvn.run(&mut func));
    }

    #[test]
    fn test_gvn_different_operands() {
        // v2 = add v0, v1
        // v3 = add v0, v4   -> different operands, not eliminated
        // ret
        let a1 = MachInst::new(AArch64Opcode::AddRR, vec![vreg(2), vreg(0), vreg(1)]);
        let a2 = MachInst::new(AArch64Opcode::AddRR, vec![vreg(3), vreg(0), vreg(4)]);
        let ret = MachInst::new(AArch64Opcode::Ret, vec![]);
        let mut func = make_func_with_insts(vec![a1, a2, ret]);

        let mut gvn = GlobalValueNumbering;
        assert!(!gvn.run(&mut func));
    }

    #[test]
    fn test_gvn_different_opcodes() {
        // v2 = add v0, v1
        // v3 = sub v0, v1   -> different opcode
        // ret
        let a1 = MachInst::new(AArch64Opcode::AddRR, vec![vreg(2), vreg(0), vreg(1)]);
        let s1 = MachInst::new(AArch64Opcode::SubRR, vec![vreg(3), vreg(0), vreg(1)]);
        let ret = MachInst::new(AArch64Opcode::Ret, vec![]);
        let mut func = make_func_with_insts(vec![a1, s1, ret]);

        let mut gvn = GlobalValueNumbering;
        assert!(!gvn.run(&mut func));
    }

    #[test]
    fn test_gvn_immediate_operands() {
        // v1 = add v0, #5
        // v2 = add v0, #5   -> eliminated
        // ret
        let a1 = MachInst::new(AArch64Opcode::AddRI, vec![vreg(1), vreg(0), imm(5)]);
        let a2 = MachInst::new(AArch64Opcode::AddRI, vec![vreg(2), vreg(0), imm(5)]);
        let ret = MachInst::new(AArch64Opcode::Ret, vec![]);
        let mut func = make_func_with_insts(vec![a1, a2, ret]);

        let mut gvn = GlobalValueNumbering;
        assert!(gvn.run(&mut func));

        let block = func.block(func.entry);
        assert_eq!(block.insts.len(), 2); // a1, ret
    }

    #[test]
    fn test_gvn_mul_commutative() {
        // v2 = mul v0, v1
        // v3 = mul v1, v0   -> eliminated (commutative)
        let m1 = MachInst::new(AArch64Opcode::MulRR, vec![vreg(2), vreg(0), vreg(1)]);
        let m2 = MachInst::new(AArch64Opcode::MulRR, vec![vreg(3), vreg(1), vreg(0)]);
        let ret = MachInst::new(AArch64Opcode::Ret, vec![]);
        let mut func = make_func_with_insts(vec![m1, m2, ret]);

        let mut gvn = GlobalValueNumbering;
        assert!(gvn.run(&mut func));

        let block = func.block(func.entry);
        assert_eq!(block.insts.len(), 2); // m1, ret
    }

    // ---- Dominator-based tests ----

    #[test]
    fn test_gvn_dominator_based() {
        // Diamond CFG:
        //   bb0: v2 = add v0, v1
        //   bb1: v3 = add v0, v1  -> eliminated (bb0 dominates bb1)
        //   bb2: v4 = add v0, v1  -> eliminated (bb0 dominates bb2)
        //   bb3: ret
        let mut func = MachFunction::new(
            "test_gvn_dom".to_string(),
            Signature::new(vec![], vec![]),
        );
        let bb0 = func.entry;
        let bb1 = func.create_block();
        let bb2 = func.create_block();
        let bb3 = func.create_block();

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

        let ret = func.push_inst(MachInst::new(AArch64Opcode::Ret, vec![]));
        func.append_inst(bb3, ret);

        func.add_edge(bb0, bb1);
        func.add_edge(bb0, bb2);
        func.add_edge(bb1, bb3);
        func.add_edge(bb2, bb3);

        let mut gvn = GlobalValueNumbering;
        assert!(gvn.run(&mut func));

        // bb1 and bb2 should have their adds removed.
        assert_eq!(func.block(bb1).insts.len(), 1); // just branch
        assert_eq!(func.block(bb2).insts.len(), 1); // just branch
    }

    #[test]
    fn test_gvn_no_domination() {
        // Diamond: bb1 has add, bb2 has same add.
        // Neither bb1 nor bb2 dominates the other -> no elimination.
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

        let mut gvn = GlobalValueNumbering;
        assert!(!gvn.run(&mut func));

        // Both adds should remain.
        assert_eq!(func.block(bb1).insts.len(), 2);
        assert_eq!(func.block(bb2).insts.len(), 2);
    }

    // ---- Load value numbering tests ----

    #[test]
    fn test_gvn_load_value_numbering() {
        // v2 = ldr v0, #8
        // v3 = ldr v0, #8   -> eliminated (same load, no intervening store)
        // ret
        let l1 = MachInst::new(AArch64Opcode::LdrRI, vec![vreg(2), vreg(0), imm(8)]);
        let l2 = MachInst::new(AArch64Opcode::LdrRI, vec![vreg(3), vreg(0), imm(8)]);
        let ret = MachInst::new(AArch64Opcode::Ret, vec![]);
        let mut func = make_func_with_insts(vec![l1, l2, ret]);

        let mut gvn = GlobalValueNumbering;
        assert!(gvn.run(&mut func));

        let block = func.block(func.entry);
        assert_eq!(block.insts.len(), 2); // l1, ret
    }

    #[test]
    fn test_gvn_load_killed_by_store() {
        // v2 = ldr v0, #8
        // str v5, v0, #8   (store to same address)
        // v3 = ldr v0, #8   -> NOT eliminated (store kills loads)
        // ret
        let l1 = MachInst::new(AArch64Opcode::LdrRI, vec![vreg(2), vreg(0), imm(8)]);
        let st = MachInst::new(AArch64Opcode::StrRI, vec![vreg(5), vreg(0), imm(8)]);
        let l2 = MachInst::new(AArch64Opcode::LdrRI, vec![vreg(3), vreg(0), imm(8)]);
        let ret = MachInst::new(AArch64Opcode::Ret, vec![]);
        let mut func = make_func_with_insts(vec![l1, st, l2, ret]);

        let mut gvn = GlobalValueNumbering;
        assert!(!gvn.run(&mut func));

        let block = func.block(func.entry);
        assert_eq!(block.insts.len(), 4); // all remain
    }

    #[test]
    fn test_gvn_load_killed_by_call() {
        // v2 = ldr v0, #8
        // bl (call)
        // v3 = ldr v0, #8   -> NOT eliminated (call kills loads)
        // ret
        let l1 = MachInst::new(AArch64Opcode::LdrRI, vec![vreg(2), vreg(0), imm(8)]);
        let call = MachInst::new(AArch64Opcode::Bl, vec![]);
        let l2 = MachInst::new(AArch64Opcode::LdrRI, vec![vreg(3), vreg(0), imm(8)]);
        let ret = MachInst::new(AArch64Opcode::Ret, vec![]);
        let mut func = make_func_with_insts(vec![l1, call, l2, ret]);

        let mut gvn = GlobalValueNumbering;
        assert!(!gvn.run(&mut func));

        let block = func.block(func.entry);
        assert_eq!(block.insts.len(), 4);
    }

    #[test]
    fn test_gvn_loads_different_addresses() {
        // v2 = ldr v0, #8
        // v3 = ldr v0, #16   -> different offset, not eliminated
        // ret
        let l1 = MachInst::new(AArch64Opcode::LdrRI, vec![vreg(2), vreg(0), imm(8)]);
        let l2 = MachInst::new(AArch64Opcode::LdrRI, vec![vreg(3), vreg(0), imm(16)]);
        let ret = MachInst::new(AArch64Opcode::Ret, vec![]);
        let mut func = make_func_with_insts(vec![l1, l2, ret]);

        let mut gvn = GlobalValueNumbering;
        assert!(!gvn.run(&mut func));

        let block = func.block(func.entry);
        assert_eq!(block.insts.len(), 3);
    }

    #[test]
    fn test_gvn_load_store_different_addr_conservative() {
        // v2 = ldr v0, #8
        // str v5, v1, #16   (store to DIFFERENT address)
        // v3 = ldr v0, #8   -> still NOT eliminated (conservative: any store kills all loads)
        // ret
        let l1 = MachInst::new(AArch64Opcode::LdrRI, vec![vreg(2), vreg(0), imm(8)]);
        let st = MachInst::new(AArch64Opcode::StrRI, vec![vreg(5), vreg(1), imm(16)]);
        let l2 = MachInst::new(AArch64Opcode::LdrRI, vec![vreg(3), vreg(0), imm(8)]);
        let ret = MachInst::new(AArch64Opcode::Ret, vec![]);
        let mut func = make_func_with_insts(vec![l1, st, l2, ret]);

        let mut gvn = GlobalValueNumbering;
        assert!(!gvn.run(&mut func));

        let block = func.block(func.entry);
        assert_eq!(block.insts.len(), 4);
    }

    // ---- Idempotency test ----

    #[test]
    fn test_gvn_idempotent() {
        let a1 = MachInst::new(AArch64Opcode::AddRR, vec![vreg(2), vreg(0), vreg(1)]);
        let a2 = MachInst::new(AArch64Opcode::AddRR, vec![vreg(3), vreg(0), vreg(1)]);
        let ret = MachInst::new(AArch64Opcode::Ret, vec![]);
        let mut func = make_func_with_insts(vec![a1, a2, ret]);

        let mut gvn = GlobalValueNumbering;
        assert!(gvn.run(&mut func));
        // Second run should be a no-op.
        assert!(!gvn.run(&mut func));
    }

    // ---- Empty function test ----

    #[test]
    fn test_gvn_empty_function() {
        let ret = MachInst::new(AArch64Opcode::Ret, vec![]);
        let mut func = make_func_with_insts(vec![ret]);

        let mut gvn = GlobalValueNumbering;
        assert!(!gvn.run(&mut func));
    }

    // ---- Transitive value numbering test ----

    #[test]
    fn test_gvn_transitive() {
        // v2 = add v0, v1
        // v3 = mov v2         (copy: v3 gets same value number as v2)
        // v4 = add v0, v1     -> eliminated, v4 replaced with v2
        // v5 = sub v4, #1     -> v5 = sub v2, #1
        // ret
        let a1 = MachInst::new(AArch64Opcode::AddRR, vec![vreg(2), vreg(0), vreg(1)]);
        let mov = MachInst::new(AArch64Opcode::MovR, vec![vreg(3), vreg(2)]);
        let a2 = MachInst::new(AArch64Opcode::AddRR, vec![vreg(4), vreg(0), vreg(1)]);
        let sub = MachInst::new(AArch64Opcode::SubRI, vec![vreg(5), vreg(4), imm(1)]);
        let ret = MachInst::new(AArch64Opcode::Ret, vec![]);
        let mut func = make_func_with_insts(vec![a1, mov, a2, sub, ret]);

        let mut gvn = GlobalValueNumbering;
        assert!(gvn.run(&mut func));

        // a2 should be eliminated
        let block = func.block(func.entry);
        assert_eq!(block.insts.len(), 4); // a1, mov, sub, ret

        // sub should use v2 instead of v4
        let sub_inst = func.inst(block.insts[2]);
        assert_eq!(sub_inst.operands[1], vreg(2));
    }

    // ---- Chain of redundancies ----

    #[test]
    fn test_gvn_chain_redundancies() {
        // v2 = add v0, v1
        // v3 = add v0, v1   -> eliminated, v3 -> v2
        // v4 = sub v3, #1   -> v4 = sub v2, #1
        // ret
        let a1 = MachInst::new(AArch64Opcode::AddRR, vec![vreg(2), vreg(0), vreg(1)]);
        let a2 = MachInst::new(AArch64Opcode::AddRR, vec![vreg(3), vreg(0), vreg(1)]);
        let sub = MachInst::new(AArch64Opcode::SubRI, vec![vreg(4), vreg(3), imm(1)]);
        let ret = MachInst::new(AArch64Opcode::Ret, vec![]);
        let mut func = make_func_with_insts(vec![a1, a2, sub, ret]);

        let mut gvn = GlobalValueNumbering;
        assert!(gvn.run(&mut func));

        let block = func.block(func.entry);
        assert_eq!(block.insts.len(), 3); // a1, sub, ret
        let sub_inst = func.inst(block.insts[1]);
        assert_eq!(sub_inst.operands[1], vreg(2));
    }

    // ---- Proof annotation tests ----

    #[test]
    fn test_gvn_preserves_surviving_proof() {
        // v2 = add v0, v1 [NoOverflow]
        // v3 = add v0, v1 (no proof) -> eliminated
        // Surviving instruction keeps its proof.
        let a1 = MachInst::new(AArch64Opcode::AddRR, vec![vreg(2), vreg(0), vreg(1)])
            .with_proof(ProofAnnotation::NoOverflow);
        let a2 = MachInst::new(AArch64Opcode::AddRR, vec![vreg(3), vreg(0), vreg(1)]);
        let ret = MachInst::new(AArch64Opcode::Ret, vec![]);
        let mut func = make_func_with_insts(vec![a1, a2, ret]);

        let mut gvn = GlobalValueNumbering;
        assert!(gvn.run(&mut func));

        let block = func.block(func.entry);
        assert_eq!(block.insts.len(), 2);
        let surviving = func.inst(block.insts[0]);
        assert_eq!(surviving.proof, Some(ProofAnnotation::NoOverflow));
    }

    #[test]
    fn test_gvn_merges_proof_from_eliminated() {
        // v2 = add v0, v1 (no proof)
        // v3 = add v0, v1 [InBounds] -> eliminated, proof merged onto v2
        let a1 = MachInst::new(AArch64Opcode::AddRR, vec![vreg(2), vreg(0), vreg(1)]);
        let a2 = MachInst::new(AArch64Opcode::AddRR, vec![vreg(3), vreg(0), vreg(1)])
            .with_proof(ProofAnnotation::InBounds);
        let ret = MachInst::new(AArch64Opcode::Ret, vec![]);
        let mut func = make_func_with_insts(vec![a1, a2, ret]);

        let mut gvn = GlobalValueNumbering;
        assert!(gvn.run(&mut func));

        let block = func.block(func.entry);
        assert_eq!(block.insts.len(), 2);
        let surviving = func.inst(block.insts[0]);
        assert_eq!(surviving.proof, Some(ProofAnnotation::InBounds));
    }

    #[test]
    fn test_gvn_merges_same_proof() {
        // Both have the same proof -> surviving keeps it.
        let a1 = MachInst::new(AArch64Opcode::AddRR, vec![vreg(2), vreg(0), vreg(1)])
            .with_proof(ProofAnnotation::NotNull);
        let a2 = MachInst::new(AArch64Opcode::AddRR, vec![vreg(3), vreg(0), vreg(1)])
            .with_proof(ProofAnnotation::NotNull);
        let ret = MachInst::new(AArch64Opcode::Ret, vec![]);
        let mut func = make_func_with_insts(vec![a1, a2, ret]);

        let mut gvn = GlobalValueNumbering;
        assert!(gvn.run(&mut func));

        let block = func.block(func.entry);
        assert_eq!(block.insts.len(), 2);
        let surviving = func.inst(block.insts[0]);
        assert_eq!(surviving.proof, Some(ProofAnnotation::NotNull));
    }

    #[test]
    fn test_gvn_drops_conflicting_proofs() {
        // Different proofs -> conservative merge returns None.
        let a1 = MachInst::new(AArch64Opcode::AddRR, vec![vreg(2), vreg(0), vreg(1)])
            .with_proof(ProofAnnotation::NoOverflow);
        let a2 = MachInst::new(AArch64Opcode::AddRR, vec![vreg(3), vreg(0), vreg(1)])
            .with_proof(ProofAnnotation::InBounds);
        let ret = MachInst::new(AArch64Opcode::Ret, vec![]);
        let mut func = make_func_with_insts(vec![a1, a2, ret]);

        let mut gvn = GlobalValueNumbering;
        assert!(gvn.run(&mut func));

        let block = func.block(func.entry);
        assert_eq!(block.insts.len(), 2);
        let surviving = func.inst(block.insts[0]);
        // Different proofs -> conservatively dropped.
        assert!(surviving.proof.is_none());
    }

    // ---- Load proof annotation test ----

    #[test]
    fn test_gvn_load_proof_preserved() {
        // l1 = ldr v0, #8 [NotNull]
        // l2 = ldr v0, #8   -> eliminated, proof merged
        // ret
        let l1 = MachInst::new(AArch64Opcode::LdrRI, vec![vreg(2), vreg(0), imm(8)])
            .with_proof(ProofAnnotation::NotNull);
        let l2 = MachInst::new(AArch64Opcode::LdrRI, vec![vreg(3), vreg(0), imm(8)]);
        let ret = MachInst::new(AArch64Opcode::Ret, vec![]);
        let mut func = make_func_with_insts(vec![l1, l2, ret]);

        let mut gvn = GlobalValueNumbering;
        assert!(gvn.run(&mut func));

        let block = func.block(func.entry);
        assert_eq!(block.insts.len(), 2);
        let surviving = func.inst(block.insts[0]);
        assert_eq!(surviving.proof, Some(ProofAnnotation::NotNull));
    }

    // ---- Scope isolation test ----

    #[test]
    fn test_gvn_scope_isolation_loads() {
        // bb0: v2 = ldr v0, #8
        //      br bb1, bb2
        // bb1: str v5, v0, #8   (store kills loads in this scope)
        //      v3 = ldr v0, #8  (cannot reuse bb0's load because of store)
        //      br bb3
        // bb2: v4 = ldr v0, #8  (CAN reuse bb0's load — no store on this path)
        //      br bb3
        // bb3: ret
        let mut func = MachFunction::new(
            "test_scope_loads".to_string(),
            Signature::new(vec![], vec![]),
        );
        let bb0 = func.entry;
        let bb1 = func.create_block();
        let bb2 = func.create_block();
        let bb3 = func.create_block();

        // bb0
        let l0 = func.push_inst(MachInst::new(
            AArch64Opcode::LdrRI,
            vec![vreg(2), vreg(0), imm(8)],
        ));
        func.append_inst(bb0, l0);
        let br0 = func.push_inst(MachInst::new(
            AArch64Opcode::BCond,
            vec![MachOperand::Block(bb1), MachOperand::Block(bb2)],
        ));
        func.append_inst(bb0, br0);

        // bb1: store + load
        let st = func.push_inst(MachInst::new(
            AArch64Opcode::StrRI,
            vec![vreg(5), vreg(0), imm(8)],
        ));
        func.append_inst(bb1, st);
        let l1 = func.push_inst(MachInst::new(
            AArch64Opcode::LdrRI,
            vec![vreg(3), vreg(0), imm(8)],
        ));
        func.append_inst(bb1, l1);
        let br1 = func.push_inst(MachInst::new(
            AArch64Opcode::B,
            vec![MachOperand::Block(bb3)],
        ));
        func.append_inst(bb1, br1);

        // bb2: just load (no store)
        let l2 = func.push_inst(MachInst::new(
            AArch64Opcode::LdrRI,
            vec![vreg(4), vreg(0), imm(8)],
        ));
        func.append_inst(bb2, l2);
        let br2 = func.push_inst(MachInst::new(
            AArch64Opcode::B,
            vec![MachOperand::Block(bb3)],
        ));
        func.append_inst(bb2, br2);

        // bb3
        let ret = func.push_inst(MachInst::new(AArch64Opcode::Ret, vec![]));
        func.append_inst(bb3, ret);

        func.add_edge(bb0, bb1);
        func.add_edge(bb0, bb2);
        func.add_edge(bb1, bb3);
        func.add_edge(bb2, bb3);

        let mut gvn = GlobalValueNumbering;
        assert!(gvn.run(&mut func));

        // bb1: store killed the load, so bb1 still has 3 instructions
        assert_eq!(func.block(bb1).insts.len(), 3); // store, load, branch
        // bb2: load was eliminated (bb0's load dominates and no store)
        assert_eq!(func.block(bb2).insts.len(), 1); // just branch
    }

    // ---- MOVK tied def-use regression test (issue #366) ----

    /// Regression test for the #366 residual xxh3 miscompile.
    ///
    /// MOVK is a tied def-use instruction: `MOVK Rd, #imm16, LSL #shift`
    /// inserts `imm16` into Rd at position `shift` while preserving the
    /// other bits. The instruction depends on the *current* value of Rd,
    /// but that dependency is not captured in the explicit operand list.
    ///
    /// Before this fix, GVN treated two MOVKs with matching (imm, shift)
    /// as redundant, even when their destination registers held
    /// different prior values. This corrupted multi-register constant
    /// materialization chains (e.g., two unrelated 64-bit constants that
    /// happened to share some 16-bit chunks).
    ///
    /// The scenario below mimics xxh3: build two 64-bit constants
    /// 0x067e2f2a_6bfdd932 and 0x067e2f2a_6d83f618 that share the upper
    /// 32 bits (MOVKs at positions 32 and 48 have the same imm). GVN
    /// must NOT eliminate the second register's MOVKs.
    #[test]
    fn test_gvn_preserves_movk_with_different_dest() {
        // v2 = movz #0xd932
        // v2 = movk #0x6bfd, lsl 16
        // v2 = movk #0x2f2a, lsl 32  (shared with v3)
        // v2 = movk #0x067e, lsl 48  (shared with v3)
        //
        // v3 = movz #0xf618
        // v3 = movk #0x6d83, lsl 16
        // v3 = movk #0x2f2a, lsl 32  (same imm+shift as v2's but DIFFERENT dest)
        // v3 = movk #0x067e, lsl 48  (same imm+shift as v2's but DIFFERENT dest)
        // ret
        //
        // GVN must preserve all eight instructions — eliminating v3's
        // upper MOVKs would corrupt v3 to 0x0000_0000_6d83_f618.
        let m_movz_v2 = MachInst::new(AArch64Opcode::Movz, vec![vreg(2), imm(0xd932)]);
        let m_movk_v2_16 =
            MachInst::new(AArch64Opcode::Movk, vec![vreg(2), imm(0x6bfd), imm(16)]);
        let m_movk_v2_32 =
            MachInst::new(AArch64Opcode::Movk, vec![vreg(2), imm(0x2f2a), imm(32)]);
        let m_movk_v2_48 =
            MachInst::new(AArch64Opcode::Movk, vec![vreg(2), imm(0x067e), imm(48)]);

        let m_movz_v3 = MachInst::new(AArch64Opcode::Movz, vec![vreg(3), imm(0xf618)]);
        let m_movk_v3_16 =
            MachInst::new(AArch64Opcode::Movk, vec![vreg(3), imm(0x6d83), imm(16)]);
        let m_movk_v3_32 =
            MachInst::new(AArch64Opcode::Movk, vec![vreg(3), imm(0x2f2a), imm(32)]);
        let m_movk_v3_48 =
            MachInst::new(AArch64Opcode::Movk, vec![vreg(3), imm(0x067e), imm(48)]);
        let ret = MachInst::new(AArch64Opcode::Ret, vec![]);

        let mut func = make_func_with_insts(vec![
            m_movz_v2,
            m_movk_v2_16,
            m_movk_v2_32,
            m_movk_v2_48,
            m_movz_v3,
            m_movk_v3_16,
            m_movk_v3_32,
            m_movk_v3_48,
            ret,
        ]);

        let mut gvn = GlobalValueNumbering;
        // GVN may report no changes for this input — but even if it does
        // report changes (via the MOVZs, which are safe to number), the
        // MOVKs must survive.
        let _ = gvn.run(&mut func);

        let block = func.block(func.entry);

        // Count remaining MOVKs — must still be 6 (three per constant).
        let movk_count = block
            .insts
            .iter()
            .filter(|id| func.inst(**id).opcode == AArch64Opcode::Movk)
            .count();
        assert_eq!(
            movk_count, 6,
            "GVN eliminated a MOVK — this is the #366 bug. All six MOVKs must survive."
        );

        // Verify each MOVK still targets the correct vreg (no def-vreg rewrite).
        for inst_id in &block.insts {
            let inst = func.inst(*inst_id);
            if inst.opcode == AArch64Opcode::Movk {
                match &inst.operands[0] {
                    MachOperand::VReg(v) => {
                        assert!(
                            v.id == 2 || v.id == 3,
                            "MOVK destination was rewritten to unexpected vreg {}",
                            v.id
                        );
                    }
                    _ => panic!("MOVK operand[0] is not a VReg"),
                }
            }
        }
    }

    /// Tighter version: check that running GVN on a MOVK chain doesn't
    /// drop or rewrite the MOVKs of either constant.
    #[test]
    fn test_gvn_movk_chain_preserves_both_constants() {
        // Same as above but simpler: just two parallel MOVZ+MOVK pairs
        // with the SAME (imm, shift) MOVK. GVN must NOT merge them.
        let m1 = MachInst::new(AArch64Opcode::Movz, vec![vreg(2), imm(0x1111)]);
        let m2 = MachInst::new(AArch64Opcode::Movk, vec![vreg(2), imm(0xabcd), imm(16)]);
        let m3 = MachInst::new(AArch64Opcode::Movz, vec![vreg(3), imm(0x2222)]);
        let m4 = MachInst::new(AArch64Opcode::Movk, vec![vreg(3), imm(0xabcd), imm(16)]);
        let ret = MachInst::new(AArch64Opcode::Ret, vec![]);
        let mut func = make_func_with_insts(vec![m1, m2, m3, m4, ret]);

        let mut gvn = GlobalValueNumbering;
        let _ = gvn.run(&mut func);

        let block = func.block(func.entry);
        let movk_count = block
            .insts
            .iter()
            .filter(|id| func.inst(**id).opcode == AArch64Opcode::Movk)
            .count();
        assert_eq!(
            movk_count, 2,
            "GVN incorrectly merged two MOVKs with different destinations"
        );
    }

    // -------------------------------------------------------------------
    // Regression tests for #408 / #409 — BFM tied def-use and ADC/SBC
    // carry-flag reads must be correctly classified so GVN does not fold
    // semantically-different instructions into one.
    // -------------------------------------------------------------------

    /// Regression for #408: two BFMs with identical explicit operands but
    /// different prior Rd values must NOT be value-numbered together.
    /// BFM preserves the uncovered bits of Rd, so the prior dest is an
    /// implicit input just like MOVK.
    #[test]
    fn test_gvn_preserves_bfm_with_different_dest() {
        // v2 = movz #0x1111
        // v2 = bfm v2, v0, #0, #7       (insert low byte of v0 into v2)
        // v3 = movz #0x2222
        // v3 = bfm v3, v0, #0, #7       (same explicit (src, immr, imms) but
        //                                different prior dest value)
        // ret
        //
        // Eliminating the second BFM and replacing v3 with v2 silently
        // corrupts v3 — its high bits were 0x2222 but would become 0x1111.
        let m_movz_v2 = MachInst::new(AArch64Opcode::Movz, vec![vreg(2), imm(0x1111)]);
        let m_bfm_v2 = MachInst::new(
            AArch64Opcode::Bfm,
            vec![vreg(2), vreg(0), imm(0), imm(7)],
        );
        let m_movz_v3 = MachInst::new(AArch64Opcode::Movz, vec![vreg(3), imm(0x2222)]);
        let m_bfm_v3 = MachInst::new(
            AArch64Opcode::Bfm,
            vec![vreg(3), vreg(0), imm(0), imm(7)],
        );
        let ret = MachInst::new(AArch64Opcode::Ret, vec![]);
        let mut func = make_func_with_insts(vec![m_movz_v2, m_bfm_v2, m_movz_v3, m_bfm_v3, ret]);

        let mut gvn = GlobalValueNumbering;
        let _ = gvn.run(&mut func);

        let block = func.block(func.entry);
        let bfm_count = block
            .insts
            .iter()
            .filter(|id| func.inst(**id).opcode == AArch64Opcode::Bfm)
            .count();
        assert_eq!(
            bfm_count, 2,
            "GVN merged two BFMs with different prior Rd values — this is the #408 bug"
        );

        // Every surviving BFM must keep its original destination.
        for inst_id in &block.insts {
            let inst = func.inst(*inst_id);
            if inst.opcode == AArch64Opcode::Bfm {
                match &inst.operands[0] {
                    MachOperand::VReg(v) => assert!(
                        v.id == 2 || v.id == 3,
                        "BFM destination was rewritten to unexpected vreg {}",
                        v.id
                    ),
                    _ => panic!("BFM operand[0] is not a VReg"),
                }
            }
        }
    }

    /// Regression for #409: two ADCs with the same explicit operands but
    /// preceded by different flag writers (different carry inputs) must
    /// NOT be GVN'd together. The carry flag is an implicit input.
    #[test]
    fn test_gvn_preserves_adc_across_flag_writers() {
        // v10 = adds v0, v1           (flag writer #1)
        // v2  = adc  v4, v5           (reads carry from #1)
        // v11 = adds v6, v7           (flag writer #2 — different inputs)
        // v3  = adc  v4, v5           (same explicit operands as above,
        //                               but carry comes from #2)
        // ret
        //
        // Merging v3 into v2 drops the dependency on the second ADDS and
        // silently miscompiles any i128 / multi-precision arithmetic.
        let adds1 = MachInst::new(AArch64Opcode::AddsRR, vec![vreg(10), vreg(0), vreg(1)]);
        let adc1 = MachInst::new(AArch64Opcode::Adc, vec![vreg(2), vreg(4), vreg(5)]);
        let adds2 = MachInst::new(AArch64Opcode::AddsRR, vec![vreg(11), vreg(6), vreg(7)]);
        let adc2 = MachInst::new(AArch64Opcode::Adc, vec![vreg(3), vreg(4), vreg(5)]);
        let ret = MachInst::new(AArch64Opcode::Ret, vec![]);
        let mut func = make_func_with_insts(vec![adds1, adc1, adds2, adc2, ret]);

        let mut gvn = GlobalValueNumbering;
        let _ = gvn.run(&mut func);

        let block = func.block(func.entry);
        let adc_count = block
            .insts
            .iter()
            .filter(|id| func.inst(**id).opcode == AArch64Opcode::Adc)
            .count();
        assert_eq!(
            adc_count, 2,
            "GVN merged two ADCs across different flag writers — this is the #409 bug"
        );

        // Each surviving ADC must keep its original destination vreg.
        let mut seen = std::collections::HashSet::new();
        for inst_id in &block.insts {
            let inst = func.inst(*inst_id);
            if inst.opcode == AArch64Opcode::Adc
                && let MachOperand::VReg(v) = &inst.operands[0]
            {
                seen.insert(v.id);
            }
        }
        assert!(
            seen.contains(&2) && seen.contains(&3),
            "ADC destinations were rewritten: surviving dests = {:?}",
            seen
        );
    }

    /// Matching SBC regression: symmetry with ADC.
    #[test]
    fn test_gvn_preserves_sbc_across_flag_writers() {
        let subs1 = MachInst::new(AArch64Opcode::SubsRR, vec![vreg(10), vreg(0), vreg(1)]);
        let sbc1 = MachInst::new(AArch64Opcode::Sbc, vec![vreg(2), vreg(4), vreg(5)]);
        let subs2 = MachInst::new(AArch64Opcode::SubsRR, vec![vreg(11), vreg(6), vreg(7)]);
        let sbc2 = MachInst::new(AArch64Opcode::Sbc, vec![vreg(3), vreg(4), vreg(5)]);
        let ret = MachInst::new(AArch64Opcode::Ret, vec![]);
        let mut func = make_func_with_insts(vec![subs1, sbc1, subs2, sbc2, ret]);

        let mut gvn = GlobalValueNumbering;
        let _ = gvn.run(&mut func);

        let block = func.block(func.entry);
        let sbc_count = block
            .insts
            .iter()
            .filter(|id| func.inst(**id).opcode == AArch64Opcode::Sbc)
            .count();
        assert_eq!(
            sbc_count, 2,
            "GVN merged two SBCs across different flag writers — #409 regression"
        );
    }
}
