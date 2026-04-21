// llvm2-opt - Function Inlining
//
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Copyright 2026 Dropbox, Inc. | License: Apache-2.0

//! Function inlining pass for machine-level IR.
//!
//! Replaces direct call instructions (`BL`/`Bl`) to small, single-block
//! callee functions with the callee's body inlined at the call site.
//!
//! # Algorithm
//!
//! For each basic block in the caller:
//! 1. Scan instructions for direct calls (`Bl`/`BL`) with a `Symbol` operand.
//! 2. Look up the callee by name in the registered callee map.
//! 3. Check eligibility:
//!    - Callee instruction count <= `max_callee_size` (default 20).
//!    - Callee is not recursive (caller name != callee name).
//!    - Callee has exactly one basic block (single-block only in v1).
//! 4. Inline by:
//!    a. Computing a VReg offset from the caller's `next_vreg`.
//!    b. Copying each callee instruction (except the trailing `Ret`)
//!       into the caller, remapping `VReg` operands by adding the offset.
//!    c. Replacing the call instruction in the block with the inlined body.
//!    d. Advancing `caller.next_vreg` past the callee's vreg namespace.
//!
//! # Limitations (v1)
//!
//! - Only single-block callees are inlined.
//! - Indirect calls (`BLR`/`Blr`) are not inlined.
//! - No parameter/return value remapping beyond vreg renumbering.
//! - Recursive calls are unconditionally skipped.
//!
//! Reference: LLVM `InlineFunction.cpp`, `InlineCost.cpp`

use std::collections::HashMap;

use llvm2_ir::{
    AArch64Opcode, InstId, MachFunction, MachInst, MachOperand, VReg,
};

use crate::pass_manager::MachinePass;

/// Default maximum callee instruction count for inlining eligibility.
const DEFAULT_MAX_CALLEE_SIZE: usize = 20;

/// Function inlining pass.
///
/// Holds a map of known callee functions. When the pass encounters a
/// direct call to a callee in this map that is small enough, it replaces
/// the call with the callee's body.
pub struct FunctionInlining {
    /// Available callee functions for inlining lookups.
    callees: HashMap<String, MachFunction>,
    /// Maximum number of instructions in a callee to consider for inlining.
    max_callee_size: usize,
}

impl FunctionInlining {
    /// Create a new inlining pass with default settings.
    pub fn new() -> Self {
        Self {
            callees: HashMap::new(),
            max_callee_size: DEFAULT_MAX_CALLEE_SIZE,
        }
    }

    /// Register a callee function available for inlining.
    pub fn with_callee(mut self, name: String, func: MachFunction) -> Self {
        self.callees.insert(name, func);
        self
    }

    /// Set the maximum callee instruction count threshold.
    pub fn with_max_size(mut self, max_size: usize) -> Self {
        self.max_callee_size = max_size;
        self
    }

    /// Add a callee function (non-builder API).
    pub fn add_callee(&mut self, name: String, func: MachFunction) {
        self.callees.insert(name, func);
    }

    /// Check whether a callee is eligible for inlining.
    fn is_eligible(&self, caller_name: &str, callee_name: &str) -> bool {
        // Skip recursive calls.
        if caller_name == callee_name {
            return false;
        }

        let callee = match self.callees.get(callee_name) {
            Some(c) => c,
            None => return false,
        };

        // Single-block only in v1.
        if callee.num_blocks() != 1 {
            return false;
        }

        // Count instructions in the callee's entry block.
        let entry_block = callee.block(callee.entry);
        if entry_block.insts.len() > self.max_callee_size {
            return false;
        }

        true
    }

    /// Remap a single operand: if it is a VReg, add the offset to its id.
    fn remap_operand(op: &MachOperand, vreg_offset: u32) -> MachOperand {
        match op {
            MachOperand::VReg(v) => {
                MachOperand::VReg(VReg::new(v.id + vreg_offset, v.class))
            }
            other => other.clone(),
        }
    }

    /// Clone a callee instruction with remapped vreg IDs.
    fn remap_inst(inst: &MachInst, vreg_offset: u32) -> MachInst {
        let remapped_operands: Vec<MachOperand> = inst
            .operands
            .iter()
            .map(|op| Self::remap_operand(op, vreg_offset))
            .collect();

        MachInst {
            opcode: inst.opcode,
            operands: remapped_operands,
            implicit_defs: inst.implicit_defs,
            implicit_uses: inst.implicit_uses,
            flags: inst.flags,
            proof: inst.proof,
            source_loc: inst.source_loc,
        }
    }

    /// Inline a single call site. Returns the list of new InstIds that
    /// replace the call, or None if the call is not eligible.
    fn try_inline_call(
        &self,
        caller: &mut MachFunction,
        callee_name: &str,
    ) -> Option<Vec<InstId>> {
        let callee = self.callees.get(callee_name)?;

        let vreg_offset = caller.next_vreg;
        let callee_entry = callee.block(callee.entry);

        // Collect inlined instruction IDs.
        let mut inlined_ids = Vec::new();

        for &callee_inst_id in &callee_entry.insts {
            let callee_inst = callee.inst(callee_inst_id);

            // Skip the trailing Ret instruction — the caller's control flow
            // continues after the inlined body.
            if callee_inst.is_return() {
                continue;
            }

            let remapped = Self::remap_inst(callee_inst, vreg_offset);
            let new_id = caller.push_inst(remapped);
            inlined_ids.push(new_id);
        }

        // Advance the caller's vreg counter past the callee's namespace.
        if callee.next_vreg > 0 {
            caller.next_vreg += callee.next_vreg;
        }

        Some(inlined_ids)
    }
}

impl Default for FunctionInlining {
    fn default() -> Self {
        Self::new()
    }
}

impl MachinePass for FunctionInlining {
    fn name(&self) -> &str {
        "inline"
    }

    fn run(&mut self, func: &mut MachFunction) -> bool {
        let mut changed = false;
        let caller_name = func.name.clone();

        // Process each block. We iterate over block_order by index because
        // we mutate blocks during iteration.
        let block_ids: Vec<_> = func.block_order.clone();

        for block_id in block_ids {
            // Build a new instruction list for this block, replacing calls
            // with inlined bodies where eligible.
            let old_insts: Vec<InstId> = func.block(block_id).insts.clone();
            let mut new_insts = Vec::new();
            let mut block_changed = false;

            for &inst_id in &old_insts {
                let inst = func.inst(inst_id);

                // Check if this is a direct call (Bl or BL) with a Symbol operand.
                let callee_name = if matches!(
                    inst.opcode,
                    AArch64Opcode::Bl | AArch64Opcode::BL
                ) {
                    inst.operands.iter().find_map(|op| {
                        if let MachOperand::Symbol(s) = op {
                            Some(s.clone())
                        } else {
                            None
                        }
                    })
                } else {
                    None
                };

                if let Some(ref target) = callee_name {
                    if self.is_eligible(&caller_name, target) {
                        if let Some(inlined_ids) =
                            self.try_inline_call(func, target)
                        {
                            new_insts.extend(inlined_ids);
                            block_changed = true;
                            continue;
                        }
                    }
                }

                // Not inlined — keep the original instruction.
                new_insts.push(inst_id);
            }

            if block_changed {
                func.block_mut(block_id).insts = new_insts;
                changed = true;
            }
        }

        changed
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use llvm2_ir::{
        AArch64Opcode, MachFunction, MachInst, MachOperand, RegClass,
        Signature, VReg,
    };
    use crate::pass_manager::MachinePass;

    fn vreg(id: u32) -> MachOperand {
        MachOperand::VReg(VReg::new(id, RegClass::Gpr64))
    }

    fn imm(val: i64) -> MachOperand {
        MachOperand::Imm(val)
    }

    fn make_func_with_insts(name: &str, insts: Vec<MachInst>) -> MachFunction {
        let mut func = MachFunction::new(
            name.to_string(),
            Signature::new(vec![], vec![]),
        );
        let block = func.entry;
        for inst in insts {
            let id = func.push_inst(inst);
            func.append_inst(block, id);
        }
        func
    }

    /// Helper: create a simple callee function with given instructions + Ret.
    fn make_callee(name: &str, body_insts: Vec<MachInst>, next_vreg: u32) -> MachFunction {
        let mut func = make_func_with_insts(name, body_insts);
        // Append a Ret at the end.
        let ret = MachInst::new(AArch64Opcode::Ret, vec![]);
        let ret_id = func.push_inst(ret);
        func.append_inst(func.entry, ret_id);
        func.next_vreg = next_vreg;
        func
    }

    #[test]
    fn test_inline_simple_callee() {
        // Callee: "add_fn" has one add instruction + ret
        //   v0 = add v1, v2
        //   ret
        let callee_add = MachInst::new(
            AArch64Opcode::AddRR,
            vec![vreg(0), vreg(1), vreg(2)],
        );
        let callee = make_callee("add_fn", vec![callee_add], 3);

        // Caller: "main" calls add_fn, then returns
        //   bl add_fn
        //   ret
        let call = MachInst::new(
            AArch64Opcode::Bl,
            vec![MachOperand::Symbol("add_fn".to_string())],
        );
        let ret = MachInst::new(AArch64Opcode::Ret, vec![]);
        let mut caller = make_func_with_insts("main", vec![call, ret]);
        caller.next_vreg = 5; // caller uses vregs 0..4

        let mut pass = FunctionInlining::new()
            .with_callee("add_fn".to_string(), callee);

        assert!(pass.run(&mut caller));

        // After inlining: the call is replaced by the add instruction (Ret skipped),
        // followed by the original Ret.
        let block = caller.block(caller.entry);
        assert_eq!(block.insts.len(), 2); // inlined add + original ret

        // The inlined add should have remapped vregs: offset=5, so v0->v5, v1->v6, v2->v7
        let inlined_add = caller.inst(block.insts[0]);
        assert_eq!(inlined_add.opcode, AArch64Opcode::AddRR);
        assert_eq!(inlined_add.operands.len(), 3);
        assert_eq!(inlined_add.operands[0], vreg(5));
        assert_eq!(inlined_add.operands[1], vreg(6));
        assert_eq!(inlined_add.operands[2], vreg(7));

        // next_vreg should have advanced: 5 + 3 = 8
        assert_eq!(caller.next_vreg, 8);
    }

    #[test]
    fn test_inline_skips_large_callee() {
        // Create a callee with more instructions than the threshold.
        let mut body = Vec::new();
        for i in 0..5 {
            body.push(MachInst::new(
                AArch64Opcode::AddRI,
                vec![vreg(i), vreg(i), imm(1)],
            ));
        }
        let callee = make_callee("big_fn", body, 5);

        let call = MachInst::new(
            AArch64Opcode::Bl,
            vec![MachOperand::Symbol("big_fn".to_string())],
        );
        let ret = MachInst::new(AArch64Opcode::Ret, vec![]);
        let mut caller = make_func_with_insts("main", vec![call, ret]);

        // Set threshold to 3 — callee has 5 body + 1 ret = 6 instructions.
        let mut pass = FunctionInlining::new()
            .with_callee("big_fn".to_string(), callee)
            .with_max_size(3);

        assert!(!pass.run(&mut caller));

        // Call should still be present.
        let block = caller.block(caller.entry);
        assert_eq!(block.insts.len(), 2);
    }

    #[test]
    fn test_inline_skips_recursive() {
        // Caller calls itself — should not inline.
        let callee_add = MachInst::new(
            AArch64Opcode::AddRR,
            vec![vreg(0), vreg(1), vreg(2)],
        );
        let callee = make_callee("my_func", vec![callee_add], 3);

        let call = MachInst::new(
            AArch64Opcode::Bl,
            vec![MachOperand::Symbol("my_func".to_string())],
        );
        let ret = MachInst::new(AArch64Opcode::Ret, vec![]);
        let mut caller = make_func_with_insts("my_func", vec![call, ret]);

        let mut pass = FunctionInlining::new()
            .with_callee("my_func".to_string(), callee);

        assert!(!pass.run(&mut caller));

        let block = caller.block(caller.entry);
        assert_eq!(block.insts.len(), 2); // call + ret still present
    }

    #[test]
    fn test_inline_skips_indirect_call() {
        // BLR (indirect call) should not be inlined.
        let call = MachInst::new(AArch64Opcode::Blr, vec![vreg(0)]);
        let ret = MachInst::new(AArch64Opcode::Ret, vec![]);
        let mut caller = make_func_with_insts("main", vec![call, ret]);

        let callee_add = MachInst::new(
            AArch64Opcode::AddRR,
            vec![vreg(0), vreg(1), vreg(2)],
        );
        let callee = make_callee("target", vec![callee_add], 3);

        let mut pass = FunctionInlining::new()
            .with_callee("target".to_string(), callee);

        assert!(!pass.run(&mut caller));

        let block = caller.block(caller.entry);
        assert_eq!(block.insts.len(), 2);
    }

    #[test]
    fn test_inline_skips_multi_block() {
        // Create a callee with two blocks — not eligible for v1 inlining.
        let mut callee = MachFunction::new(
            "multi_block".to_string(),
            Signature::new(vec![], vec![]),
        );
        let add = MachInst::new(
            AArch64Opcode::AddRR,
            vec![vreg(0), vreg(1), vreg(2)],
        );
        let add_id = callee.push_inst(add);
        callee.append_inst(callee.entry, add_id);

        // Create a second block.
        let bb1 = callee.create_block();
        let ret = MachInst::new(AArch64Opcode::Ret, vec![]);
        let ret_id = callee.push_inst(ret);
        callee.append_inst(bb1, ret_id);
        callee.next_vreg = 3;

        let call = MachInst::new(
            AArch64Opcode::Bl,
            vec![MachOperand::Symbol("multi_block".to_string())],
        );
        let caller_ret = MachInst::new(AArch64Opcode::Ret, vec![]);
        let mut caller = make_func_with_insts("main", vec![call, caller_ret]);

        let mut pass = FunctionInlining::new()
            .with_callee("multi_block".to_string(), callee);

        assert!(!pass.run(&mut caller));

        let block = caller.block(caller.entry);
        assert_eq!(block.insts.len(), 2);
    }

    #[test]
    fn test_inline_no_calls() {
        // Function without any calls — pass should return false.
        let add = MachInst::new(
            AArch64Opcode::AddRR,
            vec![vreg(0), vreg(1), vreg(2)],
        );
        let ret = MachInst::new(AArch64Opcode::Ret, vec![]);
        let mut func = make_func_with_insts("main", vec![add, ret]);

        let mut pass = FunctionInlining::new();
        assert!(!pass.run(&mut func));
    }

    #[test]
    fn test_inline_remaps_vregs() {
        // Callee uses vregs 0, 1, 2. Caller's next_vreg is 10.
        // After inlining, callee vregs should be 10, 11, 12.
        let callee_sub = MachInst::new(
            AArch64Opcode::SubRI,
            vec![vreg(0), vreg(1), imm(42)],
        );
        let callee_add = MachInst::new(
            AArch64Opcode::AddRR,
            vec![vreg(2), vreg(0), vreg(1)],
        );
        let callee = make_callee("helper", vec![callee_sub, callee_add], 3);

        let call = MachInst::new(
            AArch64Opcode::Bl,
            vec![MachOperand::Symbol("helper".to_string())],
        );
        let ret = MachInst::new(AArch64Opcode::Ret, vec![]);
        let mut caller = make_func_with_insts("main", vec![call, ret]);
        caller.next_vreg = 10;

        let mut pass = FunctionInlining::new()
            .with_callee("helper".to_string(), callee);

        assert!(pass.run(&mut caller));

        let block = caller.block(caller.entry);
        // 2 inlined instructions + original ret = 3
        assert_eq!(block.insts.len(), 3);

        // First inlined: sub v10, v11, #42
        let inlined_sub = caller.inst(block.insts[0]);
        assert_eq!(inlined_sub.opcode, AArch64Opcode::SubRI);
        assert_eq!(inlined_sub.operands[0], vreg(10));
        assert_eq!(inlined_sub.operands[1], vreg(11));
        assert_eq!(inlined_sub.operands[2], imm(42)); // immediates unchanged

        // Second inlined: add v12, v10, v11
        let inlined_add = caller.inst(block.insts[1]);
        assert_eq!(inlined_add.opcode, AArch64Opcode::AddRR);
        assert_eq!(inlined_add.operands[0], vreg(12));
        assert_eq!(inlined_add.operands[1], vreg(10));
        assert_eq!(inlined_add.operands[2], vreg(11));

        // next_vreg: 10 + 3 = 13
        assert_eq!(caller.next_vreg, 13);
    }

    #[test]
    fn test_inline_multiple_calls() {
        // Caller has two calls to the same callee — both should be inlined.
        let callee_mov = MachInst::new(
            AArch64Opcode::MovI,
            vec![vreg(0), imm(99)],
        );
        let callee = make_callee("tiny", vec![callee_mov], 1);

        let call1 = MachInst::new(
            AArch64Opcode::Bl,
            vec![MachOperand::Symbol("tiny".to_string())],
        );
        let call2 = MachInst::new(
            AArch64Opcode::Bl,
            vec![MachOperand::Symbol("tiny".to_string())],
        );
        let ret = MachInst::new(AArch64Opcode::Ret, vec![]);
        let mut caller = make_func_with_insts("main", vec![call1, call2, ret]);
        caller.next_vreg = 5;

        let mut pass = FunctionInlining::new()
            .with_callee("tiny".to_string(), callee);

        assert!(pass.run(&mut caller));

        let block = caller.block(caller.entry);
        // Two inlined mov instructions + ret = 3
        assert_eq!(block.insts.len(), 3);

        // First inlined mov: vreg offset 5 -> v5
        let mov1 = caller.inst(block.insts[0]);
        assert_eq!(mov1.opcode, AArch64Opcode::MovI);
        assert_eq!(mov1.operands[0], vreg(5));

        // Second inlined mov: vreg offset 5+1=6 -> v6
        let mov2 = caller.inst(block.insts[1]);
        assert_eq!(mov2.opcode, AArch64Opcode::MovI);
        assert_eq!(mov2.operands[0], vreg(6));

        // next_vreg: 5 + 1 + 1 = 7
        assert_eq!(caller.next_vreg, 7);
    }

    #[test]
    fn test_inline_preserves_proof_annotations() {
        use llvm2_ir::ProofAnnotation;

        // Callee has an instruction with a proof annotation.
        let callee_add = MachInst::new(
            AArch64Opcode::AddRR,
            vec![vreg(0), vreg(1), vreg(2)],
        ).with_proof(ProofAnnotation::NoOverflow);
        let callee = make_callee("proven", vec![callee_add], 3);

        let call = MachInst::new(
            AArch64Opcode::Bl,
            vec![MachOperand::Symbol("proven".to_string())],
        );
        let ret = MachInst::new(AArch64Opcode::Ret, vec![]);
        let mut caller = make_func_with_insts("main", vec![call, ret]);
        caller.next_vreg = 0;

        let mut pass = FunctionInlining::new()
            .with_callee("proven".to_string(), callee);

        assert!(pass.run(&mut caller));

        let block = caller.block(caller.entry);
        let inlined = caller.inst(block.insts[0]);
        assert_eq!(inlined.proof, Some(ProofAnnotation::NoOverflow));
    }

    #[test]
    fn test_inline_unknown_callee() {
        // Call to a function not in the callees map — should not inline.
        let call = MachInst::new(
            AArch64Opcode::Bl,
            vec![MachOperand::Symbol("unknown".to_string())],
        );
        let ret = MachInst::new(AArch64Opcode::Ret, vec![]);
        let mut caller = make_func_with_insts("main", vec![call, ret]);

        let mut pass = FunctionInlining::new();
        assert!(!pass.run(&mut caller));

        let block = caller.block(caller.entry);
        assert_eq!(block.insts.len(), 2);
    }

    #[test]
    fn test_inline_idempotent() {
        // After inlining, running the pass again should not change anything
        // (the inlined body contains no calls).
        let callee_add = MachInst::new(
            AArch64Opcode::AddRR,
            vec![vreg(0), vreg(1), vreg(2)],
        );
        let callee = make_callee("add_fn", vec![callee_add], 3);

        let call = MachInst::new(
            AArch64Opcode::Bl,
            vec![MachOperand::Symbol("add_fn".to_string())],
        );
        let ret = MachInst::new(AArch64Opcode::Ret, vec![]);
        let mut caller = make_func_with_insts("main", vec![call, ret]);
        caller.next_vreg = 5;

        let mut pass = FunctionInlining::new()
            .with_callee("add_fn".to_string(), callee);

        assert!(pass.run(&mut caller)); // First pass inlines
        assert!(!pass.run(&mut caller)); // Second pass: no calls remain
    }
}
