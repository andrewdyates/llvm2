// llvm2-opt - Rewrite matchers
//
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Copyright 2026 Dropbox, Inc. | License: Apache-2.0

//! [`Matcher`] selects candidate instructions for a rule.
//!
//! A matcher is a cheap structural check — "this opcode" or "this
//! category". Anything that requires looking at operand values, other
//! instructions, or the whole function belongs in a [`Constraint`] so
//! rules can share the same match but differ on side conditions.
//!
//! [`Constraint`]: super::Constraint

use std::collections::HashMap;

use llvm2_ir::{AArch64Opcode, BlockId, InstId, MachFunction, MachInst, OpcodeCategory};

/// A matcher selects candidate instructions for a rule.
///
/// Matchers must be cheap — they run on every instruction the engine
/// visits. Put expensive checks in [`Constraint`].
///
/// [`Constraint`]: super::Constraint
pub trait Matcher: Send + Sync {
    /// Returns true if this instruction is a candidate for the rule.
    fn matches(&self, inst: &MachInst) -> bool;
}

/// Context passed to constraints and rewriters during rule evaluation.
///
/// Holds immutable references so multiple rules can evaluate against the
/// same context without cloning.
pub struct MatchCtx<'a> {
    /// The instruction currently being considered.
    pub inst: &'a MachInst,
    /// The instruction's id within [`MachFunction::insts`].
    pub inst_id: InstId,
    /// Enclosing block id.
    pub block_id: BlockId,
    /// Enclosing function (read-only view).
    pub func: &'a MachFunction,
    /// Def map for the current block: VReg id -> defining InstId.
    /// Rebuilt per block; see [`crate::rewrite::engine::RewriteEngine`].
    pub def_map: &'a HashMap<u32, InstId>,
}

impl<'a> MatchCtx<'a> {
    /// Look up the defining instruction of the VReg in operand position `idx`.
    ///
    /// Returns `None` if:
    /// - the operand is not a VReg, or
    /// - the VReg is not defined in the current block (i.e. it is a block
    ///   param, a call return, a phi input, or defined in another block).
    ///
    /// When a VReg is redefined multiple times within the same block, the
    /// map resolves to the **last** writer that was seen in block order as
    /// the engine walks the block. This matches the definer-before-use
    /// semantics callers typically want for local peephole patterns.
    /// Cross-block definer analysis (across phis / control flow) is not
    /// exposed here and is future work.
    pub fn operand_def(&self, idx: usize) -> Option<&MachInst> {
        let vreg = self.inst.operands.get(idx)?.as_vreg()?;
        let def_id = self.def_map.get(&vreg.id)?;
        Some(self.func.inst(*def_id))
    }

    /// Alias of [`operand_def`](Self::operand_def) named for
    /// definer-driven pattern construction.
    ///
    /// Returns the unique in-block definer of the VReg at operand
    /// position `idx`, or `None` when the operand is not a VReg, is a
    /// block param, is defined outside the current block, or comes from
    /// a phi / call-return (anything that isn't a single definer we can
    /// see walking the block).
    #[inline]
    pub fn definer_of(&self, idx: usize) -> Option<&MachInst> {
        self.operand_def(idx)
    }
}

// ---------------------------------------------------------------------------
// Primitive matchers
// ---------------------------------------------------------------------------

/// Matches any instruction with the given opcode.
pub struct OpcodeMatcher {
    pub opcode: AArch64Opcode,
}

impl Matcher for OpcodeMatcher {
    #[inline]
    fn matches(&self, inst: &MachInst) -> bool {
        inst.opcode == self.opcode
    }
}

/// Matches any instruction whose opcode falls in the given category.
pub struct CategoryMatcher {
    pub category: OpcodeCategory,
}

impl Matcher for CategoryMatcher {
    #[inline]
    fn matches(&self, inst: &MachInst) -> bool {
        inst.opcode.categorize() == self.category
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use llvm2_ir::{MachOperand, RegClass, Signature, VReg};

    fn add_ri(imm: i64) -> MachInst {
        MachInst::new(
            AArch64Opcode::AddRI,
            vec![
                MachOperand::Imm(0),
                MachOperand::Imm(1),
                MachOperand::Imm(imm),
            ],
        )
    }

    fn vreg(id: u32) -> MachOperand {
        MachOperand::VReg(VReg::new(id, RegClass::Gpr64))
    }

    #[test]
    fn opcode_matcher_matches_exact() {
        let m = OpcodeMatcher {
            opcode: AArch64Opcode::AddRI,
        };
        assert!(m.matches(&add_ri(0)));
        assert!(!m.matches(&MachInst::new(AArch64Opcode::SubRI, vec![])));
    }

    #[test]
    fn category_matcher_matches_category() {
        let m = CategoryMatcher {
            category: OpcodeCategory::AddRI,
        };
        assert!(m.matches(&add_ri(5)));
        // SubRI is a different category, should not match.
        assert!(!m.matches(&MachInst::new(AArch64Opcode::SubRI, vec![])));
    }

    // ---------------------------------------------------------------------
    // definer_of tests — block param, movi-then-add, intervening redefinition.
    //
    // The engine walks the block once and builds `def_map` incrementally.
    // For these unit tests we build the def map by hand, mirroring the
    // engine's visit-order semantics (last writer wins).
    // ---------------------------------------------------------------------

    fn make_func_with_block(insts: Vec<MachInst>) -> (MachFunction, BlockId, Vec<InstId>) {
        let mut func = MachFunction::new("t".into(), Signature::new(vec![], vec![]));
        let entry = func.entry;
        let mut ids = Vec::new();
        for inst in insts {
            let id = func.push_inst(inst);
            func.append_inst(entry, id);
            ids.push(id);
        }
        (func, entry, ids)
    }

    /// Build def_map covering the first `up_to_exclusive` insts in block
    /// order; last writer wins. Mirrors the engine's incremental def map.
    fn def_map_after(
        func: &MachFunction,
        block_id: BlockId,
        up_to_exclusive: usize,
    ) -> HashMap<u32, InstId> {
        let mut map = HashMap::new();
        let ids = func.block(block_id).insts.clone();
        for &inst_id in ids.iter().take(up_to_exclusive) {
            let inst = func.inst(inst_id);
            if inst.opcode.produces_value() {
                if let Some(MachOperand::VReg(dst)) = inst.operands.first() {
                    map.insert(dst.id, inst_id);
                }
            }
        }
        map
    }

    #[test]
    fn definer_of_block_param_returns_none() {
        // Block param v99 — no definer in the block.
        let (func, entry, ids) = make_func_with_block(vec![MachInst::new(
            AArch64Opcode::AddRI,
            vec![vreg(0), vreg(99), MachOperand::Imm(5)],
        )]);
        let def_map = def_map_after(&func, entry, 0);
        let ctx = MatchCtx {
            inst: func.inst(ids[0]),
            inst_id: ids[0],
            block_id: entry,
            func: &func,
            def_map: &def_map,
        };
        assert!(ctx.definer_of(1).is_none(), "block param has no definer");
        // Operand 0 is the dst VReg — also not yet defined.
        assert!(ctx.definer_of(0).is_none());
        // Operand 2 is an Imm — never a definer.
        assert!(ctx.definer_of(2).is_none());
    }

    #[test]
    fn definer_of_sees_movi_before_add() {
        // movi v1, #5
        // add  v2, v1, #3
        //
        // When evaluating `add`, operand[1] is v1 — its definer is movi.
        let (func, entry, ids) = make_func_with_block(vec![
            MachInst::new(
                AArch64Opcode::MovI,
                vec![vreg(1), MachOperand::Imm(5)],
            ),
            MachInst::new(
                AArch64Opcode::AddRI,
                vec![vreg(2), vreg(1), MachOperand::Imm(3)],
            ),
        ]);
        // Def map seen by the second instruction: one entry (v1 -> ids[0]).
        let def_map = def_map_after(&func, entry, 1);
        let ctx = MatchCtx {
            inst: func.inst(ids[1]),
            inst_id: ids[1],
            block_id: entry,
            func: &func,
            def_map: &def_map,
        };
        let def = ctx.definer_of(1).expect("definer should exist");
        assert_eq!(def.opcode, AArch64Opcode::MovI);
        assert!(matches!(def.operands[1], MachOperand::Imm(5)));
    }

    #[test]
    fn definer_of_intervening_redef_picks_newest() {
        // movi v1, #5          ; first writer
        // movi v1, #9          ; redefinition — this is what use sees
        // add  v2, v1, #0      ; use: definer is the second movi
        let (func, entry, ids) = make_func_with_block(vec![
            MachInst::new(
                AArch64Opcode::MovI,
                vec![vreg(1), MachOperand::Imm(5)],
            ),
            MachInst::new(
                AArch64Opcode::MovI,
                vec![vreg(1), MachOperand::Imm(9)],
            ),
            MachInst::new(
                AArch64Opcode::AddRI,
                vec![vreg(2), vreg(1), MachOperand::Imm(0)],
            ),
        ]);
        // When the Add executes, two writers have been seen — last wins.
        let def_map = def_map_after(&func, entry, 2);
        let ctx = MatchCtx {
            inst: func.inst(ids[2]),
            inst_id: ids[2],
            block_id: entry,
            func: &func,
            def_map: &def_map,
        };
        let def = ctx.definer_of(1).expect("definer should exist");
        assert_eq!(def.opcode, AArch64Opcode::MovI);
        // Second movi has imm=9, not 5.
        assert!(matches!(def.operands[1], MachOperand::Imm(9)));
        assert_eq!(
            *ctx.def_map.get(&1).unwrap(),
            ids[1],
            "newest writer must be id of the second movi"
        );
    }

    #[test]
    fn operand_def_and_definer_of_agree() {
        // Confirm the alias returns identical output.
        let (func, entry, ids) = make_func_with_block(vec![
            MachInst::new(AArch64Opcode::MovI, vec![vreg(1), MachOperand::Imm(7)]),
            MachInst::new(
                AArch64Opcode::AddRI,
                vec![vreg(2), vreg(1), MachOperand::Imm(1)],
            ),
        ]);
        let def_map = def_map_after(&func, entry, 1);
        let ctx = MatchCtx {
            inst: func.inst(ids[1]),
            inst_id: ids[1],
            block_id: entry,
            func: &func,
            def_map: &def_map,
        };
        let a = ctx.operand_def(1).map(|i| i.opcode);
        let b = ctx.definer_of(1).map(|i| i.opcode);
        assert_eq!(a, b);
    }
}
