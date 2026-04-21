// llvm2-opt - Scalar Replacement of Aggregates
//
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Copyright 2026 Andrew Yates | License: Apache-2.0

//! Scalar Replacement of Aggregates (SROA) for machine-level IR.
//!
//! SROA identifies stack slots whose address never escapes the function and
//! whose uses are limited to simple load/store patterns, and rewrites those
//! loads/stores into pure vreg-to-vreg copies. The stack slot storage is left
//! in the function (regalloc / frame lowering may prune unused slots later),
//! but the scalar SSA replacement eliminates stack traffic for the slot.
//!
//! # Scope (Phase 2b — issue #391)
//!
//! The current implementation targets the "textbook" pattern that frontends
//! emit for struct/tuple locals that are never address-escaped:
//!
//! ```text
//!   root = AddPCRel SP, StackSlot(N)          ; address of slot
//!   ; optional per-field offset derivation:
//!   f0   = MovR root                          ; field 0 (offset 0)
//!   f1   = AddRI root, #K                     ; field 1 (offset K)
//!   STR  value, f0, #0                        ; store to field
//!   val  = LDR       f1, #0                   ; load from field
//! ```
//!
//! A slot is rewritten iff **every** reference to the root vreg flows into
//! exactly one of:
//!
//! 1. a `MovR` alias (offset +0) that is itself a "derived address",
//! 2. an `AddRI` immediate offset that is itself a "derived address",
//! 3. a `LdrRI`/`StrRI` where the root/derived vreg is the address base
//!    (and the inner immediate is a known constant).
//!
//! Any use outside that envelope (call operand, compare, arithmetic other
//! than AddRI, store of the address as a value, out-of-pass block argument,
//! etc.) marks the slot **escaped** and the pass leaves it alone.
//!
//! When a slot is accepted, each distinct `(slot_byte_offset, opcode)` load
//! or store is replaced by a vreg move against a dedicated scalar vreg.
//! Intermediate `AddPCRel` / `AddRI` / `MovR` root uses become dead and are
//! removed in the same pass; subsequent DCE and copy-prop clean up the moves.
//!
//! # Non-goals
//!
//! * SROA for stack slots accessed via register-indexed addressing
//!   (`LdrRO`/`StrRO`) — array locals stay on the stack for now.
//! * SROA across aliased stack slots (partial overlap or dynamic offset) —
//!   tracked separately for future work.
//! * SROA for slots whose address crosses block boundaries via phi/block
//!   parameters — the pass bails out conservatively when it sees a use it
//!   doesn't understand.
//!
//! # Safety / correctness
//!
//! The pass is a straightforward local rewrite:
//!
//! * Only stores keep "observable" effects at the ISA level; we replace them
//!   with vreg moves (which are not observable from outside the function).
//!   Because we also eliminate every matching load from the same slot, the
//!   semantics of the function modulo the slot's memory contents are
//!   preserved.
//! * The escape analysis is conservative: any unrecognised instruction
//!   that mentions the root (or a derived address) triggers a bail-out.
//! * We never rewrite across a slot whose load/store widths disagree for
//!   the same byte offset, because mixing LDR/LDRB at offset 0 would
//!   require a bitcast we can't synthesise here.
//! * Orphan instructions are removed in a single sweep after rewriting, so
//!   the block instruction vectors stay consistent with `func.insts`.
//!
//! Reference: `designs/2026-04-18-aggregate-lowering.md` Phase 2b.

use std::collections::{HashMap, HashSet};

use llvm2_ir::{
    AArch64Opcode, InstId, MachFunction, MachInst, MachOperand, ProofAnnotation, StackSlotId, VReg,
};

use crate::pass_manager::MachinePass;

/// Scalar Replacement of Aggregates pass.
///
/// Runs at `O1+`; see [`OptimizationPipeline`](crate::pipeline::OptimizationPipeline)
/// for wiring.
#[derive(Debug, Default)]
pub struct ScalarReplacementOfAggregates;

impl MachinePass for ScalarReplacementOfAggregates {
    fn name(&self) -> &str {
        "sroa"
    }

    fn run(&mut self, func: &mut MachFunction) -> bool {
        // 1. Build vreg -> definer instruction map so we can recognise a
        //    load/store base that came from an AddPCRel or AddRI chain.
        let def_of = build_vreg_def_map(func);

        // 2. Count every use of every vreg across the function. This lets us
        //    confirm that when we eliminate a root/derived vreg, the def is
        //    truly dead (no surprise reader).
        let use_count = collect_vreg_use_counts(func);

        // 3. Find every "root" AddPCRel that materialises a stack-slot address.
        let roots = collect_stack_slot_roots(func);

        if roots.is_empty() {
            return false;
        }

        let mut rewrites: Vec<SlotRewrite> = Vec::new();

        // 4. For each slot, try to collect all accesses. Bail out (skip the
        //    slot) the moment we see a use we don't recognise.
        'slot_loop: for (slot, root_insts) in group_roots_by_slot(&roots) {
            let mut plan = SlotPlan::new(slot);

            for root_inst in &root_insts {
                let root_vreg = match def_vreg(func.inst(*root_inst)) {
                    Some(v) => v,
                    None => continue 'slot_loop,
                };
                if !plan.add_root(root_vreg, *root_inst) {
                    continue 'slot_loop;
                }
                if !trace_addr_uses(
                    func,
                    &def_of,
                    root_vreg.id,
                    0,
                    &mut plan,
                ) {
                    continue 'slot_loop;
                }
            }

            // Every root def is covered, every derived vreg is used only by
            // recognised addresses. Now confirm we touched **every** use of
            // every root/derived vreg: if `use_count` disagrees with what we
            // walked, there's an unknown reader (e.g., a backward edge we
            // didn't revisit) and we must bail.
            if !plan.all_uses_covered(&use_count) {
                continue 'slot_loop;
            }

            // Collect the rewrite entries.
            if let Some(r) = plan.finalise(func) {
                rewrites.push(r);
            }
        }

        if rewrites.is_empty() {
            return false;
        }

        apply_rewrites(func, &rewrites)
    }
}

// ---------------------------------------------------------------------------
// Helpers: vreg bookkeeping
// ---------------------------------------------------------------------------

/// Return the defining vreg (operand[0]) if the instruction's first operand
/// is a VReg; otherwise `None`. SROA only considers instructions whose
/// convention is "first operand is the destination".
fn def_vreg(inst: &MachInst) -> Option<VReg> {
    inst.operands.first().and_then(|op| match op {
        MachOperand::VReg(v) => Some(*v),
        _ => None,
    })
}

/// Map every defined vreg id to the instruction that defined it.
fn build_vreg_def_map(func: &MachFunction) -> HashMap<u32, InstId> {
    let mut out = HashMap::new();
    for block_id in &func.block_order {
        let block = func.block(*block_id);
        for &inst_id in &block.insts {
            let inst = func.inst(inst_id);
            if produces_value(inst) {
                if let Some(v) = def_vreg(inst) {
                    out.insert(v.id, inst_id);
                }
            }
        }
    }
    out
}

/// Count how many times each vreg appears as a *source* operand.
///
/// Uses the same convention as DCE: if the instruction produces a value,
/// operands[0] is the def; otherwise every operand is a use.
fn collect_vreg_use_counts(func: &MachFunction) -> HashMap<u32, u32> {
    let mut counts: HashMap<u32, u32> = HashMap::new();
    for block_id in &func.block_order {
        let block = func.block(*block_id);
        for &inst_id in &block.insts {
            let inst = func.inst(inst_id);
            let start = if produces_value(inst) { 1 } else { 0 };
            for op in &inst.operands[start..] {
                if let MachOperand::VReg(v) = op {
                    *counts.entry(v.id).or_insert(0) += 1;
                }
            }
        }
    }
    counts
}

/// An instruction produces a value iff its `InstFlags` predicate says so.
/// Mirrors `effects::inst_produces_value` but we duplicate locally to keep
/// dependencies minimal.
fn produces_value(inst: &MachInst) -> bool {
    crate::effects::inst_produces_value(inst)
}

// ---------------------------------------------------------------------------
// Root discovery
// ---------------------------------------------------------------------------

/// Collect every `AddPCRel` instruction whose third operand is a StackSlot.
/// These are the ISel-emitted roots for `Opcode::StackAddr { slot }`.
fn collect_stack_slot_roots(func: &MachFunction) -> Vec<(StackSlotId, InstId)> {
    let mut out = Vec::new();
    for block_id in &func.block_order {
        let block = func.block(*block_id);
        for &inst_id in &block.insts {
            let inst = func.inst(inst_id);
            if inst.opcode != AArch64Opcode::AddPCRel {
                continue;
            }
            // AddPCRel operands: [VReg(dst), PReg(SP), StackSlot(N)].
            if let Some(MachOperand::StackSlot(slot)) = inst.operands.get(2) {
                out.push((*slot, inst_id));
            }
        }
    }
    out
}

/// Group the flat list of roots by `StackSlotId`, preserving definition order.
fn group_roots_by_slot(
    roots: &[(StackSlotId, InstId)],
) -> Vec<(StackSlotId, Vec<InstId>)> {
    let mut order: Vec<StackSlotId> = Vec::new();
    let mut by_slot: HashMap<StackSlotId, Vec<InstId>> = HashMap::new();
    for (slot, inst) in roots {
        if !by_slot.contains_key(slot) {
            order.push(*slot);
        }
        by_slot.entry(*slot).or_default().push(*inst);
    }
    order.into_iter().map(|s| (s, by_slot.remove(&s).unwrap())).collect()
}

// ---------------------------------------------------------------------------
// Address-use tracing
// ---------------------------------------------------------------------------

/// A single access instruction (load or store) against a slot.
#[derive(Debug, Clone)]
struct Access {
    inst_id: InstId,
    byte_offset: i64,
    is_load: bool,
    /// For loads, the destination vreg (operand[0]). For stores, the source
    /// vreg holding the stored value (operand[0]).
    value_vreg: VReg,
    /// Opcode bucket for width-consistency checks.
    opcode: AArch64Opcode,
}

/// Per-slot rewrite plan accumulated during use tracing.
struct SlotPlan {
    slot: StackSlotId,
    /// All root AddPCRel instructions for this slot.
    roots: Vec<InstId>,
    /// Derived-address instructions (`AddRI` and `MovR`) to remove on commit.
    derived_defs: Vec<InstId>,
    /// Every vreg we took ownership of (root + derived). Used to confirm no
    /// outside reader exists.
    owned_vregs: HashMap<u32, u32 /* observed use count */>,
    /// Loads and stores we will rewrite.
    accesses: Vec<Access>,
    /// Set to true if tracing has found a reason to abort.
    aborted: bool,
    /// True when the slot address is passed to a `Bl` callee tagged with
    /// `ProofAnnotation::Pure` (#456 partial-escape).
    ///
    /// In this mode the slot is *not* fully scalar-replaceable: the Bl reads
    /// the slot's spilled bytes, so the root/derived-address defs and the
    /// `StrRI` stores must remain live. What we *can* do is redirect each
    /// in-function `LdrRI` against the slot to read the shadow scalar vreg
    /// written by the preceding `StrRI` at the same offset, eliminating the
    /// load/store round-trip inside the caller.
    partial_escape: bool,
}

impl SlotPlan {
    fn new(slot: StackSlotId) -> Self {
        Self {
            slot,
            roots: Vec::new(),
            derived_defs: Vec::new(),
            owned_vregs: HashMap::new(),
            accesses: Vec::new(),
            aborted: false,
            partial_escape: false,
        }
    }

    fn add_root(&mut self, vreg: VReg, inst: InstId) -> bool {
        if self.owned_vregs.contains_key(&vreg.id) {
            // Two AddPCRel instructions produced the same vreg id — unexpected.
            return false;
        }
        self.owned_vregs.insert(vreg.id, 0);
        self.roots.push(inst);
        true
    }

    fn add_derived(&mut self, vreg: VReg, inst: InstId) -> bool {
        if self.owned_vregs.contains_key(&vreg.id) {
            return false;
        }
        self.owned_vregs.insert(vreg.id, 0);
        self.derived_defs.push(inst);
        true
    }

    fn note_use(&mut self, vreg_id: u32) {
        if let Some(c) = self.owned_vregs.get_mut(&vreg_id) {
            *c = c.saturating_add(1);
        }
    }

    fn abort(&mut self) {
        self.aborted = true;
    }

    fn all_uses_covered(&self, global: &HashMap<u32, u32>) -> bool {
        if self.aborted {
            return false;
        }
        for (vreg_id, walked) in &self.owned_vregs {
            let global_count = global.get(vreg_id).copied().unwrap_or(0);
            if global_count != *walked {
                return false;
            }
        }
        true
    }

    /// Finalise: return a rewrite, or `None` if we have nothing to do.
    fn finalise(&mut self, func: &MachFunction) -> Option<SlotRewrite> {
        if self.aborted {
            return None;
        }
        if self.accesses.is_empty() && self.derived_defs.is_empty() && self.roots.is_empty() {
            return None;
        }

        // Width-consistency check: every (offset, opcode) pair must stay a
        // single opcode bucket. Mixing LdrRI with LdrbRI at the same offset
        // would need a bitcast we don't emit here.
        let mut opcode_at_offset: HashMap<i64, AArch64Opcode> = HashMap::new();
        for a in &self.accesses {
            if let Some(existing) = opcode_at_offset.get(&a.byte_offset) {
                if !accesses_compatible(*existing, a.opcode) {
                    return None;
                }
            } else {
                opcode_at_offset.insert(a.byte_offset, access_canonical(a.opcode));
            }
        }

        // Partial-escape (#456) requires every load to be backed by a store
        // at the same offset within this pass's visibility. Without a STR
        // writing the shadow scalar before the Bl, the LDR->MovR rewrite
        // would read an uninitialised vreg.
        if self.partial_escape {
            let mut offsets_with_store: HashSet<i64> = HashSet::new();
            for a in &self.accesses {
                if !a.is_load {
                    offsets_with_store.insert(a.byte_offset);
                }
            }
            for a in &self.accesses {
                if a.is_load && !offsets_with_store.contains(&a.byte_offset) {
                    // LDR at an offset with no preceding STR in this block
                    // would read stale/undef bytes after the rewrite; give up.
                    return None;
                }
            }
        }

        // Allocate one scalar "shadow" vreg per access offset, matching the
        // destination register class of the first load at that offset (or a
        // store's source class).
        let mut scalar_vreg: HashMap<i64, VReg> = HashMap::new();
        let mut next_vreg = func.next_vreg;
        for a in &self.accesses {
            if scalar_vreg.contains_key(&a.byte_offset) {
                continue;
            }
            let cls = a.value_vreg.class;
            scalar_vreg.insert(a.byte_offset, VReg::new(next_vreg, cls));
            next_vreg += 1;
        }

        Some(SlotRewrite {
            slot: self.slot,
            roots: std::mem::take(&mut self.roots),
            derived_defs: std::mem::take(&mut self.derived_defs),
            accesses: std::mem::take(&mut self.accesses),
            scalar_vreg,
            next_vreg,
            partial_escape: self.partial_escape,
        })
    }
}

/// Are two load/store opcodes compatible (same width and signedness)?
fn accesses_compatible(a: AArch64Opcode, b: AArch64Opcode) -> bool {
    access_canonical(a) == access_canonical(b)
}

/// Canonicalise loads and stores to the same opcode bucket so a load and a
/// matching store at the same offset compare equal.
fn access_canonical(op: AArch64Opcode) -> AArch64Opcode {
    use AArch64Opcode::*;
    match op {
        LdrRI | StrRI => LdrRI,
        LdrbRI | StrbRI => LdrbRI,
        LdrhRI | StrhRI => LdrhRI,
        other => other,
    }
}

/// The rewrite description produced once we have confirmed the slot is safe.
#[derive(Debug)]
struct SlotRewrite {
    #[allow(dead_code)]
    slot: StackSlotId,
    roots: Vec<InstId>,
    derived_defs: Vec<InstId>,
    accesses: Vec<Access>,
    scalar_vreg: HashMap<i64, VReg>,
    /// Partial-escape mode (#456): the slot's address is passed to a pure
    /// Bl. In this mode commit keeps all roots, derived defs, and STRs alive
    /// (the Bl still needs the spilled bytes) and only rewrites LDRs.
    partial_escape: bool,
    next_vreg: u32,
}

/// Walk every use of `vreg_id` and classify it as:
///
/// * a `LdrRI` / `StrRI` against (root + `base_offset` + inner_imm) — add to
///   the plan,
/// * an `AddRI` with an immediate — recurse with offset += imm,
/// * a `MovR` / `Copy` — recurse with unchanged offset,
/// * anything else — abort.
///
/// Returns `false` and marks the plan aborted when it sees a use it doesn't
/// recognise or a use it already visited (cycle).
fn trace_addr_uses(
    func: &MachFunction,
    def_of: &HashMap<u32, InstId>,
    vreg_id: u32,
    base_offset: i64,
    plan: &mut SlotPlan,
) -> bool {
    // Find every instruction using `vreg_id` as a source.
    let users = collect_users_of(func, vreg_id);
    for user_id in users {
        let inst = func.inst(user_id);
        plan.note_use(vreg_id);

        // Ignore the definer itself — it uses its own def only in the
        // degenerate case of `a = a`, which we treat as "too weird".
        if let Some(defining) = def_of.get(&vreg_id) {
            if *defining == user_id {
                // Unreachable in well-formed IR; treat as abort.
                plan.abort();
                return false;
            }
        }

        match inst.opcode {
            AArch64Opcode::AddRI => {
                // Must be: AddRI dst, base_vreg, imm. Base must be our vreg,
                // imm must be an immediate.
                if !is_addr_user_addri(inst, vreg_id) {
                    plan.abort();
                    return false;
                }
                let imm = match inst.operands.get(2) {
                    Some(MachOperand::Imm(v)) => *v,
                    _ => {
                        plan.abort();
                        return false;
                    }
                };
                let dst = match def_vreg(inst) {
                    Some(v) => v,
                    None => {
                        plan.abort();
                        return false;
                    }
                };
                if !plan.add_derived(dst, user_id) {
                    plan.abort();
                    return false;
                }
                if !trace_addr_uses(func, def_of, dst.id, base_offset + imm, plan) {
                    return false;
                }
            }
            AArch64Opcode::MovR | AArch64Opcode::Copy => {
                // MovR/Copy dst, src: alias. Our vreg must be the source.
                if !is_mov_from_source(inst, vreg_id) {
                    plan.abort();
                    return false;
                }
                match inst.operands.first() {
                    Some(MachOperand::VReg(dst)) => {
                        // Internal alias — recurse into the derived vreg.
                        let dst = *dst;
                        if !plan.add_derived(dst, user_id) {
                            plan.abort();
                            return false;
                        }
                        if !trace_addr_uses(func, def_of, dst.id, base_offset, plan) {
                            return false;
                        }
                    }
                    Some(MachOperand::PReg(_)) => {
                        // Copy-to-PReg is ABI arg marshalling for a following
                        // call. If the same block has a subsequent `Bl` tagged
                        // `ProofAnnotation::Pure`, the slot address is used
                        // only as a call argument that does not escape the
                        // callee — we can rewrite in-function LDRs against
                        // the slot but must leave the root/derived defs and
                        // STRs alone (partial-escape, #456). Otherwise, this
                        // is a normal escape and we bail.
                        if !copy_preg_reaches_pure_bl(func, user_id) {
                            plan.abort();
                            return false;
                        }
                        plan.partial_escape = true;
                        // The Copy itself is a legitimate use of our vreg —
                        // record it so the use-count check passes without
                        // adding any derived vreg.
                    }
                    _ => {
                        plan.abort();
                        return false;
                    }
                }
            }
            AArch64Opcode::LdrRI
            | AArch64Opcode::LdrbRI
            | AArch64Opcode::LdrhRI
            | AArch64Opcode::LdrsbRI
            | AArch64Opcode::LdrshRI => {
                // Load format: [dst, base, imm]. Base must be our vreg.
                if !is_mem_base(inst, 1, vreg_id) {
                    plan.abort();
                    return false;
                }
                let imm = match inst.operands.get(2) {
                    Some(MachOperand::Imm(v)) => *v,
                    _ => {
                        plan.abort();
                        return false;
                    }
                };
                let dst = match def_vreg(inst) {
                    Some(v) => v,
                    None => {
                        plan.abort();
                        return false;
                    }
                };
                plan.accesses.push(Access {
                    inst_id: user_id,
                    byte_offset: base_offset + imm,
                    is_load: true,
                    value_vreg: dst,
                    opcode: inst.opcode,
                });
            }
            AArch64Opcode::StrRI | AArch64Opcode::StrbRI | AArch64Opcode::StrhRI => {
                // Store format: [value, base, imm]. Base must be our vreg
                // (operand[1]); crucially, operand[0] (the value) must NOT
                // be our vreg — storing the address itself is an escape.
                if !is_mem_base(inst, 1, vreg_id) {
                    plan.abort();
                    return false;
                }
                if matches!(inst.operands.first(), Some(MachOperand::VReg(v)) if v.id == vreg_id)
                {
                    plan.abort();
                    return false;
                }
                let imm = match inst.operands.get(2) {
                    Some(MachOperand::Imm(v)) => *v,
                    _ => {
                        plan.abort();
                        return false;
                    }
                };
                let value = match inst.operands.first() {
                    Some(MachOperand::VReg(v)) => *v,
                    _ => {
                        plan.abort();
                        return false;
                    }
                };
                plan.accesses.push(Access {
                    inst_id: user_id,
                    byte_offset: base_offset + imm,
                    is_load: false,
                    value_vreg: value,
                    opcode: inst.opcode,
                });
            }
            _ => {
                // Any other opcode touching the slot address means escape.
                plan.abort();
                return false;
            }
        }
    }
    true
}

/// Collect InstIds that use `vreg_id` as a *source* operand.
fn collect_users_of(func: &MachFunction, vreg_id: u32) -> Vec<InstId> {
    let mut out = Vec::new();
    for block_id in &func.block_order {
        let block = func.block(*block_id);
        for &inst_id in &block.insts {
            let inst = func.inst(inst_id);
            let start = if produces_value(inst) { 1 } else { 0 };
            for op in &inst.operands[start..] {
                if let MachOperand::VReg(v) = op {
                    if v.id == vreg_id {
                        out.push(inst_id);
                        break;
                    }
                }
            }
        }
    }
    out
}

/// Is this `AddRI dst, base, imm` with `base == vreg_id`?
fn is_addr_user_addri(inst: &MachInst, vreg_id: u32) -> bool {
    if inst.operands.len() != 3 {
        return false;
    }
    let base_ok = matches!(inst.operands.get(1), Some(MachOperand::VReg(v)) if v.id == vreg_id);
    let imm_ok = matches!(inst.operands.get(2), Some(MachOperand::Imm(_)));
    let dst_ok = matches!(inst.operands.first(), Some(MachOperand::VReg(_)));
    base_ok && imm_ok && dst_ok
}

/// Is this `MovR dst, src` / `Copy dst, src` whose *source* is `vreg_id`,
/// irrespective of whether the destination is a VReg (internal alias) or a
/// PReg (ABI arg marshalling, partial-escape #456)?
fn is_mov_from_source(inst: &MachInst, vreg_id: u32) -> bool {
    if inst.operands.len() != 2 {
        return false;
    }
    let src_ok = matches!(inst.operands.get(1), Some(MachOperand::VReg(v)) if v.id == vreg_id);
    let dst_ok = matches!(
        inst.operands.first(),
        Some(MachOperand::VReg(_)) | Some(MachOperand::PReg(_))
    );
    src_ok && dst_ok
}

/// Walk the block containing `copy_id` forward; return `true` iff the next
/// `Bl` ISA instruction in that block carries `proof == Some(Pure)`.
///
/// This is the partial-escape predicate for SROA (#456): a Copy from a slot
/// address into a PReg is non-escaping *only* when it feeds a pure call in
/// the same basic block. We conservatively require no intervening Bl/Blr with
/// non-pure proof before the pure Bl.
fn copy_preg_reaches_pure_bl(func: &MachFunction, copy_id: InstId) -> bool {
    for block_id in &func.block_order {
        let block = func.block(*block_id);
        let Some(pos) = block.insts.iter().position(|id| *id == copy_id) else {
            continue;
        };
        for &next_id in &block.insts[pos + 1..] {
            let next = func.inst(next_id);
            match next.opcode {
                AArch64Opcode::Bl => {
                    return next.proof == Some(ProofAnnotation::Pure);
                }
                AArch64Opcode::Blr => {
                    // Indirect call in the way — can't prove purity of the
                    // target, so we must conservatively treat the slot as
                    // escaped.
                    return false;
                }
                _ => {}
            }
        }
        // Copy was in this block but no Bl follows — treat as escape.
        return false;
    }
    false
}

/// For a load `[dst, base, imm]` or store `[val, base, imm]`: operand at
/// `base_idx` must be `VReg(vreg_id)`.
fn is_mem_base(inst: &MachInst, base_idx: usize, vreg_id: u32) -> bool {
    if inst.operands.len() != 3 {
        return false;
    }
    matches!(inst.operands.get(base_idx), Some(MachOperand::VReg(v)) if v.id == vreg_id)
}

// ---------------------------------------------------------------------------
// Rewrite application
// ---------------------------------------------------------------------------

/// Apply the accumulated rewrites to the function in place.
///
/// Returns `true` iff any instruction was modified or removed.
fn apply_rewrites(func: &mut MachFunction, rewrites: &[SlotRewrite]) -> bool {
    let mut changed = false;

    // Flatten "dead" instruction ids (roots + derived) into a single set so
    // one block pass can remove them all.
    let mut dead: HashSet<InstId> = HashSet::new();
    // Bump the function's vreg counter once, using the max of all rewrites.
    let mut max_next = func.next_vreg;

    // For partial-escape slots (#456), each STR must be preceded by a new
    // `MovR scalar_vreg, value` that mirrors the stored bytes into the shadow
    // vreg. We collect these insertions here and splice them in afterwards,
    // so we only pay the block rebuild cost once.
    //
    // Keyed by `original StrRI InstId` -> newly-allocated `MovR InstId`.
    let mut insert_before: HashMap<InstId, InstId> = HashMap::new();

    // Rewrite loads/stores first — this modifies `func.insts` in place.
    for rw in rewrites {
        max_next = max_next.max(rw.next_vreg);

        for acc in &rw.accesses {
            let target_vreg = rw.scalar_vreg[&acc.byte_offset];
            if rw.partial_escape {
                if acc.is_load {
                    // LdrRI dst, base, imm -> MovR dst, scalar_vreg
                    // (LDR is eliminated; STR still runs so the callee sees
                    // the spilled bytes.)
                    let inst = func.inst_mut(acc.inst_id);
                    inst.opcode = AArch64Opcode::MovR;
                    inst.operands = vec![
                        MachOperand::VReg(acc.value_vreg),
                        MachOperand::VReg(target_vreg),
                    ];
                    changed = true;
                } else {
                    // Insert a new `MovR scalar_vreg, value` *before* the
                    // original StrRI. The STR itself is left intact — the
                    // pure callee reads the spilled bytes.
                    let mov = MachInst::new(
                        AArch64Opcode::MovR,
                        vec![
                            MachOperand::VReg(target_vreg),
                            MachOperand::VReg(acc.value_vreg),
                        ],
                    );
                    let new_id = func.push_inst(mov);
                    insert_before.insert(acc.inst_id, new_id);
                    changed = true;
                }
            } else {
                let inst = func.inst_mut(acc.inst_id);
                if acc.is_load {
                    // LdrRI dst, base, imm -> MovR dst, scalar_vreg
                    inst.opcode = AArch64Opcode::MovR;
                    inst.operands = vec![
                        MachOperand::VReg(acc.value_vreg),
                        MachOperand::VReg(target_vreg),
                    ];
                } else {
                    // StrRI value, base, imm -> MovR scalar_vreg, value
                    inst.opcode = AArch64Opcode::MovR;
                    inst.operands = vec![
                        MachOperand::VReg(target_vreg),
                        MachOperand::VReg(acc.value_vreg),
                    ];
                }
                changed = true;
            }
        }

        if !rw.partial_escape {
            for id in &rw.roots {
                dead.insert(*id);
            }
            for id in &rw.derived_defs {
                dead.insert(*id);
            }
        }
        // Partial-escape: roots/derived defs stay alive (the Bl needs the
        // slot address in the ABI register), and the STRs stay alive too.
    }

    if !dead.is_empty() || !insert_before.is_empty() {
        for block_id in func.block_order.clone() {
            let block = func.block_mut(block_id);
            let before = block.insts.len();
            let mut new_insts: Vec<InstId> = Vec::with_capacity(block.insts.len());
            for &id in &block.insts {
                if let Some(&mov_id) = insert_before.get(&id) {
                    new_insts.push(mov_id);
                }
                if !dead.contains(&id) {
                    new_insts.push(id);
                }
            }
            if new_insts.len() != before {
                changed = true;
            }
            block.insts = new_insts;
        }
    }

    if max_next > func.next_vreg {
        func.next_vreg = max_next;
    }

    changed
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use llvm2_ir::{
        AArch64Opcode, BlockId, MachFunction, MachInst, MachOperand, RegClass, Signature,
        StackSlot, StackSlotId, VReg,
    };

    fn vreg(id: u32, class: RegClass) -> MachOperand {
        MachOperand::VReg(VReg::new(id, class))
    }
    fn g64(id: u32) -> MachOperand {
        vreg(id, RegClass::Gpr64)
    }
    fn g32(id: u32) -> MachOperand {
        vreg(id, RegClass::Gpr32)
    }
    fn imm(v: i64) -> MachOperand {
        MachOperand::Imm(v)
    }

    fn new_func() -> MachFunction {
        MachFunction::new(
            "sroa_test".to_string(),
            Signature::new(vec![], vec![]),
        )
    }

    /// Build a minimal entry block with SP + StackSlot(0) already allocated.
    fn with_slot(func: &mut MachFunction, size: u32, align: u32) -> StackSlotId {
        func.alloc_stack_slot(StackSlot::new(size, align))
    }

    fn push(func: &mut MachFunction, block: BlockId, inst: MachInst) -> InstId {
        let id = func.push_inst(inst);
        func.append_inst(block, id);
        id
    }

    /// Scenario: single stack slot of a struct `(i64, i64)`, store at +0,
    /// store at +8, load from +0. SROA should remove all memory instructions
    /// and replace them with register moves.
    #[test]
    fn struct_local_store_store_load_is_sroa_eliminated() {
        use llvm2_ir::regs::SP;

        let mut func = new_func();
        let entry = func.entry;
        let slot = with_slot(&mut func, 16, 8);
        func.next_vreg = 20; // reserve space for materialised values.

        // v10 = AddPCRel SP, StackSlot(0)    ; root
        push(
            &mut func,
            entry,
            MachInst::new(
                AArch64Opcode::AddPCRel,
                vec![g64(10), MachOperand::PReg(SP), MachOperand::StackSlot(slot)],
            ),
        );
        // v11 = AddRI v10, #8                ; field 1 offset
        push(
            &mut func,
            entry,
            MachInst::new(AArch64Opcode::AddRI, vec![g64(11), g64(10), imm(8)]),
        );
        // STR v0, v10, #0                    ; store field 0
        push(
            &mut func,
            entry,
            MachInst::new(AArch64Opcode::StrRI, vec![g64(0), g64(10), imm(0)]),
        );
        // STR v1, v11, #0                    ; store field 1
        push(
            &mut func,
            entry,
            MachInst::new(AArch64Opcode::StrRI, vec![g64(1), g64(11), imm(0)]),
        );
        // v2 = LDR v10, #0                   ; load field 0
        push(
            &mut func,
            entry,
            MachInst::new(AArch64Opcode::LdrRI, vec![g64(2), g64(10), imm(0)]),
        );
        // RET
        push(&mut func, entry, MachInst::new(AArch64Opcode::Ret, vec![]));

        let mut sroa = ScalarReplacementOfAggregates;
        let changed = sroa.run(&mut func);
        assert!(changed, "SROA should fire on single-slot, no-escape pattern");

        // Count memory ops / address ops remaining.
        let block = func.block(entry);
        let kinds: Vec<AArch64Opcode> =
            block.insts.iter().map(|id| func.inst(*id).opcode).collect();

        assert!(!kinds.contains(&AArch64Opcode::AddPCRel), "root AddPCRel removed");
        assert!(!kinds.contains(&AArch64Opcode::AddRI), "derived AddRI removed");
        assert!(!kinds.contains(&AArch64Opcode::LdrRI), "LDR replaced by MovR");
        assert!(!kinds.contains(&AArch64Opcode::StrRI), "STR replaced by MovR");
        // At least one MovR (store->move, load->move) remains.
        assert!(kinds.iter().any(|o| *o == AArch64Opcode::MovR));
        // Ret still present.
        assert!(kinds.contains(&AArch64Opcode::Ret));
    }

    /// Scenario: slot address escapes through a store (pointer stored into
    /// another memory location). SROA must bail out; IR unchanged.
    #[test]
    fn escaping_address_store_disables_sroa() {
        use llvm2_ir::regs::SP;

        let mut func = new_func();
        let entry = func.entry;
        let slot = with_slot(&mut func, 8, 8);
        func.next_vreg = 20;

        // v10 = AddPCRel SP, StackSlot(0)
        push(
            &mut func,
            entry,
            MachInst::new(
                AArch64Opcode::AddPCRel,
                vec![g64(10), MachOperand::PReg(SP), MachOperand::StackSlot(slot)],
            ),
        );
        // STR v10, v1, #0      ; store the address v10 as a value! (escape)
        push(
            &mut func,
            entry,
            MachInst::new(AArch64Opcode::StrRI, vec![g64(10), g64(1), imm(0)]),
        );
        push(&mut func, entry, MachInst::new(AArch64Opcode::Ret, vec![]));

        let kinds_before: Vec<AArch64Opcode> = func
            .block(entry)
            .insts
            .iter()
            .map(|id| func.inst(*id).opcode)
            .collect();

        let mut sroa = ScalarReplacementOfAggregates;
        let changed = sroa.run(&mut func);
        assert!(!changed, "SROA must decline when address escapes via store");
        let kinds_after: Vec<AArch64Opcode> = func
            .block(entry)
            .insts
            .iter()
            .map(|id| func.inst(*id).opcode)
            .collect();
        assert_eq!(kinds_before, kinds_after, "IR unchanged on escape");
    }

    /// Scenario: slot address is passed to a call (escape via argument).
    #[test]
    fn escaping_address_to_call_disables_sroa() {
        use llvm2_ir::regs::SP;

        let mut func = new_func();
        let entry = func.entry;
        let slot = with_slot(&mut func, 8, 8);
        func.next_vreg = 20;

        // v10 = AddPCRel SP, StackSlot(0)
        push(
            &mut func,
            entry,
            MachInst::new(
                AArch64Opcode::AddPCRel,
                vec![g64(10), MachOperand::PReg(SP), MachOperand::StackSlot(slot)],
            ),
        );
        // BL callee, v10      (pretend `Bl` consumes v10 as an argument)
        push(
            &mut func,
            entry,
            MachInst::new(
                AArch64Opcode::Bl,
                vec![MachOperand::Symbol("callee".to_string()), g64(10)],
            ),
        );
        push(&mut func, entry, MachInst::new(AArch64Opcode::Ret, vec![]));

        let mut sroa = ScalarReplacementOfAggregates;
        let changed = sroa.run(&mut func);
        assert!(!changed, "SROA must bail when address escapes to a call");
    }

    /// Scenario: two distinct slots, both SROA-eligible, independent rewrite.
    #[test]
    fn multiple_independent_slots_are_both_sroa_eliminated() {
        use llvm2_ir::regs::SP;

        let mut func = new_func();
        let entry = func.entry;
        let slot0 = with_slot(&mut func, 8, 8);
        let slot1 = with_slot(&mut func, 8, 8);
        func.next_vreg = 30;

        // slot0
        push(
            &mut func,
            entry,
            MachInst::new(
                AArch64Opcode::AddPCRel,
                vec![g64(10), MachOperand::PReg(SP), MachOperand::StackSlot(slot0)],
            ),
        );
        push(
            &mut func,
            entry,
            MachInst::new(AArch64Opcode::StrRI, vec![g64(0), g64(10), imm(0)]),
        );
        push(
            &mut func,
            entry,
            MachInst::new(AArch64Opcode::LdrRI, vec![g64(20), g64(10), imm(0)]),
        );
        // slot1
        push(
            &mut func,
            entry,
            MachInst::new(
                AArch64Opcode::AddPCRel,
                vec![g64(11), MachOperand::PReg(SP), MachOperand::StackSlot(slot1)],
            ),
        );
        push(
            &mut func,
            entry,
            MachInst::new(AArch64Opcode::StrRI, vec![g64(1), g64(11), imm(0)]),
        );
        push(
            &mut func,
            entry,
            MachInst::new(AArch64Opcode::LdrRI, vec![g64(21), g64(11), imm(0)]),
        );
        push(&mut func, entry, MachInst::new(AArch64Opcode::Ret, vec![]));

        let mut sroa = ScalarReplacementOfAggregates;
        let changed = sroa.run(&mut func);
        assert!(changed);

        let kinds: Vec<AArch64Opcode> = func
            .block(entry)
            .insts
            .iter()
            .map(|id| func.inst(*id).opcode)
            .collect();
        assert!(!kinds.contains(&AArch64Opcode::AddPCRel));
        assert!(!kinds.contains(&AArch64Opcode::LdrRI));
        assert!(!kinds.contains(&AArch64Opcode::StrRI));
    }

    /// Scenario: mixed widths at the same offset. A byte store then a word
    /// load from offset 0 would require a truncation we don't synthesise —
    /// SROA must bail out cleanly.
    #[test]
    fn mixed_widths_at_same_offset_disables_sroa() {
        use llvm2_ir::regs::SP;

        let mut func = new_func();
        let entry = func.entry;
        let slot = with_slot(&mut func, 8, 8);
        func.next_vreg = 20;

        // v10 = AddPCRel SP, StackSlot(0)
        push(
            &mut func,
            entry,
            MachInst::new(
                AArch64Opcode::AddPCRel,
                vec![g64(10), MachOperand::PReg(SP), MachOperand::StackSlot(slot)],
            ),
        );
        // STRB v0, v10, #0   ; byte write at offset 0
        push(
            &mut func,
            entry,
            MachInst::new(AArch64Opcode::StrbRI, vec![g32(0), g64(10), imm(0)]),
        );
        // v1 = LDR v10, #0   ; word read at offset 0 — INCONSISTENT!
        push(
            &mut func,
            entry,
            MachInst::new(AArch64Opcode::LdrRI, vec![g64(1), g64(10), imm(0)]),
        );
        push(&mut func, entry, MachInst::new(AArch64Opcode::Ret, vec![]));

        let mut sroa = ScalarReplacementOfAggregates;
        let changed = sroa.run(&mut func);
        assert!(
            !changed,
            "SROA must decline when loads/stores at the same offset have mismatched widths"
        );
    }

    /// Scenario: unknown use of the address (e.g., compare with zero).
    #[test]
    fn unknown_use_disables_sroa() {
        use llvm2_ir::regs::SP;

        let mut func = new_func();
        let entry = func.entry;
        let slot = with_slot(&mut func, 8, 8);
        func.next_vreg = 20;

        // v10 = AddPCRel SP, StackSlot(0)
        push(
            &mut func,
            entry,
            MachInst::new(
                AArch64Opcode::AddPCRel,
                vec![g64(10), MachOperand::PReg(SP), MachOperand::StackSlot(slot)],
            ),
        );
        // CMP v10, #0       ; unknown pattern — must bail
        push(
            &mut func,
            entry,
            MachInst::new(AArch64Opcode::CmpRI, vec![g64(10), imm(0)]),
        );
        push(&mut func, entry, MachInst::new(AArch64Opcode::Ret, vec![]));

        let mut sroa = ScalarReplacementOfAggregates;
        let changed = sroa.run(&mut func);
        assert!(!changed);
    }

    /// Scenario (#456 partial-escape): slot address is copied into `X0` and
    /// passed to a `Bl` tagged with `ProofAnnotation::Pure`. The in-function
    /// LDR must be rewritten to a MovR against the shadow scalar, while the
    /// STR (spill) and the root `AddPCRel` must stay live because the pure
    /// callee reads the spilled bytes.
    #[test]
    fn pure_call_enables_partial_escape_sroa() {
        use llvm2_ir::regs::{SP, X0};

        let mut func = new_func();
        let entry = func.entry;
        let slot = with_slot(&mut func, 8, 8);
        func.next_vreg = 20;

        // v10 = AddPCRel SP, StackSlot(0)    ; root
        let addpc = push(
            &mut func,
            entry,
            MachInst::new(
                AArch64Opcode::AddPCRel,
                vec![g64(10), MachOperand::PReg(SP), MachOperand::StackSlot(slot)],
            ),
        );
        // STR v0, v10, #0                    ; spill arg into slot
        let str_id = push(
            &mut func,
            entry,
            MachInst::new(AArch64Opcode::StrRI, vec![g64(0), g64(10), imm(0)]),
        );
        // v2 = LDR v10, #0                   ; in-function read of the slot
        let ldr_id = push(
            &mut func,
            entry,
            MachInst::new(AArch64Opcode::LdrRI, vec![g64(2), g64(10), imm(0)]),
        );
        // Copy X0, v10                       ; ABI arg marshalling
        push(
            &mut func,
            entry,
            MachInst::new(
                AArch64Opcode::Copy,
                vec![MachOperand::PReg(X0), g64(10)],
            ),
        );
        // Bl pure_callee                     ; proof = Pure
        push(
            &mut func,
            entry,
            MachInst::new(
                AArch64Opcode::Bl,
                vec![MachOperand::Symbol("pure_callee".to_string())],
            )
            .with_proof(ProofAnnotation::Pure),
        );
        push(&mut func, entry, MachInst::new(AArch64Opcode::Ret, vec![]));

        let mut sroa = ScalarReplacementOfAggregates;
        let changed = sroa.run(&mut func);
        assert!(changed, "pure-call partial-escape must fire");

        // Root AddPCRel must still be present (Bl needs the address).
        let block = func.block(entry);
        let opcodes_and_ids: Vec<(InstId, AArch64Opcode)> = block
            .insts
            .iter()
            .map(|id| (*id, func.inst(*id).opcode))
            .collect();
        assert!(
            opcodes_and_ids.iter().any(|(id, _)| *id == addpc),
            "AddPCRel root must survive partial-escape"
        );
        // Original StrRI must still be present (the pure callee reads the
        // spilled bytes).
        assert!(
            opcodes_and_ids.iter().any(|(id, _)| *id == str_id),
            "StrRI must survive partial-escape"
        );
        // LDR must have been rewritten to MovR.
        let ldr_inst = func.inst(ldr_id);
        assert_eq!(
            ldr_inst.opcode,
            AArch64Opcode::MovR,
            "LdrRI must become MovR from shadow scalar"
        );
        // A `MovR scalar, v0` must precede the StrRI — it is a newly-inserted
        // shadow mirror that makes the rewritten LDR read a defined vreg.
        let str_pos = opcodes_and_ids
            .iter()
            .position(|(id, _)| *id == str_id)
            .expect("str pos");
        assert!(
            str_pos >= 1,
            "a MovR scalar mirror must be spliced before the StrRI"
        );
        let mirror = &opcodes_and_ids[str_pos - 1];
        assert_eq!(mirror.1, AArch64Opcode::MovR);
    }

    /// Scenario (#456): slot address copied to X0 but the following call is
    /// NOT tagged Pure. SROA must fall back to the conservative escape path
    /// and decline.
    #[test]
    fn non_pure_call_still_escapes_sroa() {
        use llvm2_ir::regs::{SP, X0};

        let mut func = new_func();
        let entry = func.entry;
        let slot = with_slot(&mut func, 8, 8);
        func.next_vreg = 20;

        push(
            &mut func,
            entry,
            MachInst::new(
                AArch64Opcode::AddPCRel,
                vec![g64(10), MachOperand::PReg(SP), MachOperand::StackSlot(slot)],
            ),
        );
        push(
            &mut func,
            entry,
            MachInst::new(AArch64Opcode::StrRI, vec![g64(0), g64(10), imm(0)]),
        );
        push(
            &mut func,
            entry,
            MachInst::new(
                AArch64Opcode::Copy,
                vec![MachOperand::PReg(X0), g64(10)],
            ),
        );
        // Plain Bl, no proof — classic escape.
        push(
            &mut func,
            entry,
            MachInst::new(
                AArch64Opcode::Bl,
                vec![MachOperand::Symbol("impure_callee".to_string())],
            ),
        );
        push(&mut func, entry, MachInst::new(AArch64Opcode::Ret, vec![]));

        let mut sroa = ScalarReplacementOfAggregates;
        let changed = sroa.run(&mut func);
        assert!(
            !changed,
            "non-pure call must still escape — no partial-escape rewrite"
        );
    }
}
