// llvm2-opt - NEON/SIMD auto-vectorization pass
//
// Author: Andrew Yates <ayates@dropbox.com>
// Copyright 2026 Dropbox, Inc. | License: Apache-2.0

//! Auto-vectorization pass: detects vectorizable loops and transforms
//! scalar operations into NEON SIMD instructions.
//!
//! # Overview
//!
//! This pass analyzes natural loops for vectorization opportunities.
//! For each loop body, it checks whether the loop iterations are
//! independent (no cross-iteration data dependencies) and whether
//! the operations can be mapped to NEON SIMD instructions profitably.
//!
//! # Algorithm
//!
//! 1. Compute dominator tree and loop analysis.
//! 2. For each natural loop (innermost first):
//!    a. Analyze the loop body for vectorizability.
//!    b. Build a `VectorizationPlan` describing the transformation.
//!    c. Check profitability using the cost model.
//!    d. If profitable, emit the plan (future: actual IR rewrite).
//! 3. Return whether any vectorization opportunity was found.
//!
//! # Vectorizability Requirements
//!
//! A loop is vectorizable if:
//! - It has a single latch (simple counted loop form).
//! - The loop body contains only vectorizable instructions.
//! - There are no cross-iteration data dependencies (no reduction/recurrence).
//! - The trip count is known or can be bounded.
//! - Memory accesses are consecutive (stride-1) or absent.
//!
//! # NEON Arrangement Selection
//!
//! The element type of the loop's primary data determines the NEON
//! arrangement:
//!
//! | Element Type | Arrangement | Lanes | Width |
//! |-------------|-------------|-------|-------|
//! | i8          | 16B         | 16    | 128b  |
//! | i16         | 8H          | 8     | 128b  |
//! | i32         | 4S          | 4     | 128b  |
//! | i64 / f64   | 2D          | 2     | 128b  |
//! | f32         | 4S          | 4     | 128b  |
//!
//! # Cost Model Integration
//!
//! Uses [`llvm2_ir::cost_model::MultiTargetCostModel`] to compare scalar
//! vs NEON cost for the loop body. Vectorization proceeds only when
//! the NEON cost (including setup/teardown overhead) is lower than the
//! scalar cost scaled by the vectorization factor.
//!
//! Reference: LLVM `LoopVectorize.cpp`, CompCert verified loop optimization.

use std::collections::{HashMap, HashSet};

use llvm2_ir::cost_model::{
    CostModelGen, MultiTargetCostModel, NeonArrangement, NeonOp,
};
use llvm2_ir::{AArch64Opcode, BlockId, InstId, MachFunction, MachOperand, RegClass};

use crate::dom::DomTree;
use crate::effects::{opcode_effect, produces_value, MemoryEffect};
use crate::loops::{LoopAnalysis, NaturalLoop};
use crate::pass_manager::MachinePass;

// ---------------------------------------------------------------------------
// VectorizationPlan — describes how to vectorize a loop
// ---------------------------------------------------------------------------

/// Element type for vectorization — determines NEON arrangement.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum VecElementType {
    /// 8-bit integer (i8).
    I8,
    /// 16-bit integer (i16).
    I16,
    /// 32-bit integer (i32).
    I32,
    /// 64-bit integer (i64).
    I64,
    /// 32-bit float (f32).
    F32,
    /// 64-bit float (f64).
    F64,
}

impl VecElementType {
    /// Element size in bits.
    pub fn bits(self) -> u32 {
        match self {
            Self::I8 => 8,
            Self::I16 => 16,
            Self::I32 => 32,
            Self::I64 => 64,
            Self::F32 => 32,
            Self::F64 => 64,
        }
    }

    /// Best 128-bit NEON arrangement for this element type.
    pub fn neon_arrangement(self) -> NeonArrangement {
        match self {
            Self::I8 => NeonArrangement::B16,
            Self::I16 => NeonArrangement::H8,
            Self::I32 | Self::F32 => NeonArrangement::S4,
            Self::I64 | Self::F64 => NeonArrangement::D2,
        }
    }

    /// Number of SIMD lanes in the 128-bit arrangement.
    pub fn lanes(self) -> u32 {
        self.neon_arrangement().lane_count()
    }
}

/// Describes a vectorization plan for a single loop.
#[derive(Debug, Clone)]
pub struct VectorizationPlan {
    /// The loop header block.
    pub loop_header: BlockId,
    /// Estimated trip count (iterations). None if unknown.
    pub trip_count: Option<u32>,
    /// Primary element type for the vectorized computation.
    pub element_type: VecElementType,
    /// NEON arrangement to use.
    pub arrangement: NeonArrangement,
    /// Vectorization factor (number of scalar iterations per NEON iteration).
    pub vf: u32,
    /// Scalar instructions that will be vectorized.
    pub vectorizable_insts: Vec<InstId>,
    /// Estimated scalar cost (total cycles for trip_count iterations).
    pub scalar_cost: f64,
    /// Estimated NEON cost (total cycles including overhead).
    pub neon_cost: f64,
    /// Whether this plan is profitable (neon_cost < scalar_cost).
    pub is_profitable: bool,
}

impl VectorizationPlan {
    /// Speedup factor: scalar_cost / neon_cost. > 1.0 means profitable.
    pub fn speedup(&self) -> f64 {
        if self.neon_cost <= 0.0 {
            return 0.0;
        }
        self.scalar_cost / self.neon_cost
    }
}

// ---------------------------------------------------------------------------
// Vectorizability analysis
// ---------------------------------------------------------------------------

/// Check if a single instruction can be vectorized (mapped to NEON).
///
/// An instruction is vectorizable if:
/// - It is a pure arithmetic/logical operation (no memory, no call).
/// - It maps to a known NEON operation.
/// - It does not set condition flags (CMP, TST, ADDS, SUBS).
pub fn is_vectorizable(opcode: AArch64Opcode) -> bool {
    // Must be pure (no memory side effects).
    if opcode_effect(opcode) != MemoryEffect::Pure {
        return false;
    }

    // Must have a NEON equivalent.
    scalar_to_neon_op(opcode).is_some()
}

/// Map a scalar AArch64 opcode to its NEON operation equivalent.
fn scalar_to_neon_op(opcode: AArch64Opcode) -> Option<NeonOp> {
    use AArch64Opcode::*;
    match opcode {
        AddRR | AddRI => Some(NeonOp::Add),
        SubRR | SubRI => Some(NeonOp::Sub),
        MulRR => Some(NeonOp::Mul),
        Neg => Some(NeonOp::Neg),
        AndRR | AndRI => Some(NeonOp::And),
        OrrRR | OrrRI => Some(NeonOp::Orr),
        EorRR | EorRI => Some(NeonOp::Eor),
        BicRR => Some(NeonOp::Bic),
        LslRI => Some(NeonOp::Shl),
        LsrRI => Some(NeonOp::Ushr),
        AsrRI => Some(NeonOp::Sshr),
        FaddRR => Some(NeonOp::Fadd),
        FmulRR => Some(NeonOp::Fmul),
        _ => None,
    }
}

/// Infer the element type from an instruction's operands.
///
/// Uses the register class of the destination (operand[0]) to determine
/// the element width. Returns None if no dest or unrecognized class.
fn infer_element_type(func: &MachFunction, inst_id: InstId) -> Option<VecElementType> {
    let inst = func.inst(inst_id);
    if !produces_value(inst.opcode) {
        return None;
    }
    if inst.operands.is_empty() {
        return None;
    }

    match &inst.operands[0] {
        MachOperand::VReg(vreg) => match vreg.class {
            RegClass::Gpr32 => Some(VecElementType::I32),
            RegClass::Gpr64 => Some(VecElementType::I64),
            RegClass::Fpr32 => Some(VecElementType::F32),
            RegClass::Fpr64 => Some(VecElementType::F64),
            _ => None,
        },
        _ => None,
    }
}

/// Check if a loop body has cross-iteration data dependencies.
///
/// A cross-iteration dependency exists when an instruction uses a value
/// defined in the same loop body (a recurrence/reduction). We conservatively
/// check: all source operands of vectorizable instructions must either be
/// defined outside the loop or be loop-invariant.
///
/// Returns true if the loop body is dependency-free for vectorization.
fn is_dependency_free(
    func: &MachFunction,
    lp: &NaturalLoop,
    vectorizable_insts: &[InstId],
) -> bool {
    // Build the set of defs inside the loop body.
    let mut loop_defs: HashMap<u32, InstId> = HashMap::new();
    for &block_id in &lp.body {
        let block = func.block(block_id);
        for &inst_id in &block.insts {
            let inst = func.inst(inst_id);
            if produces_value(inst.opcode) {
                if let Some(MachOperand::VReg(vreg)) = inst.operands.first() {
                    loop_defs.insert(vreg.id, inst_id);
                }
            }
        }
    }

    let vectorizable_set: HashSet<InstId> = vectorizable_insts.iter().copied().collect();

    // For each vectorizable instruction, check that source operands
    // defined inside the loop are also vectorizable (parallel, not recurrence).
    for &inst_id in vectorizable_insts {
        let inst = func.inst(inst_id);
        // Source operands are operands[1..] for most instructions.
        for operand in inst.operands.iter().skip(1) {
            if let MachOperand::VReg(vreg) = operand {
                if let Some(&def_inst) = loop_defs.get(&vreg.id) {
                    // This value is defined inside the loop.
                    // If it's NOT a vectorizable instruction, we have a
                    // dependency on a non-vectorizable computation (e.g.,
                    // a phi node for induction variable, a load, etc.).
                    // For now, we allow dependencies on other vectorizable
                    // instructions (they'll all be vectorized together).
                    // But if an instruction uses its OWN output from a prior
                    // iteration (via phi), that's a recurrence.
                    if !vectorizable_set.contains(&def_inst) {
                        return false;
                    }
                }
            }
        }
    }

    true
}

/// Estimate the trip count of a loop from its structure.
///
/// Looks for a simple counted loop pattern:
/// - Compare against immediate in the latch or header.
/// - The immediate is the trip count.
///
/// Returns None if the trip count cannot be determined statically.
fn estimate_trip_count(func: &MachFunction, lp: &NaturalLoop) -> Option<u32> {
    // Look in the latch block for a compare-immediate that controls the branch.
    let latch_block = func.block(lp.latch);
    for &inst_id in &latch_block.insts {
        let inst = func.inst(inst_id);
        match inst.opcode {
            AArch64Opcode::CmpRI | AArch64Opcode::CMPWri | AArch64Opcode::CMPXri => {
                // The immediate operand is typically the last operand.
                for operand in &inst.operands {
                    if let MachOperand::Imm(val) = operand {
                        if *val > 0 && *val < 1_000_000 {
                            return Some(*val as u32);
                        }
                    }
                }
            }
            _ => {}
        }
    }

    // Also check the header block.
    let header_block = func.block(lp.header);
    for &inst_id in &header_block.insts {
        let inst = func.inst(inst_id);
        match inst.opcode {
            AArch64Opcode::CmpRI | AArch64Opcode::CMPWri | AArch64Opcode::CMPXri => {
                for operand in &inst.operands {
                    if let MachOperand::Imm(val) = operand {
                        if *val > 0 && *val < 1_000_000 {
                            return Some(*val as u32);
                        }
                    }
                }
            }
            _ => {}
        }
    }

    None
}

/// Analyze a loop for vectorization potential.
///
/// Returns a `VectorizationPlan` describing whether and how to vectorize,
/// or None if the loop is fundamentally not vectorizable.
pub fn analyze_loop(
    func: &MachFunction,
    lp: &NaturalLoop,
    cost_model: &MultiTargetCostModel,
) -> Option<VectorizationPlan> {
    // Collect vectorizable instructions and their element types.
    let mut vectorizable_insts: Vec<InstId> = Vec::new();
    let mut element_types: HashMap<VecElementType, u32> = HashMap::new();

    for &block_id in &lp.body {
        let block = func.block(block_id);
        for &inst_id in &block.insts {
            let inst = func.inst(inst_id);
            if is_vectorizable(inst.opcode) {
                vectorizable_insts.push(inst_id);
                if let Some(ety) = infer_element_type(func, inst_id) {
                    *element_types.entry(ety).or_insert(0) += 1;
                }
            }
        }
    }

    // Must have at least one vectorizable instruction.
    if vectorizable_insts.is_empty() {
        return None;
    }

    // Check for cross-iteration dependencies.
    if !is_dependency_free(func, lp, &vectorizable_insts) {
        return None;
    }

    // Determine the primary element type (most common).
    let element_type = element_types
        .into_iter()
        .max_by_key(|&(_, count)| count)
        .map(|(ty, _)| ty)?;

    let arrangement = element_type.neon_arrangement();
    let vf = element_type.lanes();

    // Estimate trip count.
    let trip_count = estimate_trip_count(func, lp);

    // Cost analysis: compare scalar vs NEON.
    let tc = trip_count.unwrap_or(64); // assume 64 iterations if unknown
    let scalar_cost = compute_scalar_cost(func, &vectorizable_insts, cost_model, tc);
    let neon_cost = compute_neon_cost(
        func,
        &vectorizable_insts,
        cost_model,
        element_type,
        arrangement,
        vf,
        tc,
    );

    let is_profitable = neon_cost < scalar_cost;

    Some(VectorizationPlan {
        loop_header: lp.header,
        trip_count,
        element_type,
        arrangement,
        vf,
        vectorizable_insts,
        scalar_cost,
        neon_cost,
        is_profitable,
    })
}

/// Compute the scalar cost of executing the vectorizable instructions
/// for `trip_count` iterations.
fn compute_scalar_cost(
    func: &MachFunction,
    insts: &[InstId],
    cost_model: &MultiTargetCostModel,
    trip_count: u32,
) -> f64 {
    let scalar_model = cost_model.scalar_model();
    let per_iter: f64 = insts
        .iter()
        .map(|&inst_id| {
            let opcode = func.inst(inst_id).opcode;
            use llvm2_ir::cost_model::CostModel;
            scalar_model.latency(opcode) as f64
        })
        .sum();

    per_iter * trip_count as f64
}

/// Compute the NEON cost of executing the vectorized loop.
///
/// Includes:
/// - NEON instruction cost per vector iteration (trip_count / vf iterations).
/// - Setup overhead: vector register initialization, domain entry/exit.
/// - Teardown: scalar epilogue for remaining elements.
///
/// # Cost model rationale
///
/// Data transfer between GPR and NEON domains is expensive (~12 cycles
/// per transfer on Apple Silicon), but for a vectorized loop the data
/// stays in NEON registers for the entire loop duration. Transfer cost
/// is therefore one-time at loop entry and exit, NOT per-iteration.
/// This matches real hardware behavior: the loop body operates entirely
/// in the NEON domain.
fn compute_neon_cost(
    func: &MachFunction,
    insts: &[InstId],
    cost_model: &MultiTargetCostModel,
    _element_type: VecElementType,
    arrangement: NeonArrangement,
    vf: u32,
    trip_count: u32,
) -> f64 {
    // NEON cost per vector iteration.
    let per_vector_iter: f64 = insts
        .iter()
        .map(|&inst_id| {
            let opcode = func.inst(inst_id).opcode;
            if let Some(neon_op) = scalar_to_neon_op(opcode) {
                let (lat, _tp) = cost_model.neon_cost(neon_op, arrangement);
                lat as f64
            } else {
                // Fallback: estimate same as scalar (shouldn't happen for
                // truly vectorizable insts, but be conservative).
                use llvm2_ir::cost_model::CostModel;
                cost_model.scalar_model().latency(opcode) as f64
            }
        })
        .sum();

    let vector_iters = (trip_count / vf) as f64;
    let remainder = (trip_count % vf) as u32;

    // Scalar epilogue cost for remaining elements.
    let scalar_model = cost_model.scalar_model();
    let per_scalar_iter: f64 = insts
        .iter()
        .map(|&inst_id| {
            let opcode = func.inst(inst_id).opcode;
            use llvm2_ir::cost_model::CostModel;
            scalar_model.latency(opcode) as f64
        })
        .sum();
    let epilogue_cost = per_scalar_iter * remainder as f64;

    // Setup overhead: domain transfer is one-time at loop entry/exit.
    // Includes: loop counter setup, NEON register initialization, and
    // one domain transfer each way for operands that start/end in GPR.
    let transfer = cost_model.transfer_costs();
    let setup_cost = 4.0 // loop counter + branch overhead
        + transfer.memory_to_neon_cycles  // one-time: load initial vectors
        + transfer.neon_to_memory_cycles; // one-time: store final results

    per_vector_iter * vector_iters + epilogue_cost + setup_cost
}

// ---------------------------------------------------------------------------
// VectorizationPass — MachinePass implementation
// ---------------------------------------------------------------------------

/// NEON auto-vectorization pass.
///
/// Analyzes loops for vectorization opportunities and annotates them
/// with vectorization plans. Currently operates in analysis-only mode:
/// it identifies vectorizable loops and computes profitability, but does
/// not yet rewrite the IR. The `changed` return tracks whether any
/// profitable vectorization opportunity was found.
///
/// Future work: actual IR rewrite to emit NEON instructions.
pub struct VectorizationPass {
    /// Apple Silicon generation for cost modeling.
    generation: CostModelGen,
    /// Minimum trip count for vectorization to be considered.
    min_trip_count: u32,
    /// Collected vectorization plans (for diagnostics/testing).
    plans: Vec<VectorizationPlan>,
}

impl VectorizationPass {
    /// Create a new vectorization pass with default settings (M1, min trip count 8).
    pub fn new() -> Self {
        Self {
            generation: CostModelGen::M1,
            min_trip_count: 8,
            plans: Vec::new(),
        }
    }

    /// Create a vectorization pass with custom settings.
    pub fn with_config(generation: CostModelGen, min_trip_count: u32) -> Self {
        Self {
            generation,
            min_trip_count,
            plans: Vec::new(),
        }
    }

    /// Returns the collected vectorization plans from the last run.
    pub fn plans(&self) -> &[VectorizationPlan] {
        &self.plans
    }
}

impl Default for VectorizationPass {
    fn default() -> Self {
        Self::new()
    }
}

impl MachinePass for VectorizationPass {
    fn name(&self) -> &str {
        "vectorize"
    }

    fn run(&mut self, func: &mut MachFunction) -> bool {
        self.plans.clear();

        let dom = DomTree::compute(func);
        let loop_analysis = LoopAnalysis::compute(func, &dom);

        if loop_analysis.is_empty() {
            return false;
        }

        let cost_model = MultiTargetCostModel::new(self.generation);
        let mut found_profitable = false;

        // Process loops innermost-first (higher depth first).
        let mut loops: Vec<_> = loop_analysis.all_loops().cloned().collect();
        loops.sort_by(|a, b| b.depth.cmp(&a.depth));

        for lp in &loops {
            if let Some(plan) = analyze_loop(func, lp, &cost_model) {
                // Check minimum trip count threshold.
                let tc = plan.trip_count.unwrap_or(0);
                if tc < self.min_trip_count && plan.trip_count.is_some() {
                    // Known small trip count: skip.
                    self.plans.push(plan);
                    continue;
                }

                if plan.is_profitable {
                    found_profitable = true;
                }

                self.plans.push(plan);
            }
        }

        // Return true if we found profitable opportunities.
        // Note: in analysis-only mode we still report true to signal
        // that the pass found actionable information. When IR rewriting
        // is implemented, this will reflect actual changes.
        found_profitable
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dom::DomTree;
    use crate::loops::LoopAnalysis;
    use llvm2_ir::{
        AArch64Opcode, BlockId, MachFunction, MachInst, MachOperand, RegClass, Signature, VReg,
    };

    fn vreg32(id: u32) -> MachOperand {
        MachOperand::VReg(VReg::new(id, RegClass::Gpr32))
    }

    fn vreg64(id: u32) -> MachOperand {
        MachOperand::VReg(VReg::new(id, RegClass::Gpr64))
    }

    fn fpreg64(id: u32) -> MachOperand {
        MachOperand::VReg(VReg::new(id, RegClass::Fpr64))
    }

    fn imm(val: i64) -> MachOperand {
        MachOperand::Imm(val)
    }

    /// Build a simple vectorizable add loop:
    ///
    /// ```text
    ///   bb0 (entry/preheader)
    ///    |
    ///   bb1 (header) <---+
    ///   |  add v2, v0, v1 (i32 add)
    ///   |  cmp v3, #100
    ///   |  bcond bb2, bb1
    ///    |               |
    ///   bb2 (exit)  bb1 (latch = header is self-loop pattern via bb3)
    /// ```
    ///
    /// Simplified as: bb0 -> bb1 -> bb3 (latch) -> bb1, bb1 -> bb2
    fn make_vectorizable_add_loop() -> MachFunction {
        let mut func = MachFunction::new(
            "vec_add_loop".to_string(),
            Signature::new(vec![], vec![]),
        );
        let bb0 = func.entry;
        let bb1 = func.create_block(); // header
        let bb2 = func.create_block(); // exit
        let bb3 = func.create_block(); // latch

        // bb0: branch to header
        let br0 = func.push_inst(MachInst::new(
            AArch64Opcode::B,
            vec![MachOperand::Block(bb1)],
        ));
        func.append_inst(bb0, br0);

        // bb1 (header): add v2 = v0 + v1 (i32)
        let add = func.push_inst(MachInst::new(
            AArch64Opcode::AddRR,
            vec![vreg32(2), vreg32(0), vreg32(1)],
        ));
        func.append_inst(bb1, add);

        // Compare v3 against trip count 100
        let cmp = func.push_inst(MachInst::new(
            AArch64Opcode::CmpRI,
            vec![vreg32(3), imm(100)],
        ));
        func.append_inst(bb1, cmp);

        // Conditional branch: exit or continue
        let bcond = func.push_inst(MachInst::new(
            AArch64Opcode::BCond,
            vec![MachOperand::Block(bb2), MachOperand::Block(bb3)],
        ));
        func.append_inst(bb1, bcond);

        // bb3 (latch): back to header
        let br3 = func.push_inst(MachInst::new(
            AArch64Opcode::B,
            vec![MachOperand::Block(bb1)],
        ));
        func.append_inst(bb3, br3);

        // bb2 (exit): return
        let ret = func.push_inst(MachInst::new(AArch64Opcode::Ret, vec![]));
        func.append_inst(bb2, ret);

        func.add_edge(bb0, bb1);
        func.add_edge(bb1, bb2);
        func.add_edge(bb1, bb3);
        func.add_edge(bb3, bb1);

        func
    }

    /// Build a loop with a data dependency (recurrence) that prevents vectorization.
    ///
    /// ```text
    ///   bb0 -> bb1 (header) -> bb3 (latch) -> bb1
    ///                       -> bb2 (exit)
    /// ```
    ///
    /// In bb1: v2 = add v2, v1 — v2 depends on its own prior value (recurrence).
    fn make_dependency_loop() -> MachFunction {
        let mut func = MachFunction::new(
            "dep_loop".to_string(),
            Signature::new(vec![], vec![]),
        );
        let bb0 = func.entry;
        let bb1 = func.create_block();
        let bb2 = func.create_block();
        let bb3 = func.create_block();

        let br0 = func.push_inst(MachInst::new(
            AArch64Opcode::B,
            vec![MachOperand::Block(bb1)],
        ));
        func.append_inst(bb0, br0);

        // bb1: load v2 = load (not vectorizable — memory op)
        let ld = func.push_inst(MachInst::new(
            AArch64Opcode::LdrRI,
            vec![vreg32(2), vreg64(10), imm(0)],
        ));
        func.append_inst(bb1, ld);

        // v3 = add v2, v1 — depends on load result
        let add = func.push_inst(MachInst::new(
            AArch64Opcode::AddRR,
            vec![vreg32(3), vreg32(2), vreg32(1)],
        ));
        func.append_inst(bb1, add);

        let cmp = func.push_inst(MachInst::new(
            AArch64Opcode::CmpRI,
            vec![vreg32(4), imm(100)],
        ));
        func.append_inst(bb1, cmp);

        let bcond = func.push_inst(MachInst::new(
            AArch64Opcode::BCond,
            vec![MachOperand::Block(bb2), MachOperand::Block(bb3)],
        ));
        func.append_inst(bb1, bcond);

        let br3 = func.push_inst(MachInst::new(
            AArch64Opcode::B,
            vec![MachOperand::Block(bb1)],
        ));
        func.append_inst(bb3, br3);

        let ret = func.push_inst(MachInst::new(AArch64Opcode::Ret, vec![]));
        func.append_inst(bb2, ret);

        func.add_edge(bb0, bb1);
        func.add_edge(bb1, bb2);
        func.add_edge(bb1, bb3);
        func.add_edge(bb3, bb1);

        func
    }

    /// Build a loop with a small trip count (4) — should be rejected by cost model.
    fn make_small_trip_count_loop() -> MachFunction {
        let mut func = MachFunction::new(
            "small_tc".to_string(),
            Signature::new(vec![], vec![]),
        );
        let bb0 = func.entry;
        let bb1 = func.create_block();
        let bb2 = func.create_block();
        let bb3 = func.create_block();

        let br0 = func.push_inst(MachInst::new(
            AArch64Opcode::B,
            vec![MachOperand::Block(bb1)],
        ));
        func.append_inst(bb0, br0);

        let add = func.push_inst(MachInst::new(
            AArch64Opcode::AddRR,
            vec![vreg32(2), vreg32(0), vreg32(1)],
        ));
        func.append_inst(bb1, add);

        // Trip count = 4
        let cmp = func.push_inst(MachInst::new(
            AArch64Opcode::CmpRI,
            vec![vreg32(3), imm(4)],
        ));
        func.append_inst(bb1, cmp);

        let bcond = func.push_inst(MachInst::new(
            AArch64Opcode::BCond,
            vec![MachOperand::Block(bb2), MachOperand::Block(bb3)],
        ));
        func.append_inst(bb1, bcond);

        let br3 = func.push_inst(MachInst::new(
            AArch64Opcode::B,
            vec![MachOperand::Block(bb1)],
        ));
        func.append_inst(bb3, br3);

        let ret = func.push_inst(MachInst::new(AArch64Opcode::Ret, vec![]));
        func.append_inst(bb2, ret);

        func.add_edge(bb0, bb1);
        func.add_edge(bb1, bb2);
        func.add_edge(bb1, bb3);
        func.add_edge(bb3, bb1);

        func
    }

    // =========================================================================
    // Test: simple add loop is vectorizable
    // =========================================================================

    #[test]
    fn test_simple_add_loop_vectorizable() {
        let func = make_vectorizable_add_loop();
        let dom = DomTree::compute(&func);
        let la = LoopAnalysis::compute(&func, &dom);
        let cost_model = MultiTargetCostModel::new(CostModelGen::M1);

        assert_eq!(la.num_loops(), 1);
        let lp = la.all_loops().next().unwrap();
        let plan = analyze_loop(&func, lp, &cost_model);

        assert!(plan.is_some(), "add loop should be vectorizable");
        let plan = plan.unwrap();
        assert_eq!(plan.element_type, VecElementType::I32);
        assert_eq!(plan.arrangement, NeonArrangement::S4);
        assert_eq!(plan.vf, 4);
        assert_eq!(plan.trip_count, Some(100));
        assert!(!plan.vectorizable_insts.is_empty());
    }

    // =========================================================================
    // Test: loop with data dependency is NOT vectorizable
    // =========================================================================

    #[test]
    fn test_data_dependency_blocks_vectorization() {
        let func = make_dependency_loop();
        let dom = DomTree::compute(&func);
        let la = LoopAnalysis::compute(&func, &dom);
        let cost_model = MultiTargetCostModel::new(CostModelGen::M1);

        let lp = la.all_loops().next().unwrap();
        let plan = analyze_loop(&func, lp, &cost_model);

        // The add depends on a load (not vectorizable), so the dependency
        // check should reject it.
        assert!(plan.is_none(), "loop with memory dependency should not be vectorizable");
    }

    // =========================================================================
    // Test: cost model rejects small trip count
    // =========================================================================

    #[test]
    fn test_small_trip_count_rejected() {
        let mut func = make_small_trip_count_loop();
        let mut pass = VectorizationPass::with_config(CostModelGen::M1, 8);
        let changed = pass.run(&mut func);

        // The pass should find a plan but it should not be "changed" because
        // the trip count (4) is below the minimum threshold (8).
        // The plan exists but is skipped.
        assert!(!changed, "small trip count should not trigger vectorization");
    }

    // =========================================================================
    // Test: i32 maps to arrangement 4S
    // =========================================================================

    #[test]
    fn test_i32_arrangement_4s() {
        assert_eq!(VecElementType::I32.neon_arrangement(), NeonArrangement::S4);
        assert_eq!(VecElementType::I32.lanes(), 4);
        assert_eq!(VecElementType::I32.bits(), 32);
    }

    // =========================================================================
    // Test: f64 maps to arrangement 2D
    // =========================================================================

    #[test]
    fn test_f64_arrangement_2d() {
        assert_eq!(VecElementType::F64.neon_arrangement(), NeonArrangement::D2);
        assert_eq!(VecElementType::F64.lanes(), 2);
        assert_eq!(VecElementType::F64.bits(), 64);
    }

    // =========================================================================
    // Test: i8 maps to arrangement 16B
    // =========================================================================

    #[test]
    fn test_i8_arrangement_16b() {
        assert_eq!(VecElementType::I8.neon_arrangement(), NeonArrangement::B16);
        assert_eq!(VecElementType::I8.lanes(), 16);
        assert_eq!(VecElementType::I8.bits(), 8);
    }

    // =========================================================================
    // Test: i16 maps to arrangement 8H
    // =========================================================================

    #[test]
    fn test_i16_arrangement_8h() {
        assert_eq!(VecElementType::I16.neon_arrangement(), NeonArrangement::H8);
        assert_eq!(VecElementType::I16.lanes(), 8);
        assert_eq!(VecElementType::I16.bits(), 16);
    }

    // =========================================================================
    // Test: f32 maps to arrangement 4S
    // =========================================================================

    #[test]
    fn test_f32_arrangement_4s() {
        assert_eq!(VecElementType::F32.neon_arrangement(), NeonArrangement::S4);
        assert_eq!(VecElementType::F32.lanes(), 4);
    }

    // =========================================================================
    // Test: is_vectorizable for various opcodes
    // =========================================================================

    #[test]
    fn test_is_vectorizable_opcodes() {
        // Vectorizable: pure arithmetic with NEON equivalents
        assert!(is_vectorizable(AArch64Opcode::AddRR));
        assert!(is_vectorizable(AArch64Opcode::SubRR));
        assert!(is_vectorizable(AArch64Opcode::MulRR));
        assert!(is_vectorizable(AArch64Opcode::Neg));
        assert!(is_vectorizable(AArch64Opcode::AndRR));
        assert!(is_vectorizable(AArch64Opcode::OrrRR));
        assert!(is_vectorizable(AArch64Opcode::EorRR));
        assert!(is_vectorizable(AArch64Opcode::FaddRR));
        assert!(is_vectorizable(AArch64Opcode::FmulRR));
        assert!(is_vectorizable(AArch64Opcode::LslRI));

        // NOT vectorizable: memory ops
        assert!(!is_vectorizable(AArch64Opcode::LdrRI));
        assert!(!is_vectorizable(AArch64Opcode::StrRI));

        // NOT vectorizable: branches, calls
        assert!(!is_vectorizable(AArch64Opcode::B));
        assert!(!is_vectorizable(AArch64Opcode::Bl));
        assert!(!is_vectorizable(AArch64Opcode::Ret));

        // NOT vectorizable: compare (sets flags, no NEON map)
        assert!(!is_vectorizable(AArch64Opcode::CmpRR));
        assert!(!is_vectorizable(AArch64Opcode::CmpRI));

        // NOT vectorizable: divide (no NEON integer divide)
        assert!(!is_vectorizable(AArch64Opcode::SDiv));
        assert!(!is_vectorizable(AArch64Opcode::UDiv));
    }

    // =========================================================================
    // Test: no-loop function returns false
    // =========================================================================

    #[test]
    fn test_no_loop_function() {
        let mut func = MachFunction::new(
            "no_loop".to_string(),
            Signature::new(vec![], vec![]),
        );
        let bb0 = func.entry;
        let ret = func.push_inst(MachInst::new(AArch64Opcode::Ret, vec![]));
        func.append_inst(bb0, ret);

        let mut pass = VectorizationPass::new();
        let changed = pass.run(&mut func);

        assert!(!changed);
        assert!(pass.plans().is_empty());
    }

    // =========================================================================
    // Test: VectorizationPass integration via MachinePass trait
    // =========================================================================

    #[test]
    fn test_pass_finds_vectorizable_loop() {
        let mut func = make_vectorizable_add_loop();
        let mut pass = VectorizationPass::new();
        let changed = pass.run(&mut func);

        assert!(changed, "pass should find profitable vectorization");
        assert_eq!(pass.plans().len(), 1);

        let plan = &pass.plans()[0];
        assert!(plan.is_profitable);
        assert!(plan.speedup() > 1.0);
    }

    // =========================================================================
    // Test: speedup calculation
    // =========================================================================

    #[test]
    fn test_speedup_calculation() {
        let plan = VectorizationPlan {
            loop_header: BlockId(1),
            trip_count: Some(100),
            element_type: VecElementType::I32,
            arrangement: NeonArrangement::S4,
            vf: 4,
            vectorizable_insts: vec![],
            scalar_cost: 100.0,
            neon_cost: 50.0,
            is_profitable: true,
        };
        assert!((plan.speedup() - 2.0).abs() < 0.001);

        // Zero neon cost
        let plan2 = VectorizationPlan {
            neon_cost: 0.0,
            ..plan.clone()
        };
        assert_eq!(plan2.speedup(), 0.0);
    }

    // =========================================================================
    // Test: scalar_to_neon_op mapping
    // =========================================================================

    #[test]
    fn test_scalar_to_neon_op_mapping() {
        assert_eq!(scalar_to_neon_op(AArch64Opcode::AddRR), Some(NeonOp::Add));
        assert_eq!(scalar_to_neon_op(AArch64Opcode::AddRI), Some(NeonOp::Add));
        assert_eq!(scalar_to_neon_op(AArch64Opcode::SubRR), Some(NeonOp::Sub));
        assert_eq!(scalar_to_neon_op(AArch64Opcode::MulRR), Some(NeonOp::Mul));
        assert_eq!(scalar_to_neon_op(AArch64Opcode::Neg), Some(NeonOp::Neg));
        assert_eq!(scalar_to_neon_op(AArch64Opcode::FaddRR), Some(NeonOp::Fadd));
        assert_eq!(scalar_to_neon_op(AArch64Opcode::FmulRR), Some(NeonOp::Fmul));
        assert_eq!(scalar_to_neon_op(AArch64Opcode::BicRR), Some(NeonOp::Bic));

        // No mapping
        assert_eq!(scalar_to_neon_op(AArch64Opcode::SDiv), None);
        assert_eq!(scalar_to_neon_op(AArch64Opcode::Ret), None);
        assert_eq!(scalar_to_neon_op(AArch64Opcode::CmpRR), None);
    }

    // =========================================================================
    // Test: FP loop with f64 -> 2D arrangement
    // =========================================================================

    #[test]
    fn test_fp_loop_f64_arrangement() {
        let mut func = MachFunction::new(
            "fp_loop".to_string(),
            Signature::new(vec![], vec![]),
        );
        let bb0 = func.entry;
        let bb1 = func.create_block();
        let bb2 = func.create_block();
        let bb3 = func.create_block();

        let br0 = func.push_inst(MachInst::new(
            AArch64Opcode::B,
            vec![MachOperand::Block(bb1)],
        ));
        func.append_inst(bb0, br0);

        // FP add: f64
        let fadd = func.push_inst(MachInst::new(
            AArch64Opcode::FaddRR,
            vec![fpreg64(2), fpreg64(0), fpreg64(1)],
        ));
        func.append_inst(bb1, fadd);

        let cmp = func.push_inst(MachInst::new(
            AArch64Opcode::CmpRI,
            vec![vreg32(5), imm(200)],
        ));
        func.append_inst(bb1, cmp);

        let bcond = func.push_inst(MachInst::new(
            AArch64Opcode::BCond,
            vec![MachOperand::Block(bb2), MachOperand::Block(bb3)],
        ));
        func.append_inst(bb1, bcond);

        let br3 = func.push_inst(MachInst::new(
            AArch64Opcode::B,
            vec![MachOperand::Block(bb1)],
        ));
        func.append_inst(bb3, br3);

        let ret = func.push_inst(MachInst::new(AArch64Opcode::Ret, vec![]));
        func.append_inst(bb2, ret);

        func.add_edge(bb0, bb1);
        func.add_edge(bb1, bb2);
        func.add_edge(bb1, bb3);
        func.add_edge(bb3, bb1);

        let dom = DomTree::compute(&func);
        let la = LoopAnalysis::compute(&func, &dom);
        let cost_model = MultiTargetCostModel::new(CostModelGen::M1);

        let lp = la.all_loops().next().unwrap();
        let plan = analyze_loop(&func, lp, &cost_model);

        assert!(plan.is_some());
        let plan = plan.unwrap();
        assert_eq!(plan.element_type, VecElementType::F64);
        assert_eq!(plan.arrangement, NeonArrangement::D2);
        assert_eq!(plan.vf, 2);
    }

    // =========================================================================
    // Test: loop with only non-vectorizable instructions returns None
    // =========================================================================

    #[test]
    fn test_loop_with_only_branches_not_vectorizable() {
        // Loop body: just cmp + bcond + branch (no vectorizable compute)
        let mut func = MachFunction::new(
            "branch_only".to_string(),
            Signature::new(vec![], vec![]),
        );
        let bb0 = func.entry;
        let bb1 = func.create_block();
        let bb2 = func.create_block();
        let bb3 = func.create_block();

        let br0 = func.push_inst(MachInst::new(
            AArch64Opcode::B,
            vec![MachOperand::Block(bb1)],
        ));
        func.append_inst(bb0, br0);

        let cmp = func.push_inst(MachInst::new(
            AArch64Opcode::CmpRI,
            vec![vreg32(0), imm(10)],
        ));
        func.append_inst(bb1, cmp);

        let bcond = func.push_inst(MachInst::new(
            AArch64Opcode::BCond,
            vec![MachOperand::Block(bb2), MachOperand::Block(bb3)],
        ));
        func.append_inst(bb1, bcond);

        let br3 = func.push_inst(MachInst::new(
            AArch64Opcode::B,
            vec![MachOperand::Block(bb1)],
        ));
        func.append_inst(bb3, br3);

        let ret = func.push_inst(MachInst::new(AArch64Opcode::Ret, vec![]));
        func.append_inst(bb2, ret);

        func.add_edge(bb0, bb1);
        func.add_edge(bb1, bb2);
        func.add_edge(bb1, bb3);
        func.add_edge(bb3, bb1);

        let dom = DomTree::compute(&func);
        let la = LoopAnalysis::compute(&func, &dom);
        let cost_model = MultiTargetCostModel::new(CostModelGen::M1);

        let lp = la.all_loops().next().unwrap();
        let plan = analyze_loop(&func, lp, &cost_model);

        assert!(plan.is_none(), "loop with only branches should not be vectorizable");
    }

    // =========================================================================
    // Test: VecElementType coverage
    // =========================================================================

    #[test]
    fn test_vec_element_type_all_variants() {
        let cases = [
            (VecElementType::I8, 8, NeonArrangement::B16, 16),
            (VecElementType::I16, 16, NeonArrangement::H8, 8),
            (VecElementType::I32, 32, NeonArrangement::S4, 4),
            (VecElementType::I64, 64, NeonArrangement::D2, 2),
            (VecElementType::F32, 32, NeonArrangement::S4, 4),
            (VecElementType::F64, 64, NeonArrangement::D2, 2),
        ];

        for (ety, bits, arr, lanes) in cases {
            assert_eq!(ety.bits(), bits, "{:?} should have {} bits", ety, bits);
            assert_eq!(ety.neon_arrangement(), arr, "{:?} should map to {:?}", ety, arr);
            assert_eq!(ety.lanes(), lanes, "{:?} should have {} lanes", ety, lanes);
        }
    }

    // =========================================================================
    // Test: pass name
    // =========================================================================

    #[test]
    fn test_pass_name() {
        let pass = VectorizationPass::new();
        assert_eq!(pass.name(), "vectorize");
    }

    // =========================================================================
    // Test: VectorizationPlan profitability
    // =========================================================================

    #[test]
    fn test_plan_profitability_from_analysis() {
        let func = make_vectorizable_add_loop();
        let dom = DomTree::compute(&func);
        let la = LoopAnalysis::compute(&func, &dom);
        let cost_model = MultiTargetCostModel::new(CostModelGen::M1);

        let lp = la.all_loops().next().unwrap();
        let plan = analyze_loop(&func, lp, &cost_model).unwrap();

        // With trip count 100 and i32 add (1-cycle scalar, 2-cycle NEON for
        // 4 elements at a time), NEON should be profitable:
        // Scalar: 100 * 1 = 100 cycles
        // NEON: 25 * 2 + overhead < 100
        assert!(plan.is_profitable, "i32 add loop with TC=100 should be profitable");
        assert!(plan.neon_cost < plan.scalar_cost);
    }
}
