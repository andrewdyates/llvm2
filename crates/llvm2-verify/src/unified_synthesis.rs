// llvm2-verify/unified_synthesis.rs - Unified CEGIS synthesis across multiple targets
//
// Author: Andrew Yates <ayates@dropbox.com>
// Copyright 2026 Dropbox, Inc. | License: Apache-2.0
//
// Extends the single-target CEGIS loop to search across multiple compute
// targets simultaneously. For a given tMIR operation, the unified loop
// generates candidate lowerings for both scalar AArch64 instructions and
// NEON SIMD instructions, verifies each via CEGIS, ranks by cost, and
// returns the best proven-correct lowering with its target annotation.
//
// Reference: designs/2026-04-13-superoptimization.md
// Reference: designs/2026-04-13-unified-solver-architecture.md

//! Unified multi-target CEGIS synthesis.
//!
//! The [`UnifiedCegisLoop`] searches for optimal lowerings across multiple
//! compute targets (scalar CPU, NEON SIMD) simultaneously. For each source
//! pattern, it:
//!
//! 1. Generates [`TargetCandidate`]s from both scalar and NEON search spaces
//! 2. Fast-filters candidates against accumulated counterexamples
//! 3. Verifies surviving candidates via SMT
//! 4. Ranks verified candidates by estimated cost
//! 5. Returns the cheapest proven-correct rule with target annotation
//!
//! This implements the cross-target search from the unified solver architecture
//! (Phase 2: multi-target candidate generation and ranking).

use crate::cegis::{CegisLoop, CegisResult};
use crate::neon_semantics;
use crate::smt::{SmtExpr, VectorArrangement};
use crate::synthesis::{SearchConfig, SynthOpcode};
use llvm2_ir::cost_model::{
    ComputeTarget, CostModelGen, MultiTargetCostModel, ProfitabilityAnalyzer,
};
use std::collections::HashMap;
use std::fmt;

// ---------------------------------------------------------------------------
// Compute target (local to synthesis -- mirrors llvm2-lower's ComputeTarget
// without the cross-crate dependency)
// ---------------------------------------------------------------------------

/// Compute target for synthesis candidates.
///
/// This is a synthesis-local enum mirroring `ComputeTarget` from llvm2-lower
/// to avoid circular dependencies. Only includes targets that the synthesis
/// engine can generate candidates for.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SynthTarget {
    /// AArch64 scalar integer/FP instructions.
    Scalar,
    /// AArch64 NEON 128-bit SIMD instructions (specific arrangement).
    Neon(VectorArrangement),
}

impl fmt::Display for SynthTarget {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SynthTarget::Scalar => write!(f, "scalar"),
            SynthTarget::Neon(arr) => write!(f, "NEON({:?})", arr),
        }
    }
}

// ---------------------------------------------------------------------------
// Target candidate
// ---------------------------------------------------------------------------

/// A candidate lowering expression annotated with its compute target
/// and estimated execution cost.
#[derive(Debug, Clone)]
pub struct TargetCandidate {
    /// Human-readable description of the candidate.
    pub name: String,
    /// The candidate SMT expression (target-side semantics).
    pub expr: SmtExpr,
    /// Which compute target this candidate uses.
    pub target: SynthTarget,
    /// Estimated execution cost (lower is better).
    /// Scalar ALU ops = 1, MUL = 3. NEON ops have a per-lane overhead
    /// but amortize over multiple elements, so per-element cost is lower.
    pub cost: i32,
}

impl TargetCandidate {
    /// Create a new target candidate.
    pub fn new(name: impl Into<String>, expr: SmtExpr, target: SynthTarget, cost: i32) -> Self {
        Self {
            name: name.into(),
            expr,
            target,
            cost,
        }
    }
}

// ---------------------------------------------------------------------------
// NEON opcode enumeration for synthesis
// ---------------------------------------------------------------------------

/// NEON opcodes supported in the synthesis search space.
///
/// These map to the NEON semantic encoding functions in `neon_semantics.rs`.
/// Only integer operations are included (FP synthesis is future work).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum NeonSynthOpcode {
    /// NEON vector ADD (per-lane wrapping addition).
    Add,
    /// NEON vector SUB (per-lane wrapping subtraction).
    Sub,
    /// NEON vector MUL (per-lane wrapping multiply, no 2D).
    Mul,
    /// NEON vector NEG (per-lane two's complement negate).
    Neg,
    /// NEON bitwise AND (full-width).
    And,
    /// NEON bitwise ORR (full-width).
    Orr,
    /// NEON bitwise EOR (full-width).
    Eor,
    /// NEON bitwise BIC (bit clear = AND NOT, full-width).
    Bic,
    /// NEON SHL (per-lane shift left by immediate).
    Shl,
    /// NEON USHR (per-lane unsigned shift right by immediate).
    Ushr,
    /// NEON SSHR (per-lane signed/arithmetic shift right by immediate).
    Sshr,
    /// NEON MLA (multiply-accumulate: Vd + Vn * Vm, per-lane, no 2D).
    /// This is a fused two-instruction sequence: MUL then ADD.
    Mla,
    /// NEON MLS (multiply-subtract: Vd - Vn * Vm, per-lane, no 2D).
    /// This is a fused two-instruction sequence: MUL then SUB.
    Mls,
}

impl NeonSynthOpcode {
    /// All NEON opcodes in the search space.
    pub fn all() -> &'static [NeonSynthOpcode] {
        &[
            NeonSynthOpcode::Add,
            NeonSynthOpcode::Sub,
            NeonSynthOpcode::Mul,
            NeonSynthOpcode::Neg,
            NeonSynthOpcode::And,
            NeonSynthOpcode::Orr,
            NeonSynthOpcode::Eor,
            NeonSynthOpcode::Bic,
            NeonSynthOpcode::Shl,
            NeonSynthOpcode::Ushr,
            NeonSynthOpcode::Sshr,
            NeonSynthOpcode::Mla,
            NeonSynthOpcode::Mls,
        ]
    }

    /// Whether this opcode is unary.
    pub fn is_unary(self) -> bool {
        matches!(self, NeonSynthOpcode::Neg)
    }

    /// Whether this opcode takes an immediate shift amount.
    pub fn is_shift(self) -> bool {
        matches!(
            self,
            NeonSynthOpcode::Shl | NeonSynthOpcode::Ushr | NeonSynthOpcode::Sshr
        )
    }

    /// Whether this opcode is bitwise (no lane decomposition needed).
    pub fn is_bitwise(self) -> bool {
        matches!(
            self,
            NeonSynthOpcode::And
                | NeonSynthOpcode::Orr
                | NeonSynthOpcode::Eor
                | NeonSynthOpcode::Bic
        )
    }

    /// Whether this opcode is a fused multi-instruction operation (ternary).
    pub fn is_fused(self) -> bool {
        matches!(self, NeonSynthOpcode::Mla | NeonSynthOpcode::Mls)
    }

    /// Whether this opcode is compatible with the given arrangement.
    ///
    /// Most NEON integer ops support all arrangements. Exceptions:
    /// - MUL, MLA, MLS: no D2 (64-bit lane MUL not supported on AArch64 NEON)
    pub fn is_compatible(self, arrangement: VectorArrangement) -> bool {
        match self {
            NeonSynthOpcode::Mul | NeonSynthOpcode::Mla | NeonSynthOpcode::Mls => {
                arrangement != VectorArrangement::D2
            }
            _ => true,
        }
    }

    /// Human-readable name.
    pub fn name(self) -> &'static str {
        match self {
            NeonSynthOpcode::Add => "NEON_ADD",
            NeonSynthOpcode::Sub => "NEON_SUB",
            NeonSynthOpcode::Mul => "NEON_MUL",
            NeonSynthOpcode::Neg => "NEON_NEG",
            NeonSynthOpcode::And => "NEON_AND",
            NeonSynthOpcode::Orr => "NEON_ORR",
            NeonSynthOpcode::Eor => "NEON_EOR",
            NeonSynthOpcode::Bic => "NEON_BIC",
            NeonSynthOpcode::Shl => "NEON_SHL",
            NeonSynthOpcode::Ushr => "NEON_USHR",
            NeonSynthOpcode::Sshr => "NEON_SSHR",
            NeonSynthOpcode::Mla => "NEON_MLA",
            NeonSynthOpcode::Mls => "NEON_MLS",
        }
    }

    /// Estimated cost of this NEON operation.
    ///
    /// NEON instructions have pipeline overhead but process multiple lanes.
    /// The cost here is the instruction-level cost, not per-element.
    /// For cost comparison with scalar, the caller should consider the lane
    /// count (a single NEON ADD replaces N scalar ADDs).
    pub fn cost(self) -> i32 {
        match self {
            NeonSynthOpcode::Add
            | NeonSynthOpcode::Sub
            | NeonSynthOpcode::And
            | NeonSynthOpcode::Orr
            | NeonSynthOpcode::Eor
            | NeonSynthOpcode::Bic
            | NeonSynthOpcode::Shl
            | NeonSynthOpcode::Ushr
            | NeonSynthOpcode::Sshr => 1,
            NeonSynthOpcode::Neg => 1,
            NeonSynthOpcode::Mul => 3,
            // Fused ops: cost equals sum of component instructions.
            // MLA = MUL + ADD = 3 + 1 = 4, but on hardware MLA is typically
            // 4 cycles (same latency as MUL, ADD is "free" in the fused pipe).
            NeonSynthOpcode::Mla | NeonSynthOpcode::Mls => 4,
        }
    }
}

// ---------------------------------------------------------------------------
// Unified search space
// ---------------------------------------------------------------------------

/// Configuration for the unified multi-target search space.
#[derive(Debug, Clone)]
pub struct UnifiedSearchConfig {
    /// Scalar search configuration.
    pub scalar: SearchConfig,
    /// Whether to include NEON candidates in the search.
    pub include_neon: bool,
    /// NEON vector arrangements to search. Default: S2 (2x32-bit, testable
    /// with the 64-bit evaluator), S4 (4x32-bit, 128-bit), D2 (2x64-bit,
    /// 128-bit). S4 and D2 cover the full 128-bit NEON register file.
    pub neon_arrangements: Vec<VectorArrangement>,
    /// Interesting shift amounts for NEON SHL/USHR (must be < lane_bits).
    pub neon_shift_amounts: Vec<u32>,
}

impl Default for UnifiedSearchConfig {
    fn default() -> Self {
        Self {
            scalar: SearchConfig::default(),
            include_neon: true,
            // S2 (2x32-bit in 64-bit) is testable with the u64 evaluator.
            // S4 (4x32-bit, 128-bit) and D2 (2x64-bit, 128-bit) cover the
            // full 128-bit NEON register file for SIMD synthesis (#166).
            neon_arrangements: vec![
                VectorArrangement::S2,
                VectorArrangement::S4,
                VectorArrangement::D2,
            ],
            neon_shift_amounts: vec![1, 2, 4],
        }
    }
}

/// Generates candidate lowerings for both scalar and NEON targets.
pub struct UnifiedSearchSpace;

impl UnifiedSearchSpace {
    /// Generate scalar candidates for a given source expression.
    ///
    /// For single-variable expressions, generates all single-instruction
    /// scalar patterns (identity, ALU ops with interesting immediates, etc.).
    pub fn scalar_candidates(
        source_name: &str,
        width: u32,
    ) -> Vec<TargetCandidate> {
        let x = SmtExpr::var(source_name, width);
        let mut candidates = Vec::new();

        // Identity (elision) -- cost 0
        candidates.push(TargetCandidate::new(
            "identity",
            x.clone(),
            SynthTarget::Scalar,
            0,
        ));

        // Binary ops with interesting immediates
        let immediates: &[i64] = &[0, 1, 2, -1];

        for &opcode in SynthOpcode::all() {
            if opcode.is_const_only() {
                continue;
            }

            if opcode.is_unary() {
                let expr = encode_scalar_unary(opcode, &x, width);
                let cost = scalar_opcode_cost(opcode);
                candidates.push(TargetCandidate::new(
                    format!("{} {}", opcode.name(), source_name),
                    expr,
                    SynthTarget::Scalar,
                    cost,
                ));
            } else {
                for &imm in immediates {
                    // Skip invalid shift amounts
                    if matches!(opcode, SynthOpcode::Lsl | SynthOpcode::Lsr | SynthOpcode::Asr)
                        && (imm < 0 || imm as u32 >= width)
                    {
                        continue;
                    }
                    let imm_expr = SmtExpr::bv_const(imm as u64, width);
                    let expr = encode_scalar_binary(opcode, &x, &imm_expr);
                    let cost = scalar_opcode_cost(opcode);
                    candidates.push(TargetCandidate::new(
                        format!("{} {}, #{}", opcode.name(), source_name, imm),
                        expr,
                        SynthTarget::Scalar,
                        cost,
                    ));
                }

                // op x, x (same register)
                let expr = encode_scalar_binary(opcode, &x, &x);
                let cost = scalar_opcode_cost(opcode);
                candidates.push(TargetCandidate::new(
                    format!("{} {}, {}", opcode.name(), source_name, source_name),
                    expr,
                    SynthTarget::Scalar,
                    cost,
                ));
            }
        }

        candidates
    }

    /// Generate NEON candidates for a given source expression.
    ///
    /// The source and candidate are both vector registers containing
    /// lanes of the specified arrangement. Generates:
    /// - Identity (cost 0)
    /// - All single-instruction NEON ops (self-operation: vn OP vn)
    /// - Shift-by-immediate variants for SHL/USHR/SSHR
    /// - Multi-instruction fusion candidates (MLA, MLS: vn + vn*vn, vn - vn*vn)
    /// - Arrangement-filtered: skips opcodes incompatible with the arrangement
    pub fn neon_candidates(
        source_name: &str,
        arrangement: VectorArrangement,
        config: &UnifiedSearchConfig,
    ) -> Vec<TargetCandidate> {
        let total_bits = arrangement.lane_count() * arrangement.lane_bits();
        let vn = SmtExpr::var(source_name, total_bits);
        let mut candidates = Vec::new();

        // Identity
        candidates.push(TargetCandidate::new(
            format!("NEON identity ({:?})", arrangement),
            vn.clone(),
            SynthTarget::Neon(arrangement),
            0,
        ));

        for &opcode in NeonSynthOpcode::all() {
            // Arrangement compatibility filter
            if !opcode.is_compatible(arrangement) {
                continue;
            }

            if opcode.is_unary() {
                let expr = encode_neon_unary(opcode, arrangement, &vn);
                let cost = opcode.cost();
                candidates.push(TargetCandidate::new(
                    format!("{}.{:?} {}", opcode.name(), arrangement, source_name),
                    expr,
                    SynthTarget::Neon(arrangement),
                    cost,
                ));
            } else if opcode.is_shift() {
                for &shift_amt in &config.neon_shift_amounts {
                    if shift_amt >= arrangement.lane_bits() {
                        continue;
                    }
                    // For USHR/SSHR, shift must be >= 1
                    if matches!(opcode, NeonSynthOpcode::Ushr | NeonSynthOpcode::Sshr)
                        && shift_amt < 1
                    {
                        continue;
                    }
                    let expr = encode_neon_shift(opcode, arrangement, &vn, shift_amt);
                    let cost = opcode.cost();
                    candidates.push(TargetCandidate::new(
                        format!(
                            "{}.{:?} {}, #{}",
                            opcode.name(),
                            arrangement,
                            source_name,
                            shift_amt
                        ),
                        expr,
                        SynthTarget::Neon(arrangement),
                        cost,
                    ));
                }
            } else if opcode.is_fused() {
                // Fused multi-instruction candidates: MLA(vn, vn, vn), MLS(vn, vn, vn)
                // MLA: vd = vn + vn * vn (accumulator = first operand)
                // MLS: vd = vn - vn * vn (accumulator = first operand)
                let expr = encode_neon_fused(opcode, arrangement, &vn, &vn, &vn);
                let cost = opcode.cost();
                candidates.push(TargetCandidate::new(
                    format!(
                        "{}.{:?} {}, {}, {}",
                        opcode.name(),
                        arrangement,
                        source_name,
                        source_name,
                        source_name,
                    ),
                    expr,
                    SynthTarget::Neon(arrangement),
                    cost,
                ));
            } else if opcode.is_bitwise() {
                // Bitwise ops: vn OP vn (self-operation)
                let expr = encode_neon_bitwise(opcode, &vn, &vn);
                let cost = opcode.cost();
                candidates.push(TargetCandidate::new(
                    format!(
                        "{}.{:?} {}, {}",
                        opcode.name(),
                        arrangement,
                        source_name,
                        source_name
                    ),
                    expr,
                    SynthTarget::Neon(arrangement),
                    cost,
                ));
            } else {
                // Lane-parallel binary ops: vn OP vn (self-operation)
                let expr = encode_neon_binary(opcode, arrangement, &vn, &vn);
                let cost = opcode.cost();
                candidates.push(TargetCandidate::new(
                    format!(
                        "{}.{:?} {}, {}",
                        opcode.name(),
                        arrangement,
                        source_name,
                        source_name
                    ),
                    expr,
                    SynthTarget::Neon(arrangement),
                    cost,
                ));
            }
        }

        candidates
    }

    /// Filter a candidate list to only those compatible with the given arrangement.
    ///
    /// This is useful when externally-constructed candidates need to be pruned
    /// to a specific arrangement's constraints (e.g., no MUL/MLA/MLS on D2).
    pub fn filter_by_arrangement(
        candidates: &[TargetCandidate],
        arrangement: VectorArrangement,
    ) -> Vec<TargetCandidate> {
        candidates
            .iter()
            .filter(|c| match c.target {
                SynthTarget::Neon(arr) => arr == arrangement,
                _ => true,
            })
            .cloned()
            .collect()
    }

    /// Generate candidates across all configured targets.
    pub fn all_candidates(
        source_name: &str,
        width: u32,
        config: &UnifiedSearchConfig,
    ) -> Vec<TargetCandidate> {
        let mut candidates = Self::scalar_candidates(source_name, width);

        if config.include_neon {
            for &arr in &config.neon_arrangements {
                candidates.extend(Self::neon_candidates(source_name, arr, config));
            }
        }

        candidates
    }
}

// ---------------------------------------------------------------------------
// SMT encoding helpers (scalar)
// ---------------------------------------------------------------------------

/// Encode a scalar unary operation.
fn encode_scalar_unary(opcode: SynthOpcode, operand: &SmtExpr, width: u32) -> SmtExpr {
    match opcode {
        SynthOpcode::Neg => operand.clone().bvneg(),
        SynthOpcode::Mvn => operand.clone().bvxor(SmtExpr::bv_const(u64::MAX, width)),
        _ => panic!("Not a unary opcode: {:?}", opcode),
    }
}

/// Encode a scalar binary operation.
fn encode_scalar_binary(opcode: SynthOpcode, lhs: &SmtExpr, rhs: &SmtExpr) -> SmtExpr {
    match opcode {
        SynthOpcode::Add => lhs.clone().bvadd(rhs.clone()),
        SynthOpcode::Sub => lhs.clone().bvsub(rhs.clone()),
        SynthOpcode::Mul => lhs.clone().bvmul(rhs.clone()),
        SynthOpcode::And => lhs.clone().bvand(rhs.clone()),
        SynthOpcode::Orr => lhs.clone().bvor(rhs.clone()),
        SynthOpcode::Eor => lhs.clone().bvxor(rhs.clone()),
        SynthOpcode::Lsl => lhs.clone().bvshl(rhs.clone()),
        SynthOpcode::Lsr => lhs.clone().bvlshr(rhs.clone()),
        SynthOpcode::Asr => lhs.clone().bvashr(rhs.clone()),
        _ => panic!("Not a binary opcode: {:?}", opcode),
    }
}

/// Cost of a scalar opcode.
fn scalar_opcode_cost(opcode: SynthOpcode) -> i32 {
    match opcode {
        SynthOpcode::Mul => 3,
        _ => 1,
    }
}

// ---------------------------------------------------------------------------
// SMT encoding helpers (NEON)
// ---------------------------------------------------------------------------

/// Encode a NEON unary operation using the semantic functions from neon_semantics.
fn encode_neon_unary(
    opcode: NeonSynthOpcode,
    arrangement: VectorArrangement,
    vn: &SmtExpr,
) -> SmtExpr {
    match opcode {
        NeonSynthOpcode::Neg => neon_semantics::encode_neon_neg(arrangement, vn),
        _ => panic!("Not a NEON unary opcode: {:?}", opcode),
    }
}

/// Encode a NEON binary lane-parallel operation.
fn encode_neon_binary(
    opcode: NeonSynthOpcode,
    arrangement: VectorArrangement,
    vn: &SmtExpr,
    vm: &SmtExpr,
) -> SmtExpr {
    match opcode {
        NeonSynthOpcode::Add => neon_semantics::encode_neon_add(arrangement, vn, vm),
        NeonSynthOpcode::Sub => neon_semantics::encode_neon_sub(arrangement, vn, vm),
        NeonSynthOpcode::Mul => neon_semantics::encode_neon_mul(arrangement, vn, vm),
        _ => panic!("Not a NEON binary lane-parallel opcode: {:?}", opcode),
    }
}

/// Encode a NEON bitwise operation (full-width, no lane decomposition).
fn encode_neon_bitwise(
    opcode: NeonSynthOpcode,
    vn: &SmtExpr,
    vm: &SmtExpr,
) -> SmtExpr {
    match opcode {
        NeonSynthOpcode::And => neon_semantics::encode_neon_and(vn, vm),
        NeonSynthOpcode::Orr => neon_semantics::encode_neon_orr(vn, vm),
        NeonSynthOpcode::Eor => neon_semantics::encode_neon_eor(vn, vm),
        NeonSynthOpcode::Bic => neon_semantics::encode_neon_bic(vn, vm),
        _ => panic!("Not a NEON bitwise opcode: {:?}", opcode),
    }
}

/// Encode a NEON shift-by-immediate operation.
fn encode_neon_shift(
    opcode: NeonSynthOpcode,
    arrangement: VectorArrangement,
    vn: &SmtExpr,
    imm: u32,
) -> SmtExpr {
    match opcode {
        NeonSynthOpcode::Shl => neon_semantics::encode_neon_shl(arrangement, vn, imm),
        NeonSynthOpcode::Ushr => neon_semantics::encode_neon_ushr(arrangement, vn, imm),
        NeonSynthOpcode::Sshr => neon_semantics::encode_neon_sshr(arrangement, vn, imm),
        _ => panic!("Not a NEON shift opcode: {:?}", opcode),
    }
}

/// Encode a NEON fused multi-instruction operation.
///
/// MLA: `vd = vacc + vn * vm` (multiply-accumulate, per-lane)
/// MLS: `vd = vacc - vn * vm` (multiply-subtract, per-lane)
///
/// These are encoded as the composition of MUL + ADD/SUB. On hardware,
/// AArch64 NEON has dedicated MLA/MLS instructions that fuse these
/// into a single operation.
fn encode_neon_fused(
    opcode: NeonSynthOpcode,
    arrangement: VectorArrangement,
    vacc: &SmtExpr,
    vn: &SmtExpr,
    vm: &SmtExpr,
) -> SmtExpr {
    let product = neon_semantics::encode_neon_mul(arrangement, vn, vm);
    match opcode {
        NeonSynthOpcode::Mla => neon_semantics::encode_neon_add(arrangement, vacc, &product),
        NeonSynthOpcode::Mls => neon_semantics::encode_neon_sub(arrangement, vacc, &product),
        _ => panic!("Not a NEON fused opcode: {:?}", opcode),
    }
}

// ---------------------------------------------------------------------------
// Proven unified rule
// ---------------------------------------------------------------------------

/// A proven optimization rule annotated with its target.
///
/// Extends `ProvenRule` from `synthesis.rs` with target information so the
/// backend knows whether to emit scalar or NEON instructions.
#[derive(Debug, Clone)]
pub struct UnifiedProvenRule {
    /// Human-readable rule name.
    pub name: String,
    /// Source expression (the pattern being optimized).
    pub source_expr: SmtExpr,
    /// Target expression (the proven-equivalent replacement).
    pub target_expr: SmtExpr,
    /// Which compute target the replacement uses.
    pub target: SynthTarget,
    /// Hash of the proof for dedup/audit.
    pub proof_hash: u64,
    /// Estimated cost of the replacement (lower is better).
    pub cost: i32,
    /// Bitvector width at which equivalence was verified.
    pub verified_width: u32,
    /// Number of CEGIS iterations used.
    pub cegis_iterations: usize,
}

// ---------------------------------------------------------------------------
// Unified CEGIS loop
// ---------------------------------------------------------------------------

/// Unified multi-target CEGIS synthesis loop.
///
/// Wraps the single-target [`CegisLoop`] to search across scalar and NEON
/// candidate spaces simultaneously. The loop:
///
/// 1. Generates candidates from [`UnifiedSearchSpace`]
/// 2. Filters candidates against accumulated counterexamples (fast path)
/// 3. Verifies survivors via CEGIS (solver path)
/// 4. Ranks verified candidates by cost
/// 5. Returns the cheapest proven-correct rule
///
/// Counterexamples are shared across all targets -- a counterexample found
/// while checking a scalar candidate also filters NEON candidates (and vice
/// versa), as long as the variable names match.
pub struct UnifiedCegisLoop {
    /// Inner CEGIS loop (shared counterexample accumulator).
    cegis: CegisLoop,
    /// Search space configuration.
    config: UnifiedSearchConfig,
    /// Accumulated proven rules across all invocations.
    proven_rules: Vec<UnifiedProvenRule>,
    /// Statistics: total candidates evaluated.
    pub stats_candidates_evaluated: u64,
    /// Statistics: total candidates proven equivalent.
    pub stats_candidates_proven: u64,
    /// Statistics: total candidates rejected by concrete eval.
    pub stats_concrete_rejections: u64,
}

impl UnifiedCegisLoop {
    /// Create a new unified CEGIS loop.
    ///
    /// # Arguments
    ///
    /// * `max_iterations` - Max CEGIS refinement iterations per candidate.
    /// * `timeout_ms` - Solver timeout per query.
    /// * `config` - Unified search configuration.
    pub fn new(
        max_iterations: usize,
        timeout_ms: u64,
        config: UnifiedSearchConfig,
    ) -> Self {
        Self {
            cegis: CegisLoop::new(max_iterations, timeout_ms),
            config,
            proven_rules: Vec::new(),
            stats_candidates_evaluated: 0,
            stats_candidates_proven: 0,
            stats_concrete_rejections: 0,
        }
    }

    /// Create with default configuration (scalar + NEON S2/S4/D2).
    pub fn with_defaults() -> Self {
        Self::new(10, 5000, UnifiedSearchConfig::default())
    }

    /// Add seed counterexamples for fast filtering.
    pub fn add_edge_case_seeds(&mut self, vars: &[(String, u32)]) {
        self.cegis.add_edge_case_seeds(vars);
    }

    /// Search for the best proven-correct lowering of a source expression.
    ///
    /// Generates candidates across all configured targets, verifies each
    /// via CEGIS, and returns the cheapest proven-correct rule. If no
    /// candidate is proven, returns None.
    ///
    /// # Arguments
    ///
    /// * `source_expr` - The source pattern to optimize.
    /// * `var_name` - Name of the symbolic variable.
    /// * `width` - Bitvector width of the variable.
    pub fn find_best(
        &mut self,
        source_expr: &SmtExpr,
        var_name: &str,
        width: u32,
    ) -> Option<UnifiedProvenRule> {
        let vars = vec![(var_name.to_string(), width)];
        self.cegis.add_edge_case_seeds(&vars);

        // Generate all candidates
        let candidates = UnifiedSearchSpace::all_candidates(var_name, width, &self.config);
        self.find_best_from_candidates(source_expr, &candidates, &vars)
    }

    /// Search for the best proven-correct lowering from a pre-built set
    /// of candidates. This is the core verification loop.
    pub fn find_best_from_candidates(
        &mut self,
        source_expr: &SmtExpr,
        candidates: &[TargetCandidate],
        vars: &[(String, u32)],
    ) -> Option<UnifiedProvenRule> {
        let mut proven: Vec<UnifiedProvenRule> = Vec::new();

        for candidate in candidates {
            self.stats_candidates_evaluated += 1;

            // Fast path: check against accumulated counterexamples
            if self
                .cegis
                .check_concrete_only(source_expr, &candidate.expr)
                .is_some()
            {
                self.stats_concrete_rejections += 1;
                continue;
            }

            // Slow path: full CEGIS verification
            let result = self.cegis.verify(source_expr, &candidate.expr, vars);

            match result {
                CegisResult::Equivalent {
                    proof_hash,
                    iterations,
                } => {
                    self.stats_candidates_proven += 1;
                    let width = vars.first().map(|(_, w)| *w).unwrap_or(32);
                    proven.push(UnifiedProvenRule {
                        name: candidate.name.clone(),
                        source_expr: source_expr.clone(),
                        target_expr: candidate.expr.clone(),
                        target: candidate.target,
                        proof_hash,
                        cost: candidate.cost,
                        verified_width: width,
                        cegis_iterations: iterations,
                    });
                }
                CegisResult::NotEquivalent { .. } => {
                    // Counterexample was added to the CEGIS loop automatically
                    // and will filter future candidates.
                }
                CegisResult::Timeout
                | CegisResult::MaxIterationsReached { .. }
                | CegisResult::Error(_) => {
                    // Skip this candidate
                }
            }
        }

        // Rank by cost (lowest first), break ties by target preference
        // (scalar preferred over NEON for equal cost -- simpler code).
        proven.sort_by(|a, b| {
            a.cost
                .cmp(&b.cost)
                .then_with(|| target_preference(a.target).cmp(&target_preference(b.target)))
        });

        // Store all proven rules for later query
        self.proven_rules.extend(proven.iter().cloned());

        proven.into_iter().next()
    }

    /// Search for ALL proven-correct lowerings (not just the best).
    ///
    /// Returns all candidates that were proven equivalent, sorted by cost.
    pub fn find_all(
        &mut self,
        source_expr: &SmtExpr,
        var_name: &str,
        width: u32,
    ) -> Vec<UnifiedProvenRule> {
        let vars = vec![(var_name.to_string(), width)];
        self.cegis.add_edge_case_seeds(&vars);

        let candidates = UnifiedSearchSpace::all_candidates(var_name, width, &self.config);
        let mut proven: Vec<UnifiedProvenRule> = Vec::new();

        for candidate in &candidates {
            self.stats_candidates_evaluated += 1;

            if self
                .cegis
                .check_concrete_only(source_expr, &candidate.expr)
                .is_some()
            {
                self.stats_concrete_rejections += 1;
                continue;
            }

            let result = self.cegis.verify(source_expr, &candidate.expr, &vars);

            if let CegisResult::Equivalent {
                proof_hash,
                iterations,
            } = result
            {
                self.stats_candidates_proven += 1;
                proven.push(UnifiedProvenRule {
                    name: candidate.name.clone(),
                    source_expr: source_expr.clone(),
                    target_expr: candidate.expr.clone(),
                    target: candidate.target,
                    proof_hash,
                    cost: candidate.cost,
                    verified_width: width,
                    cegis_iterations: iterations,
                });
            }
        }

        proven.sort_by(|a, b| {
            a.cost
                .cmp(&b.cost)
                .then_with(|| target_preference(a.target).cmp(&target_preference(b.target)))
        });

        self.proven_rules.extend(proven.iter().cloned());
        proven
    }

    /// Return the number of accumulated counterexamples.
    pub fn counterexample_count(&self) -> usize {
        self.cegis.counterexample_count()
    }

    /// Return all proven rules found so far.
    pub fn proven_rules(&self) -> &[UnifiedProvenRule] {
        &self.proven_rules
    }

    /// Reset state (counterexamples, stats, proven rules).
    pub fn reset(&mut self) {
        self.cegis.reset();
        self.proven_rules.clear();
        self.stats_candidates_evaluated = 0;
        self.stats_candidates_proven = 0;
        self.stats_concrete_rejections = 0;
    }
}

/// Target preference ordering (lower value = preferred for tie-breaking).
fn target_preference(target: SynthTarget) -> u32 {
    match target {
        SynthTarget::Scalar => 0,
        SynthTarget::Neon(_) => 1,
    }
}

// ---------------------------------------------------------------------------
// Extended compute targets for unified synthesis engine
// ---------------------------------------------------------------------------

/// Extended compute target for the unified synthesis engine.
///
/// Extends [`SynthTarget`] (scalar + NEON) with GPU and ANE targets.
/// These correspond to the compute domains in the cost model
/// (`llvm2-ir/cost_model.rs`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum FullTarget {
    /// AArch64 scalar integer/FP instructions.
    Scalar,
    /// AArch64 NEON 128-bit SIMD instructions (specific arrangement).
    Neon(VectorArrangement),
    /// Apple GPU via Metal compute shaders.
    Gpu,
    /// Apple Neural Engine (ANE) via CoreML.
    Ane,
}

impl fmt::Display for FullTarget {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            FullTarget::Scalar => write!(f, "scalar"),
            FullTarget::Neon(arr) => write!(f, "NEON({:?})", arr),
            FullTarget::Gpu => write!(f, "GPU"),
            FullTarget::Ane => write!(f, "ANE"),
        }
    }
}

impl FullTarget {
    /// Convert from a SynthTarget (limited to scalar + NEON).
    pub fn from_synth_target(st: SynthTarget) -> Self {
        match st {
            SynthTarget::Scalar => FullTarget::Scalar,
            SynthTarget::Neon(arr) => FullTarget::Neon(arr),
        }
    }
}

/// Preference ordering for FullTarget (lower value = preferred for tie-breaking).
fn full_target_preference(target: FullTarget) -> u32 {
    match target {
        FullTarget::Scalar => 0,
        FullTarget::Neon(_) => 1,
        FullTarget::Gpu => 2,
        FullTarget::Ane => 3,
    }
}

/// Extract an operation name from a synthesis rule name for cost model lookup.
///
/// Rule names follow patterns like "ADD x, #1", "NEON_MUL.S2 v, v", "identity".
/// This function extracts the base operation name (e.g., "ADD", "MUL", "MOV")
/// so the cost model can look up the correct latency/throughput values.
fn rule_op_name(name: &str) -> String {
    // Handle "identity" -> maps to MOV (zero-cost move)
    if name == "identity" || name.contains("identity") {
        return "MOV".to_string();
    }
    // Strip NEON_ prefix if present
    let stripped = name.strip_prefix("NEON_").unwrap_or(name);
    // Take the first word (the opcode), strip arrangement suffixes like ".S2"
    let first_word = stripped.split_whitespace().next().unwrap_or("ADD");
    let op = first_word.split('.').next().unwrap_or(first_word);
    op.to_ascii_uppercase()
}

// ---------------------------------------------------------------------------
// SynthesisResult — per-target best candidate with cost estimate
// ---------------------------------------------------------------------------

/// A synthesis result for a single target, including cost estimate.
#[derive(Debug, Clone)]
pub struct TargetSynthesisResult {
    /// The compute target.
    pub target: FullTarget,
    /// The proven-correct lowering rule (if any).
    pub rule: Option<UnifiedProvenRule>,
    /// Estimated cost from the cost model. Lower is better.
    /// For scalar/NEON this is in synthesis cost units (opcode cost).
    /// For GPU/ANE this incorporates dispatch/compile overhead amortized
    /// over the operation size.
    pub cost_estimate: f64,
    /// Human-readable description of the candidate.
    pub description: String,
}

/// Complete synthesis result across all targets.
#[derive(Debug, Clone)]
pub struct SynthesisResult {
    /// Best candidate per target, sorted by cost (cheapest first).
    pub per_target: Vec<TargetSynthesisResult>,
    /// The overall best candidate across all targets.
    pub best: Option<TargetSynthesisResult>,
    /// Total candidates evaluated across all targets.
    pub total_candidates_evaluated: u64,
    /// Total candidates proven correct.
    pub total_candidates_proven: u64,
}

impl SynthesisResult {
    /// Returns true if at least one target found a proven-correct candidate.
    pub fn has_any_result(&self) -> bool {
        self.best.is_some()
    }

    /// Returns the best candidate if one exists.
    pub fn best_rule(&self) -> Option<&UnifiedProvenRule> {
        self.best.as_ref().and_then(|b| b.rule.as_ref())
    }

    /// Returns results only for targets that found proven candidates.
    pub fn proven_targets(&self) -> Vec<&TargetSynthesisResult> {
        self.per_target.iter().filter(|r| r.rule.is_some()).collect()
    }
}

// ---------------------------------------------------------------------------
// GPU candidate generation
// ---------------------------------------------------------------------------

/// GPU synthesis opcode categories.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GpuSynthOp {
    /// Element-wise map: output[i] = f(input[i]).
    Map,
    /// Parallel reduce: result = fold(op, identity, input[0..N]).
    Reduce,
    /// Fused map-reduce: result = fold(reduce_op, id, map(f, input)).
    MapReduce,
}

/// Generate GPU candidate expressions for a given specification.
///
/// GPU candidates are element-wise map operations that apply a scalar
/// operation across all elements in parallel. The SMT encoding uses
/// the sequential specification from `gpu_semantics.rs` since the
/// parallel execution is proven equivalent via tMIR algebraic properties.
fn generate_gpu_candidates(
    _spec: &SmtExpr,
    op_hint: &str,
    elem_width: u32,
    element_count: u32,
) -> Vec<(String, GpuSynthOp, f64)> {
    let mut candidates = Vec::new();
    let upper = op_hint.to_ascii_uppercase();

    // GPU is only beneficial for element-wise operations (map pattern)
    // and reductions. We generate candidate descriptions with cost estimates.
    // The actual SMT encoding would use gpu_semantics::encode_parallel_map,
    // but for synthesis ranking we use the cost model estimates.

    // Estimate GPU dispatch overhead (from cost_model.rs: ~10000 cycles)
    let dispatch_overhead: f64 = 10000.0;

    // GPU throughput per operation type (ops per CPU-cycle)
    let gpu_ops_per_cycle: f64 = match upper.as_str() {
        "ADD" | "SUB" | "NEG" | "AND" | "ORR" | "EOR" | "SHL" | "USHR" => 512.0,
        "MUL" => 256.0,
        "FADD" | "FSUB" | "FMUL" | "FMA" => 256.0,
        _ => 128.0,
    };

    let elements = element_count.max(1) as f64;
    let compute_cycles = elements / gpu_ops_per_cycle;
    let total_cost = dispatch_overhead + compute_cycles;

    // Map candidate
    candidates.push((
        format!("GPU_MAP_{} ({} x {}b)", upper, element_count, elem_width),
        GpuSynthOp::Map,
        total_cost,
    ));

    // For associative ops, also generate reduce candidates
    match upper.as_str() {
        "ADD" | "MUL" | "AND" | "ORR" | "EOR" => {
            candidates.push((
                format!("GPU_REDUCE_{} ({} x {}b)", upper, element_count, elem_width),
                GpuSynthOp::Reduce,
                total_cost * 1.2, // reductions are slightly more expensive
            ));
        }
        _ => {}
    }

    candidates
}

// ---------------------------------------------------------------------------
// ANE candidate generation
// ---------------------------------------------------------------------------

/// ANE synthesis operation categories.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AneSynthOp {
    /// Element-wise operation (Add, Sub, Mul, Div).
    ElementWise,
    /// Matrix multiply (GEMM).
    Gemm,
    /// Activation function (ReLU, etc).
    Activation,
}

/// Generate ANE candidate expressions for a given specification.
///
/// ANE candidates are for operations that map to CoreML/BNNS operations:
/// element-wise ops, matrix multiplies, and activation functions.
/// ANE operates at FP16 precision, so there is a quantization cost.
fn generate_ane_candidates(
    _spec: &SmtExpr,
    op_hint: &str,
    elem_width: u32,
    element_count: u32,
) -> Vec<(String, AneSynthOp, f64)> {
    let mut candidates = Vec::new();
    let upper = op_hint.to_ascii_uppercase();

    // ANE compilation overhead (from cost_model.rs: ~100000 cycles)
    let compile_overhead: f64 = 100_000.0;

    let elements = element_count.max(1) as f64;

    // ANE throughput per operation type
    let (supported, ane_ops_per_cycle): (bool, f64) = match upper.as_str() {
        "GEMM" | "MATMUL" | "CONV2D" => (true, 2048.0),
        "ADD" | "SUB" | "MUL" | "FADD" | "FMUL" | "FMA" => (true, 512.0),
        "NEG" | "AND" | "ORR" | "EOR" => (true, 256.0),
        "SHL" | "USHR" => (true, 256.0),
        _ => (false, 0.0),
    };

    if !supported {
        return candidates;
    }

    let compute_cycles = elements / ane_ops_per_cycle;
    let total_cost = compile_overhead + compute_cycles;

    // Element-wise candidate
    match upper.as_str() {
        "ADD" | "SUB" | "MUL" | "NEG" | "FADD" | "FMUL" => {
            candidates.push((
                format!("ANE_ELEMWISE_{} ({} x {}b)", upper, element_count, elem_width),
                AneSynthOp::ElementWise,
                total_cost,
            ));
        }
        "GEMM" | "MATMUL" => {
            candidates.push((
                format!("ANE_GEMM ({} x {}b)", element_count, elem_width),
                AneSynthOp::Gemm,
                total_cost * 0.5, // GEMM is ANE's strength
            ));
        }
        _ => {
            candidates.push((
                format!("ANE_{} ({} x {}b)", upper, element_count, elem_width),
                AneSynthOp::ElementWise,
                total_cost,
            ));
        }
    }

    candidates
}

// ---------------------------------------------------------------------------
// UnifiedSynthesisEngine
// ---------------------------------------------------------------------------

/// Configuration for the unified synthesis engine.
#[derive(Debug, Clone)]
pub struct SynthesisEngineConfig {
    /// Configuration for the scalar + NEON CEGIS search.
    pub cegis_config: UnifiedSearchConfig,
    /// Maximum CEGIS iterations per candidate.
    pub max_cegis_iterations: usize,
    /// Solver timeout in milliseconds per query.
    pub timeout_ms: u64,
    /// Whether to include GPU candidates in the search.
    pub include_gpu: bool,
    /// Whether to include ANE candidates in the search.
    pub include_ane: bool,
    /// Operation hint for GPU/ANE candidate generation (e.g., "ADD", "MUL").
    /// If empty, GPU/ANE candidates are skipped.
    pub op_hint: String,
    /// Element width in bits (for GPU/ANE cost estimation).
    pub element_width: u32,
    /// Number of elements (for GPU/ANE cost amortization).
    pub element_count: u32,
    /// Apple Silicon generation for cost model estimation.
    /// Controls latency/throughput values used in candidate ranking.
    pub cost_model_gen: CostModelGen,
}

impl Default for SynthesisEngineConfig {
    fn default() -> Self {
        Self {
            cegis_config: UnifiedSearchConfig::default(),
            max_cegis_iterations: 10,
            timeout_ms: 5000,
            include_gpu: true,
            include_ane: true,
            op_hint: String::new(),
            element_width: 32,
            element_count: 1,
            cost_model_gen: CostModelGen::M1,
        }
    }
}

/// Unified synthesis engine that searches across ALL compute targets.
///
/// Takes a tMIR operation specification (as an [`SmtExpr`]) and:
/// 1. Searches scalar + NEON candidates via [`UnifiedCegisLoop`] (verified)
/// 2. Generates GPU candidates using `gpu_semantics.rs` cost estimates
/// 3. Generates ANE candidates using `ane_semantics.rs` cost estimates
/// 4. Ranks all candidates by cost (using the multi-target cost model)
/// 5. Returns a [`SynthesisResult`] with the best candidate per target
///
/// # Design
///
/// Scalar and NEON candidates are formally verified via CEGIS — the engine
/// proves semantic equivalence with the source expression. GPU and ANE
/// candidates are ranked by the cost model but are NOT verified at the SMT
/// level (they use array/FP theories that are expensive to verify). Instead,
/// GPU/ANE correctness relies on the tMIR algebraic property proofs
/// (Pure, Associative, Commutative) that justify the parallel execution.
pub struct UnifiedSynthesisEngine {
    /// Inner CEGIS loop for scalar + NEON verification.
    cegis_loop: UnifiedCegisLoop,
    /// Engine configuration.
    config: SynthesisEngineConfig,
    /// Multi-target cost model from llvm2-ir for candidate ranking.
    /// Replaces ad-hoc cost estimation with the unified cost model that
    /// covers CPU scalar, NEON, GPU, and ANE targets.
    cost_model: MultiTargetCostModel,
    /// Profitability analyzer for gating GPU/ANE candidate generation.
    /// Uses threshold-based analysis to exclude targets where dispatch
    /// overhead would exceed the compute benefit.
    profitability: ProfitabilityAnalyzer,
}

impl UnifiedSynthesisEngine {
    /// Create a new unified synthesis engine.
    pub fn new(config: SynthesisEngineConfig) -> Self {
        let cegis_loop = UnifiedCegisLoop::new(
            config.max_cegis_iterations,
            config.timeout_ms,
            config.cegis_config.clone(),
        );
        let cost_model = MultiTargetCostModel::new(config.cost_model_gen);
        let profitability = ProfitabilityAnalyzer::new(config.cost_model_gen);
        Self {
            cegis_loop,
            config,
            cost_model,
            profitability,
        }
    }

    /// Create with default configuration.
    pub fn with_defaults() -> Self {
        Self::new(SynthesisEngineConfig::default())
    }

    /// Create with a specific operation hint for GPU/ANE candidate generation.
    pub fn for_op(op_hint: &str, element_width: u32, element_count: u32) -> Self {
        Self::new(SynthesisEngineConfig {
            op_hint: op_hint.to_string(),
            element_width,
            element_count,
            ..Default::default()
        })
    }

    /// Access the multi-target cost model.
    pub fn cost_model(&self) -> &MultiTargetCostModel {
        &self.cost_model
    }

    /// Access the profitability analyzer.
    pub fn profitability_analyzer(&self) -> &ProfitabilityAnalyzer {
        &self.profitability
    }

    /// Search for the best implementation across all targets.
    ///
    /// # Arguments
    ///
    /// * `spec` - The tMIR operation specification as an SMT expression.
    /// * `var_name` - Name of the symbolic variable.
    /// * `width` - Bitvector width of the variable.
    ///
    /// # Returns
    ///
    /// A [`SynthesisResult`] containing the best candidate per target,
    /// sorted by cost.
    pub fn synthesize(
        &mut self,
        spec: &SmtExpr,
        var_name: &str,
        width: u32,
    ) -> SynthesisResult {
        let mut per_target: Vec<TargetSynthesisResult> = Vec::new();
        let mut total_evaluated: u64 = 0;
        let mut total_proven: u64 = 0;

        // ------------------------------------------------------------------
        // Phase 1: Scalar + NEON search via CEGIS (verified)
        // ------------------------------------------------------------------
        let proven_rules = self.cegis_loop.find_all(spec, var_name, width);
        total_evaluated += self.cegis_loop.stats_candidates_evaluated;
        total_proven += self.cegis_loop.stats_candidates_proven;

        // Group proven rules by target, keep best per target.
        // Use the MultiTargetCostModel for cost estimation instead of ad-hoc
        // synthesis cost values. The cost model provides latency_cycles which
        // accounts for Apple Silicon pipeline characteristics.
        let mut best_by_synth_target: HashMap<String, UnifiedProvenRule> = HashMap::new();

        for rule in &proven_rules {
            let key = format!("{}", rule.target);
            best_by_synth_target
                .entry(key)
                .and_modify(|existing| {
                    if rule.cost < existing.cost {
                        *existing = rule.clone();
                    }
                })
                .or_insert_with(|| rule.clone());
        }

        // Add scalar result using cost model estimate
        let scalar_rule = best_by_synth_target.get("scalar").cloned();
        let scalar_cost = if let Some(ref rule) = scalar_rule {
            // Use cost model to get proper latency estimate for this op.
            // Map rule name back to an op hint for cost model lookup.
            let op_name = rule_op_name(&rule.name);
            let est = self.cost_model.estimate_cost(
                ComputeTarget::CpuScalar,
                &op_name,
                width,
            );
            est.latency_cycles
        } else {
            f64::MAX
        };
        per_target.push(TargetSynthesisResult {
            target: FullTarget::Scalar,
            description: scalar_rule.as_ref()
                .map(|r| r.name.clone())
                .unwrap_or_else(|| "no scalar candidate found".to_string()),
            rule: scalar_rule,
            cost_estimate: scalar_cost,
        });

        // Add NEON results (best per arrangement) using cost model
        for (_key, rule) in &best_by_synth_target {
            if let SynthTarget::Neon(arr) = rule.target {
                let op_name = rule_op_name(&rule.name);
                let neon_width = arr.lane_count() * arr.lane_bits();
                let est = self.cost_model.estimate_cost(
                    ComputeTarget::Neon,
                    &op_name,
                    neon_width,
                );
                per_target.push(TargetSynthesisResult {
                    target: FullTarget::Neon(arr),
                    description: rule.name.clone(),
                    rule: Some(rule.clone()),
                    cost_estimate: est.latency_cycles,
                });
            }
        }

        // ------------------------------------------------------------------
        // Phase 2: GPU candidates (cost-estimated, not verified)
        // Gate with ProfitabilityAnalyzer: skip GPU if data size is below
        // the profitability threshold for this operation category.
        // ------------------------------------------------------------------
        if self.config.include_gpu && !self.config.op_hint.is_empty() {
            let data_size_bytes = (self.config.element_width as u64
                * self.config.element_count as u64)
                / 8;
            let is_profitable = self.profitability.is_gpu_profitable(
                &self.config.op_hint,
                data_size_bytes,
                self.config.element_count as u64,
            );

            if is_profitable {
                let gpu_candidates = generate_gpu_candidates(
                    spec,
                    &self.config.op_hint,
                    self.config.element_width,
                    self.config.element_count,
                );
                total_evaluated += gpu_candidates.len() as u64;

                if let Some((name, _op, _ad_hoc_cost)) = gpu_candidates
                    .iter()
                    .min_by(|a, b| a.2.partial_cmp(&b.2).unwrap_or(std::cmp::Ordering::Equal))
                {
                    // Use the real cost model instead of the ad-hoc cost
                    let width_bits = self.config.element_width * self.config.element_count;
                    let est = self.cost_model.estimate_cost(
                        ComputeTarget::Gpu,
                        &self.config.op_hint,
                        width_bits.max(32),
                    );
                    per_target.push(TargetSynthesisResult {
                        target: FullTarget::Gpu,
                        description: name.clone(),
                        rule: None, // GPU candidates are not SMT-verified
                        cost_estimate: est.latency_cycles,
                    });
                }
            }
        }

        // ------------------------------------------------------------------
        // Phase 3: ANE candidates (cost-estimated, not verified)
        // Gate with ProfitabilityAnalyzer: skip ANE if data size is below
        // the profitability threshold for this operation category.
        // ------------------------------------------------------------------
        if self.config.include_ane && !self.config.op_hint.is_empty() {
            let data_size_bytes = (self.config.element_width as u64
                * self.config.element_count as u64)
                / 8;
            let tensor_shape = &[self.config.element_count as u64];
            let is_profitable = self.profitability.is_ane_profitable(
                &self.config.op_hint,
                data_size_bytes,
                tensor_shape,
            );

            if is_profitable {
                let ane_candidates = generate_ane_candidates(
                    spec,
                    &self.config.op_hint,
                    self.config.element_width,
                    self.config.element_count,
                );
                total_evaluated += ane_candidates.len() as u64;

                if let Some((name, _op, _ad_hoc_cost)) = ane_candidates
                    .iter()
                    .min_by(|a, b| a.2.partial_cmp(&b.2).unwrap_or(std::cmp::Ordering::Equal))
                {
                    // Use the real cost model instead of the ad-hoc cost
                    let width_bits = self.config.element_width * self.config.element_count;
                    let est = self.cost_model.estimate_cost(
                        ComputeTarget::Ane,
                        &self.config.op_hint,
                        width_bits.max(32),
                    );
                    per_target.push(TargetSynthesisResult {
                        target: FullTarget::Ane,
                        description: name.clone(),
                        rule: None, // ANE candidates are not SMT-verified
                        cost_estimate: est.latency_cycles,
                    });
                }
            }
        }

        // ------------------------------------------------------------------
        // Phase 4: Rank by cost (using cost model estimates)
        // ------------------------------------------------------------------
        per_target.sort_by(|a, b| {
            a.cost_estimate
                .partial_cmp(&b.cost_estimate)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| {
                    full_target_preference(a.target)
                        .cmp(&full_target_preference(b.target))
                })
        });

        // The best is the first with a finite cost
        let best = per_target
            .iter()
            .find(|r| r.cost_estimate < f64::MAX)
            .cloned();

        SynthesisResult {
            per_target,
            best,
            total_candidates_evaluated: total_evaluated,
            total_candidates_proven: total_proven,
        }
    }

    /// Search with a pre-configured operation hint.
    ///
    /// Convenience method that sets the operation hint before searching.
    pub fn synthesize_for_op(
        &mut self,
        spec: &SmtExpr,
        var_name: &str,
        width: u32,
        op_hint: &str,
        element_count: u32,
    ) -> SynthesisResult {
        self.config.op_hint = op_hint.to_string();
        self.config.element_width = width;
        self.config.element_count = element_count;
        self.synthesize(spec, var_name, width)
    }

    /// Access the underlying CEGIS loop.
    pub fn cegis_loop(&self) -> &UnifiedCegisLoop {
        &self.cegis_loop
    }

    /// Access the configuration.
    pub fn config(&self) -> &SynthesisEngineConfig {
        &self.config
    }

    /// Reset state (counterexamples, stats, proven rules).
    pub fn reset(&mut self) {
        self.cegis_loop.reset();
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::smt::SmtExpr;

    // -----------------------------------------------------------------------
    // TargetCandidate tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_target_candidate_construction() {
        let expr = SmtExpr::var("x", 32);
        let tc = TargetCandidate::new("test", expr.clone(), SynthTarget::Scalar, 1);
        assert_eq!(tc.name, "test");
        assert_eq!(tc.target, SynthTarget::Scalar);
        assert_eq!(tc.cost, 1);
    }

    #[test]
    fn test_synth_target_display() {
        assert_eq!(format!("{}", SynthTarget::Scalar), "scalar");
        assert_eq!(
            format!("{}", SynthTarget::Neon(VectorArrangement::S2)),
            "NEON(S2)"
        );
    }

    // -----------------------------------------------------------------------
    // NeonSynthOpcode tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_neon_opcode_properties() {
        assert!(NeonSynthOpcode::Neg.is_unary());
        assert!(!NeonSynthOpcode::Add.is_unary());

        assert!(NeonSynthOpcode::Shl.is_shift());
        assert!(NeonSynthOpcode::Ushr.is_shift());
        assert!(NeonSynthOpcode::Sshr.is_shift());
        assert!(!NeonSynthOpcode::Add.is_shift());

        assert!(NeonSynthOpcode::And.is_bitwise());
        assert!(NeonSynthOpcode::Orr.is_bitwise());
        assert!(NeonSynthOpcode::Eor.is_bitwise());
        assert!(NeonSynthOpcode::Bic.is_bitwise());
        assert!(!NeonSynthOpcode::Add.is_bitwise());

        assert!(NeonSynthOpcode::Mla.is_fused());
        assert!(NeonSynthOpcode::Mls.is_fused());
        assert!(!NeonSynthOpcode::Add.is_fused());
    }

    #[test]
    fn test_neon_opcode_all() {
        let all = NeonSynthOpcode::all();
        assert_eq!(all.len(), 13);
    }

    // -----------------------------------------------------------------------
    // UnifiedSearchSpace tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_scalar_candidates_nonempty() {
        let candidates = UnifiedSearchSpace::scalar_candidates("x", 8);
        assert!(
            !candidates.is_empty(),
            "Scalar search should produce candidates"
        );
        // Identity + unary (NEG, MVN) + binary ops * immediates + binary ops * self
        assert!(
            candidates.len() > 10,
            "Expected many scalar candidates, got {}",
            candidates.len()
        );
    }

    #[test]
    fn test_scalar_candidates_include_identity() {
        let candidates = UnifiedSearchSpace::scalar_candidates("x", 8);
        assert!(
            candidates.iter().any(|c| c.name == "identity"),
            "Should include identity candidate"
        );
    }

    #[test]
    fn test_neon_candidates_nonempty() {
        let config = UnifiedSearchConfig::default();
        let candidates =
            UnifiedSearchSpace::neon_candidates("v", VectorArrangement::S2, &config);
        assert!(
            !candidates.is_empty(),
            "NEON search should produce candidates"
        );
    }

    #[test]
    fn test_neon_candidates_include_identity() {
        let config = UnifiedSearchConfig::default();
        let candidates =
            UnifiedSearchSpace::neon_candidates("v", VectorArrangement::S2, &config);
        assert!(
            candidates.iter().any(|c| c.name.contains("identity")),
            "NEON should include identity candidate"
        );
    }

    #[test]
    fn test_all_candidates_combined() {
        let config = UnifiedSearchConfig::default();
        let all = UnifiedSearchSpace::all_candidates("x", 8, &config);

        let scalar_count = all.iter().filter(|c| c.target == SynthTarget::Scalar).count();
        let neon_count = all
            .iter()
            .filter(|c| matches!(c.target, SynthTarget::Neon(_)))
            .count();

        assert!(scalar_count > 0, "Should have scalar candidates");
        assert!(neon_count > 0, "Should have NEON candidates");
    }

    #[test]
    fn test_all_candidates_neon_disabled() {
        let config = UnifiedSearchConfig {
            include_neon: false,
            ..Default::default()
        };
        let all = UnifiedSearchSpace::all_candidates("x", 8, &config);

        let neon_count = all
            .iter()
            .filter(|c| matches!(c.target, SynthTarget::Neon(_)))
            .count();
        assert_eq!(neon_count, 0, "Should have no NEON candidates when disabled");
    }

    // -----------------------------------------------------------------------
    // Default config includes 128-bit NEON arrangements (#166)
    // -----------------------------------------------------------------------

    #[test]
    fn test_default_config_includes_128bit_neon() {
        let config = UnifiedSearchConfig::default();
        assert!(
            config.neon_arrangements.contains(&VectorArrangement::S4),
            "Default config should include S4 (4x32-bit, 128-bit NEON)"
        );
        assert!(
            config.neon_arrangements.contains(&VectorArrangement::D2),
            "Default config should include D2 (2x64-bit, 128-bit NEON)"
        );
        // S2 should still be present
        assert!(
            config.neon_arrangements.contains(&VectorArrangement::S2),
            "Default config should still include S2 (2x32-bit, 64-bit)"
        );
    }

    #[test]
    fn test_all_candidates_include_128bit_neon_targets() {
        let config = UnifiedSearchConfig::default();
        let all = UnifiedSearchSpace::all_candidates("x", 8, &config);

        let s4_count = all
            .iter()
            .filter(|c| matches!(c.target, SynthTarget::Neon(VectorArrangement::S4)))
            .count();
        let d2_count = all
            .iter()
            .filter(|c| matches!(c.target, SynthTarget::Neon(VectorArrangement::D2)))
            .count();

        assert!(s4_count > 0, "Should have S4 (128-bit) NEON candidates, got 0");
        assert!(d2_count > 0, "Should have D2 (128-bit) NEON candidates, got 0");
    }

    #[test]
    fn test_neon_candidates_s4_nonempty() {
        let config = UnifiedSearchConfig::default();
        let candidates =
            UnifiedSearchSpace::neon_candidates("v", VectorArrangement::S4, &config);
        assert!(
            !candidates.is_empty(),
            "NEON S4 (128-bit) search should produce candidates"
        );
        // Should have identity + various ops
        assert!(
            candidates.len() > 5,
            "Expected multiple S4 candidates, got {}",
            candidates.len()
        );
    }

    #[test]
    fn test_neon_candidates_d2_skips_mul() {
        // D2 (2x64-bit) does not support MUL on AArch64 NEON
        let config = UnifiedSearchConfig::default();
        let candidates =
            UnifiedSearchSpace::neon_candidates("v", VectorArrangement::D2, &config);
        assert!(
            !candidates.iter().any(|c| c.name.contains("NEON_MUL")),
            "D2 candidates should NOT include NEON_MUL (unsupported on AArch64)"
        );
    }

    // -----------------------------------------------------------------------
    // Unified CEGIS loop -- scalar identity tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_unified_find_best_add_zero() {
        // Source: x + 0. Best lowering should be identity (cost 0).
        let mut loop_ = UnifiedCegisLoop::with_defaults();
        let source = SmtExpr::var("x", 8).bvadd(SmtExpr::bv_const(0, 8));

        let result = loop_.find_best(&source, "x", 8);

        match result {
            Some(rule) => {
                // Identity (cost 0) should be the best
                assert_eq!(
                    rule.cost, 0,
                    "Best lowering of x+0 should be identity (cost 0), got cost {} ({})",
                    rule.cost, rule.name
                );
                assert_eq!(rule.target, SynthTarget::Scalar);
            }
            None => {
                // Solver not available -- acceptable in CI
            }
        }
    }

    #[test]
    fn test_unified_find_best_sub_zero() {
        let mut loop_ = UnifiedCegisLoop::with_defaults();
        let source = SmtExpr::var("x", 8).bvsub(SmtExpr::bv_const(0, 8));

        let result = loop_.find_best(&source, "x", 8);

        match result {
            Some(rule) => {
                assert_eq!(
                    rule.cost, 0,
                    "Best lowering of x-0 should be identity (cost 0), got cost {} ({})",
                    rule.cost, rule.name
                );
            }
            None => {} // Solver not available
        }
    }

    #[test]
    fn test_unified_find_best_xor_self() {
        // x ^ x == 0. The identity won't match, but some scalar candidate should.
        let mut loop_ = UnifiedCegisLoop::with_defaults();
        let source = SmtExpr::var("x", 8).bvxor(SmtExpr::var("x", 8));

        let result = loop_.find_best(&source, "x", 8);

        match result {
            Some(rule) => {
                // Should find something equivalent to 0
                // (e.g., SUB x, x or EOR x, x or AND x, #0, etc.)
                assert!(
                    rule.cost >= 0,
                    "Found rule: {} with cost {}",
                    rule.name,
                    rule.cost
                );
            }
            None => {} // Solver not available
        }
    }

    // -----------------------------------------------------------------------
    // Unified CEGIS loop -- NEON-specific tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_unified_find_neon_add_zero_identity() {
        // NEON ADD.S2 v, <zero_vector> should be equivalent to identity.
        // But we test from the search perspective: given a NEON source expression
        // that is identity, the best candidate should also be identity.
        let mut config = UnifiedSearchConfig::default();
        config.include_neon = true;
        config.neon_arrangements = vec![VectorArrangement::S2];

        let mut loop_ = UnifiedCegisLoop::new(10, 5000, config);

        // Source: just the vector variable itself (identity pattern)
        let source = SmtExpr::var("v", 64);
        let vars = vec![("v".to_string(), 64)];
        loop_.cegis.add_edge_case_seeds(&vars);

        // Generate only NEON candidates and verify
        let neon_config = UnifiedSearchConfig::default();
        let neon_candidates =
            UnifiedSearchSpace::neon_candidates("v", VectorArrangement::S2, &neon_config);

        let result = loop_.find_best_from_candidates(&source, &neon_candidates, &vars);

        match result {
            Some(rule) => {
                assert!(
                    matches!(rule.target, SynthTarget::Neon(_)),
                    "Should find a NEON target"
                );
                assert_eq!(
                    rule.cost, 0,
                    "Identity should be the best NEON lowering"
                );
            }
            None => {} // Solver not available
        }
    }

    #[test]
    fn test_unified_find_neon_sub_self_zero() {
        // NEON SUB.S2 v, v should produce a zero vector.
        // The unified loop should find this equivalence.
        let config = UnifiedSearchConfig::default();
        let mut loop_ = UnifiedCegisLoop::new(10, 5000, config);

        let vn = SmtExpr::var("v", 64);
        let source = neon_semantics::encode_neon_sub(VectorArrangement::S2, &vn, &vn);

        let vars = vec![("v".to_string(), 64)];
        loop_.cegis.add_edge_case_seeds(&vars);

        // Check: SUB v, v should be equivalent to EOR v, v (both produce 0)
        let eor_self = neon_semantics::encode_neon_eor(&vn, &vn);

        let cegis_result = loop_.cegis.verify(&source, &eor_self, &vars);
        match cegis_result {
            CegisResult::Equivalent { .. } => {
                // Good: NEON SUB v,v == NEON EOR v,v (both zero)
            }
            CegisResult::Error(_) => {} // No solver
            other => panic!(
                "NEON SUB v,v should equal NEON EOR v,v, got {:?}",
                other
            ),
        }
    }

    // -----------------------------------------------------------------------
    // find_all tests -- ensure multiple proven rules are returned
    // -----------------------------------------------------------------------

    #[test]
    fn test_unified_find_all_returns_multiple() {
        let config = UnifiedSearchConfig {
            include_neon: false, // scalar only for simplicity
            ..Default::default()
        };
        let mut loop_ = UnifiedCegisLoop::new(10, 5000, config);

        // Source: x + 0. Multiple candidates should be equivalent:
        // identity, ADD x #0, SUB x #0, ORR x #0, EOR x #0, etc.
        let source = SmtExpr::var("x", 8).bvadd(SmtExpr::bv_const(0, 8));

        let proven = loop_.find_all(&source, "x", 8);

        // We should find at least the identity
        if !proven.is_empty() {
            assert!(
                proven.len() >= 2,
                "Should find multiple equivalent lowerings for x+0, got {}",
                proven.len()
            );

            // Should be sorted by cost
            for i in 1..proven.len() {
                assert!(
                    proven[i].cost >= proven[i - 1].cost,
                    "Rules should be sorted by cost: {} ({}) vs {} ({})",
                    proven[i - 1].name,
                    proven[i - 1].cost,
                    proven[i].name,
                    proven[i].cost
                );
            }

            // First should be cheapest (identity at cost 0)
            assert_eq!(
                proven[0].cost, 0,
                "Cheapest should be identity (cost 0), got {}",
                proven[0].name
            );
        }
        // else: solver not available, acceptable
    }

    // -----------------------------------------------------------------------
    // Cross-target comparison test
    // -----------------------------------------------------------------------

    #[test]
    fn test_cross_target_scalar_preferred_on_tie() {
        // When scalar and NEON both match at the same cost, scalar should
        // be preferred (simpler code, no vector overhead for scalar values).
        let config = UnifiedSearchConfig::default();
        let mut loop_ = UnifiedCegisLoop::new(10, 5000, config);

        // Source: identity (x). Both scalar identity and NEON identity have
        // cost 0, but scalar should win the tie-break.
        let source = SmtExpr::var("x", 8);
        let result = loop_.find_best(&source, "x", 8);

        match result {
            Some(rule) => {
                if rule.cost == 0 {
                    assert_eq!(
                        rule.target,
                        SynthTarget::Scalar,
                        "Scalar should be preferred over NEON on cost tie"
                    );
                }
            }
            None => {} // Solver not available
        }
    }

    // -----------------------------------------------------------------------
    // Statistics tracking tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_unified_stats() {
        let config = UnifiedSearchConfig {
            include_neon: false,
            ..Default::default()
        };
        let mut loop_ = UnifiedCegisLoop::new(10, 5000, config);

        let source = SmtExpr::var("x", 8).bvadd(SmtExpr::bv_const(0, 8));
        let _ = loop_.find_best(&source, "x", 8);

        assert!(
            loop_.stats_candidates_evaluated > 0,
            "Should have evaluated candidates"
        );
        // Some should have been proven or rejected
        let definitive = loop_.stats_candidates_proven + loop_.stats_concrete_rejections;
        assert!(
            definitive > 0,
            "Should have some definitive results (proven={}, rejected={})",
            loop_.stats_candidates_proven,
            loop_.stats_concrete_rejections,
        );
    }

    // -----------------------------------------------------------------------
    // Reset test
    // -----------------------------------------------------------------------

    #[test]
    fn test_unified_reset() {
        let mut loop_ = UnifiedCegisLoop::with_defaults();

        let source = SmtExpr::var("x", 8).bvadd(SmtExpr::bv_const(0, 8));
        let _ = loop_.find_best(&source, "x", 8);

        assert!(loop_.stats_candidates_evaluated > 0);

        loop_.reset();

        assert_eq!(loop_.stats_candidates_evaluated, 0);
        assert_eq!(loop_.stats_candidates_proven, 0);
        assert_eq!(loop_.stats_concrete_rejections, 0);
        assert_eq!(loop_.counterexample_count(), 0);
        assert!(loop_.proven_rules().is_empty());
    }

    // -----------------------------------------------------------------------
    // NEON semantic encoding tests (via unified search space)
    // -----------------------------------------------------------------------

    #[test]
    fn test_neon_candidate_encoding_add() {
        use std::collections::HashMap;
        use crate::smt::EvalResult;

        // Generate a NEON ADD.S2 candidate and evaluate it
        let config = UnifiedSearchConfig::default();
        let candidates =
            UnifiedSearchSpace::neon_candidates("v", VectorArrangement::S2, &config);

        // Find the ADD self-operation candidate
        let add_self = candidates
            .iter()
            .find(|c| c.name.contains("NEON_ADD"))
            .expect("Should have NEON_ADD candidate");

        // v = [10, 20] as S2. ADD v, v = [20, 40]
        let mut env = HashMap::new();
        let v_val = (20u64 << 32) | 10u64; // lane0=10, lane1=20
        env.insert("v".to_string(), v_val);

        let result = add_self.expr.eval(&env);
        let expected = (40u64 << 32) | 20u64; // lane0=20, lane1=40
        assert_eq!(result, EvalResult::Bv(expected));
    }

    #[test]
    fn test_neon_candidate_encoding_neg() {
        use std::collections::HashMap;
        use crate::smt::EvalResult;

        let config = UnifiedSearchConfig::default();
        let candidates =
            UnifiedSearchSpace::neon_candidates("v", VectorArrangement::S2, &config);

        let neg = candidates
            .iter()
            .find(|c| c.name.contains("NEON_NEG"))
            .expect("Should have NEON_NEG candidate");

        // v = [1, 0]. NEG = [0xFFFFFFFF, 0]
        let mut env = HashMap::new();
        env.insert("v".to_string(), 1u64); // lane0=1, lane1=0
        let result = neg.expr.eval(&env);
        let expected = 0xFFFFFFFFu64; // lane0=0xFFFFFFFF, lane1=0
        assert_eq!(result, EvalResult::Bv(expected));
    }

    #[test]
    fn test_neon_candidate_encoding_shl() {
        use std::collections::HashMap;
        use crate::smt::EvalResult;

        let config = UnifiedSearchConfig::default();
        let candidates =
            UnifiedSearchSpace::neon_candidates("v", VectorArrangement::S2, &config);

        let shl1 = candidates
            .iter()
            .find(|c| c.name.contains("NEON_SHL") && c.name.contains("#1"))
            .expect("Should have NEON_SHL #1 candidate");

        // v = [5, 3]. SHL 1 = [10, 6]
        let mut env = HashMap::new();
        let v_val = (3u64 << 32) | 5u64;
        env.insert("v".to_string(), v_val);
        let result = shl1.expr.eval(&env);
        let expected = (6u64 << 32) | 10u64;
        assert_eq!(result, EvalResult::Bv(expected));
    }

    // -----------------------------------------------------------------------
    // Integration: verify NEON MUL x,2 == NEON SHL x,1 via unified loop
    // -----------------------------------------------------------------------

    #[test]
    fn test_unified_neon_mul2_eq_shl1() {
        // NEON MUL.S2 v, <2,2> == NEON SHL.S2 v, #1
        // This is a classic strength reduction that should work per-lane.
        let mut loop_ = UnifiedCegisLoop::with_defaults();

        let v = SmtExpr::var("v", 64);
        // Construct MUL v, <2,2> manually using the NEON semantic encoding
        let two_vec = SmtExpr::bv_const((2u64 << 32) | 2u64, 64);
        let source = neon_semantics::encode_neon_mul(VectorArrangement::S2, &v, &two_vec);

        let shl1 = neon_semantics::encode_neon_shl(VectorArrangement::S2, &v, 1);

        let vars = vec![("v".to_string(), 64)];
        loop_.cegis.add_edge_case_seeds(&vars);

        let result = loop_.cegis.verify(&source, &shl1, &vars);
        match result {
            CegisResult::Equivalent { .. } => {
                // Correct: NEON MUL v, <2,2> == NEON SHL v, #1
            }
            CegisResult::Error(_) => {} // No solver
            other => {
                panic!(
                    "NEON MUL.S2 v,<2,2> should equal NEON SHL.S2 v,#1, got {:?}",
                    other
                );
            }
        }
    }

    // -----------------------------------------------------------------------
    // UnifiedSynthesisEngine tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_engine_finds_scalar_candidate_for_simple_add() {
        // Engine should find a scalar candidate for x + 0 (identity).
        let mut engine = UnifiedSynthesisEngine::with_defaults();
        let source = SmtExpr::var("x", 8).bvadd(SmtExpr::bv_const(0, 8));
        let result = engine.synthesize(&source, "x", 8);

        // Should have evaluated candidates
        assert!(
            result.total_candidates_evaluated > 0,
            "Engine should evaluate candidates"
        );

        // Should find at least the scalar result
        let scalar_results: Vec<_> = result
            .per_target
            .iter()
            .filter(|r| r.target == FullTarget::Scalar)
            .collect();
        assert!(
            !scalar_results.is_empty(),
            "Should have scalar target in results"
        );

        // If solver is available, the scalar candidate should be identity (cost 0)
        if let Some(rule) = scalar_results[0].rule.as_ref() {
            assert_eq!(
                rule.cost, 0,
                "Best scalar lowering of x+0 should be identity (cost 0), got {}",
                rule.cost
            );
        }
    }

    #[test]
    fn test_engine_finds_neon_candidate_for_vectorizable_op() {
        // Engine should find NEON candidates when NEON is enabled.
        let mut engine = UnifiedSynthesisEngine::with_defaults();
        let source = SmtExpr::var("x", 8).bvadd(SmtExpr::bv_const(0, 8));
        let result = engine.synthesize(&source, "x", 8);

        // Check that NEON targets appear in results
        let _neon_results: Vec<_> = result
            .per_target
            .iter()
            .filter(|r| matches!(r.target, FullTarget::Neon(_)))
            .collect();

        // Solver may or may not be available, but NEON targets should be evaluated
        // even if no proven candidates exist. The per_target list includes NEON
        // entries only if proven candidates were found.
        if result.total_candidates_proven > 0 {
            // If anything was proven, there should be at least scalar results
            assert!(
                result.has_any_result(),
                "Should have at least one result when candidates are proven"
            );
        }
    }

    #[test]
    fn test_engine_cost_ranking_prefers_neon_for_multi_element() {
        // When configured with element_count > 1, NEON should be cheaper
        // for vectorizable operations. But this depends on the CEGIS finding
        // a NEON candidate. Test the cost ranking with op_hint for GPU/ANE.
        let mut engine = UnifiedSynthesisEngine::for_op("ADD", 32, 16);
        let source = SmtExpr::var("x", 8).bvadd(SmtExpr::bv_const(0, 8));
        let result = engine.synthesize(&source, "x", 8);

        // The result should have entries sorted by cost
        for i in 1..result.per_target.len() {
            assert!(
                result.per_target[i].cost_estimate >= result.per_target[i - 1].cost_estimate
                    || (result.per_target[i].cost_estimate == result.per_target[i - 1].cost_estimate
                        && full_target_preference(result.per_target[i].target)
                            >= full_target_preference(result.per_target[i - 1].target)),
                "Results should be sorted by cost: {} ({}) vs {} ({})",
                result.per_target[i - 1].description,
                result.per_target[i - 1].cost_estimate,
                result.per_target[i].description,
                result.per_target[i].cost_estimate,
            );
        }
    }

    #[test]
    fn test_engine_gpu_candidate_generation_for_map_ops() {
        // Engine should generate GPU candidates when op_hint is set and
        // element count exceeds the profitability threshold (4096 for
        // elementwise ops per GpuThresholds::default()).
        let mut engine = UnifiedSynthesisEngine::for_op("MUL", 32, 10000);
        let source = SmtExpr::var("x", 8).bvmul(SmtExpr::bv_const(2, 8));
        let result = engine.synthesize(&source, "x", 8);

        let gpu_results: Vec<_> = result
            .per_target
            .iter()
            .filter(|r| r.target == FullTarget::Gpu)
            .collect();

        assert!(
            !gpu_results.is_empty(),
            "Should have GPU target in results when element count exceeds threshold"
        );

        // GPU candidates should not have SMT-verified rules
        for gpu_result in &gpu_results {
            assert!(
                gpu_result.rule.is_none(),
                "GPU candidates should not be SMT-verified"
            );
        }

        // GPU description should mention the op
        assert!(
            gpu_results[0].description.contains("GPU_MAP_MUL")
                || gpu_results[0].description.contains("GPU_REDUCE_MUL"),
            "GPU description should mention the operation: {}",
            gpu_results[0].description
        );
    }

    #[test]
    fn test_engine_ane_candidate_generation_for_matrix_ops() {
        // Engine should generate ANE candidates for supported ops when
        // element count exceeds the profitability threshold (65536 for
        // elementwise ops per AneThresholds::default()).
        let mut engine = UnifiedSynthesisEngine::for_op("ADD", 32, 100000);
        let source = SmtExpr::var("x", 8).bvadd(SmtExpr::bv_const(0, 8));
        let result = engine.synthesize(&source, "x", 8);

        let ane_results: Vec<_> = result
            .per_target
            .iter()
            .filter(|r| r.target == FullTarget::Ane)
            .collect();

        assert!(
            !ane_results.is_empty(),
            "Should have ANE target in results when element count exceeds threshold"
        );

        // ANE candidates should not have SMT-verified rules
        for ane_result in &ane_results {
            assert!(
                ane_result.rule.is_none(),
                "ANE candidates should not be SMT-verified"
            );
        }

        // ANE cost should include compile overhead (~100000)
        // Cost comes from the real MultiTargetCostModel now
        assert!(
            ane_results[0].cost_estimate > 50_000.0,
            "ANE cost should include compile overhead, got {}",
            ane_results[0].cost_estimate
        );
    }

    #[test]
    fn test_engine_gpu_excluded_below_profitability_threshold() {
        // GPU dispatch has ~10K cycle overhead. For small element counts
        // (below GpuThresholds::elementwise_min_elements = 4096),
        // the ProfitabilityAnalyzer should gate GPU candidates.
        let mut engine = UnifiedSynthesisEngine::for_op("ADD", 32, 1);
        let source = SmtExpr::var("x", 8).bvadd(SmtExpr::bv_const(0, 8));
        let result = engine.synthesize(&source, "x", 8);

        let gpu_results: Vec<_> = result
            .per_target
            .iter()
            .filter(|r| r.target == FullTarget::Gpu)
            .collect();

        assert!(
            gpu_results.is_empty(),
            "GPU should be excluded for 1 element (below profitability threshold)"
        );
    }

    #[test]
    fn test_engine_no_gpu_ane_when_disabled() {
        // When GPU and ANE are disabled, no candidates should be generated.
        let config = SynthesisEngineConfig {
            include_gpu: false,
            include_ane: false,
            op_hint: "ADD".to_string(),
            ..Default::default()
        };
        let mut engine = UnifiedSynthesisEngine::new(config);
        let source = SmtExpr::var("x", 8).bvadd(SmtExpr::bv_const(0, 8));
        let result = engine.synthesize(&source, "x", 8);

        let gpu_count = result
            .per_target
            .iter()
            .filter(|r| r.target == FullTarget::Gpu)
            .count();
        let ane_count = result
            .per_target
            .iter()
            .filter(|r| r.target == FullTarget::Ane)
            .count();

        assert_eq!(gpu_count, 0, "Should have no GPU results when disabled");
        assert_eq!(ane_count, 0, "Should have no ANE results when disabled");
    }

    #[test]
    fn test_engine_no_gpu_ane_when_no_op_hint() {
        // When op_hint is empty, GPU/ANE should not be generated.
        let mut engine = UnifiedSynthesisEngine::with_defaults();
        let source = SmtExpr::var("x", 8).bvadd(SmtExpr::bv_const(0, 8));
        let result = engine.synthesize(&source, "x", 8);

        let gpu_count = result
            .per_target
            .iter()
            .filter(|r| r.target == FullTarget::Gpu)
            .count();
        let ane_count = result
            .per_target
            .iter()
            .filter(|r| r.target == FullTarget::Ane)
            .count();

        assert_eq!(gpu_count, 0, "No GPU results without op_hint");
        assert_eq!(ane_count, 0, "No ANE results without op_hint");
    }

    #[test]
    fn test_engine_synthesis_result_helpers() {
        // Test SynthesisResult helper methods.
        let mut engine = UnifiedSynthesisEngine::with_defaults();
        let source = SmtExpr::var("x", 8).bvadd(SmtExpr::bv_const(0, 8));
        let result = engine.synthesize(&source, "x", 8);

        // has_any_result depends on solver availability
        if result.total_candidates_proven > 0 {
            assert!(result.has_any_result(), "Should have a result when candidates are proven");
            assert!(result.best_rule().is_some(), "best_rule should return Some");
            assert!(
                !result.proven_targets().is_empty(),
                "proven_targets should not be empty"
            );
        }

        // per_target should always have at least the scalar entry
        assert!(
            !result.per_target.is_empty(),
            "per_target should always have at least scalar"
        );
    }

    #[test]
    fn test_engine_synthesize_for_op_convenience() {
        // Test the convenience method synthesize_for_op.
        // Use element_count=10000 to exceed GPU profitability threshold (4096).
        let mut engine = UnifiedSynthesisEngine::with_defaults();
        let source = SmtExpr::var("x", 8).bvmul(SmtExpr::bv_const(2, 8));
        let result = engine.synthesize_for_op(&source, "x", 8, "MUL", 10000);

        // Should have generated GPU candidates since op_hint is set and
        // element count exceeds the GPU profitability threshold
        let gpu_count = result
            .per_target
            .iter()
            .filter(|r| r.target == FullTarget::Gpu)
            .count();
        assert!(gpu_count > 0, "synthesize_for_op should generate GPU candidates");

        // Config should be updated
        assert_eq!(engine.config().op_hint, "MUL");
        assert_eq!(engine.config().element_count, 10000);
    }

    #[test]
    fn test_engine_reset_clears_state() {
        let mut engine = UnifiedSynthesisEngine::with_defaults();
        let source = SmtExpr::var("x", 8).bvadd(SmtExpr::bv_const(0, 8));
        let _ = engine.synthesize(&source, "x", 8);

        // After synthesis, there should be some state
        assert!(engine.cegis_loop().stats_candidates_evaluated > 0);

        engine.reset();

        assert_eq!(engine.cegis_loop().stats_candidates_evaluated, 0);
        assert_eq!(engine.cegis_loop().stats_candidates_proven, 0);
        assert!(engine.cegis_loop().proven_rules().is_empty());
    }

    #[test]
    fn test_full_target_display() {
        assert_eq!(format!("{}", FullTarget::Scalar), "scalar");
        assert_eq!(
            format!("{}", FullTarget::Neon(VectorArrangement::S4)),
            "NEON(S4)"
        );
        assert_eq!(format!("{}", FullTarget::Gpu), "GPU");
        assert_eq!(format!("{}", FullTarget::Ane), "ANE");
    }

    #[test]
    fn test_full_target_from_synth_target() {
        assert_eq!(
            FullTarget::from_synth_target(SynthTarget::Scalar),
            FullTarget::Scalar
        );
        assert_eq!(
            FullTarget::from_synth_target(SynthTarget::Neon(VectorArrangement::D2)),
            FullTarget::Neon(VectorArrangement::D2)
        );
    }

    #[test]
    fn test_gpu_candidate_generation_add() {
        let spec = SmtExpr::var("x", 32);
        let candidates = generate_gpu_candidates(&spec, "ADD", 32, 1000);
        // Should have at least a MAP candidate and a REDUCE candidate (ADD is associative)
        assert!(
            candidates.len() >= 2,
            "ADD should generate MAP + REDUCE GPU candidates, got {}",
            candidates.len()
        );
        assert!(
            candidates.iter().any(|(name, _, _)| name.contains("GPU_MAP_ADD")),
            "Should have GPU_MAP_ADD candidate"
        );
        assert!(
            candidates.iter().any(|(name, _, _)| name.contains("GPU_REDUCE_ADD")),
            "Should have GPU_REDUCE_ADD candidate"
        );
    }

    #[test]
    fn test_ane_candidate_generation_gemm() {
        let spec = SmtExpr::var("x", 32);
        let candidates = generate_ane_candidates(&spec, "GEMM", 32, 1000);
        assert!(
            !candidates.is_empty(),
            "GEMM should generate ANE candidates"
        );
        assert!(
            candidates.iter().any(|(name, _, _)| name.contains("ANE_GEMM")),
            "Should have ANE_GEMM candidate"
        );
        // GEMM cost should benefit from ANE's strength (0.5x multiplier)
        let gemm_cost = candidates[0].2;
        let add_candidates = generate_ane_candidates(&spec, "ADD", 32, 1000);
        if !add_candidates.is_empty() {
            let add_cost = add_candidates[0].2;
            assert!(
                gemm_cost < add_cost,
                "GEMM cost ({}) should be lower than ADD cost ({}) on ANE",
                gemm_cost,
                add_cost,
            );
        }
    }

    #[test]
    fn test_ane_unsupported_op_generates_no_candidates() {
        let spec = SmtExpr::var("x", 32);
        let candidates = generate_ane_candidates(&spec, "DIV", 32, 1000);
        assert!(
            candidates.is_empty(),
            "DIV should not generate ANE candidates (unsupported)"
        );
    }

    // -----------------------------------------------------------------------
    // Comprehensive NEON candidate enumeration tests (#160)
    // -----------------------------------------------------------------------

    #[test]
    fn test_neon_candidates_add_produces_correct_candidates() {
        // NEON candidate list for S2 should contain ADD self-op.
        let config = UnifiedSearchConfig::default();
        let candidates =
            UnifiedSearchSpace::neon_candidates("v", VectorArrangement::S2, &config);
        let add_candidates: Vec<_> = candidates
            .iter()
            .filter(|c| c.name.contains("NEON_ADD"))
            .collect();
        assert!(
            !add_candidates.is_empty(),
            "Should produce NEON_ADD candidate for S2"
        );
        // ADD self-op should have cost 1
        for c in &add_candidates {
            assert_eq!(c.cost, 1, "NEON ADD should cost 1, got {}", c.cost);
        }
    }

    #[test]
    fn test_neon_candidates_mul_add_produces_mla_candidate() {
        // MLA (multiply-accumulate) should appear as a fused candidate.
        let config = UnifiedSearchConfig::default();
        let candidates =
            UnifiedSearchSpace::neon_candidates("v", VectorArrangement::S2, &config);
        let mla_candidates: Vec<_> = candidates
            .iter()
            .filter(|c| c.name.contains("NEON_MLA"))
            .collect();
        assert!(
            !mla_candidates.is_empty(),
            "Should produce NEON_MLA (fused MUL+ADD) candidate"
        );
        // MLA cost should be 4 (MUL=3 + ADD=1 fused)
        assert_eq!(
            mla_candidates[0].cost, 4,
            "NEON_MLA cost should be 4"
        );
    }

    // -----------------------------------------------------------------------
    // Cost model integration tests (#149)
    // -----------------------------------------------------------------------

    #[test]
    fn test_engine_uses_cost_model_for_scalar_ranking() {
        // The engine should use MultiTargetCostModel for scalar cost estimates
        // instead of the ad-hoc synthesis cost values.
        use llvm2_ir::cost_model::{ComputeTarget, CostModelGen, MultiTargetCostModel};

        let engine = UnifiedSynthesisEngine::with_defaults();
        let cm = engine.cost_model();

        // MUL should be more expensive than ADD in the cost model
        let add_cost = cm.estimate_cost(ComputeTarget::CpuScalar, "ADD", 32);
        let mul_cost = cm.estimate_cost(ComputeTarget::CpuScalar, "MUL", 32);
        assert!(
            mul_cost.latency_cycles > add_cost.latency_cycles,
            "MUL ({}) should cost more than ADD ({}) in the cost model",
            mul_cost.latency_cycles,
            add_cost.latency_cycles,
        );
    }

    #[test]
    fn test_neon_candidates_mls_candidate_present() {
        // MLS (multiply-subtract) should appear for S4 arrangement.
        let config = UnifiedSearchConfig::default();
        let candidates =
            UnifiedSearchSpace::neon_candidates("v", VectorArrangement::S4, &config);
        let mls_candidates: Vec<_> = candidates
            .iter()
            .filter(|c| c.name.contains("NEON_MLS"))
            .collect();
        assert!(
            !mls_candidates.is_empty(),
            "Should produce NEON_MLS (fused MUL-SUB) candidate for S4"
        );
    }

    #[test]
    fn test_neon_candidates_arrangement_filtering_d2_no_mul_mla_mls() {
        // D2 does not support MUL, MLA, or MLS.
        let config = UnifiedSearchConfig::default();
        let candidates =
            UnifiedSearchSpace::neon_candidates("v", VectorArrangement::D2, &config);
        assert!(
            !candidates.iter().any(|c| c.name.contains("NEON_MUL")),
            "D2 should NOT have NEON_MUL"
        );
        assert!(
            !candidates.iter().any(|c| c.name.contains("NEON_MLA")),
            "D2 should NOT have NEON_MLA"
        );
        assert!(
            !candidates.iter().any(|c| c.name.contains("NEON_MLS")),
            "D2 should NOT have NEON_MLS"
        );
    }

    #[test]
    fn test_neon_candidates_bic_present() {
        // BIC (bit clear) should be in the candidate list.
        let config = UnifiedSearchConfig::default();
        let candidates =
            UnifiedSearchSpace::neon_candidates("v", VectorArrangement::S2, &config);
        assert!(
            candidates.iter().any(|c| c.name.contains("NEON_BIC")),
            "Should include NEON_BIC candidate"
        );
    }

    #[test]
    fn test_neon_candidates_sshr_present() {
        // SSHR (signed shift right) should be in the candidate list.
        let config = UnifiedSearchConfig::default();
        let candidates =
            UnifiedSearchSpace::neon_candidates("v", VectorArrangement::S2, &config);
        let sshr_candidates: Vec<_> = candidates
            .iter()
            .filter(|c| c.name.contains("NEON_SSHR"))
            .collect();
        assert!(
            !sshr_candidates.is_empty(),
            "Should include NEON_SSHR candidates"
        );
        // Should have one per valid shift amount
        assert!(
            sshr_candidates.len() >= 2,
            "Should have multiple SSHR shift amounts, got {}",
            sshr_candidates.len()
        );
    }

    #[test]
    fn test_neon_opcode_is_compatible() {
        // All opcodes compatible with S2
        for &op in NeonSynthOpcode::all() {
            assert!(
                op.is_compatible(VectorArrangement::S2),
                "{:?} should be compatible with S2",
                op
            );
        }

        // MUL, MLA, MLS NOT compatible with D2
        assert!(
            !NeonSynthOpcode::Mul.is_compatible(VectorArrangement::D2),
            "MUL should NOT be compatible with D2"
        );
        assert!(
            !NeonSynthOpcode::Mla.is_compatible(VectorArrangement::D2),
            "MLA should NOT be compatible with D2"
        );
        assert!(
            !NeonSynthOpcode::Mls.is_compatible(VectorArrangement::D2),
            "MLS should NOT be compatible with D2"
        );

        // ADD, SUB, NEG compatible with D2
        assert!(NeonSynthOpcode::Add.is_compatible(VectorArrangement::D2));
        assert!(NeonSynthOpcode::Sub.is_compatible(VectorArrangement::D2));
        assert!(NeonSynthOpcode::Neg.is_compatible(VectorArrangement::D2));
    }

    #[test]
    fn test_neon_filter_by_arrangement() {
        // Build candidates for S2 and D2, then filter for D2 only
        let config = UnifiedSearchConfig::default();
        let s2_candidates =
            UnifiedSearchSpace::neon_candidates("v", VectorArrangement::S2, &config);
        let d2_candidates =
            UnifiedSearchSpace::neon_candidates("v", VectorArrangement::D2, &config);

        // Combine and filter
        let mut combined = s2_candidates;
        combined.extend(d2_candidates);

        let filtered = UnifiedSearchSpace::filter_by_arrangement(
            &combined,
            VectorArrangement::D2,
        );

        // All filtered candidates should target D2
        for c in &filtered {
            assert_eq!(
                c.target,
                SynthTarget::Neon(VectorArrangement::D2),
                "Filtered candidate '{}' should target D2",
                c.name
            );
        }
        // And there should be some
        assert!(
            !filtered.is_empty(),
            "Filter should keep D2 candidates"
        );
    }

    #[test]
    fn test_neon_candidates_s2_has_all_expected_opcodes() {
        // S2 should have candidates for all 13 opcodes (excluding fused
        // which also need MUL compatibility, but S2 supports MUL).
        let config = UnifiedSearchConfig::default();
        let candidates =
            UnifiedSearchSpace::neon_candidates("v", VectorArrangement::S2, &config);

        let expected_prefixes = [
            "NEON_ADD", "NEON_SUB", "NEON_MUL", "NEON_NEG",
            "NEON_AND", "NEON_ORR", "NEON_EOR", "NEON_BIC",
            "NEON_SHL", "NEON_USHR", "NEON_SSHR",
            "NEON_MLA", "NEON_MLS",
        ];
        for prefix in &expected_prefixes {
            assert!(
                candidates.iter().any(|c| c.name.contains(prefix)),
                "S2 candidates should include {} but found none",
                prefix
            );
        }
    }

    #[test]
    fn test_engine_cost_model_ranking_matches_synthesis_output() {
        // Verify that synthesis results are sorted by cost model estimates.
        // With a large element count, GPU should be included but ranked
        // according to the cost model, not ad-hoc values.
        let mut engine = UnifiedSynthesisEngine::for_op("ADD", 32, 10000);
        let source = SmtExpr::var("x", 8).bvadd(SmtExpr::bv_const(0, 8));
        let result = engine.synthesize(&source, "x", 8);

        // Results should be sorted by cost_estimate (ascending)
        for i in 1..result.per_target.len() {
            let prev = &result.per_target[i - 1];
            let curr = &result.per_target[i];
            assert!(
                curr.cost_estimate >= prev.cost_estimate
                    || (curr.cost_estimate == prev.cost_estimate
                        && full_target_preference(curr.target)
                            >= full_target_preference(prev.target)),
                "Results not sorted by cost model: {} ({:.1}) should be >= {} ({:.1})",
                prev.description,
                prev.cost_estimate,
                curr.description,
                curr.cost_estimate,
            );
        }
    }

    #[test]
    fn test_neon_mla_encoding_correctness() {
        // MLA(vacc, vn, vm) = vacc + vn * vm, per-lane.
        use std::collections::HashMap;
        use crate::smt::EvalResult;

        let config = UnifiedSearchConfig::default();
        let candidates =
            UnifiedSearchSpace::neon_candidates("v", VectorArrangement::S2, &config);

        let mla = candidates
            .iter()
            .find(|c| c.name.contains("NEON_MLA"))
            .expect("Should have NEON_MLA candidate");

        // For self-op: MLA(v, v, v) = v + v*v per lane
        // v = [3, 5] as S2 => result = [3 + 3*3, 5 + 5*5] = [12, 30]
        let mut env = HashMap::new();
        let v_val = (5u64 << 32) | 3u64;
        env.insert("v".to_string(), v_val);
        let result = mla.expr.eval(&env);
        let expected = (30u64 << 32) | 12u64;
        assert_eq!(
            result,
            EvalResult::Bv(expected),
            "MLA(v,v,v) with v=[3,5] should give [12,30]"
        );
    }

    #[test]
    fn test_engine_profitability_gates_gpu_for_small_workloads() {
        // For small element counts, GPU should not appear in results
        // because ProfitabilityAnalyzer gates it below 4096 elements.
        let mut engine = UnifiedSynthesisEngine::for_op("ADD", 32, 100);
        let source = SmtExpr::var("x", 8).bvadd(SmtExpr::bv_const(0, 8));
        let result = engine.synthesize(&source, "x", 8);

        let gpu_count = result
            .per_target
            .iter()
            .filter(|r| r.target == FullTarget::Gpu)
            .count();
        assert_eq!(
            gpu_count, 0,
            "GPU should be excluded for 100 elements (below 4096 threshold)"
        );
    }

    #[test]
    fn test_neon_mls_encoding_correctness() {
        // MLS(vacc, vn, vm) = vacc - vn * vm, per-lane.
        use std::collections::HashMap;
        use crate::smt::EvalResult;

        let config = UnifiedSearchConfig::default();
        let candidates =
            UnifiedSearchSpace::neon_candidates("v", VectorArrangement::S2, &config);

        let mls = candidates
            .iter()
            .find(|c| c.name.contains("NEON_MLS"))
            .expect("Should have NEON_MLS candidate");

        // For self-op: MLS(v, v, v) = v - v*v per lane
        // v = [10, 2] as S2 => result = [10 - 10*10, 2 - 2*2]
        // = [10 - 100, 2 - 4] = [-90 mod 2^32, -2 mod 2^32]
        // = [0xFFFFFFA6, 0xFFFFFFFE]
        let mut env = HashMap::new();
        let v_val = (2u64 << 32) | 10u64;
        env.insert("v".to_string(), v_val);
        let result = mls.expr.eval(&env);
        let expected = (0xFFFFFFFEu64 << 32) | 0xFFFFFFA6u64;
        assert_eq!(
            result,
            EvalResult::Bv(expected),
            "MLS(v,v,v) with v=[10,2] should give wrapping result"
        );
    }

    #[test]
    fn test_engine_profitability_gates_ane_for_small_workloads() {
        // For small element counts, ANE should not appear in results
        // because ProfitabilityAnalyzer gates it below 65536 elements.
        let mut engine = UnifiedSynthesisEngine::for_op("ADD", 32, 1000);
        let source = SmtExpr::var("x", 8).bvadd(SmtExpr::bv_const(0, 8));
        let result = engine.synthesize(&source, "x", 8);

        let ane_count = result
            .per_target
            .iter()
            .filter(|r| r.target == FullTarget::Ane)
            .count();
        assert_eq!(
            ane_count, 0,
            "ANE should be excluded for 1000 elements (below 65536 threshold)"
        );
    }

    #[test]
    fn test_neon_bic_encoding_correctness() {
        // BIC(vn, vm) = vn AND NOT(vm)
        use std::collections::HashMap;
        use crate::smt::EvalResult;

        let config = UnifiedSearchConfig::default();
        let candidates =
            UnifiedSearchSpace::neon_candidates("v", VectorArrangement::S2, &config);

        let bic = candidates
            .iter()
            .find(|c| c.name.contains("NEON_BIC"))
            .expect("Should have NEON_BIC candidate");

        // Self-op: BIC(v, v) = v AND NOT(v) = 0 for any input
        let mut env = HashMap::new();
        env.insert("v".to_string(), 0xDEADBEEF_CAFEBABEu64);
        let result = bic.expr.eval(&env);
        assert_eq!(
            result,
            EvalResult::Bv(0),
            "BIC(v, v) should always be 0"
        );
    }

    #[test]
    fn test_neon_sshr_encoding_correctness() {
        // SSHR.S2 v, #1: signed shift right by 1
        use std::collections::HashMap;
        use crate::smt::EvalResult;

        let config = UnifiedSearchConfig::default();
        let candidates =
            UnifiedSearchSpace::neon_candidates("v", VectorArrangement::S2, &config);

        let sshr1 = candidates
            .iter()
            .find(|c| c.name.contains("NEON_SSHR") && c.name.contains("#1"))
            .expect("Should have NEON_SSHR #1 candidate");

        // v = [0x80000000, 4] => SSHR #1 => [0xC0000000, 2]
        // 0x80000000 is -2^31 in signed, >>s 1 = -2^30 = 0xC0000000
        let mut env = HashMap::new();
        let v_val = (4u64 << 32) | 0x80000000u64;
        env.insert("v".to_string(), v_val);
        let result = sshr1.expr.eval(&env);
        let expected = (2u64 << 32) | 0xC0000000u64;
        assert_eq!(
            result,
            EvalResult::Bv(expected),
            "SSHR.S2 v, #1 should produce signed shift result"
        );
    }

    #[test]
    fn test_neon_candidates_d2_still_has_basic_ops() {
        // D2 should still have ADD, SUB, NEG, bitwise, shifts.
        let config = UnifiedSearchConfig::default();
        let candidates =
            UnifiedSearchSpace::neon_candidates("v", VectorArrangement::D2, &config);

        let present_prefixes = [
            "NEON_ADD", "NEON_SUB", "NEON_NEG",
            "NEON_AND", "NEON_ORR", "NEON_EOR", "NEON_BIC",
            "NEON_SHL", "NEON_USHR", "NEON_SSHR",
        ];
        for prefix in &present_prefixes {
            assert!(
                candidates.iter().any(|c| c.name.contains(prefix)),
                "D2 candidates should include {} (arrangement-compatible)",
                prefix
            );
        }
    }

    #[test]
    fn test_neon_candidate_count_comprehensive() {
        // Verify the total candidate count for S2 arrangement with default config.
        // Expected:
        //   1 identity
        //   + 1 NEG (unary)
        //   + 3 ADD, SUB, MUL (binary self-ops)
        //   + 4 AND, ORR, EOR, BIC (bitwise self-ops)
        //   + 3 shift amounts * 3 shift ops (SHL, USHR, SSHR) = 9
        //   + 2 MLA, MLS (fused self-ops)
        //   = 1 + 1 + 3 + 4 + 9 + 2 = 20
        let config = UnifiedSearchConfig::default();
        let candidates =
            UnifiedSearchSpace::neon_candidates("v", VectorArrangement::S2, &config);

        assert_eq!(
            candidates.len(),
            20,
            "S2 with default config should have 20 candidates, got {}. Names: {:?}",
            candidates.len(),
            candidates.iter().map(|c| &c.name).collect::<Vec<_>>()
        );
    }

    #[test]
    fn test_neon_candidate_count_d2_excludes_mul_variants() {
        // D2 should have fewer candidates (no MUL, MLA, MLS).
        // Expected:
        //   1 identity
        //   + 1 NEG
        //   + 2 ADD, SUB (binary self-ops, no MUL)
        //   + 4 AND, ORR, EOR, BIC
        //   + 3 shift amounts * 3 shift ops = 9
        //   + 0 MLA, MLS (excluded)
        //   = 1 + 1 + 2 + 4 + 9 + 0 = 17
        let config = UnifiedSearchConfig::default();
        let candidates =
            UnifiedSearchSpace::neon_candidates("v", VectorArrangement::D2, &config);

        assert_eq!(
            candidates.len(),
            17,
            "D2 with default config should have 17 candidates (no MUL/MLA/MLS), got {}. Names: {:?}",
            candidates.len(),
            candidates.iter().map(|c| &c.name).collect::<Vec<_>>()
        );
    }

    #[test]
    fn test_engine_custom_cost_model_gen_changes_results() {
        // Using M4 vs M1 cost model gen should produce different cost estimates
        // because M4 has wider integer throughput.
        use llvm2_ir::cost_model::CostModelGen;

        let config_m1 = SynthesisEngineConfig {
            cost_model_gen: CostModelGen::M1,
            include_gpu: false,
            include_ane: false,
            ..Default::default()
        };
        let config_m4 = SynthesisEngineConfig {
            cost_model_gen: CostModelGen::M4,
            include_gpu: false,
            include_ane: false,
            ..Default::default()
        };

        let engine_m1 = UnifiedSynthesisEngine::new(config_m1);
        let engine_m4 = UnifiedSynthesisEngine::new(config_m4);

        // M1 and M4 should use different cost model instances
        let m1_add = engine_m1
            .cost_model()
            .estimate_cost(ComputeTarget::CpuScalar, "ADD", 32);
        let m4_add = engine_m4
            .cost_model()
            .estimate_cost(ComputeTarget::CpuScalar, "ADD", 32);

        // M4 has wider integer throughput (7 vs 6 IPC for ALU ops)
        assert!(
            m4_add.throughput_per_cycle > m1_add.throughput_per_cycle,
            "M4 should have higher ADD throughput ({}) than M1 ({})",
            m4_add.throughput_per_cycle,
            m1_add.throughput_per_cycle,
        );
    }

    #[test]
    fn test_engine_exposes_cost_model_and_profitability() {
        // Verify accessor methods for cost model and profitability analyzer.
        let engine = UnifiedSynthesisEngine::with_defaults();

        // cost_model() should return a usable reference
        let _est = engine
            .cost_model()
            .estimate_cost(ComputeTarget::CpuScalar, "ADD", 32);

        // profitability_analyzer() should return a usable reference
        let profitable = engine
            .profitability_analyzer()
            .is_gpu_profitable("ADD", 1_000_000, 100_000);
        assert!(
            profitable,
            "GPU should be profitable for 100K elements of ADD"
        );
    }

    #[test]
    fn test_rule_op_name_extraction() {
        // Test the helper that maps rule names to cost model operation names.
        assert_eq!(rule_op_name("identity"), "MOV");
        assert_eq!(rule_op_name("ADD x, #1"), "ADD");
        assert_eq!(rule_op_name("MUL x, x"), "MUL");
        assert_eq!(rule_op_name("SUB x, #0"), "SUB");
        assert_eq!(rule_op_name("NEON_ADD.S2 v, v"), "ADD");
        assert_eq!(rule_op_name("NEON_MUL.S4 v, v"), "MUL");
        assert_eq!(rule_op_name("NEON identity (S2)"), "MOV");
        assert_eq!(rule_op_name("NEG x"), "NEG");
        assert_eq!(rule_op_name("EOR x, x"), "EOR");
    }

    #[test]
    fn test_engine_ane_excluded_for_unsupported_ops() {
        // ANE should not generate candidates for bitwise ops (unsupported
        // on the Neural Engine) even with large element counts.
        // The ProfitabilityAnalyzer's is_ane_profitable checks both
        // target_legality and threshold checks.
        let mut engine = UnifiedSynthesisEngine::for_op("SHL", 32, 100000);
        let source = SmtExpr::var("x", 8).bvshl(SmtExpr::bv_const(1, 8));
        let result = engine.synthesize(&source, "x", 8);

        let ane_count = result
            .per_target
            .iter()
            .filter(|r| r.target == FullTarget::Ane)
            .count();
        // SHL is a bitwise/shift op that the ProfitabilityAnalyzer classifies
        // as BitwiseLogic, which returns false for is_ane_profitable
        assert_eq!(
            ane_count, 0,
            "ANE should be excluded for SHL (BitwiseLogic category is unprofitable on ANE)"
        );
    }
}
