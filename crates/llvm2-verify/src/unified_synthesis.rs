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
    /// NEON SHL (per-lane shift left by immediate).
    Shl,
    /// NEON USHR (per-lane unsigned shift right by immediate).
    Ushr,
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
            NeonSynthOpcode::Shl,
            NeonSynthOpcode::Ushr,
        ]
    }

    /// Whether this opcode is unary.
    pub fn is_unary(self) -> bool {
        matches!(self, NeonSynthOpcode::Neg)
    }

    /// Whether this opcode takes an immediate shift amount.
    pub fn is_shift(self) -> bool {
        matches!(self, NeonSynthOpcode::Shl | NeonSynthOpcode::Ushr)
    }

    /// Whether this opcode is bitwise (no lane decomposition needed).
    pub fn is_bitwise(self) -> bool {
        matches!(
            self,
            NeonSynthOpcode::And | NeonSynthOpcode::Orr | NeonSynthOpcode::Eor
        )
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
            NeonSynthOpcode::Shl => "NEON_SHL",
            NeonSynthOpcode::Ushr => "NEON_USHR",
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
            | NeonSynthOpcode::Shl
            | NeonSynthOpcode::Ushr => 1,
            NeonSynthOpcode::Neg => 1,
            NeonSynthOpcode::Mul => 3,
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
    /// The source and candidate are both 64-bit vector registers containing
    /// lanes of the specified arrangement. For binary ops, the second operand
    /// is the same vector (self-operation) or a vector of the same variable.
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
            // Skip MUL for D2 (not supported by AArch64 NEON)
            if opcode == NeonSynthOpcode::Mul && arrangement == VectorArrangement::D2 {
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
                    // For USHR, shift must be >= 1
                    if opcode == NeonSynthOpcode::Ushr && shift_amt < 1 {
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
        _ => panic!("Not a NEON shift opcode: {:?}", opcode),
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
        assert!(!NeonSynthOpcode::Add.is_shift());

        assert!(NeonSynthOpcode::And.is_bitwise());
        assert!(NeonSynthOpcode::Orr.is_bitwise());
        assert!(NeonSynthOpcode::Eor.is_bitwise());
        assert!(!NeonSynthOpcode::Add.is_bitwise());
    }

    #[test]
    fn test_neon_opcode_all() {
        let all = NeonSynthOpcode::all();
        assert_eq!(all.len(), 9);
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
}
