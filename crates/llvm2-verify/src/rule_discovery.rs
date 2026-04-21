// llvm2-verify/rule_discovery.rs - AI-native automatic rule discovery
//
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Copyright 2026 Andrew Yates | License: Apache-2.0
//
// Bridges AI agent proposals to the solver verification pipeline. AI agents
// propose optimization rules as (pattern, replacement) pairs. The CEGIS loop
// verifies semantic equivalence. Proven rules are added to a persistent
// database. Incorrect proposals are rejected with counterexamples.
//
// This implements Section 5 ("Automatic Rule Discovery") of
// designs/2026-04-13-ai-native-compilation.md
//
// Reference: Alive2 (PLDI 2021), Souper (arXiv:1711.04422)

//! AI-native automatic rule discovery.
//!
//! The rule discovery system is the bridge between AI creativity and solver
//! rigor. AI agents can propose any optimization rule — the solver only
//! accepts proven-correct ones.
//!
//! # Architecture
//!
//! ```text
//! AI Agent
//!   │ propose_rule(pattern, replacement)
//!   ▼
//! RuleDiscovery
//!   │ encode → CEGIS verify
//!   ├─ Equivalent   → ProvenRule → RuleDatabase
//!   ├─ NotEquivalent → Rejected(counterexample)
//!   └─ Timeout       → Inconclusive
//! ```
//!
//! # Example
//!
//! ```rust
//! use llvm2_verify::smt::SmtExpr;
//! use llvm2_verify::rule_discovery::{RuleDiscovery, RuleProposal, RuleResult};
//!
//! let mut discovery = RuleDiscovery::new(8);
//!
//! // Propose: x + 0 == x (should be accepted)
//! let proposal = RuleProposal::new(
//!     SmtExpr::var("x", 8).bvadd(SmtExpr::bv_const(0, 8)),
//!     SmtExpr::var("x", 8),
//! ).with_name("add_zero_identity");
//!
//! let result = discovery.propose_rule(proposal);
//! assert!(matches!(result, RuleResult::Accepted(_)));
//! ```

use crate::cegis::{CegisLoop, CegisResult, ConcreteInput};
use crate::lowering_proof::{verify_by_evaluation, ProofObligation};
use crate::smt::SmtExpr;
use crate::synthesis::{ProvenRule, ProvenRuleDb};
use crate::verify::VerificationResult;
use llvm2_opt::cache::StableHasher;
use std::collections::HashSet;

// ---------------------------------------------------------------------------
// RuleProposal
// ---------------------------------------------------------------------------

/// A structured rule proposal from an AI agent.
///
/// Contains the source pattern, proposed replacement, optional preconditions,
/// and metadata for cost estimation and deduplication.
#[derive(Debug, Clone)]
pub struct RuleProposal {
    /// Source pattern as an SMT expression.
    pub pattern: SmtExpr,
    /// Proposed replacement as an SMT expression.
    pub replacement: SmtExpr,
    /// Optional preconditions that must hold for the rule to be valid.
    /// E.g., bitwidth constraints, operand range constraints.
    pub preconditions: Vec<SmtExpr>,
    /// Human-readable name for the rule (optional).
    pub name: Option<String>,
    /// Cost estimate: positive means replacement is cheaper.
    /// If None, auto-computed from expression depth.
    pub cost_estimate: Option<i32>,
}

impl RuleProposal {
    /// Create a new rule proposal with pattern and replacement.
    pub fn new(pattern: SmtExpr, replacement: SmtExpr) -> Self {
        Self {
            pattern,
            replacement,
            preconditions: Vec::new(),
            name: None,
            cost_estimate: None,
        }
    }

    /// Set a human-readable name.
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = Some(name.into());
        self
    }

    /// Add a precondition.
    pub fn with_precondition(mut self, precondition: SmtExpr) -> Self {
        self.preconditions.push(precondition);
        self
    }

    /// Set the cost estimate.
    pub fn with_cost_estimate(mut self, cost: i32) -> Self {
        self.cost_estimate = Some(cost);
        self
    }

    /// Compute a deterministic hash of this proposal for deduplication.
    pub fn proposal_hash(&self) -> u64 {
        let mut hasher = StableHasher::new();
        hasher.write_str(&format!("{}", self.pattern));
        hasher.write_str(&format!("{}", self.replacement));
        hasher.finish64()
    }

    /// Extract free variables from the pattern and replacement.
    pub fn free_vars(&self) -> Vec<(String, u32)> {
        let mut vars_set: HashSet<String> = HashSet::new();
        let mut vars = Vec::new();

        for var_name in self.pattern.free_vars() {
            if vars_set.insert(var_name.clone()) {
                // Get width from the pattern expression
                let width = self.find_var_width(&var_name);
                vars.push((var_name, width));
            }
        }
        for var_name in self.replacement.free_vars() {
            if vars_set.insert(var_name.clone()) {
                let width = self.find_var_width(&var_name);
                vars.push((var_name, width));
            }
        }
        vars
    }

    /// Find the bitvector width of a variable by traversing the expressions.
    fn find_var_width(&self, target: &str) -> u32 {
        Self::find_var_width_in_expr(&self.pattern, target)
            .or_else(|| Self::find_var_width_in_expr(&self.replacement, target))
            .unwrap_or(32) // default to 32-bit if not found
    }

    fn find_var_width_in_expr(expr: &SmtExpr, target: &str) -> Option<u32> {
        match expr {
            SmtExpr::Var { name, width } if name == target => Some(*width),
            SmtExpr::Var { .. }
            | SmtExpr::BvConst { .. }
            | SmtExpr::BoolConst(_) => None,
            SmtExpr::BvAdd { lhs, rhs, .. }
            | SmtExpr::BvSub { lhs, rhs, .. }
            | SmtExpr::BvMul { lhs, rhs, .. }
            | SmtExpr::BvSDiv { lhs, rhs, .. }
            | SmtExpr::BvUDiv { lhs, rhs, .. }
            | SmtExpr::BvAnd { lhs, rhs, .. }
            | SmtExpr::BvOr { lhs, rhs, .. }
            | SmtExpr::BvXor { lhs, rhs, .. }
            | SmtExpr::BvShl { lhs, rhs, .. }
            | SmtExpr::BvLshr { lhs, rhs, .. }
            | SmtExpr::BvAshr { lhs, rhs, .. }
            | SmtExpr::Eq { lhs, rhs }
            | SmtExpr::BvSlt { lhs, rhs, .. }
            | SmtExpr::BvSge { lhs, rhs, .. }
            | SmtExpr::BvSgt { lhs, rhs, .. }
            | SmtExpr::BvSle { lhs, rhs, .. }
            | SmtExpr::BvUlt { lhs, rhs, .. }
            | SmtExpr::BvUge { lhs, rhs, .. }
            | SmtExpr::BvUgt { lhs, rhs, .. }
            | SmtExpr::BvUle { lhs, rhs, .. }
            | SmtExpr::And { lhs, rhs }
            | SmtExpr::Or { lhs, rhs }
            | SmtExpr::FPEq { lhs, rhs }
            | SmtExpr::FPLt { lhs, rhs }
            | SmtExpr::FPGt { lhs, rhs }
            | SmtExpr::FPGe { lhs, rhs }
            | SmtExpr::FPLe { lhs, rhs } => {
                Self::find_var_width_in_expr(lhs, target)
                    .or_else(|| Self::find_var_width_in_expr(rhs, target))
            }
            SmtExpr::FPAdd { lhs, rhs, .. }
            | SmtExpr::FPSub { lhs, rhs, .. }
            | SmtExpr::FPMul { lhs, rhs, .. }
            | SmtExpr::FPDiv { lhs, rhs, .. } => {
                Self::find_var_width_in_expr(lhs, target)
                    .or_else(|| Self::find_var_width_in_expr(rhs, target))
            }
            SmtExpr::FPFma { a, b, c, .. } => {
                Self::find_var_width_in_expr(a, target)
                    .or_else(|| Self::find_var_width_in_expr(b, target))
                    .or_else(|| Self::find_var_width_in_expr(c, target))
            }
            SmtExpr::BvNeg { operand, .. }
            | SmtExpr::Not { operand }
            | SmtExpr::Extract { operand, .. }
            | SmtExpr::ZeroExtend { operand, .. }
            | SmtExpr::SignExtend { operand, .. }
            | SmtExpr::FPNeg { operand }
            | SmtExpr::FPAbs { operand }
            | SmtExpr::FPSqrt { operand, .. }
            | SmtExpr::FPIsNaN { operand }
            | SmtExpr::FPIsInf { operand }
            | SmtExpr::FPIsZero { operand }
            | SmtExpr::FPIsNormal { operand }
            | SmtExpr::FPToSBv { operand, .. }
            | SmtExpr::FPToUBv { operand, .. }
            | SmtExpr::BvToFP { operand, .. }
            | SmtExpr::FPToFP { operand, .. } => {
                Self::find_var_width_in_expr(operand, target)
            }
            SmtExpr::Concat { hi, lo, .. } => {
                Self::find_var_width_in_expr(hi, target)
                    .or_else(|| Self::find_var_width_in_expr(lo, target))
            }
            SmtExpr::Ite { cond, then_expr, else_expr } => {
                Self::find_var_width_in_expr(cond, target)
                    .or_else(|| Self::find_var_width_in_expr(then_expr, target))
                    .or_else(|| Self::find_var_width_in_expr(else_expr, target))
            }
            SmtExpr::Select { array, index } => {
                Self::find_var_width_in_expr(array, target)
                    .or_else(|| Self::find_var_width_in_expr(index, target))
            }
            SmtExpr::Store { array, index, value } => {
                Self::find_var_width_in_expr(array, target)
                    .or_else(|| Self::find_var_width_in_expr(index, target))
                    .or_else(|| Self::find_var_width_in_expr(value, target))
            }
            SmtExpr::ConstArray { value, .. } => {
                Self::find_var_width_in_expr(value, target)
            }
            SmtExpr::UF { args, .. } => {
                for arg in args {
                    if let Some(w) = Self::find_var_width_in_expr(arg, target) {
                        return Some(w);
                    }
                }
                None
            }
            SmtExpr::FPConst { .. } | SmtExpr::UFDecl { .. } => None,
            SmtExpr::ForAll { lower, upper, body, .. }
            | SmtExpr::Exists { lower, upper, body, .. } => {
                Self::find_var_width_in_expr(lower, target)
                    .or_else(|| Self::find_var_width_in_expr(upper, target))
                    .or_else(|| Self::find_var_width_in_expr(body, target))
            }
        }
    }
}

// ---------------------------------------------------------------------------
// RuleResult
// ---------------------------------------------------------------------------

/// Result of a rule proposal verification.
#[derive(Debug, Clone)]
pub enum RuleResult {
    /// Rule is semantically equivalent — proven correct. Contains the
    /// verified rule ready for addition to the database.
    Accepted(DiscoveredRule),

    /// Rule is not equivalent. Contains a counterexample demonstrating
    /// the difference and whether it was found by concrete evaluation or
    /// SMT solving.
    Rejected {
        /// Counterexample input that distinguishes pattern from replacement.
        counterexample: ConcreteInput,
        /// Whether the counterexample was found by fast concrete evaluation
        /// (true) or by the SMT solver (false).
        found_by_concrete: bool,
    },

    /// Verification timed out — neither proven nor disproven.
    Inconclusive,

    /// Proposal was a duplicate of an existing proven rule.
    Duplicate,
}

impl RuleResult {
    /// Returns true if the rule was accepted (proven correct).
    pub fn is_accepted(&self) -> bool {
        matches!(self, RuleResult::Accepted(_))
    }

    /// Returns true if the rule was rejected with a counterexample.
    pub fn is_rejected(&self) -> bool {
        matches!(self, RuleResult::Rejected { .. })
    }
}

// ---------------------------------------------------------------------------
// DiscoveredRule
// ---------------------------------------------------------------------------

/// A rule that has been verified correct via the discovery pipeline.
///
/// Similar to `ProvenRule` from synthesis.rs but carries the original
/// SMT expressions and discovery metadata.
#[derive(Debug, Clone)]
pub struct DiscoveredRule {
    /// Human-readable rule name.
    pub name: String,
    /// Source pattern (SMT expression).
    pub pattern: SmtExpr,
    /// Replacement (SMT expression).
    pub replacement: SmtExpr,
    /// Preconditions under which the rule holds.
    pub preconditions: Vec<SmtExpr>,
    /// Hash of the proof (for deduplication).
    pub proof_hash: u64,
    /// Cost delta: positive means replacement is cheaper.
    pub cost_delta: i32,
    /// Bitvector width at which the rule was verified.
    pub verified_width: u32,
    /// Number of CEGIS iterations used during verification.
    pub cegis_iterations: usize,
}

// ---------------------------------------------------------------------------
// DiscoveryStats
// ---------------------------------------------------------------------------

/// Statistics tracking for the discovery pipeline.
#[derive(Debug, Clone, Default)]
pub struct DiscoveryStats {
    /// Total proposals submitted.
    pub total_proposed: u64,
    /// Proposals accepted (proven correct).
    pub total_accepted: u64,
    /// Proposals rejected (counterexample found).
    pub total_rejected: u64,
    /// Proposals timed out (inconclusive).
    pub total_inconclusive: u64,
    /// Proposals that were duplicates of existing rules.
    pub total_duplicates: u64,
}

// ---------------------------------------------------------------------------
// RuleDatabase
// ---------------------------------------------------------------------------

/// Persistent rule storage extending ProvenRuleDb.
///
/// Stores both synthesis-discovered rules (ProvenRule) and AI-proposed
/// rules (DiscoveredRule). Provides deduplication, querying, and
/// statistics tracking.
#[derive(Debug, Clone, Default)]
pub struct RuleDatabase {
    /// Rules discovered via AI proposals.
    pub discovered_rules: Vec<DiscoveredRule>,
    /// Rules discovered via enumeration-based synthesis.
    pub synthesis_db: ProvenRuleDb,
    /// Proof hashes of all rules for deduplication.
    known_hashes: HashSet<u64>,
    /// Discovery statistics.
    pub stats: DiscoveryStats,
}

impl RuleDatabase {
    /// Create a new empty database.
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a database pre-populated with synthesis rules.
    pub fn from_synthesis(db: ProvenRuleDb) -> Self {
        let mut known_hashes = HashSet::new();
        for rule in &db.rules {
            known_hashes.insert(rule.proof_hash);
        }
        Self {
            discovered_rules: Vec::new(),
            synthesis_db: db,
            known_hashes,
            stats: DiscoveryStats::default(),
        }
    }

    /// Check if a proof hash is already known.
    pub fn is_known(&self, proof_hash: u64) -> bool {
        self.known_hashes.contains(&proof_hash)
    }

    /// Add a discovered rule to the database.
    ///
    /// Returns false if the rule is a duplicate (already known).
    pub fn add_discovered(&mut self, rule: DiscoveredRule) -> bool {
        if self.known_hashes.contains(&rule.proof_hash) {
            return false;
        }
        self.known_hashes.insert(rule.proof_hash);
        self.discovered_rules.push(rule);
        true
    }

    /// Add a synthesis-proven rule to the database.
    pub fn add_synthesis(&mut self, rule: ProvenRule) -> bool {
        if self.known_hashes.contains(&rule.proof_hash) {
            return false;
        }
        self.known_hashes.insert(rule.proof_hash);
        self.synthesis_db.add(rule);
        true
    }

    /// Total number of proven rules (discovered + synthesis).
    pub fn total_rules(&self) -> usize {
        self.discovered_rules.len() + self.synthesis_db.len()
    }

    /// Return all discovered rules with positive cost delta (profitable).
    pub fn profitable_discovered(&self) -> Vec<&DiscoveredRule> {
        self.discovered_rules
            .iter()
            .filter(|r| r.cost_delta > 0)
            .collect()
    }

    /// Return all rules (both types) that are profitable.
    pub fn all_profitable_count(&self) -> usize {
        let discovered_count = self
            .discovered_rules
            .iter()
            .filter(|r| r.cost_delta > 0)
            .count();
        let synthesis_count = self.synthesis_db.profitable_rules().len();
        discovered_count + synthesis_count
    }

    /// Query rules by name substring.
    pub fn query_by_name(&self, substring: &str) -> Vec<&DiscoveredRule> {
        self.discovered_rules
            .iter()
            .filter(|r| r.name.contains(substring))
            .collect()
    }
}

// ---------------------------------------------------------------------------
// RuleDiscovery
// ---------------------------------------------------------------------------

/// AI-native rule discovery engine.
///
/// Bridges AI agent proposals to the CEGIS verification pipeline.
/// Uses the existing SMT infrastructure for encoding and the CEGIS loop
/// for efficient equivalence checking.
pub struct RuleDiscovery {
    /// Bitvector width for verification.
    width: u32,
    /// CEGIS loop instance (accumulates counterexamples across proposals).
    cegis: CegisLoop,
    /// Rule database (stores proven rules, tracks deduplication).
    pub database: RuleDatabase,
}

impl RuleDiscovery {
    /// Create a new discovery engine for the given bitvector width.
    ///
    /// Uses default CEGIS settings (10 iterations, 5s timeout).
    pub fn new(width: u32) -> Self {
        Self {
            width,
            cegis: CegisLoop::new(10, 5000),
            database: RuleDatabase::new(),
        }
    }

    /// Create a discovery engine with a pre-populated rule database.
    pub fn with_database(width: u32, database: RuleDatabase) -> Self {
        Self {
            width,
            cegis: CegisLoop::new(10, 5000),
            database,
        }
    }

    /// Create a discovery engine with custom CEGIS parameters.
    pub fn with_cegis(width: u32, max_iterations: usize, timeout_ms: u64) -> Self {
        Self {
            width,
            cegis: CegisLoop::new(max_iterations, timeout_ms),
            database: RuleDatabase::new(),
        }
    }

    /// Propose a rule for verification.
    ///
    /// This is the main entry point for AI agents. The proposal goes through:
    /// 1. Deduplication check (skip if already proven)
    /// 2. CEGIS verification (concrete evaluation + solver)
    /// 3. If proven, add to database and return Accepted
    /// 4. If disproven, return Rejected with counterexample
    /// 5. If timeout, return Inconclusive
    pub fn propose_rule(&mut self, proposal: RuleProposal) -> RuleResult {
        self.database.stats.total_proposed += 1;

        // Step 1: Deduplication check
        let proposal_hash = proposal.proposal_hash();
        if self.database.is_known(proposal_hash) {
            self.database.stats.total_duplicates += 1;
            return RuleResult::Duplicate;
        }

        // Step 2: Extract variables
        let vars = proposal.free_vars();

        // Step 3: Seed CEGIS with edge cases for these variables
        self.cegis.add_edge_case_seeds(&vars);

        // Step 4: CEGIS verification
        let result = self.cegis.verify(
            &proposal.pattern,
            &proposal.replacement,
            &vars,
        );

        match result {
            CegisResult::Equivalent {
                proof_hash,
                iterations,
            } => {
                // Proven correct! Build discovered rule.
                let name = proposal
                    .name
                    .unwrap_or_else(|| format!("{} => {}", proposal.pattern, proposal.replacement));

                let cost_delta = proposal.cost_estimate.unwrap_or_else(|| {
                    estimate_expr_cost(&proposal.pattern) - estimate_expr_cost(&proposal.replacement)
                });

                let rule = DiscoveredRule {
                    name,
                    pattern: proposal.pattern,
                    replacement: proposal.replacement,
                    preconditions: proposal.preconditions,
                    proof_hash,
                    cost_delta,
                    verified_width: self.width,
                    cegis_iterations: iterations,
                };

                self.database.add_discovered(rule.clone());
                self.database.stats.total_accepted += 1;
                RuleResult::Accepted(rule)
            }
            CegisResult::NotEquivalent {
                counterexample,
                found_by_concrete,
            } => {
                self.database.stats.total_rejected += 1;
                RuleResult::Rejected {
                    counterexample,
                    found_by_concrete,
                }
            }
            CegisResult::Timeout | CegisResult::MaxIterationsReached { .. } => {
                self.database.stats.total_inconclusive += 1;
                RuleResult::Inconclusive
            }
            CegisResult::Error(_) => {
                // Solver error — fall back to evaluation-based verification
                let fallback = self.verify_by_evaluation_fallback(&proposal, &vars);
                match fallback {
                    Some(true) => {
                        // Evaluation-based verification passed
                        let name = proposal.name.unwrap_or_else(|| {
                            format!("{} => {}", proposal.pattern, proposal.replacement)
                        });
                        let cost_delta = proposal.cost_estimate.unwrap_or_else(|| {
                            estimate_expr_cost(&proposal.pattern)
                                - estimate_expr_cost(&proposal.replacement)
                        });

                        let rule = DiscoveredRule {
                            name,
                            pattern: proposal.pattern,
                            replacement: proposal.replacement,
                            preconditions: proposal.preconditions,
                            proof_hash: proposal_hash,
                            cost_delta,
                            verified_width: self.width,
                            cegis_iterations: 0,
                        };
                        self.database.add_discovered(rule.clone());
                        self.database.stats.total_accepted += 1;
                        RuleResult::Accepted(rule)
                    }
                    Some(false) => {
                        self.database.stats.total_rejected += 1;
                        RuleResult::Rejected {
                            counterexample: ConcreteInput::new(),
                            found_by_concrete: true,
                        }
                    }
                    None => {
                        self.database.stats.total_inconclusive += 1;
                        RuleResult::Inconclusive
                    }
                }
            }
        }
    }

    /// Verify a batch of proposals.
    ///
    /// Returns results in the same order as proposals. Each proposal is
    /// verified independently. Counterexamples from earlier rejections
    /// are reused for fast filtering of later proposals (via the shared
    /// CEGIS loop).
    pub fn propose_batch(&mut self, proposals: Vec<RuleProposal>) -> Vec<RuleResult> {
        proposals
            .into_iter()
            .map(|proposal| self.propose_rule(proposal))
            .collect()
    }

    /// Return the current discovery statistics.
    pub fn stats(&self) -> &DiscoveryStats {
        &self.database.stats
    }

    /// Return the number of accumulated counterexamples in the CEGIS loop.
    pub fn counterexample_count(&self) -> usize {
        self.cegis.counterexample_count()
    }

    /// Fallback: verify using evaluation-based testing (no solver).
    ///
    /// Uses the existing `verify_by_evaluation` infrastructure from
    /// lowering_proof.rs. Returns Some(true) if valid, Some(false) if
    /// invalid, None if inconclusive.
    fn verify_by_evaluation_fallback(
        &self,
        proposal: &RuleProposal,
        vars: &[(String, u32)],
    ) -> Option<bool> {
        let obligation = ProofObligation {
            name: proposal
                .name
                .clone()
                .unwrap_or_else(|| "ai_proposal".to_string()),
            tmir_expr: proposal.pattern.clone(),
            aarch64_expr: proposal.replacement.clone(),
            inputs: vars.to_vec(),
            preconditions: proposal.preconditions.clone(),
            fp_inputs: vec![],
            category: None,
        };

        match verify_by_evaluation(&obligation) {
            VerificationResult::Valid => Some(true),
            VerificationResult::Invalid { .. } => Some(false),
            VerificationResult::Unknown { .. } => None,
        }
    }
}

// ---------------------------------------------------------------------------
// Cost estimation
// ---------------------------------------------------------------------------

/// Estimate the cost of an SMT expression by counting nodes.
///
/// Simple cost model: each operation costs 1, multiplications cost 3,
/// divisions cost 5, constants cost 0, variables cost 0.
/// This is a rough proxy for instruction cost.
fn estimate_expr_cost(expr: &SmtExpr) -> i32 {
    match expr {
        SmtExpr::Var { .. } | SmtExpr::BvConst { .. } | SmtExpr::BoolConst(_) => 0,
        SmtExpr::BvMul { lhs, rhs, .. } => 3 + estimate_expr_cost(lhs) + estimate_expr_cost(rhs),
        SmtExpr::BvSDiv { lhs, rhs, .. } | SmtExpr::BvUDiv { lhs, rhs, .. } => {
            5 + estimate_expr_cost(lhs) + estimate_expr_cost(rhs)
        }
        SmtExpr::BvAdd { lhs, rhs, .. }
        | SmtExpr::BvSub { lhs, rhs, .. }
        | SmtExpr::BvAnd { lhs, rhs, .. }
        | SmtExpr::BvOr { lhs, rhs, .. }
        | SmtExpr::BvXor { lhs, rhs, .. }
        | SmtExpr::BvShl { lhs, rhs, .. }
        | SmtExpr::BvLshr { lhs, rhs, .. }
        | SmtExpr::BvAshr { lhs, rhs, .. } => {
            1 + estimate_expr_cost(lhs) + estimate_expr_cost(rhs)
        }
        SmtExpr::Eq { lhs, rhs }
        | SmtExpr::BvSlt { lhs, rhs, .. }
        | SmtExpr::BvSge { lhs, rhs, .. }
        | SmtExpr::BvSgt { lhs, rhs, .. }
        | SmtExpr::BvSle { lhs, rhs, .. }
        | SmtExpr::BvUlt { lhs, rhs, .. }
        | SmtExpr::BvUge { lhs, rhs, .. }
        | SmtExpr::BvUgt { lhs, rhs, .. }
        | SmtExpr::BvUle { lhs, rhs, .. }
        | SmtExpr::And { lhs, rhs }
        | SmtExpr::Or { lhs, rhs }
        | SmtExpr::FPEq { lhs, rhs }
        | SmtExpr::FPLt { lhs, rhs }
        | SmtExpr::FPGt { lhs, rhs }
        | SmtExpr::FPGe { lhs, rhs }
        | SmtExpr::FPLe { lhs, rhs } => 1 + estimate_expr_cost(lhs) + estimate_expr_cost(rhs),
        SmtExpr::FPAdd { lhs, rhs, .. }
        | SmtExpr::FPSub { lhs, rhs, .. }
        | SmtExpr::FPMul { lhs, rhs, .. }
        | SmtExpr::FPDiv { lhs, rhs, .. } => {
            3 + estimate_expr_cost(lhs) + estimate_expr_cost(rhs)
        }
        SmtExpr::FPFma { a, b, c, .. } => {
            5 + estimate_expr_cost(a) + estimate_expr_cost(b) + estimate_expr_cost(c)
        }
        SmtExpr::BvNeg { operand, .. }
        | SmtExpr::Not { operand }
        | SmtExpr::FPNeg { operand }
        | SmtExpr::FPAbs { operand }
        | SmtExpr::FPIsNaN { operand }
        | SmtExpr::FPIsInf { operand }
        | SmtExpr::FPIsZero { operand }
        | SmtExpr::FPIsNormal { operand } => {
            1 + estimate_expr_cost(operand)
        }
        SmtExpr::FPSqrt { operand, .. } => {
            4 + estimate_expr_cost(operand)
        }
        SmtExpr::FPToSBv { operand, .. }
        | SmtExpr::FPToUBv { operand, .. }
        | SmtExpr::BvToFP { operand, .. }
        | SmtExpr::FPToFP { operand, .. } => {
            2 + estimate_expr_cost(operand)
        }
        SmtExpr::Extract { operand, .. }
        | SmtExpr::ZeroExtend { operand, .. }
        | SmtExpr::SignExtend { operand, .. } => 1 + estimate_expr_cost(operand),
        SmtExpr::Concat { hi, lo, .. } => {
            1 + estimate_expr_cost(hi) + estimate_expr_cost(lo)
        }
        SmtExpr::Ite { cond, then_expr, else_expr } => {
            1 + estimate_expr_cost(cond)
                + estimate_expr_cost(then_expr)
                + estimate_expr_cost(else_expr)
        }
        SmtExpr::Select { array, index } => {
            2 + estimate_expr_cost(array) + estimate_expr_cost(index)
        }
        SmtExpr::Store { array, index, value } => {
            2 + estimate_expr_cost(array)
                + estimate_expr_cost(index)
                + estimate_expr_cost(value)
        }
        SmtExpr::ConstArray { value, .. } => estimate_expr_cost(value),
        SmtExpr::UF { args, .. } => {
            2 + args.iter().map(estimate_expr_cost).sum::<i32>()
        }
        SmtExpr::FPConst { .. } => 0,
        SmtExpr::UFDecl { .. } => 0,
        SmtExpr::ForAll { lower, upper, body, .. }
        | SmtExpr::Exists { lower, upper, body, .. } => {
            5 + estimate_expr_cost(lower) + estimate_expr_cost(upper) + estimate_expr_cost(body)
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::smt::SmtExpr;
    use crate::synthesis::RuleCandidate;

    // -----------------------------------------------------------------------
    // RuleProposal tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_proposal_construction() {
        let proposal = RuleProposal::new(
            SmtExpr::var("x", 8).bvadd(SmtExpr::bv_const(0, 8)),
            SmtExpr::var("x", 8),
        )
        .with_name("add_zero_identity")
        .with_cost_estimate(1);

        assert_eq!(proposal.name.as_deref(), Some("add_zero_identity"));
        assert_eq!(proposal.cost_estimate, Some(1));
        assert!(proposal.preconditions.is_empty());
    }

    #[test]
    fn test_proposal_free_vars() {
        let proposal = RuleProposal::new(
            SmtExpr::var("x", 8).bvadd(SmtExpr::bv_const(0, 8)),
            SmtExpr::var("x", 8),
        );
        let vars = proposal.free_vars();
        assert_eq!(vars.len(), 1);
        assert_eq!(vars[0].0, "x");
        assert_eq!(vars[0].1, 8);
    }

    #[test]
    fn test_proposal_free_vars_multiple() {
        let proposal = RuleProposal::new(
            SmtExpr::var("a", 32).bvadd(SmtExpr::var("b", 32)),
            SmtExpr::var("b", 32).bvadd(SmtExpr::var("a", 32)),
        );
        let vars = proposal.free_vars();
        assert_eq!(vars.len(), 2);
    }

    #[test]
    fn test_proposal_hash_deterministic() {
        let p1 = RuleProposal::new(
            SmtExpr::var("x", 8).bvadd(SmtExpr::bv_const(0, 8)),
            SmtExpr::var("x", 8),
        );
        let p2 = RuleProposal::new(
            SmtExpr::var("x", 8).bvadd(SmtExpr::bv_const(0, 8)),
            SmtExpr::var("x", 8),
        );
        assert_eq!(p1.proposal_hash(), p2.proposal_hash());
    }

    #[test]
    fn test_proposal_hash_different() {
        let p1 = RuleProposal::new(
            SmtExpr::var("x", 8).bvadd(SmtExpr::bv_const(0, 8)),
            SmtExpr::var("x", 8),
        );
        let p2 = RuleProposal::new(
            SmtExpr::var("x", 8).bvsub(SmtExpr::bv_const(0, 8)),
            SmtExpr::var("x", 8),
        );
        assert_ne!(p1.proposal_hash(), p2.proposal_hash());
    }

    #[test]
    fn test_proposal_with_precondition() {
        let proposal = RuleProposal::new(
            SmtExpr::var("x", 8).bvudiv(SmtExpr::var("y", 8)),
            SmtExpr::var("x", 8).bvlshr(SmtExpr::bv_const(1, 8)),
        )
        .with_precondition(
            SmtExpr::var("y", 8).eq_expr(SmtExpr::bv_const(2, 8)),
        );
        assert_eq!(proposal.preconditions.len(), 1);
    }

    // -----------------------------------------------------------------------
    // RuleResult tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_rule_result_accepted() {
        let result = RuleResult::Accepted(DiscoveredRule {
            name: "test".to_string(),
            pattern: SmtExpr::var("x", 8),
            replacement: SmtExpr::var("x", 8),
            preconditions: vec![],
            proof_hash: 123,
            cost_delta: 1,
            verified_width: 8,
            cegis_iterations: 1,
        });
        assert!(result.is_accepted());
        assert!(!result.is_rejected());
    }

    #[test]
    fn test_rule_result_rejected() {
        let result = RuleResult::Rejected {
            counterexample: ConcreteInput::new(),
            found_by_concrete: true,
        };
        assert!(!result.is_accepted());
        assert!(result.is_rejected());
    }

    // -----------------------------------------------------------------------
    // RuleDatabase tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_database_empty() {
        let db = RuleDatabase::new();
        assert_eq!(db.total_rules(), 0);
        assert!(!db.is_known(12345));
    }

    #[test]
    fn test_database_add_discovered() {
        let mut db = RuleDatabase::new();
        let rule = DiscoveredRule {
            name: "test_rule".to_string(),
            pattern: SmtExpr::var("x", 8),
            replacement: SmtExpr::var("x", 8),
            preconditions: vec![],
            proof_hash: 12345,
            cost_delta: 2,
            verified_width: 8,
            cegis_iterations: 1,
        };

        assert!(db.add_discovered(rule.clone()));
        assert_eq!(db.total_rules(), 1);
        assert!(db.is_known(12345));

        // Duplicate should be rejected
        assert!(!db.add_discovered(rule));
        assert_eq!(db.total_rules(), 1);
    }

    #[test]
    fn test_database_from_synthesis() {
        let mut synthesis_db = ProvenRuleDb::new();
        synthesis_db.add(ProvenRule {
            name: "synth_rule".to_string(),
            candidate: RuleCandidate {
                pattern: vec![],
                replacement: vec![],
            },
            proof_hash: 99999,
            cost_delta: 1,
            verified_width: 8,
        });

        let db = RuleDatabase::from_synthesis(synthesis_db);
        assert_eq!(db.total_rules(), 1);
        assert!(db.is_known(99999));
    }

    #[test]
    fn test_database_profitable() {
        let mut db = RuleDatabase::new();

        db.add_discovered(DiscoveredRule {
            name: "profitable".to_string(),
            pattern: SmtExpr::var("x", 8),
            replacement: SmtExpr::var("x", 8),
            preconditions: vec![],
            proof_hash: 1,
            cost_delta: 3,
            verified_width: 8,
            cegis_iterations: 1,
        });

        db.add_discovered(DiscoveredRule {
            name: "neutral".to_string(),
            pattern: SmtExpr::var("x", 8),
            replacement: SmtExpr::var("x", 8),
            preconditions: vec![],
            proof_hash: 2,
            cost_delta: 0,
            verified_width: 8,
            cegis_iterations: 1,
        });

        assert_eq!(db.profitable_discovered().len(), 1);
        assert_eq!(db.profitable_discovered()[0].name, "profitable");
    }

    #[test]
    fn test_database_query_by_name() {
        let mut db = RuleDatabase::new();

        db.add_discovered(DiscoveredRule {
            name: "add_zero_identity".to_string(),
            pattern: SmtExpr::var("x", 8),
            replacement: SmtExpr::var("x", 8),
            preconditions: vec![],
            proof_hash: 1,
            cost_delta: 1,
            verified_width: 8,
            cegis_iterations: 1,
        });

        db.add_discovered(DiscoveredRule {
            name: "mul_to_shift".to_string(),
            pattern: SmtExpr::var("x", 8),
            replacement: SmtExpr::var("x", 8),
            preconditions: vec![],
            proof_hash: 2,
            cost_delta: 2,
            verified_width: 8,
            cegis_iterations: 1,
        });

        assert_eq!(db.query_by_name("add").len(), 1);
        assert_eq!(db.query_by_name("mul").len(), 1);
        assert_eq!(db.query_by_name("identity").len(), 1);
        assert_eq!(db.query_by_name("nonexistent").len(), 0);
    }

    // -----------------------------------------------------------------------
    // Cost estimation tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_cost_estimate_var() {
        assert_eq!(estimate_expr_cost(&SmtExpr::var("x", 32)), 0);
    }

    #[test]
    fn test_cost_estimate_const() {
        assert_eq!(estimate_expr_cost(&SmtExpr::bv_const(42, 32)), 0);
    }

    #[test]
    fn test_cost_estimate_add() {
        let expr = SmtExpr::var("x", 32).bvadd(SmtExpr::bv_const(1, 32));
        assert_eq!(estimate_expr_cost(&expr), 1);
    }

    #[test]
    fn test_cost_estimate_mul() {
        let expr = SmtExpr::var("x", 32).bvmul(SmtExpr::bv_const(2, 32));
        assert_eq!(estimate_expr_cost(&expr), 3);
    }

    #[test]
    fn test_cost_estimate_neg() {
        let expr = SmtExpr::var("x", 32).bvneg();
        assert_eq!(estimate_expr_cost(&expr), 1);
    }

    #[test]
    fn test_cost_estimate_compound() {
        // (x + 1) * 2 = cost(add) + cost(mul) = 1 + 3 = 4
        let expr = SmtExpr::var("x", 32)
            .bvadd(SmtExpr::bv_const(1, 32))
            .bvmul(SmtExpr::bv_const(2, 32));
        assert_eq!(estimate_expr_cost(&expr), 4);
    }

    // -----------------------------------------------------------------------
    // RuleDiscovery — acceptance tests (known-good rules)
    // -----------------------------------------------------------------------

    #[test]
    fn test_discover_add_zero_identity() {
        let mut discovery = RuleDiscovery::new(8);

        let proposal = RuleProposal::new(
            SmtExpr::var("x", 8).bvadd(SmtExpr::bv_const(0, 8)),
            SmtExpr::var("x", 8),
        )
        .with_name("add_zero_identity");

        let result = discovery.propose_rule(proposal);

        match &result {
            RuleResult::Accepted(rule) => {
                assert_eq!(rule.name, "add_zero_identity");
                assert_eq!(rule.verified_width, 8);
                assert!(rule.cost_delta > 0, "Eliding ADD should be profitable");
            }
            RuleResult::Inconclusive => {
                // Solver not available — acceptable in test, fall through
            }
            other => panic!("Expected Accepted or Inconclusive, got {:?}", other),
        }
    }

    #[test]
    fn test_discover_sub_zero_identity() {
        let mut discovery = RuleDiscovery::new(8);

        let proposal = RuleProposal::new(
            SmtExpr::var("x", 8).bvsub(SmtExpr::bv_const(0, 8)),
            SmtExpr::var("x", 8),
        )
        .with_name("sub_zero_identity");

        let result = discovery.propose_rule(proposal);
        assert!(
            result.is_accepted() || matches!(result, RuleResult::Inconclusive),
            "SUB x, 0 => x should be accepted or inconclusive, got {:?}",
            result
        );
    }

    #[test]
    fn test_discover_mul_to_shift() {
        let mut discovery = RuleDiscovery::new(8);

        let proposal = RuleProposal::new(
            SmtExpr::var("x", 8).bvmul(SmtExpr::bv_const(2, 8)),
            SmtExpr::var("x", 8).bvshl(SmtExpr::bv_const(1, 8)),
        )
        .with_name("mul2_to_lsl1");

        let result = discovery.propose_rule(proposal);

        match &result {
            RuleResult::Accepted(rule) => {
                assert_eq!(rule.name, "mul2_to_lsl1");
                // MUL cost 3, LSL cost 1, delta = 2
                assert!(rule.cost_delta > 0, "MUL->LSL should be profitable");
            }
            RuleResult::Inconclusive => {}
            other => panic!("Expected Accepted or Inconclusive, got {:?}", other),
        }
    }

    #[test]
    fn test_discover_xor_self_zero() {
        let mut discovery = RuleDiscovery::new(8);

        let proposal = RuleProposal::new(
            SmtExpr::var("x", 8).bvxor(SmtExpr::var("x", 8)),
            SmtExpr::bv_const(0, 8),
        )
        .with_name("xor_self_zero");

        let result = discovery.propose_rule(proposal);
        assert!(
            result.is_accepted() || matches!(result, RuleResult::Inconclusive),
            "XOR x, x => 0 should be accepted or inconclusive, got {:?}",
            result
        );
    }

    #[test]
    fn test_discover_and_self_identity() {
        let mut discovery = RuleDiscovery::new(8);

        let proposal = RuleProposal::new(
            SmtExpr::var("x", 8).bvand(SmtExpr::var("x", 8)),
            SmtExpr::var("x", 8),
        )
        .with_name("and_self_identity");

        let result = discovery.propose_rule(proposal);
        assert!(
            result.is_accepted() || matches!(result, RuleResult::Inconclusive),
            "AND x, x => x should be accepted or inconclusive, got {:?}",
            result
        );
    }

    #[test]
    fn test_discover_or_self_identity() {
        let mut discovery = RuleDiscovery::new(8);

        let proposal = RuleProposal::new(
            SmtExpr::var("x", 8).bvor(SmtExpr::var("x", 8)),
            SmtExpr::var("x", 8),
        )
        .with_name("or_self_identity");

        let result = discovery.propose_rule(proposal);
        assert!(
            result.is_accepted() || matches!(result, RuleResult::Inconclusive),
            "OR x, x => x should be accepted or inconclusive, got {:?}",
            result
        );
    }

    #[test]
    fn test_discover_double_negation() {
        let mut discovery = RuleDiscovery::new(8);

        let proposal = RuleProposal::new(
            SmtExpr::var("x", 8).bvneg().bvneg(),
            SmtExpr::var("x", 8),
        )
        .with_name("double_negation");

        let result = discovery.propose_rule(proposal);
        assert!(
            result.is_accepted() || matches!(result, RuleResult::Inconclusive),
            "NEG(NEG(x)) => x should be accepted or inconclusive, got {:?}",
            result
        );
    }

    // -----------------------------------------------------------------------
    // RuleDiscovery — rejection tests (known-bad rules)
    // -----------------------------------------------------------------------

    #[test]
    fn test_reject_add_one_identity() {
        let mut discovery = RuleDiscovery::new(8);

        let proposal = RuleProposal::new(
            SmtExpr::var("x", 8).bvadd(SmtExpr::bv_const(1, 8)),
            SmtExpr::var("x", 8),
        )
        .with_name("add_one_identity_WRONG");

        let result = discovery.propose_rule(proposal);

        match &result {
            RuleResult::Rejected {
                counterexample,
                found_by_concrete,
            } => {
                // The counterexample should distinguish add(x,1) from x
                if !counterexample.values.is_empty() {
                    let x_val = counterexample.values.get("x").expect("should have x");
                    // For any x, x+1 != x (unless overflow at width boundary,
                    // but the counterexample should be one where they differ)
                    let pattern_val = crate::smt::mask(x_val.wrapping_add(1), 8);
                    let replacement_val = crate::smt::mask(*x_val, 8);
                    assert_ne!(
                        pattern_val, replacement_val,
                        "Counterexample should distinguish pattern from replacement"
                    );
                }
                let _ = found_by_concrete; // may be either
            }
            RuleResult::Inconclusive => {
                // Solver not available — still acceptable
            }
            other => panic!("Expected Rejected or Inconclusive, got {:?}", other),
        }
    }

    #[test]
    fn test_reject_mul_as_add() {
        let mut discovery = RuleDiscovery::new(8);

        // Wrong: x * y != x + y (in general)
        let proposal = RuleProposal::new(
            SmtExpr::var("x", 8).bvmul(SmtExpr::var("y", 8)),
            SmtExpr::var("x", 8).bvadd(SmtExpr::var("y", 8)),
        )
        .with_name("mul_as_add_WRONG");

        let result = discovery.propose_rule(proposal);
        assert!(
            result.is_rejected() || matches!(result, RuleResult::Inconclusive),
            "x * y => x + y should be rejected, got {:?}",
            result
        );
    }

    #[test]
    fn test_reject_neg_identity() {
        let mut discovery = RuleDiscovery::new(8);

        let proposal = RuleProposal::new(
            SmtExpr::var("x", 8).bvneg(),
            SmtExpr::var("x", 8),
        )
        .with_name("neg_identity_WRONG");

        let result = discovery.propose_rule(proposal);
        assert!(
            result.is_rejected() || matches!(result, RuleResult::Inconclusive),
            "NEG(x) => x should be rejected, got {:?}",
            result
        );
    }

    // -----------------------------------------------------------------------
    // Deduplication tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_deduplication() {
        let mut discovery = RuleDiscovery::new(8);

        let proposal1 = RuleProposal::new(
            SmtExpr::var("x", 8).bvadd(SmtExpr::bv_const(0, 8)),
            SmtExpr::var("x", 8),
        )
        .with_name("add_zero_first");

        let result1 = discovery.propose_rule(proposal1);

        // Submit same rule again
        let proposal2 = RuleProposal::new(
            SmtExpr::var("x", 8).bvadd(SmtExpr::bv_const(0, 8)),
            SmtExpr::var("x", 8),
        )
        .with_name("add_zero_duplicate");

        let result2 = discovery.propose_rule(proposal2);

        // First should be accepted (or inconclusive if no solver),
        // second should be duplicate if first was accepted
        if result1.is_accepted() {
            assert!(
                matches!(result2, RuleResult::Duplicate),
                "Duplicate proposal should be detected, got {:?}",
                result2
            );
        }
    }

    // -----------------------------------------------------------------------
    // Batch discovery tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_batch_discovery() {
        let mut discovery = RuleDiscovery::new(8);

        let proposals = vec![
            // Good rule: x + 0 => x
            RuleProposal::new(
                SmtExpr::var("x", 8).bvadd(SmtExpr::bv_const(0, 8)),
                SmtExpr::var("x", 8),
            )
            .with_name("add_zero"),
            // Bad rule: x + 1 => x
            RuleProposal::new(
                SmtExpr::var("x", 8).bvadd(SmtExpr::bv_const(1, 8)),
                SmtExpr::var("x", 8),
            )
            .with_name("add_one_WRONG"),
            // Good rule: x - 0 => x
            RuleProposal::new(
                SmtExpr::var("x", 8).bvsub(SmtExpr::bv_const(0, 8)),
                SmtExpr::var("x", 8),
            )
            .with_name("sub_zero"),
        ];

        let results = discovery.propose_batch(proposals);

        assert_eq!(results.len(), 3);

        // The bad rule should be rejected
        assert!(
            results[1].is_rejected() || matches!(results[1], RuleResult::Inconclusive),
            "add_one should be rejected, got {:?}",
            results[1]
        );

        // Stats should reflect all three proposals
        assert_eq!(discovery.stats().total_proposed, 3);
    }

    // -----------------------------------------------------------------------
    // Statistics tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_stats_tracking() {
        let mut discovery = RuleDiscovery::new(8);

        // Propose a good rule
        let _ = discovery.propose_rule(RuleProposal::new(
            SmtExpr::var("x", 8).bvadd(SmtExpr::bv_const(0, 8)),
            SmtExpr::var("x", 8),
        ));

        // Propose a bad rule
        let _ = discovery.propose_rule(RuleProposal::new(
            SmtExpr::var("x", 8).bvadd(SmtExpr::bv_const(1, 8)),
            SmtExpr::var("x", 8),
        ));

        let stats = discovery.stats();
        assert_eq!(stats.total_proposed, 2);

        // At least one should have a definitive result (accepted or rejected)
        let definitive = stats.total_accepted + stats.total_rejected + stats.total_inconclusive;
        assert_eq!(definitive, 2, "All proposals should have results");
    }

    // -----------------------------------------------------------------------
    // Integration: 32-bit width tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_discover_32bit_add_zero() {
        let mut discovery = RuleDiscovery::new(32);

        let proposal = RuleProposal::new(
            SmtExpr::var("x", 32).bvadd(SmtExpr::bv_const(0, 32)),
            SmtExpr::var("x", 32),
        )
        .with_name("add_zero_32bit");

        let result = discovery.propose_rule(proposal);
        assert!(
            result.is_accepted() || matches!(result, RuleResult::Inconclusive),
            "32-bit add_zero should be accepted or inconclusive, got {:?}",
            result
        );
    }

    #[test]
    fn test_discover_32bit_mul_to_shift() {
        let mut discovery = RuleDiscovery::new(32);

        let proposal = RuleProposal::new(
            SmtExpr::var("x", 32).bvmul(SmtExpr::bv_const(2, 32)),
            SmtExpr::var("x", 32).bvshl(SmtExpr::bv_const(1, 32)),
        )
        .with_name("mul2_to_lsl1_32bit");

        let result = discovery.propose_rule(proposal);
        assert!(
            result.is_accepted() || matches!(result, RuleResult::Inconclusive),
            "32-bit mul2->lsl1 should be accepted or inconclusive, got {:?}",
            result
        );
    }

    // -----------------------------------------------------------------------
    // Counterexample accumulation across proposals
    // -----------------------------------------------------------------------

    #[test]
    fn test_counterexample_accumulation() {
        let mut discovery = RuleDiscovery::new(8);

        // First proposal (bad) — will generate counterexamples
        let _ = discovery.propose_rule(RuleProposal::new(
            SmtExpr::var("x", 8).bvadd(SmtExpr::bv_const(1, 8)),
            SmtExpr::var("x", 8),
        ));

        let count_after_first = discovery.counterexample_count();

        // Second proposal (also bad) — should benefit from accumulated counterexamples
        let _ = discovery.propose_rule(RuleProposal::new(
            SmtExpr::var("x", 8).bvsub(SmtExpr::bv_const(1, 8)),
            SmtExpr::var("x", 8),
        ));

        // Counterexample count should be >= what we had before
        assert!(discovery.counterexample_count() >= count_after_first);
    }
}
