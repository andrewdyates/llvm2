// llvm2-verify/synthesis.rs - Offline peephole synthesis via SMT
//
// Author: Andrew Yates <ayates@dropbox.com>
// Copyright 2026 Dropbox, Inc. | License: Apache-2.0
//
// Enumerates AArch64 instruction patterns and uses SMT bitvector
// verification to discover proven-correct peephole optimization rules.
//
// Approach: Alive2-style (PLDI 2021) equivalence checking. For each
// candidate rewrite (pattern -> replacement), encode both sides as
// bitvector SMT expressions and check whether they compute the same
// result for all inputs. If equivalent, the rule is proven correct.
//
// Reference: designs/2026-04-13-superoptimization.md

//! Offline peephole synthesis framework.
//!
//! This module provides solver-driven discovery of peephole optimization
//! rules. Instead of hand-writing rules, [`SynthesisEngine`] enumerates
//! instruction pattern pairs and uses the existing SMT infrastructure to
//! verify semantic equivalence.
//!
//! # Architecture
//!
//! ```text
//! SearchSpace::enumerate()
//!     → Vec<RuleCandidate>
//!         → SynthesisEngine::verify_candidate()
//!             → ProofObligation + verify_by_evaluation
//!                 → ProvenRule (if UNSAT / Valid)
//!                     → ProvenRuleDb
//! ```
//!
//! # Example
//!
//! ```rust
//! use llvm2_verify::synthesis::{SynthesisEngine, SearchSpace, SearchConfig};
//!
//! let config = SearchConfig { max_pattern_len: 1, max_replacement_len: 1, width: 8 };
//! let candidates = SearchSpace::enumerate(&config);
//! let mut engine = SynthesisEngine::new(config.width);
//! for candidate in &candidates {
//!     if let Some(rule) = engine.verify_candidate(candidate) {
//!         println!("Discovered: {}", rule.name);
//!     }
//! }
//! ```

use crate::lowering_proof::{verify_by_evaluation, ProofObligation};
use crate::smt::SmtExpr;
use crate::verify::VerificationResult;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

// ---------------------------------------------------------------------------
// Opcode enumeration
// ---------------------------------------------------------------------------

/// AArch64 opcodes supported in the synthesis search space.
///
/// This is a simplified enumeration of opcodes whose semantics we can
/// encode as bitvector SMT expressions. Each maps to a corresponding
/// SmtExpr builder.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SynthOpcode {
    /// ADD Rd, Rn, Rm/imm -- wrapping addition.
    Add,
    /// SUB Rd, Rn, Rm/imm -- wrapping subtraction.
    Sub,
    /// MUL Rd, Rn, Rm/imm -- wrapping multiplication.
    Mul,
    /// AND Rd, Rn, Rm/imm -- bitwise AND.
    And,
    /// ORR Rd, Rn, Rm/imm -- bitwise OR.
    Orr,
    /// EOR Rd, Rn, Rm/imm -- bitwise XOR.
    Eor,
    /// LSL Rd, Rn, Rm/imm -- logical shift left.
    Lsl,
    /// LSR Rd, Rn, Rm/imm -- logical shift right.
    Lsr,
    /// ASR Rd, Rn, Rm/imm -- arithmetic shift right.
    Asr,
    /// NEG Rd, Rn -- two's complement negation (unary).
    Neg,
    /// MVN Rd, Rn -- bitwise NOT (unary).
    Mvn,
    /// MOV Rd, #imm -- move immediate (identity/constant).
    MovImm,
}

impl SynthOpcode {
    /// All opcodes in the search space.
    pub fn all() -> &'static [SynthOpcode] {
        &[
            SynthOpcode::Add,
            SynthOpcode::Sub,
            SynthOpcode::Mul,
            SynthOpcode::And,
            SynthOpcode::Orr,
            SynthOpcode::Eor,
            SynthOpcode::Lsl,
            SynthOpcode::Lsr,
            SynthOpcode::Asr,
            SynthOpcode::Neg,
            SynthOpcode::Mvn,
            SynthOpcode::MovImm,
        ]
    }

    /// Whether this opcode is unary (takes one register operand).
    pub fn is_unary(self) -> bool {
        matches!(self, SynthOpcode::Neg | SynthOpcode::Mvn)
    }

    /// Whether this opcode is a constant-producing instruction.
    pub fn is_const_only(self) -> bool {
        matches!(self, SynthOpcode::MovImm)
    }

    /// Whether this binary opcode is commutative.
    pub fn is_commutative(self) -> bool {
        matches!(
            self,
            SynthOpcode::Add | SynthOpcode::Mul | SynthOpcode::And | SynthOpcode::Orr | SynthOpcode::Eor
        )
    }

    /// Human-readable name.
    pub fn name(self) -> &'static str {
        match self {
            SynthOpcode::Add => "ADD",
            SynthOpcode::Sub => "SUB",
            SynthOpcode::Mul => "MUL",
            SynthOpcode::And => "AND",
            SynthOpcode::Orr => "ORR",
            SynthOpcode::Eor => "EOR",
            SynthOpcode::Lsl => "LSL",
            SynthOpcode::Lsr => "LSR",
            SynthOpcode::Asr => "ASR",
            SynthOpcode::Neg => "NEG",
            SynthOpcode::Mvn => "MVN",
            SynthOpcode::MovImm => "MOV",
        }
    }
}

// ---------------------------------------------------------------------------
// Operand patterns
// ---------------------------------------------------------------------------

/// Describes what value fills an operand slot in a synthesis pattern.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum OperandPattern {
    /// The primary input register (symbolic variable "x").
    InputReg,
    /// Same as another operand in this instruction (for idempotent patterns).
    /// The index refers to the operand position (0 = first operand).
    SameAsInput(usize),
    /// A specific immediate constant value.
    Immediate(i64),
    /// Any immediate -- used during enumeration to try a set of interesting
    /// constants (0, 1, 2, -1, etc.).
    AnyImmediate,
}

// ---------------------------------------------------------------------------
// Instruction pattern
// ---------------------------------------------------------------------------

/// A single instruction in a synthesis pattern.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct InstrPattern {
    pub opcode: SynthOpcode,
    pub operands: Vec<OperandPattern>,
}

impl InstrPattern {
    /// Format this instruction for display.
    pub fn display(&self) -> String {
        let ops: Vec<String> = self
            .operands
            .iter()
            .map(|op| match op {
                OperandPattern::InputReg => "x".to_string(),
                OperandPattern::SameAsInput(i) => format!("op{}", i),
                OperandPattern::Immediate(v) => format!("#{}", v),
                OperandPattern::AnyImmediate => "#?".to_string(),
            })
            .collect();
        format!("{} {}", self.opcode.name(), ops.join(", "))
    }
}

// ---------------------------------------------------------------------------
// Rule candidate
// ---------------------------------------------------------------------------

/// A candidate rewrite rule: pattern -> replacement.
///
/// The pattern and replacement are sequences of instructions. For
/// single-instruction peephole rules, both are length 1.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct RuleCandidate {
    /// The instruction sequence to match.
    pub pattern: Vec<InstrPattern>,
    /// The replacement instruction sequence.
    pub replacement: Vec<InstrPattern>,
}

impl RuleCandidate {
    /// Human-readable description of this rule.
    pub fn display(&self) -> String {
        let pat: Vec<String> = self.pattern.iter().map(|i| i.display()).collect();
        let rep: Vec<String> = self.replacement.iter().map(|i| i.display()).collect();
        format!("{} => {}", pat.join(" ; "), rep.join(" ; "))
    }
}

// ---------------------------------------------------------------------------
// Proven rule
// ---------------------------------------------------------------------------

/// A peephole rule that has been verified correct via SMT.
#[derive(Debug, Clone)]
pub struct ProvenRule {
    /// Human-readable rule name.
    pub name: String,
    /// The verified rewrite rule.
    pub candidate: RuleCandidate,
    /// Hash of the proof obligation (for deduplication and audit trail).
    pub proof_hash: u64,
    /// Cost delta: positive means the replacement is cheaper (fewer
    /// instructions or cheaper instructions).
    pub cost_delta: i32,
    /// Bitvector width at which the rule was verified.
    pub verified_width: u32,
}

// ---------------------------------------------------------------------------
// Proven rule database
// ---------------------------------------------------------------------------

/// Collection of proven peephole rules.
#[derive(Debug, Clone, Default)]
pub struct ProvenRuleDb {
    pub rules: Vec<ProvenRule>,
}

impl ProvenRuleDb {
    pub fn new() -> Self {
        Self { rules: Vec::new() }
    }

    pub fn add(&mut self, rule: ProvenRule) {
        self.rules.push(rule);
    }

    pub fn len(&self) -> usize {
        self.rules.len()
    }

    pub fn is_empty(&self) -> bool {
        self.rules.is_empty()
    }

    /// Return rules that strictly reduce cost (cost_delta > 0).
    pub fn profitable_rules(&self) -> Vec<&ProvenRule> {
        self.rules.iter().filter(|r| r.cost_delta > 0).collect()
    }
}

// ---------------------------------------------------------------------------
// Search configuration
// ---------------------------------------------------------------------------

/// Configuration for the search space enumeration.
#[derive(Debug, Clone)]
pub struct SearchConfig {
    /// Maximum number of instructions in a pattern (1 or 2).
    pub max_pattern_len: usize,
    /// Maximum number of instructions in a replacement (1 or 2).
    pub max_replacement_len: usize,
    /// Bitvector width for verification (8 for exhaustive, 32/64 for sampling).
    pub width: u32,
}

impl Default for SearchConfig {
    fn default() -> Self {
        Self {
            max_pattern_len: 1,
            max_replacement_len: 1,
            width: 8,
        }
    }
}

// ---------------------------------------------------------------------------
// Search space enumeration
// ---------------------------------------------------------------------------

/// Enumerates candidate peephole rewrites from the AArch64 instruction space.
pub struct SearchSpace;

impl SearchSpace {
    /// Interesting immediate values to try during enumeration.
    fn interesting_immediates() -> &'static [i64] {
        &[0, 1, 2, -1]
    }

    /// Enumerate all single-instruction patterns operating on input "x".
    fn single_instruction_patterns(width: u32) -> Vec<InstrPattern> {
        let mut patterns = Vec::new();

        for &opcode in SynthOpcode::all() {
            if opcode.is_const_only() {
                // MOV #imm -- produces a constant, not interesting as pattern
                // (a pattern must consume the input)
                continue;
            }

            if opcode.is_unary() {
                // NEG x, MVN x
                patterns.push(InstrPattern {
                    opcode,
                    operands: vec![OperandPattern::InputReg],
                });
            } else {
                // Binary: op x, #imm  (register-immediate)
                for &imm in Self::interesting_immediates() {
                    // Skip shift amounts >= width (undefined behavior)
                    if matches!(opcode, SynthOpcode::Lsl | SynthOpcode::Lsr | SynthOpcode::Asr)
                        && (imm < 0 || imm as u32 >= width)
                    {
                        continue;
                    }
                    patterns.push(InstrPattern {
                        opcode,
                        operands: vec![
                            OperandPattern::InputReg,
                            OperandPattern::Immediate(imm),
                        ],
                    });
                }

                // Binary: op x, x  (same register, for idempotent patterns)
                patterns.push(InstrPattern {
                    opcode,
                    operands: vec![
                        OperandPattern::InputReg,
                        OperandPattern::SameAsInput(0),
                    ],
                });
            }
        }

        // Also include the identity: just "x" (no-op / elide instruction).
        // Represented as a zero-length replacement.
        // We handle this specially in enumerate().

        patterns
    }

    /// Enumerate candidate rewrites.
    ///
    /// For max_pattern_len=1, max_replacement_len=1:
    /// Generates all pairs (pattern, replacement) of single-instruction
    /// patterns. Also generates (pattern, identity) where the replacement
    /// is "just return x" (instruction elision).
    ///
    /// Pruning rules:
    /// 1. Skip identity rewrites (pattern == replacement)
    /// 2. Skip commutative duplicates (ADD x,#1 -> ADD x,#1 same as reverse)
    /// 3. Skip patterns that don't consume the input
    pub fn enumerate(config: &SearchConfig) -> Vec<RuleCandidate> {
        let mut candidates = Vec::new();
        let patterns = Self::single_instruction_patterns(config.width);

        if config.max_pattern_len == 1 && config.max_replacement_len == 1 {
            // Pattern: single instruction -> identity (elide)
            // The "replacement" is the input itself (no instructions).
            // We represent this as an empty replacement vec.
            for pat in &patterns {
                candidates.push(RuleCandidate {
                    pattern: vec![pat.clone()],
                    replacement: vec![],
                });
            }

            // Pattern: single instruction -> different single instruction
            for (i, pat) in patterns.iter().enumerate() {
                for (j, rep) in patterns.iter().enumerate() {
                    if i == j {
                        continue; // Skip identity rewrites
                    }

                    // Skip if same opcode and same operands (already pruned by i==j
                    // but guard against structural equality too)
                    if pat == rep {
                        continue;
                    }

                    candidates.push(RuleCandidate {
                        pattern: vec![pat.clone()],
                        replacement: vec![rep.clone()],
                    });
                }
            }
        }

        candidates
    }
}

// ---------------------------------------------------------------------------
// SMT encoding of instruction patterns
// ---------------------------------------------------------------------------

/// Encode a single instruction pattern as an SmtExpr.
///
/// The input variable is "x" with the given bit width.
fn encode_instr_pattern(instr: &InstrPattern, input: &SmtExpr, width: u32) -> SmtExpr {
    let resolve_operand = |op: &OperandPattern| -> SmtExpr {
        match op {
            OperandPattern::InputReg => input.clone(),
            OperandPattern::SameAsInput(0) => input.clone(),
            OperandPattern::SameAsInput(_) => input.clone(), // Only one input for now
            OperandPattern::Immediate(v) => {
                // Convert i64 to u64 with proper masking
                SmtExpr::bv_const(*v as u64, width)
            }
            OperandPattern::AnyImmediate => {
                // Should not appear in concrete patterns
                panic!("AnyImmediate should be resolved before encoding")
            }
        }
    };

    match instr.opcode {
        SynthOpcode::Add => {
            let lhs = resolve_operand(&instr.operands[0]);
            let rhs = resolve_operand(&instr.operands[1]);
            lhs.bvadd(rhs)
        }
        SynthOpcode::Sub => {
            let lhs = resolve_operand(&instr.operands[0]);
            let rhs = resolve_operand(&instr.operands[1]);
            lhs.bvsub(rhs)
        }
        SynthOpcode::Mul => {
            let lhs = resolve_operand(&instr.operands[0]);
            let rhs = resolve_operand(&instr.operands[1]);
            lhs.bvmul(rhs)
        }
        SynthOpcode::And => {
            let lhs = resolve_operand(&instr.operands[0]);
            let rhs = resolve_operand(&instr.operands[1]);
            lhs.bvand(rhs)
        }
        SynthOpcode::Orr => {
            let lhs = resolve_operand(&instr.operands[0]);
            let rhs = resolve_operand(&instr.operands[1]);
            lhs.bvor(rhs)
        }
        SynthOpcode::Eor => {
            let lhs = resolve_operand(&instr.operands[0]);
            let rhs = resolve_operand(&instr.operands[1]);
            lhs.bvxor(rhs)
        }
        SynthOpcode::Lsl => {
            let lhs = resolve_operand(&instr.operands[0]);
            let rhs = resolve_operand(&instr.operands[1]);
            lhs.bvshl(rhs)
        }
        SynthOpcode::Lsr => {
            let lhs = resolve_operand(&instr.operands[0]);
            let rhs = resolve_operand(&instr.operands[1]);
            lhs.bvlshr(rhs)
        }
        SynthOpcode::Asr => {
            let lhs = resolve_operand(&instr.operands[0]);
            let rhs = resolve_operand(&instr.operands[1]);
            lhs.bvashr(rhs)
        }
        SynthOpcode::Neg => {
            let operand = resolve_operand(&instr.operands[0]);
            operand.bvneg()
        }
        SynthOpcode::Mvn => {
            // MVN = bitwise NOT = XOR with all-ones
            let operand = resolve_operand(&instr.operands[0]);
            operand.bvxor(SmtExpr::bv_const(u64::MAX, width))
        }
        SynthOpcode::MovImm => {
            // MOV #imm -- just the immediate value
            if let Some(OperandPattern::Immediate(v)) = instr.operands.first() {
                SmtExpr::bv_const(*v as u64, width)
            } else {
                // MOV x -- identity
                input.clone()
            }
        }
    }
}

/// Encode a sequence of instruction patterns as an SmtExpr.
///
/// For a single instruction, the result is that instruction applied to input.
/// For a sequence, each instruction feeds its result into the next.
/// For an empty sequence, the result is the identity (input itself).
fn encode_pattern_sequence(instrs: &[InstrPattern], input: &SmtExpr, width: u32) -> SmtExpr {
    if instrs.is_empty() {
        return input.clone(); // Identity / elision
    }

    let mut current = input.clone();
    for instr in instrs {
        current = encode_instr_pattern(instr, &current, width);
    }
    current
}

// ---------------------------------------------------------------------------
// Synthesis engine
// ---------------------------------------------------------------------------

/// Verifies candidate rewrites using SMT and collects proven rules.
pub struct SynthesisEngine {
    width: u32,
    /// Statistics: total candidates checked.
    pub candidates_checked: usize,
    /// Statistics: candidates verified as correct.
    pub candidates_proven: usize,
    /// Statistics: candidates disproven (counterexample found).
    pub candidates_disproven: usize,
}

impl SynthesisEngine {
    pub fn new(width: u32) -> Self {
        Self {
            width,
            candidates_checked: 0,
            candidates_proven: 0,
            candidates_disproven: 0,
        }
    }

    /// Verify a single candidate rewrite.
    ///
    /// Returns `Some(ProvenRule)` if the candidate is semantically
    /// equivalent (proven correct). Returns `None` if a counterexample
    /// is found or verification is inconclusive.
    pub fn verify_candidate(&mut self, candidate: &RuleCandidate) -> Option<ProvenRule> {
        self.candidates_checked += 1;

        let x = SmtExpr::var("x", self.width);

        let pattern_expr = encode_pattern_sequence(&candidate.pattern, &x, self.width);
        let replacement_expr = encode_pattern_sequence(&candidate.replacement, &x, self.width);

        let obligation = ProofObligation {
            name: format!("Synthesis: {}", candidate.display()),
            tmir_expr: pattern_expr,
            aarch64_expr: replacement_expr,
            inputs: vec![("x".to_string(), self.width)],
            preconditions: vec![],
        };

        let result = verify_by_evaluation(&obligation);

        match result {
            VerificationResult::Valid => {
                self.candidates_proven += 1;

                // Compute proof hash for audit trail
                let mut hasher = DefaultHasher::new();
                candidate.hash(&mut hasher);
                self.width.hash(&mut hasher);
                let proof_hash = hasher.finish();

                // Compute cost delta: positive means replacement is cheaper
                let pattern_cost = Self::estimate_cost(&candidate.pattern);
                let replacement_cost = Self::estimate_cost(&candidate.replacement);
                let cost_delta = pattern_cost - replacement_cost;

                Some(ProvenRule {
                    name: candidate.display(),
                    candidate: candidate.clone(),
                    proof_hash,
                    cost_delta,
                    verified_width: self.width,
                })
            }
            VerificationResult::Invalid { .. } => {
                self.candidates_disproven += 1;
                None
            }
            VerificationResult::Unknown { .. } => None,
        }
    }

    /// Run synthesis over all candidates from a search space configuration.
    ///
    /// Returns a ProvenRuleDb containing all discovered rules.
    pub fn run(&mut self, config: &SearchConfig) -> ProvenRuleDb {
        let candidates = SearchSpace::enumerate(config);
        let mut db = ProvenRuleDb::new();

        for candidate in &candidates {
            if let Some(rule) = self.verify_candidate(candidate) {
                db.add(rule);
            }
        }

        db
    }

    /// Estimate the execution cost of an instruction sequence.
    ///
    /// Simple model: each instruction costs 1, identity (empty) costs 0.
    /// Shift instructions cost 1 (same as ALU on AArch64).
    /// MUL costs 3 (higher latency on most microarchitectures).
    fn estimate_cost(instrs: &[InstrPattern]) -> i32 {
        instrs
            .iter()
            .map(|instr| match instr.opcode {
                SynthOpcode::Mul => 3,
                _ => 1,
            })
            .sum()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------------
    // Encoding tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_encode_add_imm() {
        let x = SmtExpr::var("x", 8);
        let instr = InstrPattern {
            opcode: SynthOpcode::Add,
            operands: vec![OperandPattern::InputReg, OperandPattern::Immediate(0)],
        };
        let expr = encode_instr_pattern(&instr, &x, 8);
        // ADD x, #0 should equal x for all inputs
        let mut env = std::collections::HashMap::new();
        env.insert("x".to_string(), 42u64);
        assert_eq!(expr.eval(&env).as_u64(), 42);
    }

    #[test]
    fn test_encode_neg() {
        let x = SmtExpr::var("x", 8);
        let instr = InstrPattern {
            opcode: SynthOpcode::Neg,
            operands: vec![OperandPattern::InputReg],
        };
        let expr = encode_instr_pattern(&instr, &x, 8);
        let mut env = std::collections::HashMap::new();
        env.insert("x".to_string(), 1u64);
        // NEG(1) in 8-bit = 255
        assert_eq!(expr.eval(&env).as_u64(), 255);
    }

    #[test]
    fn test_encode_mvn() {
        let x = SmtExpr::var("x", 8);
        let instr = InstrPattern {
            opcode: SynthOpcode::Mvn,
            operands: vec![OperandPattern::InputReg],
        };
        let expr = encode_instr_pattern(&instr, &x, 8);
        let mut env = std::collections::HashMap::new();
        env.insert("x".to_string(), 0u64);
        // MVN(0) in 8-bit = 0xFF
        assert_eq!(expr.eval(&env).as_u64(), 0xFF);
    }

    #[test]
    fn test_encode_identity_sequence() {
        let x = SmtExpr::var("x", 8);
        let expr = encode_pattern_sequence(&[], &x, 8);
        let mut env = std::collections::HashMap::new();
        env.insert("x".to_string(), 99u64);
        assert_eq!(expr.eval(&env).as_u64(), 99);
    }

    #[test]
    fn test_encode_mul_by_2() {
        let x = SmtExpr::var("x", 8);
        let instr = InstrPattern {
            opcode: SynthOpcode::Mul,
            operands: vec![OperandPattern::InputReg, OperandPattern::Immediate(2)],
        };
        let expr = encode_instr_pattern(&instr, &x, 8);
        let mut env = std::collections::HashMap::new();
        env.insert("x".to_string(), 5u64);
        assert_eq!(expr.eval(&env).as_u64(), 10);
    }

    #[test]
    fn test_encode_lsl_by_1() {
        let x = SmtExpr::var("x", 8);
        let instr = InstrPattern {
            opcode: SynthOpcode::Lsl,
            operands: vec![OperandPattern::InputReg, OperandPattern::Immediate(1)],
        };
        let expr = encode_instr_pattern(&instr, &x, 8);
        let mut env = std::collections::HashMap::new();
        env.insert("x".to_string(), 5u64);
        assert_eq!(expr.eval(&env).as_u64(), 10);
    }

    // -----------------------------------------------------------------------
    // Search space tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_search_space_nonempty() {
        let config = SearchConfig {
            max_pattern_len: 1,
            max_replacement_len: 1,
            width: 8,
        };
        let candidates = SearchSpace::enumerate(&config);
        assert!(
            !candidates.is_empty(),
            "search space should produce candidates"
        );
        // Rough bound: we have ~40 patterns, so ~40 identity + 40*39 pairs
        assert!(
            candidates.len() > 100,
            "expected many candidates, got {}",
            candidates.len()
        );
    }

    #[test]
    fn test_search_space_no_self_rewrite() {
        let config = SearchConfig::default();
        let candidates = SearchSpace::enumerate(&config);
        for c in &candidates {
            if c.pattern.len() == 1 && c.replacement.len() == 1 {
                assert_ne!(
                    c.pattern[0], c.replacement[0],
                    "should not have identity rewrite: {}",
                    c.display()
                );
            }
        }
    }

    // -----------------------------------------------------------------------
    // Known rule discovery tests
    // -----------------------------------------------------------------------

    /// Verify that synthesis discovers: ADD x, #0 => identity
    #[test]
    fn test_discover_add_zero_identity() {
        let candidate = RuleCandidate {
            pattern: vec![InstrPattern {
                opcode: SynthOpcode::Add,
                operands: vec![OperandPattern::InputReg, OperandPattern::Immediate(0)],
            }],
            replacement: vec![],
        };

        let mut engine = SynthesisEngine::new(8);
        let rule = engine.verify_candidate(&candidate);
        assert!(rule.is_some(), "ADD x, #0 => identity should be proven");
        let rule = rule.unwrap();
        assert!(rule.cost_delta > 0, "eliding ADD should be profitable");
    }

    /// Verify that synthesis discovers: SUB x, #0 => identity
    #[test]
    fn test_discover_sub_zero_identity() {
        let candidate = RuleCandidate {
            pattern: vec![InstrPattern {
                opcode: SynthOpcode::Sub,
                operands: vec![OperandPattern::InputReg, OperandPattern::Immediate(0)],
            }],
            replacement: vec![],
        };

        let mut engine = SynthesisEngine::new(8);
        let rule = engine.verify_candidate(&candidate);
        assert!(rule.is_some(), "SUB x, #0 => identity should be proven");
    }

    /// Verify that synthesis discovers: MUL x, #1 => identity
    #[test]
    fn test_discover_mul_one_identity() {
        let candidate = RuleCandidate {
            pattern: vec![InstrPattern {
                opcode: SynthOpcode::Mul,
                operands: vec![OperandPattern::InputReg, OperandPattern::Immediate(1)],
            }],
            replacement: vec![],
        };

        let mut engine = SynthesisEngine::new(8);
        let rule = engine.verify_candidate(&candidate);
        assert!(rule.is_some(), "MUL x, #1 => identity should be proven");
        let rule = rule.unwrap();
        // MUL costs 3, identity costs 0, delta = 3
        assert_eq!(rule.cost_delta, 3, "eliding MUL should save cost 3");
    }

    /// Verify that synthesis discovers: MUL x, #2 => LSL x, #1
    #[test]
    fn test_discover_mul2_to_lsl1() {
        let candidate = RuleCandidate {
            pattern: vec![InstrPattern {
                opcode: SynthOpcode::Mul,
                operands: vec![OperandPattern::InputReg, OperandPattern::Immediate(2)],
            }],
            replacement: vec![InstrPattern {
                opcode: SynthOpcode::Lsl,
                operands: vec![OperandPattern::InputReg, OperandPattern::Immediate(1)],
            }],
        };

        let mut engine = SynthesisEngine::new(8);
        let rule = engine.verify_candidate(&candidate);
        assert!(
            rule.is_some(),
            "MUL x, #2 => LSL x, #1 should be proven"
        );
        let rule = rule.unwrap();
        // MUL costs 3, LSL costs 1, delta = 2
        assert_eq!(rule.cost_delta, 2, "MUL->LSL should save cost 2");
    }

    /// Verify that synthesis discovers: ORR x, x => identity
    #[test]
    fn test_discover_orr_self_identity() {
        let candidate = RuleCandidate {
            pattern: vec![InstrPattern {
                opcode: SynthOpcode::Orr,
                operands: vec![OperandPattern::InputReg, OperandPattern::SameAsInput(0)],
            }],
            replacement: vec![],
        };

        let mut engine = SynthesisEngine::new(8);
        let rule = engine.verify_candidate(&candidate);
        assert!(rule.is_some(), "ORR x, x => identity should be proven");
    }

    /// Verify that synthesis discovers: AND x, x => identity
    #[test]
    fn test_discover_and_self_identity() {
        let candidate = RuleCandidate {
            pattern: vec![InstrPattern {
                opcode: SynthOpcode::And,
                operands: vec![OperandPattern::InputReg, OperandPattern::SameAsInput(0)],
            }],
            replacement: vec![],
        };

        let mut engine = SynthesisEngine::new(8);
        let rule = engine.verify_candidate(&candidate);
        assert!(rule.is_some(), "AND x, x => identity should be proven");
    }

    /// Verify that synthesis discovers: EOR x, #0 => identity
    #[test]
    fn test_discover_eor_zero_identity() {
        let candidate = RuleCandidate {
            pattern: vec![InstrPattern {
                opcode: SynthOpcode::Eor,
                operands: vec![OperandPattern::InputReg, OperandPattern::Immediate(0)],
            }],
            replacement: vec![],
        };

        let mut engine = SynthesisEngine::new(8);
        let rule = engine.verify_candidate(&candidate);
        assert!(rule.is_some(), "EOR x, #0 => identity should be proven");
    }

    /// Verify that synthesis discovers: LSL x, #0 => identity
    #[test]
    fn test_discover_lsl_zero_identity() {
        let candidate = RuleCandidate {
            pattern: vec![InstrPattern {
                opcode: SynthOpcode::Lsl,
                operands: vec![OperandPattern::InputReg, OperandPattern::Immediate(0)],
            }],
            replacement: vec![],
        };

        let mut engine = SynthesisEngine::new(8);
        let rule = engine.verify_candidate(&candidate);
        assert!(rule.is_some(), "LSL x, #0 => identity should be proven");
    }

    /// Verify that synthesis discovers: LSR x, #0 => identity
    #[test]
    fn test_discover_lsr_zero_identity() {
        let candidate = RuleCandidate {
            pattern: vec![InstrPattern {
                opcode: SynthOpcode::Lsr,
                operands: vec![OperandPattern::InputReg, OperandPattern::Immediate(0)],
            }],
            replacement: vec![],
        };

        let mut engine = SynthesisEngine::new(8);
        let rule = engine.verify_candidate(&candidate);
        assert!(rule.is_some(), "LSR x, #0 => identity should be proven");
    }

    /// Verify that synthesis discovers: ASR x, #0 => identity
    #[test]
    fn test_discover_asr_zero_identity() {
        let candidate = RuleCandidate {
            pattern: vec![InstrPattern {
                opcode: SynthOpcode::Asr,
                operands: vec![OperandPattern::InputReg, OperandPattern::Immediate(0)],
            }],
            replacement: vec![],
        };

        let mut engine = SynthesisEngine::new(8);
        let rule = engine.verify_candidate(&candidate);
        assert!(rule.is_some(), "ASR x, #0 => identity should be proven");
    }

    /// Verify that synthesis discovers: SUB x, x => (produces 0)
    /// Specifically: SUB x, x is equivalent to AND x, #0
    /// Also: EOR x, x => produces 0
    #[test]
    fn test_discover_sub_self_zero() {
        // SUB x, x should be equivalent to producing 0
        // We test this by checking SUB x, x == EOR x, x (both produce 0)
        let candidate = RuleCandidate {
            pattern: vec![InstrPattern {
                opcode: SynthOpcode::Sub,
                operands: vec![OperandPattern::InputReg, OperandPattern::SameAsInput(0)],
            }],
            replacement: vec![InstrPattern {
                opcode: SynthOpcode::Eor,
                operands: vec![OperandPattern::InputReg, OperandPattern::SameAsInput(0)],
            }],
        };

        let mut engine = SynthesisEngine::new(8);
        let rule = engine.verify_candidate(&candidate);
        assert!(
            rule.is_some(),
            "SUB x, x => EOR x, x (both produce 0) should be proven"
        );
    }

    // -----------------------------------------------------------------------
    // Negative tests: rules that should NOT be proven
    // -----------------------------------------------------------------------

    /// ADD x, #1 => identity should NOT be proven (not equivalent)
    #[test]
    fn test_reject_add_one_identity() {
        let candidate = RuleCandidate {
            pattern: vec![InstrPattern {
                opcode: SynthOpcode::Add,
                operands: vec![OperandPattern::InputReg, OperandPattern::Immediate(1)],
            }],
            replacement: vec![],
        };

        let mut engine = SynthesisEngine::new(8);
        let rule = engine.verify_candidate(&candidate);
        assert!(
            rule.is_none(),
            "ADD x, #1 => identity should be rejected"
        );
    }

    /// MUL x, #2 => identity should NOT be proven
    #[test]
    fn test_reject_mul2_identity() {
        let candidate = RuleCandidate {
            pattern: vec![InstrPattern {
                opcode: SynthOpcode::Mul,
                operands: vec![OperandPattern::InputReg, OperandPattern::Immediate(2)],
            }],
            replacement: vec![],
        };

        let mut engine = SynthesisEngine::new(8);
        let rule = engine.verify_candidate(&candidate);
        assert!(rule.is_none(), "MUL x, #2 => identity should be rejected");
    }

    /// NEG x => identity should NOT be proven (unless x is always 0)
    #[test]
    fn test_reject_neg_identity() {
        let candidate = RuleCandidate {
            pattern: vec![InstrPattern {
                opcode: SynthOpcode::Neg,
                operands: vec![OperandPattern::InputReg],
            }],
            replacement: vec![],
        };

        let mut engine = SynthesisEngine::new(8);
        let rule = engine.verify_candidate(&candidate);
        assert!(rule.is_none(), "NEG x => identity should be rejected");
    }

    // -----------------------------------------------------------------------
    // Full synthesis run test
    // -----------------------------------------------------------------------

    /// Run synthesis over the full single-instruction search space
    /// and verify that known rules are discovered.
    #[test]
    fn test_full_synthesis_run() {
        let config = SearchConfig {
            max_pattern_len: 1,
            max_replacement_len: 1,
            width: 8,
        };

        let mut engine = SynthesisEngine::new(config.width);
        let db = engine.run(&config);

        // We should discover many rules
        assert!(
            !db.is_empty(),
            "synthesis should discover at least some rules"
        );

        // Check that profitable rules exist
        let profitable = db.profitable_rules();
        assert!(
            !profitable.is_empty(),
            "should discover profitable optimizations"
        );

        // Print summary for debugging
        eprintln!(
            "Synthesis stats: {} checked, {} proven, {} disproven",
            engine.candidates_checked, engine.candidates_proven, engine.candidates_disproven
        );
        eprintln!("Proven rules: {}", db.len());
        eprintln!("Profitable rules: {}", profitable.len());

        for rule in &db.rules {
            eprintln!(
                "  [cost_delta={}] {}",
                rule.cost_delta, rule.name
            );
        }

        // Verify specific known rules were discovered
        let rule_names: Vec<&str> = db.rules.iter().map(|r| r.name.as_str()).collect();

        // ADD x, #0 => identity
        assert!(
            rule_names.iter().any(|n| n.contains("ADD") && n.contains("#0") && !n.contains(";")),
            "should discover ADD x, #0 => identity. Found rules: {:?}",
            rule_names
        );

        // SUB x, #0 => identity
        assert!(
            rule_names.iter().any(|n| n.contains("SUB") && n.contains("#0") && !n.contains(";")),
            "should discover SUB x, #0 => identity"
        );

        // MUL x, #1 => identity
        assert!(
            rule_names.iter().any(|n| n.contains("MUL") && n.contains("#1")),
            "should discover MUL x, #1 => identity"
        );
    }

    // -----------------------------------------------------------------------
    // ProvenRuleDb tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_proven_rule_db() {
        let mut db = ProvenRuleDb::new();
        assert!(db.is_empty());
        assert_eq!(db.len(), 0);
        assert!(db.profitable_rules().is_empty());

        db.add(ProvenRule {
            name: "test rule".to_string(),
            candidate: RuleCandidate {
                pattern: vec![],
                replacement: vec![],
            },
            proof_hash: 12345,
            cost_delta: 2,
            verified_width: 8,
        });

        assert!(!db.is_empty());
        assert_eq!(db.len(), 1);
        assert_eq!(db.profitable_rules().len(), 1);
    }

    // -----------------------------------------------------------------------
    // Wider-width verification tests (sampling-based)
    // -----------------------------------------------------------------------

    /// Verify ADD x, #0 => identity at 32-bit width (random sampling).
    #[test]
    fn test_add_zero_identity_32bit() {
        let candidate = RuleCandidate {
            pattern: vec![InstrPattern {
                opcode: SynthOpcode::Add,
                operands: vec![OperandPattern::InputReg, OperandPattern::Immediate(0)],
            }],
            replacement: vec![],
        };

        let mut engine = SynthesisEngine::new(32);
        let rule = engine.verify_candidate(&candidate);
        assert!(
            rule.is_some(),
            "ADD x, #0 => identity should be proven at 32-bit"
        );
    }

    /// Verify MUL x, #2 => LSL x, #1 at 32-bit width (random sampling).
    #[test]
    fn test_mul2_to_lsl1_32bit() {
        let candidate = RuleCandidate {
            pattern: vec![InstrPattern {
                opcode: SynthOpcode::Mul,
                operands: vec![OperandPattern::InputReg, OperandPattern::Immediate(2)],
            }],
            replacement: vec![InstrPattern {
                opcode: SynthOpcode::Lsl,
                operands: vec![OperandPattern::InputReg, OperandPattern::Immediate(1)],
            }],
        };

        let mut engine = SynthesisEngine::new(32);
        let rule = engine.verify_candidate(&candidate);
        assert!(
            rule.is_some(),
            "MUL x, #2 => LSL x, #1 should be proven at 32-bit"
        );
    }

    /// Verify MUL x, #2 => LSL x, #1 at 64-bit width (random sampling).
    #[test]
    fn test_mul2_to_lsl1_64bit() {
        let candidate = RuleCandidate {
            pattern: vec![InstrPattern {
                opcode: SynthOpcode::Mul,
                operands: vec![OperandPattern::InputReg, OperandPattern::Immediate(2)],
            }],
            replacement: vec![InstrPattern {
                opcode: SynthOpcode::Lsl,
                operands: vec![OperandPattern::InputReg, OperandPattern::Immediate(1)],
            }],
        };

        let mut engine = SynthesisEngine::new(64);
        let rule = engine.verify_candidate(&candidate);
        assert!(
            rule.is_some(),
            "MUL x, #2 => LSL x, #1 should be proven at 64-bit"
        );
    }

    // -----------------------------------------------------------------------
    // Cost model tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_cost_model() {
        assert_eq!(SynthesisEngine::estimate_cost(&[]), 0);

        let add = InstrPattern {
            opcode: SynthOpcode::Add,
            operands: vec![OperandPattern::InputReg, OperandPattern::Immediate(0)],
        };
        assert_eq!(SynthesisEngine::estimate_cost(&[add.clone()]), 1);

        let mul = InstrPattern {
            opcode: SynthOpcode::Mul,
            operands: vec![OperandPattern::InputReg, OperandPattern::Immediate(1)],
        };
        assert_eq!(SynthesisEngine::estimate_cost(&[mul]), 3);
        assert_eq!(
            SynthesisEngine::estimate_cost(&[add.clone(), add]),
            2
        );
    }

    // -----------------------------------------------------------------------
    // Display tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_instr_pattern_display() {
        let instr = InstrPattern {
            opcode: SynthOpcode::Add,
            operands: vec![OperandPattern::InputReg, OperandPattern::Immediate(0)],
        };
        assert_eq!(instr.display(), "ADD x, #0");

        let instr = InstrPattern {
            opcode: SynthOpcode::Neg,
            operands: vec![OperandPattern::InputReg],
        };
        assert_eq!(instr.display(), "NEG x");
    }

    #[test]
    fn test_rule_candidate_display() {
        let candidate = RuleCandidate {
            pattern: vec![InstrPattern {
                opcode: SynthOpcode::Mul,
                operands: vec![OperandPattern::InputReg, OperandPattern::Immediate(2)],
            }],
            replacement: vec![InstrPattern {
                opcode: SynthOpcode::Lsl,
                operands: vec![OperandPattern::InputReg, OperandPattern::Immediate(1)],
            }],
        };
        assert_eq!(candidate.display(), "MUL x, #2 => LSL x, #1");
    }

    #[test]
    fn test_rule_candidate_display_identity() {
        let candidate = RuleCandidate {
            pattern: vec![InstrPattern {
                opcode: SynthOpcode::Add,
                operands: vec![OperandPattern::InputReg, OperandPattern::Immediate(0)],
            }],
            replacement: vec![],
        };
        assert_eq!(candidate.display(), "ADD x, #0 => ");
    }
}
