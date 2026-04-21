// llvm2-opt - Rewrite engine (fixed-point driver)
//
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Copyright 2026 Dropbox, Inc. | License: Apache-2.0

//! [`RewriteEngine`] drives a set of [`Rule`]s against a [`MachFunction`]
//! until no rule fires (or `max_iterations` is reached).
//!
//! The engine:
//! - visits each instruction in block order,
//! - evaluates every rule and picks the highest-benefit firing rule,
//! - applies the action (`Replace` / `Delete`),
//! - preserves the original instruction's proof annotation and source
//!   location when replacing, matching the behavior of the hand-written
//!   peephole pass,
//! - invalidates the per-block def map after any change, and
//! - iterates to fixed point.

use std::collections::HashMap;
use std::collections::HashSet;

use llvm2_ir::{BlockId, InstId, MachFunction, MachOperand};

use crate::rewrite::matcher::MatchCtx;
use crate::rewrite::rewriter::RewriteAction;
use crate::rewrite::rule::Rule;

/// Per-engine statistics.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct RewriteStats {
    /// Number of rewrites applied (Replace + Delete).
    pub rewrites: u32,
    /// Number of fixed-point iterations executed.
    pub iterations: u32,
    /// Per-rule firing count (indexed by rule registration order).
    pub rule_fires: Vec<u32>,
}

/// Fixed-point rewrite driver.
pub struct RewriteEngine {
    rules: Vec<Rule>,
}

impl RewriteEngine {
    /// Create an empty engine.
    pub fn new() -> Self {
        Self { rules: Vec::new() }
    }

    /// Register a rule. Rules are evaluated in registration order; if
    /// multiple rules match with the same benefit, the earlier-registered
    /// rule wins.
    pub fn register(&mut self, rule: Rule) {
        self.rules.push(rule);
    }

    /// Registered rule count.
    #[inline]
    pub fn num_rules(&self) -> usize {
        self.rules.len()
    }

    /// Run all rules to fixed point.
    pub fn run_to_fixpoint(
        &self,
        func: &mut MachFunction,
        max_iterations: u32,
    ) -> RewriteStats {
        let mut stats = RewriteStats {
            rule_fires: vec![0; self.rules.len()],
            ..Default::default()
        };
        for iter in 0..max_iterations {
            stats.iterations = iter + 1;
            let iter_changes = self.run_once(func, &mut stats);
            if iter_changes == 0 {
                break;
            }
        }
        stats
    }

    /// Single pass over the function. Returns the number of rewrites
    /// applied this pass.
    fn run_once(&self, func: &mut MachFunction, stats: &mut RewriteStats) -> u32 {
        let mut changes: u32 = 0;
        let mut to_delete: HashSet<InstId> = HashSet::new();

        for block_id in func.block_order.clone() {
            // Build def map for this block.
            let mut def_map = build_def_map(func, block_id);
            let inst_ids = func.block(block_id).insts.clone();
            for inst_id in inst_ids {
                if to_delete.contains(&inst_id) {
                    continue;
                }

                // Pick the best-firing rule.
                let best = {
                    let inst = func.inst(inst_id);
                    let ctx = MatchCtx {
                        inst,
                        inst_id,
                        block_id,
                        func,
                        def_map: &def_map,
                    };
                    let mut best: Option<(usize, RewriteAction)> = None;
                    for (idx, rule) in self.rules.iter().enumerate() {
                        if let Some(action) = rule.evaluate(&ctx) {
                            let better = match &best {
                                None => true,
                                Some((b_idx, _)) => {
                                    rule.benefit > self.rules[*b_idx].benefit
                                }
                            };
                            if better {
                                best = Some((idx, action));
                            }
                        }
                    }
                    best
                };

                if let Some((rule_idx, action)) = best {
                    // Snapshot proof + source_loc from the original inst so
                    // we can transfer them onto the replacement.
                    let (orig_proof, orig_loc) = {
                        let inst = func.inst(inst_id);
                        (inst.proof, inst.source_loc)
                    };
                    match action {
                        RewriteAction::None => {}
                        RewriteAction::Replace(mut new_inst) => {
                            if new_inst.proof.is_none() {
                                new_inst.proof = orig_proof;
                            }
                            if new_inst.source_loc.is_none() {
                                new_inst.source_loc = orig_loc;
                            }
                            // Update def map if the new instruction defs a
                            // VReg in operand 0.
                            if let Some(MachOperand::VReg(dst)) = new_inst.operands.first() {
                                def_map.insert(dst.id, inst_id);
                            }
                            *func.inst_mut(inst_id) = new_inst;
                            stats.rewrites += 1;
                            stats.rule_fires[rule_idx] += 1;
                            changes += 1;
                        }
                        RewriteAction::Delete => {
                            to_delete.insert(inst_id);
                            // Remove from def map so downstream rules in this
                            // block can't see the deleted definer.
                            let inst = func.inst(inst_id);
                            if let Some(MachOperand::VReg(dst)) = inst.operands.first() {
                                if def_map.get(&dst.id) == Some(&inst_id) {
                                    def_map.remove(&dst.id);
                                }
                            }
                            stats.rewrites += 1;
                            stats.rule_fires[rule_idx] += 1;
                            changes += 1;
                        }
                    }
                }
            }
        }

        if !to_delete.is_empty() {
            for block_id in func.block_order.clone() {
                let block = func.block_mut(block_id);
                block.insts.retain(|id| !to_delete.contains(id));
            }
        }

        changes
    }
}

impl Default for RewriteEngine {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Build a VReg-id → defining InstId map for the given block.
fn build_def_map(func: &MachFunction, block_id: BlockId) -> HashMap<u32, InstId> {
    let mut map = HashMap::new();
    for &inst_id in &func.block(block_id).insts {
        let inst = func.inst(inst_id);
        if inst.opcode.produces_value() {
            if let Some(MachOperand::VReg(dst)) = inst.operands.first() {
                map.insert(dst.id, inst_id);
            }
        }
    }
    map
}
