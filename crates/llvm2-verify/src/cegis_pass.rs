// llvm2-verify/cegis_pass.rs - CEGIS superopt pass wrapper
//
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Copyright 2026 Dropbox, Inc. | License: Apache-2.0
//
// Wraps the `CegisLoop` primitive as a `MachinePass` so callers can request
// CEGIS-based superoptimization from the optimization pipeline with a
// per-function wall-clock budget. Results (whether the candidate rewrite was
// proven equivalent or rejected) are keyed into the shared compilation cache
// under a per-function hash derived from (instructions + target triple + cpu
// + features). Repeat compilations reuse cached outcomes so that expensive
// CEGIS searches are paid for only once per input tuple.
//
// tla2 supremacy blocker #8 (part of epic #390, issue #395).

//! CEGIS superoptimization pass wrapper.
//!
//! This module exposes the existing [`crate::CegisLoop`] verification loop as
//! a [`llvm2_opt::MachinePass`] implementation. It is feature-gated off by
//! default via [`CegisSuperoptConfig::budget_sec = 0`] and only activates
//! when the pipeline threads through a non-zero budget.
//!
//! # Algorithm per function
//!
//! 1. Compute a deterministic 128-bit hash of the function body + target
//!    triple + CPU + features via [`crate::CegisLoop`]-independent key.
//! 2. Lookup the cache. On a hit: deserialize [`CegisCacheEntry`] and apply
//!    the stored rewrites (skip all solver work).
//! 3. On a miss: walk instructions, identify candidate rewrite sites, and
//!    call [`crate::CegisLoop::verify`] with a per-query timeout derived
//!    from `per_query_ms`. The total wall clock spent in this pass is
//!    capped by `budget_sec`.
//! 4. On successful equivalence proofs, record rewrites in
//!    [`CegisCacheEntry`], serialize via `rmp-serde`, and store under the
//!    function hash.
//!
//! The first payload layer recognizes one hand-seeded single-instruction
//! rewrite: `MulRR x, (Movz #0)` within a block can become `Movz #0` when the
//! replacement is strictly cheaper and CEGIS proves equivalence.

use std::collections::HashMap;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::Arc;
use std::time::{Duration, Instant};

use llvm2_ir::cost_model::{AppleSiliconCostModel, CostModel, CostModelGen};
use llvm2_ir::provenance::PassId as TracePassId;
use llvm2_ir::trace::{CompilationTrace, EventKind, Justification, RuleId};
use llvm2_ir::{AArch64Opcode, MachFunction, MachInst, MachOperand, RegClass, VReg};
use llvm2_opt::{CacheBackend, CacheKey, MachinePass, StableHasher};
use serde::{Deserialize, Serialize};

use crate::cegis::{CegisLoop, CegisResult};
use crate::smt::SmtExpr;
use crate::synthesis::ProvenRuleDb;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Count how many instructions in `func` read `target` as a VReg operand.
///
/// "Uses" are any operand after the first that match the VReg id. The
/// leading operand is treated as the definition (conventional for our
/// `MachInst` layout where `operands[0]` is the destination). This matches
/// the assumption used by the Layer A / Layer B matchers.
fn count_vreg_uses(func: &MachFunction, target: VReg) -> u32 {
    let mut uses: u32 = 0;
    for inst in &func.insts {
        // Skip the destination slot (operand 0) — that is a def, not a use.
        for operand in inst.operands.iter().skip(1) {
            if let Some(v) = operand.as_vreg()
                && v.id == target.id
            {
                uses = uses.saturating_add(1);
            }
        }
    }
    uses
}

/// Low-`width` bitmask for constant immediates fed into the SMT evaluator.
fn mask_u64(width: u32) -> u64 {
    if width >= 64 {
        u64::MAX
    } else {
        (1u64 << width) - 1
    }
}

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for [`CegisSuperoptPass`].
///
/// The pass is effectively disabled when `budget_sec == 0`, which is the
/// default. The CLI flag `--cegis-superopt=<secs>` sets this field to a
/// non-zero value.
#[derive(Clone)]
pub struct CegisSuperoptConfig {
    /// Total per-function wall-clock budget in seconds. `0` disables the pass.
    pub budget_sec: u64,
    /// Per solver query timeout (milliseconds).
    pub per_query_ms: u64,
    /// Target triple (used for cache keying).
    pub target_triple: String,
    /// CPU variant (e.g. "apple-m1"; used for cache keying).
    pub cpu: String,
    /// Target features (used for cache keying; order-invariant).
    pub features: Vec<String>,
    /// Optimization level (0-3, used for cache keying).
    pub opt_level: u8,
    /// Optional cache backend. If `None`, the pass runs but never caches.
    pub cache: Option<Arc<dyn CacheBackend>>,
    /// Optional structured compilation trace collector. When set, the pass
    /// emits a per-function summary event via [`CompilationTrace::emit`] at
    /// the end of [`CegisSuperoptPass::run`]. Level filtering is delegated
    /// to the trace itself (see [`llvm2_ir::trace::TraceLevel`]); passing
    /// a trace at level `None` is effectively a no-op.
    pub trace: Option<Arc<CompilationTrace>>,
}

impl CegisSuperoptConfig {
    /// Build a disabled-default configuration.
    pub fn disabled() -> Self {
        Self {
            budget_sec: 0,
            per_query_ms: 5_000,
            target_triple: String::new(),
            cpu: String::new(),
            features: Vec::new(),
            opt_level: 2,
            cache: None,
            trace: None,
        }
    }

    /// Returns true if this configuration will actually run CEGIS queries.
    #[inline]
    pub fn is_enabled(&self) -> bool {
        self.budget_sec > 0
    }
}

impl Default for CegisSuperoptConfig {
    fn default() -> Self {
        Self::disabled()
    }
}

impl std::fmt::Debug for CegisSuperoptConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CegisSuperoptConfig")
            .field("budget_sec", &self.budget_sec)
            .field("per_query_ms", &self.per_query_ms)
            .field("target_triple", &self.target_triple)
            .field("cpu", &self.cpu)
            .field("features", &self.features)
            .field("opt_level", &self.opt_level)
            .field("cache", &self.cache.as_ref().map(|_| "<CacheBackend>"))
            .field("trace", &self.trace.as_ref().map(|t| t.level()))
            .finish()
    }
}

// ---------------------------------------------------------------------------
// Cache entry
// ---------------------------------------------------------------------------

/// Which matcher / payload layer produced a rewrite.
///
/// Stored alongside each cached [`ProvenRewrite`] so the cache-hit replay
/// path can re-run only the relevant matcher at the recorded `inst_index`
/// and apply the same replacement without re-invoking CEGIS.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RewriteLayer {
    /// Layer A — single-instruction rewrite (e.g. `MulRR x, Movz #0` → `Movz #0`).
    A,
    /// Layer B — two-instruction window fusion (e.g. `Movz+AddRR` → `AddRI`).
    B,
}

/// A single CEGIS rewrite proof recorded in the cache.
///
/// `inst_index` is the flat [`MachFunction::insts`] index that was verified
/// equivalent to a replacement. The replacement body itself is NOT serialized;
/// instead we record which matcher layer produced it, and the cache-hit replay
/// path (`apply_cached_rewrite`) re-runs that matcher at the stored index to
/// reconstruct the replacement. Because the cache key hashes the function body
/// (opcodes + operands + block layout), a cache hit guarantees the same matcher
/// input, so the reconstruction is deterministic.
///
/// This keeps the cache entry small and format-stable while still letting the
/// replay path mutate the function identically to the cold path — which is
/// required for correctness (#491).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProvenRewrite {
    /// Instruction index in `MachFunction::insts`.
    pub inst_index: u32,
    /// Which payload layer produced this rewrite (used on replay).
    pub layer: RewriteLayer,
    /// Proof hash from [`CegisResult::Equivalent::proof_hash`].
    pub proof_hash: u64,
    /// Number of CEGIS iterations used to prove equivalence.
    pub iterations: u32,
}

/// Serializable cache entry stored under the per-function cache key.
///
/// Encoded as MessagePack via `rmp-serde`. Format version is bumped whenever
/// fields are added; older cache entries are discarded by the consumer when
/// version changes (equivalent to a cache miss).
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct CegisCacheEntry {
    /// Format version. Bump when fields change.
    pub version: u32,
    /// All rewrites proven equivalent on this function.
    pub proven_rewrites: Vec<ProvenRewrite>,
    /// Number of candidate sites attempted.
    pub attempted: u64,
    /// Number of sites where equivalence was proven (== proven_rewrites.len()).
    pub verified: u64,
    /// Number of sites rejected (not equivalent / timeout / error).
    pub rejected: u64,
}

impl CegisCacheEntry {
    /// Format version.
    ///
    /// History:
    /// - v1: Layer A only; no replay.
    /// - v2: Layer B added (two-instruction window fusion); still no replay —
    ///   cache hits silently skipped rewrites, causing un-replayed mutation
    ///   and phantom verification counts (#491).
    /// - v3: `ProvenRewrite` gained a `layer` tag and the cache-hit path now
    ///   replays rewrites via `apply_cached_rewrite` (#491). Stats increments
    ///   on replay reflect rewrites actually applied this run, not cached
    ///   counts.
    ///
    /// Older entries are silently rejected as a miss; this is semantics-
    /// preserving because the pass simply re-runs enumeration on the next
    /// invocation.
    pub const VERSION: u32 = 3;

    /// Fresh empty entry with the current version.
    pub fn empty() -> Self {
        Self {
            version: Self::VERSION,
            ..Default::default()
        }
    }

    /// Encode to MessagePack bytes.
    pub fn encode(&self) -> Result<Vec<u8>, rmp_serde::encode::Error> {
        rmp_serde::to_vec(self)
    }

    /// Decode from MessagePack bytes. Returns `None` if the version mismatches
    /// (treated as a cache miss by callers).
    pub fn decode(bytes: &[u8]) -> Option<Self> {
        let entry: Self = rmp_serde::from_slice(bytes).ok()?;
        if entry.version != Self::VERSION {
            return None;
        }
        Some(entry)
    }
}

// ---------------------------------------------------------------------------
// Stats
// ---------------------------------------------------------------------------

/// Runtime statistics for a single [`CegisSuperoptPass`] execution.
///
/// These are the canonical observability counters for the CEGIS pass. Fields
/// are grouped into four categories:
///
/// - **Coverage** (`functions_seen`, cache hit/miss/put counts).
/// - **Per-layer work** (`layer_a_candidates`, `layer_a_committed`,
///   `layer_b_candidates`, `layer_b_committed`). `candidates` and `verified`
///   are roll-ups across all layers and are retained for backwards
///   compatibility with the #486 Layer A acceptance criteria.
/// - **Timing** (`total_wall_ms`, `solver_ms`). Wall time is measured from
///   the first call to [`CegisSuperoptPass::run`] for this function; solver
///   time is the cumulative sum of elapsed time inside solver calls.
/// - **Failure modes** (`rejected`, `budget_exhausted`, `timeouts`,
///   `verifier_errors`, `panics`). Each failure is counted in exactly one
///   bucket: a candidate that timed out bumps `timeouts` + `rejected`; a
///   verifier error bumps `verifier_errors` + `rejected`; a panic bumps
///   `panics` + `rejected`. Plain "not equivalent" / cost-not-profitable
///   rejections bump `rejected` only.
///
/// The struct derives `Serialize`/`Deserialize` so that harnesses can
/// roundtrip stats through JSON for weekly-report ingestion (#486 §10).
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CegisPassStats {
    /// Number of functions the pass ran on (regardless of result).
    pub functions_seen: u64,
    /// Number of cache hits (result was read from the cache).
    pub cache_hits: u64,
    /// Number of cache misses (CEGIS actually ran).
    pub cache_misses: u64,
    /// Number of cache puts after a successful verification pass.
    pub cache_puts: u64,
    /// Number of candidate rewrite sites considered (all layers).
    pub candidates: u64,
    /// Number of candidate sites proven equivalent (all layers).
    pub verified: u64,
    /// Number of candidate sites rejected (not equivalent / timeout / error).
    pub rejected: u64,
    /// Number of times the wall-clock budget was exhausted mid-function.
    pub budget_exhausted: u64,
    /// Number of solver calls made (across all CegisLoop instances).
    pub solver_calls: u64,
    /// Number of verifier panics caught and contained.
    pub panics: u64,
    /// Number of Layer A candidate sites considered (single-instruction
    /// rewrites such as `MulRR x, Movz #0` → `Movz #0`).
    pub layer_a_candidates: u64,
    /// Number of Layer A rewrites actually committed (proven equivalent AND
    /// strictly cost-better, OR replayed from the cache). Strict subset of
    /// `layer_a_candidates` on the cold path; independent on the hot path.
    pub layer_a_committed: u64,
    /// Number of Layer B candidate sites considered (two-instruction window
    /// fusion such as `Movz+AddRR` → `AddRI`).
    pub layer_b_candidates: u64,
    /// Number of Layer B rewrites actually committed.
    pub layer_b_committed: u64,
    /// Total wall-clock time spent in the pass across all invocations this
    /// pass instance has seen, in milliseconds. Measured around the entire
    /// `run_inner` body, so it includes cache lookups, replay, and CEGIS.
    pub total_wall_ms: u64,
    /// Cumulative time spent inside `CegisLoop::verify` (i.e. wall time
    /// elapsed across SMT / concrete-eval work), in milliseconds. Strict
    /// subset of `total_wall_ms` on any single invocation.
    pub solver_ms: u64,
    /// Number of candidate sites whose CEGIS query returned a solver timeout
    /// ([`CegisResult::Timeout`] or [`CegisResult::MaxIterationsReached`]).
    /// Subset of `rejected`.
    pub timeouts: u64,
    /// Number of candidate sites whose CEGIS query returned a verifier error
    /// ([`CegisResult::Error`]). Distinct from `timeouts` and from
    /// "not equivalent" rejections. Subset of `rejected`.
    pub verifier_errors: u64,
}

// ---------------------------------------------------------------------------
// The pass
// ---------------------------------------------------------------------------

/// CEGIS-driven superoptimization pass.
///
/// This is a thin wrapper around [`crate::CegisLoop`] that threads a
/// per-function time budget and a shared compilation cache. On a cache hit it
/// skips all solver work. On a miss it walks the function instructions and
/// invokes `CegisLoop::verify` for each candidate rewrite site (if any),
/// aborting early when the budget is exhausted.
///
/// The first payload layer rewrites `MulRR x, (Movz #0)` to `Movz #0` when the
/// verifier proves equivalence and the replacement is strictly cheaper.
pub struct CegisSuperoptPass {
    config: CegisSuperoptConfig,
    stats: CegisPassStats,
}

impl CegisSuperoptPass {
    /// Create a new pass with the given configuration.
    pub fn new(config: CegisSuperoptConfig) -> Self {
        Self {
            config,
            stats: CegisPassStats::default(),
        }
    }

    /// Return the collected statistics.
    pub fn stats(&self) -> &CegisPassStats {
        &self.stats
    }

    /// Compute the deterministic per-function cache key.
    ///
    /// Hashes:
    /// - function name (framed)
    /// - number of instructions (framed)
    /// - each instruction's opcode Debug repr + operand count
    /// - block layout (entry + block_order)
    /// - opt_level, target_triple, cpu, features
    ///
    /// The resulting key is the same structure used by
    /// [`llvm2_opt::CacheKey`] so callers can share a single on-disk cache
    /// between whole-module and per-function entries.
    pub fn compute_function_key(&self, func: &MachFunction) -> CacheKey {
        let mut h = StableHasher::new();
        h.write_str(&func.name);
        h.write_u64(func.insts.len() as u64);
        for inst in &func.insts {
            // Debug repr of the opcode enum is stable within a compiler
            // version. This matches what the cache key version field gates.
            let op = format!("{:?}", inst.opcode);
            h.write_str(&op);
            h.write_u64(inst.operands.len() as u64);
            for operand in &inst.operands {
                // Using Debug is coarse but deterministic within a compiler
                // build. Fine-grained operand hashing is left for a future
                // key-version bump (see CACHE_KEY_VERSION in llvm2-opt).
                let s = format!("{:?}", operand);
                h.write_str(&s);
            }
        }
        h.write_u64(func.blocks.len() as u64);
        h.write_u64(func.entry.0 as u64);
        h.write_u64(func.block_order.len() as u64);
        for b in &func.block_order {
            h.write_u64(b.0 as u64);
        }
        let module_hash = h.finish128();
        CacheKey::new(
            module_hash,
            self.config.opt_level,
            self.config.target_triple.clone(),
            self.config.cpu.clone(),
            self.config.features.clone(),
        )
    }

    fn match_layer_a_candidate(
        func: &MachFunction,
        inst: &MachInst,
        def_map: &HashMap<u32, llvm2_ir::InstId>,
    ) -> Option<(u32, MachInst)> {
        if inst.opcode != AArch64Opcode::MulRR || inst.operands.len() < 3 {
            return None;
        }

        let dst = inst.operands.first()?.as_vreg()?;
        let src1 = inst.operands.get(1)?.as_vreg()?;
        let src2 = inst.operands.get(2)?.as_vreg()?;
        let def_id = def_map.get(&src2.id)?;
        let def_inst = func.inst(*def_id);

        if def_inst.opcode != AArch64Opcode::Movz || def_inst.operands.len() < 2 {
            return None;
        }
        if def_inst.operands.first()?.as_vreg()? != src2 {
            return None;
        }
        if def_inst.operands.get(1)?.as_imm()? != 0 {
            return None;
        }

        let width = if src1.class == RegClass::Gpr32 {
            32
        } else {
            64
        };
        let mut replacement = MachInst::new(
            AArch64Opcode::Movz,
            vec![MachOperand::VReg(dst), MachOperand::Imm(0)],
        );
        replacement.proof = inst.proof.clone();
        replacement.source_loc = inst.source_loc.clone();

        Some((width, replacement))
    }

    fn enumerate_and_verify_layer_a(
        &mut self,
        func: &mut MachFunction,
        deadline: Instant,
    ) -> (bool, CegisCacheEntry) {
        let mut entry = CegisCacheEntry::empty();
        let mut committed = false;

        if ProvenRuleDb::seed_layer_a().is_empty() {
            return (false, entry);
        }

        let cost_model = AppleSiliconCostModel::new(CostModelGen::M1);
        let mut cegis = CegisLoop::new(1, self.config.per_query_ms);
        let mut exhausted = false;

        'blocks: for block_id in func.block_order.clone() {
            let inst_ids = func.block(block_id).insts.clone();
            let mut def_map = HashMap::new();

            for inst_id in inst_ids {
                if Instant::now() >= deadline {
                    exhausted = true;
                    break 'blocks;
                }

                let inst = func.inst(inst_id).clone();
                if let Some((width, replacement)) =
                    Self::match_layer_a_candidate(func, &inst, &def_map)
                {
                    self.stats.candidates += 1;
                    self.stats.layer_a_candidates += 1;
                    entry.attempted += 1;

                    let src_cost = cost_model.latency(AArch64Opcode::MulRR) as i32;
                    let tgt_cost = cost_model.latency(AArch64Opcode::Movz) as i32;
                    if tgt_cost >= src_cost {
                        self.stats.rejected += 1;
                        entry.rejected += 1;
                    } else {
                        let vars = vec![("x".to_string(), width)];
                        let src = SmtExpr::var("x", width).bvmul(SmtExpr::bv_const(0, width));
                        let tgt = SmtExpr::bv_const(0, width);
                        // Clear counterexamples from prior candidates so a
                        // CX proving a different obligation cannot trigger a
                        // spurious `NotEquivalent` fast-path rejection here
                        // (#493). Stats accumulate across candidates via
                        // `clear_counterexamples` (not `reset`).
                        cegis.clear_counterexamples();
                        cegis.add_edge_case_seeds(&vars);

                        // Wall-clock the solver call so we can populate
                        // `stats.solver_ms`. CegisLoop::verify does concrete
                        // eval + potentially several z4 queries; we attribute
                        // the entire elapsed duration to "solver" since that
                        // is what the #486 acceptance criterion wants
                        // (anything that is not pure enumeration/mutation).
                        let solver_start = Instant::now();
                        let result = catch_unwind(AssertUnwindSafe(|| {
                            cegis.verify(&src, &tgt, &vars)
                        }));
                        self.stats.solver_ms = self
                            .stats
                            .solver_ms
                            .saturating_add(solver_start.elapsed().as_millis() as u64);

                        match result {
                            Ok(CegisResult::Equivalent {
                                proof_hash,
                                iterations,
                            }) => {
                                *func.inst_mut(inst_id) = replacement;
                                entry.verified += 1;
                                self.stats.verified += 1;
                                self.stats.layer_a_committed += 1;
                                entry.proven_rewrites.push(ProvenRewrite {
                                    inst_index: inst_id.0,
                                    layer: RewriteLayer::A,
                                    proof_hash,
                                    iterations: iterations as u32,
                                });
                                committed = true;
                            }
                            Ok(CegisResult::NotEquivalent { .. }) => {
                                self.stats.rejected += 1;
                                entry.rejected += 1;
                            }
                            Ok(
                                CegisResult::Timeout
                                | CegisResult::MaxIterationsReached { .. },
                            ) => {
                                self.stats.timeouts += 1;
                                self.stats.rejected += 1;
                                entry.rejected += 1;
                            }
                            Ok(CegisResult::Error(_)) => {
                                self.stats.verifier_errors += 1;
                                self.stats.rejected += 1;
                                entry.rejected += 1;
                            }
                            Err(_) => {
                                self.stats.panics += 1;
                                self.stats.rejected += 1;
                                entry.rejected += 1;
                            }
                        }
                    }
                }

                let current_inst = func.inst(inst_id);
                if current_inst.produces_value()
                    && let Some(dst) = current_inst.operands.first().and_then(MachOperand::as_vreg)
                {
                    def_map.insert(dst.id, inst_id);
                }
            }
        }

        if exhausted {
            self.stats.budget_exhausted += 1;
        }
        self.stats.solver_calls += cegis.stats_solver_calls();

        (committed, entry)
    }

    /// Match a Layer B two-instruction window.
    ///
    /// Pattern:
    /// ```text
    /// Movz  v,   #imm          (earlier in same block, only use of v)
    /// AddRR dst, src, v        (current instruction)
    /// ```
    /// Replacement (single instruction):
    /// ```text
    /// AddRI dst, src, imm
    /// ```
    ///
    /// The matcher returns `(width, movz_inst_id, replacement)` on a
    /// successful match. `movz_inst_id` identifies the Movz to splice out of
    /// the block, and `replacement` is the new `AddRI` body that will
    /// overwrite the current `AddRR` instruction. The existing `AddRR`'s
    /// `InstId` (and therefore its destination `VReg`) is preserved; only the
    /// opcode and operand shape change. SSA is preserved because:
    ///
    /// 1. The `AddRR` keeps its original `dst` VReg — consumers of `dst`
    ///    elsewhere in the function are unaffected.
    /// 2. The `Movz`'s `dst` VReg becomes dead after the splice. We verified
    ///    above that it has exactly one use (the current `AddRR`), so no
    ///    other consumer is left dangling.
    /// 3. The `AddRR`'s `src` operand is kept verbatim.
    ///
    /// Constraints enforced:
    /// - `Movz` immediate must be non-negative and fit in `u32::MAX`
    ///   (downstream encoders further restrict to ADD's 12-bit imm field,
    ///   but that is a concern for codegen, not for the semantic proof).
    /// - The Movz result VReg must be used exactly once in the function.
    /// - The register class of AddRR's `dst`, `src`, and `v` must agree
    ///   (Gpr32 or Gpr64).
    fn match_layer_b_candidate(
        func: &MachFunction,
        add_inst: &MachInst,
        def_map: &HashMap<u32, llvm2_ir::InstId>,
    ) -> Option<(u32, llvm2_ir::InstId, MachInst)> {
        if add_inst.opcode != AArch64Opcode::AddRR || add_inst.operands.len() < 3 {
            return None;
        }

        let dst = add_inst.operands.first()?.as_vreg()?;
        let src = add_inst.operands.get(1)?.as_vreg()?;
        let movz_vreg = add_inst.operands.get(2)?.as_vreg()?;

        if dst.class != src.class || dst.class != movz_vreg.class {
            return None;
        }
        let width = if dst.class == RegClass::Gpr32 {
            32
        } else if dst.class == RegClass::Gpr64 {
            64
        } else {
            return None;
        };

        let movz_id = *def_map.get(&movz_vreg.id)?;
        let movz_inst = func.inst(movz_id);

        if movz_inst.opcode != AArch64Opcode::Movz || movz_inst.operands.len() < 2 {
            return None;
        }
        if movz_inst.operands.first()?.as_vreg()? != movz_vreg {
            return None;
        }
        let imm = movz_inst.operands.get(1)?.as_imm()?;
        if imm < 0 || imm > u32::MAX as i64 {
            return None;
        }

        // Single-use check: if any other instruction in the function reads
        // the Movz destination, we cannot splice it out.
        if count_vreg_uses(func, movz_vreg) != 1 {
            return None;
        }

        let mut replacement = MachInst::new(
            AArch64Opcode::AddRI,
            vec![
                MachOperand::VReg(dst),
                MachOperand::VReg(src),
                MachOperand::Imm(imm),
            ],
        );
        replacement.proof = add_inst.proof.clone();
        replacement.source_loc = add_inst.source_loc.clone();

        Some((width, movz_id, replacement))
    }

    /// Enumerate Layer B candidates in `func` and commit rewrites that pass
    /// both the cost gate and CEGIS verification. Mirrors
    /// [`Self::enumerate_and_verify_layer_a`] but operates on two-instruction
    /// windows with SSA-preserving splice semantics.
    fn enumerate_and_verify_layer_b(
        &mut self,
        func: &mut MachFunction,
        deadline: Instant,
        entry: &mut CegisCacheEntry,
    ) -> bool {
        let mut committed = false;

        if ProvenRuleDb::seed_layer_b().is_empty() {
            return false;
        }

        let cost_model = AppleSiliconCostModel::new(CostModelGen::M1);
        let mut cegis = CegisLoop::new(1, self.config.per_query_ms);
        let mut exhausted = false;

        'blocks: for block_id in func.block_order.clone() {
            let inst_ids = func.block(block_id).insts.clone();
            let mut def_map: HashMap<u32, llvm2_ir::InstId> = HashMap::new();
            let mut to_remove: Vec<llvm2_ir::InstId> = Vec::new();

            for inst_id in inst_ids {
                if Instant::now() >= deadline {
                    exhausted = true;
                    break 'blocks;
                }

                let inst = func.inst(inst_id).clone();
                if let Some((width, movz_id, replacement)) =
                    Self::match_layer_b_candidate(func, &inst, &def_map)
                {
                    self.stats.candidates += 1;
                    self.stats.layer_b_candidates += 1;
                    entry.attempted += 1;

                    // Cost gate: sum of source latencies strictly greater
                    // than replacement latency.
                    let src_cost = cost_model.latency(AArch64Opcode::Movz) as i32
                        + cost_model.latency(AArch64Opcode::AddRR) as i32;
                    let tgt_cost = cost_model.latency(AArch64Opcode::AddRI) as i32;
                    if tgt_cost >= src_cost {
                        self.stats.rejected += 1;
                        entry.rejected += 1;
                    } else {
                        let imm_val = func.inst(movz_id)
                            .operands
                            .get(1)
                            .and_then(MachOperand::as_imm)
                            .unwrap_or(0);
                        let imm_u = (imm_val as u64) & mask_u64(width);

                        let vars = vec![("y".to_string(), width)];
                        let src_expr =
                            SmtExpr::var("y", width).bvadd(SmtExpr::bv_const(imm_u, width));
                        let tgt_expr =
                            SmtExpr::var("y", width).bvadd(SmtExpr::bv_const(imm_u, width));
                        // Scope CX state to this candidate (#493). Different
                        // `imm` values per candidate mean each `src_expr`
                        // uses a distinct constant; leaving prior CXs in
                        // place would let them reject this obligation on
                        // the concrete fast path.
                        cegis.clear_counterexamples();
                        cegis.add_edge_case_seeds(&vars);

                        let solver_start = Instant::now();
                        let result = catch_unwind(AssertUnwindSafe(|| {
                            cegis.verify(&src_expr, &tgt_expr, &vars)
                        }));
                        self.stats.solver_ms = self
                            .stats
                            .solver_ms
                            .saturating_add(solver_start.elapsed().as_millis() as u64);

                        match result {
                            Ok(CegisResult::Equivalent {
                                proof_hash,
                                iterations,
                            }) => {
                                *func.inst_mut(inst_id) = replacement;
                                to_remove.push(movz_id);
                                entry.verified += 1;
                                self.stats.verified += 1;
                                self.stats.layer_b_committed += 1;
                                entry.proven_rewrites.push(ProvenRewrite {
                                    inst_index: inst_id.0,
                                    layer: RewriteLayer::B,
                                    proof_hash,
                                    iterations: iterations as u32,
                                });
                                committed = true;
                            }
                            Ok(CegisResult::NotEquivalent { .. }) => {
                                self.stats.rejected += 1;
                                entry.rejected += 1;
                            }
                            Ok(
                                CegisResult::Timeout
                                | CegisResult::MaxIterationsReached { .. },
                            ) => {
                                self.stats.timeouts += 1;
                                self.stats.rejected += 1;
                                entry.rejected += 1;
                            }
                            Ok(CegisResult::Error(_)) => {
                                self.stats.verifier_errors += 1;
                                self.stats.rejected += 1;
                                entry.rejected += 1;
                            }
                            Err(_) => {
                                self.stats.panics += 1;
                                self.stats.rejected += 1;
                                entry.rejected += 1;
                            }
                        }
                    }
                }

                // Update def_map with the CURRENT body (possibly just
                // rewritten from AddRR to AddRI — both produce `dst`).
                let current_inst = func.inst(inst_id);
                if current_inst.produces_value()
                    && let Some(dst) = current_inst.operands.first().and_then(MachOperand::as_vreg)
                {
                    def_map.insert(dst.id, inst_id);
                }
            }

            // Apply pending splices: drop spliced-out insts from the block
            // schedule. The arena entries remain (orphaned) but are no
            // longer scheduled; regalloc / codegen will not emit them.
            if !to_remove.is_empty() {
                let remove_set: std::collections::HashSet<_> = to_remove.into_iter().collect();
                func.block_mut(block_id)
                    .insts
                    .retain(|id| !remove_set.contains(id));
            }
        }

        if exhausted {
            self.stats.budget_exhausted += 1;
        }
        self.stats.solver_calls += cegis.stats_solver_calls();

        committed
    }

    /// Apply a single cached rewrite to `func` by re-running the matcher at
    /// the recorded `inst_index` and (for Layer B) splicing out the Movz.
    ///
    /// Returns `true` if the rewrite was applied. Because the cache key
    /// encodes the function body, a cache hit implies the matcher input is
    /// byte-identical to the cold-path input, so the matcher result is
    /// deterministic. A matcher miss on replay indicates cache corruption or
    /// a matcher bug; the caller treats that as "skip this rewrite" rather
    /// than failing loudly, preserving forward progress (the pass is still
    /// semantics-preserving, just less optimal on this one site).
    fn apply_cached_rewrite(func: &mut MachFunction, rewrite: &ProvenRewrite) -> bool {
        let target_id = llvm2_ir::InstId(rewrite.inst_index);
        // Validate that the index is in-bounds before touching the function.
        if (target_id.0 as usize) >= func.insts.len() {
            return false;
        }

        // Reconstruct the def_map that would have been visible to the
        // matcher at the moment it originally fired. We rebuild it by
        // walking the block schedule up to (but not including) `target_id`.
        //
        // Because the replay runs on a byte-identical function, this
        // def_map is identical to the one the cold path saw.
        let Some(block_id) = func.block_order.iter().copied().find(|b| {
            func.block(*b)
                .insts
                .iter()
                .any(|id| *id == target_id)
        }) else {
            return false;
        };

        let mut def_map: HashMap<u32, llvm2_ir::InstId> = HashMap::new();
        for &inst_id in &func.block(block_id).insts {
            if inst_id == target_id {
                break;
            }
            let inst = func.inst(inst_id);
            if inst.produces_value()
                && let Some(dst) = inst.operands.first().and_then(MachOperand::as_vreg)
            {
                def_map.insert(dst.id, inst_id);
            }
        }

        let inst = func.inst(target_id).clone();
        match rewrite.layer {
            RewriteLayer::A => {
                if let Some((_width, replacement)) =
                    Self::match_layer_a_candidate(func, &inst, &def_map)
                {
                    *func.inst_mut(target_id) = replacement;
                    true
                } else {
                    false
                }
            }
            RewriteLayer::B => {
                if let Some((_width, movz_id, replacement)) =
                    Self::match_layer_b_candidate(func, &inst, &def_map)
                {
                    *func.inst_mut(target_id) = replacement;
                    // Splice the Movz out of its block's schedule, mirroring
                    // the cold-path behavior in `enumerate_and_verify_layer_b`.
                    let movz_block = func
                        .block_order
                        .iter()
                        .copied()
                        .find(|b| func.block(*b).insts.contains(&movz_id));
                    if let Some(mb) = movz_block {
                        func.block_mut(mb).insts.retain(|id| *id != movz_id);
                    }
                    true
                } else {
                    false
                }
            }
        }
    }

    /// Replay all cached rewrites in order. Returns
    /// `(committed, applied_a, applied_b)` where the per-layer `applied_*`
    /// counters record how many rewrites actually mutated the function this
    /// run. Per-layer counts let the caller bump `layer_a_committed` and
    /// `layer_b_committed` separately on cache hits.
    ///
    /// Layer A rewrites must be replayed before Layer B, because Layer B may
    /// reference instructions whose def/use graph depends on Layer A's
    /// rewrites having been applied first (same ordering the cold path uses).
    fn replay_cached_rewrites(
        func: &mut MachFunction,
        entry: &CegisCacheEntry,
    ) -> (bool, u64, u64) {
        let mut applied_a: u64 = 0;
        let mut applied_b: u64 = 0;
        let mut committed = false;

        // Sort rewrites: Layer A first, then Layer B. Within a layer,
        // preserve recorded order (which mirrors cold-path enumeration order).
        let mut layer_a: Vec<&ProvenRewrite> = Vec::new();
        let mut layer_b: Vec<&ProvenRewrite> = Vec::new();
        for r in &entry.proven_rewrites {
            match r.layer {
                RewriteLayer::A => layer_a.push(r),
                RewriteLayer::B => layer_b.push(r),
            }
        }

        for r in layer_a.into_iter() {
            if Self::apply_cached_rewrite(func, r) {
                applied_a += 1;
                committed = true;
            }
        }
        for r in layer_b.into_iter() {
            if Self::apply_cached_rewrite(func, r) {
                applied_b += 1;
                committed = true;
            }
        }

        (committed, applied_a, applied_b)
    }

    /// Body of `run` split out so that it can be unit-tested without needing
    /// to implement `MachinePass` in tests.
    fn run_inner(&mut self, func: &mut MachFunction) -> bool {
        if !self.config.is_enabled() {
            return false;
        }
        self.stats.functions_seen += 1;

        // Wall-clock the entire pass body (cache lookup + replay OR
        // enumeration + CEGIS + cache put) so that #486's `total_wall_ms`
        // reflects the full cost observed by the pipeline. `solver_ms` is
        // accumulated independently inside enumerate_and_verify_layer_{a,b}
        // and represents a strict subset of `total_wall_ms`.
        let wall_start = Instant::now();

        // Snapshot pre-invocation counters so the trace event can report the
        // delta attributable to THIS call (rather than the running totals
        // across all functions the pass instance has seen).
        let candidates_before = self.stats.candidates;
        let verified_before = self.stats.verified;
        let rejected_before = self.stats.rejected;
        let layer_a_candidates_before = self.stats.layer_a_candidates;
        let layer_a_committed_before = self.stats.layer_a_committed;
        let layer_b_candidates_before = self.stats.layer_b_candidates;
        let layer_b_committed_before = self.stats.layer_b_committed;
        let timeouts_before = self.stats.timeouts;
        let verifier_errors_before = self.stats.verifier_errors;
        let panics_before = self.stats.panics;
        let solver_ms_before = self.stats.solver_ms;
        let solver_calls_before = self.stats.solver_calls;

        let key = self.compute_function_key(func);

        // Cache hit / miss outcome for trace reporting.
        let mut from_cache = false;
        let committed;

        // Cache hit path --------------------------------------------------
        if let Some(backend) = self.config.cache.as_ref()
            && let Some(bytes) = backend.get(&key)
            && let Some(entry) = CegisCacheEntry::decode(&bytes)
        {
            self.stats.cache_hits += 1;
            from_cache = true;
            // Replay the cached rewrites so the function actually mutates.
            // Only bump `verified` / layer-committed by the number of
            // rewrites we actually applied this run (not `entry.verified`,
            // which would be a phantom count — see #491).
            let (c, applied_a, applied_b) = Self::replay_cached_rewrites(func, &entry);
            committed = c;
            let applied = applied_a + applied_b;
            self.stats.verified += applied;
            self.stats.candidates += applied;
            self.stats.layer_a_committed += applied_a;
            self.stats.layer_a_candidates += applied_a;
            self.stats.layer_b_committed += applied_b;
            self.stats.layer_b_candidates += applied_b;
            // `rejected` is a cold-path outcome (CEGIS said no / timeout /
            // error). Replay never "rejects" — it either applies or can't
            // find the site. We therefore do NOT re-credit `entry.rejected`
            // on a hit; the observability is about work done this run.
        } else {
            // Cache miss path ---------------------------------------------
            self.stats.cache_misses += 1;

            let deadline = Instant::now() + Duration::from_secs(self.config.budget_sec);
            let (committed_a, mut entry) =
                self.enumerate_and_verify_layer_a(func, deadline);
            // Layer B runs after Layer A so that any Layer-A-rewrites are
            // visible to the window enumerator (e.g. a MulRR->Movz collapse
            // could unblock a subsequent two-inst fusion).
            let committed_b =
                self.enumerate_and_verify_layer_b(func, deadline, &mut entry);
            committed = committed_a || committed_b;

            // Persist the (possibly empty) entry so repeat runs see a cache hit.
            if let Some(backend) = self.config.cache.as_ref() {
                match entry.encode() {
                    Ok(bytes) => {
                        backend.put(&key, &bytes);
                        self.stats.cache_puts += 1;
                    }
                    Err(_) => {
                        // Serialization should never fail for a well-formed
                        // entry; if it does, drop the put silently — the pass
                        // is still semantically correct, just slower next time.
                    }
                }
            }
        }

        // Update wall-clock stats BEFORE emitting the trace event so that
        // the event's payload reflects final numbers for this invocation.
        let wall_ms = wall_start.elapsed().as_millis() as u64;
        self.stats.total_wall_ms = self.stats.total_wall_ms.saturating_add(wall_ms);

        // Emit a single structured trace event summarising this invocation
        // (issue #486 / #492 §10). The event name slot carries the full
        // stats payload encoded as `key=value` pairs so it round-trips
        // through `CompilationTrace::to_json` without needing new enum
        // variants. Downstream tooling can parse the payload with a trivial
        // regex; the goal is discoverability under `cargo run -- --trace`
        // more than typed access.
        if let Some(trace) = self.config.trace.as_ref() {
            // Per-invocation deltas (since we only ever accumulate).
            let d_candidates = self.stats.candidates - candidates_before;
            let d_verified = self.stats.verified - verified_before;
            let d_rejected = self.stats.rejected - rejected_before;
            let d_layer_a_candidates =
                self.stats.layer_a_candidates - layer_a_candidates_before;
            let d_layer_a_committed =
                self.stats.layer_a_committed - layer_a_committed_before;
            let d_layer_b_candidates =
                self.stats.layer_b_candidates - layer_b_candidates_before;
            let d_layer_b_committed =
                self.stats.layer_b_committed - layer_b_committed_before;
            let d_timeouts = self.stats.timeouts - timeouts_before;
            let d_verifier_errors = self.stats.verifier_errors - verifier_errors_before;
            let d_panics = self.stats.panics - panics_before;
            let d_solver_ms = self.stats.solver_ms - solver_ms_before;
            let d_solver_calls = self.stats.solver_calls - solver_calls_before;

            let payload = format!(
                "CegisSuperoptPass{{func={func},cache={cache},wall_ms={wall_ms},solver_ms={solver_ms},candidates={candidates},verified={verified},rejected={rejected},layer_a_candidates={lac},layer_a_committed={laco},layer_b_candidates={lbc},layer_b_committed={lbco},timeouts={timeouts},verifier_errors={verr},panics={panics},solver_calls={calls},committed={committed}}}",
                func = func.name,
                cache = if from_cache { "hit" } else { "miss" },
                wall_ms = wall_ms,
                solver_ms = d_solver_ms,
                candidates = d_candidates,
                verified = d_verified,
                rejected = d_rejected,
                lac = d_layer_a_candidates,
                laco = d_layer_a_committed,
                lbc = d_layer_b_candidates,
                lbco = d_layer_b_committed,
                timeouts = d_timeouts,
                verr = d_verifier_errors,
                panics = d_panics,
                calls = d_solver_calls,
                committed = committed,
            );

            // Rule IDs: 486 (Applied) and 486 (Rejected) are stable
            // sentinel values tied to issue #486. They are intentionally
            // the same so post-processing can filter by "rule=486" and
            // then discriminate on EventKind (Applied vs Rejected).
            let (kind, justification) = if committed {
                (
                    EventKind::Applied {
                        rule: RuleId(486),
                        before: Vec::new(),
                        after: Vec::new(),
                    },
                    Justification::SolverProved {
                        proof_hash: d_verified,
                    },
                )
            } else {
                (
                    EventKind::Rejected {
                        rule: RuleId(486),
                        reason: if from_cache {
                            "cache-hit: no rewrites replayed".to_string()
                        } else if d_candidates == 0 {
                            "no candidate sites matched".to_string()
                        } else {
                            "all candidates rejected by cost gate or CEGIS".to_string()
                        },
                    },
                    Justification::CostModel {
                        before: d_candidates as f64,
                        after: d_verified as f64,
                    },
                )
            };

            trace.emit(TracePassId::new(payload), kind, Vec::new(), justification);
        }

        committed
    }
}

impl MachinePass for CegisSuperoptPass {
    fn name(&self) -> &str {
        "CegisSuperoptPass"
    }

    fn run(&mut self, func: &mut MachFunction) -> bool {
        self.run_inner(func)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use llvm2_opt::{CacheBackend, CacheKey, InMemoryCache};
    use std::sync::Arc;

    // Build a minimally-valid MachFunction for tests via the public
    // constructor. We only need the name + insts + blocks + block_order
    // + entry fields for the pass to compute a cache key.
    fn empty_func(name: &str) -> MachFunction {
        use llvm2_ir::function::Signature;
        MachFunction::new(name.to_string(), Signature::new(vec![], vec![]))
    }

    fn make_config(cache: Option<Arc<dyn CacheBackend>>, budget_sec: u64) -> CegisSuperoptConfig {
        CegisSuperoptConfig {
            budget_sec,
            per_query_ms: 1000,
            target_triple: "aarch64-apple-darwin".to_string(),
            cpu: "apple-m1".to_string(),
            features: vec!["neon".to_string(), "fp-armv8".to_string()],
            opt_level: 2,
            cache,
            trace: None,
        }
    }

    #[test]
    fn test_config_disabled_default() {
        let cfg = CegisSuperoptConfig::default();
        assert!(!cfg.is_enabled());
        assert_eq!(cfg.budget_sec, 0);
    }

    #[test]
    fn test_disabled_pass_is_noop() {
        let mut pass = CegisSuperoptPass::new(CegisSuperoptConfig::default());
        let mut func = empty_func("noop");
        assert_eq!(pass.run(&mut func), false);
        assert_eq!(pass.stats().functions_seen, 0);
        assert_eq!(pass.stats().cache_hits, 0);
        assert_eq!(pass.stats().cache_misses, 0);
    }

    #[test]
    fn test_enabled_pass_records_miss_then_hit() {
        let cache: Arc<dyn CacheBackend> = Arc::new(InMemoryCache::new());
        let cfg = make_config(Some(cache.clone()), 1);

        // First run: cache miss, put.
        let mut pass = CegisSuperoptPass::new(cfg.clone());
        let mut func = empty_func("f");
        let _ = pass.run(&mut func);
        assert_eq!(pass.stats().functions_seen, 1);
        assert_eq!(pass.stats().cache_misses, 1);
        assert_eq!(pass.stats().cache_hits, 0);
        assert_eq!(pass.stats().cache_puts, 1);

        // Second run on an identical function: cache hit.
        let mut pass2 = CegisSuperoptPass::new(cfg);
        let mut func2 = empty_func("f");
        let _ = pass2.run(&mut func2);
        assert_eq!(pass2.stats().functions_seen, 1);
        assert_eq!(pass2.stats().cache_hits, 1);
        assert_eq!(pass2.stats().cache_misses, 0);
    }

    #[test]
    fn test_key_is_deterministic_across_instances() {
        let cfg = make_config(None, 1);
        let p1 = CegisSuperoptPass::new(cfg.clone());
        let p2 = CegisSuperoptPass::new(cfg);
        let f1 = empty_func("same");
        let f2 = empty_func("same");
        let k1 = p1.compute_function_key(&f1);
        let k2 = p2.compute_function_key(&f2);
        assert_eq!(k1, k2);
    }

    #[test]
    fn test_key_differs_with_function_name() {
        let cfg = make_config(None, 1);
        let pass = CegisSuperoptPass::new(cfg);
        let f1 = empty_func("alpha");
        let f2 = empty_func("beta");
        assert_ne!(
            pass.compute_function_key(&f1),
            pass.compute_function_key(&f2)
        );
    }

    #[test]
    fn test_key_differs_with_features() {
        let mut cfg_a = make_config(None, 1);
        let mut cfg_b = make_config(None, 1);
        cfg_b.features.push("extra-feature".to_string());
        let pa = CegisSuperoptPass::new(cfg_a.clone());
        let pb = CegisSuperoptPass::new(cfg_b.clone());
        let f = empty_func("f");
        assert_ne!(pa.compute_function_key(&f), pb.compute_function_key(&f));
        // And feature order does NOT matter (sort+dedup inside CacheKey::new).
        cfg_a.features = vec!["b".to_string(), "a".to_string()];
        cfg_b.features = vec!["a".to_string(), "b".to_string()];
        let pa = CegisSuperoptPass::new(cfg_a);
        let pb = CegisSuperoptPass::new(cfg_b);
        assert_eq!(pa.compute_function_key(&f), pb.compute_function_key(&f));
    }

    #[test]
    fn test_cache_entry_roundtrip() {
        let entry = CegisCacheEntry {
            version: CegisCacheEntry::VERSION,
            proven_rewrites: vec![ProvenRewrite {
                inst_index: 3,
                layer: RewriteLayer::A,
                proof_hash: 0xDEAD_BEEF,
                iterations: 1,
            }],
            attempted: 10,
            verified: 1,
            rejected: 9,
        };
        let bytes = entry.encode().expect("encode");
        let decoded = CegisCacheEntry::decode(&bytes).expect("decode");
        assert_eq!(decoded, entry);
    }

    #[test]
    fn test_cache_entry_wrong_version_rejected() {
        let entry = CegisCacheEntry {
            version: 999,
            ..Default::default()
        };
        let bytes = entry.encode().expect("encode");
        assert!(CegisCacheEntry::decode(&bytes).is_none());
    }

    #[test]
    fn test_cache_entry_older_versions_rejected() {
        // v1 (Layer A only) and v2 (pre-replay) must both decode to None
        // after the v3 bump so the pass re-enumerates under the replay-
        // capable enumerator on the next run.
        for old_version in [1u32, 2u32] {
            let entry = CegisCacheEntry {
                version: old_version,
                ..Default::default()
            };
            let bytes = entry.encode().expect("encode");
            assert!(
                CegisCacheEntry::decode(&bytes).is_none(),
                "v{} cache entries must be rejected after VERSION=3 bump",
                old_version,
            );
        }
    }

    #[test]
    fn test_pass_name() {
        let p = CegisSuperoptPass::new(CegisSuperoptConfig::default());
        assert_eq!(p.name(), "CegisSuperoptPass");
    }

    #[test]
    fn test_pass_without_cache_still_runs() {
        let cfg = make_config(None, 1);
        let mut pass = CegisSuperoptPass::new(cfg);
        let mut func = empty_func("nocache");
        let _ = pass.run(&mut func);
        assert_eq!(pass.stats().functions_seen, 1);
        assert_eq!(pass.stats().cache_misses, 1);
        assert_eq!(pass.stats().cache_puts, 0); // no backend
    }

    #[test]
    fn test_cache_key_round_trip_through_backend() {
        let cache = InMemoryCache::new();
        let key = CacheKey::new(
            0x12345678_ABCDEF00_u128,
            2,
            "aarch64-apple-darwin".to_string(),
            "apple-m1".to_string(),
            vec!["neon".to_string()],
        );
        let entry = CegisCacheEntry::empty();
        let bytes = entry.encode().unwrap();
        cache.put(&key, &bytes);
        let got = cache.get(&key).expect("hit");
        let decoded = CegisCacheEntry::decode(&got).expect("decode");
        assert_eq!(decoded, entry);
    }
}
