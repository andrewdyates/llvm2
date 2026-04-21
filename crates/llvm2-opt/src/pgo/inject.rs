// llvm2-opt/pgo/inject.rs - Basic-block counter-injection pass
//
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Copyright 2026 Andrew Yates | License: Apache-2.0
//
// Reference: designs/2026-04-18-pgo-workflow.md (section 1, Instrumentation)
//
// Phase 1 MVP. Emits a `Bl` pseudo-call to a per-site symbol
// (`__llvm2_profile_bb_enter$<func>$<bb>`) at the top of every basic
// block. The symbol encodes the (function, block) pair so the runtime
// helper can be generated, or the call sites can be counted statically.
//
// The pass records a [`CounterMap`] describing every instrumentation
// site in emission order. Writers use the map to produce a
// [`super::schema::ProfData`] layout even when no run has happened yet
// (zero-count blocks). Consumers that prefer to call
// `__llvm2_profile_bb_enter(func_id, bb_id)` with integer args can
// layer lowering on top of this pass.

//! Basic-block counter-injection pass.
//!
//! Behind [`PipelineConfig::profile_generate`](super::PipelineConfig), the
//! pipeline injects a profiling site at the top of every basic block so
//! the runtime can record hit counts per block. The pass is AArch64-only
//! for now (matches the existing per-function trampoline, see
//! `crates/llvm2-codegen/src/jit.rs:1477`).

use llvm2_ir::{AArch64Opcode, BlockId, InstId, MachFunction, MachInst, MachOperand};

use crate::pass_manager::{AnalysisCache, MachinePass};

/// Symbol prefix for generated per-block counter sites.
///
/// A block counter site for function `foo`, block `3` is emitted as:
///
/// ```text
///   BL  __llvm2_profile_bb_enter$foo$3
/// ```
///
/// The `$` separator is chosen to avoid collisions with Rust / Swift /
/// C++ mangled names (which use other ASCII punctuation but not `$` in
/// the middle of a legal identifier).
pub const COUNTER_SYMBOL_PREFIX: &str = "__llvm2_profile_bb_enter";

/// A single counter-site record.
///
/// Produced by [`inject_block_counters`] and consumed by
/// [`super::schema::ProfData`] writers. The `counter_index` is a dense,
/// zero-based index assigned in emission order — suitable for indexing
/// a flat `Vec<AtomicU64>` in the runtime.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CounterSite {
    /// Function symbol name (copied from [`MachFunction::name`]).
    pub function: String,
    /// Block id as a `u32`.
    pub block_id: u32,
    /// Dense counter index within the enclosing function.
    pub counter_index: u32,
    /// Mangled symbol used for the emitted `BL` target.
    pub symbol: String,
}

/// Map from counter sites to their dense indices, per function.
///
/// Returned by [`inject_block_counters`]. The counter map is persisted
/// alongside the compiled artifact so the runtime and the writer agree
/// on the counter layout.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct CounterMap {
    /// All sites across all functions, in emission order.
    pub sites: Vec<CounterSite>,
}

impl CounterMap {
    /// Create an empty counter map.
    pub fn new() -> Self {
        Self::default()
    }

    /// Number of recorded counter sites.
    pub fn len(&self) -> usize {
        self.sites.len()
    }

    /// Whether no counter sites have been recorded.
    pub fn is_empty(&self) -> bool {
        self.sites.is_empty()
    }

    /// Iterate over the sites belonging to `function`, in the order they
    /// were emitted.
    pub fn sites_for<'a>(
        &'a self,
        function: &'a str,
    ) -> impl Iterator<Item = &'a CounterSite> + 'a {
        self.sites.iter().filter(move |s| s.function == function)
    }

    /// Absorb the sites from another counter map at the tail.
    pub fn extend(&mut self, other: CounterMap) {
        self.sites.extend(other.sites);
    }
}

/// Build the mangled symbol for a `(function, block)` pair.
///
/// Exposed for tests and for runtime helpers that need to recognise
/// counter sites by name.
pub fn counter_symbol(function: &str, block_id: u32) -> String {
    format!("{}${}${}", COUNTER_SYMBOL_PREFIX, function, block_id)
}

/// Inject per-block counter sites into `func`.
///
/// Returns a [`CounterMap`] describing every site that was emitted.
/// The function's block ordering and CFG are not disturbed — the pass
/// only prepends a single `BL <symbol>` instruction to each block.
///
/// AArch64-only for now. Callers must not invoke this on functions
/// destined for x86-64; the compiler pipeline is expected to gate on
/// [`llvm2_ir::target_info`] (same gating as the per-function
/// trampoline).
pub fn inject_block_counters(func: &mut MachFunction) -> CounterMap {
    let mut map = CounterMap::new();

    // Snapshot the block order; we will prepend instructions but not
    // reshape the CFG.
    let blocks: Vec<BlockId> = func.block_order.clone();
    let mut next_index: u32 = 0;
    for block_id in blocks {
        let symbol = counter_symbol(&func.name, block_id.0);

        let inst = MachInst::new(
            AArch64Opcode::Bl,
            vec![MachOperand::Symbol(symbol.clone()), MachOperand::Imm(block_id.0 as i64)],
        );
        let inst_id: InstId = func.push_inst(inst);
        prepend_inst(func, block_id, inst_id);

        map.sites.push(CounterSite {
            function: func.name.clone(),
            block_id: block_id.0,
            counter_index: next_index,
            symbol,
        });
        next_index += 1;
    }

    map
}

fn prepend_inst(func: &mut MachFunction, block_id: BlockId, inst_id: InstId) {
    let block = func.block_mut(block_id);
    block.insts.insert(0, inst_id);
}

/// A [`MachinePass`] wrapper around [`inject_block_counters`] that
/// publishes its [`CounterMap`] via an external shared cell.
///
/// The pipeline holds the shared cell so it can assemble the
/// cross-function counter map as passes execute on each function.
pub struct CounterInjectionPass {
    sink: std::sync::Arc<std::sync::Mutex<CounterMap>>,
}

impl CounterInjectionPass {
    /// Construct a pass that appends all emitted sites to `sink`.
    pub fn new(sink: std::sync::Arc<std::sync::Mutex<CounterMap>>) -> Self {
        Self { sink }
    }

    /// Pass name, exposed for tests.
    pub const NAME: &'static str = "pgo-counter-injection";
}

impl MachinePass for CounterInjectionPass {
    fn name(&self) -> &str {
        Self::NAME
    }

    fn run(&mut self, func: &mut MachFunction) -> bool {
        let local = inject_block_counters(func);
        if local.is_empty() {
            return false;
        }
        let mut guard = match self.sink.lock() {
            Ok(g) => g,
            Err(poisoned) => poisoned.into_inner(),
        };
        guard.extend(local);
        true
    }

    fn run_with_analyses(
        &mut self,
        func: &mut MachFunction,
        _analyses: &mut AnalysisCache,
    ) -> bool {
        self.run(func)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use llvm2_ir::{AArch64Opcode, MachFunction, MachInst, MachOperand, Signature};

    fn empty_func(name: &str) -> MachFunction {
        MachFunction::new(name.to_string(), Signature::new(vec![], vec![]))
    }

    fn func_with_two_blocks() -> MachFunction {
        let mut f = empty_func("two_blocks");
        let bb0 = f.entry;
        let bb1 = f.create_block();

        let br = f.push_inst(MachInst::new(
            AArch64Opcode::B,
            vec![MachOperand::Block(bb1)],
        ));
        f.append_inst(bb0, br);

        let ret = f.push_inst(MachInst::new(AArch64Opcode::Ret, vec![]));
        f.append_inst(bb1, ret);

        f.add_edge(bb0, bb1);
        f
    }

    #[test]
    fn counter_symbol_format_is_stable() {
        assert_eq!(
            counter_symbol("foo", 3),
            "__llvm2_profile_bb_enter$foo$3"
        );
    }

    #[test]
    fn inject_emits_one_call_per_block() {
        let mut f = func_with_two_blocks();
        let pre0 = f.block(BlockId(0)).insts.len();
        let pre1 = f.block(BlockId(1)).insts.len();
        let map = inject_block_counters(&mut f);

        assert_eq!(map.sites.len(), 2);
        assert_eq!(map.sites[0].function, "two_blocks");
        assert_eq!(map.sites[0].block_id, 0);
        assert_eq!(map.sites[0].counter_index, 0);
        assert_eq!(map.sites[1].block_id, 1);
        assert_eq!(map.sites[1].counter_index, 1);

        assert_eq!(f.block(BlockId(0)).insts.len(), pre0 + 1);
        assert_eq!(f.block(BlockId(1)).insts.len(), pre1 + 1);

        // The prepended instruction must be a BL whose first operand is
        // the counter symbol.
        for site in &map.sites {
            let block_id = BlockId(site.block_id);
            let first = f.block(block_id).insts[0];
            let inst = f.inst(first);
            assert_eq!(inst.opcode, AArch64Opcode::Bl);
            match &inst.operands[0] {
                MachOperand::Symbol(s) => assert_eq!(s, &site.symbol),
                other => panic!("expected Symbol operand, got {:?}", other),
            }
        }
    }

    #[test]
    fn inject_preserves_existing_instructions() {
        let mut f = func_with_two_blocks();

        let original_bb0_first_inst = f.block(BlockId(0)).insts[0];
        inject_block_counters(&mut f);

        // The original first instruction must now be the second.
        let new_second = f.block(BlockId(0)).insts[1];
        assert_eq!(new_second, original_bb0_first_inst);
    }

    #[test]
    fn counter_map_sites_for_filters_by_function() {
        let mut a = empty_func("alpha");
        a.create_block();
        let mut b = empty_func("beta");
        let mut map = inject_block_counters(&mut a);
        map.extend(inject_block_counters(&mut b));
        assert!(map.sites_for("alpha").count() >= 1);
        assert!(map.sites_for("beta").count() >= 1);
        assert_eq!(map.sites_for("missing").count(), 0);
    }

    #[test]
    fn pass_runs_and_publishes_sites() {
        let sink = std::sync::Arc::new(std::sync::Mutex::new(CounterMap::new()));
        let mut pass = CounterInjectionPass::new(sink.clone());
        let mut f = func_with_two_blocks();
        let changed = pass.run(&mut f);
        assert!(changed);
        let guard = sink.lock().unwrap();
        assert_eq!(guard.sites.len(), 2);
    }

    #[test]
    fn pass_on_empty_function_reports_change_for_entry_block() {
        // Every MachFunction has at least an entry block, so even a
        // newly-constructed function has 1 site.
        let sink = std::sync::Arc::new(std::sync::Mutex::new(CounterMap::new()));
        let mut pass = CounterInjectionPass::new(sink.clone());
        let mut f = empty_func("noop");
        assert!(pass.run(&mut f));
        assert_eq!(sink.lock().unwrap().sites.len(), 1);
    }
}
