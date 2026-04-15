# Unified Solver Integration Plan

**Author:** Andrew Yates
**Date:** 2026-04-14
**Status:** Integration plan (Part of #121)
**Depends on:** designs/2026-04-13-unified-solver-architecture.md (master design)

---

## Implementation Status (as of 2026-04-15)

**Overall: Individual modules exist (semantics encoders, synthesis loops, cost model, dispatch). Integration gaps between modules remain -- the unified solver does not yet drive actual compilation decisions end-to-end.**

| Component | Status | Details |
|-----------|--------|---------|
| **Per-target semantic modules** | IMPLEMENTED | CPU, NEON, GPU, ANE encodings all present. |
| **Unified synthesis loop** | IMPLEMENTED | `unified_synthesis.rs` (5.1K LOC) searches across targets. |
| **End-to-end data flow** | NOT CONNECTED | Synthesis results do not automatically feed into the compilation pipeline's ISel/dispatch decisions. |
| **z4 solver integration** | NOT CONNECTED | See #34, #121, #236. |

---

## Purpose

This document maps the end-to-end data flow through the unified solver system,
identifies every integration gap between the modules built in Waves 1-3, specifies
exact module interfaces, proposes implementation order, and lists concrete issues
for each gap.

---

## 1. End-to-End Data Flow

```
tMIR Module (tmir_func::Module)
  |
  |  [A] GraphBuilder::build_from_module()
  |      crates/llvm2-lower/src/compute_graph.rs:GraphBuilder
  v
ComputeGraph { nodes: Vec<ComputeNode>, edges: Vec<DataEdge> }
  |
  |  [B] ProofAnalyzer::analyze()
  |      crates/llvm2-lower/src/target_analysis.rs:ProofAnalyzer
  |      Input: SubgraphDescriptor per node
  |      Output: TargetLegality per node (legal targets + justification)
  v
ComputeGraph with annotated legal_targets per node
  |
  |  [C] UnifiedSynthesisLoop (DOES NOT EXIST YET)
  |      For each (node, legal_target) pair:
  |        1. Encode tMIR subgraph semantics -> SmtExpr (tmir_semantics.rs)
  |        2. Enumerate candidates on that target (synthesis.rs + new encoders)
  |        3. CEGIS verify each candidate (cegis.rs)
  |        4. Rank proven candidates by cost (cost_model.rs)
  |        5. Optionally accept AI proposals (rule_discovery.rs)
  v
Vec<ProvenCandidate> per node, each with:
  - target: ComputeTarget
  - implementation: SmtExpr (proven equivalent)
  - cost: u64 (from AppleSiliconCostModel or GPU/ANE model)
  - proof_hash: u64
  |
  |  [D] DispatchPlanner (DOES NOT EXIST YET)
  |      Assigns best target per node, accounts for transfer costs
  |      Uses ComputeGraph::partition_cost() for global optimization
  v
TargetAssignment: HashMap<ComputeNodeId, (ComputeTarget, ProvenCandidate)>
  |
  |  [E] Per-target lowering (partially exists)
  |      CPU scalar: isel.rs -> regalloc -> codegen (EXISTS)
  |      NEON SIMD: neon_semantics.rs (semantics exist, ISel gap)
  |      GPU/ANE: (NOT STARTED)
  v
MachineFunction per CPU subgraph, Metal IR per GPU subgraph, etc.
  |
  |  [F] Verification of each lowered piece (partially exists)
  |      z4_bridge.rs: BV proofs (EXISTS)
  |      memory_proofs.rs: Load/Store proofs (EXISTS)
  |      neon_semantics.rs + CEGIS: NEON proofs (PARTIAL)
  v
ProofCertificate per lowering rule
  |
  |  [G] Code emission (EXISTS for AArch64)
  |      codegen crate -> Mach-O
  v
Output: verified binary + proof certificates
```

### Legend for gaps

| Label | Status | Module(s) involved |
|-------|--------|--------------------|
| **[A]** | EXISTS but incomplete | `compute_graph.rs` — GraphBuilder uses tMIR stubs; needs real tMIR integration |
| **[B]** | EXISTS | `target_analysis.rs` — ProofAnalyzer, SubgraphDescriptor, TargetLegality all working |
| **[C]** | **GAP** | No unified synthesis loop exists; CEGIS and synthesis are single-target only |
| **[D]** | **GAP** | No dispatch planner; partition_cost() exists but no optimizer calls it |
| **[E]** | PARTIAL | CPU scalar ISel exists; NEON ISel not wired; GPU/ANE not started |
| **[F]** | PARTIAL | BV proofs and memory proofs exist; NEON proofs not integrated into pipeline |
| **[G]** | EXISTS | AArch64 codegen and Mach-O emission working |

---

## 2. Integration Gaps

### Gap 1: compute_graph.rs -> target_analysis.rs bridge

**What exists:**
- `compute_graph.rs` produces `ComputeGraph` with `ComputeNode` items, each having
  `legal_targets: Vec<ComputeTarget>` and `costs: HashMap<ComputeTarget, ComputeCost>`.
- `target_analysis.rs` has `ProofAnalyzer::analyze(descriptor: &SubgraphDescriptor) -> TargetLegality`.
- `SubgraphDescriptor` needs: id, values, value_types, subgraph_proofs, data_size_bytes.

**What is missing:**
- No code converts a `ComputeNode` into a `SubgraphDescriptor`. The `GraphBuilder`
  populates `legal_targets` directly using heuristics, bypassing `ProofAnalyzer` entirely.
- `ComputeNode.instructions` are `TmirInstId` references, but there is no method to
  extract `Value`, `Type`, or `SubgraphProof` from the tMIR module for those instructions.
- The `TargetProofContext` wraps `ProofContext` from the adapter layer, but the adapter's
  `ProofContext` is populated during lowering (adapter.rs), not during graph construction.

**Required connecting code:**
```rust
// In compute_graph.rs or a new bridge module:
pub fn build_descriptor(
    node: &ComputeNode,
    module: &TmirModule,
    proof_ctx: &TargetProofContext,
) -> SubgraphDescriptor {
    // 1. Map TmirInstId -> tMIR instructions -> extract values and types
    // 2. Query proof_ctx for subgraph-level proofs
    // 3. Build SubgraphDescriptor
}

pub fn annotate_graph_with_proofs(
    graph: &mut ComputeGraph,
    module: &TmirModule,
    proof_ctx: &TargetProofContext,
    analyzer: &ProofAnalyzer,
) {
    for node in &mut graph.nodes {
        let desc = build_descriptor(node, module, proof_ctx);
        let legality = analyzer.analyze(&desc);
        node.legal_targets = legality.legal_targets();
        // Also set per-target costs from legality.data_size_bytes + cost model
    }
}
```

### Gap 2: target_analysis.rs -> CEGIS search space constraint

**What exists:**
- `target_analysis.rs` produces `TargetLegality` per subgraph, listing legal targets.
- `cegis.rs` has `CegisLoop::verify(source, target, vars)` for single-pair equivalence.
- `synthesis.rs` has `SearchSpace::enumerate(config)` producing scalar AArch64 candidates.

**What is missing:**
- No multi-target enumeration. `SearchSpace` only produces scalar AArch64 `SynthOpcode`
  patterns. There is no NEON, GPU, or ANE candidate enumeration.
- No way to constrain the CEGIS search to only targets deemed legal by `TargetLegality`.
- The synthesis engine (`SynthesisEngine`) verifies one candidate at a time using
  `verify_by_evaluation()` (exhaustive/sampling), not the CEGIS loop.

**Required connecting code:**
```rust
// New: unified_synthesis.rs in llvm2-verify
pub struct UnifiedSynthesisConfig {
    pub legal_targets: Vec<ComputeTarget>,
    pub source_smt: SmtExpr,
    pub source_vars: Vec<(String, u32)>,
    pub cost_model: Box<dyn CostModel>,
    pub max_candidates_per_target: usize,
}

pub struct UnifiedSynthesisEngine {
    cegis: CegisLoop,
    scalar_search: SearchSpace,       // existing
    neon_search: NeonSearchSpace,     // NEW
    // gpu_search: GpuSearchSpace,    // FUTURE
    rule_db: RuleDatabase,
}

impl UnifiedSynthesisEngine {
    pub fn synthesize(&mut self, config: UnifiedSynthesisConfig) -> Vec<ProvenCandidate> {
        let mut proven = Vec::new();
        for target in &config.legal_targets {
            let candidates = match target {
                ComputeTarget::CpuScalar => self.enumerate_scalar(&config),
                ComputeTarget::CpuSimd => self.enumerate_neon(&config),
                ComputeTarget::Gpu => vec![], // future
                ComputeTarget::NeuralEngine => vec![], // future
            };
            for candidate in candidates {
                match self.cegis.verify(&config.source_smt, &candidate.smt, &config.source_vars) {
                    CegisResult::Equivalent { proof_hash, .. } => {
                        let cost = config.cost_model.estimate(&candidate, target);
                        proven.push(ProvenCandidate { target, smt: candidate.smt, cost, proof_hash });
                    }
                    _ => {} // discard
                }
            }
        }
        proven.sort_by_key(|c| c.cost);
        proven
    }
}
```

### Gap 3: Cost model integration across targets

**What exists:**
- `cost_model.rs` in llvm2-ir: `AppleSiliconCostModel` with per-`AArch64Opcode` latency/throughput.
  Supports M1 and M4 generations.
- `compute_graph.rs`: `ComputeCost { latency_cycles, throughput_ops_per_kcycle }` per node per target.
- `synthesis.rs`: `SynthesisEngine::estimate_cost()` — naive instruction count (MUL=3, else=1).
- `target_analysis.rs`: `CostConfig` with `gpu_launch_threshold_bytes` and `ane_launch_threshold_bytes`.

**What is missing:**
- `AppleSiliconCostModel` only knows `AArch64Opcode`. It cannot price NEON sequences,
  GPU kernel dispatch, or ANE matrix ops. There is no `CostModel` implementation for
  non-scalar targets.
- `synthesis.rs::estimate_cost()` is completely separate from `AppleSiliconCostModel` —
  they are not connected. Synthesis uses naive instruction counting; the real cost model
  exists in a different crate.
- `compute_graph.rs` populates `ComputeCost` with hardcoded estimates, not by calling
  the cost model.
- No GPU cost model at all (Metal kernel launch overhead, memory bandwidth, etc.).

**Required connecting code:**
```rust
// Extend CostModel trait to handle all targets:
pub trait MultiTargetCostModel: CostModel {
    fn estimate_neon_sequence(&self, instrs: &[NeonOpcode]) -> u64;
    fn estimate_gpu_kernel(&self, data_size_bytes: u64, op_count: u64) -> u64;
    fn estimate_ane_op(&self, data_size_bytes: u64, op_kind: AneOpKind) -> u64;
    fn estimate_transfer(&self, from: ComputeTarget, to: ComputeTarget, bytes: u64) -> u64;
}

// Bridge between synthesis engine and real cost model:
pub fn rank_candidates(
    candidates: &[ProvenCandidate],
    cost_model: &dyn MultiTargetCostModel,
) -> Vec<(usize, u64)> { ... }
```

### Gap 4: Memory proofs integration with main verification pipeline

**What exists:**
- `memory_proofs.rs`: 21 proofs for Load/Store lowering using array theory (QF_ABV).
  Creates SmtExpr using Select/Store/ConstArray. Verifies via `verify_by_evaluation()`.
- `z4_bridge.rs`: `infer_logic()` correctly detects QF_ABV. `generate_smt2_query()`
  produces valid SMT-LIB2 for array expressions.
- `z4_bridge.rs`: The z4 native API backend returns `Err("Array theory not yet supported
  in z4 native API")` for array expressions — falls back to CLI.

**What is missing:**
- Memory proofs are standalone proof obligations verified in isolation. They are NOT
  integrated into the main lowering pipeline (isel.rs -> verify path).
- When the lowering pass (isel.rs) lowers a tMIR Load/Store to an AArch64 LDR/STR,
  it does not generate or check the corresponding memory proof obligation.
- The CEGIS loop does not support array-sorted variables. `ConcreteInput` stores
  `HashMap<String, u64>`, which cannot represent array values.
- The `verify_all_with_z4()` function in z4_bridge.rs runs arithmetic, NZCV, and
  peephole proofs — but NOT memory proofs.

**Required connecting code:**
```rust
// 1. Add memory proofs to verify_all_with_z4():
pub fn verify_all_with_z4(config: &Z4Config) -> Vec<(String, Z4Result)> {
    let mut results = Vec::new();
    // ... existing proofs ...
    for obligation in crate::memory_proofs::all_memory_proofs() {
        let result = verify_with_z4(&obligation, config);
        results.push((obligation.name.clone(), result));
    }
    results
}

// 2. Extend ConcreteInput for array values:
pub enum ConcreteValue {
    Bv(u64),
    Array(HashMap<u64, u64>, u64), // entries + default
}

// 3. Wire memory proof generation into the lowering pipeline (isel.rs)
```

### Gap 5: NEON semantics -> synthesis/CEGIS integration

**What exists:**
- `neon_semantics.rs`: 11 NEON operations encoded as SmtExpr (ADD, SUB, MUL, NEG,
  AND, ORR, EOR, SHL, SSHR, ABS, MOVI). Uses lane decomposition via `map_lanes_*`.
- `cegis.rs`: Handles SmtExpr verification for any bitvector expression.
- `synthesis.rs`: Only enumerates scalar AArch64 opcodes.

**What is missing:**
- No NEON search space. `SearchSpace` does not enumerate NEON instruction patterns.
  A `NeonSearchSpace` would enumerate `(NEON_opcode, VectorArrangement)` pairs.
- NEON semantics produce 128-bit (or 64-bit) SmtExpr. The CEGIS loop's `ConcreteInput`
  stores u64, which cannot hold 128-bit vector values.
- No bridge from `neon_semantics.rs` functions to `RuleCandidate` / `ProvenRule` types.
- The `ProvenRuleDb` and `RuleDatabase` have no way to annotate a rule with its target
  (scalar vs NEON vs GPU).

**Required connecting code:**
```rust
// New: neon_synthesis.rs
pub struct NeonSearchSpace;

impl NeonSearchSpace {
    pub fn enumerate(config: &NeonSearchConfig) -> Vec<NeonCandidate> {
        // For each arrangement (4S, 2D, 8H, etc.):
        //   For each NEON opcode (ADD, SUB, MUL, SHL, ...):
        //     Generate candidates as SmtExpr via neon_semantics::encode_neon_*
    }
}

// Extend ProvenRule with target annotation:
pub struct ProvenRule {
    // ... existing fields ...
    pub target: ComputeTarget,  // NEW
}

// Extend ConcreteInput for 128-bit vectors:
// Use u128 or [u64; 2] for NEON vector values.
```

### Gap 6: AI rule_discovery.rs -> multi-target proposals

**What exists:**
- `rule_discovery.rs`: `RuleDiscovery` accepts `RuleProposal(pattern: SmtExpr, replacement: SmtExpr)`
  and verifies via CEGIS. Stores proven rules in `RuleDatabase`.

**What is missing:**
- `RuleProposal` has no field for target. An AI proposing a NEON rule looks identical
  to an AI proposing a scalar rule. The discovery engine cannot route the proposal to
  the correct search space or cost model.
- `RuleDatabase` has no target-aware querying. Cannot ask "give me all proven NEON rules."
- No way for the AI to propose cross-target rules (e.g., "this scalar sequence is equivalent
  to this NEON sequence"). Such proposals require different variable widths on source vs target.

**Required connecting code:**
```rust
pub struct RuleProposal {
    // ... existing fields ...
    pub source_target: Option<ComputeTarget>,      // NEW
    pub replacement_target: Option<ComputeTarget>,  // NEW
}

impl RuleDatabase {
    pub fn query_by_target(&self, target: ComputeTarget) -> Vec<&DiscoveredRule> { ... }
    pub fn cross_target_rules(&self) -> Vec<&DiscoveredRule> { ... }
}
```

---

## 3. Module Interface Specifications

### 3.1 compute_graph.rs -> target_analysis.rs

**Direction:** compute_graph produces `ComputeNode`; target_analysis consumes `SubgraphDescriptor`.

**Interface contract:**

```
Input: ComputeNode {
    id: ComputeNodeId,
    instructions: Vec<TmirInstId>,
    kind: NodeKind,
    data_size_bytes: u64,
    produced_values: Vec<ValueId>,
    consumed_values: Vec<ValueId>,
}

Transform: build_subgraph_descriptor(node, tmir_module, proof_context) -> SubgraphDescriptor

Output: SubgraphDescriptor {
    id: SubgraphId(node.id.0),
    values: [resolved from produced_values + consumed_values],
    value_types: [resolved from tmir_module type info],
    subgraph_proofs: [extracted from proof_context for the node's instructions],
    data_size_bytes: node.data_size_bytes,
}
```

**Key resolution:** `ComputeNodeId(u32)` maps 1:1 to `SubgraphId(u32)`. The `id.0` value is
shared. This is safe because both are scoped to a single function's graph.

### 3.2 target_analysis.rs -> unified CEGIS search space

**Direction:** target_analysis produces `TargetLegality`; synthesis consumes legal target set.

**Interface contract:**

```
Input: TargetLegality {
    subgraph: SubgraphId,
    judgments: HashMap<ComputeTarget, TargetJudgment>,
    parallel_reduction_legal: bool,
    data_size_bytes: u64,
}

Filter: legal_targets = legality.legal_targets()
        // Always includes CpuScalar, may include CpuSimd, Gpu, NeuralEngine

Constrain synthesis:
    for target in legal_targets:
        candidates = enumerate_candidates(target, source_smt)
        // Only enumerate for legal targets — skip GPU if not legal
```

**Key invariant:** `CpuScalar` is always legal. The synthesis engine MUST always produce
at least one scalar candidate as fallback.

### 3.3 Cost model ranking across targets

**Direction:** Cost model takes `(candidate, target)` pairs; returns ranked list.

**Interface contract:**

```
Input: Vec<ProvenCandidate> from synthesis
    Each has: target: ComputeTarget, smt: SmtExpr, proof_hash: u64

Ranking:
    For CpuScalar candidates:
        cost = AppleSiliconCostModel::latency(opcode) summed over instruction sequence
    For CpuSimd candidates:
        cost = AppleSiliconCostModel + NEON-specific latency data
    For Gpu candidates:
        cost = gpu_launch_overhead + data_transfer_cost + kernel_compute_cost
    For NeuralEngine candidates:
        cost = ane_launch_overhead + data_transfer_cost + ane_compute_cost

Output: candidates sorted by total_cost ascending
    Winner = candidates[0]
    Fallback = best CpuScalar candidate (always exists)
```

**Key principle:** Transfer costs between targets are critical. A GPU kernel that computes
in 50 cycles but needs 200 cycles of data transfer loses to a NEON sequence at 150 cycles.
The cost model MUST include transfer overhead via `estimate_transfer_cost()` from compute_graph.rs.

### 3.4 Memory proofs in the verification pipeline

**Direction:** Memory proofs integrate into both batch verification and per-lowering verification.

**Interface contract:**

```
Batch path (verify_all_with_z4):
    collect all_memory_proofs() -> Vec<ProofObligation>
    for each: verify_with_z4(obligation, config) -> Z4Result
    aggregate into VerificationSummary

Per-lowering path (during isel.rs):
    When lowering tMIR Load(addr, ty) -> AArch64 LDR:
        tmir_smt = encode_tmir_load(memory, addr, ty)
        aarch64_smt = encode_aarch64_ldr(memory, effective_addr, ty)
        obligation = ProofObligation { tmir_expr: tmir_smt, aarch64_expr: aarch64_smt, ... }
        verify_by_evaluation(&obligation) OR verify_with_z4(&obligation, config)
```

**Theory requirement:** Memory proofs use QF_ABV (arrays of bitvectors). The z4_bridge
correctly infers this logic. The CLI fallback (z3 subprocess) handles QF_ABV natively.
The z4 native API does NOT support arrays yet — must use CLI fallback for memory proofs.

---

## 4. Implementation Order

Dependencies form a DAG. The implementation order respects this DAG:

### Phase A: Foundation bridges (no new theory needed)

**A1. compute_graph -> target_analysis bridge** (Gap 1)
- Implement `build_subgraph_descriptor()` and `annotate_graph_with_proofs()`.
- Wire `ProofAnalyzer::analyze()` into graph construction.
- Dependencies: none (both modules exist).
- Effort: ~200 LOC. Pure wiring.

**A2. Memory proofs in verify_all_with_z4** (Gap 4, batch path)
- Add `all_memory_proofs()` collector function in memory_proofs.rs.
- Include in `verify_all_with_z4()`.
- Dependencies: none (memory proofs already produce ProofObligation).
- Effort: ~50 LOC.

**A3. ProvenRule/RuleDatabase target annotation** (Gap 6, partial)
- Add `target: ComputeTarget` field to `ProvenRule` and `DiscoveredRule`.
- Add `query_by_target()` to `RuleDatabase`.
- Dependencies: none.
- Effort: ~100 LOC.

### Phase B: NEON synthesis (needs 128-bit concrete eval)

**B1. ConcreteInput u128 support** (Gaps 4, 5)
- Extend `ConcreteInput` values from `HashMap<String, u64>` to support u128.
- Option: use `ConcreteValue` enum { Bv64(u64), Bv128(u128) }.
- Or: store all values as u128, cast down for narrower widths.
- Dependencies: A1 complete (for integration testing).
- Effort: ~150 LOC. Touches cegis.rs, ConcreteInput, smt.rs eval.

**B2. NEON search space enumeration** (Gap 5)
- New module `neon_synthesis.rs` in llvm2-verify.
- Enumerate NEON instruction patterns using `neon_semantics::encode_neon_*`.
- Produce `SmtExpr` candidates for each `(opcode, arrangement)` pair.
- Dependencies: B1 complete (for 128-bit CEGIS verification).
- Effort: ~400 LOC.

**B3. Wire NEON search into RuleDiscovery** (Gap 5, Gap 6)
- `RuleProposal` gets `source_target` / `replacement_target` fields.
- NEON candidates flow through existing CEGIS + RuleDatabase pipeline.
- Dependencies: B2, A3.
- Effort: ~100 LOC.

### Phase C: Unified synthesis engine (orchestration layer)

**C1. UnifiedSynthesisEngine** (Gap 2)
- New module `unified_synthesis.rs` in llvm2-verify.
- Orchestrates: for each (subgraph, legal_target), enumerate + CEGIS + rank.
- Depends on scalar SearchSpace, NEON NeonSearchSpace, CegisLoop, CostModel.
- Dependencies: B2, A1 complete.
- Effort: ~500 LOC.

**C2. MultiTargetCostModel trait** (Gap 3)
- Extend `CostModel` trait with NEON/GPU/ANE cost estimation methods.
- Implement for `AppleSiliconCostModel` — NEON using existing Firestorm data,
  GPU/ANE as stubs returning configurable estimates.
- Bridge `synthesis.rs::estimate_cost()` to use `AppleSiliconCostModel`.
- Dependencies: none (can be done in parallel with B-phase).
- Effort: ~300 LOC.

**C3. DispatchPlanner** (Gap between synthesis and lowering)
- Given `UnifiedSynthesisEngine` output + `ComputeGraph`, find optimal partition.
- Greedy or ILP approach: minimize total cost = compute + transfer.
- Use existing `ComputeGraph::partition_cost()` for evaluation.
- Dependencies: C1, C2 complete.
- Effort: ~400 LOC.

### Phase D: Pipeline wiring (end-to-end)

**D1. Memory proofs in lowering pipeline** (Gap 4, per-lowering path)
- When isel.rs lowers Load/Store, generate proof obligation.
- Call `verify_by_evaluation()` or optionally `verify_with_z4()`.
- Dependencies: A2.
- Effort: ~200 LOC in isel.rs + adapter.

**D2. End-to-end integration test**
- Test: tMIR module -> graph -> analysis -> synthesis -> dispatch -> lowering -> verify.
- Initially with tMIR stubs; then with real tMIR when available.
- Dependencies: C3, D1 complete.
- Effort: ~300 LOC of test infrastructure.

### Dependency DAG

```
A1 ─────────┬─> B1 ──> B2 ──> B3
A2 ──> D1   |                   |
A3 ─────────┤                   |
            └─> C1 <──── B2    |
C2 ────────────> C1            |
C1 + C2 ──────> C3            |
C3 + D1 ──────> D2            |
```

### Estimated total: ~2800 LOC across 10 work items

---

## 5. Concrete GitHub Issues

### Issue 1: compute_graph -> target_analysis bridge (Phase A1)

**Title:** Wire ProofAnalyzer into ComputeGraph construction
**Priority:** P1
**Why:** Without this bridge, target analysis is disconnected from graph construction.
The graph builder hardcodes legal_targets using heuristics instead of proof-guided
analysis, defeating the core insight of the unified architecture.

### Issue 2: Unified multi-target synthesis engine (Phase C1)

**Title:** Implement UnifiedSynthesisEngine for cross-target CEGIS
**Priority:** P1
**Why:** The existing synthesis engine only searches scalar AArch64. The unified
architecture requires searching across all legal targets simultaneously. This is
the central component that makes the unified solver real.

### Issue 3: NEON search space for synthesis (Phase B2)

**Title:** Implement NEON candidate enumeration for synthesis
**Priority:** P2
**Why:** NEON is the first non-scalar target. The neon_semantics module encodes 11
NEON operations as SmtExpr, but no synthesis engine enumerates NEON candidates.
This blocks multi-target synthesis for the most common non-scalar optimization target.

### Issue 4: Multi-target cost model (Phase C2)

**Title:** Extend CostModel to support NEON, GPU, and ANE cost estimation
**Priority:** P2
**Why:** The current cost model only handles scalar AArch64 opcodes. Without
multi-target cost estimation, the synthesis engine cannot rank candidates across
targets. NEON data is available from the existing Firestorm microarchitecture
research.

### Issue 5: Memory proofs pipeline integration (Phase A2 + D1)

**Title:** Integrate memory proofs into verify_all and lowering pipeline
**Priority:** P2
**Why:** 21 memory proofs exist but run in isolation. They are not included in
verify_all_with_z4() and are not generated during Load/Store lowering. This
means memory correctness is tested but not enforced in the compilation pipeline.

---

## 6. Risks and Open Questions

### Risk: z4 array theory support

The z4 native Rust API does not support array theory (QF_ABV). Memory proofs and
GPU kernel proofs require arrays. The CLI fallback (z3 subprocess) handles this, but
subprocess overhead is significant for per-instruction verification during compilation.

**Mitigation:** File upstream issue for z4 array support. In the meantime, use the
CLI fallback for array proofs and the native API for BV-only proofs.

### Risk: 128-bit concrete evaluation for NEON CEGIS

The CEGIS loop stores counterexamples as `HashMap<String, u64>`. NEON semantics
produce 128-bit vectors. Extending to u128 is straightforward but touches the
hot path of concrete evaluation.

**Mitigation:** Use `u128` storage with `as u64` downcast for scalar expressions.
The SmtExpr evaluator already uses u64 internally — needs extension to u128 for
`Concat` results wider than 64 bits.

### Risk: GPU/ANE encoding complexity

GPU kernel semantics (parallel map/reduce over arrays) and ANE matrix operations
are significantly more complex than scalar/NEON bitvector encoding. The current
SmtExpr AST supports arrays and FP, but encoding a full Metal kernel dispatch
requires quantifiers (bounded forall) which the AST does not yet support.

**Mitigation:** Start with NEON (bitvector, no quantifiers needed). GPU/ANE
encoding is Phase 2+ work. File separate design doc for GPU semantic encoding
when NEON integration is complete.

### Open question: Synthesis enumeration explosion

The unified search space `(all targets) x (all sequences) x (all arrangements)` is
orders of magnitude larger than the current scalar-only search. Even with proof-guided
pruning, enumeration time may be prohibitive.

**Approach:** Start with CEGIS-driven AI proposals (rule_discovery.rs) rather than
blind enumeration. AI proposes NEON candidates; solver verifies. Enumeration-based
synthesis reserved for scalar peepholes where the search space is manageable.

---

## References

- `designs/2026-04-13-unified-solver-architecture.md` — Master design
- `designs/2026-04-14-z4-integration-guide.md` — z4 API reference
- `designs/2026-04-14-cost-model-calibration.md` — Apple Silicon microarch data
- `designs/2026-04-13-z4-theory-extensions.md` — z4 theory capabilities
- Sasnauskas et al., "Souper: A Synthesizing Superoptimizer" (arXiv:1711.04422)
- Lopes et al., "Alive2: Bounded Translation Validation for LLVM" (PLDI 2021)
- ARM Architecture Reference Manual (DDI 0487), Sections C6-C7
- Dougall Johnson, "Apple M1 Firestorm Microarchitecture"
