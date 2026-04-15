# Cross-Target Dispatch Plan Verification

**Date:** 2026-04-14
**Status:** Draft
**Issue:** #143
**Author:** Andrew Yates <ayates@dropbox.com>

---

## Implementation Status (as of 2026-04-15)

**Overall: Dispatch plan generation and target analysis are implemented. Formal verification of dispatch correctness uses mock evaluation.**

| Component | Status | Details |
|-----------|--------|---------|
| **Dispatch plan generation** (`dispatch.rs`) | IMPLEMENTED | 2.8K LOC. Target assignment, data transfer, synchronization. |
| **Target analysis** (`target_analysis.rs`) | IMPLEMENTED | 1.2K LOC. Proof-guided legality analysis. |
| **Dispatch correctness proofs** | MOCK ONLY | No z4-backed formal proof that dispatching preserves semantics. |
| **Data race freedom proofs** | NOT IMPLEMENTED | No formal synchronization correctness verification. |

---

## Problem

LLVM2's heterogeneous compute pipeline (`llvm2-lower/src/compute_graph.rs`
and `llvm2-lower/src/target_analysis.rs`) dispatches computation nodes to
different targets (CPU scalar, NEON SIMD, GPU, Neural Engine) based on
proof-guided analysis. The dispatch decisions determine program correctness
and performance, but there is currently no formal verification that:

1. Every node has at least one legal target (liveness)
2. Data transfers between targets are correctly placed (completeness)
3. Synchronization between targets is sufficient (ordering)
4. The dispatch respects tMIR proof annotations (soundness)

This document defines verification obligations for the dispatch pipeline and
proposes how to prove each property using z4 and testing.

---

## 1. Architecture Overview

### 1.1 Dispatch Pipeline

```text
tMIR Module
    |
    v
GraphBuilder::build_from_module()   [compute_graph.rs:803]
    |  - Groups instructions into ComputeNodes
    |  - Detects patterns (Scalar, DataParallel, MatrixHeavy)
    |  - Runs ProofAnalyzer on each node
    v
ComputeGraph { nodes, edges }
    |
    v
ProofAnalyzer::analyze()            [target_analysis.rs]
    |  - Checks Pure, InBounds, ValidBorrow proofs
    |  - Determines legal targets per node
    |  - Checks Associative+Commutative for parallel reduction
    v
TargetLegality per node
    |
    v
target_recommendations()            [compute_graph.rs:348]
    |  - Picks cheapest legal target per node
    v
Assignment: HashMap<ComputeNodeId, ComputeTarget>
    |
    v
partition_cost()                    [compute_graph.rs:256]
    |  - Sums compute costs + transfer costs
    v
Total cost (u64)
```

### 1.2 Key Data Structures

| Type | Source | Role |
|------|--------|------|
| `ComputeGraph` | `compute_graph.rs:217` | DAG of computation nodes + data edges |
| `ComputeNode` | `compute_graph.rs:178` | Group of tMIR instructions with costs and legality |
| `DataEdge` | `compute_graph.rs:203` | Data dependency between nodes with transfer cost |
| `ComputeTarget` | `target_analysis.rs:73` | CpuScalar, CpuSimd, Gpu, NeuralEngine |
| `TargetLegality` | `target_analysis.rs` | Per-node legal targets with justifications |
| `ProofAnalyzer` | `target_analysis.rs` | Proof-guided target pruning engine |
| `TargetProofContext` | `target_analysis.rs` | Module-level proof annotations |

---

## 2. Verification Obligations

### 2.1 Property 1: CPU Fallback Liveness

**Statement**: For every computation node `n` in a well-formed `ComputeGraph`,
`n.legal_targets` is non-empty and contains `CpuScalar`.

**Why it matters**: If a node has no legal targets, the dispatch has no valid
assignment and compilation fails. If `CpuScalar` is missing, the fallback
path does not exist, which means a missing proof annotation makes the program
uncompilable rather than merely unoptimized.

**Current implementation**: The `ProofAnalyzer::analyze()` function always
includes `CpuScalar` in the legal target set (see `target_analysis.rs`
pruning algorithm: CPU is the "always legal" baseline). NEON (`CpuSimd`) is
also always legal for scalar computations.

**Verification approach**:

1. **Invariant enforcement in `ProofAnalyzer`**:
   ```rust
   fn analyze(&self, desc: &SubgraphDescriptor, ctx: &TargetProofContext) -> TargetLegality {
       let mut legal = vec![ComputeTarget::CpuScalar, ComputeTarget::CpuSimd];
       // ... add GPU/ANE based on proofs ...
       assert!(!legal.is_empty(), "CPU fallback must always be present");
       // ...
   }
   ```
   This is a runtime assertion. For formal verification:

2. **z4 proof obligation**:
   ```
   forall desc: SubgraphDescriptor, ctx: TargetProofContext.
     let result = analyze(desc, ctx) in
       CpuScalar in result.legal_targets
   ```
   Encoding: model `analyze` as a function over bitvector-encoded proof
   sets. The `legal_targets` output is a 4-bit vector (one bit per target).
   Prove that bit 0 (CpuScalar) is always set.

3. **Property-based testing** (immediate, no z4 required):
   Generate random `SubgraphDescriptor` values (varying proof combinations,
   data sizes, value types) and assert `CpuScalar` is always present in
   the result. Use the existing test infrastructure in `compute_graph.rs`
   (see `test_without_proofs_cpu_only`, `test_scalar_module_produces_scalar_node`).

### 2.2 Property 2: Data Transfer Completeness

**Statement**: For every edge `(u, v)` in the compute graph where
`assignment[u] != assignment[v]`, a data transfer is accounted for in the
cost model, and the transfer is physically realizable.

**Why it matters**: If a data edge crosses a target boundary (e.g., CPU to GPU)
but no transfer is inserted, the consumer node reads stale or invalid data.

**Current implementation**: `partition_cost()` (`compute_graph.rs:256-287`)
iterates all edges and adds transfer costs when source and destination targets
differ. The `estimate_transfer_cost()` function (`compute_graph.rs:522-577`)
computes a non-zero cost for all cross-target pairs (except CPU<->SIMD which
share memory).

**Verification approach**:

1. **Exhaustive target-pair coverage**:
   Prove that `estimate_transfer_cost(bytes, from, to)` returns a well-defined
   cost for all 16 (from, to) combinations of `ComputeTarget`:

   ```
   forall from, to in ComputeTarget::ALL.
     estimate_transfer_cost(1, from, to) is defined
   ```

   This is already covered by the exhaustive match in `estimate_transfer_cost`
   (which has a match arm for every pair), but should be tested:

   ```rust
   #[test]
   fn test_all_target_pairs_have_defined_cost() {
       for &from in &ComputeTarget::ALL {
           for &to in &ComputeTarget::ALL {
               let cost = estimate_transfer_cost(100, from, to);
               // All pairs must return a defined cost (no panic)
               if from == to {
                   assert_eq!(cost.total_cycles, 0);
               }
           }
       }
   }
   ```

2. **Transfer symmetry**: For correctness, transfer cost should be symmetric
   (transferring CPU->GPU costs the same as GPU->CPU for the same data size).
   This is currently true by inspection of the match arms. Formalize:

   ```
   forall from, to in ComputeTarget::ALL, bytes: u64.
     estimate_transfer_cost(bytes, from, to).total_cycles
       == estimate_transfer_cost(bytes, to, from).total_cycles
   ```

3. **Edge coverage proof**: Prove that `partition_cost` examines every edge:
   ```
   forall graph: ComputeGraph, assignment: HashMap.
     partition_cost(graph, assignment).is_some() implies
       forall edge in graph.edges.
         if assignment[edge.from] != assignment[edge.to] then
           transfer_cost(edge) > 0
   ```
   This follows from the structure of the `partition_cost` loop, which
   iterates `for edge in &self.edges`. A z4 encoding would model the graph
   as a bounded list of edges and verify the loop invariant.

### 2.3 Property 3: Synchronization Sufficiency

**Statement**: The dispatch plan respects data dependency ordering: a consumer
node does not execute before all its producer nodes have completed and their
data has been transferred.

**Why it matters**: On heterogeneous systems, different targets may execute
asynchronously. If node B on GPU reads data produced by node A on CPU, the
GPU kernel must wait for the CPU computation and data transfer to complete.

**Current implementation**: The `ComputeGraph` is a DAG (directed acyclic
graph) with edges representing data dependencies. The execution model is
implicitly topological: a node can execute only after all its predecessor
nodes (connected by incoming edges) have completed.

However, the current implementation does NOT generate explicit synchronization
primitives (barriers, fences, semaphores). This is acceptable at the current
abstraction level because:

- The dispatch plan is a **logical schedule**, not a runtime schedule.
- The actual execution backend (Metal command buffer, CoreML inference) is
  responsible for enforcing data dependencies.
- CPU<->SIMD transitions need no synchronization (same core).

**What needs verification**:

1. **DAG property**: The compute graph must be acyclic. A cycle would mean
   a node depends on its own output, which is unrealizable.

   ```
   forall graph: ComputeGraph.
     is_dag(graph.nodes, graph.edges) == true
   ```

   Verification: topological sort of the graph must succeed. If it does not,
   the graph builder has a bug.

   ```rust
   fn verify_dag(graph: &ComputeGraph) -> bool {
       // Kahn's algorithm: remove nodes with no incoming edges until empty
       let mut in_degree: HashMap<ComputeNodeId, usize> = HashMap::new();
       for node in &graph.nodes {
           in_degree.entry(node.id).or_insert(0);
       }
       for edge in &graph.edges {
           *in_degree.entry(edge.to).or_insert(0) += 1;
       }
       let mut queue: Vec<ComputeNodeId> = in_degree.iter()
           .filter(|(_, &deg)| deg == 0)
           .map(|(&id, _)| id)
           .collect();
       let mut visited = 0;
       while let Some(node_id) = queue.pop() {
           visited += 1;
           for edge in graph.outgoing_edges(node_id) {
               let deg = in_degree.get_mut(&edge.to).unwrap();
               *deg -= 1;
               if *deg == 0 {
                   queue.push(edge.to);
               }
           }
       }
       visited == graph.nodes.len()
   }
   ```

2. **Dependency completeness**: Every value consumed by a node must have a
   corresponding producer node connected by an edge.

   ```
   forall node in graph.nodes, vid in node.consumed_values.
     exists producer in graph.nodes, edge in graph.edges.
       vid in producer.produced_values
       && edge.from == producer.id
       && edge.to == node.id
   ```

   Exception: function parameters (block entry values) have no producer within
   the graph -- they are external inputs.

3. **GPU/ANE synchronization primitives** (future work): When the dispatch
   plan assigns nodes to GPU or ANE, the code generator must insert:
   - Metal command buffer commit + waitUntilCompleted for GPU->CPU transfers
   - CoreML prediction waitForCompletion for ANE->CPU transfers
   - Metal shared event signal/wait for GPU->GPU dependencies (future)

   This is out of scope for the current dispatch verification but should be
   tracked as a requirement for the heterogeneous codegen backend.

### 2.4 Property 4: Proof Annotation Soundness

**Statement**: The dispatch respects tMIR proof annotations: a node is
assigned to a target only if the required proofs are present in the
`TargetProofContext`.

**Why it matters**: If the dispatch assigns a node to GPU without the required
`Pure` + `InBounds` + `ValidBorrow` proofs, the GPU kernel may:
- Read/write aliased memory (violating `ValidBorrow`)
- Access out-of-bounds memory (violating `InBounds`)
- Have observable side effects reordered (violating `Pure`)

**Current implementation**: The `ProofAnalyzer::analyze()` function
(`target_analysis.rs`) implements the pruning algorithm from the design doc:

```
Pure?           -> NO: CPU only
Array data?     -> NO: CPU/SIMD only
InBounds+Valid? -> NO: CPU/SIMD only
                -> YES: GPU/ANE legal
Assoc+Commut?   -> NO: GPU sequential only
                -> YES: GPU parallel legal
Data size?      -> Small: SIMD
                -> Large: GPU/ANE
```

**Verification approach**:

1. **Proof requirement matrix**: Define a formal table of which proofs are
   required for each target:

   | Target | Required Proofs |
   |--------|----------------|
   | CpuScalar | none |
   | CpuSimd | none |
   | Gpu | Pure AND InBounds AND ValidBorrow (for all array values) |
   | NeuralEngine | Pure AND InBounds AND ValidBorrow AND sufficient data size |

2. **z4 encoding of the pruning algorithm**:

   Model the proof context as a bitvector where each bit represents the
   presence of a specific proof type:
   ```
   bit 0: Pure
   bit 1: InBounds (exists for at least one array value)
   bit 2: ValidBorrow (exists for at least one array value)
   bit 3: Associative
   bit 4: Commutative
   bit 5: has_array_data
   bit 6: data_size >= GPU_THRESHOLD
   ```

   Model the output `legal_targets` as a 4-bit vector. Encode the pruning
   algorithm as a bitvector function and prove:

   ```
   forall proofs: bv7.
     let legal = prune(proofs) in
       // GPU legal implies Pure AND InBounds AND ValidBorrow AND has_array
       (legal & GPU_BIT != 0) implies (proofs & 0b0100111 == 0b0100111)
       // ANE legal implies GPU legal AND sufficient data
       (legal & ANE_BIT != 0) implies (legal & GPU_BIT != 0) AND (proofs & 0b1000000 != 0)
       // CPU always legal
       (legal & CPU_SCALAR_BIT != 0) == true
       (legal & CPU_SIMD_BIT != 0) == true
   ```

3. **Property-based testing**: Generate random proof contexts and verify that
   the output satisfies the implication constraints above. The existing tests
   in `compute_graph.rs` partially cover this (see `test_without_proofs_cpu_only`,
   `test_with_proof_context_unlocks_gpu`), but should be expanded with
   randomized proof combinations.

---

## 3. Integration with tMIR Proof Annotations

### 3.1 How Proofs Flow

```text
tMIR Source (tRust/tSwift/tC)
    |
    v
tMIR Module with proof annotations
    |  - Per-value: InBounds, ValidBorrow, NoOverflow, NonNull
    |  - Per-subgraph: Pure, Associative, Commutative, Deterministic
    v
adapter.rs: translate_module()      [llvm2-lower/src/adapter.rs]
    |  - Extracts ProofContext from tMIR module
    |  - Maps tMIR ValueIds to internal Values
    v
ProofContext { value_proofs, ... }
    |
    v
TargetProofContext::new(proof_ctx)  [target_analysis.rs]
    |  - Wraps ProofContext with subgraph-level proofs
    |  - add_subgraph_proof(SubgraphId, SubgraphProof)
    v
GraphBuilder::build_from_module(module)
    |  - For each node: SubgraphDescriptor <- proofs from context
    |  - ProofAnalyzer::analyze(desc, ctx)
    v
Per-node TargetLegality
```

### 3.2 Proof Preservation Verification

The critical verification obligation is that proofs from tMIR are faithfully
propagated through the dispatch pipeline:

1. **No proof amplification**: The dispatch must not claim a proof exists that
   tMIR did not provide. This would be a soundness violation.

   ```
   forall node in graph.nodes.
     node.target_legality.proofs is subset of ctx.proofs_for(node)
   ```

2. **No proof loss**: All proofs from tMIR that are relevant to a node should
   be available to the `ProofAnalyzer`. Proof loss would cause unnecessary
   pessimization (not unsoundness, but lost optimization opportunity).

   The current implementation propagates proofs via
   `proof_ctx.subgraph_proofs_for(subgraph_id)` in `build_nodes_for_block`
   (`compute_graph.rs:1021`). This is correct as long as the `SubgraphId`
   mapping from `ComputeNodeId` is consistent.

3. **Subgraph-to-node mapping consistency**: The `SubgraphId(node_id.0)`
   mapping (`compute_graph.rs:953`) must be stable across graph construction.
   If node IDs are reassigned (e.g., after graph optimization), the subgraph
   proof mapping would break.

   Verification: assert that `SubgraphId` values in the `TargetProofContext`
   correspond to actual `ComputeNodeId` values in the graph.

### 3.3 Annotate-After-Construction Soundness

The `annotate_with_proofs()` method (`compute_graph.rs:416-508`) allows
re-analyzing nodes after initial construction. This must satisfy:

- New proofs may only expand the legal target set (monotonicity)
- Removing a proof annotation must cause the corresponding target to become
  illegal

Formal statement:
```
forall graph, ctx1, ctx2: TargetProofContext.
  ctx1 is subset of ctx2 implies
    forall node.
      graph.annotate(ctx1).legal_targets is subset of graph.annotate(ctx2).legal_targets
```

This is the monotonicity property of the pruning algorithm: more proofs
means more targets, never fewer.

---

## 4. Proposed Verification Plan

### Phase 1: Property-Based Testing (Immediate)

Add to `compute_graph.rs` tests:

1. `test_all_target_pairs_have_defined_cost` -- exhaustive coverage
2. `test_cpu_fallback_always_present` -- randomized SubgraphDescriptor
3. `test_dag_property` -- verify topological sort succeeds
4. `test_proof_soundness_gpu_requires_pure` -- remove Pure, verify GPU illegal
5. `test_proof_soundness_gpu_requires_inbounds` -- remove InBounds, verify
6. `test_proof_monotonicity` -- more proofs never reduces legal targets
7. `test_edge_covers_all_cross_node_values` -- dependency completeness

### Phase 2: z4 Formal Proofs (Follow-up)

1. **CPU fallback liveness**: Encode `ProofAnalyzer::analyze` as a z4
   bitvector function. Prove `CpuScalar` always in output.

2. **Pruning algorithm soundness**: Encode the proof requirement matrix as
   implications. Prove GPU/ANE targets are only present when required proofs
   are present.

3. **Transfer cost well-definedness**: Prove `estimate_transfer_cost` returns
   a value for all input combinations (no panics, no overflow).

4. **Monotonicity**: Prove that adding proofs to the context can only expand
   the legal target set.

### Phase 3: Integration with Unified Synthesis (Future)

The `unified_synthesis.rs` module searches for lowerings across scalar and
NEON targets simultaneously. Dispatch verification should integrate with
synthesis to prove:

- If synthesis finds a NEON lowering for a node, the node's legal targets
  must include `CpuSimd`
- If synthesis finds a scalar lowering, the node's legal targets must include
  `CpuScalar`
- The chosen lowering's semantics must match the source tMIR semantics
  (this is already verified by CEGIS in `unified_synthesis.rs`)

---

## 5. Risk Analysis

| Risk | Impact | Mitigation |
|------|--------|------------|
| Graph builder produces cycles | Compilation hang or incorrect schedule | DAG property check in `build_from_module` |
| Missing edge for cross-block dependency | Silent data corruption | Dependency completeness test + second-pass branch resolution |
| Proof amplification in annotate_with_proofs | Unsound GPU dispatch | Monotonicity property test |
| SubgraphId/ComputeNodeId mapping drift | Wrong proofs applied to wrong node | Assertion that IDs are consistent |
| Transfer cost underestimate | Performance regression (not correctness) | Calibration against measured hardware |
| Transfer cost overflow (u64 saturation) | Wrong cost comparison | Use `saturating_add` (already done) |

---

## References

- `crates/llvm2-lower/src/compute_graph.rs` -- ComputeGraph, GraphBuilder, cost model
- `crates/llvm2-lower/src/target_analysis.rs` -- ProofAnalyzer, TargetLegality, ComputeTarget
- `crates/llvm2-verify/src/unified_synthesis.rs` -- Cross-target CEGIS synthesis
- `designs/2026-04-13-heterogeneous-compute.md` -- Heterogeneous compute design
- `designs/2026-04-13-unified-solver-architecture.md` -- Unified solver architecture
- `designs/2026-04-13-verification-architecture.md` -- z4 verification architecture
