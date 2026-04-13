# Heterogeneous Compute Allocation

**Author:** Andrew Yates
**Date:** 2026-04-13
**Status:** Design
**Epic:** Automatic heterogeneous compute allocation

---

## Introduction

Today's compilers allocate **registers** and **memory** automatically — the programmer writes `let x = a + b` and the compiler decides whether `x` lives in X0, X3, or on the stack. Nobody manually assigns registers anymore (outside of inline assembly).

But when it comes to **compute resources**, programmers are still doing manual allocation. Want to use the GPU? Write Metal/CUDA shaders. Want the Neural Engine? Use CoreML. Want SIMD? Write intrinsics or hope the auto-vectorizer fires. This is 1960s-era register allocation applied to 2026 hardware.

**LLVM2's insight:** Compute resource allocation is the same problem as register allocation, just at a higher level. The compiler should decide which computations run on which hardware — CPU cores, GPU, Neural Engine, SIMD units — the same way it decides which values live in which registers.

The programmer writes pure math. The compiler maps it to the best hardware. With proofs.

---

## The Analogy

| Concept | Register Allocation | Compute Allocation |
|---------|--------------------|--------------------|
| Resources | Registers (X0-X30) | Compute units (CPU, GPU, ANE, SIMD) |
| Values | Virtual registers | Computation subgraphs |
| Constraints | Live ranges, calling conventions | Data transfer cost, kernel launch overhead |
| Spilling | Register → stack (slow) | GPU → CPU fallback (slow) |
| Cost model | Instruction latency | Compute + transfer time |
| Coloring | Graph coloring / linear scan | Partitioning + scheduling |
| Result | Each value in a physical register | Each computation on optimal hardware |

The compiler already solves the hard version of this (register allocation is NP-hard). Compute allocation is the same structure at a coarser granularity.

---

## Target Hardware (Apple Silicon)

Apple M-series chips have **four** compute resources in a single package:

| Resource | Strength | Weakness | Access |
|----------|----------|----------|--------|
| **CPU P-cores** | General-purpose, low latency, complex control flow | Lower throughput for data-parallel | Direct |
| **CPU E-cores** | Energy-efficient, background work | Slower than P-cores | Direct |
| **GPU** | Massive data parallelism, matrix ops, texture | Kernel launch overhead, data transfer | Metal |
| **Neural Engine (ANE)** | 16-bit matrix multiply, convolutions, ~15 TOPS | Fixed function, limited ops, model compilation | CoreML/BNNS |
| **AMX (SIMD)** | Matrix multiply on CPU, no launch overhead | Undocumented Apple-internal ISA | Private API |
| **NEON (SIMD)** | 128-bit vector ops, standard ARM | Limited width vs GPU | Inline |

Today, the programmer manually targets each. This is absurd — the compiler has more information than the programmer about:
- Data sizes and shapes
- Memory layout and transfer costs
- Kernel launch overhead vs computation time
- Which operations the ANE supports
- P-core vs E-core scheduling

---

## Prior Art

### Halide (Separation of Algorithm and Schedule)

Halide separates *what* to compute from *how* to compute it. The algorithm is pure math; the schedule controls tiling, vectorization, parallelism, and compute placement. An autoscheduler searches the schedule space.

**Key insight:** Once you separate algorithm from schedule, automated search over compute placement becomes possible. Halide targets CPU, GPU, and HVX (Hexagon vector) from the same algorithm.

### OpenCL / SYCL / oneAPI (Explicit Heterogeneous)

These give the programmer explicit control over device selection. Better than nothing, but still manual allocation. The programmer must reason about data transfer, kernel granularity, and device capabilities.

### TVM/Ansor (ML Workload Autotuning)

TVM maps tensor computations to CPU/GPU/accelerator with learned cost models. Ansor's evolutionary search explores the space of implementations.

**Limitation:** Domain-specific to tensor/ML workloads. Not general-purpose.

### Tapir/OpenCilk (Parallel CPU)

Cilk extends C with `spawn`/`sync` for fork-join parallelism on CPU cores. The compiler and runtime handle scheduling.

**Key insight:** Parallelism as a compiler concept, not a library. But CPU-only — no GPU or accelerator targeting.

### XLA / IREE (ML Compiler Infrastructure)

XLA compiles HLO (High-Level Operations) to CPU, GPU, and TPU. IREE provides a HAL (Hardware Abstraction Layer) that dispatches work to heterogeneous backends.

**Key insight:** HAL abstraction over heterogeneous hardware is necessary. But these are ML-specific.

---

## LLVM2 Heterogeneous Compute Architecture

### Core Design: Compute Allocation as a Compiler Pass

Compute allocation is a pass in the LLVM2 pipeline, running after tMIR optimization but before target-specific lowering:

```
tMIR → tMIR optimization → [COMPUTE ALLOCATION] → per-target lowering → encoding
                                    │
                            ┌───────┼───────┐
                            ▼       ▼       ▼
                          CPU     GPU     ANE
                         lower   lower   lower
                            │       │       │
                            ▼       ▼       ▼
                        AArch64  Metal   CoreML
                         .o      .metallib  .mlmodel
```

### Computation Graph Analysis

The compiler analyzes the tMIR program as a **computation graph** — a DAG of operations with data dependencies.

```rust
pub struct ComputeGraph {
    /// Nodes are computation subgraphs
    nodes: Vec<ComputeNode>,
    /// Edges are data dependencies (with transfer cost)
    edges: Vec<DataEdge>,
}

pub struct ComputeNode {
    /// The tMIR instructions in this subgraph
    pub instructions: Vec<TmirInstId>,
    /// Estimated cost on each compute target
    pub costs: HashMap<ComputeTarget, ComputeCost>,
    /// Which targets can execute this (legality)
    pub legal_targets: Vec<ComputeTarget>,
}

pub struct DataEdge {
    pub from: ComputeNodeId,
    pub to: ComputeNodeId,
    /// Bytes to transfer if nodes are on different devices
    pub transfer_bytes: u64,
    /// Transfer cost (latency + bandwidth)
    pub transfer_cost: TransferCost,
}

pub enum ComputeTarget {
    CpuPCore,
    CpuECore,
    Gpu,
    NeuralEngine,
    Simd,  // NEON / AMX
}
```

### Cost Model: Multi-Target

```rust
pub trait HeterogeneousCostModel {
    /// Cost of executing a compute node on a target
    fn compute_cost(&self, node: &ComputeNode, target: ComputeTarget) -> ComputeCost;
    
    /// Cost of transferring data between targets
    fn transfer_cost(&self, bytes: u64, from: ComputeTarget, to: ComputeTarget) -> TransferCost;
    
    /// Overhead of launching a kernel on a target
    fn launch_overhead(&self, target: ComputeTarget) -> Duration;
    
    /// Is this computation legal on this target?
    fn is_legal(&self, node: &ComputeNode, target: ComputeTarget) -> bool;
}

pub struct ComputeCost {
    pub latency: Duration,
    pub throughput: f64,       // ops/second
    pub energy: f64,           // picojoules
    pub occupancy: f64,        // fraction of target utilized
}

pub struct TransferCost {
    pub latency: Duration,     // Fixed overhead
    pub bandwidth: f64,        // bytes/second
    pub total: Duration,       // latency + bytes/bandwidth
}
```

### Allocation Algorithm

The allocation problem: given a computation graph with per-target costs and data transfer costs, assign each node to a compute target to minimize total execution time.

This is a **graph partitioning** problem — NP-hard in general, but tractable with good heuristics (same as register allocation).

**Algorithm: Greedy Heterogeneous Allocation**

```
1. Build computation graph from tMIR
2. For each node, compute cost on every legal target
3. Identify data-parallel subgraphs (candidates for GPU/SIMD)
4. Identify matrix-heavy subgraphs (candidates for ANE)
5. Greedy assignment:
   a. Start with everything on CPU
   b. For each subgraph, check: does moving it to GPU/ANE/SIMD
      save more compute time than it costs in data transfer?
   c. If yes, move it. Update transfer costs for neighbors.
6. Refine: merge small GPU kernels to amortize launch overhead
7. Output: partitioned computation graph with target assignments
```

**Phase 2: Solver-Driven Allocation**

Use z4 to find the optimal allocation:
- Encode the graph partitioning as an integer programming problem
- z4 finds the assignment minimizing total cost
- Proven optimal (not just heuristic) for graphs up to ~1000 nodes

### Per-Target Lowering

Once allocated, each partition is lowered to its target:

| Target | Lowering | Output |
|--------|----------|--------|
| CPU P-core | Normal AArch64 ISel + RegAlloc | `.o` (Mach-O) |
| CPU SIMD | AArch64 NEON instructions | `.o` (Mach-O, same file) |
| GPU | Metal Shading Language IR | `.metallib` |
| Neural Engine | CoreML model graph | `.mlmodelc` |

The host program (CPU) contains dispatch code that launches GPU/ANE work and synchronizes.

### tMIR Proof Integration

tMIR proofs enable optimizations traditional heterogeneous compilers can't do:

- **InBounds** proofs → GPU kernels don't need bounds checks
- **NoOverflow** proofs → Use unchecked arithmetic on GPU (faster)
- **ValidBorrow** proofs → Data can be transferred without copying (zero-copy DMA)
- **Pure** annotations → Computation is side-effect-free, safe to move to any target
- **Commutative/Associative** proofs → Safe to reorder for GPU-friendly access patterns

### Synchronization and Data Transfer

The compiler generates synchronization code:

```rust
pub struct DispatchPlan {
    /// Ordered list of compute dispatches
    pub dispatches: Vec<Dispatch>,
    /// Data transfers between dispatches
    pub transfers: Vec<DataTransfer>,
    /// Synchronization barriers
    pub barriers: Vec<Barrier>,
}

pub struct Dispatch {
    pub target: ComputeTarget,
    pub computation: ComputeNodeId,
    pub inputs: Vec<BufferId>,
    pub outputs: Vec<BufferId>,
}

pub struct DataTransfer {
    pub from_target: ComputeTarget,
    pub to_target: ComputeTarget,
    pub buffer: BufferId,
    pub size: u64,
    pub can_overlap: bool,  // Can this transfer overlap with computation?
}
```

---

## Energy-Aware Compilation

Apple Silicon's E-cores use significantly less power than P-cores. The compiler can optimize for:

| Goal | Strategy |
|------|----------|
| Maximum performance | P-cores + GPU + ANE |
| Balanced | P-cores for latency-sensitive, E-cores for background |
| Maximum battery | E-cores + ANE (avoid GPU for small workloads) |

```rust
pub enum CompilationGoal {
    MaxPerformance,
    Balanced,
    MaxBattery,
    Custom { compute_weight: f64, energy_weight: f64 },
}
```

---

## Implementation Plan

### Phase 1: Computation Graph Analysis
- Build computation graph from tMIR
- Identify data-parallel regions
- Identify matrix-heavy regions
- Cost estimation (CPU-only initially)

### Phase 2: SIMD Auto-Targeting
- NEON vectorization for identified parallel regions
- This is the easiest heterogeneous target (same address space, no launch overhead)
- Extend AArch64 ISel with NEON instructions

### Phase 3: GPU Targeting (Metal)
- Metal IR emission for identified GPU-suitable subgraphs
- Data transfer code generation (shared memory on Apple Silicon)
- Kernel launch and synchronization code
- Cost model for GPU vs CPU decision

### Phase 4: Neural Engine Targeting
- CoreML/BNNS lowering for matrix-heavy subgraphs
- ANE capability detection (which ops are supported)
- Fallback to GPU/CPU when ANE can't handle the computation

### Phase 5: Solver-Driven Optimal Allocation
- Encode graph partitioning as z4 optimization problem
- Proven-optimal allocation for tractable graph sizes
- Heuristic fallback for larger programs

### Phase 6: Verified Heterogeneous Compilation
- Prove that compute allocation preserves semantics
- Prove that data transfers maintain memory consistency
- Prove that synchronization is correct (no data races)
- All in tRust — the allocator is itself verified

---

## Why This Matters

In 2026, a MacBook has:
- 12 CPU cores (8P + 4E)
- 40 GPU cores
- 16 Neural Engine cores
- ~40 TOPS of compute

Most programs use **one CPU core**. The rest sits idle.

A programmer shouldn't need to know Metal, CoreML, NEON intrinsics, or GCD to use their hardware. The compiler already decides which register to use — it should decide which compute unit to use.

**With proofs.** Every compute allocation decision is formally verified to preserve program semantics. Moving a computation from CPU to GPU doesn't change the answer — and we can prove it.

---

## References

1. Ragan-Kelley et al. "Halide: A Language and Compiler for Optimizing Parallelism, Locality, and Recomputation." PLDI 2013.
2. Chen et al. "TVM: An Automated End-to-End Optimizing Compiler for Deep Learning." OSDI 2018.
3. Zheng et al. "Ansor: Generating High-Performance Tensor Programs for Deep Learning." OSDI 2020.
4. Schardl, Moses, Leiserson. "Tapir: Embedding Recursive Fork-Join Parallelism into LLVM's Intermediate Representation." PPoPP 2017.
5. Lattner et al. "MLIR: Scaling Compiler Infrastructure for Domain Specific Computation." CGO 2021.
6. Apple. "Metal Shading Language Specification." 2024.
7. Apple. "Core ML Framework Documentation." 2024.
8. Apple. "Accelerate Framework (BNNS) Documentation." 2024.
