# Unified Solver Architecture

**Author:** Andrew Yates
**Date:** 2026-04-13
**Status:** Design (MASTER DESIGN — supersedes per-pillar designs for architecture)

---

## Implementation Status (as of 2026-04-15)

**Overall: Per-target semantic encodings and the unified synthesis loop are implemented. The "same solver, all targets" vision is structurally present but runs on mock evaluation, not a real SMT solver.**

| Component | Status | Details |
|-----------|--------|---------|
| **CPU scalar semantics** (`aarch64_semantics.rs`) | IMPLEMENTED | AArch64 instruction semantics as SmtExpr. |
| **NEON SIMD semantics** (`neon_semantics.rs`) | IMPLEMENTED | 1.6K LOC. Lane decomposition for vector instructions. |
| **GPU semantics** (`gpu_semantics.rs`) | IMPLEMENTED | Metal parallel map/reduce/scatter/gather encoding. |
| **ANE semantics** (`ane_semantics.rs`) | IMPLEMENTED | 1.4K LOC. GEMM, Conv2D, activations, FP16 quantization. |
| **Unified CEGIS synthesis** (`unified_synthesis.rs`) | IMPLEMENTED | 5.1K LOC. Cross-target candidate ranking (CPU/NEON/GPU/ANE), fast counterexample filtering, cost-based ranking. |
| **Multi-target cost model** (`cost_model.rs`) | IMPLEMENTED | 2.7K LOC. CPU/NEON/GPU/ANE latency and throughput. |
| **SMT solver backend** | NOT CONNECTED | All verification uses mock Rust evaluation. z4 not linked. See #34, #121, #236. |
| **Cross-target equivalence proofs** | MOCK ONLY | The solver can compare across targets using mock evaluation, but no formal z4-backed proofs exist. |
| **Integration with compilation pipeline** | PARTIAL | Unified synthesis can propose cross-target lowerings, but the compilation pipeline does not yet invoke it to make actual dispatch decisions during compilation. |

---

## The Core Insight

LLVM2's four vision pillars — superoptimization, transparency, AI-native compilation, and heterogeneous compute — are not separate features. They are **the same system** viewed at different granularities. The unifying mechanism is the **solver**.

The solver doesn't optimize "CPU code." It optimizes **computation**. The target hardware is just a parameter to the semantic encoding.

```
Traditional compiler:
  Human writes peephole rules for CPU.
  Human writes GPU kernel dispatch.
  Human writes SIMD intrinsics.
  Human writes cost model.
  Trust all of it.

LLVM2:
  Solver searches over ALL targets simultaneously.
  Solver proves each candidate correct.
  Cost model ranks proven-correct candidates.
  Best one wins.
  Everything has a certificate.
```

---

## Architecture: One Solver, All Targets

### The Universal Optimization Query

Every optimization in LLVM2 — scalar, vector, GPU, ANE — is the same solver query:

```
Given:
  source: tMIR semantic encoding of the computation
  candidate: semantic encoding of an implementation on target T
  
Query:
  ∀ inputs: source(inputs) = candidate(inputs)?
  
If UNSAT (no counterexample): proven equivalent → rank by cost
If SAT (counterexample found): not equivalent → discard
```

The solver doesn't know or care what "target T" is. It just compares two semantic encodings. This means we can search over:

| Target | Semantic Encoding | What the solver sees |
|--------|-------------------|---------------------|
| AArch64 scalar | Bitvector ops on registers | `BV64 → BV64` functions |
| NEON SIMD | Bitvector ops on 128-bit vectors | `BV128 → BV128` functions |
| Metal GPU kernel | Parallel map over array | `Array BV64 → Array BV64` functions |
| ANE matrix op | Matrix multiply semantics | `Matrix → Matrix` functions |
| Multi-target | Composition of the above | `BV64 → dispatch → BV64` functions |

**All use the same solver. All produce the same kind of proof certificate.**

### The Search Space

Traditional superoptimization searches over scalar instructions:
```
{ADD, SUB, MUL, LSL, LSR, AND, ORR, ...} × {register, immediate} × length 1-3
```

Unified search extends this to:
```
Scalar:  {ADD, SUB, MUL, ...} × {reg, imm}
SIMD:    {FADD.4S, FMUL.4S, ...} × {vreg}
GPU:     {map, reduce, scan, matmul} × {threadgroup_size}
ANE:     {conv, matmul, elementwise} × {precision}
Hybrid:  {CPU_compute + GPU_compute + transfer} × {partition}
```

The search space is larger, but tMIR proofs make it **tractable** — see below.

### tMIR Proofs: The Information Advantage

This is why LLVM2 can do what no traditional compiler can. tMIR carries proofs that eliminate the conservative assumptions forcing traditional compilers to keep everything on CPU:

| tMIR Proof | Conservative assumption it eliminates | What it unlocks |
|------------|--------------------------------------|-----------------|
| `Pure` | "This might have side effects" | Safe to move to GPU/ANE (no side effects to preserve) |
| `ValidBorrow` | "These pointers might alias" | Zero-copy DMA between CPU↔GPU (proven non-aliased) |
| `InBounds` | "This index might be out of range" | Skip bounds checks on GPU (proven safe) |
| `NoOverflow` | "This arithmetic might overflow" | Use fast unchecked GPU/ANE arithmetic |
| `Commutative` | "Order might matter" | Reorder for GPU-friendly access patterns |
| `Associative` | "Grouping might matter" | Parallelize reductions across GPU threads |
| `Deterministic` | "Result might vary by execution order" | Safe to distribute across any hardware |

Without these proofs, a compiler CANNOT safely move `a[i] + b[i]` to GPU because:
- `a` and `b` might alias → GPU copy would be wrong
- `i` might be out of bounds → GPU would fault differently than CPU
- The addition might overflow → GPU wrapping semantics might differ

With tMIR proofs: `a` and `b` are `ValidBorrow` (non-aliasing), `i` is `InBounds`, the add is `NoOverflow`. The solver can prove the GPU kernel equivalent and the compiler can dispatch it — **no programmer annotation needed**.

### Pruning the Cross-Target Search

The search space `(all targets) × (all sequences)` is huge. tMIR proofs prune it:

```
1. Is the computation Pure?
   NO  → CPU only (side effects must stay on CPU)
   YES → continue

2. Does it operate on arrays/matrices?
   NO  → Scalar or SIMD only (GPU launch overhead dominates for scalars)
   YES → continue

3. Are array accesses InBounds + ValidBorrow?
   NO  → CPU/SIMD only (can't prove GPU memory safety)
   YES → GPU/ANE candidates are legal

4. Is it associative + commutative?
   NO  → Sequential GPU ok, but no parallel reduction
   YES → Full GPU parallelism legal

5. Cost model: is data large enough to amortize GPU launch?
   NO  → SIMD on CPU
   YES → GPU or ANE
```

Most of the search tree is pruned by steps 1-3 using tMIR proofs. The solver only runs on the surviving candidates.

---

## Unified Solver Pipeline

```
tMIR input (with proofs)
    │
    ▼
┌──────────────────────────────────────────────────┐
│  Phase 1: Proof-Guided Target Analysis            │
│                                                    │
│  For each computation subgraph:                    │
│  - Check tMIR proofs (Pure? InBounds? ValidBorrow?)│
│  - Determine legal targets (CPU, SIMD, GPU, ANE)   │
│  - Estimate per-target cost (cost model)           │
└──────────────────────────────────────────────────┘
    │
    ▼
┌──────────────────────────────────────────────────┐
│  Phase 2: Solver-Driven Synthesis                  │
│                                                    │
│  For each subgraph × legal target:                 │
│  - Enumerate candidate implementations             │
│  - z4 CEGIS: prove equivalence to tMIR source      │
│  - Rank proven candidates by cost                  │
│  - Select cheapest proven-correct implementation    │
└──────────────────────────────────────────────────┘
    │
    ▼
┌──────────────────────────────────────────────────┐
│  Phase 3: Dispatch Plan Generation                 │
│                                                    │
│  - Partition computation across selected targets    │
│  - Generate data transfer code (CPU↔GPU)           │
│  - Generate synchronization barriers               │
│  - z4 proves dispatch plan preserves semantics      │
└──────────────────────────────────────────────────┘
    │
    ▼
┌──────────────────────────────────────────────────┐
│  Phase 4: Per-Target Lowering                      │
│                                                    │
│  CPU subgraphs → AArch64 ISel → RegAlloc → encode  │
│  SIMD subgraphs → NEON ISel → RegAlloc → encode    │
│  GPU subgraphs → Metal IR → .metallib              │
│  ANE subgraphs → CoreML → .mlmodelc                │
│                                                    │
│  Each lowering verified by z4 (per-instruction)     │
└──────────────────────────────────────────────────┘
    │
    ▼
┌──────────────────────────────────────────────────┐
│  Phase 5: Transparency + Certificates              │
│                                                    │
│  - Full event log: every decision with WHY          │
│  - Provenance: tMIR → target instruction mapping    │
│  - Proof certificates for every lowering + dispatch │
│  - Cost justification for every target choice       │
└──────────────────────────────────────────────────┘
    │
    ▼
  Output: binary + proof certificates + compilation trace
```

---

## Concrete Example

Source (tMIR):
```
fn dot_product(a: &[f64; 1000], b: &[f64; 1000]) -> f64 {
    // Proofs: Pure, InBounds, ValidBorrow(a, b), Associative(+), Commutative(+)
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}
```

**Phase 1: Proof analysis**
- Pure ✓ → any target legal
- InBounds ✓ → GPU indexing safe
- ValidBorrow(a, b) ✓ → zero-copy DMA, a and b don't alias
- Associative+Commutative ✓ → parallel reduction legal

Legal targets: CPU scalar, NEON SIMD, GPU, ANE

**Phase 2: Solver synthesis**

Candidate A — CPU scalar:
```asm
; 1000 iterations × (LDR + LDR + FMUL + FADD) = 4000 instructions
loop: LDR D0, [X0, X2, LSL #3]
      LDR D1, [X1, X2, LSL #3]
      FMUL D0, D0, D1
      FADD D3, D3, D0
      ADD X2, X2, #1
      CMP X2, #1000
      B.NE loop
```
z4 proves: equivalent ✓. Cost: ~4000 cycles.

Candidate B — NEON SIMD:
```asm
; 500 iterations × (LDP + LDP + FMUL.2D + FADD.2D) + horizontal add
loop: LDP D0, D1, [X0, X2, LSL #3]
      LDP D2, D3, [X1, X2, LSL #3]
      FMUL.2D V0, V0, V2
      FADD.2D V4, V4, V0
      ADD X2, X2, #2
      CMP X2, #1000
      B.NE loop
      FADDP D0, V4.2D
```
z4 proves: equivalent ✓ (using Associative+Commutative proofs). Cost: ~2000 cycles.

Candidate C — GPU (Metal):
```metal
kernel void dot(const device float* a,
                const device float* b,
                device float* result,
                uint id [[thread_position_in_grid]]) {
    // parallel map: each thread computes a[id] * b[id]
    // parallel reduce: sum across threads
}
```
z4 proves: equivalent ✓ (using Pure + InBounds + ValidBorrow proofs).
Cost: ~50 cycles compute + ~200 cycles transfer = ~250 cycles.

Candidate D — ANE:
```
CoreML elementwise multiply + reduce sum
```
z4 proves: equivalent ✓. Cost: ~30 cycles compute + ~300 cycles transfer = ~330 cycles.

**Phase 2 result:** For N=1000, GPU wins (250 cycles). Emit GPU path with CPU-SIMD fallback for N<100.

**Phase 3:** Generate dispatch: allocate Metal buffer, copy a/b (or zero-copy via ValidBorrow), dispatch kernel, read result.

**Phase 4:** Lower dispatch code to AArch64 (Metal API calls), lower GPU kernel to Metal IR.

**Phase 5:** Certificate proves: `∀ a, b: dot_product_tmir(a,b) = dot_product_gpu(a,b)`. Event log records: "Chose GPU over NEON because: data size 1000 × f64 = 8KB > GPU threshold 4KB; GPU cost 250 cycles < NEON cost 2000 cycles."

---

## Semantic Encoding Per Target

### AArch64 Scalar (exists today)

Bitvector operations on 64-bit values:
```
encode(ADD Xd, Xn, Xm) = λ(Xn, Xm). BVAdd(Xn, Xm)
encode(LSL Xd, Xn, #k) = λ(Xn). BVShl(Xn, k)
```

### NEON SIMD (new)

Bitvector operations on 128-bit vectors, decomposed into lanes:
```
encode(FADD.2D Vd, Vn, Vm) = λ(Vn, Vm). 
  concat(FPAdd(Vn[63:0], Vm[63:0]), FPAdd(Vn[127:64], Vm[127:64]))
```

Requires: FP theory in z4 (or bitwise FP encoding via IEEE 754).

### GPU Kernel (new)

Parallel map/reduce over arrays:
```
encode(parallel_map(f, arr)) = λ(arr). 
  ∀ i ∈ [0, len(arr)): result[i] = f(arr[i])

encode(parallel_reduce(+, arr)) = λ(arr).
  fold(+, 0, arr)  // legal because Associative+Commutative
```

Requires: Array theory in z4 (QF_ABV).

### ANE Operations (new)

Matrix operations with fixed-function semantics:
```
encode(matmul(A, B)) = λ(A, B).
  ∀ i,j: result[i][j] = Σ_k A[i][k] * B[k][j]
```

Requires: Nested array theory or unrolled bitvector encoding for small matrices.

---

## What z4 Needs

| Feature | Theory | Status | Needed For |
|---------|--------|--------|-----------|
| Bitvectors | QF_BV | Exists | Scalar AArch64 |
| Arrays | QF_ABV | Needed | GPU kernels, memory ops |
| Floating point | QF_FP | Needed | NEON FP, GPU FP |
| Uninterpreted functions | QF_UF | Needed | External calls, ANE ops |
| Quantifiers (bounded) | BV+∀ | Needed | "for all indices" proofs |

The hardest part is floating-point: IEEE 754 semantics in SMT are expensive. Practical approach: prove integer/bitwise operations exactly, use bounded testing for FP (validate on concrete inputs up to some coverage threshold, not full SMT proof). This is what STOKE does for FP.

---

## The Self-Improving Loop (Unified)

The AI-native pillar isn't separate either. It's the same solver loop with an AI agent providing candidates:

```
1. AI agent observes: tMIR computation + available targets + cost model
2. AI agent proposes: "try this GPU kernel" or "try this NEON sequence"
3. Solver verifies: proves equivalence (or returns counterexample)
4. If proven: add to rule database with cost annotation
5. Cost model ranks: select best proven implementation
6. Measure runtime: update cost model with real performance data
7. AI agent learns: better proposals next time
```

The solver is the **gatekeeper**. The AI can be arbitrarily creative or wrong. Only proven-correct implementations are used. The system can only improve, never regress in correctness.

---

## Transparency Is Built In

Every step above produces audit data:
- Phase 1: "Computation X is Pure (proof: abc), InBounds (proof: def) → GPU legal"
- Phase 2: "Candidate GPU kernel proved equivalent (proof: ghi), cost 250 cycles"
- Phase 3: "Chose GPU over NEON: 250 < 2000 cycles, data size 8KB > 4KB threshold"
- Phase 4: "GPU kernel lowered to Metal IR, 12 instructions"
- Phase 5: certificate = {proof_phase2: ghi, proof_phase4: jkl, cost_justification: ...}

Transparency isn't a feature bolted on — it's an inherent property of solver-driven compilation. The solver produces proofs. The cost model produces justifications. The event log just records what already exists.

---

## Implementation Priority

The unified architecture changes the build order:

1. **z4 extensions** — Array theory (QF_ABV) for GPU/memory, FP theory for NEON
2. **Multi-target semantic encoding** — NEON, Metal kernel, ANE operation encodings
3. **Unified synthesis loop** — CEGIS over all targets, not just scalar AArch64
4. **Proof-guided target analysis** — use tMIR proofs to determine legal targets
5. **Cost model** — multi-target cost estimation (CPU, SIMD, GPU, ANE)
6. **Dispatch codegen** — data transfer, synchronization, kernel launch
7. **Transparency** — event log, provenance, certificates (mostly falls out naturally)
8. **AI integration** — pluggable candidate proposers, self-improving loop

The critical path is z4 extensions (step 1) and multi-target encoding (step 2). Everything else builds on those.

---

## Why This Is Different From Everything Else

| System | Approach | Limitation |
|--------|----------|------------|
| LLVM | Hand-written rules per target | No cross-target optimization |
| TVM | ML autotuning for tensors | Domain-specific, no proofs |
| Halide | Schedule search for stencils | Domain-specific, no proofs |
| MLGO | ML replaces 2 heuristics | Narrow scope, no proofs |
| OpenCL/SYCL | Programmer picks target | Manual, no automatic allocation |
| LLVM2 | **Solver searches all targets, proves everything, tMIR proofs enable what others can't** | Needs z4 extensions |

LLVM2 is the only system where:
1. The solver searches across CPU, GPU, and accelerator simultaneously
2. Every optimization is proven correct
3. Proof-carrying IR enables optimizations no other compiler can make
4. The system is self-improving (AI proposes, solver verifies)
5. Every decision is transparent and auditable

---

## References

1. Schkufza et al. "Stochastic Superoptimization." ASPLOS 2013.
2. Sasnauskas et al. "Souper: A Synthesizing Superoptimizer." 2017.
3. Ragan-Kelley et al. "Halide." PLDI 2013.
4. Chen et al. "TVM." OSDI 2018.
5. Cummins et al. "MLGO." 2022.
6. Lopes et al. "Alive2." PLDI 2021.
7. Leroy. "CompCert." J. Automated Reasoning, 2009.
8. Apple. "Metal Shading Language Specification." 2024.
9. Apple. "Core ML Framework." 2024.
