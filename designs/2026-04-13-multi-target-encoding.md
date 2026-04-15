# Multi-Target Semantic Encoding

**Author:** Andrew Yates
**Date:** 2026-04-13
**Status:** Design
**Part of:** #121 (Master design: Unified solver architecture)

---

## Implementation Status (as of 2026-04-15)

**Overall: All three target semantic encodings (NEON, GPU, ANE) are implemented as code modules. Verification uses mock evaluation, not a real SMT solver.**

| Component | Status | Details |
|-----------|--------|---------|
| **NEON SIMD encoding** (`neon_semantics.rs`) | IMPLEMENTED | 1.6K LOC. 128-bit vector lane decomposition, arrangement selection. |
| **NEON lowering proofs** (`neon_lowering_proofs.rs`, `neon_encoding_proofs.rs`) | IMPLEMENTED (mock) | tMIR vector ops to NEON verified via mock evaluation. |
| **Metal GPU encoding** (`gpu_semantics.rs`) | IMPLEMENTED | Parallel map/reduce/scatter/gather with algebraic property justification. |
| **ANE encoding** (`ane_semantics.rs`) | IMPLEMENTED | 1.4K LOC. GEMM, Conv2D, activations, FP16 quantization. |
| **ANE precision proofs** (`ane_precision_proofs.rs`) | IMPLEMENTED (mock) | FP32-to-FP16 bounded error via f64 mock evaluation. |
| **Cross-target equivalence** | MOCK ONLY | Unified synthesis loop (`unified_synthesis.rs`) can compare across targets but uses mock evaluation, not z4. |

See #121 (Unified solver architecture epic).

---

## Overview

The unified solver architecture requires semantic encodings for each compilation target so the solver can prove equivalence across targets. This document specifies the SMT encoding strategy for:

1. **NEON SIMD** -- 128-bit vector operations decomposed into lanes
2. **Metal GPU kernels** -- parallel map/reduce over arrays
3. **ANE matrix operations** -- matrix multiply with fixed-function semantics

It also researches how TVM, Halide, and MLIR handle multi-target lowering to inform our approach.

---

## Prior Art: Multi-Target Compilation Frameworks

### TVM (Apache, 2018-2020)

TVM maps tensor computations to CPU/GPU/accelerator through a layered IR stack:

```
Relax (graph-level) -> TensorIR (loop-level) -> Target codegen
```

**Key architecture decisions:**
- **Separation of computation and schedule**: The computation (what to compute) is target-independent. The schedule (how to compute it -- tiling, vectorization, thread binding) is target-specific.
- **Target-specific lowering**: LLVM-based targets use IRBuilder for in-memory LLVM IR. GPU targets generate source-level languages (CUDA, Metal). External dispatch routes sub-graphs to vendor libraries (cuBLAS, cuDNN).
- **MetaSchedule**: Automated performance tuning via search over schedule transformations. Uses learned cost models (XGBoost) to predict execution time, avoiding hardware measurement during search.

**Relevance to LLVM2:** TVM's schedule separation maps to our proof-guided target analysis (Phase 1 of the unified pipeline). TVM's learned cost model is analogous to our Apple M-series cost model. But TVM is domain-specific (tensors) and unverified -- LLVM2 extends this to general computation with proofs.

**Ref:** Chen et al. "TVM: An Automated End-to-End Optimizing Compiler for Deep Learning." OSDI 2018. Zheng et al. "Ansor: Generating High-Performance Tensor Programs for Deep Learning." OSDI 2020.

### Halide (MIT/Google, 2012-)

Halide pioneered the algorithm-schedule separation for image processing pipelines:

**Target representation:** The `Target` class encodes:
- Architecture (X86, ARM, MIPS, WebAssembly, etc.)
- OS (Linux, macOS, Windows, iOS, Android)
- Feature flags (SSE41, AVX, AVX2, NEON, SVE, Metal, OpenCL, CUDA)

**Multi-target lowering:** From a single algorithm description:
- CPU: Generates LLVM IR with vectorization directives
- GPU: Generates CUDA/Metal/OpenCL kernel source
- Hexagon HVX: Generates Hexagon Vector Extensions
- The autoscheduler searches over schedules for a given target

**Key insight:** Halide's "schedule" is what LLVM2 calls "dispatch plan" -- it determines how computation maps to hardware. Halide automates this search; LLVM2 uses the solver to prove the mapping correct.

**Ref:** Ragan-Kelley et al. "Halide: A Language and Compiler for Optimizing Parallelism, Locality, and Recomputation." PLDI 2013.

### MLIR (LLVM, 2020-)

MLIR provides a multi-level IR framework with **dialects** for different abstraction levels:

**Progressive lowering through dialects:**
```
linalg (structured ops) -> vector + scf (vectorized loops)
                        -> gpu (GPU abstractions)
                        -> llvm (CPU codegen)
                        -> nvvm/rocdl/spirv (GPU codegen)
```

**Key dialects for multi-target:**
- `linalg` -- Structured linear algebra (target-independent)
- `vector` -- Hardware-agnostic vector operations
- `gpu` -- GPU-specific: kernel launch, thread/block indexing, barriers
- `arm_neon` / `arm_sve` / `arm_sme` -- ARM SIMD-specific
- `nvvm` / `rocdl` / `amdgpu` -- GPU vendor-specific
- `memref` -- Memory abstraction (target-independent address spaces)

**Heterogeneous support:** MLIR's dialect system allows operations from multiple targets to coexist in the same IR. The `gpu.launch` operation marks boundaries between host and device code. Lowering passes convert target-independent operations to target-specific ones.

**Relevance to LLVM2:** MLIR's dialect system is the closest analogue to our multi-target semantic encoding. But MLIR's lowering is unverified. LLVM2 adds solver-verified correctness to each lowering step.

**Ref:** Lattner et al. "MLIR: Scaling Compiler Infrastructure for Domain Specific Computation." CGO 2021.

### Key Takeaways

| Framework | Algorithm/Schedule Separation | Verification | Target Scope |
|-----------|------------------------------|-------------|-------------|
| TVM | Yes (TensorIR schedules) | No | Tensor ops only |
| Halide | Yes (explicit schedules) | No | Image processing |
| MLIR | Partial (dialect lowering) | No | General-purpose |
| **LLVM2** | **Yes (tMIR proofs + solver)** | **Yes (z4)** | **General-purpose** |

LLVM2 is unique in combining algorithm-schedule separation (via tMIR proofs) with formal verification (via z4) for general-purpose code, not just domain-specific workloads.

---

## Encoding Strategy

### Core Principle: Lane Decomposition

Every multi-lane (SIMD, GPU) operation is decomposed into per-lane scalar operations that the solver can verify individually. The vector/parallel structure is captured by the composition pattern.

```
Multi-lane operation on target T:
  Input: array or vector of N elements
  Output: array or vector of N elements
  
SMT encoding:
  forall i in [0, N):
    output[i] = f(input[i])   -- where f is the per-lane operation
    
Verification:
  forall i in [0, N):
    f_target(input[i]) = f_source(input[i])   -- per-lane equivalence
```

This reduces multi-target verification to per-element verification, which is the same QF_BV or QF_FP problem we already solve.

---

## Target 1: NEON SIMD Encoding

### AArch64 NEON Architecture

NEON operates on 128-bit registers (V0-V31) interpreted as vectors of lanes:

| Arrangement | Lane Count | Lane Width | Example |
|-------------|-----------|------------|---------|
| `.16B` | 16 | 8-bit | `ADD.16B V0, V1, V2` |
| `.8H` | 8 | 16-bit | `ADD.8H V0, V1, V2` |
| `.4S` | 4 | 32-bit | `ADD.4S V0, V1, V2` (int32 or float32) |
| `.2D` | 2 | 64-bit | `ADD.2D V0, V1, V2` (int64 or float64) |

**Ref:** ARM Architecture Reference Manual, Chapter C7: NEON and FP instructions.

### SMT Encoding: Integer NEON

Integer NEON ops are pure bitvector operations per lane. The 128-bit register is a `BitVec(128)` decomposed into lanes via `extract`.

```rust
/// Encode NEON ADD.4S Vd, Vn, Vm
/// 4 lanes of 32-bit integer addition
fn encode_add_4s(vn: Expr, vm: Expr) -> Expr {
    let lane_width = 32;
    let num_lanes = 4;
    let mut result = Expr::bitvec_const(0u64, 128);
    
    for lane in 0..num_lanes {
        let lo = lane * lane_width;
        let hi = lo + lane_width - 1;
        
        let vn_lane = vn.clone().extract(hi, lo);  // BV32
        let vm_lane = vm.clone().extract(hi, lo);  // BV32
        let sum = vn_lane.bvadd(vm_lane);           // BV32
        
        // Insert lane into result
        result = insert_lane(result, sum, lane, lane_width);
    }
    result
}

/// Insert a lane value into a 128-bit vector
fn insert_lane(vec: Expr, lane_val: Expr, lane_idx: usize, lane_width: usize) -> Expr {
    let shift = lane_idx * lane_width;
    let mask = !((((1u128 << lane_width) - 1) << shift) as i64);
    let cleared = vec.bvand(Expr::bitvec_const(mask, 128));
    let shifted = lane_val.zero_extend(128 - lane_width).bvshl(Expr::bitvec_const(shift as i64, 128));
    cleared.bvor(shifted)
}
```

**Theory required:** QF_BV only (bitvectors). No new z4 theories needed for integer NEON.

### SMT Encoding: Floating-Point NEON

FP NEON ops are per-lane IEEE 754 operations:

```rust
/// Encode NEON FADD.2D Vd, Vn, Vm
/// 2 lanes of 64-bit FP addition (Float64)
fn encode_fadd_2d(vn: Expr, vm: Expr) -> Expr {
    let fp64 = Sort::floating_point(11, 53);
    let rne = Expr::rounding_mode(RoundingMode::RNE);
    
    // Lane 0: bits [63:0]
    let vn_lane0 = to_fp(vn.clone().extract(63, 0), &fp64);
    let vm_lane0 = to_fp(vm.clone().extract(63, 0), &fp64);
    let sum0 = Expr::fp_add(rne.clone(), vn_lane0, vm_lane0);
    let bits0 = fp_to_bv(sum0, 64);
    
    // Lane 1: bits [127:64]
    let vn_lane1 = to_fp(vn.extract(127, 64), &fp64);
    let vm_lane1 = to_fp(vm.extract(127, 64), &fp64);
    let sum1 = Expr::fp_add(rne, vn_lane1, vm_lane1);
    let bits1 = fp_to_bv(sum1, 64);
    
    // Concatenate: [lane1 : lane0]
    bits1.concat(bits0)
}
```

**Theory required:** QF_FPBV (floating-point + bitvectors). Uses z4's existing FP theory.

### SMT Encoding: Cross-Lane Operations

Some NEON instructions operate across lanes (horizontal operations):

```rust
/// Encode FADDP D0, V4.2D  (pairwise add: result = lane0 + lane1)
fn encode_faddp_2d(vn: Expr) -> Expr {
    let fp64 = Sort::floating_point(11, 53);
    let rne = Expr::rounding_mode(RoundingMode::RNE);
    
    let lane0 = to_fp(vn.clone().extract(63, 0), &fp64);
    let lane1 = to_fp(vn.extract(127, 64), &fp64);
    
    fp_to_bv(Expr::fp_add(rne, lane0, lane1), 64)
}

/// Encode SADDLV (signed add across long vector): sum all lanes
fn encode_saddlv_4s(vn: Expr) -> Expr {
    let mut sum = Expr::bitvec_const(0i64, 64);
    for lane in 0..4 {
        let lo = lane * 32;
        let lane_val = vn.clone().extract(lo + 31, lo).sign_extend(32); // BV32 -> BV64
        sum = sum.bvadd(lane_val);
    }
    sum
}
```

### NEON Instruction Coverage

| Category | Instructions | Encoding | z4 Theory |
|----------|-------------|----------|-----------|
| Integer arithmetic | ADD, SUB, MUL, NEG | BV per-lane | QF_BV |
| Integer compare | CMEQ, CMGT, CMGE, CMHI, CMHS | BV per-lane | QF_BV |
| Integer shift | SHL, SSHR, USHR, SSRA, USRA | BV per-lane | QF_BV |
| Integer widening | SADDL, UADDL, SMULL, UMULL | BV with extension | QF_BV |
| Integer narrowing | XTN, SQXTN, UQXTN | BV with truncation | QF_BV |
| Integer saturating | SQADD, UQADD, SQSUB, UQSUB | BV with min/max | QF_BV |
| Bitwise | AND, ORR, EOR, BIC, BIF, BIT, BSL | BV per-lane | QF_BV |
| FP arithmetic | FADD, FSUB, FMUL, FDIV, FABS, FNEG | FP per-lane | QF_FPBV |
| FP compare | FCMEQ, FCMGT, FCMGE, FCMLT, FCMLE | FP per-lane | QF_FPBV |
| FP conversion | FCVTZS, FCVTZU, SCVTF, UCVTF | FP-BV conversion | QF_FPBV |
| Cross-lane | FADDP, SADDLV, SMAXV, SMINV | Multi-lane reduction | QF_BV or QF_FPBV |
| Permute | TBL, TBX, ZIP, UZP, TRN, REV | BV lane rearrangement | QF_BV |
| Load/store | LD1, LD2, LD3, LD4, ST1-ST4 | Array ops | QF_ABV |

---

## Target 2: Metal GPU Kernel Encoding

### GPU Execution Model

Metal GPU kernels execute as grids of threadgroups:

```metal
kernel void f(device float* input [[buffer(0)]],
              device float* output [[buffer(1)]],
              uint id [[thread_position_in_grid]],
              uint tid [[thread_position_in_threadgroup]]) {
    // Each thread processes one element
    output[id] = transform(input[id]);
}
```

**Key concepts:**
- Grid: total threads = `gridSize.x * gridSize.y * gridSize.z`
- Threadgroup: cooperative threads sharing `threadgroup` memory
- Thread: single execution unit with `thread_position_in_grid` index

### SMT Encoding: Parallel Map

A parallel map kernel applies a function `f` to every element of an array:

```rust
/// Encode: kernel void map(device T* in, device T* out, uint id) { out[id] = f(in[id]); }
/// Where f is a pure function proven equivalent to some tMIR computation.
fn encode_gpu_parallel_map(
    input: Expr,    // Array(BV64, BV_elem)
    f: impl Fn(Expr) -> Expr,  // Per-element transform
    n: u64,         // Grid size (number of elements)
) -> Expr {
    let mut result = input.clone();  // Start with input array
    for id in 0..n {
        let idx = Expr::bitvec_const(id as i64, 64);
        let elem = Expr::select(input.clone(), idx.clone());
        let transformed = f(elem);
        result = Expr::store(result, idx, transformed);
    }
    result
}
```

**Theory required:** QF_ABV (arrays + bitvectors). For small N, unrolled. For large N, bounded quantifiers.

### SMT Encoding: Parallel Reduce

A parallel reduce combines all elements using an associative+commutative binary operator:

```rust
/// Encode: result = reduce(op, identity, arr, N)
/// Precondition: op is Associative AND Commutative (from tMIR proofs)
///
/// Because op is associative and commutative, ANY evaluation order produces
/// the same result. We encode the sequential fold (simplest):
fn encode_gpu_parallel_reduce(
    input: Expr,         // Array(BV64, BV_elem)
    op: impl Fn(Expr, Expr) -> Expr,  // Binary operator
    identity: Expr,      // Identity element
    n: u64,              // Array length
) -> Expr {
    let mut acc = identity;
    for id in 0..n {
        let idx = Expr::bitvec_const(id as i64, 64);
        let elem = Expr::select(input.clone(), idx);
        acc = op(acc, elem);
    }
    acc
}
```

**Key insight:** tMIR's `Associative` and `Commutative` proofs make this encoding sound. Without those proofs, the GPU's parallel reduction (which evaluates in tree order) might produce a different result than the sequential fold. WITH those proofs, all evaluation orders are equivalent.

### SMT Encoding: Dot Product (Concrete Example)

From the unified solver design doc, the dot product example:

```rust
/// Verify: CPU dot_product(a, b) == GPU dot_product_kernel(a, b)
fn verify_dot_product_gpu(n: u64) {
    let mut program = Z4Program::new();
    
    // Declare input arrays
    let arr_sort = Sort::array(Sort::bitvec(64), Sort::floating_point(11, 53));
    let a = program.declare_const("a", arr_sort.clone());
    let b = program.declare_const("b", arr_sort.clone());
    
    let fp64 = Sort::floating_point(11, 53);
    let rne = Expr::rounding_mode(RoundingMode::RNE);
    let zero = Expr::fp_plus_zero(&fp64);
    
    // CPU sequential: sum(a[i] * b[i] for i in 0..N)
    let mut cpu_sum = zero.clone();
    for i in 0..n {
        let idx = Expr::bitvec_const(i as i64, 64);
        let ai = Expr::select(a.clone(), idx.clone());
        let bi = Expr::select(b.clone(), idx);
        let prod = Expr::fp_mul(rne.clone(), ai, bi);
        cpu_sum = Expr::fp_add(rne.clone(), cpu_sum, prod);
    }
    
    // GPU parallel: each thread computes a[id]*b[id], then reduce-sum
    // Because tMIR proves Associative+Commutative for FP add (with NoOverflow),
    // the parallel tree reduction equals sequential fold.
    // Encode as same sequential fold (proven equivalent by associativity).
    let mut gpu_sum = zero;
    for i in 0..n {
        let idx = Expr::bitvec_const(i as i64, 64);
        let ai = Expr::select(a.clone(), idx.clone());
        let bi = Expr::select(b.clone(), idx);
        let prod = Expr::fp_mul(rne.clone(), ai, bi);
        gpu_sum = Expr::fp_add(rne.clone(), gpu_sum, prod);
    }
    
    // Prove equivalence (trivially true since encodings are identical
    // thanks to Associative+Commutative proofs)
    program.assert(cpu_sum.fp_eq(gpu_sum).not());
    // Expected: UNSAT
}
```

**Note on FP associativity:** IEEE 754 FP addition is NOT associative in general (due to rounding). However, tMIR's `Associative` proof annotation means the source language has proven that associativity holds for this specific computation (e.g., the values don't cause rounding divergence, or the algorithm tolerates it). LLVM2 trusts tMIR proofs and uses them as preconditions.

### GPU Memory Transfer Encoding

Data transfers between CPU and GPU must be modeled:

```rust
/// Encode CPU-to-GPU DMA transfer
/// Precondition: ValidBorrow (source and destination don't alias)
fn encode_dma_transfer(
    cpu_mem: Expr,      // CPU memory (Array)
    gpu_mem: Expr,      // GPU memory (Array)
    src_base: Expr,     // Source base address (BV64)
    dst_base: Expr,     // Destination base address (BV64)
    size: u64,          // Transfer size in elements
) -> Expr {
    let mut result = gpu_mem;
    for i in 0..size {
        let offset = Expr::bitvec_const(i as i64, 64);
        let src_addr = src_base.clone().bvadd(offset.clone());
        let dst_addr = dst_base.clone().bvadd(offset);
        let value = Expr::select(cpu_mem.clone(), src_addr);
        result = Expr::store(result, dst_addr, value);
    }
    result
}
```

**Theory required:** QF_ABV. The `ValidBorrow` proof from tMIR ensures non-aliasing, which means the DMA transfer is a correct copy (no write-write conflicts).

### GPU Encoding Complexity Analysis

| Kernel Pattern | SMT Formula Size | Solver Time (est.) |
|---------------|-----------------|-------------------|
| Map, N=4 | ~50 terms | < 1s |
| Map, N=64 | ~800 terms | < 5s |
| Map, N=1024 | ~13K terms | < 30s |
| Reduce, N=4 | ~30 terms | < 1s |
| Reduce, N=64 | ~500 terms | < 5s |
| Reduce, N=1024 | ~8K terms | < 30s |
| Dot product, N=1000 | ~15K terms | 30-60s |
| Map+Reduce, N=4096 | ~65K terms | timeout risk |

For N > 1024, bounded quantifiers are needed to avoid formula blowup.

---

## Target 3: ANE (Apple Neural Engine) Encoding

### ANE Architecture

The Apple Neural Engine is a fixed-function accelerator optimized for:
- Matrix multiplication (INT8, INT16, FP16)
- Convolutions (1D, 2D, depthwise)
- Elementwise operations (add, mul, activation functions)
- Pooling (max, average)

ANE operations are accessed via CoreML model compilation (`.mlmodelc`).

### SMT Encoding: Matrix Multiply

Matrix multiplication `C = A * B` where A is MxK and B is KxN:

```rust
/// Encode ANE matrix multiply: C[i][j] = sum(A[i][k] * B[k][j] for k in 0..K)
fn encode_matmul(
    a: Expr,     // Array(BV64, FP16) -- flattened M*K matrix
    b: Expr,     // Array(BV64, FP16) -- flattened K*N matrix
    m: u64, k: u64, n: u64,
) -> Expr {
    let fp16 = Sort::floating_point(5, 11);
    let rne = Expr::rounding_mode(RoundingMode::RNE);
    let arr_sort = Sort::array(Sort::bitvec(64), fp16.clone());
    let mut c = program.declare_const("c_init", arr_sort);  // result array
    
    for i in 0..m {
        for j in 0..n {
            let mut sum = Expr::fp_plus_zero(&fp16);
            for kk in 0..k {
                let a_idx = Expr::bitvec_const((i * k + kk) as i64, 64);
                let b_idx = Expr::bitvec_const((kk * n + j) as i64, 64);
                let a_elem = Expr::select(a.clone(), a_idx);
                let b_elem = Expr::select(b.clone(), b_idx);
                let prod = Expr::fp_mul(rne.clone(), a_elem, b_elem);
                sum = Expr::fp_add(rne.clone(), sum, prod);
            }
            let c_idx = Expr::bitvec_const((i * n + j) as i64, 64);
            c = Expr::store(c, c_idx, sum);
        }
    }
    c
}
```

**Practical limit:** For small matrices (M,K,N <= 8), this is tractable. For larger matrices, use UF abstraction.

### SMT Encoding: ANE as Uninterpreted Functions

For operations whose ANE semantics are known but complex, use UF with axiomatic constraints:

```rust
/// Model ANE conv2d as an uninterpreted function with known properties
fn encode_ane_conv2d(
    input: Expr,
    kernel: Expr,
    // Properties known from ANE specification:
    // - Linearity: conv(a*x + b*y, k) = a*conv(x,k) + b*conv(y,k)
    // - Associativity with compose: conv(conv(x,k1),k2) = conv(x, k1*k2)
) -> Expr {
    // Declare as uninterpreted function
    let conv = program.declare_fun("ane_conv2d",
        vec![arr_sort.clone(), arr_sort.clone()],
        arr_sort.clone());
    
    conv.apply(vec![input, kernel])
}
```

**Theory required:** QF_UFBV or QF_UFABV. The UF abstraction lets us prove high-level properties (e.g., "ANE matmul computes the same matrix as CPU matmul") without encoding ANE's full implementation.

### ANE Precision Considerations

ANE typically operates at FP16 or INT8 precision. When the source computation is FP32 or FP64, the lowering introduces precision loss. The verification must account for this:

```
Proof obligation for ANE lowering:
  NOT: exact equivalence (fp32_result == ane_fp16_result)
  BUT: bounded error (|fp32_result - fp16_to_fp32(ane_fp16_result)| < epsilon)
```

This requires encoding the FP16-to-FP32 conversion and bounding the error, which z4's FP theory can handle via `fp.to_fp` conversion functions.

---

## Unified Semantic Encoding API

### Proposed `llvm2-verify` API

```rust
/// Semantic encoding for a computation on a specific target
pub trait TargetEncoding {
    /// The SMT logic required (e.g., QF_BV, QF_ABV, QF_FPBV)
    fn required_logic(&self) -> SmtLogic;
    
    /// Encode the computation's semantics as an SMT expression
    fn encode(&self, inputs: &[Expr], program: &mut Z4Program) -> Expr;
    
    /// Estimated solver time for this encoding
    fn estimated_solve_time(&self) -> Duration;
}

/// AArch64 scalar encoding (existing)
pub struct ScalarEncoding {
    pub opcode: AArch64Opcode,
    pub operand_types: Vec<Type>,
}

/// NEON SIMD encoding (new)
pub struct NeonEncoding {
    pub opcode: NeonOpcode,
    pub arrangement: VectorArrangement,
}

/// GPU kernel encoding (new)
pub struct GpuKernelEncoding {
    pub pattern: GpuPattern,
    pub grid_size: u64,
    pub element_type: Type,
}

/// ANE operation encoding (new)
pub struct AneEncoding {
    pub operation: AneOp,
    pub precision: AnePrecision,
    pub dimensions: MatrixDim,
}

/// GPU kernel patterns
pub enum GpuPattern {
    ParallelMap { f: Box<dyn Fn(Expr) -> Expr> },
    ParallelReduce { op: BinaryOp, identity: Expr },
    ParallelMapReduce { map_f: Box<dyn Fn(Expr) -> Expr>, reduce_op: BinaryOp, identity: Expr },
    Scatter { /* ... */ },
    Gather { /* ... */ },
}

/// NEON vector arrangements
pub enum VectorArrangement {
    B16,  // 16 x 8-bit
    H8,   // 8 x 16-bit
    S4,   // 4 x 32-bit
    D2,   // 2 x 64-bit
}
```

### Cross-Target Equivalence Proof

The key verification query compares encodings across targets:

```rust
/// Prove that a computation on target T produces the same result
/// as the tMIR source semantics.
pub fn prove_target_equivalence(
    tmir_encoding: &dyn TargetEncoding,
    target_encoding: &dyn TargetEncoding,
    inputs: &[Expr],
    program: &mut Z4Program,
) -> VerificationResult {
    let tmir_result = tmir_encoding.encode(inputs, program);
    let target_result = target_encoding.encode(inputs, program);
    
    // Assert negation of equivalence
    program.assert(tmir_result.eq(target_result).not());
    
    match program.check_sat() {
        Unsat => VerificationResult::Valid,      // Proven equivalent
        Sat(model) => VerificationResult::Invalid { counterexample: model },
        Unknown(reason) => VerificationResult::Unknown { reason },
    }
}
```

---

## Integration with Unified Solver Pipeline

### Phase 1: Proof-Guided Target Analysis

The target analysis pass determines which targets are legal for each computation subgraph:

```rust
pub fn analyze_legal_targets(
    subgraph: &ComputeSubgraph,
    tmir_proofs: &ProofAnnotations,
) -> Vec<ComputeTarget> {
    let mut targets = vec![ComputeTarget::CpuScalar]; // Always legal
    
    // NEON: legal for data-parallel operations on vectors
    if subgraph.is_data_parallel() && subgraph.element_count() >= 2 {
        targets.push(ComputeTarget::Neon(best_arrangement(subgraph)));
    }
    
    // GPU: legal if Pure + InBounds + ValidBorrow
    if tmir_proofs.has(Pure) && tmir_proofs.has(InBounds) && tmir_proofs.has(ValidBorrow) {
        if subgraph.element_count() >= gpu_threshold() {
            targets.push(ComputeTarget::Gpu);
        }
    }
    
    // ANE: legal for matrix ops if Pure + supported operation
    if tmir_proofs.has(Pure) && is_ane_compatible(subgraph) {
        targets.push(ComputeTarget::Ane);
    }
    
    targets
}
```

### Phase 2: Solver-Driven Synthesis

For each legal target, generate a candidate encoding and verify:

```rust
pub fn synthesize_best_implementation(
    tmir_encoding: &TmirEncoding,
    legal_targets: &[ComputeTarget],
    cost_model: &dyn CostModel,
) -> (ComputeTarget, ProofCertificate) {
    let mut best: Option<(ComputeTarget, f64, ProofCertificate)> = None;
    
    for target in legal_targets {
        let target_encoding = generate_encoding(target, tmir_encoding);
        let result = prove_target_equivalence(tmir_encoding, &target_encoding, ...);
        
        if let VerificationResult::Valid = result {
            let cost = cost_model.estimate_cost(target, tmir_encoding);
            if best.is_none() || cost < best.as_ref().unwrap().1 {
                best = Some((*target, cost, result.certificate()));
            }
        }
    }
    
    best.map(|(t, _, cert)| (t, cert)).unwrap_or((ComputeTarget::CpuScalar, ...))
}
```

---

## Implementation Plan

### Phase 1: NEON Integer Encoding (P1)

1. Define `NeonEncoding` struct in `llvm2-verify/src/neon_semantics.rs`
2. Implement lane decomposition for integer NEON ops (ADD, SUB, MUL, shift, compare)
3. Implement cross-lane operations (ADDV, SADDLV)
4. Implement permute operations (TBL, ZIP, TRN)
5. Write verification tests for each NEON instruction

**Theory:** QF_BV only
**Estimated effort:** 1-2 techlead sessions

### Phase 2: NEON FP Encoding (P2)

1. Extend `neon_semantics.rs` with FP lane operations
2. Implement FADD, FSUB, FMUL, FDIV per-lane encoding
3. Implement FADDP (pairwise add) cross-lane encoding
4. Implement FP conversion operations (FCVTZS, SCVTF)
5. Add FP rounding mode handling (AArch64 FPCR)

**Theory:** QF_FPBV
**Estimated effort:** 1-2 techlead sessions

### Phase 3: GPU Map/Reduce Encoding (P2)

1. Define `GpuKernelEncoding` in `llvm2-verify/src/gpu_semantics.rs`
2. Implement parallel_map encoding (unrolled for small N)
3. Implement parallel_reduce encoding (using Associative proof)
4. Implement DMA transfer encoding
5. Write dot_product verification example

**Theory:** QF_ABV (+ bounded quantifiers for large N)
**Estimated effort:** 2-3 techlead sessions

### Phase 4: ANE Matrix Encoding (P3)

1. Define `AneEncoding` in `llvm2-verify/src/ane_semantics.rs`
2. Implement matrix multiply encoding (small matrices: unrolled)
3. Implement UF abstraction for large/complex ANE ops
4. Handle precision mismatch (FP16 ANE vs FP32 source)
5. Write matmul verification example

**Theory:** QF_UFABV or QF_FPBV
**Estimated effort:** 2-3 techlead sessions

### Phase 5: Unified Encoding API (P2)

1. Define `TargetEncoding` trait
2. Implement `prove_target_equivalence`
3. Integrate with proof-guided target analysis
4. Add cost model hooks for target selection
5. Connect to unified synthesis loop

**Estimated effort:** 1-2 techlead sessions

---

## Complexity and Scalability

| Encoding | Formula Size | Solver Time | Scalability Limit |
|----------|-------------|-------------|-------------------|
| NEON 4-lane BV | O(4 * inst_size) | < 1s | None (small formulas) |
| NEON 2-lane FP | O(2 * fp_formula) | 1-10s | FP formula size |
| GPU map N=64 | O(N * inst_size) | < 10s | N < ~1000 without quantifiers |
| GPU reduce N=64 | O(N * op_size) | < 10s | N < ~1000 without quantifiers |
| GPU N=large | O(quantifier) | Depends on z4 | z4 quantifier support |
| ANE matmul 8x8 | O(M*K*N * fp_size) | 10-60s | Matrix dims < ~16 |
| ANE matmul large | O(UF constraints) | < 5s | UF is lightweight |

For production use, the encoding falls into two regimes:
1. **Small instances (NEON, small GPU):** Unrolled, exact proofs. Fast and complete.
2. **Large instances (big GPU, ANE):** UF abstraction or bounded quantifiers. Sound but approximate.

---

## References

1. Chen et al. "TVM: An Automated End-to-End Optimizing Compiler for Deep Learning." OSDI 2018.
2. Zheng et al. "Ansor: Generating High-Performance Tensor Programs for Deep Learning." OSDI 2020.
3. Ragan-Kelley et al. "Halide: A Language and Compiler for Optimizing Parallelism, Locality, and Recomputation." PLDI 2013.
4. Lattner et al. "MLIR: Scaling Compiler Infrastructure for Domain Specific Computation." CGO 2021.
5. ARM Architecture Reference Manual, DDI 0487. Chapter C7: NEON and FP.
6. Apple. "Metal Shading Language Specification." 2024.
7. Apple. "Core ML Framework Documentation." 2024.
8. LLVM2 unified solver architecture: `designs/2026-04-13-unified-solver-architecture.md`
9. LLVM2 z4 theory extensions: `designs/2026-04-13-z4-theory-extensions.md`
10. LLVM2 verification architecture: `designs/2026-04-13-verification-architecture.md`
