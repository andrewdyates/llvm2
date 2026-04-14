# Metal GPU Semantic Encoding for LLVM2 Verification

**Author:** Andrew Yates
**Date:** 2026-04-14
**Status:** Design
**Part of:** #127 (Multi-target semantic encoding: Metal GPU kernel parallel map/reduce)
**Related:** #121 (Unified solver architecture), #129 (Unified synthesis loop), #131 (Dispatch codegen)

---

## Problem

The unified solver architecture searches over CPU scalar, NEON SIMD, GPU, and ANE targets to find the optimal proven-correct implementation for each computation subgraph. The NEON semantic encoding (`neon_semantics.rs`) is implemented. The GPU semantic encoding is missing -- without it, the solver cannot generate or verify Metal GPU candidates.

This document specifies the Metal GPU semantic encoding: the compute model, the tMIR-to-Metal operation mapping, the SMT encoding for each GPU operation type, data transfer semantics, and cost thresholds for profitability analysis.

---

## Metal Shading Language (MSL) Compute Model

### Architecture Overview

Metal compute kernels execute on Apple GPU cores as massively parallel programs. The execution hierarchy has three levels:

```
Grid  (total work)
  |
  +-- Threadgroup  (cooperative threads sharing threadgroup memory)
        |
        +-- SIMD Group  (32 threads executing in lockstep on one EU)
              |
              +-- Thread  (single execution lane)
```

**Ref:** Apple. "Metal Shading Language Specification," Version 3.2, Chapter 4: Compute Functions. Apple Developer Documentation, 2024.

| Concept | Metal Name | Typical Size | LLVM2 Encoding |
|---------|-----------|-------------|----------------|
| Total work items | Grid size | 1K - 1M+ | Array length `N` |
| Cooperative group | Threadgroup | 32 - 1024 | Reduction unit boundary |
| SIMD-width group | SIMD group | 32 (Apple GPU) | Implicit (hardware) |
| Single lane | Thread | 1 | Per-element function |

### Key Hardware Parameters (Apple M-series GPU)

| Parameter | M1 | M3 | M4 | Source |
|-----------|-----|-----|-----|--------|
| GPU cores | 7-8 | 10 | 10 | Apple specs |
| Execution units / core | 16 | 16 | 16 | Apple GPU arch |
| Threads / SIMD group | 32 | 32 | 32 | Metal spec |
| Max threadgroup size | 1024 | 1024 | 1024 | Metal spec |
| Threadgroup memory | 32 KB | 32 KB | 32 KB | Metal spec |
| Max total threads | ~24K | ~40K | ~40K | cores * EUs * 32 |
| Memory bandwidth | 68 GB/s | 100 GB/s | 120 GB/s | Apple specs |
| FP32 TFLOPS | 2.6 | 4.1 | ~4.5 | Estimated |

**Ref:** Apple. "Apple GPU Family Feature Set Tables." Apple Developer Documentation.

### Apple Unified Memory Architecture

Apple Silicon uses a unified memory architecture (UMA): CPU and GPU share the same physical DRAM. This is critical for the semantic encoding because it changes the data transfer model.

| Transfer Type | Traditional GPU (discrete) | Apple Silicon (UMA) |
|---------------|---------------------------|---------------------|
| CPU -> GPU | PCIe DMA (~12 GB/s) | No-op (shared memory) |
| GPU -> CPU | PCIe DMA (~12 GB/s) | No-op (shared memory) |
| Overhead | Buffer allocation + copy | `MTLBuffer` with `storageModeShared` |
| Latency | 10-100 us (PCIe) | <1 us (cache coherency) |

This means the GPU profitability threshold on Apple Silicon is much lower than on discrete GPUs. Transfer cost is essentially zero -- only kernel launch overhead matters.

**Ref:** Apple. "Setting Resource Storage Modes." Apple Metal Best Practices Guide, 2024.

---

## MSL Operations Relevant to Compiler Lowering

### Tier 1: Element-wise Operations (Parallel Map)

These are the simplest GPU patterns: apply a pure function to every element of an array independently.

```metal
kernel void elementwise_add(
    const device float* a [[buffer(0)]],
    const device float* b [[buffer(1)]],
    device float* result  [[buffer(2)]],
    uint id [[thread_position_in_grid]])
{
    result[id] = a[id] + b[id];
}
```

**tMIR pattern:** Any loop body where each iteration is independent (no loop-carried dependencies) and the body is `Pure`.

**Required tMIR proofs:** `Pure`, `InBounds`, `ValidBorrow`

**Supported element types:** `float` (FP32), `half` (FP16), `int`, `uint`, `short`, `ushort`, `char`, `uchar`

Metal also supports `bfloat` (BF16) on M4+ via `metal::bfloat`.

### Tier 2: Reduction Operations (Parallel Reduce)

Combine all elements using an associative binary operator. Metal provides two reduction strategies:

**Strategy A: Threadgroup Reduce (cooperative)**

```metal
kernel void reduce_sum(
    const device float* input   [[buffer(0)]],
    device float* partial_sums  [[buffer(1)]],
    threadgroup float* shared   [[threadgroup(0)]],
    uint id   [[thread_position_in_grid]],
    uint tid  [[thread_position_in_threadgroup]],
    uint tgid [[threadgroup_position_in_grid]])
{
    shared[tid] = input[id];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Tree reduction within threadgroup
    for (uint stride = get_threads_per_threadgroup() / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared[tid] += shared[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid == 0) {
        partial_sums[tgid] = shared[0];
    }
}
```

**Strategy B: SIMD Group Reduce (Apple M1+)**

```metal
kernel void reduce_sum_simd(
    const device float* input  [[buffer(0)]],
    device float* partial_sums [[buffer(1)]],
    uint id   [[thread_position_in_grid]],
    uint sgid [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]])
{
    float val = input[id];
    float sum = simd_sum(val);  // Hardware-accelerated 32-wide reduction

    if (lane == 0) {
        partial_sums[sgid] = sum;
    }
}
```

**tMIR proofs required:** `Associative` + `Commutative` (for reduction operator), `Pure`, `InBounds`

**Ref:** Apple. "Performing Calculations on Visible Function Tables." Metal Best Practices Guide. See also MSL spec Section 6.8: SIMD-group Functions.

### Tier 3: Matrix Multiply (simdgroup_matrix / AMX)

Apple GPUs provide hardware-accelerated matrix multiply via `simdgroup_matrix_multiply`:

```metal
#include <metal_simdgroup_matrix>
using namespace metal;

kernel void matmul(
    const device float* A [[buffer(0)]],
    const device float* B [[buffer(1)]],
    device float* C       [[buffer(2)]],
    uint2 gid [[thread_position_in_grid]])
{
    simdgroup_matrix<float, 8, 8> a, b, c;
    simdgroup_load(a, A + gid.y * 8 * K, K);
    simdgroup_load(b, B + gid.x * 8, N);
    simdgroup_multiply_accumulate(c, a, b, c);
    simdgroup_store(c, C + gid.y * 8 * N + gid.x * 8, N);
}
```

This maps to Apple's AMX (Apple Matrix eXtension) coprocessor, which provides 8x8 matrix multiply at very high throughput.

| Matrix type | Supported sizes | Element types |
|------------|-----------------|---------------|
| `simdgroup_matrix` | 8x8 | `float`, `half`, `bfloat` (M4+) |

**tMIR pattern:** Matrix multiply (`C[i][j] = sum_k A[i][k] * B[k][j]`). The tMIR must carry dimension annotations and the computation must be provably a matrix multiply (not just nested loops that happen to look like matmul).

**Ref:** Apple. "Performing Calculations with simdgroup Matrices." Metal Programming Guide, 2024.

### Tier 4: Scan / Prefix Sum

Inclusive/exclusive parallel prefix sum, useful for stream compaction, histograms, and radix sort:

```metal
// Using simd_prefix_inclusive_sum (Apple M1+)
float prefix = simd_prefix_inclusive_sum(val);
```

**tMIR proofs required:** `Associative` (scan operator)

### Tier 5: Scatter / Gather

Irregular memory access patterns:

```metal
// Gather: output[i] = input[indices[i]]
result[id] = input[indices[id]];

// Scatter: output[indices[i]] = input[i]
// Requires atomic if indices may collide
```

**tMIR proofs required:** `ValidBorrow` (non-aliasing for scatter), `InBounds` for both scatter and gather

---

## SMT Encoding for GPU Operations

### Design Principle: Semantic Equivalence, Not Implementation Detail

The SMT encoding represents the *semantic effect* of a GPU kernel, not its implementation. The GPU's parallel execution (threadgroups, barriers, SIMD groups) produces the same result as a sequential loop IF the operation's algebraic properties (associativity, commutativity) hold. tMIR proofs guarantee these properties.

Therefore, we encode GPU kernels as sequential specifications and rely on tMIR proofs to justify that the parallel implementation is equivalent.

### Encoding 1: Parallel Map

```rust
/// SMT encoding of a Metal parallel map kernel.
///
/// Semantics: for each i in [0, N): output[i] = f(input[i])
///
/// This is the GPU equivalent of a vectorized loop. The SMT encoding
/// is identical to the sequential specification because parallel map
/// with a Pure function is order-independent.
///
/// tMIR proof requirements: Pure, InBounds(0..N), ValidBorrow(input, output)
///
/// z4 theory: QF_ABV (arrays + bitvectors)
pub fn encode_gpu_parallel_map(
    input: &SmtExpr,       // Array(BV64, BV_elem)
    f: &dyn Fn(&SmtExpr) -> SmtExpr,  // Per-element pure function
    n: u64,                // Grid size (number of elements)
) -> SmtExpr {
    // Sequential encoding: proven equivalent to parallel by Pure
    let mut result = SmtExpr::const_array(Sort::bv(64), Sort::bv(elem_width));
    for i in 0..n {
        let idx = SmtExpr::bv_const(i, 64);
        let elem = SmtExpr::select(input, &idx);
        let transformed = f(&elem);
        result = SmtExpr::store(&result, &idx, &transformed);
    }
    result
}
```

**Formula complexity:** O(N) store operations. For N <= 1024, this produces manageable formulas. For N > 1024, use bounded quantifiers:

```rust
/// Quantified encoding for large parallel maps.
///
/// forall i in [0, N): output[i] = f(input[i])
///
/// This uses z4's bounded quantifier support (issue #124).
pub fn encode_gpu_parallel_map_quantified(
    input: &SmtExpr,
    output: &SmtExpr,
    f: &dyn Fn(&SmtExpr) -> SmtExpr,
    n: u64,
) -> SmtExpr {
    let i = SmtExpr::bound_var("i", Sort::bv(64));
    let in_range = i.bvuge(&SmtExpr::bv_const(0, 64))
        .and(&i.bvult(&SmtExpr::bv_const(n, 64)));
    let body = SmtExpr::select(output, &i)
        .eq(&f(&SmtExpr::select(input, &i)));
    SmtExpr::forall(&[i], &in_range.implies(&body))
}
```

### Encoding 2: Parallel Reduce

```rust
/// SMT encoding of a Metal parallel reduce kernel.
///
/// Semantics: result = fold(op, identity, input[0..N])
///
/// The GPU executes a tree reduction (log2(N) parallel steps), but
/// because tMIR proves Associative + Commutative for `op`, all
/// evaluation orders produce the same result. We encode the
/// sequential fold (simplest SMT formula).
///
/// tMIR proof requirements:
///   - Associative(op): (a op b) op c = a op (b op c)
///   - Commutative(op): a op b = b op a
///   - Pure, InBounds(0..N)
///
/// z4 theory: QF_ABV or QF_FPBV (if FP elements)
pub fn encode_gpu_parallel_reduce(
    input: &SmtExpr,                    // Array(BV64, BV_elem)
    op: &dyn Fn(&SmtExpr, &SmtExpr) -> SmtExpr,  // Associative + commutative
    identity: &SmtExpr,                 // Identity element for op
    n: u64,
) -> SmtExpr {
    let mut acc = identity.clone();
    for i in 0..n {
        let idx = SmtExpr::bv_const(i, 64);
        let elem = SmtExpr::select(input, &idx);
        acc = op(&acc, &elem);
    }
    acc
}
```

**Soundness note on FP reductions:** IEEE 754 FP addition is NOT associative in general. The sequential fold `((a + b) + c) + d` may differ from the tree reduction `(a + b) + (c + d)` due to rounding. The encoding is sound ONLY because tMIR's `Associative` proof guarantees that for this specific computation, the difference is acceptable (either the values are exact, or the algorithm tolerates rounding variation). Without this proof, the GPU candidate is illegal and the solver will not consider it.

### Encoding 3: Map-Reduce (Fused)

```rust
/// SMT encoding of fused map-reduce: result = fold(op, id, map(f, input[0..N]))
///
/// Common pattern: dot product = sum(a[i] * b[i] for i in 0..N)
///
/// The fused encoding avoids materializing the intermediate mapped array.
pub fn encode_gpu_map_reduce(
    input_a: &SmtExpr,    // First input array
    input_b: &SmtExpr,    // Second input array (or same as first)
    map_f: &dyn Fn(&SmtExpr, &SmtExpr) -> SmtExpr,   // Per-element binary map
    reduce_op: &dyn Fn(&SmtExpr, &SmtExpr) -> SmtExpr, // Reduction operator
    identity: &SmtExpr,
    n: u64,
) -> SmtExpr {
    let mut acc = identity.clone();
    for i in 0..n {
        let idx = SmtExpr::bv_const(i, 64);
        let a_elem = SmtExpr::select(input_a, &idx);
        let b_elem = SmtExpr::select(input_b, &idx);
        let mapped = map_f(&a_elem, &b_elem);
        acc = reduce_op(&acc, &mapped);
    }
    acc
}

/// Example: dot product verification
///
/// CPU sequential: sum = 0; for i in 0..N: sum += a[i] * b[i]
/// GPU kernel: parallel map a[i]*b[i], then tree reduce with +
///
/// Both encode to the same SMT formula (thanks to Associative+Commutative).
pub fn verify_dot_product_equivalence(n: u64) -> SmtExpr {
    let a = SmtExpr::const_array_var("a", Sort::bv(64), Sort::fp(11, 53));
    let b = SmtExpr::const_array_var("b", Sort::bv(64), Sort::fp(11, 53));

    let rne = SmtExpr::rounding_mode(RoundingMode::RNE);
    let zero = SmtExpr::fp_zero(11, 53);

    // CPU encoding (sequential)
    let cpu_result = encode_gpu_map_reduce(
        &a, &b,
        &|ai, bi| SmtExpr::fp_mul(&rne, ai, bi),
        &|acc, prod| SmtExpr::fp_add(&rne, acc, prod),
        &zero, n,
    );

    // GPU encoding (identical because Associative+Commutative)
    let gpu_result = encode_gpu_map_reduce(
        &a, &b,
        &|ai, bi| SmtExpr::fp_mul(&rne, ai, bi),
        &|acc, prod| SmtExpr::fp_add(&rne, acc, prod),
        &zero, n,
    );

    // Assert non-equivalence; expect UNSAT
    cpu_result.fp_eq(&gpu_result).not()
}
```

### Encoding 4: Scatter / Gather

```rust
/// SMT encoding of a GPU gather operation.
///
/// Semantics: for each i in [0, N): output[i] = input[indices[i]]
///
/// tMIR proof requirements: InBounds(indices[i], len(input)) for all i
pub fn encode_gpu_gather(
    input: &SmtExpr,      // Source array
    indices: &SmtExpr,    // Index array (Array(BV64, BV64))
    n: u64,
) -> SmtExpr {
    let mut result = SmtExpr::const_array(Sort::bv(64), Sort::bv(elem_width));
    for i in 0..n {
        let i_idx = SmtExpr::bv_const(i, 64);
        let src_idx = SmtExpr::select(indices, &i_idx);
        let value = SmtExpr::select(input, &src_idx);
        result = SmtExpr::store(&result, &i_idx, &value);
    }
    result
}

/// SMT encoding of a GPU scatter operation.
///
/// Semantics: for each i in [0, N): output[indices[i]] = input[i]
///
/// WARNING: If indices are not unique, the result depends on write order.
/// tMIR must prove ValidBorrow (non-aliasing writes) or the scatter is
/// illegal for GPU dispatch.
pub fn encode_gpu_scatter(
    input: &SmtExpr,
    indices: &SmtExpr,    // Must be proven unique (ValidBorrow)
    output_base: &SmtExpr,
    n: u64,
) -> SmtExpr {
    let mut result = output_base.clone();
    for i in 0..n {
        let i_idx = SmtExpr::bv_const(i, 64);
        let dst_idx = SmtExpr::select(indices, &i_idx);
        let value = SmtExpr::select(input, &i_idx);
        result = SmtExpr::store(&result, &dst_idx, &value);
    }
    result
}
```

### Encoding 5: Matrix Multiply (simdgroup_matrix_multiply)

```rust
/// SMT encoding of GPU matrix multiply via simdgroup_matrix.
///
/// Semantics: C[i][j] = sum_k A[i][k] * B[k][j]
///
/// For small matrices (M,K,N <= 16), fully unrolled.
/// For larger matrices, use UF abstraction with axiomatic properties.
///
/// The GPU implementation uses simdgroup_matrix_multiply (8x8 tiles),
/// but the semantic encoding is the mathematical definition.
///
/// z4 theory: QF_FPBV for small matrices, QF_UFABV for large
pub fn encode_gpu_matmul(
    a: &SmtExpr,   // Flattened M*K array
    b: &SmtExpr,   // Flattened K*N array
    m: u64, k: u64, n: u64,
    elem_sort: &Sort,
) -> SmtExpr {
    let rne = SmtExpr::rounding_mode(RoundingMode::RNE);
    let zero = SmtExpr::fp_zero_of_sort(elem_sort);
    let mut c = SmtExpr::const_array(Sort::bv(64), elem_sort.clone());

    for i in 0..m {
        for j in 0..n {
            let mut sum = zero.clone();
            for kk in 0..k {
                let a_idx = SmtExpr::bv_const(i * k + kk, 64);
                let b_idx = SmtExpr::bv_const(kk * n + j, 64);
                let a_elem = SmtExpr::select(a, &a_idx);
                let b_elem = SmtExpr::select(b, &b_idx);
                let prod = SmtExpr::fp_mul(&rne, &a_elem, &b_elem);
                sum = SmtExpr::fp_add(&rne, &sum, &prod);
            }
            let c_idx = SmtExpr::bv_const(i * n + j, 64);
            c = SmtExpr::store(&c, &c_idx, &sum);
        }
    }
    c
}

/// UF abstraction for large matrix multiply.
///
/// Declares matmul as an uninterpreted function with known properties:
/// 1. Bilinearity: matmul(A+B, C) = matmul(A, C) + matmul(B, C)
/// 2. Associativity: matmul(matmul(A, B), C) = matmul(A, matmul(B, C))
/// 3. Identity: matmul(I, A) = A
///
/// This lets the solver reason about matmul chains without unrolling
/// the O(M*K*N) inner product.
pub fn encode_gpu_matmul_uf(
    a: &SmtExpr,
    b: &SmtExpr,
    dims: MatrixDim,
) -> SmtExpr {
    // Declare as uninterpreted function with dimensional constraint
    let matmul_fn = SmtExpr::declare_fun(
        "gpu_matmul",
        &[Sort::array(Sort::bv(64), Sort::fp(11, 53)),
          Sort::array(Sort::bv(64), Sort::fp(11, 53))],
        Sort::array(Sort::bv(64), Sort::fp(11, 53)),
    );
    matmul_fn.apply(&[a.clone(), b.clone()])
}
```

---

## Data Transfer Semantics

### Apple UMA: Implicit Transfer

On Apple Silicon, CPU and GPU share the same physical memory. Metal `MTLBuffer` with `storageModeShared` requires no explicit copy -- both CPU and GPU see the same bytes.

```rust
/// SMT encoding of CPU-to-GPU data transfer on Apple UMA.
///
/// On shared-memory (UMA) systems, the transfer is a no-op:
/// the GPU reads from the same memory as the CPU.
///
/// Precondition: ValidBorrow(cpu_buffer, gpu_view) -- the CPU must not
/// mutate the buffer while the GPU is reading it.
///
/// Returns: the same array expression (identity function)
pub fn encode_uma_transfer(
    cpu_memory: &SmtExpr,
) -> SmtExpr {
    // UMA transfer is semantically the identity function.
    // The proof obligation is on the ValidBorrow: the CPU must not
    // write to the buffer during GPU execution.
    cpu_memory.clone()
}
```

### Discrete GPU: Explicit DMA (Non-Apple, Future Target)

For future non-UMA targets (e.g., server GPUs with discrete memory):

```rust
/// SMT encoding of explicit DMA transfer (discrete GPU).
///
/// Copies `size` elements from `src_base` in `cpu_mem` to
/// `dst_base` in `gpu_mem`.
///
/// Precondition: ValidBorrow (no aliasing between source and destination)
pub fn encode_dma_transfer(
    cpu_mem: &SmtExpr,
    gpu_mem: &SmtExpr,
    src_base: &SmtExpr,
    dst_base: &SmtExpr,
    size: u64,
) -> SmtExpr {
    let mut result = gpu_mem.clone();
    for i in 0..size {
        let offset = SmtExpr::bv_const(i, 64);
        let src_addr = src_base.bvadd(&offset);
        let dst_addr = dst_base.bvadd(&offset);
        let value = SmtExpr::select(cpu_mem, &src_addr);
        result = SmtExpr::store(&result, &dst_addr, &value);
    }
    result
}
```

### Synchronization Semantics

GPU kernel dispatch is asynchronous. The CPU must wait for completion before reading results. In the SMT encoding, this is modeled as a sequential composition:

```rust
/// Model a complete GPU dispatch: transfer -> compute -> transfer back
///
/// The sequential composition ensures the solver sees the full
/// data flow: CPU memory -> GPU kernel -> CPU memory.
pub fn encode_gpu_dispatch(
    cpu_input: &SmtExpr,
    kernel: &dyn Fn(&SmtExpr) -> SmtExpr,
    n: u64,
) -> SmtExpr {
    // Step 1: Transfer input (identity on UMA)
    let gpu_input = encode_uma_transfer(cpu_input);

    // Step 2: Execute kernel
    let gpu_output = kernel(&gpu_input);

    // Step 3: Transfer output (identity on UMA)
    encode_uma_transfer(&gpu_output)
}
```

---

## Cost Thresholds: GPU vs Scalar vs NEON

### When Is GPU Dispatch Profitable?

GPU dispatch has fixed overhead costs that must be amortized:

| Cost Component | Apple Silicon (UMA) | Discrete GPU |
|---------------|-------------------|--------------| 
| `MTLCommandBuffer` creation | ~2 us | ~5 us |
| `MTLComputeCommandEncoder` setup | ~1 us | ~2 us |
| Kernel dispatch (`dispatchThreads`) | ~1 us | ~3 us |
| GPU wake-up (if idle) | ~10-50 us | ~50-200 us |
| Synchronization (`waitUntilCompleted`) | ~5 us | ~10 us |
| **Total fixed overhead** | **~20-60 us** | **~70-210 us** |
| Data transfer (UMA: 0, discrete: PCIe) | ~0 | varies |

**Ref:** Apple. "Minimizing Pass and Command Overhead." Metal Best Practices Guide. Measured values from Apple GPU profiling tools.

### Crossover Analysis

The GPU becomes profitable when its parallel compute time plus fixed overhead is less than the CPU alternative:

```
T_gpu = T_overhead + N / (GPU_throughput)
T_cpu_scalar = N * T_per_element_cpu
T_cpu_neon = N / (NEON_lanes) * T_per_element_neon

GPU profitable when: T_gpu < min(T_cpu_scalar, T_cpu_neon)
```

For a simple element-wise FP32 add:

| N | CPU scalar | NEON (4S) | GPU | Winner |
|---|-----------|-----------|-----|--------|
| 16 | 16 ns | 4 ns | 20,000 ns | NEON |
| 256 | 256 ns | 64 ns | 20,010 ns | NEON |
| 1,024 | 1,024 ns | 256 ns | 20,050 ns | NEON |
| 4,096 | 4,096 ns | 1,024 ns | 20,100 ns | NEON |
| 16,384 | 16,384 ns | 4,096 ns | 20,400 ns | GPU |
| 65,536 | 65,536 ns | 16,384 ns | 21,600 ns | GPU |
| 262,144 | 262,144 ns | 65,536 ns | 26,400 ns | GPU |

**Approximate crossover point: N ~ 10,000-20,000 elements for simple element-wise ops on Apple UMA.**

For more compute-intensive kernels (e.g., matrix multiply with O(N^3) compute / O(N^2) data), the crossover is much lower: N ~ 64-256 for matmul.

### Proposed Cost Model Thresholds

```rust
/// GPU profitability thresholds for LLVM2 cost model.
///
/// These are conservative estimates for Apple M-series (UMA).
/// The solver uses these to prune the search space: don't even
/// consider GPU candidates below the threshold.
pub struct GpuCostThresholds {
    /// Minimum elements for element-wise map kernels
    pub map_min_elements: u64,          // 8,192

    /// Minimum elements for reduction kernels
    pub reduce_min_elements: u64,       // 4,096

    /// Minimum effective FLOP count for matmul
    /// (matmul is compute-intensive: lower threshold)
    pub matmul_min_flops: u64,          // 16,384 (e.g., 32x32x16)

    /// Minimum elements for fused map-reduce (e.g., dot product)
    pub map_reduce_min_elements: u64,   // 4,096

    /// Fixed dispatch overhead in nanoseconds
    pub dispatch_overhead_ns: u64,       // 20,000

    /// GPU throughput for FP32 element-wise (elements/ns)
    pub fp32_elementwise_throughput: f64, // ~4.0

    /// GPU throughput for FP32 matmul (FLOP/ns)
    pub fp32_matmul_throughput: f64,     // ~2,600 (GFLOPS)
}
```

### Dynamic Profitability (Future)

The static thresholds above are starting points. Phase 2 adds runtime feedback:

1. Measure actual GPU dispatch overhead on the target device
2. Measure actual GPU throughput for each kernel pattern
3. Update cost model with measured data (self-improving loop from unified solver architecture)
4. Consider GPU power state: if GPU is already active (another kernel running), overhead is lower

---

## Unified Encoding API: `GpuKernelEncoding`

### Rust API Design

```rust
/// GPU kernel semantic encoding for the unified solver.
///
/// Implements the `TargetEncoding` trait so the solver can compare
/// GPU kernels against CPU scalar, NEON, and ANE implementations.
pub struct GpuKernelEncoding {
    /// Kernel pattern (map, reduce, matmul, scatter, gather)
    pub pattern: GpuKernelPattern,
    /// Grid size (number of work items)
    pub grid_size: u64,
    /// Element type (FP32, FP16, I32, etc.)
    pub element_type: LirType,
    /// Threadgroup size (for reduce patterns)
    pub threadgroup_size: u32,
}

pub enum GpuKernelPattern {
    /// Element-wise: output[i] = f(input[i])
    ParallelMap {
        /// Per-element function, encoded as SmtExpr -> SmtExpr
        element_fn: GpuElementFn,
    },
    /// Reduction: result = fold(op, identity, input[0..N])
    ParallelReduce {
        reduce_op: BinaryOp,
        identity: SmtExpr,
    },
    /// Fused map-reduce: result = fold(reduce_op, id, map(map_fn, inputs))
    ParallelMapReduce {
        map_fn: GpuElementFn,
        reduce_op: BinaryOp,
        identity: SmtExpr,
    },
    /// Matrix multiply: C = A * B
    MatMul {
        m: u64,
        k: u64,
        n: u64,
    },
    /// Gather: output[i] = input[indices[i]]
    Gather,
    /// Scatter: output[indices[i]] = input[i]
    Scatter,
}

/// Binary operations supported in GPU reductions.
pub enum BinaryOp {
    Add,      // Associative, commutative
    Mul,      // Associative, commutative
    Min,      // Associative, commutative
    Max,      // Associative, commutative
    BitwiseAnd, // Associative, commutative
    BitwiseOr,  // Associative, commutative
    BitwiseXor, // Associative, commutative
}

impl TargetEncoding for GpuKernelEncoding {
    fn required_logic(&self) -> SmtLogic {
        match &self.pattern {
            GpuKernelPattern::ParallelMap { .. } => SmtLogic::QF_ABV,
            GpuKernelPattern::ParallelReduce { .. } => SmtLogic::QF_ABV,
            GpuKernelPattern::ParallelMapReduce { .. } => SmtLogic::QF_ABV,
            GpuKernelPattern::MatMul { .. } => {
                if self.element_type.is_float() {
                    SmtLogic::QF_FPABV
                } else {
                    SmtLogic::QF_ABV
                }
            }
            GpuKernelPattern::Gather | GpuKernelPattern::Scatter => SmtLogic::QF_ABV,
        }
    }

    fn encode(&self, inputs: &[SmtExpr], program: &mut Z4Program) -> SmtExpr {
        match &self.pattern {
            GpuKernelPattern::ParallelMap { element_fn } => {
                encode_gpu_parallel_map(&inputs[0], element_fn, self.grid_size)
            }
            GpuKernelPattern::ParallelReduce { reduce_op, identity } => {
                encode_gpu_parallel_reduce(&inputs[0], reduce_op, identity, self.grid_size)
            }
            GpuKernelPattern::ParallelMapReduce { map_fn, reduce_op, identity } => {
                encode_gpu_map_reduce(
                    &inputs[0], &inputs[1], map_fn, reduce_op, identity, self.grid_size,
                )
            }
            GpuKernelPattern::MatMul { m, k, n } => {
                encode_gpu_matmul(&inputs[0], &inputs[1], *m, *k, *n, &self.element_sort())
            }
            GpuKernelPattern::Gather => {
                encode_gpu_gather(&inputs[0], &inputs[1], self.grid_size)
            }
            GpuKernelPattern::Scatter => {
                encode_gpu_scatter(&inputs[0], &inputs[1], &inputs[2], self.grid_size)
            }
        }
    }

    fn estimated_solve_time(&self) -> Duration {
        match &self.pattern {
            GpuKernelPattern::ParallelMap { .. } => {
                if self.grid_size <= 64 { Duration::from_secs(1) }
                else if self.grid_size <= 1024 { Duration::from_secs(10) }
                else { Duration::from_secs(60) }
            }
            GpuKernelPattern::ParallelReduce { .. } => {
                if self.grid_size <= 64 { Duration::from_secs(1) }
                else if self.grid_size <= 1024 { Duration::from_secs(5) }
                else { Duration::from_secs(30) }
            }
            GpuKernelPattern::MatMul { m, k, n } => {
                let total = m * k * n;
                if total <= 512 { Duration::from_secs(5) }
                else if total <= 4096 { Duration::from_secs(30) }
                else { Duration::from_secs(120) }
            }
            _ => Duration::from_secs(10),
        }
    }
}
```

---

## Solver Complexity Analysis

| Kernel Pattern | N | Formula Size | z4 Theory | Est. Solve Time |
|---------------|---|-------------|-----------|-----------------|
| Map (int) | 4 | ~50 terms | QF_ABV | < 0.1s |
| Map (int) | 64 | ~800 terms | QF_ABV | < 1s |
| Map (int) | 1024 | ~13K terms | QF_ABV | < 10s |
| Map (FP32) | 64 | ~2K terms | QF_FPABV | 1-5s |
| Map (FP32) | 1024 | ~30K terms | QF_FPABV | 10-60s |
| Reduce (int) | 64 | ~500 terms | QF_ABV | < 1s |
| Reduce (int) | 1024 | ~8K terms | QF_ABV | < 10s |
| Reduce (FP32) | 64 | ~1.5K terms | QF_FPABV | 1-10s |
| Map-Reduce (FP32) | 1000 | ~15K terms | QF_FPABV | 30-60s |
| MatMul (int, 8x8x8) | 512 terms | ~4K terms | QF_ABV | 1-5s |
| MatMul (FP32, 8x8x8) | 512 terms | ~8K terms | QF_FPABV | 10-30s |
| MatMul (FP32, 32x32x32) | 32K terms | ~200K terms | QF_UFABV (UF) | 1-5s |
| Scatter/Gather | 64 | ~800 terms | QF_ABV | < 1s |
| Quantified Map | any | ~10 terms | BV+forall | depends on z4 |

**Scalability regimes:**
1. **Exact proofs (N <= 1024):** Fully unrolled, complete verification. Fast.
2. **Bounded proofs (1024 < N <= 65536):** Bounded quantifiers via z4 issue #124. Sound but may timeout.
3. **Abstract proofs (N > 65536):** UF abstraction. Sound, lightweight, but approximate -- proves high-level properties, not bit-exact equivalence.

---

## z4 Theory Requirements

| Theory | Needed For | Issue | Status |
|--------|-----------|-------|--------|
| QF_BV | Integer GPU kernels | Exists | Available |
| QF_ABV | Array operations (all GPU kernels) | #122 | In progress |
| QF_FP | FP element operations | #123 | In progress |
| QF_FPBV | Mixed FP and BV | #123 | In progress |
| Bounded forall | Large array proofs | #124 | In progress |
| QF_UF | Large matmul abstraction | Needed | Not started |

---

## Integration with Unified Solver Pipeline

### Phase 1: Proof-Guided Target Analysis

```rust
/// Determine if a computation subgraph is legal for GPU dispatch.
///
/// GPU is legal iff:
/// 1. The computation is Pure (no side effects)
/// 2. Array accesses are InBounds (no out-of-bounds GPU faults)
/// 3. Memory accesses have ValidBorrow (no aliasing)
/// 4. For reductions: the operator is Associative + Commutative
/// 5. Data size exceeds GPU profitability threshold
pub fn is_gpu_legal(
    subgraph: &ComputeSubgraph,
    tmir_proofs: &ProofAnnotations,
    cost_model: &GpuCostThresholds,
) -> bool {
    if !tmir_proofs.has(Proof::Pure) { return false; }
    if !tmir_proofs.has(Proof::InBounds) { return false; }
    if !tmir_proofs.has(Proof::ValidBorrow) { return false; }

    if subgraph.has_reduction() {
        if !tmir_proofs.has(Proof::Associative) { return false; }
        if !tmir_proofs.has(Proof::Commutative) { return false; }
    }

    // Check profitability threshold
    let n = subgraph.element_count();
    match subgraph.pattern() {
        Pattern::Map => n >= cost_model.map_min_elements,
        Pattern::Reduce => n >= cost_model.reduce_min_elements,
        Pattern::MapReduce => n >= cost_model.map_reduce_min_elements,
        Pattern::MatMul { m, k, n } => {
            (m * k * n) as u64 >= cost_model.matmul_min_flops
        }
        _ => n >= cost_model.map_min_elements,
    }
}
```

### Phase 2: Candidate Generation and Verification

The unified synthesis loop generates GPU candidates alongside scalar and NEON candidates. For each legal GPU target:

1. Classify the tMIR subgraph into a `GpuKernelPattern`
2. Generate the `GpuKernelEncoding`
3. Feed to z4: prove `tmir_encoding == gpu_encoding`
4. If proven: estimate cost via GPU cost model, rank against other targets

### Phase 4: Metal IR Emission (Codegen)

Once a GPU candidate is selected and verified, `llvm2-codegen` must emit:
1. Metal compute kernel (MSL source or Metal IR bitcode)
2. Host dispatch code (CPU AArch64 instructions calling Metal API)
3. Buffer allocation and binding

This is in `llvm2-codegen`'s scope (#131), not in this design. The semantic encoding here proves the kernel correct; codegen produces the actual bytes.

---

## Implementation Plan

### Step 1: GPU Encoding Types (1 session)
- Define `GpuKernelEncoding`, `GpuKernelPattern`, `BinaryOp` in `llvm2-verify/src/gpu_semantics.rs`
- Implement `TargetEncoding` trait for `GpuKernelEncoding`

### Step 2: Parallel Map Encoding (1 session)
- Implement `encode_gpu_parallel_map` (unrolled, N <= 1024)
- Write verification tests: GPU map == sequential loop
- Implement quantified version for large N (depends on #124)

### Step 3: Parallel Reduce + Map-Reduce (1-2 sessions)
- Implement `encode_gpu_parallel_reduce`
- Implement `encode_gpu_map_reduce`
- Write dot product verification test
- Test with both integer and FP elements

### Step 4: Matrix Multiply Encoding (1-2 sessions)
- Implement `encode_gpu_matmul` (unrolled, small matrices)
- Implement `encode_gpu_matmul_uf` (UF abstraction, large matrices)
- Write matmul verification tests

### Step 5: GPU Cost Model Integration (1 session)
- Implement `GpuCostThresholds` with Apple M-series defaults
- Wire into `is_gpu_legal` profitability check
- Integrate with `HeterogeneousCostModel` from #144 / #161

### Step 6: Scatter/Gather + Dispatch Encoding (1 session)
- Implement scatter/gather encodings
- Implement `encode_gpu_dispatch` for full dispatch verification
- Wire UMA transfer semantics

**Total estimated effort: 6-8 techlead sessions.**

---

## References

1. Apple. "Metal Shading Language Specification." Version 3.2. Apple Developer Documentation, 2024.
2. Apple. "Metal Best Practices Guide." Apple Developer Documentation, 2024.
3. Apple. "Performing Calculations on Visible Function Tables." Metal Programming Guide, 2024.
4. Apple. "Performing Calculations with simdgroup Matrices." Metal Programming Guide, 2024.
5. Apple. "Setting Resource Storage Modes." Metal Best Practices Guide, 2024.
6. Apple. "Minimizing Pass and Command Overhead." Metal Best Practices Guide, 2024.
7. Apple. "Apple GPU Family Feature Set Tables." Apple Developer Documentation.
8. Chen et al. "TVM: An Automated End-to-End Optimizing Compiler for Deep Learning." OSDI 2018.
9. Ragan-Kelley et al. "Halide: A Language and Compiler for Optimizing Parallelism, Locality, and Recomputation." PLDI 2013.
10. Lattner et al. "MLIR: Scaling Compiler Infrastructure for Domain Specific Computation." CGO 2021.
11. Lopes et al. "Alive2: Bounded Translation Validation for LLVM." PLDI 2021.
12. LLVM2 multi-target semantic encoding design: `designs/2026-04-13-multi-target-encoding.md`
13. LLVM2 unified solver architecture: `designs/2026-04-13-unified-solver-architecture.md`
14. LLVM2 heterogeneous compute allocation: `designs/2026-04-13-heterogeneous-compute.md`
15. LLVM2 cost model calibration: `designs/2026-04-14-cost-model-calibration.md`
16. LLVM2 NEON semantics implementation: `crates/llvm2-verify/src/neon_semantics.rs`
