# GPU/ANE Profitability Cost Thresholds and Target Legality Checks

**Author:** Andrew Yates
**Date:** 2026-04-14
**Status:** Design
**Part of:** #171 (GPU/ANE profitability cost thresholds and target legality checks)
**Related:** #144 (Heterogeneous cost model), #161 (Multi-target cost model), `designs/2026-04-13-heterogeneous-compute.md`, `designs/2026-04-14-cost-model-calibration.md`, `designs/2026-04-14-ane-semantics.md`

---

## Problem

The current cost model (`crates/llvm2-ir/src/cost_model.rs`) provides per-operation cost estimates for each compute target (CPU scalar, NEON, GPU, ANE), and the compute graph (`crates/llvm2-lower/src/compute_graph.rs`) recommends targets based on lowest latency among legal targets. However, the system lacks:

1. **Profitability thresholds** -- minimum data sizes below which accelerator dispatch overhead exceeds compute savings.
2. **Target legality checks** -- which operations each target actually supports, beyond proof-based legality.
3. **Data transfer cost accounting** -- a decision model that considers end-to-end cost including marshalling.
4. **Apple Silicon specifics** -- unified memory advantages, ANE constraints, GPU thread group sizing.

Without these, the system may dispatch tiny workloads to GPU (wasting 10K+ cycles on dispatch overhead) or attempt ANE operations that the hardware does not support (triggering silent CPU fallback with wasted compilation time).

---

## Target Legality: What Each Target Supports

### CPU Scalar (AArch64)

**Supports:** All operations. The CPU is the universal fallback.

| Category | Operations | Notes |
|----------|-----------|-------|
| Integer arithmetic | add, sub, mul, sdiv, udiv, neg | Full 8/16/32/64-bit |
| Bitwise | and, or, xor, bic, orn, shifts | |
| FP arithmetic | fadd, fsub, fmul, fdiv, fma | FP32 and FP64 |
| FP conversion | fcvt, scvtf, ucvtf | All conversions |
| Memory | load, store, atomic | Full memory model |
| Control flow | branch, call, return | |
| Comparison | all | |

**Constraints:** None. Always legal.

### NEON (ARM Advanced SIMD)

**Supports:** Data-parallel operations on 64-bit or 128-bit vectors.

| Category | Operations | Arrangements | Notes |
|----------|-----------|-------------|-------|
| Integer add/sub | ADD, SUB, NEG | 8B/16B/4H/8H/2S/4S/2D | |
| Integer multiply | MUL | 8B/16B/4H/8H/2S/4S | No 64-bit multiply |
| Bitwise | AND, ORR, EOR, BIC | 8B/16B | Byte-wise only |
| Shifts | SHL, USHR, SSHR | All | Immediate only for some |
| FP add/sub | FADD, FSUB | 2S/4S/2D | FP32 and FP64 |
| FP multiply | FMUL, FMLA | 2S/4S/2D | Includes FMA |
| Min/max | SMIN, SMAX, UMIN, UMAX | All | |
| Comparison | CMEQ, CMGT, CMGE | All | Returns mask |
| Reduction | ADDV, SADDLV | Limited | Horizontal only |

**Constraints:**
- No integer division (must use scalar or reciprocal approximation)
- No FP division (FDIV exists but is slow, prefer reciprocal + Newton)
- Vector width must be 64 or 128 bits
- Data must be naturally aligned for best performance

**Legality check:**
```rust
pub fn is_neon_legal(op: &str, element_bits: u32) -> bool {
    match (op.to_ascii_uppercase().as_str(), element_bits) {
        ("MUL", 64) => false,       // No 64-bit integer vector MUL
        ("SDIV" | "UDIV" | "DIV", _) => false,  // No vector division
        (_, bits) if bits > 64 => false,  // No elements wider than 64 bits
        _ => true,
    }
}
```

### GPU (Metal Compute Shaders)

**Supports:** Arbitrary computation expressed as Metal Shading Language kernels.

| Category | Operations | Data Types | Notes |
|----------|-----------|-----------|-------|
| Integer arithmetic | all | int, uint, short, char | |
| FP arithmetic | all | float, half, float3/4 | |
| Matrix multiply | simdgroup_matrix_multiply | float, half | M1+ hardware matrix multiply |
| Atomic | all | int, uint, float | Shared/device memory |
| Texture | sample, read, write | Various | Fixed-function texture units |
| Threadgroup | barrier, shuffle, reduce | | Cooperative operations |

**Constraints (legality):**
- Maximum threads per threadgroup: 1024
- Maximum threadgroup memory: 32KB (M1), 64KB (M4)
- No recursion in shaders
- No dynamic allocation (all buffers must be pre-allocated)
- Metal Shading Language subset (no full C++ features)
- Thread divergence reduces SIMD utilization

**Legality check:**
```rust
pub fn is_gpu_legal(
    op: &str,
    data_size_bytes: u64,
    has_side_effects: bool,
    requires_proof: bool,
) -> GpuLegality {
    // GPU requires Pure proof (no side effects)
    if has_side_effects && !requires_proof {
        return GpuLegality::Illegal("side effects not proven absent");
    }

    // GPU requires static data shape
    // (dynamic shapes need runtime buffer allocation)

    // All standard arithmetic operations are legal on GPU
    GpuLegality::Legal
}
```

### ANE (Apple Neural Engine)

**Supports:** Fixed-function tensor operations only.

| Category | Operations | Precision | Constraints |
|----------|-----------|-----------|-------------|
| GEMM | matmul, batched_matmul | FP16, INT8 | Static shapes, <= 4D |
| Convolution | conv1d, conv2d, depthwise, transposed | FP16, INT8 | Static kernel |
| Pooling | max_pool, avg_pool, global_pool | FP16 | 2D only |
| Normalization | batch_norm, layer_norm, group_norm | FP16 | Fused with conv |
| Activation | relu, leaky_relu, sigmoid, tanh, gelu, silu | FP16 | Fused preferred |
| Element-wise | add, sub, mul, div | FP16 | Tensor-tensor only |
| Reshape | reshape, transpose, permute, concat, split | N/A | Data movement |
| Reduction | sum, mean, max, min | FP16 | Along specified axes |
| Attention | scaled_dot_product (M4+) | FP16, BF16 | Hardware unit |

**Operations NOT supported (fall back to CPU):**
- General integer arithmetic (non-tensor)
- Scatter/gather with irregular indices
- Control flow with data-dependent bounds
- Custom activation functions
- Dynamic shapes (some operations)
- FP32/FP64 operations (ANE is FP16/INT8 only)
- Arbitrary user-defined operations

**Legality check:**
```rust
pub fn is_ane_legal(
    op: &str,
    tensor_dims: u32,
    element_precision: Precision,
    has_static_shape: bool,
) -> AneLegality {
    // ANE requires static shapes
    if !has_static_shape {
        return AneLegality::Illegal("dynamic shapes not supported");
    }

    // ANE requires <= 4D tensors
    if tensor_dims > 4 {
        return AneLegality::Illegal("tensor > 4 dimensions");
    }

    // ANE requires FP16 or INT8 (BF16 on M4+)
    match element_precision {
        Precision::FP16 | Precision::INT8 => {}
        Precision::BF16 => {
            // Only legal on M4+; M1-M3 will silently fall back to CPU
        }
        Precision::FP32 | Precision::FP64 => {
            return AneLegality::RequiresQuantization {
                source: element_precision,
                target: Precision::FP16,
            };
        }
        _ => return AneLegality::Illegal("unsupported precision"),
    }

    // Check operation support
    let op_upper = op.to_ascii_uppercase();
    match op_upper.as_str() {
        "GEMM" | "MATMUL" | "CONV1D" | "CONV2D" | "DEPTHWISE_CONV2D"
        | "TRANSPOSED_CONV2D" | "MAXPOOL" | "AVGPOOL" | "GLOBAL_AVGPOOL"
        | "GLOBAL_MAXPOOL" | "BATCH_NORM" | "LAYER_NORM" | "GROUP_NORM"
        | "RELU" | "LEAKY_RELU" | "SIGMOID" | "TANH" | "GELU" | "SILU"
        | "ADD" | "SUB" | "MUL" | "DIV"
        | "REDUCE_SUM" | "REDUCE_MEAN" | "REDUCE_MAX" | "REDUCE_MIN"
        | "RESHAPE" | "TRANSPOSE" | "PERMUTE" | "CONCAT" | "SPLIT"
        => AneLegality::Legal,

        "ATTENTION" | "SCALED_DOT_PRODUCT_ATTENTION"
        => AneLegality::LegalOnGeneration(CostModelGen::M4),

        _ => AneLegality::Illegal("operation not supported on ANE"),
    }
}
```

---

## Profitability Thresholds

### The Core Equation

For accelerator dispatch to be profitable:

```
cost_accelerator + cost_transfer < cost_baseline

where:
  cost_accelerator = dispatch_overhead + compute_time(data_size)
  cost_transfer    = marshal_to + marshal_from
  cost_baseline    = cpu_compute_time(data_size)  [or NEON for vector ops]
```

This simplifies to a minimum data size threshold:

```
data_size_min = dispatch_overhead / (baseline_throughput - accelerator_throughput)
```

### GPU Profitability

**Dispatch overhead** (from `cost_model.rs`, lines 797-798): ~10,000 CPU-equivalent cycles. This includes Metal command buffer encoding, GPU scheduling, and synchronization fence.

**Transfer costs** (from `cost_model.rs`, lines 556-559, and `compute_graph.rs`, lines 542-547):
- CPU->GPU: 5000 cycles overhead + ~1 nanocycle/byte
- GPU->CPU: same (unified memory, symmetric)
- Total round-trip for N bytes: 10000 + 2*N nanocycles

**GPU throughput** (from `cost_model.rs`, lines 802-809):
- Integer ALU: ~512 ops/cycle
- FP arithmetic: ~256 ops/cycle
- GEMM: ~512 ops/cycle

**Baseline comparison:** NEON for vectorizable ops, CPU scalar for the rest.

#### GPU Threshold Calculations

| Operation | NEON Throughput | GPU Throughput | Dispatch+Transfer | Min Elements |
|-----------|---------------|---------------|-------------------|-------------|
| Integer ADD | 4 elements/cycle (4S) | 512 ops/cycle | ~10K cycles | ~80 (64B) |
| FP32 MUL | 4 elements/cycle (4S) | 256 ops/cycle | ~10K cycles | ~160 (640B) |
| FP32 GEMM (NxN) | 4 FMA/cycle | 512 ops/cycle | ~10K cycles | N >= 32 (~4KB) |
| Reduction | 1 element/cycle | 64 ops/cycle | ~10K cycles | ~640 (2.5KB) |

**Practical thresholds:**

```rust
/// GPU profitability thresholds for Apple Silicon.
///
/// Below these thresholds, NEON is faster than GPU dispatch.
/// Values derived from: dispatch overhead ~10K cycles, NEON 4-way FP/SIMD,
/// GPU 256-512 ALU ops/cycle, transfer overhead ~10K cycles round-trip.
pub struct GpuThresholds {
    /// Minimum elements for element-wise operations (add, mul, etc.)
    /// Below this: use NEON. Above: GPU is profitable.
    pub elementwise_min_elements: u64,       // 4096

    /// Minimum total FLOP count for GEMM dispatch.
    /// GEMM FLOPS = 2*M*N*K. Below this: use NEON FMA loop.
    pub gemm_min_flops: u64,                 // 32768 (e.g., 32x32x16)

    /// Minimum elements for reduction operations.
    /// GPU parallel reduce has high overhead; NEON horizontal adds are fast.
    pub reduction_min_elements: u64,          // 8192

    /// Minimum data size in bytes for ANY GPU dispatch.
    /// Absolute floor below which GPU overhead can never be amortized.
    pub absolute_min_bytes: u64,              // 4096 (4KB)

    /// GPU dispatch overhead in cycles (command buffer + scheduling + fence).
    pub dispatch_overhead_cycles: u64,        // 10000

    /// GPU round-trip transfer overhead in cycles (for unified memory).
    pub transfer_overhead_cycles: u64,        // 10000
}

impl Default for GpuThresholds {
    fn default() -> Self {
        Self {
            elementwise_min_elements: 4096,
            gemm_min_flops: 32768,
            reduction_min_elements: 8192,
            absolute_min_bytes: 4096,
            dispatch_overhead_cycles: 10000,
            transfer_overhead_cycles: 10000,
        }
    }
}
```

### ANE Profitability

**Dispatch overhead**: Much higher than GPU. CoreML compilation + model load: ~100K CPU-equivalent cycles (from `cost_model.rs`, line 844).

**Transfer costs** (from `compute_graph.rs`, lines 552-557):
- CPU->ANE: 50000 cycles overhead + ~2 nanocycles/byte
- ANE->CPU: same

**ANE throughput** (from `cost_model.rs`, lines 847-858):
- GEMM/Conv2D: ~2048 ops/cycle
- Element-wise: ~512 ops/cycle
- Shift/bitwise: ~256 ops/cycle

**ANE-specific consideration:** ANE is only profitable when the operation can be fused into a larger model. Standalone small operations pay the full CoreML compilation overhead.

#### ANE Threshold Calculations

| Operation | CPU/NEON Throughput | ANE Throughput | Overhead | Min Elements |
|-----------|-------------------|---------------|----------|-------------|
| GEMM (FP16) | 4 FMA/cycle (NEON) | 2048 ops/cycle | ~200K cycles | N >= 64 (~8KB) |
| Conv2D 3x3 | ~4 MAC/cycle (NEON) | 2048 ops/cycle | ~200K cycles | C*H*W >= 16K |
| Element-wise | 4 elements/cycle | 512 ops/cycle | ~200K cycles | ~16K elements |
| Fused Conv-BN-ReLU | ~4 MAC/cycle | 2048 ops/cycle | ~200K cycles | C*H*W >= 8K |

**Key insight:** ANE fused operations (Conv-BN-ReLU) have a lower effective threshold because the fusion eliminates intermediate memory accesses that the CPU/NEON path would require.

```rust
/// ANE profitability thresholds for Apple Silicon.
///
/// ANE has very high dispatch overhead (CoreML compilation) but extreme
/// throughput for supported operations. Only profitable for large workloads.
pub struct AneThresholds {
    /// Minimum FLOP count for standalone GEMM.
    pub gemm_min_flops: u64,                 // 131072 (e.g., 64x64x32)

    /// Minimum FLOP count for fused Conv-BN-ReLU patterns.
    /// Lower than standalone because fusion amortizes overhead.
    pub fused_conv_min_flops: u64,           // 65536

    /// Minimum elements for standalone element-wise operations.
    /// Very high: ANE dispatch overhead makes small element-wise unprofitable.
    pub elementwise_min_elements: u64,       // 65536

    /// Minimum elements for reduction operations.
    pub reduction_min_elements: u64,          // 32768

    /// Minimum data size in bytes for ANY ANE dispatch.
    pub absolute_min_bytes: u64,              // 32768 (32KB)

    /// ANE dispatch overhead in cycles (CoreML compile + model load).
    pub dispatch_overhead_cycles: u64,        // 100000

    /// ANE round-trip transfer overhead in cycles.
    pub transfer_overhead_cycles: u64,        // 100000

    /// Per-dispatch overhead in nanoseconds (for latency estimation).
    pub dispatch_overhead_ns: u64,            // 50000 (50 us)

    /// ANE FP16 GEMM throughput (GFLOPS).
    pub fp16_gemm_gflops: f64,               // ~11000 (M1), ~38000 (M4)

    /// ANE FP16 element-wise throughput (elements/ns).
    pub fp16_elementwise_throughput: f64,     // ~1.0
}
```

### Apple Silicon Specifics

#### Unified Memory Architecture (UMA)

Apple Silicon shares a single memory pool between CPU, GPU, and ANE. This fundamentally changes the profitability calculation:

**Advantage:** No PCIe DMA copies. CPU and GPU can access the same physical memory pages. The "transfer cost" is actually cache coherency synchronization, not data movement.

**Quantified benefit:**
- Discrete GPU (PCIe): ~5-10 us per transfer + bandwidth-limited copy
- Apple UMA (GPU): ~1-3 us synchronization fence, zero copy
- Apple UMA (ANE): ~10-30 us CoreML marshalling, near-zero copy

This means GPU dispatch is profitable at smaller data sizes on Apple Silicon than on discrete GPU systems.

```rust
/// Apple Silicon unified memory transfer cost model.
///
/// Unlike discrete GPU systems where transfer cost = DMA copy time,
/// Apple Silicon transfer cost = cache coherency synchronization.
pub struct UnifiedMemoryModel {
    /// GPU buffer synchronization fence cost (cycles).
    /// Much lower than PCIe DMA: no data movement, just cache flush.
    pub gpu_sync_cycles: u64,                // 350 (from cost_model.rs line 666)

    /// GPU buffer readback synchronization (cycles).
    pub gpu_readback_cycles: u64,            // 550 (from cost_model.rs line 667)

    /// ANE input tensor marshalling (cycles).
    /// Higher than GPU due to CoreML input format conversion.
    pub ane_marshal_cycles: u64,             // 3000 (from cost_model.rs line 668)

    /// ANE output tensor readback (cycles).
    pub ane_readback_cycles: u64,            // 1500 (from cost_model.rs line 669)

    /// Whether data is already in GPU-accessible memory (IOSurface-backed).
    /// If true, gpu_sync_cycles drops to near-zero.
    pub data_in_gpu_memory: bool,
}
```

#### ANE Hardware Limitations

The ANE is a fixed-function accelerator with specific constraints that affect profitability:

1. **Batch size sensitivity:** ANE throughput scales with batch size. Batch=1 inference often runs faster on GPU or NEON. Batch >= 4 is typically the ANE crossover point.

2. **Tensor dimension alignment:** ANE operates most efficiently on tensors with dimensions that are multiples of 8 (for FP16) or 16 (for INT8). Unaligned tensors require padding, wasting compute.

3. **Operator fusion dependency:** Standalone element-wise operations on ANE have high per-dispatch overhead. The same operation fused into a Conv-BN-ReLU chain is essentially free. The profitability model must consider fusion context.

4. **Model compilation caching:** CoreML caches compiled models. The first invocation pays ~100K cycles; subsequent invocations with the same model shape pay only ~1K cycles (runtime dispatch). The profitability model must distinguish first-time vs cached execution.

```rust
/// ANE hardware constraint checks.
pub struct AneConstraints {
    /// Minimum batch size for ANE profitability.
    pub min_batch_size: u32,                 // 1 (legal), 4 (profitable)

    /// Preferred dimension alignment for maximum throughput.
    pub dimension_alignment: u32,            // 8 (FP16), 16 (INT8)

    /// Whether the operation is part of a fusion chain.
    pub is_fused: bool,

    /// Whether a compiled CoreML model is cached for this shape.
    pub model_cached: bool,
}
```

---

## Decision Tree for Target Selection

### Algorithm: Proof-Annotated Target Selection

The decision tree integrates legality checks, profitability thresholds, and proof annotations:

```
Input: ComputeNode from compute_graph.rs, with:
  - kind: NodeKind (Scalar, DataParallel, MatrixHeavy)
  - data_size_bytes: u64
  - legal_targets: Vec<ComputeTarget> (from ProofAnalyzer)
  - target_legality: Option<TargetLegality>

Output: ComputeTarget recommendation

1. LEGALITY FILTER
   For each candidate target, check:
   a. Is it in legal_targets? (proof-based legality from ProofAnalyzer)
   b. Is the operation supported by the hardware? (target_supports_op)
   c. Are precision requirements met? (precision_check)
   d. Are shape/dimension constraints satisfied? (shape_check)

2. PROFITABILITY FILTER
   For each legal target, check:
   a. Does data_size_bytes exceed the minimum threshold?
   b. Does the operation FLOP count exceed the minimum threshold?
   c. Is the total cost (compute + transfer) less than the baseline?

3. TARGET RANKING
   For targets that pass both filters, rank by:
   a. Total estimated cost (compute + transfer in + transfer out)
   b. Energy efficiency (tie-breaker for equal cost)

4. PROOF-GUIDED BOOST
   If tMIR annotations provide:
   a. Bounded inputs -> ANE FP16 precision is verified -> boost ANE rank
   b. Associative+Commutative -> parallel reduction legal -> boost GPU rank
   c. Static shape proof -> ANE compatibility confirmed -> boost ANE rank
```

### Implementation: `ProfitabilityAnalyzer`

```rust
/// Profitability analyzer that combines legality, thresholds, and cost model.
///
/// Sits between the ProofAnalyzer (which determines what is LEGAL) and the
/// cost model (which determines what is CHEAPEST). This analyzer filters
/// out legal-but-unprofitable targets.
pub struct ProfitabilityAnalyzer {
    gpu_thresholds: GpuThresholds,
    ane_thresholds: AneThresholds,
    cost_model: MultiTargetCostModel,
    uma_model: UnifiedMemoryModel,
}

impl ProfitabilityAnalyzer {
    /// Analyze a computation node and return profitability-filtered recommendations.
    pub fn analyze(&self, node: &ComputeNode) -> TargetRecommendation {
        let mut candidates: Vec<(ComputeTarget, f64)> = Vec::new();

        for &target in &node.legal_targets {
            // Step 1: Hardware legality check
            if !self.is_hardware_legal(target, node) {
                continue;
            }

            // Step 2: Profitability threshold check
            if !self.is_profitable(target, node) {
                continue;
            }

            // Step 3: Compute total cost (including transfers)
            let total_cost = self.total_cost(target, node);
            candidates.push((target, total_cost));
        }

        // Step 4: Rank by total cost
        candidates.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

        // Return cheapest, or CpuScalar as fallback
        let recommended = candidates.first()
            .map(|(t, _)| *t)
            .unwrap_or(ComputeTarget::CpuScalar);

        TargetRecommendation {
            node_id: node.id,
            recommended_target: recommended,
            legal_targets: candidates.iter().map(|(t, _)| *t).collect(),
            reason: self.explain(recommended, node),
            parallel_reduction_legal: node.target_legality
                .as_ref()
                .map(|l| l.parallel_reduction_legal)
                .unwrap_or(false),
        }
    }

    /// Check if the operation is supported by the target hardware.
    fn is_hardware_legal(&self, target: ComputeTarget, node: &ComputeNode) -> bool {
        match target {
            ComputeTarget::CpuScalar => true,  // Always legal
            ComputeTarget::CpuSimd => {
                // NEON: no 64-bit integer MUL, no integer division
                // Check based on node's operations
                true // simplified; real check inspects instructions
            }
            ComputeTarget::Gpu => {
                // GPU: no recursion, no dynamic allocation
                // Already checked by ProofAnalyzer (Pure proof required)
                true
            }
            ComputeTarget::NeuralEngine => {
                // ANE: restricted operation set, static shapes, FP16/INT8 only
                // This is the most restrictive legality check
                node.kind == NodeKind::MatrixHeavy
                    || node.kind == NodeKind::DataParallel
                // Scalar ops are never profitable on ANE
            }
        }
    }

    /// Check if the data size exceeds the profitability threshold.
    fn is_profitable(&self, target: ComputeTarget, node: &ComputeNode) -> bool {
        match target {
            ComputeTarget::CpuScalar | ComputeTarget::CpuSimd => true,
            ComputeTarget::Gpu => {
                node.data_size_bytes >= self.gpu_thresholds.absolute_min_bytes
            }
            ComputeTarget::NeuralEngine => {
                node.data_size_bytes >= self.ane_thresholds.absolute_min_bytes
            }
        }
    }

    /// Compute total cost including data transfers.
    fn total_cost(&self, target: ComputeTarget, node: &ComputeNode) -> f64 {
        let compute_cost = node.costs.get(&target)
            .map(|c| c.latency_cycles as f64)
            .unwrap_or(f64::MAX);

        let transfer_cost = match target {
            ComputeTarget::CpuScalar | ComputeTarget::CpuSimd => 0.0,
            ComputeTarget::Gpu => {
                self.uma_model.gpu_sync_cycles as f64
                + self.uma_model.gpu_readback_cycles as f64
            }
            ComputeTarget::NeuralEngine => {
                self.uma_model.ane_marshal_cycles as f64
                + self.uma_model.ane_readback_cycles as f64
            }
        };

        compute_cost + transfer_cost
    }

    /// Generate human-readable explanation for a recommendation.
    fn explain(&self, target: ComputeTarget, node: &ComputeNode) -> String {
        match target {
            ComputeTarget::CpuScalar => {
                "CPU scalar: data too small for accelerator dispatch".to_string()
            }
            ComputeTarget::CpuSimd => {
                format!(
                    "NEON SIMD: data size {}B profitable for vectorization, \
                     below GPU threshold {}B",
                    node.data_size_bytes,
                    self.gpu_thresholds.absolute_min_bytes,
                )
            }
            ComputeTarget::Gpu => {
                format!(
                    "GPU dispatch: data size {}B exceeds threshold {}B, \
                     compute savings exceed dispatch overhead",
                    node.data_size_bytes,
                    self.gpu_thresholds.absolute_min_bytes,
                )
            }
            ComputeTarget::NeuralEngine => {
                format!(
                    "ANE dispatch: data size {}B exceeds threshold {}B, \
                     matrix-heavy pattern benefits from fixed-function hardware",
                    node.data_size_bytes,
                    self.ane_thresholds.absolute_min_bytes,
                )
            }
        }
    }
}
```

---

## Decision Matrix: Target Selection by Workload

### Quick Reference

| Workload | Data Size | Winner | Reason |
|----------|-----------|--------|--------|
| Scalar arithmetic | any | CPU Scalar | No vectorization possible |
| Element-wise, < 64 elements | < 256B | CPU Scalar | Dispatch overhead dominates |
| Element-wise, 64-1024 elements | 256B-4KB | NEON | SIMD amortizes, GPU too expensive |
| Element-wise, 1K-4K elements | 4KB-16KB | NEON or GPU | GPU starts to amortize |
| Element-wise, > 4K elements | > 16KB | GPU | GPU throughput dominates |
| GEMM, N < 16 | < 2KB | NEON FMA loop | Dispatch overhead too high |
| GEMM, 16 <= N < 64 | 2KB-32KB | GPU | GPU matrix multiply |
| GEMM, N >= 64 (FP16) | >= 32KB | ANE | ANE dedicated matrix hardware |
| Conv2D 3x3, small spatial | C*H*W < 4K | NEON | Dispatch overhead |
| Conv2D 3x3, medium spatial | 4K <= C*H*W < 16K | GPU | Parallel convolution |
| Conv2D 3x3, large spatial | C*H*W >= 16K | ANE | ANE conv pipeline |
| Fused Conv-BN-ReLU | C*H*W >= 8K | ANE | Fusion amortizes overhead |
| Reduction (sum/max) | < 8K elements | NEON horizontal | Low overhead |
| Reduction (sum/max) | >= 8K elements | GPU parallel reduce | Massive parallelism |
| Attention (M4+) | seq_len >= 128 | ANE | Hardware attention unit |

### Crossover Points (M1 Apple Silicon)

```
CPU Scalar -> NEON:      4 elements (any operation)
NEON -> GPU:             ~4096 elements (element-wise)
                         ~32x32 matrix (GEMM)
                         ~4K spatial elements (Conv2D)
GPU -> ANE:              ~64x64 matrix (GEMM, FP16)
                         ~16K spatial elements (Conv2D)
                         ~64K elements (element-wise, FP16)
```

These crossover points shift on M4:
- ANE peak performance: 38 TOPS (vs 11 TOPS on M1) -> ANE thresholds lower by ~3x
- GPU peak performance: ~50% higher -> GPU thresholds lower by ~33%
- CPU/NEON: similar to M1

---

## Integration with Existing Code

### File: `crates/llvm2-lower/src/profitability.rs` (new)

Contains `GpuThresholds`, `AneThresholds`, `UnifiedMemoryModel`, `ProfitabilityAnalyzer`, and legality check functions.

### Integration Point: `compute_graph.rs`

The `target_recommendations()` method currently picks the cheapest legal target by `latency_cycles`. It should be extended to use `ProfitabilityAnalyzer`:

```rust
// In compute_graph.rs, ComputeGraph::target_recommendations():
//   Before: pick cheapest legal target by latency
//   After:  filter through ProfitabilityAnalyzer, then pick cheapest

pub fn target_recommendations_with_profitability(
    &self,
    analyzer: &ProfitabilityAnalyzer,
) -> Vec<TargetRecommendation> {
    self.nodes.iter().map(|node| analyzer.analyze(node)).collect()
}
```

### Integration Point: `cost_model.rs`

The `MultiTargetCostModel::estimate_cost` method should be updated to use the threshold values rather than hardcoded overhead numbers. The threshold constants should be centralized in `profitability.rs` and referenced from `cost_model.rs`.

### Integration Point: `target_analysis.rs`

The `ProofAnalyzer` determines proof-based legality (Pure, InBounds, etc.). The `ProfitabilityAnalyzer` adds a second layer: even if an operation is legally dispatchable to GPU, it may not be profitable. The two analyzers should be composed:

```
ProofAnalyzer (what is LEGAL) + ProfitabilityAnalyzer (what is PROFITABLE)
  = final target recommendation
```

---

## Implementation Plan

### Step 1: Threshold Definitions (1 session)
- Define `GpuThresholds`, `AneThresholds`, `AneConstraints`
- Define `UnifiedMemoryModel`
- Implement `Default` for M1 and M4 presets
- File: `crates/llvm2-lower/src/profitability.rs`

### Step 2: Hardware Legality Checks (1 session)
- Implement `is_neon_legal`, `is_gpu_legal`, `is_ane_legal`
- Wire into target legality alongside existing proof-based checks
- Tests: verify known-illegal operations are rejected

### Step 3: ProfitabilityAnalyzer (1-2 sessions)
- Implement the full `ProfitabilityAnalyzer` with decision tree
- Integrate with `ComputeGraph::target_recommendations`
- Tests: verify crossover points match expected thresholds

### Step 4: Cost Model Refinement (1 session)
- Update `MultiTargetCostModel` to use centralized thresholds
- Add M4-specific threshold presets
- Calibrate against known Apple Silicon benchmarks

### Step 5: End-to-End Integration (1 session)
- Wire profitability into the full lowering pipeline
- Test: tMIR module -> compute graph -> profitability-filtered recommendations
- Verify that small workloads stay on CPU/NEON, large workloads dispatch to GPU/ANE

**Total estimated effort: 5-7 techlead sessions.**

---

## References

1. Dougall Johnson. "Apple M1 Firestorm Microarchitecture." https://dougallj.github.io/applecpu/firestorm.html
2. Dougall Johnson. "Apple M1 Firestorm SIMD/FP Instructions." https://dougallj.github.io/applecpu/firestorm-simd.html
3. Apple. "Metal Performance Shaders." Apple Developer Documentation, 2024.
4. Apple. "Core ML Framework." Apple Developer Documentation, 2024.
5. Apple. "Apple Silicon Architecture Overview." Apple Developer Documentation.
6. LLVM2 cost model: `crates/llvm2-ir/src/cost_model.rs`
7. LLVM2 compute graph: `crates/llvm2-lower/src/compute_graph.rs`
8. LLVM2 target analysis: `crates/llvm2-lower/src/target_analysis.rs`
9. LLVM2 heterogeneous compute design: `designs/2026-04-13-heterogeneous-compute.md`
10. LLVM2 cost model calibration: `designs/2026-04-14-cost-model-calibration.md`
11. LLVM2 ANE semantics: `designs/2026-04-14-ane-semantics.md`
