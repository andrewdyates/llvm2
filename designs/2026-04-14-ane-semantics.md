# Apple Neural Engine (ANE) Semantic Encoding for LLVM2 Verification

**Author:** Andrew Yates
**Date:** 2026-04-14
**Status:** Design
**Part of:** #128 (Multi-target semantic encoding: ANE matrix operation semantics)
**Related:** #120 (Neural Engine targeting via CoreML lowering), #121 (Unified solver architecture), #129 (Unified synthesis loop)

---

## Implementation Status (as of 2026-04-15)

**Overall: ANE semantic encoding is implemented. CoreML MIL emission exists. No end-to-end ANE execution tested.**

| Component | Status | Details |
|-----------|--------|---------|
| **ANE semantics** (`ane_semantics.rs`) | IMPLEMENTED | 1.4K LOC. GEMM, Conv2D, activations, FP16 quantization encoding. |
| **ANE precision proofs** (`ane_precision_proofs.rs`) | IMPLEMENTED (mock) | FP32-to-FP16 bounded error proofs via f64 mock evaluation. |
| **CoreML MIL emitter** (`coreml_emitter.rs`) | IMPLEMENTED | 1.8K LOC. Generates CoreML model graphs for ANE execution. |
| **ANE capability detection** | PARTIAL | Encoded in semantic module. Not tested against real hardware. |
| **End-to-end ANE execution** | NOT TESTED | Emitted CoreML models have not been loaded and executed on ANE hardware. |

---

## Problem

The unified solver architecture requires semantic encodings for every compilation target so the solver can prove equivalence across them. ANE is a fixed-function accelerator on Apple Silicon that excels at matrix-heavy and neural network workloads. Without an ANE semantic encoding, the solver cannot consider ANE as a candidate target -- leaving significant compute capacity unused on Apple hardware.

This document specifies the ANE's capabilities, the mapping from tMIR operations to ANE-compatible computations, the SMT encoding approach, precision handling, and profitability analysis.

---

## Apple Neural Engine Architecture

### Hardware Overview

The Apple Neural Engine (ANE) is a dedicated machine learning accelerator integrated into Apple Silicon. Unlike the GPU (which is programmable), the ANE is a fixed-function pipeline optimized for a specific set of tensor operations.

| Parameter | M1 (2020) | M2 (2022) | M3 (2023) | M4 (2024) |
|-----------|-----------|-----------|-----------|-----------|
| Neural Engine cores | 16 | 16 | 16 | 16 |
| Peak performance | 11 TOPS | 15.8 TOPS | 18 TOPS | 38 TOPS |
| Supported precisions | FP16, INT8 | FP16, INT8 | FP16, INT8 | FP16, INT8, BF16 |
| Power efficiency | ~3 TOPS/W | ~4 TOPS/W | ~4.5 TOPS/W | ~5+ TOPS/W |
| Memory bandwidth (shared) | 68 GB/s | 100 GB/s | 100 GB/s | 120 GB/s |

**Ref:** Apple. "Apple Silicon Chip Specifications." Apple Developer Documentation. Performance figures from Apple keynotes and developer documentation.

### Programming Model

The ANE is NOT directly programmable. It is accessed exclusively through:

1. **Core ML** -- Apple's ML framework. Compiles `.mlmodel` to `.mlmodelc` (compiled format that can target ANE, GPU, or CPU).
2. **BNNS (Basic Neural Network Subroutines)** -- Part of Accelerate framework. Lower-level API for individual tensor operations. Less overhead than Core ML.
3. **Metal Performance Shaders (MPS)** -- Some operations dispatch to ANE transparently when running on Apple Silicon.

```
User code -> Core ML / BNNS API -> ANE driver -> ANE hardware
                                 -> GPU (fallback)
                                 -> CPU (fallback)
```

The ANE compiler within Core ML decides which operations run on ANE vs GPU vs CPU based on operation type, data shape, and precision.

**Ref:** Apple. "Core ML Framework." Apple Developer Documentation, 2024. Apple. "Accelerate Framework (BNNS)." Apple Developer Documentation, 2024.

### Supported Operations

The ANE supports a restricted set of tensor operations. Operations outside this set fall back to GPU or CPU.

| Category | Operations | Precision | Notes |
|----------|-----------|-----------|-------|
| **Convolution** | Conv1D, Conv2D, DepthwiseConv2D, TransposedConv2D | FP16, INT8 | Core ANE strength |
| **Matrix multiply** | GEMM (General Matrix Multiply), BatchedGEMM | FP16, INT8 | High throughput |
| **Pooling** | MaxPool, AvgPool, GlobalAvgPool, GlobalMaxPool | FP16 | 2D only |
| **Normalization** | BatchNorm, InstanceNorm, LayerNorm, GroupNorm | FP16 | Fused with conv |
| **Activation** | ReLU, LeakyReLU, PReLU, GELU, Sigmoid, Tanh, SiLU, Swish | FP16 | Fused with preceding op |
| **Element-wise** | Add, Sub, Mul, Div (tensor-tensor and tensor-scalar) | FP16 | Limited standalone |
| **Reshape** | Reshape, Transpose, Permute, Concat, Split | N/A | Data movement only |
| **Reduction** | ReduceSum, ReduceMean, ReduceMax, ReduceMin | FP16 | Along specified axes |
| **Attention** | ScaledDotProductAttention (M4+) | FP16, BF16 | Hardware attention unit |

**Operations NOT supported on ANE (fall back to GPU/CPU):**
- General-purpose integer arithmetic (non-tensor)
- Scatter/gather with irregular indices
- Control flow (if/else, loops with data-dependent bounds)
- Custom activation functions not in the supported list
- Dynamic shapes (some operations require static shapes for ANE)
- FP32 operations (ANE operates at FP16 or INT8 precision)

**Ref:** Apple. "Optimizing Core ML Performance." Apple Developer Documentation. Apple. "Core ML Model Intermediate Language (MIL) Specification."

---

## ANE Operator Fusion

The ANE achieves high performance through operator fusion: multiple logical operations execute as a single hardware pass. Understanding fusion is critical for the semantic encoding because fused operations have different performance characteristics than unfused ones.

### Common Fusion Patterns

| Fused Pattern | Logical Operations | ANE Implementation |
|--------------|-------------------|-------------------|
| Conv-BN-ReLU | Conv2D + BatchNorm + ReLU | Single ANE pass |
| Conv-Add-ReLU | Conv2D + residual Add + ReLU | Single ANE pass |
| MatMul-Add | GEMM + bias add | Single ANE pass |
| MatMul-GELU | GEMM + GELU activation | Single ANE pass |
| Pool-Activation | MaxPool + activation | Single ANE pass |

### Fusion Rules for the Solver

The solver must understand fusion to correctly estimate ANE cost:

```rust
/// ANE operator fusion rules.
///
/// The ANE fuses eligible sequences into single hardware operations.
/// An unfused element-wise op on ANE is relatively slow (high fixed
/// overhead per ANE dispatch). A fused sequence is fast.
///
/// The solver should prefer fused ANE patterns over unfused GPU patterns
/// when the fusion criteria are met.
pub enum AneFusionPattern {
    /// Convolution + optional normalization + optional activation
    ConvNormAct {
        conv: ConvParams,
        norm: Option<NormParams>,
        act: Option<ActivationFn>,
    },
    /// Matrix multiply + optional bias + optional activation
    GemmBiasAct {
        gemm: GemmParams,
        bias: Option<()>,  // bias add
        act: Option<ActivationFn>,
    },
    /// Pooling + optional activation
    PoolAct {
        pool: PoolParams,
        act: Option<ActivationFn>,
    },
    /// Standalone element-wise (NOT fused -- higher overhead)
    ElementWise {
        op: ElementWiseOp,
    },
    /// Standalone reduction
    Reduce {
        op: ReduceOp,
        axes: Vec<usize>,
    },
}
```

---

## Mapping tMIR Operations to ANE

### Which tMIR Patterns Are ANE-Compatible?

The key question for the solver: given a tMIR computation subgraph, CAN it run on ANE?

```rust
/// Determine if a tMIR subgraph is ANE-compatible.
///
/// ANE compatibility requires:
/// 1. Pure (no side effects)
/// 2. Operations are in the ANE-supported set
/// 3. Data fits ANE tensor format (4D: NCHW or NHWC, up to INT8/FP16)
/// 4. Shapes are static (known at compile time)
/// 5. Data size exceeds profitability threshold
pub fn is_ane_compatible(
    subgraph: &ComputeSubgraph,
    tmir_proofs: &ProofAnnotations,
) -> AneCompatibility {
    if !tmir_proofs.has(Proof::Pure) {
        return AneCompatibility::Incompatible("not Pure");
    }

    // Check that all operations are ANE-supported
    for op in subgraph.operations() {
        if !is_ane_supported_op(op) {
            return AneCompatibility::Incompatible("unsupported operation");
        }
    }

    // Check tensor dimensionality: ANE requires <= 4D tensors
    for tensor in subgraph.tensors() {
        if tensor.ndim() > 4 {
            return AneCompatibility::Incompatible("tensor > 4 dimensions");
        }
        if !tensor.has_static_shape() {
            return AneCompatibility::Incompatible("dynamic shape");
        }
    }

    // Check precision: ANE supports FP16 and INT8 only (BF16 on M4+)
    // If source is FP32/FP64, lowering introduces quantization
    let precision = subgraph.max_precision();
    if precision > Precision::FP16 {
        return AneCompatibility::RequiresQuantization {
            source: precision,
            target: Precision::FP16,
        };
    }

    // Check for fusion opportunities
    let fusion = detect_ane_fusion(subgraph);

    AneCompatibility::Compatible { fusion }
}

pub enum AneCompatibility {
    /// Can run on ANE directly
    Compatible { fusion: Vec<AneFusionPattern> },
    /// Can run on ANE if quantized (with precision loss)
    RequiresQuantization { source: Precision, target: Precision },
    /// Cannot run on ANE
    Incompatible(&'static str),
}
```

### tMIR-to-ANE Operation Mapping

| tMIR Pattern | ANE Operation | Conditions |
|-------------|--------------|------------|
| Nested loop: `C[i][j] += A[i][k] * B[k][j]` | GEMM | Static dims, FP16/INT8 |
| Sliding window: `sum(input[i+k] * weight[k])` | Conv1D | Static kernel size |
| 2D convolution pattern | Conv2D | Static kernel, <= 4D |
| Element-wise: `a[i] + b[i]` | Add (if fused) | Tensor shapes match |
| Element-wise: `a[i] * b[i]` | Mul (if fused) | Tensor shapes match |
| `max(0, x)` | ReLU (fused) | Following conv/gemm |
| `1 / (1 + exp(-x))` | Sigmoid (fused) | Following conv/gemm |
| Reduction: `sum(a[i])` along axis | ReduceSum | Static axis |
| Reduction: `max(a[i])` along axis | ReduceMax | Static axis |
| Pooling: `max(window)` over 2D grid | MaxPool2D | Static window, stride |
| Pooling: `mean(window)` over 2D grid | AvgPool2D | Static window, stride |

---

## SMT Encoding for ANE Operations

### Design Principle: Mathematical Specification with Precision Constraints

ANE operations are encoded as their mathematical definitions with explicit precision modeling. The key difference from GPU encoding: ANE always operates at reduced precision (FP16/INT8), so the verification must account for quantization error.

### Encoding 1: Matrix Multiply (GEMM)

```rust
/// SMT encoding of ANE GEMM: C = A * B (in FP16)
///
/// The mathematical operation is standard matrix multiply, but
/// all intermediate values are FP16. If the source computation
/// is FP32, the encoding includes the FP32->FP16 conversion at
/// input and FP16->FP32 conversion at output.
///
/// z4 theory: QF_FPBV (FP16 = floating_point(5, 11))
pub fn encode_ane_gemm(
    a: &SmtExpr,   // Flattened M*K matrix (FP16 elements)
    b: &SmtExpr,   // Flattened K*N matrix (FP16 elements)
    m: u64, k: u64, n: u64,
) -> SmtExpr {
    let fp16 = Sort::fp(5, 11);  // IEEE 754 half-precision
    let rne = SmtExpr::rounding_mode(RoundingMode::RNE);
    let zero = SmtExpr::fp_zero(5, 11);
    let mut c = SmtExpr::const_array(Sort::bv(64), fp16.clone());

    for i in 0..m {
        for j in 0..n {
            let mut sum = zero.clone();
            for kk in 0..k {
                let a_idx = SmtExpr::bv_const(i * k + kk, 64);
                let b_idx = SmtExpr::bv_const(kk * n + j, 64);
                let a_elem = SmtExpr::select(a, &a_idx);
                let b_elem = SmtExpr::select(b, &b_idx);
                // FP16 multiply
                let prod = SmtExpr::fp_mul(&rne, &a_elem, &b_elem);
                // FP16 accumulate (note: ANE may use FP32 accumulator
                // internally; we model worst-case FP16 accumulation)
                sum = SmtExpr::fp_add(&rne, &sum, &prod);
            }
            let c_idx = SmtExpr::bv_const(i * n + j, 64);
            c = SmtExpr::store(&c, &c_idx, &sum);
        }
    }
    c
}
```

**Practical limits:** For small matrices (M,K,N <= 8), fully unrolled encoding is tractable (~512 FP operations, solvable in 10-30s). For larger matrices, use UF abstraction (see below).

### Encoding 2: Convolution (Conv2D)

```rust
/// SMT encoding of ANE Conv2D.
///
/// Standard 2D convolution: output[n][c_out][h][w] =
///   sum over (c_in, kh, kw) of input[n][c_in][h+kh][w+kw] * kernel[c_out][c_in][kh][kw]
///   + bias[c_out]
///
/// For small kernels (3x3, 5x5) with small spatial dims, this is tractable.
/// For larger cases, use UF abstraction.
pub fn encode_ane_conv2d(
    input: &SmtExpr,     // 4D tensor [N, C_in, H, W], flattened
    kernel: &SmtExpr,    // 4D tensor [C_out, C_in, KH, KW], flattened
    bias: Option<&SmtExpr>,  // 1D tensor [C_out]
    params: &Conv2dParams,
) -> SmtExpr {
    let fp16 = Sort::fp(5, 11);
    let rne = SmtExpr::rounding_mode(RoundingMode::RNE);
    let zero = SmtExpr::fp_zero(5, 11);

    let out_h = (params.in_h + 2 * params.pad_h - params.kernel_h) / params.stride_h + 1;
    let out_w = (params.in_w + 2 * params.pad_w - params.kernel_w) / params.stride_w + 1;

    let mut output = SmtExpr::const_array(Sort::bv(64), fp16.clone());

    for n_idx in 0..params.batch {
        for c_out in 0..params.channels_out {
            for oh in 0..out_h {
                for ow in 0..out_w {
                    let mut sum = match bias {
                        Some(b) => SmtExpr::select(b, &SmtExpr::bv_const(c_out, 64)),
                        None => zero.clone(),
                    };

                    for c_in in 0..params.channels_in {
                        for kh in 0..params.kernel_h {
                            for kw in 0..params.kernel_w {
                                let ih = oh * params.stride_h + kh - params.pad_h;
                                let iw = ow * params.stride_w + kw - params.pad_w;

                                // Skip padding positions (zero-padded)
                                if ih < params.in_h && iw < params.in_w {
                                    let in_idx = flatten_4d(
                                        n_idx, c_in, ih, iw,
                                        params.channels_in, params.in_h, params.in_w,
                                    );
                                    let k_idx = flatten_4d(
                                        c_out, c_in, kh, kw,
                                        params.channels_in, params.kernel_h, params.kernel_w,
                                    );
                                    let in_elem = SmtExpr::select(input, &SmtExpr::bv_const(in_idx, 64));
                                    let k_elem = SmtExpr::select(kernel, &SmtExpr::bv_const(k_idx, 64));
                                    let prod = SmtExpr::fp_mul(&rne, &in_elem, &k_elem);
                                    sum = SmtExpr::fp_add(&rne, &sum, &prod);
                                }
                            }
                        }
                    }

                    let out_idx = flatten_4d(
                        n_idx, c_out, oh, ow,
                        params.channels_out, out_h, out_w,
                    );
                    output = SmtExpr::store(&output, &SmtExpr::bv_const(out_idx, 64), &sum);
                }
            }
        }
    }
    output
}

fn flatten_4d(n: u64, c: u64, h: u64, w: u64, c_dim: u64, h_dim: u64, w_dim: u64) -> u64 {
    n * c_dim * h_dim * w_dim + c * h_dim * w_dim + h * w_dim + w
}
```

### Encoding 3: Element-Wise Operations

```rust
/// SMT encoding of ANE element-wise operations.
///
/// ANE element-wise ops operate in FP16. The encoding is the same
/// as the mathematical definition at FP16 precision.
pub fn encode_ane_elementwise(
    input_a: &SmtExpr,
    input_b: &SmtExpr,
    op: ElementWiseOp,
    n: u64,  // Total number of elements
) -> SmtExpr {
    let fp16 = Sort::fp(5, 11);
    let rne = SmtExpr::rounding_mode(RoundingMode::RNE);
    let mut result = SmtExpr::const_array(Sort::bv(64), fp16.clone());

    for i in 0..n {
        let idx = SmtExpr::bv_const(i, 64);
        let a = SmtExpr::select(input_a, &idx);
        let b = SmtExpr::select(input_b, &idx);
        let value = match op {
            ElementWiseOp::Add => SmtExpr::fp_add(&rne, &a, &b),
            ElementWiseOp::Sub => SmtExpr::fp_sub(&rne, &a, &b),
            ElementWiseOp::Mul => SmtExpr::fp_mul(&rne, &a, &b),
            ElementWiseOp::Div => SmtExpr::fp_div(&rne, &a, &b),
        };
        result = SmtExpr::store(&result, &idx, &value);
    }
    result
}
```

### Encoding 4: Activation Functions

Activation functions are nonlinear element-wise transforms. Some have exact SMT encodings; others require piecewise approximation.

```rust
/// SMT encoding of ANE activation functions.
pub fn encode_ane_activation(
    input: &SmtExpr,
    act: ActivationFn,
    n: u64,
) -> SmtExpr {
    let fp16 = Sort::fp(5, 11);
    let rne = SmtExpr::rounding_mode(RoundingMode::RNE);
    let mut result = SmtExpr::const_array(Sort::bv(64), fp16.clone());
    let zero = SmtExpr::fp_zero(5, 11);

    for i in 0..n {
        let idx = SmtExpr::bv_const(i, 64);
        let x = SmtExpr::select(input, &idx);
        let value = match act {
            // ReLU: max(0, x) -- exact encoding via ite
            ActivationFn::ReLU => {
                SmtExpr::ite(&x.fp_geq(&zero), &x, &zero)
            }
            // LeakyReLU: x >= 0 ? x : alpha * x
            ActivationFn::LeakyReLU { alpha } => {
                let alpha_expr = SmtExpr::fp_const(alpha, 5, 11);
                let neg_branch = SmtExpr::fp_mul(&rne, &alpha_expr, &x);
                SmtExpr::ite(&x.fp_geq(&zero), &x, &neg_branch)
            }
            // Sigmoid: 1 / (1 + exp(-x))
            // Cannot encode exp() in QF_FPBV. Use UF abstraction.
            ActivationFn::Sigmoid => {
                let sigmoid_fn = SmtExpr::declare_fun(
                    "ane_sigmoid", &[fp16.clone()], fp16.clone(),
                );
                // Axiomatic constraints:
                // - Range: 0 <= sigmoid(x) <= 1
                // - Monotonicity: x1 < x2 => sigmoid(x1) < sigmoid(x2)
                // - sigmoid(0) = 0.5
                sigmoid_fn.apply(&[x])
            }
            // GELU: x * 0.5 * (1 + erf(x / sqrt(2)))
            // Use UF abstraction with monotonicity constraint.
            ActivationFn::GELU => {
                let gelu_fn = SmtExpr::declare_fun(
                    "ane_gelu", &[fp16.clone()], fp16.clone(),
                );
                gelu_fn.apply(&[x])
            }
            // Tanh: (exp(x) - exp(-x)) / (exp(x) + exp(-x))
            // Use UF abstraction.
            ActivationFn::Tanh => {
                let tanh_fn = SmtExpr::declare_fun(
                    "ane_tanh", &[fp16.clone()], fp16.clone(),
                );
                tanh_fn.apply(&[x])
            }
        };
        result = SmtExpr::store(&result, &idx, &value);
    }
    result
}
```

### Encoding 5: Pooling

```rust
/// SMT encoding of ANE MaxPool2D.
pub fn encode_ane_maxpool2d(
    input: &SmtExpr,   // 4D tensor [N, C, H, W], flattened
    params: &Pool2dParams,
) -> SmtExpr {
    let fp16 = Sort::fp(5, 11);
    let neg_inf = SmtExpr::fp_neg_inf(5, 11);

    let out_h = (params.in_h - params.kernel_h) / params.stride_h + 1;
    let out_w = (params.in_w - params.kernel_w) / params.stride_w + 1;

    let mut output = SmtExpr::const_array(Sort::bv(64), fp16.clone());

    for n_idx in 0..params.batch {
        for c in 0..params.channels {
            for oh in 0..out_h {
                for ow in 0..out_w {
                    let mut max_val = neg_inf.clone();

                    for kh in 0..params.kernel_h {
                        for kw in 0..params.kernel_w {
                            let ih = oh * params.stride_h + kh;
                            let iw = ow * params.stride_w + kw;
                            let in_idx = flatten_4d(
                                n_idx, c, ih, iw,
                                params.channels, params.in_h, params.in_w,
                            );
                            let elem = SmtExpr::select(input, &SmtExpr::bv_const(in_idx, 64));
                            max_val = SmtExpr::ite(
                                &elem.fp_gt(&max_val), &elem, &max_val,
                            );
                        }
                    }

                    let out_idx = flatten_4d(
                        n_idx, c, oh, ow,
                        params.channels, out_h, out_w,
                    );
                    output = SmtExpr::store(&output, &SmtExpr::bv_const(out_idx, 64), &max_val);
                }
            }
        }
    }
    output
}
```

### Encoding 6: UF Abstraction for Large Operations

For operations too large to unroll (or transcendental functions like sigmoid/tanh), use uninterpreted functions with axiomatic constraints.

```rust
/// UF-based ANE encoding for large or complex operations.
///
/// Declares the ANE operation as an uninterpreted function and
/// asserts known algebraic properties as axioms. This lets the
/// solver reason about composition (matmul chains, fused patterns)
/// without unrolling individual elements.
pub fn encode_ane_operation_uf(
    operation: &AneOperation,
    inputs: &[SmtExpr],
) -> (SmtExpr, Vec<SmtExpr>) {  // (result, axioms)
    let mut axioms = Vec::new();

    match operation {
        AneOperation::Gemm { m, k, n } => {
            let matmul = SmtExpr::declare_fun("ane_matmul", &[arr_sort(), arr_sort()], arr_sort());
            let result = matmul.apply(inputs);

            // Axiom: matmul is bilinear
            // matmul(alpha * A, B) = alpha * matmul(A, B)
            // (encoded as: for specific test vectors, verify bilinearity)

            // Axiom: matmul associativity
            // matmul(matmul(A, B), C) = matmul(A, matmul(B, C))

            // Axiom: identity
            // matmul(I, A) = A

            (result, axioms)
        }
        AneOperation::Conv2d { params } => {
            let conv = SmtExpr::declare_fun("ane_conv2d", &[arr_sort(), arr_sort()], arr_sort());
            let result = conv.apply(inputs);

            // Axiom: linearity
            // conv(alpha * x, k) = alpha * conv(x, k)

            (result, axioms)
        }
        _ => {
            let f = SmtExpr::declare_fun("ane_op", &[arr_sort()], arr_sort());
            (f.apply(inputs), axioms)
        }
    }
}
```

---

## Precision Handling: FP16 Quantization

### The Fundamental Challenge

ANE operates at FP16 (and INT8). When the source tMIR computation uses FP32 or FP64, lowering to ANE introduces quantization error. The verification must handle this correctly.

### Three Verification Strategies

**Strategy 1: Exact Equivalence (FP16 source)**

If the tMIR computation already uses FP16 precision (e.g., half-precision training), the ANE encoding is exact. Standard equivalence checking applies:

```
Proof obligation: forall inputs: tmir_fp16(inputs) == ane_fp16(inputs)
```

**Strategy 2: Bounded Error (FP32 source, lossy lowering)**

If the tMIR uses FP32 but the algorithm tolerates FP16 precision (e.g., inference), the verification proves bounded error:

```rust
/// Verify that ANE FP16 result is within epsilon of CPU FP32 result.
///
/// Proof obligation:
///   forall inputs: |fp32_result - fp16_to_fp32(ane_result)| < epsilon
///
/// The epsilon bound comes from error analysis of the specific
/// operation (e.g., matmul of M*K*N with FP16 has worst-case
/// error ~ K * 2^{-10} * max(|A|) * max(|B|)).
pub fn verify_ane_bounded_error(
    tmir_fp32_encoding: &SmtExpr,  // CPU FP32 result
    ane_fp16_encoding: &SmtExpr,   // ANE FP16 result
    epsilon: f64,                   // Error bound
) -> SmtExpr {
    let fp32 = Sort::fp(8, 24);
    let rne = SmtExpr::rounding_mode(RoundingMode::RNE);

    // Convert ANE FP16 result to FP32 for comparison
    let ane_as_fp32 = SmtExpr::fp_to_fp(&rne, ane_fp16_encoding, &fp32);

    // Compute absolute difference
    let diff = SmtExpr::fp_sub(&rne, tmir_fp32_encoding, &ane_as_fp32);
    let abs_diff = SmtExpr::fp_abs(&diff);

    // Assert: abs_diff >= epsilon (negation; expect UNSAT)
    let eps_expr = SmtExpr::fp_const(epsilon, 8, 24);
    abs_diff.fp_geq(&eps_expr)  // If UNSAT: error is always < epsilon
}
```

**Strategy 3: Structural Equivalence (large operations)**

For large operations where element-wise comparison is intractable, verify structural properties:

```
Proof obligation:
  1. The ANE operation computes the same mathematical function (e.g., matrix multiply)
  2. The precision is within specified bounds
  3. Fusion does not change semantics (e.g., Conv-BN-ReLU == ReLU(BN(Conv(x))))
```

This uses the UF abstraction approach: declare the ANE operation and assert it satisfies the required algebraic properties.

### FP16 Error Bounds

| Operation | Worst-Case FP16 Error | Relative Error |
|-----------|----------------------|----------------|
| Element-wise add/sub | 2^{-10} * max(\|a\|, \|b\|) | ~0.1% |
| Element-wise mul | 2^{-10} * \|a\| * \|b\| | ~0.1% |
| GEMM (M x K) | K * 2^{-10} * max(\|A\|) * max(\|B\|) | grows with K |
| Conv2D (C_in x KH x KW) | C_in*KH*KW * 2^{-10} | grows with kernel |
| ReLU | 0 (exact) | 0% |
| Sigmoid | 2^{-10} (per-element) | ~0.1% |

FP16 has 10 bits of mantissa (vs FP32's 23 bits). The unit roundoff is epsilon = 2^{-10} ~ 9.77e-4.

**Ref:** Higham, N. J. "Accuracy and Stability of Numerical Algorithms." SIAM, 2002. Chapter 3: Floating-Point Arithmetic.

---

## CoreML Model Compilation Overhead

### Compilation Pipeline

When LLVM2 targets ANE, it must compile the computation to a CoreML model:

```
tMIR subgraph -> CoreML MIL (Model Intermediate Language)
                -> ANE compiler (within CoreML)
                -> .mlmodelc (compiled model)
                -> ANE execution
```

### Overhead Analysis

| Phase | Latency | When Paid |
|-------|---------|-----------|
| MIL generation | ~1 ms | Compile time |
| CoreML compilation (`compileModel`) | 100-5000 ms | Compile time (or first load) |
| Model loading (`MLModel.init`) | 10-100 ms | Runtime (first invocation) |
| ANE dispatch (per inference) | 10-100 us | Runtime (each call) |
| Result readback | <10 us (UMA) | Runtime |

**Key insight:** CoreML compilation is expensive (100ms - 5s depending on model complexity). This cost is paid once at compile time or first load. It is NOT paid per-invocation. Therefore:

- For **ahead-of-time (AOT) compilation**: CoreML compilation happens during LLVM2 compilation. No runtime overhead beyond dispatch.
- For **JIT compilation**: CoreML compilation adds significant first-use latency. Only justified for hot paths that will be called many times.

**Ref:** Apple. "Creating a Core ML Model." Apple Developer Documentation. Measured via Instruments profiling.

### When Is ANE Profitable?

ANE is profitable when its compute savings exceed the per-dispatch overhead AND the computation fits ANE's supported operations:

```rust
/// ANE profitability thresholds.
pub struct AneCostThresholds {
    /// Minimum FLOP count for standalone GEMM dispatch
    pub gemm_min_flops: u64,                // 65,536 (e.g., 64x64x16)

    /// Minimum FLOP count for fused Conv-BN-ReLU
    pub fused_conv_min_flops: u64,          // 32,768

    /// Minimum elements for standalone element-wise ops
    /// (high: ANE dispatch overhead makes small element-wise unprofitable)
    pub elementwise_min_elements: u64,      // 65,536

    /// Minimum elements for reduction operations
    pub reduce_min_elements: u64,           // 16,384

    /// Per-dispatch overhead in nanoseconds
    pub dispatch_overhead_ns: u64,           // 50,000 (50 us)

    /// ANE throughput for FP16 GEMM (GFLOPS)
    pub fp16_gemm_gflops: f64,              // ~11,000 (M1), ~38,000 (M4)

    /// ANE throughput for FP16 element-wise (elements/ns)
    pub fp16_elementwise_throughput: f64,    // ~1.0
}
```

### ANE vs GPU vs NEON Decision Matrix

| Computation | Size | Winner | Reason |
|------------|------|--------|--------|
| GEMM (FP16) | 8x8 | NEON | ANE dispatch overhead dominates |
| GEMM (FP16) | 64x64 | GPU or ANE | Both efficient, ANE more power-efficient |
| GEMM (FP16) | 256x256+ | ANE | ANE peak TOPS > GPU for matmul |
| Conv2D 3x3 | 32x32x16 | ANE | ANE conv fused pipeline |
| Conv2D 3x3 | 8x8x3 | NEON | Too small for ANE dispatch |
| Element-wise add | 1K elements | NEON | ANE dispatch overhead |
| Element-wise add | 100K elements | GPU | GPU throughput for simple ops |
| Reduction sum | 10K elements | GPU | GPU parallel reduce |
| Attention block | 128 tokens | ANE (M4+) | Hardware attention unit |

---

## Unified Encoding API: `AneEncoding`

```rust
/// ANE operation semantic encoding for the unified solver.
///
/// Implements the `TargetEncoding` trait so the solver can compare
/// ANE implementations against CPU, NEON, and GPU candidates.
pub struct AneEncoding {
    /// The ANE operation pattern
    pub operation: AneOperation,
    /// Precision (FP16 or INT8)
    pub precision: AnePrecision,
    /// Tensor dimensions
    pub dims: TensorDims,
    /// Whether this is a fused pattern
    pub fusion: Option<AneFusionPattern>,
}

pub enum AneOperation {
    /// General matrix multiply
    Gemm { m: u64, k: u64, n: u64 },
    /// 2D convolution
    Conv2d { params: Conv2dParams },
    /// Element-wise binary operation
    ElementWise { op: ElementWiseOp, n: u64 },
    /// Activation function (usually fused)
    Activation { act: ActivationFn, n: u64 },
    /// Pooling
    Pool2d { params: Pool2dParams, pool_type: PoolType },
    /// Reduction along axes
    Reduce { op: ReduceOp, axes: Vec<usize>, shape: Vec<u64> },
}

pub enum AnePrecision {
    FP16,          // Half-precision float (standard)
    INT8,          // 8-bit integer (quantized)
    BF16,          // BFloat16 (M4+ only)
}

impl TargetEncoding for AneEncoding {
    fn required_logic(&self) -> SmtLogic {
        match (&self.operation, &self.precision) {
            // Small operations: full FP encoding
            (AneOperation::Gemm { m, k, n }, AnePrecision::FP16)
                if m * k * n <= 512 => SmtLogic::QF_FPABV,

            // INT8 operations: bitvector only
            (_, AnePrecision::INT8) => SmtLogic::QF_ABV,

            // Large operations: UF abstraction
            _ => SmtLogic::QF_UFABV,
        }
    }

    fn encode(&self, inputs: &[SmtExpr], program: &mut Z4Program) -> SmtExpr {
        match &self.operation {
            AneOperation::Gemm { m, k, n } => {
                if m * k * n <= 512 {
                    encode_ane_gemm(&inputs[0], &inputs[1], *m, *k, *n)
                } else {
                    let (result, axioms) = encode_ane_operation_uf(
                        &self.operation, inputs,
                    );
                    for axiom in axioms { program.assert(&axiom); }
                    result
                }
            }
            AneOperation::Conv2d { params } => {
                if params.total_ops() <= 4096 {
                    encode_ane_conv2d(&inputs[0], &inputs[1], inputs.get(2), params)
                } else {
                    let (result, axioms) = encode_ane_operation_uf(
                        &self.operation, inputs,
                    );
                    for axiom in axioms { program.assert(&axiom); }
                    result
                }
            }
            AneOperation::ElementWise { op, n } => {
                encode_ane_elementwise(&inputs[0], &inputs[1], *op, *n)
            }
            AneOperation::Activation { act, n } => {
                encode_ane_activation(&inputs[0], *act, *n)
            }
            AneOperation::Pool2d { params, pool_type } => {
                match pool_type {
                    PoolType::Max => encode_ane_maxpool2d(&inputs[0], params),
                    PoolType::Avg => encode_ane_avgpool2d(&inputs[0], params),
                }
            }
            AneOperation::Reduce { op, axes, shape } => {
                encode_ane_reduce(&inputs[0], *op, axes, shape)
            }
        }
    }

    fn estimated_solve_time(&self) -> Duration {
        match &self.operation {
            AneOperation::Gemm { m, k, n } => {
                let total = m * k * n;
                if total <= 64 { Duration::from_secs(1) }
                else if total <= 512 { Duration::from_secs(30) }
                else { Duration::from_secs(5) }  // UF: fast
            }
            AneOperation::Conv2d { params } => {
                if params.total_ops() <= 1024 { Duration::from_secs(10) }
                else { Duration::from_secs(5) }  // UF
            }
            AneOperation::ElementWise { n, .. } => {
                if *n <= 1024 { Duration::from_secs(5) }
                else { Duration::from_secs(30) }
            }
            AneOperation::Activation { n, .. } => {
                if *n <= 256 { Duration::from_secs(2) }
                else { Duration::from_secs(10) }
            }
            _ => Duration::from_secs(10),
        }
    }
}
```

---

## Solver Complexity Analysis

| Operation | Size | Formula | z4 Theory | Est. Time |
|-----------|------|---------|-----------|-----------|
| GEMM (FP16) | 4x4x4 | ~64 FP ops | QF_FPBV | 1-5s |
| GEMM (FP16) | 8x8x8 | ~512 FP ops | QF_FPBV | 10-30s |
| GEMM (FP16) | 16x16x16 | ~4K FP ops | QF_FPBV | timeout |
| GEMM (FP16, UF) | any | ~10 terms | QF_UFABV | < 1s |
| GEMM (INT8) | 8x8x8 | ~512 BV ops | QF_ABV | < 5s |
| Conv2D 3x3 | 8x8x1 | ~576 FP ops | QF_FPBV | 5-20s |
| Conv2D 3x3 (UF) | any | ~10 terms | QF_UFABV | < 1s |
| ReLU | 64 elems | ~64 ite ops | QF_FPBV | < 1s |
| MaxPool 2x2 | 8x8 | ~64 ite ops | QF_FPBV | 1-5s |
| Bounded error check | 64 elems | ~200 FP ops | QF_FPBV | 5-30s |

**Scalability regime:**
1. **Exact proofs (small):** Fully unrolled FP encoding for operations with < ~500 total FP ops. Complete verification.
2. **UF proofs (large):** Uninterpreted function abstraction for larger operations. Proves algebraic properties (associativity, linearity) but not bit-exact FP equivalence.
3. **Bounded error proofs:** For FP32->FP16 quantization, prove the error is bounded. Tractable for individual elements; use UF for entire operations.

---

## z4 Theory Requirements

| Theory | Needed For | Issue | Status |
|--------|-----------|-------|--------|
| QF_BV | INT8 operations | Exists | Available |
| QF_ABV | Array-based tensor encoding | #122 | In progress |
| QF_FP (FP16) | Half-precision ANE ops | #123 | In progress |
| QF_FPBV | Mixed FP and BV | #123 | In progress |
| QF_UF | Large operation abstraction | Needed | Not started |
| QF_UFABV | UF + arrays | Needed | Not started |

---

## Implementation Plan

### Step 1: ANE Type Definitions (1 session)
- Define `AneEncoding`, `AneOperation`, `AnePrecision`, `AneFusionPattern`
- Define `Conv2dParams`, `Pool2dParams`, `ActivationFn`, `ElementWiseOp`
- Implement `TargetEncoding` trait skeleton
- File: `crates/llvm2-verify/src/ane_semantics.rs`

### Step 2: GEMM Encoding (1-2 sessions)
- Implement `encode_ane_gemm` (FP16, unrolled for small matrices)
- Implement `encode_ane_operation_uf` (UF abstraction for large matrices)
- Write verification tests: ANE GEMM == CPU matmul (small matrices)
- Write bounded-error tests: ANE FP16 GEMM vs CPU FP32 GEMM

### Step 3: Conv2D Encoding (1-2 sessions)
- Implement `encode_ane_conv2d` (unrolled for small kernels)
- Handle padding, stride, dilation
- Write verification tests for 3x3 convolution

### Step 4: Activation + Element-wise (1 session)
- Implement `encode_ane_activation` (ReLU exact, sigmoid/GELU via UF)
- Implement `encode_ane_elementwise`
- Write tests for ReLU, LeakyReLU (exact), sigmoid (UF with constraints)

### Step 5: Pooling + Reduction (1 session)
- Implement `encode_ane_maxpool2d`, `encode_ane_avgpool2d`
- Implement `encode_ane_reduce`
- Write verification tests

### Step 6: Precision Verification (1 session)
- Implement `verify_ane_bounded_error`
- Implement FP32->FP16 conversion encoding
- Write epsilon-bound verification tests

### Step 7: ANE Cost Model + Profitability (1 session)
- Implement `AneCostThresholds` with M-series defaults
- Implement `is_ane_compatible` detection logic
- Implement fusion pattern detection
- Wire into `HeterogeneousCostModel` (#144 / #161)

**Total estimated effort: 7-9 techlead sessions.**

---

## References

1. Apple. "Core ML Framework." Apple Developer Documentation, 2024.
2. Apple. "Accelerate Framework (BNNS)." Apple Developer Documentation, 2024.
3. Apple. "Optimizing Core ML Performance." Apple Developer Documentation, 2024.
4. Apple. "Core ML Model Intermediate Language (MIL) Specification." Internal documentation.
5. Apple. "Creating a Core ML Model." Apple Developer Documentation, 2024.
6. Apple. "Apple Silicon Chip Specifications." Apple Developer Documentation.
7. Higham, N. J. "Accuracy and Stability of Numerical Algorithms." SIAM, 2002.
8. Leroy, X. "Formal Verification of a Realistic Compiler." Communications of the ACM, 2009 (CompCert).
9. Lopes et al. "Alive2: Bounded Translation Validation for LLVM." PLDI 2021.
10. LLVM2 multi-target semantic encoding: `designs/2026-04-13-multi-target-encoding.md`
11. LLVM2 unified solver architecture: `designs/2026-04-13-unified-solver-architecture.md`
12. LLVM2 heterogeneous compute allocation: `designs/2026-04-13-heterogeneous-compute.md`
13. LLVM2 Metal GPU semantic encoding: `designs/2026-04-14-metal-gpu-semantics.md`
14. LLVM2 cost model calibration: `designs/2026-04-14-cost-model-calibration.md`
15. LLVM2 NEON semantics implementation: `crates/llvm2-verify/src/neon_semantics.rs`
