// llvm2-verify/ane_semantics.rs - Apple Neural Engine semantic encoding
//
// Author: Andrew Yates <ayates@dropbox.com>
// Copyright 2026 Dropbox, Inc. | License: Apache-2.0
//
// Encodes Apple Neural Engine (ANE) operation semantics as SMT expressions
// using array theory for tensor representation and floating-point theory for
// FP16/FP32 precision modeling. Covers GEMM, Conv2D, activation functions,
// element-wise operations, and FP16 quantization with bounded error.
//
// Reference: designs/2026-04-14-ane-semantics.md
// Reference: Apple. "Core ML Framework." Apple Developer Documentation, 2024.
// Reference: Higham, N. J. "Accuracy and Stability of Numerical Algorithms."
//            SIAM, 2002. Chapter 3: Floating-Point Arithmetic.

//! Apple Neural Engine (ANE) operation semantics encoded as [`SmtExpr`] formulas.
//!
//! The ANE is a fixed-function accelerator on Apple Silicon that operates at
//! FP16 (and INT8) precision. This module encodes ANE operations as their
//! mathematical definitions with explicit FP16 precision modeling.
//!
//! Tensors are represented as SMT arrays (`(Array (_ BitVec 64) (_ FloatingPoint 5 11))`)
//! with row-major flattened indexing. All arithmetic uses FP16 with round-to-nearest-even.
//!
//! # Supported Operations
//!
//! - **GEMM**: General matrix multiply `C = A * B` in FP16
//! - **Conv2D**: 2D convolution with stride, padding, optional bias
//! - **Activation**: ReLU (exact), LeakyReLU (exact), Sigmoid (UF), Tanh (UF), GELU (UF)
//! - **Element-wise**: Add, Sub, Mul, Div on tensors
//! - **FP16 quantization**: FP32-to-FP16 conversion with bounded error encoding

use crate::smt::{SmtExpr, SmtSort, RoundingMode};

// ---------------------------------------------------------------------------
// ANE element type
// ---------------------------------------------------------------------------

/// Element type for ANE tensors.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AneElementType {
    /// IEEE 754 half-precision: `(_ FloatingPoint 5 11)`.
    FP16,
    /// IEEE 754 single-precision: `(_ FloatingPoint 8 24)`.
    FP32,
}

impl AneElementType {
    /// Exponent bits for this element type.
    pub fn eb(self) -> u32 {
        match self {
            AneElementType::FP16 => 5,
            AneElementType::FP32 => 8,
        }
    }

    /// Significand bits (including implicit bit) for this element type.
    pub fn sb(self) -> u32 {
        match self {
            AneElementType::FP16 => 11,
            AneElementType::FP32 => 24,
        }
    }

    /// SMT sort for this element type.
    pub fn sort(self) -> SmtSort {
        SmtSort::FloatingPoint(self.eb(), self.sb())
    }
}

// ---------------------------------------------------------------------------
// ANE tensor shape
// ---------------------------------------------------------------------------

/// Tensor dimensions for ANE operations.
///
/// ANE requires tensors with at most 4 dimensions (NCHW layout).
/// All shapes must be statically known at compile time.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AneTensorShape {
    /// Batch dimension (N). 1 for non-batched.
    pub batch: u64,
    /// Channel dimension (C).
    pub channels: u64,
    /// Height dimension (H).
    pub height: u64,
    /// Width dimension (W).
    pub width: u64,
    /// Element type.
    pub elem_type: AneElementType,
}

impl AneTensorShape {
    /// Create a 2D matrix shape (M x N) with given element type.
    pub fn matrix(rows: u64, cols: u64, elem_type: AneElementType) -> Self {
        AneTensorShape {
            batch: 1,
            channels: 1,
            height: rows,
            width: cols,
            elem_type,
        }
    }

    /// Create a 4D tensor shape (N, C, H, W).
    pub fn tensor_4d(
        batch: u64,
        channels: u64,
        height: u64,
        width: u64,
        elem_type: AneElementType,
    ) -> Self {
        AneTensorShape { batch, channels, height, width, elem_type }
    }

    /// Total number of elements.
    pub fn numel(&self) -> u64 {
        self.batch * self.channels * self.height * self.width
    }

    /// Flatten a 4D index (n, c, h, w) to a linear index (row-major NCHW).
    pub fn flatten(&self, n: u64, c: u64, h: u64, w: u64) -> u64 {
        n * self.channels * self.height * self.width
            + c * self.height * self.width
            + h * self.width
            + w
    }
}

// ---------------------------------------------------------------------------
// Conv2D parameters
// ---------------------------------------------------------------------------

/// Parameters for ANE Conv2D operation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Conv2dParams {
    /// Input batch size.
    pub batch: u64,
    /// Input channels.
    pub channels_in: u64,
    /// Output channels (number of filters).
    pub channels_out: u64,
    /// Input height.
    pub in_h: u64,
    /// Input width.
    pub in_w: u64,
    /// Kernel height.
    pub kernel_h: u64,
    /// Kernel width.
    pub kernel_w: u64,
    /// Stride in height dimension.
    pub stride_h: u64,
    /// Stride in width dimension.
    pub stride_w: u64,
    /// Padding in height dimension.
    pub pad_h: u64,
    /// Padding in width dimension.
    pub pad_w: u64,
}

impl Conv2dParams {
    /// Output height.
    pub fn out_h(&self) -> u64 {
        (self.in_h + 2 * self.pad_h - self.kernel_h) / self.stride_h + 1
    }

    /// Output width.
    pub fn out_w(&self) -> u64 {
        (self.in_w + 2 * self.pad_w - self.kernel_w) / self.stride_w + 1
    }

    /// Total FP operations (multiply-accumulate count).
    pub fn total_ops(&self) -> u64 {
        self.batch
            * self.channels_out
            * self.out_h()
            * self.out_w()
            * self.channels_in
            * self.kernel_h
            * self.kernel_w
    }
}

// ---------------------------------------------------------------------------
// Activation function type
// ---------------------------------------------------------------------------

/// ANE activation function type.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ActivationFn {
    /// `max(0, x)` -- exact SMT encoding via `ite`.
    ReLU,
    /// `x >= 0 ? x : alpha * x` -- exact via `ite` + `fp_mul`.
    LeakyReLU { alpha: f64 },
    /// `1 / (1 + exp(-x))` -- UF abstraction (transcendental).
    Sigmoid,
    /// `(exp(x) - exp(-x)) / (exp(x) + exp(-x))` -- UF abstraction.
    Tanh,
    /// `x * 0.5 * (1 + erf(x / sqrt(2)))` -- UF abstraction.
    GELU,
}

// ---------------------------------------------------------------------------
// Element-wise operation type
// ---------------------------------------------------------------------------

/// ANE element-wise binary operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ElementWiseOp {
    Add,
    Sub,
    Mul,
    Div,
}

// ---------------------------------------------------------------------------
// FP16 zero constant helper
// ---------------------------------------------------------------------------

/// Create an FP16 zero constant.
fn fp16_zero() -> SmtExpr {
    SmtExpr::fp_const(0, 5, 11)
}

/// Create an FP16 constant from an f64 value.
///
/// Converts via f32 -> f16 bit pattern for SmtExpr encoding.
/// For evaluation purposes, the concrete evaluator interprets FPConst
/// with (5,11) as FP16, but internally stores as f64.
fn fp16_const_from_f64(val: f64) -> SmtExpr {
    // For the concrete evaluator, FPConst with eb=5,sb=11 is treated specially.
    // We store the f32 bits since the evaluator checks for (8,24) and falls through
    // to f64 otherwise. For FP16, store as f64 bits.
    SmtExpr::fp_const(val.to_bits(), 5, 11)
}

/// Create an FP32 constant from an f64 value.
fn fp32_const_from_f64(val: f64) -> SmtExpr {
    SmtExpr::fp32_const(val as f32)
}

/// Array sort for FP16 tensors: `(Array (_ BitVec 64) (_ FloatingPoint 5 11))`.
fn fp16_array_sort() -> SmtSort {
    SmtSort::Array(
        Box::new(SmtSort::BitVec(64)),
        Box::new(SmtSort::FloatingPoint(5, 11)),
    )
}

// ---------------------------------------------------------------------------
// GEMM encoding
// ---------------------------------------------------------------------------

/// SMT encoding of ANE GEMM: `C = A * B` in FP16.
///
/// Matrix A is M x K, matrix B is K x N, result C is M x N.
/// All elements are FP16. Intermediate multiply-accumulate uses FP16
/// (worst-case model; real ANE may use FP32 accumulator internally).
///
/// Tensors are represented as SMT arrays with BV64 indices and FP16 elements,
/// using row-major flattened indexing.
///
/// For small matrices (M*K*N <= 512), the fully unrolled encoding is tractable.
/// For larger matrices, use `encode_ane_gemm_uf` instead.
///
/// # Arguments
///
/// * `a` - Flattened M*K matrix (FP16 array)
/// * `b` - Flattened K*N matrix (FP16 array)
/// * `m` - Number of rows in A / rows in C
/// * `k` - Shared dimension (columns of A, rows of B)
/// * `n` - Number of columns in B / columns in C
pub fn encode_ane_gemm(
    a: &SmtExpr,
    b: &SmtExpr,
    m: u64,
    k: u64,
    n: u64,
) -> SmtExpr {
    let zero = fp16_zero();
    let mut c = SmtExpr::const_array(SmtSort::BitVec(64), zero.clone());

    for i in 0..m {
        for j in 0..n {
            let mut sum = zero.clone();
            for kk in 0..k {
                let a_idx = SmtExpr::bv_const(i * k + kk, 64);
                let b_idx = SmtExpr::bv_const(kk * n + j, 64);
                let a_elem = SmtExpr::select(a.clone(), a_idx);
                let b_elem = SmtExpr::select(b.clone(), b_idx);
                let prod = SmtExpr::fp_mul(RoundingMode::RNE, a_elem, b_elem);
                sum = SmtExpr::fp_add(RoundingMode::RNE, sum, prod);
            }
            let c_idx = SmtExpr::bv_const(i * n + j, 64);
            c = SmtExpr::store(c, c_idx, sum);
        }
    }
    c
}

/// UF-abstracted GEMM for large matrices.
///
/// Declares an uninterpreted function `ane_matmul` and returns its application.
/// The caller should assert algebraic axioms (bilinearity, associativity, identity)
/// as additional constraints.
pub fn encode_ane_gemm_uf(
    a: &SmtExpr,
    b: &SmtExpr,
) -> SmtExpr {
    let arr_sort = fp16_array_sort();
    SmtExpr::uf(
        "ane_matmul",
        vec![a.clone(), b.clone()],
        arr_sort,
    )
}

// ---------------------------------------------------------------------------
// Conv2D encoding
// ---------------------------------------------------------------------------

/// Flatten a 4D index (n, c, h, w) to linear index given dimensions.
fn flatten_4d(n: u64, c: u64, h: u64, w: u64, c_dim: u64, h_dim: u64, w_dim: u64) -> u64 {
    n * c_dim * h_dim * w_dim + c * h_dim * w_dim + h * w_dim + w
}

/// SMT encoding of ANE Conv2D.
///
/// Standard 2D convolution with zero-padding:
/// ```text
/// output[n][c_out][oh][ow] =
///   sum over (c_in, kh, kw) of
///     input[n][c_in][oh*stride_h + kh - pad_h][ow*stride_w + kw - pad_w]
///     * kernel[c_out][c_in][kh][kw]
///   + bias[c_out]  (if provided)
/// ```
///
/// All arithmetic in FP16 with round-to-nearest-even.
///
/// # Arguments
///
/// * `input` - 4D tensor [N, C_in, H, W], flattened to FP16 array
/// * `kernel` - 4D tensor [C_out, C_in, KH, KW], flattened to FP16 array
/// * `bias` - Optional 1D tensor [C_out], FP16 array
/// * `params` - Convolution parameters (dimensions, stride, padding)
pub fn encode_ane_conv2d(
    input: &SmtExpr,
    kernel: &SmtExpr,
    bias: Option<&SmtExpr>,
    params: &Conv2dParams,
) -> SmtExpr {
    let zero = fp16_zero();
    let out_h = params.out_h();
    let out_w = params.out_w();

    let mut output = SmtExpr::const_array(SmtSort::BitVec(64), zero.clone());

    for n_idx in 0..params.batch {
        for c_out in 0..params.channels_out {
            for oh in 0..out_h {
                for ow in 0..out_w {
                    let mut sum = match bias {
                        Some(b) => SmtExpr::select(
                            b.clone(),
                            SmtExpr::bv_const(c_out, 64),
                        ),
                        None => zero.clone(),
                    };

                    for c_in in 0..params.channels_in {
                        for kh in 0..params.kernel_h {
                            for kw in 0..params.kernel_w {
                                // Compute input coordinates with stride and padding.
                                let ih_signed =
                                    (oh * params.stride_h + kh) as i64 - params.pad_h as i64;
                                let iw_signed =
                                    (ow * params.stride_w + kw) as i64 - params.pad_w as i64;

                                // Skip out-of-bounds positions (zero-padded).
                                if ih_signed >= 0
                                    && (ih_signed as u64) < params.in_h
                                    && iw_signed >= 0
                                    && (iw_signed as u64) < params.in_w
                                {
                                    let ih = ih_signed as u64;
                                    let iw = iw_signed as u64;
                                    let in_idx = flatten_4d(
                                        n_idx,
                                        c_in,
                                        ih,
                                        iw,
                                        params.channels_in,
                                        params.in_h,
                                        params.in_w,
                                    );
                                    let k_idx = flatten_4d(
                                        c_out,
                                        c_in,
                                        kh,
                                        kw,
                                        params.channels_in,
                                        params.kernel_h,
                                        params.kernel_w,
                                    );
                                    let in_elem = SmtExpr::select(
                                        input.clone(),
                                        SmtExpr::bv_const(in_idx, 64),
                                    );
                                    let k_elem = SmtExpr::select(
                                        kernel.clone(),
                                        SmtExpr::bv_const(k_idx, 64),
                                    );
                                    let prod = SmtExpr::fp_mul(
                                        RoundingMode::RNE,
                                        in_elem,
                                        k_elem,
                                    );
                                    sum = SmtExpr::fp_add(RoundingMode::RNE, sum, prod);
                                }
                            }
                        }
                    }

                    let out_idx = flatten_4d(
                        n_idx,
                        c_out,
                        oh,
                        ow,
                        params.channels_out,
                        out_h,
                        out_w,
                    );
                    output =
                        SmtExpr::store(output, SmtExpr::bv_const(out_idx, 64), sum);
                }
            }
        }
    }
    output
}

// ---------------------------------------------------------------------------
// Activation function encoding
// ---------------------------------------------------------------------------

/// SMT encoding of ANE activation functions applied element-wise.
///
/// ReLU and LeakyReLU have exact SMT encodings via `ite`. Sigmoid, Tanh,
/// and GELU use uninterpreted function (UF) abstraction since they involve
/// transcendental functions not expressible in QF_FPBV.
///
/// # Arguments
///
/// * `input` - FP16 array of `n` elements
/// * `activation` - Activation function type
/// * `n` - Number of elements
pub fn encode_ane_activation(
    input: &SmtExpr,
    activation: ActivationFn,
    n: u64,
) -> SmtExpr {
    let zero = fp16_zero();
    let fp16_sort = SmtSort::FloatingPoint(5, 11);
    let mut result = SmtExpr::const_array(SmtSort::BitVec(64), zero.clone());

    for i in 0..n {
        let idx = SmtExpr::bv_const(i, 64);
        let x = SmtExpr::select(input.clone(), idx.clone());
        let value = match activation {
            ActivationFn::ReLU => {
                // max(0, x): exact encoding via ite on fp comparison.
                // fp_lt returns Bool: x < 0 means x is negative.
                let x_lt_zero = x.clone().fp_lt(zero.clone());
                SmtExpr::ite(x_lt_zero, zero.clone(), x)
            }
            ActivationFn::LeakyReLU { alpha } => {
                let alpha_expr = fp16_const_from_f64(alpha);
                let neg_branch =
                    SmtExpr::fp_mul(RoundingMode::RNE, alpha_expr, x.clone());
                let x_lt_zero = x.clone().fp_lt(zero.clone());
                SmtExpr::ite(x_lt_zero, neg_branch, x)
            }
            ActivationFn::Sigmoid => {
                // UF abstraction: sigmoid is not expressible in QF_FPBV.
                SmtExpr::uf(
                    "ane_sigmoid",
                    vec![x],
                    fp16_sort.clone(),
                )
            }
            ActivationFn::Tanh => {
                SmtExpr::uf(
                    "ane_tanh",
                    vec![x],
                    fp16_sort.clone(),
                )
            }
            ActivationFn::GELU => {
                SmtExpr::uf(
                    "ane_gelu",
                    vec![x],
                    fp16_sort.clone(),
                )
            }
        };
        result = SmtExpr::store(result, SmtExpr::bv_const(i, 64), value);
    }
    result
}

// ---------------------------------------------------------------------------
// Element-wise operation encoding
// ---------------------------------------------------------------------------

/// SMT encoding of ANE element-wise binary operations.
///
/// All operations use FP16 precision with round-to-nearest-even.
///
/// # Arguments
///
/// * `input_a` - First FP16 array operand
/// * `input_b` - Second FP16 array operand
/// * `op` - Element-wise operation (Add, Sub, Mul, Div)
/// * `n` - Number of elements
pub fn encode_ane_elementwise(
    input_a: &SmtExpr,
    input_b: &SmtExpr,
    op: ElementWiseOp,
    n: u64,
) -> SmtExpr {
    let zero = fp16_zero();
    let mut result = SmtExpr::const_array(SmtSort::BitVec(64), zero);

    for i in 0..n {
        let idx = SmtExpr::bv_const(i, 64);
        let a = SmtExpr::select(input_a.clone(), idx.clone());
        let b = SmtExpr::select(input_b.clone(), idx.clone());
        let value = match op {
            ElementWiseOp::Add => SmtExpr::fp_add(RoundingMode::RNE, a, b),
            ElementWiseOp::Sub => {
                // No FPSub in SmtExpr; encode as a + (-b).
                let neg_b = b.fp_neg();
                SmtExpr::fp_add(RoundingMode::RNE, a, neg_b)
            }
            ElementWiseOp::Mul => SmtExpr::fp_mul(RoundingMode::RNE, a, b),
            ElementWiseOp::Div => SmtExpr::fp_div(RoundingMode::RNE, a, b),
        };
        result = SmtExpr::store(result, SmtExpr::bv_const(i, 64), value);
    }
    result
}

// ---------------------------------------------------------------------------
// FP16 quantization encoding
// ---------------------------------------------------------------------------

/// Encode FP32-to-FP16 quantization of a single element.
///
/// This constructs the SMT expression for converting an FP32 value to FP16
/// by rounding to nearest-even. The result is an FP16 value.
///
/// In the concrete evaluator, this is approximated by f64 arithmetic
/// (which has more than enough precision to represent both FP32 and FP16).
pub fn encode_fp16_quantize_elem(fp32_val: &SmtExpr) -> SmtExpr {
    // Model FP32->FP16 conversion as: truncate to FP16 precision.
    // In SMT-LIB2 this would be `((_ to_fp 5 11) RNE fp32_val)`.
    // For the concrete evaluator, we model this as an identity (f64 carries both).
    // The bounded-error verification proves the quantization error is within bounds.
    //
    // For symbolic encoding, we use a UF that captures the rounding behavior.
    SmtExpr::uf(
        "fp32_to_fp16",
        vec![fp32_val.clone()],
        SmtSort::FloatingPoint(5, 11),
    )
}

/// Encode FP32-to-FP16 quantization of an entire array.
///
/// Applies element-wise FP16 quantization to each element of an FP32 array.
///
/// # Arguments
///
/// * `fp32_array` - Array of FP32 values
/// * `n` - Number of elements
pub fn encode_fp16_quantize(fp32_array: &SmtExpr, n: u64) -> SmtExpr {
    let zero = fp16_zero();
    let mut result = SmtExpr::const_array(SmtSort::BitVec(64), zero);

    for i in 0..n {
        let idx = SmtExpr::bv_const(i, 64);
        let fp32_elem = SmtExpr::select(fp32_array.clone(), idx.clone());
        let fp16_elem = encode_fp16_quantize_elem(&fp32_elem);
        result = SmtExpr::store(result, SmtExpr::bv_const(i, 64), fp16_elem);
    }
    result
}

/// Encode bounded error check: `|fp32_result - fp16_result| < epsilon`.
///
/// Constructs an SMT formula asserting the absolute difference between
/// two FP values exceeds epsilon. If this formula is UNSAT, then the
/// error is always within bounds.
///
/// For concrete evaluation, this returns a Bool expression.
///
/// # Arguments
///
/// * `fp32_val` - The FP32 reference value
/// * `fp16_val` - The FP16 approximate value (as f64 for evaluation)
/// * `epsilon` - Error bound
pub fn encode_bounded_error_check(
    fp32_val: &SmtExpr,
    fp16_val: &SmtExpr,
    epsilon: f64,
) -> SmtExpr {
    // Compute |fp32 - fp16|.
    // diff = fp32 - fp16 = fp_add(fp32, fp_neg(fp16))
    let neg_fp16 = fp16_val.clone().fp_neg();
    let diff = SmtExpr::fp_add(RoundingMode::RNE, fp32_val.clone(), neg_fp16);

    // abs_diff: if diff < 0 then -diff else diff
    let zero_f64 = SmtExpr::fp64_const(0.0);
    let diff_lt_zero = diff.clone().fp_lt(zero_f64);
    let abs_diff = SmtExpr::ite(diff_lt_zero, diff.clone().fp_neg(), diff);

    // Check: abs_diff < epsilon
    let eps_expr = SmtExpr::fp64_const(epsilon);
    abs_diff.fp_lt(eps_expr)
}

// ---------------------------------------------------------------------------
// Helper: build FP16 array from f64 slice (for testing)
// ---------------------------------------------------------------------------

/// Build an FP16 SMT array from a slice of f64 values.
///
/// Useful for constructing test inputs. Values are stored as FP16 constants.
pub fn fp16_array_from_f64(values: &[f64]) -> SmtExpr {
    let zero = fp16_zero();
    let mut arr = SmtExpr::const_array(SmtSort::BitVec(64), zero);
    for (i, &v) in values.iter().enumerate() {
        let idx = SmtExpr::bv_const(i as u64, 64);
        let val = fp16_const_from_f64(v);
        arr = SmtExpr::store(arr, idx, val);
    }
    arr
}

/// Build an FP32 SMT array from a slice of f64 values.
pub fn fp32_array_from_f64(values: &[f64]) -> SmtExpr {
    let zero = SmtExpr::fp32_const(0.0f32);
    let mut arr = SmtExpr::const_array(SmtSort::BitVec(64), zero);
    for (i, &v) in values.iter().enumerate() {
        let idx = SmtExpr::bv_const(i as u64, 64);
        let val = fp32_const_from_f64(v);
        arr = SmtExpr::store(arr, idx, val);
    }
    arr
}

/// Read element at index `i` from an FP array and evaluate to f64.
#[cfg(test)]
fn read_array_elem_f64(arr: &SmtExpr, i: u64, env: &std::collections::HashMap<String, u64>) -> f64 {
    let idx = SmtExpr::bv_const(i, 64);
    let elem = SmtExpr::select(arr.clone(), idx);
    elem.try_eval(env).unwrap().as_f64()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::smt::EvalResult;
    use std::collections::HashMap;

    fn empty_env() -> HashMap<String, u64> {
        HashMap::new()
    }

    // =======================================================================
    // AneTensorShape tests
    // =======================================================================

    #[test]
    fn test_tensor_shape_matrix() {
        let shape = AneTensorShape::matrix(4, 3, AneElementType::FP16);
        assert_eq!(shape.numel(), 12);
        assert_eq!(shape.flatten(0, 0, 0, 0), 0);
        assert_eq!(shape.flatten(0, 0, 0, 2), 2);
        assert_eq!(shape.flatten(0, 0, 3, 0), 9);
        assert_eq!(shape.flatten(0, 0, 3, 2), 11);
    }

    #[test]
    fn test_tensor_shape_4d() {
        let shape = AneTensorShape::tensor_4d(2, 3, 4, 5, AneElementType::FP16);
        assert_eq!(shape.numel(), 120);
        // Flatten (0,0,0,0) = 0
        assert_eq!(shape.flatten(0, 0, 0, 0), 0);
        // Flatten (1,0,0,0) = 1*3*4*5 = 60
        assert_eq!(shape.flatten(1, 0, 0, 0), 60);
        // Flatten (0,1,0,0) = 0 + 1*4*5 = 20
        assert_eq!(shape.flatten(0, 1, 0, 0), 20);
        // Flatten (0,0,1,0) = 5
        assert_eq!(shape.flatten(0, 0, 1, 0), 5);
    }

    #[test]
    fn test_element_type_sort() {
        assert_eq!(AneElementType::FP16.sort(), SmtSort::FloatingPoint(5, 11));
        assert_eq!(AneElementType::FP32.sort(), SmtSort::FloatingPoint(8, 24));
        assert_eq!(AneElementType::FP16.eb(), 5);
        assert_eq!(AneElementType::FP16.sb(), 11);
        assert_eq!(AneElementType::FP32.eb(), 8);
        assert_eq!(AneElementType::FP32.sb(), 24);
    }

    // =======================================================================
    // Conv2dParams tests
    // =======================================================================

    #[test]
    fn test_conv2d_params_output_dims() {
        let params = Conv2dParams {
            batch: 1,
            channels_in: 1,
            channels_out: 1,
            in_h: 5,
            in_w: 5,
            kernel_h: 3,
            kernel_w: 3,
            stride_h: 1,
            stride_w: 1,
            pad_h: 0,
            pad_w: 0,
        };
        assert_eq!(params.out_h(), 3);
        assert_eq!(params.out_w(), 3);
    }

    #[test]
    fn test_conv2d_params_with_padding() {
        let params = Conv2dParams {
            batch: 1,
            channels_in: 1,
            channels_out: 1,
            in_h: 5,
            in_w: 5,
            kernel_h: 3,
            kernel_w: 3,
            stride_h: 1,
            stride_w: 1,
            pad_h: 1,
            pad_w: 1,
        };
        // With padding=1 and kernel=3, output = input size
        assert_eq!(params.out_h(), 5);
        assert_eq!(params.out_w(), 5);
    }

    #[test]
    fn test_conv2d_params_with_stride() {
        let params = Conv2dParams {
            batch: 1,
            channels_in: 1,
            channels_out: 1,
            in_h: 6,
            in_w: 6,
            kernel_h: 3,
            kernel_w: 3,
            stride_h: 2,
            stride_w: 2,
            pad_h: 0,
            pad_w: 0,
        };
        assert_eq!(params.out_h(), 2);
        assert_eq!(params.out_w(), 2);
    }

    #[test]
    fn test_conv2d_total_ops() {
        let params = Conv2dParams {
            batch: 1,
            channels_in: 3,
            channels_out: 8,
            in_h: 32,
            in_w: 32,
            kernel_h: 3,
            kernel_w: 3,
            stride_h: 1,
            stride_w: 1,
            pad_h: 1,
            pad_w: 1,
        };
        // out_h = 32, out_w = 32
        // total = 1 * 8 * 32 * 32 * 3 * 3 * 3 = 221184
        assert_eq!(params.total_ops(), 221184);
    }

    // =======================================================================
    // GEMM tests
    // =======================================================================

    #[test]
    fn test_gemm_1x1x1() {
        // Scalar multiply: C[0,0] = A[0,0] * B[0,0]
        let a = fp16_array_from_f64(&[3.0]);
        let b = fp16_array_from_f64(&[4.0]);
        let c = encode_ane_gemm(&a, &b, 1, 1, 1);

        let env = empty_env();
        let result = read_array_elem_f64(&c, 0, &env);
        assert!((result - 12.0).abs() < 1e-6, "expected 12.0, got {}", result);
    }

    #[test]
    fn test_gemm_2x2_identity() {
        // A * I = A for 2x2 identity matrix.
        // A = [[1, 2], [3, 4]], I = [[1, 0], [0, 1]]
        let a = fp16_array_from_f64(&[1.0, 2.0, 3.0, 4.0]);
        let identity = fp16_array_from_f64(&[1.0, 0.0, 0.0, 1.0]);
        let c = encode_ane_gemm(&a, &identity, 2, 2, 2);

        let env = empty_env();
        assert!((read_array_elem_f64(&c, 0, &env) - 1.0).abs() < 1e-6);
        assert!((read_array_elem_f64(&c, 1, &env) - 2.0).abs() < 1e-6);
        assert!((read_array_elem_f64(&c, 2, &env) - 3.0).abs() < 1e-6);
        assert!((read_array_elem_f64(&c, 3, &env) - 4.0).abs() < 1e-6);
    }

    #[test]
    fn test_gemm_2x2_basic() {
        // A = [[1, 2], [3, 4]], B = [[5, 6], [7, 8]]
        // C = [[1*5+2*7, 1*6+2*8], [3*5+4*7, 3*6+4*8]]
        //   = [[19, 22], [43, 50]]
        let a = fp16_array_from_f64(&[1.0, 2.0, 3.0, 4.0]);
        let b = fp16_array_from_f64(&[5.0, 6.0, 7.0, 8.0]);
        let c = encode_ane_gemm(&a, &b, 2, 2, 2);

        let env = empty_env();
        assert!((read_array_elem_f64(&c, 0, &env) - 19.0).abs() < 1e-6);
        assert!((read_array_elem_f64(&c, 1, &env) - 22.0).abs() < 1e-6);
        assert!((read_array_elem_f64(&c, 2, &env) - 43.0).abs() < 1e-6);
        assert!((read_array_elem_f64(&c, 3, &env) - 50.0).abs() < 1e-6);
    }

    #[test]
    fn test_gemm_2x3_times_3x2() {
        // A (2x3) = [[1,2,3],[4,5,6]]
        // B (3x2) = [[7,8],[9,10],[11,12]]
        // C (2x2) = [[1*7+2*9+3*11, 1*8+2*10+3*12], [4*7+5*9+6*11, 4*8+5*10+6*12]]
        //         = [[58, 64], [139, 154]]
        let a = fp16_array_from_f64(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let b = fp16_array_from_f64(&[7.0, 8.0, 9.0, 10.0, 11.0, 12.0]);
        let c = encode_ane_gemm(&a, &b, 2, 3, 2);

        let env = empty_env();
        assert!((read_array_elem_f64(&c, 0, &env) - 58.0).abs() < 1e-6);
        assert!((read_array_elem_f64(&c, 1, &env) - 64.0).abs() < 1e-6);
        assert!((read_array_elem_f64(&c, 2, &env) - 139.0).abs() < 1e-6);
        assert!((read_array_elem_f64(&c, 3, &env) - 154.0).abs() < 1e-6);
    }

    #[test]
    fn test_gemm_3x3_basic() {
        // A = [[1,0,0],[0,2,0],[0,0,3]] (diagonal)
        // B = [[1,1,1],[1,1,1],[1,1,1]] (all ones)
        // C = [[1,1,1],[2,2,2],[3,3,3]]
        let a = fp16_array_from_f64(&[1.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 3.0]);
        let b = fp16_array_from_f64(&[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]);
        let c = encode_ane_gemm(&a, &b, 3, 3, 3);

        let env = empty_env();
        // Row 0: [1, 1, 1]
        assert!((read_array_elem_f64(&c, 0, &env) - 1.0).abs() < 1e-6);
        assert!((read_array_elem_f64(&c, 1, &env) - 1.0).abs() < 1e-6);
        assert!((read_array_elem_f64(&c, 2, &env) - 1.0).abs() < 1e-6);
        // Row 1: [2, 2, 2]
        assert!((read_array_elem_f64(&c, 3, &env) - 2.0).abs() < 1e-6);
        assert!((read_array_elem_f64(&c, 4, &env) - 2.0).abs() < 1e-6);
        assert!((read_array_elem_f64(&c, 5, &env) - 2.0).abs() < 1e-6);
        // Row 2: [3, 3, 3]
        assert!((read_array_elem_f64(&c, 6, &env) - 3.0).abs() < 1e-6);
        assert!((read_array_elem_f64(&c, 7, &env) - 3.0).abs() < 1e-6);
        assert!((read_array_elem_f64(&c, 8, &env) - 3.0).abs() < 1e-6);
    }

    #[test]
    fn test_gemm_zero_matrix() {
        // A * 0 = 0
        let a = fp16_array_from_f64(&[1.0, 2.0, 3.0, 4.0]);
        let zero = fp16_array_from_f64(&[0.0, 0.0, 0.0, 0.0]);
        let c = encode_ane_gemm(&a, &zero, 2, 2, 2);

        let env = empty_env();
        for i in 0..4 {
            assert!(
                read_array_elem_f64(&c, i, &env).abs() < 1e-6,
                "element {} should be 0",
                i
            );
        }
    }

    // =======================================================================
    // Conv2D tests
    // =======================================================================

    #[test]
    fn test_conv2d_1x1_identity_kernel() {
        // 1x1 convolution with identity kernel = copy.
        // Input: 1x1x3x3, Kernel: 1x1x1x1 = [1.0]
        let input = fp16_array_from_f64(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]);
        let kernel = fp16_array_from_f64(&[1.0]);
        let params = Conv2dParams {
            batch: 1,
            channels_in: 1,
            channels_out: 1,
            in_h: 3,
            in_w: 3,
            kernel_h: 1,
            kernel_w: 1,
            stride_h: 1,
            stride_w: 1,
            pad_h: 0,
            pad_w: 0,
        };
        let output = encode_ane_conv2d(&input, &kernel, None, &params);
        let env = empty_env();

        // Output should be same as input.
        for i in 0..9 {
            let expected = (i + 1) as f64;
            let actual = read_array_elem_f64(&output, i, &env);
            assert!(
                (actual - expected).abs() < 1e-6,
                "element {}: expected {}, got {}",
                i,
                expected,
                actual
            );
        }
    }

    #[test]
    fn test_conv2d_3x3_single_channel() {
        // Input: 1x1x3x3, Kernel: 1x1x3x3 (all ones) -> output 1x1x1x1
        // = sum of all input elements
        let input_data: Vec<f64> = (1..=9).map(|x| x as f64).collect();
        let input = fp16_array_from_f64(&input_data);
        let kernel = fp16_array_from_f64(&[1.0; 9]);
        let params = Conv2dParams {
            batch: 1,
            channels_in: 1,
            channels_out: 1,
            in_h: 3,
            in_w: 3,
            kernel_h: 3,
            kernel_w: 3,
            stride_h: 1,
            stride_w: 1,
            pad_h: 0,
            pad_w: 0,
        };
        let output = encode_ane_conv2d(&input, &kernel, None, &params);
        let env = empty_env();

        // Sum of 1..9 = 45
        let result = read_array_elem_f64(&output, 0, &env);
        assert!(
            (result - 45.0).abs() < 1e-6,
            "expected 45.0, got {}",
            result
        );
    }

    #[test]
    fn test_conv2d_with_bias() {
        // 1x1 kernel with bias.
        let input = fp16_array_from_f64(&[1.0, 2.0, 3.0, 4.0]);
        let kernel = fp16_array_from_f64(&[2.0]);
        let bias = fp16_array_from_f64(&[10.0]);
        let params = Conv2dParams {
            batch: 1,
            channels_in: 1,
            channels_out: 1,
            in_h: 2,
            in_w: 2,
            kernel_h: 1,
            kernel_w: 1,
            stride_h: 1,
            stride_w: 1,
            pad_h: 0,
            pad_w: 0,
        };
        let output = encode_ane_conv2d(&input, &kernel, Some(&bias), &params);
        let env = empty_env();

        // output[i] = input[i] * 2.0 + 10.0
        assert!((read_array_elem_f64(&output, 0, &env) - 12.0).abs() < 1e-6);
        assert!((read_array_elem_f64(&output, 1, &env) - 14.0).abs() < 1e-6);
        assert!((read_array_elem_f64(&output, 2, &env) - 16.0).abs() < 1e-6);
        assert!((read_array_elem_f64(&output, 3, &env) - 18.0).abs() < 1e-6);
    }

    #[test]
    fn test_conv2d_with_stride() {
        // Input: 1x1x4x4, Kernel: 1x1x2x2 (all ones), stride=2, no padding
        // Output: 1x1x2x2
        let input = fp16_array_from_f64(&[
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0,
            16.0,
        ]);
        let kernel = fp16_array_from_f64(&[1.0, 1.0, 1.0, 1.0]);
        let params = Conv2dParams {
            batch: 1,
            channels_in: 1,
            channels_out: 1,
            in_h: 4,
            in_w: 4,
            kernel_h: 2,
            kernel_w: 2,
            stride_h: 2,
            stride_w: 2,
            pad_h: 0,
            pad_w: 0,
        };
        let output = encode_ane_conv2d(&input, &kernel, None, &params);
        let env = empty_env();

        // out[0,0] = 1+2+5+6 = 14
        // out[0,1] = 3+4+7+8 = 22
        // out[1,0] = 9+10+13+14 = 46
        // out[1,1] = 11+12+15+16 = 54
        assert!((read_array_elem_f64(&output, 0, &env) - 14.0).abs() < 1e-6);
        assert!((read_array_elem_f64(&output, 1, &env) - 22.0).abs() < 1e-6);
        assert!((read_array_elem_f64(&output, 2, &env) - 46.0).abs() < 1e-6);
        assert!((read_array_elem_f64(&output, 3, &env) - 54.0).abs() < 1e-6);
    }

    // =======================================================================
    // Activation function tests
    // =======================================================================

    #[test]
    fn test_relu_positive_values() {
        let input = fp16_array_from_f64(&[1.0, 2.0, 3.0, 0.5]);
        let output = encode_ane_activation(&input, ActivationFn::ReLU, 4);
        let env = empty_env();

        // Positive values pass through unchanged.
        assert!((read_array_elem_f64(&output, 0, &env) - 1.0).abs() < 1e-6);
        assert!((read_array_elem_f64(&output, 1, &env) - 2.0).abs() < 1e-6);
        assert!((read_array_elem_f64(&output, 2, &env) - 3.0).abs() < 1e-6);
        assert!((read_array_elem_f64(&output, 3, &env) - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_relu_negative_values() {
        let input = fp16_array_from_f64(&[-1.0, -2.0, -0.5, -100.0]);
        let output = encode_ane_activation(&input, ActivationFn::ReLU, 4);
        let env = empty_env();

        // Negative values become 0.
        for i in 0..4 {
            assert!(
                read_array_elem_f64(&output, i, &env).abs() < 1e-6,
                "ReLU of negative should be 0 at index {}",
                i
            );
        }
    }

    #[test]
    fn test_relu_mixed_values() {
        let input = fp16_array_from_f64(&[-1.0, 0.0, 1.0, -0.5, 2.0]);
        let output = encode_ane_activation(&input, ActivationFn::ReLU, 5);
        let env = empty_env();

        assert!(read_array_elem_f64(&output, 0, &env).abs() < 1e-6); // -1 -> 0
        assert!(read_array_elem_f64(&output, 1, &env).abs() < 1e-6); // 0 -> 0
        assert!((read_array_elem_f64(&output, 2, &env) - 1.0).abs() < 1e-6); // 1 -> 1
        assert!(read_array_elem_f64(&output, 3, &env).abs() < 1e-6); // -0.5 -> 0
        assert!((read_array_elem_f64(&output, 4, &env) - 2.0).abs() < 1e-6); // 2 -> 2
    }

    #[test]
    fn test_relu_zero() {
        let input = fp16_array_from_f64(&[0.0]);
        let output = encode_ane_activation(&input, ActivationFn::ReLU, 1);
        let env = empty_env();
        // ReLU(0) = 0 (0 is not < 0, so we take the x branch)
        assert!(read_array_elem_f64(&output, 0, &env).abs() < 1e-6);
    }

    #[test]
    fn test_leaky_relu_positive() {
        let input = fp16_array_from_f64(&[1.0, 2.0, 3.0]);
        let output = encode_ane_activation(&input, ActivationFn::LeakyReLU { alpha: 0.1 }, 3);
        let env = empty_env();

        // Positive values pass through unchanged.
        assert!((read_array_elem_f64(&output, 0, &env) - 1.0).abs() < 1e-6);
        assert!((read_array_elem_f64(&output, 1, &env) - 2.0).abs() < 1e-6);
        assert!((read_array_elem_f64(&output, 2, &env) - 3.0).abs() < 1e-6);
    }

    #[test]
    fn test_leaky_relu_negative() {
        let input = fp16_array_from_f64(&[-10.0, -1.0]);
        let output = encode_ane_activation(&input, ActivationFn::LeakyReLU { alpha: 0.1 }, 2);
        let env = empty_env();

        // Negative values are scaled by alpha.
        assert!((read_array_elem_f64(&output, 0, &env) - (-1.0)).abs() < 1e-6);
        assert!((read_array_elem_f64(&output, 1, &env) - (-0.1)).abs() < 1e-6);
    }

    #[test]
    fn test_sigmoid_uses_uf() {
        // Sigmoid uses UF abstraction -- cannot be concretely evaluated.
        // Verify the expression is constructed without panicking,
        // and that evaluation returns an error (UF is not evaluable).
        let input = fp16_array_from_f64(&[0.0]);
        let output = encode_ane_activation(&input, ActivationFn::Sigmoid, 1);

        // Trying to evaluate should fail because of UF.
        let idx = SmtExpr::bv_const(0, 64);
        let elem = SmtExpr::select(output, idx);
        let result = elem.try_eval(&empty_env());
        assert!(result.is_err(), "UF evaluation should fail");
    }

    #[test]
    fn test_tanh_uses_uf() {
        let input = fp16_array_from_f64(&[0.0]);
        let output = encode_ane_activation(&input, ActivationFn::Tanh, 1);
        let idx = SmtExpr::bv_const(0, 64);
        let elem = SmtExpr::select(output, idx);
        assert!(elem.try_eval(&empty_env()).is_err());
    }

    #[test]
    fn test_gelu_uses_uf() {
        let input = fp16_array_from_f64(&[0.0]);
        let output = encode_ane_activation(&input, ActivationFn::GELU, 1);
        let idx = SmtExpr::bv_const(0, 64);
        let elem = SmtExpr::select(output, idx);
        assert!(elem.try_eval(&empty_env()).is_err());
    }

    // =======================================================================
    // Element-wise operation tests
    // =======================================================================

    #[test]
    fn test_elementwise_add() {
        let a = fp16_array_from_f64(&[1.0, 2.0, 3.0]);
        let b = fp16_array_from_f64(&[4.0, 5.0, 6.0]);
        let result = encode_ane_elementwise(&a, &b, ElementWiseOp::Add, 3);
        let env = empty_env();

        assert!((read_array_elem_f64(&result, 0, &env) - 5.0).abs() < 1e-6);
        assert!((read_array_elem_f64(&result, 1, &env) - 7.0).abs() < 1e-6);
        assert!((read_array_elem_f64(&result, 2, &env) - 9.0).abs() < 1e-6);
    }

    #[test]
    fn test_elementwise_sub() {
        let a = fp16_array_from_f64(&[10.0, 20.0, 30.0]);
        let b = fp16_array_from_f64(&[1.0, 2.0, 3.0]);
        let result = encode_ane_elementwise(&a, &b, ElementWiseOp::Sub, 3);
        let env = empty_env();

        assert!((read_array_elem_f64(&result, 0, &env) - 9.0).abs() < 1e-6);
        assert!((read_array_elem_f64(&result, 1, &env) - 18.0).abs() < 1e-6);
        assert!((read_array_elem_f64(&result, 2, &env) - 27.0).abs() < 1e-6);
    }

    #[test]
    fn test_elementwise_mul() {
        let a = fp16_array_from_f64(&[2.0, 3.0, 4.0]);
        let b = fp16_array_from_f64(&[5.0, 6.0, 7.0]);
        let result = encode_ane_elementwise(&a, &b, ElementWiseOp::Mul, 3);
        let env = empty_env();

        assert!((read_array_elem_f64(&result, 0, &env) - 10.0).abs() < 1e-6);
        assert!((read_array_elem_f64(&result, 1, &env) - 18.0).abs() < 1e-6);
        assert!((read_array_elem_f64(&result, 2, &env) - 28.0).abs() < 1e-6);
    }

    #[test]
    fn test_elementwise_div() {
        let a = fp16_array_from_f64(&[10.0, 21.0, 8.0]);
        let b = fp16_array_from_f64(&[2.0, 3.0, 4.0]);
        let result = encode_ane_elementwise(&a, &b, ElementWiseOp::Div, 3);
        let env = empty_env();

        assert!((read_array_elem_f64(&result, 0, &env) - 5.0).abs() < 1e-6);
        assert!((read_array_elem_f64(&result, 1, &env) - 7.0).abs() < 1e-6);
        assert!((read_array_elem_f64(&result, 2, &env) - 2.0).abs() < 1e-6);
    }

    // =======================================================================
    // FP16 quantization tests
    // =======================================================================

    #[test]
    fn test_fp16_quantize_elem_is_uf() {
        // encode_fp16_quantize_elem uses UF abstraction.
        let fp32_val = SmtExpr::fp32_const(1.5f32);
        let fp16_val = encode_fp16_quantize_elem(&fp32_val);
        // Should produce a UF node.
        assert!(fp16_val.try_eval(&empty_env()).is_err());
    }

    #[test]
    fn test_fp16_quantize_array_structure() {
        // Verify the quantization produces an array with n elements.
        let fp32_arr = fp32_array_from_f64(&[1.0, 2.0, 3.0]);
        let fp16_arr = encode_fp16_quantize(&fp32_arr, 3);
        // Cannot evaluate (UF), but verify structure doesn't panic.
        let _sort = fp16_arr.sort();
    }

    #[test]
    fn test_bounded_error_check_small_diff() {
        // Check that |1.0 - 1.001| < 0.01 is true.
        let fp32_val = SmtExpr::fp64_const(1.0);
        let fp16_val = SmtExpr::fp64_const(1.001);
        let check = encode_bounded_error_check(&fp32_val, &fp16_val, 0.01);
        let result = check.try_eval(&empty_env()).unwrap();
        assert_eq!(result, EvalResult::Bool(true));
    }

    #[test]
    fn test_bounded_error_check_large_diff() {
        // Check that |1.0 - 2.0| < 0.5 is false (diff = 1.0 > 0.5).
        let fp32_val = SmtExpr::fp64_const(1.0);
        let fp16_val = SmtExpr::fp64_const(2.0);
        let check = encode_bounded_error_check(&fp32_val, &fp16_val, 0.5);
        let result = check.try_eval(&empty_env()).unwrap();
        assert_eq!(result, EvalResult::Bool(false));
    }

    #[test]
    fn test_bounded_error_check_negative_diff() {
        // Check that |1.0 - 0.5| < 0.6 is true (diff = 0.5 < 0.6).
        let fp32_val = SmtExpr::fp64_const(1.0);
        let fp16_val = SmtExpr::fp64_const(0.5);
        let check = encode_bounded_error_check(&fp32_val, &fp16_val, 0.6);
        let result = check.try_eval(&empty_env()).unwrap();
        assert_eq!(result, EvalResult::Bool(true));
    }

    #[test]
    fn test_bounded_error_check_exact_match() {
        // Check that |5.0 - 5.0| < 0.001 is true.
        let fp32_val = SmtExpr::fp64_const(5.0);
        let fp16_val = SmtExpr::fp64_const(5.0);
        let check = encode_bounded_error_check(&fp32_val, &fp16_val, 0.001);
        let result = check.try_eval(&empty_env()).unwrap();
        assert_eq!(result, EvalResult::Bool(true));
    }

    // =======================================================================
    // Integration tests: GEMM + activation (fused pattern)
    // =======================================================================

    #[test]
    fn test_gemm_then_relu() {
        // GEMM producing negative values, then ReLU clamps them.
        // A = [[-1, 2]], B = [[1], [1]] => C = [[-1+2]] = [[1]]
        // But let's use values that produce a negative:
        // A = [[-3, 1]], B = [[1], [1]] => C = [[-3+1]] = [[-2]]
        // Then ReLU(-2) = 0.
        let a = fp16_array_from_f64(&[-3.0, 1.0]);
        let b = fp16_array_from_f64(&[1.0, 1.0]);
        let gemm_result = encode_ane_gemm(&a, &b, 1, 2, 1);
        let relu_result = encode_ane_activation(&gemm_result, ActivationFn::ReLU, 1);

        let env = empty_env();
        let val = read_array_elem_f64(&relu_result, 0, &env);
        assert!(
            val.abs() < 1e-6,
            "ReLU of -2 should be 0, got {}",
            val
        );
    }

    #[test]
    fn test_gemm_then_relu_positive() {
        // A = [[1, 2]], B = [[3], [4]] => C = [[1*3+2*4]] = [[11]]
        // ReLU(11) = 11
        let a = fp16_array_from_f64(&[1.0, 2.0]);
        let b = fp16_array_from_f64(&[3.0, 4.0]);
        let gemm_result = encode_ane_gemm(&a, &b, 1, 2, 1);
        let relu_result = encode_ane_activation(&gemm_result, ActivationFn::ReLU, 1);

        let env = empty_env();
        let val = read_array_elem_f64(&relu_result, 0, &env);
        assert!(
            (val - 11.0).abs() < 1e-6,
            "expected 11.0, got {}",
            val
        );
    }

    // =======================================================================
    // Helper function tests
    // =======================================================================

    #[test]
    fn test_fp16_array_from_f64_roundtrip() {
        let values = &[1.0, -2.5, 0.0, 100.0, -0.125];
        let arr = fp16_array_from_f64(values);
        let env = empty_env();

        for (i, &expected) in values.iter().enumerate() {
            let actual = read_array_elem_f64(&arr, i as u64, &env);
            assert!(
                (actual - expected).abs() < 1e-6,
                "index {}: expected {}, got {}",
                i,
                expected,
                actual
            );
        }
    }

    #[test]
    fn test_fp32_array_from_f64_roundtrip() {
        let values = &[1.5, -3.0, 0.0, 42.0];
        let arr = fp32_array_from_f64(values);
        let env = empty_env();

        for (i, &expected) in values.iter().enumerate() {
            let actual = read_array_elem_f64(&arr, i as u64, &env);
            assert!(
                (actual - expected).abs() < 1e-3,
                "index {}: expected {}, got {}",
                i,
                expected,
                actual
            );
        }
    }

    #[test]
    fn test_flatten_4d_basic() {
        // NCHW: (0,0,0,0) in (1,1,3,3) = 0
        assert_eq!(flatten_4d(0, 0, 0, 0, 1, 3, 3), 0);
        // (0,0,0,1) = 1
        assert_eq!(flatten_4d(0, 0, 0, 1, 1, 3, 3), 1);
        // (0,0,1,0) = 3
        assert_eq!(flatten_4d(0, 0, 1, 0, 1, 3, 3), 3);
        // (0,0,2,2) = 8
        assert_eq!(flatten_4d(0, 0, 2, 2, 1, 3, 3), 8);
    }

    #[test]
    fn test_gemm_uf_produces_uf_node() {
        let a = fp16_array_from_f64(&[1.0]);
        let b = fp16_array_from_f64(&[2.0]);
        let result = encode_ane_gemm_uf(&a, &b);
        // Should be a UF application node -- cannot evaluate.
        assert!(result.try_eval(&empty_env()).is_err());
    }
}
