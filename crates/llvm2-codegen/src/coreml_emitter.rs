// llvm2-codegen/coreml_emitter.rs - CoreML MIL operation emitter for ANE targeting
//
// Author: Andrew Yates <ayates@dropbox.com>
// Copyright 2026 Dropbox, Inc. | License: Apache-2.0
//
// Generates CoreML Model Intermediate Language (MIL) operation graphs from
// AneTensorShape and ANE operation descriptors. The MIL program is serialized
// to protobuf (.mlmodel) for compilation via CoreML's compileModel API.
//
// Design doc: designs/2026-04-14-coreml-ane-lowering.md
// Reference: Apple. "Core ML Model Specification." coremltools documentation.
// Reference: Apple. "MIL Ops." coremltools source, converters/mil/mil/ops.

//! CoreML Model Intermediate Language (MIL) operation generation for ANE targeting.
//!
//! This module implements the MIL program emitter for Neural Engine dispatch.
//! It translates verified ANE operations (from `ane_semantics.rs`) into CoreML's
//! SSA-based tensor operation graph (MIL), which CoreML then compiles and routes
//! to the Apple Neural Engine.
//!
//! # Supported Operations
//!
//! - **GEMM**: `mil.matmul(A, B)` -- general matrix multiply
//! - **Conv2D**: `mil.conv(input, weight, bias, ...)` -- 2D convolution
//! - **Activations**: relu, leaky_relu, sigmoid, tanh, gelu
//! - **Element-wise**: add, sub, mul, real_div
//! - **Reduce**: reduce_sum, reduce_mean, reduce_max, reduce_min
//!
//! # Fused Patterns
//!
//! The emitter detects and emits fused operation patterns that CoreML maps
//! to single ANE passes: GEMM+bias+ReLU, Conv+BatchNorm+ReLU, MatMul+GELU.

use std::collections::HashMap;
use std::fmt;

use llvm2_lower::compute_graph::{ComputeNode, ComputeNodeId, NodeKind};
use llvm2_lower::target_analysis::ComputeTarget;

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

/// Errors that can occur during CoreML MIL emission from ComputeGraph nodes.
#[derive(Debug, Clone)]
pub enum CoreMLEmitError {
    /// A node is not legal on the NeuralEngine target.
    NotAneCompatible { node_id: ComputeNodeId, kind: NodeKind },
    /// Empty node list provided.
    EmptyNodeList,
    /// A node references a predecessor that was not emitted yet.
    MissingPredecessor { node_id: ComputeNodeId, missing_dep: String },
    /// Unsupported dominant operation for ANE lowering.
    UnsupportedOp { node_id: ComputeNodeId, op: String },
}

impl fmt::Display for CoreMLEmitError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CoreMLEmitError::NotAneCompatible { node_id, kind } => {
                write!(f, "node {} ({}) is not ANE-compatible", node_id, kind)
            }
            CoreMLEmitError::EmptyNodeList => write!(f, "empty node list"),
            CoreMLEmitError::MissingPredecessor { node_id, missing_dep } => {
                write!(
                    f,
                    "node {} references missing predecessor '{}'",
                    node_id, missing_dep
                )
            }
            CoreMLEmitError::UnsupportedOp { node_id, op } => {
                write!(f, "node {} has unsupported op '{}' for ANE", node_id, op)
            }
        }
    }
}

impl std::error::Error for CoreMLEmitError {}

// ---------------------------------------------------------------------------
// MIL data types
// ---------------------------------------------------------------------------

/// CoreML MIL tensor element data type.
///
/// Maps to CoreML's `MLMultiArrayDataType` and MIL type annotations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MilDataType {
    /// IEEE 754 half-precision (FP16). Primary ANE type.
    Float16,
    /// IEEE 754 single-precision (FP32). Requires quantization for ANE.
    Float32,
    /// 32-bit signed integer. Limited ANE support.
    Int32,
}

impl fmt::Display for MilDataType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MilDataType::Float16 => write!(f, "fp16"),
            MilDataType::Float32 => write!(f, "fp32"),
            MilDataType::Int32 => write!(f, "i32"),
        }
    }
}

// ---------------------------------------------------------------------------
// Tensor shape for MIL
// ---------------------------------------------------------------------------

/// Static tensor shape for MIL operations (NCHW layout).
///
/// All dimensions must be known at compile time for ANE execution.
/// Mirrors `AneTensorShape` from `ane_semantics.rs` but decoupled from
/// the SMT encoding layer.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct MilTensorShape {
    /// Batch dimension (N). 1 for non-batched.
    pub batch: u64,
    /// Channel dimension (C).
    pub channels: u64,
    /// Height dimension (H).
    pub height: u64,
    /// Width dimension (W).
    pub width: u64,
}

impl MilTensorShape {
    /// Create a 2D matrix shape (M x N), stored as (1, 1, M, N) in NCHW.
    pub fn matrix(rows: u64, cols: u64) -> Self {
        MilTensorShape { batch: 1, channels: 1, height: rows, width: cols }
    }

    /// Create a 4D tensor shape.
    pub fn tensor_4d(batch: u64, channels: u64, height: u64, width: u64) -> Self {
        MilTensorShape { batch, channels, height, width }
    }

    /// Create a 1D vector shape (1, 1, 1, N).
    pub fn vector(length: u64) -> Self {
        MilTensorShape { batch: 1, channels: 1, height: 1, width: length }
    }

    /// Total number of elements.
    pub fn numel(&self) -> u64 {
        self.batch * self.channels * self.height * self.width
    }

    /// Return shape as a 4-element array [N, C, H, W].
    pub fn dims(&self) -> [u64; 4] {
        [self.batch, self.channels, self.height, self.width]
    }

    /// Rank of the shape (number of non-trivial dimensions from the left).
    pub fn rank(&self) -> u32 {
        if self.batch > 1 { 4 }
        else if self.channels > 1 { 3 }
        else if self.height > 1 { 2 }
        else { 1 }
    }
}

impl fmt::Display for MilTensorShape {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "({}, {}, {}, {})", self.batch, self.channels, self.height, self.width)
    }
}

// ---------------------------------------------------------------------------
// MIL SSA values
// ---------------------------------------------------------------------------

/// An SSA value reference in the MIL program.
///
/// MIL is SSA-based: every value is defined exactly once and referenced by name.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct MilValue {
    /// SSA name (e.g., "x_0", "matmul_1", "relu_2").
    pub name: String,
    /// Tensor shape (static).
    pub shape: MilTensorShape,
    /// Element data type.
    pub dtype: MilDataType,
}

impl MilValue {
    pub fn new(name: &str, shape: MilTensorShape, dtype: MilDataType) -> Self {
        MilValue { name: name.to_string(), shape, dtype }
    }
}

// ---------------------------------------------------------------------------
// MIL operations
// ---------------------------------------------------------------------------

/// A single MIL operation in the program graph.
///
/// Each operation consumes SSA inputs and produces an SSA output.
/// The operation names and semantics follow Apple's MIL specification.
///
/// Ref: https://github.com/apple/coremltools/tree/main/coremltools/converters/mil/mil/ops
#[derive(Debug, Clone)]
pub enum MilOperation {
    /// `mil.matmul(x, y) -> tensor`
    MatMul {
        output: String,
        x: String,
        y: String,
        transpose_x: bool,
        transpose_y: bool,
    },

    /// `mil.conv(x, weight, bias, strides, pad_type, dilations, groups) -> tensor`
    Conv {
        output: String,
        x: String,
        weight: String,
        bias: Option<String>,
        strides: [u64; 2],
        pad_type: PadType,
        dilations: [u64; 2],
        groups: u64,
    },

    /// Element-wise binary: `mil.add`, `mil.sub`, `mil.mul`, `mil.real_div`.
    ElementWise {
        output: String,
        op: MilElementWiseOp,
        x: String,
        y: String,
    },

    /// Activation function: `mil.relu`, `mil.sigmoid`, etc.
    Activation {
        output: String,
        op: MilActivationOp,
        x: String,
    },

    /// Reduction: `mil.reduce_sum`, `mil.reduce_mean`, etc.
    Reduce {
        output: String,
        op: MilReduceOp,
        x: String,
        axes: Vec<i64>,
        keep_dims: bool,
    },

    /// Reshape: `mil.reshape(x, shape) -> tensor`
    Reshape {
        output: String,
        x: String,
        shape: Vec<i64>,
    },

    /// Transpose: `mil.transpose(x, perm) -> tensor`
    Transpose {
        output: String,
        x: String,
        perm: Vec<u32>,
    },
}

impl MilOperation {
    /// Return the output SSA name for this operation.
    pub fn output_name(&self) -> &str {
        match self {
            MilOperation::MatMul { output, .. } => output,
            MilOperation::Conv { output, .. } => output,
            MilOperation::ElementWise { output, .. } => output,
            MilOperation::Activation { output, .. } => output,
            MilOperation::Reduce { output, .. } => output,
            MilOperation::Reshape { output, .. } => output,
            MilOperation::Transpose { output, .. } => output,
        }
    }

    /// Return the MIL operation type name as a string.
    pub fn op_type(&self) -> &'static str {
        match self {
            MilOperation::MatMul { .. } => "matmul",
            MilOperation::Conv { .. } => "conv",
            MilOperation::ElementWise { op, .. } => op.mil_name(),
            MilOperation::Activation { op, .. } => op.mil_name(),
            MilOperation::Reduce { op, .. } => op.mil_name(),
            MilOperation::Reshape { .. } => "reshape",
            MilOperation::Transpose { .. } => "transpose",
        }
    }
}

// ---------------------------------------------------------------------------
// Sub-enums for operation variants
// ---------------------------------------------------------------------------

/// Element-wise binary operations in MIL.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MilElementWiseOp {
    Add,
    Sub,
    Mul,
    RealDiv,
}

impl MilElementWiseOp {
    /// MIL operation name.
    pub fn mil_name(&self) -> &'static str {
        match self {
            MilElementWiseOp::Add => "add",
            MilElementWiseOp::Sub => "sub",
            MilElementWiseOp::Mul => "mul",
            MilElementWiseOp::RealDiv => "real_div",
        }
    }
}

/// Activation functions supported by MIL / ANE.
#[derive(Debug, Clone, PartialEq)]
pub enum MilActivationOp {
    ReLU,
    LeakyReLU { alpha: f32 },
    Sigmoid,
    Tanh,
    GELU { mode: GeLUMode },
}

impl MilActivationOp {
    /// MIL operation name.
    pub fn mil_name(&self) -> &'static str {
        match self {
            MilActivationOp::ReLU => "relu",
            MilActivationOp::LeakyReLU { .. } => "leaky_relu",
            MilActivationOp::Sigmoid => "sigmoid",
            MilActivationOp::Tanh => "tanh",
            MilActivationOp::GELU { .. } => "gelu",
        }
    }
}

/// GELU approximation mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GeLUMode {
    /// Exact: `x * 0.5 * (1 + erf(x / sqrt(2)))`.
    Exact,
    /// Tanh approximation (faster, used in BERT etc.).
    TanhApprox,
}

/// Reduction operations in MIL.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MilReduceOp {
    Sum,
    Mean,
    Max,
    Min,
}

impl MilReduceOp {
    /// MIL operation name.
    pub fn mil_name(&self) -> &'static str {
        match self {
            MilReduceOp::Sum => "reduce_sum",
            MilReduceOp::Mean => "reduce_mean",
            MilReduceOp::Max => "reduce_max",
            MilReduceOp::Min => "reduce_min",
        }
    }
}

/// Padding type for convolution.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PadType {
    /// No padding.
    Valid,
    /// Pad to preserve spatial dimensions.
    Same,
    /// Custom padding (specified separately).
    Custom,
}

impl fmt::Display for PadType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PadType::Valid => write!(f, "valid"),
            PadType::Same => write!(f, "same"),
            PadType::Custom => write!(f, "custom"),
        }
    }
}

// ---------------------------------------------------------------------------
// CoreML compute unit preference
// ---------------------------------------------------------------------------

/// CoreML compute unit routing preference.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MlComputeUnits {
    /// Let CoreML decide (default). May use CPU, GPU, or ANE.
    All,
    /// Prefer CPU + Neural Engine (skip GPU).
    CpuAndNeuralEngine,
    /// CPU only (fallback).
    CpuOnly,
}

// ---------------------------------------------------------------------------
// CoreML feature descriptor
// ---------------------------------------------------------------------------

/// A model input or output feature description.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CoreMLFeature {
    /// Feature name (matches model spec).
    pub name: String,
    /// Tensor shape (static, NCHW).
    pub shape: MilTensorShape,
    /// Element data type.
    pub dtype: MilDataType,
}

impl CoreMLFeature {
    pub fn new(name: &str, shape: MilTensorShape, dtype: MilDataType) -> Self {
        CoreMLFeature { name: name.to_string(), shape, dtype }
    }
}

// ---------------------------------------------------------------------------
// MIL program (SSA operation graph)
// ---------------------------------------------------------------------------

/// A complete MIL program representing a CoreML model's computation.
///
/// The program is an ordered sequence of SSA operations with declared
/// inputs and outputs. It is serialized to protobuf for `.mlmodel` emission.
#[derive(Debug, Clone)]
pub struct MilProgram {
    /// Model inputs (feature descriptions).
    pub inputs: Vec<CoreMLFeature>,
    /// Model outputs (feature descriptions).
    pub outputs: Vec<CoreMLFeature>,
    /// Ordered sequence of MIL operations (SSA, topologically sorted).
    pub operations: Vec<MilOperation>,
    /// CoreML specification version (7+ for MIL support).
    pub spec_version: u32,
}

impl MilProgram {
    /// Create an empty MIL program with default spec version 8 (iOS 18).
    pub fn new() -> Self {
        MilProgram {
            inputs: Vec::new(),
            outputs: Vec::new(),
            operations: Vec::new(),
            spec_version: 8,
        }
    }

    /// Add an input feature to the model.
    pub fn add_input(&mut self, feature: CoreMLFeature) {
        self.inputs.push(feature);
    }

    /// Add an output feature to the model.
    pub fn add_output(&mut self, feature: CoreMLFeature) {
        self.outputs.push(feature);
    }

    /// Append an operation to the program.
    pub fn push_op(&mut self, op: MilOperation) {
        self.operations.push(op);
    }

    /// Return the total number of operations.
    pub fn op_count(&self) -> usize {
        self.operations.len()
    }

    /// Validate the program: check that all input references resolve to
    /// either program inputs or earlier operation outputs.
    ///
    /// Returns `Ok(())` if valid, or an error describing the first
    /// unresolved reference.
    pub fn validate(&self) -> Result<(), String> {
        let mut defined: std::collections::HashSet<&str> = std::collections::HashSet::new();

        // Program inputs are defined
        for inp in &self.inputs {
            defined.insert(&inp.name);
        }

        // Check each operation's inputs
        for op in &self.operations {
            let refs = op_input_refs(op);
            for r in &refs {
                if !defined.contains(r.as_str()) {
                    return Err(format!(
                        "MIL validation: operation '{}' references undefined value '{}'",
                        op.output_name(),
                        r,
                    ));
                }
            }
            defined.insert(op.output_name());
        }

        // Check outputs reference defined values
        for out in &self.outputs {
            if !defined.contains(out.name.as_str()) {
                return Err(format!(
                    "MIL validation: output '{}' references undefined value",
                    out.name,
                ));
            }
        }

        Ok(())
    }
}

impl Default for MilProgram {
    fn default() -> Self {
        Self::new()
    }
}

/// Collect all input SSA name references for an operation.
fn op_input_refs(op: &MilOperation) -> Vec<String> {
    match op {
        MilOperation::MatMul { x, y, .. } => vec![x.clone(), y.clone()],
        MilOperation::Conv { x, weight, bias, .. } => {
            let mut refs = vec![x.clone(), weight.clone()];
            if let Some(b) = bias {
                refs.push(b.clone());
            }
            refs
        }
        MilOperation::ElementWise { x, y, .. } => vec![x.clone(), y.clone()],
        MilOperation::Activation { x, .. } => vec![x.clone()],
        MilOperation::Reduce { x, .. } => vec![x.clone()],
        MilOperation::Reshape { x, .. } => vec![x.clone()],
        MilOperation::Transpose { x, .. } => vec![x.clone()],
    }
}

// ---------------------------------------------------------------------------
// CoreML MIL emitter
// ---------------------------------------------------------------------------

/// Emits MIL programs from ANE operation descriptors.
///
/// The emitter translates high-level ANE operations (matching those verified
/// by `ane_semantics.rs`) into MIL SSA operations. The resulting `MilProgram`
/// can be serialized to protobuf for `.mlmodel` output.
///
/// Ref: designs/2026-04-14-coreml-ane-lowering.md
pub struct CoreMLEmitter {
    /// Auto-incrementing counter for SSA value names.
    next_id: u32,
    /// Target data type for ANE operations.
    dtype: MilDataType,
}

impl CoreMLEmitter {
    /// Create a new emitter targeting FP16 by default.
    pub fn new() -> Self {
        CoreMLEmitter {
            next_id: 0,
            dtype: MilDataType::Float16,
        }
    }

    /// Create a new emitter with a specified data type.
    pub fn with_dtype(dtype: MilDataType) -> Self {
        CoreMLEmitter { next_id: 0, dtype }
    }

    /// Generate a fresh SSA name with the given prefix.
    fn fresh_name(&mut self, prefix: &str) -> String {
        let name = format!("{}_{}", prefix, self.next_id);
        self.next_id += 1;
        name
    }

    /// Emit a matrix multiply operation: `C = matmul(A, B)`.
    ///
    /// Corresponds to `encode_ane_gemm()` in `ane_semantics.rs`.
    pub fn emit_matmul(
        &mut self,
        program: &mut MilProgram,
        x: &str,
        y: &str,
        transpose_x: bool,
        transpose_y: bool,
    ) -> String {
        let name = self.fresh_name("matmul");
        program.push_op(MilOperation::MatMul {
            output: name.clone(),
            x: x.to_string(),
            y: y.to_string(),
            transpose_x,
            transpose_y,
        });
        name
    }

    /// Emit an element-wise binary operation.
    ///
    /// Corresponds to `encode_ane_elementwise()` in `ane_semantics.rs`.
    pub fn emit_elementwise(
        &mut self,
        program: &mut MilProgram,
        op: MilElementWiseOp,
        x: &str,
        y: &str,
    ) -> String {
        let name = self.fresh_name(op.mil_name());
        program.push_op(MilOperation::ElementWise {
            output: name.clone(),
            op,
            x: x.to_string(),
            y: y.to_string(),
        });
        name
    }

    /// Emit an activation function.
    ///
    /// Corresponds to `encode_ane_activation()` in `ane_semantics.rs`.
    pub fn emit_activation(
        &mut self,
        program: &mut MilProgram,
        op: MilActivationOp,
        x: &str,
    ) -> String {
        let name = self.fresh_name(op.mil_name());
        program.push_op(MilOperation::Activation {
            output: name.clone(),
            op,
            x: x.to_string(),
        });
        name
    }

    /// Emit a 2D convolution operation.
    ///
    /// Corresponds to `encode_ane_conv2d()` in `ane_semantics.rs`.
    pub fn emit_conv2d(
        &mut self,
        program: &mut MilProgram,
        x: &str,
        weight: &str,
        bias: Option<&str>,
        strides: [u64; 2],
        pad_type: PadType,
        dilations: [u64; 2],
        groups: u64,
    ) -> String {
        let name = self.fresh_name("conv");
        program.push_op(MilOperation::Conv {
            output: name.clone(),
            x: x.to_string(),
            weight: weight.to_string(),
            bias: bias.map(|s| s.to_string()),
            strides,
            pad_type,
            dilations,
            groups,
        });
        name
    }

    /// Emit a reduction operation.
    pub fn emit_reduce(
        &mut self,
        program: &mut MilProgram,
        op: MilReduceOp,
        x: &str,
        axes: &[i64],
        keep_dims: bool,
    ) -> String {
        let name = self.fresh_name(op.mil_name());
        program.push_op(MilOperation::Reduce {
            output: name.clone(),
            op,
            x: x.to_string(),
            axes: axes.to_vec(),
            keep_dims,
        });
        name
    }

    /// Emit a fused GEMM + bias + ReLU pattern (single ANE pass).
    ///
    /// This is a common fusion pattern that CoreML recognizes and maps to
    /// a single ANE pass for maximum throughput.
    pub fn emit_gemm_bias_relu(
        &mut self,
        program: &mut MilProgram,
        x: &str,
        weight: &str,
        bias: &str,
    ) -> String {
        let mm = self.emit_matmul(program, x, weight, false, false);
        let add = self.emit_elementwise(program, MilElementWiseOp::Add, &mm, bias);
        self.emit_activation(program, MilActivationOp::ReLU, &add)
    }

    /// Emit a fused MatMul + GELU pattern (single ANE pass).
    pub fn emit_matmul_gelu(
        &mut self,
        program: &mut MilProgram,
        x: &str,
        weight: &str,
    ) -> String {
        let mm = self.emit_matmul(program, x, weight, false, false);
        self.emit_activation(
            program,
            MilActivationOp::GELU { mode: GeLUMode::Exact },
            &mm,
        )
    }

    /// Return the current SSA counter (for testing/debugging).
    pub fn next_id(&self) -> u32 {
        self.next_id
    }

    /// Return the target data type.
    pub fn dtype(&self) -> MilDataType {
        self.dtype
    }
}

impl Default for CoreMLEmitter {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// ComputeGraph -> MIL program generation
// ---------------------------------------------------------------------------

/// Fusion pattern detected between consecutive ANE-targeted nodes.
///
/// CoreML maps these fused patterns to single ANE passes for maximum
/// throughput. Detection runs on consecutive node triples/pairs before
/// individual emission.
#[derive(Debug, Clone, PartialEq, Eq)]
enum FusionPattern {
    /// Conv2D -> BatchNorm -> ReLU  (single ANE pass)
    ConvBnRelu { conv_idx: usize, bn_idx: usize, relu_idx: usize },
    /// GEMM -> Bias(Add) -> Activation  (single ANE pass)
    GemmBiasAct { gemm_idx: usize, bias_idx: usize, act_idx: usize },
    /// MatMul -> GELU  (single ANE pass)
    MatMulGelu { mm_idx: usize, gelu_idx: usize },
}

/// Detect fusion patterns in a sequence of ANE-targeted nodes.
///
/// Returns a list of fusion patterns with indices into the node slice.
/// Each node index can appear in at most one pattern; earlier patterns win.
fn detect_fusion_patterns(nodes: &[ComputeNode]) -> Vec<FusionPattern> {
    let mut patterns = Vec::new();
    let mut consumed: std::collections::HashSet<usize> = std::collections::HashSet::new();

    // Pass 1: detect 3-node fusions (higher priority)
    for i in 0..nodes.len().saturating_sub(2) {
        if consumed.contains(&i) || consumed.contains(&(i + 1)) || consumed.contains(&(i + 2)) {
            continue;
        }

        let op_a = nodes[i].dominant_op.as_str();
        let op_b = nodes[i + 1].dominant_op.as_str();
        let op_c = nodes[i + 2].dominant_op.as_str();

        // Conv -> BN -> ReLU
        if is_conv_op(op_a) && is_batchnorm_op(op_b) && is_relu_op(op_c) {
            patterns.push(FusionPattern::ConvBnRelu {
                conv_idx: i,
                bn_idx: i + 1,
                relu_idx: i + 2,
            });
            consumed.insert(i);
            consumed.insert(i + 1);
            consumed.insert(i + 2);
            continue;
        }

        // GEMM -> Bias(ADD) -> Activation
        if is_gemm_op(op_a) && is_bias_op(op_b) && is_activation_op(op_c) {
            patterns.push(FusionPattern::GemmBiasAct {
                gemm_idx: i,
                bias_idx: i + 1,
                act_idx: i + 2,
            });
            consumed.insert(i);
            consumed.insert(i + 1);
            consumed.insert(i + 2);
            continue;
        }
    }

    // Pass 2: detect 2-node fusions
    for i in 0..nodes.len().saturating_sub(1) {
        if consumed.contains(&i) || consumed.contains(&(i + 1)) {
            continue;
        }

        let op_a = nodes[i].dominant_op.as_str();
        let op_b = nodes[i + 1].dominant_op.as_str();

        // MatMul -> GELU
        if is_gemm_op(op_a) && is_gelu_op(op_b) {
            patterns.push(FusionPattern::MatMulGelu {
                mm_idx: i,
                gelu_idx: i + 1,
            });
            consumed.insert(i);
            consumed.insert(i + 1);
        }
    }

    patterns
}

/// Helper: is this op a convolution?
fn is_conv_op(op: &str) -> bool {
    matches!(op, "CONV" | "CONV2D" | "Conv2D" | "conv" | "conv2d")
}

/// Helper: is this op batch normalization?
fn is_batchnorm_op(op: &str) -> bool {
    matches!(op, "BN" | "BATCHNORM" | "BatchNorm" | "batchnorm" | "bn")
}

/// Helper: is this op ReLU?
fn is_relu_op(op: &str) -> bool {
    matches!(op, "RELU" | "ReLU" | "relu")
}

/// Helper: is this op GEMM/matmul?
fn is_gemm_op(op: &str) -> bool {
    matches!(op, "GEMM" | "gemm" | "MATMUL" | "matmul" | "MatMul")
}

/// Helper: is this op a bias addition?
fn is_bias_op(op: &str) -> bool {
    matches!(op, "ADD" | "BIAS" | "add" | "bias")
}

/// Helper: is this op any activation function?
fn is_activation_op(op: &str) -> bool {
    matches!(
        op,
        "RELU" | "ReLU" | "relu" | "SIGMOID" | "sigmoid" | "TANH"
            | "tanh" | "GELU" | "gelu" | "LEAKY_RELU" | "leaky_relu"
    )
}

/// Helper: is this op GELU specifically?
fn is_gelu_op(op: &str) -> bool {
    matches!(op, "GELU" | "gelu")
}

/// Helper: is this op a reduction?
fn is_reduce_op(op: &str) -> bool {
    matches!(
        op,
        "REDUCE_SUM" | "reduce_sum" | "REDUCE_MEAN" | "reduce_mean"
            | "REDUCE_MAX" | "reduce_max" | "REDUCE_MIN" | "reduce_min"
            | "SUM" | "sum" | "MEAN" | "mean"
    )
}

/// Map a dominant_op string to a MilActivationOp.
fn activation_from_op(op: &str) -> MilActivationOp {
    match op {
        "RELU" | "ReLU" | "relu" => MilActivationOp::ReLU,
        "SIGMOID" | "sigmoid" => MilActivationOp::Sigmoid,
        "TANH" | "tanh" => MilActivationOp::Tanh,
        "GELU" | "gelu" => MilActivationOp::GELU { mode: GeLUMode::TanhApprox },
        "LEAKY_RELU" | "leaky_relu" => MilActivationOp::LeakyReLU { alpha: 0.01 },
        _ => MilActivationOp::ReLU, // default fallback
    }
}

/// Map a dominant_op string to a MilElementWiseOp.
fn elementwise_from_op(op: &str) -> MilElementWiseOp {
    match op {
        "ADD" | "add" | "BIAS" | "bias" => MilElementWiseOp::Add,
        "SUB" | "sub" => MilElementWiseOp::Sub,
        "MUL" | "mul" => MilElementWiseOp::Mul,
        "DIV" | "div" | "REAL_DIV" | "real_div" => MilElementWiseOp::RealDiv,
        _ => MilElementWiseOp::Add,
    }
}

/// Map a dominant_op string to a MilReduceOp.
fn reduce_from_op(op: &str) -> MilReduceOp {
    match op {
        "REDUCE_SUM" | "reduce_sum" | "SUM" | "sum" => MilReduceOp::Sum,
        "REDUCE_MEAN" | "reduce_mean" | "MEAN" | "mean" => MilReduceOp::Mean,
        "REDUCE_MAX" | "reduce_max" | "MAX" | "max" => MilReduceOp::Max,
        "REDUCE_MIN" | "reduce_min" | "MIN" | "min" => MilReduceOp::Min,
        _ => MilReduceOp::Sum,
    }
}

/// Determine the MIL tensor shape for a node based on its data size and kind.
///
/// Uses the node's `data_size_bytes` to estimate a reasonable shape. For
/// matrix-heavy nodes, assumes square matrices in FP16. For data-parallel,
/// assumes a 1D vector.
fn shape_from_node(node: &ComputeNode, dtype: MilDataType) -> MilTensorShape {
    let elem_bytes: u64 = match dtype {
        MilDataType::Float16 => 2,
        MilDataType::Float32 => 4,
        MilDataType::Int32 => 4,
    };

    let num_elements = node.data_size_bytes.max(elem_bytes) / elem_bytes;

    match node.kind {
        NodeKind::MatrixHeavy => {
            // Estimate square matrix dimensions
            let side = (num_elements as f64).sqrt() as u64;
            let side = side.max(1);
            MilTensorShape::matrix(side, side)
        }
        NodeKind::DataParallel => {
            MilTensorShape::vector(num_elements.max(1))
        }
        NodeKind::Scalar => {
            MilTensorShape::vector(num_elements.max(1))
        }
    }
}

/// SSA name for a ComputeNode's weight/parameter input.
fn node_weight_name(node: &ComputeNode) -> String {
    format!("weight_{}", node.id.0)
}

/// SSA name for a ComputeNode's bias input.
fn node_bias_name(node: &ComputeNode) -> String {
    format!("bias_{}", node.id.0)
}

impl CoreMLEmitter {
    /// Generate a complete MIL program from a slice of ANE-targeted ComputeNodes.
    ///
    /// Each node must have `ComputeTarget::NeuralEngine` in its `legal_targets`.
    /// The emitter:
    /// 1. Detects fusion patterns (Conv-BN-ReLU, GEMM-Bias-Act, MatMul-GELU)
    /// 2. Emits fused patterns as combined operations
    /// 3. Emits remaining nodes individually
    /// 4. Wires SSA references using node input/output relationships
    ///
    /// Returns a `MilProgram` with inputs derived from the first node and
    /// outputs derived from the last node.
    pub fn emit_program_from_nodes(
        &mut self,
        nodes: &[ComputeNode],
    ) -> Result<MilProgram, CoreMLEmitError> {
        if nodes.is_empty() {
            return Err(CoreMLEmitError::EmptyNodeList);
        }

        // Validate all nodes are ANE-compatible
        for node in nodes {
            if !node.legal_targets.contains(&ComputeTarget::NeuralEngine) {
                return Err(CoreMLEmitError::NotAneCompatible {
                    node_id: node.id,
                    kind: node.kind,
                });
            }
        }

        let mut program = MilProgram::new();

        // The first node's input is the program input
        let input_shape = shape_from_node(&nodes[0], self.dtype);
        let input_name = "input_0".to_string();
        program.add_input(CoreMLFeature::new(&input_name, input_shape, self.dtype));

        // For matrix/conv ops, we need a weight input
        let weight_name = "weight_0".to_string();
        if is_gemm_op(&nodes[0].dominant_op) || is_conv_op(&nodes[0].dominant_op) {
            let weight_shape = shape_from_node(&nodes[0], self.dtype);
            program.add_input(CoreMLFeature::new(&weight_name, weight_shape, self.dtype));
        }

        // Detect fusion patterns
        let fusions = detect_fusion_patterns(nodes);

        // Build set of indices consumed by fusions
        let mut fused_indices: std::collections::HashSet<usize> = std::collections::HashSet::new();
        for pattern in &fusions {
            match pattern {
                FusionPattern::ConvBnRelu { conv_idx, bn_idx, relu_idx } => {
                    fused_indices.insert(*conv_idx);
                    fused_indices.insert(*bn_idx);
                    fused_indices.insert(*relu_idx);
                }
                FusionPattern::GemmBiasAct { gemm_idx, bias_idx, act_idx } => {
                    fused_indices.insert(*gemm_idx);
                    fused_indices.insert(*bias_idx);
                    fused_indices.insert(*act_idx);
                }
                FusionPattern::MatMulGelu { mm_idx, gelu_idx } => {
                    fused_indices.insert(*mm_idx);
                    fused_indices.insert(*gelu_idx);
                }
            }
        }

        // Track the last SSA name produced
        let mut last_output = input_name;
        // Map node index -> SSA output name
        let mut node_outputs: HashMap<u32, String> = HashMap::new();

        // Process nodes in order, emitting fusions or individual ops
        let mut i = 0;
        while i < nodes.len() {
            if fused_indices.contains(&i) {
                // Find which fusion pattern starts at this index
                if let Some(pattern) = fusions.iter().find(|p| match p {
                    FusionPattern::ConvBnRelu { conv_idx, .. } => *conv_idx == i,
                    FusionPattern::GemmBiasAct { gemm_idx, .. } => *gemm_idx == i,
                    FusionPattern::MatMulGelu { mm_idx, .. } => *mm_idx == i,
                }) {
                    match pattern {
                        FusionPattern::ConvBnRelu { conv_idx, bn_idx, relu_idx } => {
                            // Emit Conv -> BN (as elementwise mul) -> ReLU
                            let wn = node_weight_name(&nodes[*conv_idx]);
                            let ws = shape_from_node(&nodes[*conv_idx], self.dtype);
                            program.add_input(CoreMLFeature::new(&wn, ws, self.dtype));

                            let conv_out = self.emit_conv2d(
                                &mut program,
                                &last_output,
                                &wn,
                                None,
                                [1, 1],
                                PadType::Same,
                                [1, 1],
                                1,
                            );

                            // BN is approximated as elementwise scale
                            let bn_scale_name = node_bias_name(&nodes[*bn_idx]);
                            let bn_shape = shape_from_node(&nodes[*bn_idx], self.dtype);
                            program.add_input(CoreMLFeature::new(
                                &bn_scale_name,
                                bn_shape,
                                self.dtype,
                            ));
                            let bn_out = self.emit_elementwise(
                                &mut program,
                                MilElementWiseOp::Mul,
                                &conv_out,
                                &bn_scale_name,
                            );

                            let relu_out = self.emit_activation(
                                &mut program,
                                MilActivationOp::ReLU,
                                &bn_out,
                            );

                            node_outputs.insert(nodes[*conv_idx].id.0, conv_out);
                            node_outputs.insert(nodes[*bn_idx].id.0, bn_out);
                            node_outputs.insert(nodes[*relu_idx].id.0, relu_out.clone());
                            last_output = relu_out;
                            i = relu_idx + 1;
                        }
                        FusionPattern::GemmBiasAct { gemm_idx, bias_idx, act_idx } => {
                            // Emit GEMM -> Bias -> Activation as fused pattern
                            let wn = node_weight_name(&nodes[*gemm_idx]);
                            let ws = shape_from_node(&nodes[*gemm_idx], self.dtype);
                            program.add_input(CoreMLFeature::new(&wn, ws, self.dtype));

                            let bn = node_bias_name(&nodes[*bias_idx]);
                            let bs = shape_from_node(&nodes[*bias_idx], self.dtype);
                            program.add_input(CoreMLFeature::new(&bn, bs, self.dtype));

                            let mm_out =
                                self.emit_matmul(&mut program, &last_output, &wn, false, false);
                            let add_out = self.emit_elementwise(
                                &mut program,
                                MilElementWiseOp::Add,
                                &mm_out,
                                &bn,
                            );
                            let act_op = activation_from_op(&nodes[*act_idx].dominant_op);
                            let act_out =
                                self.emit_activation(&mut program, act_op, &add_out);

                            node_outputs.insert(nodes[*gemm_idx].id.0, mm_out);
                            node_outputs.insert(nodes[*bias_idx].id.0, add_out);
                            node_outputs.insert(nodes[*act_idx].id.0, act_out.clone());
                            last_output = act_out;
                            i = act_idx + 1;
                        }
                        FusionPattern::MatMulGelu { mm_idx, gelu_idx } => {
                            let wn = node_weight_name(&nodes[*mm_idx]);
                            let ws = shape_from_node(&nodes[*mm_idx], self.dtype);
                            program.add_input(CoreMLFeature::new(&wn, ws, self.dtype));

                            let gelu_out =
                                self.emit_matmul_gelu(&mut program, &last_output, &wn);

                            node_outputs.insert(nodes[*mm_idx].id.0, gelu_out.clone());
                            node_outputs.insert(nodes[*gelu_idx].id.0, gelu_out.clone());
                            last_output = gelu_out;
                            i = gelu_idx + 1;
                        }
                    }
                } else {
                    // Consumed by a fusion that starts earlier; skip
                    i += 1;
                }
            } else {
                // Emit individual node
                let node = &nodes[i];
                let op = node.dominant_op.as_str();

                let out = if is_gemm_op(op) {
                    let wn = node_weight_name(node);
                    let ws = shape_from_node(node, self.dtype);
                    program.add_input(CoreMLFeature::new(&wn, ws, self.dtype));
                    self.emit_matmul(&mut program, &last_output, &wn, false, false)
                } else if is_conv_op(op) {
                    let wn = node_weight_name(node);
                    let ws = shape_from_node(node, self.dtype);
                    program.add_input(CoreMLFeature::new(&wn, ws, self.dtype));
                    self.emit_conv2d(
                        &mut program,
                        &last_output,
                        &wn,
                        None,
                        [1, 1],
                        PadType::Same,
                        [1, 1],
                        1,
                    )
                } else if is_activation_op(op) {
                    let act_op = activation_from_op(op);
                    self.emit_activation(&mut program, act_op, &last_output)
                } else if is_reduce_op(op) {
                    let reduce_op = reduce_from_op(op);
                    self.emit_reduce(&mut program, reduce_op, &last_output, &[-1], true)
                } else {
                    // Default: elementwise
                    let ew_op = elementwise_from_op(op);
                    // For elementwise ops, use last_output for both operands.
                    // In real usage, the node's consumed_values would identify
                    // the second operand.
                    self.emit_elementwise(&mut program, ew_op, &last_output, &last_output)
                };

                node_outputs.insert(node.id.0, out.clone());
                last_output = out;
                i += 1;
            }
        }

        // Set the last output as the program output
        let output_shape = shape_from_node(nodes.last().unwrap(), self.dtype);
        program.add_output(CoreMLFeature::new(&last_output, output_shape, self.dtype));

        Ok(program)
    }
}

// ---------------------------------------------------------------------------
// ANE compatibility validation
// ---------------------------------------------------------------------------

/// ANE-compatible MIL operation types.
///
/// Operations outside this set will be rejected by the ANE compiler and
/// fall back to CPU/GPU execution, defeating the purpose of ANE targeting.
const ANE_COMPATIBLE_OPS: &[&str] = &[
    "matmul",
    "conv",
    "add",
    "sub",
    "mul",
    "real_div",
    "relu",
    "leaky_relu",
    "sigmoid",
    "tanh",
    "gelu",
    "reduce_sum",
    "reduce_mean",
    "reduce_max",
    "reduce_min",
    "reshape",
    "transpose",
];

/// Validate that all operations in a MIL program are ANE-compatible.
///
/// Returns a list of warning messages for operations that are not known
/// to run efficiently on the Apple Neural Engine. An empty list means
/// the program is fully ANE-compatible.
///
/// Checks performed:
/// 1. All operation types are in the ANE-compatible set
/// 2. Data types are FP16 (primary ANE type) or FP32 (requires quantization)
/// 3. No unsupported reduction axes configurations
pub fn validate_ane_compatibility(program: &MilProgram) -> Vec<String> {
    let mut warnings = Vec::new();

    // Check operation types
    for (idx, op) in program.operations.iter().enumerate() {
        let op_type = op.op_type();
        if !ANE_COMPATIBLE_OPS.contains(&op_type) {
            warnings.push(format!(
                "op[{}] '{}' (type '{}') is not in ANE-compatible op set",
                idx,
                op.output_name(),
                op_type,
            ));
        }
    }

    // Check input data types
    for input in &program.inputs {
        if input.dtype == MilDataType::Int32 {
            warnings.push(format!(
                "input '{}' uses Int32 which has limited ANE support; prefer Float16",
                input.name,
            ));
        }
    }

    // Check output data types
    for output in &program.outputs {
        if output.dtype == MilDataType::Int32 {
            warnings.push(format!(
                "output '{}' uses Int32 which has limited ANE support; prefer Float16",
                output.name,
            ));
        }
    }

    // Check for excessive reduction dimensions (ANE prefers single-axis reductions)
    for (idx, op) in program.operations.iter().enumerate() {
        if let MilOperation::Reduce { axes, .. } = op {
            if axes.len() > 2 {
                warnings.push(format!(
                    "op[{}] '{}' reduces over {} axes; ANE may fall back to CPU for >2 axes",
                    idx,
                    op.output_name(),
                    axes.len(),
                ));
            }
        }
    }

    warnings
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mil_tensor_shape_matrix() {
        let shape = MilTensorShape::matrix(64, 128);
        assert_eq!(shape.numel(), 64 * 128);
        assert_eq!(shape.dims(), [1, 1, 64, 128]);
        assert_eq!(shape.rank(), 2);
    }

    #[test]
    fn test_mil_tensor_shape_4d() {
        let shape = MilTensorShape::tensor_4d(2, 3, 224, 224);
        assert_eq!(shape.numel(), 2 * 3 * 224 * 224);
        assert_eq!(shape.rank(), 4);
        assert_eq!(shape.to_string(), "(2, 3, 224, 224)");
    }

    #[test]
    fn test_mil_data_type_display() {
        assert_eq!(MilDataType::Float16.to_string(), "fp16");
        assert_eq!(MilDataType::Float32.to_string(), "fp32");
        assert_eq!(MilDataType::Int32.to_string(), "i32");
    }

    #[test]
    fn test_emitter_matmul() {
        let mut emitter = CoreMLEmitter::new();
        let mut program = MilProgram::new();
        program.add_input(CoreMLFeature::new(
            "A", MilTensorShape::matrix(64, 32), MilDataType::Float16,
        ));
        program.add_input(CoreMLFeature::new(
            "B", MilTensorShape::matrix(32, 64), MilDataType::Float16,
        ));

        let out = emitter.emit_matmul(&mut program, "A", "B", false, false);
        assert_eq!(out, "matmul_0");
        assert_eq!(program.op_count(), 1);
        assert_eq!(program.operations[0].op_type(), "matmul");
    }

    #[test]
    fn test_emitter_elementwise_chain() {
        let mut emitter = CoreMLEmitter::new();
        let mut program = MilProgram::new();
        program.add_input(CoreMLFeature::new(
            "x", MilTensorShape::vector(1024), MilDataType::Float16,
        ));
        program.add_input(CoreMLFeature::new(
            "y", MilTensorShape::vector(1024), MilDataType::Float16,
        ));

        let add = emitter.emit_elementwise(&mut program, MilElementWiseOp::Add, "x", "y");
        assert_eq!(add, "add_0");

        let relu = emitter.emit_activation(&mut program, MilActivationOp::ReLU, &add);
        assert_eq!(relu, "relu_1");
        assert_eq!(program.op_count(), 2);
    }

    #[test]
    fn test_gemm_bias_relu_fusion() {
        let mut emitter = CoreMLEmitter::new();
        let mut program = MilProgram::new();
        program.add_input(CoreMLFeature::new(
            "x", MilTensorShape::matrix(128, 64), MilDataType::Float16,
        ));
        program.add_input(CoreMLFeature::new(
            "w", MilTensorShape::matrix(64, 32), MilDataType::Float16,
        ));
        program.add_input(CoreMLFeature::new(
            "b", MilTensorShape::vector(32), MilDataType::Float16,
        ));

        let out = emitter.emit_gemm_bias_relu(&mut program, "x", "w", "b");

        // Should produce 3 operations: matmul, add, relu
        assert_eq!(program.op_count(), 3);
        assert_eq!(program.operations[0].op_type(), "matmul");
        assert_eq!(program.operations[1].op_type(), "add");
        assert_eq!(program.operations[2].op_type(), "relu");
        assert_eq!(out, "relu_2");
    }

    #[test]
    fn test_program_validate_ok() {
        let mut emitter = CoreMLEmitter::new();
        let mut program = MilProgram::new();
        program.add_input(CoreMLFeature::new(
            "x", MilTensorShape::vector(100), MilDataType::Float16,
        ));

        let relu_name = emitter.emit_activation(&mut program, MilActivationOp::ReLU, "x");
        program.add_output(CoreMLFeature::new(
            &relu_name, MilTensorShape::vector(100), MilDataType::Float16,
        ));

        assert!(program.validate().is_ok());
    }

    #[test]
    fn test_program_validate_undefined_ref() {
        let mut emitter = CoreMLEmitter::new();
        let mut program = MilProgram::new();
        // No inputs defined — "x" is undefined
        let _relu = emitter.emit_activation(&mut program, MilActivationOp::ReLU, "x");

        let result = program.validate();
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("undefined value 'x'"));
    }

    #[test]
    fn test_conv2d_emission() {
        let mut emitter = CoreMLEmitter::new();
        let mut program = MilProgram::new();
        program.add_input(CoreMLFeature::new(
            "input",
            MilTensorShape::tensor_4d(1, 3, 224, 224),
            MilDataType::Float16,
        ));
        program.add_input(CoreMLFeature::new(
            "weight",
            MilTensorShape::tensor_4d(64, 3, 7, 7),
            MilDataType::Float16,
        ));

        let out = emitter.emit_conv2d(
            &mut program,
            "input",
            "weight",
            None,
            [2, 2],
            PadType::Same,
            [1, 1],
            1,
        );

        assert_eq!(out, "conv_0");
        assert_eq!(program.op_count(), 1);
        if let MilOperation::Conv { pad_type, strides, .. } = &program.operations[0] {
            assert_eq!(*pad_type, PadType::Same);
            assert_eq!(*strides, [2, 2]);
        } else {
            panic!("expected Conv operation");
        }
    }

    // -----------------------------------------------------------------------
    // Helper to create a test ComputeNode with ANE targeting
    // -----------------------------------------------------------------------

    fn make_ane_node(id: u32, kind: NodeKind, dominant_op: &str, data_size: u64) -> ComputeNode {
        let mut costs = HashMap::new();
        costs.insert(
            ComputeTarget::NeuralEngine,
            llvm2_lower::compute_graph::ComputeCost {
                latency_cycles: 100,
                throughput_ops_per_kcycle: 5000,
            },
        );
        ComputeNode {
            id: ComputeNodeId(id),
            instructions: vec![],
            costs,
            legal_targets: vec![ComputeTarget::NeuralEngine, ComputeTarget::CpuScalar],
            kind,
            data_size_bytes: data_size,
            produced_values: vec![],
            consumed_values: vec![],
            dominant_op: dominant_op.to_string(),
            target_legality: None,
        }
    }

    fn make_cpu_only_node(id: u32) -> ComputeNode {
        let mut costs = HashMap::new();
        costs.insert(
            ComputeTarget::CpuScalar,
            llvm2_lower::compute_graph::ComputeCost {
                latency_cycles: 50,
                throughput_ops_per_kcycle: 1000,
            },
        );
        ComputeNode {
            id: ComputeNodeId(id),
            instructions: vec![],
            costs,
            legal_targets: vec![ComputeTarget::CpuScalar],
            kind: NodeKind::Scalar,
            data_size_bytes: 64,
            produced_values: vec![],
            consumed_values: vec![],
            dominant_op: "ADD".to_string(),
            target_legality: None,
        }
    }

    // -----------------------------------------------------------------------
    // emit_program_from_nodes tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_emit_program_from_nodes_empty() {
        let mut emitter = CoreMLEmitter::new();
        let result = emitter.emit_program_from_nodes(&[]);
        assert!(result.is_err());
        match result.unwrap_err() {
            CoreMLEmitError::EmptyNodeList => {}
            e => panic!("expected EmptyNodeList, got {:?}", e),
        }
    }

    #[test]
    fn test_emit_program_from_nodes_not_ane_compatible() {
        let mut emitter = CoreMLEmitter::new();
        let node = make_cpu_only_node(0);
        let result = emitter.emit_program_from_nodes(&[node]);
        assert!(result.is_err());
        match result.unwrap_err() {
            CoreMLEmitError::NotAneCompatible { node_id, .. } => {
                assert_eq!(node_id.0, 0);
            }
            e => panic!("expected NotAneCompatible, got {:?}", e),
        }
    }

    #[test]
    fn test_emit_single_gemm_node() {
        let mut emitter = CoreMLEmitter::new();
        let node = make_ane_node(0, NodeKind::MatrixHeavy, "GEMM", 8192);
        let program = emitter.emit_program_from_nodes(&[node]).unwrap();

        // Should have: matmul op
        assert_eq!(program.op_count(), 1);
        assert_eq!(program.operations[0].op_type(), "matmul");
        assert!(program.validate().is_ok());
    }

    #[test]
    fn test_emit_single_conv_node() {
        let mut emitter = CoreMLEmitter::new();
        let node = make_ane_node(0, NodeKind::DataParallel, "CONV", 4096);
        let program = emitter.emit_program_from_nodes(&[node]).unwrap();

        assert_eq!(program.op_count(), 1);
        assert_eq!(program.operations[0].op_type(), "conv");
        assert!(program.validate().is_ok());
    }

    #[test]
    fn test_emit_single_relu_node() {
        let mut emitter = CoreMLEmitter::new();
        let node = make_ane_node(0, NodeKind::DataParallel, "RELU", 2048);
        let program = emitter.emit_program_from_nodes(&[node]).unwrap();

        assert_eq!(program.op_count(), 1);
        assert_eq!(program.operations[0].op_type(), "relu");
        assert!(program.validate().is_ok());
    }

    #[test]
    fn test_emit_elementwise_add_node() {
        let mut emitter = CoreMLEmitter::new();
        let node = make_ane_node(0, NodeKind::DataParallel, "ADD", 1024);
        let program = emitter.emit_program_from_nodes(&[node]).unwrap();

        assert_eq!(program.op_count(), 1);
        assert_eq!(program.operations[0].op_type(), "add");
        assert!(program.validate().is_ok());
    }

    #[test]
    fn test_emit_reduce_mean_node() {
        let mut emitter = CoreMLEmitter::new();
        let node = make_ane_node(0, NodeKind::DataParallel, "REDUCE_MEAN", 2048);
        let program = emitter.emit_program_from_nodes(&[node]).unwrap();

        assert_eq!(program.op_count(), 1);
        assert_eq!(program.operations[0].op_type(), "reduce_mean");
        assert!(program.validate().is_ok());
    }

    #[test]
    fn test_emit_gemm_bias_act_fusion() {
        let mut emitter = CoreMLEmitter::new();
        let nodes = vec![
            make_ane_node(0, NodeKind::MatrixHeavy, "GEMM", 8192),
            make_ane_node(1, NodeKind::DataParallel, "ADD", 256),
            make_ane_node(2, NodeKind::DataParallel, "RELU", 256),
        ];
        let program = emitter.emit_program_from_nodes(&nodes).unwrap();

        // GEMM-Bias-Act fusion: matmul + add + relu = 3 ops
        assert_eq!(program.op_count(), 3);
        assert_eq!(program.operations[0].op_type(), "matmul");
        assert_eq!(program.operations[1].op_type(), "add");
        assert_eq!(program.operations[2].op_type(), "relu");
        assert!(program.validate().is_ok());
    }

    #[test]
    fn test_emit_conv_bn_relu_fusion() {
        let mut emitter = CoreMLEmitter::new();
        let nodes = vec![
            make_ane_node(0, NodeKind::DataParallel, "CONV", 4096),
            make_ane_node(1, NodeKind::DataParallel, "BN", 512),
            make_ane_node(2, NodeKind::DataParallel, "RELU", 512),
        ];
        let program = emitter.emit_program_from_nodes(&nodes).unwrap();

        // Conv-BN-ReLU fusion: conv + mul (BN scale) + relu = 3 ops
        assert_eq!(program.op_count(), 3);
        assert_eq!(program.operations[0].op_type(), "conv");
        assert_eq!(program.operations[1].op_type(), "mul");
        assert_eq!(program.operations[2].op_type(), "relu");
        assert!(program.validate().is_ok());
    }

    #[test]
    fn test_emit_matmul_gelu_fusion() {
        let mut emitter = CoreMLEmitter::new();
        let nodes = vec![
            make_ane_node(0, NodeKind::MatrixHeavy, "GEMM", 8192),
            make_ane_node(1, NodeKind::DataParallel, "GELU", 4096),
        ];
        let program = emitter.emit_program_from_nodes(&nodes).unwrap();

        // MatMul-GELU fusion: matmul + gelu = 2 ops
        assert_eq!(program.op_count(), 2);
        assert_eq!(program.operations[0].op_type(), "matmul");
        assert_eq!(program.operations[1].op_type(), "gelu");
        assert!(program.validate().is_ok());
    }

    #[test]
    fn test_emit_mixed_nodes_no_fusion() {
        let mut emitter = CoreMLEmitter::new();
        let nodes = vec![
            make_ane_node(0, NodeKind::MatrixHeavy, "GEMM", 8192),
            make_ane_node(1, NodeKind::DataParallel, "SIGMOID", 256),
        ];
        let program = emitter.emit_program_from_nodes(&nodes).unwrap();

        // No fusion pattern matches (GEMM + SIGMOID is not a recognized fusion)
        assert_eq!(program.op_count(), 2);
        assert_eq!(program.operations[0].op_type(), "matmul");
        assert_eq!(program.operations[1].op_type(), "sigmoid");
        assert!(program.validate().is_ok());
    }

    #[test]
    fn test_emit_multi_layer_network() {
        let mut emitter = CoreMLEmitter::new();
        // Simulate a 4-layer pattern: GEMM -> ADD -> RELU -> GEMM
        let nodes = vec![
            make_ane_node(0, NodeKind::MatrixHeavy, "GEMM", 8192),
            make_ane_node(1, NodeKind::DataParallel, "ADD", 256),
            make_ane_node(2, NodeKind::DataParallel, "RELU", 256),
            make_ane_node(3, NodeKind::MatrixHeavy, "GEMM", 4096),
        ];
        let program = emitter.emit_program_from_nodes(&nodes).unwrap();

        // GEMM-Bias-Act fusion on first 3 nodes (3 ops), then standalone GEMM (1 op)
        assert_eq!(program.op_count(), 4);
        assert_eq!(program.operations[0].op_type(), "matmul");
        assert_eq!(program.operations[1].op_type(), "add");
        assert_eq!(program.operations[2].op_type(), "relu");
        assert_eq!(program.operations[3].op_type(), "matmul");
        assert!(program.validate().is_ok());
    }

    // -----------------------------------------------------------------------
    // validate_ane_compatibility tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_validate_ane_all_compatible() {
        let mut emitter = CoreMLEmitter::new();
        let nodes = vec![
            make_ane_node(0, NodeKind::MatrixHeavy, "GEMM", 8192),
            make_ane_node(1, NodeKind::DataParallel, "RELU", 256),
        ];
        let program = emitter.emit_program_from_nodes(&nodes).unwrap();

        let warnings = validate_ane_compatibility(&program);
        assert!(warnings.is_empty(), "expected no warnings, got: {:?}", warnings);
    }

    #[test]
    fn test_validate_ane_int32_warning() {
        let mut program = MilProgram::new();
        program.add_input(CoreMLFeature::new(
            "x",
            MilTensorShape::vector(64),
            MilDataType::Int32,
        ));
        let mut emitter = CoreMLEmitter::with_dtype(MilDataType::Float16);
        let out = emitter.emit_activation(&mut program, MilActivationOp::ReLU, "x");
        program.add_output(CoreMLFeature::new(
            &out,
            MilTensorShape::vector(64),
            MilDataType::Float16,
        ));

        let warnings = validate_ane_compatibility(&program);
        assert_eq!(warnings.len(), 1);
        assert!(warnings[0].contains("Int32"));
    }

    #[test]
    fn test_validate_ane_multi_axis_reduce_warning() {
        let mut program = MilProgram::new();
        program.add_input(CoreMLFeature::new(
            "x",
            MilTensorShape::tensor_4d(1, 3, 8, 8),
            MilDataType::Float16,
        ));
        program.push_op(MilOperation::Reduce {
            output: "reduce_0".to_string(),
            op: MilReduceOp::Sum,
            x: "x".to_string(),
            axes: vec![1, 2, 3], // 3 axes -- triggers warning
            keep_dims: false,
        });
        program.add_output(CoreMLFeature::new(
            "reduce_0",
            MilTensorShape::vector(1),
            MilDataType::Float16,
        ));

        let warnings = validate_ane_compatibility(&program);
        assert_eq!(warnings.len(), 1);
        assert!(warnings[0].contains("3 axes"));
    }

    // -----------------------------------------------------------------------
    // Fusion pattern detection tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_detect_fusion_conv_bn_relu() {
        let nodes = vec![
            make_ane_node(0, NodeKind::DataParallel, "CONV", 4096),
            make_ane_node(1, NodeKind::DataParallel, "BN", 512),
            make_ane_node(2, NodeKind::DataParallel, "RELU", 512),
        ];
        let patterns = detect_fusion_patterns(&nodes);
        assert_eq!(patterns.len(), 1);
        assert_eq!(
            patterns[0],
            FusionPattern::ConvBnRelu { conv_idx: 0, bn_idx: 1, relu_idx: 2 }
        );
    }

    #[test]
    fn test_detect_fusion_none_for_unmatched() {
        let nodes = vec![
            make_ane_node(0, NodeKind::DataParallel, "ADD", 256),
            make_ane_node(1, NodeKind::DataParallel, "SUB", 256),
        ];
        let patterns = detect_fusion_patterns(&nodes);
        assert!(patterns.is_empty());
    }

    #[test]
    fn test_shape_from_node_matrix() {
        let node = make_ane_node(0, NodeKind::MatrixHeavy, "GEMM", 8192);
        let shape = shape_from_node(&node, MilDataType::Float16);
        // 8192 bytes / 2 bytes per FP16 = 4096 elements, sqrt(4096) = 64
        assert_eq!(shape.height, 64);
        assert_eq!(shape.width, 64);
    }

    #[test]
    fn test_shape_from_node_vector() {
        let node = make_ane_node(0, NodeKind::DataParallel, "ADD", 1024);
        let shape = shape_from_node(&node, MilDataType::Float16);
        // 1024 bytes / 2 = 512 elements as a vector
        assert_eq!(shape.width, 512);
        assert_eq!(shape.height, 1);
    }
}
