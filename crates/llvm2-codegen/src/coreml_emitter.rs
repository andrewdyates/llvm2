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

use std::fmt;

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
}
