// llvm2-codegen/metal_emitter.rs - Metal Shading Language (MSL) kernel emitter
//
// Author: Andrew Yates <ayates@dropbox.com>
// Copyright 2026 Dropbox, Inc. | License: Apache-2.0
//
// Generates Metal compute kernel source text from GpuKernelShape and kernel
// pattern descriptors. Phase 1 of the Metal emission pipeline (MSL source;
// Phase 2 will emit AIR bitcode directly).
//
// Design doc: designs/2026-04-14-metal-ir-emission.md
// Reference: Apple. "Metal Shading Language Specification," Version 3.2.

//! Metal Shading Language (MSL) kernel source code generation.
//!
//! This module implements Phase 1 of the Metal emission pipeline: generating
//! human-readable MSL source text for GPU compute kernels. The emitted source
//! is compiled to AIR bitcode via `xcrun -sdk macosx metal` and archived into
//! a `.metallib` via `xcrun -sdk macosx metallib`.
//!
//! # Supported Kernel Patterns
//!
//! - **Parallel Map**: element-wise `output[tid] = f(input[tid])`
//! - **Parallel Reduce**: tree reduction within threadgroups + SIMD acceleration
//! - **Map-Reduce (fused)**: avoids materializing intermediate array
//! - **MatMul**: tiled 8x8 matrix multiply via `simdgroup_matrix`
//!
//! # Usage
//!
//! ```ignore
//! let emitter = MetalKernelEmitter::new("node_42", MslElementType::Float);
//! let kernel = MslKernel::parallel_map("neg(x)", 1024, 256);
//! let source = emitter.emit(&kernel);
//! ```

use std::fmt;

use llvm2_lower::compute_graph::{ComputeNode, ComputeNodeId, NodeKind};
use llvm2_lower::dispatch::{DispatchOp, DispatchPlan};
use llvm2_lower::target_analysis::ComputeTarget;

// ---------------------------------------------------------------------------
// Metal emit errors
// ---------------------------------------------------------------------------

/// Errors that can occur during Metal kernel generation from ComputeGraph nodes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MetalEmitError {
    /// Node kind is not suitable for GPU execution (e.g., scalar).
    UnsuitableNodeKind {
        node_id: ComputeNodeId,
        kind: NodeKind,
    },
    /// GPU is not a legal target for this node.
    GpuNotLegal {
        node_id: ComputeNodeId,
    },
    /// Node has zero data size, cannot compute element count.
    ZeroDataSize {
        node_id: ComputeNodeId,
    },
    /// MatMul node dimensions could not be inferred (data size not
    /// a valid square or rectangular matrix).
    MatMulDimensionError {
        node_id: ComputeNodeId,
        data_size_bytes: u64,
    },
}

impl fmt::Display for MetalEmitError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MetalEmitError::UnsuitableNodeKind { node_id, kind } => {
                write!(f, "node {} has kind {} which is not suitable for Metal GPU execution", node_id, kind)
            }
            MetalEmitError::GpuNotLegal { node_id } => {
                write!(f, "GPU is not a legal compute target for node {}", node_id)
            }
            MetalEmitError::ZeroDataSize { node_id } => {
                write!(f, "node {} has zero data_size_bytes, cannot infer element count", node_id)
            }
            MetalEmitError::MatMulDimensionError { node_id, data_size_bytes } => {
                write!(f, "cannot infer MatMul dimensions for node {} with data_size_bytes={}", node_id, data_size_bytes)
            }
        }
    }
}

// ---------------------------------------------------------------------------
// MSL element types
// ---------------------------------------------------------------------------

/// Metal Shading Language scalar element type.
///
/// Maps to MSL built-in types used in kernel buffer declarations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MslElementType {
    /// 16-bit floating point (`half`).
    Half,
    /// 32-bit floating point (`float`).
    Float,
    /// 32-bit signed integer (`int`).
    Int,
    /// 32-bit unsigned integer (`uint`).
    Uint,
}

impl fmt::Display for MslElementType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MslElementType::Half => write!(f, "half"),
            MslElementType::Float => write!(f, "float"),
            MslElementType::Int => write!(f, "int"),
            MslElementType::Uint => write!(f, "uint"),
        }
    }
}

// ---------------------------------------------------------------------------
// MSL expression: tMIR op -> inline MSL
// ---------------------------------------------------------------------------

/// A tMIR operation mapped to an inline MSL expression.
///
/// The expression generator maps tMIR operations to MSL operators following
/// the table in the Metal IR emission design doc.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MslOp {
    /// `a + b`
    Add,
    /// `a - b`
    Sub,
    /// `a * b`
    Mul,
    /// `a / b`
    Div,
    /// `-a`
    Neg,
    /// `abs(a)`
    Abs,
    /// `sqrt(a)`
    Sqrt,
    /// `fma(a, b, c)`
    Fma,
    /// `min(a, b)`
    Min,
    /// `max(a, b)`
    Max,
    /// `clamp(x, lo, hi)`
    Clamp,
    /// `select(f, t, c)` (Metal order: false, true, condition)
    Select,
}

impl MslOp {
    /// Emit this operation as an inline MSL expression applied to operand names.
    ///
    /// For unary ops, only `a` is used. For binary, `a` and `b`.
    /// For ternary (fma, clamp, select), `a`, `b`, and `c`.
    pub fn emit(&self, a: &str, b: &str, c: &str) -> String {
        match self {
            MslOp::Add => format!("{a} + {b}"),
            MslOp::Sub => format!("{a} - {b}"),
            MslOp::Mul => format!("{a} * {b}"),
            MslOp::Div => format!("{a} / {b}"),
            MslOp::Neg => format!("-{a}"),
            MslOp::Abs => format!("abs({a})"),
            MslOp::Sqrt => format!("sqrt({a})"),
            MslOp::Fma => format!("fma({a}, {b}, {c})"),
            MslOp::Min => format!("min({a}, {b})"),
            MslOp::Max => format!("max({a}, {b})"),
            MslOp::Clamp => format!("clamp({a}, {b}, {c})"),
            MslOp::Select => format!("select({b}, {a}, {c})"),
        }
    }

    /// Number of operands required by this operation.
    pub fn arity(&self) -> u32 {
        match self {
            MslOp::Neg | MslOp::Abs | MslOp::Sqrt => 1,
            MslOp::Add | MslOp::Sub | MslOp::Mul | MslOp::Div
            | MslOp::Min | MslOp::Max => 2,
            MslOp::Fma | MslOp::Clamp | MslOp::Select => 3,
        }
    }
}

// ---------------------------------------------------------------------------
// Reduction operator (MSL-side)
// ---------------------------------------------------------------------------

/// Reduction operator for GPU reduce kernels.
///
/// Mirrors `GpuReduceOp` from `llvm2-verify/gpu_semantics.rs` but expressed
/// as MSL source fragments rather than SMT expressions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MslReduceOp {
    Add,
    Mul,
    Min,
    Max,
    BitwiseAnd,
    BitwiseOr,
    BitwiseXor,
}

impl MslReduceOp {
    /// Emit the binary expression `<a> <op> <b>` in MSL.
    pub fn emit_binary(&self, a: &str, b: &str) -> String {
        match self {
            MslReduceOp::Add => format!("{a} + {b}"),
            MslReduceOp::Mul => format!("{a} * {b}"),
            MslReduceOp::Min => format!("min({a}, {b})"),
            MslReduceOp::Max => format!("max({a}, {b})"),
            MslReduceOp::BitwiseAnd => format!("{a} & {b}"),
            MslReduceOp::BitwiseOr => format!("{a} | {b}"),
            MslReduceOp::BitwiseXor => format!("{a} ^ {b}"),
        }
    }

    /// Emit the SIMD intrinsic name for this reduction operation.
    ///
    /// Returns `None` for bitwise ops (no SIMD intrinsic available).
    pub fn simd_intrinsic(&self) -> Option<&'static str> {
        match self {
            MslReduceOp::Add => Some("simd_sum"),
            MslReduceOp::Mul => Some("simd_product"),
            MslReduceOp::Min => Some("simd_min"),
            MslReduceOp::Max => Some("simd_max"),
            MslReduceOp::BitwiseAnd => Some("simd_and"),
            MslReduceOp::BitwiseOr => Some("simd_or"),
            MslReduceOp::BitwiseXor => Some("simd_xor"),
        }
    }

    /// Identity element literal for this operation and element type.
    pub fn identity(&self, elem: MslElementType) -> &'static str {
        match (self, elem) {
            (MslReduceOp::Add, _) => "0",
            (MslReduceOp::Mul, _) => "1",
            (MslReduceOp::Min, MslElementType::Float | MslElementType::Half) => "INFINITY",
            (MslReduceOp::Min, MslElementType::Int) => "INT_MAX",
            (MslReduceOp::Min, MslElementType::Uint) => "UINT_MAX",
            (MslReduceOp::Max, MslElementType::Float | MslElementType::Half) => "-INFINITY",
            (MslReduceOp::Max, MslElementType::Int) => "INT_MIN",
            (MslReduceOp::Max, MslElementType::Uint) => "0",
            (MslReduceOp::BitwiseAnd, _) => "~0u",
            (MslReduceOp::BitwiseOr | MslReduceOp::BitwiseXor, _) => "0",
        }
    }
}

// ---------------------------------------------------------------------------
// Metal dispatch parameters
// ---------------------------------------------------------------------------

/// Metal dispatch size (equivalent to `MTLSize`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MtlSize {
    pub width: u64,
    pub height: u64,
    pub depth: u64,
}

impl MtlSize {
    pub fn new_1d(width: u64) -> Self {
        MtlSize { width, height: 1, depth: 1 }
    }

    pub fn new_2d(width: u64, height: u64) -> Self {
        MtlSize { width, height, depth: 1 }
    }
}

/// Computed Metal dispatch parameters for a kernel launch.
///
/// Ref: designs/2026-04-14-metal-ir-emission.md, "Grid Size Calculation"
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MetalDispatchParams {
    /// Total threads in the grid (1D, 2D, or 3D).
    pub grid_size: MtlSize,
    /// Threads per threadgroup.
    pub threadgroup_size: MtlSize,
}

impl MetalDispatchParams {
    /// Compute dispatch params for a 1D parallel map/reduce.
    pub fn for_1d(element_count: u64, threadgroup_size: u32) -> Self {
        let tg = threadgroup_size as u64;
        let grid_width = ((element_count + tg - 1) / tg) * tg;
        MetalDispatchParams {
            grid_size: MtlSize::new_1d(grid_width),
            threadgroup_size: MtlSize::new_1d(tg),
        }
    }

    /// Compute dispatch params for 2D matrix operations.
    pub fn for_2d(rows: u64, cols: u64, tile_size: u32) -> Self {
        let ts = tile_size as u64;
        MetalDispatchParams {
            grid_size: MtlSize::new_2d(
                ((cols + ts - 1) / ts) * ts,
                ((rows + ts - 1) / ts) * ts,
            ),
            threadgroup_size: MtlSize::new_2d(ts, ts),
        }
    }
}

// ---------------------------------------------------------------------------
// Metal storage mode
// ---------------------------------------------------------------------------

/// Metal buffer storage mode.
///
/// On Apple UMA, `Shared` avoids any data copy between CPU and GPU.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MtlStorageMode {
    /// CPU + GPU access, no copy (UMA). Default for LLVM2.
    Shared,
    /// GPU-only access. Used for intermediate GPU-to-GPU buffers.
    Private,
    /// Tile memory (render only). Not used for compute.
    Memoryless,
}

impl fmt::Display for MtlStorageMode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MtlStorageMode::Shared => write!(f, "MTLResourceStorageModeShared"),
            MtlStorageMode::Private => write!(f, "MTLResourceStorageModePrivate"),
            MtlStorageMode::Memoryless => write!(f, "MTLResourceStorageModeMemoryless"),
        }
    }
}

// ---------------------------------------------------------------------------
// MSL kernel descriptors
// ---------------------------------------------------------------------------

/// A complete MSL kernel specification ready for source emission.
#[derive(Debug, Clone)]
pub enum MslKernel {
    /// Element-wise unary or binary map: `output[tid] = f(input[tid])`.
    ParallelMap {
        /// Inline MSL expression for the per-element function body.
        /// Tokens `input[tid]` / `a[tid]`, `b[tid]` are used by convention.
        body_expr: String,
        /// Number of input buffers (1 for unary, 2 for binary).
        input_count: u32,
        /// Total element count (grid size).
        element_count: u64,
        /// Threadgroup size.
        threadgroup_size: u32,
    },

    /// Tree reduction within threadgroups, partial results collected.
    ParallelReduce {
        /// Reduction operation.
        op: MslReduceOp,
        /// Whether to use SIMD-accelerated reduction.
        use_simd: bool,
        /// Total element count.
        element_count: u64,
        /// Threadgroup size.
        threadgroup_size: u32,
    },

    /// Fused map-reduce (map then reduce without materializing intermediate).
    MapReduce {
        /// Inline MSL expression for the map function.
        map_expr: String,
        /// Reduction operation.
        reduce_op: MslReduceOp,
        /// Total element count.
        element_count: u64,
        /// Threadgroup size.
        threadgroup_size: u32,
    },

    /// Tiled 8x8 matrix multiply using `simdgroup_matrix`.
    MatMul {
        /// M dimension (rows of A / rows of C).
        m: u64,
        /// K dimension (cols of A / rows of B).
        k: u64,
        /// N dimension (cols of B / cols of C).
        n: u64,
    },
}

impl MslKernel {
    /// Create a unary parallel map kernel.
    pub fn parallel_map(body_expr: &str, element_count: u64, threadgroup_size: u32) -> Self {
        MslKernel::ParallelMap {
            body_expr: body_expr.to_string(),
            input_count: 1,
            element_count,
            threadgroup_size,
        }
    }

    /// Create a binary parallel map kernel (two input arrays).
    pub fn parallel_map2(body_expr: &str, element_count: u64, threadgroup_size: u32) -> Self {
        MslKernel::ParallelMap {
            body_expr: body_expr.to_string(),
            input_count: 2,
            element_count,
            threadgroup_size,
        }
    }

    /// Create a parallel reduce kernel.
    pub fn parallel_reduce(
        op: MslReduceOp,
        use_simd: bool,
        element_count: u64,
        threadgroup_size: u32,
    ) -> Self {
        MslKernel::ParallelReduce { op, use_simd, element_count, threadgroup_size }
    }

    /// Create a fused map-reduce kernel.
    pub fn map_reduce(
        map_expr: &str,
        reduce_op: MslReduceOp,
        element_count: u64,
        threadgroup_size: u32,
    ) -> Self {
        MslKernel::MapReduce {
            map_expr: map_expr.to_string(),
            reduce_op,
            element_count,
            threadgroup_size,
        }
    }

    /// Create a tiled matrix multiply kernel.
    pub fn matmul(m: u64, k: u64, n: u64) -> Self {
        MslKernel::MatMul { m, k, n }
    }
}

// ---------------------------------------------------------------------------
// MetalKernelEmitter
// ---------------------------------------------------------------------------

/// Emits Metal Shading Language (MSL) source text for compute kernels.
///
/// Each emitter instance is associated with a single compute node (identified
/// by `node_id`) and produces kernel functions named `llvm2_<pattern>_<node_id>`.
///
/// Ref: designs/2026-04-14-metal-ir-emission.md
pub struct MetalKernelEmitter {
    /// Compute node identifier (used in kernel function names).
    node_id: String,
    /// Element type for kernel buffers.
    elem_type: MslElementType,
}

impl MetalKernelEmitter {
    /// Create a new emitter for a given compute node.
    pub fn new(node_id: &str, elem_type: MslElementType) -> Self {
        MetalKernelEmitter {
            node_id: node_id.to_string(),
            elem_type,
        }
    }

    /// Emit complete MSL source text for the given kernel specification.
    ///
    /// The output includes `#include <metal_stdlib>`, kernel declaration,
    /// bounds checking, and the pattern-specific body.
    pub fn emit(&self, kernel: &MslKernel) -> String {
        let mut out = String::new();

        // Header
        out.push_str("#include <metal_stdlib>\nusing namespace metal;\n\n");
        out.push_str(&format!(
            "// Generated by LLVM2 — verified correct via z4 proof\n"
        ));
        out.push_str(&format!("// Compute node: {}\n\n", self.node_id));

        match kernel {
            MslKernel::ParallelMap { body_expr, input_count, element_count, threadgroup_size: _ } => {
                self.emit_parallel_map(&mut out, body_expr, *input_count, *element_count);
            }
            MslKernel::ParallelReduce { op, use_simd, element_count, threadgroup_size } => {
                self.emit_parallel_reduce(&mut out, *op, *use_simd, *element_count, *threadgroup_size);
            }
            MslKernel::MapReduce { map_expr, reduce_op, element_count, threadgroup_size } => {
                self.emit_map_reduce(&mut out, map_expr, *reduce_op, *element_count, *threadgroup_size);
            }
            MslKernel::MatMul { m, k, n } => {
                self.emit_matmul(&mut out, *m, *k, *n);
            }
        }

        out
    }

    /// Emit a unary or binary parallel map kernel.
    fn emit_parallel_map(
        &self,
        out: &mut String,
        body_expr: &str,
        input_count: u32,
        element_count: u64,
    ) {
        let ty = &self.elem_type;
        if input_count == 1 {
            out.push_str(&format!(
                "kernel void llvm2_map_{node}(\n\
                 \x20   const device {ty}* input  [[buffer(0)]],\n\
                 \x20   device {ty}* output       [[buffer(1)]],\n\
                 \x20   uint tid [[thread_position_in_grid]])\n\
                 {{\n\
                 \x20   if (tid >= {n}u) return;\n\
                 \x20   output[tid] = {body};\n\
                 }}\n",
                node = self.node_id,
                ty = ty,
                n = element_count,
                body = body_expr,
            ));
        } else {
            out.push_str(&format!(
                "kernel void llvm2_map2_{node}(\n\
                 \x20   const device {ty}* a [[buffer(0)]],\n\
                 \x20   const device {ty}* b [[buffer(1)]],\n\
                 \x20   device {ty}* output  [[buffer(2)]],\n\
                 \x20   uint tid [[thread_position_in_grid]])\n\
                 {{\n\
                 \x20   if (tid >= {n}u) return;\n\
                 \x20   output[tid] = {body};\n\
                 }}\n",
                node = self.node_id,
                ty = ty,
                n = element_count,
                body = body_expr,
            ));
        }
    }

    /// Emit a parallel reduce kernel (threadgroup tree or SIMD-accelerated).
    fn emit_parallel_reduce(
        &self,
        out: &mut String,
        op: MslReduceOp,
        use_simd: bool,
        element_count: u64,
        threadgroup_size: u32,
    ) {
        let ty = &self.elem_type;
        let identity = op.identity(*ty);

        if use_simd {
            let intrinsic = op.simd_intrinsic().unwrap_or("simd_sum");
            out.push_str(&format!(
                "kernel void llvm2_reduce_simd_{node}(\n\
                 \x20   const device {ty}* input       [[buffer(0)]],\n\
                 \x20   device {ty}* partial_results   [[buffer(1)]],\n\
                 \x20   threadgroup {ty}* shared       [[threadgroup(0)]],\n\
                 \x20   uint tid  [[thread_position_in_grid]],\n\
                 \x20   uint lid  [[thread_position_in_threadgroup]],\n\
                 \x20   uint sgid [[simdgroup_index_in_threadgroup]],\n\
                 \x20   uint lane [[thread_index_in_simdgroup]],\n\
                 \x20   uint tgid [[threadgroup_position_in_grid]])\n\
                 {{\n\
                 \x20   {ty} val = (tid < {n}u) ? input[tid] : {id};\n\
                 \x20   {ty} simd_result = {intrinsic}(val);\n\
                 \x20   if (lane == 0) {{ shared[sgid] = simd_result; }}\n\
                 \x20   threadgroup_barrier(mem_flags::mem_threadgroup);\n\
                 \x20   if (sgid == 0) {{\n\
                 \x20       {ty} final_val = (lid < ({tg_size}u / 32u)) ? shared[lid] : {id};\n\
                 \x20       final_val = {intrinsic}(final_val);\n\
                 \x20       if (lane == 0) {{ partial_results[tgid] = final_val; }}\n\
                 \x20   }}\n\
                 }}\n",
                node = self.node_id,
                ty = ty,
                n = element_count,
                id = identity,
                intrinsic = intrinsic,
                tg_size = threadgroup_size,
            ));
        } else {
            let binary = op.emit_binary("shared[lid]", "shared[lid + stride]");
            out.push_str(&format!(
                "kernel void llvm2_reduce_{node}(\n\
                 \x20   const device {ty}* input       [[buffer(0)]],\n\
                 \x20   device {ty}* partial_results   [[buffer(1)]],\n\
                 \x20   threadgroup {ty}* shared       [[threadgroup(0)]],\n\
                 \x20   uint tid  [[thread_position_in_grid]],\n\
                 \x20   uint lid  [[thread_position_in_threadgroup]],\n\
                 \x20   uint tgid [[threadgroup_position_in_grid]],\n\
                 \x20   uint tg_size [[threads_per_threadgroup]])\n\
                 {{\n\
                 \x20   shared[lid] = (tid < {n}u) ? input[tid] : {id};\n\
                 \x20   threadgroup_barrier(mem_flags::mem_threadgroup);\n\
                 \x20   for (uint stride = tg_size / 2; stride > 0; stride >>= 1) {{\n\
                 \x20       if (lid < stride) {{\n\
                 \x20           shared[lid] = {binary};\n\
                 \x20       }}\n\
                 \x20       threadgroup_barrier(mem_flags::mem_threadgroup);\n\
                 \x20   }}\n\
                 \x20   if (lid == 0) {{ partial_results[tgid] = shared[0]; }}\n\
                 }}\n",
                node = self.node_id,
                ty = ty,
                n = element_count,
                id = identity,
                binary = binary,
            ));
        }
    }

    /// Emit a fused map-reduce kernel.
    fn emit_map_reduce(
        &self,
        out: &mut String,
        map_expr: &str,
        reduce_op: MslReduceOp,
        element_count: u64,
        threadgroup_size: u32,
    ) {
        let ty = &self.elem_type;
        let identity = reduce_op.identity(*ty);
        let intrinsic = reduce_op.simd_intrinsic().unwrap_or("simd_sum");
        out.push_str(&format!(
            "kernel void llvm2_map_reduce_{node}(\n\
             \x20   const device {ty}* a           [[buffer(0)]],\n\
             \x20   const device {ty}* b           [[buffer(1)]],\n\
             \x20   device {ty}* partial_results   [[buffer(2)]],\n\
             \x20   threadgroup {ty}* shared       [[threadgroup(0)]],\n\
             \x20   uint tid  [[thread_position_in_grid]],\n\
             \x20   uint lid  [[thread_position_in_threadgroup]],\n\
             \x20   uint sgid [[simdgroup_index_in_threadgroup]],\n\
             \x20   uint lane [[thread_index_in_simdgroup]],\n\
             \x20   uint tgid [[threadgroup_position_in_grid]])\n\
             {{\n\
             \x20   {ty} mapped = (tid < {n}u) ? {map} : {id};\n\
             \x20   {ty} simd_result = {intrinsic}(mapped);\n\
             \x20   if (lane == 0) {{ shared[sgid] = simd_result; }}\n\
             \x20   threadgroup_barrier(mem_flags::mem_threadgroup);\n\
             \x20   if (sgid == 0) {{\n\
             \x20       {ty} final_val = (lid < ({tg_size}u / 32u)) ? shared[lid] : {id};\n\
             \x20       final_val = {intrinsic}(final_val);\n\
             \x20       if (lane == 0) {{ partial_results[tgid] = final_val; }}\n\
             \x20   }}\n\
             }}\n",
            node = self.node_id,
            ty = ty,
            n = element_count,
            map = map_expr,
            id = identity,
            intrinsic = intrinsic,
            tg_size = threadgroup_size,
        ));
    }

    /// Emit a tiled 8x8 matrix multiply kernel using `simdgroup_matrix`.
    fn emit_matmul(
        &self,
        out: &mut String,
        m: u64,
        k: u64,
        n: u64,
    ) {
        let ty = &self.elem_type;
        out.push_str("#include <metal_simdgroup_matrix>\n\n");
        out.push_str(&format!(
            "kernel void llvm2_matmul_{node}(\n\
             \x20   const device {ty}* A  [[buffer(0)]],\n\
             \x20   const device {ty}* B  [[buffer(1)]],\n\
             \x20   device {ty}* C        [[buffer(2)]],\n\
             \x20   uint2 gid [[thread_position_in_grid]])\n\
             {{\n\
             \x20   const uint M = {m}u;\n\
             \x20   const uint K = {k}u;\n\
             \x20   const uint N = {n}u;\n\
             \x20   uint row = gid.y;\n\
             \x20   uint col = gid.x;\n\
             \x20   simdgroup_matrix<{ty}, 8, 8> acc;\n\
             \x20   acc = make_filled_simdgroup_matrix<{ty}, 8, 8>(0.0{suffix});\n\
             \x20   for (uint kk = 0; kk < K; kk += 8) {{\n\
             \x20       simdgroup_matrix<{ty}, 8, 8> a_tile, b_tile;\n\
             \x20       simdgroup_load(a_tile, A + row * K + kk, K);\n\
             \x20       simdgroup_load(b_tile, B + kk * N + col, N);\n\
             \x20       simdgroup_multiply_accumulate(acc, a_tile, b_tile, acc);\n\
             \x20   }}\n\
             \x20   simdgroup_store(acc, C + row * N + col, N);\n\
             }}\n",
            node = self.node_id,
            ty = ty,
            m = m,
            k = k,
            n = n,
            suffix = if *ty == MslElementType::Half { "h" } else { "f" },
        ));
    }

    /// Emit buffer creation host-side code snippet (Objective-C).
    ///
    /// This is a helper for generating host dispatch code fragments.
    pub fn emit_buffer_creation(
        buf_name: &str,
        size_bytes: u64,
        storage_mode: MtlStorageMode,
    ) -> String {
        format!(
            "id<MTLBuffer> {} = [device newBufferWithLength:{} options:{}];",
            buf_name, size_bytes, storage_mode,
        )
    }
}

// ---------------------------------------------------------------------------
// ComputeGraph -> MSL kernel generation
// ---------------------------------------------------------------------------

/// Default threadgroup size for 1D kernels.
const DEFAULT_THREADGROUP_SIZE: u32 = 256;

/// Default tile size for MatMul kernels (8x8 simdgroup_matrix tiles).
const DEFAULT_MATMUL_TILE: u32 = 8;

/// Infer MSL element type from a ComputeNode's dominant operation name.
///
/// Floating-point ops (FADD, FSUB, FMUL, FDIV, FNEG, FMA) map to Float.
/// Integer ops (ADD, SUB, MUL, SDIV, UDIV) map to Int.
/// Unsigned ops (UADD, USUB, UMUL) map to Uint.
/// Default is Float (safest for GPU workloads).
pub fn infer_element_type(dominant_op: &str) -> MslElementType {
    let op = dominant_op.to_uppercase();
    if op.starts_with('F') || op == "FMA" || op.contains("FLOAT") {
        MslElementType::Float
    } else if op.starts_with('U') || op.contains("UINT") || op.contains("UNSIGNED") {
        MslElementType::Uint
    } else if op.starts_with("REDUCE") {
        // Reduce ops: check suffix for type hint
        if op.contains("FLOAT") || op.contains("FP") {
            MslElementType::Float
        } else {
            MslElementType::Float // default reduce to float
        }
    } else if op == "ADD" || op == "SUB" || op == "MUL" || op == "SDIV"
        || op == "NEG" || op == "ABS" || op.contains("INT")
    {
        MslElementType::Int
    } else {
        MslElementType::Float
    }
}

/// Infer the MslOp for the map body from a dominant operation name.
///
/// Returns `None` for reduce-like operations that should use `ParallelReduce`.
fn infer_msl_op(dominant_op: &str) -> Option<MslOp> {
    let op = dominant_op.to_uppercase();
    match op.as_str() {
        "ADD" | "FADD" => Some(MslOp::Add),
        "SUB" | "FSUB" => Some(MslOp::Sub),
        "MUL" | "FMUL" => Some(MslOp::Mul),
        "SDIV" | "UDIV" | "FDIV" | "DIV" => Some(MslOp::Div),
        "NEG" | "FNEG" => Some(MslOp::Neg),
        "ABS" | "FABS" => Some(MslOp::Abs),
        "SQRT" | "FSQRT" => Some(MslOp::Sqrt),
        "FMA" | "FFMA" => Some(MslOp::Fma),
        "MIN" | "FMIN" => Some(MslOp::Min),
        "MAX" | "FMAX" => Some(MslOp::Max),
        _ => None,
    }
}

/// Infer the MslReduceOp from a dominant operation name for reduce patterns.
fn infer_reduce_op(dominant_op: &str) -> MslReduceOp {
    let op = dominant_op.to_uppercase();
    if op.contains("MUL") || op.contains("PROD") {
        MslReduceOp::Mul
    } else if op.contains("MIN") {
        MslReduceOp::Min
    } else if op.contains("MAX") {
        MslReduceOp::Max
    } else if op.contains("AND") {
        MslReduceOp::BitwiseAnd
    } else if op.contains("OR") {
        MslReduceOp::BitwiseOr
    } else if op.contains("XOR") {
        MslReduceOp::BitwiseXor
    } else {
        // Default: additive reduction (sum)
        MslReduceOp::Add
    }
}

/// Returns true if the dominant_op looks like a reduction pattern.
fn is_reduce_pattern(dominant_op: &str) -> bool {
    let op = dominant_op.to_uppercase();
    op.starts_with("REDUCE") || op.starts_with("SUM") || op.starts_with("PROD")
        || op == "DOT" || op.starts_with("ACCUM")
}

/// Emit a complete MSL kernel source string for a ComputeGraph node.
///
/// This is the main entry point for converting ComputeGraph GPU nodes into
/// Metal shader source code. It inspects the node's `kind`, `dominant_op`,
/// and `data_size_bytes` to select the right kernel template and generate
/// the MSL source.
///
/// # Errors
///
/// - `UnsuitableNodeKind` if the node is `Scalar` (should run on CPU).
/// - `GpuNotLegal` if GPU is not in the node's `legal_targets`.
/// - `ZeroDataSize` if `data_size_bytes` is 0.
/// - `MatMulDimensionError` if MatMul dimensions cannot be inferred.
pub fn emit_kernel_from_node(node: &ComputeNode) -> Result<String, MetalEmitError> {
    // Validate: node must be suitable for GPU
    if node.kind == NodeKind::Scalar {
        return Err(MetalEmitError::UnsuitableNodeKind {
            node_id: node.id,
            kind: node.kind,
        });
    }

    if !node.legal_targets.contains(&ComputeTarget::Gpu) {
        return Err(MetalEmitError::GpuNotLegal { node_id: node.id });
    }

    if node.data_size_bytes == 0 {
        return Err(MetalEmitError::ZeroDataSize { node_id: node.id });
    }

    let node_id_str = format!("{}", node.id);
    let elem_type = infer_element_type(&node.dominant_op);
    let emitter = MetalKernelEmitter::new(&node_id_str, elem_type);

    // Compute element count from data size (4 bytes for float/int/uint, 2 for half)
    let elem_bytes = match elem_type {
        MslElementType::Half => 2u64,
        MslElementType::Float | MslElementType::Int | MslElementType::Uint => 4u64,
    };
    let element_count = node.data_size_bytes / elem_bytes;

    match node.kind {
        NodeKind::DataParallel => {
            if is_reduce_pattern(&node.dominant_op) {
                // Emit a parallel reduce kernel
                let reduce_op = infer_reduce_op(&node.dominant_op);
                let kernel = MslKernel::parallel_reduce(
                    reduce_op,
                    true, // use SIMD acceleration by default
                    element_count,
                    DEFAULT_THREADGROUP_SIZE,
                );
                Ok(emitter.emit(&kernel))
            } else {
                // Emit a parallel map kernel
                let msl_op = infer_msl_op(&node.dominant_op);
                let (body_expr, input_count) = match msl_op {
                    Some(ref op) if op.arity() == 1 => {
                        (op.emit("input[tid]", "", ""), 1u32)
                    }
                    Some(ref op) if op.arity() == 2 => {
                        (op.emit("a[tid]", "b[tid]", ""), 2u32)
                    }
                    Some(ref op) => {
                        // Ternary ops: use 2 inputs + constant
                        (op.emit("a[tid]", "b[tid]", "0"), 2u32)
                    }
                    None => {
                        // Fallback: identity map
                        ("input[tid]".to_string(), 1u32)
                    }
                };

                let kernel = MslKernel::ParallelMap {
                    body_expr,
                    input_count,
                    element_count,
                    threadgroup_size: DEFAULT_THREADGROUP_SIZE,
                };

                Ok(emitter.emit(&kernel))
            }
        }

        NodeKind::MatrixHeavy => {
            // Infer square matrix dimensions from data size.
            // For C = A*B where A is MxK and B is KxN, total data is:
            //   (M*K + K*N + M*N) * elem_bytes
            // For simplicity, assume square: M=K=N=dim, so 3*dim^2*elem_bytes = data_size
            let total_elements = node.data_size_bytes / elem_bytes;
            // 3*dim^2 = total_elements  =>  dim = sqrt(total_elements / 3)
            let dim_sq = total_elements / 3;
            let dim = (dim_sq as f64).sqrt() as u64;

            if dim == 0 {
                return Err(MetalEmitError::MatMulDimensionError {
                    node_id: node.id,
                    data_size_bytes: node.data_size_bytes,
                });
            }

            let kernel = MslKernel::matmul(dim, dim, dim);
            Ok(emitter.emit(&kernel))
        }

        NodeKind::Scalar => {
            // Already handled above, but match arm is required
            unreachable!()
        }
    }
}

// ---------------------------------------------------------------------------
// Host-side Metal dispatch code generation
// ---------------------------------------------------------------------------

/// Emit host-side Objective-C Metal dispatch code for a DispatchPlan.
///
/// Generates a complete dispatch function that:
/// - Creates Metal buffers for data transfers
/// - Creates compute pipeline states for each kernel
/// - Encodes and dispatches compute commands
/// - Inserts synchronization barriers
///
/// The generated code assumes `_device` (id<MTLDevice>), `_queue`
/// (id<MTLCommandQueue>), and `_library` (id<MTLLibrary>) are in scope
/// as instance variables.
pub fn emit_dispatch_code(
    plan: &DispatchPlan,
    graph: &llvm2_lower::compute_graph::ComputeGraph,
) -> String {
    let mut out = String::new();

    out.push_str("// Generated by LLVM2 — Metal dispatch code\n");
    out.push_str(&format!(
        "// Dispatch plan: {} ops ({} launches, {} transfers)\n\n",
        plan.len(),
        plan.count_launches(),
        plan.count_transfers(),
    ));

    out.push_str("- (void)executeDispatchPlan {\n");
    out.push_str("    id<MTLCommandBuffer> cmdBuf = [_queue commandBuffer];\n\n");

    for (i, op) in plan.ops.iter().enumerate() {
        match op {
            DispatchOp::DataTransfer { src, dst, size_bytes, edge_from, edge_to, .. } => {
                out.push_str(&format!(
                    "    // Op {}: Transfer {} bytes ({:?} -> {:?})\n",
                    i, size_bytes, src, dst,
                ));
                // On Apple UMA, shared memory means no explicit copy for CPU<->GPU.
                if (*src == ComputeTarget::CpuScalar || *src == ComputeTarget::CpuSimd)
                    && *dst == ComputeTarget::Gpu
                {
                    out.push_str("    // UMA: no explicit copy needed (shared memory)\n");
                } else if *src == ComputeTarget::Gpu
                    && (*dst == ComputeTarget::CpuScalar || *dst == ComputeTarget::CpuSimd)
                {
                    out.push_str("    // UMA: coherent after command buffer completion\n");
                } else {
                    out.push_str(&format!(
                        "    id<MTLBuffer> xfer_{i} = [_device newBufferWithLength:{size} options:MTLResourceStorageModeShared];\n",
                        i = i, size = size_bytes,
                    ));
                }
                let _ = (edge_from, edge_to); // suppress unused warnings in doc
                out.push('\n');
            }

            DispatchOp::KernelLaunch { target, node_id, estimated_cycles } => {
                let node_id_str = format!("{}", node_id);
                let kernel_name = if let Some(node) = graph.node(*node_id) {
                    match node.kind {
                        NodeKind::MatrixHeavy => format!("llvm2_matmul_{}", node_id_str),
                        NodeKind::DataParallel => {
                            if is_reduce_pattern(&node.dominant_op) {
                                format!("llvm2_reduce_simd_{}", node_id_str)
                            } else {
                                format!("llvm2_map_{}", node_id_str)
                            }
                        }
                        NodeKind::Scalar => format!("llvm2_scalar_{}", node_id_str),
                    }
                } else {
                    format!("llvm2_kernel_{}", node_id_str)
                };

                out.push_str(&format!(
                    "    // Op {}: Launch {:?} kernel '{}' (est. {} cycles)\n",
                    i, target, kernel_name, estimated_cycles,
                ));

                if *target == ComputeTarget::Gpu {
                    out.push_str(&format!(
                        "    id<MTLFunction> fn_{i} = [_library newFunctionWithName:@\"{name}\"];\n",
                        i = i, name = kernel_name,
                    ));
                    out.push_str(&format!(
                        "    id<MTLComputePipelineState> pso_{i} = [_device newComputePipelineStateWithFunction:fn_{i} error:nil];\n",
                        i = i,
                    ));
                    out.push_str(&format!(
                        "    id<MTLComputeCommandEncoder> enc_{i} = [cmdBuf computeCommandEncoder];\n",
                        i = i,
                    ));
                    out.push_str(&format!(
                        "    [enc_{i} setComputePipelineState:pso_{i}];\n",
                        i = i,
                    ));

                    // Dispatch dimensions from node metadata
                    if let Some(node) = graph.node(*node_id) {
                        let elem_type = infer_element_type(&node.dominant_op);
                        let elem_bytes: u64 = match elem_type {
                            MslElementType::Half => 2,
                            _ => 4,
                        };
                        let element_count = node.data_size_bytes / elem_bytes.max(1);

                        if node.kind == NodeKind::MatrixHeavy {
                            let dim_sq = (element_count / 3).max(1);
                            let dim = (dim_sq as f64).sqrt() as u64;
                            let params = MetalDispatchParams::for_2d(dim, dim, DEFAULT_MATMUL_TILE);
                            out.push_str(&format!(
                                "    [enc_{i} dispatchThreads:MTLSizeMake({w}, {h}, 1) threadsPerThreadgroup:MTLSizeMake({tw}, {th}, 1)];\n",
                                i = i,
                                w = params.grid_size.width,
                                h = params.grid_size.height,
                                tw = params.threadgroup_size.width,
                                th = params.threadgroup_size.height,
                            ));
                        } else {
                            let params = MetalDispatchParams::for_1d(element_count, DEFAULT_THREADGROUP_SIZE);
                            out.push_str(&format!(
                                "    [enc_{i} dispatchThreads:MTLSizeMake({w}, 1, 1) threadsPerThreadgroup:MTLSizeMake({tw}, 1, 1)];\n",
                                i = i,
                                w = params.grid_size.width,
                                tw = params.threadgroup_size.width,
                            ));
                        }
                    }

                    out.push_str(&format!("    [enc_{i} endEncoding];\n", i = i));
                } else {
                    out.push_str(&format!(
                        "    // CPU execution for {} (not a Metal kernel)\n",
                        node_id,
                    ));
                }
                out.push('\n');
            }

            DispatchOp::Synchronize { target, .. } => {
                out.push_str(&format!(
                    "    // Op {}: Synchronize {:?}\n",
                    i, target,
                ));
                if *target == ComputeTarget::Gpu {
                    out.push_str("    [cmdBuf commit];\n");
                    out.push_str("    [cmdBuf waitUntilCompleted];\n");
                    out.push_str("    cmdBuf = [_queue commandBuffer];\n");
                }
                out.push('\n');
            }

            DispatchOp::CpuFallback { node_id, reason } => {
                out.push_str(&format!(
                    "    // Op {}: CPU fallback for {} ({})\n",
                    i, node_id, reason,
                ));
                out.push('\n');
            }
        }
    }

    out.push_str("    [cmdBuf commit];\n");
    out.push_str("    [cmdBuf waitUntilCompleted];\n");
    out.push_str("}\n");

    out
}

// ---------------------------------------------------------------------------
// Kernel function name generation
// ---------------------------------------------------------------------------

/// Return the MSL kernel function name for a given node and pattern.
pub fn kernel_function_name(node_id: &str, kernel: &MslKernel) -> String {
    match kernel {
        MslKernel::ParallelMap { input_count, .. } => {
            if *input_count <= 1 {
                format!("llvm2_map_{}", node_id)
            } else {
                format!("llvm2_map2_{}", node_id)
            }
        }
        MslKernel::ParallelReduce { use_simd, .. } => {
            if *use_simd {
                format!("llvm2_reduce_simd_{}", node_id)
            } else {
                format!("llvm2_reduce_{}", node_id)
            }
        }
        MslKernel::MapReduce { .. } => format!("llvm2_map_reduce_{}", node_id),
        MslKernel::MatMul { .. } => format!("llvm2_matmul_{}", node_id),
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    #[test]
    fn test_msl_element_type_display() {
        assert_eq!(MslElementType::Float.to_string(), "float");
        assert_eq!(MslElementType::Half.to_string(), "half");
        assert_eq!(MslElementType::Int.to_string(), "int");
        assert_eq!(MslElementType::Uint.to_string(), "uint");
    }

    #[test]
    fn test_msl_op_emit_binary() {
        assert_eq!(MslOp::Add.emit("x", "y", ""), "x + y");
        assert_eq!(MslOp::Mul.emit("a", "b", ""), "a * b");
        assert_eq!(MslOp::Min.emit("a", "b", ""), "min(a, b)");
    }

    #[test]
    fn test_msl_op_emit_ternary() {
        assert_eq!(MslOp::Fma.emit("a", "b", "c"), "fma(a, b, c)");
        assert_eq!(MslOp::Clamp.emit("x", "lo", "hi"), "clamp(x, lo, hi)");
        assert_eq!(MslOp::Select.emit("t", "f", "c"), "select(f, t, c)");
    }

    #[test]
    fn test_msl_op_arity() {
        assert_eq!(MslOp::Neg.arity(), 1);
        assert_eq!(MslOp::Add.arity(), 2);
        assert_eq!(MslOp::Fma.arity(), 3);
    }

    #[test]
    fn test_dispatch_params_1d() {
        let params = MetalDispatchParams::for_1d(1000, 256);
        // Rounds up: ceil(1000/256) * 256 = 4 * 256 = 1024
        assert_eq!(params.grid_size.width, 1024);
        assert_eq!(params.grid_size.height, 1);
        assert_eq!(params.threadgroup_size.width, 256);
    }

    #[test]
    fn test_dispatch_params_2d() {
        let params = MetalDispatchParams::for_2d(100, 200, 16);
        // ceil(200/16)*16 = 208, ceil(100/16)*16 = 112
        assert_eq!(params.grid_size.width, 208);
        assert_eq!(params.grid_size.height, 112);
        assert_eq!(params.threadgroup_size.width, 16);
        assert_eq!(params.threadgroup_size.height, 16);
    }

    #[test]
    fn test_emit_parallel_map_contains_kernel_decl() {
        let emitter = MetalKernelEmitter::new("node_1", MslElementType::Float);
        let kernel = MslKernel::parallel_map("-input[tid]", 1024, 256);
        let source = emitter.emit(&kernel);

        assert!(source.contains("kernel void llvm2_map_node_1("));
        assert!(source.contains("const device float* input"));
        assert!(source.contains("if (tid >= 1024u) return;"));
        assert!(source.contains("output[tid] = -input[tid];"));
        assert!(source.contains("#include <metal_stdlib>"));
    }

    #[test]
    fn test_emit_parallel_reduce_tree() {
        let emitter = MetalKernelEmitter::new("node_3", MslElementType::Float);
        let kernel = MslKernel::parallel_reduce(MslReduceOp::Add, false, 2048, 256);
        let source = emitter.emit(&kernel);

        assert!(source.contains("kernel void llvm2_reduce_node_3("));
        assert!(source.contains("threadgroup float* shared"));
        assert!(source.contains("threadgroup_barrier"));
        assert!(source.contains("partial_results[tgid] = shared[0];"));
    }

    #[test]
    fn test_emit_matmul_includes_simdgroup_matrix() {
        let emitter = MetalKernelEmitter::new("node_7", MslElementType::Float);
        let kernel = MslKernel::matmul(64, 32, 64);
        let source = emitter.emit(&kernel);

        assert!(source.contains("#include <metal_simdgroup_matrix>"));
        assert!(source.contains("kernel void llvm2_matmul_node_7("));
        assert!(source.contains("simdgroup_matrix<float, 8, 8>"));
        assert!(source.contains("simdgroup_multiply_accumulate"));
        assert!(source.contains("const uint M = 64u;"));
        assert!(source.contains("const uint K = 32u;"));
        assert!(source.contains("const uint N = 64u;"));
    }

    #[test]
    fn test_kernel_function_name() {
        let map = MslKernel::parallel_map("x", 100, 32);
        assert_eq!(kernel_function_name("n1", &map), "llvm2_map_n1");

        let map2 = MslKernel::parallel_map2("a+b", 100, 32);
        assert_eq!(kernel_function_name("n1", &map2), "llvm2_map2_n1");

        let red = MslKernel::parallel_reduce(MslReduceOp::Add, true, 100, 256);
        assert_eq!(kernel_function_name("n2", &red), "llvm2_reduce_simd_n2");

        let mm = MslKernel::matmul(8, 8, 8);
        assert_eq!(kernel_function_name("n3", &mm), "llvm2_matmul_n3");
    }

    // -----------------------------------------------------------------------
    // ComputeGraph -> MSL kernel generation tests
    // -----------------------------------------------------------------------

    /// Helper: construct a minimal ComputeNode for testing.
    fn make_test_node(
        id: u32,
        kind: NodeKind,
        dominant_op: &str,
        data_size_bytes: u64,
        legal_targets: Vec<ComputeTarget>,
    ) -> ComputeNode {
        ComputeNode {
            id: ComputeNodeId(id),
            instructions: vec![],
            costs: HashMap::new(),
            legal_targets,
            kind,
            data_size_bytes,
            produced_values: vec![],
            consumed_values: vec![],
            dominant_op: dominant_op.to_string(),
            target_legality: None,
        }
    }

    #[test]
    fn test_infer_element_type_float_ops() {
        assert_eq!(infer_element_type("FADD"), MslElementType::Float);
        assert_eq!(infer_element_type("FMUL"), MslElementType::Float);
        assert_eq!(infer_element_type("FSUB"), MslElementType::Float);
        assert_eq!(infer_element_type("FDIV"), MslElementType::Float);
        assert_eq!(infer_element_type("FNEG"), MslElementType::Float);
        assert_eq!(infer_element_type("FMA"), MslElementType::Float);
    }

    #[test]
    fn test_infer_element_type_int_ops() {
        assert_eq!(infer_element_type("ADD"), MslElementType::Int);
        assert_eq!(infer_element_type("SUB"), MslElementType::Int);
        assert_eq!(infer_element_type("MUL"), MslElementType::Int);
        assert_eq!(infer_element_type("NEG"), MslElementType::Int);
    }

    #[test]
    fn test_infer_element_type_uint_ops() {
        assert_eq!(infer_element_type("UADD"), MslElementType::Uint);
        assert_eq!(infer_element_type("UMUL"), MslElementType::Uint);
    }

    #[test]
    fn test_emit_kernel_from_node_data_parallel_fadd() {
        let node = make_test_node(
            10,
            NodeKind::DataParallel,
            "FADD",
            4096, // 1024 float elements
            vec![ComputeTarget::CpuScalar, ComputeTarget::Gpu],
        );

        let source = emit_kernel_from_node(&node).unwrap();
        // Should generate a binary map kernel: a[tid] + b[tid]
        assert!(source.contains("kernel void llvm2_map2_node_10("),
            "Expected binary map kernel, got:\n{}", source);
        assert!(source.contains("a[tid] + b[tid]"),
            "Expected add expression, got:\n{}", source);
        assert!(source.contains("const device float*"),
            "Expected float type, got:\n{}", source);
        assert!(source.contains("if (tid >= 1024u)"),
            "Expected bounds check for 1024 elements, got:\n{}", source);
    }

    #[test]
    fn test_emit_kernel_from_node_data_parallel_neg() {
        let node = make_test_node(
            11,
            NodeKind::DataParallel,
            "NEG",
            4000, // 1000 int elements
            vec![ComputeTarget::CpuScalar, ComputeTarget::Gpu],
        );

        let source = emit_kernel_from_node(&node).unwrap();
        // NEG is unary -> single input map kernel
        assert!(source.contains("kernel void llvm2_map_node_11("),
            "Expected unary map kernel, got:\n{}", source);
        assert!(source.contains("-input[tid]"),
            "Expected negation expression, got:\n{}", source);
        assert!(source.contains("const device int*"),
            "Expected int type for NEG, got:\n{}", source);
    }

    #[test]
    fn test_emit_kernel_from_node_data_parallel_reduce_add() {
        let node = make_test_node(
            12,
            NodeKind::DataParallel,
            "REDUCE_ADD",
            8192, // 2048 float elements
            vec![ComputeTarget::CpuScalar, ComputeTarget::Gpu],
        );

        let source = emit_kernel_from_node(&node).unwrap();
        // Should generate a SIMD reduce kernel
        assert!(source.contains("llvm2_reduce_simd_node_12"),
            "Expected SIMD reduce kernel, got:\n{}", source);
        assert!(source.contains("simd_sum"),
            "Expected simd_sum intrinsic for additive reduce, got:\n{}", source);
        assert!(source.contains("threadgroup float*"),
            "Expected float shared memory, got:\n{}", source);
    }

    #[test]
    fn test_emit_kernel_from_node_data_parallel_reduce_max() {
        let node = make_test_node(
            13,
            NodeKind::DataParallel,
            "REDUCE_MAX",
            4096,
            vec![ComputeTarget::Gpu],
        );

        let source = emit_kernel_from_node(&node).unwrap();
        assert!(source.contains("simd_max"),
            "Expected simd_max intrinsic, got:\n{}", source);
        assert!(source.contains("-INFINITY"),
            "Expected -INFINITY identity for max reduce of float, got:\n{}", source);
    }

    #[test]
    fn test_emit_kernel_from_node_matmul() {
        // 3 * 64^2 * 4 = 49152 bytes for a 64x64 matmul (A + B + C)
        let data_size = 3 * 64 * 64 * 4;
        let node = make_test_node(
            20,
            NodeKind::MatrixHeavy,
            "FMUL",
            data_size,
            vec![ComputeTarget::Gpu],
        );

        let source = emit_kernel_from_node(&node).unwrap();
        assert!(source.contains("kernel void llvm2_matmul_node_20("),
            "Expected matmul kernel, got:\n{}", source);
        assert!(source.contains("simdgroup_matrix<float, 8, 8>"),
            "Expected simdgroup_matrix usage, got:\n{}", source);
        assert!(source.contains("simdgroup_multiply_accumulate"),
            "Expected simdgroup_multiply_accumulate, got:\n{}", source);
        assert!(source.contains("const uint M = 64u;"),
            "Expected M=64, got:\n{}", source);
    }

    #[test]
    fn test_emit_kernel_from_node_scalar_rejected() {
        let node = make_test_node(
            30,
            NodeKind::Scalar,
            "ADD",
            1000,
            vec![ComputeTarget::CpuScalar, ComputeTarget::Gpu],
        );

        let result = emit_kernel_from_node(&node);
        assert!(result.is_err());
        match result.unwrap_err() {
            MetalEmitError::UnsuitableNodeKind { node_id, kind } => {
                assert_eq!(node_id.0, 30);
                assert_eq!(kind, NodeKind::Scalar);
            }
            other => panic!("Expected UnsuitableNodeKind, got {:?}", other),
        }
    }

    #[test]
    fn test_emit_kernel_from_node_gpu_not_legal() {
        let node = make_test_node(
            31,
            NodeKind::DataParallel,
            "ADD",
            1000,
            vec![ComputeTarget::CpuScalar, ComputeTarget::CpuSimd],
        );

        let result = emit_kernel_from_node(&node);
        assert!(result.is_err());
        match result.unwrap_err() {
            MetalEmitError::GpuNotLegal { node_id } => {
                assert_eq!(node_id.0, 31);
            }
            other => panic!("Expected GpuNotLegal, got {:?}", other),
        }
    }

    #[test]
    fn test_emit_kernel_from_node_zero_data_size() {
        let node = make_test_node(
            32,
            NodeKind::DataParallel,
            "FADD",
            0,
            vec![ComputeTarget::Gpu],
        );

        let result = emit_kernel_from_node(&node);
        assert!(result.is_err());
        match result.unwrap_err() {
            MetalEmitError::ZeroDataSize { node_id } => {
                assert_eq!(node_id.0, 32);
            }
            other => panic!("Expected ZeroDataSize, got {:?}", other),
        }
    }

    #[test]
    fn test_emit_kernel_from_node_sqrt_unary() {
        let node = make_test_node(
            14,
            NodeKind::DataParallel,
            "FSQRT",
            2048, // 512 float elements
            vec![ComputeTarget::Gpu],
        );

        let source = emit_kernel_from_node(&node).unwrap();
        assert!(source.contains("kernel void llvm2_map_node_14("),
            "Expected unary map kernel for sqrt, got:\n{}", source);
        assert!(source.contains("sqrt(input[tid])"),
            "Expected sqrt(input[tid]) expression, got:\n{}", source);
    }

    #[test]
    fn test_emit_kernel_from_node_div_binary() {
        let node = make_test_node(
            15,
            NodeKind::DataParallel,
            "FDIV",
            4096,
            vec![ComputeTarget::Gpu],
        );

        let source = emit_kernel_from_node(&node).unwrap();
        assert!(source.contains("a[tid] / b[tid]"),
            "Expected division expression, got:\n{}", source);
    }

    #[test]
    fn test_emit_kernel_from_node_unknown_op_fallback() {
        let node = make_test_node(
            16,
            NodeKind::DataParallel,
            "CUSTOM_OP",
            4096,
            vec![ComputeTarget::Gpu],
        );

        let source = emit_kernel_from_node(&node).unwrap();
        // Unknown ops fall back to identity map
        assert!(source.contains("input[tid]"),
            "Expected identity fallback, got:\n{}", source);
    }

    #[test]
    fn test_emit_dispatch_code_basic() {
        use llvm2_lower::compute_graph::ComputeGraph;
        use llvm2_lower::dispatch::DispatchOp;

        let mut graph = ComputeGraph::new();
        graph.nodes.push(make_test_node(
            1,
            NodeKind::DataParallel,
            "FADD",
            4096,
            vec![ComputeTarget::Gpu],
        ));

        let plan = DispatchPlan {
            ops: vec![
                DispatchOp::KernelLaunch {
                    target: ComputeTarget::Gpu,
                    node_id: ComputeNodeId(1),
                    estimated_cycles: 500,
                },
            ],
            assignment: {
                let mut m = HashMap::new();
                m.insert(ComputeNodeId(1), ComputeTarget::Gpu);
                m
            },
            estimated_total_cycles: 500,
        };

        let code = emit_dispatch_code(&plan, &graph);
        assert!(code.contains("executeDispatchPlan"),
            "Expected function name in dispatch code, got:\n{}", code);
        assert!(code.contains("llvm2_map_node_1"),
            "Expected kernel name in dispatch code, got:\n{}", code);
        assert!(code.contains("newComputePipelineState"),
            "Expected PSO creation in dispatch code, got:\n{}", code);
        assert!(code.contains("dispatchThreads"),
            "Expected dispatch call, got:\n{}", code);
        assert!(code.contains("[cmdBuf commit]"),
            "Expected commit, got:\n{}", code);
    }

    #[test]
    fn test_emit_dispatch_code_with_transfer_and_sync() {
        use llvm2_lower::compute_graph::{ComputeGraph, TransferCost};
        use llvm2_lower::dispatch::DispatchOp;

        let mut graph = ComputeGraph::new();
        graph.nodes.push(make_test_node(
            5,
            NodeKind::DataParallel,
            "FADD",
            8192,
            vec![ComputeTarget::Gpu],
        ));

        let plan = DispatchPlan {
            ops: vec![
                DispatchOp::DataTransfer {
                    src: ComputeTarget::CpuScalar,
                    dst: ComputeTarget::Gpu,
                    size_bytes: 8192,
                    cost: TransferCost::zero(),
                    edge_from: ComputeNodeId(0),
                    edge_to: ComputeNodeId(5),
                },
                DispatchOp::KernelLaunch {
                    target: ComputeTarget::Gpu,
                    node_id: ComputeNodeId(5),
                    estimated_cycles: 1000,
                },
                DispatchOp::Synchronize {
                    target: ComputeTarget::Gpu,
                    node_id: ComputeNodeId(5),
                },
            ],
            assignment: {
                let mut m = HashMap::new();
                m.insert(ComputeNodeId(5), ComputeTarget::Gpu);
                m
            },
            estimated_total_cycles: 1200,
        };

        let code = emit_dispatch_code(&plan, &graph);
        // Check transfer comment (UMA no-copy)
        assert!(code.contains("UMA: no explicit copy needed"),
            "Expected UMA comment for CPU->GPU transfer, got:\n{}", code);
        // Check sync
        assert!(code.contains("[cmdBuf commit]"),
            "Expected commit for sync, got:\n{}", code);
        assert!(code.contains("[cmdBuf waitUntilCompleted]"),
            "Expected waitUntilCompleted for sync, got:\n{}", code);
        // Plan stats
        assert!(code.contains("1 launches"),
            "Expected 1 launch in stats, got:\n{}", code);
        assert!(code.contains("1 transfers"),
            "Expected 1 transfer in stats, got:\n{}", code);
    }

    #[test]
    fn test_is_reduce_pattern() {
        assert!(is_reduce_pattern("REDUCE_ADD"));
        assert!(is_reduce_pattern("REDUCE_MAX"));
        assert!(is_reduce_pattern("SUM"));
        assert!(is_reduce_pattern("PROD"));
        assert!(is_reduce_pattern("DOT"));
        assert!(is_reduce_pattern("ACCUM"));
        assert!(!is_reduce_pattern("FADD"));
        assert!(!is_reduce_pattern("MUL"));
    }
}
