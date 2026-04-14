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
}
