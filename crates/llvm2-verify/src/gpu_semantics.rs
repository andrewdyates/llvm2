// llvm2-verify/gpu_semantics.rs - Metal GPU parallel operation SMT encoding
//
// Author: Andrew Yates <ayates@dropbox.com>
// Copyright 2026 Dropbox, Inc. | License: Apache-2.0
//
// Encodes Metal GPU compute kernel semantics as SMT expressions for the
// unified solver architecture. GPU kernels execute massively parallel operations
// (map, reduce, scatter, gather) whose semantics are encoded as sequential
// specifications -- tMIR proofs of algebraic properties (Pure, Associative,
// Commutative) justify that the parallel execution is equivalent.
//
// Design doc: designs/2026-04-14-metal-gpu-semantics.md
// Reference: Apple. "Metal Shading Language Specification," Version 3.2.

//! Metal GPU parallel operation SMT encoding for the unified solver.
//!
//! # Design Principle: Semantic Equivalence, Not Implementation Detail
//!
//! The SMT encoding represents the *semantic effect* of a GPU kernel, not its
//! implementation. The GPU's parallel execution (threadgroups, barriers, SIMD
//! groups) produces the same result as a sequential loop IF the operation's
//! algebraic properties (associativity, commutativity) hold. tMIR proofs
//! guarantee these properties.
//!
//! Therefore, we encode GPU kernels as sequential specifications and rely on
//! tMIR proofs to justify that the parallel implementation is equivalent.
//!
//! # Supported Operations
//!
//! - **Parallel Map**: element-wise `output[i] = f(input[i])`
//! - **Parallel Reduce**: `result = fold(op, identity, input[0..N])`
//! - **Map-Reduce (fused)**: `result = fold(op, id, map(f, input))`
//! - **Threadgroup Barrier**: synchronization point (identity for verification)
//! - **SIMD Group Operations**: shuffle/reduce within a 32-thread wave

use crate::smt::{SmtExpr, SmtSort, RoundingMode};

// ---------------------------------------------------------------------------
// GpuKernelShape
// ---------------------------------------------------------------------------

/// Describes the execution shape of a Metal GPU compute kernel.
///
/// Maps to Metal's execution hierarchy:
/// - Grid: total work items (N elements)
/// - Threadgroup: cooperative threads sharing threadgroup memory
/// - SIMD group: 32 threads executing in lockstep on one EU
///
/// Reference: Metal Shading Language Specification, Chapter 4.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GpuKernelShape {
    /// Total number of work items (grid size). Must be > 0.
    pub grid_size: u64,
    /// Threads per threadgroup. Must divide grid_size evenly.
    /// Typical values: 32, 64, 128, 256, 512, 1024.
    /// Maximum: 1024 (Metal spec).
    pub threadgroup_size: u32,
    /// SIMD group width (threads per SIMD group).
    /// Always 32 on Apple GPUs.
    pub simd_width: u32,
}

impl GpuKernelShape {
    /// Create a new kernel shape with Apple GPU defaults (simd_width=32).
    pub fn new(grid_size: u64, threadgroup_size: u32) -> Self {
        Self {
            grid_size,
            threadgroup_size,
            simd_width: 32,
        }
    }

    /// Number of threadgroups needed to cover the grid.
    pub fn num_threadgroups(&self) -> u64 {
        (self.grid_size + self.threadgroup_size as u64 - 1) / self.threadgroup_size as u64
    }

    /// Number of SIMD groups per threadgroup.
    pub fn simd_groups_per_threadgroup(&self) -> u32 {
        (self.threadgroup_size + self.simd_width - 1) / self.simd_width
    }

    /// Validate the kernel shape against Metal hardware constraints.
    ///
    /// Returns `true` if the shape is valid for dispatch.
    pub fn is_valid(&self) -> bool {
        self.grid_size > 0
            && self.threadgroup_size > 0
            && self.threadgroup_size <= 1024
            && self.simd_width > 0
            && self.threadgroup_size % self.simd_width == 0
    }
}

// ---------------------------------------------------------------------------
// Binary operations for reductions
// ---------------------------------------------------------------------------

/// Associative, commutative binary operations supported in GPU reductions.
///
/// Each operation has a known identity element used to initialize the
/// accumulator. tMIR must prove Associative + Commutative for the operation
/// before the solver considers it legal for GPU dispatch.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GpuReduceOp {
    /// Addition (identity: 0).
    Add,
    /// Multiplication (identity: 1).
    Mul,
    /// Minimum (identity: type max value).
    Min,
    /// Maximum (identity: type min value).
    Max,
    /// Bitwise AND (identity: all-ones).
    BitwiseAnd,
    /// Bitwise OR (identity: 0).
    BitwiseOr,
    /// Bitwise XOR (identity: 0).
    BitwiseXor,
}

impl GpuReduceOp {
    /// Apply this operation to two bitvector SMT expressions.
    pub fn apply_bv(&self, a: SmtExpr, b: SmtExpr) -> SmtExpr {
        match self {
            GpuReduceOp::Add => a.bvadd(b),
            GpuReduceOp::Mul => a.bvmul(b),
            GpuReduceOp::Min => {
                // Unsigned min: ite(a < b, a, b)
                let cond = a.clone().bvult(b.clone());
                SmtExpr::ite(cond, a, b)
            }
            GpuReduceOp::Max => {
                // Unsigned max: ite(a > b, a, b)
                let cond = a.clone().bvugt(b.clone());
                SmtExpr::ite(cond, a, b)
            }
            GpuReduceOp::BitwiseAnd => a.bvand(b),
            GpuReduceOp::BitwiseOr => a.bvor(b),
            GpuReduceOp::BitwiseXor => a.bvxor(b),
        }
    }

    /// Apply this operation to two floating-point SMT expressions.
    ///
    /// Only valid for Add, Mul, Min, Max. Bitwise ops on FP are not meaningful.
    pub fn apply_fp(&self, a: SmtExpr, b: SmtExpr) -> SmtExpr {
        match self {
            GpuReduceOp::Add => SmtExpr::fp_add(RoundingMode::RNE, a, b),
            GpuReduceOp::Mul => SmtExpr::fp_mul(RoundingMode::RNE, a, b),
            GpuReduceOp::Min => {
                let cond = a.clone().fp_lt(b.clone());
                SmtExpr::ite(cond, a, b)
            }
            GpuReduceOp::Max => {
                let cond = b.clone().fp_lt(a.clone());
                SmtExpr::ite(cond, a, b)
            }
            _ => panic!("bitwise reduce ops not valid for floating-point"),
        }
    }

    /// Return the identity element for this operation as a bitvector constant.
    pub fn identity_bv(&self, width: u32) -> SmtExpr {
        match self {
            GpuReduceOp::Add | GpuReduceOp::BitwiseOr | GpuReduceOp::BitwiseXor => {
                SmtExpr::bv_const(0, width)
            }
            GpuReduceOp::Mul => SmtExpr::bv_const(1, width),
            GpuReduceOp::Min => {
                // All-ones = max unsigned value
                let max_val = if width >= 64 { u64::MAX } else { (1u64 << width) - 1 };
                SmtExpr::bv_const(max_val, width)
            }
            GpuReduceOp::Max => SmtExpr::bv_const(0, width),
            GpuReduceOp::BitwiseAnd => {
                let all_ones = if width >= 64 { u64::MAX } else { (1u64 << width) - 1 };
                SmtExpr::bv_const(all_ones, width)
            }
        }
    }

    /// Return the identity element for this operation as a floating-point constant.
    ///
    /// Only valid for Add, Mul, Min, Max.
    pub fn identity_fp32(&self) -> SmtExpr {
        match self {
            GpuReduceOp::Add => SmtExpr::fp32_const(0.0),
            GpuReduceOp::Mul => SmtExpr::fp32_const(1.0),
            GpuReduceOp::Min => SmtExpr::fp32_const(f32::INFINITY),
            GpuReduceOp::Max => SmtExpr::fp32_const(f32::NEG_INFINITY),
            _ => panic!("bitwise reduce ops not valid for floating-point"),
        }
    }
}

// ---------------------------------------------------------------------------
// SIMD group operations
// ---------------------------------------------------------------------------

/// Operations available within a SIMD group (32-thread wave).
///
/// These correspond to Metal's `simd_*` intrinsics which execute at hardware
/// speed within a single SIMD group.
///
/// Reference: Metal Shading Language Specification, Section 6.8.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SimdGroupOp {
    /// `simd_shuffle(val, lane)`: read value from another lane.
    Shuffle,
    /// `simd_sum(val)`: sum all lanes in the SIMD group.
    Sum,
    /// `simd_min(val)`: minimum across all lanes.
    Min,
    /// `simd_max(val)`: maximum across all lanes.
    Max,
    /// `simd_and(val)`: bitwise AND across all lanes.
    And,
    /// `simd_or(val)`: bitwise OR across all lanes.
    Or,
    /// `simd_xor(val)`: bitwise XOR across all lanes.
    Xor,
    /// `simd_prefix_inclusive_sum(val)`: inclusive prefix sum.
    PrefixInclusiveSum,
    /// `simd_broadcast(val, lane)`: broadcast one lane to all lanes.
    Broadcast,
}

// ---------------------------------------------------------------------------
// Parallel Map encoding
// ---------------------------------------------------------------------------

/// SMT encoding of a Metal parallel map kernel.
///
/// Semantics: `for each i in [0, N): output[i] = f(input[i])`
///
/// This is the GPU equivalent of a vectorized loop. The SMT encoding
/// is identical to the sequential specification because parallel map
/// with a Pure function is order-independent.
///
/// tMIR proof requirements: Pure, InBounds(0..N), ValidBorrow(input, output)
///
/// z4 theory: QF_ABV (arrays + bitvectors)
///
/// # Arguments
/// * `input` - Input array expression: `(Array BV64 BV_elem)`
/// * `f` - Per-element pure function mapping element to element
/// * `n` - Grid size (number of elements)
/// * `elem_sort` - Sort of array elements (used for the output constant array)
pub fn encode_parallel_map(
    input: &SmtExpr,
    f: &dyn Fn(&SmtExpr) -> SmtExpr,
    n: u64,
    elem_sort: &SmtSort,
) -> SmtExpr {
    let mut result = SmtExpr::const_array(
        SmtSort::BitVec(64),
        default_value_for_sort(elem_sort),
    );
    for i in 0..n {
        let idx = SmtExpr::bv_const(i, 64);
        let elem = SmtExpr::select(input.clone(), idx.clone());
        let transformed = f(&elem);
        result = SmtExpr::store(result, idx, transformed);
    }
    result
}

/// SMT encoding of a Metal parallel map kernel with two input arrays.
///
/// Semantics: `for each i in [0, N): output[i] = f(a[i], b[i])`
///
/// Used for binary element-wise ops like vector addition, multiplication, etc.
pub fn encode_parallel_map2(
    input_a: &SmtExpr,
    input_b: &SmtExpr,
    f: &dyn Fn(&SmtExpr, &SmtExpr) -> SmtExpr,
    n: u64,
    elem_sort: &SmtSort,
) -> SmtExpr {
    let mut result = SmtExpr::const_array(
        SmtSort::BitVec(64),
        default_value_for_sort(elem_sort),
    );
    for i in 0..n {
        let idx = SmtExpr::bv_const(i, 64);
        let a_elem = SmtExpr::select(input_a.clone(), idx.clone());
        let b_elem = SmtExpr::select(input_b.clone(), idx.clone());
        let transformed = f(&a_elem, &b_elem);
        result = SmtExpr::store(result, idx, transformed);
    }
    result
}

// ---------------------------------------------------------------------------
// Parallel Reduce encoding
// ---------------------------------------------------------------------------

/// SMT encoding of a Metal parallel reduce kernel.
///
/// Semantics: `result = fold(op, identity, input[0..N])`
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
///
/// # Arguments
/// * `input` - Input array expression: `(Array BV64 BV_elem)`
/// * `op` - Associative + commutative binary operator
/// * `identity` - Identity element for the operator
/// * `n` - Number of elements to reduce
pub fn encode_parallel_reduce(
    input: &SmtExpr,
    op: &dyn Fn(&SmtExpr, &SmtExpr) -> SmtExpr,
    identity: &SmtExpr,
    n: u64,
) -> SmtExpr {
    let mut acc = identity.clone();
    for i in 0..n {
        let idx = SmtExpr::bv_const(i, 64);
        let elem = SmtExpr::select(input.clone(), idx);
        acc = op(&acc, &elem);
    }
    acc
}

/// SMT encoding of a parallel reduce using a [`GpuReduceOp`] on bitvectors.
///
/// Convenience wrapper that uses the op's built-in identity element.
pub fn encode_parallel_reduce_bv(
    input: &SmtExpr,
    op: GpuReduceOp,
    elem_width: u32,
    n: u64,
) -> SmtExpr {
    let identity = op.identity_bv(elem_width);
    encode_parallel_reduce(input, &|a, b| op.apply_bv(a.clone(), b.clone()), &identity, n)
}

// ---------------------------------------------------------------------------
// Map-Reduce (fused) encoding
// ---------------------------------------------------------------------------

/// SMT encoding of fused map-reduce: `result = fold(reduce_op, id, map(f, input[0..N]))`.
///
/// Common pattern: dot product = `sum(a[i] * b[i] for i in 0..N)`.
///
/// The fused encoding avoids materializing the intermediate mapped array.
pub fn encode_map_reduce(
    input_a: &SmtExpr,
    input_b: &SmtExpr,
    map_f: &dyn Fn(&SmtExpr, &SmtExpr) -> SmtExpr,
    reduce_op: &dyn Fn(&SmtExpr, &SmtExpr) -> SmtExpr,
    identity: &SmtExpr,
    n: u64,
) -> SmtExpr {
    let mut acc = identity.clone();
    for i in 0..n {
        let idx = SmtExpr::bv_const(i, 64);
        let a_elem = SmtExpr::select(input_a.clone(), idx.clone());
        let b_elem = SmtExpr::select(input_b.clone(), idx);
        let mapped = map_f(&a_elem, &b_elem);
        acc = reduce_op(&acc, &mapped);
    }
    acc
}

// ---------------------------------------------------------------------------
// Threadgroup barrier encoding
// ---------------------------------------------------------------------------

/// SMT encoding of a Metal threadgroup barrier.
///
/// `threadgroup_barrier(mem_flags::mem_threadgroup)` is a synchronization
/// point that ensures all threads in a threadgroup have completed their
/// writes to threadgroup memory before any thread proceeds past the barrier.
///
/// In the SMT encoding, barriers are identity operations (no-ops) because:
/// 1. We encode the sequential specification, not the parallel implementation
/// 2. The tMIR proof of Pure guarantees no data races
/// 3. The barrier's effect (ordering) is implicit in the sequential encoding
///
/// The function returns its input unchanged -- it exists as a semantic marker
/// for the verification pipeline to track synchronization points.
pub fn encode_threadgroup_barrier(memory_state: &SmtExpr) -> SmtExpr {
    memory_state.clone()
}

// ---------------------------------------------------------------------------
// SIMD group operation encoding
// ---------------------------------------------------------------------------

/// SMT encoding of a SIMD group shuffle operation.
///
/// `simd_shuffle(data, lane)`: each thread reads the value from the thread
/// at position `lane` within its SIMD group.
///
/// In the SMT encoding, a SIMD group of width W is represented as an array
/// of W elements. The shuffle reads `data[lane]`.
///
/// # Arguments
/// * `simd_data` - Array of per-lane values: `(Array BV32 BV_elem)`
/// * `source_lane` - Lane index to read from (0..simd_width-1)
pub fn encode_simdgroup_shuffle(
    simd_data: &SmtExpr,
    source_lane: &SmtExpr,
) -> SmtExpr {
    SmtExpr::select(simd_data.clone(), source_lane.clone())
}

/// SMT encoding of a SIMD group broadcast operation.
///
/// `simd_broadcast(data, lane)`: all threads in the SIMD group receive
/// the value from thread `lane`.
///
/// Returns an array where every element equals `data[source_lane]`.
///
/// # Arguments
/// * `simd_data` - Array of per-lane values
/// * `source_lane` - Lane whose value is broadcast
/// * `simd_width` - Number of lanes in the SIMD group (32 on Apple GPUs)
/// * `elem_sort` - Sort of the element values
pub fn encode_simdgroup_broadcast(
    simd_data: &SmtExpr,
    source_lane: &SmtExpr,
    simd_width: u32,
    elem_sort: &SmtSort,
) -> SmtExpr {
    let broadcast_val = SmtExpr::select(simd_data.clone(), source_lane.clone());
    let mut result = SmtExpr::const_array(
        SmtSort::BitVec(32),
        default_value_for_sort(elem_sort),
    );
    for i in 0..simd_width {
        let idx = SmtExpr::bv_const(i as u64, 32);
        result = SmtExpr::store(result, idx, broadcast_val.clone());
    }
    result
}

/// SMT encoding of a SIMD group reduction operation.
///
/// `simd_sum(val)`, `simd_min(val)`, etc.: reduce all lanes in the SIMD
/// group using the given associative+commutative operation.
///
/// The result is the same for all lanes (every thread gets the reduction
/// result). We return the scalar result value.
///
/// # Arguments
/// * `simd_data` - Array of per-lane values: `(Array BV32 BV_elem)`
/// * `op` - Reduction operation
/// * `identity` - Identity element for the operation
/// * `simd_width` - Number of lanes (32)
pub fn encode_simdgroup_reduce(
    simd_data: &SmtExpr,
    op: &dyn Fn(&SmtExpr, &SmtExpr) -> SmtExpr,
    identity: &SmtExpr,
    simd_width: u32,
) -> SmtExpr {
    let mut acc = identity.clone();
    for i in 0..simd_width {
        let idx = SmtExpr::bv_const(i as u64, 32);
        let elem = SmtExpr::select(simd_data.clone(), idx);
        acc = op(&acc, &elem);
    }
    acc
}

/// SMT encoding of SIMD group inclusive prefix sum.
///
/// `simd_prefix_inclusive_sum(val)`: lane i gets the sum of lanes 0..=i.
///
/// Returns an array where `result[i] = sum(data[0..=i])`.
///
/// # Arguments
/// * `simd_data` - Array of per-lane values
/// * `simd_width` - Number of lanes
/// * `elem_sort` - Sort of element values
pub fn encode_simdgroup_prefix_sum(
    simd_data: &SmtExpr,
    simd_width: u32,
    elem_sort: &SmtSort,
) -> SmtExpr {
    let mut result = SmtExpr::const_array(
        SmtSort::BitVec(32),
        default_value_for_sort(elem_sort),
    );
    let mut running_sum = SmtExpr::bv_const(0, elem_width_from_sort(elem_sort));
    for i in 0..simd_width {
        let idx = SmtExpr::bv_const(i as u64, 32);
        let elem = SmtExpr::select(simd_data.clone(), idx.clone());
        running_sum = running_sum.bvadd(elem);
        result = SmtExpr::store(result, idx, running_sum.clone());
    }
    result
}

// ---------------------------------------------------------------------------
// UMA transfer encoding
// ---------------------------------------------------------------------------

/// SMT encoding of CPU-to-GPU data transfer on Apple UMA (Unified Memory
/// Architecture).
///
/// On shared-memory systems, the transfer is a no-op: the GPU reads from
/// the same memory as the CPU. The proof obligation is on ValidBorrow:
/// the CPU must not mutate the buffer during GPU execution.
///
/// Returns: the same array expression (identity function).
pub fn encode_uma_transfer(cpu_memory: &SmtExpr) -> SmtExpr {
    cpu_memory.clone()
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Return a default zero value for the given sort.
fn default_value_for_sort(sort: &SmtSort) -> SmtExpr {
    match sort {
        SmtSort::BitVec(w) => SmtExpr::bv_const(0, *w),
        SmtSort::FloatingPoint(8, 24) => SmtExpr::fp32_const(0.0),
        SmtSort::FloatingPoint(11, 53) => SmtExpr::fp64_const(0.0),
        SmtSort::FloatingPoint(eb, sb) => SmtExpr::fp_const(0, *eb, *sb),
        SmtSort::Bool => SmtExpr::bool_const(false),
        SmtSort::Array(idx, elem) => {
            SmtExpr::const_array((**idx).clone(), default_value_for_sort(elem))
        }
    }
}

/// Extract bitvector width from an element sort. Panics for non-BV sorts.
fn elem_width_from_sort(sort: &SmtSort) -> u32 {
    match sort {
        SmtSort::BitVec(w) => *w,
        _ => panic!("expected BitVec sort for element width, got {:?}", sort),
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::smt::EvalResult;
    use std::collections::HashMap;

    /// Helper to build an array from a slice of u64 values.
    /// Returns the array expression (no variables needed -- fully concrete).
    fn build_array_bv32(values: &[u64]) -> SmtExpr {
        let mut arr = SmtExpr::const_array(
            SmtSort::BitVec(64),
            SmtExpr::bv_const(0, 32),
        );
        for (i, &v) in values.iter().enumerate() {
            let idx = SmtExpr::bv_const(i as u64, 64);
            arr = SmtExpr::store(arr, idx, SmtExpr::bv_const(v, 32));
        }
        arr
    }

    /// Helper to build an array indexed by BV32 (for SIMD group tests).
    fn build_simd_array(values: &[u64], elem_width: u32) -> SmtExpr {
        let mut arr = SmtExpr::const_array(
            SmtSort::BitVec(32),
            SmtExpr::bv_const(0, elem_width),
        );
        for (i, &v) in values.iter().enumerate() {
            let idx = SmtExpr::bv_const(i as u64, 32);
            arr = SmtExpr::store(arr, idx, SmtExpr::bv_const(v, elem_width));
        }
        arr
    }

    /// Evaluate an expression with an empty environment.
    fn eval_empty(expr: &SmtExpr) -> EvalResult {
        expr.try_eval(&HashMap::new()).unwrap()
    }

    /// Select element at index i from array, evaluate.
    fn select_eval(arr: &SmtExpr, i: u64, idx_width: u32) -> EvalResult {
        let sel = SmtExpr::select(arr.clone(), SmtExpr::bv_const(i, idx_width));
        eval_empty(&sel)
    }

    // =======================================================================
    // GpuKernelShape tests
    // =======================================================================

    #[test]
    fn test_shape_basic() {
        let shape = GpuKernelShape::new(1024, 256);
        assert_eq!(shape.num_threadgroups(), 4);
        assert_eq!(shape.simd_groups_per_threadgroup(), 8);
        assert!(shape.is_valid());
    }

    #[test]
    fn test_shape_non_divisible_grid() {
        let shape = GpuKernelShape::new(1000, 256);
        assert_eq!(shape.num_threadgroups(), 4); // ceil(1000/256) = 4
        assert!(shape.is_valid());
    }

    #[test]
    fn test_shape_invalid_zero_grid() {
        let shape = GpuKernelShape::new(0, 256);
        assert!(!shape.is_valid());
    }

    #[test]
    fn test_shape_invalid_exceeds_max_threadgroup() {
        let shape = GpuKernelShape::new(4096, 2048);
        assert!(!shape.is_valid());
    }

    #[test]
    fn test_shape_invalid_non_aligned_threadgroup() {
        let shape = GpuKernelShape {
            grid_size: 1024,
            threadgroup_size: 33, // not a multiple of simd_width=32
            simd_width: 32,
        };
        assert!(!shape.is_valid());
    }

    // =======================================================================
    // Parallel Map tests
    // =======================================================================

    #[test]
    fn test_parallel_map_add_one() {
        // map(x -> x + 1) over [10, 20, 30, 40]
        let input = build_array_bv32(&[10, 20, 30, 40]);
        let result = encode_parallel_map(
            &input,
            &|elem| elem.clone().bvadd(SmtExpr::bv_const(1, 32)),
            4,
            &SmtSort::BitVec(32),
        );

        assert_eq!(select_eval(&result, 0, 64), EvalResult::Bv(11));
        assert_eq!(select_eval(&result, 1, 64), EvalResult::Bv(21));
        assert_eq!(select_eval(&result, 2, 64), EvalResult::Bv(31));
        assert_eq!(select_eval(&result, 3, 64), EvalResult::Bv(41));
    }

    #[test]
    fn test_parallel_map_double() {
        // map(x -> x * 2) over [5, 10, 15]
        let input = build_array_bv32(&[5, 10, 15]);
        let result = encode_parallel_map(
            &input,
            &|elem| elem.clone().bvmul(SmtExpr::bv_const(2, 32)),
            3,
            &SmtSort::BitVec(32),
        );

        assert_eq!(select_eval(&result, 0, 64), EvalResult::Bv(10));
        assert_eq!(select_eval(&result, 1, 64), EvalResult::Bv(20));
        assert_eq!(select_eval(&result, 2, 64), EvalResult::Bv(30));
    }

    #[test]
    fn test_parallel_map_negate() {
        // map(x -> -x) over [1, 0, 0xFFFFFFFF]
        let input = build_array_bv32(&[1, 0, 0xFFFFFFFF]);
        let result = encode_parallel_map(
            &input,
            &|elem| elem.clone().bvneg(),
            3,
            &SmtSort::BitVec(32),
        );

        assert_eq!(select_eval(&result, 0, 64), EvalResult::Bv(0xFFFFFFFF));
        assert_eq!(select_eval(&result, 1, 64), EvalResult::Bv(0));
        assert_eq!(select_eval(&result, 2, 64), EvalResult::Bv(1));
    }

    #[test]
    fn test_parallel_map_identity() {
        // map(x -> x) should return the same values
        let input = build_array_bv32(&[42, 99, 0]);
        let result = encode_parallel_map(
            &input,
            &|elem| elem.clone(),
            3,
            &SmtSort::BitVec(32),
        );

        assert_eq!(select_eval(&result, 0, 64), EvalResult::Bv(42));
        assert_eq!(select_eval(&result, 1, 64), EvalResult::Bv(99));
        assert_eq!(select_eval(&result, 2, 64), EvalResult::Bv(0));
    }

    #[test]
    fn test_parallel_map_wrapping() {
        // map(x -> x + 1) on max u32 should wrap to 0
        let input = build_array_bv32(&[0xFFFFFFFF]);
        let result = encode_parallel_map(
            &input,
            &|elem| elem.clone().bvadd(SmtExpr::bv_const(1, 32)),
            1,
            &SmtSort::BitVec(32),
        );

        assert_eq!(select_eval(&result, 0, 64), EvalResult::Bv(0));
    }

    #[test]
    fn test_parallel_map2_add() {
        // map2(a, b -> a + b) over [1, 2, 3] and [10, 20, 30]
        let input_a = build_array_bv32(&[1, 2, 3]);
        let input_b = build_array_bv32(&[10, 20, 30]);
        let result = encode_parallel_map2(
            &input_a,
            &input_b,
            &|a, b| a.clone().bvadd(b.clone()),
            3,
            &SmtSort::BitVec(32),
        );

        assert_eq!(select_eval(&result, 0, 64), EvalResult::Bv(11));
        assert_eq!(select_eval(&result, 1, 64), EvalResult::Bv(22));
        assert_eq!(select_eval(&result, 2, 64), EvalResult::Bv(33));
    }

    // =======================================================================
    // Parallel Reduce tests
    // =======================================================================

    #[test]
    fn test_reduce_sum() {
        // sum([10, 20, 30, 40]) = 100
        let input = build_array_bv32(&[10, 20, 30, 40]);
        let result = encode_parallel_reduce(
            &input,
            &|a, b| a.clone().bvadd(b.clone()),
            &SmtExpr::bv_const(0, 32),
            4,
        );

        assert_eq!(eval_empty(&result), EvalResult::Bv(100));
    }

    #[test]
    fn test_reduce_product() {
        // product([2, 3, 4]) = 24
        let input = build_array_bv32(&[2, 3, 4]);
        let result = encode_parallel_reduce(
            &input,
            &|a, b| a.clone().bvmul(b.clone()),
            &SmtExpr::bv_const(1, 32),
            3,
        );

        assert_eq!(eval_empty(&result), EvalResult::Bv(24));
    }

    #[test]
    fn test_reduce_min() {
        // min([100, 5, 42, 7]) = 5
        let input = build_array_bv32(&[100, 5, 42, 7]);
        let result = encode_parallel_reduce_bv(
            &input,
            GpuReduceOp::Min,
            32,
            4,
        );

        assert_eq!(eval_empty(&result), EvalResult::Bv(5));
    }

    #[test]
    fn test_reduce_max() {
        // max([10, 99, 42, 7]) = 99
        let input = build_array_bv32(&[10, 99, 42, 7]);
        let result = encode_parallel_reduce_bv(
            &input,
            GpuReduceOp::Max,
            32,
            4,
        );

        assert_eq!(eval_empty(&result), EvalResult::Bv(99));
    }

    #[test]
    fn test_reduce_bitwise_and() {
        // AND([0xFF, 0x0F, 0xF0]) = 0xFF & 0x0F & 0xF0 = 0x00
        let input = build_array_bv32(&[0xFF, 0x0F, 0xF0]);
        let result = encode_parallel_reduce_bv(
            &input,
            GpuReduceOp::BitwiseAnd,
            32,
            3,
        );

        assert_eq!(eval_empty(&result), EvalResult::Bv(0));
    }

    #[test]
    fn test_reduce_bitwise_or() {
        // OR([0x0F, 0xF0, 0x100]) = 0x1FF
        let input = build_array_bv32(&[0x0F, 0xF0, 0x100]);
        let result = encode_parallel_reduce_bv(
            &input,
            GpuReduceOp::BitwiseOr,
            32,
            3,
        );

        assert_eq!(eval_empty(&result), EvalResult::Bv(0x1FF));
    }

    #[test]
    fn test_reduce_bitwise_xor() {
        // XOR([0xFF, 0xFF, 0x0F]) = 0x0F (0xFF ^ 0xFF = 0, 0 ^ 0x0F = 0x0F)
        let input = build_array_bv32(&[0xFF, 0xFF, 0x0F]);
        let result = encode_parallel_reduce_bv(
            &input,
            GpuReduceOp::BitwiseXor,
            32,
            3,
        );

        assert_eq!(eval_empty(&result), EvalResult::Bv(0x0F));
    }

    #[test]
    fn test_reduce_empty() {
        // Reducing zero elements should return the identity
        let input = build_array_bv32(&[]);
        let result = encode_parallel_reduce_bv(
            &input,
            GpuReduceOp::Add,
            32,
            0,
        );

        assert_eq!(eval_empty(&result), EvalResult::Bv(0));
    }

    #[test]
    fn test_reduce_single_element() {
        // Reducing a single element should return that element
        let input = build_array_bv32(&[42]);
        let result = encode_parallel_reduce_bv(
            &input,
            GpuReduceOp::Add,
            32,
            1,
        );

        assert_eq!(eval_empty(&result), EvalResult::Bv(42));
    }

    // =======================================================================
    // Commutativity / associativity tests for reduce ops
    // =======================================================================

    #[test]
    fn test_reduce_add_commutative() {
        // sum([a, b]) == sum([b, a])
        let arr1 = build_array_bv32(&[7, 13]);
        let arr2 = build_array_bv32(&[13, 7]);
        let r1 = encode_parallel_reduce_bv(&arr1, GpuReduceOp::Add, 32, 2);
        let r2 = encode_parallel_reduce_bv(&arr2, GpuReduceOp::Add, 32, 2);
        assert_eq!(eval_empty(&r1), eval_empty(&r2));
    }

    #[test]
    fn test_reduce_min_commutative() {
        // min([100, 3]) == min([3, 100])
        let arr1 = build_array_bv32(&[100, 3]);
        let arr2 = build_array_bv32(&[3, 100]);
        let r1 = encode_parallel_reduce_bv(&arr1, GpuReduceOp::Min, 32, 2);
        let r2 = encode_parallel_reduce_bv(&arr2, GpuReduceOp::Min, 32, 2);
        assert_eq!(eval_empty(&r1), eval_empty(&r2));
    }

    #[test]
    fn test_reduce_max_commutative() {
        // max([5, 99]) == max([99, 5])
        let arr1 = build_array_bv32(&[5, 99]);
        let arr2 = build_array_bv32(&[99, 5]);
        let r1 = encode_parallel_reduce_bv(&arr1, GpuReduceOp::Max, 32, 2);
        let r2 = encode_parallel_reduce_bv(&arr2, GpuReduceOp::Max, 32, 2);
        assert_eq!(eval_empty(&r1), eval_empty(&r2));
    }

    #[test]
    fn test_reduce_add_associative() {
        // (a + b) + c == a + (b + c) for concrete values
        // We verify by checking that reducing [a, b, c] in different orders
        // gives the same result. Our sequential fold is always left-to-right,
        // but the commutativity test above + this check covers the tree reduction
        // reordering the GPU actually performs.
        let arr_abc = build_array_bv32(&[5, 10, 20]);
        let arr_bca = build_array_bv32(&[10, 20, 5]);
        let arr_cab = build_array_bv32(&[20, 5, 10]);

        let r1 = encode_parallel_reduce_bv(&arr_abc, GpuReduceOp::Add, 32, 3);
        let r2 = encode_parallel_reduce_bv(&arr_bca, GpuReduceOp::Add, 32, 3);
        let r3 = encode_parallel_reduce_bv(&arr_cab, GpuReduceOp::Add, 32, 3);

        let v1 = eval_empty(&r1);
        let v2 = eval_empty(&r2);
        let v3 = eval_empty(&r3);
        assert_eq!(v1, v2);
        assert_eq!(v2, v3);
        assert_eq!(v1, EvalResult::Bv(35));
    }

    // =======================================================================
    // Map-Reduce tests
    // =======================================================================

    #[test]
    fn test_map_reduce_dot_product() {
        // dot([1, 2, 3], [4, 5, 6]) = 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32
        let a = build_array_bv32(&[1, 2, 3]);
        let b = build_array_bv32(&[4, 5, 6]);
        let result = encode_map_reduce(
            &a,
            &b,
            &|ai, bi| ai.clone().bvmul(bi.clone()),
            &|acc, prod| acc.clone().bvadd(prod.clone()),
            &SmtExpr::bv_const(0, 32),
            3,
        );

        assert_eq!(eval_empty(&result), EvalResult::Bv(32));
    }

    #[test]
    fn test_map_reduce_sum_of_squares() {
        // sum(x^2 for x in [3, 4]) = 9 + 16 = 25
        // Using map_reduce with both inputs as the same array
        let a = build_array_bv32(&[3, 4]);
        let result = encode_map_reduce(
            &a,
            &a,
            &|ai, bi| ai.clone().bvmul(bi.clone()),
            &|acc, prod| acc.clone().bvadd(prod.clone()),
            &SmtExpr::bv_const(0, 32),
            2,
        );

        assert_eq!(eval_empty(&result), EvalResult::Bv(25));
    }

    // =======================================================================
    // Threadgroup barrier tests
    // =======================================================================

    #[test]
    fn test_barrier_is_identity() {
        let arr = build_array_bv32(&[1, 2, 3]);
        let after_barrier = encode_threadgroup_barrier(&arr);
        // Barrier should not change the memory state
        assert_eq!(select_eval(&after_barrier, 0, 64), EvalResult::Bv(1));
        assert_eq!(select_eval(&after_barrier, 1, 64), EvalResult::Bv(2));
        assert_eq!(select_eval(&after_barrier, 2, 64), EvalResult::Bv(3));
    }

    // =======================================================================
    // SIMD group operation tests
    // =======================================================================

    #[test]
    fn test_simdgroup_shuffle() {
        // 4-lane SIMD group (small for testing) with values [10, 20, 30, 40]
        let data = build_simd_array(&[10, 20, 30, 40], 32);
        // Shuffle: read from lane 2
        let result = encode_simdgroup_shuffle(&data, &SmtExpr::bv_const(2, 32));
        assert_eq!(eval_empty(&result), EvalResult::Bv(30));
    }

    #[test]
    fn test_simdgroup_reduce_sum() {
        // 4-lane SIMD reduce sum: 10 + 20 + 30 + 40 = 100
        let data = build_simd_array(&[10, 20, 30, 40], 32);
        let result = encode_simdgroup_reduce(
            &data,
            &|a, b| a.clone().bvadd(b.clone()),
            &SmtExpr::bv_const(0, 32),
            4,
        );
        assert_eq!(eval_empty(&result), EvalResult::Bv(100));
    }

    #[test]
    fn test_simdgroup_reduce_min() {
        // 4-lane SIMD reduce min: min(100, 5, 42, 7) = 5
        let data = build_simd_array(&[100, 5, 42, 7], 32);
        let result = encode_simdgroup_reduce(
            &data,
            &|a, b| {
                let cond = a.clone().bvult(b.clone());
                SmtExpr::ite(cond, a.clone(), b.clone())
            },
            &SmtExpr::bv_const(0xFFFFFFFF, 32),
            4,
        );
        assert_eq!(eval_empty(&result), EvalResult::Bv(5));
    }

    #[test]
    fn test_simdgroup_broadcast() {
        // Broadcast lane 1 (value=20) to all 4 lanes
        let data = build_simd_array(&[10, 20, 30, 40], 32);
        let result = encode_simdgroup_broadcast(
            &data,
            &SmtExpr::bv_const(1, 32),
            4,
            &SmtSort::BitVec(32),
        );
        // All lanes should be 20
        assert_eq!(select_eval(&result, 0, 32), EvalResult::Bv(20));
        assert_eq!(select_eval(&result, 1, 32), EvalResult::Bv(20));
        assert_eq!(select_eval(&result, 2, 32), EvalResult::Bv(20));
        assert_eq!(select_eval(&result, 3, 32), EvalResult::Bv(20));
    }

    #[test]
    fn test_simdgroup_prefix_sum() {
        // prefix_sum([1, 2, 3, 4]) = [1, 3, 6, 10]
        let data = build_simd_array(&[1, 2, 3, 4], 32);
        let result = encode_simdgroup_prefix_sum(
            &data,
            4,
            &SmtSort::BitVec(32),
        );
        assert_eq!(select_eval(&result, 0, 32), EvalResult::Bv(1));
        assert_eq!(select_eval(&result, 1, 32), EvalResult::Bv(3));
        assert_eq!(select_eval(&result, 2, 32), EvalResult::Bv(6));
        assert_eq!(select_eval(&result, 3, 32), EvalResult::Bv(10));
    }

    // =======================================================================
    // UMA transfer tests
    // =======================================================================

    #[test]
    fn test_uma_transfer_identity() {
        let arr = build_array_bv32(&[42, 99]);
        let transferred = encode_uma_transfer(&arr);
        assert_eq!(select_eval(&transferred, 0, 64), EvalResult::Bv(42));
        assert_eq!(select_eval(&transferred, 1, 64), EvalResult::Bv(99));
    }

    // =======================================================================
    // Cross-check: map then reduce == map-reduce fused
    // =======================================================================

    #[test]
    fn test_map_then_reduce_equals_fused() {
        // Verify that map-then-reduce gives the same result as fused map-reduce.
        // Computation: sum(a[i] * b[i]) for i in 0..3
        let a = build_array_bv32(&[2, 3, 4]);
        let b = build_array_bv32(&[5, 6, 7]);

        // Approach 1: separate map then reduce
        let mapped = encode_parallel_map2(
            &a,
            &b,
            &|ai, bi| ai.clone().bvmul(bi.clone()),
            3,
            &SmtSort::BitVec(32),
        );
        let result_separate = encode_parallel_reduce(
            &mapped,
            &|acc, elem| acc.clone().bvadd(elem.clone()),
            &SmtExpr::bv_const(0, 32),
            3,
        );

        // Approach 2: fused map-reduce
        let result_fused = encode_map_reduce(
            &a,
            &b,
            &|ai, bi| ai.clone().bvmul(bi.clone()),
            &|acc, prod| acc.clone().bvadd(prod.clone()),
            &SmtExpr::bv_const(0, 32),
            3,
        );

        // 2*5 + 3*6 + 4*7 = 10 + 18 + 28 = 56
        assert_eq!(eval_empty(&result_separate), EvalResult::Bv(56));
        assert_eq!(eval_empty(&result_fused), EvalResult::Bv(56));
    }

    // =======================================================================
    // GpuReduceOp identity element tests
    // =======================================================================

    #[test]
    fn test_identity_add_is_neutral() {
        // 42 + 0 = 42
        let identity = GpuReduceOp::Add.identity_bv(32);
        let result = GpuReduceOp::Add.apply_bv(SmtExpr::bv_const(42, 32), identity);
        assert_eq!(eval_empty(&result), EvalResult::Bv(42));
    }

    #[test]
    fn test_identity_mul_is_neutral() {
        // 42 * 1 = 42
        let identity = GpuReduceOp::Mul.identity_bv(32);
        let result = GpuReduceOp::Mul.apply_bv(SmtExpr::bv_const(42, 32), identity);
        assert_eq!(eval_empty(&result), EvalResult::Bv(42));
    }

    #[test]
    fn test_identity_and_is_neutral() {
        // 42 & 0xFFFFFFFF = 42
        let identity = GpuReduceOp::BitwiseAnd.identity_bv(32);
        let result = GpuReduceOp::BitwiseAnd.apply_bv(SmtExpr::bv_const(42, 32), identity);
        assert_eq!(eval_empty(&result), EvalResult::Bv(42));
    }

    #[test]
    fn test_identity_or_is_neutral() {
        // 42 | 0 = 42
        let identity = GpuReduceOp::BitwiseOr.identity_bv(32);
        let result = GpuReduceOp::BitwiseOr.apply_bv(SmtExpr::bv_const(42, 32), identity);
        assert_eq!(eval_empty(&result), EvalResult::Bv(42));
    }

    #[test]
    fn test_identity_xor_is_neutral() {
        // 42 ^ 0 = 42
        let identity = GpuReduceOp::BitwiseXor.identity_bv(32);
        let result = GpuReduceOp::BitwiseXor.apply_bv(SmtExpr::bv_const(42, 32), identity);
        assert_eq!(eval_empty(&result), EvalResult::Bv(42));
    }

    // =======================================================================
    // End-to-end GPU dispatch test
    // =======================================================================

    #[test]
    fn test_full_gpu_dispatch_map() {
        // Simulate full GPU dispatch: transfer -> map -> transfer back
        let cpu_input = build_array_bv32(&[10, 20, 30]);

        // Step 1: Transfer to GPU (identity on UMA)
        let gpu_input = encode_uma_transfer(&cpu_input);

        // Step 2: GPU kernel (map: x -> x + 5)
        let gpu_output = encode_parallel_map(
            &gpu_input,
            &|elem| elem.clone().bvadd(SmtExpr::bv_const(5, 32)),
            3,
            &SmtSort::BitVec(32),
        );

        // Step 3: Transfer back (identity on UMA)
        let cpu_output = encode_uma_transfer(&gpu_output);

        assert_eq!(select_eval(&cpu_output, 0, 64), EvalResult::Bv(15));
        assert_eq!(select_eval(&cpu_output, 1, 64), EvalResult::Bv(25));
        assert_eq!(select_eval(&cpu_output, 2, 64), EvalResult::Bv(35));
    }
}
