// llvm2-lower/compute_graph.rs - Computation graph analysis for heterogeneous compute
//
// Author: Andrew Yates <ayates@dropbox.com>
// Copyright 2026 Dropbox, Inc. | License: Apache-2.0
//
// Reference: designs/2026-04-13-heterogeneous-compute.md (Computation Graph Analysis)
//
// Builds a computation graph from tMIR programs: a DAG of ComputeNodes
// connected by DataEdges. Each node represents a group of instructions
// that can be assigned to a compute target (CPU, SIMD, GPU, ANE). Edges
// carry data dependency and transfer cost information.
//
// Pattern detection identifies:
// - Data-parallel regions (map/reduce over arrays)
// - Matrix-heavy regions (nested loops with multiply-accumulate)
// - Sequential scalar ops (grouped into CPU nodes)

//! Computation graph analysis for heterogeneous compute allocation.
//!
//! This module implements Phase 1 of the heterogeneous compute pipeline:
//! building a computation graph from tMIR and identifying regions suitable
//! for different compute targets.
//!
//! # Architecture
//!
//! ```text
//! tMIR Module
//!     |
//!     v
//! GraphBuilder::build_from_module()
//!     |
//!     v
//! ComputeGraph { nodes: Vec<ComputeNode>, edges: Vec<DataEdge> }
//!     |
//!     v
//! partition_cost() -- evaluate a target assignment
//! ```

use std::collections::{HashMap, HashSet};
use std::fmt;

use serde::{Deserialize, Serialize};

use tmir_func::Module as TmirModule;
use tmir_instrs::{BinOp, Instr, InstrNode};
use tmir_types::{BlockId, Ty, ValueId};

use crate::instructions::Value;
use crate::target_analysis::{
    ComputeTarget, ProofAnalyzer, SubgraphDescriptor, SubgraphId, TargetLegality,
    TargetProofContext,
};

// ---------------------------------------------------------------------------
// Node and edge identifiers
// ---------------------------------------------------------------------------

/// Unique identifier for a node in the computation graph.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ComputeNodeId(pub u32);

impl fmt::Display for ComputeNodeId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "node_{}", self.0)
    }
}

/// Unique identifier for an edge in the computation graph.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct DataEdgeId(pub u32);

// ---------------------------------------------------------------------------
// Instruction identifier within tMIR
// ---------------------------------------------------------------------------

/// A reference to a tMIR instruction within the computation graph.
///
/// Identifies an instruction by its containing function, block, and
/// instruction index within that block.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct TmirInstId {
    /// Index of the function in the module.
    pub func_idx: u32,
    /// Block ID within the function.
    pub block_id: u32,
    /// Instruction index within the block body.
    pub inst_idx: u32,
}

// ---------------------------------------------------------------------------
// Cost types (cycle-count based for deterministic testing)
// ---------------------------------------------------------------------------

/// Estimated computation cost on a single target, measured in cycles.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ComputeCost {
    /// Estimated execution latency in cycles.
    pub latency_cycles: u64,
    /// Estimated throughput in operations per kilocycle.
    pub throughput_ops_per_kcycle: u64,
}

impl Default for ComputeCost {
    fn default() -> Self {
        Self {
            latency_cycles: 1,
            throughput_ops_per_kcycle: 1000,
        }
    }
}

/// Cost of transferring data between compute targets.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct TransferCost {
    /// Fixed overhead in cycles (e.g., kernel launch latency).
    pub overhead_cycles: u64,
    /// Per-byte transfer cost in nanocycles (cycles * 1e9 / byte).
    /// Use nanocycles to avoid floating point.
    pub per_byte_nanocycles: u64,
    /// Total estimated cost in cycles.
    pub total_cycles: u64,
}

impl TransferCost {
    /// Compute transfer cost for a given byte count.
    pub fn for_bytes(bytes: u64, overhead: u64, per_byte_nanocycles: u64) -> Self {
        let transfer_cycles = bytes.saturating_mul(per_byte_nanocycles) / 1_000_000_000;
        Self {
            overhead_cycles: overhead,
            per_byte_nanocycles,
            total_cycles: overhead.saturating_add(transfer_cycles),
        }
    }

    /// Zero cost (same-target transfer).
    pub fn zero() -> Self {
        Self {
            overhead_cycles: 0,
            per_byte_nanocycles: 0,
            total_cycles: 0,
        }
    }
}

// ---------------------------------------------------------------------------
// Node classification
// ---------------------------------------------------------------------------

/// Classification of a computation node's workload pattern.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum NodeKind {
    /// Sequential scalar operations (best on CPU scalar).
    Scalar,
    /// Data-parallel operations (map/reduce over arrays).
    /// Candidates for SIMD or GPU.
    DataParallel,
    /// Matrix-heavy operations (multiply-accumulate patterns).
    /// Candidates for GPU or Neural Engine.
    MatrixHeavy,
}

impl fmt::Display for NodeKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            NodeKind::Scalar => write!(f, "Scalar"),
            NodeKind::DataParallel => write!(f, "DataParallel"),
            NodeKind::MatrixHeavy => write!(f, "MatrixHeavy"),
        }
    }
}

// ---------------------------------------------------------------------------
// Core graph types
// ---------------------------------------------------------------------------

/// A node in the computation graph representing a group of tMIR instructions.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComputeNode {
    /// Unique node identifier.
    pub id: ComputeNodeId,
    /// The tMIR instructions belonging to this subgraph.
    pub instructions: Vec<TmirInstId>,
    /// Estimated cost on each legal compute target.
    pub costs: HashMap<ComputeTarget, ComputeCost>,
    /// Which compute targets can legally execute this node.
    pub legal_targets: Vec<ComputeTarget>,
    /// Workload classification.
    pub kind: NodeKind,
    /// Estimated data size in bytes processed by this node.
    pub data_size_bytes: u64,
    /// Values produced by this node (used for edge construction).
    pub produced_values: Vec<ValueId>,
    /// Values consumed by this node (used for edge construction).
    pub consumed_values: Vec<ValueId>,
    /// Full target legality analysis from ProofAnalyzer, including justifications,
    /// parallel reduction legality, and per-target judgments.
    /// `None` for manually constructed nodes that bypass proof analysis.
    #[serde(skip)]
    pub target_legality: Option<TargetLegality>,
}

/// A data dependency edge between two computation nodes.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataEdge {
    /// Source node (producer).
    pub from: ComputeNodeId,
    /// Destination node (consumer).
    pub to: ComputeNodeId,
    /// Number of bytes that must be transferred if nodes are on different targets.
    pub transfer_bytes: u64,
    /// Transfer cost estimate (populated based on target pair).
    pub transfer_cost: TransferCost,
}

/// The computation graph: a DAG of nodes connected by data dependency edges.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComputeGraph {
    /// Computation nodes (subgraphs).
    pub nodes: Vec<ComputeNode>,
    /// Data dependency edges.
    pub edges: Vec<DataEdge>,
}

impl ComputeGraph {
    /// Create an empty computation graph.
    pub fn new() -> Self {
        Self {
            nodes: Vec::new(),
            edges: Vec::new(),
        }
    }

    /// Get a node by ID.
    pub fn node(&self, id: ComputeNodeId) -> Option<&ComputeNode> {
        self.nodes.iter().find(|n| n.id == id)
    }

    /// Get all edges originating from a node.
    pub fn outgoing_edges(&self, node_id: ComputeNodeId) -> Vec<&DataEdge> {
        self.edges.iter().filter(|e| e.from == node_id).collect()
    }

    /// Get all edges targeting a node.
    pub fn incoming_edges(&self, node_id: ComputeNodeId) -> Vec<&DataEdge> {
        self.edges.iter().filter(|e| e.to == node_id).collect()
    }

    /// Compute the total cost of a target assignment (partition).
    ///
    /// Given a mapping from each node to a compute target, this returns
    /// the total cost = sum of compute costs + sum of transfer costs for
    /// edges between nodes on different targets.
    ///
    /// Returns `None` if any node is missing from the assignment or if
    /// the assigned target is not legal for that node.
    pub fn partition_cost(
        &self,
        assignment: &HashMap<ComputeNodeId, ComputeTarget>,
    ) -> Option<u64> {
        let mut total: u64 = 0;

        // Add compute costs for each node.
        for node in &self.nodes {
            let target = assignment.get(&node.id)?;
            if !node.legal_targets.contains(target) {
                return None; // Illegal assignment
            }
            let cost = node.costs.get(target)?;
            total = total.saturating_add(cost.latency_cycles);
        }

        // Add transfer costs for edges between different targets.
        for edge in &self.edges {
            let from_target = assignment.get(&edge.from)?;
            let to_target = assignment.get(&edge.to)?;
            if from_target != to_target {
                let transfer = estimate_transfer_cost(
                    edge.transfer_bytes,
                    *from_target,
                    *to_target,
                );
                total = total.saturating_add(transfer.total_cycles);
            }
        }

        Some(total)
    }

    /// Number of nodes.
    pub fn num_nodes(&self) -> usize {
        self.nodes.len()
    }

    /// Number of edges.
    pub fn num_edges(&self) -> usize {
        self.edges.len()
    }
}

impl Default for ComputeGraph {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Proof-guided target recommendations
// ---------------------------------------------------------------------------

/// Per-node target recommendation produced by proof-guided analysis.
///
/// Combines the [`TargetLegality`] from [`ProofAnalyzer`] with the cost model
/// to recommend the cheapest legal target for each computation node.
#[derive(Debug, Clone)]
pub struct TargetRecommendation {
    /// The node this recommendation applies to.
    pub node_id: ComputeNodeId,
    /// The recommended compute target (lowest cost among legal targets).
    pub recommended_target: ComputeTarget,
    /// All legal targets for this node.
    pub legal_targets: Vec<ComputeTarget>,
    /// Human-readable justification for the recommendation.
    pub reason: String,
    /// Whether parallel reduction is legal for this node's subgraph.
    pub parallel_reduction_legal: bool,
}

impl ComputeGraph {
    /// Build a graph from a tMIR module with a custom proof context and analyzer.
    ///
    /// This is the primary entry point for proof-guided graph construction.
    /// Each node's `target_legality` is populated with full [`TargetLegality`]
    /// from the analyzer, including justifications and parallel reduction info.
    pub fn with_proof_context(
        module: &TmirModule,
        proof_ctx: TargetProofContext,
        analyzer: &ProofAnalyzer,
    ) -> Self {
        let mut builder = GraphBuilder::new(analyzer.clone(), proof_ctx);
        builder.build_from_module(module)
    }

    /// Return per-node target recommendations using stored proof-guided legality.
    ///
    /// For each node, picks the cheapest legal target by `latency_cycles` from
    /// the node's cost map. Nodes without `target_legality` fall back to
    /// CpuScalar.
    pub fn target_recommendations(&self) -> Vec<TargetRecommendation> {
        self.nodes.iter().map(|node| {
            let legal = if let Some(ref legality) = node.target_legality {
                legality.legal_targets()
            } else {
                node.legal_targets.clone()
            };

            // Pick cheapest legal target by latency.
            let recommended = legal.iter()
                .filter_map(|t| node.costs.get(t).map(|c| (*t, c.latency_cycles)))
                .min_by_key(|(_, cost)| *cost)
                .map(|(t, _)| t)
                .unwrap_or(ComputeTarget::CpuScalar);

            let parallel_reduction_legal = node.target_legality
                .as_ref()
                .map(|l| l.parallel_reduction_legal)
                .unwrap_or(false);

            let reason = if let Some(ref legality) = node.target_legality {
                legality.reason(recommended)
                    .unwrap_or("no justification available")
                    .to_string()
            } else {
                format!("{} selected as cheapest legal target (no proof context)", recommended)
            };

            TargetRecommendation {
                node_id: node.id,
                recommended_target: recommended,
                legal_targets: legal,
                reason,
                parallel_reduction_legal,
            }
        }).collect()
    }

    /// Compute the total cost of the proof-guided optimal assignment.
    ///
    /// For each node, assigns the cheapest legal target. Then computes total
    /// cost including transfer costs for edges between nodes on different targets.
    /// Returns `None` if any node has no legal targets or no cost data.
    pub fn proof_guided_partition_cost(&self) -> Option<u64> {
        let mut assignment = HashMap::new();
        for node in &self.nodes {
            let legal = if let Some(ref legality) = node.target_legality {
                legality.legal_targets()
            } else {
                node.legal_targets.clone()
            };

            let best = legal.iter()
                .filter_map(|t| node.costs.get(t).map(|c| (*t, c.latency_cycles)))
                .min_by_key(|(_, cost)| *cost)
                .map(|(t, _)| t)?;

            assignment.insert(node.id, best);
        }
        self.partition_cost(&assignment)
    }

    /// Re-analyze all nodes with a new proof context, updating target legality
    /// and legal_targets on each node in-place.
    ///
    /// This is useful when proof annotations become available after initial graph
    /// construction (e.g., after a verification pass discovers new proofs).
    pub fn annotate_with_proofs(
        &mut self,
        module: &TmirModule,
        proof_ctx: &TargetProofContext,
        analyzer: &ProofAnalyzer,
    ) {
        for node in &mut self.nodes {
            let subgraph_id = SubgraphId(node.id.0);

            // Build a SubgraphDescriptor from the node.
            let mut desc = SubgraphDescriptor::new(subgraph_id);
            desc.data_size_bytes = node.data_size_bytes;

            // Propagate subgraph-level proofs from the proof context.
            desc.subgraph_proofs = proof_ctx.subgraph_proofs_for(subgraph_id);

            // Map node values into the descriptor.
            let all_values: Vec<ValueId> = node.consumed_values.iter()
                .chain(node.produced_values.iter())
                .copied()
                .collect();

            // Collect type information from the module for these values.
            let mut value_types_map: HashMap<ValueId, Ty> = HashMap::new();
            for func in &module.functions {
                for block in func.iter_blocks() {
                    for (vid, ty) in &block.params {
                        if all_values.contains(vid) {
                            value_types_map.insert(*vid, ty.clone());
                        }
                    }
                    for instr_node in &block.body {
                        match &instr_node.instr {
                            Instr::BinOp { ty, lhs, rhs, .. } => {
                                if all_values.contains(lhs) {
                                    value_types_map.entry(*lhs).or_insert_with(|| ty.clone());
                                }
                                if all_values.contains(rhs) {
                                    value_types_map.entry(*rhs).or_insert_with(|| ty.clone());
                                }
                                for r in &instr_node.results {
                                    if all_values.contains(r) {
                                        value_types_map.insert(*r, ty.clone());
                                    }
                                }
                            }
                            Instr::UnOp { ty, operand, .. } => {
                                if all_values.contains(operand) {
                                    value_types_map.entry(*operand).or_insert_with(|| ty.clone());
                                }
                                for r in &instr_node.results {
                                    if all_values.contains(r) {
                                        value_types_map.insert(*r, ty.clone());
                                    }
                                }
                            }
                            _ => {}
                        }
                    }
                }
            }

            // Map ValueIds -> internal Values for the analyzer.
            let mut lir_values = Vec::new();
            for (i, vid) in all_values.iter().enumerate() {
                let val = Value(i as u32);
                lir_values.push(val);
                if let Some(ty) = value_types_map.get(vid) {
                    if let Ok(lir_ty) = crate::adapter::translate_type(ty) {
                        desc.value_types.insert(val, lir_ty);
                    }
                }
            }
            desc.values = lir_values;

            // Run proof-guided analysis.
            let legality = analyzer.analyze(&desc, proof_ctx);
            node.legal_targets = legality.legal_targets();

            // Update costs for newly legal targets.
            for &target in &node.legal_targets {
                node.costs.entry(target).or_insert_with(|| {
                    estimate_compute_cost(
                        node.kind,
                        node.instructions.len(),
                        node.data_size_bytes,
                        target,
                    )
                });
            }

            node.target_legality = Some(legality);
        }
    }
}

// ---------------------------------------------------------------------------
// Transfer cost estimation
// ---------------------------------------------------------------------------

/// Estimate the cost of transferring data between two compute targets.
///
/// Cost model (Apple Silicon):
/// - CPU <-> SIMD: zero cost (same core, same address space)
/// - CPU <-> GPU: kernel launch overhead + DMA transfer
/// - CPU <-> ANE: model compilation overhead + DMA transfer
/// - GPU <-> ANE: must go through CPU (double transfer)
pub fn estimate_transfer_cost(
    bytes: u64,
    from: ComputeTarget,
    to: ComputeTarget,
) -> TransferCost {
    if from == to {
        return TransferCost::zero();
    }

    match (from, to) {
        // CPU <-> SIMD: same core, negligible cost
        (ComputeTarget::CpuScalar, ComputeTarget::CpuSimd)
        | (ComputeTarget::CpuSimd, ComputeTarget::CpuScalar) => {
            TransferCost::for_bytes(bytes, 0, 0)
        }

        // CPU <-> GPU: Metal command buffer overhead + shared memory transfer
        // Apple Silicon uses unified memory, but there's still cache coherency cost.
        // Overhead: ~5000 cycles for kernel launch
        // Per-byte: ~1 nanocycle (very fast due to unified memory)
        (ComputeTarget::CpuScalar, ComputeTarget::Gpu)
        | (ComputeTarget::CpuSimd, ComputeTarget::Gpu)
        | (ComputeTarget::Gpu, ComputeTarget::CpuScalar)
        | (ComputeTarget::Gpu, ComputeTarget::CpuSimd) => {
            TransferCost::for_bytes(bytes, 5000, 1_000_000_000)
        }

        // CPU <-> ANE: CoreML model compilation/load overhead + transfer
        // Overhead: ~50000 cycles for model load
        // Per-byte: ~2 nanocycles
        (ComputeTarget::CpuScalar, ComputeTarget::NeuralEngine)
        | (ComputeTarget::CpuSimd, ComputeTarget::NeuralEngine)
        | (ComputeTarget::NeuralEngine, ComputeTarget::CpuScalar)
        | (ComputeTarget::NeuralEngine, ComputeTarget::CpuSimd) => {
            TransferCost::for_bytes(bytes, 50000, 2_000_000_000)
        }

        // GPU <-> ANE: must transit through CPU (double hop)
        (ComputeTarget::Gpu, ComputeTarget::NeuralEngine)
        | (ComputeTarget::NeuralEngine, ComputeTarget::Gpu) => {
            let cpu_gpu = estimate_transfer_cost(bytes, ComputeTarget::Gpu, ComputeTarget::CpuScalar);
            let cpu_ane = estimate_transfer_cost(bytes, ComputeTarget::CpuScalar, ComputeTarget::NeuralEngine);
            TransferCost {
                overhead_cycles: cpu_gpu.overhead_cycles + cpu_ane.overhead_cycles,
                per_byte_nanocycles: cpu_gpu.per_byte_nanocycles + cpu_ane.per_byte_nanocycles,
                total_cycles: cpu_gpu.total_cycles + cpu_ane.total_cycles,
            }
        }

        // Same-target: unreachable due to early return above, but needed for exhaustiveness.
        (f, t) if f == t => TransferCost::zero(),

        // Fallback: should not be reachable with current variants.
        _ => TransferCost::zero(),
    }
}

// ---------------------------------------------------------------------------
// Per-target cost estimation
// ---------------------------------------------------------------------------

/// Estimate the compute cost for a node on a specific target.
///
/// This is a simplified cost model. Real costs would come from profiling
/// data or microarchitectural models.
fn estimate_compute_cost(
    kind: NodeKind,
    num_instructions: usize,
    data_size_bytes: u64,
    target: ComputeTarget,
) -> ComputeCost {
    let base_cycles = num_instructions as u64;

    match (kind, target) {
        // Scalar on CPU scalar: 1 cycle per instruction (baseline)
        (NodeKind::Scalar, ComputeTarget::CpuScalar) => ComputeCost {
            latency_cycles: base_cycles,
            throughput_ops_per_kcycle: 1000,
        },
        // Scalar on SIMD: slight overhead for vector setup
        (NodeKind::Scalar, ComputeTarget::CpuSimd) => ComputeCost {
            latency_cycles: base_cycles + 2,
            throughput_ops_per_kcycle: 800,
        },

        // Data-parallel on CPU scalar: N iterations
        (NodeKind::DataParallel, ComputeTarget::CpuScalar) => {
            let iterations = (data_size_bytes / 8).max(1); // assume 8-byte elements
            ComputeCost {
                latency_cycles: base_cycles * iterations,
                throughput_ops_per_kcycle: 1000,
            }
        }
        // Data-parallel on SIMD: 4x speedup (128-bit NEON)
        (NodeKind::DataParallel, ComputeTarget::CpuSimd) => {
            let iterations = (data_size_bytes / 8).max(1);
            ComputeCost {
                latency_cycles: base_cycles * iterations / 4,
                throughput_ops_per_kcycle: 4000,
            }
        }
        // Data-parallel on GPU: massive parallelism, but launch overhead
        (NodeKind::DataParallel, ComputeTarget::Gpu) => {
            let iterations = (data_size_bytes / 8).max(1);
            ComputeCost {
                latency_cycles: base_cycles + iterations / 64, // 64-wide warps
                throughput_ops_per_kcycle: 64000,
            }
        }
        // Data-parallel on ANE: not ideal (no reduce support)
        (NodeKind::DataParallel, ComputeTarget::NeuralEngine) => {
            let iterations = (data_size_bytes / 8).max(1);
            ComputeCost {
                latency_cycles: base_cycles * iterations / 8,
                throughput_ops_per_kcycle: 8000,
            }
        }

        // Matrix-heavy on CPU scalar: O(n^2) or O(n^3)
        (NodeKind::MatrixHeavy, ComputeTarget::CpuScalar) => {
            let elements = (data_size_bytes / 8).max(1);
            ComputeCost {
                latency_cycles: base_cycles * elements * elements,
                throughput_ops_per_kcycle: 1000,
            }
        }
        // Matrix-heavy on SIMD: partial vectorization
        (NodeKind::MatrixHeavy, ComputeTarget::CpuSimd) => {
            let elements = (data_size_bytes / 8).max(1);
            ComputeCost {
                latency_cycles: base_cycles * elements * elements / 4,
                throughput_ops_per_kcycle: 4000,
            }
        }
        // Matrix-heavy on GPU: excellent (GEMM kernels)
        (NodeKind::MatrixHeavy, ComputeTarget::Gpu) => {
            let elements = (data_size_bytes / 8).max(1);
            ComputeCost {
                latency_cycles: base_cycles + elements * elements / 256,
                throughput_ops_per_kcycle: 256000,
            }
        }
        // Matrix-heavy on ANE: best target (dedicated matrix multiply hardware)
        (NodeKind::MatrixHeavy, ComputeTarget::NeuralEngine) => {
            let elements = (data_size_bytes / 8).max(1);
            ComputeCost {
                latency_cycles: base_cycles + elements * elements / 512,
                throughput_ops_per_kcycle: 512000,
            }
        }

        // Scalar on GPU/ANE: never efficient but compute a cost anyway
        (NodeKind::Scalar, ComputeTarget::Gpu) => ComputeCost {
            latency_cycles: base_cycles + 5000, // launch overhead dominates
            throughput_ops_per_kcycle: 100,
        },
        (NodeKind::Scalar, ComputeTarget::NeuralEngine) => ComputeCost {
            latency_cycles: base_cycles + 50000,
            throughput_ops_per_kcycle: 10,
        },
    }
}

// ---------------------------------------------------------------------------
// Pattern detection
// ---------------------------------------------------------------------------

/// Check if a sequence of tMIR instructions represents a data-parallel pattern.
///
/// Heuristic: a block operating on array-typed values with element-wise
/// binary operations (FAdd, FMul, Add, Mul) is data-parallel.
fn detect_data_parallel(instrs: &[&InstrNode], value_types: &HashMap<ValueId, Ty>) -> bool {
    // Need at least one array-typed value
    let has_array = value_types.values().any(|ty| matches!(ty, Ty::Array(_, _)));
    if !has_array {
        return false;
    }

    // Check for element-wise operations on arrays
    let has_elementwise = instrs.iter().any(|node| match &node.instr {
        Instr::BinOp { op, .. } => matches!(
            op,
            BinOp::Add | BinOp::Mul | BinOp::FAdd | BinOp::FMul | BinOp::FSub | BinOp::Sub
        ),
        _ => false,
    });

    has_elementwise
}

/// Check if a sequence of tMIR instructions represents a matrix-heavy pattern.
///
/// Heuristic: multiply-accumulate pattern (FMul followed by FAdd) operating
/// on array data indicates matrix multiplication or similar.
fn detect_matrix_heavy(instrs: &[&InstrNode], value_types: &HashMap<ValueId, Ty>) -> bool {
    // Need array-typed values
    let has_array = value_types.values().any(|ty| matches!(ty, Ty::Array(_, _)));
    if !has_array {
        return false;
    }

    // Look for multiply-accumulate pattern: FMul + FAdd in sequence
    let mut has_fmul = false;
    let mut has_fadd_after_fmul = false;

    for node in instrs {
        match &node.instr {
            Instr::BinOp { op: BinOp::FMul, .. } | Instr::BinOp { op: BinOp::Mul, .. } => {
                has_fmul = true;
            }
            Instr::BinOp { op: BinOp::FAdd, .. } | Instr::BinOp { op: BinOp::Add, .. } => {
                if has_fmul {
                    has_fadd_after_fmul = true;
                }
            }
            _ => {}
        }
    }

    has_fadd_after_fmul
}

/// Classify a group of instructions into a NodeKind.
fn classify_node(instrs: &[&InstrNode], value_types: &HashMap<ValueId, Ty>) -> NodeKind {
    if detect_matrix_heavy(instrs, value_types) {
        NodeKind::MatrixHeavy
    } else if detect_data_parallel(instrs, value_types) {
        NodeKind::DataParallel
    } else {
        NodeKind::Scalar
    }
}

// ---------------------------------------------------------------------------
// Graph builder
// ---------------------------------------------------------------------------

/// Builds a ComputeGraph from a tMIR module.
///
/// The builder walks each function's blocks, groups instructions into
/// computation nodes, detects patterns (data-parallel, matrix-heavy),
/// and creates data dependency edges between nodes.
pub struct GraphBuilder {
    /// Proof analyzer for determining target legality.
    analyzer: ProofAnalyzer,
    /// Proof context for the module.
    proof_ctx: TargetProofContext,
    /// Next node ID.
    next_node_id: u32,
}

impl GraphBuilder {
    /// Create a new graph builder with the given proof analyzer and context.
    pub fn new(analyzer: ProofAnalyzer, proof_ctx: TargetProofContext) -> Self {
        Self {
            analyzer,
            proof_ctx,
            next_node_id: 0,
        }
    }

    /// Create a graph builder with default configuration.
    pub fn with_defaults() -> Self {
        Self::new(
            ProofAnalyzer::with_defaults(),
            TargetProofContext::default(),
        )
    }

    /// Allocate a fresh node ID.
    fn fresh_node_id(&mut self) -> ComputeNodeId {
        let id = ComputeNodeId(self.next_node_id);
        self.next_node_id += 1;
        id
    }

    /// Build a computation graph from a tMIR module.
    ///
    /// Each basic block becomes one or more computation nodes. Sequential
    /// scalar ops are grouped together. Data-parallel and matrix-heavy
    /// patterns are split into their own nodes.
    pub fn build_from_module(&mut self, module: &TmirModule) -> ComputeGraph {
        let mut graph = ComputeGraph::new();

        // Track which ValueId is produced by which node
        let mut value_to_node: HashMap<ValueId, ComputeNodeId> = HashMap::new();

        // Track which BlockId maps to which ComputeNodeId (for cross-block edges)
        let mut block_to_node: HashMap<(u32, u32), ComputeNodeId> = HashMap::new();

        for (func_idx, func) in module.functions.iter().enumerate() {
            // Collect type information for all values in the function
            let mut value_types: HashMap<ValueId, Ty> = HashMap::new();
            for block in func.iter_blocks() {
                for (vid, ty) in &block.params {
                    value_types.insert(*vid, ty.clone());
                }
                for node in &block.body {
                    // Infer types from instructions
                    match &node.instr {
                        Instr::BinOp { ty, lhs, rhs, .. } => {
                            value_types.entry(*lhs).or_insert_with(|| ty.clone());
                            value_types.entry(*rhs).or_insert_with(|| ty.clone());
                            for r in &node.results {
                                value_types.insert(*r, ty.clone());
                            }
                        }
                        Instr::UnOp { ty, operand, .. } => {
                            value_types.entry(*operand).or_insert_with(|| ty.clone());
                            for r in &node.results {
                                value_types.insert(*r, ty.clone());
                            }
                        }
                        Instr::Cmp { ty, lhs, rhs, .. } => {
                            value_types.entry(*lhs).or_insert_with(|| ty.clone());
                            value_types.entry(*rhs).or_insert_with(|| ty.clone());
                            for r in &node.results {
                                value_types.insert(*r, Ty::Bool);
                            }
                        }
                        Instr::Const { ty, .. } | Instr::FConst { ty, .. } => {
                            for r in &node.results {
                                value_types.insert(*r, ty.clone());
                            }
                        }
                        Instr::Load { ty, .. } => {
                            for r in &node.results {
                                value_types.insert(*r, ty.clone());
                            }
                        }
                        _ => {}
                    }
                }
            }

            for block in func.iter_blocks() {
                let nodes = self.build_nodes_for_block(
                    func_idx as u32,
                    block.id,
                    &block.body,
                    &value_types,
                );

                for node in nodes {
                    // Track value producers
                    for vid in &node.produced_values {
                        value_to_node.insert(*vid, node.id);
                    }
                    // Track block -> node mapping
                    block_to_node.insert((func_idx as u32, block.id.0), node.id);
                    graph.nodes.push(node);
                }
            }

            // Second pass: resolve branch-arg-to-block-param data flow.
            // When a Br instruction in block A passes args to block B's params,
            // the block B params are effectively "produced" by block A's node
            // (since the branch transfers the values).
            for block in func.iter_blocks() {
                let source_node_id = block_to_node
                    .get(&(func_idx as u32, block.id.0))
                    .copied();

                for node in &block.body {
                    let targets: Vec<(BlockId, &[ValueId])> = match &node.instr {
                        Instr::Br { target, args } => {
                            vec![(*target, args.as_slice())]
                        }
                        Instr::CondBr { then_target, then_args, else_target, else_args, .. } => {
                            vec![
                                (*then_target, then_args.as_slice()),
                                (*else_target, else_args.as_slice()),
                            ]
                        }
                        _ => vec![],
                    };

                    for (target_block_id, args) in targets {
                        if let Some(target_block) = func.block(target_block_id) {
                            // Map branch args -> target block params.
                            // The target block's params are produced by the source block.
                            for (arg_vid, (param_vid, _param_ty)) in
                                args.iter().zip(target_block.params.iter())
                            {
                                if let Some(src_node) = source_node_id {
                                    // The param is "produced" by the source node
                                    // (it flows through the branch)
                                    value_to_node.insert(*param_vid, src_node);
                                    // Also ensure the arg itself is tracked
                                    let _ = arg_vid; // already tracked as consumed
                                }
                            }
                        }
                    }
                }
            }
        }

        // Build edges based on data dependencies
        self.build_edges(&mut graph, &value_to_node);

        graph
    }

    /// Build computation nodes for a single basic block.
    ///
    /// Groups instructions into nodes based on pattern detection:
    /// - Consecutive scalar ops -> single Scalar node
    /// - Data-parallel patterns -> DataParallel node
    /// - Matrix-heavy patterns -> MatrixHeavy node
    fn build_nodes_for_block(
        &mut self,
        func_idx: u32,
        block_id: BlockId,
        body: &[InstrNode],
        value_types: &HashMap<ValueId, Ty>,
    ) -> Vec<ComputeNode> {
        if body.is_empty() {
            return Vec::new();
        }

        // Collect instruction references for pattern detection
        let instr_refs: Vec<&InstrNode> = body.iter().collect();

        // Try to detect if the whole block is a single pattern
        let kind = classify_node(&instr_refs, value_types);

        // For now, group entire block into one node (future: split into
        // multiple nodes when different patterns are detected within a block)
        let node_id = self.fresh_node_id();
        let subgraph_id = SubgraphId(node_id.0);

        // Collect TmirInstIds, produced/consumed values, and data size
        let mut instructions = Vec::new();
        let mut produced_values = Vec::new();
        let mut consumed_values = Vec::new();
        let mut data_size_bytes: u64 = 0;

        for (inst_idx, node) in body.iter().enumerate() {
            instructions.push(TmirInstId {
                func_idx,
                block_id: block_id.0,
                inst_idx: inst_idx as u32,
            });

            // Track produced values
            produced_values.extend_from_slice(&node.results);

            // Track consumed values and estimate data size
            match &node.instr {
                Instr::BinOp { ty, lhs, rhs, .. } => {
                    consumed_values.push(*lhs);
                    consumed_values.push(*rhs);
                    data_size_bytes += estimate_type_bytes(ty) as u64 * 2;
                }
                Instr::UnOp { ty, operand, .. } => {
                    consumed_values.push(*operand);
                    data_size_bytes += estimate_type_bytes(ty) as u64;
                }
                Instr::Cmp { lhs, rhs, .. } => {
                    consumed_values.push(*lhs);
                    consumed_values.push(*rhs);
                }
                Instr::Load { ptr, .. } => {
                    consumed_values.push(*ptr);
                }
                Instr::Store { ptr, value, .. } => {
                    consumed_values.push(*ptr);
                    consumed_values.push(*value);
                }
                Instr::Br { args, .. } => {
                    consumed_values.extend_from_slice(args);
                }
                Instr::CondBr { cond, then_args, else_args, .. } => {
                    consumed_values.push(*cond);
                    consumed_values.extend_from_slice(then_args);
                    consumed_values.extend_from_slice(else_args);
                }
                Instr::Return { values } => {
                    consumed_values.extend_from_slice(values);
                }
                Instr::Call { args, .. } => {
                    consumed_values.extend_from_slice(args);
                }
                Instr::Index { base, index, .. } => {
                    consumed_values.push(*base);
                    consumed_values.push(*index);
                }
                _ => {}
            }
        }

        // Build SubgraphDescriptor for target legality analysis
        let mut subgraph_desc = SubgraphDescriptor::new(subgraph_id);
        subgraph_desc.data_size_bytes = data_size_bytes;

        // Propagate subgraph-level proofs from the TargetProofContext into the
        // descriptor. This bridges Gap 1: proof annotations flow from the proof
        // context into each node's legality analysis.
        subgraph_desc.subgraph_proofs = self.proof_ctx.subgraph_proofs_for(subgraph_id);

        // Map ValueIds to internal Values for the analyzer
        let mut lir_values = Vec::new();
        for (i, vid) in consumed_values.iter().chain(produced_values.iter()).enumerate() {
            let val = Value(i as u32);
            lir_values.push(val);
            if let Some(ty) = value_types.get(vid) {
                if let Ok(lir_ty) = crate::adapter::translate_type(ty) {
                    subgraph_desc.value_types.insert(val, lir_ty);
                }
            }
        }
        subgraph_desc.values = lir_values;

        // Determine target legality via ProofAnalyzer
        let legality = self.analyzer.analyze(&subgraph_desc, &self.proof_ctx);
        let legal_targets = legality.legal_targets();

        // Estimate costs for each legal target
        let mut costs = HashMap::new();
        for &target in &legal_targets {
            let cost = estimate_compute_cost(
                kind,
                instructions.len(),
                data_size_bytes,
                target,
            );
            costs.insert(target, cost);
        }

        vec![ComputeNode {
            id: node_id,
            instructions,
            costs,
            legal_targets,
            kind,
            data_size_bytes,
            produced_values,
            consumed_values,
            target_legality: Some(legality),
        }]
    }

    /// Build data dependency edges between nodes.
    fn build_edges(
        &self,
        graph: &mut ComputeGraph,
        value_to_node: &HashMap<ValueId, ComputeNodeId>,
    ) {
        let mut seen_edges: HashSet<(ComputeNodeId, ComputeNodeId)> = HashSet::new();

        for node in &graph.nodes {
            for vid in &node.consumed_values {
                if let Some(&producer_id) = value_to_node.get(vid) {
                    if producer_id != node.id && !seen_edges.contains(&(producer_id, node.id)) {
                        seen_edges.insert((producer_id, node.id));

                        // Estimate transfer size: use the node's data size as approximation
                        let transfer_bytes = node.data_size_bytes.min(
                            graph.node(producer_id)
                                .map(|n| n.data_size_bytes)
                                .unwrap_or(0)
                        ).max(8); // minimum 8 bytes (one register)

                        graph.edges.push(DataEdge {
                            from: producer_id,
                            to: node.id,
                            transfer_bytes,
                            transfer_cost: TransferCost::zero(), // filled in by partition_cost
                        });
                    }
                }
            }
        }
    }
}

/// Estimate byte size of a tMIR type.
fn estimate_type_bytes(ty: &Ty) -> u32 {
    match ty {
        Ty::Bool => 1,
        Ty::Int(w) | Ty::UInt(w) | Ty::Float(w) => (*w as u32 + 7) / 8,
        Ty::Ptr(_) => 8,
        Ty::Array(elem, count) => estimate_type_bytes(elem) * (*count as u32),
        Ty::Struct(_) => 8, // rough estimate
        Ty::Void => 0,
        Ty::Func(_) => 8,
    }
}

// ---------------------------------------------------------------------------
// Convenience constructors for testing / manual graph building
// ---------------------------------------------------------------------------

impl ComputeGraph {
    /// Build a graph from a tMIR module with default configuration.
    pub fn from_module(module: &TmirModule) -> Self {
        let mut builder = GraphBuilder::with_defaults();
        builder.build_from_module(module)
    }

    /// Build a graph with a custom proof context.
    pub fn from_module_with_proofs(
        module: &TmirModule,
        proof_ctx: TargetProofContext,
    ) -> Self {
        let analyzer = ProofAnalyzer::with_defaults();
        let mut builder = GraphBuilder::new(analyzer, proof_ctx);
        builder.build_from_module(module)
    }

    /// Add a node manually (for testing).
    pub fn add_node(&mut self, node: ComputeNode) {
        self.nodes.push(node);
    }

    /// Add an edge manually (for testing).
    pub fn add_edge(&mut self, edge: DataEdge) {
        self.edges.push(edge);
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use tmir_func::{Block as TmirBlock, Function as TmirFunction, Module};
    use tmir_instrs::{BinOp, Instr, InstrNode};
    use tmir_types::{BlockId, FuncId, FuncTy, Ty, ValueId};

    // -------------------------------------------------------------------
    // Test helpers
    // -------------------------------------------------------------------

    /// Build a simple tMIR module: fn add(a: i32, b: i32) -> i32 { a + b }
    fn build_scalar_add_module() -> Module {
        Module {
            name: "scalar_add".to_string(),
            functions: vec![TmirFunction {
                id: FuncId(0),
                name: "add".to_string(),
                ty: FuncTy {
                    params: vec![Ty::Int(32), Ty::Int(32)],
                    returns: vec![Ty::Int(32)],
                },
                entry: BlockId(0),
                blocks: vec![TmirBlock {
                    id: BlockId(0),
                    params: vec![
                        (ValueId(0), Ty::Int(32)),
                        (ValueId(1), Ty::Int(32)),
                    ],
                    body: vec![
                        InstrNode {
                            instr: Instr::BinOp {
                                op: BinOp::Add,
                                ty: Ty::Int(32),
                                lhs: ValueId(0),
                                rhs: ValueId(1),
                            },
                            results: vec![ValueId(2)],
                        },
                        InstrNode {
                            instr: Instr::Return {
                                values: vec![ValueId(2)],
                            },
                            results: vec![],
                        },
                    ],
                }],
            }],
            structs: vec![],
        }
    }

    /// Build a tMIR module with array FAdd operations (data-parallel pattern).
    fn build_data_parallel_module() -> Module {
        Module {
            name: "data_parallel".to_string(),
            functions: vec![TmirFunction {
                id: FuncId(0),
                name: "vec_add".to_string(),
                ty: FuncTy {
                    params: vec![
                        Ty::Array(Box::new(Ty::Float(64)), 1000),
                        Ty::Array(Box::new(Ty::Float(64)), 1000),
                    ],
                    returns: vec![Ty::Float(64)],
                },
                entry: BlockId(0),
                blocks: vec![TmirBlock {
                    id: BlockId(0),
                    params: vec![
                        (ValueId(0), Ty::Array(Box::new(Ty::Float(64)), 1000)),
                        (ValueId(1), Ty::Array(Box::new(Ty::Float(64)), 1000)),
                    ],
                    body: vec![
                        InstrNode {
                            instr: Instr::BinOp {
                                op: BinOp::FAdd,
                                ty: Ty::Array(Box::new(Ty::Float(64)), 1000),
                                lhs: ValueId(0),
                                rhs: ValueId(1),
                            },
                            results: vec![ValueId(2)],
                        },
                        InstrNode {
                            instr: Instr::Return {
                                values: vec![ValueId(2)],
                            },
                            results: vec![],
                        },
                    ],
                }],
            }],
            structs: vec![],
        }
    }

    /// Build a tMIR module with FMul+FAdd pattern (matrix-heavy / MAC).
    fn build_matrix_heavy_module() -> Module {
        Module {
            name: "matrix_heavy".to_string(),
            functions: vec![TmirFunction {
                id: FuncId(0),
                name: "dot_product".to_string(),
                ty: FuncTy {
                    params: vec![
                        Ty::Array(Box::new(Ty::Float(64)), 1000),
                        Ty::Array(Box::new(Ty::Float(64)), 1000),
                    ],
                    returns: vec![Ty::Float(64)],
                },
                entry: BlockId(0),
                blocks: vec![TmirBlock {
                    id: BlockId(0),
                    params: vec![
                        (ValueId(0), Ty::Array(Box::new(Ty::Float(64)), 1000)),
                        (ValueId(1), Ty::Array(Box::new(Ty::Float(64)), 1000)),
                    ],
                    body: vec![
                        // FMul: multiply elements
                        InstrNode {
                            instr: Instr::BinOp {
                                op: BinOp::FMul,
                                ty: Ty::Array(Box::new(Ty::Float(64)), 1000),
                                lhs: ValueId(0),
                                rhs: ValueId(1),
                            },
                            results: vec![ValueId(2)],
                        },
                        // FAdd: accumulate (MAC pattern)
                        InstrNode {
                            instr: Instr::BinOp {
                                op: BinOp::FAdd,
                                ty: Ty::Float(64),
                                lhs: ValueId(2),
                                rhs: ValueId(2),
                            },
                            results: vec![ValueId(3)],
                        },
                        InstrNode {
                            instr: Instr::Return {
                                values: vec![ValueId(3)],
                            },
                            results: vec![],
                        },
                    ],
                }],
            }],
            structs: vec![],
        }
    }

    /// Build a module with two functions that have a data dependency
    /// (second function consumes values from the first).
    fn build_two_block_module() -> Module {
        Module {
            name: "two_block".to_string(),
            functions: vec![TmirFunction {
                id: FuncId(0),
                name: "two_blocks".to_string(),
                ty: FuncTy {
                    params: vec![Ty::Int(32), Ty::Int(32)],
                    returns: vec![Ty::Int(32)],
                },
                entry: BlockId(0),
                blocks: vec![
                    TmirBlock {
                        id: BlockId(0),
                        params: vec![
                            (ValueId(0), Ty::Int(32)),
                            (ValueId(1), Ty::Int(32)),
                        ],
                        body: vec![
                            InstrNode {
                                instr: Instr::BinOp {
                                    op: BinOp::Add,
                                    ty: Ty::Int(32),
                                    lhs: ValueId(0),
                                    rhs: ValueId(1),
                                },
                                results: vec![ValueId(2)],
                            },
                            InstrNode {
                                instr: Instr::Br {
                                    target: BlockId(1),
                                    args: vec![ValueId(2)],
                                },
                                results: vec![],
                            },
                        ],
                    },
                    TmirBlock {
                        id: BlockId(1),
                        params: vec![(ValueId(3), Ty::Int(32))],
                        body: vec![
                            InstrNode {
                                instr: Instr::BinOp {
                                    op: BinOp::Mul,
                                    ty: Ty::Int(32),
                                    lhs: ValueId(3),
                                    rhs: ValueId(3),
                                },
                                results: vec![ValueId(4)],
                            },
                            InstrNode {
                                instr: Instr::Return {
                                    values: vec![ValueId(4)],
                                },
                                results: vec![],
                            },
                        ],
                    },
                ],
            }],
            structs: vec![],
        }
    }

    // -------------------------------------------------------------------
    // Test: Scalar module produces a Scalar node
    // -------------------------------------------------------------------

    #[test]
    fn test_scalar_module_produces_scalar_node() {
        let module = build_scalar_add_module();
        let graph = ComputeGraph::from_module(&module);

        assert_eq!(graph.num_nodes(), 1);
        assert_eq!(graph.nodes[0].kind, NodeKind::Scalar);
        assert!(graph.nodes[0].legal_targets.contains(&ComputeTarget::CpuScalar));
    }

    // -------------------------------------------------------------------
    // Test: Data-parallel module detection
    // -------------------------------------------------------------------

    #[test]
    fn test_data_parallel_detection() {
        let module = build_data_parallel_module();
        let graph = ComputeGraph::from_module(&module);

        assert_eq!(graph.num_nodes(), 1);
        assert_eq!(graph.nodes[0].kind, NodeKind::DataParallel);
    }

    // -------------------------------------------------------------------
    // Test: Matrix-heavy module detection (FMul + FAdd pattern)
    // -------------------------------------------------------------------

    #[test]
    fn test_matrix_heavy_detection() {
        let module = build_matrix_heavy_module();
        let graph = ComputeGraph::from_module(&module);

        assert_eq!(graph.num_nodes(), 1);
        assert_eq!(graph.nodes[0].kind, NodeKind::MatrixHeavy);
    }

    // -------------------------------------------------------------------
    // Test: Two-block module produces edges
    // -------------------------------------------------------------------

    #[test]
    fn test_two_block_produces_edges() {
        let module = build_two_block_module();
        let graph = ComputeGraph::from_module(&module);

        // Two blocks -> two nodes
        assert_eq!(graph.num_nodes(), 2);

        // Block 1 consumes ValueId(2) produced by block 0 (via branch args)
        // So there should be an edge from node 0 to node 1
        assert!(graph.num_edges() >= 1, "Expected at least 1 edge, got {}", graph.num_edges());

        let edge = &graph.edges[0];
        assert_eq!(edge.from, ComputeNodeId(0));
        assert_eq!(edge.to, ComputeNodeId(1));
    }

    // -------------------------------------------------------------------
    // Test: Partition cost calculation — all on CPU scalar
    // -------------------------------------------------------------------

    #[test]
    fn test_partition_cost_all_cpu() {
        let module = build_scalar_add_module();
        let graph = ComputeGraph::from_module(&module);

        let mut assignment = HashMap::new();
        assignment.insert(ComputeNodeId(0), ComputeTarget::CpuScalar);

        let cost = graph.partition_cost(&assignment);
        assert!(cost.is_some(), "Cost should be computable for legal assignment");
        assert!(cost.unwrap() > 0, "Cost should be positive");
    }

    // -------------------------------------------------------------------
    // Test: Partition cost with illegal target returns None
    // -------------------------------------------------------------------

    #[test]
    fn test_partition_cost_illegal_target() {
        let module = build_scalar_add_module();
        let graph = ComputeGraph::from_module(&module);

        // Without proofs, GPU is not legal for scalar ops
        let mut assignment = HashMap::new();
        assignment.insert(ComputeNodeId(0), ComputeTarget::Gpu);

        let cost = graph.partition_cost(&assignment);
        assert!(cost.is_none(), "GPU should be illegal without proofs");
    }

    // -------------------------------------------------------------------
    // Test: Partition cost with missing node returns None
    // -------------------------------------------------------------------

    #[test]
    fn test_partition_cost_missing_node() {
        let module = build_scalar_add_module();
        let graph = ComputeGraph::from_module(&module);

        let assignment: HashMap<ComputeNodeId, ComputeTarget> = HashMap::new();
        let cost = graph.partition_cost(&assignment);
        assert!(cost.is_none(), "Missing node assignment should return None");
    }

    // -------------------------------------------------------------------
    // Test: Transfer cost between same targets is zero
    // -------------------------------------------------------------------

    #[test]
    fn test_transfer_cost_same_target_zero() {
        let cost = estimate_transfer_cost(1000, ComputeTarget::CpuScalar, ComputeTarget::CpuScalar);
        assert_eq!(cost.total_cycles, 0);
    }

    // -------------------------------------------------------------------
    // Test: Transfer cost CPU <-> SIMD is zero (same core)
    // -------------------------------------------------------------------

    #[test]
    fn test_transfer_cost_cpu_simd_zero() {
        let cost = estimate_transfer_cost(
            8000,
            ComputeTarget::CpuScalar,
            ComputeTarget::CpuSimd,
        );
        assert_eq!(cost.total_cycles, 0);
    }

    // -------------------------------------------------------------------
    // Test: Transfer cost CPU <-> GPU has overhead
    // -------------------------------------------------------------------

    #[test]
    fn test_transfer_cost_cpu_gpu_has_overhead() {
        let cost = estimate_transfer_cost(
            8000,
            ComputeTarget::CpuScalar,
            ComputeTarget::Gpu,
        );
        assert!(cost.total_cycles >= 5000, "GPU transfer should have launch overhead");
        assert!(cost.overhead_cycles == 5000);
    }

    // -------------------------------------------------------------------
    // Test: Transfer cost CPU <-> ANE has higher overhead than GPU
    // -------------------------------------------------------------------

    #[test]
    fn test_transfer_cost_cpu_ane_higher_than_gpu() {
        let gpu_cost = estimate_transfer_cost(8000, ComputeTarget::CpuScalar, ComputeTarget::Gpu);
        let ane_cost = estimate_transfer_cost(8000, ComputeTarget::CpuScalar, ComputeTarget::NeuralEngine);
        assert!(
            ane_cost.total_cycles > gpu_cost.total_cycles,
            "ANE transfer should cost more than GPU: ANE={}, GPU={}",
            ane_cost.total_cycles,
            gpu_cost.total_cycles,
        );
    }

    // -------------------------------------------------------------------
    // Test: GPU <-> ANE double-hop cost
    // -------------------------------------------------------------------

    #[test]
    fn test_transfer_cost_gpu_ane_double_hop() {
        let gpu_ane = estimate_transfer_cost(8000, ComputeTarget::Gpu, ComputeTarget::NeuralEngine);
        let gpu_cpu = estimate_transfer_cost(8000, ComputeTarget::Gpu, ComputeTarget::CpuScalar);
        let cpu_ane = estimate_transfer_cost(8000, ComputeTarget::CpuScalar, ComputeTarget::NeuralEngine);

        assert_eq!(
            gpu_ane.total_cycles,
            gpu_cpu.total_cycles + cpu_ane.total_cycles,
            "GPU<->ANE should be sum of GPU<->CPU + CPU<->ANE"
        );
    }

    // -------------------------------------------------------------------
    // Test: Two-block partition cost with mixed targets includes transfer
    // -------------------------------------------------------------------

    #[test]
    fn test_partition_cost_with_transfer() {
        let module = build_two_block_module();
        let graph = ComputeGraph::from_module(&module);

        // Both on CpuScalar: no transfer cost
        let mut all_cpu = HashMap::new();
        for node in &graph.nodes {
            all_cpu.insert(node.id, ComputeTarget::CpuScalar);
        }
        let cost_all_cpu = graph.partition_cost(&all_cpu).unwrap();

        // One on CpuScalar, one on CpuSimd: still no transfer cost (same core)
        let mut mixed_cpu_simd = HashMap::new();
        mixed_cpu_simd.insert(graph.nodes[0].id, ComputeTarget::CpuScalar);
        mixed_cpu_simd.insert(graph.nodes[1].id, ComputeTarget::CpuSimd);
        let cost_mixed = graph.partition_cost(&mixed_cpu_simd).unwrap();

        // Both should be computable
        assert!(cost_all_cpu > 0);
        assert!(cost_mixed > 0);
    }

    // -------------------------------------------------------------------
    // Test: ComputeNodeId display
    // -------------------------------------------------------------------

    #[test]
    fn test_compute_node_id_display() {
        assert_eq!(format!("{}", ComputeNodeId(42)), "node_42");
    }

    // -------------------------------------------------------------------
    // Test: NodeKind display
    // -------------------------------------------------------------------

    #[test]
    fn test_node_kind_display() {
        assert_eq!(format!("{}", NodeKind::Scalar), "Scalar");
        assert_eq!(format!("{}", NodeKind::DataParallel), "DataParallel");
        assert_eq!(format!("{}", NodeKind::MatrixHeavy), "MatrixHeavy");
    }

    // -------------------------------------------------------------------
    // Test: Empty module produces empty graph
    // -------------------------------------------------------------------

    #[test]
    fn test_empty_module_empty_graph() {
        let module = Module::new("empty");
        let graph = ComputeGraph::from_module(&module);
        assert_eq!(graph.num_nodes(), 0);
        assert_eq!(graph.num_edges(), 0);
    }

    // -------------------------------------------------------------------
    // Test: Partition cost of empty graph
    // -------------------------------------------------------------------

    #[test]
    fn test_empty_graph_partition_cost() {
        let graph = ComputeGraph::new();
        let assignment = HashMap::new();
        let cost = graph.partition_cost(&assignment);
        assert_eq!(cost, Some(0));
    }

    // -------------------------------------------------------------------
    // Test: TransferCost::zero
    // -------------------------------------------------------------------

    #[test]
    fn test_transfer_cost_zero() {
        let zero = TransferCost::zero();
        assert_eq!(zero.total_cycles, 0);
        assert_eq!(zero.overhead_cycles, 0);
        assert_eq!(zero.per_byte_nanocycles, 0);
    }

    // -------------------------------------------------------------------
    // Test: ComputeCost default
    // -------------------------------------------------------------------

    #[test]
    fn test_compute_cost_default() {
        let cost = ComputeCost::default();
        assert_eq!(cost.latency_cycles, 1);
        assert_eq!(cost.throughput_ops_per_kcycle, 1000);
    }

    // -------------------------------------------------------------------
    // Test: Graph outgoing/incoming edges
    // -------------------------------------------------------------------

    #[test]
    fn test_graph_edge_queries() {
        let module = build_two_block_module();
        let graph = ComputeGraph::from_module(&module);

        if graph.num_edges() > 0 {
            let outgoing = graph.outgoing_edges(ComputeNodeId(0));
            let incoming = graph.incoming_edges(ComputeNodeId(1));
            assert!(!outgoing.is_empty(), "Node 0 should have outgoing edges");
            assert!(!incoming.is_empty(), "Node 1 should have incoming edges");
        }
    }

    // -------------------------------------------------------------------
    // Test: Pattern detection helper - scalar instructions
    // -------------------------------------------------------------------

    #[test]
    fn test_detect_data_parallel_requires_arrays() {
        let instrs = vec![InstrNode {
            instr: Instr::BinOp {
                op: BinOp::Add,
                ty: Ty::Int(32),
                lhs: ValueId(0),
                rhs: ValueId(1),
            },
            results: vec![ValueId(2)],
        }];
        let refs: Vec<&InstrNode> = instrs.iter().collect();

        let mut types = HashMap::new();
        types.insert(ValueId(0), Ty::Int(32));
        types.insert(ValueId(1), Ty::Int(32));

        assert!(!detect_data_parallel(&refs, &types));
    }

    // -------------------------------------------------------------------
    // Test: Pattern detection helper - array FAdd is data-parallel
    // -------------------------------------------------------------------

    #[test]
    fn test_detect_data_parallel_with_arrays() {
        let instrs = vec![InstrNode {
            instr: Instr::BinOp {
                op: BinOp::FAdd,
                ty: Ty::Array(Box::new(Ty::Float(64)), 100),
                lhs: ValueId(0),
                rhs: ValueId(1),
            },
            results: vec![ValueId(2)],
        }];
        let refs: Vec<&InstrNode> = instrs.iter().collect();

        let mut types = HashMap::new();
        types.insert(ValueId(0), Ty::Array(Box::new(Ty::Float(64)), 100));
        types.insert(ValueId(1), Ty::Array(Box::new(Ty::Float(64)), 100));

        assert!(detect_data_parallel(&refs, &types));
    }

    // -------------------------------------------------------------------
    // Test: Pattern detection - MAC pattern requires both FMul and FAdd
    // -------------------------------------------------------------------

    #[test]
    fn test_detect_matrix_heavy_requires_mac() {
        // FMul alone is not matrix-heavy
        let instrs = vec![InstrNode {
            instr: Instr::BinOp {
                op: BinOp::FMul,
                ty: Ty::Array(Box::new(Ty::Float(64)), 100),
                lhs: ValueId(0),
                rhs: ValueId(1),
            },
            results: vec![ValueId(2)],
        }];
        let refs: Vec<&InstrNode> = instrs.iter().collect();

        let mut types = HashMap::new();
        types.insert(ValueId(0), Ty::Array(Box::new(Ty::Float(64)), 100));

        assert!(!detect_matrix_heavy(&refs, &types));
    }

    // -------------------------------------------------------------------
    // Test: classify_node prioritizes MatrixHeavy over DataParallel
    // -------------------------------------------------------------------

    #[test]
    fn test_classify_node_matrix_over_parallel() {
        let instrs = vec![
            InstrNode {
                instr: Instr::BinOp {
                    op: BinOp::FMul,
                    ty: Ty::Array(Box::new(Ty::Float(64)), 100),
                    lhs: ValueId(0),
                    rhs: ValueId(1),
                },
                results: vec![ValueId(2)],
            },
            InstrNode {
                instr: Instr::BinOp {
                    op: BinOp::FAdd,
                    ty: Ty::Float(64),
                    lhs: ValueId(2),
                    rhs: ValueId(2),
                },
                results: vec![ValueId(3)],
            },
        ];
        let refs: Vec<&InstrNode> = instrs.iter().collect();

        let mut types = HashMap::new();
        types.insert(ValueId(0), Ty::Array(Box::new(Ty::Float(64)), 100));
        types.insert(ValueId(1), Ty::Array(Box::new(Ty::Float(64)), 100));

        // MAC pattern on arrays -> MatrixHeavy takes priority
        assert_eq!(classify_node(&refs, &types), NodeKind::MatrixHeavy);
    }

    // -------------------------------------------------------------------
    // Test: estimate_type_bytes
    // -------------------------------------------------------------------

    #[test]
    fn test_estimate_type_bytes() {
        assert_eq!(estimate_type_bytes(&Ty::Bool), 1);
        assert_eq!(estimate_type_bytes(&Ty::Int(8)), 1);
        assert_eq!(estimate_type_bytes(&Ty::Int(16)), 2);
        assert_eq!(estimate_type_bytes(&Ty::Int(32)), 4);
        assert_eq!(estimate_type_bytes(&Ty::Int(64)), 8);
        assert_eq!(estimate_type_bytes(&Ty::Float(32)), 4);
        assert_eq!(estimate_type_bytes(&Ty::Float(64)), 8);
        assert_eq!(
            estimate_type_bytes(&Ty::Array(Box::new(Ty::Float(64)), 100)),
            800
        );
        assert_eq!(estimate_type_bytes(&Ty::Ptr(Box::new(Ty::Int(32)))), 8);
    }

    // -------------------------------------------------------------------
    // Test: ComputeGraph manual construction
    // -------------------------------------------------------------------

    #[test]
    fn test_manual_graph_construction() {
        let mut graph = ComputeGraph::new();

        let mut costs_a = HashMap::new();
        costs_a.insert(ComputeTarget::CpuScalar, ComputeCost {
            latency_cycles: 10,
            throughput_ops_per_kcycle: 1000,
        });
        costs_a.insert(ComputeTarget::CpuSimd, ComputeCost {
            latency_cycles: 5,
            throughput_ops_per_kcycle: 2000,
        });

        graph.add_node(ComputeNode {
            id: ComputeNodeId(0),
            instructions: vec![],
            costs: costs_a,
            legal_targets: vec![ComputeTarget::CpuScalar, ComputeTarget::CpuSimd],
            kind: NodeKind::DataParallel,
            data_size_bytes: 1000,
            produced_values: vec![],
            consumed_values: vec![],
            target_legality: None,
        });

        let mut costs_b = HashMap::new();
        costs_b.insert(ComputeTarget::CpuScalar, ComputeCost {
            latency_cycles: 20,
            throughput_ops_per_kcycle: 1000,
        });
        costs_b.insert(ComputeTarget::CpuSimd, ComputeCost {
            latency_cycles: 8,
            throughput_ops_per_kcycle: 2000,
        });

        graph.add_node(ComputeNode {
            id: ComputeNodeId(1),
            instructions: vec![],
            costs: costs_b,
            legal_targets: vec![ComputeTarget::CpuScalar, ComputeTarget::CpuSimd],
            kind: NodeKind::DataParallel,
            data_size_bytes: 2000,
            produced_values: vec![],
            consumed_values: vec![],
            target_legality: None,
        });

        graph.add_edge(DataEdge {
            from: ComputeNodeId(0),
            to: ComputeNodeId(1),
            transfer_bytes: 1000,
            transfer_cost: TransferCost::zero(),
        });

        assert_eq!(graph.num_nodes(), 2);
        assert_eq!(graph.num_edges(), 1);

        // All on CpuScalar: 10 + 20 = 30 (no transfer cost between CPU nodes)
        let mut all_cpu = HashMap::new();
        all_cpu.insert(ComputeNodeId(0), ComputeTarget::CpuScalar);
        all_cpu.insert(ComputeNodeId(1), ComputeTarget::CpuScalar);
        assert_eq!(graph.partition_cost(&all_cpu), Some(30));

        // All on CpuSimd: 5 + 8 = 13
        let mut all_simd = HashMap::new();
        all_simd.insert(ComputeNodeId(0), ComputeTarget::CpuSimd);
        all_simd.insert(ComputeNodeId(1), ComputeTarget::CpuSimd);
        assert_eq!(graph.partition_cost(&all_simd), Some(13));

        // Mixed CPU/SIMD: 10 + 8 = 18 + 0 transfer (CPU<->SIMD is zero)
        let mut mixed = HashMap::new();
        mixed.insert(ComputeNodeId(0), ComputeTarget::CpuScalar);
        mixed.insert(ComputeNodeId(1), ComputeTarget::CpuSimd);
        assert_eq!(graph.partition_cost(&mixed), Some(18));
    }

    // -------------------------------------------------------------------
    // Test: Cost estimation varies by node kind and target
    // -------------------------------------------------------------------

    #[test]
    fn test_cost_estimation_scalar_vs_parallel() {
        let scalar_cpu = estimate_compute_cost(NodeKind::Scalar, 10, 80, ComputeTarget::CpuScalar);
        let parallel_cpu = estimate_compute_cost(NodeKind::DataParallel, 10, 80, ComputeTarget::CpuScalar);

        // Data-parallel on CPU scalar is more expensive than scalar (iterates over data)
        assert!(
            parallel_cpu.latency_cycles >= scalar_cpu.latency_cycles,
            "DataParallel should cost >= Scalar on CPU: parallel={}, scalar={}",
            parallel_cpu.latency_cycles,
            scalar_cpu.latency_cycles,
        );
    }

    // -------------------------------------------------------------------
    // Test: GPU cost lower than CPU for data-parallel
    // -------------------------------------------------------------------

    #[test]
    fn test_gpu_cheaper_for_data_parallel() {
        let data_size = 8000; // large enough for GPU
        let n_instrs = 5;

        let cpu_cost = estimate_compute_cost(
            NodeKind::DataParallel, n_instrs, data_size, ComputeTarget::CpuScalar,
        );
        let gpu_cost = estimate_compute_cost(
            NodeKind::DataParallel, n_instrs, data_size, ComputeTarget::Gpu,
        );

        assert!(
            gpu_cost.latency_cycles < cpu_cost.latency_cycles,
            "GPU should be cheaper for data-parallel: GPU={}, CPU={}",
            gpu_cost.latency_cycles,
            cpu_cost.latency_cycles,
        );
    }

    // -------------------------------------------------------------------
    // Test: ANE cheapest for matrix-heavy
    // -------------------------------------------------------------------

    #[test]
    fn test_ane_cheapest_for_matrix_heavy() {
        let data_size = 8000;
        let n_instrs = 5;

        let cpu_cost = estimate_compute_cost(
            NodeKind::MatrixHeavy, n_instrs, data_size, ComputeTarget::CpuScalar,
        );
        let gpu_cost = estimate_compute_cost(
            NodeKind::MatrixHeavy, n_instrs, data_size, ComputeTarget::Gpu,
        );
        let ane_cost = estimate_compute_cost(
            NodeKind::MatrixHeavy, n_instrs, data_size, ComputeTarget::NeuralEngine,
        );

        assert!(
            ane_cost.latency_cycles <= gpu_cost.latency_cycles,
            "ANE should be <= GPU for matrix-heavy: ANE={}, GPU={}",
            ane_cost.latency_cycles,
            gpu_cost.latency_cycles,
        );
        assert!(
            gpu_cost.latency_cycles < cpu_cost.latency_cycles,
            "GPU should be < CPU for matrix-heavy: GPU={}, CPU={}",
            gpu_cost.latency_cycles,
            cpu_cost.latency_cycles,
        );
    }

    // -------------------------------------------------------------------
    // Test: Multiple functions in module produce separate nodes
    // -------------------------------------------------------------------

    #[test]
    fn test_multiple_functions_produce_nodes() {
        let module = Module {
            name: "multi".to_string(),
            functions: vec![
                TmirFunction {
                    id: FuncId(0),
                    name: "foo".to_string(),
                    ty: FuncTy {
                        params: vec![Ty::Int(32)],
                        returns: vec![Ty::Int(32)],
                    },
                    entry: BlockId(0),
                    blocks: vec![TmirBlock {
                        id: BlockId(0),
                        params: vec![(ValueId(0), Ty::Int(32))],
                        body: vec![InstrNode {
                            instr: Instr::Return { values: vec![ValueId(0)] },
                            results: vec![],
                        }],
                    }],
                },
                TmirFunction {
                    id: FuncId(1),
                    name: "bar".to_string(),
                    ty: FuncTy {
                        params: vec![Ty::Int(64)],
                        returns: vec![Ty::Int(64)],
                    },
                    entry: BlockId(0),
                    blocks: vec![TmirBlock {
                        id: BlockId(0),
                        params: vec![(ValueId(10), Ty::Int(64))],
                        body: vec![InstrNode {
                            instr: Instr::Return { values: vec![ValueId(10)] },
                            results: vec![],
                        }],
                    }],
                },
            ],
            structs: vec![],
        };

        let graph = ComputeGraph::from_module(&module);
        assert_eq!(graph.num_nodes(), 2, "Two functions should produce two nodes");
    }

    // ===================================================================
    // Proof-graph bridge tests (Gap 1: ProofAnalyzer <-> ComputeGraph)
    // ===================================================================

    // -------------------------------------------------------------------
    // Test helpers for proof-guided construction
    // -------------------------------------------------------------------

    use crate::adapter::{Proof, ProofContext};
    use crate::target_analysis::{
        SubgraphProof, TargetProofContext, SubgraphId,
    };
    use crate::instructions::Value;

    /// Build a TargetProofContext with Pure subgraph proof on node 0,
    /// plus InBounds+ValidBorrow on the first two values.
    fn full_proof_context() -> TargetProofContext {
        let mut proof_ctx = ProofContext::default();
        // Add InBounds + ValidBorrow proofs on LIR Values 0 and 1.
        // These correspond to the consumed_values mapped as Value(0), Value(1)
        // by the graph builder.
        for i in 0..2 {
            proof_ctx.value_proofs.insert(
                Value(i),
                vec![
                    Proof::InBounds {
                        base: tmir_types::ValueId(i as u32),
                        index: tmir_types::ValueId(i as u32 + 100),
                    },
                    Proof::ValidBorrow {
                        borrow: tmir_types::ValueId(i as u32),
                    },
                ],
            );
        }
        let mut ctx = TargetProofContext::new(proof_ctx);
        // Add Pure proof on subgraph 0 (corresponds to first node).
        ctx.add_subgraph_proof(SubgraphId(0), SubgraphProof::Pure);
        ctx
    }

    /// Build a TargetProofContext with full proofs for GPU+parallel reduction.
    fn full_gpu_proof_context() -> TargetProofContext {
        let mut ctx = full_proof_context();
        ctx.add_subgraph_proof(SubgraphId(0), SubgraphProof::Associative);
        ctx.add_subgraph_proof(SubgraphId(0), SubgraphProof::Commutative);
        ctx
    }

    // -------------------------------------------------------------------
    // Test: with_proof_context propagates Pure proof to unlock SIMD+GPU
    // -------------------------------------------------------------------

    #[test]
    fn test_with_proof_context_unlocks_gpu() {
        let module = build_data_parallel_module();
        let proof_ctx = full_proof_context();
        let analyzer = ProofAnalyzer::with_defaults();

        let graph = ComputeGraph::with_proof_context(&module, proof_ctx, &analyzer);

        assert_eq!(graph.num_nodes(), 1);
        let node = &graph.nodes[0];

        // With Pure + InBounds + ValidBorrow, GPU should be legal for
        // data-parallel array operations (data size is large enough).
        assert!(
            node.legal_targets.contains(&ComputeTarget::Gpu),
            "GPU should be legal with full proofs, got: {:?}",
            node.legal_targets
        );
        assert!(node.legal_targets.contains(&ComputeTarget::CpuScalar));
        assert!(node.legal_targets.contains(&ComputeTarget::CpuSimd));

        // target_legality should be populated.
        assert!(node.target_legality.is_some());
        let legality = node.target_legality.as_ref().unwrap();
        assert!(legality.is_legal(ComputeTarget::Gpu));
    }

    // -------------------------------------------------------------------
    // Test: without proofs, GPU is illegal
    // -------------------------------------------------------------------

    #[test]
    fn test_without_proofs_cpu_only() {
        let module = build_data_parallel_module();
        let graph = ComputeGraph::from_module(&module);

        assert_eq!(graph.num_nodes(), 1);
        let node = &graph.nodes[0];

        // Without proofs, side effects are not proven absent -> CPU/SIMD only.
        assert!(node.legal_targets.contains(&ComputeTarget::CpuScalar));
        assert!(node.legal_targets.contains(&ComputeTarget::CpuSimd));
        assert!(
            !node.legal_targets.contains(&ComputeTarget::Gpu),
            "GPU should be illegal without proofs"
        );
        assert!(
            !node.legal_targets.contains(&ComputeTarget::NeuralEngine),
            "ANE should be illegal without proofs"
        );
    }

    // -------------------------------------------------------------------
    // Test: target_recommendations picks cheapest legal target
    // -------------------------------------------------------------------

    #[test]
    fn test_target_recommendations_prefer_gpu() {
        let module = build_data_parallel_module();
        let proof_ctx = full_proof_context();
        let analyzer = ProofAnalyzer::with_defaults();

        let graph = ComputeGraph::with_proof_context(&module, proof_ctx, &analyzer);
        let recs = graph.target_recommendations();

        assert_eq!(recs.len(), 1);
        let rec = &recs[0];

        // For data-parallel workloads with full proofs, GPU should have lower
        // latency than CPU scalar (massive parallelism).
        assert!(
            rec.legal_targets.contains(&ComputeTarget::Gpu),
            "GPU should be in legal targets"
        );

        // The recommendation should be GPU because it has the lowest latency
        // for data-parallel workloads.
        assert_eq!(
            rec.recommended_target,
            ComputeTarget::Gpu,
            "Should recommend GPU for data-parallel with full proofs, got: {}",
            rec.recommended_target
        );
    }

    // -------------------------------------------------------------------
    // Test: target_recommendations without proofs -> CPU recommendation
    // -------------------------------------------------------------------

    #[test]
    fn test_target_recommendations_no_proofs_cpu() {
        let module = build_scalar_add_module();
        let graph = ComputeGraph::from_module(&module);
        let recs = graph.target_recommendations();

        assert_eq!(recs.len(), 1);
        let rec = &recs[0];

        // Without proofs, should recommend CpuScalar (cheapest for scalar ops).
        assert_eq!(rec.recommended_target, ComputeTarget::CpuScalar);
        assert!(!rec.parallel_reduction_legal);
        assert!(
            !rec.legal_targets.contains(&ComputeTarget::Gpu),
            "GPU should not be in legal targets without proofs"
        );
    }

    // -------------------------------------------------------------------
    // Test: proof_guided_partition_cost returns a valid cost
    // -------------------------------------------------------------------

    #[test]
    fn test_proof_guided_partition_cost() {
        let module = build_data_parallel_module();
        let proof_ctx = full_proof_context();
        let analyzer = ProofAnalyzer::with_defaults();

        let graph = ComputeGraph::with_proof_context(&module, proof_ctx, &analyzer);
        let cost = graph.proof_guided_partition_cost();

        assert!(cost.is_some(), "Should produce a valid partition cost");
        assert!(cost.unwrap() > 0, "Cost should be positive");
    }

    // -------------------------------------------------------------------
    // Test: proof_guided_partition_cost on empty graph
    // -------------------------------------------------------------------

    #[test]
    fn test_proof_guided_partition_cost_empty_graph() {
        let graph = ComputeGraph::new();
        let cost = graph.proof_guided_partition_cost();
        assert_eq!(cost, Some(0));
    }

    // -------------------------------------------------------------------
    // Test: subgraph proofs from TargetProofContext propagate to nodes
    // -------------------------------------------------------------------

    #[test]
    fn test_subgraph_proofs_propagate_to_nodes() {
        let module = build_data_parallel_module();
        let proof_ctx = full_gpu_proof_context();
        let analyzer = ProofAnalyzer::with_defaults();

        let graph = ComputeGraph::with_proof_context(&module, proof_ctx, &analyzer);

        let node = &graph.nodes[0];
        let legality = node.target_legality.as_ref().unwrap();

        // With Associative + Commutative proofs, parallel reduction should be legal.
        assert!(
            legality.parallel_reduction_legal,
            "Parallel reduction should be legal with Associative + Commutative proofs"
        );
    }

    // -------------------------------------------------------------------
    // Test: target_recommendations reports parallel reduction
    // -------------------------------------------------------------------

    #[test]
    fn test_recommendations_include_parallel_reduction() {
        let module = build_data_parallel_module();
        let proof_ctx = full_gpu_proof_context();
        let analyzer = ProofAnalyzer::with_defaults();

        let graph = ComputeGraph::with_proof_context(&module, proof_ctx, &analyzer);
        let recs = graph.target_recommendations();

        assert_eq!(recs.len(), 1);
        assert!(
            recs[0].parallel_reduction_legal,
            "Recommendation should report parallel reduction legal"
        );
    }

    // -------------------------------------------------------------------
    // Test: annotate_with_proofs upgrades node legality post-construction
    // -------------------------------------------------------------------

    #[test]
    fn test_annotate_with_proofs_upgrades_legality() {
        let module = build_data_parallel_module();

        // Build graph without proofs first.
        let mut graph = ComputeGraph::from_module(&module);
        assert!(
            !graph.nodes[0].legal_targets.contains(&ComputeTarget::Gpu),
            "GPU should be illegal before annotation"
        );

        // Now annotate with full proofs.
        let proof_ctx = full_proof_context();
        let analyzer = ProofAnalyzer::with_defaults();
        graph.annotate_with_proofs(&module, &proof_ctx, &analyzer);

        assert!(
            graph.nodes[0].legal_targets.contains(&ComputeTarget::Gpu),
            "GPU should be legal after annotation with proofs"
        );
        assert!(graph.nodes[0].target_legality.is_some());
    }

    // -------------------------------------------------------------------
    // Test: matrix-heavy with full proofs recommends GPU or ANE
    // -------------------------------------------------------------------

    #[test]
    fn test_matrix_heavy_with_proofs_recommends_accelerator() {
        let module = build_matrix_heavy_module();

        let mut proof_ctx = ProofContext::default();
        for i in 0..2 {
            proof_ctx.value_proofs.insert(
                Value(i),
                vec![
                    Proof::InBounds {
                        base: tmir_types::ValueId(i as u32),
                        index: tmir_types::ValueId(i as u32 + 100),
                    },
                    Proof::ValidBorrow {
                        borrow: tmir_types::ValueId(i as u32),
                    },
                ],
            );
        }
        let mut ctx = TargetProofContext::new(proof_ctx);
        ctx.add_subgraph_proof(SubgraphId(0), SubgraphProof::Pure);

        let analyzer = ProofAnalyzer::with_defaults();
        let graph = ComputeGraph::with_proof_context(&module, ctx, &analyzer);
        let recs = graph.target_recommendations();

        assert_eq!(recs.len(), 1);
        let rec = &recs[0];

        // Matrix-heavy: GPU or ANE should be recommended (not CPU scalar).
        assert!(
            rec.recommended_target == ComputeTarget::Gpu
                || rec.recommended_target == ComputeTarget::NeuralEngine,
            "Matrix-heavy with proofs should recommend GPU or ANE, got: {}",
            rec.recommended_target
        );
    }

    // -------------------------------------------------------------------
    // Test: from_module_with_proofs builds graph with proof context
    // -------------------------------------------------------------------

    #[test]
    fn test_from_module_with_proofs() {
        let module = build_data_parallel_module();
        let proof_ctx = full_proof_context();

        let graph = ComputeGraph::from_module_with_proofs(&module, proof_ctx);

        assert_eq!(graph.num_nodes(), 1);
        // from_module_with_proofs uses the default analyzer, but passes proofs.
        assert!(graph.nodes[0].target_legality.is_some());
    }

    // -------------------------------------------------------------------
    // Test: proof_guided_partition_cost < naive CPU-only cost
    // -------------------------------------------------------------------

    #[test]
    fn test_proof_guided_cost_beats_cpu_only() {
        let module = build_data_parallel_module();
        let proof_ctx = full_proof_context();
        let analyzer = ProofAnalyzer::with_defaults();

        let graph = ComputeGraph::with_proof_context(&module, proof_ctx, &analyzer);

        // Compute proof-guided cost (should pick GPU for data-parallel).
        let guided_cost = graph.proof_guided_partition_cost().unwrap();

        // Compute CPU-only cost.
        let mut cpu_assignment = HashMap::new();
        for node in &graph.nodes {
            cpu_assignment.insert(node.id, ComputeTarget::CpuScalar);
        }
        let cpu_cost = graph.partition_cost(&cpu_assignment).unwrap();

        assert!(
            guided_cost <= cpu_cost,
            "Proof-guided cost ({}) should be <= CPU-only cost ({})",
            guided_cost,
            cpu_cost
        );
    }

    // -------------------------------------------------------------------
    // Test: target_legality carries justification strings
    // -------------------------------------------------------------------

    #[test]
    fn test_legality_justification_strings() {
        let module = build_data_parallel_module();
        let proof_ctx = full_proof_context();
        let analyzer = ProofAnalyzer::with_defaults();

        let graph = ComputeGraph::with_proof_context(&module, proof_ctx, &analyzer);
        let node = &graph.nodes[0];
        let legality = node.target_legality.as_ref().unwrap();

        // GPU reason should mention proof-related justification.
        let gpu_reason = legality.reason(ComputeTarget::Gpu);
        assert!(
            gpu_reason.is_some(),
            "GPU should have a justification string"
        );
        let reason_text = gpu_reason.unwrap();
        assert!(
            reason_text.contains("Pure") || reason_text.contains("legal") || reason_text.contains("InBounds"),
            "GPU justification should reference proofs: {}",
            reason_text
        );
    }
}
