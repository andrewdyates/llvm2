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

use tmir::{Module as TmirModule, BinOp, Inst, InstrNode, BlockId, Ty, ValueId};
use crate::tmir_compat::Operand;

use crate::instructions::Value;
use crate::target_analysis::{
    ComputeTarget, ProofAnalyzer, SubgraphDescriptor, SubgraphId, TargetLegality,
    TargetProofContext,
};

use llvm2_ir::cost_model::{
    CostModelGen,
    ProfitabilityAnalyzer,
    ComputeTarget as CostComputeTarget,
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
    /// Dominant operation name (e.g., "ADD", "MUL", "GEMM") for cost model queries.
    /// Derived from the most common instruction in the node.
    pub dominant_op: String,
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
    /// Profitability analyzer for GPU/ANE dispatch decisions.
    /// Uses proper thresholds from the cost model instead of ad-hoc checks.
    #[serde(skip)]
    pub(crate) profitability: Option<ProfitabilityAnalyzer>,
}

impl ComputeGraph {
    /// Create an empty computation graph.
    pub fn new() -> Self {
        Self {
            nodes: Vec::new(),
            edges: Vec::new(),
            profitability: None,
        }
    }

    /// Create an empty computation graph with a profitability analyzer.
    pub fn new_with_profitability(generation: CostModelGen) -> Self {
        Self {
            nodes: Vec::new(),
            edges: Vec::new(),
            profitability: Some(ProfitabilityAnalyzer::new(generation)),
        }
    }

    /// Set the profitability analyzer for this graph.
    pub fn set_profitability(&mut self, analyzer: ProfitabilityAnalyzer) {
        self.profitability = Some(analyzer);
    }

    /// Returns true if this graph has a profitability analyzer attached.
    pub fn has_profitability(&self) -> bool {
        self.profitability.is_some()
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
// Target mapping: llvm2-lower ComputeTarget <-> llvm2-ir cost_model ComputeTarget
// ---------------------------------------------------------------------------

/// Map a `llvm2-lower` ComputeTarget to the `llvm2-ir::cost_model` ComputeTarget.
/// Returns `None` for targets with no direct cost model equivalent.
fn to_cost_target(target: ComputeTarget) -> Option<CostComputeTarget> {
    match target {
        ComputeTarget::CpuScalar => Some(CostComputeTarget::CpuScalar),
        ComputeTarget::CpuSimd => Some(CostComputeTarget::Neon),
        ComputeTarget::Gpu => Some(CostComputeTarget::Gpu),
        ComputeTarget::NeuralEngine => Some(CostComputeTarget::Ane),
    }
}

/// Convert a BinOp to the operation name string used by ProfitabilityAnalyzer.
fn binop_to_op_name(op: &BinOp) -> &'static str {
    match op {
        BinOp::Add => "ADD",
        BinOp::Sub => "SUB",
        BinOp::Mul => "MUL",
        BinOp::SDiv => "SDIV",
        BinOp::UDiv => "UDIV",
        BinOp::SRem => "SDIV", // closest equivalent
        BinOp::URem => "UDIV", // closest equivalent
        BinOp::And => "AND",
        BinOp::Or => "ORR",
        BinOp::Xor => "EOR",
        BinOp::Shl => "SHL",
        BinOp::AShr => "ASR",
        BinOp::LShr => "LSR",
        BinOp::FAdd => "FADD",
        BinOp::FSub => "FSUB",
        BinOp::FMul => "FMUL",
        BinOp::FDiv => "FDIV",
    }
}

/// Derive the dominant operation name from a block's instructions.
///
/// Examines BinOp instructions and returns the most frequent operation name.
/// Falls back to a kind-based heuristic if no BinOps are found.
fn derive_dominant_op(body: &[InstrNode], kind: NodeKind) -> String {
    let mut op_counts: HashMap<&'static str, usize> = HashMap::new();

    for node in body {
        if let Inst::BinOp { op, .. } = &node.inst {
            let name = binop_to_op_name(op);
            *op_counts.entry(name).or_insert(0) += 1;
        }
    }

    // Return the most frequent BinOp if any were found.
    if let Some((&name, _)) = op_counts.iter().max_by_key(|(_, count)| **count) {
        return name.to_string();
    }

    // Fallback based on NodeKind.
    match kind {
        NodeKind::MatrixHeavy => "GEMM".to_string(),
        NodeKind::DataParallel => "ADD".to_string(),
        NodeKind::Scalar => "ADD".to_string(),
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
    /// A default M1 ProfitabilityAnalyzer is attached for target dispatch.
    pub fn with_proof_context(
        module: &TmirModule,
        proof_ctx: TargetProofContext,
        analyzer: &ProofAnalyzer,
    ) -> Self {
        let mut builder = GraphBuilder::new(analyzer.clone(), proof_ctx);
        let mut graph = builder.build_from_module(module);
        graph.profitability = Some(ProfitabilityAnalyzer::new(CostModelGen::M1));
        graph
    }

    /// Build a graph from a tMIR module with proof context and a custom
    /// ProfitabilityAnalyzer for GPU/ANE dispatch thresholds.
    pub fn with_profitability(
        module: &TmirModule,
        proof_ctx: TargetProofContext,
        analyzer: &ProofAnalyzer,
        profitability: ProfitabilityAnalyzer,
    ) -> Self {
        let mut builder = GraphBuilder::new(analyzer.clone(), proof_ctx);
        let mut graph = builder.build_from_module(module);
        graph.profitability = Some(profitability);
        graph
    }

    /// Return per-node target recommendations using stored proof-guided legality
    /// and profitability analysis.
    ///
    /// For each node, filters legal targets through the [`ProfitabilityAnalyzer`]
    /// (when available) to exclude targets that are legal but unprofitable for
    /// the node's workload size. Then picks the cheapest remaining target by
    /// `latency_cycles`. Nodes without `target_legality` fall back to CpuScalar.
    ///
    /// When no `ProfitabilityAnalyzer` is set, falls back to the previous
    /// behavior of picking the cheapest legal target without profitability
    /// filtering.
    pub fn target_recommendations(&self) -> Vec<TargetRecommendation> {
        self.nodes.iter().map(|node| {
            let legal = if let Some(ref legality) = node.target_legality {
                legality.legal_targets()
            } else {
                node.legal_targets.clone()
            };

            // Filter legal targets through ProfitabilityAnalyzer when available.
            // GPU and ANE have dispatch overhead that makes them unprofitable for
            // small workloads. ProfitabilityAnalyzer encodes these thresholds.
            let profitable = if let Some(ref pa) = self.profitability {
                let op = &node.dominant_op;
                let data_size = node.data_size_bytes;
                // Approximate element count: assume 4-byte elements as default.
                let element_count = (data_size / 4).max(1);
                // Approximate tensor shape as a flat array for profitability checks.
                let tensor_shape = [element_count];

                legal.iter().copied().filter(|&target| {
                    match target {
                        ComputeTarget::Gpu => {
                            if let Some(cost_target) = to_cost_target(target) {
                                pa.target_legality(op, cost_target)
                                    && pa.is_gpu_profitable(op, data_size, element_count)
                            } else {
                                false
                            }
                        }
                        ComputeTarget::NeuralEngine => {
                            if let Some(cost_target) = to_cost_target(target) {
                                pa.target_legality(op, cost_target)
                                    && pa.is_ane_profitable(op, data_size, &tensor_shape)
                            } else {
                                false
                            }
                        }
                        // CPU Scalar and SIMD are always considered profitable.
                        _ => true,
                    }
                }).collect::<Vec<_>>()
            } else {
                legal.clone()
            };

            // If profitability filtering removed all targets, fall back to
            // the original legal set (CPU targets will still be there).
            let effective = if profitable.is_empty() { &legal } else { &profitable };

            // Pick cheapest legal+profitable target by latency.
            let recommended = effective.iter()
                .filter_map(|t| node.costs.get(t).map(|c| (*t, c.latency_cycles)))
                .min_by_key(|(_, cost)| *cost)
                .map(|(t, _)| t)
                .unwrap_or(ComputeTarget::CpuScalar);

            let parallel_reduction_legal = node.target_legality
                .as_ref()
                .map(|l| l.parallel_reduction_legal)
                .unwrap_or(false);

            let reason = if let Some(ref legality) = node.target_legality {
                let base = legality.reason(recommended)
                    .unwrap_or("no justification available");
                if self.profitability.is_some() && !profitable.contains(&recommended) {
                    format!("{} (profitability-filtered fallback)", base)
                } else if self.profitability.is_some() {
                    format!("{} (profitability-verified)", base)
                } else {
                    base.to_string()
                }
            } else {
                format!("{} selected as cheapest legal target (no proof context)", recommended)
            };

            TargetRecommendation {
                node_id: node.id,
                recommended_target: recommended,
                legal_targets: effective.to_vec(),
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
                        match &instr_node.inst {
                            Inst::BinOp { ty, lhs, rhs, .. } => {
                                if let Some(lhs_vid) = lhs.as_value() {
                                    if all_values.contains(&lhs_vid) {
                                        value_types_map.entry(lhs_vid).or_insert_with(|| ty.clone());
                                    }
                                }
                                if let Some(rhs_vid) = rhs.as_value() {
                                    if all_values.contains(&rhs_vid) {
                                        value_types_map.entry(rhs_vid).or_insert_with(|| ty.clone());
                                    }
                                }
                                for r in &instr_node.results {
                                    if all_values.contains(r) {
                                        value_types_map.insert(*r, ty.clone());
                                    }
                                }
                            }
                            Inst::UnOp { ty, operand, .. } => {
                                if let Some(op_vid) = operand.as_value() {
                                    if all_values.contains(&op_vid) {
                                        value_types_map.entry(op_vid).or_insert_with(|| ty.clone());
                                    }
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
                if let Some(ty) = value_types_map.get(vid)
                    && let Ok(lir_ty) = crate::adapter::translate_type(ty) {
                        desc.value_types.insert(val, lir_ty);
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
                        node.instuctions.len(),
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

/// Minimum array element count to consider a pattern suitable for SIMD/GPU dispatch.
/// Below this threshold, the overhead of vectorized or GPU dispatch exceeds
/// the benefit, so we classify the region as scalar even if it operates on arrays.
const MIN_VECTORIZABLE_ELEMENTS: u64 = 4;

/// Check if a sequence of tMIR instructions represents a data-parallel pattern.
///
/// Heuristic: a block operating on array-typed values with element-wise
/// binary operations (FAdd, FMul, Add, Mul) is data-parallel, provided:
/// 1. At least one array-typed value exists
/// 2. The array has enough elements to justify vectorization
/// 3. Element-wise binary operations are present
/// 4. The operations actually consume array-typed operands (not just
///    happening to coexist with array parameters)
/// 5. No loop-carried dependency pattern is detected (an operation whose
///    result feeds back into itself via a consumed value)
fn detect_data_parallel(instrs: &[&InstrNode], value_types: &HashMap<ValueId, Ty>) -> bool {
    // Need at least one array-typed value with sufficient element count
    let has_large_array = value_types.values().any(|ty| match ty {
        Ty::Array { len, .. } => *len >= MIN_VECTORIZABLE_ELEMENTS,
        _ => false,
    });
    if !has_large_array {
        return false;
    }

    // Check for element-wise operations whose operands are actually array-typed.
    // This prevents false positives where scalar ops coexist with array parameters
    // but don't actually operate on the arrays.
    let has_array_elementwise = instrs.iter().any(|node| match &node.inst {
        Inst::BinOp { op, lhs, rhs, .. } => {
            let is_elementwise = matches!(
                op,
                BinOp::Add | BinOp::Mul | BinOp::FAdd | BinOp::FMul | BinOp::FSub | BinOp::Sub
            );
            if !is_elementwise {
                return false;
            }
            // At least one operand must be array-typed (skip constants)
            let lhs_is_array = lhs.as_value().and_then(|v| value_types.get(&v)).is_some_and(|ty| matches!(ty, Ty::Array { .. }));
            let rhs_is_array = rhs.as_value().and_then(|v| value_types.get(&v)).is_some_and(|ty| matches!(ty, Ty::Array { .. }));
            lhs_is_array || rhs_is_array
        }
        _ => false,
    });
    if !has_array_elementwise {
        return false;
    }

    // Check for loop-carried dependencies: if an operation's result is also
    // one of its own operands (via consumed values of other instructions),
    // this indicates a reduction or accumulation that may not be parallelizable
    // without associativity proofs. We allow the data-parallel classification
    // but flag it conservatively -- the ProofAnalyzer will gate actual GPU
    // dispatch on associativity/commutativity proofs.
    //
    // Note: we do NOT reject here because data-parallel with reductions is
    // still a valid pattern (e.g., map-reduce). The target legality analysis
    // handles the reduction legality separately.

    has_array_elementwise
}

/// Check if a sequence of tMIR instructions represents a matrix-heavy pattern.
///
/// Heuristic: multiply-accumulate pattern (FMul followed by FAdd) operating
/// on array data indicates matrix multiplication or similar, provided:
/// 1. Array-typed values exist with sufficient element count
/// 2. The multiply and accumulate operations have a data dependency
///    (the FAdd consumes the FMul result, not independent values)
/// 3. At least one operand of the multiply is array-typed
fn detect_matrix_heavy(instrs: &[&InstrNode], value_types: &HashMap<ValueId, Ty>) -> bool {
    // Need array-typed values with sufficient element count
    let has_large_array = value_types.values().any(|ty| match ty {
        Ty::Array { len, .. } => *len >= MIN_VECTORIZABLE_ELEMENTS,
        _ => false,
    });
    if !has_large_array {
        return false;
    }

    // Look for multiply-accumulate pattern with data dependency validation:
    // FMul/Mul must produce a result that is consumed by a subsequent FAdd/Add.
    // Additionally, at least one operand of the multiply must be array-typed.
    let mut mul_results: HashSet<ValueId> = HashSet::new();
    let mut has_mac_pattern = false;

    for node in instrs {
        match &node.inst {
            Inst::BinOp { op, lhs, rhs, .. }
                if matches!(op, BinOp::FMul | BinOp::Mul) =>
            {
                // Check that at least one operand is array-typed (skip constants)
                let lhs_is_array = lhs.as_value().and_then(|v| value_types.get(&v)).is_some_and(|ty| matches!(ty, Ty::Array { .. }));
                let rhs_is_array = rhs.as_value().and_then(|v| value_types.get(&v)).is_some_and(|ty| matches!(ty, Ty::Array { .. }));
                if lhs_is_array || rhs_is_array {
                    for r in &node.results {
                        mul_results.insert(*r);
                    }
                }
            }
            Inst::BinOp { op, lhs, rhs, .. }
                if matches!(op, BinOp::FAdd | BinOp::Add) =>
            {
                // The accumulate must consume a multiply result (skip constants)
                let lhs_match = lhs.as_value().is_some_and(|v| mul_results.contains(&v));
                let rhs_match = rhs.as_value().is_some_and(|v| mul_results.contains(&v));
                if lhs_match || rhs_match {
                    has_mac_pattern = true;
                }
            }
            _ => {}
        }
    }

    has_mac_pattern
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
                    // Infer types from instructions (skip constant operands)
                    match &node.inst {
                        Inst::BinOp { ty, lhs, rhs, .. } => {
                            if let Some(v) = lhs.as_value() { value_types.entry(v).or_insert_with(|| ty.clone()); }
                            if let Some(v) = rhs.as_value() { value_types.entry(v).or_insert_with(|| ty.clone()); }
                            for r in &node.results {
                                value_types.insert(*r, ty.clone());
                            }
                        }
                        Inst::UnOp { ty, operand, .. } => {
                            if let Some(v) = operand.as_value() { value_types.entry(v).or_insert_with(|| ty.clone()); }
                            for r in &node.results {
                                value_types.insert(*r, ty.clone());
                            }
                        }
                        Inst::Cmp { ty, lhs, rhs, .. } => {
                            if let Some(v) = lhs.as_value() { value_types.entry(v).or_insert_with(|| ty.clone()); }
                            if let Some(v) = rhs.as_value() { value_types.entry(v).or_insert_with(|| ty.clone()); }
                            for r in &node.results {
                                value_types.insert(*r, Ty::Bool);
                            }
                        }
                        Inst::Const { ty, .. } | Inst::FConst { ty, .. } => {
                            for r in &node.results {
                                value_types.insert(*r, ty.clone());
                            }
                        }
                        Inst::Load { ty, .. } => {
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
                    let targets: Vec<(BlockId, &[Operand])> = match &node.inst {
                        Inst::Br { target, args } => {
                            vec![(*target, args.as_slice())]
                        }
                        Inst::CondBr { then_target, then_args, else_target, else_args, .. } => {
                            vec![
                                (*then_target, then_args.as_slice()),
                                (*else_target, else_args.as_slice()),
                            ]
                        }
                        Inst::Invoke { normal_dest, normal_args, unwind_dest, unwind_args, .. } => {
                            vec![
                                (*normal_dest, normal_args.as_slice()),
                                (*unwind_dest, unwind_args.as_slice()),
                            ]
                        }
                        _ => vec![],
                    };

                    for (target_block_id, args) in targets {
                        if let Some(target_block) = func.block(target_block_id) {
                            // Map branch args -> target block params.
                            // The target block's params are produced by the source block.
                            for (arg_op, (param_vid, _param_ty)) in
                                args.iter().zip(target_block.params.iter())
                            {
                                if let Some(src_node) = source_node_id {
                                    // The param is "produced" by the source node
                                    // (it flows through the branch)
                                    value_to_node.insert(*param_vid, src_node);
                                    // Also ensure the arg itself is tracked
                                    let _ = arg_op; // already tracked as consumed
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

            // Track consumed values and estimate data size.
            // Operand::Constant values are skipped (they don't have ValueIds).
            let push_operand = |cv: &mut Vec<ValueId>, op: &Operand| {
                if let Some(vid) = op.as_value() {
                    cv.push(vid);
                }
            };
            let push_operands = |cv: &mut Vec<ValueId>, ops: &[Operand]| {
                for op in ops {
                    if let Some(vid) = op.as_value() {
                        cv.push(vid);
                    }
                }
            };
            match &node.inst {
                Inst::BinOp { ty, lhs, rhs, .. } => {
                    push_operand(&mut consumed_values, lhs);
                    push_operand(&mut consumed_values, rhs);
                    data_size_bytes += estimate_type_bytes(ty) as u64 * 2;
                }
                Inst::UnOp { ty, operand, .. } => {
                    push_operand(&mut consumed_values, operand);
                    data_size_bytes += estimate_type_bytes(ty) as u64;
                }
                Inst::Cmp { lhs, rhs, .. } => {
                    push_operand(&mut consumed_values, lhs);
                    push_operand(&mut consumed_values, rhs);
                }
                Inst::Load { ptr, .. } => {
                    consumed_values.push(*ptr);
                }
                Inst::Store { ptr, value, .. } => {
                    consumed_values.push(*ptr);
                    push_operand(&mut consumed_values, value);
                }
                Inst::Br { args, .. } => {
                    push_operands(&mut consumed_values, args);
                }
                Inst::CondBr { cond, then_args, else_args, .. } => {
                    push_operand(&mut consumed_values, cond);
                    push_operands(&mut consumed_values, then_args);
                    push_operands(&mut consumed_values, else_args);
                }
                Inst::Return { values } => {
                    push_operands(&mut consumed_values, values);
                }
                Inst::Call { args, .. } => {
                    push_operands(&mut consumed_values, args);
                }
                Inst::Index { base, index, .. } => {
                    consumed_values.push(*base);
                    push_operand(&mut consumed_values, index);
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
            if let Some(ty) = value_types.get(vid)
                && let Ok(lir_ty) = crate::adapter::translate_type(ty) {
                    subgraph_desc.value_types.insert(val, lir_ty);
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

        // Derive dominant operation name from instructions for profitability queries.
        let dominant_op = derive_dominant_op(body, kind);

        vec![ComputeNode {
            id: node_id,
            instructions,
            costs,
            legal_targets,
            kind,
            data_size_bytes,
            produced_values,
            consumed_values,
            dominant_op,
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
                if let Some(&producer_id) = value_to_node.get(vid)
                    && producer_id != node.id && !seen_edges.contains(&(producer_id, node.id)) {
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

/// Estimate byte size of a tMIR type.
fn estimate_type_bytes(ty: &Ty) -> u32 {
    match ty {
        Ty::Primitive(p) => match p {
            PrimitiveType::Bool => 1,
            PrimitiveType::Int { width, .. } => (width.bits() as u32).div_ceil(8),
            PrimitiveType::Float(fw) => (fw.bits() as u32).div_ceil(8),
            PrimitiveType::Unit => 0,
            PrimitiveType::Never => 0,
        },
        Ty::Ref { .. } => 8,
        Ty::Array { element, len } => estimate_type_bytes(element) * (*len as u32),
        Ty::Vector { element, lanes } => estimate_type_bytes(element) * lanes,
        Ty::Struct(_) | Ty::StructDef { .. } | Ty::Tuple(_) => 8, // rough estimate
        Ty::Enum { .. } => 8, // rough estimate
        Ty::FnPtr { .. } | Ty::Func(_) => 8,
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
    /// Attaches a default M1 ProfitabilityAnalyzer for GPU/ANE dispatch.
    pub fn from_module_with_proofs(
        module: &TmirModule,
        proof_ctx: TargetProofContext,
    ) -> Self {
        let analyzer = ProofAnalyzer::with_defaults();
        let mut builder = GraphBuilder::new(analyzer, proof_ctx);
        let mut graph = builder.build_from_module(module);
        graph.profitability = Some(ProfitabilityAnalyzer::new(CostModelGen::M1));
        graph
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
    use tmir_instrs::{BinOp, Instr, InstrNode, Operand};
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
                    params: vec![Ty::I32, Ty::I32],
                    returns: vec![Ty::I32],
                },
                entry: BlockId(0),
                blocks: vec![TmirBlock {
                    id: BlockId(0),
                    params: vec![
                        (ValueId(0), Ty::I32),
                        (ValueId(1), Ty::I32),
                    ],
                    body: vec![
                        InstrNode {
                            instr: Inst::BinOp {
                                op: BinOp::Add,
                                ty: Ty::I32,
                                lhs: Operand::Value(ValueId(0)),
                                rhs: Operand::Value(ValueId(1)),
                            },
                            results: vec![ValueId(2)],
                            proofs: vec![],
                        },
                        InstrNode {
                            instr: Inst::Return {
                                values: vec![Operand::Value(ValueId(2))],
                            },
                            results: vec![],
                            proofs: vec![],
                        },
                    ],
                }],
                proofs: vec![],
            }],
            structs: vec![],
            globals: vec![],
            data_layout: None,
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
                        Ty::array(Ty::float(64), 1000),
                        Ty::array(Ty::float(64), 1000),
                    ],
                    returns: vec![Ty::float(64)],
                },
                entry: BlockId(0),
                blocks: vec![TmirBlock {
                    id: BlockId(0),
                    params: vec![
                        (ValueId(0), Ty::array(Ty::float(64), 1000)),
                        (ValueId(1), Ty::array(Ty::float(64), 1000)),
                    ],
                    body: vec![
                        InstrNode {
                            instr: Inst::BinOp {
                                op: BinOp::FAdd,
                                ty: Ty::array(Ty::float(64), 1000),
                                lhs: Operand::Value(ValueId(0)),
                                rhs: Operand::Value(ValueId(1)),
                            },
                            results: vec![ValueId(2)],
                            proofs: vec![],
                        },
                        InstrNode {
                            instr: Inst::Return {
                                values: vec![Operand::Value(ValueId(2))],
                            },
                            results: vec![],
                            proofs: vec![],
                        },
                    ],
                }],
                proofs: vec![],
            }],
            structs: vec![],
            globals: vec![],
            data_layout: None,
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
                        Ty::array(Ty::float(64), 1000),
                        Ty::array(Ty::float(64), 1000),
                    ],
                    returns: vec![Ty::float(64)],
                },
                entry: BlockId(0),
                blocks: vec![TmirBlock {
                    id: BlockId(0),
                    params: vec![
                        (ValueId(0), Ty::array(Ty::float(64), 1000)),
                        (ValueId(1), Ty::array(Ty::float(64), 1000)),
                    ],
                    body: vec![
                        // FMul: multiply elements
                        InstrNode {
                            instr: Inst::BinOp {
                                op: BinOp::FMul,
                                ty: Ty::array(Ty::float(64), 1000),
                                lhs: Operand::Value(ValueId(0)),
                                rhs: Operand::Value(ValueId(1)),
                            },
                            results: vec![ValueId(2)],
                            proofs: vec![],
                        },
                        // FAdd: accumulate (MAC pattern)
                        InstrNode {
                            instr: Inst::BinOp {
                                op: BinOp::FAdd,
                                ty: Ty::float(64),
                                lhs: Operand::Value(ValueId(2)),
                                rhs: Operand::Value(ValueId(2)),
                            },
                            results: vec![ValueId(3)],
                            proofs: vec![],
                        },
                        InstrNode {
                            instr: Inst::Return {
                                values: vec![Operand::Value(ValueId(3))],
                            },
                            results: vec![],
                            proofs: vec![],
                        },
                    ],
                }],
                proofs: vec![],
            }],
            structs: vec![],
            globals: vec![],
            data_layout: None,
        }
    }

    /// Build a large data-parallel module (100K-element arrays).
    /// Workload large enough for GPU profitability thresholds.
    fn build_large_data_parallel_module() -> Module {
        Module {
            name: "large_data_parallel".to_string(),
            functions: vec![TmirFunction {
                id: FuncId(0),
                name: "large_vec_add".to_string(),
                ty: FuncTy {
                    params: vec![
                        Ty::array(Ty::float(64), 100_000),
                        Ty::array(Ty::float(64), 100_000),
                    ],
                    returns: vec![Ty::float(64)],
                },
                entry: BlockId(0),
                blocks: vec![TmirBlock {
                    id: BlockId(0),
                    params: vec![
                        (ValueId(0), Ty::array(Ty::float(64), 100_000)),
                        (ValueId(1), Ty::array(Ty::float(64), 100_000)),
                    ],
                    body: vec![
                        InstrNode {
                            instr: Inst::BinOp {
                                op: BinOp::FAdd,
                                ty: Ty::array(Ty::float(64), 100_000),
                                lhs: Operand::Value(ValueId(0)),
                                rhs: Operand::Value(ValueId(1)),
                            },
                            results: vec![ValueId(2)],
                            proofs: vec![],
                        },
                        InstrNode {
                            instr: Inst::Return {
                                values: vec![Operand::Value(ValueId(2))],
                            },
                            results: vec![],
                            proofs: vec![],
                        },
                    ],
                }],
                proofs: vec![],
            }],
            structs: vec![],
            globals: vec![],
            data_layout: None,
        }
    }

    /// Build a large matrix-heavy module (100K-element arrays).
    /// Workload large enough for GPU/ANE profitability thresholds.
    fn build_large_matrix_heavy_module() -> Module {
        Module {
            name: "large_matrix_heavy".to_string(),
            functions: vec![TmirFunction {
                id: FuncId(0),
                name: "large_dot_product".to_string(),
                ty: FuncTy {
                    params: vec![
                        Ty::array(Ty::float(64), 100_000),
                        Ty::array(Ty::float(64), 100_000),
                    ],
                    returns: vec![Ty::float(64)],
                },
                entry: BlockId(0),
                blocks: vec![TmirBlock {
                    id: BlockId(0),
                    params: vec![
                        (ValueId(0), Ty::array(Ty::float(64), 100_000)),
                        (ValueId(1), Ty::array(Ty::float(64), 100_000)),
                    ],
                    body: vec![
                        // FMul: multiply elements
                        InstrNode {
                            instr: Inst::BinOp {
                                op: BinOp::FMul,
                                ty: Ty::array(Ty::float(64), 100_000),
                                lhs: Operand::Value(ValueId(0)),
                                rhs: Operand::Value(ValueId(1)),
                            },
                            results: vec![ValueId(2)],
                            proofs: vec![],
                        },
                        // FAdd: accumulate (MAC pattern)
                        InstrNode {
                            instr: Inst::BinOp {
                                op: BinOp::FAdd,
                                ty: Ty::float(64),
                                lhs: Operand::Value(ValueId(2)),
                                rhs: Operand::Value(ValueId(2)),
                            },
                            results: vec![ValueId(3)],
                            proofs: vec![],
                        },
                        InstrNode {
                            instr: Inst::Return {
                                values: vec![Operand::Value(ValueId(3))],
                            },
                            results: vec![],
                            proofs: vec![],
                        },
                    ],
                }],
                proofs: vec![],
            }],
            structs: vec![],
            globals: vec![],
            data_layout: None,
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
                    params: vec![Ty::I32, Ty::I32],
                    returns: vec![Ty::I32],
                },
                entry: BlockId(0),
                blocks: vec![
                    TmirBlock {
                        id: BlockId(0),
                        params: vec![
                            (ValueId(0), Ty::I32),
                            (ValueId(1), Ty::I32),
                        ],
                        body: vec![
                            InstrNode {
                                instr: Inst::BinOp {
                                    op: BinOp::Add,
                                    ty: Ty::I32,
                                    lhs: Operand::Value(ValueId(0)),
                                    rhs: Operand::Value(ValueId(1)),
                                },
                                results: vec![ValueId(2)],
                                proofs: vec![],
                            },
                            InstrNode {
                                instr: Inst::Br {
                                    target: BlockId(1),
                                    args: vec![Operand::Value(ValueId(2))],
                                },
                                results: vec![],
                                proofs: vec![],
                            },
                        ],
                    },
                    TmirBlock {
                        id: BlockId(1),
                        params: vec![(ValueId(3), Ty::I32)],
                        body: vec![
                            InstrNode {
                                instr: Inst::BinOp {
                                    op: BinOp::Mul,
                                    ty: Ty::I32,
                                    lhs: Operand::Value(ValueId(3)),
                                    rhs: Operand::Value(ValueId(3)),
                                },
                                results: vec![ValueId(4)],
                                proofs: vec![],
                            },
                            InstrNode {
                                instr: Inst::Return {
                                    values: vec![Operand::Value(ValueId(4))],
                                },
                                results: vec![],
                                proofs: vec![],
                            },
                        ],
                    },
                ],
                proofs: vec![],
            }],
            structs: vec![],
            globals: vec![],
            data_layout: None,
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
        let instrs = [InstrNode {
            instr: Inst::BinOp {
                op: BinOp::Add,
                ty: Ty::I32,
                lhs: Operand::Value(ValueId(0)),
                rhs: Operand::Value(ValueId(1)),
            },
            results: vec![ValueId(2)],
            proofs: vec![],
        }];
        let refs: Vec<&InstrNode> = instrs.iter().collect();

        let mut types = HashMap::new();
        types.insert(ValueId(0), Ty::I32);
        types.insert(ValueId(1), Ty::I32);

        assert!(!detect_data_parallel(&refs, &types));
    }

    // -------------------------------------------------------------------
    // Test: Pattern detection helper - array FAdd is data-parallel
    // -------------------------------------------------------------------

    #[test]
    fn test_detect_data_parallel_with_arrays() {
        let instrs = [InstrNode {
            instr: Inst::BinOp {
                op: BinOp::FAdd,
                ty: Ty::array(Ty::float(64), 100),
                lhs: Operand::Value(ValueId(0)),
                rhs: Operand::Value(ValueId(1)),
            },
            results: vec![ValueId(2)],
            proofs: vec![],
        }];
        let refs: Vec<&InstrNode> = instrs.iter().collect();

        let mut types = HashMap::new();
        types.insert(ValueId(0), Ty::array(Ty::float(64), 100));
        types.insert(ValueId(1), Ty::array(Ty::float(64), 100));

        assert!(detect_data_parallel(&refs, &types));
    }

    // -------------------------------------------------------------------
    // Test: Pattern detection - MAC pattern requires both FMul and FAdd
    // -------------------------------------------------------------------

    #[test]
    fn test_detect_matrix_heavy_requires_mac() {
        // FMul alone is not matrix-heavy
        let instrs = [InstrNode {
            instr: Inst::BinOp {
                op: BinOp::FMul,
                ty: Ty::array(Ty::float(64), 100),
                lhs: Operand::Value(ValueId(0)),
                rhs: Operand::Value(ValueId(1)),
            },
            results: vec![ValueId(2)],
            proofs: vec![],
        }];
        let refs: Vec<&InstrNode> = instrs.iter().collect();

        let mut types = HashMap::new();
        types.insert(ValueId(0), Ty::array(Ty::float(64), 100));

        assert!(!detect_matrix_heavy(&refs, &types));
    }

    // -------------------------------------------------------------------
    // Test: classify_node prioritizes MatrixHeavy over DataParallel
    // -------------------------------------------------------------------

    #[test]
    fn test_classify_node_matrix_over_parallel() {
        let instrs = [InstrNode {
                instr: Inst::BinOp {
                    op: BinOp::FMul,
                    ty: Ty::array(Ty::float(64), 100),
                    lhs: Operand::Value(ValueId(0)),
                    rhs: Operand::Value(ValueId(1)),
                },
                results: vec![ValueId(2)],
                proofs: vec![],
            },
            InstrNode {
                instr: Inst::BinOp {
                    op: BinOp::FAdd,
                    ty: Ty::float(64),
                    lhs: Operand::Value(ValueId(2)),
                    rhs: Operand::Value(ValueId(2)),
                },
                results: vec![ValueId(3)],
                proofs: vec![],
            }];
        let refs: Vec<&InstrNode> = instrs.iter().collect();

        let mut types = HashMap::new();
        types.insert(ValueId(0), Ty::array(Ty::float(64), 100));
        types.insert(ValueId(1), Ty::array(Ty::float(64), 100));

        // MAC pattern on arrays -> MatrixHeavy takes priority
        assert_eq!(classify_node(&refs, &types), NodeKind::MatrixHeavy);
    }

    // -------------------------------------------------------------------
    // Test: estimate_type_bytes
    // -------------------------------------------------------------------

    #[test]
    fn test_estimate_type_bytes() {
        assert_eq!(estimate_type_bytes(&Ty::Bool), 1);
        assert_eq!(estimate_type_bytes(&Ty::int(8)), 1);
        assert_eq!(estimate_type_bytes(&Ty::int(16)), 2);
        assert_eq!(estimate_type_bytes(&Ty::I32), 4);
        assert_eq!(estimate_type_bytes(&Ty::I64), 8);
        assert_eq!(estimate_type_bytes(&Ty::float(32)), 4);
        assert_eq!(estimate_type_bytes(&Ty::float(64)), 8);
        assert_eq!(
            estimate_type_bytes(&Ty::array(Ty::float(64), 100)),
            800
        );
        assert_eq!(estimate_type_bytes(&Ty::ptr(Ty::I32)), 8);
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
            dominant_op: "ADD".to_string(),
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
            dominant_op: "ADD".to_string(),
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
                        params: vec![Ty::I32],
                        returns: vec![Ty::I32],
                    },
                    entry: BlockId(0),
                    blocks: vec![TmirBlock {
                        id: BlockId(0),
                        params: vec![(ValueId(0), Ty::I32)],
                        body: vec![InstrNode {
                            instr: Inst::Return { values: vec![Operand::Value(ValueId(0))] },
                            results: vec![],
                            proofs: vec![],
                        }],
                    }],
                    proofs: vec![],
                },
                TmirFunction {
                    id: FuncId(1),
                    name: "bar".to_string(),
                    ty: FuncTy {
                        params: vec![Ty::I64],
                        returns: vec![Ty::I64],
                    },
                    entry: BlockId(0),
                    blocks: vec![TmirBlock {
                        id: BlockId(0),
                        params: vec![(ValueId(10), Ty::I64)],
                        body: vec![InstrNode {
                            instr: Inst::Return { values: vec![Operand::Value(ValueId(10))] },
                            results: vec![],
                            proofs: vec![],
                        }],
                    }],
                    proofs: vec![],
                },
            ],
            structs: vec![],
            globals: vec![],
            data_layout: None,
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
                        base: tmir_types::ValueId(i),
                        index: tmir_types::ValueId(i + 100),
                    },
                    Proof::ValidBorrow {
                        borrow: tmir_types::ValueId(i),
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
        // Use large module so workload exceeds GPU profitability thresholds.
        let module = build_large_data_parallel_module();
        let proof_ctx = full_proof_context();
        let analyzer = ProofAnalyzer::with_defaults();

        let graph = ComputeGraph::with_proof_context(&module, proof_ctx, &analyzer);
        let recs = graph.target_recommendations();

        assert_eq!(recs.len(), 1);
        let rec = &recs[0];

        // For large data-parallel workloads with full proofs, GPU should pass
        // profitability check and be in the filtered legal targets.
        assert!(
            rec.legal_targets.contains(&ComputeTarget::Gpu),
            "GPU should be in legal targets for large workload"
        );

        // The recommendation should be GPU because it has the lowest latency
        // for large data-parallel workloads.
        assert_eq!(
            rec.recommended_target,
            ComputeTarget::Gpu,
            "Should recommend GPU for large data-parallel with full proofs, got: {}",
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
        // Use large module so workload exceeds GPU/ANE profitability thresholds.
        let module = build_large_matrix_heavy_module();

        let mut proof_ctx = ProofContext::default();
        for i in 0..2 {
            proof_ctx.value_proofs.insert(
                Value(i),
                vec![
                    Proof::InBounds {
                        base: tmir_types::ValueId(i),
                        index: tmir_types::ValueId(i + 100),
                    },
                    Proof::ValidBorrow {
                        borrow: tmir_types::ValueId(i),
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

        // Matrix-heavy with large workload: GPU should be recommended
        // (profitability thresholds exceeded).
        assert!(
            rec.recommended_target == ComputeTarget::Gpu
                || rec.recommended_target == ComputeTarget::NeuralEngine,
            "Large matrix-heavy with proofs should recommend GPU or ANE, got: {}",
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

    // ===================================================================
    // False-positive prevention tests (#159)
    // ===================================================================

    // -------------------------------------------------------------------
    // Test: scalar ops with small array params should NOT be data-parallel
    // -------------------------------------------------------------------

    #[test]
    fn test_small_array_not_data_parallel() {
        // An array of 2 elements is too small to justify vectorization
        let instrs = [InstrNode {
            instr: Inst::BinOp {
                op: BinOp::FAdd,
                ty: Ty::array(Ty::float(64), 2),
                lhs: Operand::Value(ValueId(0)),
                rhs: Operand::Value(ValueId(1)),
            },
            results: vec![ValueId(2)],
            proofs: vec![],
        }];
        let refs: Vec<&InstrNode> = instrs.iter().collect();

        let mut types = HashMap::new();
        types.insert(ValueId(0), Ty::array(Ty::float(64), 2));
        types.insert(ValueId(1), Ty::array(Ty::float(64), 2));

        assert!(
            !detect_data_parallel(&refs, &types),
            "Arrays with < MIN_VECTORIZABLE_ELEMENTS should not be classified as data-parallel"
        );
    }

    // -------------------------------------------------------------------
    // Test: scalar binary op coexisting with array param is NOT data-parallel
    // -------------------------------------------------------------------

    #[test]
    fn test_scalar_op_with_array_param_not_data_parallel() {
        // The binary op operates on scalar i32 values, even though an array
        // parameter exists in the value types. This should NOT match.
        let instrs = [InstrNode {
            instr: Inst::BinOp {
                op: BinOp::Add,
                ty: Ty::I32,
                lhs: Operand::Value(ValueId(0)),
                rhs: Operand::Value(ValueId(1)),
            },
            results: vec![ValueId(2)],
            proofs: vec![],
        }];
        let refs: Vec<&InstrNode> = instrs.iter().collect();

        let mut types = HashMap::new();
        types.insert(ValueId(0), Ty::I32); // operand is scalar
        types.insert(ValueId(1), Ty::I32); // operand is scalar
        // Some other value is array-typed but not an operand of the Add
        types.insert(ValueId(10), Ty::array(Ty::float(64), 1000));

        assert!(
            !detect_data_parallel(&refs, &types),
            "Scalar op with array in scope should not match data-parallel pattern"
        );
    }

    // -------------------------------------------------------------------
    // Test: FMul without subsequent FAdd is NOT matrix-heavy
    // (already tested, but verify the strengthened check still works)
    // -------------------------------------------------------------------

    #[test]
    fn test_fmul_only_with_array_not_matrix_heavy() {
        let instrs = [InstrNode {
            instr: Inst::BinOp {
                op: BinOp::FMul,
                ty: Ty::array(Ty::float(64), 100),
                lhs: Operand::Value(ValueId(0)),
                rhs: Operand::Value(ValueId(1)),
            },
            results: vec![ValueId(2)],
            proofs: vec![],
        }];
        let refs: Vec<&InstrNode> = instrs.iter().collect();

        let mut types = HashMap::new();
        types.insert(ValueId(0), Ty::array(Ty::float(64), 100));
        types.insert(ValueId(1), Ty::array(Ty::float(64), 100));

        assert!(
            !detect_matrix_heavy(&refs, &types),
            "FMul alone (no FAdd consuming result) should not be matrix-heavy"
        );
    }

    // -------------------------------------------------------------------
    // Test: FMul + FAdd with no data dependency is NOT matrix-heavy
    // -------------------------------------------------------------------

    #[test]
    fn test_independent_fmul_fadd_not_matrix_heavy() {
        // FMul produces ValueId(2), but FAdd does NOT consume it --
        // it consumes ValueId(3) and ValueId(4) instead.
        let instrs = vec![
            InstrNode {
                instr: Inst::BinOp {
                    op: BinOp::FMul,
                    ty: Ty::array(Ty::float(64), 100),
                    lhs: Operand::Value(ValueId(0)),
                    rhs: Operand::Value(ValueId(1)),
                },
                results: vec![ValueId(2)],
                proofs: vec![],
            },
            InstrNode {
                instr: Inst::BinOp {
                    op: BinOp::FAdd,
                    ty: Ty::float(64),
                    lhs: Operand::Value(ValueId(3)), // NOT ValueId(2)
                    rhs: Operand::Value(ValueId(4)), // NOT ValueId(2)
                },
                results: vec![ValueId(5)],
                proofs: vec![],
            },
        ];
        let refs: Vec<&InstrNode> = instrs.iter().collect();

        let mut types = HashMap::new();
        types.insert(ValueId(0), Ty::array(Ty::float(64), 100));
        types.insert(ValueId(1), Ty::array(Ty::float(64), 100));
        types.insert(ValueId(3), Ty::float(64));
        types.insert(ValueId(4), Ty::float(64));

        assert!(
            !detect_matrix_heavy(&refs, &types),
            "FMul + FAdd without data dependency should not be matrix-heavy"
        );
    }

    // -------------------------------------------------------------------
    // Test: FMul on scalar + FAdd consuming result is NOT matrix-heavy
    //       (neither FMul operand is array-typed)
    // -------------------------------------------------------------------

    #[test]
    fn test_scalar_fmul_fadd_not_matrix_heavy() {
        let instrs = [InstrNode {
                instr: Inst::BinOp {
                    op: BinOp::FMul,
                    ty: Ty::float(64),
                    lhs: Operand::Value(ValueId(0)),
                    rhs: Operand::Value(ValueId(1)),
                },
                results: vec![ValueId(2)],
                proofs: vec![],
            },
            InstrNode {
                instr: Inst::BinOp {
                    op: BinOp::FAdd,
                    ty: Ty::float(64),
                    lhs: Operand::Value(ValueId(2)),
                    rhs: Operand::Value(ValueId(2)),
                },
                results: vec![ValueId(3)],
                proofs: vec![],
            }];
        let refs: Vec<&InstrNode> = instrs.iter().collect();

        let mut types = HashMap::new();
        types.insert(ValueId(0), Ty::float(64)); // scalar
        types.insert(ValueId(1), Ty::float(64)); // scalar
        // There IS an array in scope, but the FMul operands are not array-typed
        types.insert(ValueId(10), Ty::array(Ty::float(64), 1000));

        assert!(
            !detect_matrix_heavy(&refs, &types),
            "Scalar FMul + FAdd with array in scope should not be matrix-heavy"
        );
    }

    // -------------------------------------------------------------------
    // Test: small array with MAC pattern is NOT matrix-heavy
    // -------------------------------------------------------------------

    #[test]
    fn test_small_array_mac_not_matrix_heavy() {
        // Array has only 2 elements -- below MIN_VECTORIZABLE_ELEMENTS
        let instrs = [InstrNode {
                instr: Inst::BinOp {
                    op: BinOp::FMul,
                    ty: Ty::array(Ty::float(64), 2),
                    lhs: Operand::Value(ValueId(0)),
                    rhs: Operand::Value(ValueId(1)),
                },
                results: vec![ValueId(2)],
                proofs: vec![],
            },
            InstrNode {
                instr: Inst::BinOp {
                    op: BinOp::FAdd,
                    ty: Ty::float(64),
                    lhs: Operand::Value(ValueId(2)),
                    rhs: Operand::Value(ValueId(2)),
                },
                results: vec![ValueId(3)],
                proofs: vec![],
            }];
        let refs: Vec<&InstrNode> = instrs.iter().collect();

        let mut types = HashMap::new();
        types.insert(ValueId(0), Ty::array(Ty::float(64), 2));
        types.insert(ValueId(1), Ty::array(Ty::float(64), 2));

        assert!(
            !detect_matrix_heavy(&refs, &types),
            "Small arrays (< MIN_VECTORIZABLE_ELEMENTS) should not match matrix-heavy"
        );
    }

    // -------------------------------------------------------------------
    // Test: Confirm valid MAC pattern with data dep still matches
    // -------------------------------------------------------------------

    #[test]
    fn test_valid_mac_with_dependency_matches() {
        // FMul produces ValueId(2), FAdd consumes ValueId(2) -- valid MAC
        let instrs = [InstrNode {
                instr: Inst::BinOp {
                    op: BinOp::FMul,
                    ty: Ty::array(Ty::float(64), 100),
                    lhs: Operand::Value(ValueId(0)),
                    rhs: Operand::Value(ValueId(1)),
                },
                results: vec![ValueId(2)],
                proofs: vec![],
            },
            InstrNode {
                instr: Inst::BinOp {
                    op: BinOp::FAdd,
                    ty: Ty::float(64),
                    lhs: Operand::Value(ValueId(2)),
                    rhs: Operand::Value(ValueId(2)),
                },
                results: vec![ValueId(3)],
                proofs: vec![],
            }];
        let refs: Vec<&InstrNode> = instrs.iter().collect();

        let mut types = HashMap::new();
        types.insert(ValueId(0), Ty::array(Ty::float(64), 100));
        types.insert(ValueId(1), Ty::array(Ty::float(64), 100));

        assert!(
            detect_matrix_heavy(&refs, &types),
            "Valid MAC pattern with array operands and data dep should match"
        );
    }

    // -------------------------------------------------------------------
    // Test: Module with small array ops classifies as Scalar
    // -------------------------------------------------------------------

    #[test]
    fn test_small_array_module_classifies_scalar() {
        // A module with 2-element arrays should be classified as Scalar
        // (below the MIN_VECTORIZABLE_ELEMENTS threshold)
        let module = Module {
            name: "small_array".to_string(),
            functions: vec![TmirFunction {
                id: FuncId(0),
                name: "tiny_add".to_string(),
                ty: FuncTy {
                    params: vec![
                        Ty::array(Ty::float(64), 2),
                        Ty::array(Ty::float(64), 2),
                    ],
                    returns: vec![Ty::float(64)],
                },
                entry: BlockId(0),
                blocks: vec![TmirBlock {
                    id: BlockId(0),
                    params: vec![
                        (ValueId(0), Ty::array(Ty::float(64), 2)),
                        (ValueId(1), Ty::array(Ty::float(64), 2)),
                    ],
                    body: vec![
                        InstrNode {
                            instr: Inst::BinOp {
                                op: BinOp::FAdd,
                                ty: Ty::array(Ty::float(64), 2),
                                lhs: Operand::Value(ValueId(0)),
                                rhs: Operand::Value(ValueId(1)),
                            },
                            results: vec![ValueId(2)],
                            proofs: vec![],
                        },
                        InstrNode {
                            instr: Inst::Return {
                                values: vec![Operand::Value(ValueId(2))],
                            },
                            results: vec![],
                            proofs: vec![],
                        },
                    ],
                }],
                proofs: vec![],
            }],
            structs: vec![],
            globals: vec![],
            data_layout: None,
        };

        let graph = ComputeGraph::from_module(&module);
        assert_eq!(graph.num_nodes(), 1);
        assert_eq!(
            graph.nodes[0].kind,
            NodeKind::Scalar,
            "Small array operations should be classified as Scalar, not DataParallel"
        );
    }

    // -------------------------------------------------------------------
    // ProfitabilityAnalyzer integration tests
    // -------------------------------------------------------------------

    #[test]
    fn test_small_workload_filters_gpu_ane() {
        // Small data-parallel module (1000-element arrays, ~16KB data).
        // GPU requires >= 4096 elements; ANE requires >= 32KB.
        // Both should be filtered out by ProfitabilityAnalyzer.
        let module = build_data_parallel_module();
        let proof_ctx = full_proof_context();
        let analyzer = ProofAnalyzer::with_defaults();

        let graph = ComputeGraph::with_proof_context(&module, proof_ctx, &analyzer);
        let recs = graph.target_recommendations();

        assert_eq!(recs.len(), 1);
        let rec = &recs[0];

        // GPU and ANE should be filtered out by profitability checks.
        assert!(
            !rec.legal_targets.contains(&ComputeTarget::Gpu),
            "GPU should be filtered for small workload (profitability)"
        );
        assert!(
            !rec.legal_targets.contains(&ComputeTarget::NeuralEngine),
            "ANE should be filtered for small workload (profitability)"
        );

        // Should recommend CPU target instead.
        assert!(
            rec.recommended_target == ComputeTarget::CpuScalar
                || rec.recommended_target == ComputeTarget::CpuSimd,
            "Small workload should recommend CPU, got: {}",
            rec.recommended_target
        );
    }

    #[test]
    fn test_large_workload_includes_gpu() {
        // Large data-parallel module (100K-element arrays, ~1.6MB data).
        // Well above GPU thresholds.
        let module = build_large_data_parallel_module();
        let proof_ctx = full_proof_context();
        let analyzer = ProofAnalyzer::with_defaults();

        let graph = ComputeGraph::with_proof_context(&module, proof_ctx, &analyzer);
        let recs = graph.target_recommendations();

        assert_eq!(recs.len(), 1);
        let rec = &recs[0];

        // Large workload: GPU should pass profitability check.
        assert!(
            rec.legal_targets.contains(&ComputeTarget::Gpu),
            "GPU should be profitable for large workload"
        );
    }

    #[test]
    fn test_target_legality_filters_bitwise_from_ane() {
        // Build a module with bitwise operations (AND).
        // ANE does not support bitwise ops per ProfitabilityAnalyzer::target_legality.
        let module = Module {
            name: "bitwise".to_string(),
            functions: vec![TmirFunction {
                id: FuncId(0),
                name: "bitwise_and".to_string(),
                ty: FuncTy {
                    params: vec![
                        Ty::array(Ty::I32, 100_000),
                        Ty::array(Ty::I32, 100_000),
                    ],
                    returns: vec![Ty::I32],
                },
                entry: BlockId(0),
                blocks: vec![TmirBlock {
                    id: BlockId(0),
                    params: vec![
                        (ValueId(0), Ty::array(Ty::I32, 100_000)),
                        (ValueId(1), Ty::array(Ty::I32, 100_000)),
                    ],
                    body: vec![
                        InstrNode {
                            instr: Inst::BinOp {
                                op: BinOp::And,
                                ty: Ty::array(Ty::I32, 100_000),
                                lhs: Operand::Value(ValueId(0)),
                                rhs: Operand::Value(ValueId(1)),
                            },
                            results: vec![ValueId(2)],
                            proofs: vec![],
                        },
                        InstrNode {
                            instr: Inst::Return {
                                values: vec![Operand::Value(ValueId(2))],
                            },
                            results: vec![],
                            proofs: vec![],
                        },
                    ],
                }],
                proofs: vec![],
            }],
            structs: vec![],
            globals: vec![],
            data_layout: None,
        };

        // Give full proofs so GPU/ANE are at least proof-legal.
        let proof_ctx = full_proof_context();
        let analyzer = ProofAnalyzer::with_defaults();

        let graph = ComputeGraph::with_proof_context(&module, proof_ctx, &analyzer);
        let recs = graph.target_recommendations();

        assert_eq!(recs.len(), 1);
        let rec = &recs[0];

        // ANE should be filtered: ProfitabilityAnalyzer says AND is not
        // ANE-legal (bitwise ops are not supported by the Neural Engine).
        assert!(
            !rec.legal_targets.contains(&ComputeTarget::NeuralEngine),
            "ANE should not be legal for bitwise AND (hardware limitation)"
        );
    }

    #[test]
    fn test_profitability_set_and_get() {
        // Verify that setting a ProfitabilityAnalyzer on a graph affects
        // target_recommendations() behavior.
        let module = build_data_parallel_module();
        let proof_ctx = full_proof_context();
        let analyzer = ProofAnalyzer::with_defaults();

        // Build graph WITHOUT profitability analyzer.
        let mut builder = GraphBuilder::new(analyzer.clone(), proof_ctx.clone());
        let mut graph = builder.build_from_module(&module);

        // Without profitability: GPU should appear in recommendations
        // (it's proof-legal and has lowest latency).
        let recs_without = graph.target_recommendations();
        let has_gpu_without = recs_without.iter().any(|r|
            r.legal_targets.contains(&ComputeTarget::Gpu)
        );

        // Now set the profitability analyzer.
        graph.set_profitability(ProfitabilityAnalyzer::new(CostModelGen::M1));

        // With profitability: GPU should be filtered for this small workload.
        let recs_with = graph.target_recommendations();
        let has_gpu_with = recs_with.iter().any(|r|
            r.legal_targets.contains(&ComputeTarget::Gpu)
        );

        // The small workload should see GPU removed after profitability filtering.
        if has_gpu_without {
            assert!(
                !has_gpu_with,
                "ProfitabilityAnalyzer should filter GPU for small workload"
            );
        }
    }

    #[test]
    fn test_dominant_op_derived_from_instructions() {
        // Build modules and verify the dominant_op field is set correctly.
        let module = build_data_parallel_module();
        let graph = ComputeGraph::from_module(&module);

        assert_eq!(graph.num_nodes(), 1);
        // data_parallel module has a single FAdd -> dominant op is "FADD"
        assert_eq!(
            graph.nodes[0].dominant_op, "FADD",
            "Data-parallel module should have dominant_op FADD"
        );

        // Matrix-heavy module has FMul + FAdd -> dominant is tied,
        // HashMap iteration order is nondeterministic but both are valid.
        let mat_module = build_matrix_heavy_module();
        let mat_graph = ComputeGraph::from_module(&mat_module);

        assert_eq!(mat_graph.num_nodes(), 1);
        let dom = &mat_graph.nodes[0].dominant_op;
        assert!(
            dom == "FMUL" || dom == "FADD",
            "Matrix-heavy module should have dominant_op FMUL or FADD, got: {}",
            dom
        );
    }
}
