// llvm2-lower/dispatch.rs - Dispatch codegen for heterogeneous compute
//
// Author: Andrew Yates <ayates@dropbox.com>
// Copyright 2026 Dropbox, Inc. | License: Apache-2.0
//
// Generates dispatch plans from compute graph target recommendations.
// A dispatch plan is a sequence of operations: data transfers between
// compute domains, kernel launches, synchronizations, and CPU fallbacks.
// The plan respects data dependencies from the compute graph edges and
// validates that all inputs are available before each operation executes.

//! Dispatch plan generation for heterogeneous compute.
//!
//! Given a [`ComputeGraph`] and per-node [`TargetRecommendation`]s, this
//! module produces a [`DispatchPlan`] — an ordered sequence of
//! [`DispatchOp`]s that correctly moves data, launches compute, and
//! synchronizes results across CPU, NEON, GPU, and ANE targets.
//!
//! # Architecture
//!
//! ```text
//! ComputeGraph + Vec<TargetRecommendation>
//!     |
//!     v
//! generate_dispatch_plan()
//!     |
//!     v
//! DispatchPlan { ops: Vec<DispatchOp> }
//!     |
//!     v
//! validate_dispatch_plan() -> Result<(), DispatchError>
//! ```

use std::collections::{HashMap, HashSet};
use std::fmt;

use crate::compute_graph::{
    ComputeGraph, ComputeNodeId, TargetRecommendation, TransferCost,
    estimate_transfer_cost,
};
use crate::target_analysis::ComputeTarget;

// ---------------------------------------------------------------------------
// DispatchOp — individual dispatch operations
// ---------------------------------------------------------------------------

/// A single dispatch operation in a dispatch plan.
#[derive(Debug, Clone, PartialEq)]
pub enum DispatchOp {
    /// Transfer data between compute domains.
    ///
    /// On Apple Silicon with unified memory, CPU<->GPU transfers are
    /// cache coherency operations rather than physical copies. CPU<->ANE
    /// transfers involve CoreML tensor marshalling.
    DataTransfer {
        /// Source compute target holding the data.
        src: ComputeTarget,
        /// Destination compute target that needs the data.
        dst: ComputeTarget,
        /// Data size in bytes.
        size_bytes: u64,
        /// Estimated transfer cost.
        cost: TransferCost,
        /// The graph edge that triggered this transfer (for traceability).
        edge_from: ComputeNodeId,
        /// The destination node consuming this data.
        edge_to: ComputeNodeId,
    },

    /// Launch a compute kernel on a specific target.
    ///
    /// For CPU scalar/NEON, this is inline execution (no actual "launch").
    /// For GPU, this creates a Metal compute command encoder.
    /// For ANE, this submits a CoreML prediction request.
    KernelLaunch {
        /// Target to execute on.
        target: ComputeTarget,
        /// The compute graph node being executed.
        node_id: ComputeNodeId,
        /// Estimated compute cost in cycles.
        estimated_cycles: u64,
    },

    /// Synchronize execution — wait for outstanding async operations.
    ///
    /// Required after GPU/ANE kernel launches before results can be read
    /// back to CPU or used by a different compute domain.
    Synchronize {
        /// The target to synchronize.
        target: ComputeTarget,
        /// The node whose results are being waited on.
        node_id: ComputeNodeId,
    },

    /// Fall back to CPU scalar execution.
    ///
    /// Used when the recommended target is unavailable or when the node
    /// is simple enough that dispatch overhead exceeds compute savings.
    CpuFallback {
        /// The compute graph node being executed on CPU.
        node_id: ComputeNodeId,
        /// Reason for the fallback.
        reason: String,
    },
}

impl fmt::Display for DispatchOp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DispatchOp::DataTransfer { src, dst, size_bytes, edge_from, edge_to, .. } => {
                write!(
                    f,
                    "Transfer {} -> {} ({} bytes, {} -> {})",
                    src, dst, size_bytes, edge_from, edge_to
                )
            }
            DispatchOp::KernelLaunch { target, node_id, estimated_cycles } => {
                write!(
                    f,
                    "Launch {} on {} (~{} cycles)",
                    node_id, target, estimated_cycles
                )
            }
            DispatchOp::Synchronize { target, node_id } => {
                write!(f, "Sync {} after {}", target, node_id)
            }
            DispatchOp::CpuFallback { node_id, reason } => {
                write!(f, "CpuFallback {} ({})", node_id, reason)
            }
        }
    }
}

// ---------------------------------------------------------------------------
// DispatchPlan
// ---------------------------------------------------------------------------

/// A complete dispatch plan: an ordered sequence of dispatch operations.
///
/// The operations must be executed in order. Data transfers precede the
/// kernel launches that depend on them. Synchronizations follow async
/// kernel launches before their results are consumed.
#[derive(Debug, Clone)]
pub struct DispatchPlan {
    /// Ordered sequence of dispatch operations.
    pub ops: Vec<DispatchOp>,
    /// Per-node target assignment (for validation and debugging).
    pub assignment: HashMap<ComputeNodeId, ComputeTarget>,
    /// Total estimated cost in cycles (compute + transfer + sync).
    pub estimated_total_cycles: u64,
}

impl DispatchPlan {
    /// Number of operations in the plan.
    pub fn len(&self) -> usize {
        self.ops.len()
    }

    /// Whether the plan is empty.
    pub fn is_empty(&self) -> bool {
        self.ops.is_empty()
    }

    /// Count operations of a specific kind.
    pub fn count_transfers(&self) -> usize {
        self.ops.iter().filter(|op| matches!(op, DispatchOp::DataTransfer { .. })).count()
    }

    /// Count kernel launches.
    pub fn count_launches(&self) -> usize {
        self.ops.iter().filter(|op| matches!(op, DispatchOp::KernelLaunch { .. })).count()
    }

    /// Count synchronizations.
    pub fn count_syncs(&self) -> usize {
        self.ops.iter().filter(|op| matches!(op, DispatchOp::Synchronize { .. })).count()
    }

    /// Count CPU fallbacks.
    pub fn count_fallbacks(&self) -> usize {
        self.ops.iter().filter(|op| matches!(op, DispatchOp::CpuFallback { .. })).count()
    }
}

impl fmt::Display for DispatchPlan {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "DispatchPlan ({} ops, ~{} cycles):", self.ops.len(), self.estimated_total_cycles)?;
        for (i, op) in self.ops.iter().enumerate() {
            writeln!(f, "  [{}] {}", i, op)?;
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// DispatchError
// ---------------------------------------------------------------------------

/// Errors detected during dispatch plan validation.
#[derive(Debug, Clone, PartialEq)]
pub enum DispatchError {
    /// A node in the graph has no target assignment.
    MissingAssignment {
        node_id: ComputeNodeId,
    },

    /// A data dependency is not satisfied: the source node's results are
    /// not available (via transfer or same-target execution) before the
    /// consumer node launches.
    UnsatisfiedDependency {
        /// The consumer node.
        consumer: ComputeNodeId,
        /// The producer node whose data is missing.
        producer: ComputeNodeId,
    },

    /// A kernel launch references a node not in the graph.
    UnknownNode {
        node_id: ComputeNodeId,
    },

    /// A synchronization is issued for a target that has no pending
    /// async operations.
    SpuriousSync {
        target: ComputeTarget,
        node_id: ComputeNodeId,
    },

    /// An async target (GPU/ANE) has results consumed without a prior
    /// synchronization.
    MissingSyncBeforeRead {
        /// The async target that was not synchronized.
        target: ComputeTarget,
        /// The node whose results need synchronization.
        producer: ComputeNodeId,
        /// The node trying to consume unsynced results.
        consumer: ComputeNodeId,
    },
}

impl fmt::Display for DispatchError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DispatchError::MissingAssignment { node_id } => {
                write!(f, "Node {} has no target assignment", node_id)
            }
            DispatchError::UnsatisfiedDependency { consumer, producer } => {
                write!(
                    f,
                    "Node {} depends on {} but data is not available before launch",
                    consumer, producer
                )
            }
            DispatchError::UnknownNode { node_id } => {
                write!(f, "Node {} is not in the compute graph", node_id)
            }
            DispatchError::SpuriousSync { target, node_id } => {
                write!(
                    f,
                    "Spurious sync of {} for {} — no pending async ops",
                    target, node_id
                )
            }
            DispatchError::MissingSyncBeforeRead { target, producer, consumer } => {
                write!(
                    f,
                    "Node {} reads results of {} on {} without prior synchronization",
                    consumer, producer, target
                )
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Plan generation
// ---------------------------------------------------------------------------

/// Returns true if the given target requires asynchronous dispatch and
/// explicit synchronization before results can be read.
fn is_async_target(target: ComputeTarget) -> bool {
    matches!(target, ComputeTarget::Gpu | ComputeTarget::NeuralEngine)
}

/// Topological sort of compute graph nodes by data dependencies.
///
/// Returns nodes in dependency order: a node appears only after all its
/// predecessors. Panics on cycles (the graph should be a DAG).
fn topological_order(graph: &ComputeGraph) -> Vec<ComputeNodeId> {
    let node_ids: Vec<ComputeNodeId> = graph.nodes.iter().map(|n| n.id).collect();

    // Build in-degree map.
    let mut in_degree: HashMap<ComputeNodeId, usize> = HashMap::new();
    for id in &node_ids {
        in_degree.insert(*id, 0);
    }
    for edge in &graph.edges {
        *in_degree.entry(edge.to).or_insert(0) += 1;
    }

    // Kahn's algorithm.
    let mut queue: Vec<ComputeNodeId> = node_ids
        .iter()
        .filter(|id| in_degree[id] == 0)
        .copied()
        .collect();
    // Sort for deterministic output.
    queue.sort_by_key(|id| id.0);

    let mut order = Vec::with_capacity(node_ids.len());
    while let Some(node_id) = queue.pop() {
        // Use first element (like a FIFO) for stable ordering.
        // We sort below after each round anyway.
        order.push(node_id);
        for edge in graph.outgoing_edges(node_id) {
            if let Some(deg) = in_degree.get_mut(&edge.to) {
                *deg -= 1;
                if *deg == 0 {
                    queue.push(edge.to);
                }
            }
        }
        queue.sort_by_key(|id| id.0);
    }

    order
}

/// Generate a dispatch plan from a compute graph and target recommendations.
///
/// The plan topologically sorts nodes and emits:
/// 1. Data transfers for cross-target edges (before the consumer launches)
/// 2. Kernel launches on the recommended target
/// 3. Synchronizations after async (GPU/ANE) kernel launches
///
/// Scalar-only and NEON-only nodes skip transfer and sync overhead.
pub fn generate_dispatch_plan(
    graph: &ComputeGraph,
    recommendations: &[TargetRecommendation],
) -> DispatchPlan {
    // Build node->target assignment from recommendations.
    let assignment: HashMap<ComputeNodeId, ComputeTarget> = recommendations
        .iter()
        .map(|rec| (rec.node_id, rec.recommended_target))
        .collect();

    let order = topological_order(graph);
    let mut ops = Vec::new();
    let mut total_cycles: u64 = 0;

    // Track which async nodes have been synchronized.
    // After an async kernel launch, we need to sync before the results
    // are consumed by a node on a different target.
    let mut pending_async: HashSet<ComputeNodeId> = HashSet::new();

    for node_id in &order {
        let target = assignment.get(node_id).copied().unwrap_or(ComputeTarget::CpuScalar);

        // --- Emit data transfers for incoming cross-target edges ---
        let incoming = graph.incoming_edges(*node_id);
        for edge in &incoming {
            let src_target = assignment.get(&edge.from).copied().unwrap_or(ComputeTarget::CpuScalar);

            // If the source is an async target and hasn't been synced yet,
            // emit a sync before the transfer/launch.
            if is_async_target(src_target) && pending_async.contains(&edge.from) {
                ops.push(DispatchOp::Synchronize {
                    target: src_target,
                    node_id: edge.from,
                });
                pending_async.remove(&edge.from);
            }

            // Emit transfer if targets differ and there's data to move.
            if src_target != target && edge.transfer_bytes > 0 {
                let cost = estimate_transfer_cost(edge.transfer_bytes, src_target, target);
                total_cycles = total_cycles.saturating_add(cost.total_cycles);
                ops.push(DispatchOp::DataTransfer {
                    src: src_target,
                    dst: target,
                    size_bytes: edge.transfer_bytes,
                    cost,
                    edge_from: edge.from,
                    edge_to: edge.to,
                });
            }
        }

        // --- Emit kernel launch or CPU fallback ---
        let node = graph.node(*node_id);
        let estimated_cycles = node
            .and_then(|n| n.costs.get(&target))
            .map(|c| c.latency_cycles)
            .unwrap_or(1);

        if target == ComputeTarget::CpuScalar
            && !assignment.contains_key(node_id)
        {
            // No recommendation — fall back to CPU.
            ops.push(DispatchOp::CpuFallback {
                node_id: *node_id,
                reason: "no target recommendation".to_string(),
            });
            total_cycles = total_cycles.saturating_add(estimated_cycles);
        } else {
            ops.push(DispatchOp::KernelLaunch {
                target,
                node_id: *node_id,
                estimated_cycles,
            });
            total_cycles = total_cycles.saturating_add(estimated_cycles);

            // Track async launches for later synchronization.
            if is_async_target(target) {
                pending_async.insert(*node_id);
            }
        }
    }

    // Emit final synchronizations for any remaining async nodes whose
    // results are needed by the caller (leaf nodes with async targets).
    let mut remaining_async: Vec<ComputeNodeId> = pending_async.into_iter().collect();
    remaining_async.sort_by_key(|id| id.0);
    for node_id in remaining_async {
        let target = assignment.get(&node_id).copied().unwrap_or(ComputeTarget::CpuScalar);
        ops.push(DispatchOp::Synchronize {
            target,
            node_id,
        });
    }

    DispatchPlan {
        ops,
        assignment,
        estimated_total_cycles: total_cycles,
    }
}

// ---------------------------------------------------------------------------
// Plan validation
// ---------------------------------------------------------------------------

/// Validate a dispatch plan against the compute graph.
///
/// Checks:
/// 1. Every graph node has a target assignment.
/// 2. Every kernel launch references a valid graph node.
/// 3. Data dependencies are satisfied: for each edge, the producer
///    is launched before the consumer, and if on different targets,
///    a data transfer is emitted.
/// 4. Async target results are synchronized before cross-target reads.
pub fn validate_dispatch_plan(
    graph: &ComputeGraph,
    plan: &DispatchPlan,
) -> Result<(), DispatchError> {
    let graph_node_ids: HashSet<ComputeNodeId> =
        graph.nodes.iter().map(|n| n.id).collect();

    // 1. Every graph node must have a target assignment.
    for node in &graph.nodes {
        if !plan.assignment.contains_key(&node.id) {
            return Err(DispatchError::MissingAssignment { node_id: node.id });
        }
    }

    // 2. Every kernel launch must reference a valid graph node.
    for op in &plan.ops {
        match op {
            DispatchOp::KernelLaunch { node_id, .. }
            | DispatchOp::CpuFallback { node_id, .. } => {
                if !graph_node_ids.contains(node_id) {
                    return Err(DispatchError::UnknownNode { node_id: *node_id });
                }
            }
            _ => {}
        }
    }

    // 3. Dependency satisfaction: track which nodes have been executed.
    let mut executed: HashSet<ComputeNodeId> = HashSet::new();
    // Track which nodes have had their data transferred to specific targets.
    let mut transferred: HashSet<(ComputeNodeId, ComputeTarget)> = HashSet::new();
    // Track pending async nodes (launched but not yet synced).
    let mut pending_async: HashSet<ComputeNodeId> = HashSet::new();

    for op in &plan.ops {
        match op {
            DispatchOp::DataTransfer { edge_from, edge_to, dst, .. } => {
                // The producer must have been launched.
                if !executed.contains(edge_from) {
                    return Err(DispatchError::UnsatisfiedDependency {
                        consumer: *edge_to,
                        producer: *edge_from,
                    });
                }
                transferred.insert((*edge_from, *dst));
            }
            DispatchOp::KernelLaunch { node_id, target, .. } => {
                // All predecessors must be executed (and data transferred if cross-target).
                for edge in graph.incoming_edges(*node_id) {
                    if !executed.contains(&edge.from) {
                        return Err(DispatchError::UnsatisfiedDependency {
                            consumer: *node_id,
                            producer: edge.from,
                        });
                    }
                    let src_target = plan.assignment.get(&edge.from).copied()
                        .unwrap_or(ComputeTarget::CpuScalar);
                    if src_target != *target && edge.transfer_bytes > 0 {
                        if !transferred.contains(&(edge.from, *target)) {
                            return Err(DispatchError::UnsatisfiedDependency {
                                consumer: *node_id,
                                producer: edge.from,
                            });
                        }
                    }
                    // Check async sync requirement.
                    if is_async_target(src_target) && pending_async.contains(&edge.from) {
                        return Err(DispatchError::MissingSyncBeforeRead {
                            target: src_target,
                            producer: edge.from,
                            consumer: *node_id,
                        });
                    }
                }
                executed.insert(*node_id);
                if is_async_target(*target) {
                    pending_async.insert(*node_id);
                }
            }
            DispatchOp::Synchronize { node_id, .. } => {
                pending_async.remove(node_id);
            }
            DispatchOp::CpuFallback { node_id, .. } => {
                // Same dependency check as KernelLaunch on CpuScalar.
                for edge in graph.incoming_edges(*node_id) {
                    if !executed.contains(&edge.from) {
                        return Err(DispatchError::UnsatisfiedDependency {
                            consumer: *node_id,
                            producer: edge.from,
                        });
                    }
                }
                executed.insert(*node_id);
            }
        }
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Dispatch plan property verification
// ---------------------------------------------------------------------------

/// Report from verifying all dispatch plan properties.
///
/// Collects all violations rather than returning on the first error,
/// giving a complete picture of what is wrong with a plan.
#[derive(Debug, Clone)]
pub struct VerificationReport {
    /// Whether every node in the graph has CpuScalar in its legal_targets.
    pub cpu_fallback_ok: bool,
    /// Nodes missing CpuScalar from their legal_targets.
    pub missing_cpu_fallback: Vec<ComputeNodeId>,

    /// Whether every cross-target edge has a corresponding DataTransfer op.
    pub data_transfers_ok: bool,
    /// Descriptions of missing or incorrect data transfers.
    pub data_transfer_errors: Vec<String>,

    /// Whether synchronization is correct (async results synced before use).
    pub synchronization_ok: bool,
    /// Descriptions of synchronization violations.
    pub synchronization_errors: Vec<String>,

    /// Whether the dependency ordering in the plan matches the graph DAG.
    pub dependency_order_ok: bool,
    /// Descriptions of dependency ordering violations.
    pub dependency_errors: Vec<String>,
}

impl VerificationReport {
    /// Returns true if all properties passed verification.
    pub fn all_ok(&self) -> bool {
        self.cpu_fallback_ok
            && self.data_transfers_ok
            && self.synchronization_ok
            && self.dependency_order_ok
    }

    /// Total number of individual violations across all properties.
    pub fn total_violations(&self) -> usize {
        self.missing_cpu_fallback.len()
            + self.data_transfer_errors.len()
            + self.synchronization_errors.len()
            + self.dependency_errors.len()
    }
}

impl fmt::Display for VerificationReport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "VerificationReport:")?;
        writeln!(f, "  CPU fallback: {}", if self.cpu_fallback_ok { "PASS" } else { "FAIL" })?;
        for nid in &self.missing_cpu_fallback {
            writeln!(f, "    - {} missing CpuScalar in legal_targets", nid)?;
        }
        writeln!(f, "  Data transfers: {}", if self.data_transfers_ok { "PASS" } else { "FAIL" })?;
        for msg in &self.data_transfer_errors {
            writeln!(f, "    - {}", msg)?;
        }
        writeln!(f, "  Synchronization: {}", if self.synchronization_ok { "PASS" } else { "FAIL" })?;
        for msg in &self.synchronization_errors {
            writeln!(f, "    - {}", msg)?;
        }
        writeln!(f, "  Dependency order: {}", if self.dependency_order_ok { "PASS" } else { "FAIL" })?;
        for msg in &self.dependency_errors {
            writeln!(f, "    - {}", msg)?;
        }
        Ok(())
    }
}

/// Verify that every node in the compute graph has CpuScalar in its
/// legal_targets (the "CPU fallback liveness" property).
///
/// This ensures compilation always has a valid target assignment:
/// even if no proof annotations are present, every node can run on
/// the CPU scalar path.
pub fn verify_cpu_fallback(graph: &ComputeGraph, _plan: &DispatchPlan) -> bool {
    graph.nodes.iter().all(|node| {
        node.legal_targets.contains(&ComputeTarget::CpuScalar)
    })
}

/// Verify that every cross-target edge in the plan has a corresponding
/// DataTransfer operation ("data transfer completeness" property).
///
/// For every edge where the source and destination nodes are assigned
/// to different targets and transfer_bytes > 0, there must be a
/// DataTransfer op in the plan that covers that edge.
pub fn verify_data_transfers(
    graph: &ComputeGraph,
    plan: &DispatchPlan,
) -> Result<(), Vec<String>> {
    let mut errors = Vec::new();

    // Build set of (from, to) pairs covered by DataTransfer ops.
    let covered: HashSet<(ComputeNodeId, ComputeNodeId)> = plan
        .ops
        .iter()
        .filter_map(|op| {
            if let DispatchOp::DataTransfer { edge_from, edge_to, .. } = op {
                Some((*edge_from, *edge_to))
            } else {
                None
            }
        })
        .collect();

    for edge in &graph.edges {
        let src_target = plan.assignment.get(&edge.from).copied();
        let dst_target = plan.assignment.get(&edge.to).copied();

        // Skip edges where either node lacks assignment (caught by other checks).
        let (src, dst) = match (src_target, dst_target) {
            (Some(s), Some(d)) => (s, d),
            _ => continue,
        };

        // Cross-target edge with data to move must have a transfer op.
        if src != dst && edge.transfer_bytes > 0 {
            if !covered.contains(&(edge.from, edge.to)) {
                errors.push(format!(
                    "Missing DataTransfer for edge {} -> {} ({} -> {}, {} bytes)",
                    edge.from, edge.to, src, dst, edge.transfer_bytes
                ));
            }
        }
    }

    if errors.is_empty() {
        Ok(())
    } else {
        Err(errors)
    }
}

/// Verify that synchronization points are sufficient ("synchronization
/// sufficiency" property).
///
/// Every node launched on an async target (GPU, NeuralEngine) must be
/// synchronized before its results are consumed by a node on a different
/// target. This walks the plan ops in order and tracks pending async
/// launches and sync points.
pub fn verify_synchronization(
    graph: &ComputeGraph,
    plan: &DispatchPlan,
) -> Result<(), Vec<String>> {
    let mut errors = Vec::new();

    // Track nodes that have been launched on async targets but not yet synced.
    let mut pending_async: HashSet<ComputeNodeId> = HashSet::new();
    // Track which nodes have been executed (launched or fallback).
    let mut executed: HashSet<ComputeNodeId> = HashSet::new();

    for op in &plan.ops {
        match op {
            DispatchOp::Synchronize { node_id, .. } => {
                pending_async.remove(node_id);
            }
            DispatchOp::KernelLaunch { target, node_id, .. } => {
                // Check that all predecessors on async targets have been synced.
                for edge in graph.incoming_edges(*node_id) {
                    let src_target = plan.assignment.get(&edge.from).copied()
                        .unwrap_or(ComputeTarget::CpuScalar);
                    if is_async_target(src_target) && pending_async.contains(&edge.from) {
                        errors.push(format!(
                            "Node {} consumes unsynced async result from {} (target: {})",
                            node_id, edge.from, src_target
                        ));
                    }
                }
                executed.insert(*node_id);
                if is_async_target(*target) {
                    pending_async.insert(*node_id);
                }
            }
            DispatchOp::DataTransfer { edge_from, edge_to, .. } => {
                // A transfer from an async node that hasn't been synced is an error.
                let src_target = plan.assignment.get(edge_from).copied()
                    .unwrap_or(ComputeTarget::CpuScalar);
                if is_async_target(src_target) && pending_async.contains(edge_from) {
                    errors.push(format!(
                        "DataTransfer from unsynced async node {} to {} (target: {})",
                        edge_from, edge_to, src_target
                    ));
                }
            }
            DispatchOp::CpuFallback { node_id, .. } => {
                // Check incoming edges from async sources.
                for edge in graph.incoming_edges(*node_id) {
                    let src_target = plan.assignment.get(&edge.from).copied()
                        .unwrap_or(ComputeTarget::CpuScalar);
                    if is_async_target(src_target) && pending_async.contains(&edge.from) {
                        errors.push(format!(
                            "CpuFallback {} consumes unsynced async result from {} (target: {})",
                            node_id, edge.from, src_target
                        ));
                    }
                }
                executed.insert(*node_id);
            }
        }
    }

    if errors.is_empty() {
        Ok(())
    } else {
        Err(errors)
    }
}

/// Verify all dispatch plan properties and return a comprehensive report.
///
/// This checks:
/// 1. **CPU fallback liveness**: every node has CpuScalar in legal_targets
/// 2. **Data transfer completeness**: every cross-target edge has a transfer op
/// 3. **Synchronization sufficiency**: async results are synced before consumption
/// 4. **Dependency ordering**: the plan respects the graph's data dependencies
///
/// Unlike [`validate_dispatch_plan`] which returns on the first error, this
/// function collects ALL violations into a [`VerificationReport`].
pub fn verify_dispatch_plan_properties(
    graph: &ComputeGraph,
    plan: &DispatchPlan,
) -> VerificationReport {
    // Property 1: CPU fallback liveness.
    let missing_cpu_fallback: Vec<ComputeNodeId> = graph
        .nodes
        .iter()
        .filter(|node| !node.legal_targets.contains(&ComputeTarget::CpuScalar))
        .map(|node| node.id)
        .collect();
    let cpu_fallback_ok = missing_cpu_fallback.is_empty();

    // Property 2: Data transfer completeness.
    let (data_transfers_ok, data_transfer_errors) = match verify_data_transfers(graph, plan) {
        Ok(()) => (true, Vec::new()),
        Err(errors) => (false, errors),
    };

    // Property 3: Synchronization sufficiency.
    let (synchronization_ok, synchronization_errors) =
        match verify_synchronization(graph, plan) {
            Ok(()) => (true, Vec::new()),
            Err(errors) => (false, errors),
        };

    // Property 4: Dependency ordering.
    let mut dependency_errors = Vec::new();
    let mut executed: HashSet<ComputeNodeId> = HashSet::new();
    for op in &plan.ops {
        match op {
            DispatchOp::KernelLaunch { node_id, .. }
            | DispatchOp::CpuFallback { node_id, .. } => {
                for edge in graph.incoming_edges(*node_id) {
                    if !executed.contains(&edge.from) {
                        dependency_errors.push(format!(
                            "Node {} launched before predecessor {} is executed",
                            node_id, edge.from
                        ));
                    }
                }
                executed.insert(*node_id);
            }
            _ => {}
        }
    }
    let dependency_order_ok = dependency_errors.is_empty();

    VerificationReport {
        cpu_fallback_ok,
        missing_cpu_fallback,
        data_transfers_ok,
        data_transfer_errors,
        synchronization_ok,
        synchronization_errors,
        dependency_order_ok,
        dependency_errors,
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compute_graph::{ComputeCost, ComputeNode, DataEdge, NodeKind};

    // -----------------------------------------------------------------------
    // Test helpers
    // -----------------------------------------------------------------------

    /// Build a single-node scalar graph (no edges).
    fn scalar_graph() -> (ComputeGraph, Vec<TargetRecommendation>) {
        let mut costs = HashMap::new();
        costs.insert(ComputeTarget::CpuScalar, ComputeCost {
            latency_cycles: 10,
            throughput_ops_per_kcycle: 1000,
        });

        let node = ComputeNode {
            id: ComputeNodeId(0),
            instructions: vec![],
            costs,
            legal_targets: vec![ComputeTarget::CpuScalar],
            kind: NodeKind::Scalar,
            data_size_bytes: 8,
            produced_values: vec![],
            consumed_values: vec![],
            target_legality: None,
        };

        let graph = ComputeGraph {
            nodes: vec![node],
            edges: vec![],
        };

        let recs = vec![TargetRecommendation {
            node_id: ComputeNodeId(0),
            recommended_target: ComputeTarget::CpuScalar,
            legal_targets: vec![ComputeTarget::CpuScalar],
            reason: "scalar op".to_string(),
            parallel_reduction_legal: false,
        }];

        (graph, recs)
    }

    /// Build a single-node NEON graph.
    fn neon_graph() -> (ComputeGraph, Vec<TargetRecommendation>) {
        let mut costs = HashMap::new();
        costs.insert(ComputeTarget::CpuScalar, ComputeCost {
            latency_cycles: 40,
            throughput_ops_per_kcycle: 1000,
        });
        costs.insert(ComputeTarget::CpuSimd, ComputeCost {
            latency_cycles: 10,
            throughput_ops_per_kcycle: 4000,
        });

        let node = ComputeNode {
            id: ComputeNodeId(0),
            instructions: vec![],
            costs,
            legal_targets: vec![ComputeTarget::CpuScalar, ComputeTarget::CpuSimd],
            kind: NodeKind::DataParallel,
            data_size_bytes: 128,
            produced_values: vec![],
            consumed_values: vec![],
            target_legality: None,
        };

        let graph = ComputeGraph {
            nodes: vec![node],
            edges: vec![],
        };

        let recs = vec![TargetRecommendation {
            node_id: ComputeNodeId(0),
            recommended_target: ComputeTarget::CpuSimd,
            legal_targets: vec![ComputeTarget::CpuScalar, ComputeTarget::CpuSimd],
            reason: "NEON vectorizable".to_string(),
            parallel_reduction_legal: false,
        }];

        (graph, recs)
    }

    /// Build a two-node GPU graph: CPU producer -> GPU consumer with data edge.
    fn gpu_graph() -> (ComputeGraph, Vec<TargetRecommendation>) {
        let mut cpu_costs = HashMap::new();
        cpu_costs.insert(ComputeTarget::CpuScalar, ComputeCost {
            latency_cycles: 20,
            throughput_ops_per_kcycle: 1000,
        });

        let mut gpu_costs = HashMap::new();
        gpu_costs.insert(ComputeTarget::Gpu, ComputeCost {
            latency_cycles: 5,
            throughput_ops_per_kcycle: 100_000,
        });
        gpu_costs.insert(ComputeTarget::CpuScalar, ComputeCost {
            latency_cycles: 500,
            throughput_ops_per_kcycle: 1000,
        });

        let producer = ComputeNode {
            id: ComputeNodeId(0),
            instructions: vec![],
            costs: cpu_costs,
            legal_targets: vec![ComputeTarget::CpuScalar],
            kind: NodeKind::Scalar,
            data_size_bytes: 8,
            produced_values: vec![],
            consumed_values: vec![],
            target_legality: None,
        };

        let consumer = ComputeNode {
            id: ComputeNodeId(1),
            instructions: vec![],
            costs: gpu_costs,
            legal_targets: vec![ComputeTarget::CpuScalar, ComputeTarget::Gpu],
            kind: NodeKind::DataParallel,
            data_size_bytes: 4096,
            produced_values: vec![],
            consumed_values: vec![],
            target_legality: None,
        };

        let edge = DataEdge {
            from: ComputeNodeId(0),
            to: ComputeNodeId(1),
            transfer_bytes: 4096,
            transfer_cost: TransferCost::zero(),
        };

        let graph = ComputeGraph {
            nodes: vec![producer, consumer],
            edges: vec![edge],
        };

        let recs = vec![
            TargetRecommendation {
                node_id: ComputeNodeId(0),
                recommended_target: ComputeTarget::CpuScalar,
                legal_targets: vec![ComputeTarget::CpuScalar],
                reason: "scalar input".to_string(),
                parallel_reduction_legal: false,
            },
            TargetRecommendation {
                node_id: ComputeNodeId(1),
                recommended_target: ComputeTarget::Gpu,
                legal_targets: vec![ComputeTarget::CpuScalar, ComputeTarget::Gpu],
                reason: "data-parallel on GPU".to_string(),
                parallel_reduction_legal: true,
            },
        ];

        (graph, recs)
    }

    /// Build a two-node ANE graph: CPU producer -> ANE consumer.
    fn ane_graph() -> (ComputeGraph, Vec<TargetRecommendation>) {
        let mut cpu_costs = HashMap::new();
        cpu_costs.insert(ComputeTarget::CpuScalar, ComputeCost {
            latency_cycles: 20,
            throughput_ops_per_kcycle: 1000,
        });

        let mut ane_costs = HashMap::new();
        ane_costs.insert(ComputeTarget::NeuralEngine, ComputeCost {
            latency_cycles: 3,
            throughput_ops_per_kcycle: 500_000,
        });
        ane_costs.insert(ComputeTarget::CpuScalar, ComputeCost {
            latency_cycles: 1000,
            throughput_ops_per_kcycle: 1000,
        });

        let producer = ComputeNode {
            id: ComputeNodeId(0),
            instructions: vec![],
            costs: cpu_costs,
            legal_targets: vec![ComputeTarget::CpuScalar],
            kind: NodeKind::Scalar,
            data_size_bytes: 8,
            produced_values: vec![],
            consumed_values: vec![],
            target_legality: None,
        };

        let consumer = ComputeNode {
            id: ComputeNodeId(1),
            instructions: vec![],
            costs: ane_costs,
            legal_targets: vec![ComputeTarget::CpuScalar, ComputeTarget::NeuralEngine],
            kind: NodeKind::MatrixHeavy,
            data_size_bytes: 16384,
            produced_values: vec![],
            consumed_values: vec![],
            target_legality: None,
        };

        let edge = DataEdge {
            from: ComputeNodeId(0),
            to: ComputeNodeId(1),
            transfer_bytes: 16384,
            transfer_cost: TransferCost::zero(),
        };

        let graph = ComputeGraph {
            nodes: vec![producer, consumer],
            edges: vec![edge],
        };

        let recs = vec![
            TargetRecommendation {
                node_id: ComputeNodeId(0),
                recommended_target: ComputeTarget::CpuScalar,
                legal_targets: vec![ComputeTarget::CpuScalar],
                reason: "scalar input".to_string(),
                parallel_reduction_legal: false,
            },
            TargetRecommendation {
                node_id: ComputeNodeId(1),
                recommended_target: ComputeTarget::NeuralEngine,
                legal_targets: vec![ComputeTarget::CpuScalar, ComputeTarget::NeuralEngine],
                reason: "matrix-heavy on ANE".to_string(),
                parallel_reduction_legal: false,
            },
        ];

        (graph, recs)
    }

    /// Build a mixed three-node graph: CPU -> GPU -> CPU readback.
    fn mixed_graph() -> (ComputeGraph, Vec<TargetRecommendation>) {
        let mut cpu_costs = HashMap::new();
        cpu_costs.insert(ComputeTarget::CpuScalar, ComputeCost {
            latency_cycles: 10,
            throughput_ops_per_kcycle: 1000,
        });

        let mut gpu_costs = HashMap::new();
        gpu_costs.insert(ComputeTarget::Gpu, ComputeCost {
            latency_cycles: 5,
            throughput_ops_per_kcycle: 100_000,
        });
        gpu_costs.insert(ComputeTarget::CpuScalar, ComputeCost {
            latency_cycles: 500,
            throughput_ops_per_kcycle: 1000,
        });

        let mut readback_costs = HashMap::new();
        readback_costs.insert(ComputeTarget::CpuScalar, ComputeCost {
            latency_cycles: 15,
            throughput_ops_per_kcycle: 1000,
        });

        let node0 = ComputeNode {
            id: ComputeNodeId(0),
            instructions: vec![],
            costs: cpu_costs,
            legal_targets: vec![ComputeTarget::CpuScalar],
            kind: NodeKind::Scalar,
            data_size_bytes: 8,
            produced_values: vec![],
            consumed_values: vec![],
            target_legality: None,
        };

        let node1 = ComputeNode {
            id: ComputeNodeId(1),
            instructions: vec![],
            costs: gpu_costs,
            legal_targets: vec![ComputeTarget::CpuScalar, ComputeTarget::Gpu],
            kind: NodeKind::DataParallel,
            data_size_bytes: 4096,
            produced_values: vec![],
            consumed_values: vec![],
            target_legality: None,
        };

        let node2 = ComputeNode {
            id: ComputeNodeId(2),
            instructions: vec![],
            costs: readback_costs,
            legal_targets: vec![ComputeTarget::CpuScalar],
            kind: NodeKind::Scalar,
            data_size_bytes: 64,
            produced_values: vec![],
            consumed_values: vec![],
            target_legality: None,
        };

        let edges = vec![
            DataEdge {
                from: ComputeNodeId(0),
                to: ComputeNodeId(1),
                transfer_bytes: 4096,
                transfer_cost: TransferCost::zero(),
            },
            DataEdge {
                from: ComputeNodeId(1),
                to: ComputeNodeId(2),
                transfer_bytes: 64,
                transfer_cost: TransferCost::zero(),
            },
        ];

        let graph = ComputeGraph {
            nodes: vec![node0, node1, node2],
            edges,
        };

        let recs = vec![
            TargetRecommendation {
                node_id: ComputeNodeId(0),
                recommended_target: ComputeTarget::CpuScalar,
                legal_targets: vec![ComputeTarget::CpuScalar],
                reason: "scalar input".to_string(),
                parallel_reduction_legal: false,
            },
            TargetRecommendation {
                node_id: ComputeNodeId(1),
                recommended_target: ComputeTarget::Gpu,
                legal_targets: vec![ComputeTarget::CpuScalar, ComputeTarget::Gpu],
                reason: "GPU compute".to_string(),
                parallel_reduction_legal: true,
            },
            TargetRecommendation {
                node_id: ComputeNodeId(2),
                recommended_target: ComputeTarget::CpuScalar,
                legal_targets: vec![ComputeTarget::CpuScalar],
                reason: "scalar readback".to_string(),
                parallel_reduction_legal: false,
            },
        ];

        (graph, recs)
    }

    // -----------------------------------------------------------------------
    // Test 1: Simple scalar dispatch (no transfers needed)
    // -----------------------------------------------------------------------

    #[test]
    fn test_scalar_dispatch_no_transfers() {
        let (graph, recs) = scalar_graph();
        let plan = generate_dispatch_plan(&graph, &recs);

        assert_eq!(plan.count_transfers(), 0, "scalar dispatch needs no transfers");
        assert_eq!(plan.count_launches(), 1, "one kernel launch");
        assert_eq!(plan.count_syncs(), 0, "no syncs for CPU scalar");
        assert_eq!(plan.count_fallbacks(), 0, "no fallbacks");

        // Validate passes.
        assert!(validate_dispatch_plan(&graph, &plan).is_ok());

        // Check the launch target.
        match &plan.ops[0] {
            DispatchOp::KernelLaunch { target, node_id, .. } => {
                assert_eq!(*target, ComputeTarget::CpuScalar);
                assert_eq!(*node_id, ComputeNodeId(0));
            }
            other => panic!("Expected KernelLaunch, got {:?}", other),
        }
    }

    // -----------------------------------------------------------------------
    // Test 2: NEON dispatch (register move only, no heavy transfer)
    // -----------------------------------------------------------------------

    #[test]
    fn test_neon_dispatch_no_heavy_transfer() {
        let (graph, recs) = neon_graph();
        let plan = generate_dispatch_plan(&graph, &recs);

        assert_eq!(plan.count_transfers(), 0, "NEON has no cross-domain transfer for standalone");
        assert_eq!(plan.count_launches(), 1);
        assert_eq!(plan.count_syncs(), 0, "NEON is synchronous");

        match &plan.ops[0] {
            DispatchOp::KernelLaunch { target, .. } => {
                assert_eq!(*target, ComputeTarget::CpuSimd);
            }
            other => panic!("Expected KernelLaunch, got {:?}", other),
        }

        assert!(validate_dispatch_plan(&graph, &plan).is_ok());
    }

    // -----------------------------------------------------------------------
    // Test 3: GPU dispatch (data transfer + kernel + sync + transfer back)
    // -----------------------------------------------------------------------

    #[test]
    fn test_gpu_dispatch_full_sequence() {
        let (graph, recs) = gpu_graph();
        let plan = generate_dispatch_plan(&graph, &recs);

        // Should have: CPU launch, transfer CPU->GPU, GPU launch, GPU sync.
        assert_eq!(plan.count_launches(), 2, "two kernel launches");
        assert_eq!(plan.count_transfers(), 1, "one CPU->GPU transfer");
        assert!(plan.count_syncs() >= 1, "at least one GPU sync");

        // Validate dependency order.
        assert!(validate_dispatch_plan(&graph, &plan).is_ok());

        // Verify transfer goes CPU -> GPU.
        let transfer = plan.ops.iter().find(|op| matches!(op, DispatchOp::DataTransfer { .. }));
        assert!(transfer.is_some(), "must have a data transfer");
        if let Some(DispatchOp::DataTransfer { src, dst, size_bytes, .. }) = transfer {
            assert_eq!(*src, ComputeTarget::CpuScalar);
            assert_eq!(*dst, ComputeTarget::Gpu);
            assert_eq!(*size_bytes, 4096);
        }
    }

    // -----------------------------------------------------------------------
    // Test 4: ANE dispatch (similar to GPU)
    // -----------------------------------------------------------------------

    #[test]
    fn test_ane_dispatch_full_sequence() {
        let (graph, recs) = ane_graph();
        let plan = generate_dispatch_plan(&graph, &recs);

        assert_eq!(plan.count_launches(), 2);
        assert_eq!(plan.count_transfers(), 1, "one CPU->ANE transfer");
        assert!(plan.count_syncs() >= 1, "at least one ANE sync");

        assert!(validate_dispatch_plan(&graph, &plan).is_ok());

        let transfer = plan.ops.iter().find(|op| matches!(op, DispatchOp::DataTransfer { .. }));
        assert!(transfer.is_some());
        if let Some(DispatchOp::DataTransfer { src, dst, size_bytes, .. }) = transfer {
            assert_eq!(*src, ComputeTarget::CpuScalar);
            assert_eq!(*dst, ComputeTarget::NeuralEngine);
            assert_eq!(*size_bytes, 16384);
        }
    }

    // -----------------------------------------------------------------------
    // Test 5: Mixed dispatch plan validation (CPU -> GPU -> CPU)
    // -----------------------------------------------------------------------

    #[test]
    fn test_mixed_dispatch_plan_valid() {
        let (graph, recs) = mixed_graph();
        let plan = generate_dispatch_plan(&graph, &recs);

        // Three nodes: CPU, GPU, CPU.
        assert_eq!(plan.count_launches(), 3);
        // Two cross-target edges: CPU->GPU and GPU->CPU.
        assert_eq!(plan.count_transfers(), 2);
        // GPU results need sync before CPU readback.
        assert!(plan.count_syncs() >= 1);

        assert!(validate_dispatch_plan(&graph, &plan).is_ok());

        // Verify total estimated cost is reasonable (positive).
        assert!(plan.estimated_total_cycles > 0);
    }

    // -----------------------------------------------------------------------
    // Test 6: Invalid plan detection — missing dependency
    // -----------------------------------------------------------------------

    #[test]
    fn test_invalid_plan_missing_dependency() {
        let (graph, _recs) = gpu_graph();

        // Manually construct an invalid plan: launch GPU node without
        // launching CPU producer first.
        let mut assignment = HashMap::new();
        assignment.insert(ComputeNodeId(0), ComputeTarget::CpuScalar);
        assignment.insert(ComputeNodeId(1), ComputeTarget::Gpu);

        let plan = DispatchPlan {
            ops: vec![
                // Skip node 0 launch — go straight to GPU launch.
                DispatchOp::KernelLaunch {
                    target: ComputeTarget::Gpu,
                    node_id: ComputeNodeId(1),
                    estimated_cycles: 5,
                },
            ],
            assignment,
            estimated_total_cycles: 5,
        };

        let result = validate_dispatch_plan(&graph, &plan);
        assert!(result.is_err(), "should detect missing dependency");
        match result.unwrap_err() {
            DispatchError::UnsatisfiedDependency { consumer, producer } => {
                assert_eq!(consumer, ComputeNodeId(1));
                assert_eq!(producer, ComputeNodeId(0));
            }
            other => panic!("Expected UnsatisfiedDependency, got {:?}", other),
        }
    }

    // -----------------------------------------------------------------------
    // Test 7: Invalid plan detection — missing assignment
    // -----------------------------------------------------------------------

    #[test]
    fn test_invalid_plan_missing_assignment() {
        let (graph, _recs) = gpu_graph();

        // Only assign node 0, not node 1.
        let mut assignment = HashMap::new();
        assignment.insert(ComputeNodeId(0), ComputeTarget::CpuScalar);

        let plan = DispatchPlan {
            ops: vec![],
            assignment,
            estimated_total_cycles: 0,
        };

        let result = validate_dispatch_plan(&graph, &plan);
        assert!(result.is_err());
        match result.unwrap_err() {
            DispatchError::MissingAssignment { node_id } => {
                assert_eq!(node_id, ComputeNodeId(1));
            }
            other => panic!("Expected MissingAssignment, got {:?}", other),
        }
    }

    // -----------------------------------------------------------------------
    // Test 8: Empty graph produces empty plan
    // -----------------------------------------------------------------------

    #[test]
    fn test_empty_graph_produces_empty_plan() {
        let graph = ComputeGraph::new();
        let recs = vec![];
        let plan = generate_dispatch_plan(&graph, &recs);

        assert!(plan.is_empty());
        assert_eq!(plan.estimated_total_cycles, 0);
        assert!(validate_dispatch_plan(&graph, &plan).is_ok());
    }

    // -----------------------------------------------------------------------
    // Test 9: Plan display format
    // -----------------------------------------------------------------------

    #[test]
    fn test_plan_display_format() {
        let (graph, recs) = gpu_graph();
        let plan = generate_dispatch_plan(&graph, &recs);

        let display = format!("{}", plan);
        assert!(display.contains("DispatchPlan"), "should contain header");
        assert!(display.contains("Launch"), "should contain Launch ops");
    }

    // -----------------------------------------------------------------------
    // Test 10: GPU sync is emitted before cross-target consumption
    // -----------------------------------------------------------------------

    #[test]
    fn test_gpu_sync_before_readback() {
        let (graph, recs) = mixed_graph();
        let plan = generate_dispatch_plan(&graph, &recs);

        // Find the GPU sync and the CPU readback launch.
        let mut gpu_sync_idx = None;
        let mut cpu_readback_idx = None;

        for (i, op) in plan.ops.iter().enumerate() {
            match op {
                DispatchOp::Synchronize { target: ComputeTarget::Gpu, .. } => {
                    gpu_sync_idx = Some(i);
                }
                DispatchOp::KernelLaunch { node_id: ComputeNodeId(2), .. } => {
                    cpu_readback_idx = Some(i);
                }
                _ => {}
            }
        }

        assert!(gpu_sync_idx.is_some(), "GPU sync must be present");
        assert!(cpu_readback_idx.is_some(), "CPU readback launch must be present");
        assert!(
            gpu_sync_idx.unwrap() < cpu_readback_idx.unwrap(),
            "GPU sync (idx {}) must come before CPU readback (idx {})",
            gpu_sync_idx.unwrap(),
            cpu_readback_idx.unwrap()
        );
    }

    // -----------------------------------------------------------------------
    // Test 11: Transfer cost is included in total estimate
    // -----------------------------------------------------------------------

    #[test]
    fn test_transfer_cost_in_total_estimate() {
        let (graph, recs) = gpu_graph();
        let plan = generate_dispatch_plan(&graph, &recs);

        // The total should be more than just compute costs alone.
        // CPU node: 20 cycles, GPU node: 5 cycles. Transfer adds overhead.
        assert!(
            plan.estimated_total_cycles > 25,
            "Total {} should exceed pure compute (25) due to transfer cost",
            plan.estimated_total_cycles
        );
    }

    // -----------------------------------------------------------------------
    // Test 12: Topological order respects dependencies
    // -----------------------------------------------------------------------

    #[test]
    fn test_topological_order_dependencies() {
        let (graph, _recs) = mixed_graph();
        let order = topological_order(&graph);

        assert_eq!(order.len(), 3);
        // Node 0 must come before Node 1, Node 1 before Node 2.
        let pos0 = order.iter().position(|id| id.0 == 0).unwrap();
        let pos1 = order.iter().position(|id| id.0 == 1).unwrap();
        let pos2 = order.iter().position(|id| id.0 == 2).unwrap();
        assert!(pos0 < pos1, "node 0 must precede node 1");
        assert!(pos1 < pos2, "node 1 must precede node 2");
    }

    // -----------------------------------------------------------------------
    // Test 13: DispatchOp Display implementations
    // -----------------------------------------------------------------------

    #[test]
    fn test_dispatch_op_display() {
        let transfer = DispatchOp::DataTransfer {
            src: ComputeTarget::CpuScalar,
            dst: ComputeTarget::Gpu,
            size_bytes: 1024,
            cost: TransferCost::zero(),
            edge_from: ComputeNodeId(0),
            edge_to: ComputeNodeId(1),
        };
        let s = format!("{}", transfer);
        assert!(s.contains("Transfer"));
        assert!(s.contains("1024"));

        let launch = DispatchOp::KernelLaunch {
            target: ComputeTarget::Gpu,
            node_id: ComputeNodeId(1),
            estimated_cycles: 100,
        };
        let s = format!("{}", launch);
        assert!(s.contains("Launch"));
        assert!(s.contains("100"));

        let sync = DispatchOp::Synchronize {
            target: ComputeTarget::Gpu,
            node_id: ComputeNodeId(1),
        };
        let s = format!("{}", sync);
        assert!(s.contains("Sync"));

        let fallback = DispatchOp::CpuFallback {
            node_id: ComputeNodeId(0),
            reason: "test reason".to_string(),
        };
        let s = format!("{}", fallback);
        assert!(s.contains("CpuFallback"));
        assert!(s.contains("test reason"));
    }

    // -----------------------------------------------------------------------
    // Test 14: DispatchError Display implementations
    // -----------------------------------------------------------------------

    #[test]
    fn test_dispatch_error_display() {
        let err = DispatchError::MissingAssignment { node_id: ComputeNodeId(5) };
        assert!(format!("{}", err).contains("node_5"));

        let err = DispatchError::UnsatisfiedDependency {
            consumer: ComputeNodeId(1),
            producer: ComputeNodeId(0),
        };
        assert!(format!("{}", err).contains("node_1"));
        assert!(format!("{}", err).contains("node_0"));

        let err = DispatchError::UnknownNode { node_id: ComputeNodeId(99) };
        assert!(format!("{}", err).contains("node_99"));
    }

    // -----------------------------------------------------------------------
    // Test 15: Unknown node detection
    // -----------------------------------------------------------------------

    #[test]
    fn test_unknown_node_in_plan() {
        let (graph, _recs) = scalar_graph();

        let mut assignment = HashMap::new();
        assignment.insert(ComputeNodeId(0), ComputeTarget::CpuScalar);

        let plan = DispatchPlan {
            ops: vec![
                DispatchOp::KernelLaunch {
                    target: ComputeTarget::CpuScalar,
                    node_id: ComputeNodeId(0),
                    estimated_cycles: 10,
                },
                DispatchOp::KernelLaunch {
                    target: ComputeTarget::CpuScalar,
                    node_id: ComputeNodeId(99), // Not in graph.
                    estimated_cycles: 10,
                },
            ],
            assignment,
            estimated_total_cycles: 20,
        };

        let result = validate_dispatch_plan(&graph, &plan);
        assert!(result.is_err());
        match result.unwrap_err() {
            DispatchError::UnknownNode { node_id } => {
                assert_eq!(node_id, ComputeNodeId(99));
            }
            other => panic!("Expected UnknownNode, got {:?}", other),
        }
    }

    // ===================================================================
    // Dispatch plan property verification tests
    // ===================================================================

    // -----------------------------------------------------------------------
    // Test V1: Valid plan passes all property checks
    // -----------------------------------------------------------------------

    #[test]
    fn test_verify_valid_plan_passes_all_properties() {
        let (graph, recs) = mixed_graph();
        let plan = generate_dispatch_plan(&graph, &recs);

        let report = verify_dispatch_plan_properties(&graph, &plan);
        assert!(report.all_ok(), "Valid plan should pass all checks:\n{}", report);
        assert_eq!(report.total_violations(), 0);
    }

    // -----------------------------------------------------------------------
    // Test V2: CPU fallback detected as missing
    // -----------------------------------------------------------------------

    #[test]
    fn test_verify_cpu_fallback_missing_detected() {
        // Build a graph where one node is missing CpuScalar from legal_targets.
        let mut costs = HashMap::new();
        costs.insert(ComputeTarget::Gpu, ComputeCost {
            latency_cycles: 5,
            throughput_ops_per_kcycle: 100_000,
        });

        let node = ComputeNode {
            id: ComputeNodeId(0),
            instructions: vec![],
            costs,
            legal_targets: vec![ComputeTarget::Gpu], // Missing CpuScalar!
            kind: NodeKind::DataParallel,
            data_size_bytes: 4096,
            produced_values: vec![],
            consumed_values: vec![],
            target_legality: None,
        };

        let graph = ComputeGraph {
            nodes: vec![node],
            edges: vec![],
        };

        let recs = vec![TargetRecommendation {
            node_id: ComputeNodeId(0),
            recommended_target: ComputeTarget::Gpu,
            legal_targets: vec![ComputeTarget::Gpu],
            reason: "GPU only".to_string(),
            parallel_reduction_legal: false,
        }];

        let plan = generate_dispatch_plan(&graph, &recs);

        // verify_cpu_fallback should return false.
        assert!(!verify_cpu_fallback(&graph, &plan));

        // Full report should flag the issue.
        let report = verify_dispatch_plan_properties(&graph, &plan);
        assert!(!report.cpu_fallback_ok);
        assert_eq!(report.missing_cpu_fallback.len(), 1);
        assert_eq!(report.missing_cpu_fallback[0], ComputeNodeId(0));
    }

    // -----------------------------------------------------------------------
    // Test V3: Missing data transfer detected
    // -----------------------------------------------------------------------

    #[test]
    fn test_verify_missing_data_transfer_detected() {
        let (graph, _recs) = gpu_graph();

        // Manually construct a plan WITHOUT the required CPU->GPU transfer.
        let mut assignment = HashMap::new();
        assignment.insert(ComputeNodeId(0), ComputeTarget::CpuScalar);
        assignment.insert(ComputeNodeId(1), ComputeTarget::Gpu);

        let plan = DispatchPlan {
            ops: vec![
                DispatchOp::KernelLaunch {
                    target: ComputeTarget::CpuScalar,
                    node_id: ComputeNodeId(0),
                    estimated_cycles: 20,
                },
                // Missing DataTransfer here!
                DispatchOp::KernelLaunch {
                    target: ComputeTarget::Gpu,
                    node_id: ComputeNodeId(1),
                    estimated_cycles: 5,
                },
            ],
            assignment,
            estimated_total_cycles: 25,
        };

        let result = verify_data_transfers(&graph, &plan);
        assert!(result.is_err());
        let errors = result.unwrap_err();
        assert_eq!(errors.len(), 1);
        assert!(errors[0].contains("Missing DataTransfer"));
        assert!(errors[0].contains("node_0"));
        assert!(errors[0].contains("node_1"));
    }

    // -----------------------------------------------------------------------
    // Test V4: Missing synchronization detected
    // -----------------------------------------------------------------------

    #[test]
    fn test_verify_missing_synchronization_detected() {
        let (graph, _recs) = mixed_graph();

        // Construct a plan where GPU node launches but there's no Sync
        // before the CPU readback consumes GPU results.
        let mut assignment = HashMap::new();
        assignment.insert(ComputeNodeId(0), ComputeTarget::CpuScalar);
        assignment.insert(ComputeNodeId(1), ComputeTarget::Gpu);
        assignment.insert(ComputeNodeId(2), ComputeTarget::CpuScalar);

        let plan = DispatchPlan {
            ops: vec![
                DispatchOp::KernelLaunch {
                    target: ComputeTarget::CpuScalar,
                    node_id: ComputeNodeId(0),
                    estimated_cycles: 10,
                },
                DispatchOp::DataTransfer {
                    src: ComputeTarget::CpuScalar,
                    dst: ComputeTarget::Gpu,
                    size_bytes: 4096,
                    cost: TransferCost::zero(),
                    edge_from: ComputeNodeId(0),
                    edge_to: ComputeNodeId(1),
                },
                DispatchOp::KernelLaunch {
                    target: ComputeTarget::Gpu,
                    node_id: ComputeNodeId(1),
                    estimated_cycles: 5,
                },
                // Missing Sync for GPU node 1!
                DispatchOp::DataTransfer {
                    src: ComputeTarget::Gpu,
                    dst: ComputeTarget::CpuScalar,
                    size_bytes: 64,
                    cost: TransferCost::zero(),
                    edge_from: ComputeNodeId(1),
                    edge_to: ComputeNodeId(2),
                },
                DispatchOp::KernelLaunch {
                    target: ComputeTarget::CpuScalar,
                    node_id: ComputeNodeId(2),
                    estimated_cycles: 15,
                },
            ],
            assignment,
            estimated_total_cycles: 30,
        };

        let result = verify_synchronization(&graph, &plan);
        assert!(result.is_err());
        let errors = result.unwrap_err();
        assert!(!errors.is_empty());
        assert!(errors[0].contains("unsynced"));
    }

    // -----------------------------------------------------------------------
    // Test V5: Scalar plan has correct verification report
    // -----------------------------------------------------------------------

    #[test]
    fn test_verify_scalar_plan_all_pass() {
        let (graph, recs) = scalar_graph();
        let plan = generate_dispatch_plan(&graph, &recs);

        let report = verify_dispatch_plan_properties(&graph, &plan);
        assert!(report.all_ok());
        assert!(report.cpu_fallback_ok);
        assert!(report.data_transfers_ok);
        assert!(report.synchronization_ok);
        assert!(report.dependency_order_ok);
    }

    // -----------------------------------------------------------------------
    // Test V6: GPU plan has correct verification report
    // -----------------------------------------------------------------------

    #[test]
    fn test_verify_gpu_plan_all_pass() {
        let (graph, recs) = gpu_graph();
        let plan = generate_dispatch_plan(&graph, &recs);

        let report = verify_dispatch_plan_properties(&graph, &plan);
        assert!(report.all_ok(), "GPU plan should pass all checks:\n{}", report);
    }

    // -----------------------------------------------------------------------
    // Test V7: ANE plan has correct verification report
    // -----------------------------------------------------------------------

    #[test]
    fn test_verify_ane_plan_all_pass() {
        let (graph, recs) = ane_graph();
        let plan = generate_dispatch_plan(&graph, &recs);

        let report = verify_dispatch_plan_properties(&graph, &plan);
        assert!(report.all_ok(), "ANE plan should pass all checks:\n{}", report);
    }

    // -----------------------------------------------------------------------
    // Test V8: Dependency ordering violation detected
    // -----------------------------------------------------------------------

    #[test]
    fn test_verify_dependency_order_violation_detected() {
        let (graph, _recs) = gpu_graph();

        // Construct a plan where node 1 launches before node 0 (wrong order).
        let mut assignment = HashMap::new();
        assignment.insert(ComputeNodeId(0), ComputeTarget::CpuScalar);
        assignment.insert(ComputeNodeId(1), ComputeTarget::CpuScalar);

        let plan = DispatchPlan {
            ops: vec![
                // Launch consumer before producer.
                DispatchOp::KernelLaunch {
                    target: ComputeTarget::CpuScalar,
                    node_id: ComputeNodeId(1),
                    estimated_cycles: 5,
                },
                DispatchOp::KernelLaunch {
                    target: ComputeTarget::CpuScalar,
                    node_id: ComputeNodeId(0),
                    estimated_cycles: 20,
                },
            ],
            assignment,
            estimated_total_cycles: 25,
        };

        let report = verify_dispatch_plan_properties(&graph, &plan);
        assert!(!report.dependency_order_ok);
        assert!(!report.dependency_errors.is_empty());
        assert!(report.dependency_errors[0].contains("node_1"));
        assert!(report.dependency_errors[0].contains("node_0"));
    }

    // -----------------------------------------------------------------------
    // Test V9: Complex mixed-target graph verified correctly
    // -----------------------------------------------------------------------

    #[test]
    fn test_verify_complex_mixed_target_graph() {
        // Build a 4-node diamond graph:
        //   node 0 (CPU) -> node 1 (GPU)
        //   node 0 (CPU) -> node 2 (NEON)
        //   node 1 (GPU) -> node 3 (CPU)
        //   node 2 (NEON) -> node 3 (CPU)
        let mut cpu_costs = HashMap::new();
        cpu_costs.insert(ComputeTarget::CpuScalar, ComputeCost {
            latency_cycles: 10,
            throughput_ops_per_kcycle: 1000,
        });

        let mut gpu_costs = HashMap::new();
        gpu_costs.insert(ComputeTarget::Gpu, ComputeCost {
            latency_cycles: 5,
            throughput_ops_per_kcycle: 100_000,
        });
        gpu_costs.insert(ComputeTarget::CpuScalar, ComputeCost {
            latency_cycles: 500,
            throughput_ops_per_kcycle: 1000,
        });

        let mut neon_costs = HashMap::new();
        neon_costs.insert(ComputeTarget::CpuSimd, ComputeCost {
            latency_cycles: 8,
            throughput_ops_per_kcycle: 4000,
        });
        neon_costs.insert(ComputeTarget::CpuScalar, ComputeCost {
            latency_cycles: 30,
            throughput_ops_per_kcycle: 1000,
        });

        let mut sink_costs = HashMap::new();
        sink_costs.insert(ComputeTarget::CpuScalar, ComputeCost {
            latency_cycles: 15,
            throughput_ops_per_kcycle: 1000,
        });

        let nodes = vec![
            ComputeNode {
                id: ComputeNodeId(0),
                instructions: vec![],
                costs: cpu_costs.clone(),
                legal_targets: vec![ComputeTarget::CpuScalar],
                kind: NodeKind::Scalar,
                data_size_bytes: 8,
                produced_values: vec![],
                consumed_values: vec![],
                target_legality: None,
            },
            ComputeNode {
                id: ComputeNodeId(1),
                instructions: vec![],
                costs: gpu_costs,
                legal_targets: vec![ComputeTarget::CpuScalar, ComputeTarget::Gpu],
                kind: NodeKind::DataParallel,
                data_size_bytes: 4096,
                produced_values: vec![],
                consumed_values: vec![],
                target_legality: None,
            },
            ComputeNode {
                id: ComputeNodeId(2),
                instructions: vec![],
                costs: neon_costs,
                legal_targets: vec![ComputeTarget::CpuScalar, ComputeTarget::CpuSimd],
                kind: NodeKind::DataParallel,
                data_size_bytes: 128,
                produced_values: vec![],
                consumed_values: vec![],
                target_legality: None,
            },
            ComputeNode {
                id: ComputeNodeId(3),
                instructions: vec![],
                costs: sink_costs,
                legal_targets: vec![ComputeTarget::CpuScalar],
                kind: NodeKind::Scalar,
                data_size_bytes: 8,
                produced_values: vec![],
                consumed_values: vec![],
                target_legality: None,
            },
        ];

        let edges = vec![
            DataEdge {
                from: ComputeNodeId(0),
                to: ComputeNodeId(1),
                transfer_bytes: 4096,
                transfer_cost: TransferCost::zero(),
            },
            DataEdge {
                from: ComputeNodeId(0),
                to: ComputeNodeId(2),
                transfer_bytes: 128,
                transfer_cost: TransferCost::zero(),
            },
            DataEdge {
                from: ComputeNodeId(1),
                to: ComputeNodeId(3),
                transfer_bytes: 64,
                transfer_cost: TransferCost::zero(),
            },
            DataEdge {
                from: ComputeNodeId(2),
                to: ComputeNodeId(3),
                transfer_bytes: 32,
                transfer_cost: TransferCost::zero(),
            },
        ];

        let graph = ComputeGraph { nodes, edges };

        let recs = vec![
            TargetRecommendation {
                node_id: ComputeNodeId(0),
                recommended_target: ComputeTarget::CpuScalar,
                legal_targets: vec![ComputeTarget::CpuScalar],
                reason: "source".to_string(),
                parallel_reduction_legal: false,
            },
            TargetRecommendation {
                node_id: ComputeNodeId(1),
                recommended_target: ComputeTarget::Gpu,
                legal_targets: vec![ComputeTarget::CpuScalar, ComputeTarget::Gpu],
                reason: "GPU compute".to_string(),
                parallel_reduction_legal: true,
            },
            TargetRecommendation {
                node_id: ComputeNodeId(2),
                recommended_target: ComputeTarget::CpuSimd,
                legal_targets: vec![ComputeTarget::CpuScalar, ComputeTarget::CpuSimd],
                reason: "NEON vectorize".to_string(),
                parallel_reduction_legal: false,
            },
            TargetRecommendation {
                node_id: ComputeNodeId(3),
                recommended_target: ComputeTarget::CpuScalar,
                legal_targets: vec![ComputeTarget::CpuScalar],
                reason: "sink".to_string(),
                parallel_reduction_legal: false,
            },
        ];

        let plan = generate_dispatch_plan(&graph, &recs);

        // All properties should pass for generated plan.
        let report = verify_dispatch_plan_properties(&graph, &plan);
        assert!(report.all_ok(), "Diamond graph plan should pass:\n{}", report);

        // Verify that GPU sync occurs before node 3.
        assert!(validate_dispatch_plan(&graph, &plan).is_ok());
    }

    // -----------------------------------------------------------------------
    // Test V10: Empty graph passes all verifications
    // -----------------------------------------------------------------------

    #[test]
    fn test_verify_empty_graph_all_pass() {
        let graph = ComputeGraph::new();
        let plan = generate_dispatch_plan(&graph, &[]);

        let report = verify_dispatch_plan_properties(&graph, &plan);
        assert!(report.all_ok());
        assert_eq!(report.total_violations(), 0);
    }

    // -----------------------------------------------------------------------
    // Test V11: Multiple CPU fallback violations reported
    // -----------------------------------------------------------------------

    #[test]
    fn test_verify_multiple_cpu_fallback_violations() {
        let nodes = vec![
            ComputeNode {
                id: ComputeNodeId(0),
                instructions: vec![],
                costs: HashMap::new(),
                legal_targets: vec![ComputeTarget::Gpu], // Missing CpuScalar!
                kind: NodeKind::DataParallel,
                data_size_bytes: 4096,
                produced_values: vec![],
                consumed_values: vec![],
                target_legality: None,
            },
            ComputeNode {
                id: ComputeNodeId(1),
                instructions: vec![],
                costs: HashMap::new(),
                legal_targets: vec![ComputeTarget::NeuralEngine], // Missing CpuScalar!
                kind: NodeKind::MatrixHeavy,
                data_size_bytes: 16384,
                produced_values: vec![],
                consumed_values: vec![],
                target_legality: None,
            },
            ComputeNode {
                id: ComputeNodeId(2),
                instructions: vec![],
                costs: HashMap::new(),
                legal_targets: vec![ComputeTarget::CpuScalar], // OK
                kind: NodeKind::Scalar,
                data_size_bytes: 8,
                produced_values: vec![],
                consumed_values: vec![],
                target_legality: None,
            },
        ];

        let graph = ComputeGraph {
            nodes,
            edges: vec![],
        };

        let recs = vec![
            TargetRecommendation {
                node_id: ComputeNodeId(0),
                recommended_target: ComputeTarget::Gpu,
                legal_targets: vec![ComputeTarget::Gpu],
                reason: "gpu".to_string(),
                parallel_reduction_legal: false,
            },
            TargetRecommendation {
                node_id: ComputeNodeId(1),
                recommended_target: ComputeTarget::NeuralEngine,
                legal_targets: vec![ComputeTarget::NeuralEngine],
                reason: "ane".to_string(),
                parallel_reduction_legal: false,
            },
            TargetRecommendation {
                node_id: ComputeNodeId(2),
                recommended_target: ComputeTarget::CpuScalar,
                legal_targets: vec![ComputeTarget::CpuScalar],
                reason: "cpu".to_string(),
                parallel_reduction_legal: false,
            },
        ];

        let plan = generate_dispatch_plan(&graph, &recs);
        let report = verify_dispatch_plan_properties(&graph, &plan);

        assert!(!report.cpu_fallback_ok);
        assert_eq!(report.missing_cpu_fallback.len(), 2);
        assert!(report.missing_cpu_fallback.contains(&ComputeNodeId(0)));
        assert!(report.missing_cpu_fallback.contains(&ComputeNodeId(1)));
    }

    // -----------------------------------------------------------------------
    // Test V12: VerificationReport display format
    // -----------------------------------------------------------------------

    #[test]
    fn test_verification_report_display() {
        let (graph, recs) = mixed_graph();
        let plan = generate_dispatch_plan(&graph, &recs);

        let report = verify_dispatch_plan_properties(&graph, &plan);
        let display = format!("{}", report);
        assert!(display.contains("VerificationReport"));
        assert!(display.contains("CPU fallback: PASS"));
        assert!(display.contains("Data transfers: PASS"));
        assert!(display.contains("Synchronization: PASS"));
        assert!(display.contains("Dependency order: PASS"));
    }

    // -----------------------------------------------------------------------
    // Test V13: verify_data_transfers passes for same-target edges
    // -----------------------------------------------------------------------

    #[test]
    fn test_verify_data_transfers_same_target_no_transfer_needed() {
        // Two CPU nodes with an edge -- no transfer needed.
        let mut costs = HashMap::new();
        costs.insert(ComputeTarget::CpuScalar, ComputeCost {
            latency_cycles: 10,
            throughput_ops_per_kcycle: 1000,
        });

        let nodes = vec![
            ComputeNode {
                id: ComputeNodeId(0),
                instructions: vec![],
                costs: costs.clone(),
                legal_targets: vec![ComputeTarget::CpuScalar],
                kind: NodeKind::Scalar,
                data_size_bytes: 8,
                produced_values: vec![],
                consumed_values: vec![],
                target_legality: None,
            },
            ComputeNode {
                id: ComputeNodeId(1),
                instructions: vec![],
                costs,
                legal_targets: vec![ComputeTarget::CpuScalar],
                kind: NodeKind::Scalar,
                data_size_bytes: 8,
                produced_values: vec![],
                consumed_values: vec![],
                target_legality: None,
            },
        ];

        let edges = vec![DataEdge {
            from: ComputeNodeId(0),
            to: ComputeNodeId(1),
            transfer_bytes: 64,
            transfer_cost: TransferCost::zero(),
        }];

        let graph = ComputeGraph { nodes, edges };

        let recs = vec![
            TargetRecommendation {
                node_id: ComputeNodeId(0),
                recommended_target: ComputeTarget::CpuScalar,
                legal_targets: vec![ComputeTarget::CpuScalar],
                reason: "cpu".to_string(),
                parallel_reduction_legal: false,
            },
            TargetRecommendation {
                node_id: ComputeNodeId(1),
                recommended_target: ComputeTarget::CpuScalar,
                legal_targets: vec![ComputeTarget::CpuScalar],
                reason: "cpu".to_string(),
                parallel_reduction_legal: false,
            },
        ];

        let plan = generate_dispatch_plan(&graph, &recs);
        assert!(verify_data_transfers(&graph, &plan).is_ok());
        assert_eq!(plan.count_transfers(), 0);
    }

    // -----------------------------------------------------------------------
    // Test V14: Comprehensive report captures multiple failure categories
    // -----------------------------------------------------------------------

    #[test]
    fn test_verify_report_captures_multiple_failures() {
        // Build a graph with a GPU-only node (missing CPU fallback)
        // and construct a plan with missing transfer and missing sync.
        let mut gpu_costs = HashMap::new();
        gpu_costs.insert(ComputeTarget::Gpu, ComputeCost {
            latency_cycles: 5,
            throughput_ops_per_kcycle: 100_000,
        });

        let mut cpu_costs = HashMap::new();
        cpu_costs.insert(ComputeTarget::CpuScalar, ComputeCost {
            latency_cycles: 10,
            throughput_ops_per_kcycle: 1000,
        });

        let nodes = vec![
            ComputeNode {
                id: ComputeNodeId(0),
                instructions: vec![],
                costs: cpu_costs,
                legal_targets: vec![ComputeTarget::CpuScalar],
                kind: NodeKind::Scalar,
                data_size_bytes: 8,
                produced_values: vec![],
                consumed_values: vec![],
                target_legality: None,
            },
            ComputeNode {
                id: ComputeNodeId(1),
                instructions: vec![],
                costs: gpu_costs,
                legal_targets: vec![ComputeTarget::Gpu], // Missing CpuScalar!
                kind: NodeKind::DataParallel,
                data_size_bytes: 4096,
                produced_values: vec![],
                consumed_values: vec![],
                target_legality: None,
            },
        ];

        let edges = vec![DataEdge {
            from: ComputeNodeId(0),
            to: ComputeNodeId(1),
            transfer_bytes: 4096,
            transfer_cost: TransferCost::zero(),
        }];

        let graph = ComputeGraph { nodes, edges };

        // Manually construct plan without transfer.
        let mut assignment = HashMap::new();
        assignment.insert(ComputeNodeId(0), ComputeTarget::CpuScalar);
        assignment.insert(ComputeNodeId(1), ComputeTarget::Gpu);

        let plan = DispatchPlan {
            ops: vec![
                DispatchOp::KernelLaunch {
                    target: ComputeTarget::CpuScalar,
                    node_id: ComputeNodeId(0),
                    estimated_cycles: 10,
                },
                // Missing DataTransfer!
                DispatchOp::KernelLaunch {
                    target: ComputeTarget::Gpu,
                    node_id: ComputeNodeId(1),
                    estimated_cycles: 5,
                },
            ],
            assignment,
            estimated_total_cycles: 15,
        };

        let report = verify_dispatch_plan_properties(&graph, &plan);
        assert!(!report.all_ok());

        // CPU fallback violation.
        assert!(!report.cpu_fallback_ok);
        assert_eq!(report.missing_cpu_fallback.len(), 1);

        // Data transfer violation.
        assert!(!report.data_transfers_ok);
        assert!(!report.data_transfer_errors.is_empty());

        // Total violations should be at least 2.
        assert!(report.total_violations() >= 2);
    }
}
