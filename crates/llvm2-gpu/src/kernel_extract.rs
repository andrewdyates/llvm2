// llvm2-gpu/kernel_extract.rs - Extract parallel regions as kernels
//
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Copyright 2026 Dropbox, Inc. | License: Apache-2.0
//
// Reference: designs/2026-04-18-gpu-passes-pipeline.md (Pass 1).

//! Pass 1: `KernelExtract`.
//!
//! Walks the compute graph, selects nodes whose workload classification is
//! GPU-eligible (DataParallel, MatrixHeavy), and groups contiguous
//! compatible nodes into [`KernelRegion`]s.
//!
//! # GPU eligibility gate (tMIR#39, LLVM2#428/#433)
//!
//! The GPU gate in this pass is the **composed** predicate — the workload
//! classification (`NodeKind::DataParallel | NodeKind::MatrixHeavy`) AND the
//! function-level `tmir::Function::is_gpu_eligible()` predicate supplied by
//! the caller. That tMIR predicate is intentionally frozen as:
//!
//! 1. `is_safe_for_gpu()` — `Pure + NoPanic + Deterministic` (tMIR#39 contract
//!    frozen; **LLVM2 must not tighten this gate on its own**).
//! 2. `ParallelMap` proof is present.
//! 3. `DivergenceClass` is `Uniform` or `Low` (missing or `High` disqualifies).
//!
//! LLVM2 must **not** add its own divergence check, parallel-map re-inference,
//! or Pure/NoPanic/Deterministic enforcement — the authoritative source is the
//! tMIR `is_gpu_eligible()` composition. See tMIR `Function::is_gpu_eligible`
//! docstring and `designs/2026-04-18-tla2-supremacy-tmir-scope.md` §3.2.
//!
//! Callers that don't have per-node tMIR function handles can use [`run`] (the
//! pre-#433 behavior, no function-level gate — tests and fixtures). Callers
//! that do have tMIR functions should call [`run_with_gpu_gate`] with a
//! closure that returns `function.is_gpu_eligible()` for the owning function
//! of each node.

use std::fmt;

use serde::{Deserialize, Serialize};

use llvm2_lower::compute_graph::{ComputeGraph, ComputeNodeId, NodeKind};

use crate::region::{BufferId, KernelRegion, RegionId};

// ---------------------------------------------------------------------------
// KernelPattern
// ---------------------------------------------------------------------------

/// Hint describing the shape of a kernel region.
///
/// Maps directly to one of the patterns supported by
/// [`llvm2_codegen::metal_emitter::MslKernel`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum KernelPattern {
    /// Element-wise `output[i] = f(input[i])`.
    ParallelMap,
    /// Tree reduction across the input.
    ParallelReduce,
    /// Fused map-reduce (avoids materializing the intermediate).
    MapReduce,
    /// Tiled matrix multiply.
    MatMul,
}

impl fmt::Display for KernelPattern {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            KernelPattern::ParallelMap => write!(f, "ParallelMap"),
            KernelPattern::ParallelReduce => write!(f, "ParallelReduce"),
            KernelPattern::MapReduce => write!(f, "MapReduce"),
            KernelPattern::MatMul => write!(f, "MatMul"),
        }
    }
}

impl KernelPattern {
    /// Derive the default kernel pattern from a `NodeKind`.
    pub fn from_kind(kind: NodeKind) -> Option<Self> {
        match kind {
            NodeKind::DataParallel => Some(KernelPattern::ParallelMap),
            NodeKind::MatrixHeavy => Some(KernelPattern::MatMul),
            NodeKind::Scalar => None,
        }
    }
}

// ---------------------------------------------------------------------------
// Pass
// ---------------------------------------------------------------------------

/// Pass 1: extract GPU-eligible regions from a [`ComputeGraph`].
#[derive(Debug, Default, Clone)]
pub struct KernelExtract;

impl KernelExtract {
    /// Run the pass without a function-level GPU gate.
    ///
    /// Returns one [`KernelRegion`] per node whose `NodeKind` is GPU-eligible
    /// (`DataParallel` or `MatrixHeavy`). Callers that have tMIR function
    /// handles should prefer [`Self::run_with_gpu_gate`], which additionally
    /// consults `tmir::Function::is_gpu_eligible()` per node (tMIR#39 frozen
    /// composition — see module docstring).
    ///
    /// The scaffolding keeps one node per region. A future depth pass can
    /// merge adjacent compatible regions; merging is safe only when
    /// address space, element type, and divergence class agree.
    pub fn run(&self, graph: &ComputeGraph) -> Vec<KernelRegion> {
        // `|_| true` here means "no function-level gate" — fixtures and
        // pre-tMIR#39 call sites. Production call sites that own a
        // `tmir::Function` must pass the real `is_gpu_eligible` predicate.
        self.run_with_gpu_gate(graph, |_| true)
    }

    /// Run the pass, consulting a caller-supplied function-level GPU gate
    /// for each compute node.
    ///
    /// `gpu_eligible(node) -> bool` should return `true` iff the tMIR
    /// function owning `node` satisfies `tmir::Function::is_gpu_eligible()`.
    /// See the module docstring for the exact composed predicate. LLVM2 does
    /// NOT compose divergence / parallel-map / purity here — it defers to
    /// the frozen tMIR predicate.
    ///
    /// A node is extracted as a [`KernelRegion`] iff:
    ///
    /// 1. Its [`NodeKind`] has a [`KernelPattern`] mapping (DataParallel or
    ///    MatrixHeavy), AND
    /// 2. `gpu_eligible(node)` returns `true`.
    pub fn run_with_gpu_gate<F>(
        &self,
        graph: &ComputeGraph,
        mut gpu_eligible: F,
    ) -> Vec<KernelRegion>
    where
        F: FnMut(&llvm2_lower::compute_graph::ComputeNode) -> bool,
    {
        let mut regions = Vec::new();
        let mut next_region = 0u32;
        let mut next_buf = 0u32;

        for node in &graph.nodes {
            let Some(pattern) = KernelPattern::from_kind(node.kind) else {
                continue;
            };
            // tMIR#39 composed-predicate gate. LLVM2 must NOT add its own
            // divergence / purity / ParallelMap checks here — defer to the
            // frozen `tmir::Function::is_gpu_eligible` predicate.
            if !gpu_eligible(node) {
                continue;
            }

            // A region has one input per consumed value and one output per
            // produced value. Buffers are synthesized when the node has
            // no named values (test fixtures frequently do this).
            let input_count = node.consumed_values.len().max(1) as u32;
            let output_count = node.produced_values.len().max(1) as u32;

            let mut inputs = Vec::with_capacity(input_count as usize);
            for _ in 0..input_count {
                inputs.push(BufferId(next_buf));
                next_buf += 1;
            }
            let mut outputs = Vec::with_capacity(output_count as usize);
            for _ in 0..output_count {
                outputs.push(BufferId(next_buf));
                next_buf += 1;
            }

            // Default element stride is 4 bytes (f32/i32). Scalar-type
            // inference is a follow-up; this mirrors what the existing
            // Metal emitter assumes.
            let element_stride: u64 = 4;
            let element_count = if element_stride == 0 {
                0
            } else {
                node.data_size_bytes / element_stride
            };

            let region = KernelRegion::new(
                RegionId(next_region),
                format!("kernel_{}", node.id),
                vec![node.id],
                element_count,
                node.data_size_bytes,
                pattern,
                inputs,
                outputs,
            );
            regions.push(region);
            next_region += 1;
        }

        regions
    }

    /// Convenience: sum of `data_size_bytes` across extracted regions.
    pub fn total_bytes(regions: &[KernelRegion]) -> u64 {
        regions.iter().map(|r| r.data_size_bytes).sum()
    }

    /// Convenience: list all compute nodes referenced by the regions.
    pub fn covered_nodes(regions: &[KernelRegion]) -> Vec<ComputeNodeId> {
        let mut out = Vec::new();
        for r in regions {
            out.extend_from_slice(&r.nodes);
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use llvm2_lower::compute_graph::{
        ComputeCost, ComputeNode, ComputeNodeId, NodeKind,
    };
    use llvm2_lower::target_analysis::ComputeTarget;
    use std::collections::HashMap;

    fn mk_node(id: u32, kind: NodeKind, bytes: u64) -> ComputeNode {
        let mut costs = HashMap::new();
        costs.insert(
            ComputeTarget::CpuScalar,
            ComputeCost { latency_cycles: 10, throughput_ops_per_kcycle: 1000 },
        );
        costs.insert(
            ComputeTarget::Gpu,
            ComputeCost { latency_cycles: 100, throughput_ops_per_kcycle: 5000 },
        );
        ComputeNode {
            id: ComputeNodeId(id),
            instructions: vec![],
            costs,
            legal_targets: vec![ComputeTarget::CpuScalar, ComputeTarget::Gpu],
            kind,
            data_size_bytes: bytes,
            produced_values: vec![],
            consumed_values: vec![],
            dominant_op: "ADD".to_string(),
            target_legality: None,
            matmul_shape: None,
        }
    }

    #[test]
    fn extracts_data_parallel_region() {
        let mut graph = ComputeGraph::new();
        graph.add_node(mk_node(0, NodeKind::DataParallel, 4096));
        let regions = KernelExtract.run(&graph);
        assert_eq!(regions.len(), 1);
        assert_eq!(regions[0].nodes, vec![ComputeNodeId(0)]);
        assert_eq!(regions[0].pattern, KernelPattern::ParallelMap);
        assert_eq!(regions[0].element_count, 1024);
        assert_eq!(regions[0].data_size_bytes, 4096);
    }

    #[test]
    fn skips_scalar_node() {
        let mut graph = ComputeGraph::new();
        graph.add_node(mk_node(0, NodeKind::Scalar, 8));
        assert!(KernelExtract.run(&graph).is_empty());
    }

    #[test]
    fn matrix_heavy_becomes_matmul() {
        let mut graph = ComputeGraph::new();
        graph.add_node(mk_node(0, NodeKind::MatrixHeavy, 65536));
        let regions = KernelExtract.run(&graph);
        assert_eq!(regions.len(), 1);
        assert_eq!(regions[0].pattern, KernelPattern::MatMul);
    }

    #[test]
    fn total_bytes_and_covered_nodes() {
        let mut graph = ComputeGraph::new();
        graph.add_node(mk_node(0, NodeKind::DataParallel, 2048));
        graph.add_node(mk_node(1, NodeKind::DataParallel, 4096));
        graph.add_node(mk_node(2, NodeKind::Scalar, 8));
        let regions = KernelExtract.run(&graph);
        assert_eq!(regions.len(), 2);
        assert_eq!(KernelExtract::total_bytes(&regions), 6144);
        assert_eq!(
            KernelExtract::covered_nodes(&regions),
            vec![ComputeNodeId(0), ComputeNodeId(1)]
        );
    }

    // tMIR#39 composed-predicate gate (LLVM2#428/#433). The caller-supplied
    // `gpu_eligible` closure stands in for `tmir::Function::is_gpu_eligible()`.
    #[test]
    fn gpu_gate_rejects_ineligible_function() {
        let mut graph = ComputeGraph::new();
        graph.add_node(mk_node(0, NodeKind::DataParallel, 4096));
        graph.add_node(mk_node(1, NodeKind::DataParallel, 4096));

        // Only node 1 is GPU-eligible per the tMIR function predicate.
        let regions = KernelExtract.run_with_gpu_gate(&graph, |n| n.id.0 == 1);
        assert_eq!(regions.len(), 1, "only node 1 should pass the gate");
        assert_eq!(regions[0].nodes, vec![ComputeNodeId(1)]);
    }

    #[test]
    fn gpu_gate_rejects_all_when_gate_false() {
        let mut graph = ComputeGraph::new();
        graph.add_node(mk_node(0, NodeKind::DataParallel, 4096));
        graph.add_node(mk_node(1, NodeKind::MatrixHeavy, 65536));

        let regions = KernelExtract.run_with_gpu_gate(&graph, |_| false);
        assert!(regions.is_empty(), "gate=false must prune every node");
    }

    #[test]
    fn gpu_gate_true_matches_bare_run() {
        let mut graph = ComputeGraph::new();
        graph.add_node(mk_node(0, NodeKind::DataParallel, 2048));
        graph.add_node(mk_node(1, NodeKind::MatrixHeavy, 65536));
        graph.add_node(mk_node(2, NodeKind::Scalar, 8));

        let bare = KernelExtract.run(&graph);
        let gated = KernelExtract.run_with_gpu_gate(&graph, |_| true);
        assert_eq!(
            bare.iter().map(|r| r.nodes.clone()).collect::<Vec<_>>(),
            gated.iter().map(|r| r.nodes.clone()).collect::<Vec<_>>(),
            "gate=true must match the bare run"
        );
    }
}
