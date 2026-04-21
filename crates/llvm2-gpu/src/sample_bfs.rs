// llvm2-gpu/sample_bfs.rs - End-to-end BFS-style parallel_map sample
//
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Copyright 2026 Dropbox, Inc. | License: Apache-2.0
//
// Reference: designs/2026-04-18-gpu-passes-pipeline.md
// Issue: andrewdyates/LLVM2#394 (Part of #390)

//! End-to-end sample wiring the GPU passes together for a BFS-style
//! `parallel_map` kernel, including MSL source emission.
//!
//! BFS frontier expansion is a `parallel_map` (apply-to-each-neighbor)
//! followed by a `parallel_reduce` (atomic compaction). This module
//! exercises the map half. The reduce half follows in a depth task and
//! reuses the existing `MslKernel::MapReduce` path.
//!
//! The sample also surfaces a 2-input `FADD` parallel-map variant so
//! integration tests can verify the argument-buffer layout that
//! KernelExtract + AddressSpaceInfer + MemoryPartition produces for
//! multi-input kernels.

use std::collections::HashMap;

use llvm2_codegen::metal_emitter::{
    emit_kernel_from_node, MetalKernelEmitter, MslElementType, MslKernel,
};
use llvm2_lower::compute_graph::{
    ComputeCost, ComputeGraph, ComputeNode, ComputeNodeId, NodeKind,
};
use llvm2_lower::dispatch::{
    DispatchOp, DispatchPlan, generate_dispatch_plan, validate_dispatch_plan,
};
use llvm2_lower::target_analysis::ComputeTarget;
use tmir::ValueId;

use crate::pipeline::{GpuPipeline, GpuPipelineOutput};
use crate::region::KernelRegion;

// ---------------------------------------------------------------------------
// SampleBfs
// ---------------------------------------------------------------------------

/// Inputs needed to build a BFS-style sample compute graph.
#[derive(Debug, Clone, Copy)]
pub struct SampleBfsSpec {
    /// Frontier size (number of vertices in the active wavefront).
    pub frontier_size: u64,
    /// Element stride in bytes (4 = i32 / f32 payload).
    pub element_stride: u64,
}

impl Default for SampleBfsSpec {
    fn default() -> Self {
        Self { frontier_size: 1024, element_stride: 4 }
    }
}

impl SampleBfsSpec {
    fn data_bytes(&self) -> u64 {
        self.frontier_size.saturating_mul(self.element_stride)
    }
}

// ---------------------------------------------------------------------------
// SampleBfsResult
// ---------------------------------------------------------------------------

/// Everything produced by the BFS-style end-to-end sample:
/// compute graph, pipeline output, dispatch plan, and MSL source.
#[derive(Debug, Clone)]
pub struct SampleBfsResult {
    /// Compute graph used as pipeline input.
    pub graph: ComputeGraph,
    /// Output of the six-pass pipeline.
    pub pipeline: GpuPipelineOutput,
    /// Dispatch plan produced from the pipeline's recommendations.
    pub dispatch: DispatchPlan,
    /// MSL source for the one extracted kernel.
    pub msl_source: String,
}

// ---------------------------------------------------------------------------
// run_sample_bfs
// ---------------------------------------------------------------------------

/// Build a one-node BFS-style parallel_map compute graph, run the six
/// GPU passes, generate a dispatch plan, emit MSL source. Returns every
/// intermediate artifact for inspection.
///
/// Panics only on programmer errors (inconsistent graph construction);
/// all runtime failures are represented as empty pipeline outputs.
pub fn run_sample_bfs(spec: SampleBfsSpec) -> SampleBfsResult {
    let graph = build_bfs_graph(spec);

    // Run the GPU passes.
    let pipeline = GpuPipeline::default().run(&graph);

    // Feed the per-region recommendations into the existing dispatch-plan
    // generator. Each recommendation's `node_id` is the region's first
    // compute-node id, so the generator can thread it through its
    // assignment map without any translation layer.
    let dispatch = generate_dispatch_plan(&graph, &pipeline.recommendations);

    // Emit MSL source for the first region. The emitter chooses the
    // template based on the region's input-buffer count (which comes from
    // KernelExtract), threadgroup size (from LaunchSynth), and element
    // type (from the node's dominant op).
    let msl_source = if let Some(region) = pipeline.regions.first() {
        emit_msl_for_region(&graph, region, &pipeline)
    } else {
        String::new()
    };

    SampleBfsResult { graph, pipeline, dispatch, msl_source }
}

// ---------------------------------------------------------------------------
// run_sample_map2 - 2-input FADD parallel-map kernel
// ---------------------------------------------------------------------------

/// Build a two-input FADD parallel-map compute graph, run the pass
/// pipeline, emit MSL. Distinct from [`run_sample_bfs`] in that it uses
/// two consumed values so KernelExtract produces a region with two input
/// buffers and the emitter selects the `parallel_map2` template. Used by
/// integration tests to verify the multi-input argument-buffer layout.
pub fn run_sample_map2(spec: SampleBfsSpec) -> SampleBfsResult {
    let graph = build_map2_graph(spec);
    let pipeline = GpuPipeline::default().run(&graph);
    let dispatch = generate_dispatch_plan(&graph, &pipeline.recommendations);
    let msl_source = if let Some(region) = pipeline.regions.first() {
        emit_msl_for_region(&graph, region, &pipeline)
    } else {
        String::new()
    };
    SampleBfsResult { graph, pipeline, dispatch, msl_source }
}

// ---------------------------------------------------------------------------
// emit_msl_for_region
// ---------------------------------------------------------------------------

/// Emit MSL source for a [`KernelRegion`] using the existing
/// [`emit_kernel_from_node`] entry point when possible, falling back to a
/// direct [`MslKernel::parallel_map`] / [`MslKernel::parallel_map2`]
/// emission when the node's `dominant_op` is not recognized by the
/// emitter (e.g. synthetic `"BFS_STEP"` names in test fixtures).
///
/// Input count is taken from the region's `input_buffers` length, which
/// is populated by KernelExtract from the node's `consumed_values` and
/// reflects AddressSpaceInfer + MemoryPartition decisions via the
/// region's attached `buffer_plans`.
pub fn emit_msl_for_region(
    graph: &ComputeGraph,
    region: &KernelRegion,
    pipeline: &GpuPipelineOutput,
) -> String {
    // Prefer the production path: emit_kernel_from_node understands node
    // classification and picks the right template + element type.
    if let Some(node_id) = region.nodes.first().copied() {
        if let Some(node) = graph.nodes.iter().find(|n| n.id == node_id) {
            if let Ok(src) = emit_kernel_from_node(node) {
                return src;
            }
        }
    }

    // Fallback for synthetic test fixtures: honour the region's input
    // buffer count to pick the right parallel_map template.
    let threadgroup_size = pipeline
        .launches
        .iter()
        .find(|l| l.region_id == region.id)
        .map(|l| l.dispatch.threadgroup_size.width as u32)
        .unwrap_or(256);
    let emitter = MetalKernelEmitter::new(&region.name, MslElementType::Int);
    let kernel = if region.input_buffers.len() >= 2 {
        MslKernel::parallel_map2("a[tid] + b[tid]", region.element_count, threadgroup_size)
    } else {
        MslKernel::parallel_map("input[tid]", region.element_count, threadgroup_size)
    };
    emitter.emit(&kernel)
}

// ---------------------------------------------------------------------------
// build_bfs_graph
// ---------------------------------------------------------------------------

/// Construct a minimal compute graph that matches the shape of the BFS
/// frontier-expansion kernel (one DataParallel node, legal on both CPU
/// and GPU).
fn build_bfs_graph(spec: SampleBfsSpec) -> ComputeGraph {
    let mut costs = HashMap::new();
    // Costs are illustrative — the profitability analyzer isn't attached
    // in this sample, so the pipeline fallback picks the cheapest legal
    // target.
    costs.insert(
        ComputeTarget::CpuScalar,
        ComputeCost { latency_cycles: spec.frontier_size * 100, throughput_ops_per_kcycle: 1000 },
    );
    costs.insert(
        ComputeTarget::Gpu,
        ComputeCost { latency_cycles: spec.frontier_size / 4, throughput_ops_per_kcycle: 8000 },
    );

    let mut graph = ComputeGraph::new();
    graph.add_node(ComputeNode {
        id: ComputeNodeId(0),
        instructions: vec![],
        costs,
        legal_targets: vec![ComputeTarget::CpuScalar, ComputeTarget::Gpu],
        kind: NodeKind::DataParallel,
        data_size_bytes: spec.data_bytes(),
        produced_values: vec![],
        consumed_values: vec![],
        dominant_op: "BFS_STEP".to_string(),
        target_legality: None,
        matmul_shape: None,
    });
    graph
}

// ---------------------------------------------------------------------------
// build_map2_graph
// ---------------------------------------------------------------------------

/// Construct a two-input FADD parallel-map compute graph:
/// `output[tid] = a[tid] + b[tid]`.
///
/// Populates `consumed_values` with two `ValueId`s and `produced_values`
/// with one so KernelExtract materialises a region with two input buffers
/// and one output buffer. The `dominant_op = "FADD"` lets the Metal
/// emitter pick `MslElementType::Float` and `MslOp::Add`, producing a
/// recognisable `parallel_map2` kernel source.
fn build_map2_graph(spec: SampleBfsSpec) -> ComputeGraph {
    let mut costs = HashMap::new();
    costs.insert(
        ComputeTarget::CpuScalar,
        ComputeCost { latency_cycles: spec.frontier_size * 100, throughput_ops_per_kcycle: 1000 },
    );
    costs.insert(
        ComputeTarget::Gpu,
        ComputeCost { latency_cycles: spec.frontier_size / 4, throughput_ops_per_kcycle: 8000 },
    );

    let mut graph = ComputeGraph::new();
    graph.add_node(ComputeNode {
        id: ComputeNodeId(0),
        instructions: vec![],
        costs,
        legal_targets: vec![ComputeTarget::CpuScalar, ComputeTarget::Gpu],
        kind: NodeKind::DataParallel,
        data_size_bytes: spec.data_bytes(),
        produced_values: vec![ValueId::new(100)],
        consumed_values: vec![ValueId::new(1), ValueId::new(2)],
        dominant_op: "FADD".to_string(),
        target_legality: None,
        matmul_shape: None,
    });
    graph
}

// ---------------------------------------------------------------------------
// Convenience assertions for tests
// ---------------------------------------------------------------------------

/// Count the number of `KernelLaunch` ops in the dispatch plan that
/// target the GPU.
pub fn count_gpu_launches(plan: &DispatchPlan) -> usize {
    plan.ops
        .iter()
        .filter(|op| {
            matches!(
                op,
                DispatchOp::KernelLaunch { target: ComputeTarget::Gpu, .. }
            )
        })
        .count()
}

/// Validate the dispatch plan against the graph and return any error.
///
/// Thin wrapper around [`validate_dispatch_plan`] so callers can avoid
/// the direct dependency in tests.
pub fn validate(graph: &ComputeGraph, plan: &DispatchPlan) -> Result<(), String> {
    validate_dispatch_plan(graph, plan).map_err(|e| e.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sample_emits_one_region_one_launch_and_msl() {
        let result = run_sample_bfs(SampleBfsSpec::default());
        assert_eq!(result.pipeline.region_count(), 1);
        assert_eq!(result.pipeline.launches.len(), 1);
        assert!(result.pipeline.has_gpu_work());

        // MSL source must have the Metal header and a kernel.
        assert!(result.msl_source.contains("metal_stdlib"));
        assert!(result.msl_source.contains("kernel"));

        // Dispatch plan must include a GPU launch and validate cleanly.
        assert_eq!(count_gpu_launches(&result.dispatch), 1);
        validate(&result.graph, &result.dispatch).expect("dispatch plan validates");
    }

    #[test]
    fn small_frontier_still_produces_pipeline_output() {
        let spec = SampleBfsSpec { frontier_size: 32, element_stride: 4 };
        let result = run_sample_bfs(spec);
        assert_eq!(result.pipeline.region_count(), 1);
        // With only 32 elements and no profitability analyzer attached,
        // the fallback still picks Gpu as the cheapest legal target.
        assert_eq!(count_gpu_launches(&result.dispatch), 1);
    }
}
