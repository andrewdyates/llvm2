// llvm2-gpu/pipeline.rs - Orchestrates the six GPU passes
//
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Copyright 2026 Andrew Yates | License: Apache-2.0
//
// Reference: designs/2026-04-18-gpu-passes-pipeline.md

//! The orchestrator that runs the six GPU passes in order and returns
//! a single bundle that carries everything downstream consumers
//! (dispatch-plan generation, MSL emission, test harnesses) need.

use llvm2_lower::compute_graph::{ComputeGraph, TargetRecommendation};

use crate::address_space::AddressSpaceInfer;
use crate::divergence_flatten::{DivergenceFlatten, DivergenceStats};
use crate::hetero_partition::HeteroPartition;
use crate::kernel_extract::KernelExtract;
use crate::launch_synth::{LaunchSynth, MetalLaunch};
use crate::memory_partition::{BufferPlan, MemoryPartition};
use crate::region::KernelRegion;

// ---------------------------------------------------------------------------
// GpuPipelineConfig
// ---------------------------------------------------------------------------

/// Per-pass opt-in toggles.
///
/// Each toggle defaults to `true`, matching the scaffolding's "run all
/// six passes" expectation. Depth experiments can disable individual
/// passes for A/B comparisons without rewiring the code.
#[derive(Debug, Clone)]
pub struct GpuPipelineConfig {
    pub kernel_extract: bool,
    pub address_space: bool,
    pub memory_partition: bool,
    pub divergence_flatten: bool,
    pub hetero_partition: bool,
    pub launch_synth: bool,
    /// Threadgroup size passed to [`LaunchSynth`].
    pub threadgroup_size: u32,
}

impl Default for GpuPipelineConfig {
    fn default() -> Self {
        Self {
            kernel_extract: true,
            address_space: true,
            memory_partition: true,
            divergence_flatten: true,
            hetero_partition: true,
            launch_synth: true,
            threadgroup_size: crate::launch_synth::DEFAULT_THREADGROUP_SIZE,
        }
    }
}

impl GpuPipelineConfig {
    /// A configuration with every pass disabled except KernelExtract.
    /// Useful for pure pattern-detection experiments.
    pub fn extract_only() -> Self {
        Self {
            kernel_extract: true,
            address_space: false,
            memory_partition: false,
            divergence_flatten: false,
            hetero_partition: false,
            launch_synth: false,
            threadgroup_size: crate::launch_synth::DEFAULT_THREADGROUP_SIZE,
        }
    }
}

// ---------------------------------------------------------------------------
// GpuPipelineOutput
// ---------------------------------------------------------------------------

/// What the pipeline produces.
#[derive(Debug, Clone)]
pub struct GpuPipelineOutput {
    /// All kernel regions extracted from the compute graph (annotated
    /// with address-space / buffer-plan / divergence metadata).
    pub regions: Vec<KernelRegion>,
    /// Flat buffer plan view (also attached per-region).
    pub buffer_plans: Vec<BufferPlan>,
    /// Per-region target recommendation (compatible with
    /// [`llvm2_lower::dispatch::generate_dispatch_plan`]).
    pub recommendations: Vec<TargetRecommendation>,
    /// Metal launch descriptors for GPU-bound regions.
    pub launches: Vec<MetalLaunch>,
    /// Aggregated divergence stats.
    pub divergence_stats: DivergenceStats,
}

impl GpuPipelineOutput {
    /// Whether any launch was synthesized.
    pub fn has_gpu_work(&self) -> bool {
        !self.launches.is_empty()
    }

    /// Number of extracted regions.
    pub fn region_count(&self) -> usize {
        self.regions.len()
    }
}

// ---------------------------------------------------------------------------
// GpuPipeline
// ---------------------------------------------------------------------------

/// The six-pass GPU pipeline.
#[derive(Debug, Default, Clone)]
pub struct GpuPipeline {
    pub config: GpuPipelineConfig,
}

impl GpuPipeline {
    /// Construct with an explicit config.
    pub fn new(config: GpuPipelineConfig) -> Self {
        Self { config }
    }

    /// Run the pipeline over a [`ComputeGraph`].
    pub fn run(&self, graph: &ComputeGraph) -> GpuPipelineOutput {
        // 1. KernelExtract.
        let mut regions = if self.config.kernel_extract {
            KernelExtract.run(graph)
        } else {
            Vec::new()
        };

        // 2. AddressSpaceInfer.
        if self.config.address_space {
            AddressSpaceInfer.run(&mut regions);
        }

        // 3. MemoryPartition.
        let buffer_plans = if self.config.memory_partition {
            MemoryPartition.run(&mut regions)
        } else {
            Vec::new()
        };

        // 4. DivergenceFlatten.
        let divergence_stats = if self.config.divergence_flatten {
            DivergenceFlatten.run(&mut regions)
        } else {
            DivergenceStats::default()
        };

        // 5. HeteroPartition.
        let recommendations = if self.config.hetero_partition {
            HeteroPartition.run(graph, &regions)
        } else {
            Vec::new()
        };

        // 6. LaunchSynth.
        let launches = if self.config.launch_synth {
            LaunchSynth { threadgroup_size: self.config.threadgroup_size }
                .run(&regions, &recommendations)
        } else {
            Vec::new()
        };

        GpuPipelineOutput {
            regions,
            buffer_plans,
            recommendations,
            launches,
            divergence_stats,
        }
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

    fn mk_graph() -> ComputeGraph {
        let mut costs = HashMap::new();
        costs.insert(
            ComputeTarget::CpuScalar,
            ComputeCost { latency_cycles: 100, throughput_ops_per_kcycle: 1000 },
        );
        costs.insert(
            ComputeTarget::Gpu,
            ComputeCost { latency_cycles: 20, throughput_ops_per_kcycle: 5000 },
        );
        let mut g = ComputeGraph::new();
        g.add_node(ComputeNode {
            id: ComputeNodeId(0),
            instructions: vec![],
            costs,
            legal_targets: vec![ComputeTarget::CpuScalar, ComputeTarget::Gpu],
            kind: NodeKind::DataParallel,
            data_size_bytes: 4096,
            produced_values: vec![],
            consumed_values: vec![],
            dominant_op: "ADD".to_string(),
            target_legality: None,
            matmul_shape: None,
        });
        g
    }

    #[test]
    fn full_pipeline_produces_regions_plans_launches() {
        let graph = mk_graph();
        let out = GpuPipeline::default().run(&graph);
        assert_eq!(out.region_count(), 1);
        assert_eq!(out.buffer_plans.len(), 2);
        assert_eq!(out.recommendations.len(), 1);
        assert_eq!(out.recommendations[0].recommended_target, ComputeTarget::Gpu);
        assert!(out.has_gpu_work());
        assert_eq!(out.launches.len(), 1);
    }

    #[test]
    fn extract_only_config_skips_later_passes() {
        let graph = mk_graph();
        let out = GpuPipeline::new(GpuPipelineConfig::extract_only()).run(&graph);
        assert_eq!(out.region_count(), 1);
        assert!(out.buffer_plans.is_empty());
        assert!(out.recommendations.is_empty());
        assert!(out.launches.is_empty());
    }

    #[test]
    fn empty_graph_produces_empty_output() {
        let graph = ComputeGraph::new();
        let out = GpuPipeline::default().run(&graph);
        assert!(out.regions.is_empty());
        assert!(out.buffer_plans.is_empty());
        assert!(out.recommendations.is_empty());
        assert!(out.launches.is_empty());
    }
}
