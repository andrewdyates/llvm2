// llvm2-gpu/hetero_partition.rs - CPU vs GPU region dispatch
//
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Copyright 2026 Andrew Yates | License: Apache-2.0
//
// Reference: designs/2026-04-18-gpu-passes-pipeline.md (Pass 5).

//! Pass 5: `HeteroPartition`.
//!
//! Picks CPU vs GPU per kernel region using the existing compute-graph
//! target analysis (`target_recommendations`) plus the profitability
//! analyzer from `llvm2-ir::cost_model`.

use llvm2_lower::compute_graph::{ComputeGraph, TargetRecommendation};
use llvm2_lower::target_analysis::ComputeTarget;

use crate::region::{KernelRegion, RegionId};

// ---------------------------------------------------------------------------
// Pass
// ---------------------------------------------------------------------------

/// Pass 5: per-region target dispatch.
#[derive(Debug, Default, Clone)]
pub struct HeteroPartition;

impl HeteroPartition {
    /// Select a [`ComputeTarget`] per region.
    ///
    /// Walks the compute graph, gathers per-node recommendations, then
    /// rolls them up to the region level: the region inherits the
    /// weakest (most conservative) target across its constituent nodes.
    ///
    /// Returns one [`TargetRecommendation`] per region, using the first
    /// node's id as the recommendation's `node_id` slot so callers can
    /// still feed the vector into
    /// [`llvm2_lower::dispatch::generate_dispatch_plan`].
    pub fn run(
        &self,
        graph: &ComputeGraph,
        regions: &[KernelRegion],
    ) -> Vec<TargetRecommendation> {
        let node_recs = graph.target_recommendations();
        let mut out = Vec::with_capacity(regions.len());
        for region in regions {
            let mut region_target: Option<ComputeTarget> = None;
            let mut region_legal: Option<Vec<ComputeTarget>> = None;
            let mut reason = format!(
                "{}: no recommendations found (defaulting to CpuScalar)",
                region.id
            );
            let mut parallel_reduction_legal = true;

            for node_id in &region.nodes {
                let rec = node_recs.iter().find(|r| r.node_id == *node_id);
                let Some(rec) = rec else { continue };

                // Take the min-cost target for the region: a node that
                // downgrades to CpuScalar forces the whole region to CPU.
                region_target = Some(match region_target {
                    None => rec.recommended_target,
                    Some(current) => most_conservative(current, rec.recommended_target),
                });

                // Legal set = intersection of node legal sets.
                region_legal = Some(match region_legal {
                    None => rec.legal_targets.clone(),
                    Some(current) => intersect(&current, &rec.legal_targets),
                });

                parallel_reduction_legal &= rec.parallel_reduction_legal;
                reason = format!(
                    "{}: node {} -> {} ({})",
                    region.id, node_id, rec.recommended_target, rec.reason
                );
            }

            let target = region_target.unwrap_or(ComputeTarget::CpuScalar);
            let legal_targets = region_legal.unwrap_or_else(|| vec![ComputeTarget::CpuScalar]);
            let node_id_for_rec = region.nodes.first().copied().unwrap_or(
                // If the region has no nodes (shouldn't happen), fall back
                // to a synthetic id; the dispatch planner will catch it.
                llvm2_lower::compute_graph::ComputeNodeId(0),
            );

            out.push(TargetRecommendation {
                node_id: node_id_for_rec,
                recommended_target: target,
                legal_targets,
                reason,
                parallel_reduction_legal,
            });
        }
        out
    }

    /// Convenience: boolean "did this region land on GPU?"
    pub fn region_runs_on_gpu(rec: &TargetRecommendation) -> bool {
        matches!(rec.recommended_target, ComputeTarget::Gpu)
    }

    /// Convenience: index lookup by region id. Returns a reference into
    /// `recommendations` in the order they were produced by `run`.
    pub fn recommendation_for<'a>(
        regions: &[KernelRegion],
        recommendations: &'a [TargetRecommendation],
        region_id: RegionId,
    ) -> Option<&'a TargetRecommendation> {
        let idx = regions.iter().position(|r| r.id == region_id)?;
        recommendations.get(idx)
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Ordering on compute targets that picks the most conservative (lowest-
/// capability) option. Used when combining per-node recommendations into
/// a single per-region recommendation.
fn most_conservative(a: ComputeTarget, b: ComputeTarget) -> ComputeTarget {
    if rank(a) <= rank(b) { a } else { b }
}

fn rank(t: ComputeTarget) -> u8 {
    match t {
        ComputeTarget::CpuScalar => 0,
        ComputeTarget::CpuSimd => 1,
        ComputeTarget::Gpu => 2,
        ComputeTarget::NeuralEngine => 3,
    }
}

fn intersect(a: &[ComputeTarget], b: &[ComputeTarget]) -> Vec<ComputeTarget> {
    a.iter().filter(|t| b.contains(t)).copied().collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernel_extract::KernelPattern;
    use crate::region::{BufferId, KernelRegion, RegionId};
    use llvm2_lower::compute_graph::{ComputeCost, ComputeNode, ComputeNodeId, NodeKind};
    use std::collections::HashMap;

    fn mk_graph_with_legal(legal: Vec<ComputeTarget>) -> ComputeGraph {
        let mut costs = HashMap::new();
        for t in &legal {
            let latency = match t {
                ComputeTarget::CpuScalar => 100,
                ComputeTarget::CpuSimd => 40,
                ComputeTarget::Gpu => 20,
                ComputeTarget::NeuralEngine => 15,
            };
            costs.insert(*t, ComputeCost { latency_cycles: latency, throughput_ops_per_kcycle: 1000 });
        }
        let mut g = ComputeGraph::new();
        g.add_node(ComputeNode {
            id: ComputeNodeId(0),
            instructions: vec![],
            costs,
            legal_targets: legal,
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

    fn mk_region() -> KernelRegion {
        KernelRegion::new(
            RegionId(0),
            "kernel_0".into(),
            vec![ComputeNodeId(0)],
            1024,
            4096,
            KernelPattern::ParallelMap,
            vec![BufferId(0)],
            vec![BufferId(1)],
        )
    }

    #[test]
    fn all_targets_legal_prefers_gpu() {
        let graph = mk_graph_with_legal(vec![
            ComputeTarget::CpuScalar,
            ComputeTarget::CpuSimd,
            ComputeTarget::Gpu,
        ]);
        let regions = vec![mk_region()];
        let recs = HeteroPartition.run(&graph, &regions);
        assert_eq!(recs.len(), 1);
        // Without profitability analyzer, fallback picks cheapest (GPU).
        assert_eq!(recs[0].recommended_target, ComputeTarget::Gpu);
        assert!(HeteroPartition::region_runs_on_gpu(&recs[0]));
    }

    #[test]
    fn only_cpu_legal_stays_cpu() {
        let graph = mk_graph_with_legal(vec![ComputeTarget::CpuScalar]);
        let regions = vec![mk_region()];
        let recs = HeteroPartition.run(&graph, &regions);
        assert_eq!(recs[0].recommended_target, ComputeTarget::CpuScalar);
        assert!(!HeteroPartition::region_runs_on_gpu(&recs[0]));
    }

    #[test]
    fn recommendation_for_returns_correct_entry() {
        let graph = mk_graph_with_legal(vec![ComputeTarget::CpuScalar, ComputeTarget::Gpu]);
        let regions = vec![mk_region()];
        let recs = HeteroPartition.run(&graph, &regions);
        let r = HeteroPartition::recommendation_for(&regions, &recs, RegionId(0));
        assert!(r.is_some());
        assert_eq!(r.unwrap().recommended_target, ComputeTarget::Gpu);
    }

    #[test]
    fn most_conservative_downgrades_to_cpu() {
        assert_eq!(
            most_conservative(ComputeTarget::Gpu, ComputeTarget::CpuScalar),
            ComputeTarget::CpuScalar,
        );
        assert_eq!(
            most_conservative(ComputeTarget::CpuSimd, ComputeTarget::Gpu),
            ComputeTarget::CpuSimd,
        );
    }
}
