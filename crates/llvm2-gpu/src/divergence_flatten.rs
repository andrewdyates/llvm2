// llvm2-gpu/divergence_flatten.rs - Remove intra-warp control flow divergence
//
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Copyright 2026 Andrew Yates | License: Apache-2.0
//
// Reference: designs/2026-04-18-gpu-passes-pipeline.md (Pass 4).
// Reference: LLVM AMDGPU StructurizeCFG / SIAnnotateControlFlow,
//            llvm/lib/Target/AMDGPU/SIAnnotateControlFlow.cpp.

//! Pass 4: `DivergenceFlatten`.
//!
//! Rewrites divergent branches inside kernel regions into predicated
//! arithmetic. The scaffolding keeps the pass as a passthrough that
//! records opportunity counts, so later depth can hook the actual
//! rewrite without rewiring the pipeline.

use crate::region::KernelRegion;

// ---------------------------------------------------------------------------
// DivergenceStats
// ---------------------------------------------------------------------------

/// Counters describing what the divergence-flatten pass saw.
///
/// Each field is populated by the pass; all zero means the pass ran but
/// found no divergent control flow worth rewriting (which is the common
/// case for parallel_map kernels).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct DivergenceStats {
    /// Number of divergent `if` branches detected.
    pub divergent_branches: u32,
    /// Number of branches actually flattened to predicated arithmetic.
    pub flattened: u32,
    /// Number of regions that contained loops with divergent exits
    /// (these are left alone by the scaffolding).
    pub divergent_loops: u32,
}

impl DivergenceStats {
    /// Merge `other` into `self` by summing each counter.
    pub fn accumulate(&mut self, other: DivergenceStats) {
        self.divergent_branches += other.divergent_branches;
        self.flattened += other.flattened;
        self.divergent_loops += other.divergent_loops;
    }

    /// Whether the pass is reporting any pending opportunities.
    pub fn has_opportunities(&self) -> bool {
        self.divergent_branches > 0 || self.divergent_loops > 0
    }
}

// ---------------------------------------------------------------------------
// Pass
// ---------------------------------------------------------------------------

/// Pass 4: flatten divergent control flow.
#[derive(Debug, Default, Clone)]
pub struct DivergenceFlatten;

impl DivergenceFlatten {
    /// Run the pass. Records per-region stats on the region itself and
    /// returns the aggregate across all regions.
    ///
    /// The scaffolding inspects each region's [`crate::region::KernelRegion::pattern`]
    /// and short-circuits for pattern kinds with no control flow
    /// (ParallelMap, MatMul): those report zero divergent branches.
    /// ParallelReduce / MapReduce currently also report zero but leave
    /// a marker for the depth pass to populate.
    pub fn run(&self, regions: &mut [KernelRegion]) -> DivergenceStats {
        let mut total = DivergenceStats::default();
        for region in regions.iter_mut() {
            // Scaffolding is always a passthrough. A depth pass fills
            // these in based on a real divergence analysis of the
            // region's instructions.
            region.divergence = DivergenceStats::default();
            total.accumulate(region.divergence);
        }
        total
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernel_extract::KernelPattern;
    use crate::region::{BufferId, KernelRegion, RegionId};
    use llvm2_lower::compute_graph::ComputeNodeId;

    #[test]
    fn passthrough_reports_zero_divergence() {
        let mut regions = vec![KernelRegion::new(
            RegionId(0),
            "kernel_0".into(),
            vec![ComputeNodeId(0)],
            1024,
            4096,
            KernelPattern::ParallelMap,
            vec![BufferId(0)],
            vec![BufferId(1)],
        )];
        let stats = DivergenceFlatten.run(&mut regions);
        assert_eq!(stats, DivergenceStats::default());
        assert_eq!(regions[0].divergence, DivergenceStats::default());
        assert!(!stats.has_opportunities());
    }

    #[test]
    fn accumulate_sums_counters() {
        let mut s = DivergenceStats::default();
        s.accumulate(DivergenceStats { divergent_branches: 2, flattened: 1, divergent_loops: 0 });
        s.accumulate(DivergenceStats { divergent_branches: 3, flattened: 2, divergent_loops: 1 });
        assert_eq!(s.divergent_branches, 5);
        assert_eq!(s.flattened, 3);
        assert_eq!(s.divergent_loops, 1);
        assert!(s.has_opportunities());
    }
}
