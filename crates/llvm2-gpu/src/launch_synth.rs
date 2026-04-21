// llvm2-gpu/launch_synth.rs - Synthesize Metal launch glue
//
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Copyright 2026 Dropbox, Inc. | License: Apache-2.0
//
// Reference: designs/2026-04-18-gpu-passes-pipeline.md (Pass 6).

//! Pass 6: `LaunchSynth`.
//!
//! Produces Metal launch descriptors — grid/threadgroup dimensions plus
//! an ordered argument table — for every region that HeteroPartition
//! assigned to the GPU.

use std::fmt;

use llvm2_codegen::metal_emitter::{MetalDispatchParams, MtlStorageMode};
use llvm2_lower::compute_graph::TargetRecommendation;
use llvm2_lower::target_analysis::ComputeTarget;

use crate::address_space::AddressSpace;
use crate::memory_partition::{BufferPlan, BufferRole};
use crate::region::{BufferId, KernelRegion, RegionId};

/// Default threadgroup size (threads per group) for 1D kernels.
///
/// 256 is a common sweet spot on Apple Silicon and matches the existing
/// `DEFAULT_THREADGROUP_SIZE` used in `llvm2_codegen::metal_emitter`
/// tests.
pub const DEFAULT_THREADGROUP_SIZE: u32 = 256;

// ---------------------------------------------------------------------------
// LaunchArgument
// ---------------------------------------------------------------------------

/// A single entry in a Metal argument table.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LaunchArgument {
    /// Binding slot index (0..N in order of appearance).
    pub binding: u32,
    /// Which buffer this argument carries.
    pub buffer: BufferId,
    /// Metal storage mode for the buffer.
    pub storage: MtlStorageMode,
    /// Address space qualifier in the kernel signature.
    pub address_space: AddressSpace,
    /// Buffer role (input or output).
    pub role: BufferRole,
}

impl fmt::Display for LaunchArgument {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "arg[{}] = {} ({} {}, {})",
            self.binding, self.buffer, self.address_space, self.role, self.storage
        )
    }
}

// ---------------------------------------------------------------------------
// MetalLaunch
// ---------------------------------------------------------------------------

/// Everything the host side needs to launch one Metal kernel.
#[derive(Debug, Clone)]
pub struct MetalLaunch {
    /// Region id this launch corresponds to.
    pub region_id: RegionId,
    /// Kernel function name (matches what the MSL emitter produces).
    pub kernel_name: String,
    /// Grid + threadgroup dimensions.
    pub dispatch: MetalDispatchParams,
    /// Ordered argument table.
    pub arguments: Vec<LaunchArgument>,
    /// Whether the dispatch requires a synchronize after it completes
    /// (derived from region consumer topology; populated by pipeline).
    pub requires_sync: bool,
}

impl MetalLaunch {
    /// Number of arguments.
    pub fn arg_count(&self) -> usize {
        self.arguments.len()
    }

    /// Iterate input-only arguments.
    pub fn inputs(&self) -> impl Iterator<Item = &LaunchArgument> {
        self.arguments.iter().filter(|a| a.role == BufferRole::Input)
    }

    /// Iterate output-only arguments.
    pub fn outputs(&self) -> impl Iterator<Item = &LaunchArgument> {
        self.arguments.iter().filter(|a| a.role == BufferRole::Output)
    }
}

// ---------------------------------------------------------------------------
// Pass
// ---------------------------------------------------------------------------

/// Pass 6: launch glue synthesis.
#[derive(Debug, Clone)]
pub struct LaunchSynth {
    /// Threadgroup size used for 1D launches (defaults to
    /// [`DEFAULT_THREADGROUP_SIZE`]).
    pub threadgroup_size: u32,
}

impl Default for LaunchSynth {
    fn default() -> Self {
        Self { threadgroup_size: DEFAULT_THREADGROUP_SIZE }
    }
}

impl LaunchSynth {
    /// Build a launch descriptor per GPU-bound region.
    ///
    /// Regions recommended for CPU (by [`crate::HeteroPartition`]) are
    /// skipped — the caller continues to run them via the existing
    /// dispatch plan's CPU fallback.
    pub fn run(
        &self,
        regions: &[KernelRegion],
        recommendations: &[TargetRecommendation],
    ) -> Vec<MetalLaunch> {
        let mut launches = Vec::new();
        for (idx, region) in regions.iter().enumerate() {
            let rec = match recommendations.get(idx) {
                Some(r) => r,
                None => continue,
            };
            if rec.recommended_target != ComputeTarget::Gpu {
                continue;
            }

            // Grid = element_count rounded up to threadgroup size. We
            // reuse the existing metal_emitter helper so grid math is
            // identical between host and kernel.
            let dispatch = MetalDispatchParams::for_1d(region.element_count, self.threadgroup_size);

            let mut arguments = Vec::new();
            let mut binding: u32 = 0;
            for buf in region.input_buffers.iter().copied() {
                let plan = find_plan(region, buf);
                arguments.push(LaunchArgument {
                    binding,
                    buffer: buf,
                    storage: plan.map(|p| p.storage).unwrap_or(MtlStorageMode::Shared),
                    address_space: plan.map(|p| p.address_space).unwrap_or(AddressSpace::Device),
                    role: BufferRole::Input,
                });
                binding += 1;
            }
            for buf in region.output_buffers.iter().copied() {
                let plan = find_plan(region, buf);
                arguments.push(LaunchArgument {
                    binding,
                    buffer: buf,
                    storage: plan.map(|p| p.storage).unwrap_or(MtlStorageMode::Shared),
                    address_space: plan.map(|p| p.address_space).unwrap_or(AddressSpace::Device),
                    role: BufferRole::Output,
                });
                binding += 1;
            }

            launches.push(MetalLaunch {
                region_id: region.id,
                kernel_name: format!("llvm2_{}_{}", region.name, region.id.0),
                dispatch,
                arguments,
                // Scaffolding marks every GPU launch as requiring a sync
                // before its results are visible on the CPU. A depth pass
                // can elide this when consumers are also on the GPU.
                requires_sync: true,
            });
        }
        launches
    }
}

fn find_plan(region: &KernelRegion, buf: BufferId) -> Option<&BufferPlan> {
    region.buffer_plans.iter().find(|p| p.id == buf)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::address_space::AddressSpaceInfer;
    use crate::kernel_extract::KernelPattern;
    use crate::memory_partition::MemoryPartition;
    use crate::region::RegionId;
    use llvm2_lower::compute_graph::ComputeNodeId;

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
    fn gpu_recommendation_produces_one_launch() {
        let mut regions = vec![mk_region()];
        AddressSpaceInfer.run(&mut regions);
        MemoryPartition.run(&mut regions);

        let recs = vec![TargetRecommendation {
            node_id: ComputeNodeId(0),
            recommended_target: ComputeTarget::Gpu,
            legal_targets: vec![ComputeTarget::CpuScalar, ComputeTarget::Gpu],
            reason: "test".to_string(),
            parallel_reduction_legal: true,
        }];

        let launches = LaunchSynth::default().run(&regions, &recs);
        assert_eq!(launches.len(), 1);
        let launch = &launches[0];
        assert_eq!(launch.region_id, RegionId(0));
        assert_eq!(launch.arg_count(), 2);
        assert!(launch.kernel_name.contains("kernel_0"));
        assert_eq!(launch.dispatch.grid_size.width, 1024);
        assert_eq!(launch.dispatch.threadgroup_size.width, DEFAULT_THREADGROUP_SIZE as u64);
        assert!(launch.requires_sync);
        assert_eq!(launch.inputs().count(), 1);
        assert_eq!(launch.outputs().count(), 1);
    }

    #[test]
    fn cpu_recommendation_produces_no_launch() {
        let mut regions = vec![mk_region()];
        AddressSpaceInfer.run(&mut regions);
        MemoryPartition.run(&mut regions);
        let recs = vec![TargetRecommendation {
            node_id: ComputeNodeId(0),
            recommended_target: ComputeTarget::CpuScalar,
            legal_targets: vec![ComputeTarget::CpuScalar],
            reason: "test".to_string(),
            parallel_reduction_legal: false,
        }];
        let launches = LaunchSynth::default().run(&regions, &recs);
        assert!(launches.is_empty());
    }

    #[test]
    fn arguments_are_bound_in_order() {
        let mut regions = vec![KernelRegion::new(
            RegionId(0),
            "kernel_0".into(),
            vec![ComputeNodeId(0)],
            1024,
            4096,
            KernelPattern::ParallelMap,
            vec![BufferId(0), BufferId(1)],
            vec![BufferId(2)],
        )];
        AddressSpaceInfer.run(&mut regions);
        MemoryPartition.run(&mut regions);
        let recs = vec![TargetRecommendation {
            node_id: ComputeNodeId(0),
            recommended_target: ComputeTarget::Gpu,
            legal_targets: vec![ComputeTarget::Gpu],
            reason: "test".to_string(),
            parallel_reduction_legal: true,
        }];
        let launches = LaunchSynth::default().run(&regions, &recs);
        let args = &launches[0].arguments;
        assert_eq!(args.len(), 3);
        assert_eq!(args[0].binding, 0);
        assert_eq!(args[0].buffer, BufferId(0));
        assert_eq!(args[0].role, BufferRole::Input);
        assert_eq!(args[1].binding, 1);
        assert_eq!(args[1].buffer, BufferId(1));
        assert_eq!(args[1].role, BufferRole::Input);
        assert_eq!(args[2].binding, 2);
        assert_eq!(args[2].buffer, BufferId(2));
        assert_eq!(args[2].role, BufferRole::Output);
    }
}
