// llvm2-gpu/memory_partition.rs - Split host/device allocations
//
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Copyright 2026 Andrew Yates | License: Apache-2.0
//
// Reference: designs/2026-04-18-gpu-passes-pipeline.md (Pass 3).

//! Pass 3: `MemoryPartition`.
//!
//! Picks the Metal storage mode for each buffer. On Apple UMA the default
//! is `Shared` (zero-copy between CPU and GPU). `Private` is used for
//! buffers that live entirely on the GPU side; we keep it as an explicit
//! follow-up marker here — the scaffolding defaults every buffer to
//! `Shared` but the policy hook is in place so a depth pass can promote.

use std::collections::HashMap;
use std::fmt;

use llvm2_codegen::metal_emitter::MtlStorageMode;

use crate::address_space::AddressSpace;
use crate::region::{BufferId, KernelRegion, RegionId};

// ---------------------------------------------------------------------------
// BufferPlan
// ---------------------------------------------------------------------------

/// A concrete allocation plan for one buffer.
///
/// Emitted by MemoryPartition and consumed by LaunchSynth.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BufferPlan {
    /// Buffer identifier (matches `KernelRegion::input_buffers` /
    /// `output_buffers`).
    pub id: BufferId,
    /// Metal storage mode (`Shared`, `Private`, `Memoryless`).
    pub storage: MtlStorageMode,
    /// Address space qualifier (informational; matches AddressSpaceInfer
    /// output).
    pub address_space: AddressSpace,
    /// Size in bytes.
    pub size_bytes: u64,
    /// Region id that produces this buffer (if any).
    pub producer_region: Option<RegionId>,
    /// Region ids that consume this buffer.
    pub consumer_regions: Vec<RegionId>,
    /// Whether the buffer is a kernel input or output.
    pub role: BufferRole,
}

/// Whether a buffer is consumed or produced by its region.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BufferRole {
    Input,
    Output,
}

impl fmt::Display for BufferRole {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            BufferRole::Input => write!(f, "input"),
            BufferRole::Output => write!(f, "output"),
        }
    }
}

// ---------------------------------------------------------------------------
// Pass
// ---------------------------------------------------------------------------

/// Pass 3: memory partitioning.
#[derive(Debug, Default, Clone)]
pub struct MemoryPartition;

impl MemoryPartition {
    /// Generate buffer plans for every region.
    ///
    /// Policy:
    /// - Every input and output is a [`MtlStorageMode::Shared`] buffer
    ///   by default (Apple UMA, zero-copy).
    /// - `Constant` address space maps to `Shared` with a host-side
    ///   immutable flag (follow-up once the flag plumbing lands).
    ///
    /// The resulting plans are attached to each region and also returned
    /// as a flat vector for callers that want the global view.
    pub fn run(&self, regions: &mut [KernelRegion]) -> Vec<BufferPlan> {
        // Size estimate: divide the region's total bytes evenly across
        // the buffers that touch it. A depth pass can derive exact sizes
        // from tMIR type info and shape proofs.
        let mut all_plans = Vec::new();
        // Build a map from buffer id to producer/consumer regions so we
        // can set producer_region / consumer_regions.
        let mut producer: HashMap<BufferId, RegionId> = HashMap::new();
        let mut consumers: HashMap<BufferId, Vec<RegionId>> = HashMap::new();
        for region in regions.iter() {
            for buf in region.output_buffers.iter().copied() {
                producer.entry(buf).or_insert(region.id);
            }
            for buf in region.input_buffers.iter().copied() {
                consumers.entry(buf).or_default().push(region.id);
            }
        }

        for region in regions.iter_mut() {
            let buf_count = region.buffer_count().max(1) as u64;
            let per_buf = region.data_size_bytes / buf_count;
            let mut plans_for_region = Vec::new();

            let collect = |buf: BufferId, role: BufferRole| {
                let space = region
                    .address_space
                    .get(buf)
                    .unwrap_or(AddressSpace::Device);
                let storage = match space {
                    AddressSpace::Device | AddressSpace::Constant => {
                        MtlStorageMode::Shared
                    }
                    AddressSpace::Threadgroup | AddressSpace::Thread => {
                        // Threadgroup/Thread live inside the kernel and
                        // do not back an MTLBuffer, but we record them
                        // anyway as Shared with size 0 so that LaunchSynth
                        // can skip binding them. A depth pass can filter.
                        MtlStorageMode::Shared
                    }
                };
                BufferPlan {
                    id: buf,
                    storage,
                    address_space: space,
                    size_bytes: per_buf,
                    producer_region: producer.get(&buf).copied(),
                    consumer_regions: consumers
                        .get(&buf)
                        .cloned()
                        .unwrap_or_default(),
                    role,
                }
            };

            for buf in region.input_buffers.iter().copied() {
                plans_for_region.push(collect(buf, BufferRole::Input));
            }
            for buf in region.output_buffers.iter().copied() {
                plans_for_region.push(collect(buf, BufferRole::Output));
            }

            all_plans.extend(plans_for_region.iter().cloned());
            region.buffer_plans = plans_for_region;
        }

        all_plans
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::address_space::AddressSpaceInfer;
    use crate::kernel_extract::KernelPattern;
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
    fn defaults_to_shared_storage() {
        let mut regions = vec![mk_region()];
        AddressSpaceInfer.run(&mut regions);
        let plans = MemoryPartition.run(&mut regions);
        assert_eq!(plans.len(), 2);
        for p in &plans {
            assert_eq!(p.storage, MtlStorageMode::Shared);
            assert_eq!(p.address_space, AddressSpace::Device);
        }
        // Plans are also attached to the region.
        assert_eq!(regions[0].buffer_plans.len(), 2);
    }

    #[test]
    fn producer_consumer_tracking_across_regions() {
        // Region 0 produces buf 1; region 1 consumes buf 1.
        let region0 = mk_region();
        let region1 = KernelRegion::new(
            RegionId(1),
            "kernel_1".into(),
            vec![ComputeNodeId(1)],
            1024,
            4096,
            KernelPattern::ParallelMap,
            vec![BufferId(1)],
            vec![BufferId(2)],
        );
        let mut regions = vec![region0, region1];
        AddressSpaceInfer.run(&mut regions);
        let plans = MemoryPartition.run(&mut regions);

        // Find the plan for buf 1 in region 1 (the input side).
        let p = plans
            .iter()
            .find(|p| p.id == BufferId(1) && p.role == BufferRole::Input)
            .expect("buf 1 input plan present");
        assert_eq!(p.producer_region, Some(RegionId(0)));
        assert_eq!(p.consumer_regions, vec![RegionId(1)]);
    }
}
