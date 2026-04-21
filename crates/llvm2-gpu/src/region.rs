// llvm2-gpu/region.rs - Shared kernel-region data structures
//
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Copyright 2026 Andrew Yates | License: Apache-2.0

//! Shared data structures used by the GPU pass pipeline.
//!
//! A [`KernelRegion`] is the unit of work that flows between passes. It
//! starts as a bundle of one or more `ComputeNodeId`s produced by
//! [`crate::kernel_extract::KernelExtract`] and is progressively
//! annotated with address spaces (AddressSpaceInfer), buffer plans
//! (MemoryPartition), divergence stats (DivergenceFlatten), target
//! recommendation (HeteroPartition), and launch glue (LaunchSynth).

use std::fmt;

use serde::{Deserialize, Serialize};

use llvm2_lower::compute_graph::ComputeNodeId;

use crate::address_space::AddressSpaceMap;
use crate::divergence_flatten::DivergenceStats;
use crate::kernel_extract::KernelPattern;
use crate::memory_partition::BufferPlan;

// ---------------------------------------------------------------------------
// Region identifier
// ---------------------------------------------------------------------------

/// Unique identifier for a kernel region within a GPU pipeline run.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct RegionId(pub u32);

impl fmt::Display for RegionId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "region_{}", self.0)
    }
}

// ---------------------------------------------------------------------------
// Buffer identifier
// ---------------------------------------------------------------------------

/// Unique identifier for a buffer introduced by the GPU pipeline.
///
/// Buffers may correspond to real tMIR `ValueId`s (when a region reads
/// or writes a named value) or to compiler-synthesized temporaries
/// produced by MemoryPartition.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct BufferId(pub u32);

impl fmt::Display for BufferId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "buf_{}", self.0)
    }
}

// ---------------------------------------------------------------------------
// KernelRegion
// ---------------------------------------------------------------------------

/// A GPU kernel region: a contiguous group of compute-graph nodes that
/// will be compiled to one Metal kernel.
#[derive(Debug, Clone)]
pub struct KernelRegion {
    /// Unique region id within a pipeline run.
    pub id: RegionId,
    /// Human-readable slug suitable for MSL kernel names.
    pub name: String,
    /// Compute-graph nodes contained in this region, in topological order.
    pub nodes: Vec<ComputeNodeId>,
    /// Total element count across the region (defaults to the max of
    /// per-node data_size_bytes divided by the element stride).
    pub element_count: u64,
    /// Estimated payload size in bytes (sum of node data_size_bytes).
    pub data_size_bytes: u64,
    /// Kernel pattern hint (parallel map/reduce/etc.).
    pub pattern: KernelPattern,
    /// Input buffer ids (buffers consumed by the region).
    pub input_buffers: Vec<BufferId>,
    /// Output buffer ids (buffers produced by the region).
    pub output_buffers: Vec<BufferId>,
    /// Address-space annotations (populated by AddressSpaceInfer).
    pub address_space: AddressSpaceMap,
    /// Buffer plans (populated by MemoryPartition).
    pub buffer_plans: Vec<BufferPlan>,
    /// Divergence stats (populated by DivergenceFlatten).
    pub divergence: DivergenceStats,
}

impl KernelRegion {
    /// Construct a minimal region carrying only the extract-pass outputs.
    ///
    /// Subsequent passes populate the annotation fields.
    pub fn new(
        id: RegionId,
        name: String,
        nodes: Vec<ComputeNodeId>,
        element_count: u64,
        data_size_bytes: u64,
        pattern: KernelPattern,
        input_buffers: Vec<BufferId>,
        output_buffers: Vec<BufferId>,
    ) -> Self {
        Self {
            id,
            name,
            nodes,
            element_count,
            data_size_bytes,
            pattern,
            input_buffers,
            output_buffers,
            address_space: AddressSpaceMap::default(),
            buffer_plans: Vec::new(),
            divergence: DivergenceStats::default(),
        }
    }

    /// Number of distinct buffers touched (inputs + outputs).
    pub fn buffer_count(&self) -> usize {
        self.input_buffers.len() + self.output_buffers.len()
    }
}
