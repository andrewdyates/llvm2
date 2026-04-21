// llvm2-gpu/address_space.rs - Tag pointers with MSL address spaces
//
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Copyright 2026 Dropbox, Inc. | License: Apache-2.0
//
// Reference: designs/2026-04-18-gpu-passes-pipeline.md (Pass 2).
// Reference: Metal Shading Language Specification v3.2 Section 4
//            (Address Spaces).

//! Pass 2: `AddressSpaceInfer`.
//!
//! Tags each kernel-region buffer with its MSL address space so later
//! passes (MemoryPartition, LaunchSynth) can pick the right storage mode
//! and argument-table layout.

use std::collections::HashMap;
use std::fmt;

use serde::{Deserialize, Serialize};

use crate::region::{BufferId, KernelRegion};

// ---------------------------------------------------------------------------
// AddressSpace
// ---------------------------------------------------------------------------

/// MSL address space qualifier.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum AddressSpace {
    /// `device` — cross-region global buffers. This is the default for
    /// all kernel inputs and outputs.
    Device,
    /// `threadgroup` — shared across lanes in a threadgroup; used for
    /// scratch larger than a register tile.
    Threadgroup,
    /// `constant` — read-only buffers the kernel never writes. Maps to
    /// argument-buffer constants on Metal.
    Constant,
    /// `thread` — private to a single lane (a local variable).
    Thread,
}

impl fmt::Display for AddressSpace {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AddressSpace::Device => write!(f, "device"),
            AddressSpace::Threadgroup => write!(f, "threadgroup"),
            AddressSpace::Constant => write!(f, "constant"),
            AddressSpace::Thread => write!(f, "thread"),
        }
    }
}

// ---------------------------------------------------------------------------
// AddressSpaceMap
// ---------------------------------------------------------------------------

/// Per-buffer address space annotation for a kernel region.
#[derive(Debug, Clone, Default)]
pub struct AddressSpaceMap {
    inner: HashMap<BufferId, AddressSpace>,
}

impl AddressSpaceMap {
    /// New empty map.
    pub fn new() -> Self {
        Self::default()
    }

    /// Insert an annotation. Overrides any previous annotation for the
    /// buffer (later passes may refine the initial conservative default).
    pub fn insert(&mut self, id: BufferId, space: AddressSpace) {
        self.inner.insert(id, space);
    }

    /// Look up the annotation for a buffer, if any.
    pub fn get(&self, id: BufferId) -> Option<AddressSpace> {
        self.inner.get(&id).copied()
    }

    /// Number of annotated buffers.
    pub fn len(&self) -> usize {
        self.inner.len()
    }

    /// Whether the map is empty.
    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    /// Iterate (buffer, address space) pairs.
    pub fn iter(&self) -> impl Iterator<Item = (BufferId, AddressSpace)> + '_ {
        self.inner.iter().map(|(k, v)| (*k, *v))
    }
}

// ---------------------------------------------------------------------------
// Pass
// ---------------------------------------------------------------------------

/// Pass 2: infer MSL address spaces for every buffer in every region.
#[derive(Debug, Default, Clone)]
pub struct AddressSpaceInfer;

impl AddressSpaceInfer {
    /// Annotate each region in place.
    ///
    /// Policy (conservative default):
    /// - Inputs -> `Device` (later depth can promote read-only inputs
    ///   with proven-immutable markers to `Constant`).
    /// - Outputs -> `Device`.
    ///
    /// Threadgroup/Thread tagging happens inside the kernel body and is
    /// handled in the MSL emitter, not here.
    pub fn run(&self, regions: &mut [KernelRegion]) {
        for region in regions.iter_mut() {
            for buf in region.input_buffers.iter().copied() {
                region.address_space.insert(buf, AddressSpace::Device);
            }
            for buf in region.output_buffers.iter().copied() {
                region.address_space.insert(buf, AddressSpace::Device);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernel_extract::KernelPattern;
    use crate::region::{KernelRegion, RegionId};
    use llvm2_lower::compute_graph::ComputeNodeId;

    #[test]
    fn inputs_and_outputs_default_to_device() {
        let region = KernelRegion::new(
            RegionId(0),
            "kernel_0".into(),
            vec![ComputeNodeId(0)],
            1024,
            4096,
            KernelPattern::ParallelMap,
            vec![BufferId(0)],
            vec![BufferId(1)],
        );
        let mut regions = vec![region.clone()];
        AddressSpaceInfer.run(&mut regions);
        let m = &regions[0].address_space;
        assert_eq!(m.get(BufferId(0)), Some(AddressSpace::Device));
        assert_eq!(m.get(BufferId(1)), Some(AddressSpace::Device));
        assert_eq!(m.len(), 2);

        // Empty initial map.
        assert!(region.address_space.is_empty());
    }

    #[test]
    fn display_matches_msl_keywords() {
        assert_eq!(AddressSpace::Device.to_string(), "device");
        assert_eq!(AddressSpace::Threadgroup.to_string(), "threadgroup");
        assert_eq!(AddressSpace::Constant.to_string(), "constant");
        assert_eq!(AddressSpace::Thread.to_string(), "thread");
    }
}
