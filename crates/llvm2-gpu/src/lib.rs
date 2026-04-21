// llvm2-gpu - GPU passes (Metal first)
//
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Copyright 2026 Andrew Yates | License: Apache-2.0
//
// Reference: designs/2026-04-18-gpu-passes-pipeline.md
// Issue: andrewdyates/LLVM2#394 (Part of #390, tla2 supremacy blocker 5)

//! GPU pass pipeline for LLVM2 (Metal first).
//!
//! This crate implements six structural passes that sit between the
//! existing [`llvm2_lower::compute_graph`] heterogeneous compute analysis
//! and the existing [`llvm2_codegen::metal_emitter`] MSL emitter:
//!
//! 1. [`kernel_extract::KernelExtract`] — identify parallel regions and
//!    extract them as self-contained kernels.
//! 2. [`address_space::AddressSpaceInfer`] — tag buffers with MSL
//!    address spaces (`device`, `threadgroup`, `constant`, `thread`).
//! 3. [`memory_partition::MemoryPartition`] — split host/device/shared
//!    allocations and pick Metal storage modes.
//! 4. [`divergence_flatten::DivergenceFlatten`] — flatten warp-divergent
//!    control flow into predicated arithmetic.
//! 5. [`hetero_partition::HeteroPartition`] — pick CPU vs GPU per region
//!    using the cost-model profitability analyzer.
//! 6. [`launch_synth::LaunchSynth`] — synthesize Metal launch glue
//!    (grid/threadgroup dims, argument tables, storage modes).
//!
//! The passes run in the order above. Each is idempotent. Each is opt-in
//! via [`GpuPipelineConfig`] toggles and the crate-level `gpu-passes`
//! cargo feature.
//!
//! See [`pipeline::GpuPipeline`] for the orchestrator and
//! [`sample_bfs`] for the end-to-end BFS-style parallel_map sample that
//! wires KernelExtract -> LaunchSynth -> MSL emission.

pub mod address_space;
pub mod divergence_flatten;
pub mod hetero_partition;
pub mod kernel_extract;
pub mod launch_synth;
pub mod memory_partition;
pub mod pipeline;
pub mod region;
pub mod sample_bfs;

pub use address_space::{AddressSpace, AddressSpaceInfer, AddressSpaceMap};
pub use divergence_flatten::{DivergenceFlatten, DivergenceStats};
pub use hetero_partition::HeteroPartition;
pub use kernel_extract::{KernelExtract, KernelPattern};
pub use launch_synth::{LaunchSynth, MetalLaunch, LaunchArgument};
pub use memory_partition::{BufferPlan, MemoryPartition};
pub use pipeline::{GpuPipeline, GpuPipelineConfig, GpuPipelineOutput};
pub use region::{BufferId, KernelRegion, RegionId};
pub use sample_bfs::{emit_msl_for_region, run_sample_bfs, run_sample_map2};
