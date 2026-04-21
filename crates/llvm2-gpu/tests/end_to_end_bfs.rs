// llvm2-gpu/tests/end_to_end_bfs.rs - End-to-end BFS-style parallel_map test
//
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Copyright 2026 Dropbox, Inc. | License: Apache-2.0
//
// Reference: designs/2026-04-18-gpu-passes-pipeline.md
// Issue: andrewdyates/LLVM2#394

use llvm2_codegen::metal_emitter::MtlStorageMode;
use llvm2_gpu::address_space::AddressSpace;
use llvm2_gpu::memory_partition::BufferRole;
use llvm2_gpu::sample_bfs::{
    count_gpu_launches, run_sample_bfs, validate, SampleBfsSpec,
};

#[test]
fn pipeline_extracts_one_region() {
    let result = run_sample_bfs(SampleBfsSpec::default());
    assert_eq!(result.pipeline.region_count(), 1);
    assert_eq!(result.pipeline.regions[0].element_count, 1024);
}

#[test]
fn address_space_infer_marks_device() {
    let result = run_sample_bfs(SampleBfsSpec::default());
    let region = &result.pipeline.regions[0];
    assert!(!region.address_space.is_empty());
    for (_buf, space) in region.address_space.iter() {
        assert_eq!(space, AddressSpace::Device);
    }
}

#[test]
fn memory_partition_emits_shared_buffers() {
    let result = run_sample_bfs(SampleBfsSpec::default());
    assert_eq!(result.pipeline.buffer_plans.len(), 2);
    for plan in &result.pipeline.buffer_plans {
        assert_eq!(plan.storage, MtlStorageMode::Shared);
        assert_eq!(plan.address_space, AddressSpace::Device);
    }
}

#[test]
fn divergence_flatten_is_passthrough_for_parallel_map() {
    let result = run_sample_bfs(SampleBfsSpec::default());
    let stats = result.pipeline.divergence_stats;
    assert_eq!(stats.divergent_branches, 0);
    assert_eq!(stats.flattened, 0);
    assert!(!stats.has_opportunities());
}

#[test]
fn hetero_partition_recommends_gpu() {
    let result = run_sample_bfs(SampleBfsSpec::default());
    assert_eq!(result.pipeline.recommendations.len(), 1);
    assert_eq!(
        result.pipeline.recommendations[0].recommended_target,
        llvm2_lower::target_analysis::ComputeTarget::Gpu
    );
}

#[test]
fn launch_synth_produces_one_launch_with_correct_dispatch() {
    let result = run_sample_bfs(SampleBfsSpec::default());
    assert_eq!(result.pipeline.launches.len(), 1);
    let launch = &result.pipeline.launches[0];
    assert_eq!(launch.dispatch.grid_size.width, 1024);
    assert_eq!(launch.arg_count(), 2);
    assert!(launch.requires_sync);
    assert_eq!(launch.inputs().count(), 1);
    assert_eq!(launch.outputs().count(), 1);
    let input = launch.inputs().next().unwrap();
    assert_eq!(input.role, BufferRole::Input);
    assert_eq!(input.address_space, AddressSpace::Device);
}

#[test]
fn msl_emission_produces_compute_kernel() {
    let result = run_sample_bfs(SampleBfsSpec::default());
    let src = &result.msl_source;
    assert!(src.contains("#include <metal_stdlib>"));
    assert!(src.contains("using namespace metal;"));
    assert!(src.contains("kernel"));
}

#[test]
fn dispatch_plan_validates() {
    let result = run_sample_bfs(SampleBfsSpec::default());
    assert_eq!(count_gpu_launches(&result.dispatch), 1);
    validate(&result.graph, &result.dispatch).expect("plan validates");
}
