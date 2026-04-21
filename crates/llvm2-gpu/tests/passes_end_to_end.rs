// llvm2-gpu/tests/passes_end_to_end.rs - KernelExtract + AddressSpaceInfer
//                                         + MemoryPartition -> MSL integration
//
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Copyright 2026 Andrew Yates | License: Apache-2.0
//
// Reference: reports/2026-04-18-metal-current-state.md
// Issue: andrewdyates/LLVM2#394
//
// This test pipes a tMIR-shaped compute graph through the three passes
// that are in scope for the current slice of #394:
//   1. KernelExtract: extracts a DataParallel node into a KernelRegion.
//   2. AddressSpaceInfer: tags every input/output buffer Device.
//   3. MemoryPartition: produces BufferPlans with Shared storage.
//
// The resulting pass state is then fed to the existing
// `llvm2_codegen::metal_emitter` and the emitted MSL source is asserted
// on in detail: header, compute kernel declaration, address-space
// qualifiers on buffer args, and binding slots `[[buffer(0)]]` /
// `[[buffer(1)]]` / `[[buffer(2)]]` matching the LaunchSynth argument
// table.
//
// DivergenceFlatten is deliberately left as a pass-through (gated on
// #393). HeteroPartition and LaunchSynth are exercised here transitively
// because `GpuPipeline::default()` runs the full six-pass sequence.

use llvm2_codegen::metal_emitter::MtlStorageMode;
use llvm2_gpu::address_space::{AddressSpace, AddressSpaceInfer};
use llvm2_gpu::kernel_extract::{KernelExtract, KernelPattern};
use llvm2_gpu::memory_partition::{BufferRole, MemoryPartition};
use llvm2_gpu::sample_bfs::{
    emit_msl_for_region, run_sample_bfs, run_sample_map2, SampleBfsSpec,
};
use llvm2_gpu::{GpuPipeline, GpuPipelineConfig};
use llvm2_lower::target_analysis::ComputeTarget;

// ---------------------------------------------------------------------------
// Pass-by-pass happy path: KernelExtract -> AddressSpaceInfer ->
// MemoryPartition producing a Region with two input buffers and one
// output buffer, covered by a deterministic argument-buffer layout.
// ---------------------------------------------------------------------------

#[test]
fn kernel_extract_address_space_infer_memory_partition_three_buffers() {
    let result = run_sample_map2(SampleBfsSpec::default());

    // 1. KernelExtract produced one region covering one ComputeNode.
    assert_eq!(result.pipeline.region_count(), 1);
    let region = &result.pipeline.regions[0];
    assert_eq!(region.nodes.len(), 1, "one node per region in the current slice");
    assert_eq!(region.pattern, KernelPattern::ParallelMap);
    assert_eq!(region.input_buffers.len(), 2, "FADD has two inputs");
    assert_eq!(region.output_buffers.len(), 1, "one fused output");

    // 2. AddressSpaceInfer tagged every buffer `Device`.
    assert_eq!(region.address_space.len(), 3);
    for (_buf, space) in region.address_space.iter() {
        assert_eq!(space, AddressSpace::Device, "default policy: inputs and outputs are device");
    }

    // 3. MemoryPartition produced a BufferPlan per buffer, all Shared
    //    storage (UMA default).
    assert_eq!(result.pipeline.buffer_plans.len(), 3);
    let mut input_plans = 0;
    let mut output_plans = 0;
    for plan in &result.pipeline.buffer_plans {
        assert_eq!(plan.storage, MtlStorageMode::Shared);
        assert_eq!(plan.address_space, AddressSpace::Device);
        match plan.role {
            BufferRole::Input => input_plans += 1,
            BufferRole::Output => output_plans += 1,
        }
    }
    assert_eq!(input_plans, 2);
    assert_eq!(output_plans, 1);
}

// ---------------------------------------------------------------------------
// LaunchSynth argument table is consistent with MemoryPartition plans.
// ---------------------------------------------------------------------------

#[test]
fn launch_synth_argument_table_mirrors_memory_partition() {
    let result = run_sample_map2(SampleBfsSpec::default());
    assert_eq!(result.pipeline.launches.len(), 1);
    let launch = &result.pipeline.launches[0];

    // Three arguments: two inputs + one output, bound in that order.
    assert_eq!(launch.arg_count(), 3);
    assert_eq!(launch.inputs().count(), 2);
    assert_eq!(launch.outputs().count(), 1);

    // Argument bindings are dense and sequential starting at 0, matching
    // the MSL emitter's `[[buffer(N)]]` slot conventions.
    for (expected, arg) in launch.arguments.iter().enumerate() {
        assert_eq!(arg.binding as usize, expected);
        assert_eq!(arg.storage, MtlStorageMode::Shared);
        assert_eq!(arg.address_space, AddressSpace::Device);
    }

    // The first two arguments are inputs, the last is the output.
    assert_eq!(launch.arguments[0].role, BufferRole::Input);
    assert_eq!(launch.arguments[1].role, BufferRole::Input);
    assert_eq!(launch.arguments[2].role, BufferRole::Output);

    // HeteroPartition picked GPU for this DataParallel region.
    assert_eq!(
        result.pipeline.recommendations[0].recommended_target,
        ComputeTarget::Gpu
    );
}

// ---------------------------------------------------------------------------
// MSL emission covers the compute-shader header + argument-buffer layout
// + thread_position_in_grid binding.
// ---------------------------------------------------------------------------

#[test]
fn msl_source_has_compute_shader_header_and_argument_layout() {
    let result = run_sample_map2(SampleBfsSpec::default());
    let src = &result.msl_source;

    // --- Header ---
    assert!(
        src.contains("#include <metal_stdlib>"),
        "missing Metal stdlib include: {src}"
    );
    assert!(
        src.contains("using namespace metal;"),
        "missing metal namespace: {src}"
    );
    assert!(
        src.contains("Generated by LLVM2"),
        "missing LLVM2 provenance comment: {src}"
    );

    // --- Argument buffer layout (2 inputs + 1 output) ---
    // FADD dominant op -> float element type -> `const device float*`
    // for inputs and `device float*` for output.
    assert!(
        src.contains("kernel void llvm2_map2_"),
        "expected parallel_map2 kernel name: {src}"
    );
    assert!(
        src.contains("const device float* a [[buffer(0)]]"),
        "missing buffer(0) input argument: {src}"
    );
    assert!(
        src.contains("const device float* b [[buffer(1)]]"),
        "missing buffer(1) input argument: {src}"
    );
    assert!(
        src.contains("device float* output  [[buffer(2)]]"),
        "missing buffer(2) output argument: {src}"
    );
    assert!(
        src.contains("uint tid [[thread_position_in_grid]]"),
        "missing thread-position grid binding: {src}"
    );

    // --- Body: a + b elementwise ---
    assert!(
        src.contains("output[tid] = a[tid] + b[tid]"),
        "missing element-wise FADD body: {src}"
    );
    assert!(
        src.contains("if (tid >= 1024u) return;"),
        "missing grid bounds guard for 1024 elements: {src}"
    );

    // MSL byte count is a deterministic regression indicator: the
    // emitter template is fixed and the pass pipeline is deterministic
    // over a fixed input graph. Today the 2-input FADD @1024 elements
    // kernel lands at 394 bytes; if this jumps we want to see the
    // diff in review.
    let len = src.len();
    assert_eq!(
        len, 394,
        "MSL source byte count drifted — review the emitter/template diff \
         before updating this assertion. Current source:\n---\n{src}\n---"
    );
}

// ---------------------------------------------------------------------------
// Existing BFS single-input sample still emits a compute-kernel shape
// even though the `"BFS_STEP"` dominant-op falls through the emitter's
// identity-map fallback.
// ---------------------------------------------------------------------------

#[test]
fn single_input_bfs_sample_still_emits_parallel_map_kernel() {
    let result = run_sample_bfs(SampleBfsSpec::default());
    let src = &result.msl_source;
    assert!(src.contains("#include <metal_stdlib>"));
    assert!(src.contains("using namespace metal;"));
    assert!(src.contains("kernel void llvm2_map_"));
    // Single-input parallel_map exposes one input buffer + one output
    // buffer.
    assert!(src.contains("[[buffer(0)]]"));
    assert!(src.contains("[[buffer(1)]]"));
    // No second input buffer should appear.
    assert!(!src.contains("[[buffer(2)]]"));
}

// ---------------------------------------------------------------------------
// Re-run the three passes manually over the same region and verify the
// pipeline orchestrator's output matches the hand-run order. Guards
// against silent pass-reordering regressions.
// ---------------------------------------------------------------------------

#[test]
fn manual_three_pass_order_matches_pipeline() {
    let result = run_sample_map2(SampleBfsSpec::default());
    let graph = &result.graph;

    // Hand-run the passes.
    let mut regions = KernelExtract.run(graph);
    AddressSpaceInfer.run(&mut regions);
    let manual_plans = MemoryPartition.run(&mut regions);

    // Region counts and shape agree with the orchestrator's run.
    assert_eq!(regions.len(), result.pipeline.regions.len());
    assert_eq!(regions[0].input_buffers, result.pipeline.regions[0].input_buffers);
    assert_eq!(regions[0].output_buffers, result.pipeline.regions[0].output_buffers);

    // Address-space annotations match element-for-element.
    for (buf, space) in regions[0].address_space.iter() {
        assert_eq!(
            result.pipeline.regions[0].address_space.get(buf),
            Some(space)
        );
    }

    // Buffer plan count matches.
    assert_eq!(manual_plans.len(), result.pipeline.buffer_plans.len());
}

// ---------------------------------------------------------------------------
// Pipeline sans HeteroPartition + LaunchSynth still produces a valid MSL
// kernel when emit_msl_for_region is driven manually. This proves the
// three passes (KernelExtract + AddressSpaceInfer + MemoryPartition)
// carry enough context on their own to reach MSL.
// ---------------------------------------------------------------------------

#[test]
fn three_passes_alone_suffice_for_msl_emission() {
    let cfg = GpuPipelineConfig {
        kernel_extract: true,
        address_space: true,
        memory_partition: true,
        divergence_flatten: false,
        hetero_partition: false,
        launch_synth: false,
        threadgroup_size: 256,
    };
    let pipeline = GpuPipeline::new(cfg);
    let result = run_sample_map2(SampleBfsSpec::default());
    let three_pass_out = pipeline.run(&result.graph);

    assert_eq!(three_pass_out.region_count(), 1);
    assert!(three_pass_out.launches.is_empty(), "LaunchSynth disabled");
    assert!(three_pass_out.recommendations.is_empty(), "HeteroPartition disabled");

    let region = &three_pass_out.regions[0];
    let src = emit_msl_for_region(&result.graph, region, &three_pass_out);
    assert!(src.contains("#include <metal_stdlib>"));
    assert!(src.contains("kernel void llvm2_map2_"));
    assert!(src.contains("[[buffer(0)]]"));
    assert!(src.contains("[[buffer(1)]]"));
    assert!(src.contains("[[buffer(2)]]"));
}
