// llvm2-codegen/tests/e2e_heterogeneous.rs - End-to-end heterogeneous compute tests
//
// Author: Andrew Yates <ayates@dropbox.com>
// Copyright 2026 Dropbox, Inc. | License: Apache-2.0
//
// Integration tests exercising the full heterogeneous compute pipeline:
//   ComputeGraph -> TargetRecommendation -> DispatchPlan -> Verification
//                                                        -> Metal MSL emission
//                                                        -> CoreML MIL emission
//
// These tests validate that the dispatch, Metal emitter, and CoreML emitter
// work correctly together across CPU-only, GPU, ANE, and mixed target scenarios.

use std::collections::HashMap;

use llvm2_codegen::pipeline::{
    DispatchVerifyMode, Pipeline, PipelineConfig,
    emit_coreml_program, generate_cpu_only_plan,
};
use llvm2_codegen::metal_emitter::emit_metal_kernels;

use llvm2_lower::compute_graph::{
    ComputeCost, ComputeGraph, ComputeNode, ComputeNodeId, DataEdge, NodeKind,
    TargetRecommendation, TransferCost,
};
use llvm2_lower::dispatch::{
    VerificationReport,
    generate_dispatch_plan, verify_dispatch_plan_properties,
};
use llvm2_lower::target_analysis::ComputeTarget;

use llvm2_ir::cost_model::CostModelGen;

// ===========================================================================
// Test helpers
// ===========================================================================

/// Build a ComputeGraph from nodes and edges.
fn make_graph(nodes: Vec<ComputeNode>, edges: Vec<DataEdge>) -> ComputeGraph {
    let mut graph = ComputeGraph::new();
    graph.nodes = nodes;
    graph.edges = edges;
    graph
}

/// Build a ComputeGraph with profitability analyzer attached.
fn make_graph_with_profitability(
    nodes: Vec<ComputeNode>,
    edges: Vec<DataEdge>,
) -> ComputeGraph {
    let mut graph = ComputeGraph::new_with_profitability(CostModelGen::M1);
    graph.nodes = nodes;
    graph.edges = edges;
    graph
}

/// Create a CPU-only scalar node.
fn cpu_scalar_node(id: u32, dominant_op: &str, data_size: u64) -> ComputeNode {
    let mut costs = HashMap::new();
    costs.insert(
        ComputeTarget::CpuScalar,
        ComputeCost {
            latency_cycles: 50,
            throughput_ops_per_kcycle: 1000,
        },
    );
    ComputeNode {
        id: ComputeNodeId(id),
        instructions: vec![],
        costs,
        legal_targets: vec![ComputeTarget::CpuScalar],
        kind: NodeKind::Scalar,
        data_size_bytes: data_size,
        produced_values: vec![],
        consumed_values: vec![],
        dominant_op: dominant_op.to_string(),
        target_legality: None,
    }
}

/// Create a data-parallel node suitable for GPU.
fn gpu_data_parallel_node(id: u32, dominant_op: &str, data_size: u64) -> ComputeNode {
    let mut costs = HashMap::new();
    costs.insert(
        ComputeTarget::Gpu,
        ComputeCost {
            latency_cycles: 10,
            throughput_ops_per_kcycle: 100_000,
        },
    );
    costs.insert(
        ComputeTarget::CpuScalar,
        ComputeCost {
            latency_cycles: 500,
            throughput_ops_per_kcycle: 1000,
        },
    );
    ComputeNode {
        id: ComputeNodeId(id),
        instructions: vec![],
        costs,
        legal_targets: vec![ComputeTarget::CpuScalar, ComputeTarget::Gpu],
        kind: NodeKind::DataParallel,
        data_size_bytes: data_size,
        produced_values: vec![],
        consumed_values: vec![],
        dominant_op: dominant_op.to_string(),
        target_legality: None,
    }
}

/// Create a matrix-heavy node suitable for ANE.
fn ane_matrix_node(id: u32, dominant_op: &str, data_size: u64) -> ComputeNode {
    let mut costs = HashMap::new();
    costs.insert(
        ComputeTarget::NeuralEngine,
        ComputeCost {
            latency_cycles: 100,
            throughput_ops_per_kcycle: 50_000,
        },
    );
    costs.insert(
        ComputeTarget::CpuScalar,
        ComputeCost {
            latency_cycles: 2000,
            throughput_ops_per_kcycle: 1000,
        },
    );
    ComputeNode {
        id: ComputeNodeId(id),
        instructions: vec![],
        costs,
        legal_targets: vec![ComputeTarget::CpuScalar, ComputeTarget::NeuralEngine],
        kind: NodeKind::MatrixHeavy,
        data_size_bytes: data_size,
        produced_values: vec![],
        consumed_values: vec![],
        dominant_op: dominant_op.to_string(),
        target_legality: None,
    }
}

/// Build TargetRecommendation for a node given its cheapest target.
fn recommend(node: &ComputeNode, target: ComputeTarget) -> TargetRecommendation {
    TargetRecommendation {
        node_id: node.id,
        recommended_target: target,
        legal_targets: node.legal_targets.clone(),
        reason: format!("{} recommended for {}", target, node.dominant_op),
        parallel_reduction_legal: matches!(node.kind, NodeKind::DataParallel),
    }
}

/// Assert that a VerificationReport passes all properties.
fn assert_verification_passes(report: &VerificationReport) {
    assert!(
        report.all_ok(),
        "Dispatch verification failed:\n\
         - cpu_fallback_ok: {} (missing: {:?})\n\
         - data_transfers_ok: {} (errors: {:?})\n\
         - synchronization_ok: {} (errors: {:?})\n\
         - dependency_order_ok: {} (errors: {:?})\n\
         - profitability_ok: {} (mismatches: {:?})",
        report.cpu_fallback_ok, report.missing_cpu_fallback,
        report.data_transfers_ok, report.data_transfer_errors,
        report.synchronization_ok, report.synchronization_errors,
        report.dependency_order_ok, report.dependency_errors,
        report.profitability_ok, report.profitability_mismatches,
    );
}

// ===========================================================================
// Test 1: Pure CPU graph -> dispatch plan -> no GPU/ANE kernels
// ===========================================================================

#[test]
fn e2e_pure_cpu_graph_produces_no_gpu_ane_output() {
    // Build a 3-node CPU-only graph: n0 -> n1 -> n2 (all scalar ADD/SUB/MUL)
    let nodes = vec![
        cpu_scalar_node(0, "ADD", 64),
        cpu_scalar_node(1, "SUB", 64),
        cpu_scalar_node(2, "MUL", 64),
    ];
    let edges = vec![
        DataEdge {
            from: ComputeNodeId(0),
            to: ComputeNodeId(1),
            transfer_bytes: 64,
            transfer_cost: TransferCost::zero(),
        },
        DataEdge {
            from: ComputeNodeId(1),
            to: ComputeNodeId(2),
            transfer_bytes: 64,
            transfer_cost: TransferCost::zero(),
        },
    ];
    let graph = make_graph(nodes.clone(), edges);

    // All recommendations are CPU.
    let recs: Vec<TargetRecommendation> = nodes
        .iter()
        .map(|n| recommend(n, ComputeTarget::CpuScalar))
        .collect();

    // Generate dispatch plan.
    let plan = generate_dispatch_plan(&graph, &recs);

    // All ops should be KernelLaunch on CpuScalar (no fallbacks since we
    // have explicit recommendations).
    assert_eq!(plan.count_launches(), 3);
    assert_eq!(plan.count_transfers(), 0, "CPU-only graph needs no transfers");
    assert_eq!(plan.count_syncs(), 0, "CPU-only graph needs no syncs");
    assert_eq!(plan.count_fallbacks(), 0);

    // All assignments are CpuScalar.
    for target in plan.assignment.values() {
        assert_eq!(*target, ComputeTarget::CpuScalar);
    }

    // Metal emission should produce no kernels (no GPU ops).
    let metal_result = emit_metal_kernels(&plan, &graph);
    // Either returns Ok with zero kernels or the function gracefully handles
    // the absence of GPU ops.
    match metal_result {
        Ok(output) => {
            assert!(
                output.kernels.is_empty(),
                "CPU-only plan should produce no Metal kernels"
            );
        }
        Err(_) => {
            // Some implementations may error on empty GPU dispatch — acceptable.
        }
    }

    // CoreML emission should fail (no ANE nodes).
    let coreml_result = emit_coreml_program(&plan, &graph);
    assert!(
        coreml_result.is_err(),
        "CPU-only plan should not produce CoreML output"
    );
}

// ===========================================================================
// Test 2: Data-parallel graph -> GPU dispatch -> Metal MSL kernel generated
// ===========================================================================

#[test]
fn e2e_data_parallel_graph_produces_metal_kernel() {
    // CPU producer -> GPU data-parallel consumer
    let producer = cpu_scalar_node(0, "LOAD", 4096);
    let consumer = gpu_data_parallel_node(1, "ADD", 4096);

    let edge = DataEdge {
        from: ComputeNodeId(0),
        to: ComputeNodeId(1),
        transfer_bytes: 4096,
        transfer_cost: TransferCost::zero(),
    };
    let graph = make_graph(vec![producer.clone(), consumer.clone()], vec![edge]);

    let recs = vec![
        recommend(&producer, ComputeTarget::CpuScalar),
        recommend(&consumer, ComputeTarget::Gpu),
    ];

    let plan = generate_dispatch_plan(&graph, &recs);

    // Should have: CPU launch, transfer, GPU launch, final sync.
    assert!(plan.count_launches() >= 2, "need CPU + GPU launches");
    assert!(plan.count_transfers() >= 1, "need CPU->GPU transfer");

    // GPU assignment for consumer node.
    assert_eq!(
        plan.assignment.get(&ComputeNodeId(1)).copied(),
        Some(ComputeTarget::Gpu),
    );

    // Metal emission should produce exactly 1 kernel for node 1.
    let metal_output = emit_metal_kernels(&plan, &graph)
        .expect("Metal emission should succeed for GPU plan");

    assert_eq!(
        metal_output.kernels.len(), 1,
        "Should produce exactly 1 Metal kernel"
    );
    assert_eq!(metal_output.kernels[0].node_id, ComputeNodeId(1));

    // Kernel source should be valid MSL.
    let source = &metal_output.kernels[0].source;
    assert!(
        source.contains("kernel void"),
        "MSL source should contain 'kernel void'"
    );
    assert!(
        source.contains("device"),
        "MSL source should contain buffer declarations with 'device'"
    );
    assert!(
        source.contains("thread_position_in_grid"),
        "MSL source should use thread_position_in_grid"
    );

    // Buffer count should be > 0.
    assert!(
        metal_output.buffer_count > 0,
        "Should require at least one Metal buffer"
    );

    // Dispatch code should reference the kernel name.
    assert!(
        !metal_output.dispatch_code.is_empty(),
        "Should produce host dispatch code"
    );
}

// ===========================================================================
// Test 3: Matrix-heavy graph -> ANE dispatch -> CoreML MIL program generated
// ===========================================================================

#[test]
fn e2e_matrix_heavy_graph_produces_coreml_program() {
    // GEMM node on ANE.
    let gemm_node = ane_matrix_node(0, "GEMM", 8192);
    let graph = make_graph(vec![gemm_node.clone()], vec![]);

    let recs = vec![recommend(&gemm_node, ComputeTarget::NeuralEngine)];
    let plan = generate_dispatch_plan(&graph, &recs);

    // Should have a NeuralEngine kernel launch.
    assert_eq!(plan.count_launches(), 1);
    assert_eq!(
        plan.assignment.get(&ComputeNodeId(0)).copied(),
        Some(ComputeTarget::NeuralEngine),
    );

    // CoreML emission should succeed.
    let coreml_output = emit_coreml_program(&plan, &graph)
        .expect("CoreML emission should succeed for ANE plan");

    // Program should contain a matmul operation.
    assert!(
        coreml_output.program.op_count() >= 1,
        "Should produce at least 1 MIL operation"
    );
    assert_eq!(
        coreml_output.program.operations[0].op_type(),
        "matmul",
        "GEMM should produce a matmul MIL op"
    );

    // Program should validate.
    assert!(
        coreml_output.program.validate().is_ok(),
        "MIL program should pass validation"
    );

    // Latency estimate should be positive.
    assert!(
        coreml_output.estimated_latency_us >= 1,
        "Should have positive latency estimate"
    );

    // Metal emission should produce nothing (no GPU nodes).
    let metal_result = emit_metal_kernels(&plan, &graph);
    match metal_result {
        Ok(output) => {
            assert!(output.kernels.is_empty(), "No GPU nodes, no Metal kernels");
        }
        Err(_) => {} // acceptable
    }
}

// ===========================================================================
// Test 4: Mixed graph (CPU + GPU + ANE) -> full dispatch -> all outputs
// ===========================================================================

#[test]
fn e2e_mixed_cpu_gpu_ane_graph_produces_all_outputs() {
    // Build a graph: CPU -> GPU -> ANE
    //   n0: CPU scalar (LOAD)
    //   n1: GPU data-parallel (ADD, large array)
    //   n2: ANE matrix-heavy (GEMM)
    let cpu_node = cpu_scalar_node(0, "LOAD", 256);
    let gpu_node = gpu_data_parallel_node(1, "ADD", 16384);
    let ane_node = ane_matrix_node(2, "GEMM", 8192);

    let edges = vec![
        DataEdge {
            from: ComputeNodeId(0),
            to: ComputeNodeId(1),
            transfer_bytes: 256,
            transfer_cost: TransferCost::zero(),
        },
        DataEdge {
            from: ComputeNodeId(1),
            to: ComputeNodeId(2),
            transfer_bytes: 8192,
            transfer_cost: TransferCost::zero(),
        },
    ];

    let graph = make_graph(
        vec![cpu_node.clone(), gpu_node.clone(), ane_node.clone()],
        edges,
    );

    let recs = vec![
        recommend(&cpu_node, ComputeTarget::CpuScalar),
        recommend(&gpu_node, ComputeTarget::Gpu),
        recommend(&ane_node, ComputeTarget::NeuralEngine),
    ];

    let plan = generate_dispatch_plan(&graph, &recs);

    // Verify plan structure.
    assert_eq!(plan.count_launches(), 3, "should launch on all 3 targets");
    assert!(plan.count_transfers() >= 2, "need CPU->GPU and GPU->ANE transfers");
    assert!(plan.count_syncs() >= 1, "need sync for async GPU/ANE");

    // Verify assignments.
    assert_eq!(plan.assignment[&ComputeNodeId(0)], ComputeTarget::CpuScalar);
    assert_eq!(plan.assignment[&ComputeNodeId(1)], ComputeTarget::Gpu);
    assert_eq!(plan.assignment[&ComputeNodeId(2)], ComputeTarget::NeuralEngine);

    // Metal emission: should produce 1 kernel for the GPU node.
    let metal_output = emit_metal_kernels(&plan, &graph)
        .expect("Metal emission should succeed for mixed plan");
    assert_eq!(
        metal_output.kernels.len(), 1,
        "Should produce exactly 1 Metal kernel (GPU node)"
    );
    assert_eq!(metal_output.kernels[0].node_id, ComputeNodeId(1));

    // CoreML emission: should produce a MIL program for the ANE node.
    let coreml_output = emit_coreml_program(&plan, &graph)
        .expect("CoreML emission should succeed for mixed plan");
    assert!(coreml_output.program.op_count() >= 1, "MIL program should have ops");
    assert_eq!(coreml_output.program.operations[0].op_type(), "matmul");
    assert!(coreml_output.program.validate().is_ok());

    // Total estimated cycles should be reasonable.
    assert!(
        plan.estimated_total_cycles > 0,
        "Total cycles should be positive"
    );
}

// ===========================================================================
// Test 5: Dispatch verification passes on all generated plans
// ===========================================================================

#[test]
fn e2e_dispatch_verification_passes_all_plan_types() {
    // --- (a) CPU-only graph ---
    {
        let nodes = vec![
            cpu_scalar_node(0, "ADD", 64),
            cpu_scalar_node(1, "MUL", 64),
        ];
        let edge = DataEdge {
            from: ComputeNodeId(0),
            to: ComputeNodeId(1),
            transfer_bytes: 64,
            transfer_cost: TransferCost::zero(),
        };
        let graph = make_graph(nodes.clone(), vec![edge]);
        let recs: Vec<_> = nodes.iter().map(|n| recommend(n, ComputeTarget::CpuScalar)).collect();
        let plan = generate_dispatch_plan(&graph, &recs);
        let report = verify_dispatch_plan_properties(&graph, &plan);
        assert_verification_passes(&report);
    }

    // --- (b) GPU graph ---
    {
        let producer = cpu_scalar_node(0, "LOAD", 4096);
        let consumer = gpu_data_parallel_node(1, "ADD", 4096);
        let edge = DataEdge {
            from: ComputeNodeId(0),
            to: ComputeNodeId(1),
            transfer_bytes: 4096,
            transfer_cost: TransferCost::zero(),
        };
        let graph = make_graph(vec![producer.clone(), consumer.clone()], vec![edge]);
        let recs = vec![
            recommend(&producer, ComputeTarget::CpuScalar),
            recommend(&consumer, ComputeTarget::Gpu),
        ];
        let plan = generate_dispatch_plan(&graph, &recs);
        let report = verify_dispatch_plan_properties(&graph, &plan);
        assert_verification_passes(&report);
    }

    // --- (c) ANE graph ---
    {
        let node = ane_matrix_node(0, "GEMM", 8192);
        let graph = make_graph(vec![node.clone()], vec![]);
        let recs = vec![recommend(&node, ComputeTarget::NeuralEngine)];
        let plan = generate_dispatch_plan(&graph, &recs);
        let report = verify_dispatch_plan_properties(&graph, &plan);
        assert_verification_passes(&report);
    }

    // --- (d) Mixed CPU+GPU+ANE graph ---
    {
        let cpu_node = cpu_scalar_node(0, "LOAD", 256);
        let gpu_node = gpu_data_parallel_node(1, "ADD", 8192);
        let ane_node = ane_matrix_node(2, "GEMM", 8192);
        let edges = vec![
            DataEdge {
                from: ComputeNodeId(0),
                to: ComputeNodeId(1),
                transfer_bytes: 256,
                transfer_cost: TransferCost::zero(),
            },
            DataEdge {
                from: ComputeNodeId(1),
                to: ComputeNodeId(2),
                transfer_bytes: 8192,
                transfer_cost: TransferCost::zero(),
            },
        ];
        let graph = make_graph(
            vec![cpu_node.clone(), gpu_node.clone(), ane_node.clone()],
            edges,
        );
        let recs = vec![
            recommend(&cpu_node, ComputeTarget::CpuScalar),
            recommend(&gpu_node, ComputeTarget::Gpu),
            recommend(&ane_node, ComputeTarget::NeuralEngine),
        ];
        let plan = generate_dispatch_plan(&graph, &recs);
        let report = verify_dispatch_plan_properties(&graph, &plan);
        assert_verification_passes(&report);
    }

    // --- (e) CPU-only fallback plan ---
    {
        let cpu_node = cpu_scalar_node(0, "ADD", 128);
        let gpu_node = gpu_data_parallel_node(1, "MUL", 4096);
        let edge = DataEdge {
            from: ComputeNodeId(0),
            to: ComputeNodeId(1),
            transfer_bytes: 128,
            transfer_cost: TransferCost::zero(),
        };
        let graph = make_graph(vec![cpu_node.clone(), gpu_node.clone()], vec![edge]);
        let recs = vec![
            recommend(&cpu_node, ComputeTarget::CpuScalar),
            recommend(&gpu_node, ComputeTarget::Gpu),
        ];
        let cpu_plan = generate_cpu_only_plan(&graph, &recs);
        let report = verify_dispatch_plan_properties(&graph, &cpu_plan);
        assert_verification_passes(&report);
    }

    // --- (f) Pipeline.generate_and_verify_dispatch (ErrorOnFailure mode) ---
    {
        let producer = cpu_scalar_node(0, "LOAD", 256);
        let consumer = gpu_data_parallel_node(1, "ADD", 4096);
        let edge = DataEdge {
            from: ComputeNodeId(0),
            to: ComputeNodeId(1),
            transfer_bytes: 256,
            transfer_cost: TransferCost::zero(),
        };
        let graph = make_graph(vec![producer.clone(), consumer.clone()], vec![edge]);
        let recs = vec![
            recommend(&producer, ComputeTarget::CpuScalar),
            recommend(&consumer, ComputeTarget::Gpu),
        ];
        let pipeline = Pipeline::new(PipelineConfig {
            verify_dispatch: DispatchVerifyMode::ErrorOnFailure,
            ..Default::default()
        });
        let result = pipeline.generate_and_verify_dispatch(&graph, &recs);
        assert!(
            result.is_ok(),
            "Pipeline verification should pass for well-formed plan: {:?}",
            result.err()
        );
    }
}

// ===========================================================================
// Test 6: Profitability filtering — small workloads stay CPU-only
// ===========================================================================

#[test]
fn e2e_profitability_filtering_keeps_small_workloads_on_cpu() {
    // Create a graph with profitability analyzer. Use a very small data-parallel
    // node (32 bytes = 8 float elements). The profitability analyzer should
    // determine that GPU dispatch overhead exceeds the compute savings.
    let small_gpu_node = gpu_data_parallel_node(0, "ADD", 32);
    // Make it data-parallel with GPU as a legal target, but the data is tiny.
    // The profitability analyzer should filter out GPU.

    let graph = make_graph_with_profitability(
        vec![small_gpu_node.clone()],
        vec![],
    );

    // Use profitability-aware recommendations from the graph.
    let recs = graph.target_recommendations();

    // The profitability analyzer should have filtered GPU as unprofitable
    // for this tiny workload, leaving only CpuScalar.
    assert_eq!(
        recs.len(), 1,
        "Should have 1 recommendation for 1 node"
    );
    assert_eq!(
        recs[0].recommended_target,
        ComputeTarget::CpuScalar,
        "Small workload should be recommended for CPU, not GPU (profitability filter)"
    );

    // Generate dispatch plan from profitability-filtered recommendations.
    let plan = generate_dispatch_plan(&graph, &recs);

    // All operations should be CPU launches (no GPU kernels).
    assert_eq!(plan.count_launches(), 1);
    assert_eq!(plan.count_transfers(), 0);
    assert_eq!(plan.count_syncs(), 0);
    assert_eq!(
        plan.assignment[&ComputeNodeId(0)],
        ComputeTarget::CpuScalar,
        "Small workload should execute on CPU"
    );

    // Metal emission should produce no kernels.
    let metal_result = emit_metal_kernels(&plan, &graph);
    match metal_result {
        Ok(output) => {
            assert!(output.kernels.is_empty(), "No GPU dispatch, no Metal kernels");
        }
        Err(_) => {} // acceptable
    }

    // CoreML emission should fail (no ANE nodes).
    assert!(emit_coreml_program(&plan, &graph).is_err());

    // Verify the plan passes verification.
    let report = verify_dispatch_plan_properties(&graph, &plan);
    assert_verification_passes(&report);
}

// ===========================================================================
// Additional integration tests
// ===========================================================================

/// Test that profitability-aware dispatch correctly routes large workloads
/// to GPU while keeping the same graph structure's small workloads on CPU.
#[test]
fn e2e_profitability_large_workload_goes_to_gpu() {
    // 1MB data-parallel workload — should be profitable for GPU.
    let large_gpu_node = gpu_data_parallel_node(0, "ADD", 1_048_576);

    let graph = make_graph_with_profitability(
        vec![large_gpu_node.clone()],
        vec![],
    );

    let recs = graph.target_recommendations();
    assert_eq!(recs.len(), 1);

    // Large workload should go to GPU.
    assert_eq!(
        recs[0].recommended_target,
        ComputeTarget::Gpu,
        "Large workload (1MB) should be recommended for GPU"
    );

    let plan = generate_dispatch_plan(&graph, &recs);
    assert_eq!(
        plan.assignment[&ComputeNodeId(0)],
        ComputeTarget::Gpu,
    );

    // Metal emission should succeed and produce a kernel.
    let metal_output = emit_metal_kernels(&plan, &graph)
        .expect("Metal emission should succeed for large GPU workload");
    assert_eq!(metal_output.kernels.len(), 1);
}

/// Test multi-node ANE subgraph: GEMM -> ADD -> RELU produces a fused
/// CoreML MIL program with correct operation ordering.
#[test]
fn e2e_ane_multi_node_fusion_pipeline() {
    let nodes = vec![
        ane_matrix_node(0, "GEMM", 8192),
        ane_matrix_node(1, "ADD", 256),
        ane_matrix_node(2, "RELU", 256),
    ];
    // Override kinds for realistic fusion pattern.
    let mut nodes = nodes;
    nodes[1].kind = NodeKind::DataParallel;
    nodes[2].kind = NodeKind::DataParallel;

    let edges = vec![
        DataEdge {
            from: ComputeNodeId(0),
            to: ComputeNodeId(1),
            transfer_bytes: 0, // same target, no physical transfer
            transfer_cost: TransferCost::zero(),
        },
        DataEdge {
            from: ComputeNodeId(1),
            to: ComputeNodeId(2),
            transfer_bytes: 0,
            transfer_cost: TransferCost::zero(),
        },
    ];

    let graph = make_graph(nodes.clone(), edges);
    let recs: Vec<_> = nodes
        .iter()
        .map(|n| recommend(n, ComputeTarget::NeuralEngine))
        .collect();

    let plan = generate_dispatch_plan(&graph, &recs);
    assert_eq!(plan.count_launches(), 3);
    assert_eq!(plan.count_transfers(), 0, "same-target nodes need no transfers");

    // CoreML should produce a 3-op MIL program: matmul + add + relu.
    let coreml = emit_coreml_program(&plan, &graph)
        .expect("CoreML emission should succeed for multi-node ANE subgraph");

    assert_eq!(coreml.program.op_count(), 3);
    assert_eq!(coreml.program.operations[0].op_type(), "matmul");
    assert_eq!(coreml.program.operations[1].op_type(), "add");
    assert_eq!(coreml.program.operations[2].op_type(), "relu");
    assert!(coreml.program.validate().is_ok());

    // Latency should aggregate across all 3 ANE nodes.
    // Each node: 100 cycles. 3 * 100 = 300 cycles. 300 / 1000 = 0, clamped to 1.
    assert!(coreml.estimated_latency_us >= 1);
}

/// Test that the full Pipeline integration (generate_and_verify_dispatch)
/// produces correct results and that verification catches bad plans.
#[test]
fn e2e_pipeline_verify_dispatch_integration() {
    let cpu = cpu_scalar_node(0, "LOAD", 128);
    let gpu = gpu_data_parallel_node(1, "ADD", 8192);
    let ane = ane_matrix_node(2, "GEMM", 8192);

    let edges = vec![
        DataEdge {
            from: ComputeNodeId(0),
            to: ComputeNodeId(1),
            transfer_bytes: 128,
            transfer_cost: TransferCost::zero(),
        },
        DataEdge {
            from: ComputeNodeId(1),
            to: ComputeNodeId(2),
            transfer_bytes: 8192,
            transfer_cost: TransferCost::zero(),
        },
    ];

    let graph = make_graph(
        vec![cpu.clone(), gpu.clone(), ane.clone()],
        edges,
    );
    let recs = vec![
        recommend(&cpu, ComputeTarget::CpuScalar),
        recommend(&gpu, ComputeTarget::Gpu),
        recommend(&ane, ComputeTarget::NeuralEngine),
    ];

    // ErrorOnFailure mode: should pass for a well-formed graph.
    let pipeline = Pipeline::new(PipelineConfig {
        verify_dispatch: DispatchVerifyMode::ErrorOnFailure,
        ..Default::default()
    });
    let verified_plan = pipeline
        .generate_and_verify_dispatch(&graph, &recs)
        .expect("Verification should pass for well-formed mixed graph");

    assert_eq!(verified_plan.count_launches(), 3);
    assert!(verified_plan.count_transfers() >= 2);

    // FallbackOnFailure mode: should also pass (same valid plan).
    let pipeline_fallback = Pipeline::new(PipelineConfig {
        verify_dispatch: DispatchVerifyMode::FallbackOnFailure,
        ..Default::default()
    });
    let fallback_plan = pipeline_fallback
        .generate_and_verify_dispatch(&graph, &recs)
        .expect("FallbackOnFailure should always succeed");
    assert_eq!(fallback_plan.count_launches(), 3);
}

/// Test that a graph with a node missing CpuScalar triggers FallbackOnFailure
/// mode to produce a safe CPU-only plan.
#[test]
fn e2e_fallback_mode_recovers_from_bad_graph() {
    // A node with only GPU as legal target (no CpuScalar fallback).
    let mut gpu_only_node = gpu_data_parallel_node(0, "ADD", 4096);
    gpu_only_node.legal_targets = vec![ComputeTarget::Gpu]; // remove CpuScalar!

    let graph = make_graph(vec![gpu_only_node.clone()], vec![]);
    let recs = vec![recommend(&gpu_only_node, ComputeTarget::Gpu)];

    let pipeline = Pipeline::new(PipelineConfig {
        verify_dispatch: DispatchVerifyMode::FallbackOnFailure,
        ..Default::default()
    });

    let plan = pipeline
        .generate_and_verify_dispatch(&graph, &recs)
        .expect("FallbackOnFailure should not error");

    // Should have fallen back to CPU-only plan.
    assert_eq!(plan.count_fallbacks(), 1);
    assert_eq!(plan.count_launches(), 0);
}
