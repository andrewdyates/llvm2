// llvm2-lower/target_analysis.rs - Proof-guided target analysis pass
//
// Author: Andrew Yates <ayates@dropbox.com>
// Copyright 2026 Dropbox, Inc. | License: Apache-2.0
//
// Reference: designs/2026-04-13-unified-solver-architecture.md (Phase 1)
//
// Implements the proof-guided target analysis pass from the unified solver
// architecture. For each computation subgraph, this pass examines tMIR proof
// annotations to determine which compute targets (CPU scalar, SIMD, GPU,
// Neural Engine) are legal. The key insight: tMIR proofs PRUNE the search
// space. Most computations will be CPU-only because they lack the proofs
// needed for GPU/ANE. This makes the cross-target search tractable.

//! Proof-guided target analysis pass for LLVM2.
//!
//! This module implements Phase 1 of the unified solver pipeline: for each
//! computation subgraph, determine which compute targets are legal based on
//! tMIR proof annotations, and estimate per-target cost.
//!
//! # Pruning Algorithm (from the design doc)
//!
//! ```text
//! 1. Is the computation Pure?
//!    NO  -> CPU only (side effects must stay on CPU)
//!    YES -> continue
//!
//! 2. Does it operate on arrays/matrices?
//!    NO  -> Scalar or SIMD only (GPU launch overhead dominates for scalars)
//!    YES -> continue
//!
//! 3. Are array accesses InBounds + ValidBorrow?
//!    NO  -> CPU/SIMD only (can't prove GPU memory safety)
//!    YES -> GPU/ANE candidates are legal
//!
//! 4. Is it Associative + Commutative?
//!    NO  -> Sequential GPU ok, but no parallel reduction
//!    YES -> Full GPU parallelism legal
//!
//! 5. Cost model: is data large enough to amortize GPU launch?
//!    NO  -> SIMD on CPU
//!    YES -> GPU or ANE
//! ```
//!
//! # Integration with tMIR proofs
//!
//! The existing [`Proof`] and [`ProofContext`] types from the adapter layer
//! carry per-value proofs (NoOverflow, InBounds, ValidBorrow, etc.). This
//! module adds a [`TargetProofContext`] that wraps `ProofContext` with
//! subgraph-level proof annotations (Pure, Associative, Commutative,
//! Deterministic) that tMIR attaches to computation subgraphs rather than
//! individual values.

use std::collections::{HashMap, HashSet};
use std::fmt;

use serde::{Deserialize, Serialize};

use crate::adapter::{Proof, ProofContext};
use crate::instructions::Value;
use crate::types::Type;

// ---------------------------------------------------------------------------
// Compute targets
// ---------------------------------------------------------------------------

/// Hardware compute targets available on Apple Silicon.
///
/// These correspond to the target categories in the unified solver
/// architecture design doc. Each target has different capabilities,
/// costs, and proof requirements.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ComputeTarget {
    /// CPU scalar execution (AArch64 integer/FP instructions).
    /// Always legal -- the baseline target.
    CpuScalar,

    /// CPU SIMD execution (AArch64 NEON 128-bit vector instructions).
    /// Legal for data-parallel computations on scalars and small arrays.
    CpuSimd,

    /// GPU execution via Metal compute kernels.
    /// Requires: Pure + InBounds + ValidBorrow proofs, sufficient data size.
    Gpu,

    /// Apple Neural Engine execution via CoreML.
    /// Requires: same as GPU, plus matrix/tensor-shaped data.
    NeuralEngine,
}

impl ComputeTarget {
    /// All available compute targets in priority order.
    pub const ALL: [ComputeTarget; 4] = [
        ComputeTarget::CpuScalar,
        ComputeTarget::CpuSimd,
        ComputeTarget::Gpu,
        ComputeTarget::NeuralEngine,
    ];
}

impl fmt::Display for ComputeTarget {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ComputeTarget::CpuScalar => write!(f, "CPU (scalar)"),
            ComputeTarget::CpuSimd => write!(f, "CPU (SIMD/NEON)"),
            ComputeTarget::Gpu => write!(f, "GPU (Metal)"),
            ComputeTarget::NeuralEngine => write!(f, "ANE (Neural Engine)"),
        }
    }
}

// ---------------------------------------------------------------------------
// Subgraph identification
// ---------------------------------------------------------------------------

/// Unique identifier for a computation subgraph within a function.
///
/// A subgraph is a connected set of instructions that can potentially be
/// assigned to a single compute target. The partitioning into subgraphs
/// is done by the compute graph analysis (not this module).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct SubgraphId(pub u32);

impl fmt::Display for SubgraphId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "subgraph_{}", self.0)
    }
}

// ---------------------------------------------------------------------------
// Target legality
// ---------------------------------------------------------------------------

/// Legality determination for a single compute target.
#[derive(Debug, Clone, PartialEq)]
pub struct TargetJudgment {
    /// Whether this target is legal for the subgraph.
    pub legal: bool,

    /// Human-readable justification for the legality decision.
    /// Used for transparency/audit trail (Phase 5 of the unified pipeline).
    pub reason: String,
}

/// Result of target analysis for a single computation subgraph.
///
/// Maps each [`ComputeTarget`] to a legality judgment with justification.
/// This is the output of Phase 1 of the unified solver pipeline.
#[derive(Debug, Clone)]
pub struct TargetLegality {
    /// The subgraph this analysis applies to.
    pub subgraph: SubgraphId,

    /// Legality judgment per target.
    pub judgments: HashMap<ComputeTarget, TargetJudgment>,

    /// Whether parallel reduction is legal (requires Associative + Commutative).
    pub parallel_reduction_legal: bool,

    /// Estimated data size in bytes (for cost threshold decisions).
    pub data_size_bytes: u64,
}

impl TargetLegality {
    /// Returns the set of legal targets for this subgraph.
    pub fn legal_targets(&self) -> Vec<ComputeTarget> {
        self.judgments
            .iter()
            .filter(|(_, j)| j.legal)
            .map(|(t, _)| *t)
            .collect()
    }

    /// Check if a specific target is legal.
    pub fn is_legal(&self, target: ComputeTarget) -> bool {
        self.judgments
            .get(&target)
            .is_some_and(|j| j.legal)
    }

    /// Get the justification for a target's legality decision.
    pub fn reason(&self, target: ComputeTarget) -> Option<&str> {
        self.judgments.get(&target).map(|j| j.reason.as_str())
    }
}

// ---------------------------------------------------------------------------
// Subgraph-level proof annotations
// ---------------------------------------------------------------------------

/// Subgraph-level proof properties that tMIR attaches to computation
/// subgraphs (as opposed to individual values).
///
/// These extend the per-value [`Proof`] annotations from the adapter layer
/// with properties that apply to entire subgraphs. In the real tMIR crate,
/// these would be part of the function metadata; here we model them as a
/// separate context that wraps [`ProofContext`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SubgraphProof {
    /// Computation is pure (no side effects, no I/O, no global state mutation).
    /// Enables: safe to move to any compute target.
    Pure,

    /// Operation is associative: (a op b) op c = a op (b op c).
    /// Enables: parallel reduction by regrouping.
    Associative,

    /// Operation is commutative: a op b = b op a.
    /// Enables: reordering for GPU-friendly access patterns.
    Commutative,

    /// Computation is deterministic (same inputs always produce same outputs).
    /// Enables: safe to distribute across any hardware.
    Deterministic,
}

impl fmt::Display for SubgraphProof {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SubgraphProof::Pure => write!(f, "Pure"),
            SubgraphProof::Associative => write!(f, "Associative"),
            SubgraphProof::Commutative => write!(f, "Commutative"),
            SubgraphProof::Deterministic => write!(f, "Deterministic"),
        }
    }
}

// ---------------------------------------------------------------------------
// Subgraph descriptor
// ---------------------------------------------------------------------------

/// Descriptor for a computation subgraph, containing the information
/// needed by the [`ProofAnalyzer`] to determine target legality.
#[derive(Debug, Clone)]
pub struct SubgraphDescriptor {
    /// Unique identifier.
    pub id: SubgraphId,

    /// Values involved in this subgraph (for querying per-value proofs).
    pub values: Vec<Value>,

    /// Types of the values operated on (for detecting array/matrix operations).
    pub value_types: HashMap<Value, Type>,

    /// Subgraph-level proofs (Pure, Associative, Commutative, Deterministic).
    pub subgraph_proofs: HashSet<SubgraphProof>,

    /// Estimated total data size in bytes (element_size * count for arrays,
    /// scalar size for scalars). Used for the cost threshold decision.
    pub data_size_bytes: u64,
}

impl SubgraphDescriptor {
    /// Create a new subgraph descriptor.
    pub fn new(id: SubgraphId) -> Self {
        Self {
            id,
            values: Vec::new(),
            value_types: HashMap::new(),
            subgraph_proofs: HashSet::new(),
            data_size_bytes: 0,
        }
    }

    /// Check if this subgraph has a specific subgraph-level proof.
    pub fn has_proof(&self, proof: SubgraphProof) -> bool {
        self.subgraph_proofs.contains(&proof)
    }

    /// Check if this subgraph operates on arrays or matrices.
    ///
    /// A subgraph operates on arrays if any of its values have Array type.
    pub fn operates_on_arrays(&self) -> bool {
        self.value_types.values().any(|ty| matches!(ty, Type::Array(_, _)))
    }
}

// ---------------------------------------------------------------------------
// Cost configuration
// ---------------------------------------------------------------------------

/// Configuration for the cost-based target analysis thresholds.
///
/// These thresholds control when the analyzer considers GPU/ANE targets
/// worthwhile despite having sufficient proofs. The default values come
/// from the design doc example (4KB GPU threshold from the dot product
/// example: "data size 1000 x f64 = 8KB > GPU threshold 4KB").
#[derive(Debug, Clone, PartialEq)]
pub struct CostConfig {
    /// Minimum data size in bytes for GPU to be considered cost-effective.
    /// Below this threshold, GPU launch overhead dominates.
    /// Default: 4096 bytes (4KB, from the design doc dot product example).
    pub gpu_launch_threshold_bytes: u64,

    /// Minimum data size in bytes for ANE to be considered cost-effective.
    /// ANE has higher launch overhead than GPU but better throughput for
    /// matrix operations.
    /// Default: 8192 bytes (8KB).
    pub ane_launch_threshold_bytes: u64,

    /// Minimum data size in bytes for SIMD to be considered over scalar.
    /// Below this, scalar is typically sufficient.
    /// Default: 16 bytes (one NEON register width).
    pub simd_threshold_bytes: u64,
}

/// Default GPU launch threshold from the design doc: 4KB.
pub const DEFAULT_GPU_LAUNCH_THRESHOLD_BYTES: u64 = 4096;

/// Default ANE launch threshold: 8KB.
pub const DEFAULT_ANE_LAUNCH_THRESHOLD_BYTES: u64 = 8192;

/// Default SIMD threshold: 16 bytes (128-bit NEON register).
pub const DEFAULT_SIMD_THRESHOLD_BYTES: u64 = 16;

impl Default for CostConfig {
    fn default() -> Self {
        Self {
            gpu_launch_threshold_bytes: DEFAULT_GPU_LAUNCH_THRESHOLD_BYTES,
            ane_launch_threshold_bytes: DEFAULT_ANE_LAUNCH_THRESHOLD_BYTES,
            simd_threshold_bytes: DEFAULT_SIMD_THRESHOLD_BYTES,
        }
    }
}

// ---------------------------------------------------------------------------
// Target proof context
// ---------------------------------------------------------------------------

/// Extended proof context that wraps [`ProofContext`] with subgraph-level
/// proof annotations needed for target analysis.
///
/// The adapter layer's `ProofContext` carries per-value proofs (NoOverflow,
/// InBounds, etc.). This struct adds subgraph-level proofs (Pure,
/// Associative, Commutative, Deterministic) that apply to entire
/// computation subgraphs rather than individual values.
#[derive(Debug, Clone, Default)]
pub struct TargetProofContext {
    /// Per-value proofs from the adapter layer.
    pub proof_ctx: ProofContext,

    /// Subgraph-level proofs.
    pub subgraph_proofs: HashMap<SubgraphId, HashSet<SubgraphProof>>,
}

impl TargetProofContext {
    /// Create a new target proof context wrapping an existing proof context.
    pub fn new(proof_ctx: ProofContext) -> Self {
        Self {
            proof_ctx,
            subgraph_proofs: HashMap::new(),
        }
    }

    /// Add a subgraph-level proof.
    pub fn add_subgraph_proof(&mut self, subgraph: SubgraphId, proof: SubgraphProof) {
        self.subgraph_proofs
            .entry(subgraph)
            .or_default()
            .insert(proof);
    }

    /// Check if a subgraph has a specific proof.
    pub fn has_subgraph_proof(&self, subgraph: SubgraphId, proof: SubgraphProof) -> bool {
        self.subgraph_proofs
            .get(&subgraph)
            .is_some_and(|proofs| proofs.contains(&proof))
    }

    /// Get all subgraph-level proofs for a subgraph.
    pub fn subgraph_proofs_for(&self, subgraph: SubgraphId) -> HashSet<SubgraphProof> {
        self.subgraph_proofs
            .get(&subgraph)
            .cloned()
            .unwrap_or_default()
    }
}

// ---------------------------------------------------------------------------
// Proof analyzer
// ---------------------------------------------------------------------------

/// Proof-guided target analyzer.
///
/// Implements the pruning algorithm from the unified solver architecture
/// design doc. For each computation subgraph, examines tMIR proof
/// annotations to determine which compute targets are legal.
///
/// # Usage
///
/// ```rust,ignore
/// let analyzer = ProofAnalyzer::new(CostConfig::default());
/// let legality = analyzer.analyze(&subgraph, &target_proof_ctx);
/// for target in legality.legal_targets() {
///     println!("{}: legal ({})", target, legality.reason(target).unwrap());
/// }
/// ```
#[derive(Clone)]
pub struct ProofAnalyzer {
    /// Cost configuration thresholds.
    config: CostConfig,
}

impl ProofAnalyzer {
    /// Create a new proof analyzer with the given cost configuration.
    pub fn new(config: CostConfig) -> Self {
        Self { config }
    }

    /// Create a new proof analyzer with default cost configuration.
    pub fn with_defaults() -> Self {
        Self::new(CostConfig::default())
    }

    /// Analyze a computation subgraph and determine target legality.
    ///
    /// Implements the 5-step pruning algorithm from the design doc:
    ///
    /// 1. Pure? -> CPU only if not
    /// 2. Operates on arrays? -> Scalar/SIMD only if not
    /// 3. InBounds + ValidBorrow? -> CPU/SIMD only if not, GPU/ANE legal if yes
    /// 4. Associative + Commutative? -> parallel reduction legal if yes
    /// 5. Cost threshold -> SIMD vs GPU/ANE based on data size
    pub fn analyze(
        &self,
        subgraph: &SubgraphDescriptor,
        ctx: &TargetProofContext,
    ) -> TargetLegality {
        let mut judgments = HashMap::new();
        let data_size = subgraph.data_size_bytes;

        // CpuScalar is always legal -- the baseline target.
        judgments.insert(
            ComputeTarget::CpuScalar,
            TargetJudgment {
                legal: true,
                reason: "CPU scalar is always legal (baseline target)".to_string(),
            },
        );

        // Step 1: Is the computation Pure?
        let is_pure = subgraph.has_proof(SubgraphProof::Pure)
            || ctx.has_subgraph_proof(subgraph.id, SubgraphProof::Pure);

        if !is_pure {
            // Not pure -> CPU only. Side effects must stay on CPU.
            judgments.insert(
                ComputeTarget::CpuSimd,
                TargetJudgment {
                    legal: true,
                    reason: "SIMD legal: side effects constrain to CPU, but SIMD is on-CPU"
                        .to_string(),
                },
            );
            judgments.insert(
                ComputeTarget::Gpu,
                TargetJudgment {
                    legal: false,
                    reason: "GPU illegal: computation is not proven Pure (side effects must stay on CPU)"
                        .to_string(),
                },
            );
            judgments.insert(
                ComputeTarget::NeuralEngine,
                TargetJudgment {
                    legal: false,
                    reason: "ANE illegal: computation is not proven Pure (side effects must stay on CPU)"
                        .to_string(),
                },
            );

            return TargetLegality {
                subgraph: subgraph.id,
                judgments,
                parallel_reduction_legal: false,
                data_size_bytes: data_size,
            };
        }

        // Step 2: Does it operate on arrays/matrices?
        let operates_on_arrays = subgraph.operates_on_arrays();

        if !operates_on_arrays {
            // Scalar computation: Scalar or SIMD only.
            // GPU launch overhead dominates for non-array computations.
            let simd_legal = data_size >= self.config.simd_threshold_bytes;
            judgments.insert(
                ComputeTarget::CpuSimd,
                TargetJudgment {
                    legal: simd_legal,
                    reason: if simd_legal {
                        format!(
                            "SIMD legal: Pure computation, data size {} bytes >= {} byte threshold",
                            data_size, self.config.simd_threshold_bytes
                        )
                    } else {
                        format!(
                            "SIMD not cost-effective: data size {} bytes < {} byte threshold",
                            data_size, self.config.simd_threshold_bytes
                        )
                    },
                },
            );
            judgments.insert(
                ComputeTarget::Gpu,
                TargetJudgment {
                    legal: false,
                    reason: "GPU illegal: computation does not operate on arrays (launch overhead dominates)"
                        .to_string(),
                },
            );
            judgments.insert(
                ComputeTarget::NeuralEngine,
                TargetJudgment {
                    legal: false,
                    reason: "ANE illegal: computation does not operate on arrays (launch overhead dominates)"
                        .to_string(),
                },
            );

            return TargetLegality {
                subgraph: subgraph.id,
                judgments,
                parallel_reduction_legal: false,
                data_size_bytes: data_size,
            };
        }

        // Step 3: Are array accesses InBounds + ValidBorrow?
        let has_in_bounds = subgraph
            .values
            .iter()
            .any(|v| ctx.proof_ctx.has_in_bounds(v));

        let has_valid_borrow = subgraph
            .values
            .iter()
            .any(|v| {
                ctx.proof_ctx
                    .value_proofs
                    .get(v)
                    .is_some_and(|proofs| {
                        proofs.iter().any(|p| matches!(p, Proof::ValidBorrow { .. }))
                    })
            });

        let memory_safe = has_in_bounds && has_valid_borrow;

        // Step 4: Associative + Commutative?
        let is_associative = subgraph.has_proof(SubgraphProof::Associative)
            || ctx.has_subgraph_proof(subgraph.id, SubgraphProof::Associative);
        let is_commutative = subgraph.has_proof(SubgraphProof::Commutative)
            || ctx.has_subgraph_proof(subgraph.id, SubgraphProof::Commutative);
        let parallel_reduction_legal = is_associative && is_commutative;

        // SIMD is legal for Pure + array operations (on-CPU, no memory safety issue).
        judgments.insert(
            ComputeTarget::CpuSimd,
            TargetJudgment {
                legal: true,
                reason: "SIMD legal: Pure computation on arrays, on-CPU execution".to_string(),
            },
        );

        if !memory_safe {
            // Can't prove GPU memory safety -> CPU/SIMD only.
            let mut missing = Vec::new();
            if !has_in_bounds {
                missing.push("InBounds");
            }
            if !has_valid_borrow {
                missing.push("ValidBorrow");
            }
            let missing_str = missing.join(" + ");

            judgments.insert(
                ComputeTarget::Gpu,
                TargetJudgment {
                    legal: false,
                    reason: format!(
                        "GPU illegal: missing {} proof(s) (can't prove GPU memory safety)",
                        missing_str
                    ),
                },
            );
            judgments.insert(
                ComputeTarget::NeuralEngine,
                TargetJudgment {
                    legal: false,
                    reason: format!(
                        "ANE illegal: missing {} proof(s) (can't prove ANE memory safety)",
                        missing_str
                    ),
                },
            );

            return TargetLegality {
                subgraph: subgraph.id,
                judgments,
                parallel_reduction_legal,
                data_size_bytes: data_size,
            };
        }

        // Step 5: Cost threshold -- is data large enough to amortize launch overhead?
        let gpu_cost_effective = data_size >= self.config.gpu_launch_threshold_bytes;
        let ane_cost_effective = data_size >= self.config.ane_launch_threshold_bytes;

        judgments.insert(
            ComputeTarget::Gpu,
            TargetJudgment {
                legal: gpu_cost_effective,
                reason: if gpu_cost_effective {
                    format!(
                        "GPU legal: Pure + InBounds + ValidBorrow proven, data size {} bytes >= {} byte threshold{}",
                        data_size,
                        self.config.gpu_launch_threshold_bytes,
                        if parallel_reduction_legal { ", parallel reduction legal (Associative + Commutative)" } else { "" }
                    )
                } else {
                    format!(
                        "GPU not cost-effective: data size {} bytes < {} byte GPU launch threshold (use SIMD instead)",
                        data_size, self.config.gpu_launch_threshold_bytes
                    )
                },
            },
        );

        judgments.insert(
            ComputeTarget::NeuralEngine,
            TargetJudgment {
                legal: ane_cost_effective,
                reason: if ane_cost_effective {
                    format!(
                        "ANE legal: Pure + InBounds + ValidBorrow proven, data size {} bytes >= {} byte threshold{}",
                        data_size,
                        self.config.ane_launch_threshold_bytes,
                        if parallel_reduction_legal { ", parallel reduction legal (Associative + Commutative)" } else { "" }
                    )
                } else {
                    format!(
                        "ANE not cost-effective: data size {} bytes < {} byte ANE launch threshold (use GPU or SIMD instead)",
                        data_size, self.config.ane_launch_threshold_bytes
                    )
                },
            },
        );

        TargetLegality {
            subgraph: subgraph.id,
            judgments,
            parallel_reduction_legal,
            data_size_bytes: data_size,
        }
    }

    /// Analyze multiple subgraphs and return legality results for each.
    pub fn analyze_all(
        &self,
        subgraphs: &[SubgraphDescriptor],
        ctx: &TargetProofContext,
    ) -> Vec<TargetLegality> {
        subgraphs.iter().map(|sg| self.analyze(sg, ctx)).collect()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::adapter::Proof;
    use crate::instructions::Value;
    use crate::types::Type;
    use tmir_types::ValueId;

    /// Helper: build a minimal scalar subgraph (no arrays, no proofs).
    fn scalar_subgraph(id: u32) -> SubgraphDescriptor {
        let mut sg = SubgraphDescriptor::new(SubgraphId(id));
        sg.values = vec![Value(0), Value(1), Value(2)];
        sg.value_types.insert(Value(0), Type::I32);
        sg.value_types.insert(Value(1), Type::I32);
        sg.value_types.insert(Value(2), Type::I32);
        sg.data_size_bytes = 12; // 3 x i32
        sg
    }

    /// Helper: build an array subgraph with specified data size.
    fn array_subgraph(id: u32, data_size: u64) -> SubgraphDescriptor {
        let mut sg = SubgraphDescriptor::new(SubgraphId(id));
        sg.values = vec![Value(0), Value(1), Value(2)];
        sg.value_types
            .insert(Value(0), Type::Array(Box::new(Type::F64), 1000));
        sg.value_types
            .insert(Value(1), Type::Array(Box::new(Type::F64), 1000));
        sg.value_types.insert(Value(2), Type::F64);
        sg.data_size_bytes = data_size;
        sg
    }

    /// Helper: create a TargetProofContext with InBounds + ValidBorrow proofs
    /// on specified values.
    fn ctx_with_memory_proofs(values: &[Value]) -> TargetProofContext {
        let mut proof_ctx = ProofContext::default();
        for &v in values {
            proof_ctx.value_proofs.insert(
                v,
                vec![
                    Proof::InBounds {
                        base: ValueId(v.0),
                        index: ValueId(v.0 + 100),
                    },
                    Proof::ValidBorrow {
                        borrow: ValueId(v.0),
                    },
                ],
            );
        }
        TargetProofContext::new(proof_ctx)
    }

    // -----------------------------------------------------------------------
    // Test: No proofs at all -> CPU only
    // -----------------------------------------------------------------------

    #[test]
    fn test_no_proofs_cpu_only() {
        let analyzer = ProofAnalyzer::with_defaults();
        let sg = scalar_subgraph(0);
        let ctx = TargetProofContext::default();

        let legality = analyzer.analyze(&sg, &ctx);

        // CPU scalar always legal.
        assert!(legality.is_legal(ComputeTarget::CpuScalar));
        // SIMD legal (side effects constrain to CPU but SIMD is on-CPU).
        assert!(legality.is_legal(ComputeTarget::CpuSimd));
        // GPU and ANE illegal: not Pure.
        assert!(!legality.is_legal(ComputeTarget::Gpu));
        assert!(!legality.is_legal(ComputeTarget::NeuralEngine));
        assert!(!legality.parallel_reduction_legal);
    }

    // -----------------------------------------------------------------------
    // Test: Pure scalar -> CPU + SIMD, no GPU/ANE
    // -----------------------------------------------------------------------

    #[test]
    fn test_pure_scalar_no_gpu() {
        let analyzer = ProofAnalyzer::with_defaults();
        let mut sg = scalar_subgraph(1);
        sg.subgraph_proofs.insert(SubgraphProof::Pure);
        sg.data_size_bytes = 32; // > SIMD threshold of 16

        let ctx = TargetProofContext::default();
        let legality = analyzer.analyze(&sg, &ctx);

        assert!(legality.is_legal(ComputeTarget::CpuScalar));
        assert!(legality.is_legal(ComputeTarget::CpuSimd));
        // GPU/ANE illegal: no arrays.
        assert!(!legality.is_legal(ComputeTarget::Gpu));
        assert!(!legality.is_legal(ComputeTarget::NeuralEngine));
    }

    // -----------------------------------------------------------------------
    // Test: Pure scalar below SIMD threshold -> SIMD not cost-effective
    // -----------------------------------------------------------------------

    #[test]
    fn test_pure_scalar_below_simd_threshold() {
        let analyzer = ProofAnalyzer::with_defaults();
        let mut sg = scalar_subgraph(2);
        sg.subgraph_proofs.insert(SubgraphProof::Pure);
        sg.data_size_bytes = 4; // < SIMD threshold of 16

        let ctx = TargetProofContext::default();
        let legality = analyzer.analyze(&sg, &ctx);

        assert!(legality.is_legal(ComputeTarget::CpuScalar));
        assert!(!legality.is_legal(ComputeTarget::CpuSimd));
        assert!(!legality.is_legal(ComputeTarget::Gpu));
        assert!(!legality.is_legal(ComputeTarget::NeuralEngine));
    }

    // -----------------------------------------------------------------------
    // Test: Pure + arrays but no InBounds/ValidBorrow -> CPU/SIMD only
    // -----------------------------------------------------------------------

    #[test]
    fn test_pure_arrays_no_memory_proofs() {
        let analyzer = ProofAnalyzer::with_defaults();
        let mut sg = array_subgraph(3, 8000);
        sg.subgraph_proofs.insert(SubgraphProof::Pure);

        let ctx = TargetProofContext::default();
        let legality = analyzer.analyze(&sg, &ctx);

        assert!(legality.is_legal(ComputeTarget::CpuScalar));
        assert!(legality.is_legal(ComputeTarget::CpuSimd));
        // GPU/ANE illegal: no InBounds + ValidBorrow.
        assert!(!legality.is_legal(ComputeTarget::Gpu));
        assert!(!legality.is_legal(ComputeTarget::NeuralEngine));

        // Check justification mentions missing proofs.
        let gpu_reason = legality.reason(ComputeTarget::Gpu).unwrap();
        assert!(
            gpu_reason.contains("InBounds") || gpu_reason.contains("ValidBorrow"),
            "GPU reason should mention missing proofs: {}",
            gpu_reason
        );
    }

    // -----------------------------------------------------------------------
    // Test: Pure + arrays + InBounds only (no ValidBorrow) -> CPU/SIMD only
    // -----------------------------------------------------------------------

    #[test]
    fn test_pure_arrays_inbounds_only() {
        let analyzer = ProofAnalyzer::with_defaults();
        let mut sg = array_subgraph(4, 8000);
        sg.subgraph_proofs.insert(SubgraphProof::Pure);

        let mut proof_ctx = ProofContext::default();
        proof_ctx.value_proofs.insert(
            Value(0),
            vec![Proof::InBounds {
                base: ValueId(0),
                index: ValueId(100),
            }],
        );
        let ctx = TargetProofContext::new(proof_ctx);

        let legality = analyzer.analyze(&sg, &ctx);

        assert!(legality.is_legal(ComputeTarget::CpuScalar));
        assert!(legality.is_legal(ComputeTarget::CpuSimd));
        assert!(!legality.is_legal(ComputeTarget::Gpu));
        assert!(!legality.is_legal(ComputeTarget::NeuralEngine));

        // Reason should mention ValidBorrow is missing.
        let gpu_reason = legality.reason(ComputeTarget::Gpu).unwrap();
        assert!(
            gpu_reason.contains("ValidBorrow"),
            "GPU reason should mention missing ValidBorrow: {}",
            gpu_reason
        );
    }

    // -----------------------------------------------------------------------
    // Test: Full proofs but data below GPU threshold -> SIMD yes, GPU no
    // -----------------------------------------------------------------------

    #[test]
    fn test_full_proofs_below_gpu_threshold() {
        let analyzer = ProofAnalyzer::with_defaults();
        // Data size 2000 bytes < 4096 GPU threshold.
        let mut sg = array_subgraph(5, 2000);
        sg.subgraph_proofs.insert(SubgraphProof::Pure);

        let ctx = ctx_with_memory_proofs(&[Value(0), Value(1)]);
        let legality = analyzer.analyze(&sg, &ctx);

        assert!(legality.is_legal(ComputeTarget::CpuScalar));
        assert!(legality.is_legal(ComputeTarget::CpuSimd));
        // GPU: proofs sufficient but data too small.
        assert!(!legality.is_legal(ComputeTarget::Gpu));
        assert!(!legality.is_legal(ComputeTarget::NeuralEngine));

        let gpu_reason = legality.reason(ComputeTarget::Gpu).unwrap();
        assert!(
            gpu_reason.contains("not cost-effective"),
            "GPU reason should mention cost: {}",
            gpu_reason
        );
    }

    // -----------------------------------------------------------------------
    // Test: Full proofs, data above GPU threshold -> all targets legal
    // -----------------------------------------------------------------------

    #[test]
    fn test_full_proofs_above_gpu_threshold() {
        let analyzer = ProofAnalyzer::with_defaults();
        // Data size 8000 bytes > 4096 GPU threshold.
        let mut sg = array_subgraph(6, 8000);
        sg.subgraph_proofs.insert(SubgraphProof::Pure);

        let ctx = ctx_with_memory_proofs(&[Value(0), Value(1)]);
        let legality = analyzer.analyze(&sg, &ctx);

        assert!(legality.is_legal(ComputeTarget::CpuScalar));
        assert!(legality.is_legal(ComputeTarget::CpuSimd));
        assert!(legality.is_legal(ComputeTarget::Gpu));
        // ANE threshold is 8192, data is 8000 -> not cost-effective.
        assert!(!legality.is_legal(ComputeTarget::NeuralEngine));
    }

    // -----------------------------------------------------------------------
    // Test: Full proofs, data above ANE threshold -> everything legal
    // -----------------------------------------------------------------------

    #[test]
    fn test_full_proofs_above_ane_threshold() {
        let analyzer = ProofAnalyzer::with_defaults();
        // Data size 16000 bytes > 8192 ANE threshold.
        let mut sg = array_subgraph(7, 16000);
        sg.subgraph_proofs.insert(SubgraphProof::Pure);

        let ctx = ctx_with_memory_proofs(&[Value(0), Value(1)]);
        let legality = analyzer.analyze(&sg, &ctx);

        assert!(legality.is_legal(ComputeTarget::CpuScalar));
        assert!(legality.is_legal(ComputeTarget::CpuSimd));
        assert!(legality.is_legal(ComputeTarget::Gpu));
        assert!(legality.is_legal(ComputeTarget::NeuralEngine));
    }

    // -----------------------------------------------------------------------
    // Test: Parallel reduction (Associative + Commutative)
    // -----------------------------------------------------------------------

    #[test]
    fn test_parallel_reduction_legal() {
        let analyzer = ProofAnalyzer::with_defaults();
        let mut sg = array_subgraph(8, 16000);
        sg.subgraph_proofs.insert(SubgraphProof::Pure);
        sg.subgraph_proofs.insert(SubgraphProof::Associative);
        sg.subgraph_proofs.insert(SubgraphProof::Commutative);

        let ctx = ctx_with_memory_proofs(&[Value(0), Value(1)]);
        let legality = analyzer.analyze(&sg, &ctx);

        assert!(legality.parallel_reduction_legal);
        assert!(legality.is_legal(ComputeTarget::Gpu));

        // GPU reason should mention parallel reduction.
        let gpu_reason = legality.reason(ComputeTarget::Gpu).unwrap();
        assert!(
            gpu_reason.contains("parallel reduction"),
            "GPU reason should mention parallel reduction: {}",
            gpu_reason
        );
    }

    // -----------------------------------------------------------------------
    // Test: Associative only (no Commutative) -> no parallel reduction
    // -----------------------------------------------------------------------

    #[test]
    fn test_associative_only_no_parallel_reduction() {
        let analyzer = ProofAnalyzer::with_defaults();
        let mut sg = array_subgraph(9, 16000);
        sg.subgraph_proofs.insert(SubgraphProof::Pure);
        sg.subgraph_proofs.insert(SubgraphProof::Associative);
        // Intentionally NOT adding Commutative.

        let ctx = ctx_with_memory_proofs(&[Value(0), Value(1)]);
        let legality = analyzer.analyze(&sg, &ctx);

        assert!(!legality.parallel_reduction_legal);
        // GPU is still legal (sequential GPU ok), just no parallel reduction.
        assert!(legality.is_legal(ComputeTarget::Gpu));
    }

    // -----------------------------------------------------------------------
    // Test: Dot product example from the design doc
    // -----------------------------------------------------------------------

    #[test]
    fn test_dot_product_example() {
        // From designs/2026-04-13-unified-solver-architecture.md:
        // fn dot_product(a: &[f64; 1000], b: &[f64; 1000]) -> f64
        // Proofs: Pure, InBounds, ValidBorrow(a, b), Associative(+), Commutative(+)
        // Data: 1000 x f64 = 8000 bytes > 4KB GPU threshold
        // Expected: all targets legal, parallel reduction legal

        let analyzer = ProofAnalyzer::with_defaults();

        let mut sg = SubgraphDescriptor::new(SubgraphId(10));
        sg.values = vec![Value(0), Value(1), Value(2)];
        sg.value_types
            .insert(Value(0), Type::Array(Box::new(Type::F64), 1000));
        sg.value_types
            .insert(Value(1), Type::Array(Box::new(Type::F64), 1000));
        sg.value_types.insert(Value(2), Type::F64);
        sg.data_size_bytes = 1000 * 8; // 1000 x f64 = 8000 bytes
        sg.subgraph_proofs.insert(SubgraphProof::Pure);
        sg.subgraph_proofs.insert(SubgraphProof::Associative);
        sg.subgraph_proofs.insert(SubgraphProof::Commutative);

        let ctx = ctx_with_memory_proofs(&[Value(0), Value(1)]);
        let legality = analyzer.analyze(&sg, &ctx);

        // All targets legal.
        assert!(legality.is_legal(ComputeTarget::CpuScalar));
        assert!(legality.is_legal(ComputeTarget::CpuSimd));
        assert!(legality.is_legal(ComputeTarget::Gpu));
        // ANE: 8000 < 8192 threshold -> not cost-effective with default config.
        assert!(!legality.is_legal(ComputeTarget::NeuralEngine));

        // Parallel reduction legal.
        assert!(legality.parallel_reduction_legal);

        // Verify the legal_targets() method.
        let legal = legality.legal_targets();
        assert!(legal.contains(&ComputeTarget::CpuScalar));
        assert!(legal.contains(&ComputeTarget::CpuSimd));
        assert!(legal.contains(&ComputeTarget::Gpu));
    }

    // -----------------------------------------------------------------------
    // Test: Custom cost config
    // -----------------------------------------------------------------------

    #[test]
    fn test_custom_cost_config() {
        let config = CostConfig {
            gpu_launch_threshold_bytes: 1024,
            ane_launch_threshold_bytes: 2048,
            simd_threshold_bytes: 8,
        };
        let analyzer = ProofAnalyzer::new(config);

        // Small array (1500 bytes) -- legal for GPU with lower threshold.
        let mut sg = array_subgraph(11, 1500);
        sg.subgraph_proofs.insert(SubgraphProof::Pure);

        let ctx = ctx_with_memory_proofs(&[Value(0), Value(1)]);
        let legality = analyzer.analyze(&sg, &ctx);

        assert!(legality.is_legal(ComputeTarget::Gpu));
        // ANE: 1500 < 2048 threshold -> still not cost-effective.
        assert!(!legality.is_legal(ComputeTarget::NeuralEngine));
    }

    // -----------------------------------------------------------------------
    // Test: Subgraph proofs from TargetProofContext (not SubgraphDescriptor)
    // -----------------------------------------------------------------------

    #[test]
    fn test_proofs_from_target_proof_context() {
        let analyzer = ProofAnalyzer::with_defaults();
        let sg = array_subgraph(12, 16000);
        // Note: NOT setting subgraph_proofs on the descriptor itself.

        let mut ctx = ctx_with_memory_proofs(&[Value(0), Value(1)]);
        // Add proofs via the TargetProofContext instead.
        ctx.add_subgraph_proof(SubgraphId(12), SubgraphProof::Pure);
        ctx.add_subgraph_proof(SubgraphId(12), SubgraphProof::Associative);
        ctx.add_subgraph_proof(SubgraphId(12), SubgraphProof::Commutative);

        let legality = analyzer.analyze(&sg, &ctx);

        assert!(legality.is_legal(ComputeTarget::Gpu));
        assert!(legality.is_legal(ComputeTarget::NeuralEngine));
        assert!(legality.parallel_reduction_legal);
    }

    // -----------------------------------------------------------------------
    // Test: analyze_all with multiple subgraphs
    // -----------------------------------------------------------------------

    #[test]
    fn test_analyze_all() {
        let analyzer = ProofAnalyzer::with_defaults();

        // Subgraph 0: no proofs (CPU only).
        let sg0 = scalar_subgraph(0);

        // Subgraph 1: Pure + arrays + memory proofs (GPU legal).
        let mut sg1 = array_subgraph(1, 8000);
        sg1.subgraph_proofs.insert(SubgraphProof::Pure);

        let ctx = ctx_with_memory_proofs(&[Value(0), Value(1)]);

        let results = analyzer.analyze_all(&[sg0, sg1], &ctx);
        assert_eq!(results.len(), 2);

        // Subgraph 0: CPU only.
        assert!(!results[0].is_legal(ComputeTarget::Gpu));

        // Subgraph 1: GPU legal.
        assert!(results[1].is_legal(ComputeTarget::Gpu));
    }

    // -----------------------------------------------------------------------
    // Test: TargetProofContext queries
    // -----------------------------------------------------------------------

    #[test]
    fn test_target_proof_context_queries() {
        let mut ctx = TargetProofContext::default();

        let sg_id = SubgraphId(42);
        assert!(!ctx.has_subgraph_proof(sg_id, SubgraphProof::Pure));

        ctx.add_subgraph_proof(sg_id, SubgraphProof::Pure);
        ctx.add_subgraph_proof(sg_id, SubgraphProof::Deterministic);

        assert!(ctx.has_subgraph_proof(sg_id, SubgraphProof::Pure));
        assert!(ctx.has_subgraph_proof(sg_id, SubgraphProof::Deterministic));
        assert!(!ctx.has_subgraph_proof(sg_id, SubgraphProof::Associative));

        let proofs = ctx.subgraph_proofs_for(sg_id);
        assert_eq!(proofs.len(), 2);
        assert!(proofs.contains(&SubgraphProof::Pure));
        assert!(proofs.contains(&SubgraphProof::Deterministic));

        // Non-existent subgraph returns empty set.
        let empty = ctx.subgraph_proofs_for(SubgraphId(999));
        assert!(empty.is_empty());
    }

    // -----------------------------------------------------------------------
    // Test: Display implementations
    // -----------------------------------------------------------------------

    #[test]
    fn test_display_compute_target() {
        assert_eq!(format!("{}", ComputeTarget::CpuScalar), "CPU (scalar)");
        assert_eq!(format!("{}", ComputeTarget::CpuSimd), "CPU (SIMD/NEON)");
        assert_eq!(format!("{}", ComputeTarget::Gpu), "GPU (Metal)");
        assert_eq!(
            format!("{}", ComputeTarget::NeuralEngine),
            "ANE (Neural Engine)"
        );
    }

    #[test]
    fn test_display_subgraph_id() {
        assert_eq!(format!("{}", SubgraphId(42)), "subgraph_42");
    }

    #[test]
    fn test_display_subgraph_proof() {
        assert_eq!(format!("{}", SubgraphProof::Pure), "Pure");
        assert_eq!(format!("{}", SubgraphProof::Associative), "Associative");
        assert_eq!(format!("{}", SubgraphProof::Commutative), "Commutative");
        assert_eq!(
            format!("{}", SubgraphProof::Deterministic),
            "Deterministic"
        );
    }

    // -----------------------------------------------------------------------
    // Test: SubgraphDescriptor operates_on_arrays
    // -----------------------------------------------------------------------

    #[test]
    fn test_subgraph_operates_on_arrays() {
        let scalar = scalar_subgraph(0);
        assert!(!scalar.operates_on_arrays());

        let array = array_subgraph(1, 8000);
        assert!(array.operates_on_arrays());

        // Mixed: one scalar, one array.
        let mut mixed = SubgraphDescriptor::new(SubgraphId(2));
        mixed.value_types.insert(Value(0), Type::I32);
        mixed
            .value_types
            .insert(Value(1), Type::Array(Box::new(Type::I32), 100));
        assert!(mixed.operates_on_arrays());
    }

    // -----------------------------------------------------------------------
    // Test: CostConfig defaults match design doc
    // -----------------------------------------------------------------------

    #[test]
    fn test_cost_config_defaults() {
        let config = CostConfig::default();
        assert_eq!(config.gpu_launch_threshold_bytes, 4096);
        assert_eq!(config.ane_launch_threshold_bytes, 8192);
        assert_eq!(config.simd_threshold_bytes, 16);
    }

    // -----------------------------------------------------------------------
    // Test: Edge case -- empty subgraph
    // -----------------------------------------------------------------------

    #[test]
    fn test_empty_subgraph() {
        let analyzer = ProofAnalyzer::with_defaults();
        let sg = SubgraphDescriptor::new(SubgraphId(99));
        let ctx = TargetProofContext::default();

        let legality = analyzer.analyze(&sg, &ctx);

        // Empty subgraph with no proofs -> CPU only.
        assert!(legality.is_legal(ComputeTarget::CpuScalar));
        assert!(!legality.is_legal(ComputeTarget::Gpu));
        assert!(!legality.is_legal(ComputeTarget::NeuralEngine));
    }

    // -----------------------------------------------------------------------
    // Test: ValidBorrow only (no InBounds) -> CPU/SIMD only
    // -----------------------------------------------------------------------

    #[test]
    fn test_valid_borrow_only_no_gpu() {
        let analyzer = ProofAnalyzer::with_defaults();
        let mut sg = array_subgraph(13, 8000);
        sg.subgraph_proofs.insert(SubgraphProof::Pure);

        let mut proof_ctx = ProofContext::default();
        proof_ctx.value_proofs.insert(
            Value(0),
            vec![Proof::ValidBorrow {
                borrow: ValueId(0),
            }],
        );
        let ctx = TargetProofContext::new(proof_ctx);

        let legality = analyzer.analyze(&sg, &ctx);

        assert!(legality.is_legal(ComputeTarget::CpuScalar));
        assert!(legality.is_legal(ComputeTarget::CpuSimd));
        assert!(!legality.is_legal(ComputeTarget::Gpu));
        assert!(!legality.is_legal(ComputeTarget::NeuralEngine));

        // Reason should mention InBounds is missing.
        let gpu_reason = legality.reason(ComputeTarget::Gpu).unwrap();
        assert!(
            gpu_reason.contains("InBounds"),
            "GPU reason should mention missing InBounds: {}",
            gpu_reason
        );
    }

    // -----------------------------------------------------------------------
    // Test: ComputeTarget::ALL contains all 4 targets
    // -----------------------------------------------------------------------

    #[test]
    fn test_compute_target_all() {
        assert_eq!(ComputeTarget::ALL.len(), 4);
        assert!(ComputeTarget::ALL.contains(&ComputeTarget::CpuScalar));
        assert!(ComputeTarget::ALL.contains(&ComputeTarget::CpuSimd));
        assert!(ComputeTarget::ALL.contains(&ComputeTarget::Gpu));
        assert!(ComputeTarget::ALL.contains(&ComputeTarget::NeuralEngine));
    }
}
