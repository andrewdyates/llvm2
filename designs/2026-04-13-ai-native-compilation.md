# AI-Native Compilation

**Author:** Andrew Yates
**Date:** 2026-04-13
**Status:** Design
**Epic:** AI-native compilation

---

## Implementation Status (as of 2026-04-15)

**Overall: The self-improving compilation vision has foundational infrastructure. AI-native rule discovery and CEGIS verification loops exist. The programmatic API and interactive compilation explorer are not yet built.**

| Component | Status | Details |
|-----------|--------|---------|
| **AI-native rule discovery** (`rule_discovery.rs`) | IMPLEMENTED | 1.4K LOC. Agent-proposed optimization rules verified via CEGIS loop. |
| **Provenance tracking** (`llvm2-ir/provenance.rs`) | IMPLEMENTED | 1.1K LOC. tMIR-to-binary offset mapping for instruction traceability. |
| **Compilation trace** (`llvm2-ir/trace.rs`) | IMPLEMENTED | 700 LOC. Structured event log for glass-box transparency. |
| **CEGIS verification** (`cegis.rs`, `synthesis.rs`) | IMPLEMENTED | Proven-correct rule addition to database. |
| **Programmatic API** | NOT IMPLEMENTED | Compiler is a library (Rust crates), but no stable public API designed for external AI agent consumption. |
| **Interactive compilation explorer** | NOT IMPLEMENTED | No tool to query "why was this instruction chosen?" |
| **Decision point traits** | NOT IMPLEMENTED | Heuristics are not yet exposed as overridable traits for AI agents. |
| **Machine-readable pipeline output** | PARTIAL | Provenance and trace exist, but not all pipeline stages emit structured data. |

See #108 (Epic: AI-native compilation).

---

## Introduction

LLVM was designed in 2003 for human compiler engineers. The interface is a CLI (`clang`), debugging is printf + GDB, and extending it means writing C++ passes and rebuilding.

In 2026, AI writes most code and AI agents need to interact with compilers programmatically. LLVM2 is designed from the ground up for this world:

- **Machine-readable output** at every pipeline stage
- **Programmatic API** — the compiler is a library, not just a CLI
- **Self-improving** — AI proposes optimization rules, solver verifies, proven rules are added automatically
- **Decision points as APIs** — every heuristic is a trait an AI agent can implement

---

## Prior Art

### MLGO (Google, 2022)

RL framework embedded in LLVM. Replaces hand-written heuristics with neural networks for inlining and register allocation eviction. Model trained via policy gradient, compiled to native code via XLA AOT — zero TensorFlow dependency at runtime.

**Results:** 3-7% binary size reduction (inlining); 0.3-1.5% QPS improvement (regalloc).

**Key insight:** Narrow scope (2 decisions out of hundreds), but proves ML can outperform hand-tuned heuristics.

Ref: arxiv.org/abs/2101.04808

### CompilerGym (Meta, 2021)

OpenAI Gym interface wrapping LLVM. Actions = optimization passes. Observations = program features (Autophase vector, IR graphs). Reward = instruction count reduction. Standard `reset()/step()` API.

**Results:** PPO+guided search: 1.07x geomean code size reduction. 27x more compute-efficient than prior phase-ordering work.

**Key insight:** Compiler-as-RL-environment. The action/observation/reward pattern is the right abstraction for AI-compiler interaction.

Ref: arxiv.org/abs/2109.08267

### TVM/Ansor (Apache, 2018-2020)

Two-level compiler for tensor computations. Graph-level optimization + operator-level autotuning. Learned cost model (XGBoost/neural net) predicts execution time, replacing hardware measurement during search. Ansor removes templates — hierarchical sketch generation + evolutionary search.

**Results:** Ansor achieves 3.8x over AutoTVM on Intel CPU. Competitive with cuDNN/MKL.

**Key insight:** Separation of search space from cost model enables rapid exploration. Learned cost models are 100x faster than hardware measurement.

Ref: arxiv.org/abs/2006.06762

### Halide (MIT/Adobe/Google, 2012-)

DSL separating algorithm from schedule. Autoscheduler uses beam search with learned cost model.

**Key insight:** Separating "what to compute" from "how to compute it" enables automated optimization search over a well-defined space.

### Ithemal (MIT, 2019)

Hierarchical LSTM predicts throughput for x86-64 basic blocks. Less than half the error of llvm-mca and Intel IACA. Same prediction speed. Trivially portable across microarchitectures (retrain, don't rewrite).

**Key insight:** Learned cost models beat hand-written tables and are easier to port.

### LLM Compiler (Meta, 2024)

Code Llama fine-tuned on 546B tokens of LLVM-IR and assembly. 77% of autotuning search optimality for flag tuning. 45% round-trip LLVM-IR reconstruction.

**Key insight:** LLMs can learn compiler semantics from IR/assembly corpora.

---

## LLVM2 AI-Native Architecture

### 1. Compilation Library API

The compiler is a Rust library first, CLI second:

```rust
pub struct Compiler {
    config: CompilerConfig,
    passes: PassPipeline,
    cost_model: Box<dyn CostModel>,
}

impl Compiler {
    /// Compile tMIR to machine code with full structured output
    pub fn compile(&self, module: &TmirModule) -> CompilationResult;
    
    /// Compile with step-by-step observation
    pub fn compile_observed(&self, module: &TmirModule) -> ObservedCompilation;
    
    /// Get the IR state at any pipeline stage
    pub fn snapshot_at(&self, stage: PipelineStage) -> IrSnapshot;
}

pub struct CompilationResult {
    pub binary: Vec<u8>,              // The compiled object
    pub metrics: CompilationMetrics,   // Code size, estimated cycles, etc.
    pub trace: Option<CompilationTrace>, // If transparency enabled
    pub proofs: Option<ProofCertificates>, // If verification enabled
}

pub struct ObservedCompilation {
    /// IR state before each pass
    pub snapshots: Vec<(PassId, IrSnapshot)>,
    /// Events from each pass
    pub events: Vec<CompilationEvent>,
    /// Final result
    pub result: CompilationResult,
}
```

### 2. Structured IR Dumps

Machine-readable output at every stage:

```rust
pub trait Serializable {
    fn to_json(&self) -> serde_json::Value;
    fn to_cbor(&self) -> Vec<u8>;
    fn to_binary(&self) -> Vec<u8>; // Compact custom format
}

// Every IR type implements Serializable:
impl Serializable for MachFunction { ... }
impl Serializable for MachInst { ... }
impl Serializable for MachOperand { ... }
```

**Formats:**
- JSON for AI agent consumption and debugging
- CBOR for compact storage and proof certificates
- Custom binary for round-trip (serialize → deserialize → continue compilation)

### 3. Pass Management API

Passes are composable, reorderable, and injectable:

```rust
pub struct PassPipeline {
    passes: Vec<Box<dyn Pass>>,
}

impl PassPipeline {
    /// Standard optimization levels
    pub fn o0() -> Self;
    pub fn o1() -> Self;
    pub fn o2() -> Self;
    pub fn o3() -> Self;
    
    /// Programmatic composition
    pub fn insert_before(&mut self, target: PassId, pass: Box<dyn Pass>);
    pub fn insert_after(&mut self, target: PassId, pass: Box<dyn Pass>);
    pub fn remove(&mut self, pass: PassId);
    pub fn reorder(&mut self, order: &[PassId]);
    
    /// Serialize/deserialize pipeline configuration
    pub fn to_config(&self) -> PipelineConfig;
    pub fn from_config(config: &PipelineConfig) -> Self;
}

pub trait Pass {
    fn id(&self) -> PassId;
    fn name(&self) -> &str;
    fn run(&self, func: &mut MachFunction, ctx: &mut PassContext) -> PassResult;
    fn preserves(&self) -> PreservedAnalyses;
}
```

### 4. Decision Point Hooks

Every heuristic is a trait that AI agents can implement:

```rust
/// Register allocation decisions
pub trait RegAllocPolicy {
    fn choose_register(&self, vreg: VRegId, candidates: &[PReg], ctx: &AllocContext) -> PReg;
    fn should_spill(&self, vreg: VRegId, ctx: &AllocContext) -> bool;
    fn eviction_priority(&self, vreg: VRegId, ctx: &AllocContext) -> f64;
}

/// Inlining decisions
pub trait InlinePolicy {
    fn should_inline(&self, call: &CallSite, callee: &MachFunction) -> InlineDecision;
}

/// Peephole rule selection
pub trait PeepholePolicy {
    fn should_apply(&self, rule: &PeepholeRule, context: &PeepholeContext) -> bool;
}

/// Cost model (pluggable)
pub trait CostModel {
    fn predict_throughput(&self, block: &MachBlock) -> f64;
    fn latency(&self, inst: &MachInst) -> u32;
}
```

**Defaults:** Hand-tuned heuristics for each trait. AI agents can replace any or all.

### 5. Automatic Rule Discovery

The killer feature: AI proposes optimization rules, solver verifies, proven rules are added.

```rust
pub struct RuleDiscovery {
    solver: Z4Solver,
    rule_db: RuleDatabase,
}

impl RuleDiscovery {
    /// AI agent proposes a new peephole rule
    pub fn propose_rule(&self, pattern: Pattern, replacement: Pattern) -> RuleResult {
        // 1. Encode pattern semantics as SMT
        let source_smt = self.encode_pattern(&pattern);
        let target_smt = self.encode_pattern(&replacement);
        
        // 2. Ask z4: are these equivalent for all inputs?
        match self.solver.check_equivalence(source_smt, target_smt) {
            Equivalent(proof) => {
                // 3. Proven correct! Add to rule database
                let rule = ProvenRule { pattern, replacement, proof };
                self.rule_db.add(rule);
                RuleResult::Accepted(rule)
            }
            NotEquivalent(counterexample) => {
                // 4. Not equivalent — return counterexample
                RuleResult::Rejected(counterexample)
            }
            Timeout => RuleResult::Inconclusive,
        }
    }
}
```

**The solver is the gatekeeper.** AI agents can propose anything — the solver only accepts proven-correct rules. No human review needed for correctness (soundness guaranteed by z4). Human review is only for performance impact.

### 6. Observation Space (for RL agents)

Following CompilerGym's pattern, define what AI agents can see:

```rust
pub struct Observation {
    /// Program features (fixed-size vector for RL)
    pub autophase: [f64; 56],  // Instruction mix features
    /// IR graph (for GNN-based agents)
    pub ir_graph: IrGraph,
    /// Cost estimate
    pub estimated_cycles: f64,
    /// Code size
    pub code_size_bytes: u32,
    /// Available actions
    pub legal_actions: Vec<ActionId>,
}
```

### 7. Action Space

What AI agents can do:

```rust
pub enum CompilerAction {
    /// Apply/skip a specific optimization pass
    TogglePass(PassId, bool),
    /// Reorder passes
    ReorderPasses(Vec<PassId>),
    /// Set a heuristic parameter
    SetParameter(ParamId, f64),
    /// Propose a new peephole rule
    ProposeRule(Pattern, Pattern),
    /// Choose a register for allocation
    AssignRegister(VRegId, PReg),
    /// Force/prevent inlining of a call site
    InlineDecision(CallSiteId, bool),
}
```

---

## Self-Improving Compiler Loop

The endgame: a compiler that improves itself with formal correctness guarantees.

```
┌─────────────────────────────────────┐
│         Compilation Loop             │
│                                      │
│  Compile program ──► Run binary      │
│       │                    │         │
│       │              Measure perf    │
│       │                    │         │
│       ▼                    ▼         │
│  AI agent observes    Runtime data   │
│  IR + cost feedback                  │
│       │                              │
│       ▼                              │
│  Propose new rule                    │
│       │                              │
│       ▼                              │
│  z4 verifies ──► PROVEN? ──► Add     │
│       │              │               │
│       │          NOT PROVEN           │
│       │              │               │
│       ▼              ▼               │
│   Discard      Return counterexample │
│                to AI agent           │
└─────────────────────────────────────┘
```

**Key property:** The solver is the formal gatekeeper. The AI agent can be arbitrarily creative or wrong — only proven-correct rules enter the rule database. The compiler cannot regress in correctness, only in performance (which is measurable and reversible).

---

## Implementation Plan

### Phase 1: Library API
- `Compiler` struct with `compile()` and `compile_observed()`
- `CompilationResult` with metrics
- JSON IR serialization

### Phase 2: Pass Pipeline API
- `PassPipeline` with insert/remove/reorder
- Pipeline serialization/deserialization
- Config-driven compilation

### Phase 3: Decision Point Hooks
- `CostModel` trait with hand-tuned default
- `RegAllocPolicy` trait
- `PeepholePolicy` trait

### Phase 4: Rule Discovery
- `RuleDiscovery` with z4 verification
- `RuleDatabase` persistence
- AI agent → propose → verify → add loop

### Phase 5: Observation/Action Space
- CompilerGym-compatible observation format
- Action space definition
- RL training harness

---

## Security Considerations

### Untrusted AI Agents

AI agents proposing rules may be:
- Buggy (propose incorrect rules)
- Adversarial (propose rules that introduce backdoors)

**Mitigation:** The solver is the gatekeeper. A rule that introduces a backdoor would not be semantically equivalent to the original — z4 would reject it. The soundness guarantee is: if z4 says "equivalent," then for ALL inputs, the output is identical. An adversary cannot smuggle behavior through a proven-equivalent rewrite.

**Remaining risk:** Performance degradation. An adversary could propose rules that are correct but slow. **Mitigation:** Performance regression testing on benchmark suite before accepting rules into the production database.

### Solver Soundness

If z4 has a bug, the entire guarantee collapses. **Mitigation:** Cross-check with exhaustive testing for small bitwidths. Verify the verifier (tRust).

---

## References

1. Cummins et al. "MLGO: a Machine Learning Guided Compiler Optimizations Framework." 2022.
2. Cummins et al. "CompilerGym: Robust, Performant Compiler Optimization Environments for AI Research." CGO 2022.
3. Chen et al. "TVM: An Automated End-to-End Optimizing Compiler for Deep Learning." OSDI 2018.
4. Zheng et al. "Ansor: Generating High-Performance Tensor Programs for Deep Learning." OSDI 2020.
5. Ragan-Kelley et al. "Halide: A Language and Compiler for Optimizing Parallelism, Locality, and Recomputation." PLDI 2013.
6. Mendis et al. "Ithemal: Accurate, Portable and Fast Basic Block Throughput Estimation using Deep Neural Networks." ICML 2019.
7. Cummins et al. "Meta Large Language Model Compiler." 2024.
8. Haj-Ali et al. "AutoPhase: Juggling HLS Phase Orderings in Random Forests with Deep Reinforcement Learning." MLSys 2020.
