# Solver-Driven Superoptimization

**Author:** Andrew Yates
**Date:** 2026-04-13
**Status:** Design
**Epic:** Solver-driven superoptimization

---

## Implementation Status (as of 2026-04-15)

**Overall: Offline synthesis framework is implemented and operational using mock evaluation. CEGIS loop works. No real z4 solver connected, so proofs are mock-verified (exhaustive for small widths, sampled for larger). Online synthesis and solver-driven search are not yet implemented.**

| Component | Status | Details |
|-----------|--------|---------|
| **CEGIS loop** (`llvm2-verify/cegis.rs`) | IMPLEMENTED | Counter-example guided inductive synthesis. Generates counterexamples, refines candidates, verifies via mock evaluation. |
| **Offline peephole synthesis** (`llvm2-verify/synthesis.rs`) | IMPLEMENTED | 1.7K LOC. SearchSpace enumeration over AArch64 instruction patterns, SynthesisEngine with candidate verification, ProvenRuleDb. SearchConfig controls max pattern/replacement length and width. |
| **Unified multi-target CEGIS** (`llvm2-verify/unified_synthesis.rs`) | IMPLEMENTED | 5.1K LOC. Cross-target candidate ranking across CPU scalar, NEON SIMD, GPU, and ANE. Generates TargetCandidates, fast-filters against counterexamples, ranks by cost. |
| **AI-native rule discovery** (`llvm2-verify/rule_discovery.rs`) | IMPLEMENTED | 1.4K LOC. Agent-proposed rules verified via CEGIS loop. |
| **Proof database** (`llvm2-verify/proof_database.rs`) | IMPLEMENTED | Stores discovered proven-correct rules with proof certificates. |
| **Cost model (Apple M-series)** (`llvm2-ir/cost_model.rs`) | IMPLEMENTED | 2.7K LOC. Hand-written latency/throughput tables for AArch64 instructions. Phase 1 of cost model complete. |
| **Peephole pass integration** (`llvm2-opt/peephole.rs`) | IMPLEMENTED | Peephole optimization pass exists in the pipeline. Applies hand-written rules. |
| **SMT solver connection** | NOT CONNECTED | Synthesis uses mock evaluation (Rust concrete arithmetic), not a real SMT solver. z4 bridge serializes to SMT-LIB2 but native z4 API not linked. See #34, #236. |
| **Online synthesis (compile-time)** | NOT IMPLEMENTED | No hot-path detection, no time-budgeted stochastic search during compilation. |
| **Dataflow pruning** | NOT IMPLEMENTED | Known-bits, value ranges (Souper-style) for search space pruning not yet built. |
| **STOKE-style mutations** | NOT IMPLEMENTED | Stochastic mutations with simulated annealing not yet built. |
| **Learned cost model (Phase 2)** | NOT IMPLEMENTED | No Ithemal-style learned model. Current cost model is hand-written tables. |
| **Verified superoptimizer (tRust)** | NOT IMPLEMENTED | The superoptimizer itself is not yet written in tRust or formally verified. |

**Implementation plan progress:**
- **Phase 1 (Offline rule mining)**: Implemented with mock evaluation. SearchSpace, CegisLoop, rule serialization, and peephole integration all exist. No real z4-backed proofs.
- **Phase 2 (Enhanced search)**: Cost model (hand-written) implemented. Dataflow pruning and STOKE mutations not yet built.
- **Phase 3 (Online synthesis)**: Not implemented.
- **Phase 4 (Verified superoptimizer)**: Not implemented.

---

## Introduction

Traditional compilers use hand-written pattern matching — thousands of peephole rules accumulated over decades by human experts. LLVM has ~2,000 InstCombine patterns. Each must be written, tested, and maintained by hand. Bugs in these patterns cause miscompilations (Alive2 has found hundreds).

LLVM2 takes a fundamentally different approach: **use the SMT solver to synthesize optimal instruction sequences**. Instead of a human writing `add x, x → shl x, 1`, the solver discovers it, proves it correct, and adds it to the rule database. No human in the loop. Every rule is proven correct by construction.

This is superoptimization — and LLVM2 is uniquely positioned to do it because we already have SMT encodings of both source (tMIR) and target (AArch64) semantics.

**The system itself will be written in tRust and formally verified.** The superoptimizer that finds optimizations is proven correct — we verify the verifier.

---

## Prior Art

### STOKE (Stanford, ASPLOS 2013)

Schkufza, Sharma, Aiken. "Stochastic Superoptimization." ASPLOS 2013.
GitHub: https://github.com/StanfordPL/stoke

**Algorithm:** MCMC with simulated annealing. Applies random mutations (7 operators: add/delete/replace instruction, swap opcode, change operand, local/global swap) to candidate programs. Cost function combines correctness (Hamming distance on test outputs) and performance (latency model). Metropolis-Hastings acceptance criterion.

**Verification:** Three tiers — (1) testing on random inputs (fast, unsound), (2) bounded verification to fixed loop depth, (3) full SMT bitvector proof. Per-instruction-class SMT handlers encode x86-64 semantics.

**Results:** Starting from `clang -O0`, produces code matching or exceeding `gcc -O3`. On floating-point, trades precision for 2-6x speedups.

**Limitations:** x86-64 only. Best on straight-line code (tens of instructions). Control flow and memory aliasing are hard. Research prototype, not production.

**Key insight for LLVM2:** STOKE verifies post-hoc. LLVM2 can integrate verification *into* the search loop — reject candidates z4 can't prove correct during search, not after.

### Souper (Google, 2017)

Sasnauskas, Chen, Collingbourne, Ketema, Lup, Taneja, Regehr. "Souper: A Synthesizing Superoptimizer." arXiv:1711.04422.
GitHub: https://github.com/google/souper

**Algorithm:** Enumerative synthesis with CEGIS (Counter-Example Guided Inductive Synthesis). Enumerates candidates in strictly increasing cost order. Batches of 300 candidates verified against Z3 (QF_BV bitvector theory).

**IR:** Custom DAG-based IR (~60 instruction kinds) with dataflow annotations: KnownZeros/KnownOnes, ConstantRange, DemandedBits, NumSignBits, NonZero, NonNegative, PowOfTwo. These annotations are critical for search pruning.

**Pruning:** Commutativity canonicalization, self-operation elimination, constant-on-constant rejection, width-mismatch filtering, 15+ boolean specializations. Cost-bounded: RHS cost must be less than LHS cost.

**LLVM integration:** Runs as a FunctionPass after InstCombine. Extracts candidates from optimized IR, queries solver, replaces via `replaceAllUsesWith()`. Redis caching avoids redundant queries.

**Results:** 4.4% (3 MB) reduction in Clang binary size. Found optimizations LLVM missed: even multiplication can't produce odd results, path-condition-enabled constant folding.

**Limitations:** Integer/bitwise only — no FP, no memory, no vectors. Synthesis depth limited to 1-3 instructions for practical compilation times.

**Key insight for LLVM2:** Dataflow annotations (known bits, ranges) are essential for pruning the search space to tractable size. We need these in llvm2-opt.

### Denali (Joshi et al., 2002)

Equality saturation approach to superoptimization. Represents all equivalent programs simultaneously in an e-graph, applies rewrite rules to saturation, then extracts the optimal program. Recently revived by the egg library (Willsey et al., 2021).

**Key insight for LLVM2:** Equality saturation is complementary to synthesis — it applies known-correct rules exhaustively, while synthesis discovers new rules.

### Optgen (Buchwald, CGO 2015)

Automatic generation of peephole optimizations from ISA specifications. Given machine-readable ISA semantics, enumerates instruction pairs and uses SMT to check if one can replace the other.

**Key insight for LLVM2:** We already have machine-readable AArch64 semantics in `aarch64_encoder.rs`. Optgen's approach is directly applicable.

### Alive2 (Lopes et al., PLDI 2021)

Verifies LLVM IR transformations by encoding source and target as SMT (Z3), checking refinement. Found 40+ miscompilation bugs in LLVM. Web UI at alive2.llvm.org.

**Key insight for LLVM2:** We use the same approach (SMT equivalence checking) but go further — we use it to *find* optimizations, not just verify human-written ones.

---

## LLVM2 Superoptimization Architecture

### Overview

Two-mode system: **offline synthesis** (build-time rule mining) and **online synthesis** (compile-time optimization of hot paths).

```
┌─────────────────────────────────────────────┐
│           Offline Synthesis (build-time)      │
│                                               │
│  Enumerate AArch64 patterns (length 1-3)      │
│         │                                     │
│         ▼                                     │
│  z4 equivalence check (CEGIS)                │
│         │                                     │
│    ┌────┴────┐                                │
│    │ Proved  │ ──► Rule Database (proven)     │
│    │ Failed  │ ──► Discard + counterexample   │
│    └─────────┘                                │
└─────────────────────────────────────────────┘

┌─────────────────────────────────────────────┐
│           Online Synthesis (compile-time)     │
│                                               │
│  Hot path identification (profile/heuristic)  │
│         │                                     │
│         ▼                                     │
│  Extract tMIR pattern from hot code           │
│         │                                     │
│         ▼                                     │
│  Search for shorter/faster AArch64 sequence   │
│  (guided by cost model + rule DB)             │
│         │                                     │
│         ▼                                     │
│  z4 proves equivalence ──► Apply if proven    │
└─────────────────────────────────────────────┘
```

### Offline Peephole Synthesis

Mine the AArch64 ISA for peephole rules at build time. This runs once, produces a database of proven-correct rules.

**Search space definition:**
- Enumerate all 1-instruction, 2-instruction, 3-instruction AArch64 sequences from the ~50 opcodes we support
- Operand patterns: register (any class), immediate (parametric), shifted register
- Filter by type compatibility (Gpr32↔Gpr32, Gpr64↔Gpr64, etc.)

**Pruning (from Souper):**
- Commutativity: only enumerate one operand ordering for commutative ops
- Self-elimination: skip `sub x, x`, `and x, x`, etc.
- Type mismatch: skip sequences with incompatible register classes
- Cost bound: replacement must be strictly cheaper than original
- Dataflow: use known-bits analysis to prune impossible patterns

**CEGIS loop:**
1. Propose candidate replacement: `pattern_A → pattern_B`
2. Ask z4: "∃ input where A(input) ≠ B(input)?"
3. If UNSAT → proven equivalent. Add rule to database with proof certificate.
4. If SAT → z4 returns counterexample input. Add to test suite, try next candidate.
5. Repeat until search space exhausted or time budget spent.

**Output:** `rules.json` — serialized database of `(pattern, replacement, proof_hash, cost_delta)`.

### Online Synthesis

For hot paths during compilation, search for better instruction sequences on the fly.

**When to trigger:**
- Functions marked `#[hot]` or with profile data showing high execution count
- Sequences where offline rules found no improvement (novel patterns)
- User-requested: `--superoptimize` flag

**Search strategy:** Hybrid STOKE/Souper approach:
1. Start from ISel output for the hot region
2. Apply known rules from offline database
3. If no improvement, run bounded stochastic search (STOKE-style mutations, z4 verification)
4. Time-budgeted: 100ms per basic block (configurable)

### Cost Model: Apple M-Series

Instruction ranking requires accurate latency/throughput data for Apple Silicon.

```rust
pub trait CostModel {
    /// Predicted throughput in cycles for a basic block
    fn predict_throughput(&self, block: &[MachInst]) -> f64;
    
    /// Latency of a single instruction
    fn latency(&self, inst: &MachInst) -> u32;
    
    /// Reciprocal throughput (instructions per cycle)
    fn throughput(&self, inst: &MachInst) -> f64;
}
```

**Phase 1:** Hand-written tables from Dougall Johnson's M1 microarchitecture research and Apple optimization guides.
**Phase 2:** Learned cost model (Ithemal-style LSTM or transformer) trained on measured Apple M-series data.

### Memory and NZCV Flag Handling

The hard problems in superoptimization:

**Memory operations:**
- Loads/stores have side effects — can't freely reorder
- Aliasing analysis required: two memory ops to provably different addresses can be reordered
- tMIR proofs (ValidBorrow, InBounds) provide alias info traditional compilers lack
- SMT encoding: model memory as an array sort `(Array (_ BitVec 64) (_ BitVec 8))`

**NZCV flags (condition codes):**
- Many AArch64 instructions implicitly read/write NZCV
- Flag-producing instructions (CMP, ADDS, SUBS) must be correctly paired with flag-consuming instructions (B.cond, CSEL, CSET)
- SMT encoding: model NZCV as 4 separate boolean variables, encode flag semantics per instruction

**AArch64-specific traps:**
- SP vs XZR: register 31 is SP or ZR depending on context — synthesis must respect this
- Implicit zero-extension: 32-bit writes zero upper 32 bits — affects equivalence checking
- Scaled vs unscaled offsets: different encoding for different offset types

### Integration Point

Superoptimization runs in two places in the LLVM2 pipeline:

1. **Post-ISel peephole** (in `llvm2-opt`): Apply offline rules after instruction selection
2. **Late combine** (in `llvm2-opt`): After register allocation, apply register-aware rules

```
tMIR → ISel → [superopt peephole] → RegAlloc → [late superopt] → Encode → Mach-O
```

---

## SMT Encoding Requirements

What z4 must support for superoptimization:

| Feature | z4 Theory | Use |
|---------|-----------|-----|
| Integer arithmetic | QF_BV (bitvectors) | Core instruction semantics |
| Memory operations | QF_ABV (arrays + bitvectors) | Load/store equivalence |
| Condition codes | QF_BV (4-bit NZCV) | Flag-producing/consuming instructions |
| Uninterpreted functions | QF_UF | Abstract external calls |
| Quantifier-free | All | Keep queries decidable |

Current z4 status: bitvector support exists. Array theory and uninterpreted functions needed for memory and calls.

---

## Implementation Plan

### Phase 1: Offline Rule Mining (MVP)

**Module:** `crates/llvm2-verify/src/synthesis.rs`

1. Define `SearchSpace` — enumerable set of AArch64 instruction patterns
2. Implement `CegisLoop` — counterexample-guided equivalence checking against z4
3. Mine rules for our ~50 opcodes (length 1-2 replacements)
4. Serialize proven rules to `rules.json` with proof certificates
5. Integrate rule application into `llvm2-opt` peephole pass

### Phase 2: Enhanced Search

1. Add dataflow pruning (known bits, value ranges) from Souper
2. Extend to length-3 replacements
3. Add STOKE-style stochastic mutations for harder patterns
4. Implement Apple M-series cost model (hand-written tables)

### Phase 3: Online Synthesis

1. Hot path detection (profile data or heuristic)
2. Time-budgeted stochastic search during compilation
3. Parallel synthesis (spawn search on background thread)
4. Cache synthesis results across compilation units

### Phase 4: Verified Superoptimizer (tRust)

**The superoptimizer itself is written in tRust and formally verified.**

1. Prove that the CEGIS loop is sound: if it reports "equivalent," the two patterns are semantically equivalent
2. Prove that the search space enumeration is complete: no valid patterns are skipped
3. Prove that rule application preserves program semantics
4. The verifier verifies itself — bootstrapping correctness

---

## Expected Results

Based on STOKE and Souper findings:

| Optimization Type | Example | Expected Source |
|-------------------|---------|-----------------|
| Strength reduction | `mul x, 2` → `lsl x, #1` | Offline mining |
| Identity elimination | `add x, #0` → (elide) | Offline mining |
| Instruction fusion | `cmp` + `b.eq` → `cbz` | Offline mining |
| Constant folding | `movz` + `add` → single `movz` | Offline mining |
| Cross-instruction | `and` + `tst` → `ands` | Offline mining |
| Novel patterns | Context-dependent sequences | Online synthesis |
| tMIR-proof-enabled | Unchecked arithmetic (NoOverflow) | Online + proofs |

Conservative estimate: 2-5% code size reduction over hand-written rules alone, with every optimization proven correct.

---

## Risks and Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| Search space explosion | Synthesis takes too long | Cost-bounded search, aggressive pruning, time budgets |
| z4 solver timeouts | Can't verify complex patterns | Timeout per query (1s), skip and log unverifiable patterns |
| Memory operation complexity | SMT encoding blowup | Start with pure integer/register patterns, add memory later |
| Cost model inaccuracy | Select slower sequences | Validate against hardware measurements |
| Compilation time overhead | Online synthesis too slow | Offline-first strategy, online only for marked hot paths |
| Soundness of verifier | Bugs in SMT encoding | Verify the verifier (tRust), cross-check with exhaustive testing for small bitwidths |

---

## References

1. Schkufza, Sharma, Aiken. "Stochastic Superoptimization." ASPLOS 2013.
2. Sasnauskas et al. "Souper: A Synthesizing Superoptimizer." arXiv:1711.04422, 2017.
3. Joshi et al. "Denali: A Goal-directed Superoptimizer." PLDI 2002.
4. Buchwald. "Optgen: A Generator for Local Optimizations." CGO 2015.
5. Lopes et al. "Alive2: Bounded Translation Validation for LLVM." PLDI 2021.
6. Willsey et al. "egg: Fast and Extensible Equality Saturation." POPL 2021.
7. Bansal, Aiken. "Automatic Generation of Peephole Superoptimizers." ASPLOS 2006.
8. Phothilimthana et al. "Scaling Up Superoptimization." ASPLOS 2016.
