# Wave 1 Integration Audit Report

**Date:** 2026-04-14
**Auditor:** Factory Auditor Agent
**Scope:** 10 new modules merged in wave 1 (~10K LOC)

## Executive Summary

The wave 1 code is **solid**. All 483 tests pass, the build is clean (zero errors), and the architecture is well-structured. The new modules integrate correctly at the compilation level. Five issues were filed for cross-module consistency problems and test coverage gaps, none of which are blocking.

## 1. Build and Test Results

### Build

```
cargo build --workspace: SUCCESS (0 errors, 0 warnings)
```

### Tests

```
cargo test --workspace: 483 passed, 0 failed, 0 ignored
Doc-tests: 8 passed, 3 ignored (expected: external setup needed)
```

### Warnings (4 total, all in llvm2-lower)

4 unreachable pattern warnings in `crates/llvm2-lower/src/isel.rs` lines 5887, 6021-6022. These are duplicate `AArch64Opcode::StrRI` variants in `matches!` macros -- copy-paste errors. **Already tracked:** #136.

## 2. Integration Review

### Re-exports

All new modules are correctly declared in their respective `lib.rs` files:

| Module | Crate | lib.rs declaration | Re-exports |
|--------|-------|--------------------|------------|
| `synthesis.rs` | llvm2-verify | `pub mod synthesis` | None at root (used via module path) |
| `cegis.rs` | llvm2-verify | `pub mod cegis` | `CegisLoop, CegisResult, ConcreteInput` |
| `rule_discovery.rs` | llvm2-verify | `pub mod rule_discovery` | `RuleDiscovery, RuleProposal, RuleResult, RuleDatabase, DiscoveryStats` |
| `neon_semantics.rs` | llvm2-verify | `pub mod neon_semantics` | None (internal) |
| `cost_model.rs` | llvm2-ir | `pub mod cost_model` | None (accessed via module path) |
| `trace.rs` | llvm2-ir | `pub mod trace` | `CompilationEvent, CompilationTrace, EventKind, Justification, RuleId, TraceLevel` |
| `provenance.rs` | llvm2-ir | `pub mod provenance` | `PassId, ProvenanceEntry, ProvenanceMap, ProvenanceStats, ProvenanceStatus, TmirInstId, TransformKind, TransformRecord` |
| `compiler.rs` | llvm2-codegen | (in codegen crate) | Via crate API |
| `target_analysis.rs` | llvm2-lower | (in lower crate) | Via module path |

### Synthesis/CEGIS/Rule Discovery Integration

These three modules form a clean pipeline:

- `synthesis.rs` defines `SynthOpcode`, `RuleCandidate`, `ProvenRule`, `ProvenRuleDb`
- `cegis.rs` provides `CegisLoop` using `SmtExpr`, `ProofObligation`, and `Z4Config`
- `rule_discovery.rs` imports from both `synthesis` and `cegis`, composing them correctly: `RuleDatabase` wraps `ProvenRuleDb`, and `RuleDiscovery` uses `CegisLoop`

The composition is proper -- no duplication between `ProvenRuleDb` and `RuleDatabase`. The latter extends the former with AI-discovery-specific features (deduplication, statistics, query-by-name).

### Cost Model Integration

`cost_model.rs` in llvm2-ir is self-contained and does not integrate with anything. It provides a `CostModel` trait and `AppleSiliconCostModel` but is not consumed by any of the new synthesis/CEGIS modules. **Filed:** #149.

## 3. Cross-Module Consistency

### FINDING: Duplicate PassId and TmirInstId (P2)

Both `trace.rs` and `provenance.rs` define `PassId` and `TmirInstId` with different representations:

| Type | trace.rs | provenance.rs |
|------|----------|---------------|
| `PassId` | `PassId(pub u32)` | `PassId(pub String)` |
| `TmirInstId` | `TmirInstId(pub u32)` | `TmirInstId(pub u32)` |

Only the provenance versions are re-exported at crate root. The trace versions shadow them within the `trace` module. These need to be unified. **Filed:** #147.

### FINDING: Duplicate TraceLevel and CompilationTrace (P3)

Both `llvm2-ir::trace` and `llvm2-codegen::compiler` define `TraceLevel` (with different variants) and `CompilationTrace` (with different structures). They serve different purposes (structured event log vs. timing trace) but the name collision will cause confusion. **Filed:** #148.

### NEON Semantics and SMT Compatibility

NEON semantics (`neon_semantics.rs`) correctly uses `SmtExpr` operations including `Concat`, `Extract`, `ZeroExtend`, and `SignExtend` through the `smt.rs` lane helper functions (`map_lanes_binary`, `map_lanes_unary`, `lane_extract`, `concat_lanes`). These are fully compatible with the CEGIS loop -- both operate on the same `SmtExpr` AST.

### Synthesis and Rule Discovery Storage

Not duplicated. `RuleDatabase` (rule_discovery.rs) wraps `ProvenRuleDb` (synthesis.rs) via composition. The `DiscoveredRule` type carries additional metadata (SMT expressions, CEGIS iteration count) beyond what `ProvenRule` stores. This is a clean extension.

## 4. Test Quality Assessment

### synthesis.rs (30 tests)

**Strengths:**
- Comprehensive positive tests for known-good rules (ADD x,#0, SUB x,#0, MUL x,#1, MUL x,#2 -> LSL x,#1, ORR x,x, AND x,x, EOR x,#0, shifts by 0, SUB x,x)
- Negative tests for known-bad rules (ADD x,#1, MUL x,#2 identity, NEG x identity)
- Full synthesis run test verifying that the search space produces expected rules
- Multi-width tests (8-bit exhaustive, 32-bit sampling, 64-bit sampling)
- Cost model and display tests

**Gaps:**
- No test for multi-instruction patterns (max_pattern_len > 1)
- No edge case test for `SameAsInput` with index > 0

### cegis.rs (22 tests)

**Strengths:**
- Tests both concrete-only and full CEGIS paths
- Counterexample accumulation tested across queries
- Integration with existing ProofObligation infrastructure
- Graceful handling when solver is unavailable (Error variant accepted)
- Peephole identity tests (double negation, strength reduction, xor-self, and-self, or-zero, sub-self)
- Proof hash determinism test
- Edge case seed generation tested (single-var, two-var cross-product)
- Reset functionality tested

**Gaps:**
- MaxIterationsReached path not directly tested (would require a mock solver)
- No test for the counterexample validation sanity check (lines 350-359)

### rule_discovery.rs (26 tests)

**Strengths:**
- Acceptance tests for correct rules
- Rejection tests for incorrect rules with counterexample validation
- Deduplication tested (same proposal submitted twice)
- Batch discovery tested
- Statistics tracking verified
- 32-bit width tests
- Counterexample accumulation across proposals
- Database CRUD operations tested
- Cost estimation for various expression types

**No tautological tests detected.** All assertions test meaningful conditions.

### neon_semantics.rs (20 tests)

**Strengths:**
- All three supported arrangements tested (S2, H4, B8) for ADD, SUB, MUL, NEG
- Wrapping behavior tested for ADD and MUL
- Bitwise operations tested (AND, ORR, EOR, BIC)
- Shift operations tested (SHL, USHR, SSHR) with various arrangements
- Cross-check: ADD then SUB = identity
- Cross-check: SHL then USHR clears bits correctly
- Signed shift tested (sign extension verified)

**Major gap:** No 128-bit vector tests (S4, H8, B16, D2). **Filed:** #150.

### cost_model.rs (21 tests)

**Strengths:**
- Exhaustive opcode coverage test (all opcodes return valid costs)
- Relative ordering tests (div > mul > add, load > alu, fp_div > fp_add)
- M4 vs M1 comparison tests
- Block throughput prediction with various scenarios (empty, single, latency-bound, throughput-bound, load-heavy, mixed FP)
- Pseudo-instruction zero-latency/infinite-throughput tested

**No gaps identified.** The cost model is well-tested.

### compiler.rs (11 tests)

**Strengths:**
- Configuration defaults tested
- All trace levels tested
- Proof emission placeholder tested
- End-to-end `compile_ir_function` with actual code emission tested

### target_analysis.rs

Tests not fully reviewed due to file length, but the module structure follows the pruning algorithm from the design doc and integrates with the existing `Proof` and `ProofContext` adapter types.

## 5. Issues Filed

| # | Title | Priority | Finding |
|---|-------|----------|---------|
| #147 | Duplicate PassId and TmirInstId types in trace.rs and provenance.rs | P2 | Type confusion, different representations for same concept |
| #148 | Duplicate TraceLevel and CompilationTrace types across codegen and ir | P3 | Name collision between unrelated types |
| #149 | cost_model.rs not integrated with synthesis cost estimation | P2 | Detailed cost data available but unused by synthesis |
| #150 | NEON semantics limited to 64-bit vectors (no 128-bit test coverage) | P2 | 128-bit NEON operations have zero test coverage |
| #151 | synthesis.rs uses evaluation-only, not CEGIS | P3 | Weaker verification guarantee for wider widths |

Pre-existing: #136 (unreachable pattern warnings in isel.rs, already tracked).

## 6. Recommendations

### Immediate (before next wave)
1. Fix the PassId/TmirInstId duplication (#147) -- this is a landmine for anyone adding cross-module integration
2. Integrate the cost model with synthesis (#149) -- the data is already there

### Near-term
3. Extend SmtExpr evaluator to support 128-bit bitvectors for NEON testing (#150)
4. Add CEGIS option to synthesis engine for formal verification of wider widths (#151)
5. Rename codegen TraceLevel/CompilationTrace to avoid confusion (#148)

### Architecture Note
The synthesis/CEGIS/rule_discovery pipeline is well-designed. The layered approach (synthesis for enumeration, CEGIS for formal verification, rule_discovery for AI-proposed rules) follows the Alive2/Souper model correctly. The main gap is that the synthesis layer should optionally use CEGIS for final acceptance rather than evaluation-only.
