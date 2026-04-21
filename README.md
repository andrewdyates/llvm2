# LLVM2 — Verified Compiler Backend

**Author:** Andrew Yates
**Copyright:** 2026 Andrew Yates
**License:** Apache 2.0
**Status:** Alpha — compiles integer programs to correct machine code on AArch64 and x86-64

Verified codegen from tMIR to machine code. Every instruction lowering is mathematically proven to preserve semantics via z4 SMT.

LLVM2 is the final stage of the t\* verified compilation pipeline. Not a fork of LLVM or Cranelift — a purpose-built backend designed for proof-carrying IR.

> **Honest assessment:** LLVM2 is a working compiler for integer programs (~208K LOC, 6,177 tests, 1,182 SMT proof functions). Programs like fibonacci, GCD, collatz, factorial, and recursive power compile to native binaries and produce correct results — verified by differential testing against clang. It is **not** a production compiler: no aggregate types, limited floating-point, no SIMD codegen, and optimization passes above O0 have known regressions. But the core pipeline — ISel, register allocation, encoding, Mach-O/ELF emission — is functional end-to-end. See [Current Status](#current-status) for details.

```
Verified source (tRust/tSwift/tC)
         |
         v
       tMIR (proof-carrying IR)
         |
         v
+-------------------------------------+
|              LLVM2                   |
|                                      |
|  tMIR -> MachIR -> Machine Code     |
|         |            |               |
|         +------+-----+              |
|                v                     |
|         z4 proves each               |
|         lowering correct             |
+-------------------------------------+
         |
         v
   Verified Binary + Certificate
```

## Why Not LLVM?

| Problem | Impact |
|---------|--------|
| 20M+ LOC unverified C++ | Too large to verify |
| General-purpose | Bloat for our use case |
| No proof awareness | Can't leverage tMIR proofs |
| Optimizations unverified | "Trust us" for correctness |

## Why Not Cranelift?

| Problem | Impact |
|---------|--------|
| Verification-unaware | Doesn't preserve proofs |
| Different goals | Fast compilation, not verified compilation |
| Extensive modification needed | Easier to build purpose-built |

## Verification Approach

Every lowering `tMIR_instruction -> MachineCode` is verified:

1. **Encode semantics** — tMIR instruction as SMT formula
2. **Encode result** — machine instruction as SMT formula
3. **Prove equivalence** — z4 proves the two are semantically identical for all inputs

This follows Alive2 and CompCert approaches, applied systematically to the entire backend.

## Current Status

**LLVM2 compiles integer programs to correct native binaries on AArch64 and x86-64.** The full pipeline — instruction selection, register allocation, optimization, binary encoding, and object file emission — works end-to-end. Correctness is validated by differential testing against clang and triple-oracle validation (tMIR interpreter + LLVM2 + clang must all agree).

### What actually works

- **Integer programs compile and run correctly** — fibonacci, GCD (Euclidean), collatz sequence, factorial, recursive power, sum, count_digits, nested loops. Verified against clang golden truth on AArch64
- **Triple-oracle validation** — tMIR interpreter, LLVM2-compiled binary, and clang-compiled binary independently compute the same results (39 test cases across 6 functions)
- **Differential testing vs clang** — compile the same function through LLVM2 and clang, run both, assert identical stdout
- **132 AArch64 opcodes** encoded (integer, memory, FP, branch, NEON, system)
- **74 x86-64 opcodes** encoded and functional under Rosetta 2 (24 correctness tests pass)
- **1,182 SMT proof functions** verify lowering correctness, peephole identities, NEON semantics, memory model, GPU/ANE dispatch
- **6,177 tests** across 200 source files (unit tests + 22 E2E test suites)
- **tMIR frontend connected** — tRust produces tMIR, trust-llvm2-bridge translates to LLVM2 LIR, full compile-and-run pipeline works
- **tMIR interpreter** — direct IR evaluation for golden-truth validation without codegen (1,300+ LOC, supports arithmetic, branches, loops, recursive calls)
- **DWARF debug info** — __debug_abbrev, __debug_info, __debug_line sections + CFI for both AArch64 and x86-64
- **Optimization pipeline** — 18 passes including DCE, constant folding, copy propagation, peephole, CSE, LICM, NEON auto-vectorization, address mode formation, compare/select combines. O0 produces fully correct code. O1/O2 have known regressions under investigation
- **Register allocation** — linear scan + greedy allocator, liveness analysis, interval splitting, spill generation, phi elimination, copy coalescing, rematerialization
- **Object file emission** — Mach-O (AArch64), Mach-O (x86-64 via Rosetta 2), ELF stubs for Linux. Compact unwind, branch relaxation, relocations
- **Heterogeneous compute design** — Metal GPU kernel emission (MSL), CoreML/ANE model emission, computation graph analysis, cost-model-driven dispatch planning
- **CEGIS superoptimization** — counter-example guided synthesis for peephole rules, unified multi-target solver, AI-native rule discovery

### What does NOT work yet

- **No aggregate types.** Structs, arrays, unions — cannot compile code that uses them
- **~91% of AArch64 ISA missing.** No crypto extensions, no SVE, limited SIMD codegen
- **O1/O2 optimization regressions.** SRem-based loops break at O1, loop-carried accumulators break at O2. O0 is fully correct
- **No dynamic linking, no TLS, no exception handling**
- **x86-64 is functional but secondary.** 74 opcodes, runs under Rosetta 2 — not tested on real x86 hardware
- **RISC-V scaffolding only.** Opcode enum + register definitions, no encoding or pipeline
- **tMIR bridge is scalar-only.** No vector types, no aggregate passing through the bridge
- **No benchmark suite.** Optimization quality untested against clang -O2

### Honest comparison to LLVM

| Dimension | LLVM | LLVM2 today |
|-----------|------|-------------|
| ISA coverage | ~1,500 AArch64 opcodes | ~132 opcodes |
| Optimization passes | 100+ mature passes | 18 passes (O0 correct, O1/O2 WIP) |
| Targets | AArch64, x86-64, RISC-V, ARM, MIPS, ... | AArch64 (primary), x86-64 (functional) |
| Can compile C/Rust? | Yes, production quality | Integer programs only |
| Verification | None (trust-based) | 1,182 SMT proofs (our key advantage) |
| Correctness testing | Decades of fuzzing + production use | Differential testing vs clang + triple oracle |
| Maturity | 20+ years, millions of users | Alpha, 5 days old |

### What this project IS

LLVM2 is a **verified compiler backend that produces correct machine code for integer programs**, with a clear path to full-language support. The value is in:
1. **The verification approach** — every lowering proven correct via SMT, with 1,182 proof functions and growing
2. **The architecture** — 6-crate design with clean separation, ~208K LOC of Rust
3. **Working end-to-end pipeline** — tMIR → ISel → RegAlloc → Encoding → Mach-O → link → run → correct output
4. **Differential testing** — every program verified against clang and an independent tMIR interpreter
5. **The foundation** — designed to be built upon toward full verified compilation

## Vision: Designed for the 2026 AI Era

LLVM2 isn't just "LLVM rewritten in Rust." It's designed from scratch for a world where AI writes most code and correctness is non-negotiable.

### 1. Solver-Driven Optimization (Superoptimization)

Traditional compilers use hand-written pattern matching — thousands of peephole rules written by humans over decades. LLVM2 uses z4 SMT to **find** optimal instruction sequences, not just verify them.

Given a tMIR expression, the solver can:
- Enumerate candidate AArch64 instruction sequences
- Prove equivalence against the source semantics
- Select the shortest/fastest correct sequence
- Generate new optimization rules that are **provably correct by construction**

This is superoptimization (cf. [STOKE](https://github.com/StanfordPL/stoke), [Souper](https://github.com/google/souper)) applied systematically to a full backend. Every optimization we discover comes with a proof certificate. No hand-written rules, no "trust us" — the solver finds it and proves it.

**Status:** CEGIS synthesis loop implemented. Peephole synthesis (Alive2-style enumeration) and unified multi-target CEGIS solver functional. AI-native rule discovery pipeline in place.

### 2. Radical Debugging & Transparency

Every compilation decision is logged, justified, and traceable:

- **Full transformation audit trail** — from tMIR input to final binary, every lowering, optimization, and register assignment is recorded with *why* that choice was made
- **Proof certificates** — independently verifiable artifacts proving each lowering is correct. Ship the certificate alongside the binary
- **Provenance tracking** — tMIR-to-binary offset mapping traces every machine instruction back to its source
- **Compilation trace** — structured event log for glass-box transparency

The compiler is a glass box, not a black box.

**Status:** Provenance tracking and compilation trace infrastructure implemented. Interactive explorer not yet built.

### 3. AI-Native Compilation

LLVM2 is designed for AI agents to interact with, not just humans:

- **Machine-readable compilation artifacts** — structured JSON/binary output at every pipeline stage, not just text dumps
- **API-first design** — the compiler is a library, not just a CLI. AI agents can query, modify, and extend the pipeline programmatically
- **Self-documenting pipeline** — every pass, every IR node, every proof is introspectable
- **Feedback loops** — AI agents propose new optimization rules, the solver verifies them, proven-correct rules are added automatically

**Status:** Library-based crate architecture supports this. AI-native rule discovery pipeline functional. CLI with JSON output exists.

### 4. Automatic Heterogeneous Compute Allocation

A compiler already decides which register to use and when to spill to memory. **Why should the programmer manually target GPU, Neural Engine, or SIMD?** Compute resource allocation is the same problem as register allocation at a coarser granularity.

An M-series MacBook has 12 CPU cores, 40 GPU cores, 16 Neural Engine cores, ~40 TOPS of compute. Most programs use one CPU core. The compiler should fix this:

- **Computation graph analysis** — identify data-parallel and matrix-heavy subgraphs in tMIR
- **Automatic dispatch** — compiler decides: CPU, GPU (Metal), Neural Engine (CoreML), or SIMD (NEON)
- **Cost-model driven** — latency, throughput, energy, and data transfer costs determine placement
- **Proven correct** — every compute allocation decision verified to preserve program semantics
- **Energy-aware** — optimize for performance, battery life, or a custom balance

The programmer writes pure math. The compiler maps it to the best hardware. With proofs.

**Status:** Computation graph analysis, dispatch planning, Metal MSL emission, CoreML MIL emission, and Apple Silicon cost model implemented. NEON auto-vectorization pass functional. End-to-end heterogeneous dispatch not yet wired up.

**Design docs:** designs/2026-04-13-superoptimization.md, designs/2026-04-13-debugging-transparency.md, designs/2026-04-13-ai-native-compilation.md, designs/2026-04-13-heterogeneous-compute.md

## Crates

| Crate | Lines | Tests | Description |
|-------|------:|------:|-------------|
| `llvm2-ir` | 13,059 | 393 | Shared machine model: MachInst, MachFunction, registers (GPR/FPR/SIMD), operands, stack slots, calling conventions, multi-target cost model, provenance tracking, compilation trace |
| `llvm2-lower` | 32,284 | 784 | tMIR-to-MachIR instruction selection, Apple AArch64 ABI, x86-64 ISel, type system, legalization, tMIR adapter, computation graph analysis, dispatch planning |
| `llvm2-opt` | 21,709 | 452 | 18 optimization passes: DCE, constant folding, copy propagation, peephole, CSE, LICM, CFG simplification, NEON auto-vectorization, address mode formation, compare/select combines, pass manager (O0-O3) |
| `llvm2-regalloc` | 16,408 | 399 | Linear scan + greedy allocator, liveness analysis, interval splitting, spill generation, phi elimination, copy coalescing, post-RA coalescing, rematerialization, call-clobber handling |
| `llvm2-codegen` | 49,217 | 1,930 | AArch64 + x86-64 binary encoding, Mach-O writer, ELF stubs, Metal MSL emitter, CoreML MIL emitter, DWARF debug info + CFI, frame lowering, compact unwind, branch relaxation, tMIR interpreter, end-to-end pipeline |
| `llvm2-verify` | 75,541 | 2,219 | 1,182 SMT proof functions: lowering proofs, peephole proofs, NEON semantics, memory model (concrete + symbolic QF_ABV), Metal/ANE semantics, CEGIS synthesis, AI-native rule discovery, z4 bridge |

**Totals:** ~208,000 lines of Rust, 6,177 tests across 200 source files.

## Installation

```bash
git clone https://github.com/andrewdyates/LLVM2
cd LLVM2
cargo build
cargo test  # 6,177 tests
```

See designs/2026-04-12-aarch64-backend.md for the full backend design (AI Model-reviewed).

## The t\* Stack

```
+-------------------------------------------------------------+
|                    Source Languages                           |
+-------------------------------------------------------------+
|  tRust           tSwift              tC                       |
|     |               |                   |                     |
|     +---------------+-------------------+                     |
|                         |                                     |
|                         v                                     |
|                      tMIR                                     |
|          (universal proof-carrying IR)                        |
|                         |                                     |
|            +------------+------------+                        |
|            v                         v                        |
|     +-------------+          +-------------+                  |
|     |  tla2 + z4  |          |    LLVM2    |  <-- this repo   |
|     |  (verify)   |          |  (codegen)  |                  |
|     +-------------+          +-------------+                  |
|                                     |                         |
|                                     v                         |
|                       Verified Machine Code                   |
+-------------------------------------------------------------+
```

## Related Projects

| Project | Role |
|---------|------|
| [tMIR](https://github.com/andrewdyates/tMIR) | Input IR (proof-carrying) |
| [tRust](https://github.com/andrewdyates/tRust) | Rust frontend |
| [tSwift](https://github.com/andrewdyates/tSwift) | Swift frontend |
| [tC](https://github.com/andrewdyates/tC) | C verification |
| [z4](https://github.com/andrewdyates/z4) | SMT solver backend |

## License

Apache 2.0 — see [LICENSE](LICENSE).
