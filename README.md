# LLVM2 — Verified Compiler Backend

**Author:** Andrew Yates
**Copyright:** 2026 Dropbox, Inc.
**License:** Apache 2.0
**Status:** Preview — proof-of-concept skeleton, not a production compiler

Verified codegen from tMIR to machine code. Every instruction lowering is mathematically proven to preserve semantics via z4 SMT.

LLVM2 is the final stage of the t\* verified compilation pipeline. Not a fork of LLVM or Cranelift — a purpose-built backend designed for proof-carrying IR.

> **Honest assessment:** LLVM2 is a well-architected proof-of-concept (~61K LOC, 1,688 tests, 174 SMT proofs). It is **not** a production compiler and cannot replace LLVM, Cranelift, or GCC today. The architecture, verification approach, and core algorithms are real — but significant work remains before it can compile non-trivial programs. See [Current Status](#current-status) for details.

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

**LLVM2 is a proof-of-concept skeleton.** The architecture is designed, core algorithms are implemented, and the verification framework is real — but this is not a production compiler. It cannot compile real-world programs today.

### What actually works

- **Toy integer programs** compile to valid Mach-O and execute correctly on Apple Silicon (add, sub, mul, comparison, branching, function calls)
- **~50 AArch64 opcodes** are encoded (out of ~1,500+ in the full ISA)
- **174 SMT proof functions** verify lowering correctness for the instructions we support
- **1,688 tests** pass (unit tests for each component — not end-to-end program tests)
- Individual pipeline stages work in isolation: instruction selection, register allocation, encoding, Mach-O emission

### What does NOT work yet

- **No real frontend.** tMIR stubs are in-tree; the real tMIR repo is not connected
- **No aggregate types.** Structs, arrays, unions — cannot compile code that uses them
- **~97% of AArch64 opcodes are missing.** No SIMD/NEON, no crypto extensions, no SVE, limited FP
- **No address mode formation.** Base+offset, pre/post-index, scaled addressing — not implemented
- **No debug info.** No DWARF, no source maps, no line tables
- **No x86-64 or RISC-V.** Scaffolding only (opcode enums, register definitions)
- **Optimization quality is untested against clang.** No benchmark suite exists
- **No dynamic linking, no TLS, no exception handling, no PIC beyond basics**
- **Greedy register allocator not implemented** (linear scan only)
- **No proof-consuming optimizations** (the proofs exist but don't drive optimization yet)

### Honest comparison to LLVM

| Dimension | LLVM | LLVM2 today |
|-----------|------|-------------|
| ISA coverage | ~1,500 AArch64 opcodes | ~50 opcodes |
| Optimization passes | 100+ mature passes | 11 skeleton passes |
| Targets | AArch64, x86-64, RISC-V, ARM, MIPS, ... | AArch64 only (partial) |
| Can compile C/Rust? | Yes, production quality | No |
| Verification | None (trust-based) | 174 SMT proofs (our key advantage) |
| Maturity | 20+ years, millions of users | Proof-of-concept, 2 weeks old |

### What this project IS

LLVM2 is an **architecture and verification research prototype**. The value is in:
1. **The verification approach** — proving every lowering correct via SMT is novel and real
2. **The architecture** — 6-crate design with clean separation is sound and extensible
3. **The proof-of-concept** — demonstrates that verified compilation from proof-carrying IR is feasible
4. **The foundation** — designed to be built upon, not thrown away

## Vision: Designed for the 2026 AI Era

LLVM2 isn't just "LLVM rewritten in Rust." It's designed from scratch for a world where AI writes most code and correctness is non-negotiable.

### 1. Solver-Driven Optimization (Superoptimization)

Traditional compilers use hand-written pattern matching — thousands of peephole rules written by humans over decades. LLVM2 will use z4 SMT to **find** optimal instruction sequences, not just verify them.

Given a tMIR expression, the solver can:
- Enumerate candidate AArch64 instruction sequences
- Prove equivalence against the source semantics
- Select the shortest/fastest correct sequence
- Generate new optimization rules that are **provably correct by construction**

This is superoptimization (cf. [STOKE](https://github.com/StanfordPL/stoke), [Souper](https://github.com/google/souper)) applied systematically to a full backend. Every optimization we discover comes with a proof certificate. No hand-written rules, no "trust us" — the solver finds it and proves it.

**Status:** SMT encoding framework exists for both tMIR and AArch64 semantics. Superoptimization synthesis loop is not yet implemented.

### 2. Radical Debugging & Transparency

Every compilation decision is logged, justified, and traceable:

- **Full transformation audit trail** — from tMIR input to final binary, every lowering, optimization, and register assignment is recorded with *why* that choice was made
- **Proof certificates** — independently verifiable artifacts proving each lowering is correct. Ship the certificate alongside the binary
- **Interactive compilation explorer** — query any instruction in the output: "why was this generated? what tMIR instruction produced it? what alternatives were considered?"
- **Regression diagnostics** — when output quality degrades, the audit trail pinpoints exactly which transformation is responsible

The compiler is a glass box, not a black box.

**Status:** Infrastructure exists (SMT proofs, typed IR at each stage). Audit trail logging and explorer are not yet implemented.

### 3. AI-Native Compilation

LLVM2 is designed for AI agents to interact with, not just humans:

- **Machine-readable compilation artifacts** — structured JSON/binary output at every pipeline stage, not just text dumps
- **API-first design** — the compiler is a library, not just a CLI. AI agents can query, modify, and extend the pipeline programmatically
- **Self-documenting pipeline** — every pass, every IR node, every proof is introspectable
- **Feedback loops** — AI agents can propose new optimization rules, the solver verifies them, proven-correct rules are added automatically

**Status:** Library-based crate architecture supports this. API surface and structured output are not yet implemented.

### 4. Automatic Heterogeneous Compute Allocation

A compiler already decides which register to use and when to spill to memory. **Why should the programmer manually target GPU, Neural Engine, or SIMD?** Compute resource allocation is the same problem as register allocation at a coarser granularity.

An M-series MacBook has 12 CPU cores, 40 GPU cores, 16 Neural Engine cores, ~40 TOPS of compute. Most programs use one CPU core. The compiler should fix this:

- **Computation graph analysis** — identify data-parallel and matrix-heavy subgraphs in tMIR
- **Automatic dispatch** — compiler decides: CPU, GPU (Metal), Neural Engine (CoreML), or SIMD (NEON)
- **Cost-model driven** — latency, throughput, energy, and data transfer costs determine placement
- **Proven correct** — every compute allocation decision verified to preserve program semantics
- **Energy-aware** — optimize for performance, battery life, or a custom balance

The programmer writes pure math. The compiler maps it to the best hardware. With proofs.

**Status:** Design complete. Implementation not started.

**Design docs:** `designs/2026-04-13-superoptimization.md`, `designs/2026-04-13-debugging-transparency.md`, `designs/2026-04-13-ai-native-compilation.md`, `designs/2026-04-13-heterogeneous-compute.md`

## Crates

| Crate | Lines | Tests | Description |
|-------|------:|------:|-------------|
| `llvm2-ir` | 5,659 | 203 | Shared machine model: MachInst, registers (GPR/FPR/SIMD), operands, stack slots, calling conventions |
| `llvm2-lower` | 9,937 | 180 | tMIR-to-MachIR instruction selection, Apple AArch64 ABI lowering, legalization |
| `llvm2-opt` | 8,970 | 172 | 11 optimization passes: DCE, constant folding, copy propagation, peephole, CSE, LICM, dominator tree, loop analysis, memory-effects model |
| `llvm2-regalloc` | 7,488 | 134 | Linear scan RA, liveness analysis, interval splitting, spill generation, phi elimination, copy coalescing, rematerialization, call-clobber handling |
| `llvm2-codegen` | 18,604 | 676 | AArch64 binary encoding (integer/memory/FP), Mach-O writer (headers, sections, symbols, relocations, fixups), frame lowering, compact unwind, branch relaxation |
| `llvm2-verify` | 10,325 | 323 | 174 SMT proof functions: lowering proofs, peephole identity proofs, tMIR/AArch64 semantic encoders, memory-effects model |

**Totals:** ~61,000 lines of Rust, 1,688 tests across 89 source files. Plus 4 tMIR stub crates for development.

## Quick Start

```bash
git clone git@github.com:dropbox-ai-prototypes/LLVM2.git
cd LLVM2
cargo build
cargo test  # 1,688 tests, should all pass
```

See `designs/2026-04-12-aarch64-backend.md` for the full backend design (codex-reviewed).

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
| [tMIR](https://github.com/dropbox-ai-prototypes/tMIR) | Input IR (proof-carrying) |
| [tRust](https://github.com/dropbox-ai-prototypes/tRust) | Rust frontend |
| [tSwift](https://github.com/dropbox-ai-prototypes/tSwift) | Swift frontend |
| [tC](https://github.com/dropbox-ai-prototypes/tC) | C verification |
| [z4](https://github.com/dropbox-ai-prototypes/z4) | SMT solver backend |

## License

Apache 2.0 — see [LICENSE](LICENSE).
