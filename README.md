# LLVM2 — Verified Compiler Backend

> [!WARNING]
> LLVM2 is still under active development. The AArch64 path is the most mature,
> x86-64 is functional but secondary, and RISC-V is still experimental. This is
> not a drop-in replacement for LLVM or Cranelift yet.

**Author:** Andrew Yates
**Copyright:** 2026 Dropbox, Inc.
**License:** Apache 2.0
**Status:** Alpha — working verified compiler backend under active development

Verified codegen from tMIR to machine code. Every instruction lowering is mathematically proven to preserve semantics via z4 SMT.

LLVM2 is the final stage of the t\* verified compilation pipeline. Not a fork of LLVM or Cranelift — a purpose-built backend designed for proof-carrying IR.

> **Honest assessment:** LLVM2 is a real compiler backend for integer-heavy
> programs, with end-to-end code generation, differential testing against clang,
> proof infrastructure, and native object-file emission. It is **not** a
> production compiler yet: aggregate types remain incomplete, optimization above
> O0 still has known regressions, and the non-AArch64 targets are behind the
> primary path. See [Current Status](#current-status) for details.

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

**LLVM2 compiles integer programs to correct native binaries today.** The full
pipeline — instruction selection, register allocation, optimization, binary
encoding, and object-file emission — works end-to-end on the primary path.
Correctness is checked with differential testing against clang and with
triple-oracle validation (`tmir` interpreter + LLVM2 + clang).

### What actually works

- **End-to-end compilation on the primary path** — integer-heavy programs like
  fibonacci, GCD, collatz, factorial, recursive power, and loop-heavy examples
  compile and run correctly.
- **Differential validation** — the repo compares LLVM2-generated binaries
  against clang output and cross-checks results with a direct `tmir`
  interpreter.
- **Real `tmir` integration** — the backend is connected to the real `tmir`
  crate and the CLI supports binary `.tmbc`, debug JSON, and human-readable
  `.tmir` text input modes.
- **AArch64 is the strongest target** — this is the main correctness path today,
  with the most end-to-end confidence.
- **x86-64 is functional** — the repo has a real x86-64 lowering, encoding, and
  Mach-O pipeline, with validation under Rosetta 2.
- **RISC-V is past the pure-scaffolding stage** — there is a real RV64 pipeline
  and ELF emission path, but it is still an experimental backend.
- **Proof infrastructure is real** — the verifier generates proof certificates,
  supports solver-backed paths via the z3/z4 bridge, and remains central to the
  compiler design.
- **Debug and object emission infrastructure exists** — Mach-O, ELF, unwind, and
  DWARF-related code paths are all part of the in-tree backend.
- **Research-track extensions are in-tree** — heterogeneous dispatch planning,
  Metal emission, CoreML/ANE work, and CEGIS-based optimization infrastructure
  already exist in code.

### What does NOT work yet

- **No aggregate types.** Structs, arrays, unions — cannot compile code that uses them
- **Optimization above O0 still has known regressions.** The backend has real
  optimization infrastructure, but higher optimization levels are not yet in the
  same confidence band as O0.
- **AArch64 is still substantially incomplete as a full ISA implementation.**
  There is real coverage, but not anywhere near full production-target breadth.
- **No dynamic linking, no TLS, no exception handling**
- **x86-64 and RISC-V are not at AArch64 maturity.** They are real backends, not
  just placeholders, but they remain secondary and experimental.
- **tMIR bridge is scalar-only.** No vector types, no aggregate passing through the bridge
- **No serious public performance story yet.** The optimization and benchmark
  story is not ready to compete with mature production compilers.

### Near-term roadmap

1. Stabilize the AArch64 path further, especially the `tmir -> object -> link -> run`
   flow at higher optimization levels.
2. Keep pushing solver-backed verification deeper into normal workflows instead
   of relying primarily on fast mock evaluation.
3. Move x86-64 and RISC-V from "real but secondary" toward genuinely usable
   backends, while keeping the proof and audit surface coherent.

### Honest comparison to LLVM

| Dimension | LLVM | LLVM2 today |
|-----------|------|-------------|
| ISA coverage | ~1,500 AArch64 opcodes | Partial AArch64 coverage with experimental secondary targets |
| Optimization passes | 100+ mature passes | Real pipeline present, O0 strongest, O1/O2 still being hardened |
| Targets | AArch64, x86-64, RISC-V, ARM, MIPS, ... | AArch64 (primary), x86-64 (functional), RISC-V (experimental) |
| Can compile C/Rust? | Yes, production quality | Integer programs only |
| Verification | None (trust-based) | Proof-driven architecture with certificates and solver integration |
| Correctness testing | Decades of fuzzing + production use | Differential testing vs clang + triple oracle |
| Maturity | 20+ years, millions of users | Alpha, active development |

### What this project IS

LLVM2 is a **verified compiler backend that produces correct machine code for integer programs**, with a clear path to full-language support. The value is in:
1. **The verification approach** — lowering proofs, certificate generation, and solver-backed verification are first-class concerns
2. **The architecture** — a decomposed multi-crate Rust workspace with explicit lowering, optimization, codegen, verification, GPU, fuzzing, and test layers
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

## Workspace Crates

| Crate | Role |
|-------|------|
| `llvm2-ir` | Shared machine IR, target register models, operands, provenance, tracing |
| `llvm2-dialect` | Dialect framework and cross-dialect conversion infrastructure |
| `llvm2-lower` | `tmir` adapter, instruction selection, ABI lowering, dispatch planning |
| `llvm2-opt` | Optimization passes, pass manager, rewrite and PGO infrastructure |
| `llvm2-verify` | Proof obligations, verification runner, certificates, z3/z4 bridge |
| `llvm2-codegen` | Target pipelines, encoders, object emission, debug/unwind support |
| `llvm2-regalloc` | Register allocation, liveness, spills, coalescing |
| `llvm2-gpu` | GPU/heterogeneous partitioning pipeline work |
| `llvm2-cli` | Command-line frontend for compile, link, format conversion, and proof emission |
| `llvm2-fuzz` | Fuzzing harnesses and generators |
| `llvm2-llvm-import` | Import and translation tooling from LLVM-side inputs |
| `llvm2-test` | Shared test utilities and integration support |

## Installation

```bash
git clone https://github.com/andrewdyates/LLVM2
cd LLVM2
cargo build
cargo test
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
