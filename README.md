# LLVM2 — Verified Compiler Backend

**Author:** Andrew Yates
**Copyright:** 2026 Dropbox, Inc.
**License:** Apache 2.0
**Status:** Preview (active development)

Verified codegen from tMIR to machine code. Every instruction lowering is mathematically proven to preserve semantics via z4 SMT.

LLVM2 is the final stage of the t\* verified compilation pipeline. Not a fork of LLVM or Cranelift — a purpose-built backend designed for proof-carrying IR.

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

## Current Capabilities

LLVM2 is in active development targeting AArch64 macOS (Apple Silicon) as the primary backend.

**What works today:**
- Full AArch64 instruction encoding (integer, memory, FP/SIMD formats)
- Mach-O object file emission with relocations, symbols, fixups, and compact unwind
- Instruction selection from tMIR covering arithmetic, comparisons, branches, calls, memory ops
- Apple AArch64 ABI lowering (integer/FP arg classification, callee-saved registers)
- Linear scan register allocation with interval splitting, spill generation, and rematerialization
- Frame lowering with prologue/epilogue emission and branch relaxation
- 11 optimization passes: DCE, constant folding, copy propagation, peephole, CSE, LICM, dominator analysis, loop detection, side-effect modeling
- Phi elimination with parallel-copy resolution and copy coalescing
- SMT verification framework (lowering proofs, tMIR/AArch64 semantic encodings)

**Not yet implemented:**
- End-to-end tMIR-to-binary pipeline (adapter layer in progress)
- x86-64 and RISC-V targets
- DWARF debug info (compact unwind only)
- Greedy register allocator (Phase 2 RA)
- Proof-consuming optimizations (NoOverflow, InBounds, NotNull)
- Benchmark suite vs clang -O2

## Crates

| Crate | Lines | Tests | Description |
|-------|------:|------:|-------------|
| `llvm2-ir` | 2,493 | 16 | Shared machine model: MachInst, registers (GPR/FPR/SIMD), operands, stack slots, calling conventions |
| `llvm2-lower` | 3,536 | 54 | tMIR-to-MachIR instruction selection, Apple AArch64 ABI lowering, legalization |
| `llvm2-opt` | 4,377 | 82 | 11 optimization passes: DCE, constant folding, copy propagation, peephole, CSE, LICM, dominator tree, loop analysis, memory-effects model |
| `llvm2-regalloc` | 3,941 | 41 | Linear scan RA, liveness analysis, interval splitting, spill generation, phi elimination, copy coalescing, rematerialization, call-clobber handling |
| `llvm2-codegen` | 9,700 | 234 | AArch64 binary encoding (integer/memory/FP), Mach-O writer (headers, sections, symbols, relocations, fixups), frame lowering, compact unwind, branch relaxation |
| `llvm2-verify` | 1,496 | 31 | SMT encoding framework, lowering proof structure, tMIR and AArch64 semantic encoders |

**Totals:** ~28,500 lines of Rust, 587 tests across 72 source files. Plus 4 tMIR stub crates (~665 lines) for development.

## Quick Start

```bash
git clone git@github.com:dropbox-ai-prototypes/LLVM2.git
cd LLVM2
cargo build
cargo test
```

## Status

**Preview** -- AArch64 macOS backend under active development. 4 design documents in `designs/`. See `designs/2026-04-12-aarch64-backend.md` for full design (codex-reviewed).

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
