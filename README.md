# LLVM2 — Verified Compiler Backend

**Author:** Andrew Yates
**Copyright:** 2026 Dropbox, Inc.
**License:** Apache 2.0

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
|  tMIR -> LIR -> Machine Code        |
|         |           |                |
|         +-----+-----+               |
|               v                      |
|        z4 proves each                |
|        lowering correct              |
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

## Crates

| Crate | Description |
|-------|-------------|
| `llvm2-lower` | Lowering from tMIR to low-level IR |
| `llvm2-opt` | Verified optimizations |
| `llvm2-verify` | SMT encoding and semantic equivalence proofs |
| `llvm2-codegen` | Machine code generation (x86-64, AArch64, RISC-V) |

## Quick Start

```bash
git clone git@github.com:dropbox-ai-prototypes/LLVM2.git
cd LLVM2
cargo build
cargo test
```

## Status

**WIP** — Core lowering and verification infrastructure implemented. Active development.

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
