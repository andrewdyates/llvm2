# LLVM2

Verified compiler backend - formally proven correct code generation from tMIR to machine code.

**Author:** Andrew Yates
**Copyright:** 2026 Dropbox, Inc. | **License:** Apache 2.0
**Repo:** https://github.com/ayates_dbx/LLVM2
**Location:** `~/LLVM2/`
**Director:** LANG

---

## Mission

Build a purpose-built verified codegen for tMIR. Not a fork of LLVM or Cranelift - a focused compiler backend designed specifically for the t* stack.

**Key properties:**

| Property | Benefit |
|----------|---------|
| **Written in tRust** | Self-hosting. The compiler is verified by the system it compiles. |
| **tMIR-native** | Optimizations tuned for tMIR semantics, not general-purpose IR. |
| **Faster compilation** | Focused scope means less complexity than LLVM's 20+ year codebase. |
| **Faster output** | tMIR carries proof information that enables optimizations LLVM can't do. |
| **Proof-preserving** | Verification chain from source to binary is unbroken. |

**Why not LLVM?**
- LLVM is 20M+ LOC of unverified C++
- General-purpose means bloat for our use case
- Can't prove LLVM correct (too large, wrong language)
- LLVM optimizations don't leverage tMIR's proof information

**Why not Cranelift?**
- Cranelift is verification-unaware
- Designed for fast compilation, not verified compilation
- Would need extensive modification to preserve proofs

**The vision:** Every instruction lowering and optimization is proven correct using z4. The binary provably does what the tMIR says.

```
tRust  (Rust + proofs)  ──► tMIR ──┐
tSwift (Swift + proofs) ──► tMIR ──┼──► LLVM2 ──► verified machine code
tC     (C + proofs)     ──► tMIR ──┘
```

---

## Project-Specific Configuration

**Primary languages:** Rust

**Dependencies:**
- **tMIR** (ayates_dbx/tMIR) - Input IR definition
- **z4** (ayates_dbx/z4) - SMT solver for verification (optional, feature-gated)
- **LLVM source** (`~/llvm-project-ref/`) - Reference implementation for algorithm study

**Goals:**
1. Proven-correct instruction lowering from tMIR to machine code
2. Verified optimizations (peephole, constant folding, dead code elimination)
3. Support AArch64 (primary), x86-64, RISC-V targets
4. Universal backend for tRust, tSwift, tC
5. At least as fast as LLVM in compilation speed and output code quality
6. 100% Rust, zero external dependencies for core backend

---

## Architecture

```
tmir-* ──> llvm2-lower (instruction selection, ABI lowering)
                |
                v
           llvm2-ir (MachineFunction, MachInst, operands, blocks, stack)
                |
        +-------+-------+-----------+
        v       v       v           v
   llvm2-opt  llvm2-regalloc  llvm2-codegen  llvm2-verify
   (passes)   (liveness+RA)   (encode+Mach-O)  (z4, optional)
```

**Targets:** AArch64 (primary, fully functional), x86-64 (scaffolding: opcode enum, register definitions, encoding stub)

**Crates:**
- `llvm2-ir` — Shared machine model (10 modules): MachInst, MachFunction, MachBlock, registers (PReg/VReg, GPR/FPR/SIMD), operands, stack slots, condition codes, typed index wrappers (InstId/BlockId/VRegId), calling conventions. AArch64 physical register definitions + x86-64 register/opcode scaffolding.
- `llvm2-lower` — tMIR-to-MachIR lowering (7 modules): SSA tree-pattern instruction selection, Apple AArch64 ABI lowering, type system (I8-I128, F32/F64, B1), legalization, tMIR adapter layer (module/function/type translation, proof extraction).
- `llvm2-opt` — Verified optimizations (14 modules): DCE, constant folding, copy propagation, peephole, CSE, LICM, dominator tree, loop analysis, CFG simplification, proof-guided optimization, memory-effects model, pass manager, pipeline (O0-O3 levels).
- `llvm2-regalloc` — Register allocation (11 modules): linear scan allocation, greedy allocator, liveness analysis, interval splitting, spill generation, phi elimination, copy coalescing, rematerialization, call-clobber handling, spill-slot reuse, machine type adapters.
- `llvm2-verify` — Formal verification (8 modules, 40+ proofs): SMT bitvector expression AST + evaluator, tMIR semantic encoder, AArch64 semantic encoder, NZCV flag proofs, lowering correctness proofs (add/sub/mul/neg, 10 comparison ops, 4 conditional branches), peephole identity proofs (11 rules), verification driver. Mock evaluation (exhaustive for small widths, random sampling for 32/64-bit); z4 integration future.
- `llvm2-codegen` — Code generation (10+ modules): AArch64 binary encoding (integer/memory/FP/branch), x86-64 encoding stub, Mach-O object file writer (header, segments, sections, symbols, string table, relocations, fixups), frame lowering, compact unwind, branch relaxation, code layout, end-to-end compilation pipeline.

**tMIR stubs (in-tree):** `tmir-types`, `tmir-instrs`, `tmir-func`, `tmir-semantics` — development stubs until real tMIR repo is integrated.

**Design docs:**
- `designs/2026-04-12-aarch64-backend.md` — Main backend design (codex-reviewed)
- `designs/2026-04-13-tmir-integration.md` — tMIR adapter layer design
- `designs/2026-04-13-verification-architecture.md` — z4 verification architecture
- `designs/2026-04-13-type-system.md` — Type system design (LIR scalar types, Ptr, no ZST/Never)
- `designs/2026-04-13-macho-format.md` — Mach-O object file format design

## LLVM Source Reference

Reference implementation for algorithm study: `~/llvm-project-ref/`

| Area | LLVM Path |
|------|-----------|
| AArch64 backend | `llvm/lib/Target/AArch64/` |
| Machine IR model | `llvm/include/llvm/CodeGen/MachineInstr.h` |
| Register allocation | `llvm/lib/CodeGen/RegAllocGreedy.cpp` |
| Mach-O emission | `llvm/lib/MC/MachObjectWriter.cpp` |
| AArch64 encoding | `llvm/lib/Target/AArch64/MCTargetDesc/AArch64MCCodeEmitter.cpp` |

---

## Verification Approach

Each lowering rule `tMIR_instr → MachineCode` is verified by:
1. Encoding tMIR instruction semantics as SMT formula
2. Encoding machine instruction semantics as SMT formula
3. Proving semantic equivalence: `∀ inputs: tMIR_semantics(inputs) = Machine_semantics(inputs)`

This follows the approach from:
- **Alive2** (LLVM IR verification)
- **CompCert** (verified C compiler)

---

## The t* Stack

```
┌─────────────────────────────────────────────────────────────────┐
│                        Source Languages                          │
├─────────────────────────────────────────────────────────────────┤
│  tRust           tSwift              tC                         │
│  Rust+proofs     Swift+proofs        C+proofs                   │
│     │               │                   │                       │
│     └───────────────┴───────────────────┘                       │
│                         │                                       │
│                         ▼                                       │
│                      tMIR                                       │
│              (universal IR)                                     │
│                         │                                       │
│            ┌────────────┴────────────┐                          │
│            ▼                         ▼                          │
│     ┌─────────────┐          ┌─────────────┐                    │
│     │     z4      │          │    LLVM2    │                    │
│     │  (verify)   │          │  (codegen)  │                    │
│     └─────────────┘          └─────────────┘                    │
│                                     │                           │
│                                     ▼                           │
│                       Verified Machine Code                     │
└─────────────────────────────────────────────────────────────────┘
```

---

## Development Rules

- **Verify before merge:** No lowering rule lands without proof
- **Semantics first:** Define instruction semantics before implementing codegen
- **Test against baselines:** Use LLVM/Cranelift output as reference for expected behavior
- **Track unsupported:** File issues for instructions not yet verified
