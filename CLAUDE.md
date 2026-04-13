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

**Crates:**
- `llvm2-ir` - Shared machine model: MachInst, registers (GPR/FPR/SIMD), operands, stack slots, calling conventions, AArch64 physical register definitions
- `llvm2-lower` - tMIR-to-MachIR instruction selection (SSA tree-pattern matching), Apple AArch64 ABI lowering, legalization
- `llvm2-opt` - 11 optimization passes: DCE, constant folding, copy propagation, peephole, CSE, LICM, dominator tree, loop analysis, memory-effects model, pass manager, pipeline
- `llvm2-regalloc` - Linear scan register allocation, liveness analysis, interval splitting, spill generation, phi elimination, copy coalescing, rematerialization, call-clobber handling, spill-slot reuse
- `llvm2-verify` - SMT encoding framework, lowering proof structure, tMIR and AArch64 semantic encoders (optional, z4)
- `llvm2-codegen` - AArch64 binary encoding (integer/memory/FP), Mach-O writer (headers, sections, symbols, relocations, fixups), frame lowering, compact unwind, branch relaxation, code layout

**tMIR stubs (in-tree):** `tmir-types`, `tmir-instrs`, `tmir-func`, `tmir-semantics` -- development stubs until real tMIR repo is integrated.

**Design docs:**
- `designs/2026-04-12-aarch64-backend.md` - Main backend design (codex-reviewed)
- `designs/2026-04-13-tmir-integration.md` - tMIR adapter layer design
- `designs/2026-04-13-verification-architecture.md` - z4 verification architecture

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
