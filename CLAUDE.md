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
- **z4** (ayates_dbx/z4) - SMT solver for verification

**Goals:**
1. Proven-correct instruction lowering from tMIR to machine code
2. Verified optimizations (peephole, constant folding, dead code elimination)
3. Support x86-64, AArch64, RISC-V targets
4. Universal backend for tRust, tSwift, tC

---

## Architecture

```
                         ┌─────────────────┐
                         │      tMIR       │
                         │   (input IR)    │
                         └────────┬────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────┐
│                           LLVM2                                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────────────┐ │
│  │   llvm2-    │    │   llvm2-    │    │      llvm2-         │ │
│  │   lower     │───►│   opt       │───►│      codegen        │ │
│  │ (tMIR→LIR)  │    │ (optimize)  │    │  (LIR→machine code) │ │
│  └─────────────┘    └──────┬──────┘    └──────────┬──────────┘ │
│                            │                      │             │
│                            ▼                      ▼             │
│                     ┌─────────────┐        ┌─────────────┐     │
│                     │   llvm2-    │        │    z4       │     │
│                     │   verify    │◄───────│   (SMT)     │     │
│                     │ (proofs)    │        │             │     │
│                     └─────────────┘        └─────────────┘     │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
                    ┌─────────────────────────┐
                    │   Verified Machine Code  │
                    │  (x86-64, AArch64, RISC-V)│
                    └─────────────────────────┘
```

**Crates:**
- `llvm2-lower` - tMIR to Low-level IR (LIR) lowering
- `llvm2-opt` - Verified optimizations
- `llvm2-verify` - SMT encoding and proof generation
- `llvm2-codegen` - LIR to machine code emission per target

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
