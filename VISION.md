# LLVM2 Vision: Verified Codegen for tMIR

## The Core Problem

**Verification stops at the compiler backend.**

Traditional verified compilation:
```
Source → (verified) → IR → (UNVERIFIED) → Binary
                            ^^^^^^^^^^^^
                            Trust LLVM/GCC
```

LLVM is 20M+ lines of unverified C++. Compiler bugs cause silent miscompilation - correct source produces incorrect binaries. This breaks the verification chain.

**The whole stack must be verified.** LLVM2 closes the gap.

---

## The Solution: LLVM2

LLVM2 is a purpose-built verified codegen for tMIR. Not a fork of LLVM or Cranelift - a focused compiler backend designed specifically for the t* stack.

```
Verified source (tRust/tSwift/tC)
         │
         ▼
       tMIR (proof-carrying IR)
         │
         ▼
┌─────────────────────────────────────────┐
│              LLVM2                       │
│                                          │
│  tMIR → LIR → Machine Code              │
│         │           │                    │
│         └─────┬─────┘                    │
│               ▼                          │
│        z4 proves each                    │
│        lowering correct                  │
└─────────────────────────────────────────┘
         │
         ▼
   Verified Binary + Certificate
```

---

## Why Not LLVM?

| Problem | Impact |
|---------|--------|
| 20M+ LOC unverified C++ | Too large to verify |
| General-purpose | Bloat for our use case |
| No proof awareness | Can't leverage tMIR proofs |
| Optimizations unverified | "Trust us" for correctness |

LLVM is a remarkable engineering achievement, but it's not verifiable.

## Why Not Cranelift?

| Problem | Impact |
|---------|--------|
| Verification-unaware | Doesn't preserve proofs |
| Different goals | Fast compilation, not verified compilation |
| Extensive modification needed | Easier to build purpose-built |

Cranelift is designed for fast JIT compilation (Wasmtime). We need verified AOT compilation.

---

## Key Properties

| Property | Benefit |
|----------|---------|
| **Written in tRust** | Self-hosting. LLVM2 is verified by the system it compiles. |
| **tMIR-native** | Optimizations tuned for tMIR semantics, not general-purpose IR. |
| **Faster compilation** | Focused scope means less complexity than LLVM's 20+ year codebase. |
| **Faster output** | tMIR carries proof information that enables optimizations LLVM can't do. |
| **Proof-preserving** | Verification chain from source to binary is unbroken. |

---

## Verification Approach

Every lowering `tMIR_instruction → MachineCode` is verified:

1. **Encode semantics** - tMIR instruction as SMT formula
2. **Encode result** - Machine instruction as SMT formula
3. **Prove equivalence** - `∀ inputs: tMIR_semantics(inputs) = Machine_semantics(inputs)`

This follows Alive2 (LLVM IR verification) and CompCert (verified C compiler) approaches, but applied systematically to the entire backend.

---

## The t* Stack

LLVM2 is the final stage of the verified compilation pipeline:

```
┌─────────────────────────────────────────────────────────────┐
│                    Source Languages                          │
├─────────────────────────────────────────────────────────────┤
│  tRust           tSwift              tC                      │
│     │               │                   │                    │
│     └───────────────┴───────────────────┘                    │
│                         │                                    │
│                         ▼                                    │
│                      tMIR                                    │
│          (universal proof-carrying IR)                       │
│                         │                                    │
│            ┌────────────┴────────────┐                       │
│            ▼                         ▼                       │
│     ┌─────────────┐          ┌─────────────┐                 │
│     │  tla2 + z4  │          │    LLVM2    │ ◄── this repo   │
│     │  (verify)   │          │  (codegen)  │                 │
│     └─────────────┘          └─────────────┘                 │
│                                     │                        │
│                                     ▼                        │
│                       Verified Machine Code                  │
│                       (x86-64, AArch64, RISC-V)              │
└─────────────────────────────────────────────────────────────┘
```

---

## Design Reference

The complete t* stack architecture is documented in:

**`tla2/designs/2026-01-28-tmir-trusted-rust.md`**

This design covers:
- Trusted Rust syntax and semantics
- tMIR data structures and proof obligations
- Verification pipeline
- Trust boundaries and information flow
- The role of LLVM2 in the stack

---

## The Guarantee

If LLVM2 produces a binary from verified tMIR:

**The binary provably does what the tMIR says.**

Combined with tRust → tMIR verification:

**The binary provably does what the source specification says.**

No gaps. No trust assumptions. Mathematical proof end-to-end.
