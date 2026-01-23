# tCrane - Trusted Cranelift

Verified compiler backend in Rust - formally proven correct code generation.

**Author:** Andrew Yates
**Copyright:** 2026 Dropbox, Inc. | **License:** Apache 2.0
**Repo:** https://github.com/ayates_dbx/tCrane
**Location:** `~/tCrane/`
**Director:** LANG
**Baseline Remote:** https://github.com/bytecodealliance/wasmtime/tree/main/cranelift

---

## Mission

Build a formally verified compiler backend based on Cranelift IR. Every instruction lowering and optimization is proven correct using SMT solving (z4).

**Why this matters:** Compiler bugs cause silent miscompilation - code that compiles but behaves incorrectly. tCrane eliminates this class of bugs through mathematical proof.

---

## Project-Specific Configuration

**Primary languages:** Rust

**Dependencies:**
- **z4** (ayates_dbx/z4) - SMT solver for verification
- **tRust** (reference) - Similar verification approach

**Goals:**
1. Proven-correct instruction lowering from CLIF IR to machine code
2. Verified optimizations (peephole, legalization)
3. Support x86-64, AArch64, RISC-V targets
4. Integration with tRust for end-to-end verified compilation

---

## Architecture

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│ tcrane-frontend │ --> │    tcrane-ir     │ --> │ tcrane-codegen  │
│  (CLIF parser)  │     │ (IR + semantics) │     │ (machine code)  │
└─────────────────┘     └────────┬─────────┘     └────────┬────────┘
                                 │                        │
                                 v                        v
                        ┌────────────────┐       ┌────────────────┐
                        │ tcrane-verify  │ <---- │   z4 (SMT)     │
                        │ (correctness)  │       │   solver       │
                        └────────────────┘       └────────────────┘
```

**Crates:**
- `tcrane-ir` - Cranelift IR types, compatible with CLIF format
- `tcrane-verify` - SMT encoding and verification backend
- `tcrane-codegen` - Verified code generation per target
- `tcrane-frontend` - CLIF parser and builder API

---

## Verification Approach

Each lowering rule `IR_instr -> MachineCode` is verified by:
1. Encoding IR instruction semantics as SMT formula
2. Encoding machine instruction semantics as SMT formula
3. Proving semantic equivalence: `forall inputs: IR_semantics(inputs) = Machine_semantics(inputs)`

This follows the approach from:
- **Alive2** (LLVM IR verification)
- **CompCert** (verified C compiler)

---

## Development Rules

- **Verify before merge:** No lowering rule lands without proof
- **Semantics first:** Define instruction semantics in tcrane-ir before implementing codegen
- **Test against Cranelift:** Use upstream Cranelift as reference for expected behavior
- **Track unsupported:** File issues for instructions not yet verified
