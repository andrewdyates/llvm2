# tMIR Transport Architecture

**Date:** 2026-04-16
**Author:** Andrew Yates <ayates@dropbox.com>
**Status:** Design
**Supersedes:** All prior JSON wire format decisions

## Problem

The t* stack has no functioning data path from source code to verified machine code. tRust produces `trust_types::VerifiableFunction`. LLVM2 consumes `tmir_func::Module`. These are incompatible types with no bridge between them. The current "wire format" is JSON via `serde_json` — a development expedient that is unacceptable for a production compiler system.

Specific problems with the current state:

1. **No producer exists.** tRust does not emit tMIR. The tMIR repo (844 LOC, 0 tests) is a skeleton. LLVM2's stubs (1,714 LOC, 19 tests) are more complete but disconnected from tRust.
2. **JSON wire format.** The CLI reads `serde_json`-serialized tMIR. This is 10-50x slower and larger than binary formats, with no schema enforcement.
3. **No in-process API.** When tRust calls LLVM2 as a library (the primary use case for self-hosting), there should be zero serialization. Today the only path is through JSON.
4. **1,026 unwrap() calls in production code.** 272 in ISel alone. A compiler must not panic.
5. **Placeholder architecture.** The adapter uses `Iadd` as a COPY pseudo-op. Proof certificates are `Vec::new()`. Instruction count is `code_size / 4` ("rough estimate"). The optimization pass count is hardcoded per opt level rather than measured.
6. **Three incompatible type systems.** `trust_types::Ty`, `tmir_types::Ty`, and `llvm2_ir::Type` each represent the same concepts differently with no shared definition.

## Design Principles

Following z4's architecture: **hash-consed typed indices, arena-backed storage, zero serialization for in-process use, text format as a separate tooling concern.**

1. **In-process first.** The primary API is Rust types passed by reference. Serialization is an optional layer for tooling, caching, and distributed compilation — never on the hot path.
2. **Typed indices everywhere.** `TermId(u32)` into a `Vec<TermEntry>` — not `Box<Node>`, not `Arc<Node>`, not `String`. Index-based representations are cache-friendly, trivially serializable, and support zero-copy deserialization.
3. **Single source of truth.** One definition of tMIR types, shared by all producers and consumers. Not stubs, not copies, not "compatible" reimplementations.
4. **Binary format for persistence.** Custom binary format inspired by LLVM bitcode and WASM binary format: LEB128 integers, section-based, streaming-capable, deterministic/canonical encoding, provably correct round-trip.
5. **Text format for humans.** A structured text format (like LLVM `.ll` or WASM `.wat`) for debugging and testing. Not JSON.
6. **Proofs are first-class.** Proof annotations, verification certificates, and provenance metadata are part of the IR — not bolted on as optional fields or empty vecs.

## Architecture

### Layer 0: Shared Types (`tmir` crate)

A single Rust crate defining the tMIR type system, instruction set, and module structure. This crate has **zero dependencies** beyond `core`/`alloc`. It is the contract between all producers and consumers.

```
tmir/
├── src/
│   ├── lib.rs          # Module, Function, Block, re-exports
│   ├── ty.rs           # Ty, IntWidth, FloatWidth, PtrTy, ArrayTy, StructTy, FuncTy
│   ├── inst.rs         # Inst (instruction enum), InstData, Opcode
│   ├── value.rs        # ValueId, BlockId, FuncId, StructId (typed u32 indices)
│   ├── proof.rs        # ProofAnnotation, ProofObligation, ProofCertificate
│   ├── constant.rs     # Constant (Int, Float, Bool, Null, Undef, Aggregate)
│   ├── semantics.rs    # Formal instruction semantics (executable specification)
│   └── display.rs      # Text format printer (human-readable, like LLVM .ll)
```

**Design decisions:**

- **Flat Ty enum with explicit widths.** `Ty::I8`, `Ty::I16`, `Ty::I32`, `Ty::I64`, `Ty::I128`, `Ty::F32`, `Ty::F64`, `Ty::Bool`, `Ty::Ptr`, `Ty::Void`. Not nested enums, not `Int { width: IntWidth, signed: bool }`. Explicit variants are faster to match and impossible to construct with invalid widths.
- **Signed/unsigned at the instruction level, not the type level.** Like LLVM IR: `i32` is just 32 bits. `SDiv` vs `UDiv` carries the signedness. This is cleaner than having both `Int(32)` and `UInt(32)` types.
- **Block-parameter SSA.** Not MIR-style locals with Place projections. Block parameters are the modern standard (MLIR, Cranelift, WASM).
- **ValueId(u32) indexing.** All values are indices into the function's value table. No pointers, no strings, no heap allocation for value references.
- **InstrNode separates instruction from metadata.** `InstrNode { inst: Inst, results: SmallVec<[ValueId; 2]>, proofs: SmallVec<[ProofAnnotation; 1]>, span: Option<SourceSpan> }`. The instruction enum is the pure operation; metadata wraps it.
- **SmallVec for common cases.** Most instructions produce 0-2 results and have 0-1 proof annotations. SmallVec avoids heap allocation for the common case.

### Layer 1: In-Process API (Rust library calls)

**The fast path.** When tRust compiles Rust code through LLVM2, no serialization occurs.

```rust
// In tRust's codegen driver:
use tmir::{Module, Function, Block, Inst, Ty, ValueId};
use llvm2_codegen::Compiler;

// tRust builds tMIR types directly in memory
let module: tmir::Module = bridge::translate(&verifiable_functions);

// LLVM2 receives a &Module — zero copy, zero serialization
let result = compiler.compile(&module)?;
```

The bridge code lives in tRust (`trust-tmir-bridge` crate) because it depends on `trust_types`. It translates `VerifiableFunction` → `tmir::Module` with these key transformations:

- **SSA conversion:** MIR locals + Place projections → block-parameter SSA with ValueIds
- **Signedness splitting:** `BinOp::Div` → `SDiv` or `UDiv` based on operand type
- **Comparison elaboration:** `BinOp::Lt` → `Slt` or `Ult` based on operand signedness
- **Cast elaboration:** `Cast(op, ty)` → explicit `ZExt`/`SExt`/`Trunc`/`FPToSI`/etc.
- **Memory materialization:** `Place` projections → explicit `Load`/`Store`/`GEP` instructions
- **Proof extraction:** `preconditions`/`postconditions`/`Formula` → per-instruction `ProofAnnotation`

### Layer 2: Binary Format (`tmir-bitcode`, `.tmbc`)

For separate compilation, caching, distributed compilation, and the CLI tool.

**Not JSON. Not protobuf. Not capnp. Custom binary format.** Reasons:

- **Zero external dependencies.** The format is defined and parsed by the `tmir` crate itself.
- **Formally verifiable.** The encoder/decoder pair can be proven correct with z4 (round-trip property: `decode(encode(x)) == x`).
- **Optimized for tMIR.** Common patterns (i64 add, i32 compare, branch) get short encodings.
- **Deterministic.** One tMIR module = one byte sequence. Required for reproducible builds and proof caching.
- **Streaming.** Functions can be compiled as they arrive. No need to read the entire module first.
- **Lazy loading.** Function bodies can be skipped until needed (for link-time optimization).

**Format structure (inspired by WASM binary + LLVM bitcode):**

```
┌──────────────────────────────────────┐
│ Magic: b"tMBC" (4 bytes)             │
│ Version: u32-leb128                  │
├──────────────────────────────────────┤
│ Section: Type     (id=1)             │
│   - Deduplicated type table          │
│   - Types referenced by TyIdx(u32)   │
├──────────────────────────────────────┤
│ Section: Struct   (id=2)             │
│   - Struct definitions               │
├──────────────────────────────────────┤
│ Section: Import   (id=3)             │
│   - External function declarations   │
├──────────────────────────────────────┤
│ Section: Function (id=4)             │
│   - Function signatures (type refs)  │
│   - Function names                   │
├──────────────────────────────────────┤
│ Section: Code     (id=5)             │
│   - Function bodies (one per func)   │
│   - Each body: blocks → instructions │
│   - Values are u32 indices           │
│   - Types are TyIdx references       │
├──────────────────────────────────────┤
│ Section: Proof    (id=6)             │
│   - Proof annotations per function   │
│   - Proof certificates (if attached) │
├──────────────────────────────────────┤
│ Section: Debug    (id=7)             │
│   - Source spans, variable names     │
│   - Provenance mappings              │
├──────────────────────────────────────┤
│ Section: Custom   (id=0)             │
│   - Named custom sections            │
│   - Extensibility without version    │
│     bumps                            │
└──────────────────────────────────────┘
```

**Encoding details:**

- **Integers:** LEB128 (unsigned) and SLEB128 (signed). Same as WASM and DWARF.
- **Section headers:** `section_id: u8, payload_size: u32-leb128`. Size enables skipping.
- **Type table:** Deduplicated. Each unique type gets a `TyIdx`. Instructions reference types by index.
- **Instructions:** `opcode: u8` (256 opcodes is plenty) + operands as LEB128 value indices.
- **Strings:** Length-prefixed UTF-8. Deduplicated via string table section.
- **Proof annotations:** Separate section, indexed by function/instruction. Can be stripped without affecting code correctness.

**Size estimate:** A simple `add(i64, i64) -> i64` function:
- JSON: ~400 bytes (current)
- tMIR bitcode: ~30 bytes (magic + version + type section + function section + code section)
- Ratio: **13x smaller**

### Layer 3: Text Format (`tmir-text`, `.tmir`)

Human-readable format for debugging, testing, and documentation. **Not JSON.** A proper IR text format with types, SSA names, and proof annotations.

```
; tMIR text format
module "test"

fn @add(i64, i64) -> i64 {
bb0(%0: i64, %1: i64):
    %2 = add i64 %0, %1       ; #proof: no_overflow
    ret %2
}

fn @factorial(i64) -> i64 {
bb0(%n: i64):
    %cmp = sle i64 %n, 1
    condbr %cmp, bb1(%n), bb2(%n)

bb1(%n1: i64):
    ret 1i64

bb2(%n2: i64):
    %n_minus_1 = sub i64 %n2, 1
    %rec = call @factorial(%n_minus_1)
    %result = mul i64 %n2, %rec
    ret %result
}
```

**Design:** Follows LLVM IR conventions (`%` for values, `@` for globals/functions, `bb` for blocks). This is familiar to anyone who has worked with LLVM IR, MLIR, or Cranelift IR. The format is:
- **Parseable** — a proper grammar, not ad-hoc regex matching
- **Round-trippable** — `parse(print(module)) == module`
- **Useful for tests** — write test cases as `.tmir` text files, not JSON blobs or builder API calls
- **Annotatable** — proof annotations, source spans, and metadata as comments or inline attributes

### Layer 4: JSON (Development/Debug Only)

Retained **only** as `--format=json` flag on the CLI for scripting and quick inspection. Not used by any production code path. Not the default. Implemented via `serde_json` on the tMIR types (which already derive `Serialize`/`Deserialize`).

## Data Flow

### Self-Hosting Path (Primary)

```
tRust (rustc fork)
  │
  ├─ trust-mir-extract ──► VerifiableFunction (in-memory)
  │
  ├─ trust-tmir-bridge ──► tmir::Module (in-memory, zero-copy)
  │
  └─ llvm2_codegen::Compiler::compile(&module) ──► Mach-O bytes
```

**Zero serialization.** tRust links against the `tmir` crate and `llvm2-codegen` as library dependencies. Everything is in-process.

### CLI Path (Tooling)

```
tRust --emit=tmir-bc module.tmbc      # binary output
  or
tRust --emit=tmir-text module.tmir    # text output (debugging)

llvm2 module.tmbc -o module.o          # binary input (fast)
  or
llvm2 module.tmir -o module.o          # text input (debugging)
```

### Distributed Compilation Path (Future)

```
tRust --emit=tmir-bc -o /dev/stdout | distcc-tmir | llvm2 -o module.o
```

The binary format is streaming-capable: functions can be compiled as they arrive over the pipe.

### Proof Certificate Chain

```
tRust
  ├─ Source proofs: TrustDisposition (Trusted/Certified/Failed/Unknown)
  │   from trust-proof-cert (23K LOC)
  │
  ├─ tMIR bridge attaches ProofAnnotation per instruction:
  │   - NoOverflow, InBounds, NotNull, ValidBorrow, ...
  │   - Each annotation references the source proof that justifies it
  │
LLVM2
  ├─ Lowering proofs: z4 SMT verification
  │   - For each ISel rule: ∀ inputs: tMIR_semantics(inputs) = Machine_semantics(inputs)
  │   - ProofCertificate { rule_name, z4_proof_hash, verified: bool }
  │
  └─ Output: Mach-O + ProofBundle
       - ProofBundle chains source proofs → tMIR proofs → lowering proofs
       - Verifiable end-to-end: source specification → binary behavior
```

## What Must Change in LLVM2

### Critical (blocks tRust integration)

1. **Replace stubs with real `tmir` crate.** The stubs at `LLVM2/stubs/` become the `tmir` crate (either in a standalone repo or as crates within tRust). LLVM2 depends on it via git dependency.

2. **Implement binary format reader/writer** in the `tmir` crate. Parser for `.tmbc`, printer for `.tmbc`. Proven round-trip correct with z4.

3. **Implement text format reader/writer** in the `tmir` crate. Parser for `.tmir`, printer for `.tmir`. Proven round-trip correct with z4.

4. **Remove JSON from the hot path.** The CLI should default to binary format input. JSON becomes a `--format=json` flag.

5. **Add a COPY pseudo-instruction.** Stop using `Iadd` with one argument as a copy placeholder. This is a fundamental IR modeling error — the instruction semantics are wrong (add vs copy) and it makes verification harder.

6. **Real instruction count.** Replace `total_code_size / 4` with actual instruction count from the encoder (it knows exactly how many instructions it emitted).

7. **Real optimization pass count.** Count actual passes executed, not hardcoded estimates per opt level.

8. **Real proof certificates.** Replace `Some(Vec::new())` with actual z4 proof results collected during compilation.

### High Priority (code quality)

9. **Audit and fix all 1,026 unwrap() calls in production code.** A compiler must not panic on valid input. Replace with `Result` propagation or `expect()` with diagnostic messages for internal invariants.

10. **Eliminate placeholder patterns.** Every `"for now"` and `"placeholder"` comment should become an issue or a proper implementation.

11. **Unify type systems.** `tmir::Ty` should be the single source of truth. `llvm2_ir::Type` should be a lowered machine type (not duplicating the IR type). The adapter should translate `tmir::Ty` → `llvm2_ir::MachType` once.

### Medium Priority (architecture)

12. **Arena-backed IR storage.** Following z4's TermStore pattern: functions, blocks, and instructions stored in arena-backed Vecs with typed index access. Eliminates scattered heap allocations.

13. **Proof-guided optimization.** The optimizer should read `ProofAnnotation`s to enable/disable transformations. If a value is proven non-null, null checks can be eliminated. If arithmetic is proven non-overflowing, wrapping can be replaced with faster non-checking variants.

14. **Provenance tracking through optimization.** When the optimizer transforms instructions, the provenance chain must be maintained so the final binary can be traced back to source locations.

## Verification of the Format

The `tmir` crate's binary encoder and decoder should be verified:

1. **Round-trip property:** `∀ m: Module, decode(encode(m)) = m` — verified by z4 for small modules, tested exhaustively for all instruction types.

2. **Canonical encoding:** `∀ m: Module, encode(m)` produces a unique byte sequence — no padding, no alignment-dependent output, no hash-map iteration order dependency.

3. **Streaming safety:** Partial reads of a `.tmbc` file never produce silently wrong results — either a complete valid section is returned, or an explicit error.

4. **Size bounds:** `∀ m: Module, |encode(m)| ≤ C * |m|` for a known constant C — the encoding never blows up.

## Migration Plan

### Phase 1: Define tmir crate (this is the P0)
- Extract LLVM2 stubs into a proper `tmir` crate
- Add COPY pseudo-instruction
- Add text format printer/parser
- Add binary format encoder/decoder
- Verify round-trip property
- LLVM2 switches from stubs to `tmir` crate dependency

### Phase 2: Bridge tRust → tMIR
- Implement `trust-tmir-bridge` in tRust repo
- SSA conversion, type mapping, proof extraction
- Test: tRust can compile a simple Rust function and produce a `tmir::Module`

### Phase 3: Connect tRust → LLVM2
- tRust links against `llvm2-codegen` as library dependency
- Compile Rust → tMIR → AArch64 machine code in a single process
- Test: `fn add(a: i64, b: i64) -> i64 { a + b }` compiles through the full stack

### Phase 4: Self-hosting
- tRust compiles LLVM2's source code
- The resulting binary is a verified LLVM2 compiler
- Proof chain: tRust source proofs + LLVM2 lowering proofs = verified compiler binary

## References

- z4 TermStore: `~/z4/crates/z4-core/src/term/mod.rs` — hash-consed DAG with TermId(u32) indices
- LLVM Bitcode: `~/llvm-project-ref/llvm/lib/Bitcode/` — section-based binary format with abbreviations
- WASM Binary: https://webassembly.github.io/spec/core/binary/ — LEB128, sections, streaming
- Cranelift IR: CLIF text format with SSA values and block parameters
- tRust bridge design: `~/tRust/designs/2026-04-14-trust-llvm2-integration.tex`
- tRust machine semantics: `~/tRust/crates/trust-machine-sem/` (5,655 LOC)
- LLVM2 stubs: `/Users/ayates/LLVM2/stubs/` (1,714 LOC, 19 tests)
- Real tMIR repo: `~/tMIR/` (844 LOC, 0 tests)
