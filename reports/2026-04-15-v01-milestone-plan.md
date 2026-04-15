# LLVM2 v0.1 Milestone Plan: "First Real Program"

**Date:** 2026-04-15
**Author:** Researcher (W27-R1)
**Epic:** Part of #24

---

## 1. Definition of Done: What v0.1 Means

**v0.1** is the point where LLVM2 can compile a non-trivial single-file Rust
program (equivalent tMIR) through the complete pipeline and produce a correct,
running AArch64 macOS binary. The canonical target program is:

```rust
// Fibonacci (iterative) with function calls and stack allocation
fn fibonacci(n: i32) -> i32 {
    if n <= 1 { return n; }
    let mut a = 0;
    let mut b = 1;
    let mut i = 2;
    while i <= n {
        let tmp = a + b;
        a = b;
        b = tmp;
        i += 1;
    }
    b
}

fn main() -> i32 {
    fibonacci(10) // returns 55
}
```

**Stretch goal:** A program that sorts an array (requires pointer arithmetic,
memory loads/stores, and an inner comparison function call):

```rust
fn insertion_sort(arr: *mut i32, len: i32) {
    let mut i = 1;
    while i < len {
        let key = *arr.offset(i);
        let mut j = i - 1;
        while j >= 0 && *arr.offset(j) > key {
            *arr.offset(j + 1) = *arr.offset(j);
            j -= 1;
        }
        *arr.offset(j + 1) = key;
        i += 1;
    }
}
```

### v0.1 Acceptance Criteria

- [ ] CLI (`llvm2 input.tmir -o output.o`) compiles a multi-function tMIR
      module containing `fibonacci` + `main` into a single linkable `.o`
- [ ] Linking with `cc driver.c output.o -o program` succeeds
- [ ] Running `./program` produces correct output (exit code 55 or prints "55")
- [ ] At least 3 non-trivial tMIR JSON test fixtures compile and run correctly
- [ ] Compilation at O0 and O2 both produce correct output
- [ ] All existing tests continue to pass

---

## 2. Current Capabilities Assessment

### What LLVM2 can do today (verified by E2E tests)

| Capability | Evidence |
|-----------|---------|
| Single-function compilation via tMIR | `e2e_aarch64_link.rs`: add, const, sub |
| Single-function compilation via IR | `e2e_run.rs`: add, sub, const, max, factorial |
| Conditional branches | `build_max_function()` in e2e_run.rs |
| Loops with MUL | `build_factorial_function()` in e2e_run.rs (iterative factorial) |
| Multiple separate .o files linked together | `test_e2e_multiple_functions` |
| Frame lowering (prologue/epilogue) | `test_full_pipeline_frame_lowering_encoding_gaps` |
| Mach-O output accepted by macOS linker | All link tests pass |
| Integer arithmetic (ADD, SUB, MUL, SDIV, UDIV) | ISel + encoder |
| Logical ops (AND, OR, XOR, shifts) | ISel + encoder |
| FP arithmetic (FADD, FSUB, FMUL, FDIV, conversions) | ISel + encoder |
| Comparisons + conditional select (CSEL, CSET) | ISel + encoder |
| Memory loads/stores (LDR, STR, byte/half variants) | ISel + encoder |
| Atomics (LDAR, STLR, LDADD, CAS, etc.) | ISel + encoder |
| Function calls (BL, BLR) with ABI | ISel + encoder |
| Stack allocation (StackAddr) | ISel + encoder |
| Global symbol references (ADRP+ADD) | ISel + encoder |
| Linear scan + greedy register allocation | llvm2-regalloc |
| 26 optimization passes (DCE, const fold, copy prop, peephole, CSE, LICM, etc.) | llvm2-opt |
| CLI with JSON tMIR input | llvm2-cli reads serde_json Module |
| 168 AArch64 opcodes defined and encodable | llvm2-ir + llvm2-codegen |
| ~5,200 tests, ~1,800 proof tests | All crates |

### What the E2E tests do NOT cover

The E2E tests in `e2e_run.rs` build `MachFunction` structs directly with
hardcoded physical registers and manual branch offsets. They **bypass** the
tMIR adapter, ISel, register allocation, and frame lowering. Only the
`e2e_aarch64_link.rs` tests use the full tMIR-to-binary pipeline, and those
only test trivial single-block functions (add, const, sub).

**Critical finding:** No existing E2E test compiles a multi-block tMIR function
(with branches, loops, or function calls) through the **full** pipeline
(tMIR adapter -> ISel -> regalloc -> frame lowering -> encoding -> Mach-O).

---

## 3. Gap Analysis: What's Missing for v0.1

### P0 Blockers (Must fix -- cannot compile any real program without these)

#### 3.1 Multi-function module compilation is broken

**File:** `crates/llvm2-codegen/src/compiler.rs`, line 247

The `compile()` method loops over functions but overwrites `last_obj_bytes`
each iteration:
```rust
for (lir_func, _proof_ctx) in &lir_functions {
    let obj_bytes = pipeline.compile_function(lir_func)?;
    last_obj_bytes = obj_bytes; // BUG: discards all previous functions!
}
```

**Impact:** Any module with 2+ functions produces an object file containing
only the last function. Calling the first function from the second will fail
at link time ("undefined symbol").

**Fix:** Concatenate all function code into a single `__text` section with
multiple symbols at the correct offsets. The `MachOWriter` already supports
`add_symbol(name, section, offset, external)`, so this is a matter of
accumulating code bytes and emitting per-function symbols.

**Estimated effort:** 1 wave (low -- architectural change is localized)

#### 3.2 No full-pipeline E2E test for multi-block tMIR programs

**Impact:** Without this test, we have no evidence that the tMIR adapter ->
ISel -> regalloc -> frame lowering -> encoding pipeline works for programs
with branches, loops, or function calls. The factorial E2E test works only
because it manually constructs `MachFunction` with pre-allocated registers.

**Fix:** Write tMIR test programs (as JSON fixtures or inline `tmir_func::Module`
construction) that exercise:
- Control flow (if/else)
- Loops (while)
- Function calls between functions in the same module
- Stack allocation for local variables

**Estimated effort:** 1 wave

#### 3.3 BL (function call) relocation not emitted in Mach-O

When one function calls another within the same object file, the `BL`
instruction needs a relocation entry so the linker can fix up the offset.
Currently, `emit_macho()` in `pipeline.rs` adds only the function symbol --
no relocations for BL instructions. Cross-function calls within the same `.o`
will encode the wrong branch offset.

**Fix:** During encoding, track BL instructions that reference other symbols.
Emit `ARM64_RELOC_BRANCH26` relocations in the Mach-O output. The relocation
infrastructure already exists (`Relocation::branch26()` in `macho/reloc.rs`).

**Estimated effort:** 1-2 waves (moderate -- requires threading symbol info
through the encoding pipeline)

### P1 Blockers (Required for a convincing v0.1 demo)

#### 3.4 No tMIR JSON test fixtures for the CLI

Issue #237 tracks this. The CLI reads JSON but there are no test fixture files.
To demonstrate v0.1, we need hand-written JSON tMIR programs that exercise the
full pipeline. The existing `build_tmir_add_module()` in the link tests shows
the pattern -- we need the same as JSON files.

**Estimated effort:** 1 wave

#### 3.5 Stack slot allocation through tMIR adapter

The tMIR `Alloc` instruction maps to stack allocation, but the full-pipeline
test coverage is thin. For programs with local variables (like `let mut a = 0`),
the tMIR -> ISel path must correctly allocate stack slots and generate
`StackAddr` + `LDR`/`STR` sequences.

**Fix:** Verify and fix the `Alloc -> StackAddr -> LDR/STR` path with focused
integration tests.

**Estimated effort:** 1 wave

#### 3.6 Missing ABI: large struct passing, HFA

Issue #140 tracks missing ABI features. For v0.1, we need at minimum:
- Structs <= 16 bytes passed in registers (partially implemented)
- Structs > 16 bytes passed by reference (not implemented)

This is not strictly needed for the fibonacci demo but is needed for the
insertion_sort stretch goal and any program passing structs.

**Estimated effort:** 1-2 waves

### P2 Nice-to-haves (Not blocking but improve v0.1 quality)

#### 3.7 Constant pool / read-only data section

Programs that use large constants (e.g., lookup tables, string literals) need
a `__const` or `__cstring` section. The Mach-O writer has `add_data_section()`
but the pipeline does not emit constant pools.

**Estimated effort:** 2 waves

#### 3.8 tMIR JSON wire format reader module

Issue #237. A proper `tmir_reader.rs` module with validation, error messages,
and round-trip tests. The CLI already uses `serde_json::from_slice` directly,
so this is a quality improvement.

**Estimated effort:** 1 wave

#### 3.9 Real tMIR integration (replacing stubs)

Issue #227. Connecting to the real tMIR repo. This is the ultimate goal but
is blocked on the tMIR repo being ready. The v0.1 milestone should be
achievable with the current stubs.

**Estimated effort:** 2-3 waves (when tMIR repo is ready)

#### 3.10 DWARF debug info for the compiled program

The DWARF modules exist (`dwarf_info.rs`, `dwarf_cfi.rs`) but are not
automatically emitted for all compiled functions. Good for debugging the
v0.1 demo but not strictly required.

**Estimated effort:** 1 wave

---

## 4. Ranked Feature List

| Rank | Feature | Priority | Est. Waves | Blocking? | Issue |
|------|---------|----------|-----------|-----------|-------|
| 1 | Multi-function module compilation | P0 | 1 | YES | NEW |
| 2 | BL relocation emission | P0 | 1-2 | YES | NEW |
| 3 | Full-pipeline E2E test for multi-block tMIR | P0 | 1 | YES | NEW |
| 4 | tMIR JSON test fixtures | P1 | 1 | for CLI demo | #237 |
| 5 | Stack allocation through full pipeline | P1 | 1 | for local vars | NEW |
| 6 | Large struct ABI | P1 | 1-2 | for struct programs | #140 |
| 7 | Constant pool emission | P2 | 2 | for large constants | NEW |
| 8 | tMIR reader module | P2 | 1 | for quality | #237 |
| 9 | Real tMIR integration | P2 | 2-3 | for production | #227 |
| 10 | DWARF debug auto-emission | P2 | 1 | for debugging | NEW |

---

## 5. Recommended Wave-by-Wave Execution Plan

### Wave 28: Multi-function module compilation (P0 blocker)

**Goal:** Fix `Compiler::compile()` to produce a single `.o` with multiple
function symbols.

- Fix `compiler.rs` to accumulate code bytes and emit per-function symbols
- Fix `Pipeline::compile_function()` to return `(code_bytes, func_name)` tuple
  instead of complete Mach-O bytes
- Add a `Pipeline::compile_module()` method that takes all functions
- Write E2E test: two-function module (foo calls bar) compiles and links

**Acceptance:** `test_e2e_multi_function_module_link_and_run` passes.

### Wave 29: BL relocation emission (P0 blocker)

**Goal:** Cross-function calls within a `.o` file work correctly.

- During encoding, detect BL instructions with symbol operands
- Emit `ARM64_RELOC_BRANCH26` relocation entries
- Wire relocation tracking from ISel (where function names are known) through
  encoding to Mach-O emission
- Test: `main()` calls `fibonacci()` within the same `.o`

**Acceptance:** Two-function program with cross-function BL links and runs.

### Wave 30: Full-pipeline E2E for multi-block tMIR (P0 blocker)

**Goal:** Prove the tMIR -> ISel -> regalloc -> frame -> encode pipeline works
for non-trivial programs.

- Write `fibonacci` as `tmir_func::Module` (programmatic construction)
- Write `if_else` function as tMIR module
- Compile through `Compiler::compile()`, link with C driver, run
- Fix any bugs discovered (expect ISel edge cases with regalloc interaction)

**Acceptance:** Fibonacci(10) = 55 through full tMIR pipeline.

### Wave 31: Stack allocation and local variables (P1)

**Goal:** Programs with local mutable variables work through full pipeline.

- Test `Alloc` -> `StackAddr` -> `Store` -> `Load` path end-to-end
- Verify frame lowering correctly handles stack slots from ISel
- Fix any mismatches between ISel stack slot numbering and frame lowering

**Acceptance:** Program with 3+ local variables compiles and runs correctly.

### Wave 32: tMIR JSON fixtures and CLI demo (P1)

**Goal:** The `llvm2` CLI can compile hand-written JSON programs.

- Write 4+ JSON tMIR programs in `tests/tmir_fixtures/`:
  - `add_i32.json` (trivial)
  - `fibonacci.json` (loops + calls)
  - `control_flow.json` (if/else + comparisons)
  - `pointer_arithmetic.json` (loads + stores + GEP)
- Add integration test: `llvm2 fibonacci.json -o fib.o && cc driver.c fib.o -o fib && ./fib`
- Add round-trip test (serialize -> deserialize -> compile)

**Acceptance:** `llvm2 fibonacci.json -o fib.o` produces correct binary.

### Wave 33: Polish and announce v0.1 (P1)

**Goal:** Clean up, tag release, update README.

- Run full test suite, fix any regressions
- Update README with "Getting Started" showing CLI usage
- Tag `v0.1.0` release on GitHub
- File follow-up issues for P2 items

**Acceptance:** `git tag v0.1.0` with passing CI-equivalent test suite.

---

## 6. Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Regalloc bugs on multi-block tMIR programs | HIGH | Blocks v0.1 | Wave 30 is specifically designed to flush these out early |
| Branch resolution breaks with multiple functions | MEDIUM | Blocks v0.1 | Wave 28-29 address this explicitly |
| ISel misses edge cases in tMIR adapter | MEDIUM | Delays v0.1 | Adapter has 82 tests but no multi-block integration tests |
| Frame lowering stack slot conflicts | LOW | Delays v0.1 | Wave 31 tests this specifically |
| tMIR stubs diverge from real tMIR API | LOW | Delays real integration | v0.1 uses stubs; real integration is Wave 33+ |

---

## 7. What v0.1 Does NOT Include

These are explicitly deferred to post-v0.1:

- **Real tMIR repo integration** (#227) -- stubs are sufficient for v0.1
- **x86-64 target** (#232) -- scaffolding only, not functional
- **RISC-V target** (#209) -- scaffolding only
- **z4 verification integration** (#34) -- mock verification is sufficient
- **Exception handling / LSDA** (#140) -- not needed for simple programs
- **Heterogeneous compute** (GPU/ANE) -- far-future
- **Vectorization** -- optimization, not correctness

---

## 8. Success Metrics

| Metric | Current | v0.1 Target |
|--------|---------|-------------|
| Max functions per .o | 1 | N (unlimited) |
| Non-trivial tMIR programs compiled E2E | 0 | 4+ |
| Cross-function calls tested | 0 | 2+ |
| JSON fixtures | 0 | 4+ |
| CLI demo programs | 0 | 3+ |
| Full-pipeline E2E tests (tMIR->binary) | 3 (trivial) | 10+ (non-trivial) |
