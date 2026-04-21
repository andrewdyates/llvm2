# llvm2-llvm-import

**Test infrastructure only. Not a product input path.**

LLVM2's only product input is tMIR (`rustc MIR → tMIR → LLVM2 → machine code`,
or equivalent for tSwift / tC). This crate is a minimal LLVM IR text (`.ll`)
→ tMIR importer used solely to:

- Feed the llvm/llvm-test-suite SingleSource correctness corpus through the
  LLVM2 codegen pipeline (WS2, issue #439), so we can cite a differential
  pass-rate against programs LLVM2 did not generate itself.
- Bridge csmith / yarpgen random-C output into tMIR so WS3 differential
  fuzzing (#436) exercises shapes humans wouldn't hand-write.

No production compile ever routes LLVM IR through LLVM2. This crate exists
to generate tMIR test inputs from an external corpus; the codegen pipeline
underneath is identical to the one rustc will feed via `rustc_codegen_llvm2`.

## Why not reuse the `llvm-ir` crate?

The `llvm-ir` crate tops out at LLVM 16 support. The host's system
`clang` reports version 21.1.8, and our reference tree
(`~/llvm-project-ref/`) tracks head. Using `llvm-ir` would force us to
pin an older clang just to get the import step to compile, which
defeats the purpose of proving LLVM2 against modern LLVM output.

Instead, this crate implements a tiny line-oriented parser that
handles precisely the subset of `clang -O0 -S -emit-llvm` output
produced by the SingleSource corpus we care about. Anything outside
that subset returns `Error::Unsupported(String)` so the driver can
classify the program as `unsupported` — not a crash, not a miscompile.

## Supported subset

Per issue #439's acceptance criteria the importer handles:

- **Integer types:** `i1`, `i8`, `i16`, `i32`, `i64`, `i128`.
- **Floating-point types:** `float` (→ tMIR `F32`), `double`
  (→ tMIR `F64`). `half` / `bfloat` / `fp128` / `x86_fp80` /
  `ppc_fp128` stay `Unsupported` — tMIR has no matching type.
- **Arithmetic:** `add`, `sub`, `mul`, `and`, `or`, `xor`, `shl`,
  `lshr`, `ashr`, `sdiv`, `udiv`, `srem`, `urem` (with `nsw`/`nuw`
  flags silently dropped).
- **Floating-point arithmetic:** `fadd`, `fsub`, `fmul`, `fdiv`,
  `frem`, `fneg`. Fast-math flags (`fast`, `nnan`, `ninf`, `nsz`,
  `arcp`, `contract`, `reassoc`, `afn`) are accepted and silently
  dropped — LLVM2 does not yet model fast-math semantics.
- **Compares:** `icmp` for all 10 integer predicates; `fcmp` for all
  16 FP predicates. The 12 ordered/unordered predicates map 1:1 to
  `tmir::FCmpOp`; `ord` is synthesized as `FCmp::OEq(a,a) AND
  FCmp::OEq(b,b)` (both operands non-NaN); `uno` as `FCmp::UNe(a,a)
  OR FCmp::UNe(b,b)` (either NaN); `true` and `false` fold to an
  `Inst::Const { ty: Bool, .. }`.
- **Casts:** `sext`, `zext`, `trunc`, `bitcast`, `ptrtoint`,
  `inttoptr`; FP casts `sitofp`, `uitofp`, `fptosi`, `fptoui`,
  `fpext`, `fptrunc`.
- **FP literals:** decimal (`1.5`, `-3.14`, `1.000000e+00`), hex bit-
  pattern (`0x3FF8000000000000`), and the specials `inf` / `-inf` /
  `nan`. Extended-precision tags (`0xK...` for x86_fp80, `0xL...` for
  ppc_fp128, `0xM...` for f128, `0xH...` for f16, `0xR...` for bf16)
  are rejected as `Unsupported` because tMIR only models f32/f64.
- **Memory:** `alloca`, `load`, `store` (non-volatile).
- **Control flow:** unconditional `br`, conditional `br i1`, `ret`,
  `unreachable`, `switch` (integer selectors `i1`/`i8`/`i16`/`i32`/`i64`;
  case values must be integer literals; block arguments on edges are
  not expressible in textual LLVM IR so `default_args` / case `args`
  are always empty — the codegen pipeline picks linear-scan / BST /
  jump-table per #323).
- **Calls:** direct calls to named functions. Declarations from
  `declare` lines are honored; unseen callees synthesize a conservative
  vararg signature so the module still round-trips.
- **Select:** `select i1 <cond>, <ty> <t>, <ty> <f>`.
- **Globals:** private / internal `[N x i8]` string constants
  (`c"..."`). Other global forms are `Unsupported`.

## Explicitly unsupported (all return `Error::Unsupported`)

- Vector or aggregate (struct) types in any instruction.
- Half / bfloat / extended-precision FP (`half`, `bfloat`, `fp128`,
  `x86_fp80`, `ppc_fp128`) and their hex-bit-pattern literals
  (`0xK...`, `0xL...`, `0xM...`, `0xH...`, `0xR...`) — tMIR only
  has `F32` / `F64`.
- `phi` nodes (tMIR uses block parameters; a future revision will
  add an SSA-deconstruction pass — see "Expansion plan" below).
- `invoke`, `landingpad`, `resume`.
- Atomics (`atomicrmw`, `cmpxchg`, `fence`) and volatile memory.
- Inline module assembly.
- `getelementptr` on global string heads (needs global-address
  materialization — tracked as the first expansion step).
- Indirect calls (`call <ty> %fp(...)`).
- Named struct types (`%struct.X = type ...`).

Every rejection is a *typed* no, not a silent stub: the driver records
`status: unsupported` with the exact reason string so future work can
be targeted.

## Expansion plan

Ordered roughly by payoff per LOC in this importer:

1. **Global-address materialization.** Pattern `getelementptr inbounds
   [N x i8], ptr @.str, i64 0, i64 0` -> tMIR constant pointer into
   the module global. Unlocks any program using `printf("literal")`.
2. **SSA-deconstruction pass.** Convert LLVM `phi` nodes into tMIR
   block parameters + edge-specific `br`/`CondBr` arguments. Unlocks
   almost every loop in the corpus.
3. ~~**Floating-point types + ops.**~~ *Landed.* `float` / `double`
   map to `Ty::F32` / `Ty::F64`; `fadd` / `fsub` / `fmul` / `fdiv` /
   `frem` lower to `BinOp::F*`; `fneg` to `UnOp::FNeg`; 12 `fcmp`
   predicates map to `FCmpOp` with `ord` / `uno` synthesized via
   self-comparisons and `true` / `false` folded to `Bool` constants;
   all six FP casts (`sitofp` / `uitofp` / `fptosi` / `fptoui` /
   `fpext` / `fptrunc`) round-trip.
4. ~~**Switch lowering.**~~ *Landed.* `switch` -> `Inst::Switch` with
   empty edge args; codegen picks linear-scan / BST / jump-table per #323.
5. **Named struct types + `getelementptr` on structs.**

## Binary

`cargo build -p llvm2-llvm-import --release` produces
`llvm2-ws2-import`, the driver helper consumed by
scripts/run_llvm_test_suite.sh. Contract:

```
llvm2-ws2-import <input.ll> <output.o>
```

Exit 0 on success; exit 1 with stderr starting `unsupported: <reason>`
when the importer or codegen pipeline rejects a construct. Any other
stderr is a hard failure (parse error / I/O) which the driver logs as
`crash`.

## Corpus

The initial 5-program corpus lives in
`scripts/run_llvm_test_suite.sh::CORPUS`. Programs are sourced from
`~/llvm-test-suite-ref/SingleSource/UnitTests/`. See the script for
the exact list and the rationale ("smallest N integer programs that
exercise distinct subset features").
