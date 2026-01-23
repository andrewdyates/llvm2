# tCrane

Trusted Cranelift - verified compiler backend in Rust.

## Motivation

Compiler bugs cause silent miscompilation: code that compiles but behaves incorrectly. Traditional testing can miss these bugs because they often appear only with specific input patterns or optimization levels.

tCrane eliminates this class of bugs through formal verification. Every instruction lowering is mathematically proven to preserve semantics.

## Goal

A formally verified compiler backend compatible with Cranelift IR (CLIF format) where:
- Every lowering rule has an SMT proof of correctness
- Optimizations are verified to preserve semantics
- Generated machine code is provably equivalent to the IR

## Architecture

```
CLIF IR --> tcrane-ir --> tcrane-codegen --> Machine Code
                |              |
                v              v
           tcrane-verify <-- z4 (SMT)
```

## Crates

| Crate | Description |
|-------|-------------|
| `tcrane-ir` | Cranelift IR types and semantics |
| `tcrane-verify` | SMT encoding and verification |
| `tcrane-codegen` | Verified code generation |
| `tcrane-frontend` | CLIF parser and builder |

## Setup

```bash
git clone https://github.com/ayates_dbx/tCrane
cd tCrane
cargo build
```

## Usage

```rust
use tcrane_frontend::FunctionBuilder;
use tcrane_ir::function::Signature;
use tcrane_ir::types::Type;

// Build a simple function
let sig = Signature {
    params: vec![Type::I64, Type::I64],
    returns: vec![Type::I64],
};
let mut builder = FunctionBuilder::new("add", sig);
let func = builder.build();
```

## License

Apache-2.0
