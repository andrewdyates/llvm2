# LLVM2 Type System Design

Author: Andrew Yates <ayates@dropbox.com>
Date: 2026-04-13
Status: Implemented

## Overview

LLVM2 uses a minimal scalar type system designed for machine code generation from tMIR. Types exist at two levels: the lowering IR (`llvm2-lower::Type`) used during instruction selection, and the machine IR (`llvm2-ir::Type`) used in function signatures and stack slots after lowering.

## Type Enum

### Lowering IR Types (`llvm2-lower::Type`)

```rust
pub enum Type {
    I8,    // 8-bit integer
    I16,   // 16-bit integer
    I32,   // 32-bit integer
    I64,   // 64-bit integer
    I128,  // 128-bit integer
    F32,   // 32-bit IEEE 754 float
    F64,   // 64-bit IEEE 754 float
    B1,    // Boolean (1-bit semantic width, 1-byte storage)
}
```

### Machine IR Types (`llvm2-ir::Type`)

```rust
pub enum Type {
    I8, I16, I32, I64, I128,
    F32, F64,
    B1,
    Ptr,  // Pointer-sized integer (8 bytes on 64-bit targets)
}
```

The machine IR adds `Ptr` for pointer-width values in function signatures and stack slot sizing.

### Size and Alignment

| Type | bits() | bytes() | align() |
|------|--------|---------|---------|
| B1   | 1      | 1       | 1       |
| I8   | 8      | 1       | 1       |
| I16  | 16     | 2       | 2       |
| I32  | 32     | 4       | 4       |
| I64  | 64     | 8       | 8       |
| I128 | 128    | 16      | 16      |
| F32  | 32     | 4       | 4       |
| F64  | 64     | 8       | 8       |
| Ptr  | 64     | 8       | 8       |

Note: `B1.bits()` returns 1 (semantic width), but `B1.bytes()` returns 1 (storage size). This distinction matters for instruction selection -- comparisons produce B1 results, but registers always hold at least a byte.

## Pointer Types

`Ptr` is an untyped, address-width integer. It is 8 bytes on all supported 64-bit targets (AArch64, x86-64, RISC-V 64).

**Design rationale:** LLVM2 does not need typed pointers at the machine IR level. Type safety of pointer operations (what you can dereference, aliasing rules, lifetime validity) is established by tMIR's proof annotations before code reaches the backend. By the time tMIR is lowered to machine IR, pointer operations are just integer arithmetic on addresses.

This is a deliberate contrast with LLVM, which historically maintained typed pointers (`i32*`, `%struct.Foo*`) until the opaque pointer migration (LLVM 15+). LLVM needed typed pointers because it performed high-level alias analysis and type-based optimizations in the backend. LLVM2 does not need this -- tMIR's proof system provides stronger guarantees than type-based heuristics.

`Ptr` appears in:
- Function signatures (argument and return types)
- Stack slot declarations
- Frame pointer operations

`Ptr` does NOT appear in instruction operands or the lowering IR's `Type` enum -- at that level, pointers are simply `I64`.

## Zero-Sized Types (ZST)

LLVM2 does not support zero-sized types. All types have `bytes() >= 1`.

**Rationale:** ZSTs are a high-level language concept (Rust's `()`, `PhantomData<T>`, zero-variant enums). They exist to carry type-level information that affects compilation but occupy no runtime storage. By the time code reaches tMIR, ZSTs have been erased:

1. ZST function arguments are removed from the calling convention
2. ZST struct fields occupy no space in the layout
3. ZST local variables are not allocated stack slots
4. Operations on ZSTs (moves, copies) are no-ops that produce no instructions

The tMIR-to-LIR adapter in `llvm2-lower::adapter` handles this: ZST values in tMIR are never translated to LIR operands. If a tMIR function returns a ZST, the lowered function returns void.

## Never Type

There is no `Never` type in LLVM2's type system. The never type (`!` in Rust, `Never` in Swift) represents computations that do not complete -- diverging functions, infinite loops, `unreachable` intrinsics.

**Mapping in LLVM2:** tMIR's `Unreachable` instruction maps to a trap instruction in machine code. There is no need for a type to represent "this value will never exist" because:

1. Unreachable code paths are eliminated by tMIR's optimizer before reaching LLVM2
2. Any remaining `Unreachable` instructions are lowered to `BRK #1` (AArch64) / `UD2` (x86-64) trap instructions
3. Dead code elimination in `llvm2-opt` removes instructions after unconditional traps

The instruction selector in `llvm2-lower::isel` handles `Unreachable` as a special case: it emits a trap instruction and marks the rest of the block as dead. No type is needed because no value is produced.

## Vector Types (Future)

The current type system does not include vector types. AArch64 NEON/SVE vectors (Vec128, Vec256) will be added when SIMD instruction lowering is implemented. The anticipated additions:

- `V128` -- 128-bit NEON vector (maps to Q registers)
- Potentially `V64` for D-register NEON operations

Vector element types will be inferred from the instruction opcode rather than encoded in the type, following AArch64's convention where the same Q register can hold 4xF32, 2xF64, 16xI8, etc.

## Source Files

- `crates/llvm2-lower/src/types.rs` -- Lowering IR `Type` enum
- `crates/llvm2-ir/src/function.rs` -- Machine IR `Type` enum
- `crates/llvm2-lower/src/adapter.rs` -- tMIR-to-LIR type translation
