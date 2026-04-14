# ABI Gap Analysis: Exception Handling, C++ Interop, Stack Unwinding

**Date:** 2026-04-14
**Status:** Draft
**Issue:** #140
**Author:** Andrew Yates <ayates@dropbox.com>

---

## Problem

LLVM2's ABI implementation (`llvm2-lower/src/abi.rs`) and frame lowering
(`llvm2-codegen/src/frame.rs`) cover the happy path of the Apple AArch64
DarwinPCS calling convention but lack several features required for interop
with system libraries, C++ code, and correct exception handling. This document
inventories what exists, what is missing, and proposes a priority ordering for
closing the gaps.

---

## 1. Current ABI Status

### 1.1 What Is Implemented

| Feature | Location | Status |
|---------|----------|--------|
| GPR argument passing (X0-X7) | `abi.rs:122-165` | Complete |
| FPR argument passing (V0-V7) | `abi.rs:168-181` | Complete |
| Stack overflow arguments (16-byte aligned) | `abi.rs:133-141` | Complete |
| I128 two-GPR passing | `abi.rs:146-165` | Partial (uses `Indirect` as workaround) |
| Return value classification (GPR/FPR/sret) | `abi.rs:240-285` | Complete for scalars |
| Large aggregate indirect return via X8 | `abi.rs:278-280` | Complete |
| Small aggregate passing (<=16 bytes) | `abi.rs:185-221` | Complete |
| Large aggregate indirect passing (>16 bytes) | `abi.rs:205-216` | Complete |
| Variadic function support (Darwin) | `abi.rs:329-378` | Complete |
| Callee-saved GPRs (X19-X28, FP, LR) | `abi.rs:76-80` | Complete |
| Callee-saved FPRs (V8-V15, lower 64 bits) | `abi.rs:83-86` | Complete |
| Call-clobber set (X0-X18) | `abi.rs:90-96` | Complete |
| X18 reserved (Darwin platform register) | `abi.rs:89` | Complete |
| Frame pointer mandatory (X29) | `frame.rs:279` | Complete |
| Prologue/epilogue generation | `frame.rs:403-518` | Complete |
| Callee-saved pair save/restore (STP/LDP) | `frame.rs:286-339` | Complete |
| Frame index elimination (FP-relative) | `frame.rs:538-603` | Complete |
| Compact unwind encoding | `frame.rs:649-711` | Complete |
| DWARF CFI fallback (dynamic alloca) | `dwarf_cfi.rs` | Complete |
| Red zone optimization (leaf functions) | `frame.rs:354-359` | Complete |
| Compact unwind entries in Mach-O | `unwind.rs` | Complete |

### 1.2 What Is Missing or Incomplete

| Feature | Severity | Notes |
|---------|----------|-------|
| Exception handling (C++ `_Unwind_*`) | P1 | No personality function, no LSDA |
| Stack unwinding for exceptions | P1 | Compact unwind personality=0, LSDA=0 |
| HFA (Homogeneous Floating-point Aggregate) | P2 | Not detected; treated as generic struct |
| HVA (Homogeneous Vector Aggregate) | P2 | Not detected |
| `__cxa_throw` / `__cxa_begin_catch` interop | P1 | No landing pad lowering |
| `setjmp`/`longjmp` interop | P2 | No special handling |
| `_Unwind_Resume` integration | P1 | Not implemented |
| Asynchronous exception safety (signal handlers) | P3 | Not relevant for initial targets |
| Thread-local storage (TLS) ABI | P2 | No `tpidr_el0` handling |
| Tail call optimization | P3 | Not implemented |
| `swiftself`, `swiftasync` attributes | P3 | No Swift ABI extensions |
| PAC (Pointer Authentication) on LR | P2 | No `PACIBSP`/`AUTIBSP` in prologue/epilogue |
| Dynamic stack alignment (>16 bytes) | P3 | No `STACKALLIGN` support |
| Stack probing for large frames | P2 | No guard page probing |
| I128 proper two-register return | P2 | Currently misuses `Indirect` |
| Aggregate return: small struct in registers | P2 | Always indirect via X8 |

---

## 2. Exception Handling

### 2.1 Background: Itanium ABI Exception Handling on Darwin

The standard C++ exception handling model on macOS/Darwin follows the Itanium
ABI (also used by Rust panics when `panic=unwind`). The key components are:

1. **Personality function**: Per-function pointer (`__gxx_personality_v0` for
   C++, `rust_eh_personality` for Rust). Tells the unwinder how to interpret
   the LSDA for this function.

2. **LSDA (Language-Specific Data Area)**: Per-function table describing:
   - Call site ranges (which instruction ranges have active try/catch)
   - Action table entries (which exceptions are caught, cleanup actions)
   - Type table (exception type match info)

3. **Landing pads**: Code addresses where control transfers when an exception
   is caught. The compiler emits `invoke` instead of `call` for calls that
   can throw, with a landing pad block for the exceptional path.

4. **Unwind tables**: Compact unwind or DWARF CFI entries that tell the
   unwinder how to restore the caller's registers and stack pointer.

### 2.2 Current State

LLVM2 emits compact unwind entries with `personality = 0` and `lsda = 0`
(see `unwind.rs:25-28`). This means:

- Stack unwinding works for the **cleanup phase** (restoring callee-saved
  registers and SP) because the compact unwind encoding is correct.
- The **search phase** (finding a catch handler) will always skip these
  functions because there is no personality function to call.
- If an exception propagates through an LLVM2-compiled function, the frame
  will be correctly destroyed (registers restored) but no catch/cleanup
  actions will run.

**Impact**: Any tMIR program that calls C++ code which throws, or any tRust
program with `panic=unwind`, will have undefined behavior when exceptions
propagate through LLVM2-compiled frames. In practice, `std::terminate()` will
be called because the unwinder finds no personality to ask.

### 2.3 Required Changes

#### Phase 1: Basic cleanup actions (P1)

1. **tMIR representation**: tMIR needs `invoke` + landing pad blocks
   (analogous to LLVM IR `invoke`/`landingpad`). The tMIR stubs
   (`tmir-instrs`) need `Instr::Invoke` and `Instr::LandingPad` variants.

2. **Personality function emission**: The compact unwind entry's `personality`
   field must be set to the address of the personality function
   (`__gxx_personality_v0` or `rust_eh_personality`). This requires:
   - A relocation in the compact unwind entry (offset 16, 8 bytes, type
     `ARM64_RELOC_UNSIGNED`) pointing to the personality symbol.
   - The Mach-O writer must add the personality symbol to the symbol table
     as an undefined external.

3. **LSDA generation**: For each function with landing pads, emit an LSDA in
   `__TEXT,__gcc_except_tab` (yes, the section name is `__gcc_except_tab`
   even on Darwin). The LSDA format:
   ```
   u8: LPStart encoding (DW_EH_PE_omit = 0xFF for default)
   u8: TType encoding (DW_EH_PE_absptr or DW_EH_PE_indirect)
   uleb128: TType base offset (if TType != omit)
   u8: call site encoding (DW_EH_PE_uleb128)
   uleb128: call site table length
   [call site entries...]
   [action table entries...]
   [type table entries...]
   ```

4. **Call site table**: Each `invoke` instruction generates an entry:
   ```
   uleb128: start of call site (relative to function start)
   uleb128: length of call site
   uleb128: landing pad offset (relative to function start, 0 = no LP)
   uleb128: action (1-based index into action table, 0 = cleanup only)
   ```

5. **Compact unwind LSDA pointer**: Set the compact unwind entry's `lsda`
   field to point to the function's LSDA. This needs another
   `ARM64_RELOC_UNSIGNED` relocation.

#### Phase 2: Catch handlers and type matching (P2)

6. **Action table**: Encode which exception types each landing pad catches.
   Each action entry is a pair of `sleb128` values:
   ```
   sleb128: type filter (positive = catch, negative = filter, 0 = cleanup)
   sleb128: offset to next action (0 = end of chain)
   ```

7. **Type table**: Array of pointers to `std::type_info` objects (for C++)
   or type metadata (for tRust). Grows backwards from the TType base.

8. **Landing pad lowering**: At each landing pad, the unwinder provides
   the exception object pointer in X0 and the type selector in X1.
   The landing pad code must:
   - Check the selector to determine which catch clause matches
   - Call `__cxa_begin_catch(exception_ptr)` to mark the exception as handled
   - Execute the handler body
   - Call `__cxa_end_catch()` when done
   - Or call `_Unwind_Resume()` to continue unwinding

### 2.4 C++ Interop Requirements

For calling C++ functions that may throw:

| Requirement | Mechanism |
|-------------|-----------|
| Stack frame must be unwindable | Already works (compact unwind is correct) |
| Destructors must run (RAII cleanup) | Requires landing pads with cleanup actions |
| Catch blocks must execute | Requires LSDA with call site + action + type tables |
| Re-throw must work | Requires `_Unwind_Resume` call at landing pad end |
| Exception objects must be freed | `__cxa_end_catch` handles deallocation |

---

## 3. Stack Unwinding

### 3.1 Compact Unwind vs DWARF CFI

LLVM2 already supports both mechanisms:

| Mechanism | When Used | Status |
|-----------|-----------|--------|
| Compact unwind (frame mode) | Standard FP-based frames | Complete |
| Compact unwind (frameless) | Leaf functions without FP | Defined but not emitted |
| DWARF CFI (`__eh_frame`) | Dynamic alloca, non-standard frames | Complete (basic) |
| DWARF CFI + LSDA | Exception handling | **Missing** |

### 3.2 When Each Is Needed

- **Compact unwind alone**: Sufficient for C-like functions without exception
  handling. The current implementation covers this case fully.

- **Compact unwind + LSDA**: Required when the function has catch/cleanup
  actions. The compact unwind entry's `personality` and `lsda` fields must
  be populated. The linker reads `__LD,__compact_unwind` and produces
  `__TEXT,__unwind_info` which includes LSDA references.

- **DWARF CFI fallback**: Required when compact unwind cannot describe the
  frame (dynamic alloca, unusual callee-save patterns). The DWARF CFI
  (`__TEXT,__eh_frame`) provides full register restoration rules.
  Currently emitted for `has_dynamic_alloc` frames.

- **DWARF CFI + LSDA**: The FDE in `__eh_frame` can also carry personality
  and LSDA pointers via the augmentation string (`"zPLR"`). This is the
  fallback when compact unwind uses `UNWIND_ARM64_MODE_DWARF` but the
  function still needs exception handling.

### 3.3 Gaps in DWARF CFI

The current `dwarf_cfi.rs` implementation:
- Emits CIE with augmentation string and register save rules
- Emits FDEs with per-instruction CFI directives
- Does NOT emit personality pointer in the CIE augmentation data
- Does NOT emit LSDA pointer in the FDE augmentation data
- Does NOT emit the `"P"` (personality) or `"L"` (LSDA) augmentation codes

These are required for DWARF-fallback frames that also need exception handling.

---

## 4. Variadic Function Support

### 4.1 Current State: Complete

The variadic function ABI is fully implemented in `abi.rs:329-378`:

- Fixed parameters follow normal classification (GPR X0-X7, FPR V0-V7)
- All variadic arguments are placed on the stack (Apple Darwin convention)
- Variadic floats are on the stack, not in FPR registers
- Stack alignment is 8 bytes per argument slot

This matches the Apple ARM64 Function Calling Conventions specification and
is tested with 8 unit tests covering edge cases (zero fixed, overflow, floats).

### 4.2 Missing: `va_list` / `va_arg` Runtime Support

The ABI lowering for variadic _calls_ is complete, but lowering of variadic
_definitions_ (functions that receive variadic arguments) needs:

- `va_start`: Initialize `va_list` to point to the first variadic argument
  on the stack (offset after the last fixed argument's stack area).
- `va_arg`: Load the next argument from the `va_list` and advance the pointer.
- `va_end`: No-op on AArch64.
- `va_copy`: Memcpy of the `va_list` structure.

On Apple AArch64, `va_list` is simply a `char*` (stack pointer), unlike
standard AAPCS64 which uses a struct with separate GPR/FPR save areas.

**Priority**: P3 (variadic definitions are uncommon in tMIR-generated code).

---

## 5. Aggregate Passing and Returning

### 5.1 Current State

| Case | Status | Notes |
|------|--------|-------|
| Small struct (<=8 bytes) in 1 GPR | Complete | `abi.rs:189-191` |
| Small struct (9-16 bytes) in 2 GPRs | Partial | Uses `Indirect` as proxy |
| Large struct (>16 bytes) indirect | Complete | `abi.rs:205-216` |
| Struct return via X8 (sret) | Complete | `abi.rs:278-280` |
| Small struct return in X0/X1 | **Missing** | Always indirect via X8 |

### 5.2 HFA (Homogeneous Floating-point Aggregate) -- Missing

An HFA is a struct of 1-4 members of the same floating-point type:
```
struct Vec3 { float x, y, z; }  // HFA: 3x F32
struct Complex { double re, im; }  // HFA: 2x F64
```

Per AAPCS64 and Darwin ABI:
- HFA arguments are passed in consecutive FPR registers (V0-V7)
- HFA return values are in V0-V3
- If insufficient FPR registers, the entire HFA goes on the stack

**Current behavior**: HFAs are treated as generic structs and passed in GPRs
or indirectly. This is ABI-incompatible with system libraries and C code.

**Detection algorithm**: When classifying a struct type, check if all fields
are the same float type (F32 or F64) and count <= 4. If so, treat as HFA
and use FPR classification.

### 5.3 HVA (Homogeneous Vector Aggregate) -- Missing

Same as HFA but with SIMD vector types. Not relevant until LLVM2 adds
explicit vector type support. **Priority**: P3.

### 5.4 Small Struct Return in Registers

The current `classify_returns` always uses `Indirect { ptr_reg: X8 }` for
aggregate returns. Per AAPCS64:

- Structs <= 16 bytes should be returned in X0 (<=8 bytes) or X0+X1 (<=16 bytes)
- HFA structs should be returned in V0-V3
- Only structs > 16 bytes (or with non-trivial constructors in C++) should use sret

**Priority**: P2 (correctness issue for any function returning small structs).

---

## 6. Darwin-Specific Requirements

### 6.1 Already Handled

| Requirement | Status |
|-------------|--------|
| X18 reserved | Complete (`abi.rs:89`) |
| Frame pointer mandatory | Complete (`frame.rs:279`) |
| Variadic args on stack | Complete (`abi.rs:329-378`) |
| 16-byte stack alignment | Complete (`frame.rs:78`) |
| Compact unwind section | Complete (`unwind.rs`) |

### 6.2 Missing

| Requirement | Priority | Description |
|-------------|----------|-------------|
| PAC (Pointer Authentication) | P2 | macOS 11+ uses `PACIBSP` in prologue and `AUTIBSP` in epilogue for return address signing. Without this, LLVM2 binaries are incompatible with hardened macOS builds. |
| BTI (Branch Target Identification) | P3 | ARM v8.5-A feature, not yet enforced on macOS but future-proofing. |
| Stack guard pages / probing | P2 | Frames larger than one page (4096 bytes) must probe each page to avoid skipping the guard page. Without probing, large stack allocations can corrupt adjacent memory. |
| `__objc_methname` / ObjC ABI | P3 | Required for Objective-C interop. Not relevant for tMIR initially. |
| `__DATA,__thread_bss` TLS | P2 | Thread-local storage via `tpidr_el0`. Needed for any program using thread-locals. |

### 6.3 PAC Detail

On macOS 12+ with Apple Silicon, pointer authentication is used to protect
return addresses on the stack:

```asm
; Prologue
pacibsp               ; Sign LR with SP as context
stp x29, x30, [sp, #-16]!
mov x29, sp

; Epilogue
ldp x29, x30, [sp], #16
autibsp               ; Verify LR signature
ret
```

The compact unwind encoding does not need changes for PAC (the unwinder
knows to strip PAC bits from return addresses). However, the prologue/epilogue
emission in `frame.rs` must insert `PACIBSP`/`AUTIBSP` instructions.

The `__TEXT,__unwind_info` linker output handles PAC transparently, but
the `__eh_frame` FDE's return address register must be correctly encoded.

---

## 7. Priority Ordering

### P1: Blocking for real-world use

1. **Exception handling (personality + LSDA)** -- Without this, any call to
   C++ or `panic=unwind` Rust causes undefined behavior.
2. **`_Unwind_Resume` lowering** -- Required for exception propagation.
3. **Landing pad lowering in ISel** -- `invoke` must produce split edges.

### P2: ABI compatibility issues

4. **HFA detection and FPR passing** -- Calling any C function returning
   `struct { float x, y; }` will produce wrong results.
5. **Small struct return in registers** -- Same class of bug as HFA.
6. **PAC (PACIBSP/AUTIBSP)** -- Required for hardened macOS builds.
7. **I128 proper two-register handling** -- Current `Indirect` workaround
   is semantically wrong.
8. **Stack probing for large frames** -- Correctness issue for large allocations.
9. **TLS support** -- Required for multi-threaded programs.

### P3: Nice-to-have / future

10. **`va_list`/`va_arg` for variadic definitions**
11. **BTI (Branch Target Identification)**
12. **Tail call optimization**
13. **Swift ABI extensions (`swiftself`, `swiftasync`)**
14. **ObjC ABI / `__objc_methname`**
15. **HVA (vector aggregate passing)**
16. **Dynamic stack alignment beyond 16 bytes**

---

## 8. Implementation Estimates

| Item | Effort | Crate | Key Files |
|------|--------|-------|-----------|
| tMIR `Invoke`/`LandingPad` | Medium | tmir-instrs | `lib.rs` |
| Personality function in compact unwind | Small | llvm2-codegen | `unwind.rs` |
| LSDA generation | Large | llvm2-codegen | new `lsda.rs` |
| Call site table emission | Medium | llvm2-codegen | new `lsda.rs` |
| Landing pad ISel | Medium | llvm2-lower | `isel.rs` |
| HFA detection | Small | llvm2-lower | `abi.rs` |
| Small struct register return | Small | llvm2-lower | `abi.rs` |
| PAC prologue/epilogue | Small | llvm2-codegen | `frame.rs` |
| Stack probing | Medium | llvm2-codegen | `frame.rs` |
| DWARF personality/LSDA augmentation | Medium | llvm2-codegen | `dwarf_cfi.rs` |

---

## 9. Verification Impact

Each ABI change must be proven correct:

- **HFA/HVA classification**: Verify that the register assignment matches
  AAPCS64 for all HFA shapes (1-4 members, F32/F64).
- **LSDA encoding**: The call site table must cover all invoke sites. A missed
  call site means a thrown exception will skip cleanup actions (memory leak,
  resource leak, undefined state).
- **PAC**: `PACIBSP` must execute before any store of LR to memory. `AUTIBSP`
  must execute after the final load of LR from memory. Reordering either
  instruction creates a security vulnerability.
- **Stack probing**: The probe must touch each page between old SP and new SP.
  Missing a probe page allows stack overflow to skip the guard page.

---

## References

- ARM Architecture Reference Manual, AAPCS64: https://developer.arm.com/documentation/ihi0055
- Apple ARM64 Function Calling Conventions: https://developer.apple.com/documentation/xcode/writing-arm64-code-for-apple-platforms
- Itanium C++ ABI, Exception Handling: https://itanium-cxx-abi.github.io/cxx-abi/abi-eh.html
- Apple compact_unwind_encoding.h (Mach-O headers)
- LLVM source: `~/llvm-project-ref/llvm/lib/Target/AArch64/AArch64ISelLowering.cpp`
- LLVM source: `~/llvm-project-ref/llvm/lib/CodeGen/AsmPrinter/EHStreamer.cpp` (LSDA emission)
- LLVM source: `~/llvm-project-ref/llvm/lib/Target/AArch64/AArch64FrameLowering.cpp` (PAC support)
