# LLVM2 Mach-O Object File Format Design

Author: Andrew Yates <ayates@dropbox.com>
Date: 2026-04-13
Status: Implemented

References:
- Apple `<mach-o/loader.h>`, `<mach-o/reloc.h>`, `<mach-o/arm64/reloc.h>`
- LLVM `llvm/lib/MC/MachObjectWriter.cpp`
- LLVM `llvm/lib/Target/AArch64/MCTargetDesc/AArch64AsmBackend.cpp`

## Overview

LLVM2 emits Mach-O 64-bit relocatable object files (`.o`) for macOS AArch64. The implementation lives in `crates/llvm2-codegen/src/macho/` (8 modules: `constants`, `fixup`, `header`, `reloc`, `section`, `symbol`, `writer`, `mod`). The writer produces files consumable by the macOS system linker (`ld`).

## File Layout

The object file follows the standard Mach-O MH_OBJECT layout:

```
Offset 0:    mach_header_64         (32 bytes)
             Load commands:
               LC_SEGMENT_64        (72 + 80*nsects bytes)
               LC_BUILD_VERSION     (24 bytes)
               LC_SYMTAB            (24 bytes)
               LC_DYSYMTAB          (80 bytes)
             Section data           (aligned per section)
             Relocation entries     (8 bytes each)
             Symbol table (nlist)   (16 bytes each)
             String table           (variable)
```

### Section Ordering and Alignment

Sections are emitted in the order they are added to the writer:
1. `__TEXT,__text` -- machine code (4-byte aligned, `2^2`)
2. `__DATA,__data` -- initialized data (8-byte aligned, `2^3`)
3. `__LD,__compact_unwind` -- compact unwind entries (8-byte aligned, `2^3`)
4. Any custom sections added via `add_custom_section()`

All sections live in a single unnamed segment (standard for MH_OBJECT files). Section data starts immediately after the header and load commands, with inter-section padding to satisfy alignment requirements. The alignment is encoded as a power of 2 in the section header's `align` field.

**Alignment padding:** Before each section's data, zero bytes are inserted until the current file offset is aligned to `2^align`. This means:
- `__text`: aligned to 4-byte boundary (ARM64 instruction alignment)
- `__data`: aligned to 8-byte boundary
- `__compact_unwind`: aligned to 8-byte boundary

### Virtual Memory Layout

For MH_OBJECT files, `vmaddr` starts at 0. Each section's virtual address is computed sequentially with alignment padding, matching the file layout. The segment's `vmsize` equals the end of the last section's virtual address range. `fileoff` points to the first section's file offset, and `filesize` covers all section data.

## MH_SUBSECTIONS_VIA_SYMBOLS

The `MH_SUBSECTIONS_VIA_SYMBOLS` flag (0x2000) is **always set** in the Mach-O header flags. This tells the linker that it is safe to divide sections into sub-sections at symbol boundaries for dead code stripping.

**Implementation:** `MachHeader::new_arm64_object()` unconditionally sets this flag. This is correct because LLVM2 emits one symbol per function and does not generate cross-function fall-through code. Each function is an independent unit that the linker can strip if unreferenced.

**Source:** `crates/llvm2-codegen/src/macho/header.rs:44`

## Load Commands

Four load commands are emitted:

1. **LC_SEGMENT_64** -- Contains all section headers. For object files, uses an unnamed segment (16 zero bytes for segname). Protection is `VM_PROT_ALL` (rwx) for both `maxprot` and `initprot`.

2. **LC_BUILD_VERSION** -- Specifies macOS 14.0 as minimum deployment target (`minos = 0x000E0000`), SDK 14.0 (`sdk = 0x000E0000`), platform = `PLATFORM_MACOS` (1), no tool entries.

3. **LC_SYMTAB** -- Symbol table location: offset, count, string table offset, string table size.

4. **LC_DYSYMTAB** -- Dynamic symbol table with the required three-partition ordering: local symbols, external defined symbols, undefined symbols. Remaining fields (TOC, module table, indirect symbols, external/local relocations) are all zero for simple object files.

## Relocation Types

All 12 ARM64 Mach-O relocation types are defined in the `AArch64RelocKind` enum:

| Type | Value | PC-rel | Size | Use |
|------|-------|--------|------|-----|
| `ARM64_RELOC_UNSIGNED` | 0 | No | 8B | Absolute 64-bit pointers in data |
| `ARM64_RELOC_SUBTRACTOR` | 1 | No | var | Symbol difference (A - B), paired with UNSIGNED |
| `ARM64_RELOC_BRANCH26` | 2 | Yes | 4B | B/BL branch instructions (+/-128 MB range) |
| `ARM64_RELOC_PAGE21` | 3 | Yes | 4B | ADRP page address (+/-4 GB range) |
| `ARM64_RELOC_PAGEOFF12` | 4 | No | 4B | ADD/LDR page offset (12-bit, within 4 KB page) |
| `ARM64_RELOC_GOT_LOAD_PAGE21` | 5 | Yes | 4B | ADRP to GOT entry page |
| `ARM64_RELOC_GOT_LOAD_PAGEOFF12` | 6 | No | 4B | LDR from GOT entry (page offset) |
| `ARM64_RELOC_POINTER_TO_GOT` | 7 | Yes | 4B | 32-bit PC-relative GOT pointer in data |
| `ARM64_RELOC_TLVP_LOAD_PAGE21` | 8 | Yes | 4B | ADRP for TLV descriptor |
| `ARM64_RELOC_TLVP_LOAD_PAGEOFF12` | 9 | No | 4B | LDR for TLV descriptor (page offset) |
| `ARM64_RELOC_ADDEND` | 10 | No | 4B | Addend for BRANCH26/PAGE21/PAGEOFF12 (paired) |
| `ARM64_RELOC_AUTHENTICATED_POINTER` | 11 | No | 8B | arm64e pointer authentication |

### Commonly Used Relocations

The typical function call and data access patterns use these relocation combinations:

**Function call (B/BL):** Single `BRANCH26` relocation.

**Global variable access (ADRP + ADD/LDR):** Paired `PAGE21` + `PAGEOFF12` relocations at consecutive instructions.

**GOT access (ADRP + LDR from GOT):** Paired `GOT_LOAD_PAGE21` + `GOT_LOAD_PAGEOFF12`.

**Absolute data pointer:** Single `UNSIGNED` relocation (length=3 for 64-bit).

**Symbol difference (A - B):** Paired `SUBTRACTOR` + `UNSIGNED` at the same offset.

### Relocation Encoding

Each relocation is 8 bytes (`relocation_info` struct):
```
r_word0 (4 bytes): r_address -- byte offset within section
r_word1 (4 bytes): packed bitfield
  bits  0-23: r_symbolnum (symbol index or section ordinal)
  bit     24: r_pcrel (1 = PC-relative)
  bits 25-26: r_length (log2 of size: 0=1B, 1=2B, 2=4B, 3=8B)
  bit     27: r_extern (1 = symbol table index, 0 = section ordinal)
  bits 28-31: r_type (relocation type from table above)
```

Relocations with non-zero addends on BRANCH26, PAGE21, or PAGEOFF12 emit a preceding `ARM64_RELOC_ADDEND` relocation. The addend value (signed 24-bit) is stored in the `r_symbolnum` field of the addend relocation.

## Fixup Layer

The fixup layer (`crates/llvm2-codegen/src/macho/fixup.rs`) provides deferred relocation resolution. During instruction encoding, fixups are recorded with placeholder values. After final layout (when section offsets are known), fixups are resolved into relocations and instruction bytes are patched.

Fixup targets can be:
- **Symbol** -- external relocation referencing a symbol table entry
- **Section** -- local relocation referencing a section ordinal
- **SymbolPlusOffset** -- symbol with section-relative adjustment

The fixup module also provides instruction patching functions:
- `apply_branch26()` -- patch B/BL imm26 field (signed word offset)
- `apply_page21()` -- patch ADRP immhi:immlo fields (signed page offset)
- `apply_pageoff12()` -- patch ADD/LDR imm12 field (scaled page offset)

## Symbol Table

Symbols are emitted as `nlist_64` entries (16 bytes each):
```
n_strx:  u32  -- string table offset
n_type:  u8   -- type flags (N_SECT|N_EXT for external defined, N_UNDF for undefined)
n_sect:  u8   -- 1-based section ordinal (0 for undefined)
n_desc:  u16  -- descriptor (0 for simple symbols)
n_value: u64  -- virtual address (section base + offset for defined symbols)
```

**Partitioning:** The dysymtab requires symbols in strict order:
1. Local symbols (not external)
2. External defined symbols (external + section != 0)
3. Undefined symbols (external + section == 0)

The string table starts with a null byte (index 0 = empty string), followed by null-terminated symbol names.

## Compact Unwind

The `__LD,__compact_unwind` section is required for usable macOS binaries. Implementation is in `crates/llvm2-codegen/src/unwind.rs`.

Each function gets a 32-byte compact unwind entry:
```
function_offset:   u64  -- function start address (relocated via ARM64_RELOC_UNSIGNED)
function_length:   u32  -- function size in bytes
compact_encoding:  u32  -- unwind encoding
personality:       u64  -- personality function (0 for C/Rust without exceptions)
lsda:              u64  -- language-specific data area (0 for C/Rust without exceptions)
```

### Compact Unwind Encoding

Three encoding modes (defined in `crates/llvm2-codegen/src/frame.rs`):

| Mode | Value | Description |
|------|-------|-------------|
| `UNWIND_ARM64_MODE_FRAMELESS` | 0x02000000 | No frame pointer, stack size encoded |
| `UNWIND_ARM64_MODE_DWARF` | 0x03000000 | Fallback to DWARF unwind info |
| `UNWIND_ARM64_MODE_FRAME` | 0x04000000 | Standard FP/LR frame, callee-saved register bitmap |

The FRAME mode encodes which callee-saved register pairs are saved as a bitmap in bits 12-20 of the encoding. For example, `UNWIND_ARM64_FRAME_X19_X20_PAIR` (bit 0 of the bitmap) indicates x19/x20 are saved on the stack.

### Section Attributes

The compact unwind section uses:
- Section name: `__compact_unwind`
- Segment name: `__LD`
- Alignment: `2^3` = 8 bytes
- Flags: `S_ATTR_DEBUG` (0x02000000) -- debug section, not loaded at runtime

Each entry's `function_offset` field requires an `ARM64_RELOC_UNSIGNED` relocation (length=3, quad) pointing to the function symbol. Non-zero `personality` and `lsda` fields also require relocations.

## LC_DATA_IN_CODE

LLVM2 does **not** emit `LC_DATA_IN_CODE` load commands.

**Rationale:** `LC_DATA_IN_CODE` is used to mark regions within `__text` sections that contain data rather than instructions (jump tables, constant pools embedded in code). LLVM2's code generation model does not embed data in code sections:

- Jump tables are not yet implemented (switch lowering uses if/else chains)
- Constant pools are placed in `__DATA,__data`, not inline in `__TEXT,__text`
- Literal pools are not needed because AArch64's ADRP+ADD/LDR pattern can reach any address

If jump tables or inline constant pools are added in the future, `LC_DATA_IN_CODE` will be needed to prevent the disassembler from misinterpreting data bytes as instructions.

## Source Files

| File | Purpose |
|------|---------|
| `macho/mod.rs` | Module root, re-exports |
| `macho/constants.rs` | All Mach-O format constants (magic numbers, CPU types, load commands, section types, relocation types, struct sizes) |
| `macho/header.rs` | `MachHeader` -- mach_header_64 serialization |
| `macho/section.rs` | `Section64`, `SegmentCommand64` -- section and segment serialization |
| `macho/symbol.rs` | `NList64`, `SymbolTable`, `DysymtabParams` -- symbol table types |
| `macho/reloc.rs` | `AArch64RelocKind`, `Relocation` -- relocation encoding/decoding, addend/subtractor pair helpers |
| `macho/fixup.rs` | `Fixup`, `FixupList` -- deferred relocation resolution layer, instruction patching |
| `macho/writer.rs` | `MachOWriter` -- assembles complete .o file from sections, symbols, relocations |
| `unwind.rs` | `CompactUnwindEntry`, `CompactUnwindSection` -- compact unwind emission |
| `frame.rs` | `FrameLayout`, `encode_compact_unwind()` -- frame lowering and encoding |
