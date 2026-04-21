// fuzz/fuzz_targets/fuzz_macho_fixup.rs
//
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Copyright 2026 Dropbox, Inc. | License: Apache-2.0
//
// libFuzzer target shadowing `panic_fuzz_macho_fixup.rs`. Feeds the raw
// byte buffer into `MachOParser::parse` and, separately, synthesises a
// small `FixupList` from the tail of the buffer and drives it through
// `resolve_named_symbols` + `resolve_to_relocations`. The contract is:
// every public API in the fixup / parser layer must return a typed
// `Result` — never panic.

#![no_main]

use libfuzzer_sys::fuzz_target;

use llvm2_codegen::macho::fixup::{Fixup, FixupList, FixupTarget};
use llvm2_codegen::macho::linker::MachOParser;
use llvm2_codegen::macho::reloc::AArch64RelocKind;

const RELOC_KINDS: &[AArch64RelocKind] = &[
    AArch64RelocKind::Unsigned,
    AArch64RelocKind::Subtractor,
    AArch64RelocKind::Branch26,
    AArch64RelocKind::Page21,
    AArch64RelocKind::Pageoff12,
    AArch64RelocKind::GotLoadPage21,
    AArch64RelocKind::GotLoadPageoff12,
    AArch64RelocKind::PointerToGot,
    AArch64RelocKind::TlvpLoadPage21,
    AArch64RelocKind::TlvpLoadPageoff12,
    AArch64RelocKind::Addend,
    AArch64RelocKind::AuthenticatedPointer,
];

fn take_u32(data: &[u8], pos: &mut usize) -> u32 {
    let mut buf = [0u8; 4];
    for slot in buf.iter_mut() {
        if *pos < data.len() {
            *slot = data[*pos];
            *pos += 1;
        }
    }
    u32::from_le_bytes(buf)
}

fn take_i64(data: &[u8], pos: &mut usize) -> i64 {
    let mut buf = [0u8; 8];
    for slot in buf.iter_mut() {
        if *pos < data.len() {
            *slot = data[*pos];
            *pos += 1;
        }
    }
    i64::from_le_bytes(buf)
}

fuzz_target!(|data: &[u8]| {
    // First: drive the Mach-O parser against the whole buffer.
    let _ = MachOParser::parse(data);

    // Second: synthesise a fixup list from the tail of the buffer.
    let mut pos = 0;
    let mut list = FixupList::new();
    let nfixups = (data.first().copied().unwrap_or(0) % 8) as usize;
    if !data.is_empty() {
        pos = 1;
    }
    for _ in 0..nfixups {
        let offset = take_u32(data, &mut pos);
        let kind_idx = if pos < data.len() {
            let v = data[pos];
            pos += 1;
            v
        } else {
            0
        };
        let kind = RELOC_KINDS[kind_idx as usize % RELOC_KINDS.len()];
        let target_tag = if pos < data.len() {
            let v = data[pos];
            pos += 1;
            v
        } else {
            0
        };
        let target = match target_tag % 4 {
            0 => FixupTarget::Symbol(take_u32(data, &mut pos)),
            1 => FixupTarget::Section(take_u32(data, &mut pos)),
            2 => FixupTarget::SymbolPlusOffset {
                symbol_index: take_u32(data, &mut pos),
                section_offset: take_i64(data, &mut pos) as u64,
            },
            _ => FixupTarget::NamedSymbol("_fuzzsym".into()),
        };
        let addend = take_i64(data, &mut pos);
        list.push(Fixup {
            offset,
            kind,
            target,
            addend,
        });
    }

    let _ = list.resolve_named_symbols(|_name: &str| Some(0));
    let _ = list.resolve_to_relocations();
});
