// llvm2-codegen - Exception handling LSDA table generation
//
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Copyright 2026 Andrew Yates | License: Apache-2.0
//
// Reference: Itanium C++ ABI Exception Handling
//            (https://itanium-cxx-abi.github.io/cxx-abi/abi-eh.html)
// Reference: ~/llvm-project-ref/llvm/lib/CodeGen/AsmPrinter/EHStreamer.cpp
//            (LSDA emission)
// Reference: DWARF 4 spec, Section 7.3 (DWARF Expression encoding)

//! Language-Specific Data Area (LSDA) table generation for C++ exception
//! handling on AArch64 macOS.
//!
//! The LSDA is emitted in the `__TEXT,__gcc_except_table` section of a Mach-O
//! object file. It is referenced by the compact unwind entry's LSDA pointer
//! and is used by the personality routine (`__gxx_personality_v0` for C++,
//! `__rust_eh_personality` for Rust) to dispatch exceptions to the correct
//! landing pad.
//!
//! # LSDA layout (Itanium ABI)
//!
//! ```text
//! +-----------------------+
//! | Header                |
//! |  - LPStart encoding   |  u8: DW_EH_PE_omit => use function start
//! |  - TType encoding     |  u8: DW_EH_PE_omit if no type table
//! |  - TType base offset  |  ULEB128 (only if TType encoding != omit)
//! |  - Call site encoding  |  u8: DW_EH_PE_udata4 for AArch64
//! |  - Call site length    |  ULEB128
//! +-----------------------+
//! | Call Site Table        |
//! |  For each call site:   |
//! |  - region start        |  encoded (offset from function start)
//! |  - region length       |  encoded
//! |  - landing pad offset  |  encoded (0 = no landing pad)
//! |  - action index        |  ULEB128 (0 = cleanup only)
//! +-----------------------+
//! | Action Table           |
//! |  For each action:      |
//! |  - type filter index   |  SLEB128 (>0 catch, 0 cleanup, <0 filter)
//! |  - next action offset  |  SLEB128 (0 = end of chain)
//! +-----------------------+
//! | Type Table             |
//! |  (grows backward from  |
//! |   TType base offset)   |
//! |  - type info pointers  |  4 bytes each (udata4 encoding)
//! +-----------------------+
//! ```
//!
//! # Personality routines
//!
//! The personality routine is specified in the compact unwind entry and the
//! CIE augmentation data. Common personality routines:
//! - `__gxx_personality_v0` — C++ (libcxxabi / libstdc++)
//! - `__rust_eh_personality` — Rust panic unwinding
//! - `__gcc_personality_v0` — C cleanup-only

// ---------------------------------------------------------------------------
// DWARF pointer encoding constants (DW_EH_PE_*)
// ---------------------------------------------------------------------------

/// DWARF Exception Handling Pointer Encoding format.
///
/// Defines how pointers in the LSDA (and eh_frame) are encoded.
/// The encoding byte has two parts:
/// - Low 4 bits: value format (absptr, udata2, udata4, etc.)
/// - High 4 bits: application (absptr, pcrel, datarel, etc.)
///
/// Reference: DWARF 4 spec, Table 7.9 (Pointer encoding)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum DwEhPe {
    /// DW_EH_PE_absptr: absolute pointer (native size).
    AbsPtr = 0x00,
    /// DW_EH_PE_udata2: unsigned 2-byte value.
    UData2 = 0x02,
    /// DW_EH_PE_udata4: unsigned 4-byte value.
    UData4 = 0x03,
    /// DW_EH_PE_udata8: unsigned 8-byte value.
    UData8 = 0x04,
    /// DW_EH_PE_sdata4: signed 4-byte value.
    SData4 = 0x0B,
    /// DW_EH_PE_omit: value is omitted (not present).
    Omit = 0xFF,
}

impl DwEhPe {
    /// Size in bytes of a value encoded with this format.
    ///
    /// Returns `None` for `Omit` (no value present) and `AbsPtr` (size
    /// depends on the target's pointer width).
    pub fn encoded_size(&self) -> Option<u32> {
        match self {
            DwEhPe::UData2 => Some(2),
            DwEhPe::UData4 | DwEhPe::SData4 => Some(4),
            DwEhPe::UData8 => Some(8),
            DwEhPe::AbsPtr => None, // depends on target pointer size
            DwEhPe::Omit => None,
        }
    }
}

// ---------------------------------------------------------------------------
// LSDA data types
// ---------------------------------------------------------------------------

/// A single call site entry in the LSDA call site table.
///
/// Each entry describes a contiguous region of instructions that may throw
/// (or invoke cleanup code). The unwinder scans this table to find the
/// landing pad for a given instruction pointer.
///
/// All offsets are relative to the function start (LPStart).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CallSiteEntry {
    /// Start of the call site region, as a byte offset from function start.
    pub region_start: u32,
    /// Length of the call site region in bytes.
    pub region_length: u32,
    /// Offset of the landing pad from function start.
    /// 0 means no landing pad (exception propagates to caller).
    pub landing_pad: u32,
    /// 1-based index into the action table.
    /// 0 means no action (cleanup-only or no handling).
    pub action_idx: u32,
}

impl CallSiteEntry {
    /// Create a call site with a landing pad and action.
    pub fn new(region_start: u32, region_length: u32, landing_pad: u32, action_idx: u32) -> Self {
        Self {
            region_start,
            region_length,
            landing_pad,
            action_idx,
        }
    }

    /// Create a call site with no landing pad (exception propagates).
    pub fn no_landing_pad(region_start: u32, region_length: u32) -> Self {
        Self {
            region_start,
            region_length,
            landing_pad: 0,
            action_idx: 0,
        }
    }

    /// Create a cleanup-only call site (landing pad but action_idx = 0).
    pub fn cleanup(region_start: u32, region_length: u32, landing_pad: u32) -> Self {
        Self {
            region_start,
            region_length,
            landing_pad,
            action_idx: 0,
        }
    }
}

/// A single action entry in the LSDA action table.
///
/// Actions form a linked list (via `next_action_offset`). Each action
/// specifies a type filter that the personality routine checks against
/// the thrown exception type.
///
/// The type filter values have special meaning:
/// - Positive: index into the type table (catch clause)
/// - Zero: cleanup action (always matches, like a destructor call)
/// - Negative: index into the filter table (exception spec)
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ActionEntry {
    /// Type filter index.
    /// Positive = catch (index into type table).
    /// Zero = cleanup.
    /// Negative = exception specification filter.
    pub type_filter: i32,
    /// Byte offset to the next action in the chain.
    /// 0 = end of chain (no more actions for this call site).
    pub next_action_offset: i32,
}

impl ActionEntry {
    /// Create a catch action for a specific type.
    ///
    /// `type_index` is a 1-based index into the type table.
    pub fn catch(type_index: u32) -> Self {
        Self {
            type_filter: type_index as i32,
            next_action_offset: 0,
        }
    }

    /// Create a cleanup action (runs destructors, then re-throws).
    pub fn cleanup() -> Self {
        Self {
            type_filter: 0,
            next_action_offset: 0,
        }
    }

    /// Create a catch-all action (catches any exception).
    ///
    /// In the Itanium ABI, catch-all uses type filter index 0, which is
    /// also the cleanup marker. The personality routine distinguishes
    /// catch-all from cleanup based on the call site's action index
    /// versus the action table entry.
    pub fn catch_all() -> Self {
        Self {
            type_filter: 0,
            next_action_offset: 0,
        }
    }
}

/// A type info entry in the LSDA type table.
///
/// Each entry holds a reference to a type info object (e.g.,
/// `std::type_info` for C++). The index is used by action entries
/// to identify which exception types to catch.
///
/// Index 0 is reserved for "cleanup" (no specific type match).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TypeInfo {
    /// Type info index. For now this is an opaque index that will be
    /// resolved to a symbol reference or relocation during object emission.
    /// 0 = catch-all / cleanup (matches any type).
    pub type_info_index: u32,
}

impl TypeInfo {
    /// Create a type info entry.
    pub fn new(index: u32) -> Self {
        Self {
            type_info_index: index,
        }
    }

    /// Create a catch-all type info entry (index 0).
    pub fn catch_all() -> Self {
        Self {
            type_info_index: 0,
        }
    }
}

// ---------------------------------------------------------------------------
// ExceptionTable — complete LSDA data for one function
// ---------------------------------------------------------------------------

/// Complete exception handling table for a single function.
///
/// Contains all the data needed to generate an LSDA: call site entries,
/// action entries, and type info entries. The `generate_lsda()` function
/// serializes this to the binary format expected by the personality routine.
#[derive(Debug, Clone)]
pub struct ExceptionTable {
    /// Call site entries (instruction regions that may throw).
    pub call_sites: Vec<CallSiteEntry>,
    /// Action entries (type filter chains for landing pads).
    pub actions: Vec<ActionEntry>,
    /// Type info entries (exception type references).
    pub type_infos: Vec<TypeInfo>,
    /// Personality routine symbol name (e.g., "__gxx_personality_v0").
    pub personality: Option<String>,
}

impl ExceptionTable {
    /// Create a new empty exception table.
    pub fn new() -> Self {
        Self {
            call_sites: Vec::new(),
            actions: Vec::new(),
            type_infos: Vec::new(),
            personality: None,
        }
    }

    /// Create an exception table with a C++ personality routine.
    pub fn with_cxx_personality() -> Self {
        Self {
            call_sites: Vec::new(),
            actions: Vec::new(),
            type_infos: Vec::new(),
            personality: Some("__gxx_personality_v0".to_string()),
        }
    }

    /// Create an exception table with a Rust personality routine.
    pub fn with_rust_personality() -> Self {
        Self {
            call_sites: Vec::new(),
            actions: Vec::new(),
            type_infos: Vec::new(),
            personality: Some("__rust_eh_personality".to_string()),
        }
    }

    /// Add a call site entry.
    pub fn add_call_site(&mut self, entry: CallSiteEntry) {
        self.call_sites.push(entry);
    }

    /// Add an action entry. Returns the 1-based action index.
    pub fn add_action(&mut self, entry: ActionEntry) -> u32 {
        self.actions.push(entry);
        self.actions.len() as u32
    }

    /// Add a type info entry. Returns the 1-based type index.
    pub fn add_type_info(&mut self, entry: TypeInfo) -> u32 {
        self.type_infos.push(entry);
        self.type_infos.len() as u32
    }

    /// Returns true if there are no call sites (no exception handling needed).
    pub fn is_empty(&self) -> bool {
        self.call_sites.is_empty()
    }
}

impl Default for ExceptionTable {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// LEB128 encoding helpers
// ---------------------------------------------------------------------------

/// Encode a value as ULEB128 (unsigned LEB128).
///
/// Returns the encoded bytes as a Vec. This is the public, allocating
/// interface; use `encode_uleb128_into` for the append-to-buffer variant.
pub fn encode_uleb128(value: u64) -> Vec<u8> {
    let mut out = Vec::new();
    encode_uleb128_into(value, &mut out);
    out
}

/// Encode a value as SLEB128 (signed LEB128).
///
/// Returns the encoded bytes as a Vec. This is the public, allocating
/// interface; use `encode_sleb128_into` for the append-to-buffer variant.
pub fn encode_sleb128(value: i64) -> Vec<u8> {
    let mut out = Vec::new();
    encode_sleb128_into(value, &mut out);
    out
}

/// Encode a value as ULEB128, appending to `out`.
fn encode_uleb128_into(mut value: u64, out: &mut Vec<u8>) {
    loop {
        let mut byte = (value & 0x7F) as u8;
        value >>= 7;
        if value != 0 {
            byte |= 0x80; // more bytes follow
        }
        out.push(byte);
        if value == 0 {
            break;
        }
    }
}

/// Encode a value as SLEB128, appending to `out`.
fn encode_sleb128_into(mut value: i64, out: &mut Vec<u8>) {
    let mut more = true;
    while more {
        let mut byte = (value & 0x7F) as u8;
        value >>= 7;
        // If the sign bit of the current byte matches the remaining value,
        // we're done.
        if (value == 0 && byte & 0x40 == 0) || (value == -1 && byte & 0x40 != 0) {
            more = false;
        } else {
            byte |= 0x80;
        }
        out.push(byte);
    }
}

// ---------------------------------------------------------------------------
// Call site table emission
// ---------------------------------------------------------------------------

/// Emit the call site table as raw bytes.
///
/// Each entry is encoded according to `encoding`:
/// - `DwEhPe::UData4`: each field is a 4-byte unsigned little-endian value
/// - Other encodings: currently only UData4 is supported for AArch64
///
/// The action index is always ULEB128-encoded per the Itanium ABI.
fn emit_call_site_table(call_sites: &[CallSiteEntry], encoding: DwEhPe) -> Vec<u8> {
    let mut data = Vec::new();

    for cs in call_sites {
        match encoding {
            DwEhPe::UData4 => {
                data.extend_from_slice(&cs.region_start.to_le_bytes());
                data.extend_from_slice(&cs.region_length.to_le_bytes());
                data.extend_from_slice(&cs.landing_pad.to_le_bytes());
            }
            _ => {
                // Fallback: encode as udata4 (the standard for AArch64).
                data.extend_from_slice(&cs.region_start.to_le_bytes());
                data.extend_from_slice(&cs.region_length.to_le_bytes());
                data.extend_from_slice(&cs.landing_pad.to_le_bytes());
            }
        }
        // Action index is always ULEB128 per Itanium ABI.
        encode_uleb128_into(cs.action_idx as u64, &mut data);
    }

    data
}

// ---------------------------------------------------------------------------
// Action table emission
// ---------------------------------------------------------------------------

/// Emit the action table as raw bytes.
///
/// Each action entry consists of two SLEB128 values:
/// 1. Type filter index (positive = catch, 0 = cleanup, negative = filter)
/// 2. Next action offset (byte displacement to next action, 0 = end of chain)
fn emit_action_table(actions: &[ActionEntry]) -> Vec<u8> {
    let mut data = Vec::new();

    for action in actions {
        encode_sleb128_into(action.type_filter as i64, &mut data);
        encode_sleb128_into(action.next_action_offset as i64, &mut data);
    }

    data
}

// ---------------------------------------------------------------------------
// Type table emission
// ---------------------------------------------------------------------------

/// Emit the type table as raw bytes.
///
/// Type info entries are emitted in reverse order (the Itanium ABI
/// specifies that the type table grows backward from the TType base).
/// Each entry is a 4-byte value (for DW_EH_PE_udata4 encoding).
///
/// The entries are emitted in the order they appear in the `type_infos`
/// slice, but the personality routine indexes them backwards from the
/// TType base offset. Entry at index 1 is the last 4 bytes before the
/// TType base, entry 2 is 8 bytes before, etc.
fn emit_type_table(type_infos: &[TypeInfo]) -> Vec<u8> {
    let mut data = Vec::new();

    // Type table entries are stored in reverse order so that
    // type_info[1] is at (TType_base - 4), type_info[2] at (TType_base - 8), etc.
    for ti in type_infos.iter().rev() {
        data.extend_from_slice(&ti.type_info_index.to_le_bytes());
    }

    data
}

// ---------------------------------------------------------------------------
// LSDA generation — main entry point
// ---------------------------------------------------------------------------

/// Generate a complete LSDA (Language-Specific Data Area) for one function.
///
/// The output bytes are intended for the `__TEXT,__gcc_except_table` Mach-O
/// section. The personality routine reads this data to dispatch exceptions
/// to the correct landing pad.
///
/// # Binary layout
///
/// See the module-level documentation for the full LSDA layout.
///
/// # Arguments
///
/// * `table` — The exception table containing call sites, actions, and type info.
///
/// # Returns
///
/// A `Vec<u8>` containing the serialized LSDA bytes.
pub fn generate_lsda(table: &ExceptionTable) -> Vec<u8> {
    let call_site_encoding = DwEhPe::UData4;
    let has_type_table = !table.type_infos.is_empty();

    // Pre-emit the call site table, action table, and type table to compute sizes.
    let call_site_data = emit_call_site_table(&table.call_sites, call_site_encoding);
    let action_data = emit_action_table(&table.actions);
    let type_data = emit_type_table(&table.type_infos);

    let mut lsda = Vec::new();

    // --- Header ---

    // LPStart encoding: DW_EH_PE_omit (0xFF) = use function start as LPStart.
    lsda.push(DwEhPe::Omit as u8);

    // TType encoding and base offset.
    if has_type_table {
        // TType encoding: DW_EH_PE_udata4.
        lsda.push(DwEhPe::UData4 as u8);

        // TType base offset: byte offset from the end of this ULEB128 field
        // to the end of the type table (i.e., past the action table + type table).
        // We need to account for:
        //   - Call site table length (ULEB128)
        //   - Call site encoding byte
        //   - Call site data
        //   - Action data
        //   - Type data
        //
        // The TType base offset points to the END of the type table.
        // From the byte after the TType base offset field, the remaining
        // LSDA content is:
        //   call_site_encoding (1 byte) + call_site_length (ULEB128) +
        //   call_site_data + action_data + type_data

        let cs_length_encoded = encode_uleb128(call_site_data.len() as u64);
        let ttype_base = 1 // call site encoding byte
            + cs_length_encoded.len()
            + call_site_data.len()
            + action_data.len()
            + type_data.len();

        encode_uleb128_into(ttype_base as u64, &mut lsda);
    } else {
        // No type table: TType encoding = DW_EH_PE_omit.
        lsda.push(DwEhPe::Omit as u8);
    }

    // Call site encoding.
    lsda.push(call_site_encoding as u8);

    // Call site table length (ULEB128).
    encode_uleb128_into(call_site_data.len() as u64, &mut lsda);

    // --- Call site table ---
    lsda.extend_from_slice(&call_site_data);

    // --- Action table ---
    lsda.extend_from_slice(&action_data);

    // --- Type table ---
    lsda.extend_from_slice(&type_data);

    lsda
}

// ---------------------------------------------------------------------------
// EH-to-LSDA bridge — convert ABI-level EH info to codegen-level LSDA
// ---------------------------------------------------------------------------

/// Parameters describing a function's exception handling for LSDA generation.
///
/// This is a standalone bridge type that can be constructed from the ABI-level
/// `ExceptionHandlingInfo` in `llvm2-lower` without requiring a cross-crate
/// dependency. The ISel or pipeline adapter populates this struct and passes
/// it to `build_exception_table()` to produce an `ExceptionTable` ready for
/// `generate_lsda()`.
///
/// # Usage
///
/// ```ignore
/// let eh_params = EhBridgeParams {
///     personality_symbol: Some("__gxx_personality_v0".to_string()),
///     call_sites: vec![
///         EhCallSite { start_offset: 0x10, length: 0x08, landing_pad_offset: 0x50, action_index: 1 },
///     ],
///     actions: vec![
///         EhAction { type_filter: 1, next_offset: 0 },
///     ],
///     type_indices: vec![42], // opaque indices into the type table
/// };
/// let table = build_exception_table(&eh_params);
/// let lsda_bytes = generate_lsda(&table);
/// ```
#[derive(Debug, Clone)]
pub struct EhBridgeParams {
    /// Personality routine symbol name (e.g., "__gxx_personality_v0").
    /// `None` for functions without exception handling.
    pub personality_symbol: Option<String>,
    /// Call site entries mapping PC ranges to landing pads.
    pub call_sites: Vec<EhCallSite>,
    /// Action table entries (type filter chains).
    pub actions: Vec<EhAction>,
    /// Type info indices (opaque references to type descriptors).
    /// These are 1-based indices as used in the action table's type_filter.
    /// Index 0 is implicitly catch-all/cleanup.
    pub type_indices: Vec<u32>,
}

/// A call site for the EH bridge.
#[derive(Debug, Clone)]
pub struct EhCallSite {
    /// Start of the call site region (offset from function start).
    pub start_offset: u32,
    /// Length of the call site region in bytes.
    pub length: u32,
    /// Landing pad offset from function start (0 = no landing pad).
    pub landing_pad_offset: u32,
    /// 1-based index into the action table (0 = no action / cleanup-only).
    pub action_index: u32,
}

/// An action entry for the EH bridge.
#[derive(Debug, Clone)]
pub struct EhAction {
    /// Type filter: positive = catch, 0 = cleanup, negative = filter.
    pub type_filter: i32,
    /// Byte offset to next action in the chain (0 = end of chain).
    pub next_offset: i32,
}

/// Build an `ExceptionTable` from bridge parameters.
///
/// Converts the ABI-level EH metadata into the codegen-level LSDA types:
/// - Maps call site entries to `CallSiteEntry` (codegen level)
/// - Maps action entries to `ActionEntry` (codegen level) with correct
///   byte offsets for the action chain
/// - Populates the type table from the type index list
///
/// The resulting `ExceptionTable` can be passed directly to `generate_lsda()`.
pub fn build_exception_table(params: &EhBridgeParams) -> ExceptionTable {
    let mut table = ExceptionTable::new();
    table.personality = params.personality_symbol.clone();

    // Map call sites.
    for cs in &params.call_sites {
        table.add_call_site(CallSiteEntry::new(
            cs.start_offset,
            cs.length,
            cs.landing_pad_offset,
            cs.action_index,
        ));
    }

    // Map actions. The Itanium ABI specifies action chains as byte-offset
    // linked lists. Each action entry is 2 SLEB128 values: type_filter and
    // next_action_offset. The next_action_offset is a byte displacement from
    // the start of the current entry to the start of the next entry.
    //
    // For our bridge, we compute byte offsets from the incoming next_offset
    // field. If next_offset is 0, it means end of chain. Otherwise, it's
    // treated as a 1-based index into the action table (the byte offset is
    // computed from the SLEB128 sizes of the intervening entries).
    //
    // However, the simpler approach (matching LLVM's EHStreamer) is to treat
    // the action entries as a flat array where the caller has already computed
    // byte offsets. We pass them through directly.
    for action in &params.actions {
        table.actions.push(ActionEntry {
            type_filter: action.type_filter,
            next_action_offset: action.next_offset,
        });
    }

    // Map type table.
    for &idx in &params.type_indices {
        table.add_type_info(TypeInfo::new(idx));
    }

    table
}

/// Build an `ExceptionTable` from landing pad metadata.
///
/// This is a higher-level builder that takes landing pad descriptors and
/// automatically constructs the call site table, action table, and type
/// table. It handles:
///
/// - **Catch typed**: Creates catch actions with positive type filter indices
/// - **Catch-all**: Creates actions with type filter 0 (distinguished from
///   cleanup by having a non-zero action index in the call site)
/// - **Cleanup**: Creates cleanup call sites (action_idx = 0, landing pad set)
/// - **Action chains**: When a landing pad has multiple catch types, chains
///   them via next_action_offset
///
/// # Arguments
///
/// * `personality` — Personality function symbol name
/// * `landing_pads` — Landing pad descriptors
/// * `call_site_ranges` — PC ranges mapping to landing pads. Each tuple is
///   `(start_offset, length, landing_pad_offset)`.
///
/// # Returns
///
/// A fully populated `ExceptionTable` ready for `generate_lsda()`.
pub fn build_exception_table_from_pads(
    personality: &str,
    landing_pads: &[LandingPadDesc],
    call_site_ranges: &[(u32, u32, u32)],
) -> ExceptionTable {
    let mut table = ExceptionTable::new();
    table.personality = Some(personality.to_string());

    // Collect all unique type indices and build the type table.
    let mut type_index_set: Vec<u32> = Vec::new();
    for lp in landing_pads {
        for &ti in &lp.catch_type_indices {
            if ti != 0 && !type_index_set.contains(&ti) {
                type_index_set.push(ti);
            }
        }
    }

    // Add type infos to the table.
    for &ti in &type_index_set {
        table.add_type_info(TypeInfo::new(ti));
    }

    // Build action entries for each landing pad.
    // Each landing pad gets an action chain. The first action's 1-based index
    // is stored in the call site entry.
    let mut lp_to_first_action: std::collections::HashMap<u32, u32> = std::collections::HashMap::new();

    for lp in landing_pads {
        if lp.is_cleanup && lp.catch_type_indices.is_empty() {
            // Cleanup-only: action_idx = 0 in the call site (no action chain).
            lp_to_first_action.insert(lp.landing_pad_offset, 0);
            continue;
        }

        // Build the action chain for this landing pad's catch types.
        let chain_start = table.actions.len() as u32 + 1; // 1-based
        let num_catches = lp.catch_type_indices.len();

        for (i, &catch_idx) in lp.catch_type_indices.iter().enumerate() {
            let type_filter = if catch_idx == 0 {
                // Catch-all: use type filter 0, but the personality distinguishes
                // this from cleanup via the call site's non-zero action index.
                0i32
            } else {
                // Map the opaque type index to its 1-based position in our type table.
                type_index_set.iter().position(|&ti| ti == catch_idx)
                    .map(|pos| (pos + 1) as i32)
                    .unwrap_or(0)
            };

            // Compute next_action_offset. Each action entry is 2 SLEB128 values.
            // For simplicity, we use a fixed-size encoding estimate: each SLEB128
            // value for small integers fits in 1 byte, so each action entry is
            // typically 2 bytes.
            let is_last = i == num_catches - 1;
            let next_offset = if is_last && !lp.is_cleanup {
                0 // end of chain
            } else if is_last && lp.is_cleanup {
                // Chain to a cleanup action.
                2 // next entry is 2 bytes ahead (SLEB128(type_filter) + SLEB128(next))
            } else {
                // Point to next catch in the chain.
                // Each entry is SLEB128(filter) + SLEB128(next_offset).
                // For small values, each SLEB128 is 1 byte, so entry size = 2.
                2i32
            };

            table.actions.push(ActionEntry {
                type_filter,
                next_action_offset: next_offset,
            });
        }

        // If the landing pad has both catches and cleanup, add a cleanup action
        // at the end of the chain.
        if lp.is_cleanup && !lp.catch_type_indices.is_empty() {
            table.actions.push(ActionEntry {
                type_filter: 0, // cleanup
                next_action_offset: 0, // end of chain
            });
        }

        lp_to_first_action.insert(lp.landing_pad_offset, chain_start);
    }

    // Build call site entries.
    for &(start, length, lp_offset) in call_site_ranges {
        let action_idx = if lp_offset == 0 {
            0 // no landing pad
        } else {
            lp_to_first_action.get(&lp_offset).copied().unwrap_or(0)
        };

        table.add_call_site(CallSiteEntry::new(
            start, length, lp_offset, action_idx,
        ));
    }

    table
}

/// Descriptor for a landing pad, used by `build_exception_table_from_pads`.
#[derive(Debug, Clone)]
pub struct LandingPadDesc {
    /// Offset of the landing pad from the function start.
    pub landing_pad_offset: u32,
    /// Type indices this landing pad catches. 0 = catch-all.
    pub catch_type_indices: Vec<u32>,
    /// Whether this landing pad runs cleanup (destructors/drops).
    pub is_cleanup: bool,
}

impl LandingPadDesc {
    /// Create a catch-all landing pad.
    pub fn catch_all(offset: u32) -> Self {
        Self {
            landing_pad_offset: offset,
            catch_type_indices: vec![0],
            is_cleanup: false,
        }
    }

    /// Create a typed catch landing pad.
    pub fn catch_typed(offset: u32, type_index: u32) -> Self {
        Self {
            landing_pad_offset: offset,
            catch_type_indices: vec![type_index],
            is_cleanup: false,
        }
    }

    /// Create a cleanup-only landing pad.
    pub fn cleanup(offset: u32) -> Self {
        Self {
            landing_pad_offset: offset,
            catch_type_indices: Vec::new(),
            is_cleanup: true,
        }
    }

    /// Create a landing pad that catches a type and also runs cleanup.
    pub fn catch_and_cleanup(offset: u32, type_index: u32) -> Self {
        Self {
            landing_pad_offset: offset,
            catch_type_indices: vec![type_index],
            is_cleanup: true,
        }
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // --- ULEB128 encoding tests ---

    #[test]
    fn test_encode_uleb128_zero() {
        assert_eq!(encode_uleb128(0), vec![0x00]);
    }

    #[test]
    fn test_encode_uleb128_single_byte() {
        assert_eq!(encode_uleb128(1), vec![0x01]);
        assert_eq!(encode_uleb128(63), vec![63]);
        assert_eq!(encode_uleb128(127), vec![0x7F]);
    }

    #[test]
    fn test_encode_uleb128_multi_byte() {
        // 128 = 0x80 => [0x80, 0x01]
        assert_eq!(encode_uleb128(128), vec![0x80, 0x01]);
        // 624485 = 0x98765 => [0xE5, 0x8E, 0x26]
        assert_eq!(encode_uleb128(624485), vec![0xE5, 0x8E, 0x26]);
    }

    #[test]
    fn test_encode_uleb128_large() {
        // 256 = 0x100 => [0x80, 0x02]
        assert_eq!(encode_uleb128(256), vec![0x80, 0x02]);
        // 16384 = 0x4000 => [0x80, 0x80, 0x01]
        assert_eq!(encode_uleb128(16384), vec![0x80, 0x80, 0x01]);
    }

    // --- SLEB128 encoding tests ---

    #[test]
    fn test_encode_sleb128_zero() {
        assert_eq!(encode_sleb128(0), vec![0x00]);
    }

    #[test]
    fn test_encode_sleb128_positive() {
        assert_eq!(encode_sleb128(1), vec![0x01]);
        assert_eq!(encode_sleb128(63), vec![63]);
        // 64 needs 2 bytes because bit 6 (sign bit) would be set in a single byte.
        assert_eq!(encode_sleb128(64), vec![0xC0, 0x00]);
    }

    #[test]
    fn test_encode_sleb128_negative() {
        // -1 => [0x7F]
        assert_eq!(encode_sleb128(-1), vec![0x7F]);
        // -8 => [0x78]
        assert_eq!(encode_sleb128(-8), vec![0x78]);
        // -64 => [0x40]
        assert_eq!(encode_sleb128(-64), vec![0x40]);
        // -65 => [0xBF, 0x7F]
        assert_eq!(encode_sleb128(-65), vec![0xBF, 0x7F]);
    }

    // --- DwEhPe tests ---

    #[test]
    fn test_dw_eh_pe_encoded_size() {
        assert_eq!(DwEhPe::UData2.encoded_size(), Some(2));
        assert_eq!(DwEhPe::UData4.encoded_size(), Some(4));
        assert_eq!(DwEhPe::SData4.encoded_size(), Some(4));
        assert_eq!(DwEhPe::UData8.encoded_size(), Some(8));
        assert_eq!(DwEhPe::AbsPtr.encoded_size(), None);
        assert_eq!(DwEhPe::Omit.encoded_size(), None);
    }

    #[test]
    fn test_dw_eh_pe_values() {
        assert_eq!(DwEhPe::AbsPtr as u8, 0x00);
        assert_eq!(DwEhPe::UData4 as u8, 0x03);
        assert_eq!(DwEhPe::Omit as u8, 0xFF);
    }

    // --- CallSiteEntry tests ---

    #[test]
    fn test_call_site_entry_constructors() {
        let cs = CallSiteEntry::new(0x10, 0x20, 0x100, 1);
        assert_eq!(cs.region_start, 0x10);
        assert_eq!(cs.region_length, 0x20);
        assert_eq!(cs.landing_pad, 0x100);
        assert_eq!(cs.action_idx, 1);

        let no_lp = CallSiteEntry::no_landing_pad(0x10, 0x20);
        assert_eq!(no_lp.landing_pad, 0);
        assert_eq!(no_lp.action_idx, 0);

        let cleanup = CallSiteEntry::cleanup(0x10, 0x20, 0x100);
        assert_eq!(cleanup.landing_pad, 0x100);
        assert_eq!(cleanup.action_idx, 0);
    }

    // --- ActionEntry tests ---

    #[test]
    fn test_action_entry_catch() {
        let action = ActionEntry::catch(1);
        assert_eq!(action.type_filter, 1);
        assert_eq!(action.next_action_offset, 0);
    }

    #[test]
    fn test_action_entry_cleanup() {
        let action = ActionEntry::cleanup();
        assert_eq!(action.type_filter, 0);
        assert_eq!(action.next_action_offset, 0);
    }

    #[test]
    fn test_action_entry_chain() {
        // An action chain: catch type 1, then cleanup.
        let mut action1 = ActionEntry::catch(1);
        action1.next_action_offset = 2; // points to next entry (2 bytes ahead)
        let action2 = ActionEntry::cleanup();

        let data = emit_action_table(&[action1, action2]);
        // action1: SLEB128(1) = [0x01], SLEB128(2) = [0x02]
        // action2: SLEB128(0) = [0x00], SLEB128(0) = [0x00]
        assert_eq!(data, vec![0x01, 0x02, 0x00, 0x00]);
    }

    // --- TypeInfo tests ---

    #[test]
    fn test_type_info_entries() {
        let ti = TypeInfo::new(42);
        assert_eq!(ti.type_info_index, 42);

        let catch_all = TypeInfo::catch_all();
        assert_eq!(catch_all.type_info_index, 0);
    }

    #[test]
    fn test_type_table_emission_order() {
        // Type infos are emitted in reverse order.
        let type_infos = vec![TypeInfo::new(1), TypeInfo::new(2), TypeInfo::new(3)];
        let data = emit_type_table(&type_infos);

        // Reverse order: 3, 2, 1 — each 4 bytes LE.
        assert_eq!(data.len(), 12);
        assert_eq!(u32::from_le_bytes(data[0..4].try_into().unwrap()), 3);
        assert_eq!(u32::from_le_bytes(data[4..8].try_into().unwrap()), 2);
        assert_eq!(u32::from_le_bytes(data[8..12].try_into().unwrap()), 1);
    }

    // --- ExceptionTable tests ---

    #[test]
    fn test_exception_table_new() {
        let table = ExceptionTable::new();
        assert!(table.is_empty());
        assert!(table.personality.is_none());
    }

    #[test]
    fn test_exception_table_with_cxx_personality() {
        let table = ExceptionTable::with_cxx_personality();
        assert_eq!(table.personality.as_deref(), Some("__gxx_personality_v0"));
    }

    #[test]
    fn test_exception_table_with_rust_personality() {
        let table = ExceptionTable::with_rust_personality();
        assert_eq!(table.personality.as_deref(), Some("__rust_eh_personality"));
    }

    #[test]
    fn test_exception_table_add_entries() {
        let mut table = ExceptionTable::new();

        table.add_call_site(CallSiteEntry::new(0, 16, 32, 1));
        assert_eq!(table.call_sites.len(), 1);

        let action_idx = table.add_action(ActionEntry::catch(1));
        assert_eq!(action_idx, 1);

        let type_idx = table.add_type_info(TypeInfo::new(42));
        assert_eq!(type_idx, 1);

        assert!(!table.is_empty());
    }

    #[test]
    fn test_exception_table_default() {
        let table = ExceptionTable::default();
        assert!(table.is_empty());
    }

    // --- LSDA generation tests ---

    #[test]
    fn test_empty_lsda() {
        // Empty exception table: no call sites, no actions, no type infos.
        let table = ExceptionTable::new();
        let lsda = generate_lsda(&table);

        // Header:
        // [0]: LPStart encoding = 0xFF (omit)
        // [1]: TType encoding = 0xFF (omit, no type table)
        // [2]: Call site encoding = 0x03 (udata4)
        // [3]: Call site table length = 0x00 (ULEB128, zero entries)
        assert_eq!(lsda.len(), 4);
        assert_eq!(lsda[0], 0xFF); // LPStart = omit
        assert_eq!(lsda[1], 0xFF); // TType = omit
        assert_eq!(lsda[2], 0x03); // call site encoding = udata4
        assert_eq!(lsda[3], 0x00); // call site table length = 0
    }

    #[test]
    fn test_lsda_header_layout() {
        // LSDA with call sites but no type table.
        let mut table = ExceptionTable::new();
        table.add_call_site(CallSiteEntry::cleanup(0, 16, 32));

        let lsda = generate_lsda(&table);

        // Header check:
        assert_eq!(lsda[0], 0xFF); // LPStart = omit
        assert_eq!(lsda[1], 0xFF); // TType = omit (no type infos)
        assert_eq!(lsda[2], 0x03); // call site encoding = udata4

        // Call site table length: one entry with udata4 encoding.
        // Each entry = 3 * 4 (udata4 fields) + ULEB128(action_idx).
        // action_idx = 0 => ULEB128 = [0x00] = 1 byte.
        // Total = 12 + 1 = 13 bytes.
        assert_eq!(lsda[3], 13); // call site table length
    }

    #[test]
    fn test_single_call_site_with_landing_pad() {
        let mut table = ExceptionTable::new();
        table.add_call_site(CallSiteEntry::new(0x10, 0x08, 0x50, 1));
        table.add_action(ActionEntry::catch(1));
        table.add_type_info(TypeInfo::new(100));

        let lsda = generate_lsda(&table);

        // Should have a non-empty LSDA with type table.
        assert!(lsda.len() > 4);

        // LPStart = omit.
        assert_eq!(lsda[0], 0xFF);

        // TType encoding should NOT be omit (we have type infos).
        assert_eq!(lsda[1], DwEhPe::UData4 as u8);

        // Verify the LSDA contains the call site data somewhere.
        // The call site region_start = 0x10 should appear as LE bytes.
        let region_start_bytes = 0x10u32.to_le_bytes();
        assert!(
            lsda.windows(4).any(|w| w == region_start_bytes),
            "LSDA should contain region_start bytes"
        );
    }

    #[test]
    fn test_multiple_call_sites() {
        let mut table = ExceptionTable::new();
        table.add_call_site(CallSiteEntry::new(0x00, 0x10, 0x80, 1));
        table.add_call_site(CallSiteEntry::new(0x10, 0x08, 0x90, 1));
        table.add_call_site(CallSiteEntry::no_landing_pad(0x18, 0x04));
        table.add_action(ActionEntry::catch(1));
        table.add_type_info(TypeInfo::new(1));

        let lsda = generate_lsda(&table);

        // Should produce a valid LSDA.
        assert!(lsda.len() > 4);
        assert_eq!(lsda[0], 0xFF); // LPStart = omit
    }

    #[test]
    fn test_cleanup_only() {
        // Landing pad with action_idx = 0 means cleanup-only (no catch clause).
        let mut table = ExceptionTable::new();
        table.add_call_site(CallSiteEntry::cleanup(0x00, 0x20, 0x100));

        let lsda = generate_lsda(&table);

        // No type table (no type_infos).
        assert_eq!(lsda[1], 0xFF); // TType = omit

        // Parse back the call site table.
        // Header: [0xFF, 0xFF, 0x03, <cs_len>]
        let cs_len = lsda[3] as usize;
        let cs_start = 4;
        let cs_data = &lsda[cs_start..cs_start + cs_len];

        // Call site: region_start(4) + region_length(4) + landing_pad(4) + action(ULEB128)
        assert!(cs_data.len() >= 13);

        // region_start = 0
        assert_eq!(u32::from_le_bytes(cs_data[0..4].try_into().unwrap()), 0);
        // region_length = 0x20
        assert_eq!(u32::from_le_bytes(cs_data[4..8].try_into().unwrap()), 0x20);
        // landing_pad = 0x100
        assert_eq!(u32::from_le_bytes(cs_data[8..12].try_into().unwrap()), 0x100);
        // action_idx = 0 (cleanup)
        assert_eq!(cs_data[12], 0x00);
    }

    #[test]
    fn test_lsda_with_type_table_offset() {
        // Verify the TType base offset is correctly computed when type infos exist.
        let mut table = ExceptionTable::new();
        table.add_call_site(CallSiteEntry::new(0, 8, 16, 1));
        table.add_action(ActionEntry::catch(1));
        table.add_type_info(TypeInfo::new(42));

        let lsda = generate_lsda(&table);

        // Header:
        // [0]: 0xFF (LPStart omit)
        // [1]: 0x03 (TType = udata4)
        // [2..]: ULEB128 TType base offset
        assert_eq!(lsda[0], 0xFF);
        assert_eq!(lsda[1], DwEhPe::UData4 as u8);

        // Decode the TType base offset (ULEB128 starting at byte 2).
        let (ttype_offset, ttype_offset_len) = decode_test_uleb128(&lsda[2..]);

        // After the TType base offset field, the remaining data is:
        // call_site_enc(1) + cs_length(ULEB128) + cs_data + action_data + type_data
        let after_ttype_offset = 2 + ttype_offset_len;
        let remaining_len = lsda.len() - after_ttype_offset;

        // TType base offset should equal the total remaining bytes.
        assert_eq!(
            ttype_offset as usize, remaining_len,
            "TType base offset ({}) should equal remaining LSDA size ({})",
            ttype_offset, remaining_len
        );
    }

    #[test]
    fn test_lsda_roundtrip_structure() {
        // Build a complete exception table and verify LSDA structural integrity.
        let mut table = ExceptionTable::with_cxx_personality();
        table.add_call_site(CallSiteEntry::new(0, 12, 24, 1));
        table.add_call_site(CallSiteEntry::cleanup(12, 8, 36));
        table.add_action(ActionEntry::catch(1));
        table.add_type_info(TypeInfo::new(7));

        let lsda = generate_lsda(&table);

        // Basic structural checks.
        assert!(lsda.len() > 10, "LSDA too short: {} bytes", lsda.len());
        assert_eq!(lsda[0], 0xFF); // LPStart omit
        assert_ne!(lsda[1], 0xFF); // TType NOT omit (has type infos)

        // The type info (7) should appear somewhere in the LSDA as a 4-byte LE value.
        let type_bytes = 7u32.to_le_bytes();
        assert!(
            lsda.windows(4).any(|w| w == type_bytes),
            "Type info value 7 should appear in LSDA"
        );
    }

    #[test]
    fn test_call_site_table_encoding() {
        // Verify the raw call site table bytes.
        let call_sites = vec![
            CallSiteEntry::new(0x04, 0x08, 0x20, 1),
        ];
        let data = emit_call_site_table(&call_sites, DwEhPe::UData4);

        // 3 udata4 fields + 1 ULEB128 action index.
        // region_start = 4, region_length = 8, landing_pad = 0x20, action = 1
        assert_eq!(u32::from_le_bytes(data[0..4].try_into().unwrap()), 4);
        assert_eq!(u32::from_le_bytes(data[4..8].try_into().unwrap()), 8);
        assert_eq!(u32::from_le_bytes(data[8..12].try_into().unwrap()), 0x20);
        assert_eq!(data[12], 1); // action index = 1 (ULEB128)
        assert_eq!(data.len(), 13);
    }

    #[test]
    fn test_empty_call_site_table() {
        let data = emit_call_site_table(&[], DwEhPe::UData4);
        assert!(data.is_empty());
    }

    #[test]
    fn test_empty_action_table() {
        let data = emit_action_table(&[]);
        assert!(data.is_empty());
    }

    #[test]
    fn test_empty_type_table() {
        let data = emit_type_table(&[]);
        assert!(data.is_empty());
    }

    #[test]
    fn test_action_table_single_entry() {
        let data = emit_action_table(&[ActionEntry::catch(2)]);
        // SLEB128(2) = [0x02], SLEB128(0) = [0x00]
        assert_eq!(data, vec![0x02, 0x00]);
    }

    #[test]
    fn test_action_table_negative_filter() {
        // Exception specification filter (negative type_filter).
        let action = ActionEntry {
            type_filter: -1,
            next_action_offset: 0,
        };
        let data = emit_action_table(&[action]);
        // SLEB128(-1) = [0x7F], SLEB128(0) = [0x00]
        assert_eq!(data, vec![0x7F, 0x00]);
    }

    #[test]
    fn test_type_table_single_entry() {
        let data = emit_type_table(&[TypeInfo::new(42)]);
        // Single entry: 42 as u32 LE.
        assert_eq!(data, 42u32.to_le_bytes().to_vec());
    }

    #[test]
    fn test_multiple_call_sites_byte_layout() {
        let call_sites = vec![
            CallSiteEntry::new(0, 4, 8, 0),
            CallSiteEntry::new(4, 4, 0, 0),
        ];
        let data = emit_call_site_table(&call_sites, DwEhPe::UData4);

        // Entry 1: 0(4) + 4(4) + 8(4) + ULEB(0) = 13 bytes
        // Entry 2: 4(4) + 4(4) + 0(4) + ULEB(0) = 13 bytes
        assert_eq!(data.len(), 26);

        // Verify entry 2 region_start = 4.
        assert_eq!(u32::from_le_bytes(data[13..17].try_into().unwrap()), 4);
        // Verify entry 2 landing_pad = 0.
        assert_eq!(u32::from_le_bytes(data[21..25].try_into().unwrap()), 0);
    }

    // --- Helper: ULEB128 decoder for testing ---

    /// Decode a ULEB128 value from a byte slice. Returns (value, bytes_consumed).
    fn decode_test_uleb128(data: &[u8]) -> (u64, usize) {
        let mut result: u64 = 0;
        let mut shift = 0;
        for (i, &byte) in data.iter().enumerate() {
            result |= ((byte & 0x7F) as u64) << shift;
            shift += 7;
            if byte & 0x80 == 0 {
                return (result, i + 1);
            }
        }
        (result, data.len())
    }

    // =======================================================================
    // EH Bridge tests
    // =======================================================================

    #[test]
    fn test_build_exception_table_empty() {
        let params = EhBridgeParams {
            personality_symbol: None,
            call_sites: Vec::new(),
            actions: Vec::new(),
            type_indices: Vec::new(),
        };
        let table = build_exception_table(&params);
        assert!(table.is_empty());
        assert!(table.personality.is_none());
    }

    #[test]
    fn test_build_exception_table_cxx_personality() {
        let params = EhBridgeParams {
            personality_symbol: Some("__gxx_personality_v0".to_string()),
            call_sites: vec![
                EhCallSite {
                    start_offset: 0x10,
                    length: 0x08,
                    landing_pad_offset: 0x50,
                    action_index: 1,
                },
            ],
            actions: vec![
                EhAction { type_filter: 1, next_offset: 0 },
            ],
            type_indices: vec![42],
        };
        let table = build_exception_table(&params);

        assert_eq!(table.personality.as_deref(), Some("__gxx_personality_v0"));
        assert_eq!(table.call_sites.len(), 1);
        assert_eq!(table.call_sites[0].region_start, 0x10);
        assert_eq!(table.call_sites[0].region_length, 0x08);
        assert_eq!(table.call_sites[0].landing_pad, 0x50);
        assert_eq!(table.call_sites[0].action_idx, 1);
        assert_eq!(table.actions.len(), 1);
        assert_eq!(table.actions[0].type_filter, 1);
        assert_eq!(table.type_infos.len(), 1);
        assert_eq!(table.type_infos[0].type_info_index, 42);
    }

    #[test]
    fn test_build_exception_table_rust_personality() {
        let params = EhBridgeParams {
            personality_symbol: Some("__rust_eh_personality".to_string()),
            call_sites: vec![
                EhCallSite {
                    start_offset: 0,
                    length: 0x20,
                    landing_pad_offset: 0x30,
                    action_index: 0, // cleanup only
                },
            ],
            actions: Vec::new(),
            type_indices: Vec::new(),
        };
        let table = build_exception_table(&params);

        assert_eq!(table.personality.as_deref(), Some("__rust_eh_personality"));
        assert_eq!(table.call_sites.len(), 1);
        assert_eq!(table.call_sites[0].action_idx, 0);
        assert!(table.actions.is_empty());
        assert!(table.type_infos.is_empty());
    }

    #[test]
    fn test_build_exception_table_generates_valid_lsda() {
        let params = EhBridgeParams {
            personality_symbol: Some("__gxx_personality_v0".to_string()),
            call_sites: vec![
                EhCallSite {
                    start_offset: 0,
                    length: 16,
                    landing_pad_offset: 32,
                    action_index: 1,
                },
            ],
            actions: vec![
                EhAction { type_filter: 1, next_offset: 0 },
            ],
            type_indices: vec![100],
        };
        let table = build_exception_table(&params);
        let lsda = generate_lsda(&table);

        // Should produce a valid non-empty LSDA.
        assert!(lsda.len() > 4);
        assert_eq!(lsda[0], 0xFF); // LPStart omit
        assert_ne!(lsda[1], 0xFF); // TType present (has type infos)
    }

    #[test]
    fn test_build_exception_table_action_chain() {
        // Two catch types chained together.
        let params = EhBridgeParams {
            personality_symbol: Some("__gxx_personality_v0".to_string()),
            call_sites: vec![
                EhCallSite {
                    start_offset: 0,
                    length: 8,
                    landing_pad_offset: 16,
                    action_index: 1,
                },
            ],
            actions: vec![
                EhAction { type_filter: 1, next_offset: 2 },  // catch type 1, chain to next
                EhAction { type_filter: 2, next_offset: 0 },  // catch type 2, end of chain
            ],
            type_indices: vec![10, 20],
        };
        let table = build_exception_table(&params);

        assert_eq!(table.actions.len(), 2);
        assert_eq!(table.actions[0].type_filter, 1);
        assert_eq!(table.actions[0].next_action_offset, 2);
        assert_eq!(table.actions[1].type_filter, 2);
        assert_eq!(table.actions[1].next_action_offset, 0);
    }

    #[test]
    fn test_build_exception_table_multiple_call_sites() {
        let params = EhBridgeParams {
            personality_symbol: Some("__gxx_personality_v0".to_string()),
            call_sites: vec![
                EhCallSite {
                    start_offset: 0x00,
                    length: 0x10,
                    landing_pad_offset: 0x80,
                    action_index: 1,
                },
                EhCallSite {
                    start_offset: 0x10,
                    length: 0x08,
                    landing_pad_offset: 0x90,
                    action_index: 1,
                },
                EhCallSite {
                    start_offset: 0x18,
                    length: 0x04,
                    landing_pad_offset: 0,
                    action_index: 0,
                },
            ],
            actions: vec![
                EhAction { type_filter: 1, next_offset: 0 },
            ],
            type_indices: vec![1],
        };
        let table = build_exception_table(&params);
        assert_eq!(table.call_sites.len(), 3);

        // Third call site has no landing pad.
        assert_eq!(table.call_sites[2].landing_pad, 0);
        assert_eq!(table.call_sites[2].action_idx, 0);

        let lsda = generate_lsda(&table);
        assert!(lsda.len() > 4);
    }

    // =======================================================================
    // build_exception_table_from_pads tests
    // =======================================================================

    #[test]
    fn test_from_pads_cleanup_only() {
        let table = build_exception_table_from_pads(
            "__rust_eh_personality",
            &[LandingPadDesc::cleanup(0x100)],
            &[(0x00, 0x20, 0x100)],
        );

        assert_eq!(table.personality.as_deref(), Some("__rust_eh_personality"));
        assert_eq!(table.call_sites.len(), 1);
        assert_eq!(table.call_sites[0].landing_pad, 0x100);
        assert_eq!(table.call_sites[0].action_idx, 0); // cleanup = action 0
        assert!(table.actions.is_empty()); // no action chain for cleanup-only
        assert!(table.type_infos.is_empty());

        let lsda = generate_lsda(&table);
        assert!(lsda.len() > 4);
    }

    #[test]
    fn test_from_pads_catch_typed() {
        let table = build_exception_table_from_pads(
            "__gxx_personality_v0",
            &[LandingPadDesc::catch_typed(0x50, 1)],
            &[(0x10, 0x08, 0x50)],
        );

        assert_eq!(table.call_sites.len(), 1);
        assert_eq!(table.call_sites[0].action_idx, 1); // first action
        assert_eq!(table.actions.len(), 1);
        assert_eq!(table.actions[0].type_filter, 1); // type index 1
        assert_eq!(table.actions[0].next_action_offset, 0); // end of chain
        assert_eq!(table.type_infos.len(), 1);
        assert_eq!(table.type_infos[0].type_info_index, 1);
    }

    #[test]
    fn test_from_pads_catch_all() {
        let table = build_exception_table_from_pads(
            "__gxx_personality_v0",
            &[LandingPadDesc::catch_all(0x40)],
            &[(0x00, 0x10, 0x40)],
        );

        assert_eq!(table.call_sites.len(), 1);
        assert_ne!(table.call_sites[0].action_idx, 0); // non-zero = has action
        assert_eq!(table.actions.len(), 1);
        assert_eq!(table.actions[0].type_filter, 0); // catch-all
    }

    #[test]
    fn test_from_pads_catch_and_cleanup() {
        let table = build_exception_table_from_pads(
            "__gxx_personality_v0",
            &[LandingPadDesc::catch_and_cleanup(0x60, 1)],
            &[(0x00, 0x10, 0x60)],
        );

        // Should have catch action + cleanup action chained.
        assert_eq!(table.actions.len(), 2);
        assert_eq!(table.actions[0].type_filter, 1); // catch type 1
        assert_ne!(table.actions[0].next_action_offset, 0); // chains to cleanup
        assert_eq!(table.actions[1].type_filter, 0); // cleanup
        assert_eq!(table.actions[1].next_action_offset, 0); // end
    }

    #[test]
    fn test_from_pads_multiple_landing_pads() {
        let table = build_exception_table_from_pads(
            "__gxx_personality_v0",
            &[
                LandingPadDesc::catch_typed(0x80, 1),
                LandingPadDesc::cleanup(0xC0),
            ],
            &[
                (0x00, 0x10, 0x80),
                (0x10, 0x08, 0xC0),
                (0x18, 0x04, 0),    // no landing pad
            ],
        );

        assert_eq!(table.call_sites.len(), 3);
        // First call site -> catch landing pad
        assert_ne!(table.call_sites[0].action_idx, 0);
        // Second call site -> cleanup landing pad
        assert_eq!(table.call_sites[1].action_idx, 0);
        // Third call site -> no landing pad
        assert_eq!(table.call_sites[2].landing_pad, 0);
    }

    #[test]
    fn test_from_pads_end_to_end_lsda() {
        // Build a complete C++-style exception table and generate LSDA.
        let table = build_exception_table_from_pads(
            "__gxx_personality_v0",
            &[
                LandingPadDesc::catch_typed(0x50, 1),
                LandingPadDesc::catch_typed(0x80, 2),
            ],
            &[
                (0x00, 0x10, 0x50),
                (0x10, 0x08, 0x80),
            ],
        );

        let lsda = generate_lsda(&table);
        assert!(lsda.len() > 10);
        assert_eq!(lsda[0], 0xFF); // LPStart omit
        assert_ne!(lsda[1], 0xFF); // TType present

        // Type table should have 2 entries.
        assert_eq!(table.type_infos.len(), 2);
    }

    #[test]
    fn test_landing_pad_desc_constructors() {
        let ca = LandingPadDesc::catch_all(0x40);
        assert_eq!(ca.landing_pad_offset, 0x40);
        assert_eq!(ca.catch_type_indices, vec![0]);
        assert!(!ca.is_cleanup);

        let ct = LandingPadDesc::catch_typed(0x50, 3);
        assert_eq!(ct.catch_type_indices, vec![3]);
        assert!(!ct.is_cleanup);

        let cl = LandingPadDesc::cleanup(0x60);
        assert!(cl.catch_type_indices.is_empty());
        assert!(cl.is_cleanup);

        let cc = LandingPadDesc::catch_and_cleanup(0x70, 5);
        assert_eq!(cc.catch_type_indices, vec![5]);
        assert!(cc.is_cleanup);
    }
}
