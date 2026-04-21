// llvm2-llvm-import / parser.rs
//
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Copyright 2026 Dropbox, Inc. | License: Apache-2.0
//
// Hand-written line-oriented LLVM IR text parser. See the crate README
// for the supported subset and rationale for writing this instead of
// using the `llvm-ir` crate (system-LLVM version drift).
//
// Parsing strategy:
//   * Pre-process: strip comments (`; ...`), `!dbg !N` / `!tbaa !N`
//     attachments, and normalise whitespace.
//   * Classify each line:
//       - module header / directives (target triple, datalayout)
//       - global declaration (`@name = ... constant [N x i8] c"..."`)
//       - function declaration (`declare` ...)
//       - function definition (`define` ...)
//       - inside a function: block label, terminator, instruction,
//         or metadata.
//   * Translate supported constructs directly to `tmir::Inst`. Fail
//     fast with `Error::Unsupported("<description>")` for anything
//     outside the subset.
//
// This is deliberately narrow. The goal is to unblock the WS2 driver,
// not to implement a full LLVM frontend. See the expansion plan in the
// README for the sequence of features to add.

use std::collections::HashMap;
use std::fs;
use std::path::Path;

use tmir::{
    BinOp, BlockId, CallingConv, CastOp, Constant, FCmpOp, FuncId, FuncTy, FuncTyId, Function,
    Global, ICmpOp, InstrNode, Linkage, Module, SwitchCase, Ty, UnOp, ValueId,
    inst::Inst,
};

use crate::{Error, Result};

// --------------------------------------------------------------------------
// Public entry points
// --------------------------------------------------------------------------

/// Read the file at `path` and import its LLVM IR text into a `tmir::Module`.
pub fn import_module(path: &Path) -> Result<tmir::Module> {
    let text = fs::read_to_string(path)?;
    let name = path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("module")
        .to_string();
    import_text(&text, &name)
}

/// Import an in-memory LLVM IR text string into a `tmir::Module`.
pub fn import_text(text: &str, module_name: &str) -> Result<tmir::Module> {
    let mut parser = Parser::new(text, module_name.to_string());
    parser.parse()?;
    Ok(parser.module)
}

// --------------------------------------------------------------------------
// Parser state
// --------------------------------------------------------------------------

struct Parser {
    module: Module,
    /// Map from LLVM function name (without `@`) to `FuncId`.
    func_ids: HashMap<String, FuncId>,
    /// Map from LLVM function name to its `FuncTyId`.
    func_tys: HashMap<String, FuncTyId>,
    /// Map from LLVM global name (without `@`) to the index in
    /// `module.globals`. Used to flag the "string head" GEP pattern.
    globals: HashMap<String, usize>,
    /// Named LLVM aggregate layouts keyed by the full `%name` spelling.
    struct_layouts: HashMap<String, StructLayout>,
    /// Lines of the input, for error context.
    lines: Vec<String>,
}

#[allow(dead_code)]
#[derive(Clone, Debug)]
struct StructLayout {
    /// (field_ty, byte_offset)
    fields: Vec<(Ty, u64)>,
    /// Total size rounded up to the struct alignment.
    size: u64,
    /// Maximum field alignment, at least 1.
    align: u64,
}

/// Per-function scratch state: SSA value map, block map, in-progress
/// block list.
struct FuncScratch {
    /// LLVM SSA name (without `%`) -> tMIR ValueId.
    value_map: HashMap<String, ValueId>,
    /// LLVM block label (without `%`) -> tMIR BlockId.
    block_map: HashMap<String, BlockId>,
    /// Block bodies indexed by BlockId order of appearance.
    blocks: Vec<tmir::Block>,
    /// Next fresh ValueId.
    next_value: u32,
    /// Current block being built (index into blocks).
    current: Option<usize>,
}

impl FuncScratch {
    fn new() -> Self {
        Self {
            value_map: HashMap::new(),
            block_map: HashMap::new(),
            blocks: Vec::new(),
            next_value: 0,
            current: None,
        }
    }

    fn fresh_value(&mut self) -> ValueId {
        let v = ValueId::new(self.next_value);
        self.next_value += 1;
        v
    }

    fn intern_value(&mut self, name: &str) -> ValueId {
        if let Some(v) = self.value_map.get(name) {
            return *v;
        }
        let v = self.fresh_value();
        self.value_map.insert(name.to_string(), v);
        v
    }

    fn intern_block(&mut self, label: &str) -> BlockId {
        if let Some(b) = self.block_map.get(label) {
            return *b;
        }
        let id = BlockId::new(self.blocks.len() as u32);
        self.blocks.push(tmir::Block::new(id));
        self.block_map.insert(label.to_string(), id);
        id
    }

    fn push_inst(&mut self, node: InstrNode) {
        if let Some(idx) = self.current {
            self.blocks[idx].body.push(node);
        }
    }

    fn set_current(&mut self, id: BlockId) {
        self.current = Some(id.as_usize());
    }
}

// --------------------------------------------------------------------------
// Parsing
// --------------------------------------------------------------------------

impl Parser {
    fn new(text: &str, module_name: String) -> Self {
        Self {
            module: Module::new(module_name),
            func_ids: HashMap::new(),
            func_tys: HashMap::new(),
            globals: HashMap::new(),
            struct_layouts: HashMap::new(),
            lines: text.lines().map(|s| s.to_string()).collect(),
        }
    }

    fn err_unsupported(&self, what: &str) -> Error {
        Error::Unsupported(what.to_string())
    }

    fn err_parse(&self, line: usize, msg: &str) -> Error {
        Error::Parse {
            line,
            message: msg.to_string(),
        }
    }

    fn parse_ty_ctx(&self, s: &str) -> Result<Ty> {
        match parse_ty(s) {
            Ok(ty) => Ok(ty),
            Err(Error::Unsupported(_)) if s.trim().starts_with('%') => self
                .struct_layouts
                .contains_key(s.trim())
                .then_some(Ty::Ptr)
                .ok_or_else(|| self.err_unsupported(&format!("type `{}`", s.trim()))),
            Err(e) => Err(e),
        }
    }

    fn parse(&mut self) -> Result<()> {
        // Snapshot lines so we can iterate with look-ahead for function
        // bodies without fighting the borrow checker. Strip comments and
        // metadata attachments up front.
        let lines: Vec<(usize, String)> = self
            .lines
            .iter()
            .enumerate()
            .map(|(i, l)| (i + 1, strip_line(l)))
            .collect();

        let mut i = 0;
        while i < lines.len() {
            let (lineno, raw) = &lines[i];
            let line = raw.trim();
            if line.is_empty() {
                i += 1;
                continue;
            }
            if line.starts_with("target ") || line.starts_with("source_filename") {
                // Informational only.
                i += 1;
                continue;
            }
            if line.starts_with("module asm") {
                return Err(self.err_unsupported("inline module asm"));
            }
            if line.starts_with("attributes ") {
                // attribute group decl — ignore.
                i += 1;
                continue;
            }
            if line.starts_with('!') {
                // Metadata definition — ignore.
                i += 1;
                continue;
            }
            if line.starts_with('@') {
                self.parse_global(line, *lineno)?;
                i += 1;
                continue;
            }
            if line.starts_with("declare") {
                self.parse_declare(line, *lineno)?;
                i += 1;
                continue;
            }
            if line.starts_with("define") {
                // Consume the define ... { line plus the body up to the
                // matching closing `}`.
                let (end, body) = collect_function_body(&lines, i, *lineno)?;
                self.parse_define(line, &body, *lineno)?;
                i = end + 1;
                continue;
            }
            if line.starts_with('%') && line.contains("= type") {
                self.parse_struct_type_def(line, *lineno)?;
                i += 1;
                continue;
            }
            // Unknown top-level line; most commonly `%struct.X = type ...`
            if line.contains("= type") {
                return Err(self.err_unsupported("named struct types"));
            }
            return Err(self.err_parse(*lineno, &format!("unexpected top-level line: {}", line)));
        }

        Ok(())
    }

    // --- Globals -----------------------------------------------------------

    fn parse_struct_type_def(&mut self, line: &str, lineno: usize) -> Result<()> {
        let (name, rest) = split_eq(line).ok_or_else(|| self.err_parse(lineno, "bad type def"))?;
        let name = name.trim();
        if !name.starts_with('%') {
            return Err(self.err_unsupported("named type definition without `%name`"));
        }

        let body = rest
            .trim()
            .strip_prefix("type")
            .ok_or_else(|| self.err_parse(lineno, "type def missing `type` keyword"))?
            .trim();

        if !body.starts_with('{') || !body.ends_with('}') {
            return Err(self.err_unsupported(&format!(
                "named type `{}` with non-struct body `{}`",
                name, body
            )));
        }

        let inner = &body[1..body.len() - 1];
        let mut fields = Vec::new();
        let mut offset = 0u64;
        let mut struct_align = 1u64;
        if !inner.trim().is_empty() {
            for field_str in split_call_args(inner) {
                let ty = parse_ty(field_str.trim())?;
                let (field_size, field_align) = scalar_layout(&ty).ok_or_else(|| {
                    self.err_unsupported(&format!(
                        "named struct field type `{}` in `{}`",
                        field_str.trim(),
                        name
                    ))
                })?;
                offset = align_up(offset, field_align);
                fields.push((ty, offset));
                offset += field_size;
                struct_align = struct_align.max(field_align);
            }
        }

        let layout = StructLayout {
            fields,
            size: align_up(offset, struct_align),
            align: struct_align,
        };
        self.struct_layouts.insert(name.to_string(), layout);
        Ok(())
    }

    fn parse_string_global(
        &mut self,
        name: String,
        rest: &str,
        lineno: usize,
        mutable: bool,
        linkage: Linkage,
    ) -> Result<()> {
        let ty_start = rest
            .find('[')
            .ok_or_else(|| self.err_unsupported("global without [N x T] type"))?;
        let ty_end = rest[ty_start..]
            .find(']')
            .ok_or_else(|| self.err_parse(lineno, "unterminated global type"))?;
        let ty_body = &rest[ty_start + 1..ty_start + ty_end];
        if !ty_body.contains("x i8") {
            return Err(self.err_unsupported(&format!(
                "non-string global @{} (type [{}] not [N x i8])",
                name, ty_body
            )));
        }

        let init_start = rest[ty_start + ty_end..]
            .find("c\"")
            .ok_or_else(|| self.err_unsupported("global with non-c-string initializer"))?;
        let init_tail = &rest[ty_start + ty_end + init_start + 2..];
        let close = find_ll_string_end(init_tail)
            .ok_or_else(|| self.err_parse(lineno, "unterminated c-string"))?;
        let raw = &init_tail[..close];
        let bytes = decode_ll_string(raw);

        let elems: Vec<Constant> = bytes.iter().map(|b| Constant::Int(*b as i128)).collect();
        let idx = self.module.globals.len();
        self.module.globals.push(Global {
            name: name.clone(),
            ty: Ty::Ptr,
            mutable,
            initializer: Some(Constant::Aggregate(elems)),
            linkage,
        });
        self.globals.insert(name, idx);
        Ok(())
    }

    fn parse_scalar_global_initializer(&self, ty: &Ty, tok: &str) -> Result<Constant> {
        let tok = tok.trim();
        match ty {
            Ty::Bool | Ty::I8 | Ty::I16 | Ty::I32 | Ty::I64 | Ty::I128 => {
                if tok == "zeroinitializer" || tok == "null" {
                    return Ok(Constant::Int(0));
                }
                if matches!(ty, Ty::Bool) && tok == "true" {
                    return Ok(Constant::Int(1));
                }
                if matches!(ty, Ty::Bool) && tok == "false" {
                    return Ok(Constant::Int(0));
                }
                parse_int_literal(tok)
                    .map(Constant::Int)
                    .ok_or_else(|| self.err_unsupported(&format!("scalar global initializer `{}`", tok)))
            }
            Ty::F32 | Ty::F64 => {
                if tok == "zeroinitializer" {
                    return Ok(Constant::Float(0.0));
                }
                match parse_fp_literal(tok).ok_or_else(|| {
                    self.err_unsupported(&format!("scalar global initializer `{}`", tok))
                })? {
                    FpLit::Double(d) => Ok(Constant::Float(d)),
                    FpLit::Extended(tag) => Err(self.err_unsupported(&format!(
                        "extended-precision float literal `0x{}...` (tMIR only has f32/f64)",
                        tag
                    ))),
                }
            }
            Ty::Ptr => {
                if tok == "null" || tok == "zeroinitializer" {
                    Ok(Constant::Int(0))
                } else {
                    Err(self.err_unsupported(&format!(
                        "pointer global initializer `{}`",
                        tok
                    )))
                }
            }
            _ => Err(self.err_unsupported(&format!(
                "non-scalar global initializer for type `{:?}`",
                ty
            ))),
        }
    }

    fn parse_global(&mut self, line: &str, lineno: usize) -> Result<()> {
        let (name, rest) = split_eq(line).ok_or_else(|| self.err_parse(lineno, "bad global"))?;
        let name = name.trim_start_matches('@').trim().to_string();
        let lower = rest.to_lowercase();
        let linkage = parse_linkage(&lower);
        let (mutable, after_storage) = split_global_storage(rest).ok_or_else(|| {
            self.err_unsupported(&format!(
                "global @{} without `global`/`constant` storage class",
                name
            ))
        })?;

        if after_storage.trim_start().starts_with('[') {
            return self.parse_string_global(name, rest, lineno, mutable, linkage);
        }

        let init_part = after_storage.split(',').next().unwrap_or("").trim();
        let (ty_str, value_str) = split_ty_operand(init_part)?;
        let ty = parse_ty(&ty_str)?;
        if !is_scalar_global_ty(&ty) {
            return Err(self.err_unsupported(&format!(
                "non-scalar global @{} (type `{}`)",
                name, ty_str
            )));
        }
        let initializer = self.parse_scalar_global_initializer(&ty, &value_str)?;
        let idx = self.module.globals.len();
        self.module.globals.push(Global {
            name: name.clone(),
            ty,
            mutable,
            initializer: Some(initializer),
            linkage,
        });
        self.globals.insert(name, idx);
        Ok(())
    }

    // --- Declare / define --------------------------------------------------

    fn parse_declare(&mut self, line: &str, lineno: usize) -> Result<()> {
        let sig = parse_function_signature(line, lineno, /*is_define=*/ false)?;
        self.register_function(sig)?;
        Ok(())
    }

    fn parse_define(
        &mut self,
        header: &str,
        body: &[(usize, String)],
        lineno: usize,
    ) -> Result<()> {
        // Strip trailing "{" from header.
        let header = header.trim_end_matches('{').trim();
        let sig = parse_function_signature(header, lineno, /*is_define=*/ true)?;
        let fid = self.register_function(sig.clone())?;

        // Walk the body, assigning ValueIds and BlockIds.
        let mut scratch = FuncScratch::new();

        // Entry block and its parameters.
        let entry_id = scratch.intern_block("entry");
        // Seed %0, %1, ... for parameters in order.
        for (i, (arg_name, ty)) in sig.params.iter().enumerate() {
            let aname = arg_name
                .clone()
                .unwrap_or_else(|| format!("__param_{}", i));
            let v = scratch.intern_value(&aname);
            scratch.blocks[entry_id.as_usize()].params.push((v, ty.clone()));
        }
        scratch.set_current(entry_id);

        let mut bi = 0usize;
        while bi < body.len() {
            let (ln, raw) = &body[bi];
            let line = raw.trim();
            if line.is_empty() {
                bi += 1;
                continue;
            }
            if line == "{" {
                bi += 1;
                continue;
            }
            if line == "}" {
                break;
            }
            // Block label: `foo:` (possibly with preds comment)
            if let Some(label) = parse_block_label(line) {
                let id = scratch.intern_block(&label);
                scratch.set_current(id);
                bi += 1;
                continue;
            }
            // A `switch` instruction's case-list spans multiple physical
            // lines: header `switch i32 %x, label %d [` followed by one
            // `i32 K, label %L` per line and a closing `]`. Collect the
            // whole thing into a single string before parsing.
            //
            // Detect the switch by leading opcode (switch is a terminator
            // with no result, so it cannot appear on the RHS of `=`).
            let opcode = line.split_whitespace().next().unwrap_or("");
            if opcode == "switch" {
                let mut collected = String::new();
                collected.push_str(line);
                let start_ln = *ln;
                let mut end = bi;
                // After seeing `[`, keep consuming lines until we see the
                // matching `]` (we don't support nested brackets in a
                // switch case list — LLVM doesn't produce them).
                let mut saw_open = collected.contains('[');
                let mut closed = saw_open && collected.contains(']');
                while !closed && end + 1 < body.len() {
                    end += 1;
                    let next = body[end].1.trim();
                    if next.is_empty() {
                        continue;
                    }
                    collected.push(' ');
                    collected.push_str(next);
                    if !saw_open && collected.contains('[') {
                        saw_open = true;
                    }
                    if saw_open && collected.contains(']') {
                        closed = true;
                    }
                }
                if !closed {
                    return Err(self.err_parse(
                        start_ln,
                        "switch: unterminated case list (missing `]`)",
                    ));
                }
                self.parse_switch(&collected, start_ln, &mut scratch)?;
                bi = end + 1;
                continue;
            }
            self.parse_body_line(line, *ln, &mut scratch)?;
            bi += 1;
        }

        // Install blocks into the function. Every block must have a
        // terminator; if not, that's a bug in the importer (all clang -O0
        // blocks end in ret/br/unreachable).
        for b in &scratch.blocks {
            if b.terminator().is_none() {
                return Err(Error::Unsupported(format!(
                    "block in @{} has no terminator (phi or fallthrough not supported)",
                    sig.name,
                )));
            }
        }

        let mut func = Function::new(fid, sig.name.clone(), self.func_tys[&sig.name], entry_id);
        func.blocks = scratch.blocks;
        func.calling_conv = CallingConv::C;
        func.linkage = if sig.internal {
            Linkage::Internal
        } else {
            Linkage::External
        };
        self.module.add_function(func);
        Ok(())
    }

    fn register_function(&mut self, sig: FuncSignature) -> Result<FuncId> {
        if let Some(existing) = self.func_ids.get(&sig.name) {
            return Ok(*existing);
        }
        let ft = FuncTy {
            params: sig.params.iter().map(|(_, t)| t.clone()).collect(),
            returns: sig
                .ret
                .as_ref()
                .map(|t| vec![t.clone()])
                .unwrap_or_default(),
            is_vararg: sig.is_vararg,
        };
        let ftid = self.module.add_func_type(ft);
        let fid = FuncId::new(self.func_ids.len() as u32);
        self.func_ids.insert(sig.name.clone(), fid);
        self.func_tys.insert(sig.name, ftid);
        Ok(fid)
    }

    fn emit_i64_const(&self, value: i128, f: &mut FuncScratch) -> ValueId {
        let v = f.fresh_value();
        f.push_inst(
            InstrNode::new(Inst::Const {
                ty: Ty::I64,
                value: Constant::Int(value),
            })
            .with_result(v),
        );
        v
    }

    fn coerce_int_to_i64(
        &self,
        value: ValueId,
        ty: &Ty,
        f: &mut FuncScratch,
    ) -> Result<ValueId> {
        match ty {
            Ty::I64 => Ok(value),
            Ty::Bool => {
                let widened = f.fresh_value();
                f.push_inst(
                    InstrNode::new(Inst::Cast {
                        op: CastOp::ZExt,
                        src_ty: Ty::Bool,
                        dst_ty: Ty::I64,
                        operand: value,
                    })
                    .with_result(widened),
                );
                Ok(widened)
            }
            Ty::I8 | Ty::I16 | Ty::I32 => {
                let widened = f.fresh_value();
                f.push_inst(
                    InstrNode::new(Inst::Cast {
                        op: CastOp::SExt,
                        src_ty: ty.clone(),
                        dst_ty: Ty::I64,
                        operand: value,
                    })
                    .with_result(widened),
                );
                Ok(widened)
            }
            Ty::I128 => Err(self.err_unsupported(
                "dynamic struct alloca count with i128 type",
            )),
            other => Err(self.err_unsupported(&format!(
                "non-integer struct alloca count type `{:?}`",
                other
            ))),
        }
    }

    // --- Instructions ------------------------------------------------------

    fn parse_body_line(
        &mut self,
        line: &str,
        lineno: usize,
        f: &mut FuncScratch,
    ) -> Result<()> {
        // Form: `  %name = <opcode> ...`  or  `  <opcode-without-result> ...`
        let (result_name, rest) = if let Some((lhs, rhs)) = split_eq_not_icmp(line) {
            let lhs = lhs.trim();
            if !lhs.starts_with('%') {
                return Err(self.err_parse(lineno, "LHS of `=` must be %name"));
            }
            (Some(lhs.trim_start_matches('%').to_string()), rhs.trim())
        } else {
            (None, line)
        };

        // Dispatch on the leading opcode keyword.
        let opcode = rest.split_whitespace().next().unwrap_or("");
        match opcode {
            "ret" => self.parse_ret(rest, lineno, f),
            "br" => self.parse_br(rest, lineno, f),
            "unreachable" => {
                f.push_inst(InstrNode::new(Inst::Unreachable));
                Ok(())
            }
            "add" | "sub" | "mul" | "and" | "or" | "xor" | "shl" | "lshr" | "ashr"
            | "sdiv" | "udiv" | "srem" | "urem" => {
                self.parse_binop(opcode, rest, result_name, lineno, f)
            }
            "icmp" => self.parse_icmp(rest, result_name, lineno, f),
            "alloca" => self.parse_alloca(rest, result_name, lineno, f),
            "load" => self.parse_load(rest, result_name, lineno, f),
            "store" => self.parse_store(rest, lineno, f),
            "call" => self.parse_call(rest, result_name, lineno, f),
            "getelementptr" => self.parse_gep(rest, result_name, lineno, f),
            "sext" | "zext" | "trunc" | "bitcast" | "ptrtoint" | "inttoptr"
            | "sitofp" | "fptosi" | "uitofp" | "fptoui" | "fpext" | "fptrunc" => {
                self.parse_cast(opcode, rest, result_name, lineno, f)
            }
            "select" => self.parse_select(rest, result_name, lineno, f),
            "phi" => Err(self
                .err_unsupported("phi nodes (importer needs SSA-deconstruction — see README)")),
            "switch" => Err(self.err_parse(
                lineno,
                "internal: switch should be collected multi-line by parse_define",
            )),
            "invoke" => Err(self.err_unsupported("invoke / exceptions")),
            "landingpad" | "resume" => Err(self.err_unsupported("exception handling")),
            "fadd" | "fsub" | "fmul" | "fdiv" | "frem" => {
                self.parse_fbinop(opcode, rest, result_name, lineno, f)
            }
            "fneg" => self.parse_fneg(rest, result_name, lineno, f),
            "fcmp" => self.parse_fcmp(rest, result_name, lineno, f),
            "atomicrmw" | "cmpxchg" | "fence" => Err(self.err_unsupported("atomics")),
            "" => Err(self.err_parse(lineno, "empty instruction")),
            other => Err(self.err_unsupported(&format!("opcode `{}`", other))),
        }
    }

    fn parse_ret(&mut self, rest: &str, _lineno: usize, f: &mut FuncScratch) -> Result<()> {
        // `ret void`  or  `ret i32 %x`  or  `ret i32 42`.
        let tail = rest.trim_start_matches("ret").trim();
        if tail == "void" {
            f.push_inst(InstrNode::new(Inst::Return { values: vec![] }));
            return Ok(());
        }
        let (ty_str, val_str) = split_ty_operand(tail)?;
        let ty = parse_ty(&ty_str)?;
        let v = self.lookup_operand(&val_str, &ty, f)?;
        f.push_inst(InstrNode::new(Inst::Return { values: vec![v] }));
        Ok(())
    }

    fn parse_br(&mut self, rest: &str, lineno: usize, f: &mut FuncScratch) -> Result<()> {
        // `br label %L`  or  `br i1 %c, label %T, label %F`
        let tail = rest.trim_start_matches("br").trim();
        if tail.starts_with("label ") {
            let label = tail
                .trim_start_matches("label")
                .trim()
                .trim_start_matches('%')
                .to_string();
            let id = f.intern_block(&label);
            f.push_inst(InstrNode::new(Inst::Br {
                target: id,
                args: vec![],
            }));
            Ok(())
        } else if tail.starts_with("i1 ") {
            // i1 %c, label %T, label %F
            let mut parts = tail.splitn(2, ',');
            let cond_part = parts
                .next()
                .ok_or_else(|| self.err_parse(lineno, "br: missing cond"))?;
            let rest2 = parts
                .next()
                .ok_or_else(|| self.err_parse(lineno, "br: missing labels"))?;
            let cond_name = cond_part
                .trim_start_matches("i1")
                .trim()
                .trim_start_matches('%');
            let cond = f.intern_value(cond_name);
            let (tlabel, flabel) = split_two_labels(rest2)
                .ok_or_else(|| self.err_parse(lineno, "br: malformed label pair"))?;
            let then_id = f.intern_block(&tlabel);
            let else_id = f.intern_block(&flabel);
            f.push_inst(InstrNode::new(Inst::CondBr {
                cond,
                then_target: then_id,
                then_args: vec![],
                else_target: else_id,
                else_args: vec![],
            }));
            Ok(())
        } else {
            Err(self.err_parse(lineno, &format!("unrecognised br: {}", rest)))
        }
    }

    /// Parse a (multi-line-collected) `switch` instruction.
    ///
    /// Canonical shape:
    /// ```text
    /// switch <ty> <val>, label %<default> [
    ///     <ty> <case_val_0>, label %<case_block_0>
    ///     <ty> <case_val_1>, label %<case_block_1>
    ///     ...
    /// ]
    /// ```
    ///
    /// All case types must match the selector type (LLVM semantics) and
    /// must be one of `i1`, `i8`, `i16`, `i32`, or `i64` — anything else
    /// returns `Error::Unsupported`. Case values must be integer literals;
    /// LLVM does not permit non-constant case labels, so we reject any
    /// `%name` or `@name` tokens in case-value position.
    ///
    /// tMIR target: `Inst::Switch { value, default, default_args: vec![], cases }`.
    /// Block arguments on switch edges are not expressible in textual
    /// LLVM IR (phi nodes carry that information, and phis are
    /// rejected earlier by the importer), so `default_args` and each
    /// case's `args` are always empty here. The codegen-side lowering
    /// (see `llvm2-lower/src/switch.rs`, #323) picks the best strategy
    /// (linear scan / BST / jump table) regardless.
    fn parse_switch(
        &mut self,
        collected: &str,
        lineno: usize,
        f: &mut FuncScratch,
    ) -> Result<()> {
        // Strip leading `switch`.
        let tail = collected.trim_start_matches("switch").trim();

        // Split header and case list at the opening `[`.
        let lbrack = tail
            .find('[')
            .ok_or_else(|| self.err_parse(lineno, "switch: missing `[`"))?;
        let rbrack = tail
            .rfind(']')
            .ok_or_else(|| self.err_parse(lineno, "switch: missing `]`"))?;
        if rbrack <= lbrack {
            return Err(self.err_parse(lineno, "switch: `]` before `[`"));
        }
        let header = tail[..lbrack].trim().trim_end_matches(',').trim();
        let body = tail[lbrack + 1..rbrack].trim();

        // Header: `<ty> <val>, label %<default>`
        let (sel_part, default_part) = split_comma(header)
            .ok_or_else(|| self.err_parse(lineno, "switch: expected `<ty> <val>, label %default`"))?;
        let (sel_ty_str, sel_val_tok) = split_ty_operand(&sel_part)?;
        let sel_ty = parse_ty(&sel_ty_str)?;
        if !matches!(
            sel_ty,
            Ty::Bool | Ty::I8 | Ty::I16 | Ty::I32 | Ty::I64
        ) {
            return Err(self.err_unsupported(&format!(
                "switch selector type `{}` (only i1/i8/i16/i32/i64 are supported)",
                sel_ty_str
            )));
        }
        let value = self.lookup_operand(&sel_val_tok, &sel_ty, f)?;

        let default_label = default_part
            .trim()
            .trim_start_matches("label")
            .trim()
            .trim_start_matches('%')
            .to_string();
        if default_label.is_empty() {
            return Err(self.err_parse(lineno, "switch: empty default label"));
        }
        let default_block = f.intern_block(&default_label);

        // Case list: zero or more `<ty> <K>, label %<L>` entries, separated
        // by whitespace/newlines. We walk the body by whitespace-tokens
        // and group into 4-token chunks: [ty, K_comma, "label", %L]. clang -O0
        // always emits it this way, and comments / `!dbg` metadata have
        // already been stripped by the top-level preprocessor.
        let mut cases: Vec<SwitchCase> = Vec::new();
        let toks: Vec<&str> = body.split_whitespace().collect();
        let mut i = 0;
        while i < toks.len() {
            // Expect: <ty>  <K>[,]  label  %<L>[,]
            if i + 3 >= toks.len() {
                return Err(self.err_parse(
                    lineno,
                    &format!("switch: truncated case at token `{}`", toks[i]),
                ));
            }
            let case_ty_str = toks[i];
            let case_ty = parse_ty(case_ty_str)?;
            if case_ty != sel_ty {
                return Err(self.err_unsupported(&format!(
                    "switch case type `{}` does not match selector `{}`",
                    case_ty_str, sel_ty_str
                )));
            }
            // Case value: may have a trailing comma.
            let raw_val = toks[i + 1].trim_end_matches(',');
            // Reject non-literal case values — LLVM's textual syntax only
            // allows constant integer case labels, but an importer user
            // could feed garbage and we want a typed "no".
            if raw_val.starts_with('%') || raw_val.starts_with('@') {
                return Err(self.err_parse(
                    lineno,
                    &format!("switch case value must be a constant, got `{}`", raw_val),
                ));
            }
            let case_val = parse_int_literal(raw_val).ok_or_else(|| {
                self.err_parse(
                    lineno,
                    &format!("switch: bad case value literal `{}`", raw_val),
                )
            })?;
            if toks[i + 2] != "label" {
                return Err(self.err_parse(
                    lineno,
                    &format!(
                        "switch: expected `label` after case value, got `{}`",
                        toks[i + 2]
                    ),
                ));
            }
            let label_tok = toks[i + 3].trim_end_matches(',');
            let label = label_tok.trim_start_matches('%').to_string();
            if label.is_empty() {
                return Err(self.err_parse(lineno, "switch: empty case label"));
            }
            let target = f.intern_block(&label);
            cases.push(SwitchCase {
                value: Constant::Int(case_val),
                target,
                args: vec![],
            });
            i += 4;
        }

        f.push_inst(InstrNode::new(Inst::Switch {
            value,
            default: default_block,
            default_args: vec![],
            cases,
        }));
        Ok(())
    }

    fn parse_binop(
        &mut self,
        opcode: &str,
        rest: &str,
        result: Option<String>,
        lineno: usize,
        f: &mut FuncScratch,
    ) -> Result<()> {
        let op = match opcode {
            "add" => BinOp::Add,
            "sub" => BinOp::Sub,
            "mul" => BinOp::Mul,
            "and" => BinOp::And,
            "or" => BinOp::Or,
            "xor" => BinOp::Xor,
            "shl" => BinOp::Shl,
            "lshr" => BinOp::LShr,
            "ashr" => BinOp::AShr,
            "sdiv" => BinOp::SDiv,
            "udiv" => BinOp::UDiv,
            "srem" => BinOp::SRem,
            "urem" => BinOp::URem,
            _ => unreachable!(),
        };
        // Strip leading opcode and any flags like nsw / nuw / exact.
        let tail = strip_binop_flags(rest, opcode);
        // "i32 %a, %b"  OR  "i32 %a, 5"
        let (ty_str, operands) = split_ty_operand(tail)?;
        let ty = parse_ty(&ty_str)?;
        let (lhs_str, rhs_str) = split_comma(&operands)
            .ok_or_else(|| self.err_parse(lineno, "binop: expected `%a, %b`"))?;
        let lhs = self.lookup_operand(&lhs_str, &ty, f)?;
        let rhs = self.lookup_operand(&rhs_str, &ty, f)?;
        let name = result.ok_or_else(|| self.err_parse(lineno, "binop without result"))?;
        let dest = f.intern_value(&name);
        f.push_inst(
            InstrNode::new(Inst::BinOp { op, ty, lhs, rhs }).with_result(dest),
        );
        Ok(())
    }

    fn parse_icmp(
        &mut self,
        rest: &str,
        result: Option<String>,
        lineno: usize,
        f: &mut FuncScratch,
    ) -> Result<()> {
        // `icmp <pred> <ty> %a, %b`
        let tail = rest.trim_start_matches("icmp").trim();
        let mut parts = tail.splitn(2, char::is_whitespace);
        let pred_str = parts
            .next()
            .ok_or_else(|| self.err_parse(lineno, "icmp: missing predicate"))?;
        let rest = parts
            .next()
            .ok_or_else(|| self.err_parse(lineno, "icmp: missing operands"))?
            .trim();
        let pred = match pred_str {
            "eq" => ICmpOp::Eq,
            "ne" => ICmpOp::Ne,
            "ugt" => ICmpOp::Ugt,
            "uge" => ICmpOp::Uge,
            "ult" => ICmpOp::Ult,
            "ule" => ICmpOp::Ule,
            "sgt" => ICmpOp::Sgt,
            "sge" => ICmpOp::Sge,
            "slt" => ICmpOp::Slt,
            "sle" => ICmpOp::Sle,
            _ => {
                return Err(
                    self.err_unsupported(&format!("icmp predicate `{}`", pred_str))
                );
            }
        };
        let (ty_str, operands) = split_ty_operand(rest)?;
        let ty = parse_ty(&ty_str)?;
        let (lhs_str, rhs_str) = split_comma(&operands)
            .ok_or_else(|| self.err_parse(lineno, "icmp: expected `%a, %b`"))?;
        let lhs = self.lookup_operand(&lhs_str, &ty, f)?;
        let rhs = self.lookup_operand(&rhs_str, &ty, f)?;
        let name = result.ok_or_else(|| self.err_parse(lineno, "icmp without result"))?;
        let dest = f.intern_value(&name);
        f.push_inst(
            InstrNode::new(Inst::ICmp {
                op: pred,
                ty,
                lhs,
                rhs,
            })
            .with_result(dest),
        );
        Ok(())
    }

    /// Parse an FP binary op: `fadd|fsub|fmul|fdiv|frem`.
    ///
    /// Canonical clang -O0 shape:
    ///
    ///   %r = fadd float %a, %b
    ///   %r = fadd fast double %a, %b
    ///   %r = fmul reassoc nnan ninf nsz arcp contract afn float %a, 1.500000e+00
    ///
    /// Fast-math flags (`fast`, `reassoc`, `nnan`, `ninf`, `nsz`, `arcp`,
    /// `contract`, `afn`) are accepted and silently dropped, matching the
    /// integer-side behaviour for `nsw`/`nuw`.
    fn parse_fbinop(
        &mut self,
        opcode: &str,
        rest: &str,
        result: Option<String>,
        lineno: usize,
        f: &mut FuncScratch,
    ) -> Result<()> {
        let op = match opcode {
            "fadd" => BinOp::FAdd,
            "fsub" => BinOp::FSub,
            "fmul" => BinOp::FMul,
            "fdiv" => BinOp::FDiv,
            "frem" => BinOp::FRem,
            _ => unreachable!("parse_fbinop opcode `{}`", opcode),
        };
        let tail = strip_fmath_flags(rest, opcode);
        let (ty_str, operands) = split_ty_operand(tail)?;
        let ty = parse_ty(&ty_str)?;
        if !matches!(ty, Ty::F32 | Ty::F64) {
            return Err(self.err_unsupported(&format!(
                "`{}` on non-float type `{}`",
                opcode, ty_str
            )));
        }
        let (lhs_str, rhs_str) = split_comma(&operands)
            .ok_or_else(|| self.err_parse(lineno, "fbinop: expected `%a, %b`"))?;
        let lhs = self.lookup_operand(&lhs_str, &ty, f)?;
        let rhs = self.lookup_operand(&rhs_str, &ty, f)?;
        let name = result.ok_or_else(|| self.err_parse(lineno, "fbinop without result"))?;
        let dest = f.intern_value(&name);
        f.push_inst(
            InstrNode::new(Inst::BinOp { op, ty, lhs, rhs }).with_result(dest),
        );
        Ok(())
    }

    /// Parse `fneg [fast-math-flags]? <ty> <operand>`. LLVM's `fneg`
    /// corresponds to tMIR's `UnOp::FNeg`.
    fn parse_fneg(
        &mut self,
        rest: &str,
        result: Option<String>,
        lineno: usize,
        f: &mut FuncScratch,
    ) -> Result<()> {
        let tail = strip_fmath_flags(rest, "fneg");
        let (ty_str, operand_tok) = split_ty_operand(tail)?;
        let ty = parse_ty(&ty_str)?;
        if !matches!(ty, Ty::F32 | Ty::F64) {
            return Err(self.err_unsupported(&format!(
                "fneg on non-float type `{}`",
                ty_str
            )));
        }
        let operand = self.lookup_operand(&operand_tok, &ty, f)?;
        let name = result.ok_or_else(|| self.err_parse(lineno, "fneg without result"))?;
        let dest = f.intern_value(&name);
        f.push_inst(
            InstrNode::new(Inst::UnOp {
                op: UnOp::FNeg,
                ty,
                operand,
            })
            .with_result(dest),
        );
        Ok(())
    }

    /// Parse `fcmp [fast-math-flags]? <pred> <ty> <a>, <b>`. All 16 LLVM
    /// predicates are supported: 12 ordered/unordered comparisons plus
    /// `true`, `false`, `ord`, `uno`.
    ///
    ///   * `ord`  → not-NaN on either side: encoded as `FCmpOp::UEq`
    ///     applied to `%a == %a` AND `%b == %b` using a NotUnordered
    ///     predicate built from two comparisons? Too clever. Instead we
    ///     emit `FCmp { op: OEq, lhs: a, rhs: a }` AND `FCmp { op: OEq,
    ///     rhs: b }` and `and` them — that matches LLVM's documented
    ///     semantics (`ord` ≡ neither operand is a QNAN) without relying
    ///     on a tMIR-level `ord`/`uno` predicate that doesn't exist.
    ///   * `uno` → `not ord`: same pattern, with `FCmpOp::UNe` — again
    ///     because tMIR's `UNe` is true when either operand is NaN OR
    ///     they differ, which is the correct signal for `uno` when we
    ///     compare %a vs itself.
    ///   * `true` / `false` → `Inst::Const` i1 1/0.
    fn parse_fcmp(
        &mut self,
        rest: &str,
        result: Option<String>,
        lineno: usize,
        f: &mut FuncScratch,
    ) -> Result<()> {
        // Strip leading `fcmp` and any fast-math flags before the
        // predicate keyword.
        let tail = strip_fmath_flags(rest, "fcmp");
        let mut parts = tail.splitn(2, char::is_whitespace);
        let pred_str = parts
            .next()
            .ok_or_else(|| self.err_parse(lineno, "fcmp: missing predicate"))?;
        let rest2 = parts
            .next()
            .ok_or_else(|| self.err_parse(lineno, "fcmp: missing operands"))?
            .trim();

        // Handle the two trivial constant predicates without touching
        // the operand types — the operands are still required to be
        // well-typed but their values are dead.
        if pred_str == "true" || pred_str == "false" {
            let (ty_str, operands) = split_ty_operand(rest2)?;
            let ty = parse_ty(&ty_str)?;
            if !matches!(ty, Ty::F32 | Ty::F64) {
                return Err(self.err_unsupported(&format!(
                    "fcmp on non-float type `{}`",
                    ty_str
                )));
            }
            // Force-evaluate both operands so any forward-declared SSA
            // names are still interned (matches integer `icmp` handling).
            if let Some((lhs_s, rhs_s)) = split_comma(&operands) {
                let _ = self.lookup_operand(&lhs_s, &ty, f);
                let _ = self.lookup_operand(&rhs_s, &ty, f);
            }
            let name =
                result.ok_or_else(|| self.err_parse(lineno, "fcmp without result"))?;
            let dest = f.intern_value(&name);
            f.push_inst(
                InstrNode::new(Inst::Const {
                    ty: Ty::Bool,
                    value: Constant::Bool(pred_str == "true"),
                })
                .with_result(dest),
            );
            return Ok(());
        }

        // `ord` / `uno`: implement via the self-comparison trick.
        //
        //   %o = fcmp ord TY %a, %b
        //     ≡ (a == a) && (b == b)    (both non-NaN)
        //     ≡ FCmp OEq a,a  AND  FCmp OEq b,b
        //
        //   %u = fcmp uno TY %a, %b
        //     ≡ (a != a) || (b != b)    (either is NaN)
        //     ≡ FCmp UNe a,a  OR   FCmp UNe b,b
        //
        // These patterns compile through the AArch64 lowering path
        // because they are expressed entirely in terms of already-
        // supported `FCmp` + integer `BinOp::{And,Or}` on i1.
        if pred_str == "ord" || pred_str == "uno" {
            let (ty_str, operands) = split_ty_operand(rest2)?;
            let ty = parse_ty(&ty_str)?;
            if !matches!(ty, Ty::F32 | Ty::F64) {
                return Err(self.err_unsupported(&format!(
                    "fcmp on non-float type `{}`",
                    ty_str
                )));
            }
            let (lhs_str, rhs_str) = split_comma(&operands)
                .ok_or_else(|| self.err_parse(lineno, "fcmp: expected `%a, %b`"))?;
            let a = self.lookup_operand(&lhs_str, &ty, f)?;
            let b = self.lookup_operand(&rhs_str, &ty, f)?;
            let (per_side_op, combine) = if pred_str == "ord" {
                (FCmpOp::OEq, BinOp::And)
            } else {
                (FCmpOp::UNe, BinOp::Or)
            };
            let aa = f.fresh_value();
            f.push_inst(
                InstrNode::new(Inst::FCmp {
                    op: per_side_op,
                    ty: ty.clone(),
                    lhs: a,
                    rhs: a,
                })
                .with_result(aa),
            );
            let bb = f.fresh_value();
            f.push_inst(
                InstrNode::new(Inst::FCmp {
                    op: per_side_op,
                    ty,
                    lhs: b,
                    rhs: b,
                })
                .with_result(bb),
            );
            let name =
                result.ok_or_else(|| self.err_parse(lineno, "fcmp without result"))?;
            let dest = f.intern_value(&name);
            f.push_inst(
                InstrNode::new(Inst::BinOp {
                    op: combine,
                    ty: Ty::Bool,
                    lhs: aa,
                    rhs: bb,
                })
                .with_result(dest),
            );
            return Ok(());
        }

        let pred = match pred_str {
            "oeq" => FCmpOp::OEq,
            "one" => FCmpOp::ONe,
            "olt" => FCmpOp::OLt,
            "ole" => FCmpOp::OLe,
            "ogt" => FCmpOp::OGt,
            "oge" => FCmpOp::OGe,
            "ueq" => FCmpOp::UEq,
            "une" => FCmpOp::UNe,
            "ult" => FCmpOp::ULt,
            "ule" => FCmpOp::ULe,
            "ugt" => FCmpOp::UGt,
            "uge" => FCmpOp::UGe,
            other => {
                return Err(self.err_unsupported(&format!("fcmp predicate `{}`", other)));
            }
        };
        let (ty_str, operands) = split_ty_operand(rest2)?;
        let ty = parse_ty(&ty_str)?;
        if !matches!(ty, Ty::F32 | Ty::F64) {
            return Err(self.err_unsupported(&format!(
                "fcmp on non-float type `{}`",
                ty_str
            )));
        }
        let (lhs_str, rhs_str) = split_comma(&operands)
            .ok_or_else(|| self.err_parse(lineno, "fcmp: expected `%a, %b`"))?;
        let lhs = self.lookup_operand(&lhs_str, &ty, f)?;
        let rhs = self.lookup_operand(&rhs_str, &ty, f)?;
        let name = result.ok_or_else(|| self.err_parse(lineno, "fcmp without result"))?;
        let dest = f.intern_value(&name);
        f.push_inst(
            InstrNode::new(Inst::FCmp {
                op: pred,
                ty,
                lhs,
                rhs,
            })
            .with_result(dest),
        );
        Ok(())
    }

    fn parse_alloca(
        &mut self,
        rest: &str,
        result: Option<String>,
        lineno: usize,
        f: &mut FuncScratch,
    ) -> Result<()> {
        // `alloca <ty>, align N`  or  `alloca <ty>`.
        let tail = rest.trim_start_matches("alloca").trim();
        let parts = split_call_args(tail);
        let ty_str = parts.first().map(|s| s.trim()).unwrap_or("");
        if ty_str.is_empty() {
            return Err(self.err_parse(lineno, "alloca: missing type"));
        }
        if ty_str.starts_with('%') {
            let _ = self.parse_ty_ctx(ty_str)?;
            if let Some(layout) = self.struct_layouts.get(ty_str).cloned() {
                let mut align = None;
                let mut count_clause: Option<String> = None;
                for clause in parts.iter().skip(1) {
                    let clause = clause.trim();
                    if clause.is_empty() {
                        continue;
                    }
                    if clause.starts_with("align ") {
                        if align.is_some() {
                            return Err(self.err_unsupported(
                                "struct alloca with multiple align clauses",
                            ));
                        }
                        align = Some(parse_align_clause(clause).ok_or_else(|| {
                            self.err_parse(lineno, "alloca: malformed align clause")
                        })?);
                    } else if count_clause.is_none() {
                        count_clause = Some(clause.to_string());
                    } else {
                        return Err(self.err_unsupported(
                            "struct alloca with multiple count operands",
                        ));
                    }
                }

                let size_value = self.emit_i64_const(layout.size as i128, f);
                let count = if let Some(clause) = count_clause {
                    let (count_ty_str, count_tok) = split_ty_operand(&clause)?;
                    let count_ty = parse_ty(&count_ty_str)?;
                    if !is_integer_ty(&count_ty) {
                        return Err(self.err_unsupported(&format!(
                            "struct alloca count type `{}`",
                            count_ty_str
                        )));
                    }
                    if let Some(n) = parse_int_literal(&count_tok) {
                        if n < 0 {
                            return Err(self.err_unsupported("struct alloca with negative count"));
                        }
                        let total = (layout.size as i128).checked_mul(n).ok_or_else(|| {
                            self.err_unsupported("struct alloca byte count overflow")
                        })?;
                        Some(self.emit_i64_const(total, f))
                    } else {
                        let raw_count = self.lookup_operand(&count_tok, &count_ty, f)?;
                        let widened_count = self.coerce_int_to_i64(raw_count, &count_ty, f)?;
                        let total = f.fresh_value();
                        f.push_inst(
                            InstrNode::new(Inst::BinOp {
                                op: BinOp::Mul,
                                ty: Ty::I64,
                                lhs: widened_count,
                                rhs: size_value,
                            })
                            .with_result(total),
                        );
                        Some(total)
                    }
                } else {
                    Some(size_value)
                };

                let name =
                    result.ok_or_else(|| self.err_parse(lineno, "alloca without result"))?;
                let dest = f.intern_value(&name);
                f.push_inst(
                    InstrNode::new(Inst::Alloca {
                        ty: Ty::I8,
                        count,
                        align,
                    })
                    .with_result(dest),
                );
                return Ok(());
            }
        }
        let ty = parse_ty(ty_str)?;
        let name = result.ok_or_else(|| self.err_parse(lineno, "alloca without result"))?;
        let dest = f.intern_value(&name);
        f.push_inst(
            InstrNode::new(Inst::Alloca {
                ty,
                count: None,
                align: None,
            })
            .with_result(dest),
        );
        Ok(())
    }

    fn parse_load(
        &mut self,
        rest: &str,
        result: Option<String>,
        lineno: usize,
        f: &mut FuncScratch,
    ) -> Result<()> {
        // `load <ty>, ptr %p, align N`  (LLVM ≥ 15 opaque pointer form)
        let tail = rest.trim_start_matches("load").trim();
        if tail.contains("volatile") {
            return Err(self.err_unsupported("volatile load"));
        }
        let (ty_str, rest2) = split_comma(tail)
            .ok_or_else(|| self.err_parse(lineno, "load: expected `<ty>, ptr %p`"))?;
        let ty = parse_ty(&ty_str)?;
        // The next portion is `ptr %p` possibly followed by `, align N`.
        let ptr_part = rest2.split(',').next().unwrap_or("").trim();
        let (_, ptr_tok) = split_ty_operand(ptr_part)?;
        let ptr = self.lookup_operand(&ptr_tok, &Ty::Ptr, f)?;
        let name = result.ok_or_else(|| self.err_parse(lineno, "load without result"))?;
        let dest = f.intern_value(&name);
        f.push_inst(
            InstrNode::new(Inst::Load {
                ty,
                ptr,
                volatile: false,
                align: None,
            })
            .with_result(dest),
        );
        Ok(())
    }

    fn parse_store(
        &mut self,
        rest: &str,
        lineno: usize,
        f: &mut FuncScratch,
    ) -> Result<()> {
        // `store <ty> <val>, ptr %p, align N`
        let tail = rest.trim_start_matches("store").trim();
        if tail.contains("volatile") {
            return Err(self.err_unsupported("volatile store"));
        }
        let (val_part, rest2) = split_comma(tail)
            .ok_or_else(|| self.err_parse(lineno, "store: expected `<ty> <val>, ptr %p`"))?;
        let (ty_str, val_tok) = split_ty_operand(&val_part)?;
        let ty = parse_ty(&ty_str)?;
        let value = self.lookup_operand(&val_tok, &ty, f)?;
        let ptr_part = rest2.split(',').next().unwrap_or("").trim();
        let (_, ptr_tok) = split_ty_operand(ptr_part)?;
        let ptr = self.lookup_operand(&ptr_tok, &Ty::Ptr, f)?;
        f.push_inst(InstrNode::new(Inst::Store {
            ty,
            ptr,
            value,
            volatile: false,
            align: None,
        }));
        Ok(())
    }

    fn parse_call(
        &mut self,
        rest: &str,
        result: Option<String>,
        lineno: usize,
        f: &mut FuncScratch,
    ) -> Result<()> {
        // `call <ret-ty> @name(<args>)` possibly prefixed with `tail` / `notail`
        // / `musttail` / calling-conv attributes.
        //
        // We accept direct calls to named globals only. Anything fancier (call
        // void %fp(...), invoke, tail-call attrs beyond `tail`, etc.) is
        // unsupported.
        let mut tail = rest.trim_start_matches("call").trim();
        for prefix in ["tail ", "notail ", "musttail "] {
            if let Some(t) = tail.strip_prefix(prefix) {
                tail = t.trim_start();
            }
        }
        // Strip calling-conv / function attrs tokens that clang -O0 emits
        // between `call` and the return type (e.g. `void @foo(...)` vs
        // `i32 @bar(...)` vs `dso_local i32 @baz(...)`).
        //
        // Clang also emits an explicit function-type form for vararg
        // calls:
        //
        //     call i32 (ptr, ...) @printf(ptr noundef @.str, ...)
        //
        // We find the first `@` at the top level (not inside `(...)`),
        // which reliably points at the callee. Any `(...)` block before
        // it is the explicit function type and is ignored — we recover
        // the return type as the first type-looking token in the prefix.
        let at = find_top_level_at(tail)
            .ok_or_else(|| self.err_unsupported("indirect call / call on %value"))?;
        let prefix = tail[..at].trim();
        let callee_region = &tail[at..];

        // Find the return type: walk `prefix` token-by-token and pick
        // the first token recognised by `is_type_token`. This handles
        // all three shapes:
        //   `i32 @foo(...)`           -> "i32"
        //   `dso_local i32 @foo(...)` -> "i32"
        //   `i32 (ptr, ...) @printf`  -> "i32"
        //   `void @foo(...)`          -> "void"
        let ret_ty_str = prefix
            .split_whitespace()
            .find(|t| is_type_token(t) || *t == "void")
            .unwrap_or("")
            .to_string();
        let ret_ty = if ret_ty_str == "void" || ret_ty_str.is_empty() {
            None
        } else {
            Some(parse_ty(&ret_ty_str)?)
        };

        // callee_region is `@name(<args>) [#attrs]`.
        let paren_open = callee_region
            .find('(')
            .ok_or_else(|| self.err_parse(lineno, "call: missing `(`"))?;
        let callee_name = callee_region[1..paren_open].trim().to_string();
        let paren_close_rel = find_matching_paren(&callee_region[paren_open..])
            .ok_or_else(|| self.err_parse(lineno, "call: unbalanced parens"))?;
        let args_str = &callee_region[paren_open + 1..paren_open + paren_close_rel];

        // Split args by commas not inside parentheses.
        let arg_toks = split_call_args(args_str);
        let mut args: Vec<ValueId> = Vec::with_capacity(arg_toks.len());
        for tok in arg_toks {
            // Each arg is `<ty> [attrs...] <operand>`. clang emits
            // `i32 noundef %x`, `ptr noundef nonnull @.str`, etc. The
            // type is always the first token and the operand is always
            // the last; parameter attributes between them are silently
            // dropped.
            let trimmed = tok.trim();
            let toks: Vec<&str> = trimmed.split_whitespace().collect();
            if toks.len() < 2 {
                return Err(
                    self.err_parse(lineno, &format!("call arg needs `<ty> <val>`: `{}`", tok)),
                );
            }
            let aty = parse_ty(toks[0])?;
            let aval = toks.last().copied().unwrap_or("");
            let v = self.lookup_operand(aval, &aty, f)?;
            args.push(v);
        }

        // Look up or forward-declare the callee.
        let fid = if let Some(id) = self.func_ids.get(&callee_name) {
            *id
        } else {
            // Implicit declare-by-use (common for `printf` when clang emits
            // the declaration earlier; we synthesize a signature from the
            // observed call site).
            let params: Vec<(Option<String>, Ty)> = args
                .iter()
                .map(|_| (None, Ty::I32))
                .collect(); // conservative; only used to build a FuncTy shell
            // If we've seen no prior declaration this is almost certainly
            // wrong for vararg functions. But since we only need to succeed
            // in building the Module object (not to actually link), we make
            // one up.
            let sig = FuncSignature {
                name: callee_name.clone(),
                ret: ret_ty.clone(),
                params,
                is_vararg: true,
                internal: false,
            };
            self.register_function(sig)?
        };

        let node = Inst::Call { callee: fid, args };
        match (result, ret_ty) {
            (Some(name), Some(_)) => {
                let dest = f.intern_value(&name);
                f.push_inst(InstrNode::new(node).with_result(dest));
            }
            (None, _) => {
                f.push_inst(InstrNode::new(node));
            }
            (Some(_), None) => {
                return Err(self.err_parse(lineno, "call of void function has a result"));
            }
        }
        Ok(())
    }

    fn parse_gep(
        &mut self,
        rest: &str,
        result: Option<String>,
        lineno: usize,
        f: &mut FuncScratch,
    ) -> Result<()> {
        let tail = rest.trim_start_matches("getelementptr").trim();
        let tail = tail.trim_start_matches("inbounds").trim();
        let (pointee_ty_str, after_pointee) = split_comma(tail)
            .ok_or_else(|| self.err_parse(lineno, "gep: missing pointee type"))?;

        if pointee_ty_str.trim().starts_with('%') {
            let name = pointee_ty_str.trim();
            let _ = self.parse_ty_ctx(name)?;
            if let Some(layout) = self.struct_layouts.get(name).cloned() {
                let (base_part, indices_str) = split_comma(&after_pointee)
                    .ok_or_else(|| self.err_parse(lineno, "gep: missing base operand"))?;
                let (_, base_tok) = split_ty_operand(&base_part)?;
                if base_tok.starts_with('@') {
                    return Err(self.err_unsupported(
                        "address-of global in struct GEP — needs global-address materialization",
                    ));
                }
                let base = self.lookup_operand(&base_tok, &Ty::Ptr, f)?;
                let indices = split_call_args(&indices_str);
                if indices.is_empty() || indices.len() > 2 {
                    return Err(self.err_unsupported(
                        "struct GEP requires one outer index and optional field index",
                    ));
                }

                let (outer_ty_str, outer_tok) = split_ty_operand(indices[0].trim())?;
                let outer_ty = parse_ty(&outer_ty_str)?;
                if !is_integer_ty(&outer_ty) {
                    return Err(self.err_unsupported(&format!(
                        "struct GEP outer index type `{}`",
                        outer_ty_str
                    )));
                }
                let outer = parse_int_literal(&outer_tok).ok_or_else(|| {
                    self.err_unsupported("struct GEP with dynamic outer index")
                })?;

                let field_offset = if let Some(field_clause) = indices.get(1) {
                    let (field_ty_str, field_tok) = split_ty_operand(field_clause.trim())?;
                    let field_ty = parse_ty(&field_ty_str)?;
                    if !is_integer_ty(&field_ty) {
                        return Err(self.err_unsupported(&format!(
                            "struct GEP field index type `{}`",
                            field_ty_str
                        )));
                    }
                    let field_idx = parse_int_literal(&field_tok).ok_or_else(|| {
                        self.err_unsupported("struct GEP with dynamic field index")
                    })?;
                    if field_idx < 0 {
                        return Err(self.err_unsupported("struct GEP with negative field index"));
                    }
                    let field_idx = field_idx as usize;
                    layout.fields.get(field_idx).map(|(_, off)| *off).ok_or_else(|| {
                        self.err_unsupported(&format!(
                            "struct GEP field index {} out of bounds for `{}`",
                            field_idx, name
                        ))
                    })?
                } else {
                    0
                };

                let offset = outer
                    .checked_mul(layout.size as i128)
                    .and_then(|n| n.checked_add(field_offset as i128))
                    .ok_or_else(|| self.err_unsupported("struct GEP byte offset overflow"))?;
                let offset_value = self.emit_i64_const(offset, f);
                let name =
                    result.ok_or_else(|| self.err_parse(lineno, "gep without result"))?;
                let dest = f.intern_value(&name);
                f.push_inst(
                    InstrNode::new(Inst::GEP {
                        pointee_ty: Ty::I8,
                        base,
                        indices: vec![offset_value],
                    })
                    .with_result(dest),
                );
                return Ok(());
            }
        }

        if !tail.starts_with('[') {
            return Err(self.err_unsupported(
                "GEP with non-array base (only `[N x i8], ptr @str` and named-struct forms supported)",
            ));
        }
        let ptr_pos = tail
            .find("ptr ")
            .ok_or_else(|| self.err_unsupported("GEP without `ptr` operand"))?;
        let after_ptr = &tail[ptr_pos + 4..];
        let base_tok = after_ptr.split(',').next().unwrap_or("").trim();
        if base_tok.starts_with('@') {
            return Err(self.err_unsupported(
                "address-of global (GEP on `@string`) — needs global-address materialization",
            ));
        }
        let name = result.ok_or_else(|| self.err_parse(lineno, "gep without result"))?;
        let dest = f.intern_value(&name);
        let base = self.lookup_operand(base_tok, &Ty::Ptr, f)?;
        f.push_inst(
            InstrNode::new(Inst::Copy {
                ty: Ty::Ptr,
                operand: base,
            })
            .with_result(dest),
        );
        Ok(())
    }

    fn parse_cast(
        &mut self,
        opcode: &str,
        rest: &str,
        result: Option<String>,
        lineno: usize,
        f: &mut FuncScratch,
    ) -> Result<()> {
        // Form: `<op> <src-ty> <val> to <dst-ty>`
        let tail = rest.trim_start_matches(opcode).trim();
        let to_pos = tail
            .find(" to ")
            .ok_or_else(|| self.err_parse(lineno, "cast: missing `to`"))?;
        let head = &tail[..to_pos];
        let dst_ty_str = tail[to_pos + 4..].trim();
        let (src_ty_str, src_val) = split_ty_operand(head)?;
        let src_ty = parse_ty(&src_ty_str)?;
        let dst_ty = parse_ty(dst_ty_str)?;
        let op = match opcode {
            "sext" => CastOp::SExt,
            "zext" => CastOp::ZExt,
            "trunc" => CastOp::Trunc,
            "bitcast" => CastOp::Bitcast,
            "ptrtoint" => CastOp::PtrToInt,
            "inttoptr" => CastOp::IntToPtr,
            "sitofp" => CastOp::SIToFP,
            "uitofp" => CastOp::UIToFP,
            "fptosi" => CastOp::FPToSI,
            "fptoui" => CastOp::FPToUI,
            "fpext" => CastOp::FPExt,
            "fptrunc" => CastOp::FPTrunc,
            _ => unreachable!(),
        };
        let operand = self.lookup_operand(&src_val, &src_ty, f)?;
        let name = result.ok_or_else(|| self.err_parse(lineno, "cast without result"))?;
        let dest = f.intern_value(&name);
        f.push_inst(
            InstrNode::new(Inst::Cast {
                op,
                src_ty,
                dst_ty,
                operand,
            })
            .with_result(dest),
        );
        Ok(())
    }

    fn parse_select(
        &mut self,
        rest: &str,
        result: Option<String>,
        lineno: usize,
        f: &mut FuncScratch,
    ) -> Result<()> {
        // `select i1 %c, <ty> %a, <ty> %b`
        let tail = rest.trim_start_matches("select").trim();
        let parts: Vec<&str> = tail.splitn(3, ',').collect();
        if parts.len() != 3 {
            return Err(self.err_parse(lineno, "select: expected three comma-separated operands"));
        }
        let cond_part = parts[0].trim();
        let (cty, cval) = split_ty_operand(cond_part)?;
        let cond_ty = parse_ty(&cty)?;
        let cond = self.lookup_operand(&cval, &cond_ty, f)?;

        let (t_ty_str, t_val) = split_ty_operand(parts[1].trim())?;
        let t_ty = parse_ty(&t_ty_str)?;
        let t = self.lookup_operand(&t_val, &t_ty, f)?;

        let (_e_ty_str, e_val) = split_ty_operand(parts[2].trim())?;
        let e = self.lookup_operand(&e_val, &t_ty, f)?;

        let name = result.ok_or_else(|| self.err_parse(lineno, "select without result"))?;
        let dest = f.intern_value(&name);
        f.push_inst(
            InstrNode::new(Inst::Select {
                ty: t_ty,
                cond,
                then_val: t,
                else_val: e,
            })
            .with_result(dest),
        );
        Ok(())
    }

    // --- Operand resolution -----------------------------------------------

    fn lookup_operand(
        &self,
        tok: &str,
        ty: &Ty,
        f: &mut FuncScratch,
    ) -> Result<ValueId> {
        let tok = tok.trim();
        if let Some(rest) = tok.strip_prefix('%') {
            Ok(f.intern_value(rest))
        } else if tok.starts_with('@') {
            Err(self
                .err_unsupported("global address as instruction operand (needs materialization)"))
        } else if tok == "true" {
            let v = f.fresh_value();
            f.push_inst(
                InstrNode::new(Inst::Const {
                    ty: Ty::Bool,
                    value: Constant::Bool(true),
                })
                .with_result(v),
            );
            Ok(v)
        } else if tok == "false" {
            let v = f.fresh_value();
            f.push_inst(
                InstrNode::new(Inst::Const {
                    ty: Ty::Bool,
                    value: Constant::Bool(false),
                })
                .with_result(v),
            );
            Ok(v)
        } else if tok == "null" || tok == "undef" || tok == "poison" {
            // For FP types, `undef`/`poison` must materialize as a Float
            // constant or downstream lowering will see a type mismatch
            // when a `Constant::Int(0)` flows into an F32/F64 context.
            let v = f.fresh_value();
            let value = match ty {
                Ty::F32 | Ty::F64 => Constant::Float(0.0),
                _ => Constant::Int(0),
            };
            f.push_inst(
                InstrNode::new(Inst::Const {
                    ty: ty.clone(),
                    value,
                })
                .with_result(v),
            );
            Ok(v)
        } else if matches!(ty, Ty::F32 | Ty::F64) {
            // Floating-point immediate. LLVM textual IR emits these in
            // three shapes:
            //   * Decimal:  `1.5`, `-3.14`, `1.500000e+00`
            //   * Hex f64:  `0x3FF8000000000000`  (bit pattern of 1.5)
            //   * Hex f80 / f128 extended: `0xK...`, `0xL...`, `0xM...`
            //     — we reject these because tMIR only models f32/f64.
            let parsed = parse_fp_literal(tok).ok_or_else(|| Error::Parse {
                line: 0,
                message: format!("unknown float operand token `{}`", tok),
            })?;
            match parsed {
                FpLit::Double(d) => {
                    let v = f.fresh_value();
                    f.push_inst(
                        InstrNode::new(Inst::Const {
                            ty: ty.clone(),
                            value: Constant::Float(d),
                        })
                        .with_result(v),
                    );
                    Ok(v)
                }
                FpLit::Extended(tag) => Err(self.err_unsupported(&format!(
                    "extended-precision float literal `0x{}...` (tMIR only has f32/f64)",
                    tag
                ))),
            }
        } else {
            // Integer literal (possibly negative, possibly hex).
            let n = parse_int_literal(tok).ok_or_else(|| {
                Error::Parse {
                    line: 0,
                    message: format!("unknown operand token `{}`", tok),
                }
            })?;
            let v = f.fresh_value();
            f.push_inst(
                InstrNode::new(Inst::Const {
                    ty: ty.clone(),
                    value: Constant::Int(n),
                })
                .with_result(v),
            );
            Ok(v)
        }
    }
}

// --------------------------------------------------------------------------
// Small helpers / token utilities
// --------------------------------------------------------------------------

#[derive(Clone, Debug)]
struct FuncSignature {
    name: String,
    ret: Option<Ty>,
    /// Parameters: (optional ssa name, type).
    params: Vec<(Option<String>, Ty)>,
    is_vararg: bool,
    internal: bool,
}

fn strip_line(line: &str) -> String {
    // Drop everything starting with `;` comment marker.
    let no_comment = match line.find(';') {
        Some(i) => &line[..i],
        None => line,
    };
    // Drop metadata attachments like `, !dbg !12` or `, !tbaa !5` that
    // appear at the end of instruction lines.
    let trimmed = no_comment.trim_end();
    let mut s = trimmed.to_string();
    loop {
        let lower = s.to_lowercase();
        if let Some(idx) = lower.rfind(", !") {
            s.truncate(idx);
        } else {
            break;
        }
    }
    s
}

fn collect_function_body(
    lines: &[(usize, String)],
    start: usize,
    _start_lineno: usize,
) -> Result<(usize, Vec<(usize, String)>)> {
    // The `define` line may or may not end with `{` — if not, the `{` is on
    // the next line.
    let mut body = Vec::new();
    let first = &lines[start].1;
    if !first.trim_end().ends_with('{') {
        // Skip until the `{`.
    }
    for (i, (ln, raw)) in lines.iter().enumerate().skip(start + 1) {
        let t = raw.trim();
        if t == "}" {
            return Ok((i, body));
        }
        body.push((*ln, raw.clone()));
    }
    Err(Error::Parse {
        line: lines[start].0,
        message: "unterminated function body (no closing `}`)".into(),
    })
}

fn parse_function_signature(line: &str, lineno: usize, is_define: bool) -> Result<FuncSignature> {
    // Canonical shapes:
    //   define dso_local void @foo(i32 noundef %x) #0 {
    //   define internal i32 @bar(i8 %a, i8 %b) {
    //   declare i32 @printf(ptr noundef, ...) #1
    //
    // We chop leading keyword (`define` / `declare`), trailing attributes
    // (`#N`, `{`), then find the parenthesis boundaries to split ret-ty /
    // name / params.
    let head_kw = if is_define { "define" } else { "declare" };
    let mut s = line.trim_start_matches(head_kw).trim().to_string();
    if let Some(brace_idx) = s.rfind('{') {
        s.truncate(brace_idx);
    }
    // Strip trailing function attrs like `#0` / `nounwind` / `readnone` /
    // `local_unnamed_addr`.
    let paren_close = s
        .rfind(')')
        .ok_or(Error::Parse {
            line: lineno,
            message: "function header missing `)`".into(),
        })?;
    let (left, right) = s.split_at(paren_close + 1);
    let left = left.trim();
    let _attrs_after = right.trim();
    // `left` = `<attrs>... <ret-ty> @name(<params>)`
    let paren_open = left.find('(').ok_or(Error::Parse {
        line: lineno,
        message: "function header missing `(`".into(),
    })?;
    let head = &left[..paren_open];
    let params_str = &left[paren_open + 1..left.len() - 1];

    // The function name is the last `@word` token in `head`; everything
    // between the keyword and the `@name` is the return type and attributes.
    let at = head.rfind('@').ok_or(Error::Parse {
        line: lineno,
        message: "function header missing `@name`".into(),
    })?;
    let name_region = head[at + 1..].trim();
    let name = name_region
        .split(|c: char| c.is_whitespace() || c == '(')
        .next()
        .unwrap_or("")
        .to_string();
    if name.is_empty() {
        return Err(Error::Parse {
            line: lineno,
            message: "function name is empty".into(),
        });
    }

    // Return type: last token before `@`, ignoring parameter/function
    // attributes like `dso_local` / `internal` / `noundef` etc. Clang emits
    // a predictable shape: the type is always the token immediately before
    // `@name`.
    let pre_at = head[..at].trim();
    // Tokenise pre_at; the last token with a leading type char is the type.
    let mut tokens: Vec<&str> = pre_at.split_whitespace().collect();
    let mut ret_ty: Option<Ty> = None;
    while let Some(t) = tokens.pop() {
        if t == "void" {
            ret_ty = None;
            break;
        }
        if is_type_token(t) {
            ret_ty = Some(parse_ty(t)?);
            break;
        }
    }
    let internal = pre_at.contains("internal") || pre_at.contains("private");

    // Parameters.
    let mut params: Vec<(Option<String>, Ty)> = Vec::new();
    let mut is_vararg = false;
    if !params_str.trim().is_empty() {
        for p in split_call_args(params_str) {
            let p = p.trim();
            if p == "..." {
                is_vararg = true;
                continue;
            }
            // Each param is `<ty> [attrs...] [%name]`.
            let toks: Vec<&str> = p.split_whitespace().collect();
            if toks.is_empty() {
                continue;
            }
            let ty = parse_ty(toks[0])?;
            // Last token is the %name if it starts with %, else anonymous.
            let ssa = toks
                .iter()
                .rev()
                .find(|t| t.starts_with('%'))
                .map(|t| t.trim_start_matches('%').to_string());
            params.push((ssa, ty));
        }
    }

    Ok(FuncSignature {
        name,
        ret: ret_ty,
        params,
        is_vararg,
        internal,
    })
}

fn parse_block_label(line: &str) -> Option<String> {
    // `entry:` or `for.cond:  ; preds = %for.inc, %entry`.
    let first = line.split_whitespace().next()?;
    let colon = first.strip_suffix(':')?;
    if colon.is_empty() {
        return None;
    }
    // Reject things that also start with a keyword.
    if colon.contains('%') || colon.contains('@') {
        return None;
    }
    Some(colon.to_string())
}

fn split_eq(s: &str) -> Option<(&str, &str)> {
    let i = s.find('=')?;
    Some((&s[..i], &s[i + 1..]))
}

/// Split `%r = ...` at the first `=` that is not part of `==` / `icmp eq`.
/// We look for an `=` that is followed by whitespace (standard LLVM syntax).
fn split_eq_not_icmp(s: &str) -> Option<(&str, &str)> {
    let bytes = s.as_bytes();
    for (i, b) in bytes.iter().enumerate() {
        if *b != b'=' {
            continue;
        }
        // Next char should be whitespace for this to be an assignment.
        let next = bytes.get(i + 1).copied().unwrap_or(b'\0');
        if next == b' ' || next == b'\t' {
            return Some((&s[..i], &s[i + 1..]));
        }
    }
    None
}

fn split_comma(s: &str) -> Option<(String, String)> {
    // Respect balanced `[` / `(` when picking the split point.
    let bytes = s.as_bytes();
    let mut depth: i32 = 0;
    for (i, b) in bytes.iter().enumerate() {
        match *b {
            b'[' | b'(' => depth += 1,
            b']' | b')' => depth -= 1,
            b',' if depth == 0 => {
                return Some((s[..i].trim().to_string(), s[i + 1..].trim().to_string()));
            }
            _ => {}
        }
    }
    None
}

/// Split `<ty> <operand>` where `<ty>` may contain `*`, `[`, etc.
/// Used for constructs like `ret i32 %x`, `icmp slt i32 %a, 0`.
fn split_ty_operand(s: &str) -> Result<(String, String)> {
    // Find the first top-level whitespace that separates ty from operand.
    let bytes = s.as_bytes();
    let mut depth: i32 = 0;
    for (i, b) in bytes.iter().enumerate() {
        match *b {
            b'[' | b'(' | b'<' => depth += 1,
            b']' | b')' | b'>' => depth -= 1,
            b if depth == 0 && (b == b' ' || b == b'\t') => {
                let left = s[..i].trim().to_string();
                let right = s[i + 1..].trim().to_string();
                if !left.is_empty() && !right.is_empty() {
                    return Ok((left, right));
                }
            }
            _ => {}
        }
    }
    Err(Error::Parse {
        line: 0,
        message: format!("expected `<ty> <operand>` in `{}`", s),
    })
}

fn split_two_labels(s: &str) -> Option<(String, String)> {
    // Expects "label %A, label %B"
    let (a, b) = split_comma(s)?;
    let a = a.trim_start_matches("label").trim().trim_start_matches('%').to_string();
    let b = b.trim_start_matches("label").trim().trim_start_matches('%').to_string();
    Some((a, b))
}

fn strip_binop_flags<'a>(rest: &'a str, opcode: &str) -> &'a str {
    let mut s = rest.trim_start_matches(opcode).trim_start();
    loop {
        let next = s.split_whitespace().next().unwrap_or("");
        match next {
            "nsw" | "nuw" | "exact" | "disjoint" => {
                s = &s[next.len()..].trim_start();
            }
            _ => return s,
        }
    }
}

/// Strip leading opcode + any LLVM fast-math flag tokens that clang may
/// emit between the opcode and the type: `fast`, `nnan`, `ninf`, `nsz`,
/// `arcp`, `contract`, `reassoc`, `afn`. We silently drop them — LLVM2
/// does not yet model fast-math semantics and dropping matches the
/// existing `nsw`/`nuw` treatment on integer ops.
fn strip_fmath_flags<'a>(rest: &'a str, opcode: &str) -> &'a str {
    let mut s = rest.trim_start_matches(opcode).trim_start();
    loop {
        let next = s.split_whitespace().next().unwrap_or("");
        match next {
            "fast" | "nnan" | "ninf" | "nsz" | "arcp" | "contract" | "reassoc" | "afn" => {
                s = s[next.len()..].trim_start();
            }
            _ => return s,
        }
    }
}

/// Result of parsing a textual LLVM IR floating-point literal.
enum FpLit {
    /// A finite / infinite / NaN f64 value; sufficient for f32 after a
    /// narrowing cast because LLVM always round-trips f32 through a
    /// canonicalised 64-bit bit pattern in the textual IR.
    Double(f64),
    /// Extended-precision literal. LLVM uses a leading tag character
    /// after `0x`:
    ///
    /// * `0xK`  → x86 80-bit `long double`
    /// * `0xL`  → ppc_fp128 128-bit
    /// * `0xM`  → IEEE 128-bit
    /// * `0xH`  → half (f16)
    /// * `0xR`  → bfloat16
    ///
    /// tMIR has no F16 / F80 / F128 / BF16 so we surface these as a
    /// typed `Unsupported` rather than silently narrow.
    Extended(char),
}

/// Parse an LLVM IR textual float literal.
///
/// Accepted forms:
///   * Decimal with optional sign and exponent:
///     `1.5`, `-3.14`, `0.0`, `1.000000e+00`, `1e-9`, `-2.5E+10`
///   * Hex bit-pattern (f64):   `0x3FF8000000000000`  (16 hex digits)
///   * Named specials:          `inf`, `-inf`, `nan` (not emitted by
///     clang -O0 in practice but cheap to accept).
///   * Extended-precision tag:  `0xK...`, `0xL...`, `0xM...`, `0xH...`,
///     `0xR...` — returned as `FpLit::Extended(tag)` so the caller can
///     emit an `Unsupported` error with a precise reason.
fn parse_fp_literal(s: &str) -> Option<FpLit> {
    let s = s.trim();
    // Named specials (rare in clang output but legal).
    if s.eq_ignore_ascii_case("inf") || s.eq_ignore_ascii_case("+inf") {
        return Some(FpLit::Double(f64::INFINITY));
    }
    if s.eq_ignore_ascii_case("-inf") {
        return Some(FpLit::Double(f64::NEG_INFINITY));
    }
    if s.eq_ignore_ascii_case("nan") {
        return Some(FpLit::Double(f64::NAN));
    }

    // Hex bit patterns. LLVM uses an optional type tag after `0x`:
    //   0x<16 hex>            → f64 bit pattern (IEEE double)
    //   0xK<hex>              → x86 80-bit
    //   0xL<hex>              → ppc_fp128
    //   0xM<hex>              → IEEE f128
    //   0xH<hex>              → half (f16)
    //   0xR<hex>              → bfloat16
    if let Some(rest) = s.strip_prefix("0x").or_else(|| s.strip_prefix("0X")) {
        let first = rest.chars().next()?;
        if first.is_ascii_hexdigit() {
            // Plain 64-bit hex pattern.
            let bits = u64::from_str_radix(rest, 16).ok()?;
            return Some(FpLit::Double(f64::from_bits(bits)));
        } else if matches!(first, 'K' | 'L' | 'M' | 'H' | 'R' | 'k' | 'l' | 'm' | 'h' | 'r') {
            return Some(FpLit::Extended(first.to_ascii_uppercase()));
        } else {
            return None;
        }
    }

    // Decimal.
    s.parse::<f64>().ok().map(FpLit::Double)
}

fn parse_ty(s: &str) -> Result<Ty> {
    match s.trim() {
        "i1" => Ok(Ty::Bool),
        "i8" => Ok(Ty::I8),
        "i16" => Ok(Ty::I16),
        "i32" => Ok(Ty::I32),
        "i64" => Ok(Ty::I64),
        "i128" => Ok(Ty::I128),
        "ptr" => Ok(Ty::Ptr),
        "void" => Ok(Ty::Unit),
        // LLVM IR uses `float` / `double` as the textual spellings for
        // 32- and 64-bit IEEE-754 types. tMIR has native `F32` and
        // `F64` so the mapping is direct.
        "float" | "f32" => Ok(Ty::F32),
        "double" | "f64" => Ok(Ty::F64),
        // `half` (f16), `bfloat`, and the extended-precision 80/128-bit
        // types are legal in LLVM IR but tMIR only models f32 / f64
        // today. Reject with a precise reason so the WS2 driver
        // classifies the program as `unsupported` (not a crash).
        "half" => Err(Error::Unsupported(
            "half-precision float (tMIR has no f16 type)".to_string(),
        )),
        "bfloat" => Err(Error::Unsupported(
            "bfloat16 (tMIR has no bf16 type)".to_string(),
        )),
        "fp128" | "x86_fp80" | "ppc_fp128" => Err(Error::Unsupported(format!(
            "extended-precision float `{}` (tMIR only has f32/f64)",
            s.trim()
        ))),
        other if other.ends_with('*') => {
            // `i32*` or `i8*` legacy pointers — still occur occasionally.
            Ok(Ty::Ptr)
        }
        other if other.starts_with('[') || other.starts_with('<') => {
            Err(Error::Unsupported(format!(
                "aggregate / vector type `{}` (non-string context)",
                other
            )))
        }
        other => Err(Error::Unsupported(format!("type `{}`", other))),
    }
}

fn align_up(value: u64, align: u64) -> u64 {
    if align <= 1 {
        value
    } else {
        value.div_ceil(align) * align
    }
}

fn scalar_layout(ty: &Ty) -> Option<(u64, u64)> {
    match ty {
        Ty::Bool | Ty::I8 => Some((1, 1)),
        Ty::I16 => Some((2, 2)),
        Ty::I32 | Ty::F32 => Some((4, 4)),
        Ty::I64 | Ty::F64 | Ty::Ptr => Some((8, 8)),
        Ty::I128 => Some((16, 16)),
        _ => None,
    }
}

fn is_integer_ty(ty: &Ty) -> bool {
    matches!(ty, Ty::Bool | Ty::I8 | Ty::I16 | Ty::I32 | Ty::I64 | Ty::I128)
}

fn is_scalar_global_ty(ty: &Ty) -> bool {
    matches!(
        ty,
        Ty::Bool | Ty::I8 | Ty::I16 | Ty::I32 | Ty::I64 | Ty::I128 | Ty::F32 | Ty::F64 | Ty::Ptr
    )
}

fn parse_linkage(lower: &str) -> Linkage {
    if lower.contains("private") {
        Linkage::Private
    } else if lower.contains("internal") {
        Linkage::Internal
    } else {
        Linkage::External
    }
}

fn split_global_storage<'a>(rest: &'a str) -> Option<(bool, &'a str)> {
    let trimmed = rest.trim_start();
    if let Some(tail) = trimmed.strip_prefix("global ") {
        return Some((true, tail));
    }
    if let Some(tail) = trimmed.strip_prefix("global[") {
        return Some((true, tail));
    }
    if let Some(tail) = trimmed.strip_prefix("constant ") {
        return Some((false, tail));
    }
    if let Some(tail) = trimmed.strip_prefix("constant[") {
        return Some((false, tail));
    }

    let lower = rest.to_lowercase();
    for (pat, mutable) in [
        (" global ", true),
        (" global[", true),
        (" constant ", false),
        (" constant[", false),
    ] {
        if let Some(idx) = lower.find(pat) {
            return Some((mutable, &rest[idx + pat.len()..]));
        }
    }
    None
}

fn parse_align_clause(s: &str) -> Option<u64> {
    s.trim().strip_prefix("align ")?.trim().parse::<u64>().ok()
}

fn is_type_token(s: &str) -> bool {
    matches!(
        s,
        "i1" | "i8" | "i16" | "i32" | "i64" | "i128" | "ptr" | "void"
            | "f32" | "f64" | "float" | "double"
    ) || s.ends_with('*')
}

/// Find the byte offset of the first `@` at nesting depth 0, i.e. not
/// inside a `(...)` group. Used by `parse_call` so the explicit
/// function-type form `call i32 (ptr, ...) @printf(...)` is handled
/// correctly (we must not mistake the `,` inside `(ptr, ...)` for a
/// callee marker).
fn find_top_level_at(s: &str) -> Option<usize> {
    let mut depth = 0i32;
    for (i, c) in s.char_indices() {
        match c {
            '(' | '[' | '<' => depth += 1,
            ')' | ']' | '>' => depth -= 1,
            '@' if depth == 0 => return Some(i),
            _ => {}
        }
    }
    None
}

fn find_matching_paren(s: &str) -> Option<usize> {
    // s[0] must be '('. Find matching ')'.
    let mut depth = 0i32;
    for (i, c) in s.char_indices() {
        match c {
            '(' => depth += 1,
            ')' => {
                depth -= 1;
                if depth == 0 {
                    return Some(i);
                }
            }
            _ => {}
        }
    }
    None
}

fn split_call_args(s: &str) -> Vec<String> {
    // Split by top-level commas.
    let mut out = Vec::new();
    let mut depth = 0i32;
    let mut start = 0;
    let bytes = s.as_bytes();
    for (i, b) in bytes.iter().enumerate() {
        match *b {
            b'[' | b'(' | b'<' => depth += 1,
            b']' | b')' | b'>' => depth -= 1,
            b',' if depth == 0 => {
                out.push(s[start..i].trim().to_string());
                start = i + 1;
            }
            _ => {}
        }
    }
    if start < s.len() {
        let tail = s[start..].trim();
        if !tail.is_empty() {
            out.push(tail.to_string());
        }
    }
    out
}

fn find_ll_string_end(s: &str) -> Option<usize> {
    // LLVM strings terminate at an unescaped `"`. The only escape is
    // `\\xx` (hex) — `\\` and `\"` don't occur in clang-generated string
    // constants for our corpus. We just look for the first `"`.
    s.find('"')
}

fn decode_ll_string(s: &str) -> Vec<u8> {
    let mut out = Vec::with_capacity(s.len());
    let bytes = s.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'\\' && i + 2 < bytes.len() {
            let hi = (bytes[i + 1] as char).to_digit(16);
            let lo = (bytes[i + 2] as char).to_digit(16);
            if let (Some(h), Some(l)) = (hi, lo) {
                out.push(((h << 4) | l) as u8);
                i += 3;
                continue;
            }
        }
        out.push(bytes[i]);
        i += 1;
    }
    out
}

fn parse_int_literal(s: &str) -> Option<i128> {
    let s = s.trim();
    if let Some(hex) = s.strip_prefix("0x").or_else(|| s.strip_prefix("0X")) {
        return i128::from_str_radix(hex, 16).ok();
    }
    s.parse::<i128>().ok()
}

// --------------------------------------------------------------------------
// Tests
// --------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn struct_gep_offsets(f: &Function) -> Vec<i128> {
        let mut offsets = Vec::new();
        for blk in &f.blocks {
            for pair in blk.body.windows(2) {
                let [const_node, gep_node] = pair else {
                    continue;
                };
                if let (
                    Inst::Const {
                        ty: Ty::I64,
                        value: Constant::Int(offset),
                    },
                    Inst::GEP {
                        pointee_ty: Ty::I8,
                        indices,
                        ..
                    },
                ) = (&const_node.inst, &gep_node.inst)
                {
                    if indices.len() == 1
                        && const_node.results.len() == 1
                        && const_node.results[0] == indices[0]
                    {
                        offsets.push(*offset);
                    }
                }
            }
        }
        offsets
    }

    fn has_struct_alloca(f: &Function) -> bool {
        f.blocks.iter().any(|blk| {
            blk.body.iter().any(|node| {
                matches!(
                    &node.inst,
                    Inst::Alloca {
                        ty: Ty::I8,
                        count: Some(_),
                        ..
                    }
                )
            })
        })
    }

    #[test]
    fn trivial_ret_0() {
        let src = r#"
define i32 @main() {
entry:
  ret i32 0
}
"#;
        let m = import_text(src, "t").expect("parse");
        assert_eq!(m.functions.len(), 1);
        assert_eq!(m.functions[0].name, "main");
        assert_eq!(m.functions[0].blocks.len(), 1);
        assert!(m.functions[0].blocks[0].terminator().is_some());
    }

    #[test]
    fn add_sub_mul() {
        let src = r#"
define i32 @f(i32 %a, i32 %b) {
entry:
  %s = add nsw i32 %a, %b
  %d = sub nsw i32 %s, 1
  %p = mul nsw i32 %d, 2
  ret i32 %p
}
"#;
        let m = import_text(src, "t").expect("parse");
        assert_eq!(m.functions.len(), 1);
        let f = &m.functions[0];
        // 3 binops + 2 const materializations (for literals 1 and 2) + ret
        assert!(!f.blocks.is_empty());
    }

    #[test]
    fn load_store_alloca() {
        let src = r#"
define i32 @f(i32 %x) {
entry:
  %p = alloca i32, align 4
  store i32 %x, ptr %p, align 4
  %y = load i32, ptr %p, align 4
  ret i32 %y
}
"#;
        let m = import_text(src, "t").expect("parse");
        assert_eq!(m.functions.len(), 1);
    }

    #[test]
    fn icmp_and_condbr() {
        let src = r#"
define i32 @f(i32 %a) {
entry:
  %c = icmp slt i32 %a, 0
  br i1 %c, label %neg, label %pos
neg:
  ret i32 -1
pos:
  ret i32 1
}
"#;
        let m = import_text(src, "t").expect("parse");
        let f = &m.functions[0];
        assert_eq!(f.blocks.len(), 3);
    }

    #[test]
    fn rejects_phi() {
        let src = r#"
define i32 @f(i1 %c) {
entry:
  br i1 %c, label %t, label %f
t:
  br label %m
f:
  br label %m
m:
  %v = phi i32 [ 1, %t ], [ 2, %f ]
  ret i32 %v
}
"#;
        let r = import_text(src, "t");
        assert!(
            matches!(r, Err(Error::Unsupported(_))),
            "phi should be unsupported, got {:?}",
            r
        );
    }

    /// f32 `fadd` with a decimal FP immediate on the right-hand side.
    #[test]
    fn fadd_f32_const_rhs() {
        let src = r#"
define float @f(float %a) {
entry:
  %b = fadd float %a, 1.5
  ret float %b
}
"#;
        let m = import_text(src, "t").expect("parse");
        let f = &m.functions[0];
        // Body must contain a BinOp { op: FAdd, ty: F32, .. }.
        let mut saw_fadd = false;
        for blk in &f.blocks {
            for node in &blk.body {
                if let Inst::BinOp { op: BinOp::FAdd, ty: Ty::F32, .. } = &node.inst {
                    saw_fadd = true;
                }
            }
        }
        assert!(saw_fadd, "expected Inst::BinOp {{ FAdd, F32 }}");
    }

    /// f64 arithmetic covering fadd / fsub / fmul / fdiv in a single
    /// function. Confirms the dispatcher, flag-stripping, and decimal
    /// FP literal parsing all hold together.
    #[test]
    fn fbinop_f64_all_four() {
        let src = r#"
define double @f(double %a, double %b) {
entry:
  %s = fadd fast double %a, %b
  %d = fsub double %s, 1.0
  %p = fmul nnan nsz double %d, 2.0
  %q = fdiv double %p, 4.0
  ret double %q
}
"#;
        let m = import_text(src, "t").expect("parse");
        let f = &m.functions[0];
        let mut seen = std::collections::HashSet::new();
        for blk in &f.blocks {
            for node in &blk.body {
                if let Inst::BinOp { op, ty: Ty::F64, .. } = &node.inst {
                    seen.insert(*op);
                }
            }
        }
        for want in [BinOp::FAdd, BinOp::FSub, BinOp::FMul, BinOp::FDiv] {
            assert!(seen.contains(&want), "missing {:?} in {:?}", want, seen);
        }
    }

    /// `fneg` must map to `UnOp::FNeg` (not to a subtract-from-zero idiom).
    #[test]
    fn fneg_maps_to_unop() {
        let src = r#"
define double @f(double %a) {
entry:
  %b = fneg double %a
  ret double %b
}
"#;
        let m = import_text(src, "t").expect("parse");
        let f = &m.functions[0];
        let mut saw = false;
        for blk in &f.blocks {
            for node in &blk.body {
                if let Inst::UnOp { op: UnOp::FNeg, ty: Ty::F64, .. } = &node.inst {
                    saw = true;
                }
            }
        }
        assert!(saw, "expected Inst::UnOp {{ FNeg, F64 }}");
    }

    /// fcmp olt must map to FCmpOp::OLt on F64, and the result must be
    /// usable as a CondBr selector (i.e. typed Bool).
    #[test]
    fn fcmp_olt_drives_condbr() {
        let src = r#"
define i32 @f(double %x, double %y) {
entry:
  %c = fcmp olt double %x, %y
  br i1 %c, label %lt, label %ge
lt:
  ret i32 1
ge:
  ret i32 2
}
"#;
        let m = import_text(src, "t").expect("parse");
        let f = &m.functions[0];
        assert_eq!(f.blocks.len(), 3, "expected entry / lt / ge");
        let mut saw_fcmp = false;
        let mut saw_condbr = false;
        for blk in &f.blocks {
            for node in &blk.body {
                match &node.inst {
                    Inst::FCmp { op: FCmpOp::OLt, ty: Ty::F64, .. } => saw_fcmp = true,
                    Inst::CondBr { .. } => saw_condbr = true,
                    _ => {}
                }
            }
        }
        assert!(saw_fcmp, "expected FCmp OLt F64");
        assert!(saw_condbr, "expected CondBr terminator");
    }

    /// All 12 ordered / unordered FCmp predicates round-trip through
    /// `parse_fcmp`. `ord` / `uno` / `true` / `false` are covered by
    /// their own tests since they lower differently.
    #[test]
    fn fcmp_all_twelve_predicates() {
        for (ll, want) in [
            ("oeq", FCmpOp::OEq),
            ("one", FCmpOp::ONe),
            ("olt", FCmpOp::OLt),
            ("ole", FCmpOp::OLe),
            ("ogt", FCmpOp::OGt),
            ("oge", FCmpOp::OGe),
            ("ueq", FCmpOp::UEq),
            ("une", FCmpOp::UNe),
            ("ult", FCmpOp::ULt),
            ("ule", FCmpOp::ULe),
            ("ugt", FCmpOp::UGt),
            ("uge", FCmpOp::UGe),
        ] {
            let src = format!(
                "define i1 @f(double %a, double %b) {{\nentry:\n  %c = fcmp {} double %a, %b\n  ret i1 %c\n}}\n",
                ll
            );
            let m = import_text(&src, "t").expect("parse");
            let f = &m.functions[0];
            let mut got = None;
            for blk in &f.blocks {
                for node in &blk.body {
                    if let Inst::FCmp { op, .. } = &node.inst {
                        got = Some(*op);
                    }
                }
            }
            assert_eq!(got, Some(want), "fcmp {} mapped wrong", ll);
        }
    }

    /// `fcmp ord`/`uno` are synthesized as two self-comparisons combined
    /// with AND/OR because tMIR's FCmpOp enum does not include
    /// Ord/Uno. Verify we emit the expected shape.
    #[test]
    fn fcmp_ord_uno_are_synthesized() {
        let src = r#"
define i1 @f(double %a, double %b) {
entry:
  %c = fcmp ord double %a, %b
  ret i1 %c
}
"#;
        let m = import_text(src, "t").expect("parse");
        let f = &m.functions[0];
        let mut fcmps = 0usize;
        let mut ands = 0usize;
        for blk in &f.blocks {
            for node in &blk.body {
                match &node.inst {
                    Inst::FCmp { op: FCmpOp::OEq, .. } => fcmps += 1,
                    Inst::BinOp { op: BinOp::And, ty: Ty::Bool, .. } => ands += 1,
                    _ => {}
                }
            }
        }
        assert_eq!(fcmps, 2, "expected 2 self-comparisons for `ord`");
        assert_eq!(ands, 1, "expected 1 i1-And for `ord`");

        let src2 = r#"
define i1 @g(double %a, double %b) {
entry:
  %c = fcmp uno double %a, %b
  ret i1 %c
}
"#;
        let m = import_text(src2, "g").expect("parse");
        let f = &m.functions[0];
        let mut fcmps = 0usize;
        let mut ors = 0usize;
        for blk in &f.blocks {
            for node in &blk.body {
                match &node.inst {
                    Inst::FCmp { op: FCmpOp::UNe, .. } => fcmps += 1,
                    Inst::BinOp { op: BinOp::Or, ty: Ty::Bool, .. } => ors += 1,
                    _ => {}
                }
            }
        }
        assert_eq!(fcmps, 2, "expected 2 self-comparisons for `uno`");
        assert_eq!(ors, 1, "expected 1 i1-Or for `uno`");
    }

    /// `fcmp true`/`false` fold to constant i1 without consuming the
    /// operands at runtime.
    #[test]
    fn fcmp_true_false_fold_to_const() {
        for (ll, want) in [("true", true), ("false", false)] {
            let src = format!(
                "define i1 @f(double %a, double %b) {{\nentry:\n  %c = fcmp {} double %a, %b\n  ret i1 %c\n}}\n",
                ll
            );
            let m = import_text(&src, "t").expect("parse");
            let f = &m.functions[0];
            let mut saw = false;
            for blk in &f.blocks {
                for node in &blk.body {
                    if let Inst::Const { ty: Ty::Bool, value: Constant::Bool(b) } = &node.inst {
                        if *b == want {
                            saw = true;
                        }
                    }
                }
            }
            assert!(saw, "fcmp {} did not fold to Bool({})", ll, want);
        }
    }

    /// Hex-encoded f64 literals (`0x3FF8000000000000` == 1.5) round-trip
    /// through parse_fp_literal.
    #[test]
    fn fp_hex_literal_roundtrips() {
        let src = r#"
define double @f(double %a) {
entry:
  %b = fadd double %a, 0x3FF8000000000000
  ret double %b
}
"#;
        let m = import_text(src, "t").expect("parse");
        let f = &m.functions[0];
        let mut saw_const = false;
        for blk in &f.blocks {
            for node in &blk.body {
                if let Inst::Const { ty: Ty::F64, value: Constant::Float(v) } = &node.inst {
                    if (*v - 1.5).abs() < 1e-12 {
                        saw_const = true;
                    }
                }
            }
        }
        assert!(saw_const, "expected Float(1.5) materialization");
    }

    /// Extended-precision literal tags (`0xK...`) surface as a typed
    /// `Unsupported` rather than a crash.
    #[test]
    fn fp_extended_hex_literal_unsupported() {
        let src = r#"
define double @f(double %a) {
entry:
  %b = fadd double %a, 0xK3FFF8000000000000000
  ret double %b
}
"#;
        let r = import_text(src, "t");
        assert!(
            matches!(r, Err(Error::Unsupported(_))),
            "extended fp literal should be unsupported, got {:?}",
            r
        );
    }

    /// `half` / `bfloat` / `fp128` / `x86_fp80` / `ppc_fp128` stay
    /// Unsupported — tMIR has no matching type.
    #[test]
    fn other_float_widths_are_unsupported() {
        for ty in ["half", "bfloat", "fp128", "x86_fp80", "ppc_fp128"] {
            let src = format!(
                "define {} @f({} %a) {{\nentry:\n  ret {} %a\n}}\n",
                ty, ty, ty
            );
            let r = import_text(&src, "t");
            assert!(
                matches!(r, Err(Error::Unsupported(_))),
                "type {} should be unsupported, got {:?}",
                ty,
                r
            );
        }
    }

    /// FP casts: sitofp / fptosi / uitofp / fptoui / fpext / fptrunc.
    #[test]
    fn fp_casts_all_six() {
        let src = r#"
define double @f(i32 %a, double %b, float %c) {
entry:
  %x = sitofp i32 %a to double
  %y = uitofp i32 %a to double
  %z = fpext float %c to double
  %t = fptrunc double %b to float
  %u = fptosi double %b to i32
  %v = fptoui double %b to i32
  ret double %x
}
"#;
        let m = import_text(src, "t").expect("parse");
        let f = &m.functions[0];
        let mut ops = std::collections::HashSet::new();
        for blk in &f.blocks {
            for node in &blk.body {
                if let Inst::Cast { op, .. } = &node.inst {
                    ops.insert(*op);
                }
            }
        }
        for want in [
            CastOp::SIToFP,
            CastOp::UIToFP,
            CastOp::FPExt,
            CastOp::FPTrunc,
            CastOp::FPToSI,
            CastOp::FPToUI,
        ] {
            assert!(ops.contains(&want), "missing {:?} in {:?}", want, ops);
        }
    }

    /// The body form clang -O0 actually emits for
    /// `float add(float a, float b) { return a + b; }` on Apple
    /// Silicon, with anonymous-register SSA, must round-trip through
    /// the importer into a tmir::Module that has exactly one function
    /// with one block containing a `BinOp { FAdd, F32, .. }`.
    #[test]
    fn clang_style_f32_add_roundtrip() {
        let src = r#"
define float @add_f(float noundef %0, float noundef %1) {
  %3 = alloca float, align 4
  %4 = alloca float, align 4
  store float %0, ptr %3, align 4
  store float %1, ptr %4, align 4
  %5 = load float, ptr %3, align 4
  %6 = load float, ptr %4, align 4
  %7 = fadd float %5, %6
  ret float %7
}
"#;
        let m = import_text(src, "t").expect("parse");
        assert_eq!(m.functions.len(), 1);
        let f = &m.functions[0];
        assert_eq!(f.name, "add_f");
        let mut saw_fadd = false;
        for blk in &f.blocks {
            for node in &blk.body {
                if let Inst::BinOp { op: BinOp::FAdd, ty: Ty::F32, .. } = &node.inst {
                    saw_fadd = true;
                }
            }
        }
        assert!(saw_fadd);
    }

    #[test]
    fn switch_single_line_is_parsed() {
        // Regression for expansion item #4: simple inline switch.
        let src = r#"
define i32 @f(i32 %a) {
entry:
  switch i32 %a, label %d [ i32 1, label %one ]
one:
  ret i32 1
d:
  ret i32 0
}
"#;
        let m = import_text(src, "t").expect("parse");
        let f = &m.functions[0];
        // entry + one + d = 3 blocks.
        assert_eq!(f.blocks.len(), 3);
        // The entry-block terminator should be a Switch.
        let term = f.blocks[0]
            .terminator()
            .expect("entry block has a terminator");
        match &term.inst {
            Inst::Switch { cases, .. } => {
                assert_eq!(cases.len(), 1);
                assert_eq!(cases[0].value, Constant::Int(1));
            }
            other => panic!("expected Switch, got {:?}", other),
        }
    }

    #[test]
    fn switch_multiline_clang_shape() {
        // Matches what clang -O0 actually emits: header with `[`, one
        // case per line, closing `]` on its own line, numeric block
        // labels.
        let src = r#"
define i32 @dispatch(i32 %x) {
entry:
  switch i32 %x, label %d [
    i32 0, label %c0
    i32 1, label %c1
    i32 42, label %c42
  ]
c0:
  ret i32 10
c1:
  ret i32 20
c42:
  ret i32 30
d:
  ret i32 -1
}
"#;
        let m = import_text(src, "t").expect("parse");
        let f = &m.functions[0];
        // 1 entry + 4 targets = 5 blocks.
        assert_eq!(f.blocks.len(), 5);
        match &f.blocks[0].terminator().unwrap().inst {
            Inst::Switch { cases, default_args, .. } => {
                assert_eq!(cases.len(), 3);
                assert!(default_args.is_empty());
                let vals: Vec<i128> = cases
                    .iter()
                    .map(|c| match c.value {
                        Constant::Int(n) => n,
                        _ => panic!("non-int case"),
                    })
                    .collect();
                assert_eq!(vals, vec![0, 1, 42]);
                for c in cases {
                    assert!(c.args.is_empty(), "importer produces no edge args");
                }
            }
            other => panic!("expected Switch, got {:?}", other),
        }
    }

    #[test]
    fn switch_i8_widths() {
        // Exercise an i8 selector to confirm narrow-width support.
        let src = r#"
define i8 @f(i8 %x) {
entry:
  switch i8 %x, label %d [
    i8 0, label %z
    i8 1, label %o
  ]
z:
  ret i8 100
o:
  ret i8 101
d:
  ret i8 -1
}
"#;
        let m = import_text(src, "t").expect("parse");
        assert_eq!(m.functions.len(), 1);
    }

    #[test]
    fn switch_i64_widths() {
        let src = r#"
define i64 @f(i64 %x) {
entry:
  switch i64 %x, label %d [
    i64 9999999999, label %big
    i64 -1, label %neg
  ]
big:
  ret i64 1
neg:
  ret i64 2
d:
  ret i64 0
}
"#;
        let m = import_text(src, "t").expect("parse");
        assert_eq!(m.functions.len(), 1);
    }

    #[test]
    fn switch_empty_case_list_is_just_a_jump() {
        // `switch i32 %x, label %d []` is legal LLVM — equivalent to
        // `br label %d`. We lower it as a Switch with no cases; the
        // codegen side is responsible for turning that into a direct
        // jump.
        let src = r#"
define i32 @f(i32 %x) {
entry:
  switch i32 %x, label %d [ ]
d:
  ret i32 0
}
"#;
        let m = import_text(src, "t").expect("parse");
        let f = &m.functions[0];
        match &f.blocks[0].terminator().unwrap().inst {
            Inst::Switch { cases, .. } => assert!(cases.is_empty()),
            other => panic!("expected Switch, got {:?}", other),
        }
    }

    #[test]
    fn switch_case_type_mismatch_is_unsupported() {
        let src = r#"
define i32 @f(i32 %x) {
entry:
  switch i32 %x, label %d [
    i8 1, label %o
  ]
o:
  ret i32 1
d:
  ret i32 0
}
"#;
        let r = import_text(src, "t");
        assert!(
            matches!(r, Err(Error::Unsupported(_))),
            "mismatched case type should be unsupported, got {:?}",
            r
        );
    }

    #[test]
    fn struct_type_is_parsed() {
        let src = r#"
%struct.Pair = type { i32, i32 }

define void @f() {
entry:
  ret void
}
"#;
        let m = import_text(src, "t").expect("parse");
        assert!(m.structs.is_empty(), "layouts stay parser-internal");
        assert_eq!(m.functions.len(), 1);
    }

    #[test]
    fn alloca_struct_yields_i8_count() {
        let src = r#"
%struct.Pair = type { i32, i32 }

define void @f() {
entry:
  %p = alloca %struct.Pair, align 8
  ret void
}
"#;
        let m = import_text(src, "t").expect("parse");
        assert!(has_struct_alloca(&m.functions[0]));
    }

    #[test]
    fn struct_gep_field_0_offset_is_zero() {
        let src = r#"
%struct.Pair = type { i32, i32 }

define ptr @f(ptr %p) {
entry:
  %q = getelementptr inbounds %struct.Pair, ptr %p, i32 0, i32 0
  ret ptr %q
}
"#;
        let m = import_text(src, "t").expect("parse");
        assert_eq!(struct_gep_offsets(&m.functions[0]), vec![0]);
    }

    #[test]
    fn struct_gep_field_1_offset_is_field_offset() {
        let src = r#"
%struct.Pair = type { i32, i32 }
%struct.Misaligned = type { i8, i64 }

define ptr @pair(ptr %p) {
entry:
  %q = getelementptr inbounds %struct.Pair, ptr %p, i32 0, i32 1
  ret ptr %q
}

define ptr @mis(ptr %p) {
entry:
  %q = getelementptr inbounds %struct.Misaligned, ptr %p, i32 0, i32 1
  ret ptr %q
}
"#;
        let m = import_text(src, "t").expect("parse");
        assert_eq!(struct_gep_offsets(&m.functions[0]), vec![4]);
        assert_eq!(struct_gep_offsets(&m.functions[1]), vec![8]);
    }

    #[test]
    fn struct_gep_field_2_byte_offset_honors_align() {
        let src = r#"
%struct.Triple = type { i8, i32, i64 }

define ptr @f(ptr %p) {
entry:
  %field = getelementptr inbounds %struct.Triple, ptr %p, i32 0, i32 2
  %next = getelementptr inbounds %struct.Triple, ptr %p, i32 1
  ret ptr %next
}
"#;
        let m = import_text(src, "t").expect("parse");
        assert_eq!(struct_gep_offsets(&m.functions[0]), vec![8, 16]);
    }

    #[test]
    fn struct_e2e_roundtrip() {
        let src = r#"
%struct.Pair = type { i32, i32 }

define i32 @f() {
entry:
  %p = alloca %struct.Pair, align 8
  %a = getelementptr inbounds %struct.Pair, ptr %p, i32 0, i32 0
  %b = getelementptr inbounds %struct.Pair, ptr %p, i32 0, i32 1
  store i32 1, ptr %a, align 4
  store i32 2, ptr %b, align 4
  %x = load i32, ptr %a, align 4
  %y = load i32, ptr %b, align 4
  %z = add i32 %x, %y
  ret i32 %z
}
"#;
        let m = import_text(src, "t").expect("parse");
        assert_eq!(m.functions.len(), 1);
    }

    #[test]
    fn scalar_global_i32_is_parsed() {
        let src = r#"
@x = global i32 42
"#;
        let m = import_text(src, "t").expect("parse");
        assert_eq!(m.globals.len(), 1);
        assert_eq!(m.globals[0].name, "x");
        assert_eq!(m.globals[0].ty, Ty::I32);
        assert!(m.globals[0].mutable);
        assert_eq!(m.globals[0].initializer, Some(Constant::Int(42)));
    }

    #[test]
    fn scalar_global_i64_zeroinitializer() {
        let src = r#"
@y = global i64 zeroinitializer
"#;
        let m = import_text(src, "t").expect("parse");
        assert_eq!(m.globals[0].initializer, Some(Constant::Int(0)));
    }

    #[test]
    fn scalar_global_float() {
        let src = r#"
@z = global float 3.14
"#;
        let m = import_text(src, "t").expect("parse");
        match &m.globals[0].initializer {
            Some(Constant::Float(v)) => assert!((*v - 3.14).abs() < 1e-9),
            other => panic!("expected float initializer, got {:?}", other),
        }
    }

    #[test]
    fn global_with_constant_keyword_is_immutable() {
        let src = r#"
@k = constant i32 99
"#;
        let m = import_text(src, "t").expect("parse");
        assert!(!m.globals[0].mutable);
        assert_eq!(m.globals[0].initializer, Some(Constant::Int(99)));
    }

    #[test]
    fn string_global_is_parsed() {
        let src = r#"
@.str = private unnamed_addr constant [8 x i8] c"success\00", align 1

define void @f() {
entry:
  ret void
}
"#;
        let m = import_text(src, "t").expect("parse");
        assert_eq!(m.globals.len(), 1);
        assert_eq!(m.globals[0].name, ".str");
        if let Some(Constant::Aggregate(elems)) = &m.globals[0].initializer {
            assert_eq!(elems.len(), 8);
            // "success\0" = 8 bytes.
        } else {
            panic!("expected Aggregate initializer");
        }
    }

    #[test]
    fn cast_sext_zext() {
        let src = r#"
define i32 @f(i8 %a, i16 %b) {
entry:
  %x = sext i8 %a to i32
  %y = zext i16 %b to i32
  %s = add i32 %x, %y
  ret i32 %s
}
"#;
        let m = import_text(src, "t").expect("parse");
        assert_eq!(m.functions.len(), 1);
    }

    #[test]
    fn printf_call_signature_is_tolerated() {
        let src = r#"
declare i32 @printf(ptr noundef, ...)

define i32 @main() {
entry:
  ret i32 0
}
"#;
        let m = import_text(src, "t").expect("parse");
        // We register the printf declaration but do not require calls.
        assert!(m.func_types.len() >= 1);
    }

    #[test]
    fn global_address_in_call_is_unsupported() {
        let src = r#"
@.str = private unnamed_addr constant [6 x i8] c"hi!\0A\00", align 1

declare i32 @printf(ptr noundef, ...)

define i32 @main() {
entry:
  %r = call i32 (ptr, ...) @printf(ptr noundef @.str)
  ret i32 %r
}
"#;
        let r = import_text(src, "t");
        assert!(matches!(r, Err(Error::Unsupported(_))), "got {:?}", r);
    }
}
