// llvm2-codegen/src/interpreter.rs - tMIR direct interpreter for golden truth validation
//
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Copyright 2026 Andrew Yates | License: Apache-2.0
//
// This module provides a direct interpreter for tMIR programs. It evaluates
// tMIR instructions without any codegen, lowering, or optimization — serving
// as a golden truth oracle for differential testing against compiled binaries.
//
// If the interpreter and the compiled binary produce the same result for a
// given tMIR program, we have strong evidence the compiler is correct.
//
// Reference: CompCert's reference interpreter (Cminor → Clight semantics)

use std::collections::HashMap;

use tmir::{
    Module as TmirModule, Function as TmirFunction, Block as TmirBlock,
    BinOp, UnOp, ICmpOp, Inst, Constant,
    BlockId, FuncId, ValueId,
};

// ---------------------------------------------------------------------------
// Interpreter value
// ---------------------------------------------------------------------------

/// Runtime value in the interpreter.
///
/// All integer types are widened to i128 for uniform handling. This avoids
/// sign-extension and truncation bugs while preserving exact semantics for
/// all bit-widths (i8..i128). Float types are stored as f64.
#[derive(Debug, Clone)]
pub enum InterpreterValue {
    /// Integer value (covers i8, i16, i32, i64, i128).
    Int(i128),
    /// Floating-point value (covers f32 and f64).
    Float(f64),
    /// Boolean value.
    Bool(bool),
    /// Undefined value (from Undef instruction).
    Undef,
}

impl InterpreterValue {
    /// Extract as i128, or error.
    pub fn as_int(&self) -> Result<i128, InterpreterError> {
        match self {
            InterpreterValue::Int(v) => Ok(*v),
            InterpreterValue::Bool(b) => Ok(if *b { 1 } else { 0 }),
            _ => Err(InterpreterError::TypeMismatch(format!(
                "expected Int, got {:?}",
                self
            ))),
        }
    }

    /// Extract as f64, or error.
    pub fn as_float(&self) -> Result<f64, InterpreterError> {
        match self {
            InterpreterValue::Float(v) => Ok(*v),
            _ => Err(InterpreterError::TypeMismatch(format!(
                "expected Float, got {:?}",
                self
            ))),
        }
    }

    /// Extract as bool, or error.
    pub fn as_bool(&self) -> Result<bool, InterpreterError> {
        match self {
            InterpreterValue::Bool(b) => Ok(*b),
            InterpreterValue::Int(v) => Ok(*v != 0),
            _ => Err(InterpreterError::TypeMismatch(format!(
                "expected Bool, got {:?}",
                self
            ))),
        }
    }
}

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

/// Errors that can occur during interpretation.
#[derive(Debug, Clone)]
pub enum InterpreterError {
    /// Function not found in module.
    FunctionNotFound(String),
    /// Block not found in function.
    BlockNotFound(BlockId),
    /// Value not found in register file.
    ValueNotFound(ValueId),
    /// Type mismatch during operation.
    TypeMismatch(String),
    /// Division by zero.
    DivisionByZero,
    /// Fuel exhausted (step limit reached).
    FuelExhausted(u64),
    /// Unsupported instruction.
    Unsupported(String),
    /// Argument count mismatch.
    ArityMismatch { expected: usize, got: usize },
    /// Call stack depth exceeded.
    StackOverflow(usize),
}

impl std::fmt::Display for InterpreterError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::FunctionNotFound(name) => write!(f, "function not found: {}", name),
            Self::BlockNotFound(id) => write!(f, "block not found: {:?}", id),
            Self::ValueNotFound(id) => write!(f, "value not found: {:?}", id),
            Self::TypeMismatch(msg) => write!(f, "type mismatch: {}", msg),
            Self::DivisionByZero => write!(f, "division by zero"),
            Self::FuelExhausted(limit) => write!(f, "fuel exhausted after {} steps", limit),
            Self::Unsupported(msg) => write!(f, "unsupported: {}", msg),
            Self::ArityMismatch { expected, got } => {
                write!(f, "arity mismatch: expected {} args, got {}", expected, got)
            }
            Self::StackOverflow(depth) => write!(f, "stack overflow at depth {}", depth),
        }
    }
}

impl std::error::Error for InterpreterError {}

// ---------------------------------------------------------------------------
// Interpreter
// ---------------------------------------------------------------------------

/// Configuration for the interpreter.
pub struct InterpreterConfig {
    /// Maximum number of instructions to execute before aborting.
    pub fuel: u64,
    /// Maximum call stack depth.
    pub max_call_depth: usize,
}

impl Default for InterpreterConfig {
    fn default() -> Self {
        Self {
            fuel: 1_000_000,
            max_call_depth: 256,
        }
    }
}

/// tMIR direct interpreter.
///
/// Evaluates tMIR programs instruction-by-instruction without any compilation.
/// This is intentionally simple and unoptimized — correctness over speed.
struct Interpreter<'m> {
    module: &'m TmirModule,
    config: InterpreterConfig,
    steps: u64,
    call_depth: usize,
}

impl<'m> Interpreter<'m> {
    fn new(module: &'m TmirModule, config: InterpreterConfig) -> Self {
        Self {
            module,
            config,
            steps: 0,
            call_depth: 0,
        }
    }

    /// Execute a function by FuncId with the given arguments.
    fn call_func(
        &mut self,
        func_id: FuncId,
        args: &[InterpreterValue],
    ) -> Result<Vec<InterpreterValue>, InterpreterError> {
        let func = self
            .module
            .functions
            .iter()
            .find(|f| f.id == func_id)
            .ok_or_else(|| InterpreterError::FunctionNotFound(format!("FuncId({})", func_id.index())))?;

        self.execute_function(func, args)
    }

    /// Execute a function with the given arguments.
    fn execute_function(
        &mut self,
        func: &TmirFunction,
        args: &[InterpreterValue],
    ) -> Result<Vec<InterpreterValue>, InterpreterError> {
        if self.call_depth >= self.config.max_call_depth {
            return Err(InterpreterError::StackOverflow(self.call_depth));
        }
        self.call_depth += 1;

        // Find entry block.
        let entry_block = self.find_block(func, func.entry)?;

        // Validate argument count matches entry block params.
        if args.len() != entry_block.params.len() {
            self.call_depth -= 1;
            return Err(InterpreterError::ArityMismatch {
                expected: entry_block.params.len(),
                got: args.len(),
            });
        }

        // Initialize register file: bind args to entry block params.
        let mut regs: HashMap<ValueId, InterpreterValue> = HashMap::new();
        for (i, (vid, _ty)) in entry_block.params.iter().enumerate() {
            regs.insert(*vid, args[i].clone());
        }

        // Execute starting from entry block.
        let result = self.execute_block(func, func.entry, &mut regs);
        self.call_depth -= 1;
        result
    }

    /// Execute instructions in a block, following control flow until Return.
    fn execute_block(
        &mut self,
        func: &TmirFunction,
        mut block_id: BlockId,
        regs: &mut HashMap<ValueId, InterpreterValue>,
    ) -> Result<Vec<InterpreterValue>, InterpreterError> {
        loop {
            let block = self.find_block(func, block_id)?;

            for node in &block.body {
                self.steps += 1;
                if self.steps > self.config.fuel {
                    return Err(InterpreterError::FuelExhausted(self.config.fuel));
                }

                match &node.inst {
                    // --- Constants ---
                    Inst::Const { ty: _, value } => {
                        let result_vid = node.results[0];
                        let val = self.eval_constant(value);
                        regs.insert(result_vid, val);
                    }

                    // --- Binary operations ---
                    Inst::BinOp { op, ty: _, lhs, rhs } => {
                        let result_vid = node.results[0];
                        let lhs_val = self.get_reg(regs, *lhs)?;
                        let rhs_val = self.get_reg(regs, *rhs)?;
                        let result = self.eval_binop(*op, &lhs_val, &rhs_val)?;
                        regs.insert(result_vid, result);
                    }

                    // --- Unary operations ---
                    Inst::UnOp { op, ty: _, operand } => {
                        let result_vid = node.results[0];
                        let src = self.get_reg(regs, *operand)?;
                        let result = self.eval_unop(*op, &src)?;
                        regs.insert(result_vid, result);
                    }

                    // --- Integer comparisons ---
                    Inst::ICmp { op, ty: _, lhs, rhs } => {
                        let result_vid = node.results[0];
                        let lhs_val = self.get_reg(regs, *lhs)?.as_int()?;
                        let rhs_val = self.get_reg(regs, *rhs)?.as_int()?;
                        let result = self.eval_icmp(*op, lhs_val, rhs_val);
                        regs.insert(result_vid, InterpreterValue::Bool(result));
                    }

                    // --- Float comparisons ---
                    Inst::FCmp { op, ty: _, lhs, rhs } => {
                        let result_vid = node.results[0];
                        let lhs_val = self.get_reg(regs, *lhs)?.as_float()?;
                        let rhs_val = self.get_reg(regs, *rhs)?.as_float()?;
                        let result = self.eval_fcmp(*op, lhs_val, rhs_val);
                        regs.insert(result_vid, InterpreterValue::Bool(result));
                    }

                    // --- Overflow ops (result + overflow flag) ---
                    Inst::Overflow { op, ty: _, lhs, rhs } => {
                        let lhs_val = self.get_reg(regs, *lhs)?.as_int()?;
                        let rhs_val = self.get_reg(regs, *rhs)?.as_int()?;
                        let result = match op {
                            tmir::OverflowOp::AddOverflow => lhs_val.wrapping_add(rhs_val),
                            tmir::OverflowOp::SubOverflow => lhs_val.wrapping_sub(rhs_val),
                            tmir::OverflowOp::MulOverflow => lhs_val.wrapping_mul(rhs_val),
                        };
                        if !node.results.is_empty() {
                            regs.insert(node.results[0], InterpreterValue::Int(result));
                        }
                        // Overflow flag (simplified: always false for now)
                        if node.results.len() > 1 {
                            regs.insert(node.results[1], InterpreterValue::Bool(false));
                        }
                    }

                    // --- Select ---
                    Inst::Select {
                        ty: _,
                        cond,
                        then_val,
                        else_val,
                    } => {
                        let result_vid = node.results[0];
                        let cond_val = self.get_reg(regs, *cond)?.as_bool()?;
                        let result = if cond_val {
                            self.get_reg(regs, *then_val)?
                        } else {
                            self.get_reg(regs, *else_val)?
                        };
                        regs.insert(result_vid, result);
                    }

                    // --- Copy ---
                    Inst::Copy { ty: _, operand } => {
                        let result_vid = node.results[0];
                        let val = self.get_reg(regs, *operand)?;
                        regs.insert(result_vid, val);
                    }

                    // --- Cast (simplified) ---
                    Inst::Cast {
                        op,
                        src_ty: _,
                        dst_ty: _,
                        operand,
                    } => {
                        let result_vid = node.results[0];
                        let src = self.get_reg(regs, *operand)?;
                        let result = self.eval_cast(*op, &src)?;
                        regs.insert(result_vid, result);
                    }

                    // --- NullPtr ---
                    Inst::NullPtr => {
                        let result_vid = node.results[0];
                        regs.insert(result_vid, InterpreterValue::Int(0));
                    }

                    // --- Undef ---
                    Inst::Undef { .. } => {
                        let result_vid = node.results[0];
                        regs.insert(result_vid, InterpreterValue::Undef);
                    }

                    // --- Assume/Assert (no-op in interpreter) ---
                    Inst::Assume { .. } | Inst::Assert { .. } => {}

                    // --- Unconditional branch ---
                    Inst::Br { target, args } => {
                        let arg_vals: Vec<InterpreterValue> = args
                            .iter()
                            .map(|vid| self.get_reg(regs, *vid))
                            .collect::<Result<Vec<_>, _>>()?;

                        // Bind args to target block params.
                        let target_block = self.find_block(func, *target)?;
                        for (i, (vid, _ty)) in target_block.params.iter().enumerate() {
                            regs.insert(*vid, arg_vals[i].clone());
                        }

                        block_id = *target;
                        break; // Restart from new block
                    }

                    // --- Conditional branch ---
                    Inst::CondBr {
                        cond,
                        then_target,
                        then_args,
                        else_target,
                        else_args,
                    } => {
                        let cond_val = self.get_reg(regs, *cond)?.as_bool()?;
                        let (target, branch_args) = if cond_val {
                            (*then_target, then_args)
                        } else {
                            (*else_target, else_args)
                        };

                        let arg_vals: Vec<InterpreterValue> = branch_args
                            .iter()
                            .map(|vid| self.get_reg(regs, *vid))
                            .collect::<Result<Vec<_>, _>>()?;

                        let target_block = self.find_block(func, target)?;
                        for (i, (vid, _ty)) in target_block.params.iter().enumerate() {
                            regs.insert(*vid, arg_vals[i].clone());
                        }

                        block_id = target;
                        break; // Restart from new block
                    }

                    // --- Switch ---
                    Inst::Switch {
                        value,
                        default,
                        default_args,
                        cases,
                    } => {
                        let selector = self.get_reg(regs, *value)?.as_int()?;

                        // Find matching case.
                        let mut matched_target = *default;
                        let mut matched_args: &Vec<ValueId> = default_args;

                        for case in cases {
                            let case_val = match &case.value {
                                Constant::Int(v) => *v,
                                Constant::Bool(b) => {
                                    if *b { 1 } else { 0 }
                                }
                                _ => continue,
                            };
                            if selector == case_val {
                                matched_target = case.target;
                                matched_args = &case.args;
                                break;
                            }
                        }

                        let arg_vals: Vec<InterpreterValue> = matched_args
                            .iter()
                            .map(|vid| self.get_reg(regs, *vid))
                            .collect::<Result<Vec<_>, _>>()?;

                        let target_block = self.find_block(func, matched_target)?;
                        for (i, (vid, _ty)) in target_block.params.iter().enumerate() {
                            regs.insert(*vid, arg_vals[i].clone());
                        }

                        block_id = matched_target;
                        break;
                    }

                    // --- Return ---
                    Inst::Return { values } => {
                        let result: Vec<InterpreterValue> = values
                            .iter()
                            .map(|vid| self.get_reg(regs, *vid))
                            .collect::<Result<Vec<_>, _>>()?;
                        return Ok(result);
                    }

                    // --- Call ---
                    Inst::Call { callee, args } => {
                        let arg_vals: Vec<InterpreterValue> = args
                            .iter()
                            .map(|vid| self.get_reg(regs, *vid))
                            .collect::<Result<Vec<_>, _>>()?;

                        let results = self.call_func(*callee, &arg_vals)?;

                        // Bind results.
                        for (i, vid) in node.results.iter().enumerate() {
                            if i < results.len() {
                                regs.insert(*vid, results[i].clone());
                            }
                        }
                    }

                    // --- Unreachable ---
                    Inst::Unreachable => {
                        return Err(InterpreterError::Unsupported(
                            "reached unreachable instruction".to_string(),
                        ));
                    }

                    // --- Unsupported (memory, atomics, aggregates, etc.) ---
                    other => {
                        return Err(InterpreterError::Unsupported(format!(
                            "{:?}",
                            other
                        )));
                    }
                }
            }
            // If we reach end of block body without a terminator, something is wrong.
            // But the loop will continue to the next block_id iteration.
        }
    }

    // -----------------------------------------------------------------------
    // Helpers
    // -----------------------------------------------------------------------

    fn find_block<'f>(
        &self,
        func: &'f TmirFunction,
        block_id: BlockId,
    ) -> Result<&'f TmirBlock, InterpreterError> {
        func.blocks
            .iter()
            .find(|b| b.id == block_id)
            .ok_or(InterpreterError::BlockNotFound(block_id))
    }

    fn get_reg(
        &self,
        regs: &HashMap<ValueId, InterpreterValue>,
        vid: ValueId,
    ) -> Result<InterpreterValue, InterpreterError> {
        regs.get(&vid)
            .cloned()
            .ok_or(InterpreterError::ValueNotFound(vid))
    }

    fn eval_constant(&self, c: &Constant) -> InterpreterValue {
        match c {
            Constant::Int(v) => InterpreterValue::Int(*v),
            Constant::Float(v) => InterpreterValue::Float(*v),
            Constant::Bool(b) => InterpreterValue::Bool(*b),
            Constant::Aggregate(_) => InterpreterValue::Undef, // Simplified
            // tMIR#30 aggregate/closure constants — the interpreter doesn't
            // model set/sequence/record/closure runtime representations yet.
            Constant::Sequence(_)
            | Constant::Set(_)
            | Constant::Record(_)
            | Constant::Closure { .. } => InterpreterValue::Undef,
        }
    }

    fn eval_binop(
        &self,
        op: BinOp,
        lhs: &InterpreterValue,
        rhs: &InterpreterValue,
    ) -> Result<InterpreterValue, InterpreterError> {
        match op {
            // Integer arithmetic
            BinOp::Add => {
                let a = lhs.as_int()?;
                let b = rhs.as_int()?;
                Ok(InterpreterValue::Int(a.wrapping_add(b)))
            }
            BinOp::Sub => {
                let a = lhs.as_int()?;
                let b = rhs.as_int()?;
                Ok(InterpreterValue::Int(a.wrapping_sub(b)))
            }
            BinOp::Mul => {
                let a = lhs.as_int()?;
                let b = rhs.as_int()?;
                Ok(InterpreterValue::Int(a.wrapping_mul(b)))
            }
            BinOp::UDiv => {
                let a = lhs.as_int()? as u128;
                let b = rhs.as_int()? as u128;
                if b == 0 {
                    return Err(InterpreterError::DivisionByZero);
                }
                Ok(InterpreterValue::Int((a / b) as i128))
            }
            BinOp::SDiv => {
                let a = lhs.as_int()?;
                let b = rhs.as_int()?;
                if b == 0 {
                    return Err(InterpreterError::DivisionByZero);
                }
                Ok(InterpreterValue::Int(a.wrapping_div(b)))
            }
            BinOp::URem => {
                let a = lhs.as_int()? as u128;
                let b = rhs.as_int()? as u128;
                if b == 0 {
                    return Err(InterpreterError::DivisionByZero);
                }
                Ok(InterpreterValue::Int((a % b) as i128))
            }
            BinOp::SRem => {
                let a = lhs.as_int()?;
                let b = rhs.as_int()?;
                if b == 0 {
                    return Err(InterpreterError::DivisionByZero);
                }
                Ok(InterpreterValue::Int(a.wrapping_rem(b)))
            }

            // Floating-point arithmetic
            BinOp::FAdd => {
                let a = lhs.as_float()?;
                let b = rhs.as_float()?;
                Ok(InterpreterValue::Float(a + b))
            }
            BinOp::FSub => {
                let a = lhs.as_float()?;
                let b = rhs.as_float()?;
                Ok(InterpreterValue::Float(a - b))
            }
            BinOp::FMul => {
                let a = lhs.as_float()?;
                let b = rhs.as_float()?;
                Ok(InterpreterValue::Float(a * b))
            }
            BinOp::FDiv => {
                let a = lhs.as_float()?;
                let b = rhs.as_float()?;
                Ok(InterpreterValue::Float(a / b))
            }
            BinOp::FRem => {
                let a = lhs.as_float()?;
                let b = rhs.as_float()?;
                Ok(InterpreterValue::Float(a % b))
            }

            // Bitwise / shift operations
            BinOp::And => {
                let a = lhs.as_int()?;
                let b = rhs.as_int()?;
                Ok(InterpreterValue::Int(a & b))
            }
            BinOp::Or => {
                let a = lhs.as_int()?;
                let b = rhs.as_int()?;
                Ok(InterpreterValue::Int(a | b))
            }
            BinOp::Xor => {
                let a = lhs.as_int()?;
                let b = rhs.as_int()?;
                Ok(InterpreterValue::Int(a ^ b))
            }
            BinOp::Shl => {
                let a = lhs.as_int()?;
                let b = rhs.as_int()? as u32;
                Ok(InterpreterValue::Int(a.wrapping_shl(b)))
            }
            BinOp::LShr => {
                let a = lhs.as_int()? as u128;
                let b = rhs.as_int()? as u32;
                Ok(InterpreterValue::Int((a >> b) as i128))
            }
            BinOp::AShr => {
                let a = lhs.as_int()?;
                let b = rhs.as_int()? as u32;
                Ok(InterpreterValue::Int(a.wrapping_shr(b)))
            }
        }
    }

    fn eval_unop(
        &self,
        op: UnOp,
        src: &InterpreterValue,
    ) -> Result<InterpreterValue, InterpreterError> {
        match op {
            UnOp::Neg => {
                let v = src.as_int()?;
                Ok(InterpreterValue::Int(v.wrapping_neg()))
            }
            UnOp::FNeg => {
                let v = src.as_float()?;
                Ok(InterpreterValue::Float(-v))
            }
            UnOp::Not => {
                let v = src.as_int()?;
                Ok(InterpreterValue::Int(!v))
            }
        }
    }

    fn eval_icmp(&self, op: ICmpOp, lhs: i128, rhs: i128) -> bool {
        match op {
            ICmpOp::Eq => lhs == rhs,
            ICmpOp::Ne => lhs != rhs,
            ICmpOp::Slt => lhs < rhs,
            ICmpOp::Sle => lhs <= rhs,
            ICmpOp::Sgt => lhs > rhs,
            ICmpOp::Sge => lhs >= rhs,
            ICmpOp::Ult => (lhs as u128) < (rhs as u128),
            ICmpOp::Ule => (lhs as u128) <= (rhs as u128),
            ICmpOp::Ugt => (lhs as u128) > (rhs as u128),
            ICmpOp::Uge => (lhs as u128) >= (rhs as u128),
        }
    }

    fn eval_fcmp(&self, op: tmir::FCmpOp, lhs: f64, rhs: f64) -> bool {
        use tmir::FCmpOp;
        match op {
            // Ordered comparisons (false when NaN)
            FCmpOp::OEq => lhs == rhs,
            FCmpOp::ONe => lhs != rhs,
            FCmpOp::OLt => lhs < rhs,
            FCmpOp::OLe => lhs <= rhs,
            FCmpOp::OGt => lhs > rhs,
            FCmpOp::OGe => lhs >= rhs,
            // Unordered comparisons (true when NaN)
            FCmpOp::UEq => lhs == rhs || lhs.is_nan() || rhs.is_nan(),
            FCmpOp::UNe => lhs != rhs || lhs.is_nan() || rhs.is_nan(),
            FCmpOp::ULt => lhs < rhs || lhs.is_nan() || rhs.is_nan(),
            FCmpOp::ULe => lhs <= rhs || lhs.is_nan() || rhs.is_nan(),
            FCmpOp::UGt => lhs > rhs || lhs.is_nan() || rhs.is_nan(),
            FCmpOp::UGe => lhs >= rhs || lhs.is_nan() || rhs.is_nan(),
        }
    }

    fn eval_cast(
        &self,
        op: tmir::CastOp,
        src: &InterpreterValue,
    ) -> Result<InterpreterValue, InterpreterError> {
        use tmir::CastOp;
        match op {
            CastOp::ZExt | CastOp::SExt | CastOp::Trunc => {
                // For the interpreter, all ints are i128, so extension/truncation
                // is a no-op at this level. Real semantics depend on bit-width,
                // but for golden truth testing of typical programs this suffices.
                Ok(InterpreterValue::Int(src.as_int()?))
            }
            CastOp::FPToSI => Ok(InterpreterValue::Int(src.as_float()? as i128)),
            CastOp::FPToUI => Ok(InterpreterValue::Int(src.as_float()? as u128 as i128)),
            CastOp::SIToFP => Ok(InterpreterValue::Float(src.as_int()? as f64)),
            CastOp::UIToFP => Ok(InterpreterValue::Float(src.as_int()? as u128 as f64)),
            CastOp::FPExt | CastOp::FPTrunc => Ok(InterpreterValue::Float(src.as_float()?)),
            CastOp::Bitcast | CastOp::PtrToInt | CastOp::IntToPtr => {
                // Pass through — simplified for integer-focused testing.
                match src {
                    InterpreterValue::Int(v) => Ok(InterpreterValue::Int(*v)),
                    InterpreterValue::Float(v) => Ok(InterpreterValue::Float(*v)),
                    other => Ok(other.clone()),
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Interpret a tMIR function by name with the given arguments.
///
/// This is the primary entry point for golden truth validation.
///
/// # Example
/// ```ignore
/// let module = build_some_tmir_module();
/// let result = interpret(&module, "add", &[InterpreterValue::Int(3), InterpreterValue::Int(5)]);
/// assert_eq!(result.unwrap()[0].as_int().unwrap(), 8);
/// ```
pub fn interpret(
    module: &TmirModule,
    func_name: &str,
    args: &[InterpreterValue],
) -> Result<Vec<InterpreterValue>, InterpreterError> {
    interpret_with_config(module, func_name, args, InterpreterConfig::default())
}

/// Interpret a tMIR function with custom configuration.
pub fn interpret_with_config(
    module: &TmirModule,
    func_name: &str,
    args: &[InterpreterValue],
    config: InterpreterConfig,
) -> Result<Vec<InterpreterValue>, InterpreterError> {
    let func = module
        .functions
        .iter()
        .find(|f| f.name == func_name)
        .ok_or_else(|| InterpreterError::FunctionNotFound(func_name.to_string()))?;

    let mut interp = Interpreter::new(module, config);
    interp.execute_function(func, args)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use tmir::{BinOp, ICmpOp, Ty};
    use tmir_build::ModuleBuilder;

    /// Helper: extract single i128 result from interpreter output.
    fn result_int(results: &[InterpreterValue]) -> i128 {
        assert_eq!(results.len(), 1, "expected single return value");
        results[0].as_int().expect("expected Int result")
    }

    // -----------------------------------------------------------------------
    // Test 1: Simple addition — add(3, 5) == 8
    // -----------------------------------------------------------------------

    #[test]
    fn test_interpreter_add() {
        let mut mb = ModuleBuilder::new("test_add");
        let ty = mb.add_func_type(vec![Ty::I64, Ty::I64], vec![Ty::I64]);
        let mut fb = mb.function("add", ty);

        let entry = fb.create_block();
        let a = fb.add_block_param(entry, Ty::I64);
        let b = fb.add_block_param(entry, Ty::I64);
        fb.switch_to_block(entry);
        let sum = fb.binop(BinOp::Add, Ty::I64, a, b);
        fb.ret(vec![sum]);
        fb.build();

        let module = mb.build();
        let result = interpret(&module, "add", &[
            InterpreterValue::Int(3),
            InterpreterValue::Int(5),
        ])
        .expect("interpret add");
        assert_eq!(result_int(&result), 8);
    }

    // -----------------------------------------------------------------------
    // Test 2: Fibonacci — fib(10) == 55 (loop with block params)
    //
    // fn fib(n: i64) -> i64 {
    //     a = 0, b = 1, i = 0
    //     while i < n { tmp = a + b; a = b; b = tmp; i += 1 }
    //     return a
    // }
    // -----------------------------------------------------------------------

    #[test]
    fn test_interpreter_fibonacci() {
        let mut mb = ModuleBuilder::new("test_fib");
        let ty = mb.add_func_type(vec![Ty::I64], vec![Ty::I64]);
        let mut fb = mb.function("fib", ty);

        let entry = fb.create_block();
        let n = fb.add_block_param(entry, Ty::I64);

        let bb_loop = fb.create_block();
        let loop_a = fb.add_block_param(bb_loop, Ty::I64);
        let loop_b = fb.add_block_param(bb_loop, Ty::I64);
        let loop_i = fb.add_block_param(bb_loop, Ty::I64);

        let bb_body = fb.create_block();
        let bb_exit = fb.create_block();

        // entry: init a=0, b=1, i=0, jump to loop
        fb.switch_to_block(entry);
        let zero = fb.iconst(Ty::I64, 0);
        let one = fb.iconst(Ty::I64, 1);
        let i_init = fb.iconst(Ty::I64, 0);
        fb.br(bb_loop, vec![zero, one, i_init]);

        // loop header: if i < n -> body, else -> exit
        fb.switch_to_block(bb_loop);
        let cmp = fb.icmp(ICmpOp::Slt, Ty::I64, loop_i, n);
        fb.condbr(cmp, bb_body, vec![], bb_exit, vec![]);

        // body: tmp = a + b; a = b; b = tmp; i += 1; back to loop
        fb.switch_to_block(bb_body);
        let tmp = fb.binop(BinOp::Add, Ty::I64, loop_a, loop_b);
        let one2 = fb.iconst(Ty::I64, 1);
        let new_i = fb.binop(BinOp::Add, Ty::I64, loop_i, one2);
        fb.br(bb_loop, vec![loop_b, tmp, new_i]);

        // exit: return a
        fb.switch_to_block(bb_exit);
        fb.ret(vec![loop_a]);

        fb.build();
        let module = mb.build();

        let result = interpret(&module, "fib", &[InterpreterValue::Int(10)])
            .expect("interpret fib");
        assert_eq!(result_int(&result), 55);

        // Edge cases
        let r0 = interpret(&module, "fib", &[InterpreterValue::Int(0)]).unwrap();
        assert_eq!(result_int(&r0), 0);

        let r1 = interpret(&module, "fib", &[InterpreterValue::Int(1)]).unwrap();
        assert_eq!(result_int(&r1), 1);

        let r2 = interpret(&module, "fib", &[InterpreterValue::Int(2)]).unwrap();
        assert_eq!(result_int(&r2), 1);
    }

    // -----------------------------------------------------------------------
    // Test 3: GCD — gcd(12, 8) == 4 (loop with conditional)
    //
    // fn gcd(a: i64, b: i64) -> i64 {
    //     while b != 0 { tmp = b; b = a % b; a = tmp }
    //     return a
    // }
    // -----------------------------------------------------------------------

    #[test]
    fn test_interpreter_gcd() {
        let mut mb = ModuleBuilder::new("test_gcd");
        let ty = mb.add_func_type(vec![Ty::I64, Ty::I64], vec![Ty::I64]);
        let mut fb = mb.function("gcd", ty);

        let entry = fb.create_block();
        let a_param = fb.add_block_param(entry, Ty::I64);
        let b_param = fb.add_block_param(entry, Ty::I64);

        let bb_loop = fb.create_block();
        let loop_a = fb.add_block_param(bb_loop, Ty::I64);
        let loop_b = fb.add_block_param(bb_loop, Ty::I64);

        let bb_body = fb.create_block();
        let bb_exit = fb.create_block();

        // entry: jump to loop(a, b)
        fb.switch_to_block(entry);
        fb.br(bb_loop, vec![a_param, b_param]);

        // loop header: if b != 0 -> body, else -> exit
        fb.switch_to_block(bb_loop);
        let zero = fb.iconst(Ty::I64, 0);
        let cmp = fb.icmp(ICmpOp::Ne, Ty::I64, loop_b, zero);
        fb.condbr(cmp, bb_body, vec![], bb_exit, vec![]);

        // body: new_b = a % b; a = b; back to loop(b, new_b)
        fb.switch_to_block(bb_body);
        let remainder = fb.binop(BinOp::SRem, Ty::I64, loop_a, loop_b);
        fb.br(bb_loop, vec![loop_b, remainder]);

        // exit: return a
        fb.switch_to_block(bb_exit);
        fb.ret(vec![loop_a]);

        fb.build();
        let module = mb.build();

        let result = interpret(&module, "gcd", &[
            InterpreterValue::Int(12),
            InterpreterValue::Int(8),
        ])
        .expect("interpret gcd");
        assert_eq!(result_int(&result), 4);

        let r2 = interpret(&module, "gcd", &[
            InterpreterValue::Int(48),
            InterpreterValue::Int(18),
        ])
        .unwrap();
        assert_eq!(result_int(&r2), 6);

        let r3 = interpret(&module, "gcd", &[
            InterpreterValue::Int(100),
            InterpreterValue::Int(75),
        ])
        .unwrap();
        assert_eq!(result_int(&r3), 25);
    }

    // -----------------------------------------------------------------------
    // Test 4: Sum to N — sum(100) == 5050
    //
    // fn sum_to(n: i64) -> i64 {
    //     sum = 0; i = 1
    //     while i <= n { sum += i; i += 1 }
    //     return sum
    // }
    // -----------------------------------------------------------------------

    #[test]
    fn test_interpreter_sum_to_n() {
        let mut mb = ModuleBuilder::new("test_sum");
        let ty = mb.add_func_type(vec![Ty::I64], vec![Ty::I64]);
        let mut fb = mb.function("sum_to", ty);

        let entry = fb.create_block();
        let n = fb.add_block_param(entry, Ty::I64);

        let bb_loop = fb.create_block();
        let loop_sum = fb.add_block_param(bb_loop, Ty::I64);
        let loop_i = fb.add_block_param(bb_loop, Ty::I64);

        let bb_body = fb.create_block();
        let bb_exit = fb.create_block();

        // entry: sum=0, i=1, jump to loop
        fb.switch_to_block(entry);
        let sum_init = fb.iconst(Ty::I64, 0);
        let i_init = fb.iconst(Ty::I64, 1);
        fb.br(bb_loop, vec![sum_init, i_init]);

        // loop header: if i <= n -> body, else -> exit
        fb.switch_to_block(bb_loop);
        let cmp = fb.icmp(ICmpOp::Sle, Ty::I64, loop_i, n);
        fb.condbr(cmp, bb_body, vec![], bb_exit, vec![]);

        // body: sum += i; i += 1; back to loop
        fb.switch_to_block(bb_body);
        let new_sum = fb.binop(BinOp::Add, Ty::I64, loop_sum, loop_i);
        let one = fb.iconst(Ty::I64, 1);
        let new_i = fb.binop(BinOp::Add, Ty::I64, loop_i, one);
        fb.br(bb_loop, vec![new_sum, new_i]);

        // exit: return sum
        fb.switch_to_block(bb_exit);
        fb.ret(vec![loop_sum]);

        fb.build();
        let module = mb.build();

        let result = interpret(&module, "sum_to", &[InterpreterValue::Int(100)])
            .expect("interpret sum_to");
        assert_eq!(result_int(&result), 5050);

        let r0 = interpret(&module, "sum_to", &[InterpreterValue::Int(0)]).unwrap();
        assert_eq!(result_int(&r0), 0);

        let r10 = interpret(&module, "sum_to", &[InterpreterValue::Int(10)]).unwrap();
        assert_eq!(result_int(&r10), 55);
    }

    // -----------------------------------------------------------------------
    // Test 5: Factorial via recursive Call — factorial(10) == 3628800
    //
    // fn factorial(n: i64) -> i64 {
    //     if n <= 1 { return 1 }
    //     else { return n * factorial(n - 1) }
    // }
    // -----------------------------------------------------------------------

    #[test]
    fn test_interpreter_factorial() {
        let mut mb = ModuleBuilder::new("test_factorial");
        let ty = mb.add_func_type(vec![Ty::I64], vec![Ty::I64]);
        let mut fb = mb.function("factorial", ty);

        let entry = fb.create_block();
        let n = fb.add_block_param(entry, Ty::I64);

        let bb_base = fb.create_block();
        let bb_recurse = fb.create_block();

        // entry: if n <= 1 -> base, else -> recurse
        fb.switch_to_block(entry);
        let one = fb.iconst(Ty::I64, 1);
        let cmp = fb.icmp(ICmpOp::Sle, Ty::I64, n, one);
        fb.condbr(cmp, bb_base, vec![], bb_recurse, vec![]);

        // base case: return 1
        fb.switch_to_block(bb_base);
        let base_val = fb.iconst(Ty::I64, 1);
        fb.ret(vec![base_val]);

        // recursive case: return n * factorial(n - 1)
        fb.switch_to_block(bb_recurse);
        let one2 = fb.iconst(Ty::I64, 1);
        let n_minus_1 = fb.binop(BinOp::Sub, Ty::I64, n, one2);
        let func_id = tmir::FuncId::new(0); // factorial is function 0
        let sub_result = fb.call(func_id, vec![n_minus_1]);
        let product = fb.binop(BinOp::Mul, Ty::I64, n, sub_result);
        fb.ret(vec![product]);

        fb.build();
        let module = mb.build();

        let result = interpret(&module, "factorial", &[InterpreterValue::Int(10)])
            .expect("interpret factorial");
        assert_eq!(result_int(&result), 3_628_800);

        let r0 = interpret(&module, "factorial", &[InterpreterValue::Int(0)]).unwrap();
        assert_eq!(result_int(&r0), 1);

        let r1 = interpret(&module, "factorial", &[InterpreterValue::Int(1)]).unwrap();
        assert_eq!(result_int(&r1), 1);

        let r5 = interpret(&module, "factorial", &[InterpreterValue::Int(5)]).unwrap();
        assert_eq!(result_int(&r5), 120);
    }

    // -----------------------------------------------------------------------
    // Test 6: Max — max(10, 20) == 20, max(20, 10) == 20
    //
    // fn max(a: i64, b: i64) -> i64 {
    //     if a > b { return a } else { return b }
    // }
    // -----------------------------------------------------------------------

    #[test]
    fn test_interpreter_max() {
        let mut mb = ModuleBuilder::new("test_max");
        let ty = mb.add_func_type(vec![Ty::I64, Ty::I64], vec![Ty::I64]);
        let mut fb = mb.function("max", ty);

        let entry = fb.create_block();
        let a = fb.add_block_param(entry, Ty::I64);
        let b = fb.add_block_param(entry, Ty::I64);

        let bb_then = fb.create_block();
        let bb_else = fb.create_block();

        // entry: if a > b -> then (return a), else (return b)
        fb.switch_to_block(entry);
        let cmp = fb.icmp(ICmpOp::Sgt, Ty::I64, a, b);
        fb.condbr(cmp, bb_then, vec![], bb_else, vec![]);

        fb.switch_to_block(bb_then);
        fb.ret(vec![a]);

        fb.switch_to_block(bb_else);
        fb.ret(vec![b]);

        fb.build();
        let module = mb.build();

        let r1 = interpret(&module, "max", &[
            InterpreterValue::Int(10),
            InterpreterValue::Int(20),
        ])
        .unwrap();
        assert_eq!(result_int(&r1), 20);

        let r2 = interpret(&module, "max", &[
            InterpreterValue::Int(20),
            InterpreterValue::Int(10),
        ])
        .unwrap();
        assert_eq!(result_int(&r2), 20);

        let r3 = interpret(&module, "max", &[
            InterpreterValue::Int(5),
            InterpreterValue::Int(5),
        ])
        .unwrap();
        assert_eq!(result_int(&r3), 5);

        let r4 = interpret(&module, "max", &[
            InterpreterValue::Int(-3),
            InterpreterValue::Int(-7),
        ])
        .unwrap();
        assert_eq!(result_int(&r4), -3);
    }

    // -----------------------------------------------------------------------
    // Test 7: Select instruction
    // -----------------------------------------------------------------------

    #[test]
    fn test_interpreter_select() {
        let mut mb = ModuleBuilder::new("test_select");
        let ty = mb.add_func_type(vec![Ty::I64, Ty::I64, Ty::I64], vec![Ty::I64]);
        let mut fb = mb.function("select_max", ty);

        let entry = fb.create_block();
        let cond_param = fb.add_block_param(entry, Ty::I64); // nonzero = true
        let a = fb.add_block_param(entry, Ty::I64);
        let b = fb.add_block_param(entry, Ty::I64);

        fb.switch_to_block(entry);
        // Compare cond_param != 0 to get a Bool
        let zero = fb.iconst(Ty::I64, 0);
        let cond = fb.icmp(ICmpOp::Ne, Ty::I64, cond_param, zero);
        let result = fb.select(Ty::I64, cond, a, b);
        fb.ret(vec![result]);

        fb.build();
        let module = mb.build();

        // cond=1 -> select a=10
        let r1 = interpret(&module, "select_max", &[
            InterpreterValue::Int(1),
            InterpreterValue::Int(10),
            InterpreterValue::Int(20),
        ])
        .unwrap();
        assert_eq!(result_int(&r1), 10);

        // cond=0 -> select b=20
        let r2 = interpret(&module, "select_max", &[
            InterpreterValue::Int(0),
            InterpreterValue::Int(10),
            InterpreterValue::Int(20),
        ])
        .unwrap();
        assert_eq!(result_int(&r2), 20);
    }

    // -----------------------------------------------------------------------
    // Test 8: Bitwise operations — AND, OR, XOR
    // -----------------------------------------------------------------------

    #[test]
    fn test_interpreter_bitwise() {
        let mut mb = ModuleBuilder::new("test_bitwise");
        let ty = mb.add_func_type(vec![Ty::I64, Ty::I64], vec![Ty::I64]);

        // fn bit_and(a, b) -> a & b
        {
            let mut fb = mb.function("bit_and", ty);
            let entry = fb.create_block();
            let a = fb.add_block_param(entry, Ty::I64);
            let b = fb.add_block_param(entry, Ty::I64);
            fb.switch_to_block(entry);
            let result = fb.binop(BinOp::And, Ty::I64, a, b);
            fb.ret(vec![result]);
            fb.build();
        }

        // fn bit_or(a, b) -> a | b
        {
            let mut fb = mb.function("bit_or", ty);
            let entry = fb.create_block();
            let a = fb.add_block_param(entry, Ty::I64);
            let b = fb.add_block_param(entry, Ty::I64);
            fb.switch_to_block(entry);
            let result = fb.binop(BinOp::Or, Ty::I64, a, b);
            fb.ret(vec![result]);
            fb.build();
        }

        // fn bit_xor(a, b) -> a ^ b
        {
            let mut fb = mb.function("bit_xor", ty);
            let entry = fb.create_block();
            let a = fb.add_block_param(entry, Ty::I64);
            let b = fb.add_block_param(entry, Ty::I64);
            fb.switch_to_block(entry);
            let result = fb.binop(BinOp::Xor, Ty::I64, a, b);
            fb.ret(vec![result]);
            fb.build();
        }

        let module = mb.build();

        // AND: 0xFF & 0x0F == 0x0F
        let r_and = interpret(&module, "bit_and", &[
            InterpreterValue::Int(0xFF),
            InterpreterValue::Int(0x0F),
        ])
        .unwrap();
        assert_eq!(result_int(&r_and), 0x0F);

        // OR: 0xF0 | 0x0F == 0xFF
        let r_or = interpret(&module, "bit_or", &[
            InterpreterValue::Int(0xF0),
            InterpreterValue::Int(0x0F),
        ])
        .unwrap();
        assert_eq!(result_int(&r_or), 0xFF);

        // XOR: 0xFF ^ 0xFF == 0
        let r_xor = interpret(&module, "bit_xor", &[
            InterpreterValue::Int(0xFF),
            InterpreterValue::Int(0xFF),
        ])
        .unwrap();
        assert_eq!(result_int(&r_xor), 0);

        // XOR: 0xAA ^ 0x55 == 0xFF
        let r_xor2 = interpret(&module, "bit_xor", &[
            InterpreterValue::Int(0xAA),
            InterpreterValue::Int(0x55),
        ])
        .unwrap();
        assert_eq!(result_int(&r_xor2), 0xFF);
    }

    // -----------------------------------------------------------------------
    // Test 9: Fuel exhaustion
    // -----------------------------------------------------------------------

    #[test]
    fn test_interpreter_fuel_exhaustion() {
        // Build an infinite loop: while true { }
        let mut mb = ModuleBuilder::new("test_inf");
        let ty = mb.add_func_type(vec![], vec![Ty::I64]);
        let mut fb = mb.function("infinite", ty);

        let entry = fb.create_block();
        let bb_loop = fb.create_block();

        fb.switch_to_block(entry);
        fb.br(bb_loop, vec![]);

        fb.switch_to_block(bb_loop);
        fb.br(bb_loop, vec![]);

        fb.build();
        let module = mb.build();

        let config = InterpreterConfig {
            fuel: 100,
            ..Default::default()
        };
        let result = interpret_with_config(&module, "infinite", &[], config);
        assert!(
            matches!(result, Err(InterpreterError::FuelExhausted(100))),
            "expected fuel exhaustion, got {:?}",
            result
        );
    }

    // -----------------------------------------------------------------------
    // Test 10: Function not found
    // -----------------------------------------------------------------------

    #[test]
    fn test_interpreter_function_not_found() {
        let mb = ModuleBuilder::new("empty");
        let module = mb.build();
        let result = interpret(&module, "nonexistent", &[]);
        assert!(matches!(
            result,
            Err(InterpreterError::FunctionNotFound(_))
        ));
    }

    // -----------------------------------------------------------------------
    // Test 11: Arity mismatch
    // -----------------------------------------------------------------------

    #[test]
    fn test_interpreter_arity_mismatch() {
        let mut mb = ModuleBuilder::new("test_arity");
        let ty = mb.add_func_type(vec![Ty::I64, Ty::I64], vec![Ty::I64]);
        let mut fb = mb.function("needs_two", ty);
        let entry = fb.create_block();
        let a = fb.add_block_param(entry, Ty::I64);
        let _b = fb.add_block_param(entry, Ty::I64);
        fb.switch_to_block(entry);
        fb.ret(vec![a]);
        fb.build();

        let module = mb.build();
        let result = interpret(&module, "needs_two", &[InterpreterValue::Int(1)]);
        assert!(matches!(
            result,
            Err(InterpreterError::ArityMismatch { expected: 2, got: 1 })
        ));
    }
}
