// tcrane-frontend/parser.rs - CLIF text format parser
//
// Author: Andrew Yates <ayates@dropbox.com>
// Copyright 2026 Dropbox, Inc. | License: Apache-2.0

//! Parser for Cranelift's CLIF text format.

use tcrane_ir::Function;
use thiserror::Error;

/// Parse error.
#[derive(Debug, Error)]
pub enum ParseError {
    #[error("syntax error at line {line}: {message}")]
    Syntax { line: usize, message: String },
    #[error("unknown opcode: {0}")]
    UnknownOpcode(String),
}

/// Parse CLIF text format into a function.
pub fn parse_clif(_input: &str) -> Result<Function, ParseError> {
    // TODO: Implement CLIF parser
    Err(ParseError::Syntax {
        line: 1,
        message: "parser not yet implemented".to_string(),
    })
}
