// tcrane-frontend - CLIF parser and builder
//
// Author: Andrew Yates <ayates@dropbox.com>
// Copyright 2026 Dropbox, Inc. | License: Apache-2.0

//! CLIF text format parser and function builder.
//!
//! This crate provides:
//! - A parser for Cranelift's CLIF text format
//! - A builder API for constructing IR programmatically

pub mod parser;
pub mod builder;

pub use builder::FunctionBuilder;
