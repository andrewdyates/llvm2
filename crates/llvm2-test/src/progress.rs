// Progress-bar helpers.
//
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Copyright 2026 Dropbox, Inc. | License: Apache-2.0

//! Progress UI via `indicatif`.
//!
//! Progress bars are silent when `--quiet` is passed or when
//! `--format json` is selected — never print progress to stdout.

use indicatif::{ProgressBar, ProgressStyle};

use crate::OutputFormat;

/// Return a progress bar, hidden when output should stay clean.
#[must_use]
pub fn bar(len: u64, quiet: bool, format: OutputFormat) -> ProgressBar {
    let is_json = matches!(format, OutputFormat::Json | OutputFormat::Junit);
    if quiet || is_json {
        return ProgressBar::hidden();
    }
    let pb = ProgressBar::new(len);
    let style = ProgressStyle::with_template(
        "{spinner} [{elapsed_precise}] [{bar:40}] {pos}/{len} {msg}",
    )
    .unwrap_or_else(|_| ProgressStyle::default_bar());
    pb.set_style(style.progress_chars("=>-"));
    pb
}
