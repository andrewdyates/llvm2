#!/bin/bash
# Copyright 2026 Dropbox, Inc.
# Author: Andrew Yates <ayates@dropbox.com>
#
# Generate AGENTS.md stub.
#
# Codex instructions are now injected by the looper at runtime via
# build_codex_context(). AGENTS.md is a minimal stub.
#
# Usage: ./ai_template_scripts/generate_agents_md.sh [output_file]

set -euo pipefail

OUTPUT="${1:-AGENTS.md}"

cat >"$OUTPUT" <<'EOF'
<!-- Codex instructions are injected by the looper at runtime. -->
<!-- Source: CLAUDE.md + .claude/rules/*.md + .claude/codex.md -->
EOF

echo "Generated $OUTPUT (stub - rules injected by looper at runtime)"
