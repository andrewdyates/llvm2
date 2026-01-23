# Codex Instructions

> Codex reads this file automatically. The full org rules are injected via the looper prompt.

## How It Works

1. **looper.py** reads `.claude/rules/*.md` and prepends to your prompt
2. **AGENTS.md** (this file) provides supplementary project context
3. Same rules apply as Claude - single source of truth in `.claude/rules/`

## Quick Reference

- **Commit template**: See ai_template.md (injected in prompt)
- **Issue linking**: `Fixes #N` (closes), `Part of #N` (links)
- **After work**: Create git commit immediately, don't ask permission

## Codex Notes

**Size limit**: 32KB default (`project_doc_max_bytes` in ~/.codex/config.toml)

**Hierarchy**: Codex reads AGENTS.md from ~/.codex/ down to cwd, concatenating files.
