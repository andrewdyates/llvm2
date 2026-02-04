# Migration Checklist

Copyright 2026 Dropbox, Inc.
Author: Andrew Yates <ayates@dropbox.com>

Steps to align an existing repo with ai_template.

> **Note:** For new repos, `init_from_template.sh` handles hooks and labels automatically.
> This checklist is for **existing repos** that need alignment.

## Quick Alignment

1. Run audit: `./ai_template_scripts/audit_alignment.sh`
2. From `ai_template`, sync files: `./ai_template_scripts/sync_repo.sh /path/to/target_repo`
3. Re-run audit to verify

## Manual Steps

### Required Files
- [ ] `.claude/rules/ai_template.md` - Core rules
- [ ] `.claude/rules/org_chart.md` - Org reference
- [ ] `.claude/roles/*.md` - Role definitions
- [ ] `ai_template_scripts/` - Scripts

### Recommended Files
- [ ] `VISION.md` - Strategic direction
- [ ] `ideas/` directory - Future backlog
- [ ] `.pre-commit-config.yaml` - Pre-commit hooks

### Git Hooks
- [ ] Run `./ai_template_scripts/install_hooks.sh`
- [ ] Verify hooks: `ls -la .git/hooks/`

### Labels
- [ ] Run `./ai_template_scripts/init_labels.sh`
- [ ] Verify: `gh label list`

## Troubleshooting

### Hooks not working
```bash
./ai_template_scripts/install_hooks.sh
```

### Missing labels
```bash
./ai_template_scripts/init_labels.sh
```

### Preview sync changes
```bash
./ai_template_scripts/sync_repo.sh /path/to/target_repo --dry-run
```
