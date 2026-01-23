# Design: Gitignore Preservation During Sync

**Date:** 2026-01-12
**Issue:** #130
**Author:** RESEARCHER

## Problem

`sync_repo.sh` copies `.gitignore` wholesale, overwriting project-specific entries.

**Example:** tla2 had `test_specs/states/` in gitignore (model checker debug output). After sync, this was lost because ai_template's gitignore doesn't include it.

## Options Considered

### Option 1: Skip gitignore entirely
Let each project manage its own.

**Pros:** Simple, no merge logic needed.
**Cons:** Projects miss template updates (new security exclusions, runtime files). Template patterns would diverge over time.

### Option 2: Merge with deduplication
Combine entries from both files, removing duplicates.

**Pros:** All entries preserved automatically.
**Cons:** Complex implementation. Pattern ordering matters (later patterns override earlier). Could introduce subtle bugs. Hard to audit what came from where.

### Option 3: Marker-based preservation (RECOMMENDED)
Add a marker comment to template. Everything after marker in target is preserved during sync.

**Pros:**
- Clear separation: template vs project
- Easy to audit (marker is visible)
- Simple implementation
- Projects still get template updates

**Cons:**
- Marker must be present and correctly placed
- One-time migration needed for existing repos

### Option 4: Local override file
Use `.gitignore` (template) + `.gitignore.local` (project-specific).

**Pros:** Clean separation into files.
**Cons:** Git doesn't natively support includes. Non-standard. Confusing for humans reviewing `.gitignore`.

## Recommended Solution

**Option 3: Marker-based preservation**

### Template Changes

Add to end of ai_template `.gitignore`:
```
# --- END OF TEMPLATE ---
# Project-specific entries below this line are preserved during sync.
# Add your project's gitignore patterns here.
```

### sync_repo.sh Changes

Replace simple file copy with merge logic:

```bash
sync_gitignore() {
    local src="$AI_TEMPLATE_ROOT/.gitignore"
    local dst="$TARGET_REPO/.gitignore"
    local marker="# --- END OF TEMPLATE ---"

    # Get project-specific entries from target (if marker exists)
    local project_entries=""
    if [[ -f "$dst" ]] && grep -q "$marker" "$dst"; then
        project_entries=$(sed -n "/$marker/,\$p" "$dst" | tail -n +2)
    fi

    # Copy template gitignore
    cp "$src" "$dst"

    # Append project-specific entries if any
    if [[ -n "$project_entries" ]]; then
        echo "" >> "$dst"
        echo "$project_entries" >> "$dst"
    fi
}
```

### Migration

Existing repos will need marker added manually during first sync after this change. The sync script could handle this gracefully by:
1. If target has no marker, append a comment noting project entries may have been lost
2. User can review git diff and re-add project-specific entries below marker

## Implementation Tasks

1. Add marker to ai_template's `.gitignore`
2. Modify `sync_repo.sh` with merge logic
3. Update sync documentation to explain marker behavior
4. Test with dry-run on a repo that has project-specific entries
