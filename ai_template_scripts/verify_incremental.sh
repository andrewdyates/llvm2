#!/usr/bin/env bash
# Copyright 2026 Dropbox, Inc.
# Author: Andrew Yates
# Licensed under the Apache License, Version 2.0

# verify_incremental.sh - Run incremental verification across tools (Kani, TLA+, etc.)
#
# Detects available verification tools and runs only affected proofs/specs
# based on which files changed since a given commit.
#
# Usage:
#   ./ai_template_scripts/verify_incremental.sh              # Check since HEAD~1
#   ./ai_template_scripts/verify_incremental.sh --since abc123  # Check since specific commit
#   ./ai_template_scripts/verify_incremental.sh --tool kani    # Only run Kani
#   ./ai_template_scripts/verify_incremental.sh --force        # Ignore cache, verify all

set -euo pipefail

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Defaults
SINCE_COMMIT="HEAD~1"
TOOL_FILTER="all"
FORCE=false
TIER=1
UPDATE_CACHE=false
METRICS_DIR="${METRICS_DIR:-metrics/verification}"

# Detected tools
HAS_KANI=false
HAS_TLA=false

# Results tracking
TOTAL_AFFECTED=0
TOTAL_CACHED=0
TOTAL_PASSED=0
TOTAL_FAILED=0
START_TIME=$(date +%s)

# Logging (all to stderr to not pollute function return values)
log_info() { echo -e "${BLUE}[INFO]${NC} $1" >&2; }
log_ok() { echo -e "${GREEN}[OK]${NC} $1" >&2; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1" >&2; }
log_error() { echo -e "${RED}[ERROR]${NC} $1" >&2; }

# Version
version() {
    local git_hash
    git_hash=$(git -C "$(dirname "$0")/.." rev-parse --short HEAD 2>/dev/null || echo "unknown")
    local date
    date=$(date +%Y-%m-%d)
    echo "verify_incremental.sh ${git_hash} (${date})"
    exit 0
}

# Usage
usage() {
    local status="${1:-0}"
    cat <<EOF
Usage: $(basename "$0") [OPTIONS]

Run incremental verification across tools (Kani, TLA+, etc.).
Only verifies proofs/specs affected by changes since the specified commit.

Options:
  --since COMMIT    Only check changes since COMMIT (default: HEAD~1)
  --force           Ignore cache, re-verify everything
  --tool TOOL       Only run specific tool (kani, tla, all) (default: all)
  --tier TIER       Run specific tier (default: 1)
                      0 = smoke (quick sanity check)
                      1 = changed (only affected by changes)
                      2 = module (all in changed modules)
                      3 = full (complete verification)
  --update-cache    Update cache even on failure (for recording timeouts)
  -h, --help        Show this help message
  --version         Show version information

Examples:
  $(basename "$0")                     # Verify changes since HEAD~1
  $(basename "$0") --since HEAD~5      # Verify last 5 commits
  $(basename "$0") --tool kani         # Only run Kani proofs
  $(basename "$0") --force --tier 3    # Full verification, ignore cache

Output:
  Prints verification results to stdout and writes metrics JSON to:
  ${METRICS_DIR}/verify-{timestamp}.json
EOF
    exit "$status"
}

require_arg() {
    local opt="$1"
    local val="${2:-}"
    if [[ -z "$val" ]]; then
        log_error "Missing argument for $opt"
        usage 1
    fi
}

# Parse arguments
parse_args() {
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --since)
                require_arg "$1" "${2:-}"
                SINCE_COMMIT="$2"
                shift 2
                ;;
            --force)
                FORCE=true
                shift
                ;;
            --tool)
                require_arg "$1" "${2:-}"
                TOOL_FILTER="$2"
                if [[ "$TOOL_FILTER" != "all" && "$TOOL_FILTER" != "kani" && "$TOOL_FILTER" != "tla" ]]; then
                    log_error "Invalid --tool value: $TOOL_FILTER (must be: all, kani, or tla)"
                    exit 1
                fi
                shift 2
                ;;
            --tier)
                require_arg "$1" "${2:-}"
                TIER="$2"
                if ! [[ "$TIER" =~ ^[0-3]$ ]]; then
                    log_error "Invalid --tier value: $TIER (must be 0-3)"
                    exit 1
                fi
                shift 2
                ;;
            --update-cache)
                UPDATE_CACHE=true
                shift
                ;;
            --version)
                version
                ;;
            -h|--help)
                usage 0
                ;;
            *)
                log_error "Unknown option: $1"
                usage 1
                ;;
        esac
    done

    # Validate --since commit reference exists
    if ! git rev-parse --verify "$SINCE_COMMIT" &>/dev/null; then
        log_error "Invalid commit reference: $SINCE_COMMIT"
        exit 1
    fi
}

# Detect available verification tools
detect_tools() {
    log_info "Detecting verification tools..."

    # Check for Kani
    if [[ -f "Cargo.toml" ]]; then
        if grep -q 'kani' Cargo.toml 2>/dev/null || \
           [[ -d "proofs" ]] || \
           [[ -n "$(find . -name "*.rs" -exec grep -l '#\[kani::' {} \; -quit 2>/dev/null)" ]]; then
            if command -v cargo-kani &>/dev/null || command -v kani &>/dev/null; then
                HAS_KANI=true
                log_ok "Kani detected"
            else
                log_warn "Kani proofs found but kani not installed"
            fi
        fi
    fi

    # Check for TLA+
    if [[ -d "tla" ]] && [[ -n "$(find tla -name "*.tla" -type f -print -quit 2>/dev/null)" ]]; then
        if command -v tla2 &>/dev/null || command -v tlc &>/dev/null; then
            HAS_TLA=true
            log_ok "TLA+ detected"
        else
            log_warn "TLA+ specs found but tla2/tlc not installed"
        fi
    fi

    # Report if no tools found
    if [[ "$HAS_KANI" == false ]] && [[ "$HAS_TLA" == false ]]; then
        log_info "No verification tools detected in this repository"
        return 1
    fi

    return 0
}

# Get list of changed files since the specified commit
get_changed_files() {
    git diff --name-only "$SINCE_COMMIT" HEAD 2>/dev/null || git diff --name-only "$SINCE_COMMIT" 2>/dev/null || true
}

# Run Kani verification
run_kani() {
    # Handle empty arrays with set -u by using explicit check
    local changed_files=()
    if [[ $# -gt 0 ]]; then
        changed_files=("$@")
    fi
    local affected=0
    local cached=0
    local passed=0
    local failed=0
    local kani_start
    kani_start=$(date +%s)

    log_info "Running Kani verification..."

    # Check if there's a kani_incremental.py script
    local kani_script=""
    if [[ -x "scripts/kani_incremental.py" ]]; then
        kani_script="scripts/kani_incremental.py"
    elif [[ -x "ai_template_scripts/kani_incremental.py" ]]; then
        kani_script="ai_template_scripts/kani_incremental.py"
    fi

    if [[ -n "$kani_script" ]]; then
        # Use the project-specific incremental runner
        local kani_args=("--since" "$SINCE_COMMIT")
        if [[ "$FORCE" == true ]]; then
            kani_args+=("--force")
        fi
        if [[ "$UPDATE_CACHE" == true ]]; then
            kani_args+=("--update-cache")
        fi

        local kani_output
        if kani_output=$(python3 "$kani_script" "${kani_args[@]}" 2>&1); then
            # Parse output for metrics (portable - no grep -P which is GNU-only)
            affected=$(echo "$kani_output" | sed -n 's/.*affected:[[:space:]]*\([0-9][0-9]*\).*/\1/p' | head -1)
            affected=${affected:-0}
            cached=$(echo "$kani_output" | sed -n 's/.*cached:[[:space:]]*\([0-9][0-9]*\).*/\1/p' | head -1)
            cached=${cached:-0}
            passed=$(echo "$kani_output" | sed -n 's/.*passed:[[:space:]]*\([0-9][0-9]*\).*/\1/p' | head -1)
            passed=${passed:-$affected}
            log_ok "Kani: $passed passed, $cached cached"
        else
            log_error "Kani verification failed"
            failed=$((affected > 0 ? affected : 1))
            echo "$kani_output" >&2
        fi
    else
        # Fallback: run all Kani proofs if changed files include .rs
        local rust_changed=false
        for f in ${changed_files[@]+"${changed_files[@]}"}; do
            if [[ "$f" == *.rs ]]; then
                rust_changed=true
                break
            fi
        done

        if [[ "$rust_changed" == true ]] || [[ "$FORCE" == true ]]; then
            log_info "Running cargo kani..."
            if cargo kani 2>&1; then
                affected=1
                passed=1
                log_ok "Kani verification passed"
            else
                affected=1
                failed=1
                log_error "Kani verification failed"
            fi
        else
            log_info "No Rust files changed, skipping Kani"
        fi
    fi

    local kani_end
    kani_end=$(date +%s)
    local duration=$((kani_end - kani_start))

    # Update global totals
    TOTAL_AFFECTED=$((TOTAL_AFFECTED + affected))
    TOTAL_CACHED=$((TOTAL_CACHED + cached))
    TOTAL_PASSED=$((TOTAL_PASSED + passed))
    TOTAL_FAILED=$((TOTAL_FAILED + failed))

    # Return JSON fragment for metrics
    echo "{\"affected\": $affected, \"cached\": $cached, \"passed\": $passed, \"failed\": $failed, \"duration_secs\": $duration}"
}

# Run TLA+ verification
run_tla() {
    # Handle empty arrays with set -u by using explicit check
    local changed_files=()
    if [[ $# -gt 0 ]]; then
        changed_files=("$@")
    fi
    local affected=0
    local cached=0
    local passed=0
    local failed=0
    local tla_start
    tla_start=$(date +%s)

    log_info "Running TLA+ verification..."

    # Find affected TLA+ specs
    local affected_specs=()
    local tla_force_all=false
    for f in ${changed_files[@]+"${changed_files[@]}"}; do
        if [[ "$f" == tla/* ]]; then
            if [[ "$f" == *.tla ]]; then
                affected_specs+=("$f")
            elif [[ "$f" == *.cfg ]]; then
                tla_force_all=true
            fi
        fi
    done

    # If no TLA+ files changed and not forced, skip
    if [[ ${#affected_specs[@]} -eq 0 ]] && [[ "$tla_force_all" != true ]] && [[ "$FORCE" != true ]]; then
        log_info "No TLA+ specs changed, skipping"
        echo "{\"affected\": 0, \"cached\": 0, \"passed\": 0, \"failed\": 0, \"duration_secs\": 0}"
        return 0
    fi

    # Check for tla2 or tlc
    if command -v tla2 &>/dev/null; then
        local tla_args=("check")
        if [[ "$FORCE" != true ]]; then
            tla_args+=("--incremental")
        fi

        # Run affected specs or all if forced/config changed (bash 3 compatible)
        local specs_to_check=()
        if [[ "$FORCE" == true ]] || [[ "$tla_force_all" == true ]]; then
            while IFS= read -r spec; do
                [[ -n "$spec" ]] && specs_to_check+=("$spec")
            done < <(find tla -name "*.tla" -type f 2>/dev/null)
        else
            specs_to_check=("${affected_specs[@]}")
        fi

        affected=${#specs_to_check[@]}

        # Use empty array safe iteration pattern
        for spec in ${specs_to_check[@]+"${specs_to_check[@]}"}; do
            log_info "Checking $spec..."
            if tla2 "${tla_args[@]}" "$spec" 2>&1; then
                passed=$((passed + 1))
            else
                failed=$((failed + 1))
                log_error "TLA+ verification failed for $spec"
            fi
        done

        if [[ $failed -eq 0 ]]; then
            log_ok "TLA+: $passed passed"
        else
            log_error "TLA+: $failed failed, $passed passed"
        fi
    elif command -v tlc &>/dev/null; then
        # Use standard TLC
        log_warn "Using TLC (consider installing tla2 for incremental checking)"
        # Similar logic but with tlc
        affected=1
        if tlc -config tla/MC.cfg tla/MC.tla 2>&1; then
            passed=1
            log_ok "TLA+ verification passed"
        else
            failed=1
            log_error "TLA+ verification failed"
        fi
    fi

    local tla_end
    tla_end=$(date +%s)
    local duration=$((tla_end - tla_start))

    # Update global totals
    TOTAL_AFFECTED=$((TOTAL_AFFECTED + affected))
    TOTAL_CACHED=$((TOTAL_CACHED + cached))
    TOTAL_PASSED=$((TOTAL_PASSED + passed))
    TOTAL_FAILED=$((TOTAL_FAILED + failed))

    echo "{\"affected\": $affected, \"cached\": $cached, \"passed\": $passed, \"failed\": $failed, \"duration_secs\": $duration}"
}

# Write metrics JSON
write_metrics() {
    local kani_json="$1"
    local tla_json="$2"

    local end_time
    end_time=$(date +%s)
    local total_duration=$((end_time - START_TIME))
    local timestamp
    timestamp=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
    local commit
    commit=$(git rev-parse --short HEAD 2>/dev/null || echo "unknown")

    # Create metrics directory if needed
    mkdir -p "$METRICS_DIR"

    local metrics_file
    metrics_file="${METRICS_DIR}/verify-$(date +%Y%m%d-%H%M%S).json"

    cat > "$metrics_file" <<EOF
{
  "timestamp": "$timestamp",
  "commit": "$commit",
  "since": "$SINCE_COMMIT",
  "tier": $TIER,
  "force": $FORCE,
  "total_duration_secs": $total_duration,
  "summary": {
    "affected": $TOTAL_AFFECTED,
    "cached": $TOTAL_CACHED,
    "passed": $TOTAL_PASSED,
    "failed": $TOTAL_FAILED
  },
  "tools": {
    "kani": $kani_json,
    "tla": $tla_json
  }
}
EOF

    log_info "Metrics written to $metrics_file"
}

# Print summary
print_summary() {
    echo ""
    echo "=== Verification Summary ==="
    echo "Affected: $TOTAL_AFFECTED"
    echo "Cached:   $TOTAL_CACHED"
    echo "Passed:   $TOTAL_PASSED"
    echo "Failed:   $TOTAL_FAILED"

    if [[ $TOTAL_FAILED -gt 0 ]]; then
        log_error "Verification FAILED"
        return 1
    elif [[ $TOTAL_AFFECTED -eq 0 ]]; then
        log_info "No verification needed"
        return 0
    else
        log_ok "Verification PASSED"
        return 0
    fi
}

# Main
main() {
    parse_args "$@"

    echo "=== Incremental Verification ==="
    log_info "Checking changes since $SINCE_COMMIT"

    # Get changed files (bash 3 compatible)
    local changed_files=()
    while IFS= read -r line; do
        [[ -n "$line" ]] && changed_files+=("$line")
    done < <(get_changed_files)

    log_info "Changed files: ${#changed_files[@]}"
    # Use ${arr[@]+"${arr[@]}"} pattern for set -u compatibility with empty arrays
    if [[ ${#changed_files[@]} -gt 0 ]]; then
        for f in "${changed_files[@]:0:5}"; do
            echo "  - $f"
        done
        if [[ ${#changed_files[@]} -gt 5 ]]; then
            echo "  ... and $((${#changed_files[@]} - 5)) more"
        fi
    fi
    echo ""

    # Detect tools
    if ! detect_tools; then
        log_info "Nothing to verify"
        exit 0
    fi
    echo ""

    # Run verification tools
    local kani_json='{"affected": 0, "cached": 0, "passed": 0, "failed": 0, "duration_secs": 0}'
    local tla_json='{"affected": 0, "cached": 0, "passed": 0, "failed": 0, "duration_secs": 0}'

    if [[ "$HAS_KANI" == true ]] && [[ "$TOOL_FILTER" == "all" || "$TOOL_FILTER" == "kani" ]]; then
        # Use ${arr[@]+"${arr[@]}"} for empty array compatibility with set -u
        kani_json=$(run_kani ${changed_files[@]+"${changed_files[@]}"})
        echo ""
    fi

    if [[ "$HAS_TLA" == true ]] && [[ "$TOOL_FILTER" == "all" || "$TOOL_FILTER" == "tla" ]]; then
        # Use ${arr[@]+"${arr[@]}"} for empty array compatibility with set -u
        tla_json=$(run_tla ${changed_files[@]+"${changed_files[@]}"})
        echo ""
    fi

    # Write metrics
    write_metrics "$kani_json" "$tla_json"

    # Print summary and exit
    print_summary
}

main "$@"
