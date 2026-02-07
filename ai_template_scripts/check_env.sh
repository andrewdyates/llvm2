#!/usr/bin/env bash
# Copyright 2026 Dropbox, Inc.
# Author: Andrew Yates <ayates@dropbox.com>
# Licensed under the Apache License, Version 2.0

#
# check_env.sh - Verify and configure AI development environment
#
# PURPOSE: Checks Claude CLI, Codex CLI, and LLM model configurations.
# CALLED BY: sync_repo.sh, humans, looper.py startup
# REFERENCED: CLAUDE.md, install_dev_tools.sh
#
# Usage:
#   ./ai_template_scripts/check_env.sh           # Check environment
#   ./ai_template_scripts/check_env.sh --fix     # Auto-fix issues
#   ./ai_template_scripts/check_env.sh --quiet   # Exit code only
#

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Expected versions and models
# Update these when new versions are released
# Must match sync_repo.sh expectations
EXPECTED_CLAUDE_MODEL="us.anthropic.claude-opus-4-6-v1"
EXPECTED_CODEX_MODEL="gpt-5.3-codex"
EXPECTED_CODEX_EFFORT="xhigh"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info() { [[ "${QUIET:-}" != "1" ]] && echo -e "${BLUE}[INFO]${NC} $1" || :; }
log_ok() { [[ "${QUIET:-}" != "1" ]] && echo -e "${GREEN}[OK]${NC} $1" || :; }
log_warn() { [[ "${QUIET:-}" != "1" ]] && echo -e "${YELLOW}[WARN]${NC} $1" || :; }
log_error() { [[ "${QUIET:-}" != "1" ]] && echo -e "${RED}[ERROR]${NC} $1" || :; }

# Track issues
ISSUES=()
FIXED=()

version() {
    echo "check_env.sh $(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
    exit 0
}

usage() {
    cat <<EOF
Usage: $0 [OPTIONS]

Verify and configure AI development environment.

Checks:
  - Claude CLI version and update availability
  - Codex CLI version
  - LLM model settings (~/.claude/settings.json)
  - Codex model settings (~/.codex/config.toml)

Options:
  --fix        Auto-fix issues (update CLIs, configure models)
  --quiet      Suppress output, exit code only (0=ok, 1=issues)
  --version    Show version
  -h, --help   Show this help

Examples:
  $0              # Check environment
  $0 --fix        # Fix all issues
  $0 --quiet      # CI/script mode
EOF
    exit 0
}

# Parse arguments
FIX_MODE=0
QUIET=0
while [[ $# -gt 0 ]]; do
    case "$1" in
    --fix)
        FIX_MODE=1
        shift
        ;;
    --quiet)
        QUIET=1
        shift
        ;;
    --version) version ;;
    -h | --help) usage ;;
    *)
        log_error "Unknown option: $1"
        exit 1
        ;;
    esac
done

# Get installed Claude CLI version
get_claude_version() {
    if command -v claude &>/dev/null; then
        claude --version 2>/dev/null | grep -oE '[0-9]+\.[0-9]+\.[0-9]+' | head -1
    else
        echo "not_installed"
    fi
}

# Get latest Claude CLI version available
get_claude_latest() {
    # Claude's native installer checks for updates
    local output
    output=$(claude update --check 2>&1 || true)
    if echo "$output" | grep -q "up to date"; then
        get_claude_version
    elif echo "$output" | grep -qE "available.*([0-9]+\.[0-9]+\.[0-9]+)"; then
        echo "$output" | grep -oE '[0-9]+\.[0-9]+\.[0-9]+' | tail -1
    else
        echo "unknown"
    fi
}

# Get installed Codex CLI version
get_codex_version() {
    if command -v codex &>/dev/null; then
        codex --version 2>/dev/null | grep -oE '[0-9]+\.[0-9]+\.[0-9]+' | head -1
    else
        echo "not_installed"
    fi
}

# Get latest Codex CLI version from npm
get_codex_latest() {
    npm show @openai/codex version 2>/dev/null || echo "unknown"
}

# Check Claude model in settings
get_claude_model() {
    local settings="$HOME/.claude/settings.json"
    if [[ -f "$settings" ]]; then
        # Extract ANTHROPIC_MODEL from env section
        python3 -c "
import json
with open('$settings') as f:
    data = json.load(f)
    print(data.get('env', {}).get('ANTHROPIC_MODEL', 'not_set'))
" 2>/dev/null || echo "parse_error"
    else
        echo "no_settings_file"
    fi
}

# Check Codex model in config
get_codex_model() {
    local config="$HOME/.codex/config.toml"
    if [[ -f "$config" ]]; then
        python3 <<'PYEOF'
import re
import os
config_path = os.path.expanduser("~/.codex/config.toml")
with open(config_path) as f:
    for line in f:
        stripped = line.strip()
        # Match 'model = ...' but not 'model_reasoning_effort = ...' (#3080)
        if re.match(r'^model\s*=', stripped):
            val = stripped.split("=", 1)[1].strip().strip('"').strip("'")
            print(val)
            exit(0)
print("not_set")
PYEOF
    else
        echo "no_config_file"
    fi
}

# Check Codex model_reasoning_effort in config (#3080)
get_codex_effort() {
    local config="$HOME/.codex/config.toml"
    if [[ -f "$config" ]]; then
        python3 <<'PYEOF'
import re
import os
config_path = os.path.expanduser("~/.codex/config.toml")
with open(config_path) as f:
    for line in f:
        stripped = line.strip()
        if re.match(r'^model_reasoning_effort\s*=', stripped):
            val = stripped.split("=", 1)[1].strip().strip('"').strip("'")
            print(val)
            exit(0)
print("not_set")
PYEOF
    else
        echo "no_config_file"
    fi
}

# Update Claude CLI
update_claude() {
    log_info "Updating Claude CLI..."
    if claude update 2>&1; then
        log_ok "Claude CLI updated"
        FIXED+=("claude_cli")
        return 0
    else
        log_error "Failed to update Claude CLI"
        return 1
    fi
}

# Update Codex CLI
update_codex() {
    log_info "Updating Codex CLI..."
    if npm install -g @openai/codex@latest --force 2>&1; then
        log_ok "Codex CLI updated"
        FIXED+=("codex_cli")
        return 0
    else
        log_error "Failed to update Codex CLI"
        return 1
    fi
}

# Fix Claude model setting
fix_claude_model() {
    local settings="$HOME/.claude/settings.json"

    log_info "Updating Claude model in $settings..."

    if [[ ! -f "$settings" ]]; then
        log_error "Settings file not found: $settings"
        return 1
    fi

    # Use Python to safely update JSON
    # Wrap in `if` so set -e doesn't abort on python3 failure (#3083)
    if python3 -c "
import json
with open('$settings') as f:
    data = json.load(f)
if 'env' not in data:
    data['env'] = {}
data['env']['ANTHROPIC_MODEL'] = '$EXPECTED_CLAUDE_MODEL'
data['env']['CLAUDE_CODE_SUBAGENT_MODEL'] = '$EXPECTED_CLAUDE_MODEL'
with open('$settings', 'w') as f:
    json.dump(data, f, indent=4)
print('Updated ANTHROPIC_MODEL to $EXPECTED_CLAUDE_MODEL')
" 2>&1; then
        log_ok "Claude model updated"
        FIXED+=("claude_model")
        return 0
    else
        log_error "Failed to update Claude model"
        return 1
    fi
}

# Fix Codex model and model_reasoning_effort settings (#3080)
fix_codex_model() {
    local config="$HOME/.codex/config.toml"

    log_info "Updating Codex model and reasoning effort in $config..."

    # Create directory if needed
    mkdir -p "$HOME/.codex"

    if [[ -f "$config" ]]; then
        # Update existing config using Python for cross-platform compatibility
        # Wrap in `if` so set -e doesn't abort on python3 failure (#3083)
        if ! python3 -c "
import re
with open('$config', 'r') as f:
    content = f.read()
# Replace model line — must not match model_reasoning_effort (#3080)
content = re.sub(r\"\"\"^model(?!_)\s*=\s*[\"']?[^\"'\\n]+[\"']?\"\"\", 'model = \"$EXPECTED_CODEX_MODEL\"', content, flags=re.MULTILINE)
# Replace or append model_reasoning_effort (#3080)
if re.search(r'^model_reasoning_effort\s*=', content, re.MULTILINE):
    content = re.sub(r\"\"\"^model_reasoning_effort\s*=\s*[\"']?[^\"'\\n]+[\"']?\"\"\", 'model_reasoning_effort = \"$EXPECTED_CODEX_EFFORT\"', content, flags=re.MULTILINE)
else:
    # Append after model line
    content = re.sub(r\"\"\"^(model(?!_)\s*=\s*[\"']?[^\"'\\n]+[\"']?)\"\"\", r'\1\nmodel_reasoning_effort = \"$EXPECTED_CODEX_EFFORT\"', content, count=1, flags=re.MULTILINE)
with open('$config', 'w') as f:
    f.write(content)
print('Updated model to $EXPECTED_CODEX_MODEL, effort to $EXPECTED_CODEX_EFFORT')
" 2>&1; then
            log_error "Failed to update Codex model"
            return 1
        fi
    else
        # Create new config
        cat >"$config" <<EOF
# Codex configuration
model = "$EXPECTED_CODEX_MODEL"
model_reasoning_effort = "$EXPECTED_CODEX_EFFORT"
EOF
    fi

    log_ok "Codex model updated to $EXPECTED_CODEX_MODEL, effort to $EXPECTED_CODEX_EFFORT"
    FIXED+=("codex_model")
    return 0
}

# Check for model_routing overrides in .looper_config.json
check_looper_model_routing() {
    local config=".looper_config.json"

    # Check current directory first, then SCRIPT_DIR parent (ai_template root)
    if [[ ! -f "$config" ]]; then
        config="$SCRIPT_DIR/../.looper_config.json"
    fi

    if [[ ! -f "$config" ]]; then
        return 0 # No config file, nothing to check
    fi

    # Check if model_routing exists and has model overrides
    local routing_models
    routing_models=$(python3 -c "
import json
import sys

with open('$config') as f:
    data = json.load(f)

routing = data.get('model_routing', {})
if not routing:
    sys.exit(0)

# Collect all model values from model_routing
models = []
for key in ('default', 'audit'):
    section = routing.get(key, {})
    if isinstance(section, dict):
        for model_key in ('claude_model', 'codex_model'):
            if model_key in section:
                models.append(f'{key}.{model_key}={section[model_key]}')

roles = routing.get('roles', {})
if isinstance(roles, dict):
    for role, role_config in roles.items():
        if isinstance(role_config, dict):
            for model_key in ('claude_model', 'codex_model'):
                if model_key in role_config:
                    models.append(f'roles.{role}.{model_key}={role_config[model_key]}')

if models:
    print('\\n'.join(models))
" 2>/dev/null) || return 0

    if [[ -z "$routing_models" ]]; then
        return 0 # No model_routing overrides
    fi

    # Check if any models deviate from expected
    local has_deviation=0
    local deviations=""

    while IFS= read -r line; do
        if [[ "$line" == *"claude_model="* ]]; then
            local model_value="${line#*=}"
            if [[ "$model_value" != "$EXPECTED_CLAUDE_MODEL" ]]; then
                has_deviation=1
                deviations+="  - $line (expected: $EXPECTED_CLAUDE_MODEL)\n"
            fi
        elif [[ "$line" == *"codex_model="* ]]; then
            local model_value="${line#*=}"
            if [[ "$model_value" != "$EXPECTED_CODEX_MODEL" ]]; then
                has_deviation=1
                deviations+="  - $line (expected: $EXPECTED_CODEX_MODEL)\n"
            fi
        fi
    done <<<"$routing_models"

    if [[ "$has_deviation" == "1" ]]; then
        log_warn "model_routing in .looper_config.json has stale model overrides:"
        echo -e "$deviations"
        log_warn "These override global settings. Consider:"
        log_warn "  1. Remove model_routing block to use global settings"
        log_warn "  2. Update model values to match expected versions"
        ISSUES+=("looper_model_routing_stale")
    fi
}

# Update .claude-version in template
update_template_version() {
    local version_file="$SCRIPT_DIR/../.claude-version"
    local current_version
    current_version=$(get_claude_version)

    if [[ -f "$version_file" ]]; then
        local template_version
        template_version=$(cat "$version_file")
        if [[ "$template_version" != "$current_version" ]]; then
            log_info "Updating .claude-version: $template_version -> $current_version"
            echo "$current_version" >"$version_file"
            log_ok "Updated .claude-version"
        fi
    fi
}

# Update .codex-version in template
update_codex_template_version() {
    local version_file="$SCRIPT_DIR/../.codex-version"
    local current_version
    current_version=$(get_codex_version)

    if [[ "$current_version" == "not_installed" ]]; then
        return 0
    fi

    if [[ -f "$version_file" ]]; then
        local template_version
        template_version=$(cat "$version_file")
        if [[ "$template_version" != "$current_version" ]]; then
            log_info "Updating .codex-version: $template_version -> $current_version"
            echo "$current_version" >"$version_file"
            log_ok "Updated .codex-version"
        fi
    fi
}

# Main check
main() {
    [[ "$QUIET" != "1" ]] && echo "" || :
    [[ "$QUIET" != "1" ]] && echo "========================================" || :
    [[ "$QUIET" != "1" ]] && echo "AI Environment Check" || :
    [[ "$QUIET" != "1" ]] && echo "========================================" || :
    [[ "$QUIET" != "1" ]] && echo "" || :

    # Check Claude CLI
    log_info "Checking Claude CLI..."
    local claude_version
    claude_version=$(get_claude_version)

    if [[ "$claude_version" == "not_installed" ]]; then
        log_error "Claude CLI not installed"
        ISSUES+=("claude_cli_missing")
    else
        log_ok "Claude CLI: $claude_version"

        # Check for updates
        local claude_latest
        claude_latest=$(get_claude_latest)
        if [[ "$claude_latest" != "unknown" && "$claude_latest" != "$claude_version" ]]; then
            log_warn "Claude CLI update available: $claude_version -> $claude_latest"
            ISSUES+=("claude_cli_outdated")
            if [[ "$FIX_MODE" == "1" ]]; then
                update_claude
            fi
        fi
    fi

    # Check Codex CLI
    log_info "Checking Codex CLI..."
    local codex_version
    codex_version=$(get_codex_version)

    if [[ "$codex_version" == "not_installed" ]]; then
        log_warn "Codex CLI not installed (optional)"
    else
        log_ok "Codex CLI: $codex_version"

        local codex_latest
        codex_latest=$(get_codex_latest)
        if [[ "$codex_latest" != "unknown" && "$codex_latest" != "$codex_version" ]]; then
            log_warn "Codex CLI update available: $codex_version -> $codex_latest"
            ISSUES+=("codex_cli_outdated")
            if [[ "$FIX_MODE" == "1" ]]; then
                update_codex
            fi
        fi
    fi

    # Check Claude model
    log_info "Checking Claude model configuration..."
    local claude_model
    claude_model=$(get_claude_model)

    if [[ "$claude_model" == "no_settings_file" ]]; then
        log_warn "No ~/.claude/settings.json found"
        ISSUES+=("claude_settings_missing")
    elif [[ "$claude_model" == "not_set" ]]; then
        log_warn "ANTHROPIC_MODEL not set in settings"
        ISSUES+=("claude_model_not_set")
        if [[ "$FIX_MODE" == "1" ]]; then
            fix_claude_model
        fi
    elif [[ "$claude_model" != "$EXPECTED_CLAUDE_MODEL" ]]; then
        log_warn "Claude model: $claude_model"
        log_warn "Expected: $EXPECTED_CLAUDE_MODEL"
        ISSUES+=("claude_model_outdated")
        if [[ "$FIX_MODE" == "1" ]]; then
            fix_claude_model
        fi
    else
        log_ok "Claude model: $claude_model"
    fi

    # Check Codex model
    log_info "Checking Codex model configuration..."
    local codex_model
    codex_model=$(get_codex_model)
    local codex_needs_fix=0

    if [[ "$codex_model" == "no_config_file" ]]; then
        log_warn "No ~/.codex/config.toml found (optional)"
    elif [[ "$codex_model" == "not_set" ]]; then
        log_warn "Codex model not set in config"
        ISSUES+=("codex_model_not_set")
        codex_needs_fix=1
    elif [[ "$codex_model" != "$EXPECTED_CODEX_MODEL" ]]; then
        log_warn "Codex model: $codex_model (expected: $EXPECTED_CODEX_MODEL)"
        ISSUES+=("codex_model_outdated")
        codex_needs_fix=1
    else
        log_ok "Codex model: $codex_model"
    fi

    # Check Codex model_reasoning_effort (#3080)
    if [[ "$codex_model" != "no_config_file" ]]; then
        local codex_effort
        codex_effort=$(get_codex_effort)

        if [[ "$codex_effort" == "not_set" ]]; then
            log_warn "Codex model_reasoning_effort not set (expected: $EXPECTED_CODEX_EFFORT)"
            ISSUES+=("codex_effort_not_set")
            codex_needs_fix=1
        elif [[ "$codex_effort" != "$EXPECTED_CODEX_EFFORT" ]]; then
            log_warn "Codex effort: $codex_effort (expected: $EXPECTED_CODEX_EFFORT)"
            ISSUES+=("codex_effort_outdated")
            codex_needs_fix=1
        else
            log_ok "Codex effort: $codex_effort"
        fi
    fi

    if [[ "$codex_needs_fix" == "1" && "$FIX_MODE" == "1" ]]; then
        fix_codex_model
    fi

    # Check for stale model_routing overrides in .looper_config.json
    log_info "Checking .looper_config.json model_routing..."
    check_looper_model_routing

    # Update template version files if we're in ai_template
    if [[ "$FIX_MODE" == "1" ]]; then
        if [[ -f "$SCRIPT_DIR/../.claude-version" ]]; then
            update_template_version
        fi
        if [[ -f "$SCRIPT_DIR/../.codex-version" ]]; then
            update_codex_template_version
        fi
    fi

    # Summary
    [[ "$QUIET" != "1" ]] && echo "" || :
    [[ "$QUIET" != "1" ]] && echo "========================================" || :

    if [[ ${#ISSUES[@]} -eq 0 ]]; then
        log_ok "Environment OK - all checks passed"
        exit 0
    else
        if [[ ${#FIXED[@]} -gt 0 ]]; then
            log_ok "Fixed: ${FIXED[*]}"
        fi
        if [[ ${#ISSUES[@]} -gt ${#FIXED[@]} ]]; then
            local remaining=$((${#ISSUES[@]} - ${#FIXED[@]}))
            log_warn "$remaining issue(s) remain. Run with --fix to auto-fix."
        fi
        exit 1
    fi
}

main "$@"
