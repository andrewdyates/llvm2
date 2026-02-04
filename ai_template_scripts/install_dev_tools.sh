#!/usr/bin/env bash
# Copyright 2026 Dropbox, Inc.
# Author: Andrew Yates
# Licensed under the Apache License, Version 2.0

#
# install_dev_tools.sh - Install development tools for ai_template
#
# PURPOSE: Installs tools needed by code_stats.py and other scripts.
# CALLED BY: Human (one-time setup after cloning)
# REFERENCED: code_stats.py, README.md
#
# Usage:
#   ./ai_template_scripts/install_dev_tools.sh           # Install everything
#   ./ai_template_scripts/install_dev_tools.sh --check   # Check what's installed
#
# Copyright 2026 Dropbox, Inc.
# Licensed under the Apache License, Version 2.0

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Version function
version() {
    echo "install_dev_tools.sh $(git rev-parse --short HEAD 2>/dev/null || echo unknown) ($(date +%Y-%m-%d))"
    exit 0
}

# Usage function
usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Install development tools for ai_template projects."
    echo ""
    echo "Installs tools needed by code_stats.py and other scripts:"
    echo "  - Python tools (mypy, ruff, pre-commit)"
    echo "  - TLA+ tools (TLA+ toolbox, Apalache)"
    echo "  - Go tools (staticcheck)"
    echo "  - C tools (pmccabe for cyclomatic complexity)"
    echo "  - Shell tools (shellcheck)"
    echo ""
    echo "Options:"
    echo "  --check      Check what's installed (no changes)"
    echo "  -h, --help   Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0           # Install everything"
    echo "  $0 --check   # Check what's installed"
    exit 0
}

# Parse args before anything else
case "${1:-}" in
    --version) version ;;
    -h|--help) usage ;;
    --check|"") ;; # Valid options or no option
    -*) echo "ERROR: Unknown option: $1"; exit 1 ;;
    *) echo "ERROR: Unexpected argument: $1"; exit 1 ;;
esac

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Track what we installed
INSTALLED=()
FAILED=()
SKIPPED=()

log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_ok() { echo -e "${GREEN}[OK]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# Shared TLA+ tool discovery helpers.
# shellcheck source=/dev/null
source "${SCRIPT_DIR}/find_tla_tools.sh"

# Detect platform
detect_platform() {
    case "$(uname -s)" in
        Darwin*) echo "macos" ;;
        Linux*)  echo "linux" ;;
        *)       echo "unknown" ;;
    esac
}

# Check if a command exists
has_cmd() {
    command -v "$1" &> /dev/null
}

# Check whether TLA+ tools are available (java + jar or tlc wrapper)
tla_tools_ready() {
    if ! find_java >/dev/null 2>&1; then
        return 1
    fi

    if find_tla_tools >/dev/null 2>&1; then
        return 0
    fi

    if has_cmd tlc; then
        return 0
    fi

    return 1
}

# Check prerequisites
check_prerequisites() {
    log_info "Checking prerequisites..."
    local missing=()

    if ! has_cmd python3; then
        missing+=("python3")
    fi

    if ! has_cmd pip3 && ! has_cmd pip; then
        missing+=("pip")
    fi

    if [[ ${#missing[@]} -gt 0 ]]; then
        log_error "Missing required tools: ${missing[*]}"
        log_error "Please install Python 3 with pip first."
        exit 1
    fi

    log_ok "Prerequisites satisfied (python3, pip)"
}

# Get pip command (pip3 or pip)
get_pip() {
    if has_cmd pip3; then
        echo "pip3"
    else
        echo "pip"
    fi
}

# Install Python packages
install_python_tools() {
    log_info "Installing Python tools..."
    local pip_cmd
    pip_cmd=$(get_pip)

    # radon - Python complexity analysis (best-in-class)
    if has_cmd radon; then
        log_ok "radon already installed"
        SKIPPED+=("radon")
    else
        log_info "Installing radon (Python complexity analyzer)..."
        if $pip_cmd install --quiet radon; then
            log_ok "radon installed"
            INSTALLED+=("radon")
        else
            log_error "Failed to install radon"
            FAILED+=("radon")
        fi
    fi

    # lizard - Multi-language complexity (Rust, TS, Swift, ObjC, C/C++ fallback)
    if has_cmd lizard; then
        log_ok "lizard already installed"
        SKIPPED+=("lizard")
    else
        log_info "Installing lizard (multi-language complexity analyzer)..."
        if $pip_cmd install --quiet lizard; then
            log_ok "lizard installed"
            INSTALLED+=("lizard")
        else
            log_error "Failed to install lizard"
            FAILED+=("lizard")
        fi
    fi

    # ruff - Python linter (used by pre-commit hook)
    # IMPORTANT: .pre-commit-config.yaml is the SINGLE SOURCE OF TRUTH for ruff version.
    # When updating: change .pre-commit-config.yaml FIRST, then update this variable.
    local RUFF_VERSION="0.14.14"  # Must match: grep -A1 'ruff-pre-commit' .pre-commit-config.yaml
    if has_cmd ruff; then
        local installed_version
        installed_version=$(ruff --version 2>/dev/null | grep -oE '[0-9]+\.[0-9]+\.[0-9]+' | head -1)
        if [[ "$installed_version" == "$RUFF_VERSION" ]]; then
            log_ok "ruff $RUFF_VERSION already installed"
            SKIPPED+=("ruff")
        else
            log_warn "ruff version mismatch: installed=$installed_version, required=$RUFF_VERSION"
            log_info "Upgrading ruff to $RUFF_VERSION..."
            if $pip_cmd install --quiet "ruff==$RUFF_VERSION"; then
                log_ok "ruff $RUFF_VERSION installed"
                INSTALLED+=("ruff")
            else
                log_error "Failed to upgrade ruff"
                FAILED+=("ruff")
            fi
        fi
    else
        log_info "Installing ruff $RUFF_VERSION (Python linter)..."
        if $pip_cmd install --quiet "ruff==$RUFF_VERSION"; then
            log_ok "ruff $RUFF_VERSION installed"
            INSTALLED+=("ruff")
        else
            log_error "Failed to install ruff"
            FAILED+=("ruff")
        fi
    fi
}

# Install TLA+ tools (TLC model checker)
install_tla_tools() {
    log_info "Installing TLA+ tools (TLC)..."

    if [[ -x "./ai_template_scripts/install_tla_tools.sh" ]]; then
        if ./ai_template_scripts/install_tla_tools.sh; then
            INSTALLED+=("tlc")
        else
            log_error "Failed to install TLA+ tools"
            FAILED+=("tlc")
        fi
    else
        log_warn "install_tla_tools.sh not found - skipping TLA+ tools"
        SKIPPED+=("tlc")
    fi
}

# Install shell tools
install_shell_tools() {
    log_info "Installing shell tools..."
    local platform
    platform=$(detect_platform)

    # ShellCheck - Shell script linter (used by pre-commit hook)
    if has_cmd shellcheck; then
        log_ok "shellcheck already installed"
        SKIPPED+=("shellcheck")
    else
        log_info "Installing shellcheck (shell script linter)..."
        case "$platform" in
            macos)
                if has_cmd brew; then
                    if brew install shellcheck 2>/dev/null; then
                        log_ok "shellcheck installed via Homebrew"
                        INSTALLED+=("shellcheck")
                    else
                        log_error "Failed to install shellcheck"
                        FAILED+=("shellcheck")
                    fi
                else
                    log_warn "Homebrew not installed - cannot install shellcheck"
                    FAILED+=("shellcheck")
                fi
                ;;
            linux)
                if has_cmd apt-get; then
                    log_info "Installing shellcheck via apt (may require sudo)..."
                    if sudo apt-get install -y shellcheck 2>/dev/null; then
                        log_ok "shellcheck installed via apt"
                        INSTALLED+=("shellcheck")
                    else
                        log_error "Failed to install shellcheck"
                        FAILED+=("shellcheck")
                    fi
                elif has_cmd yum; then
                    if sudo yum install -y ShellCheck 2>/dev/null; then
                        log_ok "shellcheck installed via yum"
                        INSTALLED+=("shellcheck")
                    else
                        log_error "Failed to install shellcheck"
                        FAILED+=("shellcheck")
                    fi
                else
                    log_warn "Unknown package manager - cannot install shellcheck"
                    FAILED+=("shellcheck")
                fi
                ;;
            *)
                log_warn "Unknown platform - cannot install shellcheck"
                FAILED+=("shellcheck")
                ;;
        esac
    fi
}

# Install Go tools
install_go_tools() {
    log_info "Installing Go tools..."

    if ! has_cmd go; then
        log_warn "Go not installed - skipping Go tools (gocyclo)"
        log_warn "  Install Go from https://go.dev/dl/ if you need Go analysis"
        SKIPPED+=("gocyclo")
        return
    fi

    # gocyclo - Go cyclomatic complexity (best-in-class)
    if has_cmd gocyclo; then
        log_ok "gocyclo already installed"
        SKIPPED+=("gocyclo")
    else
        log_info "Installing gocyclo (Go complexity analyzer)..."
        if go install github.com/fzipp/gocyclo/cmd/gocyclo@latest 2>/dev/null; then
            # Add GOPATH/bin to PATH hint
            if [[ -d "$HOME/go/bin" ]] && [[ ":$PATH:" != *":$HOME/go/bin:"* ]]; then
                log_warn "Add \$HOME/go/bin to your PATH to use gocyclo"
            fi
            log_ok "gocyclo installed"
            INSTALLED+=("gocyclo")
        else
            log_error "Failed to install gocyclo"
            FAILED+=("gocyclo")
        fi
    fi
}

# Install C/C++ tools
install_c_tools() {
    log_info "Installing C/C++ tools..."
    local platform
    platform=$(detect_platform)

    # pmccabe - C/C++ McCabe complexity (best-in-class)
    if has_cmd pmccabe; then
        log_ok "pmccabe already installed"
        SKIPPED+=("pmccabe")
    else
        log_info "Installing pmccabe (C/C++ complexity analyzer)..."
        case "$platform" in
            macos)
                if has_cmd brew; then
                    if brew install pmccabe 2>/dev/null; then
                        log_ok "pmccabe installed via Homebrew"
                        INSTALLED+=("pmccabe")
                    else
                        log_warn "Failed to install pmccabe - will use lizard as fallback"
                        SKIPPED+=("pmccabe")
                    fi
                else
                    log_warn "Homebrew not installed - skipping pmccabe"
                    log_warn "  lizard will be used as fallback for C/C++"
                    SKIPPED+=("pmccabe")
                fi
                ;;
            linux)
                if has_cmd apt-get; then
                    log_info "Installing pmccabe via apt (may require sudo)..."
                    if sudo apt-get install -y pmccabe 2>/dev/null; then
                        log_ok "pmccabe installed via apt"
                        INSTALLED+=("pmccabe")
                    else
                        log_warn "Failed to install pmccabe - will use lizard as fallback"
                        SKIPPED+=("pmccabe")
                    fi
                elif has_cmd yum; then
                    log_warn "pmccabe not available via yum - will use lizard as fallback"
                    SKIPPED+=("pmccabe")
                else
                    log_warn "Unknown package manager - skipping pmccabe"
                    SKIPPED+=("pmccabe")
                fi
                ;;
            *)
                log_warn "Unknown platform - skipping pmccabe"
                SKIPPED+=("pmccabe")
                ;;
        esac
    fi
}

# Verify installations
verify_installations() {
    log_info "Verifying installations..."
    echo ""

    local tool_descriptions=(
        "radon:Python complexity (best-in-class)"
        "lizard:Multi-language (Rust, TS, Swift, ObjC)"
        "gocyclo:Go complexity (best-in-class)"
        "pmccabe:C/C++ complexity (best-in-class)"
        "ruff:Python linter (pre-commit)"
        "shellcheck:Shell linter (pre-commit)"
    )

    echo "Tool Status:"
    echo "------------"
    for desc in "${tool_descriptions[@]}"; do
        local tool="${desc%%:*}"
        local description="${desc#*:}"
        if has_cmd "$tool"; then
            echo -e "  ${GREEN}✓${NC} $tool - $description"
        else
            echo -e "  ${RED}✗${NC} $tool - $description"
        fi
    done
    if tla_tools_ready; then
        echo -e "  ${GREEN}✓${NC} tlc - TLA+ model checker (TLA2TOOLS_JAR)"
    else
        echo -e "  ${RED}✗${NC} tlc - TLA+ model checker (missing java or TLA2TOOLS_JAR)"
    fi
    echo ""
}

# Print summary
print_summary() {
    echo ""
    echo "========================================"
    echo "Installation Summary"
    echo "========================================"

    if [[ ${#INSTALLED[@]} -gt 0 ]]; then
        echo -e "${GREEN}Installed:${NC} ${INSTALLED[*]}"
    fi

    if [[ ${#SKIPPED[@]} -gt 0 ]]; then
        echo -e "${YELLOW}Skipped (already installed or unavailable):${NC} ${SKIPPED[*]}"
    fi

    if [[ ${#FAILED[@]} -gt 0 ]]; then
        echo -e "${RED}Failed:${NC} ${FAILED[*]}"
    fi

    echo ""
    echo "Run './ai_template_scripts/code_stats.py .' to analyze your codebase."
    echo ""
}

# Check mode - just report what's installed
check_mode() {
    log_info "Checking installed tools (no installation)..."
    echo ""
    verify_installations

    # Also run code_stats.py to show what would be analyzed
    if [[ -f "./ai_template_scripts/code_stats.py" ]]; then
        log_info "Running code_stats.py to check tool availability..."
        python3 ./ai_template_scripts/code_stats.py . --quiet 2>&1 | grep -E "(Missing tools|lizard|radon|gocyclo|pmccabe)" || true
    fi
}

# Main
main() {
    echo ""
    echo "========================================"
    echo "ai_template Development Environment Setup"
    echo "========================================"
    echo ""

    # Check for --check flag
    if [[ "${1:-}" == "--check" ]]; then
        check_mode
        exit 0
    fi

    check_prerequisites
    echo ""

    install_python_tools
    echo ""

    install_tla_tools
    echo ""

    install_go_tools
    echo ""

    install_c_tools
    echo ""

    install_shell_tools
    echo ""

    verify_installations
    print_summary
}

main "$@"
