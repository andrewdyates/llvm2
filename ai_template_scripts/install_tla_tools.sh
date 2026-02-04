#!/usr/bin/env bash
# Copyright 2026 Dropbox, Inc.
# Author: Andrew Yates <ayates@dropbox.com>
# Licensed under the Apache License, Version 2.0

# install_tla_tools.sh - Install TLA+ tools (TLC) and configure environment
#
# PURPOSE: Install tla2tools.jar and a tlc wrapper; set TLA2TOOLS_JAR globally.
# CALLED BY: check_deps.py --fix, install_dev_tools.sh
# REFERENCED: check_deps.py, README.md
#
# Usage:
#   ./ai_template_scripts/install_tla_tools.sh
#   ./ai_template_scripts/install_tla_tools.sh --version

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

version() {
    local git_hash
    git_hash=$(git -C "$SCRIPT_DIR/.." rev-parse --short HEAD 2>/dev/null || echo "unknown")
    local date
    date=$(date +%Y-%m-%d)
    echo "install_tla_tools.sh ${git_hash} (${date})"
    exit 0
}

case "${1:-}" in
    --version) version ;;
    -h|--help)
        echo "Usage: install_tla_tools.sh"
        echo ""
        echo "Install TLA+ tools (TLC) and configure environment."
        echo ""
        echo "Options:"
        echo "  --version     Show version information"
        echo "  -h, --help    Show this help message"
        exit 0
        ;;
esac

TLA_VERSION="1.8.0"
TLA_JAR_NAME="tla2tools.jar"
TLA_JAR_URL="https://github.com/tlaplus/tlaplus/releases/download/v${TLA_VERSION}/${TLA_JAR_NAME}"
TLA_JAR_PATH="${HOME}/.local/share/${TLA_JAR_NAME}"
TLA_BIN_DIR="${HOME}/.local/bin"
PROFILE_MARKER="# Added by ai_template_scripts/install_tla_tools.sh"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_ok() { echo -e "${GREEN}[OK]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

detect_platform() {
    case "$(uname -s)" in
        Darwin*) echo "macos" ;;
        Linux*)  echo "linux" ;;
        *)       echo "unknown" ;;
    esac
}

has_cmd() {
    command -v "$1" &> /dev/null
}

ensure_java() {
    if java -version &> /dev/null; then
        log_ok "Java already installed"
        return 0
    fi

    local platform
    platform=$(detect_platform)
    log_warn "Java not found - attempting to install OpenJDK"

    case "$platform" in
        macos)
            if has_cmd brew; then
                if brew install openjdk 2>/dev/null; then
                    log_ok "OpenJDK installed via Homebrew"
                else
                    log_error "Failed to install OpenJDK"
                    return 1
                fi
            else
                log_error "Homebrew not installed - cannot install OpenJDK"
                return 1
            fi
            ;;
        linux)
            if has_cmd apt-get; then
                log_info "Installing OpenJDK via apt (may require sudo)..."
                if sudo apt-get install -y openjdk-17-jre 2>/dev/null; then
                    log_ok "OpenJDK installed via apt"
                else
                    log_error "Failed to install OpenJDK via apt"
                    return 1
                fi
            elif has_cmd yum; then
                log_info "Installing OpenJDK via yum (may require sudo)..."
                if sudo yum install -y java-17-openjdk 2>/dev/null; then
                    log_ok "OpenJDK installed via yum"
                else
                    log_error "Failed to install OpenJDK via yum"
                    return 1
                fi
            else
                log_error "Unknown package manager - install Java manually"
                return 1
            fi
            ;;
        *)
            log_error "Unknown platform - install Java manually"
            return 1
            ;;
    esac
}

download_tla_tools() {
    mkdir -p "$(dirname "$TLA_JAR_PATH")"

    if [[ -f "$TLA_JAR_PATH" ]]; then
        log_ok "tla2tools.jar already installed"
        return 0
    fi

    if has_cmd curl; then
        log_info "Downloading TLA+ tools..."
        curl -fL -o "$TLA_JAR_PATH" "$TLA_JAR_URL"
    elif has_cmd wget; then
        log_info "Downloading TLA+ tools..."
        wget -O "$TLA_JAR_PATH" "$TLA_JAR_URL"
    else
        log_error "curl or wget required to download TLA+ tools"
        return 1
    fi

    log_ok "Downloaded tla2tools.jar to $TLA_JAR_PATH"
}

install_tlc_wrapper() {
    mkdir -p "$TLA_BIN_DIR"
    local wrapper="$TLA_BIN_DIR/tlc"

    cat > "$wrapper" << 'EOF'
#!/usr/bin/env bash
set -euo pipefail
jar="${TLA2TOOLS_JAR:-$HOME/.local/share/tla2tools.jar}"
java_cmd=""
if [[ -n "${JAVA_HOME:-}" ]] && [[ -x "${JAVA_HOME}/bin/java" ]]; then
    java_cmd="${JAVA_HOME}/bin/java"
elif command -v java >/dev/null 2>&1; then
    java_cmd="$(command -v java)"
elif [[ -x "/opt/homebrew/opt/openjdk/bin/java" ]]; then
    java_cmd="/opt/homebrew/opt/openjdk/bin/java"
elif [[ -x "/usr/local/opt/openjdk/bin/java" ]]; then
    java_cmd="/usr/local/opt/openjdk/bin/java"
else
    echo "java not found; install OpenJDK or set JAVA_HOME" >&2
    exit 1
fi
exec "$java_cmd" -cp "$jar" tlc2.TLC "$@"
EOF
    chmod +x "$wrapper"

    log_ok "Installed tlc wrapper at $wrapper"
}

pick_profile() {
    local shell_name
    shell_name="$(basename "${SHELL:-}")"
    local fallback="${HOME}/.profile"
    local candidates=()

    case "$shell_name" in
        zsh)
            candidates=("${HOME}/.zshrc" "${HOME}/.zprofile" "${HOME}/.profile")
            ;;
        bash)
            candidates=("${HOME}/.bashrc" "${HOME}/.bash_profile" "${HOME}/.profile")
            ;;
        *)
            candidates=("${HOME}/.profile")
            ;;
    esac

    for candidate in "${candidates[@]}"; do
        if [[ -f "$candidate" ]]; then
            echo "$candidate"
            return 0
        fi
    done

    echo "$fallback"
}

ensure_profile_exports() {
    local profile
    profile="$(pick_profile)"
    local has_jar="false"
    local has_path="false"

    if [[ -f "$profile" ]] && grep -Fq "$PROFILE_MARKER" "$profile"; then
        log_ok "Shell profile already configured ($profile)"
        return 0
    fi

    if [[ -f "$profile" ]]; then
        if grep -Fq "TLA2TOOLS_JAR" "$profile"; then
            has_jar="true"
        fi
        if grep -Fq "$TLA_BIN_DIR" "$profile"; then
            has_path="true"
        fi
        if [[ "$has_jar" == "true" && "$has_path" == "true" ]]; then
            log_ok "Shell profile already configured ($profile)"
            return 0
        fi
    fi

    {
        echo ""
        echo "$PROFILE_MARKER"
        if [[ "$has_jar" != "true" ]]; then
            echo "export TLA2TOOLS_JAR=\"$TLA_JAR_PATH\""
        fi
        if [[ "$has_path" != "true" ]]; then
            echo "export PATH=\"$TLA_BIN_DIR:\$PATH\""
        fi
    } >> "$profile"

    log_ok "Updated shell profile ($profile)"
}

main() {
    echo ""
    echo "========================================"
    echo "TLA+ Tools (TLC) Installation"
    echo "========================================"
    echo ""

    ensure_java
    download_tla_tools
    install_tlc_wrapper
    ensure_profile_exports

    echo ""
    log_ok "TLC installation complete"
    log_info "Restart your shell or source your profile to load TLA2TOOLS_JAR"
}

main "$@"
