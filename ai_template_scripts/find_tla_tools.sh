#!/usr/bin/env bash
# Copyright 2026 Dropbox, Inc.
# Author: Andrew Yates <ayates@dropbox.com>
# Licensed under the Apache License, Version 2.0

# find_tla_tools.sh - Locate Java, tla2tools.jar, and timeout binaries.
#
# Usage (source):
#   source "$PROJECT_ROOT/ai_template_scripts/find_tla_tools.sh"
#   java_bin="$(find_java)"
#   tla_jar="$(find_tla_tools)"
#   timeout_bin="$(find_timeout)"
#
# Usage (exec):
#   ./ai_template_scripts/find_tla_tools.sh --java
#   ./ai_template_scripts/find_tla_tools.sh --jar
#   ./ai_template_scripts/find_tla_tools.sh --timeout

tla_tools_err() {
    echo "$1" >&2
}

find_java() {
    if [[ -n "${JAVA_CMD:-}" ]] && [[ -x "${JAVA_CMD}" ]]; then
        echo "${JAVA_CMD}"
        return 0
    fi

    if [[ -n "${JAVA_HOME:-}" ]] && [[ -x "${JAVA_HOME}/bin/java" ]]; then
        echo "${JAVA_HOME}/bin/java"
        return 0
    fi

    if command -v java >/dev/null 2>&1; then
        command -v java
        return 0
    fi

    local candidates=(
        "/opt/homebrew/opt/openjdk/bin/java"
        "/usr/local/opt/openjdk/bin/java"
    )

    local candidate
    for candidate in "${candidates[@]}"; do
        if [[ -x "${candidate}" ]]; then
            echo "${candidate}"
            return 0
        fi
    done

    if command -v /usr/libexec/java_home >/dev/null 2>&1; then
        local java_home
        java_home="$(/usr/libexec/java_home 2>/dev/null || true)"
        if [[ -n "${java_home}" ]] && [[ -x "${java_home}/bin/java" ]]; then
            echo "${java_home}/bin/java"
            return 0
        fi
    fi

    tla_tools_err "java not found; set JAVA_HOME or install OpenJDK"
    return 1
}

find_tla_tools() {
    if [[ -n "${TLA2TOOLS_JAR:-}" ]]; then
        local env_candidate="${TLA2TOOLS_JAR/#\~/$HOME}"
        if [[ -f "${env_candidate}" ]]; then
            echo "${env_candidate}"
            return 0
        fi
    fi

    local candidates=(
        "${HOME}/.local/share/tla2tools.jar"
        "${HOME}/.tla/tla2tools.jar"
        "/usr/local/share/tla2tools.jar"
        "/opt/homebrew/share/tla2tools.jar"
        "/usr/share/tla2tools.jar"
    )

    local candidate
    for candidate in "${candidates[@]}"; do
        if [[ -f "${candidate}" ]]; then
            echo "${candidate}"
            return 0
        fi
    done

    tla_tools_err "tla2tools.jar not found; set TLA2TOOLS_JAR or run install_tla_tools.sh"
    return 1
}

find_timeout() {
    if command -v gtimeout >/dev/null 2>&1; then
        command -v gtimeout
        return 0
    fi

    if command -v timeout >/dev/null 2>&1; then
        command -v timeout
        return 0
    fi

    tla_tools_err "timeout not found; install coreutils (gtimeout) or GNU timeout"
    return 1
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    case "${1:-}" in
        --java)
            find_java
            ;;
        --jar|--tla)
            find_tla_tools
            ;;
        --timeout)
            find_timeout
            ;;
        -h|--help|"")
            cat << 'EOF'
Usage: find_tla_tools.sh [--java|--jar|--timeout]

Options:
  --java     Print resolved java binary path
  --jar      Print resolved tla2tools.jar path
  --timeout  Print resolved timeout binary (gtimeout/timeout)
EOF
            ;;
        *)
            tla_tools_err "Unknown option: $1"
            exit 1
            ;;
    esac
fi
