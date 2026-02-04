#!/usr/bin/env bash
# Copyright 2026 Dropbox, Inc.
# Author: Andrew Yates <ayates@dropbox.com>
# Licensed under the Apache License, Version 2.0
#
# Safe cleanup helper for large build artifacts (target/).
#
# Default behavior is DRY RUN: prints what would be deleted.
#
# Safety checks:
# - Refuses to run while other worker sessions are active (based on .worker_*_files.json)
# - Refuses to run while cargo wrapper build/test locks are held
# - Use --force to override safety checks (ONLY after verifying no active sessions)
#
# Modes:
# - incremental (default): remove target/**/incremental
# - build: remove target/**/incremental + target/**/build + target/**/.fingerprint
# - full: run `cargo clean` (nukes target/)
#
# Created for: #764 (target/ artifact bloat cleanup)

set -euo pipefail

MODE="incremental"
APPLY=0
FORCE=0

usage() {
  cat <<'EOF'

Usage:
  ./scripts/clean_artifacts.sh [--mode MODE] [--apply] [--force]

Safe cleanup helper for large build artifacts (target/). Default is dry-run.

Options:
  --mode MODE   One of: incremental|build|full (default: incremental)
  --apply       Actually delete files (default: dry-run)
  --force       Override safety checks (dangerous)
  -h, --help    Show this help

Examples:
  # Dry-run: show what incremental cleanup would delete
  ./scripts/clean_artifacts.sh

  # Apply incremental cleanup
  ./scripts/clean_artifacts.sh --apply

  # Apply full cleanup (cargo clean)
  ./scripts/clean_artifacts.sh --mode full --apply
EOF
}

die() {
  echo "ERROR: $*" >&2
  exit 1
}

is_pid_alive() {
  local pid="$1"
  [[ -n "$pid" ]] || return 1
  [[ "$pid" =~ ^[0-9]+$ ]] || return 1
  kill -0 "$pid" 2>/dev/null
}

repo_root() {
  git rev-parse --show-toplevel 2>/dev/null || pwd
}

check_worker_sessions() {
  local root="$1"
  local tracker
  local any_alive=0

  shopt -s nullglob
  local trackers=( "$root"/.worker_*_files.json )
  shopt -u nullglob

  [[ ${#trackers[@]} -gt 0 ]] || return 0

  for tracker in "${trackers[@]}"; do
    if ! command -v python3 >/dev/null 2>&1; then
      # Without python, conservatively refuse unless forced.
      any_alive=1
      continue
    fi

    local pid worker_id
    read -r pid worker_id < <(
      python3 - "$tracker" <<'PY' 2>/dev/null || echo "0 0"
import json
import sys

try:
    with open(sys.argv[1], "r", encoding="utf-8") as f:
        data = json.load(f)
    pid = int(data.get("pid", 0) or 0)
    worker_id = int(data.get("worker_id", 0) or 0)
    print(f"{pid} {worker_id}")
except Exception:
    print("0 0")
PY
    )

    if is_pid_alive "$pid"; then
      echo "Active worker detected: W${worker_id} (pid=${pid}) via $(basename "$tracker")" >&2
      any_alive=1
    fi
  done

  if [[ "$any_alive" -eq 1 && "$FORCE" -ne 1 ]]; then
    die "Refusing to clean while worker sessions are active. Re-run with --force to override."
  fi
}

check_cargo_locks() {
  local root="$1"
  local repo_name lock_dir
  repo_name="$(basename "$root")"
  lock_dir="$HOME/.ait_cargo_lock/$repo_name"

  local lock_pid_file test_lock_pid_file
  lock_pid_file="$lock_dir/lock.pid"
  test_lock_pid_file="$lock_dir/lock.test.pid"

  local any_locked=0

  if [[ -f "$lock_pid_file" ]]; then
    local pid
    pid="$(cat "$lock_pid_file" 2>/dev/null || true)"
    if is_pid_alive "$pid"; then
      echo "Active cargo build lock detected (pid=${pid}): $lock_pid_file" >&2
      any_locked=1
    fi
  fi

  if [[ -f "$test_lock_pid_file" ]]; then
    local pid
    pid="$(cat "$test_lock_pid_file" 2>/dev/null || true)"
    if is_pid_alive "$pid"; then
      echo "Active cargo test lock detected (pid=${pid}): $test_lock_pid_file" >&2
      any_locked=1
    fi
  fi

  if [[ "$any_locked" -eq 1 && "$FORCE" -ne 1 ]]; then
    die "Refusing to clean while cargo locks are held. Re-run with --force to override."
  fi
}

print_dir_size() {
  local path="$1"
  if command -v du >/dev/null 2>&1; then
    du -sh "$path" 2>/dev/null || true
  fi
}

collect_cleanup_paths() {
  local target_dir="$1"
  local mode="$2"

  [[ -d "$target_dir" ]] || return 0

  case "$mode" in
    incremental)
      find "$target_dir" -type d -name incremental -prune -print
      ;;
    build)
      find "$target_dir" -type d -name incremental -prune -print
      find "$target_dir" -type d -name build -prune -print
      find "$target_dir" -type d -name .fingerprint -prune -print
      ;;
    *)
      die "unknown mode: $mode"
      ;;
  esac
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --mode=*)
      MODE="${1#--mode=}"
      shift
      ;;
    --mode)
      [[ $# -ge 2 ]] || die "--mode requires an argument (incremental|build|full)"
      MODE="$2"
      shift 2
      ;;
    --apply)
      APPLY=1
      shift
      ;;
    --force)
      FORCE=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      die "unknown argument: $1 (try --help)"
      ;;
  esac
done

case "$MODE" in
  incremental|build|full) ;;
  *) die "--mode must be incremental|build|full (got: $MODE)" ;;
esac

ROOT="$(repo_root)"
TARGET_DIR="$ROOT/target"

check_worker_sessions "$ROOT"
check_cargo_locks "$ROOT"

echo "Repo: $ROOT"
if [[ -d "$TARGET_DIR" ]]; then
  echo "target/: $(print_dir_size "$TARGET_DIR" | awk '{print $1}' || true)"
else
  echo "target/: (missing)"
fi

if [[ "$MODE" == "full" ]]; then
  if [[ "$APPLY" -eq 0 ]]; then
    echo "Dry-run: would run: cargo clean"
    exit 0
  fi
  echo "Running: cargo clean"
  cargo clean
  echo "Done."
  exit 0
fi

mapfile -t PATHS < <(collect_cleanup_paths "$TARGET_DIR" "$MODE" | sort -u)

if [[ ${#PATHS[@]} -eq 0 ]]; then
  echo "Nothing to clean for mode=$MODE."
  exit 0
fi

echo "Mode: $MODE"
echo "Candidates (${#PATHS[@]}):"
for p in "${PATHS[@]}"; do
  if [[ -d "$p" ]]; then
    echo "  $(print_dir_size "$p" | awk '{print $1}' || true)  $p"
  else
    echo "  (missing) $p"
  fi
done

if [[ "$APPLY" -eq 0 ]]; then
  echo "Dry-run: no files deleted. Re-run with --apply to clean."
  exit 0
fi

echo "Deleting ${#PATHS[@]} path(s)..."
for p in "${PATHS[@]}"; do
  if [[ -d "$p" ]]; then
    printf '  rm -rf -- %q\n' "$p"
    rm -rf -- "$p"
  fi
done

echo "Done."
if [[ -d "$TARGET_DIR" ]]; then
  echo "target/ now: $(print_dir_size "$TARGET_DIR" | awk '{print $1}' || true)"
fi
