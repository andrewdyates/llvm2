#!/usr/bin/env python3
# Copyright 2026 Dropbox, Inc.
# Author: Andrew Yates
# Licensed under the Apache License, Version 2.0

"""
MCP server for setting terminal tab titles.

Simple plugin that provides set_tab_title tool for iTerm2/Terminal.app.
"""

import json
import os
import subprocess
import sys


def get_project_name() -> str:
    """Get project name from git remote or cwd."""
    try:
        result = subprocess.run(
            ["git", "remote", "get-url", "origin"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            url = result.stdout.strip()
            return url.rstrip("/").rsplit("/", 1)[-1].removesuffix(".git")
    except Exception:
        pass
    return os.path.basename(os.getcwd())


def get_role() -> str:
    """
    Get role from AI_ROLE env var (set by looper.py).
    Includes worker/prover ID if set (AI_WORKER_ID env var).
    Returns:
        W (worker), W1/W2 (multi-worker), P (prover), P1/P2 (multi-prover),
        R (researcher), M (manager), or U (user/human sessions)
    """
    role = os.environ.get("AI_ROLE", "").upper()
    role_map = {
        "WORKER": "W",
        "PROVER": "P",
        "RESEARCHER": "R",
        "MANAGER": "M",
    }
    short_role = role_map.get(role, "U")

    # Include worker/prover ID if set (multi-instance mode)
    worker_id = os.environ.get("AI_WORKER_ID", "")
    if worker_id and short_role in ("W", "P", "R"):
        return f"{short_role}{worker_id}"
    return short_role


def auto_set_title() -> None:
    """Automatically set terminal title during MCP initialize.

    Called during the MCP 'initialize' method to set the tab title
    based on current role (with worker ID if set) and project name.
    """
    role = get_role()
    project = get_project_name()
    title = f"[{role}]{project}"
    set_title(title)


def set_title_applescript(title: str) -> bool:
    """Set terminal tab title using AppleScript.

    NOTE: AppleScript is required for durable titles in iTerm2. Escape sequences
    get overwritten by later terminal activity. This complexity is intentional.

    Uses ITERM_SESSION_ID to target the correct session (not just the focused one).
    Sets session name AND tab name for maximum compatibility.

    In iTerm2:
    - Session 'name' is a session property
    - Tab also has a 'current session' whose name affects the tab title
    - Double-click edit sets both, which is why it "sticks"
    """
    # Escape quotes and backslashes for AppleScript string
    escaped_title = title.replace("\\", "\\\\").replace('"', '\\"')

    # Get session ID from environment - this identifies OUR terminal session
    session_id = os.environ.get("ITERM_SESSION_ID", "")

    if session_id:
        # Target specific session by ID - more reliable than "current session"
        # Session ID format: "w0t0p0:GUID" - extract GUID for matching
        guid = session_id.split(":")[-1] if ":" in session_id else session_id
        script = f'''tell application "iTerm2"
    repeat with w in windows
        repeat with t in tabs of w
            repeat with s in sessions of t
                try
                    if unique ID of s contains "{guid}" then
                        -- Set session name
                        set name of s to "{escaped_title}"
                        return true
                    end if
                end try
            end repeat
        end repeat
    end repeat
    return false
end tell'''
    else:
        # Fallback: target current session AND current tab
        script = f'''tell application "iTerm2"
    tell current window
        tell current tab
            tell current session
                set name to "{escaped_title}"
            end tell
        end tell
    end tell
end tell'''

    try:
        result = subprocess.run(
            ["osascript", "-e", script],
            capture_output=True,
            text=True,
            timeout=10,
        )
        return result.returncode == 0
    except Exception:
        return False


def set_title_escape(title: str) -> bool:
    """Set terminal tab title using escape sequences (fallback).

    Uses multiple escape sequences for maximum compatibility:
    - OSC 0: Standard title (icon + window)
    - OSC 1: Icon/tab name (often more "sticky" in iTerm2)
    - OSC 2: Window title

    Returns True if successful, False if no TTY available.
    """
    try:
        # Write directly to terminal, not stdout (which is for MCP protocol)
        with open("/dev/tty", "w") as tty:
            # OSC 0: Set icon name and window title
            tty.write(f"\033]0;{title}\007")
            # OSC 1: Set icon name (tab title in iTerm2).
            # This one tends to be more persistent.
            tty.write(f"\033]1;{title}\007")
            # OSC 2: Set window title
            tty.write(f"\033]2;{title}\007")
            tty.flush()
        return True
    except OSError:
        # No TTY available (CI, background processes, etc.)
        return False


def set_title(title: str) -> dict:
    """Set terminal tab title using multiple methods for reliability.

    Returns dict with results of each method tried.
    """
    results = {
        "title": title,
        "session_id": os.environ.get("ITERM_SESSION_ID", "(not set)"),
        "applescript": False,
        "escape_seq": False,
    }

    # Try AppleScript first (more reliable for "sticking")
    results["applescript"] = set_title_applescript(title)

    # Also try escape sequences as backup (directly to TTY)
    results["escape_seq"] = set_title_escape(title)

    return results


def handle_request(request: dict) -> dict | None:
    """Handle a single MCP request."""
    method = request.get("method", "")
    req_id = request.get("id")

    if method == "initialize":
        # Auto-set terminal title immediately on initialize
        # This is more reliable than waiting for notifications/initialized
        # which may not be sent by all MCP clients
        try:
            auto_set_title()
        except Exception as e:
            # Log to stderr for debugging, don't fail initialization
            print(f"[tab-title] auto_set_title failed: {e}", file=sys.stderr)
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {
                "protocolVersion": "2024-11-05",
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "tab-title", "version": "1.0.0"},
            },
        }

    if method == "notifications/initialized":
        # Keep for backwards compatibility with clients that send this
        # No-op since we already set the title during initialize
        return None

    if method == "tools/list":
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {
                "tools": [
                    {
                        "name": "set_tab_title",
                        "description": (
                            "[BOTH] Set the terminal tab/window title. "
                            "If no title provided, auto-generates from role and "
                            "project."
                        ),
                        "inputSchema": {
                            "type": "object",
                            "properties": {
                                "title": {
                                    "type": "string",
                                    "description": (
                                        "Custom title. If omitted, uses [X]<project> "
                                        "format where X is W/W1/P/P1/R/M/U."
                                    ),
                                },
                                "role": {
                                    "type": "string",
                                    "enum": [
                                        "W",
                                        "P",
                                        "R",
                                        "M",
                                        "U",
                                        "WORKER",
                                        "PROVER",
                                        "RESEARCHER",
                                        "MANAGER",
                                        "USER",
                                    ],
                                    "description": (
                                        "Role to display: W (worker), P (prover), "
                                        "R (researcher), M (manager), U (user). "
                                        "Defaults to AI_ROLE env var. "
                                        "Multi-instance IDs (W1, P2) auto-detected."
                                    ),
                                },
                                "project": {
                                    "type": "string",
                                    "description": (
                                        "Project name. Defaults to git remote name."
                                    ),
                                },
                            },
                            "required": [],
                        },
                    },
                ],
            },
        }

    if method == "tools/call":
        params = request.get("params", {})
        tool_name = params.get("name")
        args = params.get("arguments", {})

        if tool_name == "set_tab_title":
            # Get title - either custom or auto-generated
            if "title" in args:
                title = args["title"]
            else:
                role_arg = args.get("role")
                if not role_arg:
                    # No role specified - use environment-aware role with worker ID
                    role = get_role()
                else:
                    role_upper = role_arg.upper()
                    # Full role names should use env-aware role (to get worker ID)
                    full_names = {"WORKER", "PROVER", "RESEARCHER", "MANAGER", "USER"}
                    # Short role names without ID should also use env-aware role
                    short_names = {"W", "P", "R", "M", "U"}
                    if role_upper in full_names or role_upper in short_names:
                        # Use get_role() to preserve worker/prover ID from environment
                        role = get_role()
                    else:
                        # Already includes ID (W1, P2, etc.) - validate and use as-is
                        # Only allow valid patterns: single letter + optional digits
                        if (
                            len(role_upper) >= 1
                            and role_upper[0] in "WPRMU"
                            and role_upper[1:].isdigit()
                        ):
                            role = role_upper
                        else:
                            # Invalid format - fall back to env-aware role
                            role = get_role()
                project = args.get("project") or get_project_name()
                title = f"[{role}]{project}"

            try:
                results = set_title(title)
                methods_used = []
                if results["applescript"]:
                    methods_used.append("AppleScript")
                if results["escape_seq"]:
                    methods_used.append("escape sequences")

                if methods_used:
                    msg = f"Tab title set to: {title} (via {', '.join(methods_used)})"
                else:
                    msg = (
                        f"Tab title '{title}' - no methods succeeded. "
                        f"Session ID: {results['session_id']}"
                    )

                return {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": {
                        "content": [{"type": "text", "text": msg}],
                    },
                }
            except Exception as e:
                return {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": {
                        "content": [
                            {"type": "text", "text": f"Failed to set title: {e}"}
                        ],
                        "isError": True,
                    },
                }
        else:
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "error": {"code": -32601, "message": f"Unknown tool: {tool_name}"},
            }

    else:
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "error": {"code": -32601, "message": f"Unknown method: {method}"},
        }


def main() -> None:
    """Main loop - read JSON-RPC from stdin, write responses to stdout."""
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue

        try:
            request = json.loads(line)
            response = handle_request(request)
            if response is not None:
                print(json.dumps(response), flush=True)
        except json.JSONDecodeError as e:
            print(
                json.dumps(
                    {
                        "jsonrpc": "2.0",
                        "id": None,
                        "error": {"code": -32700, "message": f"Parse error: {e}"},
                    }
                ),
                flush=True,
            )


if __name__ == "__main__":
    main()
