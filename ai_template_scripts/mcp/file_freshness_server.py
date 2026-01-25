#!/usr/bin/env python3
"""
file_freshness_server.py - MCP server for file freshness tracking

Provides FreshEdit and FreshRead tools that track file read timestamps
and block edits to files modified since last read.

For use with Codex CLI or any MCP-compatible client.

Copyright 2026 Dropbox, Inc.
Author: Andrew Yates <ayates@dropbox.com>
"""

import json
import sys
import time
from pathlib import Path

# MCP protocol constants
JSONRPC_VERSION = "2.0"

# State file for tracking read timestamps
STATE_FILE = Path.home() / ".mcp_file_freshness_state.json"


def load_state() -> dict:
    """Load read timestamps from state file."""
    if STATE_FILE.exists():
        try:
            return json.loads(STATE_FILE.read_text())
        except (OSError, json.JSONDecodeError):
            pass
    return {"files_read": {}}


def save_state(state: dict) -> None:
    """Save read timestamps to state file."""
    STATE_FILE.write_text(json.dumps(state, indent=2))


def record_read(file_path: str) -> None:
    """Record that a file was read."""
    state = load_state()
    abs_path = str(Path(file_path).resolve())
    state["files_read"][abs_path] = time.time()
    save_state(state)


def check_freshness(file_path: str) -> tuple[bool, str]:
    """
    Check if file is fresh (not modified since last read).

    Returns:
        (is_fresh, message)
    """
    abs_path = str(Path(file_path).resolve())
    path = Path(abs_path)

    if not path.exists():
        return True, "File doesn't exist yet (new file)"

    state = load_state()
    last_read = state.get("files_read", {}).get(abs_path)

    if last_read is None:
        return False, f"You haven't read {file_path} in this session. Read it first."

    file_mtime = path.stat().st_mtime

    if file_mtime > last_read:
        last_read_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(last_read))
        file_mtime_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(file_mtime))
        return False, (
            f"File {file_path} was modified at {file_mtime_str} "
            f"but you last read it at {last_read_str}. Re-read before editing."
        )

    return True, "File is fresh"


def handle_tool_call(name: str, arguments: dict) -> dict:
    """Handle MCP tool calls."""

    if name == "fresh_read":
        file_path = arguments.get("file_path", "")
        if not file_path:
            return {"error": "file_path is required"}

        path = Path(file_path)
        if not path.exists():
            return {"error": f"File not found: {file_path}"}

        # Read the file
        try:
            content = path.read_text()
        except Exception as e:
            return {"error": f"Failed to read file: {e}"}

        # Record the read timestamp
        record_read(file_path)

        # Return content with line numbers
        lines = content.split("\n")
        numbered = [f"{i + 1:6d}\t{line}" for i, line in enumerate(lines)]
        return {"content": "\n".join(numbered)}

    if name == "fresh_edit":
        file_path = arguments.get("file_path", "")
        old_string = arguments.get("old_string", "")
        new_string = arguments.get("new_string", "")

        if not file_path:
            return {"error": "file_path is required"}
        if not old_string:
            return {"error": "old_string is required"}

        # Check freshness
        is_fresh, message = check_freshness(file_path)
        if not is_fresh:
            return {"error": f"BLOCKED: {message}"}

        # Perform the edit
        path = Path(file_path)
        if not path.exists():
            return {"error": f"File not found: {file_path}"}

        try:
            content = path.read_text()
        except Exception as e:
            return {"error": f"Failed to read file: {e}"}

        if old_string not in content:
            return {"error": f"old_string not found in {file_path}"}

        if content.count(old_string) > 1:
            return {
                "error": f"old_string appears multiple times in {file_path}. Make it unique."
            }

        new_content = content.replace(old_string, new_string, 1)

        try:
            path.write_text(new_content)
        except Exception as e:
            return {"error": f"Failed to write file: {e}"}

        # Update read timestamp after successful edit
        record_read(file_path)

        return {"success": True, "message": f"Edited {file_path}"}

    if name == "check_freshness":
        file_path = arguments.get("file_path", "")
        if not file_path:
            return {"error": "file_path is required"}

        is_fresh, message = check_freshness(file_path)
        return {"is_fresh": is_fresh, "message": message}

    return {"error": f"Unknown tool: {name}"}


def handle_request(request: dict) -> dict:
    """Handle incoming JSON-RPC request."""
    method = request.get("method", "")
    params = request.get("params", {})
    req_id = request.get("id")

    if method == "initialize":
        return {
            "jsonrpc": JSONRPC_VERSION,
            "id": req_id,
            "result": {
                "protocolVersion": "2024-11-05",
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "file-freshness-server", "version": "1.0.0"},
            },
        }

    if method == "tools/list":
        return {
            "jsonrpc": JSONRPC_VERSION,
            "id": req_id,
            "result": {
                "tools": [
                    {
                        "name": "fresh_read",
                        "description": "Read a file and record the read timestamp for freshness tracking.",
                        "inputSchema": {
                            "type": "object",
                            "properties": {
                                "file_path": {
                                    "type": "string",
                                    "description": "Path to the file to read",
                                }
                            },
                            "required": ["file_path"],
                        },
                    },
                    {
                        "name": "fresh_edit",
                        "description": "Edit a file, but only if it hasn't been modified since you last read it. Blocks stale edits.",
                        "inputSchema": {
                            "type": "object",
                            "properties": {
                                "file_path": {
                                    "type": "string",
                                    "description": "Path to the file to edit",
                                },
                                "old_string": {
                                    "type": "string",
                                    "description": "The exact string to replace",
                                },
                                "new_string": {
                                    "type": "string",
                                    "description": "The replacement string",
                                },
                            },
                            "required": ["file_path", "old_string", "new_string"],
                        },
                    },
                    {
                        "name": "check_freshness",
                        "description": "Check if a file is fresh (not modified since last read).",
                        "inputSchema": {
                            "type": "object",
                            "properties": {
                                "file_path": {
                                    "type": "string",
                                    "description": "Path to check",
                                }
                            },
                            "required": ["file_path"],
                        },
                    },
                ]
            },
        }

    if method == "tools/call":
        tool_name = params.get("name", "")
        arguments = params.get("arguments", {})
        result = handle_tool_call(tool_name, arguments)

        if "error" in result:
            return {
                "jsonrpc": JSONRPC_VERSION,
                "id": req_id,
                "result": {
                    "content": [{"type": "text", "text": result["error"]}],
                    "isError": True,
                },
            }
        text = result.get("content") or result.get("message") or json.dumps(result)
        return {
            "jsonrpc": JSONRPC_VERSION,
            "id": req_id,
            "result": {"content": [{"type": "text", "text": text}]},
        }

    if method == "notifications/initialized":
        # No response needed for notifications
        return None

    return {
        "jsonrpc": JSONRPC_VERSION,
        "id": req_id,
        "error": {"code": -32601, "message": f"Method not found: {method}"},
    }


def main():
    """Main loop - read JSON-RPC from stdin, write responses to stdout."""
    while True:
        try:
            line = sys.stdin.readline()
            if not line:
                break

            request = json.loads(line)
            response = handle_request(request)

            if response is not None:
                sys.stdout.write(json.dumps(response) + "\n")
                sys.stdout.flush()

        except json.JSONDecodeError as e:
            error_response = {
                "jsonrpc": JSONRPC_VERSION,
                "id": None,
                "error": {"code": -32700, "message": f"Parse error: {e}"},
            }
            sys.stdout.write(json.dumps(error_response) + "\n")
            sys.stdout.flush()
        except Exception as e:
            sys.stderr.write(f"Error: {e}\n")
            sys.stderr.flush()


if __name__ == "__main__":
    main()
