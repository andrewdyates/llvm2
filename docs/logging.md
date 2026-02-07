# Logging Guidance

Copyright 2026 Dropbox, Inc.
Author: Andrew Yates <ayates@dropbox.com>

## Sensitive Data

The following should NOT be logged directly (OWASP Logging Cheat Sheet):

| Category | Examples | Risk |
|----------|----------|------|
| API Keys | `sk-ant-*`, `sk-*`, `ghp_*`, `AKIA*` | Account compromise |
| Passwords | Password fields, auth tokens | Account compromise |
| Private Keys | PEM files, SSH keys, certificates | System compromise |
| PII | Email addresses, names, addresses | Privacy violation |
| Session Tokens | JWTs, Bearer tokens, cookies | Session hijacking |
| Cloud Credentials | AWS secret keys, GCP service accounts | Cloud account compromise |

## Log Scrubber

Use `ai_template_scripts/log_scrubber.py` to sanitize logs before sharing:

```bash
# Scrub a single log file (in-place)
./ai_template_scripts/log_scrubber.py worker_logs/worker_iter_1.jsonl

# Scrub all logs in directory
./ai_template_scripts/log_scrubber.py worker_logs/

# Preview scrubbed output without modifying
./ai_template_scripts/log_scrubber.py --stdout worker_logs/worker_iter_1.jsonl

# Include email redaction (off by default)
./ai_template_scripts/log_scrubber.py --scrub-emails worker_logs/
```

### Automatic Scrubbing

Enable automatic log scrubbing in `.looper_config.json`:

```json
{
  "scrub_logs": true
}
```

When enabled, logs are scrubbed immediately after each iteration completes.

### What Gets Scrubbed

The scrubber replaces sensitive patterns with redaction markers:

| Pattern | Replacement |
|---------|-------------|
| Anthropic API keys (`sk-ant-*`) | `[REDACTED:ANTHROPIC_KEY]` |
| OpenAI API keys (`sk-*`) | `[REDACTED:OPENAI_KEY]` |
| GitHub tokens (`ghp_*`, `gho_*`) | `[REDACTED:GITHUB_TOKEN]` |
| Slack tokens (`xoxb-*`, `xoxp-*`) | `[REDACTED:SLACK_TOKEN]` |
| AWS access keys (`AKIA*`) | `[REDACTED:AWS_ACCESS_KEY]` |
| AWS secret keys | `[REDACTED:AWS_SECRET]` |
| Bearer/JWT tokens | `[REDACTED:BEARER]` |
| Passwords | `[REDACTED:PASSWORD]` |
| Generic API keys | `[REDACTED:API_KEY]` |
| Private key blocks | `[REDACTED:PRIVATE_KEY]` |
| Home directory paths | `~` (optional, on by default) |

### What is NOT Scrubbed

The scrubber preserves these for debugging:

- Session IDs (needed for log correlation)
- Timestamps (needed for debugging)
- Tool names and commands (needed for audit)
- File paths (except home directory)

## Best Practices

1. **Enable automatic scrubbing** in shared environments
2. **Scrub before sharing** logs with others
3. **Review logs** before committing to version control
4. **Use environment variables** for secrets, not command-line arguments
5. **Avoid logging** full request/response bodies that may contain secrets

## Worker Log Schema (OpenTelemetry-aligned)

Worker logs in `worker_logs/*.jsonl` follow a schema aligned with the
[OpenTelemetry Log Data Model](https://opentelemetry.io/docs/specs/otel/logs/data-model/).

### Standard Fields

| OTel Field | JSONL Field | Type | Description |
|------------|-------------|------|-------------|
| Timestamp | (filename) | embedded in filename | Filename contains `YYYYMMDD_HHMMSS` |
| TraceId | `session_id` | string (UUID) | Session trace identifier |
| SeverityText | `type` + `subtype` | string | Event type: system/init, user, assistant, result |
| Body | `message` | object | Full message content |
| Resource | (init record) | object | model, tools, mcp_servers in init |
| Attributes | (varies) | object | Event-specific data |

Note: Unlike OTel, our logs embed timestamp in filenames rather than per-record.

### Record Types

**Init Record** (session start):
```json
{
  "type": "system",
  "subtype": "init",
  "session_id": "uuid",
  "cwd": "/path/to/repo",
  "model": "claude-opus-4-5-...",
  "tools": ["Read", "Write", ...],
  "mcp_servers": [{"name": "...", "status": "connected"}],
  "permissionMode": "bypassPermissions",
  "uuid": "unique-record-id"
}
```

**User Record** (tool result):
```json
{
  "type": "user",
  "message": {
    "role": "user",
    "content": [{"tool_use_id": "...", "type": "tool_result", ...}]
  },
  "session_id": "uuid",
  "tool_use_result": {"stdout": "...", "stderr": "..."},
  "uuid": "unique-record-id"
}
```

**Assistant Record** (AI response):
```json
{
  "type": "assistant",
  "message": {
    "model": "...",
    "content": [{"type": "text", "text": "..."}, {"type": "tool_use", ...}],
    "usage": {"input_tokens": N, "output_tokens": N}
  },
  "session_id": "uuid",
  "uuid": "unique-record-id"
}
```

**Result Record** (session end):
```json
{
  "type": "result",
  "session_id": "uuid",
  "usage": {"input_tokens": N, "output_tokens": N, "cache_read_input_tokens": N}
}
```

### Field Mapping to OpenTelemetry

| Purpose | OTel Standard | Our Implementation |
|---------|--------------|-------------------|
| Event timing | Timestamp (nanos) | ISO 8601 string |
| Trace correlation | TraceId + SpanId | session_id + uuid |
| Event type | SeverityText | type + subtype |
| Log body | Body | message object |
| Source info | Resource | model, tools, mcp_servers |
| Context | Attributes | tool_use_result, usage, etc. |

### Migration Notes

Existing logs use the schema above. To add new fields:
1. Use snake_case for new field names
2. Add field to this schema documentation
3. Update `json_to_text/` package if field needs display formatting

## References

- [OWASP Logging Cheat Sheet](https://cheatsheetseries.owasp.org/cheatsheets/Logging_Cheat_Sheet.html)
- [OpenTelemetry Log Data Model](https://opentelemetry.io/docs/specs/otel/logs/data-model/)
