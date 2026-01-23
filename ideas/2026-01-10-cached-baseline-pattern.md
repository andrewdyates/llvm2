# Cached Baseline Results Pattern

From tla2 issue #48, mail to ai_template #111.

## Problem

Running reference implementations (TLC, Python baseline) on every test is slow/flaky.

## Solution

Cache reference results with metadata:

```json
{
  "_metadata": {
    "generated_at": "...",
    "reference_version": "1.8.0",
    "machine": {"cpu": "...", "cores": 12, "memory_gb": 64}
  },
  "test_cases": {
    "case_name": {
      "expected_output": {...},
      "runtime_stats": {"mean_seconds": 360.0, "stddev_seconds": 1.6}
    }
  }
}
```

## Applicable Projects

tla2, z4, langchain_rs, docling_rs - any Rust port of reference impl.

## Status

Not adding to template - individual projects can adopt pattern directly.
