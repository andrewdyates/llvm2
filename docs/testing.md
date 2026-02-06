# Testing Guide

Local test execution for ai_template and repos derived from it.

## Prerequisites

- Python 3.11+
- pip (comes with Python)

## Setup

### Create Virtual Environment

```bash
# From repo root
python3 -m venv .venv
```

### Activate (optional but recommended)

```bash
source .venv/bin/activate
```

When activated, you can use `python` and `pytest` directly. Otherwise, prefix
commands with `.venv/bin/`.

### Install Dependencies

```bash
# With venv activated
pip install -r requirements.txt

# Or without activation
.venv/bin/pip install -r requirements.txt
```

This installs:
- `pytest` - test runner
- `pytest-cov` - coverage reporting
- `pytest-timeout` - test timeouts (prevents hangs)
- `hypothesis` - property-based testing
- `pre-commit` - git hooks (for integration tests)

## Running Tests

### Basic Commands

```bash
# All tests
.venv/bin/python -m pytest

# Verbose output (shows test names)
.venv/bin/python -m pytest -v

# Specific file
.venv/bin/python -m pytest tests/test_looper/test_rotation.py

# Specific test class
.venv/bin/python -m pytest tests/test_looper/test_rotation.py::TestGetRotationFocus

# Specific test method
.venv/bin/python -m pytest tests/test_looper/test_rotation.py::TestGetRotationFocus::test_freeform_every_third_iteration
```

### Telemetry Test Targets

`tests/test_telemetry.py` is a removed legacy path. Current telemetry coverage is split across:

- `tests/test_telemetry_core.py`
- `tests/test_telemetry_health.py`
- `tests/test_telemetry_tokens.py`
- `tests/test_looper/test_telemetry.py`

The stale path still appears in historical reports/design notes, so use `log_test.py run`
for telemetry verification to catch removed or missing pytest targets before execution:

```bash
./ai_template_scripts/log_test.py run "python3 -m pytest tests/test_telemetry_core.py tests/test_telemetry_health.py tests/test_telemetry_tokens.py tests/test_looper/test_telemetry.py -v"
```

### With Coverage

```bash
# Generate coverage report
.venv/bin/python -m pytest --cov=looper --cov-report=term-missing

# HTML report
.venv/bin/python -m pytest --cov=looper --cov-report=html
open htmlcov/index.html
```

### Timeout Configuration

Tests have a 10-second timeout by default (configured in `pyproject.toml`).

```bash
# Override timeout for slow tests
.venv/bin/python -m pytest --timeout=30

# Disable timeout (not recommended)
.venv/bin/python -m pytest --timeout=0
```

## Troubleshooting

### pytest not found

```
python3: No module named pytest
```

**Fix:** Install dependencies in the virtual environment:

```bash
.venv/bin/pip install -r requirements.txt
```

### pytest-timeout warnings

```
PytestConfigWarning: Unknown config option: timeout
```

**Fix:** The `pytest-timeout` plugin is not installed. Run:

```bash
.venv/bin/pip install pytest-timeout
```

### Import errors

```
ModuleNotFoundError: No module named 'looper'
```

**Fix:** Run pytest from the repo root or ensure `PYTHONPATH` includes the repo:

```bash
# From repo root
.venv/bin/python -m pytest

# Or with explicit path
PYTHONPATH=. .venv/bin/python -m pytest
```

### Tests hang indefinitely

The `pytest-timeout` plugin kills tests after 10 seconds. If tests hang:

1. Check if pytest-timeout is installed: `.venv/bin/pip show pytest-timeout`
2. Run with explicit timeout: `.venv/bin/python -m pytest --timeout=10`
3. For debugging, run single test with no timeout: `pytest path/to/test.py::test_name --timeout=0`

## Test Organization

```
tests/
├── conftest.py              # Shared fixtures
├── test_looper/             # Looper package tests
│   ├── test_result.py       # Result[T] tests
│   ├── test_issue_manager_base.py
│   └── ...
├── test_commit_msg_hook.py  # Git hook tests
└── ...
```

See `pyproject.toml` `[tool.pytest.ini_options]` for full pytest configuration.
