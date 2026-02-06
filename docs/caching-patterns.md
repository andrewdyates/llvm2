# Caching Patterns in Looper
<!-- Issue #2174 tracking -->

Copyright 2026 Dropbox, Inc.
Author: Andrew Yates <ayates@dropbox.com>

This document defines standard caching patterns for the looper package.

## Pattern Summary

| Use Case | Pattern | Lifecycle | Clear Method |
|----------|---------|-----------|--------------|
| Iteration-scoped data | Class cache | Per iteration | `.clear()` |
| Session-scoped config | Module global | Process lifetime | Setter function |
| Pure function results | `@functools.cache` | Process lifetime | `.cache_clear()` |

## Pattern 1: Iteration Cache (Class-level)

Use for data that should be fetched once per iteration and shared across functions.

**When to use:**
- API data (GitHub issues, metrics)
- Expensive computations that are valid for one iteration
- Data that multiple functions need within the same iteration

**Implementation:**

```python
from looper.result import Result

class IterationDataCache:
    """Cache data for a single iteration.

    Call clear() at iteration start. All functions using this cache
    share the same data until the next clear().
    """

    _data: dict | None = None
    _error: str | None = None

    @classmethod
    def clear(cls) -> None:
        """Clear cache. Call at iteration start."""
        cls._data = None
        cls._error = None

    @classmethod
    def get(cls) -> Result[dict]:
        """Get cached data, fetching if needed."""
        if cls._error is not None:
            return Result.failure(cls._error)
        if cls._data is not None:
            return Result.success(cls._data)

        # Fetch data
        result = _fetch_data()
        if result.ok:
            cls._data = result.value
        else:
            cls._error = result.error
        return result
```

**Lifecycle:**
1. Iteration starts → `IterationDataCache.clear()`
2. First access → fetches and caches
3. Subsequent access → returns cached value
4. Next iteration → repeat from step 1

**Example:** `looper/context/issue_context.py:IterationIssueCache`

## Pattern 2: Session Config (Module Global)

Use for config values loaded once per session and rarely changed.

**When to use:**
- Config file values
- Environment variable lookups
- Mode flags (local mode, debug mode)

**Implementation:**

```python
_CONFIG_VALUE: str | None = None

def set_config_value(value: str | None) -> None:
    """Set config from parsed config file."""
    global _CONFIG_VALUE
    _CONFIG_VALUE = value

def get_config_value() -> str | None:
    """Get cached config value."""
    return _CONFIG_VALUE
```

**Lifecycle:**
1. Session starts → config parsed → `set_config_value()` called
2. Functions read value via `get_config_value()`
3. Value persists until session ends

**Example:** `looper/subprocess_utils.py:_LOCAL_MODE_FROM_CONFIG`

## Pattern 3: Pure Function Cache (functools)

Use for pure functions with deterministic outputs based on inputs.

**When to use:**
- Path computations
- String transformations
- Deterministic parsing

**Implementation:**

```python
from functools import cache

@cache
def compute_expensive_value(key: str) -> int:
    """Compute value (cached per unique key)."""
    # Expensive computation
    return len(key) * 42

# To clear if needed:
# compute_expensive_value.cache_clear()
```

**Lifecycle:**
1. First call with args → computes and caches
2. Same args → returns cached value
3. Different args → computes and caches separately
4. Process ends → cache cleared

**Note:** Currently not used in looper, but appropriate for pure functions.

## Testing Caches

### Iteration Cache Testing

```python
from unittest.mock import patch

from looper.result import Result

def test_iteration_cache():
    """Test iteration cache lifecycle."""
    # Clear before test
    IterationDataCache.clear()

    # Mock the fetch
    with patch("module._fetch_data") as mock:
        mock.return_value = Result.success({"key": "value"})

        # First access fetches
        result1 = IterationDataCache.get()
        assert mock.call_count == 1

        # Second access uses cache
        result2 = IterationDataCache.get()
        assert mock.call_count == 1  # No new fetch

        # Clear and access again
        IterationDataCache.clear()
        result3 = IterationDataCache.get()
        assert mock.call_count == 2  # New fetch
```

### Session Config Testing

```python
def test_session_config():
    """Test session config pattern."""
    # Reset to known state
    set_config_value(None)
    assert get_config_value() is None

    # Set value
    set_config_value("test")
    assert get_config_value() == "test"

    # Clean up
    set_config_value(None)
```

### Pure Function Cache Testing

```python
def test_functools_cache():
    """Test functools.cache pattern."""
    # Clear cache before test
    compute_expensive_value.cache_clear()

    # Test caching
    result1 = compute_expensive_value("key")
    result2 = compute_expensive_value("key")
    assert result1 == result2

    # Verify cache was used (implementation-specific)
    info = compute_expensive_value.cache_info()
    assert info.hits == 1
    assert info.misses == 1
```

## Anti-patterns

### Don't: Mix Patterns

```python
# BAD: Global that needs iteration clearing
_cached_issues: list | None = None

def get_issues():
    global _cached_issues
    if _cached_issues is None:
        _cached_issues = fetch_issues()
    return _cached_issues
# Problem: Who clears this? When?
```

### Don't: Forget Clear Calls

```python
# BAD: Cache without clear mechanism
class BrokenCache:
    _data = None

    @classmethod
    def get(cls):
        if cls._data is None:
            cls._data = fetch()
        return cls._data
# Problem: No way to refresh stale data
```

### Don't: Cache Non-deterministic Results

```python
# BAD: Caching time-dependent value
@cache
def get_current_time():
    return datetime.now()
# Problem: Returns stale time forever
```

## Choosing a Pattern

```
Is the data iteration-scoped?
├── Yes → Pattern 1 (Iteration Cache)
└── No
    └── Is it config/mode data set once?
        ├── Yes → Pattern 2 (Session Config)
        └── No
            └── Is it a pure function?
                ├── Yes → Pattern 3 (functools.cache)
                └── No → Don't cache (or reconsider design)
```

## Related

- `looper/context/issue_context.py` - IterationIssueCache implementation
- `looper/subprocess_utils.py` - Session config example
- Issue #2174 - Pattern standardization tracking
