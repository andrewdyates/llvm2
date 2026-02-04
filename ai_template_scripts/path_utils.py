# Copyright 2026 Dropbox, Inc.
# Author: Andrew Yates <ayates@dropbox.com>
# Licensed under the Apache License, Version 2.0
"""Shared path utilities for ai_template scripts.

This module provides common utilities for handling path parameters,
including deprecation support for renamed parameters.
"""

from __future__ import annotations

import warnings
from pathlib import Path

__all__ = ["resolve_path_alias"]


def resolve_path_alias(
    new_name: str,
    old_name: str,
    positional_value: str | Path | None,
    kwargs: dict[str, str | Path | None],
    func_name: str,
) -> Path:
    """Resolve path parameter from positional arg or keyword (with deprecation warning).

    This function supports backward-compatible parameter renaming. It allows callers
    to use either the old or new parameter name during the deprecation period.

    Args:
        new_name: The new standardized parameter name (e.g., 'dir_path')
        old_name: The deprecated parameter name (e.g., 'root')
        positional_value: Value passed as positional argument (or None)
        kwargs: The **kwargs dict from the function call (will be modified)
        func_name: Name of the calling function for error messages

    Returns:
        The resolved Path value

    Raises:
        TypeError: If parameter is missing or provided multiple times

    Example:
        >>> def get_git_info(dir_path: Path | None = None, **kwargs) -> dict:
        ...     resolved = resolve_path_alias(
        ...         "dir_path", "root", dir_path, kwargs, "get_git_info"
        ...     )
        ...     # use resolved path...

        # All these work:
        >>> get_git_info(Path("."))           # positional
        >>> get_git_info(dir_path=Path("."))  # new name (preferred)
        >>> get_git_info(root=Path("."))      # deprecated (emits warning)
    """
    old_value = kwargs.pop(old_name, None)
    new_value = kwargs.pop(new_name, None)

    # Check for extra kwargs
    if kwargs:
        raise TypeError(
            f"{func_name}() got unexpected keyword arguments: {list(kwargs.keys())}"
        )

    # Count how many ways the value was provided
    provided = [v for v in [positional_value, old_value, new_value] if v is not None]

    if len(provided) == 0:
        raise TypeError(f"{func_name}() missing required argument: '{new_name}'")

    if len(provided) > 1:
        raise TypeError(
            f"{func_name}() got multiple values for path argument "
            f"(use '{new_name}' keyword only)"
        )

    # Emit deprecation warning if old name was used
    if old_value is not None:
        warnings.warn(
            f"{func_name}(): '{old_name}' is deprecated, use '{new_name}' instead",
            DeprecationWarning,
            stacklevel=3,
        )
        return Path(old_value) if not isinstance(old_value, Path) else old_value

    if new_value is not None:
        return Path(new_value) if not isinstance(new_value, Path) else new_value

    # At this point, positional_value must be non-None (len(provided) == 1)
    assert positional_value is not None
    return (
        Path(positional_value)
        if not isinstance(positional_value, Path)
        else positional_value
    )
