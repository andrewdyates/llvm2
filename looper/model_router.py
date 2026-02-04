# Copyright 2026 Dropbox, Inc.
# Author: Andrew Yates <ayates@dropbox.com>
# Licensed under the Apache License, Version 2.0

"""Model routing for per-role model selection.

Implements the model routing design from #1888 and
designs/2026-02-02-per-role-model-routing.md.

ModelRouter resolves which model to use based on:
1. model_routing.audit when audit_round > 0
2. model_routing.roles.<role> when present
3. model_routing.default when present
4. Legacy keys (claude_model, codex_model, codex_models, dasher_model)

ModelSwitchingPolicy handles session resume behavior when models change.
"""

from dataclasses import dataclass
from typing import Any, Literal

from looper.log import log_info

# Type alias for AI tools to ensure consistency across the module (#1913)
AiTool = Literal["claude", "codex", "dasher"]

__all__ = [
    "AiTool",
    "ModelRouter",
    "ModelSelection",
    "ModelSwitchingPolicy",
]


@dataclass
class ModelSelection:
    """Result of model selection.

    Attributes:
        model: Selected model string, or None if no model specified
        source: Where the selection came from for logging
        tool_key: Which config key this applies to (claude_model, codex_model, etc.)
    """

    model: str | None
    source: str
    tool_key: str


class ModelRouter:
    """Routes model selection based on role and config.

    Contracts:
        REQUIRES: config is a valid configuration dict
        ENSURES: select_model returns ModelSelection with valid source
        ENSURES: Precedence: audit > roles > default > legacy
    """

    def __init__(self, config: dict[str, Any]) -> None:
        """Initialize router with config.

        Args:
            config: Merged role configuration dict
        """
        self._config = config
        self._routing = config.get("model_routing", {})

    def update_config(self, config: dict[str, Any]) -> None:
        """Update config for the next iteration."""
        self._config = config
        self._routing = config.get("model_routing", {})

    def select_model(
        self,
        role: str,
        ai_tool: AiTool,
        audit_round: int = 0,
    ) -> ModelSelection:
        """Select the model for the given role and tool.

        Precedence (highest to lowest):
        1. model_routing.audit (when audit_round > 0)
        2. model_routing.roles.<role>
        3. model_routing.default
        4. Legacy keys (claude_model, codex_model, codex_models, dasher_model)

        Args:
            role: Role name (worker, manager, researcher, prover)
            ai_tool: Selected AI tool
            audit_round: Current audit round (0 for main iteration)

        Returns:
            ModelSelection with selected model and source
        """
        tool_key = self._get_tool_key(ai_tool)

        # 1. Check audit routing (highest priority when in audit)
        if audit_round > 0 and "audit" in self._routing:
            audit_config = self._routing["audit"]
            if isinstance(audit_config, dict) and tool_key in audit_config:
                model = audit_config[tool_key]
                return ModelSelection(
                    model=model,
                    source=f"model_routing.audit.{tool_key}",
                    tool_key=tool_key,
                )

        # 2. Check role-specific routing
        roles_config = self._routing.get("roles", {})
        if isinstance(roles_config, dict) and role in roles_config:
            role_config = roles_config[role]
            if isinstance(role_config, dict) and tool_key in role_config:
                model = role_config[tool_key]
                return ModelSelection(
                    model=model,
                    source=f"model_routing.roles.{role}.{tool_key}",
                    tool_key=tool_key,
                )

        # 3. Check default routing
        default_config = self._routing.get("default", {})
        if isinstance(default_config, dict) and tool_key in default_config:
            model = default_config[tool_key]
            return ModelSelection(
                model=model,
                source=f"model_routing.default.{tool_key}",
                tool_key=tool_key,
            )

        # 4. Fall back to legacy keys
        model = self._config.get(tool_key)
        return ModelSelection(
            model=model,
            source=f"legacy.{tool_key}" if model else "none",
            tool_key=tool_key,
        )

    def select_codex_models(
        self,
        role: str,
        audit_round: int = 0,
    ) -> tuple[list[str], str]:
        """Select codex_models list for random selection.

        Same precedence as select_model, but returns the list for random choice.

        Args:
            role: Role name
            audit_round: Current audit round

        Returns:
            Tuple of (models_list, source_string)
        """
        # 1. Check audit routing
        if audit_round > 0 and "audit" in self._routing:
            audit_config = self._routing["audit"]
            if isinstance(audit_config, dict) and "codex_models" in audit_config:
                models = audit_config["codex_models"]
                if isinstance(models, list):
                    return models, "model_routing.audit.codex_models"

        # 2. Check role-specific routing
        roles_config = self._routing.get("roles", {})
        if isinstance(roles_config, dict) and role in roles_config:
            role_config = roles_config[role]
            if isinstance(role_config, dict) and "codex_models" in role_config:
                models = role_config["codex_models"]
                if isinstance(models, list):
                    return models, f"model_routing.roles.{role}.codex_models"

        # 3. Check default routing
        default_config = self._routing.get("default", {})
        if isinstance(default_config, dict) and "codex_models" in default_config:
            models = default_config["codex_models"]
            if isinstance(models, list):
                return models, "model_routing.default.codex_models"

        # 4. Fall back to legacy key
        models = self._config.get("codex_models", [])
        return models, "legacy.codex_models" if models else "none"

    @staticmethod
    def _get_tool_key(ai_tool: AiTool) -> str:
        """Map AI tool to config key."""
        return {
            "claude": "claude_model",
            "codex": "codex_model",
            "dasher": "dasher_model",
        }[ai_tool]


@dataclass
class ModelSwitchingPolicy:
    """Policy for handling model changes in resumed sessions.

    Contracts:
        REQUIRES: enabled is bool
        REQUIRES: strategy in ("restart_session", "resume_with_model")
        ENSURES: should_restart_session returns bool
    """

    enabled: bool = False
    allowed_tools: list[str] | None = None
    strategy: Literal["restart_session", "resume_with_model"] = "restart_session"
    preserve_history: bool = False

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "ModelSwitchingPolicy":
        """Create policy from config dict.

        Args:
            config: Configuration dict (may contain model_switching key)

        Returns:
            ModelSwitchingPolicy instance
        """
        switching = config.get("model_switching", {})
        if not isinstance(switching, dict):
            return cls()

        return cls(
            enabled=switching.get("enabled", False),
            allowed_tools=switching.get("allowed_tools"),
            strategy=switching.get("strategy", "restart_session"),
            preserve_history=switching.get("preserve_history", False),
        )

    def should_restart_session(
        self,
        ai_tool: AiTool,
        old_model: str | None,
        new_model: str | None,
    ) -> bool:
        """Check if session should be restarted due to model change.

        Args:
            ai_tool: The AI tool being used
            old_model: Model from previous session (None if new session)
            new_model: Selected model for this iteration

        Returns:
            True if session should restart (drop resume_session_id)
        """
        # No change = no restart
        if old_model == new_model:
            return False

        # Model change but switching disabled = don't restart (pin to old model)
        if not self.enabled:
            return False

        # Switching enabled but tool not allowed
        if self.allowed_tools is not None and ai_tool not in self.allowed_tools:
            return False

        # Strategy determines behavior
        if self.strategy == "restart_session":
            log_info(
                f"Model switch detected ({old_model} -> {new_model}), "
                f"restarting session per model_switching.strategy=restart_session"
            )
            return True

        # resume_with_model: don't restart, pass new model to CLI
        # (This requires CLI support - currently reserved for future)
        return False

    def should_pin_model(
        self,
        ai_tool: AiTool,
        old_model: str | None,
        new_model: str | None,
    ) -> str | None:
        """Get the model to use, pinning to old model if switching disabled.

        Args:
            ai_tool: The AI tool being used
            old_model: Model from previous session
            new_model: Selected model for this iteration

        Returns:
            Model to use (may be old_model if pinned)
        """
        if old_model == new_model:
            return new_model

        # Switching disabled = pin to old model
        if not self.enabled and old_model is not None:
            log_info(
                f"Model routing selected {new_model} but model_switching disabled, "
                f"pinning to session model {old_model}"
            )
            return old_model

        # Tool not allowed for switching
        if (
            self.enabled
            and self.allowed_tools is not None
            and ai_tool not in self.allowed_tools
            and old_model is not None
        ):
            log_info(
                f"Model routing selected {new_model} but {ai_tool} not in "
                f"model_switching.allowed_tools, pinning to {old_model}"
            )
            return old_model

        return new_model
