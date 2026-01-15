from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar

from ..llm.router import ModelRouter
from ..stores import ArtifactStore
from ..subagents import get_preset, run_subagent
from ..subagents.presets import preset_input_schema
from .registry import ToolRegistry
from .runtime import ToolExecutionContext, ToolRuntime


@dataclass(slots=True)
class SubagentRunTool:
    model_router: ModelRouter
    tool_registry: ToolRegistry
    tool_runtime: ToolRuntime
    artifact_store: ArtifactStore

    name: ClassVar[str] = "subagent__run"
    description: ClassVar[str] = (
        "Run a bounded delegated task in an isolated subagent context. "
        "Use for verification (preset=verifier) or for executing a small tool chain with receipts (preset=tool_interpreter). "
        "The runner enforces a per-run tool allowlist, prevents recursion, and never nests interactive approvals: "
        "if an approval-required tool is requested, it stops and returns an actionable report."
    )
    input_schema: ClassVar[dict[str, Any]] = {
        "type": "object",
        "properties": {
            "preset": preset_input_schema(),
            "task": {"type": "string", "description": "Delegated task instruction."},
            "context": {
                "type": "object",
                "description": "Optional extra context passed to the subagent.",
                "properties": {
                    "text": {"type": "string", "description": "Extra context text."},
                    "files": {
                        "type": "array",
                        "description": "Optional file hints; the subagent may read these via project tools if needed.",
                        "items": {
                            "anyOf": [
                                {"type": "string"},
                                {
                                    "type": "object",
                                    "properties": {
                                        "path": {"type": "string"},
                                        "max_chars": {"type": "integer", "minimum": 1},
                                    },
                                    "required": ["path"],
                                    "additionalProperties": False,
                                },
                            ]
                        },
                    },
                },
                "additionalProperties": False,
            },
            "tool_allowlist": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Optional override allowlist (tool names or glob patterns).",
            },
            "max_turns": {"type": "integer", "minimum": 1, "maximum": 50, "description": "Max internal turns."},
            "max_tool_calls": {
                "type": "integer",
                "minimum": 1,
                "maximum": 200,
                "description": "Max tool calls executed inside the subagent.",
            },
        },
        "required": ["preset", "task"],
        "additionalProperties": False,
    }

    def execute(self, *, args: dict[str, Any], project_root, context: ToolExecutionContext | None = None) -> dict[str, Any]:
        preset_name = str(args.get("preset") or "").strip()
        preset = get_preset(preset_name)
        if preset is None:
            raise ValueError(f"Unknown preset: {preset_name!r}")

        task = str(args.get("task") or "").strip()
        if not task:
            raise ValueError("Missing or invalid 'task' (expected non-empty string).")

        allowlist = args.get("tool_allowlist")
        if allowlist is None:
            allowlist_patterns = list(preset.default_allowlist)
        else:
            if not isinstance(allowlist, list) or any(not isinstance(x, str) for x in allowlist):
                raise ValueError("Invalid 'tool_allowlist' (expected list of strings).")
            allowlist_patterns = [x.strip() for x in allowlist if x.strip()]

        max_turns = args.get("max_turns")
        if max_turns is None:
            max_turns_int = preset.limits.max_turns
        else:
            if isinstance(max_turns, bool) or not isinstance(max_turns, int) or max_turns < 1:
                raise ValueError("Invalid 'max_turns' (expected integer >= 1).")
            max_turns_int = min(int(max_turns), 50)

        max_tool_calls = args.get("max_tool_calls")
        if max_tool_calls is None:
            max_tool_calls_int = preset.limits.max_tool_calls
        else:
            if isinstance(max_tool_calls, bool) or not isinstance(max_tool_calls, int) or max_tool_calls < 1:
                raise ValueError("Invalid 'max_tool_calls' (expected integer >= 1).")
            max_tool_calls_int = min(int(max_tool_calls), 200)

        return run_subagent(
            preset=preset,
            task=task,
            extra_context=args.get("context"),
            tool_allowlist=allowlist_patterns,
            max_turns=max_turns_int,
            max_tool_calls=max_tool_calls_int,
            model_router=self.model_router,
            tool_registry=self.tool_registry,
            tool_runtime=self.tool_runtime,
            artifact_store=self.artifact_store,
            project_root=project_root,
            exec_context=context,
        )
