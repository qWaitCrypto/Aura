from __future__ import annotations

import importlib.resources
from dataclasses import dataclass
from typing import Any


def load_prompt_asset(name: str) -> str:
    return (
        importlib.resources.files("aura.runtime")
        .joinpath("prompts", name)
        .read_text(encoding="utf-8", errors="replace")
    )


@dataclass(frozen=True, slots=True)
class SubagentLimits:
    max_turns: int
    max_tool_calls: int


@dataclass(frozen=True, slots=True)
class SubagentPreset:
    name: str
    prompt_asset: str
    default_allowlist: list[str]
    limits: SubagentLimits

    def load_prompt(self) -> str:
        return load_prompt_asset(self.prompt_asset)


_PRESETS: dict[str, SubagentPreset] = {
    "verifier": SubagentPreset(
        name="verifier",
        prompt_asset="subagent_verifier.md",
        default_allowlist=[
            "project__list_dir",
            "project__glob",
            "project__read_text",
            "project__read_text_many",
            "project__search_text",
            "spec__query",
            "spec__get",
            "snapshot__read_text",
            "session__search",
        ],
        limits=SubagentLimits(max_turns=6, max_tool_calls=12),
    ),
    "tool_interpreter": SubagentPreset(
        name="tool_interpreter",
        prompt_asset="subagent_tool_interpreter.md",
        default_allowlist=[
            "project__list_dir",
            "project__glob",
            "project__read_text",
            "project__read_text_many",
            "project__search_text",
            "project__apply_edits",
            "project__text_stats",
            "spec__query",
            "spec__get",
            "spec__propose",
            "snapshot__list",
            "snapshot__create",
            "snapshot__diff",
            "snapshot__read_text",
            "session__search",
        ],
        limits=SubagentLimits(max_turns=12, max_tool_calls=24),
    ),
}


def get_preset(name: str) -> SubagentPreset | None:
    return _PRESETS.get(str(name or "").strip())


def list_presets() -> list[str]:
    return sorted(_PRESETS)


def preset_input_schema() -> dict[str, Any]:
    return {
        "type": "string",
        "enum": list_presets(),
        "description": "Subagent preset.",
    }

