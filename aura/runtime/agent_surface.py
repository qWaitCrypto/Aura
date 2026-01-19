from __future__ import annotations

from dataclasses import dataclass

from .llm.types import ToolSpec
from .skills import SkillMetadata
from .plan import PlanItem


@dataclass(frozen=True, slots=True)
class SpecStatusSummary:
    status: str  # open|sealed|unknown
    label: str | None = None


def build_agent_surface(
    *,
    tools: list[ToolSpec],
    skills: list[SkillMetadata],
    plan: list[PlanItem],
    spec: SpecStatusSummary,
    max_tool_lines: int = 40,
    max_skill_lines: int = 40,
    max_todo_lines: int = 20,
) -> str:
    tool_names = {t.name for t in tools}
    has_tool_calling = bool(tools)
    has_skill_tools = "skill__list" in tool_names and "skill__load" in tool_names
    has_plan_tool = "update_plan" in tool_names

    tool_lines = []
    for spec_item in tools[: max_tool_lines]:
        tool_lines.append(f"- {spec_item.name}: {spec_item.description}")
    if len(tools) > max_tool_lines:
        tool_lines.append(f"- ... ({len(tools) - max_tool_lines} more)")

    tool_notes: list[str] = []
    if "project__apply_edits" in tool_names:
        tool_notes.extend(
            [
                "- Prefer `project__apply_edits` for normal file edits; it uses structured JSON ops (no patch DSL).",
                "- For `update_file`/`insert_*`/`replace_substring_*`, copy exact lines/substrings via `project__read_text`/`project__search_text` (no guessing).",
            ]
        )
    if "project__patch" in tool_names:
        tool_notes.extend(
            [
                "- Use `project__patch` for patch-style edits; it accepts unified diff (`---/+++`, `@@`) like `git diff`.",
                "- Pass raw diff text (no ``` fences).",
            ]
        )

    skill_lines = []
    for meta in skills[: max_skill_lines]:
        skill_lines.append(f"- {meta.name}: {meta.description}")
    skills_truncated = len(skills) > max_skill_lines
    if skills_truncated:
        suffix = "; call `skill__list`" if has_skill_tools else ""
        skill_lines.append(f"- ... ({len(skills) - max_skill_lines} more{suffix})")

    todo_lines = []
    for idx, item in enumerate(plan[: max_todo_lines], start=1):
        todo_lines.append(f"{idx}. [{item.status.value}] {item.step}")
    if len(plan) > max_todo_lines:
        todo_lines.append(f"... ({len(plan) - max_todo_lines} more)")

    label = f" ({spec.label})" if spec.label else ""

    tools_section = tool_lines if tool_lines else (["- (tool calling disabled)"] if not has_tool_calling else ["- (no tools)"])
    skills_rules = (
        [
            "- Before starting a task, check available skills.",
            "- If a skill is relevant or user names it, call `skill__load` first.",
            "- If the list is truncated, call `skill__list`.",
        ]
        if has_skill_tools
        else ["- (skill tools are not available in this session)"]
    )

    todo_rules = (
        [
            "- Use `update_plan` for tasks with 2+ steps.",
            "- Keep at most one `in_progress`; update promptly; don't batch updates.",
        ]
        if has_plan_tool
        else ["- (plan tool is not available in this session)"]
    )

    return "\n".join(
        [
            "# Aura Agent Surface (v0.2)",
            "",
            "## Tools",
            *tools_section,
            *([] if not tool_notes else ["", "Notes:", *tool_notes]),
            "",
            "## Skills",
            "Rules:",
            *skills_rules,
            "",
            "<available_skills>",
            *(skill_lines if skill_lines else ["- (no skills found)"]),
            "</available_skills>",
            "",
            "## Todo",
            "Rules:",
            *todo_rules,
            "",
            "Current plan:",
            *(todo_lines if todo_lines else ["(none)"]),
            "",
            "## Spec",
            f"Status: {spec.status}{label}",
            "Rules:",
            "- Do not modify `spec/` via generic file tools when sealed; use spec workflow tools.",
        ]
    )
