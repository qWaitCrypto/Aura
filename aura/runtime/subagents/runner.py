from __future__ import annotations

import fnmatch
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..ids import new_id, now_ts_ms
from ..llm.errors import ModelResolutionError
from ..llm.router import ModelRouter
from ..llm.types import ModelRequirements, ModelRole, ToolSpec
from ..orchestrator_helpers import _summarize_text, _summarize_tool_for_ui
from ..protocol import Event, EventKind
from ..stores import ArtifactStore
from ..tools.runtime import InspectionDecision, ToolExecutionContext, ToolRuntime
from .presets import SubagentPreset


@dataclass(frozen=True, slots=True)
class SubagentReceipt:
    tool_execution_id: str | None
    tool_name: str
    tool_call_id: str
    status: str
    duration_ms: int | None
    summary: str | None
    output_ref: dict[str, Any] | None
    tool_message_ref: dict[str, Any] | None
    error_code: str | None
    error: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "tool_execution_id": self.tool_execution_id,
            "tool_name": self.tool_name,
            "tool_call_id": self.tool_call_id,
            "status": self.status,
            "duration_ms": self.duration_ms,
            "summary": self.summary,
            "output_ref": self.output_ref,
            "tool_message_ref": self.tool_message_ref,
            "error_code": self.error_code,
            "error": self.error,
        }


def _tool_allowed(tool_name: str, patterns: list[str]) -> bool:
    if not patterns:
        return False
    for pat in patterns:
        if not isinstance(pat, str):
            continue
        pat = pat.strip()
        if not pat:
            continue
        if pat == tool_name:
            return True
        if fnmatch.fnmatch(tool_name, pat):
            return True
    return False


def _filter_tool_specs(*, tool_specs: list[ToolSpec], allowlist: list[str]) -> list[ToolSpec]:
    out: list[ToolSpec] = []
    for spec in tool_specs:
        if spec.name == "subagent__run":
            continue
        if _tool_allowed(spec.name, allowlist):
            out.append(spec)
    return out


def _json_or_text(text: str) -> Any:
    try:
        return json.loads(text)
    except Exception:
        return text


def _context_to_text(extra_context: Any) -> str:
    if not isinstance(extra_context, dict):
        return ""
    parts: list[str] = []

    text = extra_context.get("text")
    if isinstance(text, str) and text.strip():
        parts.append(text.strip())

    files = extra_context.get("files")
    if isinstance(files, list) and files:
        lines: list[str] = ["File hints:"]
        for item in files:
            if isinstance(item, str) and item.strip():
                lines.append(f"- {item.strip()}")
                continue
            if isinstance(item, dict):
                path = item.get("path")
                max_chars = item.get("max_chars")
                if isinstance(path, str) and path.strip():
                    suffix = ""
                    if isinstance(max_chars, int) and not isinstance(max_chars, bool) and max_chars > 0:
                        suffix = f" (max_chars={max_chars})"
                    lines.append(f"- {path.strip()}{suffix}")
        if len(lines) > 1:
            parts.append("\n".join(lines))

    return "\n\n".join(parts).strip()


def _extract_text(out: Any) -> str:
    content = getattr(out, "content", None)
    if isinstance(content, str):
        return content

    messages = getattr(out, "messages", None)
    if isinstance(messages, list):
        for m in reversed(messages):
            if getattr(m, "role", None) == "assistant":
                if hasattr(m, "get_content_string"):
                    text = m.get_content_string()
                    if isinstance(text, str):
                        return text
                raw = getattr(m, "content", None)
                if isinstance(raw, str):
                    return raw

    if content is None:
        return ""
    return str(content)


def _update_system_message_in_run(out: Any, *, system_message: str) -> None:
    messages = getattr(out, "messages", None)
    if not isinstance(messages, list):
        return
    for m in messages:
        if getattr(m, "role", None) == "system":
            try:
                m.content = system_message
            except Exception:
                pass
            return


def _run_output_exceeded_tool_call_limit(out: Any) -> bool:
    messages = getattr(out, "messages", None)
    if not isinstance(messages, list):
        return False
    for m in messages:
        if getattr(m, "role", None) != "tool":
            continue
        content = getattr(m, "content", None)
        if isinstance(content, str) and content.startswith("Tool call limit reached."):
            return True
    return False


def run_subagent(
    *,
    preset: SubagentPreset,
    task: str,
    extra_context: Any,
    tool_allowlist: list[str],
    max_turns: int,
    max_tool_calls: int,
    model_router: ModelRouter,
    tool_registry: Any,
    tool_runtime: ToolRuntime,
    artifact_store: ArtifactStore,
    project_root: Path,
    exec_context: ToolExecutionContext | None,
) -> dict[str, Any]:
    """
    Run an isolated delegated task using agno Agent (no Team).

    This runner:
    - Restricts tools via a per-run allowlist (glob patterns).
    - Uses Aura ToolRuntime inspection for deny/approval decisions.
    - Never nests interactive approvals: if a tool needs approval, stop and return a report.
    """

    project_root = Path(project_root).expanduser().resolve()
    subagent_run_id = new_id("subag")

    session_id = exec_context.session_id if exec_context is not None else ""
    request_id = exec_context.request_id if exec_context is not None else None
    turn_id = exec_context.turn_id if exec_context is not None else None
    event_bus = exec_context.event_bus if exec_context is not None else None

    receipts: list[SubagentReceipt] = []
    executed_tool_calls = 0
    needs_approval: dict[str, Any] | None = None

    def _emit_event(*, kind: EventKind, payload: dict[str, Any]) -> None:
        nonlocal executed_tool_calls
        payload = dict(payload)
        payload.setdefault("subagent_run_id", subagent_run_id)
        payload.setdefault("preset", preset.name)

        if kind is EventKind.TOOL_CALL_END:
            executed_tool_calls += 1
            tool_name = payload.get("tool_name")
            tool_call_id = payload.get("tool_call_id")
            if isinstance(tool_name, str) and tool_name and isinstance(tool_call_id, str) and tool_call_id:
                receipts.append(
                    SubagentReceipt(
                        tool_execution_id=payload.get("tool_execution_id") if isinstance(payload.get("tool_execution_id"), str) else None,
                        tool_name=tool_name,
                        tool_call_id=tool_call_id,
                        status=str(payload.get("status") or "unknown"),
                        duration_ms=int(payload["duration_ms"]) if isinstance(payload.get("duration_ms"), int) else None,
                        summary=payload.get("summary") if isinstance(payload.get("summary"), str) else None,
                        output_ref=payload.get("output_ref") if isinstance(payload.get("output_ref"), dict) else None,
                        tool_message_ref=payload.get("tool_message_ref") if isinstance(payload.get("tool_message_ref"), dict) else None,
                        error_code=payload.get("error_code") if isinstance(payload.get("error_code"), str) else None,
                        error=payload.get("error") if isinstance(payload.get("error"), str) else None,
                    )
                )

        if event_bus is None:
            return
        step_id = payload.get("tool_execution_id")
        event = Event(
            kind=kind.value,
            payload=payload,
            session_id=session_id,
            event_id=new_id("evt"),
            timestamp=now_ts_ms(),
            request_id=request_id,
            turn_id=turn_id,
            step_id=step_id if isinstance(step_id, str) else None,
            schema_version="0.1",
        )
        event_bus.publish(event)

    # Resolve model profile for the subagent.
    requirements = ModelRequirements(needs_tools=True)
    selected_role = ModelRole.SUBAGENT
    selected_profile_id: str | None = None
    fallback_used = False
    try:
        resolved = model_router.resolve(role=ModelRole.SUBAGENT, requirements=requirements)
        selected_profile_id = getattr(resolved.profile, "profile_id", None)
    except ModelResolutionError:
        fallback_used = True
        selected_role = ModelRole.MAIN
        resolved = model_router.resolve(role=ModelRole.MAIN, requirements=requirements)
        selected_profile_id = getattr(resolved.profile, "profile_id", None)

    # Build agno model (Aura-backed).
    try:
        from ..llm.agno_aura_model import build_aura_agno_model
    except Exception as e:  # pragma: no cover - optional dependency
        raise RuntimeError(f"agno model adapter unavailable: {e}") from e

    agno_model = build_aura_agno_model(profile=resolved.profile, project_root=project_root, session_id=session_id)

    # Build toolset for the delegated run.
    all_specs = tool_registry.list_specs() if hasattr(tool_registry, "list_specs") else []
    filtered_specs = _filter_tool_specs(tool_specs=list(all_specs), allowlist=list(tool_allowlist))

    from ..agno_tools import build_agno_toolset

    # Keep subagent tool messages isolated from the parent Aura chat history.
    subagent_tool_messages: list[Any] = []

    toolset = build_agno_toolset(
        tool_specs=filtered_specs,
        tool_runtime=tool_runtime,
        emit=_emit_event,
        append_history=subagent_tool_messages.append,
        event_bus=event_bus,
    )

    # Construct system and user messages.
    allowlisted_names = [s.name for s in filtered_specs]
    allowlist_block = "\n".join(f"- {n}" for n in allowlisted_names) if allowlisted_names else "- (none)"
    system_message = preset.load_prompt().rstrip() + (
        "\n\nAllowed tools (enforced by runner):\n" + allowlist_block + "\n"
    )
    context_text = _context_to_text(extra_context)
    user_text = task.strip()
    if context_text:
        user_text = user_text + "\n\nContext:\n" + context_text

    try:
        from agno.agent.agent import Agent as AgnoAgent
        from agno.models.message import Message as AgnoMessage
        from agno.models.response import ToolExecution as AgnoToolExecution
        from agno.run.requirement import RunRequirement as AgnoRunRequirement
    except Exception as e:  # pragma: no cover - optional dependency
        raise RuntimeError(f"Agno is not available in this environment: {e}") from e

    agent = AgnoAgent(
        name=f"AuraSubagent:{preset.name}",
        model=agno_model,
        system_message=system_message,
        db=None,
        tools=toolset.functions,
        tool_call_limit=max_tool_calls,
        stream=False,
        stream_events=False,
    )

    metadata = {"aura_request_id": request_id, "aura_turn_id": turn_id, "aura_subagent_run_id": subagent_run_id}
    input_messages = [AgnoMessage(role="user", content=user_text)]

    try:
        out = agent.run(
            input_messages,
            stream=False,
            session_id=session_id,
            metadata=metadata,
            add_history_to_context=False,
        )
    except KeyboardInterrupt:
        out = None
        assistant_text = ""
        status = "failed"
        report: Any = {"ok": False, "error": "cancelled"}
    except Exception as e:
        out = None
        assistant_text = ""
        status = "failed"
        report = {"ok": False, "error": str(e)}
    else:
        assistant_text = _extract_text(out)
        report = _json_or_text(assistant_text)
        status = "completed"

    # Handle agno confirmation pauses. Aura decides allow/deny/approval; subagent never creates approvals.
    pause_guard = 0
    while out is not None and getattr(out, "is_paused", False):
        pause_guard += 1
        if pause_guard > max(4, max_turns):
            status = "failed"
            report = {
                "ok": False,
                "error": "Exceeded pause/resume limit while processing tool confirmations.",
                "error_code": "max_turns_exceeded",
            }
            break

        tools = getattr(out, "tools", None) or []
        paused = [t for t in tools if getattr(t, "requires_confirmation", False)]
        if not paused:
            status = "failed"
            report = {
                "ok": False,
                "error": "Run paused for unsupported requirement type (user_input/external_execution).",
                "error_code": "unknown_request",
            }
            break

        tool = paused[0]
        tool_call_id = getattr(tool, "tool_call_id", None)
        tool_name = getattr(tool, "tool_name", None)
        tool_args = getattr(tool, "tool_args", None)
        if not isinstance(tool_call_id, str) or not tool_call_id:
            tool_call_id = new_id("call")
        if not isinstance(tool_name, str) or not tool_name:
            tool_name = "unknown"
        if not isinstance(tool_args, dict):
            tool_args = {}

        planned = tool_runtime.plan(
            tool_execution_id=f"tool_{tool_call_id}",
            tool_name=tool_name,
            tool_call_id=tool_call_id,
            arguments=dict(tool_args),
        )
        inspection = tool_runtime.inspect(planned)

        if inspection.decision is InspectionDecision.REQUIRE_APPROVAL:
            needs_approval = {
                "tool_execution_id": planned.tool_execution_id,
                "tool_name": planned.tool_name,
                "tool_call_id": planned.tool_call_id,
                "arguments_ref": planned.arguments_ref.to_dict(),
                "summary": _summarize_tool_for_ui(planned.tool_name, planned.arguments),
                "action_summary": inspection.action_summary,
                "risk_level": inspection.risk_level or "high",
                "reason": inspection.reason,
            }
            _emit_event(
                kind=EventKind.TOOL_CALL_END,
                payload={
                    "tool_execution_id": planned.tool_execution_id,
                    "tool_name": planned.tool_name,
                    "tool_call_id": planned.tool_call_id,
                    "summary": _summarize_tool_for_ui(planned.tool_name, planned.arguments),
                    "status": "needs_approval",
                    "duration_ms": 0,
                    "output_ref": None,
                    "tool_message_ref": None,
                    "error_code": None,
                    "error": inspection.reason or inspection.action_summary,
                },
            )
            status = "needs_approval"
            report = {
                "ok": False,
                "status": "needs_approval",
                "needs_approval": [needs_approval],
                "summary": "Subagent requested a tool that requires user approval.",
            }
            break

        if inspection.decision is InspectionDecision.ALLOW:
            updated_tool = AgnoToolExecution(
                tool_call_id=tool_call_id,
                tool_name=tool_name,
                tool_args=tool_args,
                requires_confirmation=True,
                confirmed=True,
            )
            req = AgnoRunRequirement(tool_execution=updated_tool)
            _update_system_message_in_run(out, system_message=system_message)
            out = agent.continue_run(
                run_response=out,
                requirements=[req],
                stream=False,
                session_id=session_id,
                metadata=metadata,
            )
            assistant_text = _extract_text(out)
            report = _json_or_text(assistant_text)
            continue

        # DENY
        _emit_event(
            kind=EventKind.TOOL_CALL_END,
            payload={
                "tool_execution_id": planned.tool_execution_id,
                "tool_name": planned.tool_name,
                "tool_call_id": planned.tool_call_id,
                "summary": _summarize_tool_for_ui(planned.tool_name, planned.arguments),
                "status": "denied",
                "duration_ms": 0,
                "output_ref": None,
                "tool_message_ref": None,
                "error_code": (inspection.error_code.value if inspection.error_code is not None else None),
                "error": inspection.reason or inspection.action_summary,
            },
        )
        updated_tool = AgnoToolExecution(
            tool_call_id=tool_call_id,
            tool_name=tool_name,
            tool_args=tool_args,
            requires_confirmation=True,
            confirmed=False,
            confirmation_note=inspection.reason or inspection.action_summary,
        )
        req = AgnoRunRequirement(tool_execution=updated_tool)
        _update_system_message_in_run(out, system_message=system_message)
        out = agent.continue_run(
            run_response=out,
            requirements=[req],
            stream=False,
            session_id=session_id,
            metadata=metadata,
        )
        assistant_text = _extract_text(out)
        report = _json_or_text(assistant_text)

    error_code: str | None = None
    if status == "completed" and out is not None and _run_output_exceeded_tool_call_limit(out):
        status = "failed"
        error_code = "max_tool_calls_exceeded"
        report = {
            "ok": False,
            "status": "failed",
            "error_code": error_code,
            "summary": "Subagent exceeded max_tool_calls.",
        }

    # Best-effort turn counting (agno does not expose a direct max_turns guard).
    if status == "completed" and out is not None:
        messages = getattr(out, "messages", None) or []
        turns = len([m for m in messages if getattr(m, "role", None) == "assistant"])
        if isinstance(turns, int) and turns > max_turns:
            status = "failed"
            error_code = "max_turns_exceeded"
            report = {
                "ok": False,
                "status": "failed",
                "error_code": error_code,
                "summary": "Subagent exceeded max_turns.",
            }

    transcript = {
        "subagent_run_id": subagent_run_id,
        "preset": preset.name,
        "status": status,
        "selected_role": selected_role.value,
        "selected_profile_id": selected_profile_id,
        "fallback_used": fallback_used,
        "tool_allowlist": list(tool_allowlist),
        "limits": {"max_turns": int(max_turns), "max_tool_calls": int(max_tool_calls)},
        "executed_tool_calls": executed_tool_calls,
        "receipts": [r.to_dict() for r in receipts],
        "report": report,
        "assistant_text": assistant_text,
        "needs_approval": [needs_approval] if needs_approval is not None else [],
        "error_code": error_code,
    }
    transcript_ref = artifact_store.put(
        json.dumps(transcript, ensure_ascii=False, sort_keys=True, indent=2),
        kind="subagent_transcript",
        meta={"summary": f"Subagent transcript ({preset.name})", "text_summary": _summarize_text(assistant_text)},
    )

    return {
        "ok": status == "completed",
        "subagent_run_id": subagent_run_id,
        "preset": preset.name,
        "status": status,
        "selected_role": selected_role.value,
        "selected_profile_id": selected_profile_id,
        "fallback_used": fallback_used,
        "tool_allowlist": list(tool_allowlist),
        "limits": {"max_turns": int(max_turns), "max_tool_calls": int(max_tool_calls)},
        "executed_tool_calls": executed_tool_calls,
        "receipts": [r.to_dict() for r in receipts],
        "report": report,
        "transcript_ref": transcript_ref.to_dict(),
    }
