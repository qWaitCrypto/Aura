from __future__ import annotations

import json
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

from ..event_bus import EventBus

from ..protocol import ArtifactRef
from ..error_codes import ErrorCode

from ..stores import ArtifactStore
from .registry import ToolRegistry


class ToolRuntimeError(RuntimeError):
    pass


def _elide_tail(s: str, max_chars: int) -> str:
    if max_chars <= 0:
        return ""
    if len(s) <= max_chars:
        return s
    return s[: max(0, max_chars - 1)].rstrip() + "…"


def _summarize_shell_run_args(args: dict[str, Any], *, max_chars: int = 120) -> str:
    command = args.get("command")
    if not isinstance(command, str) or not command.strip():
        return "Run shell command"
    one_line = " ".join(command.splitlines()).strip()
    one_line = " ".join(one_line.split())
    one_line = _elide_tail(one_line, max_chars)
    return f"Run $ {one_line}"


class InspectionDecision(StrEnum):
    ALLOW = "allow"
    DENY = "deny"
    REQUIRE_APPROVAL = "require_approval"


class ToolApprovalMode(StrEnum):
    """
    Tool approval policy for a session.

    - strict: require approval for every tool call (including reads/search).
    - standard: require approval only for high-risk tool calls (default).
    - trusted: never require approval (dangerous).
    """

    STRICT = "strict"
    STANDARD = "standard"
    TRUSTED = "trusted"


@dataclass(frozen=True, slots=True)
class InspectionResult:
    decision: InspectionDecision
    action_summary: str
    risk_level: str | None = None
    reason: str | None = None
    error_code: ErrorCode | None = None
    diff_ref: ArtifactRef | None = None


@dataclass(frozen=True, slots=True)
class PlannedToolCall:
    tool_execution_id: str
    tool_call_id: str
    tool_name: str
    arguments: dict[str, Any]
    arguments_ref: ArtifactRef


@dataclass(frozen=True, slots=True)
class ToolExecutionContext:
    """
    Optional execution context passed to tools.

    Most tools ignore this. It's used by complex tools (e.g. subagents) that
    need to emit progress events and attach diagnostics to the current request.
    """

    session_id: str
    request_id: str | None
    turn_id: str | None
    tool_execution_id: str
    event_bus: EventBus | None = None


class ToolRuntime:
    def __init__(
        self,
        *,
        project_root: Path,
        registry: ToolRegistry,
        artifact_store: ArtifactStore,
        approval_mode: ToolApprovalMode = ToolApprovalMode.STANDARD,
    ) -> None:
        self._project_root = project_root.expanduser().resolve()
        self._registry = registry
        self._artifact_store = artifact_store
        self._approval_mode = approval_mode

    def set_approval_mode(self, mode: ToolApprovalMode) -> None:
        self._approval_mode = mode

    def get_approval_mode(self) -> ToolApprovalMode:
        return self._approval_mode

    @property
    def project_root(self) -> Path:
        return self._project_root

    @property
    def artifact_store(self) -> ArtifactStore:
        return self._artifact_store

    def get_tool(self, tool_name: str):
        return self._registry.get(tool_name)

    def plan(self, *, tool_execution_id: str, tool_name: str, tool_call_id: str, arguments: dict[str, Any]) -> PlannedToolCall:
        if not tool_call_id:
            raise ToolRuntimeError("Tool call is missing tool_call_id; cannot return tool_result.")
        if not tool_name:
            raise ToolRuntimeError("Tool call is missing tool name.")
        if not isinstance(arguments, dict):
            raise ToolRuntimeError("Tool call arguments must be an object.")

        args_ref = self._artifact_store.put(
            json.dumps(arguments, ensure_ascii=False, sort_keys=True, indent=2),
            kind="tool_args",
            meta={"summary": f"{tool_name} args"},
        )
        return PlannedToolCall(
            tool_execution_id=tool_execution_id,
            tool_call_id=tool_call_id,
            tool_name=tool_name,
            arguments=arguments,
            arguments_ref=args_ref,
        )

    def inspect(self, planned: PlannedToolCall) -> InspectionResult:
        tool = self._registry.get(planned.tool_name)
        if tool is None:
            return InspectionResult(
                decision=InspectionDecision.DENY,
                action_summary=f"Unknown tool: {planned.tool_name}",
                risk_level="high",
                reason="Tool is not registered.",
                error_code=ErrorCode.TOOL_UNKNOWN,
            )

        # Domain invariants / mandatory approvals (override all modes).
        if planned.tool_name == "snapshot__rollback":
            target = planned.arguments.get("target")
            create_backup = planned.arguments.get("create_backup")
            backup_label = planned.arguments.get("backup_label")
            preview = (
                "Rollback project files to an internal snapshot ref.\n\n"
                f"Target: {target}\n"
                f"Create backup: {create_backup}\n"
                f"Backup label: {backup_label}\n\n"
                "WARNING: This operation overwrites files in the project working tree.\n"
                "Approval is required.\n"
            )
            diff_ref = self._artifact_store.put(preview, kind="diff", meta={"summary": "Snapshot rollback preview"})
            return InspectionResult(
                decision=InspectionDecision.REQUIRE_APPROVAL,
                action_summary=f"Rollback snapshot: {target}",
                risk_level="high",
                reason="Rolling back overwrites project files and may discard uncommitted changes.",
                error_code=None,
                diff_ref=diff_ref,
            )
        sealed = self._is_spec_sealed()
        sealed_violation = self._check_sealed_spec_violation(planned, sealed=sealed)
        if sealed_violation is not None:
            return sealed_violation

        if planned.tool_name in {"spec__apply", "spec__seal"}:
            return self._inspect_spec_workflow(planned)

        if planned.tool_name == "shell__run" and _shell_run_is_allowlisted(self._project_root, planned.arguments):
            summary = _summarize_shell_run_args(planned.arguments)
            return InspectionResult(
                decision=InspectionDecision.ALLOW,
                action_summary=f"{summary} (allowlisted)",
                risk_level="high",
                reason="Matched local allowlist.",
                error_code=None,
                diff_ref=None,
            )

        if self._approval_mode is ToolApprovalMode.TRUSTED:
            return InspectionResult(
                decision=InspectionDecision.ALLOW,
                action_summary=f"Execute tool: {planned.tool_name}",
                risk_level="high",
                reason="Approval mode is trusted (auto-allow).",
                error_code=None,
                diff_ref=None,
            )

        if self._approval_mode is ToolApprovalMode.STRICT:
            return self._inspect_strict(planned)

        tool_name = planned.tool_name

        if tool_name in {"web__fetch", "web__search"}:
            diff_ref = self._build_args_preview(planned, summary=f"Preview for {tool_name}")
            return InspectionResult(
                decision=InspectionDecision.REQUIRE_APPROVAL,
                action_summary=f"Execute tool: {tool_name}",
                risk_level="high",
                reason="Web access can exfiltrate data and may be unsafe; approval required in standard mode.",
                error_code=None,
                diff_ref=diff_ref,
            )

        if tool_name == "session__export":
            diff_ref = self._build_args_preview(planned, summary="Preview for session__export (bundle output)")
            return InspectionResult(
                decision=InspectionDecision.REQUIRE_APPROVAL,
                action_summary=f"Export session: {planned.arguments.get('session_id')}",
                risk_level="high",
                reason="Export writes files into the project; approval required in standard mode.",
                error_code=None,
                diff_ref=diff_ref,
            )

        if planned.tool_name == "shell__run":
            summary = _summarize_shell_run_args(planned.arguments)
            try:
                diff_ref = self._build_shell_run_preview(planned)
            except Exception as e:
                code = _classify_tool_exception(e)
                return InspectionResult(
                    decision=InspectionDecision.DENY,
                    action_summary="Invalid shell command request.",
                    risk_level="high",
                    reason=str(e),
                    error_code=code,
                    diff_ref=None,
                )
            return InspectionResult(
                decision=InspectionDecision.REQUIRE_APPROVAL,
                action_summary=summary,
                risk_level="high",
                reason="Shell commands can modify files and system state.",
                error_code=None,
                diff_ref=diff_ref,
            )

        return InspectionResult(
            decision=InspectionDecision.ALLOW,
            action_summary=f"Execute tool: {planned.tool_name}",
            risk_level="low",
            reason=None,
            error_code=None,
            diff_ref=None,
        )

    def _is_spec_sealed(self) -> bool:
        import json

        path = self._project_root / ".aura" / "state" / "spec_state.json"
        if not path.exists():
            return False
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return False
        if not isinstance(raw, dict):
            return False
        return str(raw.get("status") or "") == "sealed"

    def _check_sealed_spec_violation(self, planned: PlannedToolCall, *, sealed: bool) -> InspectionResult | None:
        if not sealed:
            return None
        tool_name = planned.tool_name
        if tool_name in {"project__apply_patch", "project__apply_edits"}:
            try:
                if tool_name == "project__apply_patch":
                    patch_text = planned.arguments.get("patch")
                    if isinstance(patch_text, str) and patch_text.strip():
                        from .apply_patch_tool import list_patch_target_paths

                        targets = list_patch_target_paths(patch_text)
                    else:
                        targets = []
                else:
                    from .apply_edits_tool import list_apply_edits_target_paths

                    targets = list_apply_edits_target_paths(planned.arguments)

                for p in targets:
                    if _is_under_spec_dir(p):
                        return InspectionResult(
                            decision=InspectionDecision.DENY,
                            action_summary="Blocked edit touching sealed spec/",
                            risk_level="high",
                            reason="Spec is sealed; do not modify spec/ via generic file tools. Use spec workflow tools.",
                            error_code=ErrorCode.PERMISSION,
                            diff_ref=None,
                        )
            except Exception:
                # If args are invalid we let the tool fail later with a clearer error.
                pass
        return None

    def _inspect_spec_workflow(self, planned: PlannedToolCall) -> InspectionResult:
        tool_name = planned.tool_name
        if tool_name == "spec__apply":
            proposal_id = planned.arguments.get("proposal_id")
            diff_ref = None
            reason = None
            if isinstance(proposal_id, str) and proposal_id:
                try:
                    record = _load_spec_proposal_record(self._project_root, proposal_id)
                    reason_raw = record.get("reason")
                    if isinstance(reason_raw, str) and reason_raw.strip():
                        reason = reason_raw.strip()
                    raw_ref = record.get("diff_ref")
                    if isinstance(raw_ref, dict):
                        diff_ref = ArtifactRef.from_dict(raw_ref)
                except Exception:
                    diff_ref = self._build_args_preview(planned, summary="Preview for spec__apply (proposal diff unavailable)")
            else:
                diff_ref = self._build_args_preview(planned, summary="Preview for spec__apply (missing proposal_id)")
            return InspectionResult(
                decision=InspectionDecision.REQUIRE_APPROVAL,
                action_summary=f"Apply spec proposal: {proposal_id}",
                risk_level="high",
                reason=reason or "Applying a spec proposal modifies author-visible spec/ files.",
                error_code=None,
                diff_ref=diff_ref,
            )

        if tool_name == "spec__seal":
            label = planned.arguments.get("label")
            preview = (
                "Seal spec into a version label.\n\n"
                "This will create an internal snapshot (git) and mark spec as sealed.\n"
                "Default snapshot exclusions:\n"
                "- .aura/state/git\n"
                "- .aura/cache\n"
                "- .aura/index\n"
                "- .aura/tmp\n"
                "- .aura/events\n"
                "- .aura/sessions\n"
                "- .aura/artifacts\n"
            )
            if isinstance(label, str) and label.strip():
                preview = f"Label: {label.strip()}\n\n" + preview
            diff_ref = self._artifact_store.put(preview, kind="diff", meta={"summary": "Spec seal preview"})
            return InspectionResult(
                decision=InspectionDecision.REQUIRE_APPROVAL,
                action_summary=f"Seal spec: {label}",
                risk_level="high",
                reason="Sealing spec creates a version point and makes spec read-only.",
                error_code=None,
                diff_ref=diff_ref,
            )

        return InspectionResult(
            decision=InspectionDecision.DENY,
            action_summary=f"Unknown spec workflow tool: {tool_name}",
            risk_level="high",
            reason="Unsupported spec workflow operation.",
            error_code=ErrorCode.TOOL_UNKNOWN,
            diff_ref=None,
        )

    def _build_shell_run_preview(self, planned: PlannedToolCall) -> ArtifactRef:
        command = planned.arguments.get("command")
        cwd = planned.arguments.get("cwd") or "."
        timeout_s = planned.arguments.get("timeout_s")
        if not isinstance(command, str) or not command.strip():
            raise ValueError("shell__run: missing command")
        preview = f"$ {command}\n(cwd: {cwd})\n(timeout_s: {timeout_s})"
        return self._artifact_store.put(
            preview,
            kind="diff",
            meta={"summary": "Shell command preview"},
        )

    def _build_apply_patch_preview(self, planned: PlannedToolCall) -> ArtifactRef:
        patch_text = planned.arguments.get("patch")
        if not isinstance(patch_text, str) or not patch_text.strip():
            raise ValueError("project__apply_patch: missing patch")
        # The patch is itself a diff-like artifact; store it for review (bounded).
        max_chars = 20000
        preview = patch_text.strip()
        if len(preview) > max_chars:
            preview = preview[:max_chars] + "\n…(truncated)"
        return self._artifact_store.put(preview, kind="diff", meta={"summary": "Patch preview"})

    def _inspect_strict(self, planned: PlannedToolCall) -> InspectionResult:
        tool_name = planned.tool_name

        # High-risk tools: try to produce a meaningful diff/preview.
        if tool_name == "project__apply_patch":
            try:
                diff_ref = self._build_apply_patch_preview(planned)
            except Exception as e:
                code = _classify_tool_exception(e)
                return InspectionResult(
                    decision=InspectionDecision.DENY,
                    action_summary="Invalid patch request.",
                    risk_level="high",
                    reason=str(e),
                    error_code=code,
                    diff_ref=None,
                )
            return InspectionResult(
                decision=InspectionDecision.REQUIRE_APPROVAL,
                action_summary="Apply patch",
                risk_level="high",
                reason="Strict mode: approve every tool call.",
                error_code=None,
                diff_ref=diff_ref,
            )

        if tool_name == "shell__run":
            summary = _summarize_shell_run_args(planned.arguments)
            try:
                diff_ref = self._build_shell_run_preview(planned)
            except Exception as e:
                code = _classify_tool_exception(e)
                return InspectionResult(
                    decision=InspectionDecision.DENY,
                    action_summary="Invalid shell command request.",
                    risk_level="high",
                    reason=str(e),
                    error_code=code,
                    diff_ref=None,
                )
            return InspectionResult(
                decision=InspectionDecision.REQUIRE_APPROVAL,
                action_summary=summary,
                risk_level="high",
                reason="Strict mode: approve every tool call.",
                error_code=None,
                diff_ref=diff_ref,
            )

        # Low-risk tools: require approval but only preview args (do not read file contents).
        diff_ref = self._build_args_preview(planned, summary=f"Preview for {tool_name}")
        return InspectionResult(
            decision=InspectionDecision.REQUIRE_APPROVAL,
            action_summary=f"Execute tool: {tool_name}",
            risk_level="low",
            reason="Strict mode: approve every tool call.",
            error_code=None,
            diff_ref=diff_ref,
        )

    def _build_args_preview(self, planned: PlannedToolCall, *, summary: str) -> ArtifactRef:
        text = json.dumps(planned.arguments, ensure_ascii=False, sort_keys=True, indent=2)
        return self._artifact_store.put(text, kind="diff", meta={"summary": summary})


def _classify_tool_exception(exc: BaseException) -> ErrorCode:
    if isinstance(exc, ToolRuntimeError):
        msg = str(exc).lower()
        if "unknown tool" in msg:
            return ErrorCode.TOOL_UNKNOWN
        return ErrorCode.TOOL_FAILED
    if isinstance(exc, PermissionError):
        return ErrorCode.PERMISSION
    if isinstance(exc, FileNotFoundError):
        return ErrorCode.NOT_FOUND
    if isinstance(exc, TimeoutError):
        return ErrorCode.TIMEOUT
    if isinstance(exc, (ValueError, json.JSONDecodeError)):
        return ErrorCode.BAD_REQUEST
    if isinstance(exc, OSError):
        return ErrorCode.UNKNOWN
    return ErrorCode.UNKNOWN


def _is_under_spec_dir(path: str) -> bool:
    normalized = path.replace("\\", "/").lstrip("/")
    return normalized == "spec" or normalized.startswith("spec/")


def _tool_approval_policy_path(project_root: Path) -> Path:
    # Keep this in the author-visible policy dir so users can audit/edit it.
    return project_root / ".aura" / "policy" / "tool_approvals.json"


def _load_tool_approval_policy(project_root: Path) -> dict[str, Any]:
    path = _tool_approval_policy_path(project_root)
    if not path.exists():
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return raw if isinstance(raw, dict) else {}


def _save_tool_approval_policy(project_root: Path, policy: dict[str, Any]) -> None:
    path = _tool_approval_policy_path(project_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(policy, ensure_ascii=False, sort_keys=True, indent=2), encoding="utf-8")
    tmp.replace(path)


def add_shell_run_allowlist_rule(*, project_root: Path, command_prefix: str, cwd: str | None) -> None:
    command_prefix = " ".join(str(command_prefix).splitlines()).strip()
    if not command_prefix:
        return
    policy = _load_tool_approval_policy(project_root)
    rules = policy.get("shell__run_allow", [])
    if not isinstance(rules, list):
        rules = []
    entry: dict[str, Any] = {"command_prefix": command_prefix}
    if cwd is not None and str(cwd).strip():
        entry["cwd"] = str(cwd).strip()
    for existing in rules:
        if not isinstance(existing, dict):
            continue
        if existing.get("command_prefix") == entry.get("command_prefix") and existing.get("cwd") == entry.get("cwd"):
            return
    rules.append(entry)
    policy["shell__run_allow"] = rules
    _save_tool_approval_policy(project_root, policy)


def _normalize_shell_command(command: Any) -> str | None:
    if not isinstance(command, str):
        return None
    one_line = " ".join(command.strip().splitlines()).strip()
    return one_line if one_line else None


def _shell_run_is_allowlisted(project_root: Path, args: dict[str, Any]) -> bool:
    cmd = _normalize_shell_command(args.get("command"))
    if not cmd:
        return False
    cwd = args.get("cwd")
    cwd_s = str(cwd).strip() if isinstance(cwd, str) else None

    policy = _load_tool_approval_policy(project_root)
    rules = policy.get("shell__run_allow", [])
    if not isinstance(rules, list):
        return False
    for rule in rules:
        if not isinstance(rule, dict):
            continue
        prefix = rule.get("command_prefix")
        if not isinstance(prefix, str) or not prefix.strip():
            continue
        if not cmd.startswith(prefix.strip()):
            continue
        rule_cwd = rule.get("cwd")
        if isinstance(rule_cwd, str) and rule_cwd.strip():
            if cwd_s is None or cwd_s != rule_cwd.strip():
                continue
        return True
    return False


def _load_spec_proposal_record(project_root: Path, proposal_id: str) -> dict[str, Any]:
    import json

    root = project_root / ".aura" / "state" / "spec" / "proposals"
    path = root / f"{proposal_id}.json"
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("Invalid proposal record.")
    return raw
