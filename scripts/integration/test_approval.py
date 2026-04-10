from __future__ import annotations

import asyncio

import pytest

from aura.runtime.ids import new_id, now_ts_ms
from aura.runtime.llm.types import ToolCall
from aura.runtime.llm.types import LLMResponse
from aura.runtime.protocol import EventKind, Op, OpKind
from aura.runtime.run_snapshots import run_snapshot_path
from aura.runtime.tools.runtime import ToolApprovalMode
from aura.runtime.engine import ToolDecision


@pytest.mark.asyncio
async def test_tool_approval_flow(make_runtime, monkeypatch):
    async def _inline_to_thread(func, /, *args, **kwargs):
        return func(*args, **kwargs)

    # In this sandbox, filesystem calls from worker threads can hang; keep tool execution inline.
    monkeypatch.setattr(asyncio, "to_thread", _inline_to_thread, raising=True)

    rt = make_runtime(tools_enabled=True, approval_mode=ToolApprovalMode.STRICT)
    engine = rt.engine

    # Register a tool that avoids filesystem I/O inside the worker thread.
    from dataclasses import dataclass, field
    from typing import Any

    @dataclass(frozen=True, slots=True)
    class TestNoopTool:
        name: str = "test__noop"
        description: str = "No-op tool for integration tests."
        input_schema: dict[str, Any] = field(
            default_factory=lambda: {
                "type": "object",
                "properties": {"text": {"type": "string"}},
                "required": ["text"],
                "additionalProperties": False,
            }
        )

        def execute(self, *, args: dict[str, Any], project_root):
            return {"ok": True, "text": str(args.get("text") or "")}

    assert engine.tool_registry is not None  # type: ignore[attr-defined]
    engine.tool_registry.register(TestNoopTool())  # type: ignore[attr-defined]

    call_id = "call_1"
    seen: dict[str, int] = {}

    async def _stub_run_agent_once(self, *, request, profile, request_id, turn_id, **_kwargs):
        count = seen.get(request_id, 0)
        seen[request_id] = count + 1
        if count == 0:
            return LLMResponse(
                provider_kind=profile.provider_kind,
                profile_id=profile.profile_id,
                model=profile.model_name,
                text="I will compute stats.",
                tool_calls=[ToolCall(tool_call_id=call_id, name="test__noop", arguments={"text": "hello"})],
                usage=None,
                stop_reason="tool_use",
                request_id="stub",
            )
        return LLMResponse(
            provider_kind=profile.provider_kind,
            profile_id=profile.profile_id,
            model=profile.model_name,
            text="done",
            tool_calls=[],
            usage=None,
            stop_reason="stop",
            request_id="stub",
        )

    monkeypatch.setattr(engine.__class__, "_run_agent_once", _stub_run_agent_once, raising=True)

    request_id = new_id("req")
    op = Op(
        kind=OpKind.CHAT.value,
        payload={"text": "Please run a tool."},
        session_id=rt.session_id,
        request_id=request_id,
        timestamp=now_ts_ms(),
        turn_id=new_id("turn"),
    )
    result = await engine.arun(op)
    assert result.status == "needs_approval"
    assert result.pending_tools

    snap_path = run_snapshot_path(project_root=rt.project_root, run_id=request_id)
    assert snap_path.exists()

    approval_id = result.approval_id
    assert isinstance(approval_id, str) and approval_id

    approval_rec = rt.approval_store.get(approval_id)
    assert approval_rec.status.value == "pending"

    events = list(rt.event_log_store.read(rt.session_id))
    assert any(e.kind == EventKind.APPROVAL_REQUIRED.value and e.payload.get("approval_id") == approval_id for e in events)
    assert any(e.kind == EventKind.RUN_PAUSED.value and e.payload.get("run_id") == request_id for e in events)

    result2 = await engine.continue_run(
        run_id=request_id,
        decisions=[ToolDecision(tool_call_id=call_id, decision="approve")],
    )
    assert result2.status == "completed"
    assert not snap_path.exists()

    approval_rec2 = rt.approval_store.get(approval_id)
    assert approval_rec2.status.value == "granted"
