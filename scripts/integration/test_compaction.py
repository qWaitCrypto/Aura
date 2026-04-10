from __future__ import annotations

import pytest

from aura.runtime.ids import new_id, now_ts_ms
from aura.runtime.llm.types import CanonicalMessage, CanonicalMessageRole
from aura.runtime.llm.types import LLMResponse
from aura.runtime.protocol import EventKind, Op, OpKind


@pytest.mark.asyncio
async def test_compaction(make_runtime, default_model_config, monkeypatch):
    cfg = default_model_config(context_limit_tokens=200)
    rt = make_runtime(tools_enabled=False, model_config=cfg)
    engine = rt.engine

    # Seed history (compaction requires non-empty history or existing summary).
    engine._history = [  # type: ignore[attr-defined]
        CanonicalMessage(role=CanonicalMessageRole.USER, content="u" * 2000),
        CanonicalMessage(role=CanonicalMessageRole.ASSISTANT, content="a" * 2000),
        CanonicalMessage(role=CanonicalMessageRole.USER, content="u2" * 2000),
    ]

    async def _stub_run_agent_once(self, *, request, profile, request_id, turn_id, **_kwargs):
        return LLMResponse(
            provider_kind=profile.provider_kind,
            profile_id=profile.profile_id,
            model=profile.model_name,
            text="durable summary",
            tool_calls=[],
            usage=None,
            stop_reason="stop",
            request_id="stub",
        )

    monkeypatch.setattr(engine.__class__, "_run_agent_once", _stub_run_agent_once, raising=True)

    op = Op(
        kind=OpKind.COMPACT.value,
        payload={},
        session_id=rt.session_id,
        request_id=new_id("req"),
        timestamp=now_ts_ms(),
        turn_id=new_id("turn"),
    )
    result = await engine.arun(op)
    assert result.status == "completed"
    assert isinstance(engine.memory_summary, str) and engine.memory_summary

    meta = rt.session_store.get_session(rt.session_id)
    assert isinstance(meta.get("last_compacted_at"), int)
    assert meta.get("last_compaction_trigger") == "manual"

    # History should be trimmed under a small context limit.
    history = engine._history  # type: ignore[attr-defined]
    assert history is not None
    assert len(history) < 3

    events = list(rt.event_log_store.read(rt.session_id))
    assert any(
        e.kind == EventKind.OPERATION_COMPLETED.value
        and e.payload.get("op_kind") == OpKind.COMPACT.value
        and e.payload.get("trigger") == "manual"
        for e in events
    )


@pytest.mark.asyncio
async def test_auto_compact_trigger(make_runtime, default_model_config, monkeypatch):
    cfg = default_model_config(context_limit_tokens=200, auto_compact_threshold_ratio=0.5)
    rt = make_runtime(tools_enabled=False, model_config=cfg)
    engine = rt.engine

    async def _stub_run_agent_once(self, *, request, profile, request_id, turn_id, **_kwargs):
        # Used both by compaction and chat.
        if request.system is None:
            return LLMResponse(
                provider_kind=profile.provider_kind,
                profile_id=profile.profile_id,
                model=profile.model_name,
                text="auto summary",
                tool_calls=[],
                usage=None,
                stop_reason="stop",
                request_id="stub",
            )
        return LLMResponse(
            provider_kind=profile.provider_kind,
            profile_id=profile.profile_id,
            model=profile.model_name,
            text="ok",
            tool_calls=[],
            usage=None,
            stop_reason="stop",
            request_id="stub",
        )

    monkeypatch.setattr(engine.__class__, "_run_agent_once", _stub_run_agent_once, raising=True)

    op = Op(
        kind=OpKind.CHAT.value,
        payload={"text": "x" * 5000},
        session_id=rt.session_id,
        request_id=new_id("req"),
        timestamp=now_ts_ms(),
        turn_id=new_id("turn"),
    )
    result = await engine.arun(op)
    assert result.status == "completed"

    meta = rt.session_store.get_session(rt.session_id)
    assert meta.get("last_compaction_trigger") == "auto"

    events = list(rt.event_log_store.read(rt.session_id))
    assert any(
        e.kind == EventKind.OPERATION_COMPLETED.value
        and e.payload.get("op_kind") == OpKind.COMPACT.value
        and e.payload.get("trigger") == "auto"
        for e in events
    )
