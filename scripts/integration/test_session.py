from __future__ import annotations

import json

import pytest

from aura.runtime.ids import new_id, now_ts_ms
from aura.runtime.llm.types import LLMResponse
from aura.runtime.protocol import EventKind, Op, OpKind


@pytest.mark.asyncio
async def test_session_persistence(make_runtime, monkeypatch):
    rt = make_runtime(tools_enabled=False)
    engine = rt.engine

    async def _stub_run_agent_once(self, *, request, profile, request_id, turn_id, **_kwargs):
        return LLMResponse(
            provider_kind=profile.provider_kind,
            profile_id=profile.profile_id,
            model=profile.model_name,
            text="hi from stub",
            tool_calls=[],
            usage=None,
            stop_reason="stop",
            request_id="stub",
        )

    # Avoid real model calls.
    monkeypatch.setattr(engine.__class__, "_run_agent_once", _stub_run_agent_once, raising=True)

    op = Op(
        kind=OpKind.CHAT.value,
        payload={"text": "Hello"},
        session_id=rt.session_id,
        request_id=new_id("req"),
        timestamp=now_ts_ms(),
        turn_id=new_id("turn"),
    )

    result = await engine.arun(op)
    assert result.status == "completed"

    events_path = rt.paths.events_dir / f"{rt.session_id}.jsonl"
    assert events_path.exists()

    events = list(rt.event_log_store.read(rt.session_id))
    kinds = [e.kind for e in events]
    assert EventKind.OPERATION_STARTED.value in kinds
    assert EventKind.LLM_RESPONSE_COMPLETED.value in kinds
    assert EventKind.OPERATION_COMPLETED.value in kinds

    # Rehydrate via a fresh engine instance.
    rt2 = make_runtime(tools_enabled=False, session_id=rt.session_id)
    rt2.engine.load_history_from_events()

    history = rt2.engine._history  # type: ignore[attr-defined]
    assert history is not None
    assert len(history) >= 2
    assert history[0].role.value == "user"
    assert history[1].role.value == "assistant"

    # Sanity: the persisted assistant output is the stub text.
    last = next(e for e in reversed(events) if e.kind == EventKind.LLM_RESPONSE_COMPLETED.value)
    output_ref = last.payload.get("output_ref")
    assert isinstance(output_ref, dict)
    locator = output_ref.get("locator")
    assert isinstance(locator, str) and locator

    data = (rt.paths.artifacts_dir / locator).read_bytes().decode("utf-8", errors="replace")
    assert "hi from stub" in data


def test_event_log_is_valid_jsonl(make_runtime):
    rt = make_runtime(tools_enabled=False)
    (rt.paths.system_dir / "events").mkdir(parents=True, exist_ok=True)
    log = rt.paths.events_dir / f"{rt.session_id}.jsonl"
    # May be empty; just ensure parser is tolerant.
    if not log.exists():
        return
    for line in log.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        json.loads(line)
