from __future__ import annotations

import json

import pytest

from aura.runtime.ids import new_id, now_ts_ms
from aura.runtime.llm.types import LLMResponse
from aura.runtime.protocol import Op, OpKind


@pytest.mark.asyncio
async def test_event_sequence(make_runtime, monkeypatch):
    rt = make_runtime(tools_enabled=False)
    engine = rt.engine

    async def _stub_run_agent_once(self, *, request, profile, request_id, turn_id, **_kwargs):
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
        payload={"text": "Hello"},
        session_id=rt.session_id,
        request_id=new_id("req"),
        timestamp=now_ts_ms(),
        turn_id=new_id("turn"),
    )
    result = await engine.arun(op)
    assert result.status == "completed"

    path = rt.paths.events_dir / f"{rt.session_id}.jsonl"
    assert path.exists()

    seqs: list[int] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        raw = json.loads(line)
        seq = raw.get("sequence")
        if isinstance(seq, int):
            seqs.append(seq)

    assert seqs
    assert seqs == sorted(seqs)
    assert len(seqs) == len(set(seqs))
