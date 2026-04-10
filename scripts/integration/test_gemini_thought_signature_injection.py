from __future__ import annotations

from aura.runtime.llm.agno_aura_model import build_aura_agno_model
from aura.runtime.llm.types import (
    CanonicalMessage,
    CanonicalMessageRole,
    CanonicalRequest,
    CredentialRef,
    ModelCapabilities,
    ModelProfile,
    ProviderKind,
    ToolCall,
)


def test_gemini_model_reinjects_thought_signature_for_replayed_tool_calls(tmp_path):
    profile = ModelProfile(
        profile_id="gemini",
        provider_kind=ProviderKind.GEMINI,
        base_url="https://example.invalid/api",
        model_name="gemini-2.5-flash-lite",
        credential_ref=CredentialRef(kind="inline", identifier="k"),
        capabilities=ModelCapabilities(supports_tools=True, supports_streaming=True),
    )
    model = build_aura_agno_model(profile=profile, project_root=tmp_path, session_id="sess")
    # Simulate previous provider response that included a signature.
    model._thought_signatures_by_tool_call_id["gemini_1"] = "sig123"  # type: ignore[attr-defined]

    req = CanonicalRequest(
        system=None,
        messages=[
            CanonicalMessage(
                role=CanonicalMessageRole.ASSISTANT,
                content="",
                tool_calls=[ToolCall(tool_call_id="gemini_1", name="skill__list", arguments={}, raw_arguments="{}", thought_signature=None)],
            )
        ],
        tools=[],
        params={},
    )
    injected = model._inject_thought_signatures(req)  # type: ignore[attr-defined]
    assert injected.messages[0].tool_calls is not None
    assert injected.messages[0].tool_calls[0].thought_signature == "sig123"

