from __future__ import annotations

from aura.runtime.llm.providers.gemini import GeminiAdapter
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


def test_gemini_groups_tool_responses_into_single_turn():
    profile = ModelProfile(
        profile_id="gemini",
        provider_kind=ProviderKind.GEMINI,
        base_url="https://example.invalid/api",
        model_name="gemini-2.5-flash-lite",
        credential_ref=CredentialRef(kind="inline", identifier="k"),
        capabilities=ModelCapabilities(supports_tools=True, supports_streaming=True),
    )

    request = CanonicalRequest(
        system="sys",
        messages=[
            CanonicalMessage(role=CanonicalMessageRole.USER, content="hi"),
            CanonicalMessage(
                role=CanonicalMessageRole.ASSISTANT,
                content="calling tools",
                tool_calls=[
                    ToolCall(tool_call_id="c1", name="skill__list", arguments={"x": 1}),
                    ToolCall(tool_call_id="c2", name="update_plan", arguments={"y": 2}),
                ],
            ),
            CanonicalMessage(
                role=CanonicalMessageRole.TOOL,
                content='{"ok": true, "result": {"skills": []}}',
                tool_call_id="c1",
                tool_name="skill__list",
            ),
            CanonicalMessage(
                role=CanonicalMessageRole.TOOL,
                content='{"ok": true, "result": {"plan": []}}',
                tool_call_id="c2",
                tool_name="update_plan",
            ),
            CanonicalMessage(role=CanonicalMessageRole.USER, content="next"),
        ],
        tools=[],
        params={},
    )

    prepared = GeminiAdapter().prepare_request(profile, request)
    assert prepared.method == "POST"
    payload = prepared.json
    contents = payload.get("contents")
    assert isinstance(contents, list)

    # Expected sequence: system user, user hi, model tool calls, user tool responses (grouped), user next
    tool_resp_entries = [c for c in contents if isinstance(c, dict) and c.get("role") == "user" and isinstance(c.get("parts"), list) and any(isinstance(p, dict) and "functionResponse" in p for p in c["parts"])]
    assert len(tool_resp_entries) == 1
    parts = tool_resp_entries[0]["parts"]
    fn_res_parts = [p for p in parts if isinstance(p, dict) and "functionResponse" in p]
    assert len(fn_res_parts) == 2


def test_gemini_function_call_includes_thought_signature_aliases():
    profile = ModelProfile(
        profile_id="gemini",
        provider_kind=ProviderKind.GEMINI,
        base_url="https://example.invalid/api",
        model_name="gemini-2.5-flash-lite",
        credential_ref=CredentialRef(kind="inline", identifier="k"),
        capabilities=ModelCapabilities(supports_tools=True, supports_streaming=True),
    )

    request = CanonicalRequest(
        system=None,
        messages=[
            CanonicalMessage(
                role=CanonicalMessageRole.ASSISTANT,
                content="",
                tool_calls=[
                    ToolCall(
                        tool_call_id="c1",
                        name="project__list_dir",
                        arguments={"path": "."},
                        raw_arguments=None,
                        thought_signature="sig123",
                    )
                ],
            )
        ],
        tools=[],
        params={},
    )

    payload = GeminiAdapter().prepare_request(profile, request).json
    contents = payload["contents"]
    part = contents[0]["parts"][0]
    assert part.get("thoughtSignature") == "sig123"
    assert part.get("thought_signature") == "sig123"
