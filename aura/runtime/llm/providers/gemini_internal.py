from __future__ import annotations

import json
from typing import Any

from .base import PreparedRequest
from ..errors import ProviderAdapterError
from ..types import (
    CanonicalMessage,
    CanonicalMessageRole,
    CanonicalRequest,
    ModelProfile,
)


def _text_part(text: str) -> dict[str, Any]:
    return {"text": text}


def _function_call_part(*, name: str, args: dict[str, Any]) -> dict[str, Any]:
    return {"functionCall": {"name": name, "args": args}}


def _function_call_part_with_signature(
    *, name: str, args: dict[str, Any], thought_signature: str | None
) -> dict[str, Any]:
    part: dict[str, Any] = {"functionCall": {"name": name, "args": args}}
    if thought_signature:
        # The gateway expects this field to be echoed back on subsequent requests.
        part["thoughtSignature"] = thought_signature
    return part


def _function_response_part(*, name: str, response: Any) -> dict[str, Any]:
    return {"functionResponse": {"name": name, "response": response}}


def _message_to_content(msg: CanonicalMessage) -> dict[str, Any] | None:
    parts: list[dict[str, Any]] = []
    if msg.role is not CanonicalMessageRole.TOOL and msg.content:
        parts.append(_text_part(msg.content))

    if msg.role is CanonicalMessageRole.ASSISTANT and msg.tool_calls:
        for tc in msg.tool_calls:
            parts.append(
                _function_call_part_with_signature(
                    name=tc.name,
                    args=dict(tc.arguments),
                    thought_signature=tc.thought_signature,
                )
            )

    if msg.role is CanonicalMessageRole.TOOL:
        tool_name = msg.tool_name or "tool"
        try:
            parsed = json.loads(msg.content)
        except Exception:
            parsed = {"content": msg.content}
        # For Gemini-style function responses, send a structured functionResponse part.
        # Prefer the underlying tool result object if present.
        response_obj: Any = parsed
        if isinstance(parsed, dict) and "result" in parsed:
            response_obj = parsed["result"]
        if not isinstance(response_obj, dict):
            response_obj = {"result": response_obj}
        parts.append(_function_response_part(name=tool_name, response=response_obj))

    if not parts:
        return None

    if msg.role is CanonicalMessageRole.USER:
        role = "user"
    elif msg.role is CanonicalMessageRole.ASSISTANT:
        role = "model"
    elif msg.role is CanonicalMessageRole.TOOL:
        # Gemini-style function responses are represented as user content.
        role = "user"
    elif msg.role is CanonicalMessageRole.SYSTEM:
        # Treat any system messages embedded in history as user content.
        role = "user"
    else:
        role = "user"

    return {"role": role, "parts": parts}


class GeminiInternalAdapter:
    """
    Adapter for the custom "v1internal:generateContent" Gemini-like endpoint.

    Notes:
    - This provider is treated as non-streaming (LLMClient.complete only).
    - Provider-specific fields live in ModelProfile.default_params. In particular:
      - `project` (required): top-level request field.
      - `session_id` (optional): included under request.
      - other keys are merged into the nested `request` object (except `contents`/`tools`).
    """

    def prepare_request(self, profile: ModelProfile, request: CanonicalRequest) -> PreparedRequest:
        if not isinstance(profile.default_params, dict):
            raise ProviderAdapterError("gemini_internal profile.default_params must be a dict.")
        project = profile.default_params.get("project")
        if not isinstance(project, str) or not project.strip():
            raise ProviderAdapterError(
                "gemini_internal requires default_params.project (string)."
            )

        # Build contents from canonical messages.
        contents: list[dict[str, Any]] = []
        if request.system:
            contents.append({"role": "user", "parts": [_text_part(request.system)]})
        for msg in request.messages:
            content = _message_to_content(msg)
            if content is not None:
                contents.append(content)

        nested_request: dict[str, Any] = {"contents": contents}
        session_id = profile.default_params.get("session_id")
        if isinstance(session_id, str) and session_id.strip():
            nested_request["session_id"] = session_id.strip()

        # Merge provider-specific request defaults, but do not allow overriding core keys.
        for k, v in profile.default_params.items():
            if k in {"project", "contents", "tools"}:
                continue
            if k not in nested_request:
                nested_request[k] = v

        if request.tools:
            nested_request["tools"] = [
                {
                    "functionDeclarations": [
                        {
                            "name": t.name,
                            "description": t.description,
                            "parameters": t.input_schema,
                        }
                        for t in request.tools
                    ]
                }
            ]
            nested_request.setdefault(
                "toolConfig",
                {"functionCallingConfig": {"mode": "AUTO"}},
            )

        payload = {"model": profile.model_name, "project": project.strip(), "request": nested_request}

        return PreparedRequest(
            method="POST",
            url=profile.base_url,
            headers={"Content-Type": "application/json"},
            json=payload,
        )
