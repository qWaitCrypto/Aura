from __future__ import annotations

import json
from typing import Any, Iterator

from .client_common import _merge_requirements, _raise_if_cancelled
from .config import ModelConfig
from .errors import CancellationToken, LLMErrorCode, LLMRequestError, ProviderAdapterError
from .router import ModelRouter
from .secrets import resolve_credential
from .trace import LLMTrace
from .types import (
    CanonicalMessageRole,
    CanonicalRequest,
    LLMResponse,
    LLMStreamEvent,
    LLMUsage,
    ModelRequirements,
    ModelRole,
    ProviderKind,
    ToolCall,
)


def _int_or_none(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    if value <= 0:
        return None
    return int(value)


def _agno_usage_to_aura_usage(metrics: Any) -> LLMUsage | None:
    if metrics is None:
        return None
    return LLMUsage(
        input_tokens=_int_or_none(getattr(metrics, "input_tokens", None)),
        output_tokens=_int_or_none(getattr(metrics, "output_tokens", None)),
        total_tokens=_int_or_none(getattr(metrics, "total_tokens", None)),
        cache_creation_input_tokens=_int_or_none(getattr(metrics, "cache_write_tokens", None)),
        cache_read_input_tokens=_int_or_none(getattr(metrics, "cache_read_tokens", None)),
    )


def _classify_agno_model_exception(e: BaseException) -> tuple[LLMErrorCode, bool]:
    if isinstance(e, KeyboardInterrupt):
        return LLMErrorCode.CANCELLED, False
    if isinstance(e, TimeoutError):
        return LLMErrorCode.TIMEOUT, True

    status_code = getattr(e, "status_code", None)
    if isinstance(status_code, int):
        if status_code == 400:
            return LLMErrorCode.BAD_REQUEST, False
        if status_code == 401:
            return LLMErrorCode.AUTH, False
        if status_code == 403:
            return LLMErrorCode.PERMISSION, False
        if status_code == 404:
            return LLMErrorCode.NOT_FOUND, False
        if status_code == 409:
            return LLMErrorCode.CONFLICT, False
        if status_code == 422:
            return LLMErrorCode.UNPROCESSABLE, False
        if status_code == 429:
            return LLMErrorCode.RATE_LIMIT, True
        if 500 <= status_code <= 599:
            return LLMErrorCode.SERVER_ERROR, True

    name = e.__class__.__name__
    if name in {"RemoteServerUnavailableError"}:
        return LLMErrorCode.NETWORK_ERROR, True
    if name in {"ModelAuthenticationError"}:
        return LLMErrorCode.AUTH, False
    if name in {"ModelRateLimitError"}:
        return LLMErrorCode.RATE_LIMIT, True
    if name in {"ModelProviderError"}:
        return LLMErrorCode.SERVER_ERROR, True

    return LLMErrorCode.UNKNOWN, False


def _build_agno_model(profile: Any) -> Any:
    model_name = getattr(profile, "model_name", None)
    if model_name is None or not isinstance(model_name, str) or not model_name.strip():
        raise ValueError("Selected model profile is missing model_name.")

    kind = getattr(profile, "provider_kind", None)
    kind_value = getattr(kind, "value", kind)

    api_key: str | None = None
    credential_ref = getattr(profile, "credential_ref", None)
    if credential_ref is not None:
        try:
            api_key = resolve_credential(credential_ref)
        except Exception:
            api_key = None

    timeout_s = getattr(profile, "timeout_s", None)
    base_url = getattr(profile, "base_url", None)

    if kind_value == ProviderKind.OPENAI_COMPATIBLE.value:
        from agno.models.openai.like import OpenAILike

        kwargs: dict[str, Any] = {"id": model_name.strip(), "api_key": api_key}
        if isinstance(base_url, str) and base_url.strip():
            kwargs["base_url"] = base_url.strip()
        if isinstance(timeout_s, (int, float)) and not isinstance(timeout_s, bool) and float(timeout_s) > 0:
            kwargs["timeout"] = float(timeout_s)
        return OpenAILike(**kwargs)

    if kind_value == ProviderKind.ANTHROPIC.value:
        from agno.models.anthropic.claude import Claude

        kwargs = {"id": model_name.strip(), "api_key": api_key}
        if isinstance(timeout_s, (int, float)) and not isinstance(timeout_s, bool) and float(timeout_s) > 0:
            kwargs["timeout"] = float(timeout_s)
        return Claude(**kwargs)

    raise ProviderAdapterError(f"Unsupported provider_kind for agno client: {kind_value!r}")


def _build_agno_inputs(*, request: CanonicalRequest) -> tuple[list[Any], list[Any] | None]:
    from agno.models.message import Message as AgnoMessage
    from agno.tools.function import Function as AgnoFunction

    messages: list[AgnoMessage] = []
    if isinstance(request.system, str) and request.system.strip():
        messages.append(AgnoMessage(role="system", content=request.system))

    for msg in request.messages:
        if msg.role is CanonicalMessageRole.SYSTEM:
            continue
        if msg.role is CanonicalMessageRole.TOOL:
            messages.append(
                AgnoMessage(
                    role="tool",
                    content=msg.content,
                    tool_call_id=msg.tool_call_id,
                    tool_name=msg.tool_name,
                )
            )
            continue
        if msg.role is CanonicalMessageRole.ASSISTANT:
            tool_calls = None
            if msg.tool_calls:
                tool_calls = []
                for tc in msg.tool_calls:
                    if not (isinstance(tc.tool_call_id, str) and tc.tool_call_id.strip()):
                        continue
                    tool_calls.append(
                        {
                            "id": tc.tool_call_id,
                            "type": "function",
                            "function": {
                                "name": tc.name,
                                "arguments": json.dumps(tc.arguments, ensure_ascii=False),
                            },
                        }
                    )
            messages.append(AgnoMessage(role="assistant", content=msg.content, tool_calls=tool_calls))
            continue
        messages.append(AgnoMessage(role=msg.role.value, content=msg.content))

    tools: list[AgnoFunction] | None = None
    if request.tools:
        tools = []
        for spec in request.tools:
            tools.append(
                AgnoFunction(
                    name=spec.name,
                    description=spec.description,
                    parameters=spec.input_schema,
                    external_execution=True,
                )
            )

    return messages, tools


class AgnoLLMClient:
    """
    Minimal agno-backed LLM client for Aura.

    This client is intentionally limited to non-streaming completion and uses
    external tool execution so Aura remains the source of truth for tool loops,
    approvals, and event logging.
    """

    def __init__(self, config: ModelConfig) -> None:
        self._router = ModelRouter(config)

    def complete(
        self,
        *,
        role: ModelRole,
        requirements: ModelRequirements,
        request: CanonicalRequest,
        timeout_s: float | None = None,
        cancel: CancellationToken | None = None,
        trace: LLMTrace | None = None,
    ) -> LLMResponse:
        if requirements.needs_streaming:
            raise ProviderAdapterError("AgnoLLMClient.complete() does not support needs_streaming=True; use stream().")

        effective = _merge_requirements(requirements, request=request, force_streaming=False)
        resolved = self._router.resolve(role=role, requirements=effective)
        profile = resolved.profile

        if trace is not None:
            trace.record_canonical_request(request)

        _raise_if_cancelled(
            cancel,
            provider_kind=profile.provider_kind,
            profile_id=profile.profile_id,
            model=profile.model_name,
            operation="complete",
        )

        try:
            agno_model = _build_agno_model(profile)
            if timeout_s is not None:
                try:
                    setattr(agno_model, "timeout", float(timeout_s))
                except Exception:
                    pass

            messages, tools = _build_agno_inputs(request=request)
            model_response = agno_model.response(messages=messages, tools=tools)
        except Exception as e:
            code, retryable = _classify_agno_model_exception(e)
            raise LLMRequestError(
                str(e) or e.__class__.__name__,
                code=code,
                provider_kind=profile.provider_kind,
                profile_id=profile.profile_id,
                model=profile.model_name,
                retryable=retryable,
                details={"operation": "agno_complete"},
                cause=e,
            ) from e

        content = getattr(model_response, "content", None)
        if content is None:
            assistant_text = ""
        elif isinstance(content, str):
            assistant_text = content
        else:
            assistant_text = str(content)

        tool_calls: list[ToolCall] = []
        for te in getattr(model_response, "tool_executions", []) or []:
            if not getattr(te, "external_execution_required", False):
                continue
            tool_call_id = getattr(te, "tool_call_id", None)
            tool_name = getattr(te, "tool_name", None)
            tool_args = getattr(te, "tool_args", None)
            if not (isinstance(tool_call_id, str) and tool_call_id and isinstance(tool_name, str) and tool_name):
                continue
            if not isinstance(tool_args, dict):
                tool_args = {}
            tool_calls.append(
                ToolCall(tool_call_id=tool_call_id, name=tool_name, arguments=tool_args, raw_arguments=None)
            )

        usage = _agno_usage_to_aura_usage(getattr(model_response, "response_usage", None))

        return LLMResponse(
            provider_kind=profile.provider_kind,
            profile_id=profile.profile_id,
            model=profile.model_name,
            text=assistant_text,
            tool_calls=tool_calls,
            usage=usage,
            stop_reason=None,
            request_id=None,
        )

    def stream(
        self,
        *,
        role: ModelRole,
        requirements: ModelRequirements,
        request: CanonicalRequest,
        timeout_s: float | None = None,
        cancel: CancellationToken | None = None,
        trace: LLMTrace | None = None,
    ) -> Iterator[LLMStreamEvent]:
        raise ProviderAdapterError("AgnoLLMClient.stream() is not implemented.")

