from __future__ import annotations

import anthropic
import openai
from anthropic import Anthropic
from openai import OpenAI

from typing import Iterator

from .client_common import _merge_requirements, _raise_if_cancelled
from .client_exec_anthropic import complete_anthropic, stream_anthropic
from .client_exec_gemini import complete_gemini, stream_gemini
from .client_exec_openai_compatible import complete_openai_compatible, stream_openai_compatible
from .config import ModelConfig
from .errors import CancellationToken, LLMErrorCode, LLMRequestError, ProviderAdapterError
from .router import ModelRouter
from .trace import LLMTrace
from .types import CanonicalRequest, LLMResponse, LLMStreamEvent, ModelRequirements, ModelRole, ProviderKind


class LLMClient:
    def __init__(self, config: ModelConfig) -> None:
        self._router = ModelRouter(config)

    @staticmethod
    def _assert_profile_base_url(*, profile: "ModelProfile", operation: str) -> None:
        if not profile.base_url.strip():
            raise LLMRequestError(
                "Model profile base_url is empty. Edit .aura/config/models.json and set base_url for the active profile.",
                code=LLMErrorCode.BAD_REQUEST,
                provider_kind=profile.provider_kind,
                profile_id=profile.profile_id,
                model=profile.model_name,
                retryable=False,
                details={"operation": operation, "missing": "base_url"},
            )

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
            raise ProviderAdapterError("complete() does not support needs_streaming=True; use stream().")

        effective = _merge_requirements(requirements, request=request, force_streaming=False)
        resolved = self._router.resolve(role=role, requirements=effective)
        profile = resolved.profile
        self._assert_profile_base_url(profile=profile, operation="complete")
        if trace is not None:
            trace.record_canonical_request(request)
        _raise_if_cancelled(
            cancel,
            provider_kind=profile.provider_kind,
            profile_id=profile.profile_id,
            model=profile.model_name,
            operation="complete",
        )

        if profile.provider_kind is ProviderKind.OPENAI_COMPATIBLE:
            return complete_openai_compatible(
                profile=profile,
                request=request,
                timeout_s=timeout_s,
                cancel=cancel,
                trace=trace,
            )

        if profile.provider_kind is ProviderKind.GEMINI:
            return complete_gemini(
                profile=profile,
                request=request,
                timeout_s=timeout_s,
                cancel=cancel,
                trace=trace,
            )

        if profile.provider_kind is ProviderKind.ANTHROPIC:
            return complete_anthropic(
                profile=profile,
                request=request,
                timeout_s=timeout_s,
                cancel=cancel,
                trace=trace,
            )

        raise ProviderAdapterError(f"Unsupported provider_kind: {profile.provider_kind}")

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
        effective = _merge_requirements(requirements, request=request, force_streaming=True)
        resolved = self._router.resolve(role=role, requirements=effective)
        profile = resolved.profile
        self._assert_profile_base_url(profile=profile, operation="stream")
        if trace is not None:
            trace.record_canonical_request(request)
        _raise_if_cancelled(
            cancel,
            provider_kind=profile.provider_kind,
            profile_id=profile.profile_id,
            model=profile.model_name,
            operation="stream",
        )

        if profile.provider_kind is ProviderKind.OPENAI_COMPATIBLE:
            yield from stream_openai_compatible(
                profile=profile,
                request=request,
                timeout_s=timeout_s,
                cancel=cancel,
                trace=trace,
            )
            return

        if profile.provider_kind is ProviderKind.GEMINI:
            yield from stream_gemini(
                profile=profile,
                request=request,
                timeout_s=timeout_s,
                cancel=cancel,
                trace=trace,
            )
            return

        if profile.provider_kind is ProviderKind.ANTHROPIC:
            yield from stream_anthropic(
                profile=profile,
                request=request,
                timeout_s=timeout_s,
                cancel=cancel,
                trace=trace,
            )
            return

        raise ProviderAdapterError(f"Unsupported provider_kind: {profile.provider_kind}")
