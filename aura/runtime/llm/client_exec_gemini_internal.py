from __future__ import annotations

from .client_gemini_internal import _gemini_internal_to_response
from .client_httpx_errors import _wrap_httpx_like_exception
from .errors import CancellationToken, CredentialResolutionError, LLMErrorCode, LLMRequestError
from .providers.base import PreparedRequest
from .providers.gemini_internal import GeminiInternalAdapter
from .secrets import resolve_credential
from .trace import LLMTrace
from .types import CanonicalRequest, LLMResponse


def complete_gemini_internal(
    *,
    profile,
    request: CanonicalRequest,
    timeout_s: float | None,
    cancel: CancellationToken | None,
    trace: LLMTrace | None,
) -> LLMResponse:
    if profile.credential_ref is None:
        raise LLMRequestError(
            "Missing credentials for gemini_internal profile.",
            code=LLMErrorCode.AUTH,
            provider_kind=profile.provider_kind,
            profile_id=profile.profile_id,
            model=profile.model_name,
            retryable=False,
            details={"operation": "complete", "missing": "credential_ref"},
        )
    try:
        token = resolve_credential(profile.credential_ref)
    except CredentialResolutionError as e:
        raise LLMRequestError(
            str(e),
            code=LLMErrorCode.AUTH,
            provider_kind=profile.provider_kind,
            profile_id=profile.profile_id,
            model=profile.model_name,
            retryable=False,
            details={"operation": "complete", "credential_ref": getattr(e, "credential_ref", None)},
            cause=e,
        ) from e

    prepared = GeminiInternalAdapter().prepare_request(profile, request)
    auth_value = token.strip()
    if not auth_value.lower().startswith("bearer "):
        auth_value = f"Bearer {auth_value}"
    prepared = PreparedRequest(
        method=prepared.method,
        url=prepared.url,
        headers={**prepared.headers, "Authorization": auth_value},
        json=prepared.json,
    )
    request_timeout_s = timeout_s if timeout_s is not None else profile.timeout_s
    if trace is not None:
        trace.record_prepared_request(
            provider_kind=profile.provider_kind,
            profile_id=profile.profile_id,
            base_url=profile.base_url,
            model=profile.model_name,
            stream=False,
            timeout_s=request_timeout_s,
            payload=prepared.redacted().json,
        )

    if cancel is not None and cancel.cancelled:
        raise LLMRequestError(
            "Request cancelled.",
            code=LLMErrorCode.CANCELLED,
            provider_kind=profile.provider_kind,
            profile_id=profile.profile_id,
            model=profile.model_name,
            retryable=False,
            details={"operation": "complete"},
        )

    try:
        import httpx  # type: ignore

        with httpx.Client(timeout=(None if request_timeout_s is None else float(request_timeout_s))) as client:
            r = client.request(
                prepared.method,
                prepared.url,
                headers=prepared.headers,
                json=prepared.json,
            )
            r.raise_for_status()
            data = r.json()
    except Exception as e:
        # Preserve response body for easier debugging (many gateways return useful JSON on 4xx).
        try:
            import httpx  # type: ignore

            if isinstance(e, httpx.HTTPStatusError):
                body = ""
                try:
                    body = e.response.text or ""
                except Exception:
                    body = ""
                if trace is not None:
                    try:
                        trace.write_json(
                            "provider_error_response.json",
                            {
                                "status_code": int(e.response.status_code),
                                "headers": dict(e.response.headers),
                                "text": body[:4000],
                            },
                        )
                    except Exception:
                        pass
        except Exception:
            pass
        raise _wrap_httpx_like_exception(
            e,
            provider_kind=profile.provider_kind,
            profile_id=profile.profile_id,
            model=profile.model_name,
            operation="complete",
        ) from e
    if trace is not None:
        trace.write_json("provider_response.json", data)
    return _gemini_internal_to_response(profile_id=profile.profile_id, data=data)

