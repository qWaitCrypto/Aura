from __future__ import annotations

import threading
from enum import StrEnum
from typing import Any

import anthropic
import openai

from ..error_codes import ErrorCode
from .types import ProviderKind


class ModelConfigError(ValueError):
    pass


class ModelResolutionError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        role: str | None = None,
        profile_id: str | None = None,
    ) -> None:
        super().__init__(message)
        self.role = role
        self.profile_id = profile_id


class CredentialResolutionError(RuntimeError):
    def __init__(self, message: str, *, credential_ref: str | None = None) -> None:
        super().__init__(message)
        self.credential_ref = credential_ref


class ProviderAdapterError(RuntimeError):
    pass


LLMErrorCode = ErrorCode


class CancellationToken:
    def __init__(self) -> None:
        self._event = threading.Event()

    def cancel(self) -> None:
        self._event.set()

    @property
    def cancelled(self) -> bool:
        return self._event.is_set()


class LLMRequestError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        code: LLMErrorCode,
        provider_kind: ProviderKind | None = None,
        profile_id: str | None = None,
        model: str | None = None,
        status_code: int | None = None,
        request_id: str | None = None,
        retryable: bool | None = None,
        details: dict[str, Any] | None = None,
        cause: BaseException | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.provider_kind = provider_kind
        self.profile_id = profile_id
        self.model = model
        self.status_code = status_code
        self.request_id = request_id
        self.retryable = retryable
        self.details = details
        self.__cause__ = cause


def is_retryable_error_code(code: LLMErrorCode) -> bool:
    return code in {
        LLMErrorCode.TIMEOUT,
        LLMErrorCode.RATE_LIMIT,
        LLMErrorCode.SERVER_ERROR,
        LLMErrorCode.NETWORK_ERROR,
    }


def classify_provider_exception(exc: BaseException) -> LLMErrorCode:
    if isinstance(exc, LLMRequestError):
        return exc.code

    if isinstance(exc, (openai.APITimeoutError, anthropic.APITimeoutError)):
        return LLMErrorCode.TIMEOUT
    if isinstance(exc, (openai.APIConnectionError, anthropic.APIConnectionError)):
        return LLMErrorCode.NETWORK_ERROR
    if isinstance(exc, (openai.RateLimitError, anthropic.RateLimitError)):
        return LLMErrorCode.RATE_LIMIT
    if isinstance(exc, (openai.AuthenticationError, anthropic.AuthenticationError)):
        return LLMErrorCode.AUTH
    if isinstance(exc, (openai.PermissionDeniedError, anthropic.PermissionDeniedError)):
        return LLMErrorCode.PERMISSION
    if isinstance(exc, (openai.NotFoundError, anthropic.NotFoundError)):
        return LLMErrorCode.NOT_FOUND
    if isinstance(exc, (openai.ConflictError, anthropic.ConflictError)):
        return LLMErrorCode.CONFLICT
    if isinstance(exc, (openai.UnprocessableEntityError, anthropic.UnprocessableEntityError)):
        return LLMErrorCode.UNPROCESSABLE
    if isinstance(exc, (openai.BadRequestError, anthropic.BadRequestError)):
        return LLMErrorCode.BAD_REQUEST
    if isinstance(exc, (openai.InternalServerError, anthropic.InternalServerError)):
        return LLMErrorCode.SERVER_ERROR
    if isinstance(exc, (openai.APIResponseValidationError, anthropic.APIResponseValidationError)):
        return LLMErrorCode.RESPONSE_VALIDATION

    status_code = getattr(exc, "status_code", None)
    if isinstance(status_code, int):
        if status_code == 400:
            return LLMErrorCode.BAD_REQUEST
        if status_code == 401:
            return LLMErrorCode.AUTH
        if status_code == 403:
            return LLMErrorCode.PERMISSION
        if status_code == 404:
            return LLMErrorCode.NOT_FOUND
        if status_code == 409:
            return LLMErrorCode.CONFLICT
        if status_code == 422:
            return LLMErrorCode.UNPROCESSABLE
        if status_code == 429:
            return LLMErrorCode.RATE_LIMIT
        if 500 <= status_code <= 599:
            return LLMErrorCode.SERVER_ERROR

    return LLMErrorCode.UNKNOWN


def wrap_provider_exception(
    exc: BaseException,
    *,
    provider_kind: ProviderKind,
    profile_id: str,
    model: str | None,
    operation: str,
) -> LLMRequestError:
    code = classify_provider_exception(exc)
    status_code = getattr(exc, "status_code", None)
    request_id = getattr(exc, "request_id", None)
    retryable = is_retryable_error_code(code)
    message = str(exc) or exc.__class__.__name__
    details = {"operation": operation}
    return LLMRequestError(
        message,
        code=code,
        provider_kind=provider_kind,
        profile_id=profile_id,
        model=model,
        status_code=status_code if isinstance(status_code, int) else None,
        request_id=request_id if isinstance(request_id, str) else None,
        retryable=retryable,
        details=details,
        cause=exc,
    )
