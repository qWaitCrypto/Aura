from __future__ import annotations

import os

from .errors import CredentialResolutionError
from .types import CredentialRef


def resolve_credential(credential_ref: CredentialRef) -> str:
    if credential_ref.kind == "env":
        value = os.environ.get(credential_ref.identifier)
        if not value:
            raise CredentialResolutionError(
                f"Missing required environment variable '{credential_ref.identifier}'.",
                credential_ref=credential_ref.to_redacted_string(),
            )
        return value

    if credential_ref.kind in {"inline", "plaintext"}:
        if not credential_ref.identifier:
            raise CredentialResolutionError(
                "Missing inline credential value.",
                credential_ref=credential_ref.to_redacted_string(),
            )
        return credential_ref.identifier

    raise CredentialResolutionError(
        f"Unsupported credential_ref kind '{credential_ref.kind}'.",
        credential_ref=credential_ref.to_redacted_string(),
    )
