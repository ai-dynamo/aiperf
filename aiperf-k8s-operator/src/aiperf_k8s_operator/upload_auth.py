# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Domain-separated authorization for durable native result uploads."""

from __future__ import annotations

import base64
import hashlib
import hmac

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)

_KEY_DOMAIN = b"AIPERF-RESULTS-UPLOAD-KEY\x01"
_SIGNATURE_DOMAIN = b"AIPERF-RESULTS-UPLOAD-SIGNATURE\x01"
_READ_TOKEN_BYTES = 32


def _framed(domain: bytes, *fields: str | bytes) -> bytes:
    message = bytearray(domain)
    for field in fields:
        value = field if isinstance(field, bytes) else field.encode("utf-8")
        message.extend(len(value).to_bytes(8, "big"))
        message.extend(value)
    return bytes(message)


def _decode_urlsafe(value: str) -> bytes:
    padding = "=" * (-len(value) % 4)
    return base64.b64decode(value + padding, altchars=b"-_", validate=True)


def _upload_seed(
    bootstrap: bytes,
    namespace: str,
    job_id: str,
    run_id: str,
    object_uid: str,
) -> bytes:
    return hashlib.sha256(
        _framed(_KEY_DOMAIN, namespace, job_id, run_id, object_uid, bootstrap)
    ).digest()


def derive_upload_public_key(
    bootstrap: bytes,
    namespace: str,
    job_id: str,
    run_id: str,
    object_uid: str,
) -> str:
    """Derive the public verifier without retaining the private bootstrap bytes."""
    key = Ed25519PrivateKey.from_private_bytes(
        _upload_seed(bootstrap, namespace, job_id, run_id, object_uid)
    )
    public = key.public_key().public_bytes(
        serialization.Encoding.Raw, serialization.PublicFormat.Raw
    )
    return base64.urlsafe_b64encode(public).rstrip(b"=").decode("ascii")


def upload_signature_message(
    namespace: str,
    job_id: str,
    run_id: str,
    object_uid: str,
    kind: str,
    path: str,
    sha256: str,
    length: int,
) -> bytes:
    """Encode the byte-exact upload authority signed by the native sidecar."""
    return _framed(
        _SIGNATURE_DOMAIN,
        namespace,
        job_id,
        run_id,
        object_uid,
        kind,
        path,
        sha256,
        str(length),
    )


def verify_upload_signature(
    public_key: str,
    signature: str,
    namespace: str,
    job_id: str,
    run_id: str,
    object_uid: str,
    kind: str,
    path: str,
    sha256: str,
    length: int,
) -> bool:
    """Return whether one verifier authorizes the exact upload tuple."""
    try:
        verifier = Ed25519PublicKey.from_public_bytes(_decode_urlsafe(public_key))
        signature_bytes = _decode_urlsafe(signature)
        verifier.verify(
            signature_bytes,
            upload_signature_message(
                namespace,
                job_id,
                run_id,
                object_uid,
                kind,
                path,
                sha256,
                length,
            ),
        )
    except (InvalidSignature, ValueError):
        return False
    return True


def encode_results_read_token(raw: bytes) -> str:
    """Encode one dedicated results-read capability in its canonical wire form."""
    if len(raw) != _READ_TOKEN_BYTES:
        raise ValueError("results-read capability must contain exactly 32 bytes")
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii")


def results_read_token_sha256(raw: bytes) -> str:
    """Hash the raw dedicated results-read capability for verifier storage."""
    if len(raw) != _READ_TOKEN_BYTES:
        raise ValueError("results-read capability must contain exactly 32 bytes")
    return hashlib.sha256(raw).hexdigest()


def verify_results_read_token(expected_sha256: str, bearer: str) -> bool:
    """Verify one canonical bearer without retaining its private capability bytes."""
    try:
        raw = _decode_urlsafe(bearer)
    except (UnicodeEncodeError, ValueError):
        return False
    if len(raw) != _READ_TOKEN_BYTES or encode_results_read_token(raw) != bearer:
        return False
    actual = results_read_token_sha256(raw)
    return hmac.compare_digest(actual, expected_sha256)
