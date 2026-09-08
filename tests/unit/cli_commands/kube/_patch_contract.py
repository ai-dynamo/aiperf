# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A ``patch_namespaced_custom_object`` fake that decodes bodies like the apiserver.

``kubernetes_asyncio`` picks the *first* content type it advertises when the
caller omits ``_content_type``, which is ``application/json-patch+json``. A CLI
verb that sends a merge-patch object without saying so therefore ships a dict
under a header promising an RFC 6902 operation array, and the apiserver answers
400 ``cannot unmarshal object into Go value of type []handlers.jsonPatchOp``.
A plain ``AsyncMock`` accepts that call happily, so the fake below reproduces
the server's decode step instead.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock

from kubernetes_asyncio.client.exceptions import ApiException

#: The content type ``kubernetes_asyncio`` selects when ``_content_type`` is omitted.
DEFAULT_PATCH_CONTENT_TYPE = "application/json-patch+json"

JSON_PATCH_DECODE_ERROR = (
    "error decoding patch: json: cannot unmarshal object into Go value of "
    "type []handlers.jsonPatchOp"
)


def decode_patch_like_apiserver(**kwargs: Any) -> dict[str, Any]:
    """Reject a body whose shape contradicts its patch content type."""
    content_type = kwargs.get("_content_type") or DEFAULT_PATCH_CONTENT_TYPE
    body = kwargs.get("body")

    if content_type == DEFAULT_PATCH_CONTENT_TYPE and not isinstance(body, list):
        raise ApiException(status=400, reason="Bad Request")
    if content_type == "application/merge-patch+json" and not isinstance(body, dict):
        raise ApiException(status=400, reason="Bad Request")
    return {}


def strict_patch_mock() -> AsyncMock:
    """An ``AsyncMock`` that 400s on a body/content-type mismatch."""
    return AsyncMock(side_effect=decode_patch_like_apiserver)
