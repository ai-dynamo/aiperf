# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""SageMaker InvokeEndpoint(WithResponseStream) header construction.

Used by ``AioHttpTransport`` via ``--auth-type sigv4 --aws-service sagemaker``.
"""

from __future__ import annotations

from typing import Any

# Header names taken directly from botocore's sagemaker-runtime service model
# (InvokeEndpointInput / InvokeEndpointWithResponseStreamInput members'
# `location: header` / `locationName`), not guessed - AWS uses non-obvious
# header names here, and the two operations' shapes are NOT identical:
# - Accept uses a different header name for each operation ("Accept" for
#   InvokeEndpoint, "X-Amzn-SageMaker-Accept" for the streaming variant).
# - TargetModel exists only on InvokeEndpointInput, not on the streaming
#   operation's input shape at all.
HEADER_TARGET_MODEL = "X-Amzn-SageMaker-Target-Model"
HEADER_INFERENCE_ID = "X-Amzn-SageMaker-Inference-Id"
HEADER_INFERENCE_COMPONENT = "X-Amzn-SageMaker-Inference-Component"
HEADER_ACCEPT_STREAMING = "X-Amzn-SageMaker-Accept"


def sagemaker_optional_headers(
    endpoint: Any,
    *,
    streaming: bool,
    model_name: str | None = None,
    x_request_id: str | None = None,
) -> dict[str, str]:
    """Build the optional SageMaker headers (TargetModel, InferenceComponentName,
    InferenceId) from the given endpoint config.

    ``TargetModel`` defaults to ``model_name`` when not set explicitly - on
    SageMaker Multi-Model Endpoints, TargetModel *is* the model identifier to
    invoke, the same concept as AIPerf's own per-request model name.
    ``InferenceId`` is always ``x_request_id``, not user-overridable.
    ``InferenceComponentName`` is passed through only when explicitly
    configured.

    ``endpoint`` is duck-typed (``EndpointInfo`` or ``EndpointConfig``, both
    of which carry the same ``sagemaker_*`` field names) to avoid a
    transport-layer dependency on a specific config model.
    """
    headers: dict[str, str] = {}
    if endpoint.sagemaker_inference_component_name:
        headers[HEADER_INFERENCE_COMPONENT] = (
            endpoint.sagemaker_inference_component_name
        )
    if not streaming:
        target_model = endpoint.sagemaker_target_model or model_name
        if target_model:
            headers[HEADER_TARGET_MODEL] = target_model
    if x_request_id:
        headers[HEADER_INFERENCE_ID] = x_request_id
    return headers
