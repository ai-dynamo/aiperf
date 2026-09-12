# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""SageMaker Runtime ``InvokeEndpoint`` / ``InvokeEndpointWithResponseStream``.

A thin subclass of :class:`AioHttpTransport`, because SageMaker differs from a
plain OpenAI-compatible server in only four ways -- auth, URL path, headers, and
response framing -- and never in the request *body*. A vLLM container behind a
SageMaker endpoint accepts the same OpenAI-shaped JSON it would over plain HTTP,
so no endpoint plugin is involved and ``--endpoint-type chat`` (or
``completions``, or ``embeddings``) keeps working unchanged.

Auth is inherited: signing lives on :class:`AioHttpTransport` because it is a
cross-cut that also applies to a plain server behind API Gateway. Response
framing is inherited too: ``AioHttpClient`` picks its reader from the response
content type. Only the URL and headers are overridden here.

``url_schemes`` is deliberately empty so ``detect_transport_from_url`` can never
select this transport from a URL scheme -- it would otherwise compete with the
built-in ``http`` transport for ``https://``. Selection is explicit
(``--transport sagemaker``) or derived from ``--sagemaker-endpoint-name``.
"""

from __future__ import annotations

from typing import ClassVar
from urllib.parse import quote, urlsplit, urlunsplit

from aiperf.common.models import RequestInfo
from aiperf.config.endpoint import RequestContentType
from aiperf.plugin.enums import TransportType
from aiperf.plugin.schema.schemas import TransportMetadata
from aiperf.transports.aiohttp_transport import AioHttpTransport, _has_http_scheme

# Header names taken directly from botocore's sagemaker-runtime service model
# (InvokeEndpointInput / InvokeEndpointWithResponseStreamInput members'
# `location: header` / `locationName`), not guessed - AWS uses non-obvious
# header names here, and the two operations' shapes are NOT identical:
# - Accept uses a different header name for each operation ("Accept" for
#   InvokeEndpoint, "X-Amzn-SageMaker-Accept" for the streaming variant).
# - TargetModel exists only on InvokeEndpointInput, not on the streaming
#   operation's input shape at all.
HEADER_TARGET_MODEL = "X-Amzn-SageMaker-Target-Model"
HEADER_TARGET_VARIANT = "X-Amzn-SageMaker-Target-Variant"
HEADER_INFERENCE_ID = "X-Amzn-SageMaker-Inference-Id"
HEADER_INFERENCE_COMPONENT = "X-Amzn-SageMaker-Inference-Component"
HEADER_ACCEPT_STREAMING = "X-Amzn-SageMaker-Accept"

_INVOCATIONS_PATH = "invocations"
_INVOCATIONS_STREAM_PATH = "invocations-response-stream"


class SageMakerTransport(AioHttpTransport):
    """Invoke a model hosted behind a SageMaker Runtime endpoint."""

    botocore_service_id: ClassVar[str] = "sagemaker-runtime"
    """botocore service id whose model supplies the SigV4 signing name.

    Not the signing name itself: this API signs as ``sagemaker``, without the
    ``-runtime`` suffix. Resolving it through the service model rather than
    hardcoding the string is what makes the same mechanism correct for Bedrock
    (``bedrock-runtime`` signs as ``bedrock``).
    """

    @classmethod
    def metadata(cls) -> TransportMetadata:
        """Return SageMaker transport metadata.

        ``url_schemes`` is empty by design -- see the module docstring.
        """
        return TransportMetadata(
            transport_type=TransportType.SAGEMAKER,
            url_schemes=[],
        )

    def get_url(self, request_info: RequestInfo) -> str:
        """Build the SageMaker invocations URL.

        The endpoint plugin's own ``endpoint_path`` (``/v1/chat/completions``
        and friends) is deliberately ignored: SageMaker routes on the request
        *body*, not the path, and every endpoint type is invoked through the
        same ``/endpoints/{name}/invocations`` path. This is the single most
        surprising thing about the integration for a first-time user, so it is
        called out prominently in the tutorial as well.

        Streaming picks the ``-response-stream`` variant, which is a distinct
        SageMaker operation rather than a content-negotiation flag -- getting it
        wrong yields a buffered response with no stream, not an error.

        Args:
            request_info: Request context with model endpoint info

        Returns:
            Complete HTTPS URL for the invocations call
        """
        endpoint_info = request_info.model_endpoint.endpoint

        raw_base_url = endpoint_info.get_url(request_info.url_index)
        if not _has_http_scheme(raw_base_url):
            raw_base_url = f"https://{raw_base_url}"

        split = urlsplit(raw_base_url)
        base_path = split.path.rstrip("/")

        # custom_endpoint is the escape hatch, matching the base class's
        # semantics exactly: `is not None` so an empty string means "append
        # nothing" and stays distinguishable from unset.
        if endpoint_info.custom_endpoint is not None:
            sub_path = endpoint_info.custom_endpoint.lstrip("/")
        else:
            name = quote(endpoint_info.sagemaker_endpoint_name or "", safe="")
            operation = (
                _INVOCATIONS_STREAM_PATH
                if endpoint_info.streaming
                else _INVOCATIONS_PATH
            )
            sub_path = f"endpoints/{name}/{operation}"

        # Inherited: collapses an overlap so pasting a full invocations URL into
        # --url does not produce /endpoints/x/invocations/endpoints/x/invocations.
        new_path = self._dedup_path_overlap(base_path, sub_path)

        return urlunsplit(
            (split.scheme, split.netloc, new_path, split.query, split.fragment)
        )

    def get_transport_headers(self, request_info: RequestInfo) -> dict[str, str]:
        """Build SageMaker's InvokeEndpoint header set.

        Differs from the base HTTP transport in two ways that are easy to miss:
        the streaming Accept header has its own name
        (``X-Amzn-SageMaker-Accept``), and ``Accept`` is ``application/json``
        even when streaming -- the eventstream framing is the operation's, not
        something negotiated through Accept.

        Args:
            request_info: Request context with model endpoint info

        Returns:
            Content-Type, Accept, and any configured SageMaker routing headers
        """
        endpoint = request_info.model_endpoint.endpoint
        streaming = endpoint.streaming

        accept_header = HEADER_ACCEPT_STREAMING if streaming else "Accept"
        headers: dict[str, str] = {accept_header: "application/json"}

        content_type = endpoint.request_content_type
        if content_type != RequestContentType.MULTIPART_FORM_DATA:
            headers["Content-Type"] = (
                content_type or RequestContentType.APPLICATION_JSON
            )

        if endpoint.sagemaker_inference_component_name:
            headers[HEADER_INFERENCE_COMPONENT] = (
                endpoint.sagemaker_inference_component_name
            )

        if endpoint.sagemaker_target_variant:
            headers[HEADER_TARGET_VARIANT] = endpoint.sagemaker_target_variant

        # TargetModel is absent from the streaming operation's input shape
        # entirely, so sending it there is not merely redundant -- it is not
        # part of that API. On multi-model endpoints TargetModel *is* the model
        # identifier to invoke, which is the same concept as aiperf's own
        # per-request model name, hence the fallback.
        if not streaming:
            target_model = endpoint.sagemaker_target_model or (
                (request_info.turns[-1].model if request_info.turns else None)
                or self.model_endpoint.primary_model_name
            )
            if target_model:
                headers[HEADER_TARGET_MODEL] = target_model

        # Always the request id, never user-overridable: it is what correlates
        # an aiperf record with SageMaker's own captured inference data.
        if request_info.x_request_id:
            headers[HEADER_INFERENCE_ID] = request_info.x_request_id

        return headers
