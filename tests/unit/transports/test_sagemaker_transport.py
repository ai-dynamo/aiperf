# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""URL and header construction for the SageMaker transport.

These are the two things the transport actually overrides; everything else
(signing, connection reuse, cancellation, response decoding) is inherited and
covered by the base transport's own tests.
"""

import pytest
from pytest import param

from aiperf.common.enums import ModelSelectionStrategy
from aiperf.common.models import ModelEndpointInfo, ModelInfo, ModelListInfo
from aiperf.common.models.model_endpoint_info import EndpointInfo
from aiperf.plugin.enums import EndpointType, TransportType
from aiperf.transports.aws.sagemaker import (
    HEADER_ACCEPT_STREAMING,
    HEADER_INFERENCE_COMPONENT,
    HEADER_INFERENCE_ID,
    HEADER_TARGET_MODEL,
    HEADER_TARGET_VARIANT,
    SageMakerTransport,
)
from tests.unit.transports.test_aiohttp_transport import create_request_info


def _model_endpoint(
    *,
    base_url: str = "https://runtime.sagemaker.us-west-2.amazonaws.com",
    endpoint_name: str | None = "my-ep",
    streaming: bool = False,
    custom_endpoint: str | None = None,
    target_model: str | None = None,
    inference_component: str | None = None,
    target_variant: str | None = None,
    model_name: str = "test-model",
) -> ModelEndpointInfo:
    return ModelEndpointInfo(
        models=ModelListInfo(
            models=[ModelInfo(name=model_name)],
            model_selection_strategy=ModelSelectionStrategy.ROUND_ROBIN,
        ),
        endpoint=EndpointInfo(
            type=EndpointType.CHAT,
            base_urls=[base_url],
            custom_endpoint=custom_endpoint,
            streaming=streaming,
            transport=TransportType.SAGEMAKER,
            sagemaker_endpoint_name=endpoint_name,
            sagemaker_target_model=target_model,
            sagemaker_inference_component_name=inference_component,
            sagemaker_target_variant=target_variant,
        ),
    )


def _url(**kwargs) -> str:
    endpoint = _model_endpoint(**kwargs)
    transport = SageMakerTransport(model_endpoint=endpoint)
    return transport.get_url(create_request_info(endpoint))


def _headers(**kwargs) -> dict[str, str]:
    endpoint = _model_endpoint(**kwargs)
    transport = SageMakerTransport(model_endpoint=endpoint)
    return transport.get_transport_headers(create_request_info(endpoint))


class TestUrlConstruction:
    def test_non_streaming_uses_invocations(self) -> None:
        assert _url() == (
            "https://runtime.sagemaker.us-west-2.amazonaws.com/endpoints/my-ep/invocations"
        )

    def test_streaming_uses_the_response_stream_operation(self) -> None:
        """A distinct SageMaker operation, not a content-negotiation flag:
        getting it wrong yields a buffered response, not an error."""
        assert _url(streaming=True).endswith(
            "/endpoints/my-ep/invocations-response-stream"
        )

    def test_endpoint_name_is_percent_encoded(self) -> None:
        """Endpoint names are user-supplied and land in a path segment."""
        assert "/endpoints/we%2Fird/invocations" in _url(endpoint_name="we/ird")

    def test_pasting_a_full_invocations_url_does_not_double_the_path(self) -> None:
        """People copy the whole URL out of the console; the inherited
        path-overlap dedup has to absorb that."""
        url = _url(
            base_url="https://runtime.sagemaker.us-west-2.amazonaws.com/endpoints/my-ep/invocations"
        )
        assert url.count("/endpoints/my-ep/invocations") == 1

    def test_custom_endpoint_overrides_everything(self) -> None:
        assert _url(custom_endpoint="/totally/custom").endswith("/totally/custom")

    def test_endpoint_plugin_path_is_never_used(self) -> None:
        """SageMaker routes on the request body, so /v1/chat/completions must
        not appear even though --endpoint-type chat is in play. This is the
        single most surprising part of the integration."""
        assert "/v1/chat/completions" not in _url()


class TestHeaders:
    def test_streaming_uses_the_sagemaker_specific_accept_header(self) -> None:
        headers = _headers(streaming=True)
        assert headers[HEADER_ACCEPT_STREAMING] == "application/json"
        assert "Accept" not in headers

    def test_non_streaming_uses_the_plain_accept_header(self) -> None:
        headers = _headers(streaming=False)
        assert headers["Accept"] == "application/json"
        assert HEADER_ACCEPT_STREAMING not in headers

    def test_target_model_defaults_to_the_request_model(self) -> None:
        """On multi-model endpoints TargetModel *is* the model to invoke."""
        assert _headers(model_name="llama-3")[HEADER_TARGET_MODEL] == "llama-3"

    def test_explicit_target_model_wins(self) -> None:
        assert _headers(target_model="pinned.tar.gz")[HEADER_TARGET_MODEL] == (
            "pinned.tar.gz"
        )

    def test_target_model_is_absent_when_streaming(self) -> None:
        """TargetModel is not part of the streaming operation's input shape."""
        assert HEADER_TARGET_MODEL not in _headers(streaming=True, target_model="m")

    @pytest.mark.parametrize(
        "streaming", [param(False, id="unary"), param(True, id="streaming")]
    )
    def test_inference_component_is_sent_for_both_operations(
        self, streaming: bool
    ) -> None:
        headers = _headers(streaming=streaming, inference_component="comp-1")
        assert headers[HEADER_INFERENCE_COMPONENT] == "comp-1"

    @pytest.mark.parametrize(
        "streaming", [param(False, id="unary"), param(True, id="streaming")]
    )
    def test_target_variant_is_sent_for_both_operations(self, streaming: bool) -> None:
        headers = _headers(streaming=streaming, target_variant="variant-b")
        assert headers[HEADER_TARGET_VARIANT] == "variant-b"

    def test_optional_routing_headers_are_absent_when_unset(self) -> None:
        headers = _headers()
        assert HEADER_INFERENCE_COMPONENT not in headers
        assert HEADER_TARGET_VARIANT not in headers

    def test_inference_id_carries_the_request_id(self) -> None:
        """This is what correlates an aiperf record with SageMaker's own
        captured inference data."""
        assert _headers()[HEADER_INFERENCE_ID] == "test-request-id"


class TestTransportSelection:
    def test_url_schemes_is_empty_so_autodetection_cannot_pick_it(self) -> None:
        """Otherwise it would compete with the built-in http transport for
        https:// and selection would depend on plugin ordering."""
        assert SageMakerTransport.metadata().url_schemes == []
        assert SageMakerTransport.metadata().transport_type == TransportType.SAGEMAKER

    def test_declares_the_botocore_service_id_not_the_signing_name(self) -> None:
        """The signing name (`sagemaker`) is resolved from this id through
        botocore's service model, which is what makes Bedrock work for free."""
        assert SageMakerTransport.botocore_service_id == "sagemaker-runtime"
