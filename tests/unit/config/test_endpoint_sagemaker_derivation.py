# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Everything `--sagemaker-endpoint-name` derives.

The target UX is one flag plus a region:

    aiperf profile -m my-model --sagemaker-endpoint-name my-ep --aws-region us-west-2

Transport, signer, signing scope and base URL all fall out of that. These tests
pin each derivation separately so a failure names which one broke, and they run
against ``EndpointConfig`` rather than a built transport so the derived values
are also what lands in exported config artifacts.
"""

import pytest
from pydantic import ValidationError
from pytest import param

from aiperf.config.endpoint import EndpointConfig
from aiperf.plugin.enums import RequestSignerType, TransportType


def _endpoint(**overrides) -> EndpointConfig:
    data: dict = {
        "type": "chat",
        "sagemaker": {"endpoint_name": "my-ep"},
        "aws_region": "us-west-2",
    }
    data.update(overrides)
    return EndpointConfig.model_validate(data)


class TestSelectorDerivation:
    def test_endpoint_name_selects_the_sagemaker_transport(self) -> None:
        assert _endpoint().transport == TransportType.SAGEMAKER

    def test_sagemaker_transport_selects_sigv4_signing(self) -> None:
        assert _endpoint().auth_type == RequestSignerType.SIGV4

    def test_a_non_sagemaker_transport_with_an_endpoint_name_names_the_conflict(
        self,
    ) -> None:
        """``--sagemaker-endpoint-name`` selects the SageMaker transport, so
        naming a different transport is a contradiction, not an override.

        It was already rejected -- but by whichever unrelated guard fired
        first, and both blame the wrong flag. With a region it was "--aws-region
        has no effect unless --auth-type is set to 'sigv4'"; without one,
        "SageMaker endpoints require --aws-region". Neither mentions the
        transport, which is the actual problem.

        Asserting on the message is the whole point: a bare
        ``pytest.raises(ValidationError)`` passes against either of those and
        proves nothing.
        """
        for region in ("us-west-2", None):
            with pytest.raises(ValidationError) as exc_info:
                _endpoint(transport="http", aws_region=region)

            message = str(exc_info.value)
            assert "--transport" in message, message
            assert "--sagemaker-endpoint-name" in message, message
            assert "has no effect" not in message, message

    def test_missing_region_names_the_flag(self) -> None:
        with pytest.raises(ValidationError, match="--aws-region"):
            _endpoint(aws_region=None)

    def test_signing_scope_is_not_required_from_the_user(self) -> None:
        """The signing name is derivable from the transport's botocore service
        id, so requiring it here would be busywork -- unlike a bare
        `--auth-type sigv4` against an arbitrary host, which still needs it.

        Validating at all is the assertion: a missing scope would raise.
        """
        assert _endpoint().aws_service is None

    def test_explicit_signing_scope_still_wins(self) -> None:
        cfg = _endpoint(aws_service="execute-api")
        assert cfg.aws_service == "execute-api"


class TestBaseUrlDerivation:
    def test_url_is_derived_from_the_region(self) -> None:
        assert _endpoint().urls == ["https://runtime.sagemaker.us-west-2.amazonaws.com"]

    def test_china_region_uses_the_china_suffix(self) -> None:
        cfg = _endpoint(aws_region="cn-north-1")
        assert cfg.urls == ["https://runtime.sagemaker.cn-north-1.amazonaws.com.cn"]

    def test_explicit_url_wins(self) -> None:
        """Required for VPC/PrivateLink endpoints and custom domains."""
        cfg = _endpoint(
            urls=["https://vpce-123.sagemaker.us-west-2.vpce.amazonaws.com"]
        )
        assert cfg.urls == ["https://vpce-123.sagemaker.us-west-2.vpce.amazonaws.com"]

    @pytest.mark.parametrize(
        "region",
        [
            param("evil.com/x", id="path"),
            param("evil.com#", id="fragment"),
            param("us-west-2.evil.com", id="extra-labels"),
        ],
    )  # fmt: skip
    def test_a_region_that_would_change_the_host_is_rejected(self, region: str) -> None:
        """``evil.com/x`` derived ``https://runtime.sagemaker.evil.com/x.amazonaws.com``:
        a SigV4-signed request, session token included, sent to a non-AWS host."""
        with pytest.raises(ValidationError, match="--aws-region"):
            _endpoint(aws_region=region)

    def test_a_camel_case_region_is_checked_too(self) -> None:
        with pytest.raises(ValidationError, match="--aws-region"):
            EndpointConfig.model_validate(
                {
                    "type": "chat",
                    "sagemaker": {"endpointName": "my-ep"},
                    "awsRegion": "evil.com/x",
                }
            )

    def test_the_region_is_not_checked_when_it_builds_no_host(self) -> None:
        """With an explicit ``--url`` the region is only the SigV4 credential
        scope, where #771's generic signing path accepts non-AWS scopes such
        as Cloudflare R2's ``auto``."""
        cfg = _endpoint(aws_region="auto", urls=["https://example.com"])
        assert cfg.aws_region == "auto"


class TestSageMakerFieldsAreOptional:
    def test_multi_model_and_component_fields_default_to_none(self) -> None:
        cfg = _endpoint()
        assert cfg.sagemaker.target_model is None
        assert cfg.sagemaker.inference_component_name is None
        assert cfg.sagemaker.target_variant is None


class TestCamelCaseInputDerivesTheSameWay:
    """``BaseConfig`` sets ``alias_generator=to_camel`` with
    ``populate_by_name=True``, so field validation accepts ``endpointName`` and
    ``awsRegion``. The before-validator runs earlier and saw the raw mapping, so
    it read snake_case only.

    Both generated CRDs and the published JSON schema declare camelCase
    exclusively -- the exact shape the CRD's relaxed ``required`` was changed to
    admit -- so Kubernetes input either failed with "urls: Field required" or was
    spuriously rejected for a missing ``--aws-region`` that was in fact supplied.
    """

    def test_camel_case_endpoint_name_and_region_derive_the_url(self) -> None:
        cfg = EndpointConfig.model_validate(
            {
                "type": "chat",
                "sagemaker": {"endpointName": "my-ep"},
                "awsRegion": "us-west-2",
            }
        )

        assert cfg.urls == ["https://runtime.sagemaker.us-west-2.amazonaws.com"]
        assert cfg.sagemaker.endpoint_name == "my-ep"
        assert cfg.transport == TransportType.SAGEMAKER

    def test_camel_case_without_a_region_still_names_the_flag(self) -> None:
        """The region requirement must survive the alias handling, or camelCase
        input would sail past a guard that snake_case input trips."""
        with pytest.raises(ValidationError, match="--aws-region"):
            EndpointConfig.model_validate(
                {"type": "chat", "sagemaker": {"endpointName": "my-ep"}}
            )

    def test_camel_case_transport_conflict_still_names_the_conflict(self) -> None:
        with pytest.raises(ValidationError, match="--sagemaker-endpoint-name"):
            EndpointConfig.model_validate(
                {
                    "type": "chat",
                    "sagemaker": {"endpointName": "my-ep"},
                    "awsRegion": "us-west-2",
                    "transport": "http",
                }
            )


class TestEndpointNameIsRequired:
    """``--transport sagemaker`` without an endpoint name would otherwise build
    ``/endpoints//invocations`` and fail remotely with nothing pointing at the
    cause."""

    def test_sagemaker_transport_without_endpoint_name_is_rejected(self) -> None:
        with pytest.raises(ValidationError, match="--sagemaker-endpoint-name"):
            EndpointConfig.model_validate(
                {
                    "type": "chat",
                    "transport": "sagemaker",
                    "aws_region": "us-west-2",
                    "urls": ["https://runtime.sagemaker.us-west-2.amazonaws.com"],
                }
            )

    def test_empty_endpoint_name_is_rejected(self) -> None:
        with pytest.raises(ValidationError, match="--sagemaker-endpoint-name"):
            EndpointConfig.model_validate(
                {
                    "type": "chat",
                    "transport": "sagemaker",
                    "aws_region": "us-west-2",
                    "urls": ["https://runtime.sagemaker.us-west-2.amazonaws.com"],
                    "sagemaker": {"endpoint_name": ""},
                }
            )

    def test_an_explicit_path_makes_the_endpoint_name_unnecessary(self) -> None:
        """``get_url`` uses ``path`` verbatim in that case, so no name is needed."""
        cfg = EndpointConfig.model_validate(
            {
                "type": "chat",
                "transport": "sagemaker",
                "aws_region": "us-west-2",
                "urls": ["https://runtime.sagemaker.us-west-2.amazonaws.com"],
                "path": "/endpoints/other/invocations",
            }
        )
        assert cfg.path == "/endpoints/other/invocations"

    def test_without_urls_the_error_names_the_flag_not_the_field(self) -> None:
        """With no `urls`, field validation fails first with "urls: Field
        required" and no after-validator ever runs, so a YAML user never learns
        which flag was missing. The check has to happen before validation too."""
        with pytest.raises(ValidationError, match="--sagemaker-endpoint-name"):
            EndpointConfig.model_validate(
                {"type": "chat", "transport": "sagemaker", "aws_region": "us-west-2"}
            )


class TestReadinessProbeIsRejected:
    """The readiness probe builds `/v1/models` or the endpoint type's OpenAI
    path, never `/endpoints/{name}/invocations`, so against SageMaker it signs
    and sends a request to a route the endpoint does not serve. A signed 401/403
    there aborts pre-flight; any other 4xx is taken as "ready".

    The tutorial already documented `--wait-for-model-timeout` as refused for
    SageMaker; nothing refused it."""

    def test_wait_for_model_timeout_is_rejected_with_the_flag_named(self) -> None:
        with pytest.raises(ValidationError, match="--wait-for-model-timeout"):
            _endpoint(wait_for_model_timeout=30)

    def test_the_default_disabled_probe_is_still_accepted(self) -> None:
        """The probe is off by default, so the one-flag quick start must not trip
        this."""
        assert _endpoint().wait_for_model_timeout == 0
