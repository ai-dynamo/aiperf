# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import asyncio
import math
from unittest.mock import MagicMock, patch

import pytest

# botocore is an optional dependency (aiperf[aws]/aiperf[sagemaker]); skip this
# module entirely rather than failing collection when it isn't installed.
pytest.importorskip("botocore")

from aiperf.auth.base_signer import SignedRequest  # noqa: E402
from aiperf.auth.sigv4_signer import SigV4RequestSigner  # noqa: E402
from aiperf.common.enums import ModelSelectionStrategy  # noqa: E402
from aiperf.common.hooks import AIPerfHook  # noqa: E402
from aiperf.common.models.model_endpoint_info import (  # noqa: E402
    EndpointInfo,
    ModelEndpointInfo,
    ModelInfo,
    ModelListInfo,
)
from aiperf.plugin.enums import EndpointType  # noqa: E402


async def _pump_until(condition, *, max_iterations: int = 5000) -> None:
    """Pump the event loop until `condition()` is true or the iteration budget
    is exhausted.

    A fixed pump count is flaky under parallel test execution (xdist): CPU
    contention across workers changes how many event-loop iterations a given
    wall-clock/scheduling slice actually buys, so a background task under
    test may need meaningfully more yields to complete the same number of
    loop iterations. Polling a condition is only as slow as the real work
    takes, and still bounded so a genuinely stuck test fails instead of
    hanging.
    """
    for _ in range(max_iterations):
        if condition():
            return
        await asyncio.sleep(0)


def _make_model_endpoint(
    aws_region: str | None = "us-east-1",
    aws_service: str | None = "sagemaker",
    aws_profile: str | None = None,
) -> ModelEndpointInfo:
    return ModelEndpointInfo(
        models=ModelListInfo(
            models=[ModelInfo(name="test-model")],
            model_selection_strategy=ModelSelectionStrategy.ROUND_ROBIN,
        ),
        endpoint=EndpointInfo(
            type=EndpointType.CHAT,
            base_urls=["https://endpoint.sagemaker.us-east-1.amazonaws.com"],
            aws_region=aws_region,
            aws_service=aws_service,
            aws_profile=aws_profile,
        ),
    )


class TestSigV4RequestSignerInit:
    def test_stores_config(self) -> None:
        signer = SigV4RequestSigner(
            model_endpoint=_make_model_endpoint(
                aws_region="eu-west-1",
                aws_service="bedrock-runtime",
                aws_profile="prod",
            )
        )
        assert signer.region == "eu-west-1"
        assert signer.service == "bedrock-runtime"
        assert signer.profile == "prod"


class TestSigV4RequestSignerInitCredentials:
    @pytest.mark.asyncio
    async def test_init_credentials_success(self) -> None:
        signer = SigV4RequestSigner(model_endpoint=_make_model_endpoint())
        mock_creds = MagicMock()
        mock_session = MagicMock()
        mock_session.get_credentials.return_value = mock_creds

        with patch(
            "botocore.session.Session",
            return_value=mock_session,
        ):
            await signer._init_credentials()

        assert signer._credentials is mock_creds

    @pytest.mark.asyncio
    async def test_init_credentials_with_profile(self) -> None:
        signer = SigV4RequestSigner(
            model_endpoint=_make_model_endpoint(aws_profile="staging")
        )
        mock_session = MagicMock()
        mock_session.get_credentials.return_value = MagicMock()

        with patch(
            "botocore.session.Session",
            return_value=mock_session,
        ):
            await signer._init_credentials()

        mock_session.set_config_variable.assert_called_once_with("profile", "staging")

    @pytest.mark.asyncio
    async def test_init_credentials_no_profile_skips_set(self) -> None:
        signer = SigV4RequestSigner(
            model_endpoint=_make_model_endpoint(aws_profile=None)
        )
        mock_session = MagicMock()
        mock_session.get_credentials.return_value = MagicMock()

        with patch(
            "botocore.session.Session",
            return_value=mock_session,
        ):
            await signer._init_credentials()

        mock_session.set_config_variable.assert_not_called()

    @pytest.mark.asyncio
    async def test_init_credentials_none_raises(self) -> None:
        signer = SigV4RequestSigner(model_endpoint=_make_model_endpoint())
        mock_session = MagicMock()
        mock_session.get_credentials.return_value = None

        with (
            patch(
                "botocore.session.Session",
                return_value=mock_session,
            ),
            pytest.raises(ValueError, match="No AWS credentials found"),
        ):
            await signer._init_credentials()

    @pytest.mark.asyncio
    async def test_init_credentials_missing_botocore_raises_helpful_error(self) -> None:
        signer = SigV4RequestSigner(model_endpoint=_make_model_endpoint())
        with (
            patch.dict("sys.modules", {"botocore.session": None}),
            pytest.raises(ImportError, match="aiperf\\[aws\\]"),
        ):
            await signer._init_credentials()


def _setup_signer_for_sign(
    signer: SigV4RequestSigner,
    access_key: str = "AK",
    secret_key: str = "SK",
    token: str | None = None,
) -> tuple[MagicMock, MagicMock]:
    """Set up a signer with mock credentials and botocore classes for sign() tests."""
    mock_frozen = MagicMock(access_key=access_key, secret_key=secret_key, token=token)
    mock_creds = MagicMock()
    mock_creds.get_frozen_credentials.return_value = mock_frozen
    signer._credentials = mock_creds

    mock_sigv4_cls = MagicMock()
    mock_sigv4_cls.return_value.add_auth.side_effect = lambda r: None

    from botocore.awsrequest import AWSRequest
    from botocore.credentials import Credentials

    signer._SigV4Auth = mock_sigv4_cls
    signer._AWSRequest = AWSRequest
    signer._Credentials = Credentials

    return mock_creds, mock_sigv4_cls


class TestSigV4RequestSignerSign:
    @pytest.mark.asyncio
    async def test_sign_adds_authorization_header(self) -> None:
        signer = SigV4RequestSigner(model_endpoint=_make_model_endpoint())
        _, mock_sigv4_cls = _setup_signer_for_sign(
            signer,
            access_key="AKIAIOSFODNN7EXAMPLE",
            secret_key="wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY",
        )

        def add_auth_side_effect(request):
            request.headers["Authorization"] = "AWS4-HMAC-SHA256 Credential=..."
            request.headers["X-Amz-Date"] = "20260318T120000Z"

        mock_sigv4_cls.return_value.add_auth.side_effect = add_auth_side_effect

        headers = {"Content-Type": "application/json", "Host": "example.com"}
        body = b'{"prompt": "hello"}'
        result = await signer.sign("POST", "https://example.com/invoke", headers, body)

        assert isinstance(result, SignedRequest)
        assert "Authorization" in result.headers
        assert result.headers["Authorization"].startswith("AWS4-HMAC-SHA256")
        assert "X-Amz-Date" in result.headers
        assert result.url is None
        assert result.body is None

    @pytest.mark.asyncio
    async def test_sign_preserves_existing_headers(self) -> None:
        signer = SigV4RequestSigner(model_endpoint=_make_model_endpoint())
        _setup_signer_for_sign(
            signer, access_key="AKID", secret_key="SECRET", token="TOKEN"
        )

        headers = {"Content-Type": "application/json", "X-Custom": "value"}
        result = await signer.sign("GET", "https://example.com", headers, None)

        assert result.headers["Content-Type"] == "application/json"
        assert result.headers["X-Custom"] == "value"

    @pytest.mark.asyncio
    async def test_sign_calls_get_frozen_credentials_each_time(self) -> None:
        signer = SigV4RequestSigner(model_endpoint=_make_model_endpoint())
        mock_creds, _ = _setup_signer_for_sign(signer)

        await signer.sign("POST", "https://a.com", {}, b"")
        await signer.sign("POST", "https://b.com", {}, b"")

        assert mock_creds.get_frozen_credentials.call_count == 2

    @pytest.mark.asyncio
    async def test_sign_passes_correct_service_and_region(self) -> None:
        signer = SigV4RequestSigner(
            model_endpoint=_make_model_endpoint(
                aws_region="ap-southeast-1", aws_service="bedrock-runtime"
            )
        )
        _, mock_sigv4_cls = _setup_signer_for_sign(signer)

        await signer.sign("POST", "https://a.com", {}, b"")

        mock_sigv4_cls.assert_called_once()
        _, call_service, call_region = mock_sigv4_cls.call_args[0]
        assert call_service == "bedrock-runtime"
        assert call_region == "ap-southeast-1"


class TestSigV4RequestSignerPeriodicReresolution:
    """Long-running benchmarks need periodic full credential re-resolution
    (not just get_frozen_credentials()' per-call expiry refresh), so a
    rotated/leaked credential is picked up without waiting for the process
    to restart. Implemented as a real @background_task (not a lazy check
    inline in sign()), so it fires on a fixed cadence even during a quiet
    traffic lull, and runs off the event loop via asyncio.to_thread."""

    def test_reresolve_credentials_updates_credentials(self) -> None:
        signer = SigV4RequestSigner(model_endpoint=_make_model_endpoint())
        first_creds = MagicMock()
        second_creds = MagicMock()
        mock_session = MagicMock()
        mock_session.get_credentials.side_effect = [first_creds, second_creds]

        with patch("botocore.session.Session", return_value=mock_session):
            signer._reresolve_credentials()
            assert signer._credentials is first_creds

            signer._reresolve_credentials()
            assert signer._credentials is second_creds

    def test_interval_lambda_passes_through_positive_setting(self) -> None:
        signer = SigV4RequestSigner(model_endpoint=_make_model_endpoint())
        hook = next(
            h
            for h in signer.get_hooks(AIPerfHook.BACKGROUND_TASK)
            if h.func.__name__ == "_periodic_reresolve_credentials"
        )
        with patch(
            "aiperf.auth.sigv4_signer.Environment.AWS.CREDENTIAL_RERESOLVE_INTERVAL",
            42.0,
        ):
            assert hook.params.interval(signer) == 42.0

    def test_interval_lambda_is_infinite_when_disabled(self) -> None:
        signer = SigV4RequestSigner(model_endpoint=_make_model_endpoint())
        hook = next(
            h
            for h in signer.get_hooks(AIPerfHook.BACKGROUND_TASK)
            if h.func.__name__ == "_periodic_reresolve_credentials"
        )
        with patch(
            "aiperf.auth.sigv4_signer.Environment.AWS.CREDENTIAL_RERESOLVE_INTERVAL",
            0.0,
        ):
            assert hook.params.interval(signer) == math.inf

    @pytest.mark.asyncio
    async def test_background_task_reresolves_periodically(self) -> None:
        """End-to-end: start the signer's lifecycle with a short interval and
        confirm the background task re-resolves multiple times with no
        request ever being signed."""
        signer = SigV4RequestSigner(model_endpoint=_make_model_endpoint())
        mock_session = MagicMock()
        mock_session.get_credentials.return_value = MagicMock()

        with (
            patch("botocore.session.Session", return_value=mock_session),
            patch(
                "aiperf.auth.sigv4_signer.Environment.AWS.CREDENTIAL_RERESOLVE_INTERVAL",
                0.02,
            ),
        ):
            await signer.initialize()
            await signer.start()
            try:
                # One call from _init_credentials, plus several from the periodic task.
                await _pump_until(lambda: mock_session.get_credentials.call_count >= 3)
            finally:
                await signer.stop()

        assert mock_session.get_credentials.call_count >= 3

    # Note: there's no end-to-end "disabled interval never fires" lifecycle
    # test alongside the one above. This suite's global `no_sleep` autouse
    # fixture (tests/unit/conftest.py) collapses *any* asyncio.sleep delay -
    # including math.inf - to an instant real sleep(0), so it can't
    # distinguish "never fires" from "fires immediately" without a real
    # wall-clock wait. test_interval_lambda_is_infinite_when_disabled above
    # already proves the disabling logic directly and deterministically.

    @pytest.mark.asyncio
    async def test_background_task_survives_reresolution_failure(self) -> None:
        """A failed re-resolution attempt must not kill the background task -
        the framework's @background_task loop logs and continues by default -
        so the next scheduled attempt still runs."""
        signer = SigV4RequestSigner(model_endpoint=_make_model_endpoint())
        mock_session = MagicMock()
        call_count = 0

        def get_credentials(*args, **kwargs):
            # The pump condition below waits for call_count >= 3, so this
            # must never exhaust (a finite side_effect list would raise
            # StopIteration -> RuntimeError on any call past its length) -
            # only the *second* call (the first periodic re-resolution,
            # after _init_credentials) fails; every other call, however many
            # there end up being, succeeds.
            nonlocal call_count
            call_count += 1
            if call_count == 2:
                raise RuntimeError("credential_process failed")
            return MagicMock()

        mock_session.get_credentials.side_effect = get_credentials

        with (
            patch("botocore.session.Session", return_value=mock_session),
            patch(
                "aiperf.auth.sigv4_signer.Environment.AWS.CREDENTIAL_RERESOLVE_INTERVAL",
                0.02,
            ),
        ):
            await signer.initialize()
            await signer.start()
            try:
                # At least: _init_credentials, the failing periodic attempt,
                # and one more periodic attempt that succeeds - proving the
                # background task loop survived the failure and kept running.
                await _pump_until(lambda: call_count >= 3)
            finally:
                await signer.stop()

        assert call_count >= 3
