# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""End-to-end signing behaviour of ``AioHttpTransport.send_request``.

These drive the real ``SigV4RequestSigner`` against botocore rather than a mock,
because the two properties under test -- *what bytes* get hashed, and *which
headers* are in scope -- are exactly the parts a mock would paper over.

Both pin merge decisions that a future refactor could silently undo:

* Signing runs after ``body`` is final, so the pre-encoded-bytes fast path is
  signed too. Signing before that branch resolves would hash the wrong bytes.
* Signing runs on the dict ``build_headers`` returns, so the session-affinity
  headers it appends last are inside ``SignedHeaders``. Signing inside
  ``build_headers`` would leave them out and the server would reject them.
"""

import datetime
import re
from unittest.mock import AsyncMock

import orjson
import pytest

# botocore ships only in the optional aiperf[aws] extra, and CI installs
# "--extra test --no-dev" on Windows-on-ARM. Skip the module rather than
# failing collection there; these tests decode real eventstream frames.
pytest.importorskip("botocore")

from aiperf.common.models import RequestRecord
from aiperf.transports.aiohttp_transport import AioHttpTransport
from tests.unit.transports.conftest import create_model_endpoint_info
from tests.unit.transports.test_aiohttp_transport import create_request_info

_FROZEN = datetime.datetime(2026, 9, 8, 12, 0, 0, tzinfo=datetime.UTC)

_PAYLOAD: dict[str, object] = {
    "model": "test-model",
    "messages": [{"role": "user", "content": "Hi"}],
}


@pytest.fixture
def static_aws_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pin the credential chain and the clock so signatures are reproducible."""
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "AKIAIOSFODNN7EXAMPLE")
    monkeypatch.setenv(
        "AWS_SECRET_ACCESS_KEY", "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY"
    )
    monkeypatch.delenv("AWS_SESSION_TOKEN", raising=False)
    monkeypatch.delenv("AWS_PROFILE", raising=False)
    # SigV4 mixes a timestamp into the signature; freeze it so two signatures
    # taken moments apart are comparable. The symbol is botocore's consolidated
    # clock accessor, present since the 1.34.0 floor declared in pyproject.
    # Assert rather than skip: silently not freezing would make
    # test_pre_encoded_bytes_sign_identically_to_the_equivalent_dict flaky
    # instead of failing, and that test is a regression guard worth keeping loud.
    import botocore.auth

    assert hasattr(botocore.auth, "get_current_datetime"), (
        "botocore.auth.get_current_datetime is missing; the aws extra's "
        "botocore floor (>=1.34.0) no longer matches what these tests patch."
    )
    monkeypatch.setattr("botocore.auth.get_current_datetime", lambda: _FROZEN)


async def _signed_transport() -> AioHttpTransport:
    transport = AioHttpTransport(
        model_endpoint=create_model_endpoint_info(
            auth_type="sigv4",
            aws_region="us-west-2",
            aws_signing_service="sagemaker",
        )
    )
    await transport.initialize()
    transport.aiohttp_client.post_request = AsyncMock(return_value=RequestRecord())
    return transport


async def _send_and_capture(
    transport: AioHttpTransport, payload: object
) -> tuple[bytes, dict[str, str]]:
    """Send one request and return the (body, headers) handed to the client."""
    await transport.send_request(create_request_info(transport.model_endpoint), payload)
    call = transport.aiohttp_client.post_request.call_args
    return call[0][1], call[0][2]


def _signed_headers(authorization: str) -> list[str]:
    """Pull the SignedHeaders list out of a SigV4 Authorization value."""
    match = re.search(r"SignedHeaders=([^,]+)", authorization)
    assert match, f"no SignedHeaders in {authorization!r}"
    return match.group(1).split(";")


@pytest.mark.asyncio
async def test_pre_encoded_bytes_sign_identically_to_the_equivalent_dict(
    static_aws_credentials: None,
) -> None:
    """The PAYLOAD_BYTES fast path must produce the same signature as the dict
    it was encoded from -- proving the body is signed after encoding, verbatim."""
    dict_transport = await _signed_transport()
    dict_body, dict_headers = await _send_and_capture(dict_transport, _PAYLOAD)

    bytes_transport = await _signed_transport()
    pre_encoded = orjson.dumps(_PAYLOAD)
    bytes_body, bytes_headers = await _send_and_capture(bytes_transport, pre_encoded)

    # The pre-encoded bytes reach the wire untouched...
    assert bytes_body == pre_encoded
    assert dict_body == pre_encoded
    # ...and hash to the same signature as the dict form.
    assert bytes_headers["Authorization"] == dict_headers["Authorization"]
    assert "Signature=" in bytes_headers["Authorization"]


@pytest.mark.asyncio
async def test_a_different_body_yields_a_different_signature(
    static_aws_credentials: None,
) -> None:
    """Guards the test above: identical signatures must mean the body was
    actually hashed, not that signing ignores the body entirely."""
    transport = await _signed_transport()
    _, headers_a = await _send_and_capture(transport, _PAYLOAD)
    _, headers_b = await _send_and_capture(transport, {**_PAYLOAD, "model": "other"})

    assert headers_a["Authorization"] != headers_b["Authorization"]


@pytest.mark.asyncio
async def test_derived_session_headers_are_inside_signed_headers(
    static_aws_credentials: None,
) -> None:
    """``build_headers`` appends session-affinity headers last. Signing has to
    happen on its return value, or the server sees headers outside the scope."""
    transport = await _signed_transport()
    _, headers = await _send_and_capture(transport, _PAYLOAD)

    signed = _signed_headers(headers["Authorization"])

    # Derived by build_headers from the request context, not by the caller.
    assert "x-request-id" in signed
    assert "x-correlation-id" in signed
    # Appended last by build_headers, so the most likely to fall out of scope.
    assert "x-session-affinity" in signed


@pytest.mark.asyncio
async def test_unsigned_transport_adds_no_authorization_header() -> None:
    """Without --auth-type the transport must not touch Authorization."""
    transport = AioHttpTransport(model_endpoint=create_model_endpoint_info())
    await transport.initialize()
    transport.aiohttp_client.post_request = AsyncMock(return_value=RequestRecord())

    _, headers = await _send_and_capture(transport, _PAYLOAD)

    assert transport.request_signer is None
    assert "Authorization" not in headers


@pytest.mark.asyncio
async def test_signed_credentials_are_redacted_out_of_the_request_record(
    static_aws_credentials: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Signing puts long-lived credential material into the request headers;
    ``record.request_headers`` is exported to artifacts, so none of it may
    survive there.

    ``main`` already redacts these, so this is a regression pin rather than new
    behaviour. It is kept honest by asserting the same material *is* present in
    the headers actually sent -- otherwise the test would still pass if signing
    silently stopped happening.
    """
    monkeypatch.setenv("AWS_SESSION_TOKEN", "FwoGZXIvYXdzEDDDDDDDDDDDDDDDDDDD")

    transport = await _signed_transport()
    request_info = create_request_info(transport.model_endpoint)
    record = await transport.send_request(request_info, _PAYLOAD)

    sent_headers = transport.aiohttp_client.post_request.call_args[0][2]
    # Non-vacuity: the real request genuinely carries the secrets.
    assert "AKIAIOSFODNN7EXAMPLE" in sent_headers["Authorization"]
    assert "Signature=" in sent_headers["Authorization"]
    assert sent_headers["X-Amz-Security-Token"].startswith("FwoGZXIv")

    exported = record.request_headers or {}
    flattened = repr(exported)
    assert "AKIAIOSFODNN7EXAMPLE" not in flattened
    assert "Signature=" not in flattened
    assert "FwoGZXIv" not in flattened


class TestSignedRequestsDoNotFollowRedirects:
    """aiohttp strips ``Authorization`` when a redirect crosses origins, but it
    has no such rule for custom headers -- ``X-Amz-Security-Token`` would be
    replayed to the redirect target, which is a bearer credential going somewhere
    the user never configured (and possibly over cleartext).

    Following a redirect would fail anyway: SigV4 signs the Host header, so the
    replayed signature is invalid for the new origin. Refusing to follow loses
    nothing and closes the leak.
    """

    @pytest.mark.asyncio
    async def test_redirects_are_disabled_for_signed_requests(
        self, static_aws_credentials: None
    ) -> None:
        transport = await _signed_transport()
        await _send_and_capture(transport, _PAYLOAD)

        kwargs = transport.aiohttp_client.post_request.call_args.kwargs
        assert kwargs.get("allow_redirects") is False

    @pytest.mark.asyncio
    async def test_redirects_are_disabled_on_the_cancellation_path_too(
        self, static_aws_credentials: None
    ) -> None:
        """``post_request`` routes cancellable requests through a different
        helper, so the guard has to survive that branch as well."""
        transport = await _signed_transport()
        request_info = create_request_info(transport.model_endpoint)
        request_info.cancel_after_ns = 10_000_000_000

        await transport.send_request(request_info, _PAYLOAD)

        kwargs = transport.aiohttp_client.post_request.call_args.kwargs
        assert kwargs.get("allow_redirects") is False

    @pytest.mark.asyncio
    async def test_unsigned_requests_keep_following_redirects(self) -> None:
        """No credentials on the wire, so the existing behaviour is untouched."""
        transport = AioHttpTransport(model_endpoint=create_model_endpoint_info())
        await transport.initialize()
        transport.aiohttp_client.post_request = AsyncMock(return_value=RequestRecord())

        await _send_and_capture(transport, _PAYLOAD)

        kwargs = transport.aiohttp_client.post_request.call_args.kwargs
        assert "allow_redirects" not in kwargs
