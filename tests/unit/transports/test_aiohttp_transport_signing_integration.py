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

_PAYLOAD: dict[str, object] = {
    "model": "test-model",
    "messages": [{"role": "user", "content": "Hi"}],
}


@pytest.fixture
def static_aws_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pin the credential chain so signing is deterministic apart from the clock.

    Deliberately does NOT freeze time. Doing so would mean patching a botocore
    internal, and the accessor moved: ``botocore.auth.get_current_datetime``
    only exists from 1.40.2, while this extra's floor is 1.34.0 because nothing
    in production needs anything newer. The tests below assert on the bytes
    handed to the signer instead of on the resulting signature, which is both
    the property actually under test and clock-independent.
    """
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "AKIAIOSFODNN7EXAMPLE")
    monkeypatch.setenv(
        "AWS_SECRET_ACCESS_KEY", "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY"
    )
    monkeypatch.delenv("AWS_SESSION_TOKEN", raising=False)
    monkeypatch.delenv("AWS_PROFILE", raising=False)


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


async def _signed_body(transport: AioHttpTransport, payload: object) -> bytes:
    """Return the exact bytes the signer hashed for one request.

    Spies on the real signer rather than patching it: what SigV4 hashes has to
    be the same bytes that go on the wire, and asserting on the resulting
    signature instead would only prove that indirectly -- while dragging in a
    dependency on freezing botocore's clock.
    """
    real_sign = transport.request_signer.sign
    hashed: list[bytes | None] = []

    async def spy(method, url, headers, body):
        hashed.append(body)
        return await real_sign(method, url, headers, body)

    transport.request_signer.sign = spy
    wire_body, _ = await _send_and_capture(transport, payload)

    assert len(hashed) == 1, f"expected exactly one signing call, got {len(hashed)}"
    # The bytes signed and the bytes sent must be the same object-equal value;
    # signing a body that is later re-encoded is the exact bug this guards.
    assert hashed[0] == wire_body
    return wire_body


@pytest.mark.asyncio
async def test_pre_encoded_bytes_are_signed_as_the_exact_wire_bytes(
    static_aws_credentials: None,
) -> None:
    """The PAYLOAD_BYTES fast path must hash the same bytes as the dict it was
    encoded from, which is what proves signing happens after the body is final.

    Signing before that branch resolves raises TypeError on a bytes payload;
    signing a re-encoded body would hash something the server never receives.
    """
    pre_encoded = orjson.dumps(_PAYLOAD)

    dict_body = await _signed_body(await _signed_transport(), _PAYLOAD)
    bytes_body = await _signed_body(await _signed_transport(), pre_encoded)

    assert dict_body == pre_encoded
    assert bytes_body == pre_encoded


@pytest.mark.asyncio
async def test_the_signed_request_carries_a_signature(
    static_aws_credentials: None,
) -> None:
    """Guards the test above: hashing the right bytes is only meaningful if the
    request is actually signed."""
    transport = await _signed_transport()
    _, headers = await _send_and_capture(transport, _PAYLOAD)

    assert headers["Authorization"].startswith("AWS4-HMAC-SHA256 ")
    assert "Signature=" in headers["Authorization"]


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
