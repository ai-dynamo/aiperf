# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``post_request`` must pass caller kwargs through on both of its branches.

It chooses between ``_request`` and ``_request_with_cancellation`` depending on
whether a cancellation deadline is set. The second branch dropped ``**kwargs``,
so anything a caller passed -- ``allow_redirects=False`` for signed requests
being the case that matters -- silently applied to uncancellable requests only.
A guard that holds for some requests and not others is worse than none, because
it reads as covered.
"""

from unittest.mock import AsyncMock, patch

import pytest

from aiperf.common.models.record_models import RequestRecord
from aiperf.transports.aiohttp_client import AioHttpClient


@pytest.mark.asyncio
async def test_kwargs_reach_the_plain_request_path() -> None:
    client = AioHttpClient(timeout=600.0)
    with patch.object(
        client, "_request", new=AsyncMock(return_value=RequestRecord())
    ) as mock_request:
        await client.post_request(
            "https://x.example.com", b"{}", {}, allow_redirects=False
        )

    assert mock_request.call_args.kwargs["allow_redirects"] is False


@pytest.mark.asyncio
async def test_kwargs_reach_the_cancellation_path() -> None:
    client = AioHttpClient(timeout=600.0)
    with patch.object(
        client,
        "_request_with_cancellation",
        new=AsyncMock(return_value=RequestRecord()),
    ) as mock_cancel:
        await client.post_request(
            "https://x.example.com",
            b"{}",
            {},
            cancel_after_ns=10_000_000_000,
            allow_redirects=False,
        )

    assert mock_cancel.call_args.kwargs["allow_redirects"] is False
