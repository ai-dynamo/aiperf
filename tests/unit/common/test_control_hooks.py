# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, call, patch

import pytest

from aiperf.common.control_hooks import (
    PreparedEndpointControlHooks,
    prepare_endpoint_control_hooks,
    run_reset_kv_cache,
    start_server_profiler,
    stop_server_profiler,
)
from aiperf.common.control_plane_http import ControlPlaneHttpError
from aiperf.config.control_hooks import DEFAULT_CONTROL_HOOK_TIMEOUT_SECONDS
from aiperf.config.endpoint import EndpointConfig


def test_prepared_control_hooks_join_relative_paths_against_endpoint_origins() -> None:
    endpoint = EndpointConfig.model_validate(
        {
            "urls": ["http://127.0.0.1:8000/v1/chat/completions"],
            "reset_kv_cache": True,
            "server_profiler": True,
        }
    )
    hooks = prepare_endpoint_control_hooks(endpoint)
    assert hooks.reset_urls == ["http://127.0.0.1:8000/reset_prefix_cache"]
    assert hooks.profiler_start_urls == ["http://127.0.0.1:8000/start_profile"]
    assert hooks.profiler_stop_urls == ["http://127.0.0.1:8000/stop_profile"]


def test_prepared_control_hooks_timeout_does_not_inherit_large_endpoint_timeout() -> (
    None
):
    """reset_kv_cache/server_profiler timeouts must not inherit endpoint.timeout.

    endpoint.timeout defaults to 6 hours (tuned for inference requests). If
    reset_kv_cache/server_profiler fall back to it when their own
    timeout_seconds is unset, a stalled control-hook POST blocks for hours
    instead of failing fast (nvbugs 6671103).
    """
    endpoint = EndpointConfig.model_validate(
        {
            "urls": ["http://127.0.0.1:8000/v1/chat/completions"],
            "timeout": 6 * 60 * 60,
            "reset_kv_cache": True,
            "server_profiler": True,
        }
    )
    hooks = prepare_endpoint_control_hooks(endpoint)
    assert hooks.timeout_s == DEFAULT_CONTROL_HOOK_TIMEOUT_SECONDS
    assert hooks.profiler_timeout_s == DEFAULT_CONTROL_HOOK_TIMEOUT_SECONDS


def test_prepared_control_hooks_dedupe_same_origin_across_url_paths() -> None:
    endpoint = EndpointConfig.model_validate(
        {
            "urls": [
                "http://127.0.0.1:8000/v1/chat/completions",
                "http://127.0.0.1:8000/v1/completions",
                "http://other:9000/v1/chat/completions",
            ],
            "reset_kv_cache": True,
            "server_profiler": True,
        }
    )
    hooks = prepare_endpoint_control_hooks(endpoint)
    assert hooks.reset_urls == [
        "http://127.0.0.1:8000/reset_prefix_cache",
        "http://other:9000/reset_prefix_cache",
    ]
    assert hooks.profiler_start_urls == [
        "http://127.0.0.1:8000/start_profile",
        "http://other:9000/start_profile",
    ]
    assert hooks.profiler_stop_urls == [
        "http://127.0.0.1:8000/stop_profile",
        "http://other:9000/stop_profile",
    ]


def _hooks(
    *,
    reset_urls: list[str] | None = None,
    profiler_start_urls: list[str] | None = None,
    profiler_stop_urls: list[str] | None = None,
) -> PreparedEndpointControlHooks:
    return PreparedEndpointControlHooks(
        timeout_s=1.5,
        reset_urls=reset_urls or [],
        profiler_start_urls=profiler_start_urls or [],
        profiler_stop_urls=profiler_stop_urls or [],
        profiler_timeout_s=2.5,
    )


@pytest.mark.asyncio
async def test_run_reset_kv_cache_posts_to_each_reset_url() -> None:
    hooks = _hooks(
        reset_urls=[
            "http://a:8000/reset_prefix_cache",
            "http://b:8000/reset_prefix_cache",
        ]
    )
    headers = {"Authorization": "Bearer t"}
    with patch(
        "aiperf.common.control_hooks.control_plane_post",
        new_callable=AsyncMock,
    ) as post:
        await run_reset_kv_cache(hooks, headers)
        assert post.await_args_list == [
            call(
                url="http://a:8000/reset_prefix_cache",
                headers=headers,
                timeout_s=1.5,
            ),
            call(
                url="http://b:8000/reset_prefix_cache",
                headers=headers,
                timeout_s=1.5,
            ),
        ]


@pytest.mark.asyncio
async def test_run_reset_kv_cache_stops_after_first_failure() -> None:
    hooks = _hooks(
        reset_urls=[
            "http://a:8000/reset_prefix_cache",
            "http://b:8000/reset_prefix_cache",
        ]
    )
    headers = {"Authorization": "Bearer t"}
    error = ControlPlaneHttpError("reset failed")
    with (
        patch(
            "aiperf.common.control_hooks.control_plane_post",
            new_callable=AsyncMock,
            side_effect=error,
        ) as post,
        pytest.raises(ControlPlaneHttpError) as exc_info,
    ):
        await run_reset_kv_cache(hooks, headers)

    assert exc_info.value is error
    post.assert_awaited_once_with(
        url="http://a:8000/reset_prefix_cache",
        headers=headers,
        timeout_s=1.5,
    )


@pytest.mark.asyncio
async def test_stop_server_profiler_posts_to_each_stop_url() -> None:
    hooks = _hooks(
        profiler_stop_urls=[
            "http://a:8000/stop_profile",
            "http://b:8000/stop_profile",
        ]
    )
    headers = {"X-Test": "1"}
    with patch(
        "aiperf.common.control_hooks.control_plane_post",
        new_callable=AsyncMock,
    ) as post:
        await stop_server_profiler(hooks, headers)
        assert post.await_args_list == [
            call(
                url="http://a:8000/stop_profile",
                headers=headers,
                timeout_s=2.5,
            ),
            call(
                url="http://b:8000/stop_profile",
                headers=headers,
                timeout_s=2.5,
            ),
        ]


@pytest.mark.asyncio
async def test_start_server_profiler_posts_to_each_start_url() -> None:
    hooks = _hooks(
        profiler_start_urls=[
            "http://a:8000/start_profile",
            "http://b:8000/start_profile",
        ],
        profiler_stop_urls=[
            "http://a:8000/stop_profile",
            "http://b:8000/stop_profile",
        ],
    )
    headers = {"Authorization": "Bearer t"}
    with patch(
        "aiperf.common.control_hooks.control_plane_post",
        new_callable=AsyncMock,
    ) as post:
        await start_server_profiler(hooks, headers)
        assert post.await_args_list == [
            call(
                url="http://a:8000/start_profile",
                headers=headers,
                timeout_s=2.5,
            ),
            call(
                url="http://b:8000/start_profile",
                headers=headers,
                timeout_s=2.5,
            ),
        ]


@pytest.mark.asyncio
async def test_start_server_profiler_partial_failure_stops_started_then_reraises() -> (
    None
):
    hooks = _hooks(
        profiler_start_urls=[
            "http://a:8000/start_profile",
            "http://b:8000/start_profile",
        ],
        profiler_stop_urls=[
            "http://a:8000/stop_profile",
            "http://b:8000/stop_profile",
        ],
    )
    headers = {"Authorization": "Bearer t"}
    start_error = ControlPlaneHttpError("status 500 for http://b:8000/start_profile")

    async def post_side_effect(
        *, url: str, headers: dict[str, str], timeout_s: float
    ) -> None:
        del headers, timeout_s
        if url == "http://b:8000/start_profile":
            raise start_error

    with patch(
        "aiperf.common.control_hooks.control_plane_post",
        new_callable=AsyncMock,
        side_effect=post_side_effect,
    ) as post:
        with pytest.raises(ControlPlaneHttpError) as exc_info:
            await start_server_profiler(hooks, headers)

        assert exc_info.value is start_error
        assert post.await_args_list == [
            call(
                url="http://a:8000/start_profile",
                headers=headers,
                timeout_s=2.5,
            ),
            call(
                url="http://b:8000/start_profile",
                headers=headers,
                timeout_s=2.5,
            ),
            call(
                url="http://a:8000/stop_profile",
                headers=headers,
                timeout_s=2.5,
            ),
        ]


@pytest.mark.asyncio
async def test_stop_server_profiler_attempts_all_origins_then_aggregates() -> None:
    hooks = _hooks(
        profiler_stop_urls=[
            "http://a:8000/stop_profile",
            "http://b:8000/stop_profile",
            "http://c:8000/stop_profile",
        ]
    )
    headers = {"X-Test": "1"}
    calls: list[str] = []

    async def post_side_effect(
        *, url: str, headers: dict[str, str], timeout_s: float
    ) -> None:
        del headers, timeout_s
        calls.append(url)
        if url.endswith("a:8000/stop_profile") or url.endswith("c:8000/stop_profile"):
            raise ControlPlaneHttpError(f"fail {url}")

    with (
        patch(
            "aiperf.common.control_hooks.control_plane_post",
            new_callable=AsyncMock,
            side_effect=post_side_effect,
        ),
        pytest.raises(ControlPlaneHttpError, match="2 origin") as exc_info,
    ):
        await stop_server_profiler(hooks, headers)

    assert calls == [
        "http://a:8000/stop_profile",
        "http://b:8000/stop_profile",
        "http://c:8000/stop_profile",
    ]
    msg = str(exc_info.value)
    assert "fail http://a:8000/stop_profile" in msg
    assert "fail http://c:8000/stop_profile" in msg


@pytest.mark.asyncio
async def test_start_server_profiler_cleanup_failure_is_logged_then_reraises() -> None:
    hooks = _hooks(
        profiler_start_urls=[
            "http://a:8000/start_profile",
            "http://b:8000/start_profile",
        ],
        profiler_stop_urls=[
            "http://a:8000/stop_profile",
            "http://b:8000/stop_profile",
        ],
    )
    start_error = ControlPlaneHttpError("start b failed")

    async def post_side_effect(
        *, url: str, headers: dict[str, str], timeout_s: float
    ) -> None:
        del headers, timeout_s
        if url == "http://b:8000/start_profile":
            raise start_error
        if url == "http://a:8000/stop_profile":
            raise ControlPlaneHttpError("cleanup stop failed")

    with (
        patch(
            "aiperf.common.control_hooks.control_plane_post",
            new_callable=AsyncMock,
            side_effect=post_side_effect,
        ),
        patch("aiperf.common.control_hooks._logger") as logger,
    ):
        with pytest.raises(ControlPlaneHttpError) as exc_info:
            await start_server_profiler(hooks, {})
        assert exc_info.value is start_error
        assert any(
            "cleanup stop" in str(c.args[0])
            for c in logger.warning.call_args_list
            if c.args
        )


@pytest.mark.asyncio
async def test_start_server_profiler_cancellation_stops_started_then_reraises() -> None:
    hooks = _hooks(
        profiler_start_urls=[
            "http://a:8000/start_profile",
            "http://b:8000/start_profile",
        ],
        profiler_stop_urls=[
            "http://a:8000/stop_profile",
            "http://b:8000/stop_profile",
        ],
    )
    headers = {"Authorization": "Bearer t"}

    async def post_side_effect(
        *, url: str, headers: dict[str, str], timeout_s: float
    ) -> None:
        del headers, timeout_s
        if url == "http://b:8000/start_profile":
            raise asyncio.CancelledError

    with (
        patch(
            "aiperf.common.control_hooks.control_plane_post",
            new_callable=AsyncMock,
            side_effect=post_side_effect,
        ) as post,
        pytest.raises(asyncio.CancelledError),
    ):
        await start_server_profiler(hooks, headers)

    # The one successful start must still be rolled back despite cancellation.
    assert post.await_args_list[-1] == call(
        url="http://a:8000/stop_profile", headers=headers, timeout_s=2.5
    )


@pytest.mark.asyncio
async def test_stop_server_profiler_cancellation_still_stops_remaining_origins() -> (
    None
):
    hooks = _hooks(
        profiler_stop_urls=[
            "http://a:8000/stop_profile",
            "http://b:8000/stop_profile",
            "http://c:8000/stop_profile",
        ]
    )
    headers = {"Authorization": "Bearer t"}

    async def post_side_effect(
        *, url: str, headers: dict[str, str], timeout_s: float
    ) -> None:
        del headers, timeout_s
        if url == "http://a:8000/stop_profile":
            raise asyncio.CancelledError

    with (
        patch(
            "aiperf.common.control_hooks.control_plane_post",
            new_callable=AsyncMock,
            side_effect=post_side_effect,
        ) as post,
        pytest.raises(asyncio.CancelledError),
    ):
        await stop_server_profiler(hooks, headers)

    assert [c.kwargs["url"] for c in post.await_args_list] == [
        "http://a:8000/stop_profile",
        "http://b:8000/stop_profile",
        "http://c:8000/stop_profile",
    ]
