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
from aiperf.config.control_hooks import (
    DEFAULT_CONTROL_HOOK_TIMEOUT_SECONDS,
    DEFAULT_RESET_KV_CACHE_MAX_RETRY_SECONDS,
    DEFAULT_RETRY_BACKOFF_CAP_SECONDS,
    DEFAULT_RETRY_BACKOFF_MULTIPLIER,
    DEFAULT_RETRY_BACKOFF_SECONDS,
)
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
    instead of failing fast.
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


def test_prepared_control_hooks_reset_max_retry_seconds_defaults() -> None:
    endpoint = EndpointConfig.model_validate(
        {
            "urls": ["http://127.0.0.1:8000/v1/chat/completions"],
            "reset_kv_cache": True,
        }
    )
    hooks = prepare_endpoint_control_hooks(endpoint)
    assert hooks.reset_max_retry_seconds == DEFAULT_RESET_KV_CACHE_MAX_RETRY_SECONDS


def test_prepared_control_hooks_reset_max_retry_seconds_honors_override() -> None:
    endpoint = EndpointConfig.model_validate(
        {
            "urls": ["http://127.0.0.1:8000/v1/chat/completions"],
            "reset_kv_cache": {"max_retry_seconds": 5.0},
        }
    )
    hooks = prepare_endpoint_control_hooks(endpoint)
    assert hooks.reset_max_retry_seconds == 5.0


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
    reset_max_retry_seconds: float = 5.0,
) -> PreparedEndpointControlHooks:
    return PreparedEndpointControlHooks(
        timeout_s=1.5,
        reset_urls=reset_urls or [],
        profiler_start_urls=profiler_start_urls or [],
        profiler_stop_urls=profiler_stop_urls or [],
        profiler_timeout_s=2.5,
        reset_max_retry_seconds=reset_max_retry_seconds,
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
async def test_run_reset_kv_cache_does_not_retry_non_retryable_failure() -> None:
    hooks = _hooks(reset_urls=["http://a:8000/reset_prefix_cache"])
    headers = {"Authorization": "Bearer t"}
    error = ControlPlaneHttpError("status 500", retryable=False)
    with (
        patch(
            "aiperf.common.control_hooks.control_plane_post",
            new_callable=AsyncMock,
            side_effect=error,
        ) as post,
        patch(
            "aiperf.common.control_hooks.asyncio.sleep", new_callable=AsyncMock
        ) as sleep,
        pytest.raises(ControlPlaneHttpError) as exc_info,
    ):
        await run_reset_kv_cache(hooks, headers)

    assert exc_info.value is error
    post.assert_awaited_once()
    sleep.assert_not_awaited()


@pytest.mark.asyncio
async def test_run_reset_kv_cache_retries_retryable_failure_then_succeeds() -> None:
    hooks = _hooks(
        reset_urls=["http://a:8000/reset_prefix_cache"],
        reset_max_retry_seconds=100.0,
    )
    headers = {"Authorization": "Bearer t"}
    retryable_error = ControlPlaneHttpError("timeout", retryable=True)
    with (
        patch(
            "aiperf.common.control_hooks.control_plane_post",
            new_callable=AsyncMock,
            side_effect=[retryable_error, retryable_error, None],
        ) as post,
        patch(
            "aiperf.common.control_hooks.asyncio.sleep", new_callable=AsyncMock
        ) as sleep,
    ):
        await run_reset_kv_cache(hooks, headers)

    assert post.await_count == 3
    assert sleep.await_args_list == [
        call(DEFAULT_RETRY_BACKOFF_SECONDS),
        call(DEFAULT_RETRY_BACKOFF_SECONDS * DEFAULT_RETRY_BACKOFF_MULTIPLIER),
    ]


@pytest.mark.asyncio
async def test_run_reset_kv_cache_gives_up_once_retry_budget_exhausted() -> None:
    hooks = _hooks(
        reset_urls=["http://a:8000/reset_prefix_cache"],
        reset_max_retry_seconds=DEFAULT_RETRY_BACKOFF_SECONDS
        + DEFAULT_RETRY_BACKOFF_SECONDS * DEFAULT_RETRY_BACKOFF_MULTIPLIER,
    )
    headers = {"Authorization": "Bearer t"}
    retryable_error = ControlPlaneHttpError("timeout", retryable=True)

    class _FakeClock:
        def __init__(self) -> None:
            self.now = 0.0

        def monotonic(self) -> float:
            return self.now

        async def sleep(self, seconds: float) -> None:
            self.now += seconds

    clock = _FakeClock()
    with (
        patch(
            "aiperf.common.control_hooks.control_plane_post",
            new_callable=AsyncMock,
            side_effect=retryable_error,
        ) as post,
        patch("aiperf.common.control_hooks.time.monotonic", clock.monotonic),
        patch("aiperf.common.control_hooks.asyncio.sleep", clock.sleep),
        pytest.raises(ControlPlaneHttpError) as exc_info,
    ):
        await run_reset_kv_cache(hooks, headers)

    assert exc_info.value is retryable_error
    assert post.await_count == 2


@pytest.mark.asyncio
async def test_run_reset_kv_cache_backoff_growth_caps_at_defined_ceiling() -> None:
    hooks = _hooks(
        reset_urls=["http://a:8000/reset_prefix_cache"],
        reset_max_retry_seconds=1000.0,
    )
    headers = {"Authorization": "Bearer t"}
    retryable_error = ControlPlaneHttpError("timeout", retryable=True)
    # Enough retryable failures for backoff to exceed the cap by uncapped growth
    # (1 -> 2 -> 4 -> 8 -> 16), then one success.
    with (
        patch(
            "aiperf.common.control_hooks.control_plane_post",
            new_callable=AsyncMock,
            side_effect=[retryable_error] * 5 + [None],
        ),
        patch(
            "aiperf.common.control_hooks.asyncio.sleep", new_callable=AsyncMock
        ) as sleep,
    ):
        await run_reset_kv_cache(hooks, headers)

    expected_backoffs = []
    backoff = DEFAULT_RETRY_BACKOFF_SECONDS
    for _ in range(5):
        expected_backoffs.append(backoff)
        backoff = min(
            backoff * DEFAULT_RETRY_BACKOFF_MULTIPLIER,
            DEFAULT_RETRY_BACKOFF_CAP_SECONDS,
        )
    assert expected_backoffs[-1] == DEFAULT_RETRY_BACKOFF_CAP_SECONDS  # cap is reached
    assert sleep.await_args_list == [call(b) for b in expected_backoffs]


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
