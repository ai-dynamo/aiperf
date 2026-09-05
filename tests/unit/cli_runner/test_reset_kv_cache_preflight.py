# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aiperf.cli_runner._single_run import maybe_reset_kv_cache_before_run


@pytest.mark.asyncio
async def test_reset_invoked_when_enabled() -> None:
    run = MagicMock()
    run.cfg.endpoint.reset_kv_cache = MagicMock()
    run.cfg.endpoint.urls = ["http://127.0.0.1:8000"]
    run.cfg.endpoint.headers = {"X-Custom": "1"}
    run.cfg.endpoint.api_key = "sk-test"
    run.cfg.endpoint.type = "chat"

    with (
        patch(
            "aiperf.cli_runner._single_run.prepare_endpoint_control_hooks",
            return_value=MagicMock(
                reset_urls=["http://127.0.0.1:8000/reset_prefix_cache"]
            ),
        ) as prepare,
        patch(
            "aiperf.cli_runner._single_run.run_reset_kv_cache",
            new_callable=AsyncMock,
        ) as reset,
        patch(
            "aiperf.cli_runner._single_run.auth_headers_for_endpoint",
            return_value={"Authorization": "Bearer sk-test", "X-Custom": "1"},
        ) as auth,
    ):
        await maybe_reset_kv_cache_before_run(run)
        prepare.assert_called_once_with(run.cfg.endpoint)
        auth.assert_called_once_with(run.cfg.endpoint)
        reset.assert_awaited_once()
        assert reset.await_args.args[1] == {
            "Authorization": "Bearer sk-test",
            "X-Custom": "1",
        }


@pytest.mark.asyncio
async def test_reset_skipped_when_disabled() -> None:
    run = MagicMock()
    run.cfg.endpoint.reset_kv_cache = None
    with patch(
        "aiperf.cli_runner._single_run.run_reset_kv_cache",
        new_callable=AsyncMock,
    ) as reset:
        await maybe_reset_kv_cache_before_run(run)
        reset.assert_not_awaited()


@pytest.mark.asyncio
async def test_reset_precedes_inference_on_shared_recorder() -> None:
    """Reset via maybe_reset_kv_cache_before_run must precede inference on the mock."""
    from urllib.parse import urlsplit

    from aiperf_mock_server.app import asgi_app
    from aiperf_mock_server.control_state import control_state
    from httpx import ASGITransport, AsyncClient

    from aiperf.common.control_plane_http import ControlPlaneHttpError
    from aiperf.config.endpoint import EndpointConfig

    control_state.reset()
    transport = ASGITransport(app=asgi_app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:

        async def asgi_control_plane_post(
            *, url: str, headers: dict[str, str], timeout_s: float
        ) -> None:
            del timeout_s
            path = urlsplit(url).path or "/"
            resp = await client.post(path, headers=headers)
            if not (200 <= resp.status_code < 300):
                raise ControlPlaneHttpError(
                    f"control_plane POST {url} failed with status {resp.status_code}"
                )

        run = MagicMock()
        run.cfg.endpoint = EndpointConfig.model_validate(
            {
                "urls": ["http://test/v1/chat/completions"],
                "reset_kv_cache": True,
                "type": "chat",
            }
        )

        with patch(
            "aiperf.common.control_hooks.control_plane_post",
            new=asgi_control_plane_post,
        ):
            await maybe_reset_kv_cache_before_run(run)

        inference = await client.post(
            "/v1/chat/completions",
            json={
                "model": "mock-model",
                "messages": [{"role": "user", "content": "hi"}],
                "max_tokens": 8,
                "stream": False,
            },
        )
        assert inference.status_code == 200
        assert control_state.reset_count == 1
        assert control_state.events.index("reset") < control_state.events.index(
            "inference"
        )


def test_run_single_benchmark_resets_before_bootstrap() -> None:
    """_run_single_benchmark must reset once before services start."""
    from aiperf.cli_runner._single_run import _run_single_benchmark
    from aiperf.plugin.enums import UIType

    run = MagicMock()
    run.cfg.ui_type = UIType.SIMPLE
    call_order: list[str] = []

    with (
        patch("os._exit"),
        patch(
            "aiperf.config.resolution.resolvers.build_default_resolver_chain",
            return_value=MagicMock(),
        ),
        patch(
            "aiperf.cli_runner._single_run._configure_multiprocessing_start_method",
        ),
        patch(
            "aiperf.cli_runner._single_run._configure_tokenizer_preload",
        ),
        patch(
            "aiperf.cli_runner._single_run._setup_ui_queues",
            return_value=None,
        ),
        patch(
            "aiperf.cli_runner._single_run.maybe_reset_kv_cache_before_run",
            new_callable=AsyncMock,
            side_effect=lambda *_a, **_k: call_order.append("reset"),
        ) as reset,
        patch(
            "aiperf.common.bootstrap.bootstrap_and_run_service",
            side_effect=lambda **_k: call_order.append("bootstrap"),
        ),
    ):
        _run_single_benchmark(run)

    reset.assert_awaited_once_with(run)
    assert call_order == ["reset", "bootstrap"]


def test_run_single_benchmark_reset_failure_is_fatal() -> None:
    """reset_kv_cache failure must abort before bootstrap."""
    from aiperf.cli_runner._single_run import _run_single_benchmark
    from aiperf.plugin.enums import UIType

    run = MagicMock()
    run.cfg.ui_type = UIType.SIMPLE

    with (
        patch("os._exit"),
        patch(
            "aiperf.config.resolution.resolvers.build_default_resolver_chain",
            return_value=MagicMock(),
        ),
        patch(
            "aiperf.cli_runner._single_run._configure_multiprocessing_start_method",
        ),
        patch(
            "aiperf.cli_runner._single_run._configure_tokenizer_preload",
        ),
        patch(
            "aiperf.cli_runner._single_run._setup_ui_queues",
            return_value=None,
        ),
        patch(
            "aiperf.cli_runner._single_run.maybe_reset_kv_cache_before_run",
            new_callable=AsyncMock,
            side_effect=RuntimeError("reset failed"),
        ),
        patch(
            "aiperf.cli_runner._single_run.raise_startup_error_and_exit",
            side_effect=SystemExit(1),
        ) as exit_fn,
        patch(
            "aiperf.common.bootstrap.bootstrap_and_run_service",
        ) as bootstrap,
        pytest.raises(SystemExit),
    ):
        _run_single_benchmark(run)

    exit_fn.assert_called_once()
    assert "reset_kv_cache failed before benchmark start" in str(
        exit_fn.call_args.args[0]
    )
    assert exit_fn.call_args.kwargs.get("title") == "Control Hook Error"
    bootstrap.assert_not_called()
