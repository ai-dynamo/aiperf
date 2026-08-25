# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Runtime setting coverage for the Kubernetes CLI watch loop."""

from __future__ import annotations

from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

import aiperf.kubernetes.watch as watch_module


def _condition(*, status: str, message: str) -> dict[str, str]:
    return {
        "type": "Ready",
        "status": status,
        "reason": "ProbeResult",
        "message": message,
        "lastTransitionTime": "2026-08-04T12:00:00Z",
    }


def test_log_condition_updates_reports_same_length_status_transition() -> None:
    """Kubernetes updates list-map conditions in place instead of appending."""
    logger = MagicMock()
    previous = watch_module._log_condition_updates(
        logger,
        [_condition(status="False", message="controller is starting")],
        {},
        1.0,
    )
    logger.reset_mock()

    current = watch_module._log_condition_updates(
        logger,
        [_condition(status="True", message="controller is ready")],
        previous,
        2.0,
    )

    logger.info.assert_called_once()
    assert "PASS" in logger.info.call_args.args[0]
    assert "controller is ready" in logger.info.call_args.args[0]
    assert current != previous


def test_log_condition_updates_does_not_repeat_unchanged_condition() -> None:
    """Polling the same condition payload does not duplicate CLI output."""
    logger = MagicMock()
    condition = _condition(status="True", message="controller is ready")
    previous = watch_module._log_condition_updates(logger, [condition], {}, 1.0)
    logger.reset_mock()

    current = watch_module._log_condition_updates(logger, [condition], previous, 2.0)

    logger.info.assert_not_called()
    assert current == previous


@pytest.mark.asyncio
async def test_process_cr_poll_uses_configured_missing_cr_policy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A missing CR uses the configured warning grace and retry interval."""
    monkeypatch.setattr(
        watch_module,
        "K8sEnvironment",
        SimpleNamespace(
            WATCH=SimpleNamespace(
                NOT_FOUND_WARNING_GRACE_SECONDS=31.0,
                NOT_FOUND_RETRY_INTERVAL_SECONDS=17.0,
                CR_STATUS_LOG_INTERVAL_SECONDS=19.0,
            )
        ),
    )
    monkeypatch.setattr(watch_module, "_poll_cr_status", AsyncMock(return_value=None))
    sleep = AsyncMock()
    monkeypatch.setattr(watch_module.asyncio, "sleep", sleep)

    logger = MagicMock()
    result = await watch_module._process_cr_poll(
        MagicMock(),
        "namespace",
        "job",
        cli_logger=logger,
        state={"condition_signatures": {}, "last_status_log": 0.0},
        elapsed=31.1,
    )

    assert result is None
    logger.warning.assert_called_once()
    sleep.assert_awaited_once_with(17.0)


@pytest.mark.asyncio
async def test_process_cr_poll_uses_configured_status_log_interval(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A non-terminal CR emits its phase heartbeat at the configured cadence."""
    monkeypatch.setattr(
        watch_module,
        "K8sEnvironment",
        SimpleNamespace(
            WATCH=SimpleNamespace(
                NOT_FOUND_WARNING_GRACE_SECONDS=31.0,
                NOT_FOUND_RETRY_INTERVAL_SECONDS=17.0,
                CR_STATUS_LOG_INTERVAL_SECONDS=19.0,
            )
        ),
    )
    monkeypatch.setattr(
        watch_module,
        "_poll_cr_status",
        AsyncMock(return_value={"status": {"phase": "Running", "workers": {}}}),
    )
    log_heartbeat = MagicMock()
    monkeypatch.setattr(watch_module, "_log_phase_heartbeat", log_heartbeat)
    state = {"condition_signatures": {}, "last_status_log": 0.0}

    result = await watch_module._process_cr_poll(
        MagicMock(),
        "namespace",
        "job",
        cli_logger=MagicMock(),
        state=state,
        elapsed=19.0,
    )

    assert result is None
    assert state["last_status_log"] == 19.0
    log_heartbeat.assert_called_once()


@pytest.mark.asyncio
async def test_poll_until_terminal_uses_configured_cr_poll_interval(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Non-terminal CR polls wait for the configured poll interval."""
    monkeypatch.setattr(
        watch_module,
        "K8sEnvironment",
        SimpleNamespace(WATCH=SimpleNamespace(CR_POLL_INTERVAL_SECONDS=23.0)),
    )
    monkeypatch.setattr(
        watch_module,
        "_process_cr_poll",
        AsyncMock(side_effect=[None, {"phase": "Completed"}]),
    )
    sleep = AsyncMock()
    monkeypatch.setattr(watch_module.asyncio, "sleep", sleep)

    result = await watch_module._poll_until_terminal(
        MagicMock(), "namespace", "job", cli_logger=MagicMock(), timeout=60
    )

    assert result == {"phase": "Completed"}
    sleep.assert_awaited_once_with(23.0)


@pytest.mark.asyncio
async def test_watch_job_leaves_watchdog_cadences_to_watchdog_settings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The CLI does not override the watchdog's independently configurable cadence."""
    import aiperf.kubernetes.watchdog as watchdog_module

    constructor_kwargs: dict[str, object] = {}

    class _Watchdog:
        def __init__(self, *args: object, **kwargs: object) -> None:
            constructor_kwargs.update(kwargs)

        async def __aenter__(self) -> _Watchdog:
            return self

        async def __aexit__(self, *args: object) -> None:
            return None

    @asynccontextmanager
    async def _k8s_client(**kwargs: object):
        yield object()

    monkeypatch.setattr(watchdog_module, "BenchmarkWatchdog", _Watchdog)
    monkeypatch.setattr(watchdog_module, "K8sWatchdogSource", lambda api: object())
    monkeypatch.setattr(
        watch_module,
        "K8sEnvironment",
        SimpleNamespace(WATCH=SimpleNamespace(DEFAULT_TIMEOUT_SECONDS=73)),
    )
    monkeypatch.setattr(watch_module, "k8s_client", _k8s_client)
    monkeypatch.setattr(watch_module.client, "CustomObjectsApi", lambda api: object())
    monkeypatch.setattr(
        watch_module,
        "_poll_until_terminal",
        AsyncMock(return_value={"phase": "Completed"}),
    )

    result = await watch_module.watch_job("namespace", "job")

    assert result == {"phase": "Completed"}
    assert constructor_kwargs["timeout"] == 73
    assert "poll_interval" not in constructor_kwargs
    assert "status_interval" not in constructor_kwargs
