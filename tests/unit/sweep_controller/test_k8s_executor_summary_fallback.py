# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for K8sChildJobExecutor's summary-metrics fallback path.

The primary path reads ``AIPerfJob.status.summary``. When that is empty (the
``CompletedBeforeMonitor`` race, or a completion-handler bug that skips the
summary write), the executor falls back to fetching ``profile_export_aiperf.json``
from the operator's PVC-backed results API. The previous implementation hit
the child controller-pod's results-sidecar, which fails after the JobSet is
torn down.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import orjson
import pytest

from aiperf.sweep_controller.k8s_executor import K8sChildJobExecutor


def _sweep_cr() -> dict[str, Any]:
    return {
        "metadata": {
            "name": "sweep-conc-demo",
            "namespace": "aiperf-benchmarks",
            "uid": "49a887c0-ca76-400d-9e94-5a17f26245c5",
        },
        "spec": {"image": "x:latest", "benchmark": {}},
    }


def _executor() -> K8sChildJobExecutor:
    return K8sChildJobExecutor(
        api=MagicMock(),
        sweep=_sweep_cr(),
        with_trial_suffix=False,
    )


def _child(
    *, summary: dict[str, Any] | None = None, run_epoch: int | None = 1700000000
) -> dict[str, Any]:
    status: dict[str, Any] = {}
    if summary is not None:
        status["summary"] = summary
    if run_epoch is not None:
        status["runEpoch"] = run_epoch
    return {
        "metadata": {"name": "sweep-x-v00-t0", "namespace": "aiperf-benchmarks"},
        "status": status,
    }


@pytest.mark.asyncio
async def test_pull_summary_uses_status_summary_when_populated() -> None:
    """Primary path: a populated ``status.summary`` short-circuits the HTTP fallback."""
    populated = {"request_throughput": {"avg": 100.0, "unit": "req/s"}}
    child = _child(summary=populated)

    executor = _executor()
    executor._fetch_summary_from_operator = AsyncMock()  # type: ignore[method-assign]

    result = await executor._pull_summary_metrics(child)

    assert "request_throughput" in result
    executor._fetch_summary_from_operator.assert_not_called()


@pytest.mark.asyncio
async def test_pull_summary_falls_back_to_operator_when_empty() -> None:
    """Empty ``status.summary`` → operator API fallback fires and result is returned."""
    child = _child(summary={})  # explicitly empty
    expected = {"request_throughput": MagicMock()}

    executor = _executor()
    executor._fetch_summary_from_operator = AsyncMock(return_value=expected)  # type: ignore[method-assign]

    result = await executor._pull_summary_metrics(child)

    assert result is expected
    executor._fetch_summary_from_operator.assert_awaited_once_with(child)


@pytest.mark.asyncio
async def test_pull_summary_recovery_log_lists_metric_tags(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Recovery log must enumerate the recovered metric tags.

    Pre-fix the log only said ``(19 metrics)`` — useless for diagnosing an
    SLA-filter ``observed: null`` collapse, because oncall couldn't tell
    whether the offending tag (e.g. ``time_to_first_token``) was actually
    among the 19. The fix lists tags so a single grep proves the mismatch.
    """
    from aiperf.sweep_controller import k8s_executor as mod

    child = _child(summary={})
    recovered = {
        "request_throughput": MagicMock(),
        "request_latency": MagicMock(),
        "time_to_first_token": MagicMock(),
    }

    executor = _executor()
    executor._fetch_summary_from_operator = AsyncMock(return_value=recovered)  # type: ignore[method-assign]

    with caplog.at_level("INFO", logger=mod.logger.name):
        await executor._pull_summary_metrics(child)

    log_text = "\n".join(r.message for r in caplog.records)
    assert "recovered summary via operator API" in log_text
    assert "(3 metrics)" in log_text
    # Each tag must appear so SLA-filter mismatch is grep-able.
    for tag in recovered:
        assert tag in log_text, f"tag {tag!r} missing from recovery log: {log_text}"


@pytest.mark.asyncio
async def test_pull_summary_returns_empty_when_fallback_also_fails() -> None:
    """Both paths empty → return ``{}`` (caller treats as unrecoverable, run still success)."""
    child = _child(summary={})
    executor = _executor()
    executor._fetch_summary_from_operator = AsyncMock(return_value={})  # type: ignore[method-assign]

    result = await executor._pull_summary_metrics(child)

    assert result == {}


# ============================================================
# Race-aware refresh: child terminal but operator hasn't reconciled yet.
# ============================================================


@pytest.mark.asyncio
async def test_pull_summary_refreshes_when_summary_and_epoch_unset(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Both ``status.summary`` and ``status.runEpoch`` empty → refresh CR until populated.

    Reproduces the fast-adaptive-probe race: the child reached ``phase=Completed``
    so ``_wait_until_terminal`` returns, but the operator's monitor tick has
    not yet stamped ``status.summary`` or ``runEpoch``. Pre-fix, ``_pull_summary_metrics``
    saw both empty, hit ``_fetch_summary_from_operator`` which short-circuited
    on missing epoch → returned ``{}`` → SLA bracket collapsed to ``observed: null``.
    """
    from aiperf.sweep_controller import k8s_executor as mod

    initial = _child(summary={}, run_epoch=None)
    populated_summary = {
        "time_to_first_token": {"avg": 150.0, "p95": 200.0, "unit": "ms"}
    }
    refreshed = _child(summary=populated_summary, run_epoch=1700000001)

    executor = _executor()
    # First refresh returns populated child — short-circuits before exhausting
    # the SUMMARY_RACE_REFRESH_ATTEMPTS budget.
    executor._try_read_child = AsyncMock(return_value=refreshed)  # type: ignore[method-assign]
    executor._fetch_summary_from_operator = AsyncMock()  # type: ignore[method-assign]

    with caplog.at_level("INFO", logger=mod.logger.name):
        result = await executor._pull_summary_metrics(initial)

    assert "time_to_first_token" in result
    executor._try_read_child.assert_awaited()
    # Race-grace path took the primary, NOT the operator-API fallback.
    executor._fetch_summary_from_operator.assert_not_called()
    log_text = "\n".join(r.message for r in caplog.records)
    assert "status.summary populated after" in log_text


@pytest.mark.asyncio
async def test_pull_summary_refresh_falls_back_to_operator_when_only_epoch_arrives() -> (
    None
):
    """Refresh sees runEpoch populate but summary still empty → operator-API fallback fires.

    The other race ordering: the operator stamped ``runEpoch`` before
    ``status.summary``, so the primary path is still empty but the operator
    API now has a resolvable URL.
    """
    initial = _child(summary={}, run_epoch=None)
    refreshed_with_epoch = _child(summary={}, run_epoch=1700000002)
    expected = {"request_throughput": MagicMock()}

    executor = _executor()
    executor._try_read_child = AsyncMock(return_value=refreshed_with_epoch)  # type: ignore[method-assign]
    executor._fetch_summary_from_operator = AsyncMock(return_value=expected)  # type: ignore[method-assign]

    result = await executor._pull_summary_metrics(initial)

    assert result is expected
    # Fallback was called with the REFRESHED child (has runEpoch), not the
    # initial one — otherwise the operator URL builder skips on missing epoch.
    executor._fetch_summary_from_operator.assert_awaited_once()
    call_args = executor._fetch_summary_from_operator.await_args
    assert call_args.args[0]["status"]["runEpoch"] == 1700000002


@pytest.mark.asyncio
async def test_pull_summary_refresh_exhausted_returns_empty(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """All refreshes still empty → fallback fires once with original child → ``{}``."""
    from aiperf.sweep_controller import k8s_executor as mod

    initial = _child(summary={}, run_epoch=None)

    executor = _executor()
    # Every refresh returns the same empty child.
    executor._try_read_child = AsyncMock(return_value=initial)  # type: ignore[method-assign]
    executor._fetch_summary_from_operator = AsyncMock(return_value={})  # type: ignore[method-assign]

    with caplog.at_level("WARNING", logger=mod.logger.name):
        result = await executor._pull_summary_metrics(initial)

    assert result == {}
    # The retry exhausted: SUMMARY_RACE_REFRESH_ATTEMPTS reads were attempted.
    assert executor._try_read_child.await_count == mod.SUMMARY_RACE_REFRESH_ATTEMPTS, (
        f"expected {mod.SUMMARY_RACE_REFRESH_ATTEMPTS} refreshes, "
        f"got {executor._try_read_child.await_count}"
    )
    # Final fallback was called once.
    executor._fetch_summary_from_operator.assert_awaited_once()


@pytest.mark.asyncio
async def test_pull_summary_skips_refresh_when_run_epoch_already_set() -> None:
    """A child with ``runEpoch`` set but empty ``status.summary`` skips the refresh
    loop and goes straight to the operator-API fallback.

    This is the legacy fallback path — preserves pre-fix behavior so existing
    callers that rely on the immediate operator-API hit don't pay the 12s
    grace tax.
    """
    child = _child(summary={}, run_epoch=1700000000)
    expected = {"request_throughput": MagicMock()}

    executor = _executor()
    executor._try_read_child = AsyncMock()  # type: ignore[method-assign]
    executor._fetch_summary_from_operator = AsyncMock(return_value=expected)  # type: ignore[method-assign]

    result = await executor._pull_summary_metrics(child)

    assert result is expected
    # No refresh — runEpoch already populated, so skip the grace loop.
    executor._try_read_child.assert_not_called()
    executor._fetch_summary_from_operator.assert_awaited_once_with(child)


# ============================================================
# Operator API fallback
# ============================================================


def _profile_export_payload() -> dict[str, Any]:
    """Mirrors the real on-disk shape: top-level metric tags as dicts."""
    return {
        "schema_version": "1.0",
        "aiperf_version": "0.8.0",
        "request_throughput": {"avg": 95.8, "unit": "req/s"},
        "request_latency": {"avg": 2601.1, "p99": 2603.7, "unit": "ms"},
    }


class _FakeResponse:
    def __init__(self, status: int, body: bytes) -> None:
        self.status = status
        self._body = body

    async def __aenter__(self) -> _FakeResponse:
        return self

    async def __aexit__(self, *exc: Any) -> None:
        return None

    async def read(self) -> bytes:
        return self._body


class _FakeSession:
    """Captures the URL passed to ``session.get`` for assertion."""

    def __init__(self, response: _FakeResponse) -> None:
        self._response = response
        self.last_url: str | None = None

    async def __aenter__(self) -> _FakeSession:
        return self

    async def __aexit__(self, *exc: Any) -> None:
        return None

    def get(self, url: str, **_kw: Any) -> _FakeResponse:
        self.last_url = url
        return self._response


@asynccontextmanager
async def _fake_client_session_factory(session: _FakeSession):
    yield session


@pytest.mark.asyncio
async def test_fetch_from_operator_builds_url_with_run_epoch(monkeypatch) -> None:
    """URL shape: ``{BASE_URL}/api/v1/results/{ns}/{name}/runs/{epoch}/profile_export_aiperf.json``."""
    from aiperf.operator.environment import OperatorEnvironment
    from aiperf.sweep_controller import k8s_executor as mod

    monkeypatch.setattr(
        OperatorEnvironment.SERVICE,
        "BASE_URL",
        "https://op.example:9091",
    )

    payload = _profile_export_payload()
    fake_resp = _FakeResponse(200, orjson.dumps(payload))
    fake_session = _FakeSession(fake_resp)

    def _mk_session(*_a: Any, **_kw: Any) -> _FakeSession:
        return fake_session

    monkeypatch.setattr(mod.aiohttp, "ClientSession", _mk_session)

    child = _child(run_epoch=1700000000)
    result = await _executor()._fetch_summary_from_operator(child)

    assert fake_session.last_url == (
        "https://op.example:9091/api/v1/results/aiperf-benchmarks/sweep-x-v00-t0"
        "/runs/1700000000/profile_export_aiperf.json"
    )
    assert "request_throughput" in result


@pytest.mark.asyncio
async def test_fetch_from_operator_skips_when_run_epoch_missing(monkeypatch) -> None:
    """No ``runEpoch`` on the child → no HTTP call (operator's epoch allowlist
    rejects anything that isn't 9-11 digits, so synthesizing ``latest`` would
    422). Returns ``{}`` for the caller to treat as fall-through.
    """
    from aiperf.sweep_controller import k8s_executor as mod

    called = False

    class _AssertingSession:
        async def __aenter__(self) -> _AssertingSession:
            return self

        async def __aexit__(self, *exc: Any) -> None:
            return None

        def get(self, *_a: Any, **_kw: Any) -> Any:
            nonlocal called
            called = True
            raise AssertionError("operator HTTP must not be called when epoch missing")

    monkeypatch.setattr(
        mod.aiohttp, "ClientSession", lambda *a, **kw: _AssertingSession()
    )

    result = await _executor()._fetch_summary_from_operator(_child(run_epoch=None))
    assert result == {}
    assert not called


@pytest.mark.asyncio
async def test_fetch_from_operator_returns_empty_on_404(monkeypatch) -> None:
    """Operator API 404 → ``{}`` (run still treated as success with empty summary)."""
    from aiperf.sweep_controller import k8s_executor as mod

    fake_session = _FakeSession(_FakeResponse(404, b""))
    monkeypatch.setattr(mod.aiohttp, "ClientSession", lambda *a, **kw: fake_session)

    result = await _executor()._fetch_summary_from_operator(_child())
    assert result == {}


@pytest.mark.asyncio
async def test_fetch_from_operator_returns_empty_on_transport_error(
    monkeypatch,
) -> None:
    """Connection failures must not bubble up — the caller treats empty as fall-through."""
    from aiperf.sweep_controller import k8s_executor as mod

    class _BoomSession:
        async def __aenter__(self) -> _BoomSession:
            return self

        async def __aexit__(self, *exc: Any) -> None:
            return None

        def get(self, *_a: Any, **_kw: Any) -> Any:
            raise mod.aiohttp.ClientConnectionError("operator unreachable")

    monkeypatch.setattr(mod.aiohttp, "ClientSession", lambda *a, **kw: _BoomSession())

    result = await _executor()._fetch_summary_from_operator(_child())
    assert result == {}


@pytest.mark.asyncio
async def test_fetch_from_operator_returns_empty_when_metadata_missing() -> None:
    """A child CR missing namespace/name short-circuits before any HTTP call."""
    child: dict[str, Any] = {"metadata": {}, "status": {"runEpoch": 1700000000}}
    result = await _executor()._fetch_summary_from_operator(child)
    assert result == {}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "status",
    [
        pytest.param(500, id="500-internal-server-error"),
        pytest.param(503, id="503-service-unavailable"),
    ],
)
async def test_fetch_from_operator_warns_on_5xx(monkeypatch, caplog, status) -> None:
    """5xx (operator unhealthy) is logged at WARNING — distinguishes operator
    breakage from a benign 404 (run not yet on PVC), which stays at DEBUG.
    """
    from aiperf.sweep_controller import k8s_executor as mod

    fake_session = _FakeSession(_FakeResponse(status, b""))
    monkeypatch.setattr(mod.aiohttp, "ClientSession", lambda *a, **kw: fake_session)

    with caplog.at_level("WARNING", logger=mod.logger.name):
        result = await _executor()._fetch_summary_from_operator(_child())

    assert result == {}
    assert any(f"HTTP {status}" in rec.message for rec in caplog.records), (
        f"expected HTTP {status} warning in caplog records: "
        f"{[r.message for r in caplog.records]}"
    )


@pytest.mark.asyncio
async def test_fetch_from_operator_returns_empty_on_422_epoch_rejected(
    monkeypatch, caplog
) -> None:
    """422 from the operator's epoch allowlist (the regex `^\\d{9,11}$` rejects
    short/long epochs or ``latest``). Stays at DEBUG since it's a benign caller-
    side mistake, not operator brokenness.
    """
    from aiperf.sweep_controller import k8s_executor as mod

    fake_session = _FakeSession(_FakeResponse(422, b""))
    monkeypatch.setattr(mod.aiohttp, "ClientSession", lambda *a, **kw: fake_session)

    with caplog.at_level("DEBUG", logger=mod.logger.name):
        result = await _executor()._fetch_summary_from_operator(_child())

    assert result == {}
    assert not any(rec.levelname == "WARNING" for rec in caplog.records), (
        "422 must not warn (it would noise oncall on a transient client mistake)"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "exc",
    [
        pytest.param(TimeoutError("timed out"), id="timeout-error"),
        pytest.param(ConnectionError("conn reset"), id="bare-connection-error"),
    ],
)
async def test_fetch_from_operator_returns_empty_on_other_transport_errors(
    monkeypatch, exc
) -> None:
    """Beyond ``aiohttp.ClientConnectionError``, both ``TimeoutError`` (request
    aiohttp timeout coerces here) and bare ``ConnectionError`` are swallowed.
    All three are named in the except clause; this fences each.
    """
    from aiperf.sweep_controller import k8s_executor as mod

    class _BoomSession:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_a: Any) -> None:
            return None

        def get(self, *_a: Any, **_kw: Any) -> Any:
            raise exc

    monkeypatch.setattr(mod.aiohttp, "ClientSession", lambda *a, **kw: _BoomSession())

    result = await _executor()._fetch_summary_from_operator(_child())
    assert result == {}


@pytest.mark.asyncio
async def test_fetch_from_operator_returns_empty_on_invalid_json_body(
    monkeypatch,
) -> None:
    """If the response body is non-JSON (e.g. an HTML error page slipped past a
    misconfigured ingress), parse failure must NOT propagate — return ``{}``.
    """
    from aiperf.sweep_controller import k8s_executor as mod

    fake_session = _FakeSession(_FakeResponse(200, b"<html>boom</html>"))
    monkeypatch.setattr(mod.aiohttp, "ClientSession", lambda *a, **kw: fake_session)

    result = await _executor()._fetch_summary_from_operator(_child())
    assert result == {}


@pytest.mark.asyncio
async def test_fetch_from_operator_returns_empty_on_non_dict_payload(
    monkeypatch,
) -> None:
    """A JSON list/scalar at the top level (not a dict) is rejected — the
    projection requires a dict-shaped payload.
    """
    from aiperf.sweep_controller import k8s_executor as mod

    fake_session = _FakeSession(_FakeResponse(200, orjson.dumps([1, 2, 3])))
    monkeypatch.setattr(mod.aiohttp, "ClientSession", lambda *a, **kw: fake_session)

    result = await _executor()._fetch_summary_from_operator(_child())
    assert result == {}


@pytest.mark.asyncio
async def test_fetch_from_operator_strips_trailing_slash_from_base_url(
    monkeypatch,
) -> None:
    """``BASE_URL`` with a trailing slash must not produce ``//api/v1/...`` —
    the ``.rstrip("/")`` in the URL builder is load-bearing.
    """
    from aiperf.operator.environment import OperatorEnvironment
    from aiperf.sweep_controller import k8s_executor as mod

    monkeypatch.setattr(
        OperatorEnvironment.SERVICE,
        "BASE_URL",
        "https://op.example:9091/",  # trailing slash
    )

    fake_session = _FakeSession(
        _FakeResponse(200, orjson.dumps(_profile_export_payload()))
    )
    monkeypatch.setattr(mod.aiohttp, "ClientSession", lambda *a, **kw: fake_session)

    await _executor()._fetch_summary_from_operator(_child(run_epoch=1700000000))

    assert fake_session.last_url is not None
    assert "//api/v1/" not in fake_session.last_url, (
        f"trailing slash leaked into URL: {fake_session.last_url}"
    )
    assert fake_session.last_url == (
        "https://op.example:9091/api/v1/results/aiperf-benchmarks/sweep-x-v00-t0"
        "/runs/1700000000/profile_export_aiperf.json"
    )
