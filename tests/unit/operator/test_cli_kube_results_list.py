# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for ``aiperf kube results list-runs``.

The command opens an operator pod port-forward and hits
``/api/v1/results/<ns>/<name>/runs``. Tests here mock the table renderer
directly with fixture payloads, and mock the HTTP + port-forward chain for
end-to-end command coverage.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock, patch

import orjson
import pytest

from aiperf.cli_commands.kube._runs_render import print_runs_table
from aiperf.cli_commands.kube.results import _render_list_runs_payload, list_runs
from aiperf.config.kube import KubeManageOptions
from aiperf.kubernetes.console import LastBenchmarkInfo
from aiperf.kubernetes.models import AIPerfJobInfo


@pytest.fixture
def sample_payload() -> dict:
    """Response payload shape from ``/api/v1/results/<ns>/<name>/runs``."""
    return {
        "namespace": "default",
        "job_id": "foo",
        "latest_epoch": "1714150923",
        "runs": [
            {
                "epoch": "1714150923",
                "mtime_epoch": 1714150925,
                "file_count": 7,
                "total_size_bytes": 4823912,
                "is_latest": True,
            },
            {
                "epoch": "1714064523",
                "mtime_epoch": 1714064525,
                "file_count": 7,
                "total_size_bytes": 4823912,
                "is_latest": False,
            },
        ],
    }


# =============================================================================
# print_runs_table: renderer-in-isolation tests
# =============================================================================


class TestPrintRunsTable:
    """Text-formatting helper can be unit-tested without any k8s/HTTP mocks."""

    def test_renders_all_rows(self, sample_payload: dict, capsys) -> None:
        from aiperf.kubernetes.console import console as _console

        _console.width = 200
        try:
            print_runs_table(sample_payload)
        finally:
            _console.width = None

        out = capsys.readouterr().out
        assert "EPOCH" in out
        assert "TIMESTAMP" in out
        assert "FILES" in out
        assert "SIZE" in out
        assert "LATEST" in out
        assert "1714150923" in out
        assert "1714064523" in out
        # Human-readable size and UTC-formatted timestamp
        assert "4.6 MiB" in out
        assert "2024-04-26" in out

    def test_empty_runs_prints_info_message(self, capsys) -> None:
        print_runs_table({"namespace": "default", "job_id": "bar", "runs": []})
        out = capsys.readouterr().out
        assert "No runs found for default/bar" in out

    def test_marks_only_latest_row(self, sample_payload: dict, capsys) -> None:
        from aiperf.kubernetes.console import console as _console

        _console.width = 200
        try:
            print_runs_table(sample_payload)
        finally:
            _console.width = None

        out = capsys.readouterr().out
        # Checkmark rendered for exactly one row
        assert out.count("✓") == 1


# =============================================================================
# list_runs: full-command tests (HTTP + port-forward mocked)
# =============================================================================


def _mock_http_response(*, status: int = 200, json_payload: dict | None = None):
    """Return an ``aiohttp.ClientResponse``-shaped async-context mock."""
    resp = MagicMock()
    resp.status = status
    resp.raise_for_status = MagicMock()
    if status >= 400:
        resp.raise_for_status.side_effect = RuntimeError(f"HTTP {status}")
    resp.json = AsyncMock(return_value=json_payload or {})

    @asynccontextmanager
    async def _as_ctx():
        yield resp

    return _as_ctx


def _mock_session_cm(get_cm):
    session = MagicMock()
    session.get = MagicMock(return_value=get_cm())

    @asynccontextmanager
    async def _as_ctx(*_args, **_kwargs):
        yield session

    return _as_ctx


@asynccontextmanager
async def _mock_port_forward(*_args, **_kwargs):
    yield 12345


@pytest.fixture
def mock_resolve_and_pod():
    """Mock ``resolve_job`` + ``find_operator_pod`` for the list_runs flow."""
    resolved = MagicMock()
    resolved.job_id = "foo"
    resolved.namespace = "default"
    resolved.api = MagicMock()
    resolved.aclose = AsyncMock()

    async def _fake_resolve_op_ns(_api, *, explicit, default="aiperf-system"):
        return explicit if explicit is not None else default

    with (
        patch(
            "aiperf.kubernetes.cli_helpers.resolve_job",
            new=AsyncMock(return_value=resolved),
        ),
        patch(
            "aiperf.kubernetes.client.find_operator_pod",
            new=AsyncMock(return_value=("operator-pod-x", "Running")),
        ),
        patch(
            "aiperf.kubernetes.client.resolve_operator_namespace",
            new=_fake_resolve_op_ns,
        ),
        patch(
            "aiperf.kubernetes.port_forward.port_forward_with_status",
            new=_mock_port_forward,
        ),
    ):
        yield resolved


@pytest.mark.asyncio
async def test_list_runs_text_output_formats_table(
    mock_resolve_and_pod, sample_payload: dict, capsys
) -> None:
    get_cm = _mock_http_response(status=200, json_payload=sample_payload)
    session_cm = _mock_session_cm(get_cm)

    from aiperf.kubernetes.console import console as _console

    _console.width = 200
    try:
        with patch("aiohttp.ClientSession", new=session_cm):
            await list_runs(
                job_id="foo",
                manage_options=KubeManageOptions(),
                output="text",
            )
    finally:
        _console.width = None

    out = capsys.readouterr().out
    assert "EPOCH" in out
    assert "1714150923" in out
    assert "4.6 MiB" in out


def test_render_list_runs_json_does_not_wrap_long_strings(capsys) -> None:
    """JSON output must bypass Rich wrapping so stdout remains machine-parseable."""
    from aiperf.kubernetes.console import console as _console

    payload = {
        "namespace": "default",
        "job_id": "foo",
        "runs": [
            {
                "epoch": "1714150923",
                "variation_label": "long label " * 40,
            }
        ],
    }

    _console.width = 80
    try:
        _render_list_runs_payload(payload, output="json", preview=False)
    finally:
        _console.width = None

    parsed = orjson.loads(capsys.readouterr().out)
    assert parsed == payload


@pytest.mark.asyncio
async def test_list_runs_json_output_parseable(
    mock_resolve_and_pod, sample_payload: dict, capsys
) -> None:
    get_cm = _mock_http_response(status=200, json_payload=sample_payload)
    session_cm = _mock_session_cm(get_cm)

    from aiperf.kubernetes.console import console as _console

    _console.width = 200
    try:
        with patch("aiohttp.ClientSession", new=session_cm):
            await list_runs(
                job_id="foo",
                manage_options=KubeManageOptions(),
                output="json",
            )
    finally:
        _console.width = None

    out = capsys.readouterr().out.strip()
    # Rich prints the JSON; strip leading/trailing whitespace, then parse
    parsed = orjson.loads(out)
    assert parsed["namespace"] == "default"
    assert parsed["job_id"] == "foo"
    assert len(parsed["runs"]) == 2
    mock_resolve_and_pod.aclose.assert_awaited_once()


@pytest.mark.asyncio
async def test_list_runs_json_threads_quiet_to_resolver(
    sample_payload: dict,
) -> None:
    resolved = MagicMock()
    resolved.job_id = "foo"
    resolved.namespace = "default"
    resolved.api = MagicMock()
    resolved.aclose = AsyncMock()
    resolver = AsyncMock(return_value=resolved)
    get_cm = _mock_http_response(status=200, json_payload=sample_payload)
    session_cm = _mock_session_cm(get_cm)

    with (
        patch("aiperf.kubernetes.cli_helpers.resolve_job", new=resolver),
        patch(
            "aiperf.kubernetes.client.find_operator_pod",
            new=AsyncMock(return_value=("operator-pod-x", "Running")),
        ),
        patch(
            "aiperf.kubernetes.client.resolve_operator_namespace",
            new=AsyncMock(return_value="aiperf-system"),
        ),
        patch(
            "aiperf.kubernetes.port_forward.port_forward_with_status",
            new=_mock_port_forward,
        ),
        patch("aiohttp.ClientSession", new=session_cm),
    ):
        await list_runs(
            job_id=None,
            manage_options=KubeManageOptions(),
            output="json",
        )

    assert resolver.call_args.kwargs["quiet"] is True


@pytest.mark.asyncio
async def test_list_runs_json_default_job_resolution_is_quiet(
    sample_payload: dict, capsys
) -> None:
    """Default last-benchmark resolution must not contaminate JSON stdout."""
    api = MagicMock()
    api.close = AsyncMock()
    job_info = AIPerfJobInfo(
        name="foo",
        namespace="default",
        phase="Completed",
        job_id="foo",
    )
    get_cm = _mock_http_response(status=200, json_payload=sample_payload)
    session_cm = _mock_session_cm(get_cm)

    with (
        patch(
            "aiperf.kubernetes.cli_helpers.get_last_benchmark",
            return_value=LastBenchmarkInfo(job_id="foo", namespace="default"),
        ),
        patch(
            "aiperf.kubernetes.cli_helpers._open_api_client",
            new=AsyncMock(return_value=api),
        ),
        patch(
            "aiperf.kubernetes.client.find_aiperf_job",
            new=AsyncMock(return_value=job_info),
        ),
        patch(
            "aiperf.kubernetes.client.find_operator_pod",
            new=AsyncMock(return_value=("operator-pod-x", "Running")),
        ),
        patch(
            "aiperf.kubernetes.client.resolve_operator_namespace",
            new=AsyncMock(return_value="aiperf-system"),
        ),
        patch(
            "aiperf.kubernetes.port_forward.port_forward_with_status",
            new=_mock_port_forward,
        ),
        patch("aiohttp.ClientSession", new=session_cm),
    ):
        await list_runs(
            job_id=None,
            manage_options=KubeManageOptions(),
            output="json",
        )

    out = capsys.readouterr().out.strip()
    parsed = orjson.loads(out)
    assert parsed["job_id"] == "foo"
    assert "Using last benchmark" not in out
    api.close.assert_awaited_once()


@pytest.mark.asyncio
async def test_list_runs_404_raises_informative_error(
    mock_resolve_and_pod, capsys
) -> None:
    get_cm = _mock_http_response(status=404, json_payload={})
    session_cm = _mock_session_cm(get_cm)

    with (
        patch("aiohttp.ClientSession", new=session_cm),
        pytest.raises(SystemExit),
    ):
        await list_runs(
            job_id="ghost",
            manage_options=KubeManageOptions(),
            output="text",
        )

    captured = capsys.readouterr()
    # cli_utils.exit_on_error routes the panel through a stderr-pinned
    # rich Console; the user-facing "No runs found" copy lands on stderr.
    assert "No runs found" in (captured.err + captured.out)


@pytest.mark.asyncio
async def test_list_runs_text_output_includes_run_hint(
    mock_resolve_and_pod, sample_payload: dict, capsys
) -> None:
    get_cm = _mock_http_response(status=200, json_payload=sample_payload)
    session_cm = _mock_session_cm(get_cm)

    from aiperf.kubernetes.console import console as _console

    _console.width = 200
    try:
        with patch("aiohttp.ClientSession", new=session_cm):
            await list_runs(
                job_id="foo",
                manage_options=KubeManageOptions(),
                output="text",
            )
    finally:
        _console.width = None

    out = capsys.readouterr().out
    assert "--run <epoch>" in out
    assert "historical" in out


# =============================================================================
# results (--default): --run threading tests
# =============================================================================


@pytest.fixture
def mock_resolve_for_results(tmp_path):
    """Mock ``resolve_target`` + ``find_jobset`` for the default results flow."""
    from aiperf.kubernetes.cli_helpers import ResolvedJob

    job_info = MagicMock()
    job_info.namespace = "default"
    job_info.job_id = "foo"
    job_info.name = "foo"
    api = MagicMock()
    api.close = AsyncMock()
    resolved = ResolvedJob(name="foo", job_info=job_info, api=api)

    async def _fake_resolve_op_ns(_api, *, explicit, default="aiperf-system"):
        return explicit if explicit is not None else default

    with (
        patch(
            "aiperf.kubernetes.cli_helpers.resolve_target",
            new=AsyncMock(return_value=resolved),
        ),
        patch(
            "aiperf.kubernetes.client.find_jobset",
            new=AsyncMock(return_value=None),
        ),
        patch(
            "aiperf.kubernetes.client.resolve_operator_namespace",
            new=_fake_resolve_op_ns,
        ),
    ):
        yield resolved


@pytest.mark.asyncio
async def test_results_routes_through_runs_prefix_when_run_set(
    mock_resolve_for_results, tmp_path
) -> None:
    """Verify that --run threads through to the operator URL prefix."""
    from aiperf.cli_commands.kube.results import _run_results

    captured: dict = {}

    async def fake_retrieve(job_id, namespace, output_dir, api, **kwargs):  # noqa: ARG001
        captured["run"] = kwargs.get("run")
        captured["output_dir"] = output_dir
        return True

    with patch(
        "aiperf.kubernetes.results.retrieve_results_from_operator",
        new=AsyncMock(side_effect=fake_retrieve),
    ):
        await _run_results(
            job_id="foo",
            manage_options=KubeManageOptions(),
            output=tmp_path / "out",
            from_pods=False,
            all_artifacts=True,
            shutdown=False,
            port=0,
            operator_namespace="aiperf-system",
            run="1714150923",
        )

    assert captured["run"] == "1714150923"
    mock_resolve_for_results.api.close.assert_awaited_once()


@pytest.mark.asyncio
async def test_results_closes_resolved_api_on_job_path(
    mock_resolve_for_results, tmp_path
) -> None:
    """The job results path closes the ApiClient returned by resolve_target."""
    from aiperf.cli_commands.kube.results import _run_results

    with patch(
        "aiperf.kubernetes.results.retrieve_results_from_operator",
        new=AsyncMock(return_value=True),
    ):
        await _run_results(
            job_id="foo",
            manage_options=KubeManageOptions(),
            output=tmp_path / "out",
            from_pods=False,
            all_artifacts=True,
            shutdown=False,
            port=0,
            operator_namespace="aiperf-system",
            run=None,
        )

    mock_resolve_for_results.api.close.assert_awaited_once()


def test_result_base_url_helper_pins_epoch() -> None:
    """The URL helper must produce the ``/runs/<epoch>`` prefix when pinned."""
    from aiperf.kubernetes.results_operator import _result_base_url

    latest = _result_base_url("http://x", "default", "foo", None)
    pinned = _result_base_url("http://x", "default", "foo", "1714150923")

    assert latest == "http://x/api/v1/results/default/foo"
    assert pinned == "http://x/api/v1/results/default/foo/runs/1714150923"


@pytest.mark.asyncio
async def test_results_artifact_dir_includes_epoch_when_run_set(
    mock_resolve_for_results, tmp_path, monkeypatch
) -> None:
    """When --run is set and --output is omitted, the default path embeds epoch."""
    from aiperf.cli_commands.kube.results import _run_results

    monkeypatch.chdir(tmp_path)
    captured: dict = {}

    async def fake_retrieve(job_id, namespace, output_dir, api, **kwargs):  # noqa: ARG001
        captured["output_dir"] = output_dir
        return True

    with patch(
        "aiperf.kubernetes.results.retrieve_results_from_operator",
        new=AsyncMock(side_effect=fake_retrieve),
    ):
        await _run_results(
            job_id="foo",
            manage_options=KubeManageOptions(),
            output=None,
            from_pods=False,
            all_artifacts=True,
            shutdown=False,
            port=0,
            operator_namespace="aiperf-system",
            run="1714150923",
        )

    assert "1714150923" in str(captured["output_dir"])
    assert "default__foo__1714150923" in str(captured["output_dir"])


@pytest.mark.asyncio
async def test_results_run_rejects_invalid_epoch(tmp_path) -> None:
    """Garbage --run value fails fast before any k8s/HTTP traffic."""
    from aiperf.cli_commands.kube.results import _run_results

    with pytest.raises(ValueError, match="Invalid --run"):
        await _run_results(
            job_id="foo",
            manage_options=KubeManageOptions(),
            output=tmp_path / "out",
            from_pods=False,
            all_artifacts=True,
            shutdown=False,
            port=0,
            operator_namespace="aiperf-system",
            run="not-an-epoch",
        )


# =============================================================================
# --preview: annotates which runs would be reaped
# =============================================================================


def _preview_payload(now_epoch: int) -> dict:
    """Three runs: one within-keep, one outside-keep recent, one outside-keep old."""
    day = 86400
    return {
        "namespace": "default",
        "job_id": "foo",
        "latest_epoch": "1714150923",
        "runs": [
            {
                "epoch": "1714150923",
                "mtime_epoch": now_epoch,
                "file_count": 5,
                "total_size_bytes": 1000,
                "is_latest": True,
            },
            {
                "epoch": "1714100000",
                "mtime_epoch": now_epoch - 2 * day,
                "file_count": 5,
                "total_size_bytes": 1000,
                "is_latest": False,
            },
            {
                "epoch": "1714000000",
                "mtime_epoch": now_epoch - 60 * day,
                "file_count": 5,
                "total_size_bytes": 1000,
                "is_latest": False,
            },
        ],
    }


def _mock_session_two_gets(runs_payload: dict, retention_payload: dict):
    """Return a mock aiohttp session whose .get() dispatches by URL substring."""

    def _make_resp(payload: dict):
        resp = MagicMock()
        resp.status = 200
        resp.raise_for_status = MagicMock()
        resp.json = AsyncMock(return_value=payload)

        @asynccontextmanager
        async def _as_ctx():
            yield resp

        return _as_ctx

    def _get(url: str, *_a, **_kw):
        if "/config/retention" in url:
            return _make_resp(retention_payload)()
        return _make_resp(runs_payload)()

    session = MagicMock()
    session.get = MagicMock(side_effect=_get)

    @asynccontextmanager
    async def _as_ctx(*_args, **_kwargs):
        yield session

    return _as_ctx


@pytest.mark.asyncio
async def test_list_runs_preview_marks_old_for_deletion(
    mock_resolve_and_pod, capsys
) -> None:
    """retain_runs=1 + retain_days=7: everything outside count-window is marked.

    Actual server behavior matches ``enforce_retention`` in results_layout.py:
    a run is deleted when EITHER policy would reap it (i.e. kept only if
    count_keep AND age_keep). Only the single count-keeper — the latest —
    survives under retain_runs=1.
    """
    import time

    now = int(time.time())
    runs = _preview_payload(now)
    retention = {"retain_runs": 1, "retain_days": 7}
    session_cm = _mock_session_two_gets(runs, retention)

    with patch("aiohttp.ClientSession", new=session_cm):
        await list_runs(
            job_id="foo",
            manage_options=KubeManageOptions(),
            output="json",
            preview=True,
        )

    out = capsys.readouterr().out.strip()
    parsed = orjson.loads(out)
    by_epoch = {r["epoch"]: r for r in parsed["runs"]}
    # Latest is always protected and is the only count-keeper under retain_runs=1.
    assert by_epoch["1714150923"]["would_delete"] is False
    # 60-day-old: outside count-keepers AND outside age window -> reap.
    assert by_epoch["1714000000"]["would_delete"] is True
    # 2-day-old: outside count-keepers -> reap (count policy alone is enough).
    assert by_epoch["1714100000"]["would_delete"] is True
    assert parsed["retention"] == {"retain_runs": 1, "retain_days": 7}


@pytest.mark.asyncio
async def test_list_runs_preview_protects_latest(mock_resolve_and_pod, capsys) -> None:
    """Even with retain_runs=0 + retain_days=1, latest must never be marked."""
    import time

    now = int(time.time())
    runs = _preview_payload(now)
    retention = {"retain_runs": 0, "retain_days": 1}
    session_cm = _mock_session_two_gets(runs, retention)

    with patch("aiohttp.ClientSession", new=session_cm):
        await list_runs(
            job_id="foo",
            manage_options=KubeManageOptions(),
            output="json",
            preview=True,
        )

    out = capsys.readouterr().out.strip()
    parsed = orjson.loads(out)
    by_epoch = {r["epoch"]: r for r in parsed["runs"]}
    assert by_epoch["1714150923"]["would_delete"] is False
