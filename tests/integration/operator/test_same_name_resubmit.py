# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Integration test: same-named AIPerfJob resubmit preserves prior results.

The user-visible contract under test:

    kubectl apply  -> first completion writes run at epoch_old
    kubectl delete
    kubectl apply  -> second completion writes run at epoch_new
    Both run dirs coexist; /api/v1/results/<ns>/<name> serves the NEW run
    while /api/v1/results/<ns>/<name>/runs/<epoch_old> still serves the OLD.

We drive the same disk-layout API the completion handler uses
(``run_dir`` + ``write_latest`` + ``enforce_retention``) with two synthetic
AIPerfJob bodies whose ``metadata.creationTimestamp`` values parse to
distinct decimal epochs via ``epoch_key_from_body``. The HTTP assertions
then exercise the real results-server routing through ``TestClient``.

This stops short of spinning up a kind cluster: it proves the layout +
routing contract end-to-end without the overhead of a real apiserver.
"""

from __future__ import annotations

from pathlib import Path

import orjson
import pytest
from aiperf.operator.results_layout import (
    enforce_retention,
    epoch_key_from_body,
    resolve_latest,
    run_dir,
    write_latest,
)
from fastapi.testclient import TestClient

from aiperf.kubernetes.results_sidecar import write_ready_marker
from aiperf.operator.results_server import create_app

_RETAIN_RUNS_DEFAULT = 10

# Minimal realistic metrics payload — shaped like what the completion
# handler would download from the SystemController API after a run.
_PROFILE_EXPORT = orjson.dumps(
    {
        "request_throughput": {"avg": 100.0, "unit": "req/s"},
        "request_latency": {"avg": 50.0, "unit": "ms"},
        "start_time": "2024-04-25T18:00:05Z",
        "end_time": "2024-04-25T18:00:30Z",
        "input_config": {
            "models": {"items": [{"name": "llama-7b"}]},
            "endpoint": {"urls": ["http://localhost:8000"]},
        },
    }
)


def _synthetic_body(name: str, namespace: str, creation_timestamp: str) -> dict:
    """Build a minimal AIPerfJob body shaped like what kopf would deliver."""
    return {
        "apiVersion": "aiperf.nvidia.com/v1",
        "kind": "AIPerfJob",
        "metadata": {
            "name": name,
            "namespace": namespace,
            "creationTimestamp": creation_timestamp,
        },
        "spec": {"model": "llama-7b"},
    }


def _complete_run(
    results_dir: Path,
    body: dict,
    *,
    filename: str,
    payload: bytes,
) -> str:
    """Simulate the completion-handler success path for a single run.

    Mirrors ``handlers/completion.py::_apply_results_to_status`` for the
    ``has_files`` branch: write artifacts under the epoch-keyed run_dir,
    atomically update ``latest.txt``, and run a retention pass.

    Returns the epoch string written.
    """
    namespace = body["metadata"]["namespace"]
    name = body["metadata"]["name"]
    epoch = epoch_key_from_body(body)

    dest = run_dir(results_dir, namespace, name, epoch)
    dest.mkdir(parents=True, exist_ok=True)
    (dest / filename).write_bytes(payload)
    # Real completion handler gates root-file listing on this marker
    # (completion.py::_apply_results_to_status -> write_ready_marker).
    write_ready_marker(dest)

    write_latest(results_dir, namespace, name, epoch)
    enforce_retention(
        results_dir,
        namespace,
        name,
        keep=_RETAIN_RUNS_DEFAULT,
        protect_epoch=epoch,
    )
    return epoch


@pytest.mark.component_integration
def test_same_name_resubmit_preserves_prior_run(tmp_path: Path) -> None:
    """Delete + resubmit same-named AIPerfJob keeps old run, serves new as latest."""
    results_dir = tmp_path / "results"
    results_dir.mkdir()

    namespace, name = "default", "foo"

    body_first = _synthetic_body(
        name, namespace, creation_timestamp="2024-04-25T18:00:00Z"
    )
    epoch_old = _complete_run(
        results_dir,
        body_first,
        filename="profile_export_aiperf.json",
        payload=_PROFILE_EXPORT,
    )
    old_dir = results_dir / namespace / name / epoch_old
    assert old_dir.is_dir(), f"First run dir missing at {old_dir}"

    # User deletes the CR then reapplies a new one with the same name but
    # a later creationTimestamp; kopf will dispatch a fresh create handler.
    body_second = _synthetic_body(
        name, namespace, creation_timestamp="2024-04-25T18:05:00Z"
    )
    epoch_new = _complete_run(
        results_dir,
        body_second,
        filename="profile_export_aiperf.json",
        payload=_PROFILE_EXPORT,
    )
    assert epoch_new != epoch_old, (
        f"Distinct creationTimestamps must yield distinct epochs "
        f"(old={epoch_old}, new={epoch_new})"
    )

    new_dir = results_dir / namespace / name / epoch_new
    assert new_dir.is_dir(), f"New run dir missing at {new_dir}"
    assert old_dir.is_dir(), (
        f"Resubmit must NOT delete old run dir (expected {old_dir} to still exist)"
    )

    # latest.txt is atomically updated to the new epoch.
    assert resolve_latest(results_dir, namespace, name) == epoch_new
    latest_ptr = (results_dir / namespace / name / "latest.txt").read_text().strip()
    assert latest_ptr == epoch_new

    # HTTP: default route now requires an explicit epoch (HEAD removed the
    # implicit "latest at <ns>/<name>" shortcut); the pinned routes still
    # serve both runs independently — proving the disk layout truly indexes
    # by epoch and not by alias.
    with TestClient(create_app(results_dir=results_dir)) as client:
        latest_resp = client.get(f"/api/v1/results/{namespace}/{name}")
        assert latest_resp.status_code == 409, latest_resp.text
        assert "epoch" in latest_resp.json()["detail"].lower()

        historical_resp = client.get(
            f"/api/v1/results/{namespace}/{name}/runs/{epoch_old}"
        )
        assert historical_resp.status_code == 200, historical_resp.text
        historical_body = historical_resp.json()
        assert historical_body["files"], (
            f"Pinned historical route must still serve epoch {epoch_old} after resubmit"
        )

        # Also confirm pinning to the new epoch works — proves the route
        # is truly epoch-indexed.
        pinned_new_resp = client.get(
            f"/api/v1/results/{namespace}/{name}/runs/{epoch_new}"
        )
        assert pinned_new_resp.status_code == 200, pinned_new_resp.text
        assert pinned_new_resp.json()["files"]
