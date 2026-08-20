# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Swept parameter values must ride the children manifest end-to-end.

Adaptive planners label variations ``search_iter_NNNN``, which identifies the
artifact directory but describes nothing about the operating point. The values
were already stamped on the child CR's ``aiperf.nvidia.com/variation-values``
annotation; these tests pin that the same encoding also reaches the lineage
record, the CR-embedded manifest, the archived ``children.json``, and the
manifest API model that UI and CLI read back.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import orjson
import pytest
from aiperf.operator.routers._sweeps_live import children_manifest_from_live_aiperfjobs
from aiperf.operator.routers.sweeps import _children_manifest_from_doc

from aiperf.common.endpoint_credentials import (
    AIPERF_INJECTED_ENDPOINT_URLS,
    redact_sweep_display_label,
)
from aiperf.config import BenchmarkConfig, BenchmarkRun, SweepVariation
from aiperf.sweep_controller.k8s_executor import (
    VARIATION_VALUES_ANNOTATION,
    ChildRunRef,
    K8sChildJobExecutor,
)


def _sweep_cr() -> dict:
    return {
        "metadata": {"name": "s", "namespace": "ns", "uid": "uid"},
        "spec": {
            "image": "x:latest",
            "benchmark": {
                "models": ["m"],
                "endpoint": {"urls": ["http://x"], "type": "chat"},
                "datasets": [{"name": "main", "type": "synthetic"}],
                "phases": [
                    {
                        "name": "profiling",
                        "type": "concurrency",
                        "duration": 1,
                        "concurrency": 17,
                    }
                ],
            },
        },
    }


def _benchmark_run(
    var_idx: int = 8,
    trial: int = 0,
    *,
    variation_label: str = "search_iter_0008",
    variation_values: dict[str, object] | None = None,
    endpoint_url: str = "http://x",
) -> BenchmarkRun:
    cfg = BenchmarkConfig.model_validate(
        {
            "models": ["m"],
            "endpoint": {"urls": [endpoint_url], "type": "chat"},
            "datasets": [{"name": "main", "type": "synthetic"}],
            "phases": [
                {
                    "name": "profiling",
                    "type": "concurrency",
                    "duration": 1,
                    "concurrency": 17,
                }
            ],
        }
    )
    return BenchmarkRun(
        benchmark_id=f"s-v{var_idx:02d}-t{trial:01d}",
        cfg=cfg,
        # The planner-assigned label an adaptive sweep actually produces.
        variation=SweepVariation(
            index=var_idx,
            label=variation_label,
            values=variation_values or {"phases.profiling.concurrency": 17},
        ),
        trial=trial,
        label=f"run_{trial:04d}",
        artifact_dir=Path("/results"),
    )


def _executor() -> K8sChildJobExecutor:
    return K8sChildJobExecutor(
        api=None,
        sweep=_sweep_cr(),
        with_trial_suffix=True,
        sweep_run_epoch="1714000000",
    )


@pytest.mark.asyncio
async def test_terminal_child_carries_variation_values_to_the_manifest() -> None:
    executor = _executor()
    status_writer = MagicMock()
    status_writer.current_cell = AsyncMock()
    status_writer.partial_children = AsyncMock()
    executor._status_writer = status_writer
    # `execute` reads metadata.uid off whatever _get_or_create returns and
    # refuses to poll or mutate a child without one (k8s_executor.py:736-742,
    # identity fencing from b2b7f8e201). An empty dict is no longer a usable
    # stand-in for a bound child.
    executor._get_or_create = AsyncMock(  # type: ignore[method-assign]
        return_value={"metadata": {"name": "s-v08-t0", "uid": "child-uid-8"}}
    )
    executor._wait_until_terminal = AsyncMock(return_value=None)  # type: ignore[method-assign]
    executor._try_read_child = AsyncMock(  # type: ignore[method-assign]
        return_value={
            # A child with no metadata.uid is refused outright by the
            # identity fencing added in b2b7f8e201: without an immutable
            # handle a delayed mutation could hit a same-named child from a
            # previous run of this sweep.
            "metadata": {"name": "s-v08-t0", "uid": "child-uid-8"},
            "status": {"phase": "Succeeded", "runEpoch": "1714000042"},
        }
    )
    executor._collect_run_result = AsyncMock(  # type: ignore[method-assign]
        return_value=MagicMock(success=True)
    )

    await executor.execute(_benchmark_run())

    children = status_writer.partial_children.await_args.kwargs["children"]
    assert orjson.loads(children[0]["variation_values"]) == {
        "phases.profiling.concurrency": 17
    }
    assert (
        executor.terminal_children[0].variation_values
        == children[0]["variation_values"]
    )


@pytest.mark.asyncio
async def test_manifest_values_match_the_child_annotation_byte_for_byte() -> None:
    """One encoding for the annotation, ``status.runs[].values`` and the
    manifest means the read side needs exactly one formatter."""
    executor = _executor()
    run = _benchmark_run()
    executor._status_writer = None
    # `execute` reads metadata.uid off whatever _get_or_create returns and
    # refuses to poll or mutate a child without one (k8s_executor.py:736-742,
    # identity fencing from b2b7f8e201). An empty dict is no longer a usable
    # stand-in for a bound child.
    executor._get_or_create = AsyncMock(  # type: ignore[method-assign]
        return_value={"metadata": {"name": "s-v08-t0", "uid": "child-uid-8"}}
    )
    executor._wait_until_terminal = AsyncMock(return_value=None)  # type: ignore[method-assign]
    executor._try_read_child = AsyncMock(  # type: ignore[method-assign]
        return_value={
            "metadata": {"name": "s-v08-t0", "uid": "child-uid-8"},
            "status": {"phase": "Succeeded"},
        }
    )
    executor._collect_run_result = AsyncMock(  # type: ignore[method-assign]
        return_value=MagicMock(success=True)
    )

    await executor.execute(run)

    annotation = executor._build_child_metadata(run, "s-v08-t0")["annotations"][
        VARIATION_VALUES_ANNOTATION
    ]
    assert executor.terminal_children[0].variation_values == annotation


def test_child_run_ref_defaults_values_to_empty_for_older_archives() -> None:
    """Archives written before this field decode unchanged; consumers fall back
    to ``variation_label`` rather than rendering a half-formed descriptor."""
    ref = ChildRunRef(
        namespace="ns",
        name="s-v00",
        variation_index=0,
        variation_label="v0",
        trial_index=None,
        child_run_epoch="1714000042",
        label="cell-0",
        status="Succeeded",
        error="",
    )

    assert ref.variation_values == ""
    assert ref.to_dict()["variation_values"] == ""


def test_children_manifest_api_preserves_variation_values() -> None:
    """``ChildrenManifestEntry`` enumerates its fields, so an unlisted key is
    silently dropped on the archived read path."""
    response = _children_manifest_from_doc(
        {
            "sweep_run_epoch": "1714000000",
            "children": [
                {
                    "namespace": "ns",
                    "name": "s-v08-t0",
                    "variation_index": 8,
                    "variation_label": "search_iter_0008",
                    "variation_values": '{"phases.profiling.concurrency":17}',
                    "trial_index": 0,
                    "child_run_epoch": "1714000042",
                },
                {
                    "namespace": "ns",
                    "name": "s-v09-t0",
                    "variation_index": 9,
                    "variation_label": "search_iter_0009",
                    "child_run_epoch": "1714000043",
                },
            ],
        },
        epoch=None,
    )

    assert response.children[0].variation_values == (
        '{"phases.profiling.concurrency":17}'
    )
    assert response.children[1].variation_values == ""
    # camelCase on the wire, matching the rest of the response model.
    assert response.model_dump(by_alias=True)["children"][0]["variationValues"] == (
        '{"phases.profiling.concurrency":17}'
    )


@pytest.mark.asyncio
async def test_credential_values_never_reach_annotation_status_or_manifest_api() -> (
    None
):
    credential_url = "https://user:pass@host/v1?token=query-secret"
    run = _benchmark_run(
        variation_label=(
            f"endpoint.api_key=axis-secret, endpoint.urls={credential_url}"
        ),
        variation_values={
            "endpoint.apiKey": "axis-secret",
            "endpoint.urls": credential_url,
            "variables.header": "Authorization: Bearer header-secret",
        },
        endpoint_url=credential_url,
    )
    executor = _executor()
    executor.sweep["spec"]["podTemplate"] = {
        "env": [
            {
                "name": AIPERF_INJECTED_ENDPOINT_URLS,
                "valueFrom": {
                    "secretKeyRef": {"name": "endpoint-credentials", "key": "urls"}
                },
            }
        ]
    }
    child_spec = executor._build_child_spec(run)
    metadata = executor._build_child_metadata(run, "s-v08-t0")

    assert run.cfg.endpoint.urls == [credential_url]
    assert child_spec["benchmark"]["endpoint"]["urls"] == [
        "https://<redacted>@host/v1?token=<redacted>"
    ]
    assert child_spec["podTemplate"]["env"][0]["valueFrom"]["secretKeyRef"] == {
        "name": "endpoint-credentials",
        "key": "urls",
    }

    status_writer = MagicMock()
    status_writer.partial_children = AsyncMock()
    executor._status_writer = status_writer
    await executor._record_terminal_child(
        "s-v08-t0",
        run,
        {"status": {"runEpoch": "1714000042"}},
        MagicMock(success=True, was_cancelled=False, error=""),
    )

    partial = status_writer.partial_children.await_args.kwargs["children"]
    response = _children_manifest_from_doc(
        {"sweep_run_epoch": "1714000000", "children": partial}, epoch=None
    )
    exposed = orjson.dumps(
        {
            "metadata": metadata,
            "status_children": partial,
            "api": response.model_dump(by_alias=True),
        }
    ).decode()
    assert "axis-secret" not in exposed
    assert "user:pass" not in exposed
    assert "query-secret" not in exposed
    assert "header-secret" not in exposed
    assert "<redacted>" in exposed


def test_display_label_redacts_compound_credential_path_value() -> None:
    label = "concurrency=8,variables.my_auth_token=opaque-secret"

    redacted = redact_sweep_display_label(label)

    assert redacted == "concurrency=8,variables.my_auth_token=<redacted>"
    assert "opaque-secret" not in redacted


@pytest.mark.asyncio
async def test_live_manifest_reads_values_from_the_child_annotation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The live route is the ONLY manifest source while a child is still
    running -- ``status.aggregate.children`` gains an entry only at terminal
    phase."""
    listed = {
        "items": [
            {
                "metadata": {
                    "name": "s-v08-t0",
                    "labels": {
                        "aiperf.nvidia.com/sweep": "s",
                        "aiperf.nvidia.com/sweep-run-epoch": "1714000000",
                        "aiperf.nvidia.com/variation-index": "08",
                        "aiperf.nvidia.com/variation-label": "search_iter_0008",
                    },
                    "annotations": {
                        VARIATION_VALUES_ANNOTATION: (
                            '{"phases.profiling.concurrency":17}'
                        )
                    },
                },
                "status": {"phase": "Profiling", "runEpoch": "1714000042"},
            },
            {
                "metadata": {
                    "name": "s-v09-t0",
                    "labels": {
                        "aiperf.nvidia.com/sweep": "s",
                        "aiperf.nvidia.com/variation-index": "09",
                        "aiperf.nvidia.com/variation-label": "search_iter_0009",
                    },
                },
                "status": {"phase": "Pending"},
            },
        ]
    }
    custom = MagicMock()
    custom.list_namespaced_custom_object = AsyncMock(return_value=listed)
    monkeypatch.setattr(
        "aiperf.operator.routers._sweeps_live.k8s.CustomObjectsApi",
        lambda _api: custom,
    )

    response = await children_manifest_from_live_aiperfjobs(None, "ns", "s")

    assert response is not None
    assert response.children[0].variation_values == (
        '{"phases.profiling.concurrency":17}'
    )
    assert response.children[1].variation_values == ""
