# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The swept values must reach the UI on a LIVE sweep, through the real chain.

A previous pass added ``variation_values`` to the children manifest so adaptive
variations could render ``concurrency=17`` instead of ``search_iter_0008``. It
was dead on a live sweep: SweepDetail skips ``GET /sweeps/{ns}/{name}/children``
whenever ``detail.children`` is non-empty (sweep-detail.js), which is always
true on a live sweep because the sweeps router builds ``detail.children`` from
``list_all_jobs`` -- and ``AIPerfJobInfo`` had no such field, so
``resolveSweepManifest`` fell to its last-priority source and lost the values.

Every test here starts from a payload built by the PRODUCTION serializers
(``AIPerfJobCR.to_info()`` / ``_archived_from_summary``) and runs it through the
real ``resolveSweepManifest`` -> ``indexVariationValues`` chain in node. Tests
that inject a manifest literal straight into the helper cannot catch this class
of defect, which is exactly how it shipped.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import orjson
import pytest
from pytest import param

from aiperf.kubernetes.models import AIPerfJobCR
from aiperf.operator.job_union import _archived_from_summary
from tests.unit.ui.node_utils import run_node

_HELPERS = (
    Path(__file__).resolve().parents[3]
    / "src"
    / "aiperf"
    / "operator"
    / "ui"
    / "pages"
    / "sweep-detail-helpers.js"
)

_VALUES_JSON = '{"phases.profiling.concurrency":17}'


def _live_child_cr(
    *,
    name: str = "bo4-v08",
    namespace: str = "bench",
    variation_index: str = "08",
    variation_label: str = "search_iter_0008",
    annotations: dict[str, str] | None = None,
    phase: str = "Running",
) -> dict[str, Any]:
    """A live sweep child AIPerfJob CR as the apiserver returns it.

    Label and annotation keys mirror
    ``sweep_controller/k8s_executor._build_child_metadata``: the values live in
    an annotation because label values cap at 63 chars and forbid JSON
    punctuation.
    """
    return {
        "metadata": {
            "name": name,
            "namespace": namespace,
            "creationTimestamp": "2026-08-04T18:08:43Z",
            "labels": {
                "aiperf.nvidia.com/sweep": "bo4",
                "aiperf.nvidia.com/variation-index": variation_index,
                "aiperf.nvidia.com/variation-label": variation_label,
            },
            "annotations": {
                "aiperf.nvidia.com/run-identity": "abc123",
                **(
                    annotations
                    if annotations is not None
                    else {"aiperf.nvidia.com/variation-values": _VALUES_JSON}
                ),
            },
        },
        "spec": {"benchmark": {"models": ["google/gemma-4-31B-it"]}},
        "status": {"phase": phase, "jobId": name},
    }


def _sweep_detail_payload(children_crs: list[dict[str, Any]]) -> dict[str, Any]:
    """Build ``GET /sweeps/{ns}/{name}`` the way ``_get_sweep_impl`` does.

    Mirrors ``routers/sweeps._get_sweep_impl``: ``children`` is
    ``AIPerfJobInfo.model_dump(by_alias=True)`` for every job whose
    ``sweep_name`` matches. ``status`` has no ``aggregate.children`` and no
    ``runs`` -- the mid-run reality this defect is about, since the
    sweep-controller patches ``aggregate.children`` and appends ``runs[]``
    only as children reach a terminal phase.
    """
    return {
        "sweep": {"namespace": "bench", "name": "bo4", "source": "live"},
        "status": {"phase": "Running", "totalVariations": 14},
        "children": [
            AIPerfJobCR.model_validate(cr).to_info().model_dump(by_alias=True)
            for cr in children_crs
        ],
        "pods": [],
    }


def _resolve_values(detail: dict[str, Any], archived_children: Any = None) -> Any:
    """Run the page's real chain: resolveSweepManifest -> indexVariationValues.

    Returns ``{index: {valuesLabel, ...}}`` keyed by variation index, plus the
    manifest length so a test can tell "no values" from "no manifest".
    """
    script = f"""
      import {{ resolveSweepManifest, indexVariationValues }}
        from '{_HELPERS.as_posix()}';
      const detail = {json.dumps(detail)};
      const archivedChildren = {json.dumps(archived_children)};
      const manifest = resolveSweepManifest({{ detail, archivedChildren }});
      const byIndex = indexVariationValues({{
        manifest,
        statusRuns: detail?.status?.runs ?? null,
      }});
      console.log(JSON.stringify({{
        manifestLength: manifest.length,
        values: Object.fromEntries(byIndex),
      }}));
    """
    return json.loads(run_node(script))


# ---------------------------------------------------------------------------
# The live path
# ---------------------------------------------------------------------------


def test_live_sweep_detail_children_carry_swept_values_to_the_helper() -> None:
    """The regression test: values survive the chain a live sweep actually uses.

    No ``aggregate.children``, no ``archivedChildren`` (the page skipped that
    fetch), no ``status.runs`` -- so ``resolveSweepManifest`` resolves to
    ``detail.children``. Before the fix that source had no values field at all
    and this returned ``{}``.
    """
    detail = _sweep_detail_payload([_live_child_cr()])

    out = _resolve_values(detail, archived_children=None)

    assert out["manifestLength"] == 1, "detail.children must be the resolved source"
    assert out["values"]["8"]["valuesLabel"] == "concurrency=17"


def test_live_sweep_detail_children_is_the_source_that_actually_resolves() -> None:
    """Pin the precondition: the page's skip really does reach priority 4.

    If ``detail.children`` ever stops being populated for live sweeps, this
    fails and the fix above can be re-evaluated rather than silently kept.
    """
    detail = _sweep_detail_payload([_live_child_cr()])

    assert detail["children"], "sweeps router populates children for live sweeps"
    assert "aggregate" not in detail["status"]
    assert "runs" not in detail["status"]
    assert "variationValues" in detail["children"][0]


def test_multi_trial_children_share_one_variation_entry() -> None:
    """Trials of one variation collapse to a single indexed entry."""
    detail = _sweep_detail_payload(
        [
            _live_child_cr(name="bo4-v08-t0"),
            _live_child_cr(name="bo4-v08-t1"),
        ]
    )

    out = _resolve_values(detail)

    assert out["manifestLength"] == 2
    assert list(out["values"]) == ["8"]
    assert out["values"]["8"]["valuesLabel"] == "concurrency=17"


@pytest.mark.parametrize(
    "annotations",
    [
        param({}, id="annotation-absent"),
        param({"aiperf.nvidia.com/variation-values": ""}, id="annotation-empty"),
        param(
            {"aiperf.nvidia.com/variation-values": "not json"},
            id="annotation-unparseable",
        ),
        param(
            {
                "aiperf.nvidia.com/variation-values": orjson.dumps(
                    {"__aiperf_truncated__": True, "limitBytes": 2048}
                ).decode()
            },
            id="writer-side-truncation-marker",
        ),
        param(
            {"aiperf.nvidia.com/variation-values": '{"tuning":{"a":1}}'},
            id="non-scalar-values",
        ),
    ],
)  # fmt: skip
def test_absent_or_unusable_values_leave_the_label_to_stand_alone(
    annotations: dict[str, str],
) -> None:
    """No half-formed descriptor: the entry is omitted, not stored empty.

    ``annotation-unparseable`` used to be carved out here: this test asserted
    ``valuesLabel == "not json"``, on the rationale that "an operator-authored
    plain-text annotation is more useful than nothing". That was wrong on three
    counts, so the carve-out is gone and every case now produces no entry.

    1. No writer can produce it. ``_bounded_variation_values_json``
       (sweep_controller/k8s_executor.py:188-192) always emits
       ``orjson.dumps(...)``, valid JSON in both the normal and the truncated
       branch. A string that fails ``JSON.parse`` is corruption, not authorship.
    2. The failure mode is inverted. ``valuesLabel`` is what every sweep surface
       LEADS with, demoting the real ``variation_label`` beneath it. A JSON
       object clipped mid-encode would put ``{"phases.profiling.conc`` where
       ``search_iter_0008`` belongs -- the descriptor is not merely unhelpful,
       it displaces the identifier that still works.
    3. It contradicted the sibling implementations.
       ``components/live-variations-card.parseVariationValues`` returned ``[]``
       for the same input, and ``kubernetes/results.py:_cell_values`` returned
       ``""`` while its docstring claimed "Same rule as the UI's
       ``formatVariationValues``". Two of three already agreed.

    The docstring on ``formatVariationValues`` always said it "Returns null when
    there is nothing meaningful to show, so callers can fall back to the raw
    label"; returning the raw string made that fallback unreachable. Code and
    docstring now agree.
    """
    detail = _sweep_detail_payload([_live_child_cr(annotations=annotations)])

    out = _resolve_values(detail)

    assert out["manifestLength"] == 1
    assert out["values"] == {}


def test_status_runs_fills_gaps_the_live_children_do_not_cover() -> None:
    """Manifest first, ``status.runs[]`` for indices it did not cover.

    Contradicts the comment this change deleted, which claimed ``status.runs[]``
    was the only source. Variation 8 is live with an annotation; variation 3 has
    already terminated and only exists in ``status.runs[]``.
    """
    detail = _sweep_detail_payload(
        [
            _live_child_cr(),
            _live_child_cr(
                name="bo4-v03",
                variation_index="03",
                variation_label="search_iter_0003",
                annotations={},
                phase="Succeeded",
            ),
        ]
    )
    detail["status"]["runs"] = [
        {"index": 3, "values": '{"phases.profiling.concurrency":4}'}
    ]

    out = _resolve_values(detail)

    assert out["values"]["8"]["valuesLabel"] == "concurrency=17"
    assert out["values"]["3"]["valuesLabel"] == "concurrency=4"


def test_manifest_values_win_over_a_truncated_status_runs_copy() -> None:
    """The manifest's larger byte budget must not be overridden by status.

    ``status.runs[].values`` is capped at 256 bytes and degrades to a truncation
    marker sooner than the 2048-byte annotation, so a status entry must never
    displace a manifest entry for the same index.
    """
    detail = _sweep_detail_payload([_live_child_cr()])
    detail["status"]["runs"] = [
        {
            "index": 8,
            "values": orjson.dumps(
                {"__aiperf_truncated__": True, "limitBytes": 256}
            ).decode(),
        }
    ]

    out = _resolve_values(detail)

    assert out["values"]["8"]["valuesLabel"] == "concurrency=17"


def test_swept_credentials_are_redacted_before_they_reach_the_page() -> None:
    """A swept ``endpoint.apiKey`` must not render its value in the UI."""
    detail = _sweep_detail_payload(
        [
            _live_child_cr(
                annotations={
                    "aiperf.nvidia.com/variation-values": '{"endpoint.apiKey":"sk-live-secret"}'
                }
            )
        ]
    )

    assert "sk-live-secret" not in json.dumps(detail)

    out = _resolve_values(detail)

    assert "sk-live-secret" not in json.dumps(out)
    assert out["values"]["8"]["valuesLabel"].startswith("apiKey=")


# ---------------------------------------------------------------------------
# The archived path -- same last-priority source, different builder
# ---------------------------------------------------------------------------


def _archived_child_payload(tmp_path: Path, *, with_manifest: bool) -> dict[str, Any]:
    """Build ``detail.children`` for an archived sweep child on a PVC tree.

    The ``sweep.json`` marker is what makes ``detail.children`` non-empty for an
    archived sweep, which is what makes the page skip the children fetch -- so
    the archived half hits the same last-priority source as the live one.
    """
    namespace, child, sweep, epoch = "bench", "bo4-v08", "bo4", "1785866923"
    name_dir = tmp_path / namespace / child
    name_dir.mkdir(parents=True)
    (name_dir / "sweep.json").write_bytes(
        orjson.dumps(
            {
                "sweep_name": sweep,
                "variation_index": 8,
                "variation_label": "search_iter_0008",
                "trial_index": None,
                "sweep_run_epoch": epoch,
                "child_run_epoch": "1785866932362420",
            }
        )
    )
    if with_manifest:
        sweep_dir = tmp_path / namespace / "sweeps" / sweep / epoch
        sweep_dir.mkdir(parents=True)
        (sweep_dir / "children.json").write_bytes(
            orjson.dumps(
                {
                    "sweep_run_epoch": epoch,
                    "children": [
                        {
                            "namespace": namespace,
                            "name": child,
                            "variation_index": 8,
                            "variation_label": "search_iter_0008",
                            "variation_values": _VALUES_JSON,
                        }
                    ],
                }
            )
        )

    info = _archived_from_summary(
        namespace,
        child,
        {"request_throughput": {"avg": 12.5}},
        mtime_iso="2026-08-04T18:35:03Z",
        name_dir=name_dir,
        base_dir=tmp_path,
    )
    return {
        "sweep": {"namespace": namespace, "name": sweep, "source": "archived"},
        # ``synthesize_sweep_status_from_aggregate`` emits neither
        # ``aggregate`` nor ``runs``, so there is no other source at all.
        "status": {"phase": "Succeeded"},
        "children": [info.model_dump(by_alias=True)],
        "pods": [],
    }


def test_archived_sweep_child_carries_values_from_the_epoch_manifest(
    tmp_path: Path,
) -> None:
    """``sweep.json`` has no values field; ``children.json`` for that epoch does."""
    detail = _archived_child_payload(tmp_path, with_manifest=True)

    out = _resolve_values(detail)

    assert out["manifestLength"] == 1
    assert out["values"]["8"]["valuesLabel"] == "concurrency=17"


def test_archived_sweep_child_without_a_manifest_degrades_to_the_label(
    tmp_path: Path,
) -> None:
    """Pre-``variation_values`` archives must not error, just carry no values."""
    detail = _archived_child_payload(tmp_path, with_manifest=False)

    assert detail["children"][0]["variationValues"] is None

    out = _resolve_values(detail)

    assert out["manifestLength"] == 1
    assert out["values"] == {}
