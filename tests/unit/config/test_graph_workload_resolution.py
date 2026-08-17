# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Graph-ness is derived at most once per run, either eagerly by the resolver chain or lazily via the memoizing accessor."""

from __future__ import annotations

from pathlib import Path

import pytest

from aiperf.config.dataset.resolver import DatasetResolver
from aiperf.config.flags.cli_config import CLIConfig
from aiperf.config.resolution.plan import BenchmarkRun
from tests.unit.config.conftest import GRAPH_TRACE_FIXTURE as GRAPH_MIN
from tests.unit.conftest import make_run_from_cli


def _run(**cli_overrides: object) -> BenchmarkRun:
    """Build an un-chained ``BenchmarkRun`` from CLI flags (no resolver chain applied)."""
    cfg = CLIConfig(model_names=["test-model"], **cli_overrides)
    return make_run_from_cli(cfg)


def test_graph_file_resolves_ref_at_resolver_chain_time() -> None:
    """A graph trace input file resolves a ``GraphWorkloadRef`` during the resolver chain."""
    run = _run(input_file=str(GRAPH_MIN))
    DatasetResolver().resolve(run)
    assert run.resolved.graph_workload_resolved is True
    ref = run.resolved.graph_workload
    assert ref is not None
    assert ref.format == "dynamo_trace"
    assert ref.path == GRAPH_MIN


def test_synthetic_run_resolves_none_with_marker() -> None:
    """A synthetic run resolves to None but still sets the marker, distinguishing "not a graph run" from "never checked"."""
    run = _run()
    DatasetResolver().resolve(run)
    assert run.resolved.graph_workload_resolved is True
    assert run.resolved.graph_workload is None


def test_accessor_derives_once_and_memoizes(monkeypatch: pytest.MonkeyPatch) -> None:
    """On a chain-less run the accessor derives exactly once and returns the same ref thereafter."""
    from aiperf.dataset.graph import workload_detect

    run = _run(input_file=str(GRAPH_MIN))
    calls: list[Path] = []
    real_detect = workload_detect._detect_graph_workload_format

    def spy(path: Path):
        calls.append(path)
        return real_detect(path)

    monkeypatch.setattr(workload_detect, "_detect_graph_workload_format", spy)
    first = workload_detect.resolve_graph_workload(run)
    second = workload_detect.resolve_graph_workload(run)
    assert first is not None
    assert first.format == "dynamo_trace"
    assert second is first
    assert len(calls) == 1, "the second accessor call must read the memo"


def test_graph_format_override_wins_without_detection(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """An explicit ``--graph-format`` decides the format without invoking registry detection at all."""
    from aiperf.dataset.graph import workload_detect

    plain = tmp_path / "plain.jsonl"
    plain.write_text('{"messages": [{"role": "user", "content": "hi"}]}\n')
    run = _run(input_file=str(plain), graph_format="dynamo_trace")
    calls: list[Path] = []
    monkeypatch.setattr(
        workload_detect, "_detect_graph_workload_format", lambda p: calls.append(p)
    )
    ref = workload_detect.resolve_graph_workload(run)
    assert ref is not None
    assert ref.format == "dynamo_trace"
    assert ref.path == plain
    assert calls == [], "--graph-format must short-circuit registry detection"


def test_graph_format_bypasses_custom_dataset_detection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An explicit graph adapter bypasses custom dataset type detection."""
    run = _run(
        input_file=str(GRAPH_MIN),
        graph_format="dynamo_trace",
        request_count=3,
    )
    monkeypatch.setattr(
        DatasetResolver,
        "_detect_type",
        lambda _path: pytest.fail("graph_format must skip custom detection"),
    )

    DatasetResolver().resolve(run)

    assert run.resolved.graph_workload is not None
    assert run.resolved.graph_workload.format == "dynamo_trace"


def test_explicit_graph_format_makes_resolution_failure_loud(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """With ``--graph-format`` given, a resolution failure raises instead of degrading.

    Without the override the same failure returns ``None`` (best-effort sniff).
    With it the user has ASSERTED this is a graph workload, so falling back to
    the linear pipeline would be a silently wrong run.
    """
    from aiperf.dataset.graph import workload_detect

    plain = tmp_path / "plain.jsonl"
    plain.write_text('{"messages": [{"role": "user", "content": "hi"}]}\n')

    def _boom(*a: object, **k: object):
        raise RuntimeError("ref build exploded")

    # Fail AFTER the override is read, so the two runs below differ only in
    # whether --graph-format was given.
    monkeypatch.setattr(
        "aiperf.config.resolution.plan.GraphWorkloadRef", _boom, raising=True
    )

    with pytest.raises(RuntimeError, match="ref build exploded"):
        workload_detect.resolve_graph_workload(
            _run(input_file=str(plain), graph_format="dynamo_trace")
        )
    # Same failure, no assertion from the user: best-effort, degrade to None.
    assert (
        workload_detect.resolve_graph_workload(_run(input_file=str(GRAPH_MIN))) is None
    )


def test_explicit_custom_dataset_type_bypasses_graph_detection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An explicit custom loader prevents graph adapter sniffing."""
    from aiperf.common.enums import DatasetFormat
    from aiperf.dataset.graph import workload_detect

    run = _run(
        input_file=str(GRAPH_MIN),
        custom_dataset_type=DatasetFormat.MULTI_TURN,
        request_count=3,
    )
    monkeypatch.setattr(
        workload_detect,
        "_detect_graph_workload_format",
        lambda _path: pytest.fail("explicit custom type must skip graph detection"),
    )
    monkeypatch.setattr(
        DatasetResolver,
        "_count_records_and_sessions",
        lambda _self, _path, _type: (1, 1),
    )

    DatasetResolver().resolve(run)

    assert run.resolved.graph_workload_resolved is True
    assert run.resolved.graph_workload is None


def test_explicit_single_turn_bypasses_graph_detection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An explicitly selected default custom loader also bypasses detection."""
    from aiperf.common.enums import DatasetFormat
    from aiperf.dataset.graph import workload_detect

    run = _run(
        input_file=str(GRAPH_MIN),
        custom_dataset_type=DatasetFormat.SINGLE_TURN,
        request_count=3,
    )
    monkeypatch.setattr(
        workload_detect,
        "_detect_graph_workload_format",
        lambda _path: pytest.fail("explicit single_turn must skip graph detection"),
    )
    monkeypatch.setattr(
        DatasetResolver,
        "_count_records_and_sessions",
        lambda _self, _path, _type: (1, 1),
    )

    DatasetResolver().resolve(run)

    assert run.resolved.graph_workload is None


def test_graph_format_and_custom_dataset_type_raises() -> None:
    """The two explicit dataset selectors conflict before workload detection."""
    from aiperf.common.enums import DatasetFormat
    from tests.unit.conftest import make_run_from_cli

    with pytest.raises(ValueError, match="mutually exclusive"):
        make_run_from_cli(
            CLIConfig(
                model_names=["test-model"],
                input_file=str(GRAPH_MIN),
                graph_format="dynamo_trace",
                custom_dataset_type=DatasetFormat.MULTI_TURN,
                request_count=3,
            )
        )


def test_graph_detection_survives_sweep_dump_validate_round_trip() -> None:
    """Auto-detected graph mode must not flip inside a sweep subprocess.

    The sweep orchestrator writes ``run_config.json`` per cell and the
    subprocess re-validates it. ``model_fields_set`` does not survive that
    boundary -- every dumped key returns marked "set" -- so a resolved-default
    ``format`` read raw would look like an explicit custom-loader selection and
    silently suppress graph detection in the cell while the identical run
    auto-detects standalone. Both predicates therefore read the SERIALIZED
    ``format`` value (None when unset).
    """
    import orjson

    from aiperf.config.dataset import FileDataset
    from aiperf.config.resolution.predicates import is_graph_dataset
    from aiperf.dataset.graph.workload_detect import _has_explicit_custom_format

    run = _run(input_file=str(GRAPH_MIN))
    dataset = run.cfg.get_default_dataset()

    assert is_graph_dataset(dataset) is True
    assert _has_explicit_custom_format(run) is False

    # Mirror local_executor._prepare_run_artifacts / subprocess_runner exactly.
    revalidated = FileDataset.model_validate(
        orjson.loads(orjson.dumps(dataset.model_dump(mode="json", exclude_none=True)))
    )
    # This is the mechanism: an unset ``format`` is None, so ``exclude_none``
    # drops it from the dump entirely and it never comes back marked "set".
    # With the old non-None default it WAS dumped, returned marked "set", and a
    # raw ``model_fields_set`` read then saw a phantom explicit selection.
    assert "format" not in revalidated.model_fields_set
    assert revalidated.format is None
    assert is_graph_dataset(revalidated) is True


def test_explicit_custom_format_still_suppresses_graph_after_round_trip() -> None:
    """The other direction: a real explicit ``format`` must stay explicit.

    Built directly rather than through the CLI path: the point is the flag's
    survival across serialization, and routing a gzipped graph fixture through
    an explicit text loader would fail on decode before reaching the predicate.
    """
    import orjson

    from aiperf.common.enums import DatasetFormat
    from aiperf.config.dataset import FileDataset
    from aiperf.config.resolution.predicates import is_graph_dataset

    dataset = FileDataset(
        name="default",
        type="file",
        path=str(GRAPH_MIN),
        format=DatasetFormat.MOONCAKE_TRACE,
    )
    assert dataset.format is not None
    assert is_graph_dataset(dataset) is False

    revalidated = FileDataset.model_validate(
        orjson.loads(orjson.dumps(dataset.model_dump(mode="json", exclude_none=True)))
    )
    assert revalidated.format is not None
    assert is_graph_dataset(revalidated) is False


def test_graph_run_revalidates_through_the_sweep_boundary() -> None:
    """End-to-end shape of the bug: the run must survive re-validation.

    ``check_phase_dataset_compatibility`` runs from an ``AIPerfConfig``
    model_validator, so it re-executes on every ``model_validate`` -- including
    the sweep subprocess's -- with no memo to short-circuit it. It exempts graph
    datasets from the requires-a-stop-condition rule via ``is_graph_dataset``.
    When that predicate read ``model_fields_set`` instead of the serialized
    the ``format`` value, a graph run with no explicit stop condition validated
    standalone and was REJECTED after the round-trip.
    """
    import orjson

    run = _run(input_file=str(GRAPH_MIN))
    blob = orjson.dumps(run.model_dump(mode="json", exclude_none=True))
    # Must not raise: mirrors subprocess_runner's BenchmarkRun.model_validate.
    BenchmarkRun.model_validate(orjson.loads(blob))


def test_explicit_format_null_agrees_across_both_graph_predicates() -> None:
    """``format: null`` written explicitly must not flip graph-ness.

    ``is_graph_dataset`` and ``DatasetResolver._resolve_one`` are two mirrors of
    the same precedence rule. Both must read the ``format`` VALUE: a YAML that
    spells ``format: null`` puts ``"format"`` in ``model_fields_set`` while the
    value stays None, so a provenance read via ``model_fields_set`` makes the
    two disagree -- the predicate calls the trace a graph workload, the resolver
    falls through to custom-loader type detection on the same file.
    """
    from aiperf.config.dataset import FileDataset
    from aiperf.config.dataset.resolver import _DatasetResolution
    from aiperf.config.resolution.predicates import is_graph_dataset

    dataset = FileDataset.model_validate(
        {
            "name": "explicit-null",
            "type": "file",
            "path": str(GRAPH_MIN),
            "format": None,
        }
    )
    assert "format" in dataset.model_fields_set, "fixture must spell format: null"
    assert dataset.format is None

    assert is_graph_dataset(dataset) is True

    # The resolver's mirror: a graph workload short-circuits before custom
    # dataset type detection, so no type/sampling is recorded for it.
    run = _run(input_file=str(GRAPH_MIN))
    acc = _DatasetResolution()
    DatasetResolver()._resolve_one(
        run=run,
        name=dataset.name,
        ds=dataset,
        format_map={},
        acc=acc,
    )
    assert acc.types == {}, "resolver must agree the explicit-null dataset is a graph"
