# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Executable contract for the drift-free AIPerf DynoSim facade."""

from __future__ import annotations

import json
import sys
from types import SimpleNamespace
from typing import Any

import pandas as pd
import pytest

from aiperf.cli_commands import dynosim


def _invoke_app(arguments: list[str]) -> None:
    """Exercise Cyclopts' real argv parser while preserving normal status 0."""
    try:
        dynosim.app(arguments, exit_on_error=False)
    except SystemExit as error:
        if error.code not in (None, 0):
            raise


def test_run_forwards_every_raw_token_without_parsing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    authored = [
        "trace one.jsonl",
        "--replay-mode",
        "online",
        "--router-config",
        '{"router_queue_policy":"wspt","router_temperature":0.25}',
        "--aic-nextn-accept-rates=1,0.5",
        "--report-jsonl",
        "records.jsonl",
    ]
    observed: list[list[str]] = []

    def canonical_main(arguments: list[str]) -> int:
        observed.append(arguments)
        return 0

    monkeypatch.setattr(dynosim, "_import_symbol", lambda *_: canonical_main)

    dynosim.run(authored)

    assert observed == [authored]


def test_real_cyclopts_parser_preserves_the_complete_raw_vector(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    authored = [
        "trace one.jsonl",
        "trace-two.jsonl",
        "--replay-mode=online",
        "--router-config",
        '{"router_queue_policy":"wspt","router_temperature":0.25}',
        "--aic-nextn-accept-rates",
        "1,0.5,0",
        "--report-jsonl",
        "records with spaces.jsonl",
    ]
    observed: list[list[str]] = []

    def canonical_main(arguments: list[str]) -> int:
        observed.append(arguments)
        return 0

    monkeypatch.setattr(dynosim, "_import_symbol", lambda *_: canonical_main)

    _invoke_app(["run", *authored])

    assert observed == [authored]


def test_mocker_forwards_raw_tokens_and_restores_process_argv(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    authored = [
        "--model-path",
        "Qwen/Qwen3-0.6B",
        "--num-g2-blocks",
        "2048",
        "--event-plane=zmq",
        "--no-enable-prefix-caching",
    ]
    original = ["pytest", "sentinel"]
    monkeypatch.setattr(sys, "argv", original)
    observed: list[list[str]] = []

    def canonical_main() -> None:
        observed.append(list(sys.argv))

    monkeypatch.setattr(dynosim, "_import_symbol", lambda *_: canonical_main)

    dynosim.mocker(authored)

    assert observed == [["aiperf dynosim mocker", *authored]]
    assert sys.argv is original


def test_nonzero_canonical_status_is_the_aiperf_status() -> None:
    with pytest.raises(SystemExit, match="7"):
        dynosim._run_argparse_main(
            lambda _: 7,
            ["--bad"],
            accepts_argv=True,
            program="aiperf dynosim run",
        )


class _FakeSpec:
    seen: dict[str, Any] | None = None

    @classmethod
    def model_validate(cls, value: dict[str, Any]) -> dict[str, Any]:
        cls.seen = value
        return value


def _search_result() -> SimpleNamespace:
    evaluated = pd.DataFrame.from_records(
        [{"score": float("inf"), "output_throughput_tok_s": 12.5}]
    )
    feasible = pd.DataFrame.from_records(
        [{"score": 1.0, "output_throughput_tok_s": 12.5}]
    )
    return SimpleNamespace(
        best_feasible={"score": 1.0},
        best_infeasible={"score": float("inf")},
        evaluated_df=evaluated,
        feasible_df=feasible,
    )


@pytest.mark.parametrize(
    ("operation", "function_name"),
    [
        (dynosim.SweepOperation.AGG, "optimize_dense_agg_with_replay"),
        (dynosim.SweepOperation.DISAGG, "optimize_dense_disagg_with_replay"),
    ],
)
def test_sweep_validates_canonical_spec_and_emits_strict_json(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
    operation: dynosim.SweepOperation,
    function_name: str,
) -> None:
    spec_path = tmp_path / "sweep.json"
    spec_path.write_text('{"engine":{"model":"m"}}', encoding="utf-8")
    output = tmp_path / "nested" / "result.json"
    called: list[dict[str, Any]] = []

    def evaluate(spec: dict[str, Any]) -> SimpleNamespace:
        called.append(spec)
        return _search_result()

    module = SimpleNamespace(
        ReplayOptimizeSpec=_FakeSpec,
        optimize_dense_agg_with_replay=evaluate,
        optimize_dense_disagg_with_replay=evaluate,
        compare_agg_and_disagg_with_replay=evaluate,
        compare_aic_and_replay_disagg=evaluate,
    )
    monkeypatch.setattr(
        dynosim,
        "_import_symbol",
        lambda *_: _FakeSpec,
    )
    real_import = dynosim.importlib.import_module
    monkeypatch.setattr(
        dynosim.importlib,
        "import_module",
        lambda name: module
        if name == "dynamo.profiler.utils.replay_optimize"
        else real_import(name),
    )

    dynosim.sweep(spec_path, operation=operation, output=output)

    assert called == [{"engine": {"model": "m"}}]
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["operation"] == operation.value
    assert payload["best_infeasible"]["score"] is None
    assert payload["evaluated"][0]["score"] is None
    assert getattr(module, function_name) is evaluate


def test_compare_topologies_preserves_both_complete_search_tables() -> None:
    search = _search_result()
    payload = dynosim._result_payload(
        dynosim.SweepOperation.COMPARE_TOPOLOGIES,
        {
            "chosen_mode": "disagg",
            "chosen_best": {"score": 3.0},
            "agg_result": search,
            "disagg_result": search,
        },
    )

    assert payload["chosen_mode"] == "disagg"
    assert len(payload["agg"]["evaluated"]) == 1
    assert len(payload["disagg"]["feasible"]) == 1
