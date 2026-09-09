# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Regression coverage for prefixed exports across Kubernetes completion."""

from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, patch

import orjson
import pytest

from aiperf.kubernetes.results import _resolve_key_result_files
from aiperf.kubernetes.spec_converter import (
    DEFAULT_KEY_EXPORT_NAMES,
    key_export_names,
)
from aiperf.operator.artifact_names import find_summary_path
from aiperf.operator.environment import _ResultsSettings
from aiperf.operator.handlers._completion_fetch import _fetch_once_into_state
from aiperf.operator.handlers.completion import (
    _has_key_result_files,
    _key_files_materialized,
    _parse_metrics_from_files,
)


def _results_settings(path: Path) -> _ResultsSettings:
    return _ResultsSettings(DIR=path, RETAIN_RUNS=5)


class TestKeyExportNames:
    def test_default_names_are_stable(self) -> None:
        assert key_export_names({"benchmark": {}}) == DEFAULT_KEY_EXPORT_NAMES

    def test_prefix_uses_exporter_suffix_rules(self) -> None:
        names = key_export_names(
            {"benchmark": {"artifacts": {"prefix": "nightly_raw.jsonl"}}}
        )

        assert names.names == {"nightly.json", "nightly.csv"}

    def test_templated_prefix_uses_envelope_variables(self) -> None:
        names = key_export_names(
            {
                "variables": {"tag": "nightly"},
                "benchmark": {"artifacts": {"prefix": "{{ tag }}_run"}},
            }
        )

        assert names.names == {"nightly_run.json", "nightly_run.csv"}

    def test_persisted_spec_resolves_custom_summary(self, tmp_path: Path) -> None:
        (tmp_path / "job_spec.json").write_bytes(
            orjson.dumps({"benchmark": {"artifacts": {"prefix": "nightly"}}})
        )
        expected = tmp_path / "nightly.json"
        expected.write_bytes(b"{}")

        assert find_summary_path(tmp_path) == expected


class TestPrefixedCompletion:
    @pytest.mark.asyncio
    async def test_fetch_accepts_prefixed_authoritative_export(
        self, tmp_path: Path
    ) -> None:
        progress_client = AsyncMock()
        progress_client.get_metrics.return_value = {"metrics": {"x": 1}}
        progress_client.download_all_results.return_value = ["nightly.json"]
        state: dict[str, Any] = {
            "metrics": None,
            "downloaded": None,
            "checkpoints": None,
        }

        with patch(
            "aiperf.operator.handlers._completion_fetch.OperatorEnvironment.RESULTS",
            new=_results_settings(tmp_path),
        ):
            result = await _fetch_once_into_state(
                key="ns/job",
                controller_host="host",
                dest_dir=tmp_path,
                progress_client=progress_client,
                state=state,
                key_files=frozenset({"nightly.json", "nightly.csv"}),
            )

        assert result.downloaded == ["nightly.json"]

    def test_materialized_and_parse_gates_use_prefixed_name(
        self, tmp_path: Path
    ) -> None:
        epoch = "1714064523"
        run_dir = tmp_path / "ns" / "job" / epoch
        run_dir.mkdir(parents=True)
        payload = {"metrics": {"request_throughput": {"avg": 5.0}}}
        (run_dir / "nightly.json").write_bytes(orjson.dumps(payload))
        names = frozenset({"nightly.json", "nightly.csv"})

        with patch(
            "aiperf.operator.handlers.completion.OperatorEnvironment.RESULTS",
            new=_results_settings(tmp_path),
        ):
            assert _has_key_result_files(["nightly.json"], key_names=names)
            assert _key_files_materialized("ns", "job", epoch, key_names=names)
            assert (
                _parse_metrics_from_files(
                    ["nightly.json"],
                    "ns",
                    "job",
                    epoch=epoch,
                    json_name="nightly.json",
                )
                == payload
            )


@pytest.mark.asyncio
async def test_kube_results_resolves_prefixed_summary_and_console() -> None:
    cr = {"spec": {"benchmark": {"artifacts": {"prefix": "nightly"}}}}
    with patch(
        "aiperf.kubernetes.results._get_aiperfjob_cr",
        new=AsyncMock(return_value=cr),
    ):
        files = await _resolve_key_result_files(AsyncMock(), "ns", "job")

    assert files == ["metrics.json", "nightly.json", "nightly_console.txt"]
