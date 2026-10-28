# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""``--end-conversation-on-truncation`` routing and default resolution."""

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest
from pytest import param

from aiperf.config import BenchmarkConfig
from aiperf.config.dataset.truncation import end_conversation_on_truncation_enabled
from aiperf.config.flags.cli_config import CLIConfig
from aiperf.config.flags.converter import convert_cli_to_aiperf
from aiperf.config.flags.resolver import resolve_config
from aiperf.controller.system_controller import SystemController
from aiperf.plugin import plugins
from aiperf.plugin.enums import PublicDatasetType


def _cli(**kwargs: Any) -> CLIConfig:
    return CLIConfig(
        model_names=["mock-model"],
        url="http://localhost:8000",
        request_count=1,
        **kwargs,
    )


def _dataset_field(**kwargs: Any) -> bool | None:
    benchmark = convert_cli_to_aiperf(_cli(**kwargs)).benchmark
    return benchmark.get_default_dataset().end_conversation_on_truncation


def _benchmark_config(dataset: dict[str, Any], **overrides: Any) -> BenchmarkConfig:
    config = {
        "models": ["test-model"],
        "endpoint": {"urls": ["http://localhost:8000/v1/chat/completions"]},
        "datasets": [{"name": "profiling", **dataset}],
        "phases": [
            {
                "name": "profiling",
                "type": "concurrency",
                "concurrency": 1,
                "requests": 5,
            }
        ],
    }
    config.update(overrides)
    return BenchmarkConfig(**config)


_SPEED_BENCH = {"type": "public", "dataset": "speed_bench_coding"}
_SHAREGPT = {"type": "public", "dataset": "sharegpt"}
_SYNTHETIC = {"type": "synthetic", "entries": 4, "prompts": {"isl": 32}}
_SPEED_BENCH_FILE = {
    "type": "file",
    "path": "sb.jsonl",
    "format": "speed_bench_qualitative",
}


class TestCliRouting:
    @pytest.mark.parametrize(
        "kwargs",
        [
            param({}, id="synthetic"),
            param({"public_dataset": "speed_bench_coding"}, id="public"),
        ],
    )  # fmt: skip
    def test_convert_unset_flag_leaves_dataset_default(self, kwargs: dict) -> None:
        assert _dataset_field(**kwargs) is None

    @pytest.mark.parametrize(
        "value,kwargs",
        [
            param(True, {}, id="on-synthetic"),
            param(False, {"public_dataset": "speed_bench_coding"}, id="off-public"),
            param(True, {"public_dataset": "sharegpt"}, id="on-public"),
        ],
    )  # fmt: skip
    def test_convert_explicit_flag_reaches_dataset(
        self, value: bool, kwargs: dict
    ) -> None:
        assert _dataset_field(end_conversation_on_truncation=value, **kwargs) is value

    def test_resolve_config_flag_overrides_yaml(self, tmp_path: Path) -> None:
        base = tmp_path / "base.yaml"
        base.write_text(
            """\
schemaVersion: "2.0"
benchmark:
  model: test-model
  endpoint:
    url: http://localhost:8000
  dataset:
    type: synthetic
    end_conversation_on_truncation: true
  phases:
    type: concurrency
    concurrency: 1
    requests: 5
"""
        )
        cli = CLIConfig(end_conversation_on_truncation=False)
        merged = resolve_config(CLIConfig(**cli.model_dump(exclude_unset=True)), base)
        dataset = merged.benchmark.get_default_dataset()
        assert dataset.end_conversation_on_truncation is False


class TestEnabledHelper:
    @pytest.mark.parametrize(
        "dataset,expected",
        [
            param(_SPEED_BENCH, True, id="speed-bench-default-on"),
            param({"type": "public", "dataset": "spec_al_mtbench"}, True, id="mtbench-default-on"),
            param(_SHAREGPT, False, id="other-public-default-off"),
            param(_SYNTHETIC, False, id="synthetic-default-off"),
            param(_SPEED_BENCH_FILE, True, id="custom-speed-bench-default-on"),
            param({**_SPEED_BENCH_FILE, "format": "multi_turn"}, False, id="other-custom-format-default-off"),
            param({"type": "file", "path": "sb.jsonl"}, False, id="auto-detected-format-default-off"),
            param({**_SPEED_BENCH_FILE, "end_conversation_on_truncation": False}, False, id="custom-explicit-off-wins"),
            param({**_SPEED_BENCH, "end_conversation_on_truncation": False}, False, id="explicit-off-wins"),
            param({**_SYNTHETIC, "end_conversation_on_truncation": True}, True, id="explicit-on-wins"),
        ],
    )  # fmt: skip
    def test_enabled_resolves_explicit_then_loader_default(
        self, dataset: dict[str, Any], expected: bool
    ) -> None:
        cfg = _benchmark_config(dataset)
        assert end_conversation_on_truncation_enabled(cfg.get_default_dataset()) is (
            expected
        )

    def test_metadata_default_on_for_every_speed_bench_qualitative_dataset(
        self,
    ) -> None:
        qualitative = [
            name
            for name in PublicDatasetType
            if str(name).startswith("speed_bench_")
            and not str(name).startswith("speed_bench_throughput")
        ]
        assert qualitative
        for name in qualitative:
            metadata = plugins.get_public_dataset_loader_metadata(name)
            assert metadata.end_conversation_on_truncation, name


class TestSystemControllerLogging:
    def _log(self, dataset: dict[str, Any], extra: dict[str, Any]) -> MagicMock:
        controller = MagicMock()
        controller.run.cfg = _benchmark_config(
            dataset,
            endpoint={
                "urls": ["http://localhost:8000/v1/chat/completions"],
                "extra": extra,
            },
        )
        SystemController._log_end_conversation_on_truncation(controller)
        return controller

    def test_log_enabled_with_ignore_eos_warns(self) -> None:
        controller = self._log(_SPEED_BENCH, {"ignore_eos": True})
        controller.info.assert_called_once()
        assert "ignore_eos" in controller.warning.call_args.args[0]

    def test_log_enabled_without_ignore_eos_only_informs(self) -> None:
        controller = self._log(_SPEED_BENCH, {})
        assert "default for speed_bench_coding" in controller.info.call_args.args[0]
        controller.warning.assert_not_called()

    @pytest.mark.parametrize(
        "dataset,expected",
        [
            param({**_SYNTHETIC, "prompts": {"isl": 32, "osl": 100}}, True, id="option-off-warns"),
            param({**_SYNTHETIC, "prompts": {"isl": 32, "osl": 100}, "end_conversation_on_truncation": True}, False, id="option-on-suppressed"),
        ],
    )  # fmt: skip
    def test_should_warn_osl_without_ignore_eos_respects_option(
        self, dataset: dict[str, Any], expected: bool
    ) -> None:
        controller = MagicMock()
        controller.run.cfg = _benchmark_config(dataset)
        assert (
            SystemController._should_warn_osl_without_ignore_eos(controller) is expected
        )

    def test_log_disabled_is_silent(self) -> None:
        controller = self._log(_SHAREGPT, {"ignore_eos": True})
        controller.info.assert_not_called()
        controller.warning.assert_not_called()
