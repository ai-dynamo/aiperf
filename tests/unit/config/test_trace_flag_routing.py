# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Agentx trace-replay flags (--prompt-corpus, --cache-bust, --inter-turn-delay-cap-seconds, ...) route natively onto file and public trace datasets through the converter."""

from __future__ import annotations

from pathlib import Path

import pytest
from pytest import param

from aiperf.config.flags._converter_dataset import build_dataset
from aiperf.config.flags.cli_config import CLIConfig
from aiperf.config.flags.converter import convert_cli_to_aiperf
from aiperf.plugin.enums import CustomDatasetType, PublicDatasetType

_WEKA_HF = PublicDatasetType.SEMIANALYSIS_CC_TRACES_WEKA_WITH_SUBAGENTS


@pytest.fixture
def trace_jsonl(tmp_path: Path) -> Path:
    p = tmp_path / "trace.jsonl"
    p.touch()
    return p


def _file_cli(trace_jsonl: Path, **extra: object) -> CLIConfig:
    return CLIConfig(
        model_names=["test-model"],
        endpoint_type="chat",
        input_file=str(trace_jsonl),
        custom_dataset_type=CustomDatasetType.MOONCAKE_TRACE,
        **extra,
    )


def _public_cli(**extra: object) -> CLIConfig:
    return CLIConfig(
        model_names=["test-model"],
        endpoint_type="chat",
        public_dataset=_WEKA_HF,
        **extra,
    )


class TestPromptCorpusRouting:
    def test_routes_onto_file_trace(self, trace_jsonl: Path) -> None:
        out = build_dataset(_file_cli(trace_jsonl, prompt_corpus="coding"))
        assert out["type"] == "file"
        assert out.get("prompts", {}).get("corpus") == "coding"
        ds = convert_cli_to_aiperf(
            _file_cli(trace_jsonl, prompt_corpus="coding")
        ).benchmark.datasets[0]
        assert ds.prompts is not None
        assert ds.prompts.corpus == "coding"

    def test_routes_onto_public_weka_hf(self) -> None:
        out = build_dataset(_public_cli(prompt_corpus="coding"))
        assert out["type"] == "public"
        assert out.get("prompts", {}).get("corpus") == "coding"
        ds = convert_cli_to_aiperf(
            _public_cli(prompt_corpus="coding")
        ).benchmark.datasets[0]
        assert ds.prompts is not None
        assert ds.prompts.corpus == "coding"

    def test_routes_onto_synthetic_prompts_corpus(self) -> None:
        cli = CLIConfig(
            model_names=["test-model"],
            endpoint_type="chat",
            prompt_corpus="coding",
            prompt_input_tokens_mean=16,
        )
        out = build_dataset(cli)
        assert out["type"] == "synthetic"
        assert out.get("prompts", {}).get("corpus") == "coding"


class TestCacheBustRouting:
    def test_routes_onto_file_trace(self, trace_jsonl: Path) -> None:
        out = build_dataset(_file_cli(trace_jsonl, cache_bust="first_turn_prefix"))
        assert out.get("cache_bust") == {"target": "first_turn_prefix"}
        ds = convert_cli_to_aiperf(
            _file_cli(trace_jsonl, cache_bust="first_turn_prefix")
        ).benchmark.datasets[0]
        assert str(ds.cache_bust.target) == "first_turn_prefix"

    def test_routes_onto_public_weka_hf(self) -> None:
        out = build_dataset(_public_cli(cache_bust="first_turn_prefix"))
        assert out.get("cache_bust") == {"target": "first_turn_prefix"}
        ds = convert_cli_to_aiperf(
            _public_cli(cache_bust="first_turn_prefix")
        ).benchmark.datasets[0]
        assert str(ds.cache_bust.target) == "first_turn_prefix"


class TestInterTurnDelayCapRouting:
    @pytest.mark.parametrize(
        "cli_factory_id",
        [param("file", id="file"), param("public", id="public_weka_hf")],
    )
    def test_routes_onto_trace_datasets(
        self, trace_jsonl: Path, cli_factory_id: str
    ) -> None:
        def _cli() -> CLIConfig:
            return (
                _file_cli(trace_jsonl, inter_turn_delay_cap_seconds=60.0)
                if cli_factory_id == "file"
                else _public_cli(inter_turn_delay_cap_seconds=60.0)
            )

        out = build_dataset(_cli())
        assert out.get("inter_turn_delay_cap_seconds") == 60.0
        ds = convert_cli_to_aiperf(_cli()).benchmark.datasets[0]
        assert ds.inter_turn_delay_cap_seconds == 60.0


class TestTraceDelayFlagRouting:
    """The trace-delay flags route onto both file and public datasets rather than being dropped by build_dataset."""

    @pytest.mark.parametrize(
        "cli_factory_id",
        [param("file", id="file"), param("public", id="public_weka_hf")],
    )
    def test_ignore_trace_delays_routes(
        self, trace_jsonl: Path, cli_factory_id: str
    ) -> None:
        def _cli() -> CLIConfig:
            return (
                _file_cli(trace_jsonl, ignore_trace_delays=True)
                if cli_factory_id == "file"
                else _public_cli(ignore_trace_delays=True)
            )

        out = build_dataset(_cli())
        assert out.get("ignore_trace_delays") is True
        ds = convert_cli_to_aiperf(_cli()).benchmark.datasets[0]
        assert ds.ignore_trace_delays is True

    @pytest.mark.parametrize(
        "cli_factory_id",
        [param("file", id="file"), param("public", id="public_weka_hf")],
    )
    def test_use_think_time_only_routes(
        self, trace_jsonl: Path, cli_factory_id: str
    ) -> None:
        def _cli() -> CLIConfig:
            return (
                _file_cli(trace_jsonl, use_think_time_only=True)
                if cli_factory_id == "file"
                else _public_cli(use_think_time_only=True)
            )

        out = build_dataset(_cli())
        assert out.get("use_think_time_only") is True
        ds = convert_cli_to_aiperf(_cli()).benchmark.datasets[0]
        assert ds.use_think_time_only is True

    @pytest.mark.parametrize(
        "cli_factory_id",
        [param("file", id="file"), param("public", id="public_weka_hf")],
    )
    def test_use_end_to_start_delays_routes(
        self, trace_jsonl: Path, cli_factory_id: str
    ) -> None:
        def _cli() -> CLIConfig:
            return (
                _file_cli(trace_jsonl, use_end_to_start_delays=True)
                if cli_factory_id == "file"
                else _public_cli(use_end_to_start_delays=True)
            )

        out = build_dataset(_cli())
        assert out.get("use_end_to_start_delays") is True
        ds = convert_cli_to_aiperf(_cli()).benchmark.datasets[0]
        assert ds.use_end_to_start_delays is True

    @pytest.mark.parametrize(
        "cli_factory_id",
        [param("file", id="file"), param("public", id="public_weka_hf")],
    )
    def test_trace_idle_gap_cap_routes(
        self, trace_jsonl: Path, cli_factory_id: str
    ) -> None:
        def _cli() -> CLIConfig:
            return (
                _file_cli(trace_jsonl, trace_idle_gap_cap_seconds=10.0)
                if cli_factory_id == "file"
                else _public_cli(trace_idle_gap_cap_seconds=10.0)
            )

        out = build_dataset(_cli())
        assert out.get("trace_idle_gap_cap_seconds") == 10.0
        ds = convert_cli_to_aiperf(_cli()).benchmark.datasets[0]
        assert ds.trace_idle_gap_cap_seconds == 10.0


class TestSynthesisCapRouting:
    """--max-isl/--max-osl route into a synthesis sub-config on both file and public (weka_hf) datasets."""

    def test_routes_onto_file_trace(self, trace_jsonl: Path) -> None:
        ds = convert_cli_to_aiperf(
            _file_cli(trace_jsonl, synthesis_max_isl=4096, synthesis_max_osl=512)
        ).benchmark.datasets[0]
        assert ds.synthesis is not None
        assert ds.synthesis.max_isl == 4096
        assert ds.synthesis.max_osl == 512

    def test_routes_onto_public_weka_hf(self) -> None:
        ds = convert_cli_to_aiperf(
            _public_cli(synthesis_max_isl=4096, synthesis_max_osl=512)
        ).benchmark.datasets[0]
        assert ds.synthesis is not None
        assert ds.synthesis.max_isl == 4096
        assert ds.synthesis.max_osl == 512


class TestOslFallbackRouting:
    """The --osl per-record fallback routes onto the flat osl field on either file or public datasets."""

    def test_routes_onto_file_trace(self, trace_jsonl: Path) -> None:
        ds = convert_cli_to_aiperf(
            _file_cli(trace_jsonl, prompt_output_tokens_mean=128)
        ).benchmark.datasets[0]
        assert ds.osl is not None
        assert ds.osl.expected_value == 128

    def test_routes_onto_public_weka_hf(self) -> None:
        ds = convert_cli_to_aiperf(
            _public_cli(prompt_output_tokens_mean=128)
        ).benchmark.datasets[0]
        assert ds.osl is not None
        assert ds.osl.expected_value == 128


class TestWekaHfFailFast:
    """weka_hf <-> hf_weka_dataset consistency is validated at config-load time so a missing/mismatched repo fails fast."""

    def test_weka_hf_without_repo_raises(self):
        from aiperf.config.dataset.config import PublicDataset

        with pytest.raises(ValueError, match="requires"):
            PublicDataset(type="public", name="m", dataset=PublicDatasetType.WEKA_HF)

    def test_weka_hf_empty_repo_raises(self):
        from aiperf.config.dataset.config import PublicDataset

        with pytest.raises(ValueError, match="non-empty"):
            PublicDataset(
                type="public",
                name="m",
                dataset=PublicDatasetType.WEKA_HF,
                hf_weka_dataset="   ",
            )

    def test_hf_weka_dataset_on_non_weka_raises(self):
        from aiperf.config.dataset.config import PublicDataset

        with pytest.raises(ValueError, match="can only be used"):
            PublicDataset(
                type="public",
                name="m",
                dataset=PublicDatasetType.SHAREGPT,
                hf_weka_dataset="example/repo",
            )

    def test_weka_hf_with_repo_strips_and_validates(self):
        from aiperf.config.dataset.config import PublicDataset

        d = PublicDataset(
            type="public",
            name="m",
            dataset=PublicDatasetType.WEKA_HF,
            hf_weka_dataset="  semianalysisai/cc-traces-weka-061526  ",
        )
        assert d.hf_weka_dataset == "semianalysisai/cc-traces-weka-061526"

    def test_pinned_weka_alias_needs_no_repo(self):
        from aiperf.config.dataset.config import PublicDataset

        d = PublicDataset(type="public", name="m", dataset=_WEKA_HF)
        assert d.hf_weka_dataset is None  # registry-defined repo, no flag needed


class TestHfWekaDatasetConverterRouting:
    """``--hf-weka-dataset`` is copied by build_dataset and, when set alone, auto-selects --public-dataset weka_hf."""

    def test_weka_hf_with_repo_routes_hf_weka_dataset(self) -> None:
        out = build_dataset(
            CLIConfig(
                model_names=["m"],
                public_dataset=PublicDatasetType.WEKA_HF,
                hf_weka_dataset="semianalysisai/cc-traces-weka-061526",
            )
        )
        assert out["type"] == "public"
        assert out["dataset"] == PublicDatasetType.WEKA_HF
        assert out["hf_weka_dataset"] == "semianalysisai/cc-traces-weka-061526"

    def test_hf_weka_dataset_alone_auto_selects_weka_hf(self) -> None:
        out = build_dataset(
            CLIConfig(
                model_names=["m"],
                hf_weka_dataset="semianalysisai/cc-traces-weka-061526",
            )
        )
        assert out["type"] == "public"
        assert out["dataset"] == PublicDatasetType.WEKA_HF
        assert out["hf_weka_dataset"] == "semianalysisai/cc-traces-weka-061526"


class TestTraceDelayExclusivity:
    """All three trace-delay flags set Turn.delay differently, so at most one
    may be active. Regression for PR #1165 (lkomali): the prior validator only
    rejected the ignore+think_time pair.
    """

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"ignore_trace_delays": True, "use_think_time_only": True},
            {"ignore_trace_delays": True, "use_end_to_start_delays": True},
            {"use_think_time_only": True, "use_end_to_start_delays": True},
        ],
    )
    def test_file_rejects_conflicting_delay_flags(self, kwargs) -> None:
        from aiperf.config.dataset.config import FileDataset

        # The delay-exclusivity validator runs before the source (path/records)
        # validator, so a source is not needed to exercise it.
        with pytest.raises(ValueError, match="mutually exclusive"):
            FileDataset(type="file", name="m", **kwargs)
