# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regression coverage: agentx trace-replay flags must route NATIVELY onto both
FILE (mooncake/weka_trace) and PUBLIC (weka_hf) datasets through the v2
converter -- not be silently dropped by the v2 config cutover.

``--prompt-corpus`` and ``--cache-bust`` are written into the ``prompts``
subtable by ``_build_prompts`` for synthetic datasets, but ``_apply_dataset_type``
strips that whole subtable for FILE/PUBLIC. Without ``_apply_corpus_and_cache_bust``
(which routes them to the flat top-level fields after the strip) they no-op on
trace replay -- the corpus reconstruction falls back to the loader default and
KV-cache-bust experiments do nothing. ``--inter-turn-delay-cap-seconds`` was
FILE-only and dropped for weka_hf. These are the gaps a faithful agentx->v2
port must close.
"""

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
        assert out.get("prompt_corpus") == "coding"
        ds = convert_cli_to_aiperf(
            _file_cli(trace_jsonl, prompt_corpus="coding")
        ).benchmark.datasets[0]
        assert ds.prompt_corpus == "coding"

    def test_routes_onto_public_weka_hf(self) -> None:
        out = build_dataset(_public_cli(prompt_corpus="coding"))
        assert out["type"] == "public"
        assert out.get("prompt_corpus") == "coding"
        ds = convert_cli_to_aiperf(
            _public_cli(prompt_corpus="coding")
        ).benchmark.datasets[0]
        assert ds.prompt_corpus == "coding"


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


class TestSynthesisCapRouting:
    """--max-isl/--max-osl cap weka replay; the weka loader reads
    synthesis.max_isl/max_osl. PublicDataset (weka_hf) now carries a synthesis
    sub-config so the caps apply to HF Weka replay, not just file traces."""

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
    """--osl per-record fallback routes onto the flat FileDataset/PublicDataset
    osl field; the composer's _osl_distribution reads it for either type."""

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
    """weka_hf <-> hf_weka_dataset consistency is validated at config-load time
    (mirrors v1's composer-level guard), so a config file declaring
    ``dataset: weka_hf`` with no repo fails fast with a clear error instead of
    surfacing an opaque TypeError deep in the generic Weka loader."""

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
