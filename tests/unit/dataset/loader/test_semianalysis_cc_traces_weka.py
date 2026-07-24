# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for ``SemiAnalysisCCTracesWekaLoader``."""

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aiperf.common.enums import PromptCorpus
from aiperf.common.exceptions import DatasetLoaderError
from aiperf.dataset.loader.semianalysis_cc_traces_weka import (
    SemiAnalysisCCTracesWekaLoader,
)
from aiperf.dataset.loader.weka_trace import WekaTraceLoader
from aiperf.dataset.loader.weka_trace_models import WekaTrace
from aiperf.plugin import plugins
from aiperf.plugin.enums import (
    DatasetSamplingStrategy,
    PluginType,
    PublicDatasetType,
)

# Fixtures and helpers


_NO_SUBAGENTS_HF_DATASET_NAME = "semianalysisai/cc-traces-weka-no-subagents-051826"


@pytest.fixture
def user_config():
    """A real v2 BenchmarkRun (named ``user_config`` for minimal churn)."""
    from tests.unit.dataset.loader.conftest import make_weka_run

    return make_weka_run(model_names=["test-model"])


def _make_trace_dict(
    trace_id: str = "trace-1", *, with_request: bool = True
) -> dict[str, Any]:
    """Smallest valid WekaTrace row dict (matches existing model tests)."""
    requests: list[dict[str, Any]] = []
    if with_request:
        requests.append({"t": 0.0, "type": "n", "model": "m", "in": 10, "out": 1})
    return {
        "id": trace_id,
        "models": ["m"],
        "block_size": 64,
        "hash_id_scope": "local",
        "requests": requests,
    }


@pytest.fixture
async def loader(user_config) -> SemiAnalysisCCTracesWekaLoader:
    pg = MagicMock()
    return SemiAnalysisCCTracesWekaLoader(
        run=user_config,
        hf_dataset_name=_NO_SUBAGENTS_HF_DATASET_NAME,
        hf_split="train",
        prompt_generator=pg,
        default_block_size=64,
    )


# Constructor wiring


@pytest.mark.asyncio
class TestConstructorWiring:
    """The HF loader must construct a delegated WekaTraceLoader correctly."""

    async def test_constructs_inner_weka_loader_with_no_filename(
        self, loader: SemiAnalysisCCTracesWekaLoader
    ) -> None:
        assert isinstance(loader._weka, WekaTraceLoader)
        assert loader._weka.filename is None
        assert loader._weka._path is None

    async def test_propagates_prompt_generator_to_inner_loader(
        self, user_config
    ) -> None:
        pg = MagicMock()
        loader = SemiAnalysisCCTracesWekaLoader(
            run=user_config,
            hf_dataset_name=_NO_SUBAGENTS_HF_DATASET_NAME,
            prompt_generator=pg,
            default_block_size=64,
        )
        assert loader._weka.prompt_generator is pg

    async def test_propagates_default_block_size(self, user_config) -> None:
        loader = SemiAnalysisCCTracesWekaLoader(
            run=user_config,
            hf_dataset_name=_NO_SUBAGENTS_HF_DATASET_NAME,
            prompt_generator=MagicMock(),
            default_block_size=64,
        )
        assert loader._weka._block_size == 64

    async def test_streaming_forced_off_even_when_caller_passes_true(
        self, user_config
    ) -> None:
        loader = SemiAnalysisCCTracesWekaLoader(
            run=user_config,
            hf_dataset_name=_NO_SUBAGENTS_HF_DATASET_NAME,
            prompt_generator=MagicMock(),
            default_block_size=64,
            streaming=True,
        )
        assert loader.streaming is False

    async def test_records_hf_dataset_name(
        self, loader: SemiAnalysisCCTracesWekaLoader
    ) -> None:
        assert loader.hf_dataset_name == _NO_SUBAGENTS_HF_DATASET_NAME
        assert loader.hf_split == "train"


# Row validation: load_dataset


@pytest.mark.asyncio
class TestLoadDatasetRowValidation:
    """``load_dataset`` returns ``{trace_id: [WekaTrace]}`` after validating rows."""

    async def test_returns_validated_traces_keyed_by_id(
        self, loader: SemiAnalysisCCTracesWekaLoader
    ) -> None:
        rows = [_make_trace_dict("a"), _make_trace_dict("b")]
        with patch(
            "aiperf.dataset.loader.semianalysis_cc_traces_weka.BaseHFDatasetLoader.load_dataset",
            new=AsyncMock(return_value={"dataset": rows}),
        ):
            result = await loader.load_dataset()

        assert set(result.keys()) == {"a", "b"}
        for trace_id, traces in result.items():
            assert len(traces) == 1
            assert isinstance(traces[0], WekaTrace)
            assert traces[0].id == trace_id

    async def test_empty_dataset_returns_empty_dict(
        self, loader: SemiAnalysisCCTracesWekaLoader
    ) -> None:
        with patch(
            "aiperf.dataset.loader.semianalysis_cc_traces_weka.BaseHFDatasetLoader.load_dataset",
            new=AsyncMock(return_value={"dataset": []}),
        ):
            result = await loader.load_dataset()
        assert result == {}

    async def test_invalid_row_raises_dataset_loader_error_with_index(
        self, loader: SemiAnalysisCCTracesWekaLoader
    ) -> None:
        bad_row = {"id": "x"}  # missing required fields
        rows = [_make_trace_dict("good"), bad_row]
        with (
            patch(
                "aiperf.dataset.loader.semianalysis_cc_traces_weka.BaseHFDatasetLoader.load_dataset",
                new=AsyncMock(return_value={"dataset": rows}),
            ),
            pytest.raises(DatasetLoaderError, match="failed WekaTrace validation"),
        ):
            await loader.load_dataset()

    async def test_invalid_row_message_includes_row_index(
        self, loader: SemiAnalysisCCTracesWekaLoader
    ) -> None:
        rows = [_make_trace_dict("good"), {"id": "x"}]
        with (
            patch(
                "aiperf.dataset.loader.semianalysis_cc_traces_weka.BaseHFDatasetLoader.load_dataset",
                new=AsyncMock(return_value={"dataset": rows}),
            ),
            pytest.raises(DatasetLoaderError) as exc_info,
        ):
            await loader.load_dataset()
        # Bad row is at index 1.
        assert "Row 1" in str(exc_info.value)

    async def test_duplicate_trace_id_raises_dataset_loader_error(
        self, loader: SemiAnalysisCCTracesWekaLoader
    ) -> None:
        rows = [_make_trace_dict("dup"), _make_trace_dict("dup")]
        with (
            patch(
                "aiperf.dataset.loader.semianalysis_cc_traces_weka.BaseHFDatasetLoader.load_dataset",
                new=AsyncMock(return_value={"dataset": rows}),
            ),
            pytest.raises(DatasetLoaderError, match="Duplicate trace id"),
        ):
            await loader.load_dataset()


# Delegation to WekaTraceLoader


@pytest.mark.asyncio
class TestConvertToConversationsDelegation:
    """``convert_to_conversations`` delegates to the inner WekaTraceLoader so file- and HF-based replay share backing code."""

    async def test_delegates_to_inner_weka_convert(
        self, loader: SemiAnalysisCCTracesWekaLoader
    ) -> None:
        sentinel = [object()]
        loader._weka.convert_to_conversations = MagicMock(return_value=sentinel)

        data = {"trace-1": [WekaTrace.model_validate(_make_trace_dict("trace-1"))]}
        result = await loader.convert_to_conversations(data)

        assert result is sentinel
        loader._weka.convert_to_conversations.assert_called_once_with(data)


# Sampling strategy


class TestSamplingStrategy:
    def test_preferred_sampling_strategy_is_sequential(self) -> None:
        assert (
            SemiAnalysisCCTracesWekaLoader.get_preferred_sampling_strategy()
            == DatasetSamplingStrategy.SEQUENTIAL
        )


# Plugin registry integration


class TestPluginRegistry:
    def test_class_registered_under_public_dataset_loader(self) -> None:
        cls = plugins.get_class(
            PluginType.PUBLIC_DATASET_LOADER,
            PublicDatasetType.SEMIANALYSIS_CC_TRACES_WEKA,
        )
        assert cls is SemiAnalysisCCTracesWekaLoader

    def test_metadata_marks_loader_as_trace(self) -> None:
        meta = plugins.get_public_dataset_loader_metadata(
            PublicDatasetType.SEMIANALYSIS_CC_TRACES_WEKA
        )
        assert meta.is_trace is True

    def test_metadata_carries_default_block_size(self) -> None:
        meta = plugins.get_public_dataset_loader_metadata(
            PublicDatasetType.SEMIANALYSIS_CC_TRACES_WEKA
        )
        assert meta.default_block_size == 64

    def test_metadata_default_prompt_corpus_is_coding(self) -> None:
        meta = plugins.get_public_dataset_loader_metadata(
            PublicDatasetType.SEMIANALYSIS_CC_TRACES_WEKA
        )
        assert meta.default_prompt_corpus == PromptCorpus.CODING

    def test_metadata_weka_hf_is_trace(self) -> None:
        meta = plugins.get_public_dataset_loader_metadata(PublicDatasetType.WEKA_HF)
        assert meta.is_trace is True

    def test_metadata_weka_hf_has_no_pinned_dataset_name(self) -> None:
        meta = plugins.get_public_dataset_loader_metadata(PublicDatasetType.WEKA_HF)
        assert meta.hf_dataset_name is None

    def test_metadata_weka_hf_carries_default_block_size(self) -> None:
        meta = plugins.get_public_dataset_loader_metadata(PublicDatasetType.WEKA_HF)
        assert meta.default_block_size == 64

    def test_metadata_weka_hf_default_prompt_corpus_is_coding(self) -> None:
        meta = plugins.get_public_dataset_loader_metadata(PublicDatasetType.WEKA_HF)
        assert meta.default_prompt_corpus == PromptCorpus.CODING

    def test_metadata_hf_dataset_name_pinned(self) -> None:
        meta = plugins.get_public_dataset_loader_metadata(
            PublicDatasetType.SEMIANALYSIS_CC_TRACES_WEKA_NO_SUBAGENTS
        )
        assert meta.hf_dataset_name == _NO_SUBAGENTS_HF_DATASET_NAME

    def test_metadata_default_weka_alias_hf_dataset_name_pinned(self) -> None:
        meta = plugins.get_public_dataset_loader_metadata(
            PublicDatasetType.SEMIANALYSIS_CC_TRACES_WEKA
        )
        assert meta.hf_dataset_name == _NO_SUBAGENTS_HF_DATASET_NAME
