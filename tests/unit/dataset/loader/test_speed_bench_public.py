# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from unittest.mock import Mock, patch

import pytest

from aiperf.common.exceptions import DatasetLoaderError
from aiperf.config.flags.cli_config import CLIConfig
from aiperf.config.loader.errors import ConfigurationError
from aiperf.dataset.loader.speed_bench import SpeedBenchRow
from aiperf.dataset.loader.speed_bench_public import SpeedBenchPublicLoader
from aiperf.plugin.enums import DatasetSamplingStrategy
from tests.unit.conftest import make_run_from_cli

PLACEHOLDER = SpeedBenchRow.TURNS_PLACEHOLDER


def _make_run():
    return make_run_from_cli(CLIConfig(model_names=["test-model"]))


def _row(question_id: str, category: str, *contents: str) -> dict:
    return {
        "question_id": question_id,
        "category": category,
        "messages": [{"role": "user", "content": c} for c in contents],
    }


def _loader(category: str | None = None, multi_turn: bool = True, **kw):
    return SpeedBenchPublicLoader(
        run=_make_run(),
        hf_subset="qualitative",
        category=category,
        multi_turn=multi_turn,
        **kw,
    )


class TestSpeedBenchPublicLoader:
    def test_preferred_sampling_strategy_is_sequential(self):
        assert (
            SpeedBenchPublicLoader.get_preferred_sampling_strategy()
            == DatasetSamplingStrategy.SEQUENTIAL
        )

    async def test_category_filter_selects_only_matching_rows(self):
        data = {
            "dataset": [
                _row("a" * 32, "qa", "A question."),
                _row("b" * 32, "coding", "Implement binary search."),
                _row("c" * 32, "qa", "Another question."),
            ]
        }

        conversations = await _loader(category="qa").convert_to_conversations(data)

        assert len(conversations) == 2

    async def test_multi_turn_false_keeps_only_first_turn(self):
        data = {"dataset": [_row("a" * 32, "qa", "First.", "Second.")]}

        conversations = await _loader(multi_turn=False).convert_to_conversations(data)

        assert len(conversations[0].turns) == 1

    async def test_unknown_category_reports_the_mismatch(self):
        data = {"dataset": [_row("a" * 32, "qa", "A question.")]}

        with pytest.raises(DatasetLoaderError, match="matched none of the 1 rows"):
            await _loader(category="nope").convert_to_conversations(data)

    async def test_session_id_comes_from_question_id(self):
        data = {"dataset": [_row("a" * 32, "qa", "A question.")]}

        conversations = await _loader().convert_to_conversations(data)

        assert conversations[0].session_id == "a" * 32

    async def test_cached_config_is_reused_without_resolving(self, tmp_path):
        """A populated cache must not trigger a multi-GB refetch.

        This is also the documented pre-staging path: drop a resolved JSONL at
        the cache location and an offline machine can benchmark from it.
        """
        cached = tmp_path / "qualitative.jsonl"
        cached.write_text(json.dumps(_row("a" * 32, "qa", "Cached prompt.")) + "\n")

        with (
            patch.object(SpeedBenchPublicLoader, "cache_path_for", return_value=cached),
            patch.object(
                SpeedBenchPublicLoader,
                "resolve_config",
                side_effect=AssertionError("resolved despite a populated cache"),
            ),
        ):
            data = await _loader().load_dataset()

        assert data["dataset"][0]["messages"][0]["content"] == "Cached prompt."


class TestUnresolvedRowRejection:
    """Upstream reports success even when a source is unreachable.

    Exit status is therefore not evidence that the prompts are real, so the
    rows themselves are checked before the cache is published.
    """

    def test_rejects_and_removes_a_partially_resolved_cache(self, tmp_path):
        partial = tmp_path / "qualitative.jsonl.partial"
        partial.write_text(
            "\n".join(
                json.dumps(r)
                for r in [
                    _row("a" * 32, "qa", "Real prompt."),
                    _row("b" * 32, "stem", PLACEHOLDER),
                ]
            )
        )

        with pytest.raises(DatasetLoaderError, match=r"1 of 2 rows"):
            SpeedBenchPublicLoader._reject_unresolved("qualitative", partial)

        assert not partial.exists(), "a rejected cache must not be left behind"

    def test_accepts_a_fully_resolved_cache(self, tmp_path):
        partial = tmp_path / "qualitative.jsonl.partial"
        partial.write_text(json.dumps(_row("a" * 32, "qa", "Real prompt.")))

        SpeedBenchPublicLoader._reject_unresolved("qualitative", partial)

        assert partial.exists()

    def test_prefixed_placeholder_counts_as_unresolved(self, tmp_path):
        partial = tmp_path / "qualitative.jsonl.partial"
        partial.write_text(json.dumps(_row("a" * 32, "qa", f"{PLACEHOLDER}\n\nx")))

        with pytest.raises(DatasetLoaderError, match=r"1 of 1 rows"):
            SpeedBenchPublicLoader._reject_unresolved("qualitative", partial)


class TestGatePreflight:
    @pytest.fixture(autouse=True)
    def _uncached(self, tmp_path):
        """Point the cache at an empty dir.

        preflight_access short-circuits when the config is already cached, so
        without this the test passes or fails depending on whether the machine
        happens to have resolved SPEED-Bench.
        """
        with patch.object(
            SpeedBenchPublicLoader,
            "cache_path_for",
            return_value=tmp_path / "qualitative.jsonl",
        ):
            yield

    def test_gated_source_raises_actionable_configuration_error(self):
        from huggingface_hub.errors import GatedRepoError

        with (
            patch(
                "huggingface_hub.HfApi.auth_check",
                side_effect=GatedRepoError("gated", response=Mock()),
            ),
            pytest.raises(ConfigurationError) as excinfo,
        ):
            SpeedBenchPublicLoader.preflight_access(hf_subset="qualitative")

        message = str(excinfo.value)
        assert "cais/hle" in message
        assert "https://huggingface.co/datasets/cais/hle" in message
        assert "hf auth login" in message

    def test_network_failure_is_not_reported_as_missing_access(self):
        """ "I could not tell" must never render as "you lack access"."""
        with patch("huggingface_hub.HfApi.auth_check", side_effect=OSError("no route")):
            SpeedBenchPublicLoader.preflight_access(hf_subset="qualitative")

    def test_authorized_account_passes(self):
        with patch("huggingface_hub.HfApi.auth_check", return_value=None):
            SpeedBenchPublicLoader.preflight_access(hf_subset="qualitative")


class TestMovedNameMigration:
    """The 26 category selectors moved to --public-dataset.

    They were the only documented way to run those subsets, so a bare enum
    error would strand anyone with an existing script or tutorial command.
    """

    @pytest.mark.parametrize(
        "name",
        [
            "speed_bench_coding",
            "speed_bench_summarization",
            "speed_bench_throughput_1k_mixed",
            "speed_bench_throughput_32k_high_entropy",
        ],
    )
    def test_moved_name_points_at_the_new_flag(self, name):
        from pydantic import ValidationError

        with pytest.raises(ValidationError) as excinfo:
            CLIConfig(model_names=["m"], custom_dataset_type=name)

        message = str(excinfo.value)
        assert "--public-dataset" in message
        assert f"--public-dataset {name}" in message

    @pytest.mark.parametrize(
        "name", ["speed_bench_qualitative", "speed_bench_throughput_1k"]
    )
    def test_surviving_names_still_accepted(self, name):
        """The 6 base entries remain: they let AIPerf read a prepared file."""
        cfg = CLIConfig(model_names=["m"], custom_dataset_type=name)

        assert cfg.custom_dataset_type == name


class TestPreflightRunsWithoutAnEventLoop:
    """Preflight is synchronous; the loader's base class is not.

    ``BasePublicDatasetLoader.__init__`` opens an aiohttp client, which needs a
    running event loop. Constructing a loader during preflight therefore raises
    ``RuntimeError: no running event loop`` for every user, which unit tests
    hide because pytest-asyncio supplies a loop.
    """

    def test_materialize_does_not_construct_a_loader(self, tmp_path):
        cached = tmp_path / "qualitative.jsonl"
        cached.write_text(json.dumps(_row("a" * 32, "qa", "Cached.")) + "\n")

        with patch.object(
            SpeedBenchPublicLoader, "cache_path_for", return_value=cached
        ):
            # No event loop is running here, deliberately.
            SpeedBenchPublicLoader.preflight_materialize(hf_subset="qualitative")

    def test_access_check_does_not_construct_a_loader(self):
        with patch("huggingface_hub.HfApi.auth_check", return_value=None):
            SpeedBenchPublicLoader.preflight_access(hf_subset="qualitative")

    def test_cache_path_is_resolvable_without_an_instance(self):
        path = SpeedBenchPublicLoader.cache_path_for("throughput_1k")

        assert path.name == "throughput_1k.jsonl"


def test_public_loaders_construct_outside_an_event_loop():
    """Public loaders must be constructible synchronously.

    ``BasePublicDatasetLoader`` opened an aiohttp ``TCPConnector`` eagerly, so
    merely instantiating any public loader outside an event loop raised
    ``RuntimeError: no running event loop`` -- which broke the preflight phase
    for every user while unit tests, which run under pytest-asyncio, passed.
    """
    loader = SpeedBenchPublicLoader(hf_subset="qualitative")

    assert loader.config == "qualitative"
