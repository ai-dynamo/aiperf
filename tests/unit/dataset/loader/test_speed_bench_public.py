# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from aiperf.common.exceptions import DatasetLoaderError
from aiperf.config.flags.cli_config import CLIConfig
from aiperf.config.loader.errors import ConfigurationError
from aiperf.dataset.loader.speed_bench import SpeedBenchRow
from aiperf.dataset.loader.speed_bench_public import (
    HLE_ACCESS_URL,
    SpeedBenchPublicLoader,
)
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


@pytest.fixture(autouse=True)
def _isolate_from_machine_state(tmp_path_factory):
    """Keep every test in this file independent of the developer's machine.

    Two pieces of ambient state leak into these tests otherwise, and both hide
    failures locally while breaking CI:

    * a real resolved cache under ``.cache/aiperf/datasets`` makes
      ``preflight_access`` return before it checks credentials at all;
    * a cached HuggingFace token makes the no-credentials branch unreachable.

    Tests that assert on a specific credential state patch over these.
    """
    cache_dir = tmp_path_factory.mktemp("speed-bench-cache")
    with (
        patch.object(
            SpeedBenchPublicLoader,
            "cache_path_for",
            staticmethod(lambda config: cache_dir / f"{config}.jsonl"),
        ),
        patch("huggingface_hub.get_token", return_value="hf_isolated"),
    ):
        yield


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
    """Exit status is not evidence the prompts are real; the rows are."""

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
    """The gate check must distinguish credential states.

    "No token" and "token without access" need different actions: telling an
    unauthenticated user their account lacks access sends them to accept terms
    on an account they never logged into.
    """

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
            staticmethod(lambda config: tmp_path / f"{config}.jsonl"),
        ):
            yield

    def test_missing_credentials_says_so_and_lists_where_it_looked(self):
        with (
            patch("huggingface_hub.get_token", return_value=None),
            pytest.raises(ConfigurationError) as excinfo,
        ):
            SpeedBenchPublicLoader.preflight_access(hf_subset="qualitative")

        message = str(excinfo.value)
        assert "no HuggingFace credentials were found" in message
        assert "$HF_TOKEN" in message
        assert "hf auth login" in message

    def test_authenticated_but_unauthorized_says_accept_the_terms(self):
        from huggingface_hub.errors import GatedRepoError

        with (
            patch("huggingface_hub.get_token", return_value="hf_fake"),
            patch(
                "huggingface_hub.HfApi.auth_check",
                side_effect=GatedRepoError("gated", response=Mock()),
            ),
            pytest.raises(ConfigurationError) as excinfo,
        ):
            SpeedBenchPublicLoader.preflight_access(hf_subset="qualitative")

        message = str(excinfo.value)
        assert "You are authenticated" in message
        assert HLE_ACCESS_URL in message
        # Must not send an already-logged-in user back through login.
        assert "hf auth login" not in message

    def test_rejected_credentials_are_reported_not_swallowed(self):
        """`auth_check` raises RepositoryNotFoundError for a refused 401.

        GatedRepoError subclasses RepositoryNotFoundError, not the reverse, so
        catching only the former let a bad token fall into the catch-all and
        skip the fast fail this check exists for.
        """
        from huggingface_hub.errors import RepositoryNotFoundError

        with (
            patch("huggingface_hub.get_token", return_value="hf_expired"),
            patch(
                "huggingface_hub.HfApi.auth_check",
                side_effect=RepositoryNotFoundError("401", response=Mock()),
            ),
            pytest.raises(ConfigurationError) as excinfo,
        ):
            SpeedBenchPublicLoader.preflight_access(hf_subset="qualitative")

        message = str(excinfo.value)
        assert "rejected the credentials" in message
        assert "hf auth login" in message

    def test_network_failure_is_not_reported_as_missing_access(self):
        """ "I could not tell" must never render as "you lack access"."""
        with (
            patch("huggingface_hub.get_token", return_value="hf_fake"),
            patch("huggingface_hub.HfApi.auth_check", side_effect=OSError("no route")),
        ):
            SpeedBenchPublicLoader.preflight_access(hf_subset="qualitative")

    def test_authorized_account_passes(self):
        with (
            patch("huggingface_hub.get_token", return_value="hf_fake"),
            patch("huggingface_hub.HfApi.auth_check", return_value=None),
        ):
            SpeedBenchPublicLoader.preflight_access(hf_subset="qualitative")

    def test_cached_config_skips_the_probe_entirely(self, tmp_path):
        """An already-resolved config must not need credentials at all.

        This is what makes the pre-staging workflow work on an air-gapped box.
        """
        cached = tmp_path / "qualitative.jsonl"
        cached.write_text("{}\n")

        with (
            patch.object(
                SpeedBenchPublicLoader, "cache_path_for", staticmethod(lambda c: cached)
            ),
            patch("huggingface_hub.get_token", side_effect=AssertionError("probed")),
        ):
            SpeedBenchPublicLoader.preflight_access(hf_subset="qualitative")


class TestResolveConfig:
    """resolve_config drives the vendored script and publishes the cache."""

    @staticmethod
    def _fake_hf_dataset(rows):
        """Stand in for a datasets.Dataset through the calls resolve_config makes."""

        class _FakeDataset:
            def __init__(self, rows):
                self._rows = rows

            def map(self, fn, remove_columns=None):
                return _FakeDataset([{**r, **fn(r)} for r in self._rows])

            def to_json(self, path):
                Path(path).write_text(
                    "\n".join(
                        json.dumps({k: v for k, v in r.items() if k != "turns"})
                        for r in self._rows
                    )
                )

        return _FakeDataset(rows)

    def _patched_resolve(self, monkeypatch, tmp_path, rows):
        """Point the cache at tmp_path and stub out every network call.

        Patches attributes on the real modules rather than replacing them in
        ``sys.modules``: the vendored script does ``from datasets import
        Dataset, ...`` at import time, so a stub module breaks its import
        before the code under test is reached.
        """
        import datasets as datasets_mod

        from aiperf.dataset.loader.vendor import speed_bench_prepare

        monkeypatch.setattr(
            SpeedBenchPublicLoader,
            "cache_path_for",
            staticmethod(lambda config: tmp_path / f"{config}.jsonl"),
        )
        fake = self._fake_hf_dataset(rows)
        monkeypatch.setattr(datasets_mod, "load_dataset", lambda *a, **k: fake)
        monkeypatch.setattr(
            speed_bench_prepare, "_resolve_external_data", lambda ds, config: ds
        )
        return fake

    def test_publishes_a_cache_for_a_fully_resolved_config(self, monkeypatch, tmp_path):
        rows = [{"question_id": "a" * 32, "category": "qa", "turns": ["A prompt."]}]
        self._patched_resolve(monkeypatch, tmp_path, rows)

        path = SpeedBenchPublicLoader.resolve_config("qualitative")

        assert path.exists()
        written = json.loads(path.read_text().splitlines()[0])
        assert written["messages"] == [{"role": "user", "content": "A prompt."}]

    def test_leaves_no_partial_file_when_rows_are_unresolved(
        self, monkeypatch, tmp_path
    ):
        """A rejected resolve must not leave a cache the next run would trust."""
        rows = [{"question_id": "a" * 32, "category": "qa", "turns": [PLACEHOLDER]}]
        self._patched_resolve(monkeypatch, tmp_path, rows)

        with pytest.raises(DatasetLoaderError, match="unresolved"):
            SpeedBenchPublicLoader.resolve_config("qualitative")

        assert list(tmp_path.iterdir()) == [], "no cache or .partial may survive"

    def test_wraps_a_resolution_failure_with_context(self, monkeypatch, tmp_path):
        self._patched_resolve(monkeypatch, tmp_path, [])
        from aiperf.dataset.loader.vendor import speed_bench_prepare

        def _boom(dataset, config):
            raise RuntimeError("source unreachable")

        monkeypatch.setattr(speed_bench_prepare, "_resolve_external_data", _boom)

        with pytest.raises(DatasetLoaderError, match="Failed to resolve"):
            SpeedBenchPublicLoader.resolve_config("qualitative")

    def test_gated_failure_is_reported_as_a_gate_problem(self, monkeypatch, tmp_path):
        """A 403 surfacing during resolve must still name the browser step."""
        self._patched_resolve(monkeypatch, tmp_path, [])
        from aiperf.dataset.loader.vendor import speed_bench_prepare

        def _gated(dataset, config):
            raise RuntimeError("cais/hle is a gated dataset on the Hub")

        monkeypatch.setattr(speed_bench_prepare, "_resolve_external_data", _gated)

        with pytest.raises(DatasetLoaderError) as excinfo:
            SpeedBenchPublicLoader.resolve_config("qualitative")

        assert "https://huggingface.co/datasets/cais/hle" in str(excinfo.value)


class TestPreflightMaterialize:
    def test_resolves_when_the_cache_is_cold(self, monkeypatch, tmp_path):
        called = []
        monkeypatch.setattr(
            SpeedBenchPublicLoader,
            "cache_path_for",
            staticmethod(lambda config: tmp_path / f"{config}.jsonl"),
        )
        monkeypatch.setattr(
            SpeedBenchPublicLoader,
            "resolve_config",
            classmethod(lambda cls, config: called.append(config)),
        )

        SpeedBenchPublicLoader.preflight_materialize(hf_subset="qualitative")

        assert called == ["qualitative"]

    def test_surfaces_resolution_failure_as_a_configuration_error(
        self, monkeypatch, tmp_path
    ):
        """Preflight failures must render as a clean panel, not a traceback."""

        def _fail(cls, config):
            raise DatasetLoaderError("resolution blew up")

        monkeypatch.setattr(
            SpeedBenchPublicLoader,
            "cache_path_for",
            staticmethod(lambda config: tmp_path / f"{config}.jsonl"),
        )
        monkeypatch.setattr(
            SpeedBenchPublicLoader, "resolve_config", classmethod(_fail)
        )

        with pytest.raises(ConfigurationError, match="resolution blew up"):
            SpeedBenchPublicLoader.preflight_materialize(hf_subset="qualitative")


class TestConvertEdgeCases:
    async def test_rows_with_only_blank_content_are_skipped(self):
        data = {
            "dataset": [
                _row("a" * 32, "qa", "   "),
                _row("b" * 32, "qa", "Real."),
            ]
        }

        conversations = await _loader().convert_to_conversations(data)

        assert len(conversations) == 1


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
    """Preflight runs synchronously, so it must not construct a loader.

    Unit tests run under pytest-asyncio and always have a loop, so this class
    deliberately exercises the no-loop path they would otherwise hide.
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
        with (
            patch("huggingface_hub.get_token", return_value="hf_fake"),
            patch("huggingface_hub.HfApi.auth_check", return_value=None),
        ):
            SpeedBenchPublicLoader.preflight_access(hf_subset="qualitative")

    def test_cache_path_is_resolvable_without_an_instance(self):
        path = SpeedBenchPublicLoader.cache_path_for("throughput_1k")

        assert path.name == "throughput_1k.jsonl"


def test_public_loaders_construct_outside_an_event_loop():
    """Public loaders must be constructible with no event loop running."""
    loader = SpeedBenchPublicLoader(hf_subset="qualitative")

    assert loader.config == "qualitative"


class TestDeterminism:
    """The composed dataset must not vary between runs.

    Benchmark results are only comparable if the prompts are identical, so
    conversion is order-preserving and does not sample: the same cached config
    yields the same conversations regardless of seed. Resolution itself is
    likewise deterministic -- it is byte-identical to upstream's prepare step.
    """

    @staticmethod
    def _data():
        return {"dataset": [_row(f"{i:032d}", "qa", f"Prompt {i}.") for i in range(25)]}

    async def test_repeated_conversion_yields_identical_conversations(self):
        first = await _loader().convert_to_conversations(self._data())
        second = await _loader().convert_to_conversations(self._data())

        assert [c.session_id for c in first] == [c.session_id for c in second]
        assert [c.turns[0].texts[0].contents for c in first] == [
            c.turns[0].texts[0].contents for c in second
        ]

    async def test_conversion_preserves_source_order(self):
        """Order-preserving, so a truncated run is a prefix rather than a sample."""
        conversations = await _loader().convert_to_conversations(self._data())

        assert [c.session_id for c in conversations] == [f"{i:032d}" for i in range(25)]

    def test_sampling_is_sequential_not_random(self):
        assert (
            SpeedBenchPublicLoader.get_preferred_sampling_strategy()
            == DatasetSamplingStrategy.SEQUENTIAL
        )
