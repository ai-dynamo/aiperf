# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Dataset preflight: enumeration and hook dispatch.

These run in the CLI process before service bootstrap, so they must work with
no event loop and must not construct loaders (whose base class opens an
aiohttp client).
"""

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from aiperf.cli_runner._preflight import (
    _preflight_dataset_access,
    _preflight_dataset_materialize,
    _public_dataset_loaders,
)
from aiperf.config.dataset import FileDataset, PublicDataset


def _plan(dataset):
    """Minimal stand-in for BenchmarkPlan; only get_default_dataset is read."""
    return SimpleNamespace(
        configs=[SimpleNamespace(get_default_dataset=lambda: dataset)]
    )


def _public(name: str, hf_subset: str | None = None):
    return PublicDataset(name="main", type="public", dataset=name, hf_subset=hf_subset)


class TestPublicDatasetEnumeration:
    def test_yields_loader_and_kwargs_for_a_public_dataset(self):
        plan = _plan(_public("speed_bench_qa"))

        found = list(_public_dataset_loaders(plan))

        assert len(found) == 1
        loader_class, kwargs = found[0]
        assert loader_class.__name__ == "SpeedBenchPublicLoader"
        assert kwargs == {"hf_subset": "qualitative"}

    def test_only_hook_relevant_kwargs_are_forwarded(self):
        """category does not affect access or materialization, so it is not sent."""
        plan = _plan(_public("speed_bench_qualitative"))

        _, kwargs = next(iter(_public_dataset_loaders(plan)))

        assert kwargs == {"hf_subset": "qualitative"}

    def test_selectors_sharing_a_config_are_probed_once(self):
        """A category sweep names many selectors backed by one config.

        Keying dedup on the selector made preflight issue one auth probe per
        category before the run started -- 11 identical round trips for the
        qualitative sweep, and a rate-limit risk on the preflight path.
        """
        plan = SimpleNamespace(
            configs=[
                SimpleNamespace(get_default_dataset=lambda d=d: d)
                for d in (
                    _public("speed_bench_qa"),
                    _public("speed_bench_coding"),
                    _public("speed_bench_math"),
                )
            ]
        )

        assert len(list(_public_dataset_loaders(plan))) == 1

    def test_different_configs_are_probed_separately(self):
        plan = SimpleNamespace(
            configs=[
                SimpleNamespace(get_default_dataset=lambda d=d: d)
                for d in (
                    _public("speed_bench_qa"),
                    _public("speed_bench_throughput_1k"),
                )
            ]
        )

        assert len(list(_public_dataset_loaders(plan))) == 2

    def test_file_datasets_are_skipped(self, tmp_path):
        f = tmp_path / "d.jsonl"
        f.write_text('{"text": "hi"}\n')
        plan = _plan(FileDataset(name="main", type="file", path=f))

        assert list(_public_dataset_loaders(plan)) == []


class TestPreflightDispatch:
    def test_access_hook_is_invoked_with_the_loader_kwargs(self):
        plan = _plan(_public("speed_bench_qa"))

        with patch(
            "aiperf.dataset.loader.speed_bench_public.SpeedBenchPublicLoader"
            ".preflight_access"
        ) as hook:
            _preflight_dataset_access(plan)

        hook.assert_called_once_with(hf_subset="qualitative")

    def test_materialize_hook_is_invoked_with_the_loader_kwargs(self):
        plan = _plan(_public("speed_bench_qa"))

        with patch(
            "aiperf.dataset.loader.speed_bench_public.SpeedBenchPublicLoader"
            ".preflight_materialize"
        ) as hook:
            _preflight_dataset_materialize(plan)

        hook.assert_called_once_with(hf_subset="qualitative")

    def test_hook_failures_propagate(self):
        """A preflight must not swallow the error it exists to surface."""
        from aiperf.config.loader.errors import ConfigurationError

        plan = _plan(_public("speed_bench_qa"))

        with (
            patch(
                "aiperf.dataset.loader.speed_bench_public.SpeedBenchPublicLoader"
                ".preflight_access",
                side_effect=ConfigurationError("gated"),
            ),
            pytest.raises(ConfigurationError, match="gated"),
        ):
            _preflight_dataset_access(plan)

    def test_datasets_without_hooks_are_a_no_op(self):
        """Most public datasets inherit the base no-op hooks."""
        plan = _plan(_public("sharegpt"))

        _preflight_dataset_access(plan)
        _preflight_dataset_materialize(plan)
