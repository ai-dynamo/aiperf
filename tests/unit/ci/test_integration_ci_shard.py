# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The integration-suite CI shards must partition the unsharded selection.

``tests/integration/conftest.py`` keeps one round-robin shard per CI job
(``AIPERF_SHARD_INDEX`` / ``AIPERF_SHARD_TOTAL``). If the shards ever overlap
or miss an item, tests silently run twice or never.
"""

from __future__ import annotations

import pytest
from pytest import param

from tests.harness.ci_shard import ci_shard_from_env, select_ci_shard


@pytest.mark.parametrize(
    "n_items,total",
    [
        param(0, 4, id="empty"),
        param(3, 4, id="fewer-items-than-shards"),
        param(352, 4, id="ci-size"),
        param(353, 3, id="uneven"),
    ],
)  # fmt: skip
def test_select_ci_shard_shards_partition_items(n_items: int, total: int) -> None:
    items = [f"test_{i}" for i in range(n_items)]
    shards = [select_ci_shard(items, index, total)[0] for index in range(total)]

    seen = [item for shard in shards for item in shard]
    assert sorted(seen) == sorted(items)
    assert len(seen) == len(set(seen))
    assert max(map(len, shards)) - min(map(len, shards)) <= 1


def test_select_ci_shard_keep_and_drop_split_items() -> None:
    items = list(range(10))
    keep, drop = select_ci_shard(items, 1, 4)
    assert keep == [1, 5, 9]
    assert sorted(keep + drop) == items


@pytest.mark.parametrize(
    "environ,expected",
    [
        param({}, (0, 1), id="unset"),
        param({"AIPERF_SHARD_TOTAL": "1", "AIPERF_SHARD_INDEX": "3"}, (0, 1), id="total-1"),
        param({"AIPERF_SHARD_TOTAL": "4", "AIPERF_SHARD_INDEX": "3"}, (3, 4), id="last-shard"),
        param({"AIPERF_SHARD_TOTAL": "4", "AIPERF_SHARD_INDEX": ""}, (0, 4), id="empty-index"),
    ],
)  # fmt: skip
def test_ci_shard_from_env_valid_values_returns_index_and_total(
    environ: dict[str, str], expected: tuple[int, int]
) -> None:
    assert ci_shard_from_env(environ) == expected


@pytest.mark.parametrize(
    "environ,message",
    [
        param({"AIPERF_SHARD_TOTAL": "4", "AIPERF_SHARD_INDEX": "4"}, r"must be in \[0, 4\)", id="index-too-high"),
        param({"AIPERF_SHARD_TOTAL": "4", "AIPERF_SHARD_INDEX": "-1"}, r"must be in \[0, 4\)", id="negative"),
        param({"AIPERF_SHARD_TOTAL": "0"}, r"must be >= 1", id="zero-total"),
        param({"AIPERF_SHARD_TOTAL": "-2"}, r"must be >= 1", id="negative-total"),
        param({"AIPERF_SHARD_TOTAL": "four"}, "must be integers", id="non-integer"),
    ],
)  # fmt: skip
def test_ci_shard_from_env_invalid_values_raises_usage_error(
    environ: dict[str, str], message: str
) -> None:
    with pytest.raises(pytest.UsageError, match=message):
        ci_shard_from_env(environ)
