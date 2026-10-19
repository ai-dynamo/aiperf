# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import pytest

from aiperf.config.dataset.remote_ref import RemoteFixedScheduleCheck
from aiperf.config.flags import CLIConfig
from aiperf.config.loader.errors import ConfigurationError
from aiperf.dataset.remote.fixed_schedule import finalize_remote_fixed_schedule
from aiperf.plugin.enums import CustomDatasetType, PhaseType
from tests.unit.conftest import make_run_from_cli


def _profiling(cfg):
    return next(p for p in cfg.phases if p.name == "profiling")


def _cfg(**kwargs: object):
    return make_run_from_cli(
        CLIConfig(model_names=["m"], input_file="hf://o/r/t.jsonl", custom_dataset_type=CustomDatasetType.MOONCAKE_TRACE, **kwargs)
    ).cfg  # fmt: skip


def test_promoted_and_timestamped_stays_fixed_schedule() -> None:
    cfg = _cfg()
    finalize_remote_fixed_schedule(
        cfg, RemoteFixedScheduleCheck(promoted=True), promote=True
    )
    assert str(_profiling(cfg).type) == str(PhaseType.FIXED_SCHEDULE)


def test_promoted_without_timestamps_demotes_with_default_bound() -> None:
    cfg = _cfg()
    finalize_remote_fixed_schedule(
        cfg, RemoteFixedScheduleCheck(promoted=True), promote=False
    )
    phase = _profiling(cfg)
    assert str(phase.type) == str(PhaseType.CONCURRENCY)
    assert phase.requests == 10


def test_demotion_keeps_an_explicit_bound() -> None:
    cfg = _cfg(request_count=7)
    finalize_remote_fixed_schedule(
        cfg, RemoteFixedScheduleCheck(promoted=True), promote=False
    )
    assert _profiling(cfg).requests == 7


def test_demotion_keeps_concurrency_and_duration() -> None:
    cfg = _cfg(concurrency=4, benchmark_duration=30)
    finalize_remote_fixed_schedule(
        cfg, RemoteFixedScheduleCheck(promoted=True), promote=False
    )
    phase = _profiling(cfg)
    assert (phase.concurrency, phase.duration, phase.requests) == (4, 30, None)


def test_conflict_raises_only_when_timestamped() -> None:
    cfg = _cfg(request_rate=5)
    check = RemoteFixedScheduleCheck(promoted=False, conflict_error="boom: rate")
    finalize_remote_fixed_schedule(cfg, check, promote=False)
    with pytest.raises(ConfigurationError, match="boom: rate"):
        finalize_remote_fixed_schedule(cfg, check, promote=True)
