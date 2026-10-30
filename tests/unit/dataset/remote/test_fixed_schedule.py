# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import pytest
from pytest import param

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


def test_legacy_check_without_carried_bound_demotes_to_flat_floor() -> None:
    """A check serialized before the bound was carried still gets a bounded run."""
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


def _stop_condition(phase) -> dict[str, object]:
    return {k: getattr(phase, k) for k in ("requests", "duration", "sessions")}


def _demoted(**kwargs: object):
    cfg = make_run_from_cli(CLIConfig(model_names=["m"], input_file="hf://o/r/t.jsonl", **kwargs)).cfg  # fmt: skip
    check = cfg.get_default_dataset().remote_fixed_schedule_check
    assert check is not None and check.promoted
    finalize_remote_fixed_schedule(cfg, check, promote=False)
    return cfg


@pytest.mark.parametrize(
    "kwargs",
    [
        param({"concurrency": 64}, id="untyped-concurrency-64"),
        param({"concurrency": 100}, id="untyped-concurrency-100"),
        param({"concurrency": 3}, id="untyped-concurrency-below-floor"),
        param({"concurrency": 64, "custom_dataset_type": CustomDatasetType.MOONCAKE_TRACE}, id="trace-concurrency-64"),
        param({"concurrency": 8, "request_count": 5}, id="explicit-request-count"),
        param({"concurrency": 8, "benchmark_duration": 30}, id="explicit-duration"),
    ],
)  # fmt: skip
def test_demoted_remote_matches_local_file_stop_condition(
    tmp_path, kwargs: dict
) -> None:
    """A remote trace without timestamps ends up exactly where the same local file does."""
    local_file = tmp_path / "t.jsonl"
    local_file.write_text('{"input_length": 10, "output_length": 2}\n')
    local = make_run_from_cli(CLIConfig(model_names=["m"], input_file=str(local_file), **kwargs)).cfg  # fmt: skip
    remote = _demoted(**kwargs)
    assert str(_profiling(remote).type) == str(_profiling(local).type)
    assert _stop_condition(_profiling(remote)) == _stop_condition(_profiling(local))


def test_demoted_untyped_remote_gets_full_pass_when_manifest_declares_suite() -> None:
    """The manifest supplies the format after the converter ran; a suite still runs one pass."""
    cfg = make_run_from_cli(CLIConfig(model_names=["m"], input_file="hf://o/r/t.jsonl", concurrency=8)).cfg  # fmt: skip
    check = cfg.get_default_dataset().remote_fixed_schedule_check
    ds = cfg.get_default_dataset()
    cfg.datasets[0] = ds.model_copy(
        update={"format": CustomDatasetType.SPEED_BENCH_QUALITATIVE}
    )
    finalize_remote_fixed_schedule(cfg, check, promote=False)
    assert _profiling(cfg).sessions == "all"


def test_demotion_logs_the_default_it_applies(caplog) -> None:
    with caplog.at_level("WARNING"):
        _demoted(concurrency=64)
    assert "defaulting to 128 requests" in caplog.text


def test_promotion_carries_no_bound_when_the_user_set_one() -> None:
    cfg = make_run_from_cli(CLIConfig(model_names=["m"], input_file="hf://o/r/t.jsonl", concurrency=8, request_count=5)).cfg  # fmt: skip
    check = cfg.get_default_dataset().remote_fixed_schedule_check
    assert check.demoted_stop_condition is None
