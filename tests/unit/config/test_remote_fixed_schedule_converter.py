# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import pytest

from aiperf.config.flags import CLIConfig
from aiperf.plugin.enums import CustomDatasetType, PhaseType
from tests.unit.conftest import make_run_from_cli

REMOTE = "hf://org/repo/trace.jsonl"


def _run(**kwargs: object):
    return make_run_from_cli(CLIConfig(model_names=["m"], input_file=REMOTE, **kwargs))


def _profiling(run):
    return next(p for p in run.cfg.phases if p.name == "profiling")


def test_trace_type_is_optimistically_promoted() -> None:
    run = _run(custom_dataset_type=CustomDatasetType.MOONCAKE_TRACE)
    assert str(_profiling(run).type) == str(PhaseType.FIXED_SCHEDULE)
    check = run.cfg.get_default_dataset().remote_fixed_schedule_check
    assert check is not None and check.promoted and check.conflict_error is None


def test_untyped_remote_is_promoted_pending_manifest() -> None:
    run = _run()
    assert run.cfg.get_default_dataset().remote_fixed_schedule_check.promoted


def test_rate_flags_record_conflict_without_promoting() -> None:
    run = _run(custom_dataset_type=CustomDatasetType.MOONCAKE_TRACE, request_rate=5)
    check = run.cfg.get_default_dataset().remote_fixed_schedule_check
    assert str(_profiling(run).type) != str(PhaseType.FIXED_SCHEDULE)
    assert check is not None and not check.promoted and "rate" in check.conflict_error


def test_sweep_records_conflict_without_promoting() -> None:
    run = _run(custom_dataset_type=CustomDatasetType.MOONCAKE_TRACE, concurrency=[1, 2])
    check = run.cfg.get_default_dataset().remote_fixed_schedule_check
    assert str(_profiling(run).type) == str(PhaseType.CONCURRENCY)
    assert check is not None and not check.promoted
    assert "Parameter sweeps" in check.conflict_error


@pytest.mark.parametrize(
    "kwargs",
    [
        {"custom_dataset_type": CustomDatasetType.SINGLE_TURN},
        {"custom_dataset_type": CustomDatasetType.MOONCAKE_TRACE, "disable_auto_fixed_schedule": True},
        {"custom_dataset_type": CustomDatasetType.MOONCAKE_TRACE, "fixed_schedule": True},
    ],
)  # fmt: skip
def test_no_marker_when_promotion_cannot_apply(kwargs: dict) -> None:
    assert _run(**kwargs).cfg.get_default_dataset().remote_fixed_schedule_check is None
