# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for config resolution utilities (aiperf.config.resolved)."""

from __future__ import annotations

import pytest
from pytest import param

from aiperf.common.enums import DatasetFormat, DatasetType
from aiperf.common.models.dataset_models import (
    Conversation,
    ConversationMetadata,
    Turn,
    TurnMetadata,
)
from aiperf.config.dataset import (
    ComposedDataset,
    FileDataset,
    PublicDataset,
    SyntheticDataset,
)
from aiperf.config.phases import (
    ConcurrencyPhase,
    FixedSchedulePhase,
    UserCentricPhase,
)
from aiperf.config.resolved import (
    check_phase_dataset_compatibility,
    conversations_have_timing_data,
    get_dataset_entries,
    get_dataset_format,
    get_dataset_type,
    get_phase_timing,
    get_random_seed,
    get_sampling_strategy,
    get_stop_condition,
    is_file_dataset,
    is_multi_turn_dataset,
    is_public_dataset,
    is_synthetic_dataset,
    is_trace_dataset,
    requires_multi_turn,
    requires_sequential_sampling,
)
from aiperf.plugin.enums import (
    ArrivalPattern,
    DatasetSamplingStrategy,
    PhaseType,
    TimingMode,
)

# =============================================================================
# DATASET FIXTURES
# =============================================================================


def _synthetic_dataset(**kwargs) -> SyntheticDataset:
    defaults = {"type": "synthetic", "entries": 100}
    defaults.update(kwargs)
    return SyntheticDataset(**defaults)


def _file_dataset(**kwargs) -> FileDataset:
    defaults = {"type": "file", "path": "/tmp/data.jsonl"}
    defaults.update(kwargs)
    return FileDataset(**defaults)


def _public_dataset(**kwargs) -> PublicDataset:
    defaults = {"type": "public", "name": "sharegpt"}
    defaults.update(kwargs)
    return PublicDataset(**defaults)


def _composed_dataset(**kwargs) -> ComposedDataset:
    defaults = {
        "source": {"type": "file", "path": "/tmp/src.jsonl", "format": "single_turn"},
        "augment": {"osl": 512},
    }
    defaults.update(kwargs)
    return ComposedDataset(**defaults)


# =============================================================================
# PHASE FIXTURES
# =============================================================================


def _concurrency_phase(**kwargs) -> ConcurrencyPhase:
    defaults = {"type": "concurrency", "concurrency": 8, "requests": 100}
    defaults.update(kwargs)
    return ConcurrencyPhase(**defaults)


def _fixed_schedule_phase(**kwargs) -> FixedSchedulePhase:
    defaults = {"type": "fixed_schedule"}
    defaults.update(kwargs)
    return FixedSchedulePhase(**defaults)


def _user_centric_phase(**kwargs) -> UserCentricPhase:
    defaults = {"type": "user_centric", "rate": 10.0, "users": 5, "requests": 50}
    defaults.update(kwargs)
    return UserCentricPhase(**defaults)


# =============================================================================
# DATASET PROPERTY QUERIES
# =============================================================================


class TestGetDatasetType:
    @pytest.mark.parametrize(
        "dataset, expected",
        [
            param(_synthetic_dataset(), DatasetType.SYNTHETIC, id="synthetic"),
            param(_file_dataset(), DatasetType.FILE, id="file"),
            param(_public_dataset(), DatasetType.PUBLIC, id="public"),
            param(_composed_dataset(), DatasetType.COMPOSED, id="composed"),
        ],
    )  # fmt: skip
    def test_returns_correct_type(self, dataset, expected):
        assert get_dataset_type(dataset) == expected


class TestGetSamplingStrategy:
    @pytest.mark.parametrize(
        "dataset, expected",
        [
            param(
                _synthetic_dataset(),
                DatasetSamplingStrategy.SEQUENTIAL,
                id="synthetic_default",
            ),
            param(
                _synthetic_dataset(sampling="random"),
                DatasetSamplingStrategy.RANDOM,
                id="synthetic_random",
            ),
            param(
                _file_dataset(sampling="shuffle"),
                DatasetSamplingStrategy.SHUFFLE,
                id="file_shuffle",
            ),
            param(
                _public_dataset(sampling="random"),
                DatasetSamplingStrategy.RANDOM,
                id="public_random",
            ),
        ],
    )  # fmt: skip
    def test_returns_correct_strategy(self, dataset, expected):
        assert get_sampling_strategy(dataset) == expected


class TestIsFileDataset:
    def test_file_dataset_is_true(self):
        assert is_file_dataset(_file_dataset()) is True

    def test_composed_dataset_is_true(self):
        assert is_file_dataset(_composed_dataset()) is True

    def test_synthetic_dataset_is_false(self):
        assert is_file_dataset(_synthetic_dataset()) is False

    def test_public_dataset_is_false(self):
        assert is_file_dataset(_public_dataset()) is False


class TestIsSyntheticDataset:
    def test_synthetic_is_true(self):
        assert is_synthetic_dataset(_synthetic_dataset()) is True

    def test_file_is_false(self):
        assert is_synthetic_dataset(_file_dataset()) is False


class TestIsPublicDataset:
    def test_public_is_true(self):
        assert is_public_dataset(_public_dataset()) is True

    def test_synthetic_is_false(self):
        assert is_public_dataset(_synthetic_dataset()) is False


class TestIsTraceDataset:
    def test_mooncake_trace_is_true(self):
        ds = _file_dataset(format="mooncake_trace")
        assert is_trace_dataset(ds) is True

    def test_single_turn_is_false(self):
        ds = _file_dataset(format="single_turn")
        assert is_trace_dataset(ds) is False

    def test_composed_with_mooncake_source_is_true(self):
        ds = _composed_dataset(
            source={"type": "file", "path": "/tmp/t.jsonl", "format": "mooncake_trace"}
        )
        assert is_trace_dataset(ds) is True

    def test_synthetic_is_false(self):
        assert is_trace_dataset(_synthetic_dataset()) is False


class TestIsMultiTurnDataset:
    def test_multi_turn_is_true(self):
        ds = _file_dataset(format="multi_turn")
        assert is_multi_turn_dataset(ds) is True

    def test_single_turn_is_false(self):
        ds = _file_dataset(format="single_turn")
        assert is_multi_turn_dataset(ds) is False

    def test_composed_multi_turn_is_true(self):
        ds = _composed_dataset(
            source={"type": "file", "path": "/tmp/t.jsonl", "format": "multi_turn"}
        )
        assert is_multi_turn_dataset(ds) is True

    def test_synthetic_is_false(self):
        assert is_multi_turn_dataset(_synthetic_dataset()) is False


class TestGetDatasetFormat:
    @pytest.mark.parametrize(
        "dataset, expected",
        [
            param(_file_dataset(format="single_turn"), DatasetFormat.SINGLE_TURN, id="file_single_turn"),
            param(_file_dataset(format="multi_turn"), DatasetFormat.MULTI_TURN, id="file_multi_turn"),
            param(_file_dataset(format="mooncake_trace"), DatasetFormat.MOONCAKE_TRACE, id="file_mooncake"),
            param(_synthetic_dataset(), None, id="synthetic_none"),
            param(_public_dataset(), None, id="public_none"),
        ],
    )  # fmt: skip
    def test_returns_correct_format(self, dataset, expected):
        assert get_dataset_format(dataset) == expected

    def test_composed_returns_source_format(self):
        ds = _composed_dataset(
            source={"type": "file", "path": "/tmp/t.jsonl", "format": "multi_turn"}
        )
        assert get_dataset_format(ds) == DatasetFormat.MULTI_TURN


class TestGetDatasetEntries:
    def test_synthetic_returns_entries(self):
        assert get_dataset_entries(_synthetic_dataset(entries=500)) == 500

    def test_file_returns_entries(self):
        assert get_dataset_entries(_file_dataset(entries=200)) == 200

    def test_file_default_returns_none(self):
        assert get_dataset_entries(_file_dataset()) is None


class TestGetRandomSeed:
    def test_returns_seed_when_set(self):
        assert get_random_seed(_synthetic_dataset(random_seed=42)) == 42

    def test_returns_none_when_unset(self):
        assert get_random_seed(_synthetic_dataset()) is None


# =============================================================================
# TIMING DATA DETECTION
# =============================================================================


class TestConversationsHaveTimingData:
    def test_empty_list_returns_false(self):
        assert conversations_have_timing_data([]) is False

    def test_conversations_without_timing_returns_false(self):
        conversations = [
            Conversation(
                session_id="s1",
                turns=[Turn(texts=[])],
            )
        ]
        assert conversations_have_timing_data(conversations) is False

    def test_conversations_with_timestamp_returns_true(self):
        conversations = [
            Conversation(
                session_id="s1",
                turns=[Turn(timestamp=1000)],
            )
        ]
        assert conversations_have_timing_data(conversations) is True

    def test_conversations_with_delay_returns_true(self):
        conversations = [
            Conversation(
                session_id="s1",
                turns=[Turn(delay=50)],
            )
        ]
        assert conversations_have_timing_data(conversations) is True

    def test_metadata_with_timestamp_ms_returns_true(self):
        metadata = [
            ConversationMetadata(
                conversation_id="s1",
                turns=[TurnMetadata(timestamp_ms=1000)],
            )
        ]
        assert conversations_have_timing_data(metadata) is True

    def test_metadata_with_delay_ms_returns_true(self):
        metadata = [
            ConversationMetadata(
                conversation_id="s1",
                turns=[TurnMetadata(delay_ms=50)],
            )
        ]
        assert conversations_have_timing_data(metadata) is True

    def test_metadata_without_timing_returns_false(self):
        metadata = [
            ConversationMetadata(
                conversation_id="s1",
                turns=[TurnMetadata()],
            )
        ]
        assert conversations_have_timing_data(metadata) is False

    def test_mixed_conversations_returns_true_on_any(self):
        conversations = [
            Conversation(session_id="s1", turns=[Turn(texts=[])]),
            Conversation(session_id="s2", turns=[Turn(timestamp=500)]),
        ]
        assert conversations_have_timing_data(conversations) is True


# =============================================================================
# PHASE PROPERTY QUERIES
# =============================================================================


class TestGetPhaseTiming:
    @pytest.mark.parametrize(
        "phase_type, expected_mode, expected_pattern",
        [
            param(PhaseType.CONCURRENCY, TimingMode.REQUEST_RATE, ArrivalPattern.CONCURRENCY_BURST, id="concurrency"),
            param(PhaseType.POISSON, TimingMode.REQUEST_RATE, ArrivalPattern.POISSON, id="poisson"),
            param(PhaseType.GAMMA, TimingMode.REQUEST_RATE, ArrivalPattern.GAMMA, id="gamma"),
            param(PhaseType.CONSTANT, TimingMode.REQUEST_RATE, ArrivalPattern.CONSTANT, id="constant"),
            param(PhaseType.FIXED_SCHEDULE, TimingMode.FIXED_SCHEDULE, ArrivalPattern.POISSON, id="fixed_schedule"),
            param(PhaseType.USER_CENTRIC, TimingMode.USER_CENTRIC_RATE, ArrivalPattern.POISSON, id="user_centric"),
        ],
    )  # fmt: skip
    def test_returns_correct_mapping(self, phase_type, expected_mode, expected_pattern):
        mode, pattern = get_phase_timing(phase_type)
        assert mode == expected_mode
        assert pattern == expected_pattern


class TestGetStopCondition:
    def test_requests_condition(self):
        phase = _concurrency_phase(requests=100)
        assert get_stop_condition(phase) == "requests"

    def test_duration_condition(self):
        phase = _concurrency_phase(requests=None, duration=60.0)
        assert get_stop_condition(phase) == "duration"

    def test_sessions_condition(self):
        phase = _concurrency_phase(requests=None, sessions=10)
        assert get_stop_condition(phase) == "sessions"

    def test_requests_takes_priority(self):
        phase = _concurrency_phase(requests=100, duration=60.0, sessions=10)
        assert get_stop_condition(phase) == "requests"

    def test_fixed_schedule_no_condition(self):
        phase = _fixed_schedule_phase()
        assert get_stop_condition(phase) == "none"


class TestRequiresSequentialSampling:
    def test_fixed_schedule_requires_sequential(self):
        assert requires_sequential_sampling(PhaseType.FIXED_SCHEDULE) is True

    @pytest.mark.parametrize(
        "phase_type",
        [
            param(PhaseType.CONCURRENCY, id="concurrency"),
            param(PhaseType.POISSON, id="poisson"),
            param(PhaseType.GAMMA, id="gamma"),
            param(PhaseType.CONSTANT, id="constant"),
            param(PhaseType.USER_CENTRIC, id="user_centric"),
        ],
    )  # fmt: skip
    def test_other_types_do_not_require_sequential(self, phase_type):
        assert requires_sequential_sampling(phase_type) is False


class TestRequiresMultiTurn:
    def test_user_centric_requires_multi_turn(self):
        assert requires_multi_turn(PhaseType.USER_CENTRIC) is True

    @pytest.mark.parametrize(
        "phase_type",
        [
            param(PhaseType.CONCURRENCY, id="concurrency"),
            param(PhaseType.POISSON, id="poisson"),
            param(PhaseType.FIXED_SCHEDULE, id="fixed_schedule"),
        ],
    )  # fmt: skip
    def test_other_types_do_not_require_multi_turn(self, phase_type):
        assert requires_multi_turn(phase_type) is False


# =============================================================================
# PHASE-DATASET COMPATIBILITY
# =============================================================================


class TestCheckPhaseDatasetCompatibility:
    def test_concurrency_with_any_dataset_is_compatible(self):
        phase = _concurrency_phase()
        dataset = _synthetic_dataset()
        errors = check_phase_dataset_compatibility(phase, dataset, "p1", "d1")
        assert errors == []

    def test_fixed_schedule_with_sequential_file_is_compatible(self):
        phase = _fixed_schedule_phase()
        dataset = _file_dataset(format="mooncake_trace", sampling="sequential")
        errors = check_phase_dataset_compatibility(phase, dataset, "p1", "d1")
        assert errors == []

    def test_fixed_schedule_with_random_file_is_incompatible(self):
        phase = _fixed_schedule_phase()
        dataset = _file_dataset(format="mooncake_trace", sampling="random")
        errors = check_phase_dataset_compatibility(phase, dataset, "p1", "d1")
        assert len(errors) == 1
        assert "sequential sampling" in errors[0]

    def test_fixed_schedule_with_synthetic_is_compatible(self):
        phase = _fixed_schedule_phase()
        dataset = _synthetic_dataset()
        errors = check_phase_dataset_compatibility(phase, dataset, "p1", "d1")
        assert errors == []

    def test_user_centric_with_multi_turn_is_compatible(self):
        phase = _user_centric_phase()
        dataset = _file_dataset(format="multi_turn")
        errors = check_phase_dataset_compatibility(phase, dataset, "p1", "d1")
        assert errors == []

    def test_user_centric_with_single_turn_is_incompatible(self):
        phase = _user_centric_phase()
        dataset = _file_dataset(format="single_turn")
        errors = check_phase_dataset_compatibility(phase, dataset, "p1", "d1")
        assert len(errors) == 1
        assert "multi_turn" in errors[0]

    def test_user_centric_with_synthetic_is_compatible(self):
        phase = _user_centric_phase()
        dataset = _synthetic_dataset()
        errors = check_phase_dataset_compatibility(phase, dataset, "p1", "d1")
        assert errors == []

    def test_phase_name_in_error_message(self):
        phase = _fixed_schedule_phase()
        dataset = _file_dataset(sampling="random")
        errors = check_phase_dataset_compatibility(
            phase, dataset, "warmup", "trace_data"
        )
        assert "warmup" in errors[0]
        assert "trace_data" in errors[0]

    def test_composed_dataset_with_sequential_sampling(self):
        phase = _fixed_schedule_phase()
        dataset = _composed_dataset(
            source={
                "type": "file",
                "path": "/tmp/t.jsonl",
                "format": "mooncake_trace",
                "sampling": "sequential",
            }
        )
        errors = check_phase_dataset_compatibility(phase, dataset, "p1", "d1")
        assert errors == []
