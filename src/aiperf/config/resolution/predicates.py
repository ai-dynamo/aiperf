# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared config resolution utilities.

Pure functions that derive properties from config objects. These are the
single source of truth for config-derived state, used by validators,
DatasetManager, TimingConfig, and tests.

All functions are stateless - they take config objects as input and return
derived values. No side effects, no I/O, no caching.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from aiperf.common.enums import DatasetFormat, DatasetType
from aiperf.plugin.enums import (
    ArrivalPattern,
    DatasetSamplingStrategy,
    PhaseType,
    TimingMode,
)

if TYPE_CHECKING:
    from aiperf.common.models.dataset_models import (
        Conversation,
        ConversationMetadata,
    )
    from aiperf.config.dataset import DatasetConfig
    from aiperf.config.phases import BasePhaseConfig


# =============================================================================
# DATASET PROPERTY QUERIES
# =============================================================================


def get_dataset_type(dataset: DatasetConfig) -> DatasetType:
    """Get the type of a dataset config."""
    return dataset.type


def get_sampling_strategy(dataset: DatasetConfig) -> DatasetSamplingStrategy:
    """Get sampling strategy from a dataset config.

    Uses getattr for safety with the discriminated union; defaults to
    ``SEQUENTIAL`` when the variant has no ``sampling`` field.
    """
    return getattr(dataset, "sampling", DatasetSamplingStrategy.SEQUENTIAL)


def is_file_dataset(dataset: DatasetConfig) -> bool:
    """Check if dataset is file-based."""
    from aiperf.config.dataset import FileDataset

    return isinstance(dataset, FileDataset)


def is_synthetic_dataset(dataset: DatasetConfig) -> bool:
    """Check if dataset is synthetically generated."""
    from aiperf.config.dataset import SyntheticDataset

    return isinstance(dataset, SyntheticDataset)


def is_public_dataset(dataset: DatasetConfig) -> bool:
    """Check if dataset is a public benchmark dataset."""
    from aiperf.config.dataset import PublicDataset

    return isinstance(dataset, PublicDataset)


def is_trace_dataset(dataset: DatasetConfig) -> bool:
    """Check if dataset uses the Mooncake trace format.

    Narrower than ``plugins.is_trace_dataset`` (which checks the plugin
    registry's ``is_trace`` flag across all trace loaders). This predicate
    is only used by ``custom`` composer fallback logic that special-cases
    Mooncake; other trace formats (burst_gpt_trace, bailian_trace) do not
    flow through this code path.
    """
    fmt = get_dataset_format(dataset)
    if fmt is None:
        return False
    return fmt == DatasetFormat.MOONCAKE_TRACE


def is_multi_turn_dataset(dataset: DatasetConfig) -> bool:
    """Check if dataset uses multi-turn conversational format."""
    fmt = get_dataset_format(dataset)
    if fmt is None:
        return False
    return fmt == DatasetFormat.MULTI_TURN


def get_dataset_format(dataset: DatasetConfig) -> DatasetFormat | None:
    """Get the format of a file-based dataset, or None for synthetic/public."""
    from aiperf.config.dataset import FileDataset

    if isinstance(dataset, FileDataset):
        return dataset.format
    return None


def get_dataset_entries(dataset: DatasetConfig) -> int | None:
    """Get the configured entry count, or None if using all/default."""
    return getattr(dataset, "entries", None)


def get_random_seed(dataset: DatasetConfig) -> int | None:
    """Get the dataset-specific random seed."""
    return getattr(dataset, "random_seed", None)


# =============================================================================
# TIMING DATA DETECTION
# =============================================================================


def conversations_have_timing_data(
    conversations: list[Conversation] | list[ConversationMetadata],
) -> bool:
    """Determine if conversations contain timing data (timestamps or delays).

    Works with both Conversation objects (pre-materialization) and
    ConversationMetadata objects (post-materialization).

    Args:
        conversations: List of Conversation or ConversationMetadata objects.

    Returns:
        True if any turn in any conversation has a timestamp or delay value.
    """
    for conv in conversations:
        for turn in conv.turns:
            # ConversationMetadata turns have timestamp_ms/delay_ms
            # Conversation turns have timestamp/delay
            ts = getattr(turn, "timestamp_ms", None) or getattr(turn, "timestamp", None)
            delay = getattr(turn, "delay_ms", None) or getattr(turn, "delay", None)
            if ts is not None or delay is not None:
                return True
    return False


# =============================================================================
# PHASE PROPERTY QUERIES
# =============================================================================


def get_phase_timing(phase_type: PhaseType) -> tuple[TimingMode, ArrivalPattern]:
    """Map PhaseType to (TimingMode, ArrivalPattern).

    This is the single source of truth for the phase type -> timing mode mapping.
    Used by TimingConfig.from_config() and anywhere else that needs this mapping.
    """
    mapping: dict[PhaseType, tuple[TimingMode, ArrivalPattern]] = {
        PhaseType.CONCURRENCY: (
            TimingMode.REQUEST_RATE,
            ArrivalPattern.CONCURRENCY_BURST,
        ),
        PhaseType.POISSON: (TimingMode.REQUEST_RATE, ArrivalPattern.POISSON),
        PhaseType.GAMMA: (TimingMode.REQUEST_RATE, ArrivalPattern.GAMMA),
        PhaseType.CONSTANT: (TimingMode.REQUEST_RATE, ArrivalPattern.CONSTANT),
        PhaseType.FIXED_SCHEDULE: (
            TimingMode.FIXED_SCHEDULE,
            ArrivalPattern.POISSON,
        ),
        PhaseType.USER_CENTRIC: (
            TimingMode.USER_CENTRIC_RATE,
            ArrivalPattern.POISSON,
        ),
    }
    return mapping.get(phase_type, (TimingMode.REQUEST_RATE, ArrivalPattern.POISSON))


def get_stop_condition(phase: BasePhaseConfig) -> str:
    """Get which stop condition is primary for this phase.

    Returns:
        One of 'requests', 'duration', 'sessions', or 'none'.
    """
    if phase.requests is not None:
        return "requests"
    if phase.duration is not None:
        return "duration"
    if phase.sessions is not None:
        return "sessions"
    return "none"


def requires_sequential_sampling(phase_type: PhaseType) -> bool:
    """Check if a phase type requires sequential sampling."""
    return phase_type == PhaseType.FIXED_SCHEDULE


def requires_multi_turn(phase_type: PhaseType) -> bool:
    """Check if a phase type requires multi-turn datasets."""
    return phase_type == PhaseType.USER_CENTRIC


# =============================================================================
# PHASE-DATASET COMPATIBILITY
# =============================================================================


def check_phase_dataset_compatibility(
    phase: BasePhaseConfig,
    dataset: DatasetConfig,
    phase_name: str,
    dataset_name: str,
) -> list[str]:
    """Check compatibility between a phase and its dataset.

    Returns a list of error messages. Empty list means compatible.

    Args:
        phase: The phase configuration.
        dataset: The dataset configuration for this phase.
        phase_name: Name of the phase (for error messages).
        dataset_name: Name of the dataset (for error messages).

    Returns:
        List of error message strings (empty if compatible).
    """
    errors: list[str] = []

    if requires_sequential_sampling(phase.type) and is_file_dataset(dataset):
        sampling = get_sampling_strategy(dataset)
        if sampling != DatasetSamplingStrategy.SEQUENTIAL:
            errors.append(
                f"Phase '{phase_name}' uses {phase.type} which requires "
                f"sequential sampling, but dataset '{dataset_name}' uses "
                f"'{sampling}' sampling"
            )

    if (
        requires_multi_turn(phase.type)
        and is_file_dataset(dataset)
        and not is_multi_turn_dataset(dataset)
    ):
        fmt = get_dataset_format(dataset)
        errors.append(
            f"Phase '{phase_name}' uses {phase.type} which requires "
            f"multi_turn dataset format, but dataset '{dataset_name}' uses "
            f"'{fmt}' format"
        )

    return errors
